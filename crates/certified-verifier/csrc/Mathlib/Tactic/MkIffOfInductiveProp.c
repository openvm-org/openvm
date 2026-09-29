// Lean compiler output
// Module: Mathlib.Tactic.MkIffOfInductiveProp
// Imports: public import Init public meta import Init public meta import Lean.Elab.DeclarationRange public meta import Lean.Meta.Tactic.Cases public meta import Mathlib.Lean.Meta public meta import Mathlib.Lean.Name public import Mathlib.Init
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
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_DeclarationRange_ofStringPositions(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_privateToUserName_x3f(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_replaceFVar(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sortLevel_x21(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_occurs(lean_object*, lean_object*);
uint8_t lean_level_eq(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_getLast_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Expr_replaceFVars(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_instantiateTypeLevelParams(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_unzipTR___redArg(lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isProp(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_cases(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipIdxTR___redArg(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_nthConstructor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_existsi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
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
lean_object* l_Lean_Elab_Tactic_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_LocalContext_getFVarIds(lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_tryClear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_revert(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_subst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_addDecl(lean_object*, uint8_t, lean_object*, lean_object*);
extern lean_object* l_Lean_declRangeExt;
lean_object* l_Lean_MapDeclarationExtension_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_mkPrivateName(lean_object*, lean_object*);
lean_object* l_Lean_privateToUserName(lean_object*);
uint8_t lean_is_reserved_name(lean_object*, lean_object*);
lean_object* l_Lean_addProtected(lean_object*, lean_object*);
uint8_t l_Lean_Elab_Visibility_isInferredPublic(lean_object*, uint8_t);
uint8_t l_Lean_isStructure(lean_object*, lean_object*);
lean_object* l_Lean_getStructureFieldsFlattened(lean_object*, lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_Name_isAtomic(lean_object*);
lean_object* l_Lean_extractMacroScopes(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Name_replacePrefix(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MacroScopesView_review(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Name_decapitalize(lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 26, 230, 200, 188, 33, 106, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected only one new goal"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "right"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__5_value),LEAN_SCALAR_PTR_LITERAL(192, 52, 10, 58, 87, 38, 120, 247)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21_spec__0(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Mathlib.Tactic.MkIffOfInductiveProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 93, .m_capacity = 93, .m_length = 92, .m_data = "_private.Mathlib.Tactic.MkIffOfInductiveProp.0.Mathlib.Tactic.MkIff.updateLambdaBinderInfoD!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "lambda expected"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21(lean_object*);
static const lean_string_object lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__0 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__1 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__1_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__2 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__3 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__3_value;
static lean_once_cell_t lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOpList___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkAndList(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Or"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 237, 162, 225, 217, 98, 205, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__3_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_List_init(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__0_value;
static const lean_ctor_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 180, 169, 191, 74, 196, 152, 188)}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__1 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__1_value;
static const lean_string_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__2 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__2_value;
static const lean_ctor_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__3 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_constrToProp_spec__3(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0;
static const lean_array_object lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constructor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(144, 188, 57, 91, 27, 124, 155, 13)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__8_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "expected two subgoals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__3_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "expected no subgoals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_toCases_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toCases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "expected fvar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected two case subgoals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected one case subgoals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__1(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toInductive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toInductive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "mk_iff only applies to prop-valued declarations"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 155, 85, 49, 105, 137, 67, 168)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "failed to split goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__6_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "mk_iff only applies to inductive declarations"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "MkIff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mkIff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value),LEAN_SCALAR_PTR_LITERAL(150, 53, 231, 251, 140, 117, 252, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 216, 89, 204, 67, 218, 195, 67)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mk_iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIff = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "mkIffOfInductiveProp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value),LEAN_SCALAR_PTR_LITERAL(150, 53, 231, 251, 140, 117, 252, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 23, 148, 111, 219, 53, 254, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "mk_iff_of_inductive_prop "};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2;
static const lean_array_object lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "invalid declaration name `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "`, structure `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "` has field `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "` has already been declared"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "` is a reserved name"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "a private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "a non-private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0___boxed, .m_arity = 7, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "protected declarations must be in a namespace"};
static const lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 94, .m_capacity = 94, .m_length = 93, .m_data = "invalid declaration name `_root_`, `_root_` is a prefix used to refer to the 'root' namespace"};
static const lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "unrecognized syntax"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_iff"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "MkIffOfInductiveProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(249, 32, 157, 58, 64, 147, 180, 20)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__6_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(220, 135, 48, 90, 228, 250, 88, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__6_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__6_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__7_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__6_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 242, 227, 246, 221, 46, 37, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__7_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__7_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__8_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__7_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 200, 164, 92, 42, 98, 168, 229)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__8_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__8_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__9_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__8_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value),LEAN_SCALAR_PTR_LITERAL(245, 170, 235, 236, 222, 144, 207, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__9_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__9_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__10_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__10_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__10_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__11_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__9_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__10_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(28, 152, 97, 24, 125, 140, 126, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__11_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__11_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__12_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__12_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__12_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__13_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__11_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__12_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(149, 142, 77, 114, 173, 2, 100, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__13_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__13_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__14_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__13_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 25, 97, 58, 144, 193, 199, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__14_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__14_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__15_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__14_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(141, 222, 159, 170, 106, 71, 0, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__15_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__15_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__16_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__15_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(115, 181, 160, 21, 105, 190, 174, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__16_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__16_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__17_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__16_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1665152415) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(176, 251, 28, 229, 244, 17, 5, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__17_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__17_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__18_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__18_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__18_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__19_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__17_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__18_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 194, 168, 129, 244, 202, 147, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__19_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__19_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__20_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__20_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__20_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__21_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__19_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__20_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(175, 135, 48, 99, 39, 141, 27, 226)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__21_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__21_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__22_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__21_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(170, 208, 247, 236, 30, 131, 214, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__22_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__22_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__23_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed, .m_arity = 12, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__23_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__23_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__24_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__2_value),LEAN_SCALAR_PTR_LITERAL(63, 33, 238, 10, 8, 225, 65, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__24_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__24_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__25_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__24_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__25_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__25_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__26_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "Generate an `iff` lemma for an inductive `Prop`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__26_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__26_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__27_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__22_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__24_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__26_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__27_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__27_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__28_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__27_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__23_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__25_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__28_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__28_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__3));
v___x_54_ = l_Lean_stringToMessageData(v___x_53_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__7));
v___x_60_ = l_Lean_stringToMessageData(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select(lean_object* v_m_61_, lean_object* v_n_62_, lean_object* v_goal_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v_zero_69_; uint8_t v_isZero_70_; 
v_zero_69_ = lean_unsigned_to_nat(0u);
v_isZero_70_ = lean_nat_dec_eq(v_m_61_, v_zero_69_);
if (v_isZero_70_ == 1)
{
uint8_t v_isZero_71_; 
lean_dec(v_m_61_);
v_isZero_71_ = lean_nat_dec_eq(v_n_62_, v_zero_69_);
lean_dec(v_n_62_);
if (v_isZero_71_ == 1)
{
lean_object* v___x_72_; 
v___x_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_72_, 0, v_goal_63_);
return v___x_72_;
}
else
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__1));
v___x_74_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__2));
v___x_75_ = l_Lean_MVarId_nthConstructor(v___x_73_, v_zero_69_, v___x_74_, v_goal_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
if (lean_obj_tag(v___x_75_) == 0)
{
lean_object* v_a_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_92_; 
v_a_76_ = lean_ctor_get(v___x_75_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_75_);
if (v_isSharedCheck_92_ == 0)
{
v___x_78_ = v___x_75_;
v_isShared_79_ = v_isSharedCheck_92_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_a_76_);
lean_dec(v___x_75_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_92_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___y_81_; lean_object* v___y_82_; lean_object* v___y_83_; lean_object* v___y_84_; 
if (lean_obj_tag(v_a_76_) == 1)
{
lean_object* v_tail_87_; 
v_tail_87_ = lean_ctor_get(v_a_76_, 1);
if (lean_obj_tag(v_tail_87_) == 0)
{
lean_object* v_head_88_; lean_object* v___x_90_; 
v_head_88_ = lean_ctor_get(v_a_76_, 0);
lean_inc(v_head_88_);
lean_dec_ref_known(v_a_76_, 2);
if (v_isShared_79_ == 0)
{
lean_ctor_set(v___x_78_, 0, v_head_88_);
v___x_90_ = v___x_78_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_91_; 
v_reuseFailAlloc_91_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_91_, 0, v_head_88_);
v___x_90_ = v_reuseFailAlloc_91_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
return v___x_90_;
}
}
else
{
lean_dec_ref_known(v_a_76_, 2);
lean_del_object(v___x_78_);
v___y_81_ = v_a_64_;
v___y_82_ = v_a_65_;
v___y_83_ = v_a_66_;
v___y_84_ = v_a_67_;
goto v___jp_80_;
}
}
else
{
lean_del_object(v___x_78_);
lean_dec(v_a_76_);
v___y_81_ = v_a_64_;
v___y_82_ = v_a_65_;
v___y_83_ = v_a_66_;
v___y_84_ = v_a_67_;
goto v___jp_80_;
}
v___jp_80_:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4);
v___x_86_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_85_, v___y_81_, v___y_82_, v___y_83_, v___y_84_);
return v___x_86_;
}
}
}
else
{
lean_object* v_a_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_100_; 
v_a_93_ = lean_ctor_get(v___x_75_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v___x_75_);
if (v_isSharedCheck_100_ == 0)
{
v___x_95_ = v___x_75_;
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_a_93_);
lean_dec(v___x_75_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_98_; 
if (v_isShared_96_ == 0)
{
v___x_98_ = v___x_95_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_a_93_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
}
}
else
{
uint8_t v_isZero_101_; 
v_isZero_101_ = lean_nat_dec_eq(v_n_62_, v_zero_69_);
if (v_isZero_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__6));
v___x_103_ = lean_unsigned_to_nat(1u);
v___x_104_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__2));
v___x_105_ = l_Lean_MVarId_nthConstructor(v___x_102_, v___x_103_, v___x_104_, v_goal_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
if (lean_obj_tag(v___x_105_) == 0)
{
lean_object* v_a_106_; lean_object* v___y_108_; lean_object* v___y_109_; lean_object* v___y_110_; lean_object* v___y_111_; 
v_a_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_a_106_);
lean_dec_ref_known(v___x_105_, 1);
if (lean_obj_tag(v_a_106_) == 1)
{
lean_object* v_tail_114_; 
v_tail_114_ = lean_ctor_get(v_a_106_, 1);
if (lean_obj_tag(v_tail_114_) == 0)
{
lean_object* v_head_115_; lean_object* v_n_116_; lean_object* v_n_117_; 
v_head_115_ = lean_ctor_get(v_a_106_, 0);
lean_inc(v_head_115_);
lean_dec_ref_known(v_a_106_, 2);
v_n_116_ = lean_nat_sub(v_m_61_, v___x_103_);
lean_dec(v_m_61_);
v_n_117_ = lean_nat_sub(v_n_62_, v___x_103_);
lean_dec(v_n_62_);
v_m_61_ = v_n_116_;
v_n_62_ = v_n_117_;
v_goal_63_ = v_head_115_;
goto _start;
}
else
{
lean_dec_ref_known(v_a_106_, 2);
lean_dec(v_n_62_);
lean_dec(v_m_61_);
v___y_108_ = v_a_64_;
v___y_109_ = v_a_65_;
v___y_110_ = v_a_66_;
v___y_111_ = v_a_67_;
goto v___jp_107_;
}
}
else
{
lean_dec(v_a_106_);
lean_dec(v_n_62_);
lean_dec(v_m_61_);
v___y_108_ = v_a_64_;
v___y_109_ = v_a_65_;
v___y_110_ = v_a_66_;
v___y_111_ = v_a_67_;
goto v___jp_107_;
}
v___jp_107_:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__4);
v___x_113_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_112_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
return v___x_113_;
}
}
else
{
lean_object* v_a_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_126_; 
lean_dec(v_n_62_);
lean_dec(v_m_61_);
v_a_119_ = lean_ctor_get(v___x_105_, 0);
v_isSharedCheck_126_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_126_ == 0)
{
v___x_121_ = v___x_105_;
v_isShared_122_ = v_isSharedCheck_126_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_a_119_);
lean_dec(v___x_105_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_126_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
lean_object* v___x_124_; 
if (v_isShared_122_ == 0)
{
v___x_124_ = v___x_121_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v_a_119_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
}
else
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v_goal_63_);
lean_dec(v_n_62_);
lean_dec(v_m_61_);
v___x_127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___closed__8);
v___x_128_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_127_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
return v___x_128_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select___boxed(lean_object* v_m_129_, lean_object* v_n_130_, lean_object* v_goal_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select(v_m_129_, v_n_130_, v_goal_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_);
lean_dec(v_a_135_);
lean_dec_ref(v_a_134_);
lean_dec(v_a_133_);
lean_dec_ref(v_a_132_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0(lean_object* v_00_u03b1_138_, lean_object* v_msg_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v_msg_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___boxed(lean_object* v_00_u03b1_146_, lean_object* v_msg_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0(v_00_u03b1_146_, v_msg_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0(lean_object* v___y_154_){
_start:
{
lean_inc_ref(v___y_154_);
return v___y_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0___boxed(lean_object* v___y_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__0(v___y_155_);
lean_dec_ref(v___y_155_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1(lean_object* v_snd_157_, lean_object* v_head_158_, lean_object* v_fst_159_, lean_object* v___y_160_){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = lean_apply_1(v_snd_157_, v___y_160_);
v___x_162_ = l_Lean_Expr_replaceFVar(v___x_161_, v_head_158_, v_fst_159_);
lean_dec_ref(v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1___boxed(lean_object* v_snd_163_, lean_object* v_head_164_, lean_object* v_fst_165_, lean_object* v___y_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1(v_snd_163_, v_head_164_, v_fst_165_, v___y_166_);
lean_dec_ref(v_fst_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1(lean_object* v_head_168_, lean_object* v_fst_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
if (lean_obj_tag(v_a_170_) == 0)
{
lean_object* v___x_172_; 
lean_dec_ref(v_head_168_);
v___x_172_ = l_List_reverse___redArg(v_a_171_);
return v___x_172_;
}
else
{
lean_object* v_head_173_; lean_object* v_tail_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_183_; 
v_head_173_ = lean_ctor_get(v_a_170_, 0);
v_tail_174_ = lean_ctor_get(v_a_170_, 1);
v_isSharedCheck_183_ = !lean_is_exclusive(v_a_170_);
if (v_isSharedCheck_183_ == 0)
{
v___x_176_ = v_a_170_;
v_isShared_177_ = v_isSharedCheck_183_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_tail_174_);
lean_inc(v_head_173_);
lean_dec(v_a_170_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_183_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_178_; lean_object* v___x_180_; 
lean_inc_ref(v_head_168_);
v___x_178_ = l_Lean_Expr_replaceFVar(v_head_173_, v_head_168_, v_fst_169_);
lean_dec(v_head_173_);
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 1, v_a_171_);
lean_ctor_set(v___x_176_, 0, v___x_178_);
v___x_180_ = v___x_176_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_182_; 
v_reuseFailAlloc_182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_182_, 0, v___x_178_);
lean_ctor_set(v_reuseFailAlloc_182_, 1, v_a_171_);
v___x_180_ = v_reuseFailAlloc_182_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
v_a_170_ = v_tail_174_;
v_a_171_ = v___x_180_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1___boxed(lean_object* v_head_184_, lean_object* v_fst_185_, lean_object* v_a_186_, lean_object* v_a_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1(v_head_184_, v_fst_185_, v_a_186_, v_a_187_);
lean_dec_ref(v_fst_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2(lean_object* v_head_189_, lean_object* v_fst_190_, lean_object* v_a_191_, lean_object* v_a_192_){
_start:
{
if (lean_obj_tag(v_a_191_) == 0)
{
lean_object* v___x_193_; 
lean_dec_ref(v_head_189_);
v___x_193_ = l_List_reverse___redArg(v_a_192_);
return v___x_193_;
}
else
{
lean_object* v_head_194_; lean_object* v_tail_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_213_; 
v_head_194_ = lean_ctor_get(v_a_191_, 0);
v_tail_195_ = lean_ctor_get(v_a_191_, 1);
v_isSharedCheck_213_ = !lean_is_exclusive(v_a_191_);
if (v_isSharedCheck_213_ == 0)
{
v___x_197_ = v_a_191_;
v_isShared_198_ = v_isSharedCheck_213_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_tail_195_);
lean_inc(v_head_194_);
lean_dec(v_a_191_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_213_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v_fst_199_; lean_object* v_snd_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_212_; 
v_fst_199_ = lean_ctor_get(v_head_194_, 0);
v_snd_200_ = lean_ctor_get(v_head_194_, 1);
v_isSharedCheck_212_ = !lean_is_exclusive(v_head_194_);
if (v_isSharedCheck_212_ == 0)
{
v___x_202_ = v_head_194_;
v_isShared_203_ = v_isSharedCheck_212_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_snd_200_);
lean_inc(v_fst_199_);
lean_dec(v_head_194_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_212_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_204_; lean_object* v___x_206_; 
lean_inc_ref(v_head_189_);
v___x_204_ = l_Lean_Expr_replaceFVar(v_snd_200_, v_head_189_, v_fst_190_);
lean_dec(v_snd_200_);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 1, v___x_204_);
v___x_206_ = v___x_202_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_fst_199_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v___x_204_);
v___x_206_ = v_reuseFailAlloc_211_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
lean_object* v___x_208_; 
if (v_isShared_198_ == 0)
{
lean_ctor_set(v___x_197_, 1, v_a_192_);
lean_ctor_set(v___x_197_, 0, v___x_206_);
v___x_208_ = v___x_197_;
goto v_reusejp_207_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_206_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v_a_192_);
v___x_208_ = v_reuseFailAlloc_210_;
goto v_reusejp_207_;
}
v_reusejp_207_:
{
v_a_191_ = v_tail_195_;
v_a_192_ = v___x_208_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2___boxed(lean_object* v_head_214_, lean_object* v_fst_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2(v_head_214_, v_fst_215_, v_a_216_, v_a_217_);
lean_dec_ref(v_fst_215_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0(lean_object* v_head_219_, lean_object* v_a_220_, lean_object* v_a_221_){
_start:
{
if (lean_obj_tag(v_a_220_) == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_222_ = l_List_reverse___redArg(v_a_221_);
v___x_223_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_a_220_);
return v___x_223_;
}
else
{
lean_object* v_head_224_; lean_object* v_tail_225_; lean_object* v_snd_226_; uint8_t v___x_227_; 
v_head_224_ = lean_ctor_get(v_a_220_, 0);
lean_inc(v_head_224_);
v_tail_225_ = lean_ctor_get(v_a_220_, 1);
v_snd_226_ = lean_ctor_get(v_head_224_, 1);
v___x_227_ = lean_expr_eqv(v_snd_226_, v_head_219_);
if (v___x_227_ == 0)
{
lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_235_; 
lean_inc(v_tail_225_);
v_isSharedCheck_235_ = !lean_is_exclusive(v_a_220_);
if (v_isSharedCheck_235_ == 0)
{
lean_object* v_unused_236_; lean_object* v_unused_237_; 
v_unused_236_ = lean_ctor_get(v_a_220_, 1);
lean_dec(v_unused_236_);
v_unused_237_ = lean_ctor_get(v_a_220_, 0);
lean_dec(v_unused_237_);
v___x_229_ = v_a_220_;
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
else
{
lean_dec(v_a_220_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 1, v_a_221_);
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_head_224_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v_a_221_);
v___x_232_ = v_reuseFailAlloc_234_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
v_a_220_ = v_tail_225_;
v_a_221_ = v___x_232_;
goto _start;
}
}
}
else
{
lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_245_; 
v_isSharedCheck_245_ = !lean_is_exclusive(v_head_224_);
if (v_isSharedCheck_245_ == 0)
{
lean_object* v_unused_246_; lean_object* v_unused_247_; 
v_unused_246_ = lean_ctor_get(v_head_224_, 1);
lean_dec(v_unused_246_);
v_unused_247_ = lean_ctor_get(v_head_224_, 0);
lean_dec(v_unused_247_);
v___x_239_ = v_head_224_;
v_isShared_240_ = v_isSharedCheck_245_;
goto v_resetjp_238_;
}
else
{
lean_dec(v_head_224_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_245_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_241_; lean_object* v___x_243_; 
v___x_241_ = l_List_reverse___redArg(v_a_221_);
if (v_isShared_240_ == 0)
{
lean_ctor_set(v___x_239_, 1, v_a_220_);
lean_ctor_set(v___x_239_, 0, v___x_241_);
v___x_243_ = v___x_239_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_241_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_a_220_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0___boxed(lean_object* v_head_248_, lean_object* v_a_249_, lean_object* v_a_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0(v_head_248_, v_a_249_, v_a_250_);
lean_dec_ref(v_head_248_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_compactRelation(lean_object* v_x_253_, lean_object* v_x_254_){
_start:
{
if (lean_obj_tag(v_x_253_) == 0)
{
lean_object* v___f_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___f_255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___closed__0));
v___x_256_ = lean_box(0);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v_x_254_);
lean_ctor_set(v___x_257_, 1, v___f_255_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_256_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
return v___x_258_;
}
else
{
lean_object* v_head_259_; lean_object* v_tail_260_; lean_object* v___x_262_; uint8_t v_isShared_263_; uint8_t v_isSharedCheck_317_; 
v_head_259_ = lean_ctor_get(v_x_253_, 0);
v_tail_260_ = lean_ctor_get(v_x_253_, 1);
v_isSharedCheck_317_ = !lean_is_exclusive(v_x_253_);
if (v_isSharedCheck_317_ == 0)
{
v___x_262_ = v_x_253_;
v_isShared_263_ = v_isSharedCheck_317_;
goto v_resetjp_261_;
}
else
{
lean_inc(v_tail_260_);
lean_inc(v_head_259_);
lean_dec(v_x_253_);
v___x_262_ = lean_box(0);
v_isShared_263_ = v_isSharedCheck_317_;
goto v_resetjp_261_;
}
v_resetjp_261_:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v_snd_266_; 
v___x_264_ = lean_box(0);
lean_inc(v_x_254_);
v___x_265_ = lp_mathlib_List_span_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__0(v_head_259_, v_x_254_, v___x_264_);
v_snd_266_ = lean_ctor_get(v___x_265_, 1);
lean_inc(v_snd_266_);
if (lean_obj_tag(v_snd_266_) == 0)
{
lean_object* v___x_267_; lean_object* v_fst_268_; lean_object* v_snd_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_280_; 
lean_dec_ref(v___x_265_);
v___x_267_ = lp_mathlib_Mathlib_Tactic_MkIff_compactRelation(v_tail_260_, v_x_254_);
v_fst_268_ = lean_ctor_get(v___x_267_, 0);
v_snd_269_ = lean_ctor_get(v___x_267_, 1);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_267_);
if (v_isSharedCheck_280_ == 0)
{
v___x_271_ = v___x_267_;
v_isShared_272_ = v_isSharedCheck_280_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_snd_269_);
lean_inc(v_fst_268_);
lean_dec(v___x_267_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_280_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___x_273_; lean_object* v___x_275_; 
v___x_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_273_, 0, v_head_259_);
if (v_isShared_263_ == 0)
{
lean_ctor_set(v___x_262_, 1, v_fst_268_);
lean_ctor_set(v___x_262_, 0, v___x_273_);
v___x_275_ = v___x_262_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v___x_273_);
lean_ctor_set(v_reuseFailAlloc_279_, 1, v_fst_268_);
v___x_275_ = v_reuseFailAlloc_279_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
lean_object* v___x_277_; 
if (v_isShared_272_ == 0)
{
lean_ctor_set(v___x_271_, 0, v___x_275_);
v___x_277_ = v___x_271_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v_snd_269_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
else
{
lean_object* v_head_281_; lean_object* v_fst_282_; lean_object* v_tail_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_315_; 
lean_del_object(v___x_262_);
lean_dec(v_x_254_);
v_head_281_ = lean_ctor_get(v_snd_266_, 0);
lean_inc(v_head_281_);
v_fst_282_ = lean_ctor_get(v___x_265_, 0);
lean_inc(v_fst_282_);
lean_dec_ref(v___x_265_);
v_tail_283_ = lean_ctor_get(v_snd_266_, 1);
v_isSharedCheck_315_ = !lean_is_exclusive(v_snd_266_);
if (v_isSharedCheck_315_ == 0)
{
lean_object* v_unused_316_; 
v_unused_316_ = lean_ctor_get(v_snd_266_, 0);
lean_dec(v_unused_316_);
v___x_285_ = v_snd_266_;
v_isShared_286_ = v_isSharedCheck_315_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_tail_283_);
lean_dec(v_snd_266_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_315_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v_fst_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v_snd_292_; lean_object* v_fst_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_314_; 
v_fst_287_ = lean_ctor_get(v_head_281_, 0);
lean_inc(v_fst_287_);
lean_dec(v_head_281_);
lean_inc_n(v_head_259_, 2);
v___x_288_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__1(v_head_259_, v_fst_287_, v_tail_260_, v___x_264_);
v___x_289_ = l_List_appendTR___redArg(v_fst_282_, v_tail_283_);
v___x_290_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_compactRelation_spec__2(v_head_259_, v_fst_287_, v___x_289_, v___x_264_);
v___x_291_ = lp_mathlib_Mathlib_Tactic_MkIff_compactRelation(v___x_288_, v___x_290_);
v_snd_292_ = lean_ctor_get(v___x_291_, 1);
v_fst_293_ = lean_ctor_get(v___x_291_, 0);
v_isSharedCheck_314_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_314_ == 0)
{
v___x_295_ = v___x_291_;
v_isShared_296_ = v_isSharedCheck_314_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_snd_292_);
lean_inc(v_fst_293_);
lean_dec(v___x_291_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_314_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v_fst_297_; lean_object* v_snd_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_313_; 
v_fst_297_ = lean_ctor_get(v_snd_292_, 0);
v_snd_298_ = lean_ctor_get(v_snd_292_, 1);
v_isSharedCheck_313_ = !lean_is_exclusive(v_snd_292_);
if (v_isSharedCheck_313_ == 0)
{
v___x_300_ = v_snd_292_;
v_isShared_301_ = v_isSharedCheck_313_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_snd_298_);
lean_inc(v_fst_297_);
lean_dec(v_snd_292_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_313_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___f_302_; lean_object* v___x_303_; lean_object* v___x_305_; 
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_compactRelation___lam__1___boxed), 4, 3);
lean_closure_set(v___f_302_, 0, v_snd_298_);
lean_closure_set(v___f_302_, 1, v_head_259_);
lean_closure_set(v___f_302_, 2, v_fst_287_);
v___x_303_ = lean_box(0);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 1, v_fst_293_);
lean_ctor_set(v___x_285_, 0, v___x_303_);
v___x_305_ = v___x_285_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_312_, 1, v_fst_293_);
v___x_305_ = v_reuseFailAlloc_312_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
lean_object* v___x_307_; 
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 1, v___f_302_);
v___x_307_ = v___x_300_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v_fst_297_);
lean_ctor_set(v_reuseFailAlloc_311_, 1, v___f_302_);
v___x_307_ = v_reuseFailAlloc_311_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
lean_object* v___x_309_; 
if (v_isShared_296_ == 0)
{
lean_ctor_set(v___x_295_, 1, v___x_307_);
lean_ctor_set(v___x_295_, 0, v___x_305_);
v___x_309_ = v___x_295_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v___x_305_);
lean_ctor_set(v_reuseFailAlloc_310_, 1, v___x_307_);
v___x_309_ = v_reuseFailAlloc_310_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
return v___x_309_;
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21_spec__0(lean_object* v_msg_318_){
_start:
{
lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_319_ = l_Lean_instInhabitedExpr;
v___x_320_ = lean_panic_fn_borrowed(v___x_319_, v_msg_318_);
return v___x_320_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_324_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__2));
v___x_325_ = lean_unsigned_to_nat(19u);
v___x_326_ = lean_unsigned_to_nat(75u);
v___x_327_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__1));
v___x_328_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__0));
v___x_329_ = l_mkPanicMessageWithDecl(v___x_328_, v___x_327_, v___x_326_, v___x_325_, v___x_324_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21(lean_object* v_e_330_){
_start:
{
if (lean_obj_tag(v_e_330_) == 6)
{
lean_object* v_binderName_331_; lean_object* v_binderType_332_; lean_object* v_body_333_; uint8_t v___x_334_; lean_object* v___x_335_; 
v_binderName_331_ = lean_ctor_get(v_e_330_, 0);
lean_inc(v_binderName_331_);
v_binderType_332_ = lean_ctor_get(v_e_330_, 1);
lean_inc_ref(v_binderType_332_);
v_body_333_ = lean_ctor_get(v_e_330_, 2);
lean_inc_ref(v_body_333_);
lean_dec_ref_known(v_e_330_, 3);
v___x_334_ = 0;
v___x_335_ = l_Lean_Expr_lam___override(v_binderName_331_, v_binderType_332_, v_body_333_, v___x_334_);
return v___x_335_;
}
else
{
lean_object* v___x_336_; lean_object* v___x_337_; 
lean_dec_ref(v_e_330_);
v___x_336_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21___closed__3);
v___x_337_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21_spec__0(v___x_336_);
return v___x_337_;
}
}
}
static lean_object* _init_lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = lean_box(0);
v___x_345_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__3));
v___x_346_ = l_Lean_mkConst(v___x_345_, v___x_344_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0(lean_object* v_x_347_, lean_object* v_x_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_){
_start:
{
if (lean_obj_tag(v_x_348_) == 0)
{
lean_object* v___x_354_; 
v___x_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_354_, 0, v_x_347_);
return v___x_354_;
}
else
{
lean_object* v_head_355_; lean_object* v_tail_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_395_; 
v_head_355_ = lean_ctor_get(v_x_348_, 0);
v_tail_356_ = lean_ctor_get(v_x_348_, 1);
v_isSharedCheck_395_ = !lean_is_exclusive(v_x_348_);
if (v_isSharedCheck_395_ == 0)
{
v___x_358_ = v_x_348_;
v_isShared_359_ = v_isSharedCheck_395_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_tail_356_);
lean_inc(v_head_355_);
lean_dec(v_x_348_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_395_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___y_361_; lean_object* v___x_364_; 
lean_inc(v___y_352_);
lean_inc_ref(v___y_351_);
lean_inc(v___y_350_);
lean_inc_ref(v___y_349_);
lean_inc(v_head_355_);
v___x_364_ = lean_infer_type(v_head_355_, v___y_349_, v___y_350_, v___y_351_, v___y_352_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_object* v_a_365_; lean_object* v___x_366_; 
v_a_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc_n(v_a_365_, 2);
lean_dec_ref_known(v___x_364_, 1);
lean_inc(v___y_352_);
lean_inc_ref(v___y_351_);
lean_inc(v___y_350_);
lean_inc_ref(v___y_349_);
v___x_366_ = lean_infer_type(v_a_365_, v___y_349_, v___y_350_, v___y_351_, v___y_352_);
if (lean_obj_tag(v___x_366_) == 0)
{
lean_object* v_a_367_; lean_object* v___x_368_; uint8_t v___y_388_; uint8_t v___x_392_; 
v_a_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_a_367_);
lean_dec_ref_known(v___x_366_, 1);
v___x_368_ = l_Lean_Expr_sortLevel_x21(v_a_367_);
lean_dec(v_a_367_);
lean_inc(v_head_355_);
v___x_392_ = l_Lean_Expr_occurs(v_head_355_, v_x_347_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_393_ = lean_box(0);
v___x_394_ = lean_level_eq(v___x_368_, v___x_393_);
if (v___x_394_ == 0)
{
goto v___jp_369_;
}
else
{
v___y_388_ = v___x_392_;
goto v___jp_387_;
}
}
else
{
v___y_388_ = v___x_392_;
goto v___jp_387_;
}
v___jp_369_:
{
uint8_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; uint8_t v___x_374_; uint8_t v___x_375_; lean_object* v___x_376_; 
v___x_370_ = 1;
v___x_371_ = lean_unsigned_to_nat(1u);
v___x_372_ = lean_mk_empty_array_with_capacity(v___x_371_);
v___x_373_ = lean_array_push(v___x_372_, v_head_355_);
v___x_374_ = 0;
v___x_375_ = 1;
v___x_376_ = l_Lean_Meta_mkLambdaFVars(v___x_373_, v_x_347_, v___x_374_, v___x_370_, v___x_374_, v___x_370_, v___x_375_, v___y_349_, v___y_350_, v___y_351_, v___y_352_);
lean_dec_ref(v___x_373_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_381_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_a_377_);
lean_dec_ref_known(v___x_376_, 1);
v___x_378_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__1));
v___x_379_ = lean_box(0);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 1, v___x_379_);
lean_ctor_set(v___x_358_, 0, v___x_368_);
v___x_381_ = v___x_358_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v___x_368_);
lean_ctor_set(v_reuseFailAlloc_386_, 1, v___x_379_);
v___x_381_ = v_reuseFailAlloc_386_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_382_ = l_Lean_Expr_const___override(v___x_378_, v___x_381_);
v___x_383_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_updateLambdaBinderInfoD_x21(v_a_377_);
v___x_384_ = l_Lean_mkAppB(v___x_382_, v_a_365_, v___x_383_);
v_x_347_ = v___x_384_;
v_x_348_ = v_tail_356_;
goto _start;
}
}
else
{
lean_dec(v___x_368_);
lean_dec(v_a_365_);
lean_del_object(v___x_358_);
v___y_361_ = v___x_376_;
goto v___jp_360_;
}
}
v___jp_387_:
{
if (v___y_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; 
lean_dec(v___x_368_);
lean_del_object(v___x_358_);
lean_dec(v_head_355_);
v___x_389_ = lean_obj_once(&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4, &lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4_once, _init_lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4);
v___x_390_ = l_Lean_mkAppB(v___x_389_, v_a_365_, v_x_347_);
v_x_347_ = v___x_390_;
v_x_348_ = v_tail_356_;
goto _start;
}
else
{
goto v___jp_369_;
}
}
}
else
{
lean_dec(v_a_365_);
lean_del_object(v___x_358_);
lean_dec(v_head_355_);
lean_dec_ref(v_x_347_);
v___y_361_ = v___x_366_;
goto v___jp_360_;
}
}
else
{
lean_del_object(v___x_358_);
lean_dec(v_head_355_);
lean_dec_ref(v_x_347_);
v___y_361_ = v___x_364_;
goto v___jp_360_;
}
v___jp_360_:
{
if (lean_obj_tag(v___y_361_) == 0)
{
lean_object* v_a_362_; 
v_a_362_ = lean_ctor_get(v___y_361_, 0);
lean_inc(v_a_362_);
lean_dec_ref_known(v___y_361_, 1);
v_x_347_ = v_a_362_;
v_x_348_ = v_tail_356_;
goto _start;
}
else
{
lean_dec(v_tail_356_);
return v___y_361_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___boxed(lean_object* v_x_396_, lean_object* v_x_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0(v_x_396_, v_x_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(lean_object* v_args_404_, lean_object* v_inner_405_, lean_object* v_a_406_, lean_object* v_a_407_, lean_object* v_a_408_, lean_object* v_a_409_){
_start:
{
lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_411_ = l_List_reverse___redArg(v_args_404_);
v___x_412_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0(v_inner_405_, v___x_411_, v_a_406_, v_a_407_, v_a_408_, v_a_409_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList___boxed(lean_object* v_args_413_, lean_object* v_inner_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_, lean_object* v_a_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(v_args_413_, v_inner_414_, v_a_415_, v_a_416_, v_a_417_, v_a_418_);
lean_dec(v_a_418_);
lean_dec_ref(v_a_417_);
lean_dec(v_a_416_);
lean_dec_ref(v_a_415_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(lean_object* v_op_421_, lean_object* v_empty_422_, lean_object* v_x_423_){
_start:
{
if (lean_obj_tag(v_x_423_) == 0)
{
lean_dec_ref(v_op_421_);
lean_inc_ref(v_empty_422_);
return v_empty_422_;
}
else
{
lean_object* v_tail_424_; 
v_tail_424_ = lean_ctor_get(v_x_423_, 1);
if (lean_obj_tag(v_tail_424_) == 0)
{
lean_object* v_head_425_; 
lean_dec_ref(v_op_421_);
v_head_425_ = lean_ctor_get(v_x_423_, 0);
lean_inc(v_head_425_);
lean_dec_ref_known(v_x_423_, 2);
return v_head_425_;
}
else
{
lean_object* v_head_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
lean_inc(v_tail_424_);
v_head_426_ = lean_ctor_get(v_x_423_, 0);
lean_inc(v_head_426_);
lean_dec_ref_known(v_x_423_, 2);
lean_inc_ref(v_op_421_);
v___x_427_ = lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(v_op_421_, v_empty_422_, v_tail_424_);
v___x_428_ = l_Lean_mkAppB(v_op_421_, v_head_426_, v___x_427_);
return v___x_428_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOpList___boxed(lean_object* v_op_429_, lean_object* v_empty_430_, lean_object* v_x_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(v_op_429_, v_empty_430_, v_x_431_);
lean_dec_ref(v_empty_430_);
return v_res_432_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2(void){
_start:
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_436_ = lean_box(0);
v___x_437_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__1));
v___x_438_ = l_Lean_mkConst(v___x_437_, v___x_436_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkAndList(lean_object* v_a_439_){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_440_ = lean_obj_once(&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4, &lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4_once, _init_lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_mkExistsList_spec__0___closed__4);
v___x_441_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2, &lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__2);
v___x_442_ = lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(v___x_440_, v___x_441_, v_a_439_);
return v___x_442_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2(void){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_446_ = lean_box(0);
v___x_447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__1));
v___x_448_ = l_Lean_mkConst(v___x_447_, v___x_446_);
return v___x_448_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5(void){
_start:
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_452_ = lean_box(0);
v___x_453_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__4));
v___x_454_ = l_Lean_mkConst(v___x_453_, v___x_452_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkOrList(lean_object* v_a_455_){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_456_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2, &lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__2);
v___x_457_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5, &lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkOrList___closed__5);
v___x_458_ = lp_mathlib_Mathlib_Tactic_MkIff_mkOpList(v___x_456_, v___x_457_, v_a_455_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(lean_object* v_x_459_){
_start:
{
if (lean_obj_tag(v_x_459_) == 0)
{
return v_x_459_;
}
else
{
lean_object* v_tail_460_; 
v_tail_460_ = lean_ctor_get(v_x_459_, 1);
lean_inc(v_tail_460_);
if (lean_obj_tag(v_tail_460_) == 0)
{
lean_dec_ref_known(v_x_459_, 2);
return v_tail_460_;
}
else
{
lean_object* v_head_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_469_; 
v_head_461_ = lean_ctor_get(v_x_459_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v_x_459_);
if (v_isSharedCheck_469_ == 0)
{
lean_object* v_unused_470_; 
v_unused_470_ = lean_ctor_get(v_x_459_, 1);
lean_dec(v_unused_470_);
v___x_463_ = v_x_459_;
v_isShared_464_ = v_isSharedCheck_469_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_head_461_);
lean_dec(v_x_459_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_469_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_465_; lean_object* v___x_467_; 
v___x_465_ = lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(v_tail_460_);
if (v_isShared_464_ == 0)
{
lean_ctor_set(v___x_463_, 1, v___x_465_);
v___x_467_ = v___x_463_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v_head_461_);
lean_ctor_set(v_reuseFailAlloc_468_, 1, v___x_465_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_List_init(lean_object* v_00_u03b1_471_, lean_object* v_x_472_){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(v_x_472_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0(lean_object* v_k_474_, lean_object* v_b_475_, lean_object* v_c_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v___x_482_; 
lean_inc(v___y_480_);
lean_inc_ref(v___y_479_);
lean_inc(v___y_478_);
lean_inc_ref(v___y_477_);
v___x_482_ = lean_apply_7(v_k_474_, v_b_475_, v_c_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, lean_box(0));
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0___boxed(lean_object* v_k_483_, lean_object* v_b_484_, lean_object* v_c_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0(v_k_483_, v_b_484_, v_c_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_);
lean_dec(v___y_489_);
lean_dec_ref(v___y_488_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg(lean_object* v_type_492_, lean_object* v_maxFVars_x3f_493_, lean_object* v_k_494_, uint8_t v_cleanupAnnotations_495_, uint8_t v_whnfType_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_){
_start:
{
lean_object* v___f_502_; lean_object* v___x_503_; 
v___f_502_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_502_, 0, v_k_494_);
v___x_503_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_492_, v_maxFVars_x3f_493_, v___f_502_, v_cleanupAnnotations_495_, v_whnfType_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
if (lean_obj_tag(v___x_503_) == 0)
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
v_a_504_ = lean_ctor_get(v___x_503_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_503_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_503_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_503_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
if (v_isShared_507_ == 0)
{
v___x_509_ = v___x_506_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_a_504_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
else
{
lean_object* v_a_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_519_; 
v_a_512_ = lean_ctor_get(v___x_503_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_503_);
if (v_isSharedCheck_519_ == 0)
{
v___x_514_ = v___x_503_;
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_a_512_);
lean_dec(v___x_503_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_517_; 
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
return v___x_517_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___boxed(lean_object* v_type_520_, lean_object* v_maxFVars_x3f_521_, lean_object* v_k_522_, lean_object* v_cleanupAnnotations_523_, lean_object* v_whnfType_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_530_; uint8_t v_whnfType_boxed_531_; lean_object* v_res_532_; 
v_cleanupAnnotations_boxed_530_ = lean_unbox(v_cleanupAnnotations_523_);
v_whnfType_boxed_531_ = lean_unbox(v_whnfType_524_);
v_res_532_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg(v_type_520_, v_maxFVars_x3f_521_, v_k_522_, v_cleanupAnnotations_boxed_530_, v_whnfType_boxed_531_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4(lean_object* v_00_u03b1_533_, lean_object* v_type_534_, lean_object* v_maxFVars_x3f_535_, lean_object* v_k_536_, uint8_t v_cleanupAnnotations_537_, uint8_t v_whnfType_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg(v_type_534_, v_maxFVars_x3f_535_, v_k_536_, v_cleanupAnnotations_537_, v_whnfType_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___boxed(lean_object* v_00_u03b1_545_, lean_object* v_type_546_, lean_object* v_maxFVars_x3f_547_, lean_object* v_k_548_, lean_object* v_cleanupAnnotations_549_, lean_object* v_whnfType_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_556_; uint8_t v_whnfType_boxed_557_; lean_object* v_res_558_; 
v_cleanupAnnotations_boxed_556_ = lean_unbox(v_cleanupAnnotations_549_);
v_whnfType_boxed_557_ = lean_unbox(v_whnfType_550_);
v_res_558_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4(v_00_u03b1_545_, v_type_546_, v_maxFVars_x3f_547_, v_k_548_, v_cleanupAnnotations_boxed_556_, v_whnfType_boxed_557_, v___y_551_, v___y_552_, v___y_553_, v___y_554_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
lean_dec(v___y_552_);
lean_dec_ref(v___y_551_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(lean_object* v_type_559_, lean_object* v_k_560_, uint8_t v_cleanupAnnotations_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v___f_567_; uint8_t v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___f_567_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_567_, 0, v_k_560_);
v___x_568_ = 0;
v___x_569_ = lean_box(0);
v___x_570_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_568_, v___x_569_, v_type_559_, v___f_567_, v_cleanupAnnotations_561_, v___x_568_, v___y_562_, v___y_563_, v___y_564_, v___y_565_);
if (lean_obj_tag(v___x_570_) == 0)
{
lean_object* v_a_571_; lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_578_; 
v_a_571_ = lean_ctor_get(v___x_570_, 0);
v_isSharedCheck_578_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_578_ == 0)
{
v___x_573_ = v___x_570_;
v_isShared_574_ = v_isSharedCheck_578_;
goto v_resetjp_572_;
}
else
{
lean_inc(v_a_571_);
lean_dec(v___x_570_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_578_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_576_; 
if (v_isShared_574_ == 0)
{
v___x_576_ = v___x_573_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v_a_571_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
}
else
{
lean_object* v_a_579_; lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_586_; 
v_a_579_ = lean_ctor_get(v___x_570_, 0);
v_isSharedCheck_586_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_586_ == 0)
{
v___x_581_ = v___x_570_;
v_isShared_582_ = v_isSharedCheck_586_;
goto v_resetjp_580_;
}
else
{
lean_inc(v_a_579_);
lean_dec(v___x_570_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_586_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v___x_584_; 
if (v_isShared_582_ == 0)
{
v___x_584_ = v___x_581_;
goto v_reusejp_583_;
}
else
{
lean_object* v_reuseFailAlloc_585_; 
v_reuseFailAlloc_585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_585_, 0, v_a_579_);
v___x_584_ = v_reuseFailAlloc_585_;
goto v_reusejp_583_;
}
v_reusejp_583_:
{
return v___x_584_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg___boxed(lean_object* v_type_587_, lean_object* v_k_588_, lean_object* v_cleanupAnnotations_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_595_; lean_object* v_res_596_; 
v_cleanupAnnotations_boxed_595_ = lean_unbox(v_cleanupAnnotations_589_);
v_res_596_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(v_type_587_, v_k_588_, v_cleanupAnnotations_boxed_595_, v___y_590_, v___y_591_, v___y_592_, v___y_593_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
return v_res_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5(lean_object* v_00_u03b1_597_, lean_object* v_type_598_, lean_object* v_k_599_, uint8_t v_cleanupAnnotations_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(v_type_598_, v_k_599_, v_cleanupAnnotations_600_, v___y_601_, v___y_602_, v___y_603_, v___y_604_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___boxed(lean_object* v_00_u03b1_607_, lean_object* v_type_608_, lean_object* v_k_609_, lean_object* v_cleanupAnnotations_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_616_; lean_object* v_res_617_; 
v_cleanupAnnotations_boxed_616_ = lean_unbox(v_cleanupAnnotations_610_);
v_res_617_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5(v_00_u03b1_607_, v_type_608_, v_k_609_, v_cleanupAnnotations_boxed_616_, v___y_611_, v___y_612_, v___y_613_, v___y_614_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
lean_dec(v___y_612_);
lean_dec_ref(v___y_611_);
return v_res_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0(lean_object* v_params_618_, lean_object* v_fvars_619_, lean_object* v_ty_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_626_ = lean_array_mk(v_params_618_);
v___x_627_ = l_Lean_Expr_replaceFVars(v_ty_620_, v_fvars_619_, v___x_626_);
lean_dec_ref(v___x_626_);
v___x_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0___boxed(lean_object* v_params_629_, lean_object* v_fvars_630_, lean_object* v_ty_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0(v_params_629_, v_fvars_630_, v_ty_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
lean_dec(v___y_633_);
lean_dec_ref(v___y_632_);
lean_dec_ref(v_ty_631_);
lean_dec_ref(v_fvars_630_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2(lean_object* v_x_644_, lean_object* v_x_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_){
_start:
{
if (lean_obj_tag(v_x_644_) == 0)
{
lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_651_ = l_List_reverse___redArg(v_x_645_);
v___x_652_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
return v___x_652_;
}
else
{
lean_object* v_head_653_; lean_object* v_tail_654_; lean_object* v___x_656_; uint8_t v_isShared_657_; uint8_t v_isSharedCheck_714_; 
v_head_653_ = lean_ctor_get(v_x_644_, 0);
v_tail_654_ = lean_ctor_get(v_x_644_, 1);
v_isSharedCheck_714_ = !lean_is_exclusive(v_x_644_);
if (v_isSharedCheck_714_ == 0)
{
v___x_656_ = v_x_644_;
v_isShared_657_ = v_isSharedCheck_714_;
goto v_resetjp_655_;
}
else
{
lean_inc(v_tail_654_);
lean_inc(v_head_653_);
lean_dec(v_x_644_);
v___x_656_ = lean_box(0);
v_isShared_657_ = v_isSharedCheck_714_;
goto v_resetjp_655_;
}
v_resetjp_655_:
{
lean_object* v_a_659_; lean_object* v___y_665_; lean_object* v_fst_675_; lean_object* v_snd_676_; lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_713_; 
v_fst_675_ = lean_ctor_get(v_head_653_, 0);
v_snd_676_ = lean_ctor_get(v_head_653_, 1);
v_isSharedCheck_713_ = !lean_is_exclusive(v_head_653_);
if (v_isSharedCheck_713_ == 0)
{
v___x_678_ = v_head_653_;
v_isShared_679_ = v_isSharedCheck_713_;
goto v_resetjp_677_;
}
else
{
lean_inc(v_snd_676_);
lean_inc(v_fst_675_);
lean_dec(v_head_653_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_713_;
goto v_resetjp_677_;
}
v___jp_658_:
{
lean_object* v___x_661_; 
if (v_isShared_657_ == 0)
{
lean_ctor_set(v___x_656_, 1, v_x_645_);
lean_ctor_set(v___x_656_, 0, v_a_659_);
v___x_661_ = v___x_656_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v_a_659_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v_x_645_);
v___x_661_ = v_reuseFailAlloc_663_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
v_x_644_ = v_tail_654_;
v_x_645_ = v___x_661_;
goto _start;
}
}
v___jp_664_:
{
if (lean_obj_tag(v___y_665_) == 0)
{
lean_object* v_a_666_; 
v_a_666_ = lean_ctor_get(v___y_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___y_665_, 1);
v_a_659_ = v_a_666_;
goto v___jp_658_;
}
else
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_674_; 
lean_del_object(v___x_656_);
lean_dec(v_tail_654_);
lean_dec(v_x_645_);
v_a_667_ = lean_ctor_get(v___y_665_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v___y_665_);
if (v_isSharedCheck_674_ == 0)
{
v___x_669_ = v___y_665_;
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___y_665_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___x_672_; 
if (v_isShared_670_ == 0)
{
v___x_672_ = v___x_669_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_a_667_);
v___x_672_ = v_reuseFailAlloc_673_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
return v___x_672_;
}
}
}
}
v_resetjp_677_:
{
lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_680_ = l_Lean_Expr_fvarId_x21(v_fst_675_);
v___x_681_ = l_Lean_FVarId_getType___redArg(v___x_680_, v___y_646_, v___y_648_, v___y_649_);
if (lean_obj_tag(v___x_681_) == 0)
{
lean_object* v_a_682_; lean_object* v___x_683_; 
v_a_682_ = lean_ctor_get(v___x_681_, 0);
lean_inc(v_a_682_);
lean_dec_ref_known(v___x_681_, 1);
lean_inc(v___y_649_);
lean_inc_ref(v___y_648_);
lean_inc(v___y_647_);
lean_inc_ref(v___y_646_);
lean_inc(v_snd_676_);
v___x_683_ = lean_infer_type(v_snd_676_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
if (lean_obj_tag(v___x_683_) == 0)
{
lean_object* v_a_684_; lean_object* v___x_685_; 
v_a_684_ = lean_ctor_get(v___x_683_, 0);
lean_inc(v_a_684_);
lean_dec_ref_known(v___x_683_, 1);
lean_inc(v___y_649_);
lean_inc_ref(v___y_648_);
lean_inc(v___y_647_);
lean_inc_ref(v___y_646_);
lean_inc(v_a_682_);
v___x_685_ = lean_infer_type(v_a_682_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
if (lean_obj_tag(v___x_685_) == 0)
{
lean_object* v_a_686_; lean_object* v___x_687_; 
v_a_686_ = lean_ctor_get(v___x_685_, 0);
lean_inc(v_a_686_);
lean_dec_ref_known(v___x_685_, 1);
lean_inc(v_a_684_);
lean_inc(v_a_682_);
v___x_687_ = l_Lean_Meta_isExprDefEq(v_a_682_, v_a_684_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v_a_688_; lean_object* v___x_689_; uint8_t v___x_690_; 
v_a_688_ = lean_ctor_get(v___x_687_, 0);
lean_inc(v_a_688_);
lean_dec_ref_known(v___x_687_, 1);
v___x_689_ = l_Lean_Expr_sortLevel_x21(v_a_686_);
lean_dec(v_a_686_);
v___x_690_ = lean_unbox(v_a_688_);
lean_dec(v_a_688_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_694_; 
v___x_691_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__1));
v___x_692_ = lean_box(0);
if (v_isShared_679_ == 0)
{
lean_ctor_set_tag(v___x_678_, 1);
lean_ctor_set(v___x_678_, 1, v___x_692_);
lean_ctor_set(v___x_678_, 0, v___x_689_);
v___x_694_ = v___x_678_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_689_);
lean_ctor_set(v_reuseFailAlloc_697_, 1, v___x_692_);
v___x_694_ = v_reuseFailAlloc_697_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_695_ = l_Lean_Expr_const___override(v___x_691_, v___x_694_);
v___x_696_ = l_Lean_mkApp4(v___x_695_, v_a_682_, v_fst_675_, v_a_684_, v_snd_676_);
v_a_659_ = v___x_696_;
goto v___jp_658_;
}
}
else
{
lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_701_; 
lean_dec(v_a_684_);
v___x_698_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___closed__3));
v___x_699_ = lean_box(0);
if (v_isShared_679_ == 0)
{
lean_ctor_set_tag(v___x_678_, 1);
lean_ctor_set(v___x_678_, 1, v___x_699_);
lean_ctor_set(v___x_678_, 0, v___x_689_);
v___x_701_ = v___x_678_;
goto v_reusejp_700_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v___x_689_);
lean_ctor_set(v_reuseFailAlloc_704_, 1, v___x_699_);
v___x_701_ = v_reuseFailAlloc_704_;
goto v_reusejp_700_;
}
v_reusejp_700_:
{
lean_object* v___x_702_; lean_object* v___x_703_; 
v___x_702_ = l_Lean_Expr_const___override(v___x_698_, v___x_701_);
v___x_703_ = l_Lean_mkApp3(v___x_702_, v_a_682_, v_fst_675_, v_snd_676_);
v_a_659_ = v___x_703_;
goto v___jp_658_;
}
}
}
else
{
lean_object* v_a_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_712_; 
lean_dec(v_a_686_);
lean_dec(v_a_684_);
lean_dec(v_a_682_);
lean_del_object(v___x_678_);
lean_dec(v_snd_676_);
lean_dec(v_fst_675_);
lean_del_object(v___x_656_);
lean_dec(v_tail_654_);
lean_dec(v_x_645_);
v_a_705_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_712_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_712_ == 0)
{
v___x_707_ = v___x_687_;
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_a_705_);
lean_dec(v___x_687_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v___x_710_; 
if (v_isShared_708_ == 0)
{
v___x_710_ = v___x_707_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_711_; 
v_reuseFailAlloc_711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_711_, 0, v_a_705_);
v___x_710_ = v_reuseFailAlloc_711_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
return v___x_710_;
}
}
}
}
else
{
lean_dec(v_a_684_);
lean_dec(v_a_682_);
lean_del_object(v___x_678_);
lean_dec(v_snd_676_);
lean_dec(v_fst_675_);
v___y_665_ = v___x_685_;
goto v___jp_664_;
}
}
else
{
lean_dec(v_a_682_);
lean_del_object(v___x_678_);
lean_dec(v_snd_676_);
lean_dec(v_fst_675_);
v___y_665_ = v___x_683_;
goto v___jp_664_;
}
}
else
{
lean_del_object(v___x_678_);
lean_dec(v_snd_676_);
lean_dec(v_fst_675_);
v___y_665_ = v___x_681_;
goto v___jp_664_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2___boxed(lean_object* v_x_715_, lean_object* v_x_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2(v_x_715_, v_x_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
lean_dec(v___y_720_);
lean_dec_ref(v___y_719_);
lean_dec(v___y_718_);
lean_dec_ref(v___y_717_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__1(lean_object* v_a_723_, lean_object* v_a_724_){
_start:
{
if (lean_obj_tag(v_a_723_) == 0)
{
lean_object* v___x_725_; 
v___x_725_ = l_List_reverse___redArg(v_a_724_);
return v___x_725_;
}
else
{
lean_object* v_head_726_; lean_object* v_tail_727_; lean_object* v___x_729_; uint8_t v_isShared_730_; uint8_t v_isSharedCheck_740_; 
v_head_726_ = lean_ctor_get(v_a_723_, 0);
v_tail_727_ = lean_ctor_get(v_a_723_, 1);
v_isSharedCheck_740_ = !lean_is_exclusive(v_a_723_);
if (v_isSharedCheck_740_ == 0)
{
v___x_729_ = v_a_723_;
v_isShared_730_ = v_isSharedCheck_740_;
goto v_resetjp_728_;
}
else
{
lean_inc(v_tail_727_);
lean_inc(v_head_726_);
lean_dec(v_a_723_);
v___x_729_ = lean_box(0);
v_isShared_730_ = v_isSharedCheck_740_;
goto v_resetjp_728_;
}
v_resetjp_728_:
{
uint8_t v___y_732_; 
if (lean_obj_tag(v_head_726_) == 0)
{
uint8_t v___x_738_; 
v___x_738_ = 0;
v___y_732_ = v___x_738_;
goto v___jp_731_;
}
else
{
uint8_t v___x_739_; 
lean_dec_ref_known(v_head_726_, 1);
v___x_739_ = 1;
v___y_732_ = v___x_739_;
goto v___jp_731_;
}
v___jp_731_:
{
lean_object* v___x_733_; lean_object* v___x_735_; 
v___x_733_ = lean_box(v___y_732_);
if (v_isShared_730_ == 0)
{
lean_ctor_set(v___x_729_, 1, v_a_724_);
lean_ctor_set(v___x_729_, 0, v___x_733_);
v___x_735_ = v___x_729_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_737_; 
v_reuseFailAlloc_737_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_737_, 0, v___x_733_);
lean_ctor_set(v_reuseFailAlloc_737_, 1, v_a_724_);
v___x_735_ = v_reuseFailAlloc_737_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
v_a_723_ = v_tail_727_;
v_a_724_ = v___x_735_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_constrToProp_spec__3(lean_object* v_a_741_, lean_object* v_a_742_){
_start:
{
if (lean_obj_tag(v_a_741_) == 0)
{
lean_object* v___x_743_; 
v___x_743_ = lean_array_to_list(v_a_742_);
return v___x_743_;
}
else
{
lean_object* v_head_744_; 
v_head_744_ = lean_ctor_get(v_a_741_, 0);
if (lean_obj_tag(v_head_744_) == 0)
{
lean_object* v_tail_745_; 
v_tail_745_ = lean_ctor_get(v_a_741_, 1);
lean_inc(v_tail_745_);
lean_dec_ref_known(v_a_741_, 2);
v_a_741_ = v_tail_745_;
goto _start;
}
else
{
lean_object* v_tail_747_; lean_object* v_val_748_; lean_object* v___x_749_; 
lean_inc_ref(v_head_744_);
v_tail_747_ = lean_ctor_get(v_a_741_, 1);
lean_inc(v_tail_747_);
lean_dec_ref_known(v_a_741_, 2);
v_val_748_ = lean_ctor_get(v_head_744_, 0);
lean_inc(v_val_748_);
lean_dec_ref_known(v_head_744_, 1);
v___x_749_ = lean_array_push(v_a_742_, v_val_748_);
v_a_741_ = v_tail_747_;
v_a_742_ = v___x_749_;
goto _start;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0(void){
_start:
{
lean_object* v___x_751_; lean_object* v_dummy_752_; 
v___x_751_ = lean_box(0);
v_dummy_752_ = l_Lean_Expr_sort___override(v___x_751_);
return v_dummy_752_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3(void){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_757_ = lean_box(0);
v___x_758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkAndList___closed__1));
v___x_759_ = l_Lean_mkConst(v___x_758_, v___x_757_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1(lean_object* v___x_760_, lean_object* v_idxs_761_, lean_object* v___x_762_, lean_object* v_fvars_763_, lean_object* v_ty_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v___x_770_; lean_object* v_dummy_771_; lean_object* v_nargs_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v_fst_782_; lean_object* v_snd_783_; lean_object* v___x_785_; uint8_t v_isShared_786_; uint8_t v_isSharedCheck_884_; 
v___x_770_ = lean_box(0);
v_dummy_771_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__0);
v_nargs_772_ = l_Lean_Expr_getAppNumArgs(v_ty_764_);
lean_inc(v_nargs_772_);
v___x_773_ = lean_mk_array(v_nargs_772_, v_dummy_771_);
v___x_774_ = lean_unsigned_to_nat(1u);
v___x_775_ = lean_nat_sub(v_nargs_772_, v___x_774_);
lean_dec(v_nargs_772_);
v___x_776_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_ty_764_, v___x_773_, v___x_775_);
v___x_777_ = lean_array_to_list(v___x_776_);
v___x_778_ = l_List_drop___redArg(v___x_760_, v___x_777_);
lean_dec(v___x_777_);
v___x_779_ = lean_array_to_list(v_fvars_763_);
v___x_780_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_idxs_761_, v___x_778_);
v___x_781_ = lp_mathlib_Mathlib_Tactic_MkIff_compactRelation(v___x_779_, v___x_780_);
v_fst_782_ = lean_ctor_get(v___x_781_, 0);
v_snd_783_ = lean_ctor_get(v___x_781_, 1);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_884_ == 0)
{
v___x_785_ = v___x_781_;
v_isShared_786_ = v_isSharedCheck_884_;
goto v_resetjp_784_;
}
else
{
lean_inc(v_snd_783_);
lean_inc(v_fst_782_);
lean_dec(v___x_781_);
v___x_785_ = lean_box(0);
v_isShared_786_ = v_isSharedCheck_884_;
goto v_resetjp_784_;
}
v_resetjp_784_:
{
lean_object* v_fst_788_; lean_object* v_snd_789_; lean_object* v_fst_797_; lean_object* v_snd_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
v_fst_797_ = lean_ctor_get(v_snd_783_, 0);
lean_inc(v_fst_797_);
v_snd_798_ = lean_ctor_get(v_snd_783_, 1);
lean_inc(v_snd_798_);
lean_dec(v_snd_783_);
v___x_799_ = lean_box(0);
v___x_800_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__2(v_fst_797_, v___x_799_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_800_) == 0)
{
lean_object* v_a_801_; lean_object* v_bs_x27_803_; lean_object* v___y_804_; lean_object* v___y_805_; lean_object* v___y_806_; lean_object* v___y_807_; lean_object* v___x_822_; lean_object* v___x_823_; 
v_a_801_ = lean_ctor_get(v___x_800_, 0);
lean_inc(v_a_801_);
lean_dec_ref_known(v___x_800_, 1);
v___x_822_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1));
lean_inc(v_fst_782_);
v___x_823_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_constrToProp_spec__3(v_fst_782_, v___x_822_);
if (lean_obj_tag(v___x_823_) == 0)
{
if (lean_obj_tag(v_a_801_) == 0)
{
lean_object* v___x_824_; lean_object* v___x_825_; 
lean_dec(v_snd_798_);
v___x_824_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__2));
v___x_825_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3);
v_fst_788_ = v___x_824_;
v_snd_789_ = v___x_825_;
goto v___jp_787_;
}
else
{
v_bs_x27_803_ = v___x_823_;
v___y_804_ = v___y_765_;
v___y_805_ = v___y_766_;
v___y_806_ = v___y_767_;
v___y_807_ = v___y_768_;
goto v___jp_802_;
}
}
else
{
if (lean_obj_tag(v_a_801_) == 0)
{
lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; 
v___x_826_ = l_List_getLast_x21___redArg(v___x_762_, v___x_823_);
v___x_827_ = l_Lean_Expr_fvarId_x21(v___x_826_);
lean_dec(v___x_826_);
v___x_828_ = l_Lean_FVarId_getType___redArg(v___x_827_, v___y_765_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_828_) == 0)
{
lean_object* v_a_829_; lean_object* v___x_830_; 
v_a_829_ = lean_ctor_get(v___x_828_, 0);
lean_inc_n(v_a_829_, 2);
lean_dec_ref_known(v___x_828_, 1);
lean_inc(v___y_768_);
lean_inc_ref(v___y_767_);
lean_inc(v___y_766_);
lean_inc_ref(v___y_765_);
v___x_830_ = lean_infer_type(v_a_829_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_object* v_a_831_; lean_object* v___x_832_; uint8_t v___x_833_; 
v_a_831_ = lean_ctor_get(v___x_830_, 0);
lean_inc(v_a_831_);
lean_dec_ref_known(v___x_830_, 1);
v___x_832_ = l_Lean_Expr_sortLevel_x21(v_a_831_);
lean_dec(v_a_831_);
v___x_833_ = lean_level_eq(v___x_832_, v___x_770_);
lean_dec(v___x_832_);
if (v___x_833_ == 0)
{
lean_object* v___x_834_; lean_object* v___x_835_; 
lean_dec(v_a_829_);
v___x_834_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__3);
v___x_835_ = lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(v___x_823_, v___x_834_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_835_) == 0)
{
lean_object* v_a_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v_a_836_ = lean_ctor_get(v___x_835_, 0);
lean_inc(v_a_836_);
lean_dec_ref_known(v___x_835_, 1);
v___x_837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__2));
v___x_838_ = lean_apply_1(v_snd_798_, v_a_836_);
v_fst_788_ = v___x_837_;
v_snd_789_ = v___x_838_;
goto v___jp_787_;
}
else
{
lean_object* v_a_839_; lean_object* v___x_841_; uint8_t v_isShared_842_; uint8_t v_isSharedCheck_846_; 
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_839_ = lean_ctor_get(v___x_835_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v___x_835_);
if (v_isSharedCheck_846_ == 0)
{
v___x_841_ = v___x_835_;
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
else
{
lean_inc(v_a_839_);
lean_dec(v___x_835_);
v___x_841_ = lean_box(0);
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
v_resetjp_840_:
{
lean_object* v___x_844_; 
if (v_isShared_842_ == 0)
{
v___x_844_ = v___x_841_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_a_839_);
v___x_844_ = v_reuseFailAlloc_845_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
return v___x_844_;
}
}
}
}
else
{
lean_object* v___x_847_; lean_object* v___x_848_; 
v___x_847_ = lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(v___x_823_);
v___x_848_ = lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(v___x_847_, v_a_829_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v_a_849_; lean_object* v___x_850_; lean_object* v___x_851_; 
v_a_849_ = lean_ctor_get(v___x_848_, 0);
lean_inc(v_a_849_);
lean_dec_ref_known(v___x_848_, 1);
v___x_850_ = lean_box(0);
v___x_851_ = lean_apply_1(v_snd_798_, v_a_849_);
v_fst_788_ = v___x_850_;
v_snd_789_ = v___x_851_;
goto v___jp_787_;
}
else
{
lean_object* v_a_852_; lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_859_; 
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_852_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_859_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_859_ == 0)
{
v___x_854_ = v___x_848_;
v_isShared_855_ = v_isSharedCheck_859_;
goto v_resetjp_853_;
}
else
{
lean_inc(v_a_852_);
lean_dec(v___x_848_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_859_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v___x_857_; 
if (v_isShared_855_ == 0)
{
v___x_857_ = v___x_854_;
goto v_reusejp_856_;
}
else
{
lean_object* v_reuseFailAlloc_858_; 
v_reuseFailAlloc_858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_858_, 0, v_a_852_);
v___x_857_ = v_reuseFailAlloc_858_;
goto v_reusejp_856_;
}
v_reusejp_856_:
{
return v___x_857_;
}
}
}
}
}
else
{
lean_object* v_a_860_; lean_object* v___x_862_; uint8_t v_isShared_863_; uint8_t v_isSharedCheck_867_; 
lean_dec(v_a_829_);
lean_dec(v___x_823_);
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_860_ = lean_ctor_get(v___x_830_, 0);
v_isSharedCheck_867_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_867_ == 0)
{
v___x_862_ = v___x_830_;
v_isShared_863_ = v_isSharedCheck_867_;
goto v_resetjp_861_;
}
else
{
lean_inc(v_a_860_);
lean_dec(v___x_830_);
v___x_862_ = lean_box(0);
v_isShared_863_ = v_isSharedCheck_867_;
goto v_resetjp_861_;
}
v_resetjp_861_:
{
lean_object* v___x_865_; 
if (v_isShared_863_ == 0)
{
v___x_865_ = v___x_862_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v_a_860_);
v___x_865_ = v_reuseFailAlloc_866_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
return v___x_865_;
}
}
}
}
else
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_875_; 
lean_dec(v___x_823_);
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_868_ = lean_ctor_get(v___x_828_, 0);
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_875_ == 0)
{
v___x_870_ = v___x_828_;
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_828_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_873_; 
if (v_isShared_871_ == 0)
{
v___x_873_ = v___x_870_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v_a_868_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
else
{
v_bs_x27_803_ = v___x_823_;
v___y_804_ = v___y_765_;
v___y_805_ = v___y_766_;
v___y_806_ = v___y_767_;
v___y_807_ = v___y_768_;
goto v___jp_802_;
}
}
v___jp_802_:
{
lean_object* v___x_808_; lean_object* v___x_809_; 
lean_inc(v_a_801_);
v___x_808_ = lp_mathlib_Mathlib_Tactic_MkIff_mkAndList(v_a_801_);
v___x_809_ = lp_mathlib_Mathlib_Tactic_MkIff_mkExistsList(v_bs_x27_803_, v___x_808_, v___y_804_, v___y_805_, v___y_806_, v___y_807_);
if (lean_obj_tag(v___x_809_) == 0)
{
lean_object* v_a_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v_a_810_ = lean_ctor_get(v___x_809_, 0);
lean_inc(v_a_810_);
lean_dec_ref_known(v___x_809_, 1);
v___x_811_ = l_List_lengthTR___redArg(v_a_801_);
lean_dec(v_a_801_);
v___x_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_812_, 0, v___x_811_);
v___x_813_ = lean_apply_1(v_snd_798_, v_a_810_);
v_fst_788_ = v___x_812_;
v_snd_789_ = v___x_813_;
goto v___jp_787_;
}
else
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_821_; 
lean_dec(v_a_801_);
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_814_ = lean_ctor_get(v___x_809_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_809_);
if (v_isSharedCheck_821_ == 0)
{
v___x_816_ = v___x_809_;
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_809_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_819_; 
if (v_isShared_817_ == 0)
{
v___x_819_ = v___x_816_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_a_814_);
v___x_819_ = v_reuseFailAlloc_820_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
return v___x_819_;
}
}
}
}
}
else
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_883_; 
lean_dec(v_snd_798_);
lean_del_object(v___x_785_);
lean_dec(v_fst_782_);
v_a_876_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_883_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_883_ == 0)
{
v___x_878_ = v___x_800_;
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___x_800_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_881_; 
if (v_isShared_879_ == 0)
{
v___x_881_ = v___x_878_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_a_876_);
v___x_881_ = v_reuseFailAlloc_882_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
return v___x_881_;
}
}
}
v___jp_787_:
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_794_; 
v___x_790_ = lean_box(0);
v___x_791_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_constrToProp_spec__1(v_fst_782_, v___x_790_);
v___x_792_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_792_, 0, v___x_791_);
lean_ctor_set(v___x_792_, 1, v_fst_788_);
if (v_isShared_786_ == 0)
{
lean_ctor_set(v___x_785_, 1, v_snd_789_);
lean_ctor_set(v___x_785_, 0, v___x_792_);
v___x_794_ = v___x_785_;
goto v_reusejp_793_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v___x_792_);
lean_ctor_set(v_reuseFailAlloc_796_, 1, v_snd_789_);
v___x_794_ = v_reuseFailAlloc_796_;
goto v_reusejp_793_;
}
v_reusejp_793_:
{
lean_object* v___x_795_; 
v___x_795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_795_, 0, v___x_794_);
return v___x_795_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___boxed(lean_object* v___x_885_, lean_object* v_idxs_886_, lean_object* v___x_887_, lean_object* v_fvars_888_, lean_object* v_ty_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_){
_start:
{
lean_object* v_res_895_; 
v_res_895_ = lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1(v___x_885_, v_idxs_886_, v___x_887_, v_fvars_888_, v_ty_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_);
lean_dec(v___y_893_);
lean_dec_ref(v___y_892_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
lean_dec_ref(v___x_887_);
return v_res_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg(lean_object* v_ref_896_, lean_object* v_msg_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
lean_object* v_fileName_903_; lean_object* v_fileMap_904_; lean_object* v_options_905_; lean_object* v_currRecDepth_906_; lean_object* v_maxRecDepth_907_; lean_object* v_ref_908_; lean_object* v_currNamespace_909_; lean_object* v_openDecls_910_; lean_object* v_initHeartbeats_911_; lean_object* v_maxHeartbeats_912_; lean_object* v_quotContext_913_; lean_object* v_currMacroScope_914_; uint8_t v_diag_915_; lean_object* v_cancelTk_x3f_916_; uint8_t v_suppressElabErrors_917_; lean_object* v_inheritedTraceOptions_918_; lean_object* v_ref_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v_fileName_903_ = lean_ctor_get(v___y_900_, 0);
v_fileMap_904_ = lean_ctor_get(v___y_900_, 1);
v_options_905_ = lean_ctor_get(v___y_900_, 2);
v_currRecDepth_906_ = lean_ctor_get(v___y_900_, 3);
v_maxRecDepth_907_ = lean_ctor_get(v___y_900_, 4);
v_ref_908_ = lean_ctor_get(v___y_900_, 5);
v_currNamespace_909_ = lean_ctor_get(v___y_900_, 6);
v_openDecls_910_ = lean_ctor_get(v___y_900_, 7);
v_initHeartbeats_911_ = lean_ctor_get(v___y_900_, 8);
v_maxHeartbeats_912_ = lean_ctor_get(v___y_900_, 9);
v_quotContext_913_ = lean_ctor_get(v___y_900_, 10);
v_currMacroScope_914_ = lean_ctor_get(v___y_900_, 11);
v_diag_915_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*14);
v_cancelTk_x3f_916_ = lean_ctor_get(v___y_900_, 12);
v_suppressElabErrors_917_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_918_ = lean_ctor_get(v___y_900_, 13);
v_ref_919_ = l_Lean_replaceRef(v_ref_896_, v_ref_908_);
lean_inc_ref(v_inheritedTraceOptions_918_);
lean_inc(v_cancelTk_x3f_916_);
lean_inc(v_currMacroScope_914_);
lean_inc(v_quotContext_913_);
lean_inc(v_maxHeartbeats_912_);
lean_inc(v_initHeartbeats_911_);
lean_inc(v_openDecls_910_);
lean_inc(v_currNamespace_909_);
lean_inc(v_maxRecDepth_907_);
lean_inc(v_currRecDepth_906_);
lean_inc_ref(v_options_905_);
lean_inc_ref(v_fileMap_904_);
lean_inc_ref(v_fileName_903_);
v___x_920_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_920_, 0, v_fileName_903_);
lean_ctor_set(v___x_920_, 1, v_fileMap_904_);
lean_ctor_set(v___x_920_, 2, v_options_905_);
lean_ctor_set(v___x_920_, 3, v_currRecDepth_906_);
lean_ctor_set(v___x_920_, 4, v_maxRecDepth_907_);
lean_ctor_set(v___x_920_, 5, v_ref_919_);
lean_ctor_set(v___x_920_, 6, v_currNamespace_909_);
lean_ctor_set(v___x_920_, 7, v_openDecls_910_);
lean_ctor_set(v___x_920_, 8, v_initHeartbeats_911_);
lean_ctor_set(v___x_920_, 9, v_maxHeartbeats_912_);
lean_ctor_set(v___x_920_, 10, v_quotContext_913_);
lean_ctor_set(v___x_920_, 11, v_currMacroScope_914_);
lean_ctor_set(v___x_920_, 12, v_cancelTk_x3f_916_);
lean_ctor_set(v___x_920_, 13, v_inheritedTraceOptions_918_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*14, v_diag_915_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*14 + 1, v_suppressElabErrors_917_);
v___x_921_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v_msg_897_, v___y_898_, v___y_899_, v___x_920_, v___y_901_);
lean_dec_ref_known(v___x_920_, 14);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg___boxed(lean_object* v_ref_922_, lean_object* v_msg_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_){
_start:
{
lean_object* v_res_929_; 
v_res_929_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg(v_ref_922_, v_msg_923_, v___y_924_, v___y_925_, v___y_926_, v___y_927_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
lean_dec(v___y_925_);
lean_dec_ref(v___y_924_);
lean_dec(v_ref_922_);
return v_res_929_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0(void){
_start:
{
lean_object* v___x_930_; 
v___x_930_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_930_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1(void){
_start:
{
lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_931_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__0);
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v___x_931_);
return v___x_932_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2(void){
_start:
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v___x_933_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1);
v___x_934_ = lean_unsigned_to_nat(0u);
v___x_935_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_935_, 0, v___x_934_);
lean_ctor_set(v___x_935_, 1, v___x_934_);
lean_ctor_set(v___x_935_, 2, v___x_934_);
lean_ctor_set(v___x_935_, 3, v___x_934_);
lean_ctor_set(v___x_935_, 4, v___x_933_);
lean_ctor_set(v___x_935_, 5, v___x_933_);
lean_ctor_set(v___x_935_, 6, v___x_933_);
lean_ctor_set(v___x_935_, 7, v___x_933_);
lean_ctor_set(v___x_935_, 8, v___x_933_);
lean_ctor_set(v___x_935_, 9, v___x_933_);
return v___x_935_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3(void){
_start:
{
lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_936_ = lean_unsigned_to_nat(32u);
v___x_937_ = lean_mk_empty_array_with_capacity(v___x_936_);
v___x_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_938_, 0, v___x_937_);
return v___x_938_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4(void){
_start:
{
size_t v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; 
v___x_939_ = ((size_t)5ULL);
v___x_940_ = lean_unsigned_to_nat(0u);
v___x_941_ = lean_unsigned_to_nat(32u);
v___x_942_ = lean_mk_empty_array_with_capacity(v___x_941_);
v___x_943_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3);
v___x_944_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_944_, 0, v___x_943_);
lean_ctor_set(v___x_944_, 1, v___x_942_);
lean_ctor_set(v___x_944_, 2, v___x_940_);
lean_ctor_set(v___x_944_, 3, v___x_940_);
lean_ctor_set_usize(v___x_944_, 4, v___x_939_);
return v___x_944_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5(void){
_start:
{
lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_945_ = lean_box(1);
v___x_946_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4);
v___x_947_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__1);
v___x_948_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_948_, 0, v___x_947_);
lean_ctor_set(v___x_948_, 1, v___x_946_);
lean_ctor_set(v___x_948_, 2, v___x_945_);
return v___x_948_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7(void){
_start:
{
lean_object* v___x_950_; lean_object* v___x_951_; 
v___x_950_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__6));
v___x_951_ = l_Lean_stringToMessageData(v___x_950_);
return v___x_951_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9(void){
_start:
{
lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_953_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__8));
v___x_954_ = l_Lean_stringToMessageData(v___x_953_);
return v___x_954_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11(void){
_start:
{
lean_object* v___x_956_; lean_object* v___x_957_; 
v___x_956_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__10));
v___x_957_ = l_Lean_stringToMessageData(v___x_956_);
return v___x_957_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13(void){
_start:
{
lean_object* v___x_959_; lean_object* v___x_960_; 
v___x_959_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__12));
v___x_960_ = l_Lean_stringToMessageData(v___x_959_);
return v___x_960_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15(void){
_start:
{
lean_object* v___x_962_; lean_object* v___x_963_; 
v___x_962_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__14));
v___x_963_ = l_Lean_stringToMessageData(v___x_962_);
return v___x_963_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17(void){
_start:
{
lean_object* v___x_965_; lean_object* v___x_966_; 
v___x_965_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__16));
v___x_966_ = l_Lean_stringToMessageData(v___x_965_);
return v___x_966_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19(void){
_start:
{
lean_object* v___x_968_; lean_object* v___x_969_; 
v___x_968_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__18));
v___x_969_ = l_Lean_stringToMessageData(v___x_968_);
return v___x_969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg(lean_object* v_msg_970_, lean_object* v_declHint_971_, lean_object* v___y_972_){
_start:
{
lean_object* v___x_974_; lean_object* v_env_975_; uint8_t v___x_976_; 
v___x_974_ = lean_st_ref_get(v___y_972_);
v_env_975_ = lean_ctor_get(v___x_974_, 0);
lean_inc_ref(v_env_975_);
lean_dec(v___x_974_);
v___x_976_ = l_Lean_Name_isAnonymous(v_declHint_971_);
if (v___x_976_ == 0)
{
uint8_t v_isExporting_977_; 
v_isExporting_977_ = lean_ctor_get_uint8(v_env_975_, sizeof(void*)*8);
if (v_isExporting_977_ == 0)
{
lean_object* v___x_978_; 
lean_dec_ref(v_env_975_);
lean_dec(v_declHint_971_);
v___x_978_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_978_, 0, v_msg_970_);
return v___x_978_;
}
else
{
lean_object* v___x_979_; uint8_t v___x_980_; 
lean_inc_ref(v_env_975_);
v___x_979_ = l_Lean_Environment_setExporting(v_env_975_, v___x_976_);
lean_inc(v_declHint_971_);
lean_inc_ref(v___x_979_);
v___x_980_ = l_Lean_Environment_contains(v___x_979_, v_declHint_971_, v_isExporting_977_);
if (v___x_980_ == 0)
{
lean_object* v___x_981_; 
lean_dec_ref(v___x_979_);
lean_dec_ref(v_env_975_);
lean_dec(v_declHint_971_);
v___x_981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_981_, 0, v_msg_970_);
return v___x_981_;
}
else
{
lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v_c_987_; lean_object* v___x_988_; 
v___x_982_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2);
v___x_983_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5);
v___x_984_ = l_Lean_Options_empty;
v___x_985_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_985_, 0, v___x_979_);
lean_ctor_set(v___x_985_, 1, v___x_982_);
lean_ctor_set(v___x_985_, 2, v___x_983_);
lean_ctor_set(v___x_985_, 3, v___x_984_);
lean_inc(v_declHint_971_);
v___x_986_ = l_Lean_MessageData_ofConstName(v_declHint_971_, v___x_976_);
v_c_987_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_987_, 0, v___x_985_);
lean_ctor_set(v_c_987_, 1, v___x_986_);
v___x_988_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_975_, v_declHint_971_);
if (lean_obj_tag(v___x_988_) == 0)
{
lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; 
lean_dec_ref(v_env_975_);
lean_dec(v_declHint_971_);
v___x_989_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7);
v___x_990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_990_, 0, v___x_989_);
lean_ctor_set(v___x_990_, 1, v_c_987_);
v___x_991_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__9);
v___x_992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_992_, 0, v___x_990_);
lean_ctor_set(v___x_992_, 1, v___x_991_);
v___x_993_ = l_Lean_MessageData_note(v___x_992_);
v___x_994_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_994_, 0, v_msg_970_);
lean_ctor_set(v___x_994_, 1, v___x_993_);
v___x_995_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_995_, 0, v___x_994_);
return v___x_995_;
}
else
{
lean_object* v_val_996_; lean_object* v___x_998_; uint8_t v_isShared_999_; uint8_t v_isSharedCheck_1031_; 
v_val_996_ = lean_ctor_get(v___x_988_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v___x_988_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_998_ = v___x_988_;
v_isShared_999_ = v_isSharedCheck_1031_;
goto v_resetjp_997_;
}
else
{
lean_inc(v_val_996_);
lean_dec(v___x_988_);
v___x_998_ = lean_box(0);
v_isShared_999_ = v_isSharedCheck_1031_;
goto v_resetjp_997_;
}
v_resetjp_997_:
{
lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v_mod_1003_; uint8_t v___x_1004_; 
v___x_1000_ = lean_box(0);
v___x_1001_ = l_Lean_Environment_header(v_env_975_);
lean_dec_ref(v_env_975_);
v___x_1002_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1001_);
v_mod_1003_ = lean_array_get(v___x_1000_, v___x_1002_, v_val_996_);
lean_dec(v_val_996_);
lean_dec_ref(v___x_1002_);
v___x_1004_ = l_Lean_isPrivateName(v_declHint_971_);
lean_dec(v_declHint_971_);
if (v___x_1004_ == 0)
{
lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1016_; 
v___x_1005_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__11);
v___x_1006_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1006_, 0, v___x_1005_);
lean_ctor_set(v___x_1006_, 1, v_c_987_);
v___x_1007_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__13);
v___x_1008_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1008_, 0, v___x_1006_);
lean_ctor_set(v___x_1008_, 1, v___x_1007_);
v___x_1009_ = l_Lean_MessageData_ofName(v_mod_1003_);
v___x_1010_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1008_);
lean_ctor_set(v___x_1010_, 1, v___x_1009_);
v___x_1011_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__15);
v___x_1012_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1012_, 0, v___x_1010_);
lean_ctor_set(v___x_1012_, 1, v___x_1011_);
v___x_1013_ = l_Lean_MessageData_note(v___x_1012_);
v___x_1014_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1014_, 0, v_msg_970_);
lean_ctor_set(v___x_1014_, 1, v___x_1013_);
if (v_isShared_999_ == 0)
{
lean_ctor_set_tag(v___x_998_, 0);
lean_ctor_set(v___x_998_, 0, v___x_1014_);
v___x_1016_ = v___x_998_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v___x_1014_);
v___x_1016_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
return v___x_1016_;
}
}
else
{
lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1029_; 
v___x_1018_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__7);
v___x_1019_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1018_);
lean_ctor_set(v___x_1019_, 1, v_c_987_);
v___x_1020_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__17);
v___x_1021_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1021_, 0, v___x_1019_);
lean_ctor_set(v___x_1021_, 1, v___x_1020_);
v___x_1022_ = l_Lean_MessageData_ofName(v_mod_1003_);
v___x_1023_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1023_, 0, v___x_1021_);
lean_ctor_set(v___x_1023_, 1, v___x_1022_);
v___x_1024_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__19);
v___x_1025_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1023_);
lean_ctor_set(v___x_1025_, 1, v___x_1024_);
v___x_1026_ = l_Lean_MessageData_note(v___x_1025_);
v___x_1027_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1027_, 0, v_msg_970_);
lean_ctor_set(v___x_1027_, 1, v___x_1026_);
if (v_isShared_999_ == 0)
{
lean_ctor_set_tag(v___x_998_, 0);
lean_ctor_set(v___x_998_, 0, v___x_1027_);
v___x_1029_ = v___x_998_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v___x_1027_);
v___x_1029_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
return v___x_1029_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1032_; 
lean_dec_ref(v_env_975_);
lean_dec(v_declHint_971_);
v___x_1032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1032_, 0, v_msg_970_);
return v___x_1032_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___boxed(lean_object* v_msg_1033_, lean_object* v_declHint_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_){
_start:
{
lean_object* v_res_1037_; 
v_res_1037_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg(v_msg_1033_, v_declHint_1034_, v___y_1035_);
lean_dec(v___y_1035_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8(lean_object* v_msg_1038_, lean_object* v_declHint_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_){
_start:
{
lean_object* v___x_1045_; lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1055_; 
v___x_1045_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg(v_msg_1038_, v_declHint_1039_, v___y_1043_);
v_a_1046_ = lean_ctor_get(v___x_1045_, 0);
v_isSharedCheck_1055_ = !lean_is_exclusive(v___x_1045_);
if (v_isSharedCheck_1055_ == 0)
{
v___x_1048_ = v___x_1045_;
v_isShared_1049_ = v_isSharedCheck_1055_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_1045_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1055_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1053_; 
v___x_1050_ = l_Lean_unknownIdentifierMessageTag;
v___x_1051_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1051_, 0, v___x_1050_);
lean_ctor_set(v___x_1051_, 1, v_a_1046_);
if (v_isShared_1049_ == 0)
{
lean_ctor_set(v___x_1048_, 0, v___x_1051_);
v___x_1053_ = v___x_1048_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v___x_1051_);
v___x_1053_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
return v___x_1053_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8___boxed(lean_object* v_msg_1056_, lean_object* v_declHint_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8(v_msg_1056_, v_declHint_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg(lean_object* v_ref_1064_, lean_object* v_msg_1065_, lean_object* v_declHint_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_){
_start:
{
lean_object* v___x_1072_; lean_object* v_a_1073_; lean_object* v___x_1074_; 
v___x_1072_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8(v_msg_1065_, v_declHint_1066_, v___y_1067_, v___y_1068_, v___y_1069_, v___y_1070_);
v_a_1073_ = lean_ctor_get(v___x_1072_, 0);
lean_inc(v_a_1073_);
lean_dec_ref(v___x_1072_);
v___x_1074_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg(v_ref_1064_, v_a_1073_, v___y_1067_, v___y_1068_, v___y_1069_, v___y_1070_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg___boxed(lean_object* v_ref_1075_, lean_object* v_msg_1076_, lean_object* v_declHint_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg(v_ref_1075_, v_msg_1076_, v_declHint_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
lean_dec(v_ref_1075_);
return v_res_1083_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_1085_; lean_object* v___x_1086_; 
v___x_1085_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__0));
v___x_1086_ = l_Lean_stringToMessageData(v___x_1085_);
return v___x_1086_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_1088_; lean_object* v___x_1089_; 
v___x_1088_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__2));
v___x_1089_ = l_Lean_stringToMessageData(v___x_1088_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg(lean_object* v_ref_1090_, lean_object* v_constName_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_){
_start:
{
lean_object* v___x_1097_; uint8_t v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; 
v___x_1097_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__1);
v___x_1098_ = 0;
lean_inc(v_constName_1091_);
v___x_1099_ = l_Lean_MessageData_ofConstName(v_constName_1091_, v___x_1098_);
v___x_1100_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1097_);
lean_ctor_set(v___x_1100_, 1, v___x_1099_);
v___x_1101_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3);
v___x_1102_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1100_);
lean_ctor_set(v___x_1102_, 1, v___x_1101_);
v___x_1103_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg(v_ref_1090_, v___x_1102_, v_constName_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_);
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_ref_1104_, lean_object* v_constName_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
lean_object* v_res_1111_; 
v_res_1111_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg(v_ref_1104_, v_constName_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
lean_dec(v___y_1109_);
lean_dec_ref(v___y_1108_);
lean_dec(v___y_1107_);
lean_dec_ref(v___y_1106_);
lean_dec(v_ref_1104_);
return v_res_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(lean_object* v_constName_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_){
_start:
{
lean_object* v_ref_1118_; lean_object* v___x_1119_; 
v_ref_1118_ = lean_ctor_get(v___y_1115_, 5);
v___x_1119_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg(v_ref_1118_, v_constName_1112_, v___y_1113_, v___y_1114_, v___y_1115_, v___y_1116_);
return v___x_1119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg___boxed(lean_object* v_constName_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_){
_start:
{
lean_object* v_res_1126_; 
v_res_1126_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(v_constName_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
lean_dec(v___y_1124_);
lean_dec_ref(v___y_1123_);
lean_dec(v___y_1122_);
lean_dec_ref(v___y_1121_);
return v_res_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0(lean_object* v_constName_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_){
_start:
{
lean_object* v___x_1133_; lean_object* v_env_1134_; uint8_t v___x_1135_; lean_object* v___x_1136_; 
v___x_1133_ = lean_st_ref_get(v___y_1131_);
v_env_1134_ = lean_ctor_get(v___x_1133_, 0);
lean_inc_ref(v_env_1134_);
lean_dec(v___x_1133_);
v___x_1135_ = 0;
lean_inc(v_constName_1127_);
v___x_1136_ = l_Lean_Environment_find_x3f(v_env_1134_, v_constName_1127_, v___x_1135_);
if (lean_obj_tag(v___x_1136_) == 0)
{
lean_object* v___x_1137_; 
v___x_1137_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(v_constName_1127_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_);
return v___x_1137_;
}
else
{
lean_object* v_val_1138_; lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1145_; 
lean_dec(v_constName_1127_);
v_val_1138_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1145_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1145_ == 0)
{
v___x_1140_ = v___x_1136_;
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
else
{
lean_inc(v_val_1138_);
lean_dec(v___x_1136_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___x_1143_; 
if (v_isShared_1141_ == 0)
{
lean_ctor_set_tag(v___x_1140_, 0);
v___x_1143_ = v___x_1140_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v_val_1138_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0___boxed(lean_object* v_constName_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_){
_start:
{
lean_object* v_res_1152_; 
v_res_1152_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0(v_constName_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_);
lean_dec(v___y_1150_);
lean_dec_ref(v___y_1149_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
return v_res_1152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp(lean_object* v_univs_1153_, lean_object* v_params_1154_, lean_object* v_idxs_1155_, lean_object* v_c_1156_, lean_object* v_a_1157_, lean_object* v_a_1158_, lean_object* v_a_1159_, lean_object* v_a_1160_){
_start:
{
lean_object* v___x_1162_; 
v___x_1162_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0(v_c_1156_, v_a_1157_, v_a_1158_, v_a_1159_, v_a_1160_);
if (lean_obj_tag(v___x_1162_) == 0)
{
lean_object* v_a_1163_; lean_object* v___f_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; uint8_t v___x_1168_; lean_object* v___x_1169_; 
v_a_1163_ = lean_ctor_get(v___x_1162_, 0);
lean_inc(v_a_1163_);
lean_dec_ref_known(v___x_1162_, 1);
lean_inc(v_params_1154_);
v___f_1164_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1164_, 0, v_params_1154_);
v___x_1165_ = l_Lean_ConstantInfo_instantiateTypeLevelParams(v_a_1163_, v_univs_1153_);
lean_dec(v_a_1163_);
v___x_1166_ = l_List_lengthTR___redArg(v_params_1154_);
lean_dec(v_params_1154_);
lean_inc(v___x_1166_);
v___x_1167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1166_);
v___x_1168_ = 0;
v___x_1169_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__4___redArg(v___x_1165_, v___x_1167_, v___f_1164_, v___x_1168_, v___x_1168_, v_a_1157_, v_a_1158_, v_a_1159_, v_a_1160_);
if (lean_obj_tag(v___x_1169_) == 0)
{
lean_object* v_a_1170_; lean_object* v___x_1171_; lean_object* v___f_1172_; lean_object* v___x_1173_; 
v_a_1170_ = lean_ctor_get(v___x_1169_, 0);
lean_inc(v_a_1170_);
lean_dec_ref_known(v___x_1169_, 1);
v___x_1171_ = l_Lean_instInhabitedExpr;
v___f_1172_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___boxed), 10, 3);
lean_closure_set(v___f_1172_, 0, v___x_1166_);
lean_closure_set(v___f_1172_, 1, v_idxs_1155_);
lean_closure_set(v___f_1172_, 2, v___x_1171_);
v___x_1173_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(v_a_1170_, v___f_1172_, v___x_1168_, v_a_1157_, v_a_1158_, v_a_1159_, v_a_1160_);
return v___x_1173_;
}
else
{
lean_object* v_a_1174_; lean_object* v___x_1176_; uint8_t v_isShared_1177_; uint8_t v_isSharedCheck_1181_; 
lean_dec(v___x_1166_);
lean_dec(v_idxs_1155_);
v_a_1174_ = lean_ctor_get(v___x_1169_, 0);
v_isSharedCheck_1181_ = !lean_is_exclusive(v___x_1169_);
if (v_isSharedCheck_1181_ == 0)
{
v___x_1176_ = v___x_1169_;
v_isShared_1177_ = v_isSharedCheck_1181_;
goto v_resetjp_1175_;
}
else
{
lean_inc(v_a_1174_);
lean_dec(v___x_1169_);
v___x_1176_ = lean_box(0);
v_isShared_1177_ = v_isSharedCheck_1181_;
goto v_resetjp_1175_;
}
v_resetjp_1175_:
{
lean_object* v___x_1179_; 
if (v_isShared_1177_ == 0)
{
v___x_1179_ = v___x_1176_;
goto v_reusejp_1178_;
}
else
{
lean_object* v_reuseFailAlloc_1180_; 
v_reuseFailAlloc_1180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1180_, 0, v_a_1174_);
v___x_1179_ = v_reuseFailAlloc_1180_;
goto v_reusejp_1178_;
}
v_reusejp_1178_:
{
return v___x_1179_;
}
}
}
}
else
{
lean_object* v_a_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1189_; 
lean_dec(v_idxs_1155_);
lean_dec(v_params_1154_);
lean_dec(v_univs_1153_);
v_a_1182_ = lean_ctor_get(v___x_1162_, 0);
v_isSharedCheck_1189_ = !lean_is_exclusive(v___x_1162_);
if (v_isSharedCheck_1189_ == 0)
{
v___x_1184_ = v___x_1162_;
v_isShared_1185_ = v_isSharedCheck_1189_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_a_1182_);
lean_dec(v___x_1162_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1189_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v___x_1187_; 
if (v_isShared_1185_ == 0)
{
v___x_1187_ = v___x_1184_;
goto v_reusejp_1186_;
}
else
{
lean_object* v_reuseFailAlloc_1188_; 
v_reuseFailAlloc_1188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1188_, 0, v_a_1182_);
v___x_1187_ = v_reuseFailAlloc_1188_;
goto v_reusejp_1186_;
}
v_reusejp_1186_:
{
return v___x_1187_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___boxed(lean_object* v_univs_1190_, lean_object* v_params_1191_, lean_object* v_idxs_1192_, lean_object* v_c_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_){
_start:
{
lean_object* v_res_1199_; 
v_res_1199_ = lp_mathlib_Mathlib_Tactic_MkIff_constrToProp(v_univs_1190_, v_params_1191_, v_idxs_1192_, v_c_1193_, v_a_1194_, v_a_1195_, v_a_1196_, v_a_1197_);
lean_dec(v_a_1197_);
lean_dec_ref(v_a_1196_);
lean_dec(v_a_1195_);
lean_dec_ref(v_a_1194_);
return v_res_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0(lean_object* v_00_u03b1_1200_, lean_object* v_constName_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_){
_start:
{
lean_object* v___x_1207_; 
v___x_1207_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(v_constName_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
return v___x_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1208_, lean_object* v_constName_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v_res_1215_; 
v_res_1215_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0(v_00_u03b1_1208_, v_constName_1209_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
return v_res_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3(lean_object* v_00_u03b1_1216_, lean_object* v_ref_1217_, lean_object* v_constName_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_){
_start:
{
lean_object* v___x_1224_; 
v___x_1224_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg(v_ref_1217_, v_constName_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_);
return v___x_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b1_1225_, lean_object* v_ref_1226_, lean_object* v_constName_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3(v_00_u03b1_1225_, v_ref_1226_, v_constName_1227_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
lean_dec(v_ref_1226_);
return v_res_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7(lean_object* v_00_u03b1_1234_, lean_object* v_ref_1235_, lean_object* v_msg_1236_, lean_object* v_declHint_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v___x_1243_; 
v___x_1243_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___redArg(v_ref_1235_, v_msg_1236_, v_declHint_1237_, v___y_1238_, v___y_1239_, v___y_1240_, v___y_1241_);
return v___x_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7___boxed(lean_object* v_00_u03b1_1244_, lean_object* v_ref_1245_, lean_object* v_msg_1246_, lean_object* v_declHint_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_){
_start:
{
lean_object* v_res_1253_; 
v_res_1253_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7(v_00_u03b1_1244_, v_ref_1245_, v_msg_1246_, v_declHint_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_);
lean_dec(v___y_1251_);
lean_dec_ref(v___y_1250_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
lean_dec(v_ref_1245_);
return v_res_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9(lean_object* v_msg_1254_, lean_object* v_declHint_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_){
_start:
{
lean_object* v___x_1261_; 
v___x_1261_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg(v_msg_1254_, v_declHint_1255_, v___y_1259_);
return v___x_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___boxed(lean_object* v_msg_1262_, lean_object* v_declHint_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_){
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9(v_msg_1262_, v_declHint_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
return v_res_1269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9(lean_object* v_00_u03b1_1270_, lean_object* v_ref_1271_, lean_object* v_msg_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_){
_start:
{
lean_object* v___x_1278_; 
v___x_1278_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___redArg(v_ref_1271_, v_msg_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9___boxed(lean_object* v_00_u03b1_1279_, lean_object* v_ref_1280_, lean_object* v_msg_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_){
_start:
{
lean_object* v_res_1287_; 
v_res_1287_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__9(v_00_u03b1_1279_, v_ref_1280_, v_msg_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v_ref_1280_);
return v_res_1287_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0(lean_object* v_x_1288_){
_start:
{
uint8_t v___x_1289_; 
v___x_1289_ = 0;
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0___boxed(lean_object* v_x_1290_){
_start:
{
uint8_t v_res_1291_; lean_object* v_r_1292_; 
v_res_1291_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__0(v_x_1290_);
lean_dec(v_x_1290_);
v_r_1292_ = lean_box(v_res_1291_);
return v_r_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1(lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_){
_start:
{
lean_object* v_ref_1311_; uint8_t v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v_ref_1311_ = lean_ctor_get(v___y_1308_, 5);
v___x_1312_ = 0;
v___x_1313_ = l_Lean_SourceInfo_fromRef(v_ref_1311_, v___x_1312_);
v___x_1314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3));
v___x_1315_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4));
lean_inc(v___x_1313_);
v___x_1316_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1313_);
lean_ctor_set(v___x_1316_, 1, v___x_1314_);
v___x_1317_ = l_Lean_Syntax_node1(v___x_1313_, v___x_1315_, v___x_1316_);
v___x_1318_ = l_Lean_Elab_Tactic_evalTactic(v___x_1317_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_);
return v___x_1318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___boxed(lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_){
_start:
{
lean_object* v_res_1328_; 
v_res_1328_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1(v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_, v___y_1324_, v___y_1325_, v___y_1326_);
lean_dec(v___y_1326_);
lean_dec_ref(v___y_1325_);
lean_dec(v___y_1324_);
lean_dec_ref(v___y_1323_);
lean_dec(v___y_1322_);
lean_dec_ref(v___y_1321_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
return v_res_1328_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2(uint8_t v_isZero_1329_, lean_object* v_x_1330_){
_start:
{
return v_isZero_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2___boxed(lean_object* v_isZero_1331_, lean_object* v_x_1332_){
_start:
{
uint8_t v_isZero_boxed_1333_; uint8_t v_res_1334_; lean_object* v_r_1335_; 
v_isZero_boxed_1333_ = lean_unbox(v_isZero_1331_);
v_res_1334_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2(v_isZero_boxed_1333_, v_x_1332_);
lean_dec(v_x_1332_);
v_r_1335_ = lean_box(v_res_1334_);
return v_r_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3(uint8_t v_isZero_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_){
_start:
{
lean_object* v_ref_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; 
v_ref_1373_ = lean_ctor_get(v___y_1370_, 5);
v___x_1374_ = l_Lean_SourceInfo_fromRef(v_ref_1373_, v_isZero_1363_);
v___x_1375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__0));
v___x_1376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__1));
lean_inc_n(v___x_1374_, 9);
v___x_1377_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1377_, 0, v___x_1374_);
lean_ctor_set(v___x_1377_, 1, v___x_1375_);
v___x_1378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__4));
v___x_1379_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__5));
v___x_1380_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1380_, 0, v___x_1374_);
lean_ctor_set(v___x_1380_, 1, v___x_1379_);
v___x_1381_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__7));
v___x_1382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__9));
v___x_1383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__10));
v___x_1384_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1384_, 0, v___x_1374_);
lean_ctor_set(v___x_1384_, 1, v___x_1383_);
v___x_1385_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__11));
v___x_1386_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1386_, 0, v___x_1374_);
lean_ctor_set(v___x_1386_, 1, v___x_1385_);
v___x_1387_ = l_Lean_Syntax_node2(v___x_1374_, v___x_1382_, v___x_1384_, v___x_1386_);
v___x_1388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__12));
v___x_1389_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1389_, 0, v___x_1374_);
lean_ctor_set(v___x_1389_, 1, v___x_1388_);
lean_inc(v___x_1387_);
v___x_1390_ = l_Lean_Syntax_node3(v___x_1374_, v___x_1381_, v___x_1387_, v___x_1389_, v___x_1387_);
v___x_1391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___closed__13));
v___x_1392_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1392_, 0, v___x_1374_);
lean_ctor_set(v___x_1392_, 1, v___x_1391_);
v___x_1393_ = l_Lean_Syntax_node3(v___x_1374_, v___x_1378_, v___x_1380_, v___x_1390_, v___x_1392_);
v___x_1394_ = l_Lean_Syntax_node2(v___x_1374_, v___x_1376_, v___x_1377_, v___x_1393_);
v___x_1395_ = l_Lean_Elab_Tactic_evalTactic(v___x_1394_, v___y_1364_, v___y_1365_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___boxed(lean_object* v_isZero_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_){
_start:
{
uint8_t v_isZero_boxed_1406_; lean_object* v_res_1407_; 
v_isZero_boxed_1406_ = lean_unbox(v_isZero_1396_);
v_res_1407_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3(v_isZero_boxed_1406_, v___y_1397_, v___y_1398_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_);
lean_dec(v___y_1404_);
lean_dec_ref(v___y_1403_);
lean_dec(v___y_1402_);
lean_dec_ref(v___y_1401_);
lean_dec(v___y_1400_);
lean_dec_ref(v___y_1399_);
lean_dec(v___y_1398_);
lean_dec_ref(v___y_1397_);
return v_res_1407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4(uint8_t v_isZero_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_){
_start:
{
lean_object* v_ref_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; 
v_ref_1418_ = lean_ctor_get(v___y_1415_, 5);
v___x_1419_ = l_Lean_SourceInfo_fromRef(v_ref_1418_, v_isZero_1408_);
v___x_1420_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__3));
v___x_1421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__1___closed__4));
lean_inc(v___x_1419_);
v___x_1422_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1422_, 0, v___x_1419_);
lean_ctor_set(v___x_1422_, 1, v___x_1420_);
v___x_1423_ = l_Lean_Syntax_node1(v___x_1419_, v___x_1421_, v___x_1422_);
v___x_1424_ = l_Lean_Elab_Tactic_evalTactic(v___x_1423_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_, v___y_1415_, v___y_1416_);
return v___x_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4___boxed(lean_object* v_isZero_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_){
_start:
{
uint8_t v_isZero_boxed_1435_; lean_object* v_res_1436_; 
v_isZero_boxed_1435_ = lean_unbox(v_isZero_1425_);
v_res_1436_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4(v_isZero_boxed_1435_, v___y_1426_, v___y_1427_, v___y_1428_, v___y_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec(v___y_1431_);
lean_dec_ref(v___y_1430_);
lean_dec(v___y_1429_);
lean_dec_ref(v___y_1428_);
lean_dec(v___y_1427_);
lean_dec_ref(v___y_1426_);
return v_res_1436_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1(void){
_start:
{
lean_object* v___x_1438_; lean_object* v___x_1439_; 
v___x_1438_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__0));
v___x_1439_ = l_Lean_stringToMessageData(v___x_1438_);
return v___x_1439_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7(void){
_start:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; 
v___x_1448_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__6));
v___x_1449_ = l_Lean_stringToMessageData(v___x_1448_);
return v___x_1449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor(lean_object* v_mvar_1450_, lean_object* v_n_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_, lean_object* v_a_1455_){
_start:
{
lean_object* v___y_1458_; lean_object* v___y_1459_; lean_object* v___y_1460_; lean_object* v___y_1461_; lean_object* v_zero_1464_; uint8_t v_isZero_1465_; 
v_zero_1464_ = lean_unsigned_to_nat(0u);
v_isZero_1465_ = lean_nat_dec_eq(v_n_1451_, v_zero_1464_);
if (v_isZero_1465_ == 1)
{
lean_object* v___f_1466_; lean_object* v___f_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; uint8_t v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; 
lean_dec(v_n_1451_);
v___f_1466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__2));
v___f_1467_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__3));
v___x_1468_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_1468_, 0, v_mvar_1450_);
lean_closure_set(v___x_1468_, 1, v___f_1467_);
v___x_1469_ = lean_box(0);
v___x_1470_ = lean_box(0);
v___x_1471_ = lean_box(1);
v___x_1472_ = 0;
v___x_1473_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__4));
v___x_1474_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1474_, 0, v___x_1469_);
lean_ctor_set(v___x_1474_, 1, v___x_1470_);
lean_ctor_set(v___x_1474_, 2, v___x_1469_);
lean_ctor_set(v___x_1474_, 3, v___f_1466_);
lean_ctor_set(v___x_1474_, 4, v___x_1471_);
lean_ctor_set(v___x_1474_, 5, v___x_1471_);
lean_ctor_set(v___x_1474_, 6, v___x_1469_);
lean_ctor_set(v___x_1474_, 7, v___x_1473_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 1, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 2, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 3, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 4, v___x_1472_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 5, v___x_1472_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 6, v___x_1472_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 7, v___x_1472_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 8, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 9, v___x_1472_);
lean_ctor_set_uint8(v___x_1474_, sizeof(void*)*8 + 10, v_isZero_1465_);
v___x_1475_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__5));
v___x_1476_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_1468_, v___x_1474_, v___x_1475_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_);
if (lean_obj_tag(v___x_1476_) == 0)
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1488_; 
v_a_1477_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1488_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1488_ == 0)
{
v___x_1479_ = v___x_1476_;
v_isShared_1480_ = v_isSharedCheck_1488_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1476_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1488_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v_fst_1481_; 
v_fst_1481_ = lean_ctor_get(v_a_1477_, 0);
lean_inc(v_fst_1481_);
lean_dec(v_a_1477_);
if (lean_obj_tag(v_fst_1481_) == 0)
{
lean_object* v___x_1482_; lean_object* v___x_1484_; 
v___x_1482_ = lean_box(0);
if (v_isShared_1480_ == 0)
{
lean_ctor_set(v___x_1479_, 0, v___x_1482_);
v___x_1484_ = v___x_1479_;
goto v_reusejp_1483_;
}
else
{
lean_object* v_reuseFailAlloc_1485_; 
v_reuseFailAlloc_1485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1485_, 0, v___x_1482_);
v___x_1484_ = v_reuseFailAlloc_1485_;
goto v_reusejp_1483_;
}
v_reusejp_1483_:
{
return v___x_1484_;
}
}
else
{
lean_object* v___x_1486_; lean_object* v___x_1487_; 
lean_dec(v_fst_1481_);
lean_del_object(v___x_1479_);
v___x_1486_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7, &lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7);
v___x_1487_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_1486_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_);
return v___x_1487_;
}
}
}
else
{
lean_object* v_a_1489_; lean_object* v___x_1491_; uint8_t v_isShared_1492_; uint8_t v_isSharedCheck_1496_; 
v_a_1489_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1496_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1491_ = v___x_1476_;
v_isShared_1492_ = v_isSharedCheck_1496_;
goto v_resetjp_1490_;
}
else
{
lean_inc(v_a_1489_);
lean_dec(v___x_1476_);
v___x_1491_ = lean_box(0);
v_isShared_1492_ = v_isSharedCheck_1496_;
goto v_resetjp_1490_;
}
v_resetjp_1490_:
{
lean_object* v___x_1494_; 
if (v_isShared_1492_ == 0)
{
v___x_1494_ = v___x_1491_;
goto v_reusejp_1493_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v_a_1489_);
v___x_1494_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1493_;
}
v_reusejp_1493_:
{
return v___x_1494_;
}
}
}
}
else
{
lean_object* v___x_1497_; lean_object* v___f_1498_; lean_object* v___x_1499_; lean_object* v___f_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; uint8_t v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; 
v___x_1497_ = lean_box(v_isZero_1465_);
v___f_1498_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__2___boxed), 2, 1);
lean_closure_set(v___f_1498_, 0, v___x_1497_);
v___x_1499_ = lean_box(v_isZero_1465_);
v___f_1500_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__3___boxed), 10, 1);
lean_closure_set(v___f_1500_, 0, v___x_1499_);
v___x_1501_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_1501_, 0, v_mvar_1450_);
lean_closure_set(v___x_1501_, 1, v___f_1500_);
v___x_1502_ = lean_box(0);
v___x_1503_ = lean_box(0);
v___x_1504_ = 1;
v___x_1505_ = lean_box(1);
v___x_1506_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__4));
v___x_1507_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1507_, 0, v___x_1502_);
lean_ctor_set(v___x_1507_, 1, v___x_1503_);
lean_ctor_set(v___x_1507_, 2, v___x_1502_);
lean_ctor_set(v___x_1507_, 3, v___f_1498_);
lean_ctor_set(v___x_1507_, 4, v___x_1505_);
lean_ctor_set(v___x_1507_, 5, v___x_1505_);
lean_ctor_set(v___x_1507_, 6, v___x_1502_);
lean_ctor_set(v___x_1507_, 7, v___x_1506_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8, v___x_1504_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 1, v___x_1504_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 2, v___x_1504_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 3, v___x_1504_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 4, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 5, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 6, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 7, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 8, v___x_1504_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 9, v_isZero_1465_);
lean_ctor_set_uint8(v___x_1507_, sizeof(void*)*8 + 10, v___x_1504_);
v___x_1508_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__5));
lean_inc_ref(v___x_1507_);
v___x_1509_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_1501_, v___x_1507_, v___x_1508_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_);
if (lean_obj_tag(v___x_1509_) == 0)
{
lean_object* v_a_1510_; lean_object* v_fst_1511_; 
v_a_1510_ = lean_ctor_get(v___x_1509_, 0);
lean_inc(v_a_1510_);
lean_dec_ref_known(v___x_1509_, 1);
v_fst_1511_ = lean_ctor_get(v_a_1510_, 0);
lean_inc(v_fst_1511_);
lean_dec(v_a_1510_);
if (lean_obj_tag(v_fst_1511_) == 1)
{
lean_object* v_tail_1512_; 
v_tail_1512_ = lean_ctor_get(v_fst_1511_, 1);
lean_inc(v_tail_1512_);
if (lean_obj_tag(v_tail_1512_) == 1)
{
lean_object* v_tail_1513_; 
v_tail_1513_ = lean_ctor_get(v_tail_1512_, 1);
if (lean_obj_tag(v_tail_1513_) == 0)
{
lean_object* v_head_1514_; lean_object* v_head_1515_; lean_object* v___x_1516_; lean_object* v___f_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v_head_1514_ = lean_ctor_get(v_fst_1511_, 0);
lean_inc(v_head_1514_);
lean_dec_ref_known(v_fst_1511_, 2);
v_head_1515_ = lean_ctor_get(v_tail_1512_, 0);
lean_inc(v_head_1515_);
lean_dec_ref_known(v_tail_1512_, 2);
v___x_1516_ = lean_box(v_isZero_1465_);
v___f_1517_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___lam__4___boxed), 10, 1);
lean_closure_set(v___f_1517_, 0, v___x_1516_);
v___x_1518_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_1518_, 0, v_head_1514_);
lean_closure_set(v___x_1518_, 1, v___f_1517_);
v___x_1519_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_1518_, v___x_1507_, v___x_1508_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_);
if (lean_obj_tag(v___x_1519_) == 0)
{
lean_object* v_a_1520_; lean_object* v_fst_1521_; 
v_a_1520_ = lean_ctor_get(v___x_1519_, 0);
lean_inc(v_a_1520_);
lean_dec_ref_known(v___x_1519_, 1);
v_fst_1521_ = lean_ctor_get(v_a_1520_, 0);
lean_inc(v_fst_1521_);
lean_dec(v_a_1520_);
if (lean_obj_tag(v_fst_1521_) == 0)
{
lean_object* v_one_1522_; lean_object* v_n_1523_; 
v_one_1522_ = lean_unsigned_to_nat(1u);
v_n_1523_ = lean_nat_sub(v_n_1451_, v_one_1522_);
lean_dec(v_n_1451_);
v_mvar_1450_ = v_head_1515_;
v_n_1451_ = v_n_1523_;
goto _start;
}
else
{
lean_object* v___x_1525_; lean_object* v___x_1526_; 
lean_dec(v_fst_1521_);
lean_dec(v_head_1515_);
lean_dec(v_n_1451_);
v___x_1525_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7, &lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__7);
v___x_1526_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_1525_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_);
return v___x_1526_;
}
}
else
{
lean_object* v_a_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1534_; 
lean_dec(v_head_1515_);
lean_dec(v_n_1451_);
v_a_1527_ = lean_ctor_get(v___x_1519_, 0);
v_isSharedCheck_1534_ = !lean_is_exclusive(v___x_1519_);
if (v_isSharedCheck_1534_ == 0)
{
v___x_1529_ = v___x_1519_;
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_a_1527_);
lean_dec(v___x_1519_);
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
lean_dec_ref_known(v_tail_1512_, 2);
lean_dec_ref_known(v_fst_1511_, 2);
lean_dec_ref_known(v___x_1507_, 8);
lean_dec(v_n_1451_);
v___y_1458_ = v_a_1452_;
v___y_1459_ = v_a_1453_;
v___y_1460_ = v_a_1454_;
v___y_1461_ = v_a_1455_;
goto v___jp_1457_;
}
}
else
{
lean_dec(v_tail_1512_);
lean_dec_ref_known(v_fst_1511_, 2);
lean_dec_ref_known(v___x_1507_, 8);
lean_dec(v_n_1451_);
v___y_1458_ = v_a_1452_;
v___y_1459_ = v_a_1453_;
v___y_1460_ = v_a_1454_;
v___y_1461_ = v_a_1455_;
goto v___jp_1457_;
}
}
else
{
lean_dec(v_fst_1511_);
lean_dec_ref_known(v___x_1507_, 8);
lean_dec(v_n_1451_);
v___y_1458_ = v_a_1452_;
v___y_1459_ = v_a_1453_;
v___y_1460_ = v_a_1454_;
v___y_1461_ = v_a_1455_;
goto v___jp_1457_;
}
}
else
{
lean_object* v_a_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1542_; 
lean_dec_ref_known(v___x_1507_, 8);
lean_dec(v_n_1451_);
v_a_1535_ = lean_ctor_get(v___x_1509_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1509_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1537_ = v___x_1509_;
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_a_1535_);
lean_dec(v___x_1509_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1540_; 
if (v_isShared_1538_ == 0)
{
v___x_1540_ = v___x_1537_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v_a_1535_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
}
}
v___jp_1457_:
{
lean_object* v___x_1462_; lean_object* v___x_1463_; 
v___x_1462_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___closed__1);
v___x_1463_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_1462_, v___y_1458_, v___y_1459_, v___y_1460_, v___y_1461_);
return v___x_1463_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor___boxed(lean_object* v_mvar_1543_, lean_object* v_n_1544_, lean_object* v_a_1545_, lean_object* v_a_1546_, lean_object* v_a_1547_, lean_object* v_a_1548_, lean_object* v_a_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor(v_mvar_1543_, v_n_1544_, v_a_1545_, v_a_1546_, v_a_1547_, v_a_1548_);
lean_dec(v_a_1548_);
lean_dec_ref(v_a_1547_);
lean_dec(v_a_1546_);
lean_dec_ref(v_a_1545_);
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5___redArg(lean_object* v_x_1551_, lean_object* v_x_1552_, lean_object* v_x_1553_, lean_object* v_x_1554_){
_start:
{
lean_object* v_ks_1555_; lean_object* v_vs_1556_; lean_object* v___x_1558_; uint8_t v_isShared_1559_; uint8_t v_isSharedCheck_1580_; 
v_ks_1555_ = lean_ctor_get(v_x_1551_, 0);
v_vs_1556_ = lean_ctor_get(v_x_1551_, 1);
v_isSharedCheck_1580_ = !lean_is_exclusive(v_x_1551_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1558_ = v_x_1551_;
v_isShared_1559_ = v_isSharedCheck_1580_;
goto v_resetjp_1557_;
}
else
{
lean_inc(v_vs_1556_);
lean_inc(v_ks_1555_);
lean_dec(v_x_1551_);
v___x_1558_ = lean_box(0);
v_isShared_1559_ = v_isSharedCheck_1580_;
goto v_resetjp_1557_;
}
v_resetjp_1557_:
{
lean_object* v___x_1560_; uint8_t v___x_1561_; 
v___x_1560_ = lean_array_get_size(v_ks_1555_);
v___x_1561_ = lean_nat_dec_lt(v_x_1552_, v___x_1560_);
if (v___x_1561_ == 0)
{
lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1565_; 
lean_dec(v_x_1552_);
v___x_1562_ = lean_array_push(v_ks_1555_, v_x_1553_);
v___x_1563_ = lean_array_push(v_vs_1556_, v_x_1554_);
if (v_isShared_1559_ == 0)
{
lean_ctor_set(v___x_1558_, 1, v___x_1563_);
lean_ctor_set(v___x_1558_, 0, v___x_1562_);
v___x_1565_ = v___x_1558_;
goto v_reusejp_1564_;
}
else
{
lean_object* v_reuseFailAlloc_1566_; 
v_reuseFailAlloc_1566_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1566_, 0, v___x_1562_);
lean_ctor_set(v_reuseFailAlloc_1566_, 1, v___x_1563_);
v___x_1565_ = v_reuseFailAlloc_1566_;
goto v_reusejp_1564_;
}
v_reusejp_1564_:
{
return v___x_1565_;
}
}
else
{
lean_object* v_k_x27_1567_; uint8_t v___x_1568_; 
v_k_x27_1567_ = lean_array_fget_borrowed(v_ks_1555_, v_x_1552_);
v___x_1568_ = l_Lean_instBEqMVarId_beq(v_x_1553_, v_k_x27_1567_);
if (v___x_1568_ == 0)
{
lean_object* v___x_1570_; 
if (v_isShared_1559_ == 0)
{
v___x_1570_ = v___x_1558_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1574_; 
v_reuseFailAlloc_1574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1574_, 0, v_ks_1555_);
lean_ctor_set(v_reuseFailAlloc_1574_, 1, v_vs_1556_);
v___x_1570_ = v_reuseFailAlloc_1574_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
lean_object* v___x_1571_; lean_object* v___x_1572_; 
v___x_1571_ = lean_unsigned_to_nat(1u);
v___x_1572_ = lean_nat_add(v_x_1552_, v___x_1571_);
lean_dec(v_x_1552_);
v_x_1551_ = v___x_1570_;
v_x_1552_ = v___x_1572_;
goto _start;
}
}
else
{
lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1578_; 
v___x_1575_ = lean_array_fset(v_ks_1555_, v_x_1552_, v_x_1553_);
v___x_1576_ = lean_array_fset(v_vs_1556_, v_x_1552_, v_x_1554_);
lean_dec(v_x_1552_);
if (v_isShared_1559_ == 0)
{
lean_ctor_set(v___x_1558_, 1, v___x_1576_);
lean_ctor_set(v___x_1558_, 0, v___x_1575_);
v___x_1578_ = v___x_1558_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v___x_1575_);
lean_ctor_set(v_reuseFailAlloc_1579_, 1, v___x_1576_);
v___x_1578_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
return v___x_1578_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4___redArg(lean_object* v_n_1581_, lean_object* v_k_1582_, lean_object* v_v_1583_){
_start:
{
lean_object* v___x_1584_; lean_object* v___x_1585_; 
v___x_1584_ = lean_unsigned_to_nat(0u);
v___x_1585_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5___redArg(v_n_1581_, v___x_1584_, v_k_1582_, v_v_1583_);
return v___x_1585_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1586_; 
v___x_1586_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(lean_object* v_x_1587_, size_t v_x_1588_, size_t v_x_1589_, lean_object* v_x_1590_, lean_object* v_x_1591_){
_start:
{
if (lean_obj_tag(v_x_1587_) == 0)
{
lean_object* v_es_1592_; size_t v___x_1593_; size_t v___x_1594_; lean_object* v_j_1595_; lean_object* v___x_1596_; uint8_t v___x_1597_; 
v_es_1592_ = lean_ctor_get(v_x_1587_, 0);
v___x_1593_ = ((size_t)31ULL);
v___x_1594_ = lean_usize_land(v_x_1588_, v___x_1593_);
v_j_1595_ = lean_usize_to_nat(v___x_1594_);
v___x_1596_ = lean_array_get_size(v_es_1592_);
v___x_1597_ = lean_nat_dec_lt(v_j_1595_, v___x_1596_);
if (v___x_1597_ == 0)
{
lean_dec(v_j_1595_);
lean_dec(v_x_1591_);
lean_dec(v_x_1590_);
return v_x_1587_;
}
else
{
lean_object* v___x_1599_; uint8_t v_isShared_1600_; uint8_t v_isSharedCheck_1636_; 
lean_inc_ref(v_es_1592_);
v_isSharedCheck_1636_ = !lean_is_exclusive(v_x_1587_);
if (v_isSharedCheck_1636_ == 0)
{
lean_object* v_unused_1637_; 
v_unused_1637_ = lean_ctor_get(v_x_1587_, 0);
lean_dec(v_unused_1637_);
v___x_1599_ = v_x_1587_;
v_isShared_1600_ = v_isSharedCheck_1636_;
goto v_resetjp_1598_;
}
else
{
lean_dec(v_x_1587_);
v___x_1599_ = lean_box(0);
v_isShared_1600_ = v_isSharedCheck_1636_;
goto v_resetjp_1598_;
}
v_resetjp_1598_:
{
lean_object* v_v_1601_; lean_object* v___x_1602_; lean_object* v_xs_x27_1603_; lean_object* v___y_1605_; 
v_v_1601_ = lean_array_fget(v_es_1592_, v_j_1595_);
v___x_1602_ = lean_box(0);
v_xs_x27_1603_ = lean_array_fset(v_es_1592_, v_j_1595_, v___x_1602_);
switch(lean_obj_tag(v_v_1601_))
{
case 0:
{
lean_object* v_key_1610_; lean_object* v_val_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1621_; 
v_key_1610_ = lean_ctor_get(v_v_1601_, 0);
v_val_1611_ = lean_ctor_get(v_v_1601_, 1);
v_isSharedCheck_1621_ = !lean_is_exclusive(v_v_1601_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1613_ = v_v_1601_;
v_isShared_1614_ = v_isSharedCheck_1621_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_val_1611_);
lean_inc(v_key_1610_);
lean_dec(v_v_1601_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1621_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
uint8_t v___x_1615_; 
v___x_1615_ = l_Lean_instBEqMVarId_beq(v_x_1590_, v_key_1610_);
if (v___x_1615_ == 0)
{
lean_object* v___x_1616_; lean_object* v___x_1617_; 
lean_del_object(v___x_1613_);
v___x_1616_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1610_, v_val_1611_, v_x_1590_, v_x_1591_);
v___x_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1617_, 0, v___x_1616_);
v___y_1605_ = v___x_1617_;
goto v___jp_1604_;
}
else
{
lean_object* v___x_1619_; 
lean_dec(v_val_1611_);
lean_dec(v_key_1610_);
if (v_isShared_1614_ == 0)
{
lean_ctor_set(v___x_1613_, 1, v_x_1591_);
lean_ctor_set(v___x_1613_, 0, v_x_1590_);
v___x_1619_ = v___x_1613_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_x_1590_);
lean_ctor_set(v_reuseFailAlloc_1620_, 1, v_x_1591_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
v___y_1605_ = v___x_1619_;
goto v___jp_1604_;
}
}
}
}
case 1:
{
lean_object* v_node_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1634_; 
v_node_1622_ = lean_ctor_get(v_v_1601_, 0);
v_isSharedCheck_1634_ = !lean_is_exclusive(v_v_1601_);
if (v_isSharedCheck_1634_ == 0)
{
v___x_1624_ = v_v_1601_;
v_isShared_1625_ = v_isSharedCheck_1634_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_node_1622_);
lean_dec(v_v_1601_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1634_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
size_t v___x_1626_; size_t v___x_1627_; size_t v___x_1628_; size_t v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1632_; 
v___x_1626_ = ((size_t)5ULL);
v___x_1627_ = lean_usize_shift_right(v_x_1588_, v___x_1626_);
v___x_1628_ = ((size_t)1ULL);
v___x_1629_ = lean_usize_add(v_x_1589_, v___x_1628_);
v___x_1630_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(v_node_1622_, v___x_1627_, v___x_1629_, v_x_1590_, v_x_1591_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set(v___x_1624_, 0, v___x_1630_);
v___x_1632_ = v___x_1624_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v___x_1630_);
v___x_1632_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
v___y_1605_ = v___x_1632_;
goto v___jp_1604_;
}
}
}
default: 
{
lean_object* v___x_1635_; 
v___x_1635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1635_, 0, v_x_1590_);
lean_ctor_set(v___x_1635_, 1, v_x_1591_);
v___y_1605_ = v___x_1635_;
goto v___jp_1604_;
}
}
v___jp_1604_:
{
lean_object* v___x_1606_; lean_object* v___x_1608_; 
v___x_1606_ = lean_array_fset(v_xs_x27_1603_, v_j_1595_, v___y_1605_);
lean_dec(v_j_1595_);
if (v_isShared_1600_ == 0)
{
lean_ctor_set(v___x_1599_, 0, v___x_1606_);
v___x_1608_ = v___x_1599_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1609_; 
v_reuseFailAlloc_1609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1609_, 0, v___x_1606_);
v___x_1608_ = v_reuseFailAlloc_1609_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
return v___x_1608_;
}
}
}
}
}
else
{
lean_object* v_ks_1638_; lean_object* v_vs_1639_; lean_object* v___x_1641_; uint8_t v_isShared_1642_; uint8_t v_isSharedCheck_1659_; 
v_ks_1638_ = lean_ctor_get(v_x_1587_, 0);
v_vs_1639_ = lean_ctor_get(v_x_1587_, 1);
v_isSharedCheck_1659_ = !lean_is_exclusive(v_x_1587_);
if (v_isSharedCheck_1659_ == 0)
{
v___x_1641_ = v_x_1587_;
v_isShared_1642_ = v_isSharedCheck_1659_;
goto v_resetjp_1640_;
}
else
{
lean_inc(v_vs_1639_);
lean_inc(v_ks_1638_);
lean_dec(v_x_1587_);
v___x_1641_ = lean_box(0);
v_isShared_1642_ = v_isSharedCheck_1659_;
goto v_resetjp_1640_;
}
v_resetjp_1640_:
{
lean_object* v___x_1644_; 
if (v_isShared_1642_ == 0)
{
v___x_1644_ = v___x_1641_;
goto v_reusejp_1643_;
}
else
{
lean_object* v_reuseFailAlloc_1658_; 
v_reuseFailAlloc_1658_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1658_, 0, v_ks_1638_);
lean_ctor_set(v_reuseFailAlloc_1658_, 1, v_vs_1639_);
v___x_1644_ = v_reuseFailAlloc_1658_;
goto v_reusejp_1643_;
}
v_reusejp_1643_:
{
lean_object* v_newNode_1645_; uint8_t v___y_1647_; size_t v___x_1653_; uint8_t v___x_1654_; 
v_newNode_1645_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4___redArg(v___x_1644_, v_x_1590_, v_x_1591_);
v___x_1653_ = ((size_t)7ULL);
v___x_1654_ = lean_usize_dec_le(v___x_1653_, v_x_1589_);
if (v___x_1654_ == 0)
{
lean_object* v___x_1655_; lean_object* v___x_1656_; uint8_t v___x_1657_; 
v___x_1655_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1645_);
v___x_1656_ = lean_unsigned_to_nat(4u);
v___x_1657_ = lean_nat_dec_lt(v___x_1655_, v___x_1656_);
lean_dec(v___x_1655_);
v___y_1647_ = v___x_1657_;
goto v___jp_1646_;
}
else
{
v___y_1647_ = v___x_1654_;
goto v___jp_1646_;
}
v___jp_1646_:
{
if (v___y_1647_ == 0)
{
lean_object* v_ks_1648_; lean_object* v_vs_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; 
v_ks_1648_ = lean_ctor_get(v_newNode_1645_, 0);
lean_inc_ref(v_ks_1648_);
v_vs_1649_ = lean_ctor_get(v_newNode_1645_, 1);
lean_inc_ref(v_vs_1649_);
lean_dec_ref(v_newNode_1645_);
v___x_1650_ = lean_unsigned_to_nat(0u);
v___x_1651_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___closed__0);
v___x_1652_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg(v_x_1589_, v_ks_1648_, v_vs_1649_, v___x_1650_, v___x_1651_);
lean_dec_ref(v_vs_1649_);
lean_dec_ref(v_ks_1648_);
return v___x_1652_;
}
else
{
return v_newNode_1645_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg(size_t v_depth_1660_, lean_object* v_keys_1661_, lean_object* v_vals_1662_, lean_object* v_i_1663_, lean_object* v_entries_1664_){
_start:
{
lean_object* v___x_1665_; uint8_t v___x_1666_; 
v___x_1665_ = lean_array_get_size(v_keys_1661_);
v___x_1666_ = lean_nat_dec_lt(v_i_1663_, v___x_1665_);
if (v___x_1666_ == 0)
{
lean_dec(v_i_1663_);
return v_entries_1664_;
}
else
{
lean_object* v_k_1667_; lean_object* v_v_1668_; uint64_t v___x_1669_; size_t v_h_1670_; size_t v___x_1671_; lean_object* v___x_1672_; size_t v___x_1673_; size_t v___x_1674_; size_t v___x_1675_; size_t v_h_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; 
v_k_1667_ = lean_array_fget_borrowed(v_keys_1661_, v_i_1663_);
v_v_1668_ = lean_array_fget_borrowed(v_vals_1662_, v_i_1663_);
v___x_1669_ = l_Lean_instHashableMVarId_hash(v_k_1667_);
v_h_1670_ = lean_uint64_to_usize(v___x_1669_);
v___x_1671_ = ((size_t)5ULL);
v___x_1672_ = lean_unsigned_to_nat(1u);
v___x_1673_ = ((size_t)1ULL);
v___x_1674_ = lean_usize_sub(v_depth_1660_, v___x_1673_);
v___x_1675_ = lean_usize_mul(v___x_1671_, v___x_1674_);
v_h_1676_ = lean_usize_shift_right(v_h_1670_, v___x_1675_);
v___x_1677_ = lean_nat_add(v_i_1663_, v___x_1672_);
lean_dec(v_i_1663_);
lean_inc(v_v_1668_);
lean_inc(v_k_1667_);
v___x_1678_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(v_entries_1664_, v_h_1676_, v_depth_1660_, v_k_1667_, v_v_1668_);
v_i_1663_ = v___x_1677_;
v_entries_1664_ = v___x_1678_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_depth_1680_, lean_object* v_keys_1681_, lean_object* v_vals_1682_, lean_object* v_i_1683_, lean_object* v_entries_1684_){
_start:
{
size_t v_depth_boxed_1685_; lean_object* v_res_1686_; 
v_depth_boxed_1685_ = lean_unbox_usize(v_depth_1680_);
lean_dec(v_depth_1680_);
v_res_1686_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg(v_depth_boxed_1685_, v_keys_1681_, v_vals_1682_, v_i_1683_, v_entries_1684_);
lean_dec_ref(v_vals_1682_);
lean_dec_ref(v_keys_1681_);
return v_res_1686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_1687_, lean_object* v_x_1688_, lean_object* v_x_1689_, lean_object* v_x_1690_, lean_object* v_x_1691_){
_start:
{
size_t v_x_1813__boxed_1692_; size_t v_x_1814__boxed_1693_; lean_object* v_res_1694_; 
v_x_1813__boxed_1692_ = lean_unbox_usize(v_x_1688_);
lean_dec(v_x_1688_);
v_x_1814__boxed_1693_ = lean_unbox_usize(v_x_1689_);
lean_dec(v_x_1689_);
v_res_1694_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(v_x_1687_, v_x_1813__boxed_1692_, v_x_1814__boxed_1693_, v_x_1690_, v_x_1691_);
return v_res_1694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1___redArg(lean_object* v_x_1695_, lean_object* v_x_1696_, lean_object* v_x_1697_){
_start:
{
uint64_t v___x_1698_; size_t v___x_1699_; size_t v___x_1700_; lean_object* v___x_1701_; 
v___x_1698_ = l_Lean_instHashableMVarId_hash(v_x_1696_);
v___x_1699_ = lean_uint64_to_usize(v___x_1698_);
v___x_1700_ = ((size_t)1ULL);
v___x_1701_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(v_x_1695_, v___x_1699_, v___x_1700_, v_x_1696_, v_x_1697_);
return v___x_1701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(lean_object* v_mvarId_1702_, lean_object* v_val_1703_, lean_object* v___y_1704_){
_start:
{
lean_object* v___x_1706_; lean_object* v_mctx_1707_; lean_object* v_cache_1708_; lean_object* v_zetaDeltaFVarIds_1709_; lean_object* v_postponed_1710_; lean_object* v_diag_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1739_; 
v___x_1706_ = lean_st_ref_take(v___y_1704_);
v_mctx_1707_ = lean_ctor_get(v___x_1706_, 0);
v_cache_1708_ = lean_ctor_get(v___x_1706_, 1);
v_zetaDeltaFVarIds_1709_ = lean_ctor_get(v___x_1706_, 2);
v_postponed_1710_ = lean_ctor_get(v___x_1706_, 3);
v_diag_1711_ = lean_ctor_get(v___x_1706_, 4);
v_isSharedCheck_1739_ = !lean_is_exclusive(v___x_1706_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1713_ = v___x_1706_;
v_isShared_1714_ = v_isSharedCheck_1739_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_diag_1711_);
lean_inc(v_postponed_1710_);
lean_inc(v_zetaDeltaFVarIds_1709_);
lean_inc(v_cache_1708_);
lean_inc(v_mctx_1707_);
lean_dec(v___x_1706_);
v___x_1713_ = lean_box(0);
v_isShared_1714_ = v_isSharedCheck_1739_;
goto v_resetjp_1712_;
}
v_resetjp_1712_:
{
lean_object* v_depth_1715_; lean_object* v_levelAssignDepth_1716_; lean_object* v_lmvarCounter_1717_; lean_object* v_mvarCounter_1718_; lean_object* v_lDecls_1719_; lean_object* v_decls_1720_; lean_object* v_userNames_1721_; lean_object* v_lAssignment_1722_; lean_object* v_eAssignment_1723_; lean_object* v_dAssignment_1724_; lean_object* v___x_1726_; uint8_t v_isShared_1727_; uint8_t v_isSharedCheck_1738_; 
v_depth_1715_ = lean_ctor_get(v_mctx_1707_, 0);
v_levelAssignDepth_1716_ = lean_ctor_get(v_mctx_1707_, 1);
v_lmvarCounter_1717_ = lean_ctor_get(v_mctx_1707_, 2);
v_mvarCounter_1718_ = lean_ctor_get(v_mctx_1707_, 3);
v_lDecls_1719_ = lean_ctor_get(v_mctx_1707_, 4);
v_decls_1720_ = lean_ctor_get(v_mctx_1707_, 5);
v_userNames_1721_ = lean_ctor_get(v_mctx_1707_, 6);
v_lAssignment_1722_ = lean_ctor_get(v_mctx_1707_, 7);
v_eAssignment_1723_ = lean_ctor_get(v_mctx_1707_, 8);
v_dAssignment_1724_ = lean_ctor_get(v_mctx_1707_, 9);
v_isSharedCheck_1738_ = !lean_is_exclusive(v_mctx_1707_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1726_ = v_mctx_1707_;
v_isShared_1727_ = v_isSharedCheck_1738_;
goto v_resetjp_1725_;
}
else
{
lean_inc(v_dAssignment_1724_);
lean_inc(v_eAssignment_1723_);
lean_inc(v_lAssignment_1722_);
lean_inc(v_userNames_1721_);
lean_inc(v_decls_1720_);
lean_inc(v_lDecls_1719_);
lean_inc(v_mvarCounter_1718_);
lean_inc(v_lmvarCounter_1717_);
lean_inc(v_levelAssignDepth_1716_);
lean_inc(v_depth_1715_);
lean_dec(v_mctx_1707_);
v___x_1726_ = lean_box(0);
v_isShared_1727_ = v_isSharedCheck_1738_;
goto v_resetjp_1725_;
}
v_resetjp_1725_:
{
lean_object* v___x_1728_; lean_object* v___x_1730_; 
v___x_1728_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1___redArg(v_eAssignment_1723_, v_mvarId_1702_, v_val_1703_);
if (v_isShared_1727_ == 0)
{
lean_ctor_set(v___x_1726_, 8, v___x_1728_);
v___x_1730_ = v___x_1726_;
goto v_reusejp_1729_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_depth_1715_);
lean_ctor_set(v_reuseFailAlloc_1737_, 1, v_levelAssignDepth_1716_);
lean_ctor_set(v_reuseFailAlloc_1737_, 2, v_lmvarCounter_1717_);
lean_ctor_set(v_reuseFailAlloc_1737_, 3, v_mvarCounter_1718_);
lean_ctor_set(v_reuseFailAlloc_1737_, 4, v_lDecls_1719_);
lean_ctor_set(v_reuseFailAlloc_1737_, 5, v_decls_1720_);
lean_ctor_set(v_reuseFailAlloc_1737_, 6, v_userNames_1721_);
lean_ctor_set(v_reuseFailAlloc_1737_, 7, v_lAssignment_1722_);
lean_ctor_set(v_reuseFailAlloc_1737_, 8, v___x_1728_);
lean_ctor_set(v_reuseFailAlloc_1737_, 9, v_dAssignment_1724_);
v___x_1730_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1729_;
}
v_reusejp_1729_:
{
lean_object* v___x_1732_; 
if (v_isShared_1714_ == 0)
{
lean_ctor_set(v___x_1713_, 0, v___x_1730_);
v___x_1732_ = v___x_1713_;
goto v_reusejp_1731_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v___x_1730_);
lean_ctor_set(v_reuseFailAlloc_1736_, 1, v_cache_1708_);
lean_ctor_set(v_reuseFailAlloc_1736_, 2, v_zetaDeltaFVarIds_1709_);
lean_ctor_set(v_reuseFailAlloc_1736_, 3, v_postponed_1710_);
lean_ctor_set(v_reuseFailAlloc_1736_, 4, v_diag_1711_);
v___x_1732_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1731_;
}
v_reusejp_1731_:
{
lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; 
v___x_1733_ = lean_st_ref_set(v___y_1704_, v___x_1732_);
v___x_1734_ = lean_box(0);
v___x_1735_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1734_);
return v___x_1735_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg___boxed(lean_object* v_mvarId_1740_, lean_object* v_val_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_){
_start:
{
lean_object* v_res_1744_; 
v_res_1744_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(v_mvarId_1740_, v_val_1741_, v___y_1742_);
lean_dec(v___y_1742_);
return v_res_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_toCases_spec__0(lean_object* v_a_1745_, lean_object* v_a_1746_){
_start:
{
if (lean_obj_tag(v_a_1745_) == 0)
{
lean_object* v___x_1747_; 
v___x_1747_ = lean_array_to_list(v_a_1746_);
return v___x_1747_;
}
else
{
lean_object* v_head_1748_; lean_object* v_fst_1749_; uint8_t v___x_1750_; 
v_head_1748_ = lean_ctor_get(v_a_1745_, 0);
v_fst_1749_ = lean_ctor_get(v_head_1748_, 0);
v___x_1750_ = lean_unbox(v_fst_1749_);
if (v___x_1750_ == 0)
{
lean_object* v_tail_1751_; 
v_tail_1751_ = lean_ctor_get(v_a_1745_, 1);
lean_inc(v_tail_1751_);
lean_dec_ref_known(v_a_1745_, 2);
v_a_1745_ = v_tail_1751_;
goto _start;
}
else
{
lean_object* v_tail_1753_; lean_object* v_snd_1754_; lean_object* v___x_1755_; 
lean_inc(v_head_1748_);
v_tail_1753_ = lean_ctor_get(v_a_1745_, 1);
lean_inc(v_tail_1753_);
lean_dec_ref_known(v_a_1745_, 2);
v_snd_1754_ = lean_ctor_get(v_head_1748_, 1);
lean_inc(v_snd_1754_);
lean_dec(v_head_1748_);
v___x_1755_ = lean_array_push(v_a_1746_, v_snd_1754_);
v_a_1745_ = v_tail_1753_;
v_a_1746_ = v___x_1755_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2(lean_object* v_a_1757_, lean_object* v_x_1758_, lean_object* v_x_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
if (lean_obj_tag(v_x_1758_) == 0)
{
lean_object* v___x_1765_; lean_object* v___x_1766_; 
v___x_1765_ = l_List_reverse___redArg(v_x_1759_);
v___x_1766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1766_, 0, v___x_1765_);
return v___x_1766_;
}
else
{
lean_object* v_head_1767_; lean_object* v_tail_1768_; lean_object* v___x_1770_; uint8_t v_isShared_1771_; uint8_t v_isSharedCheck_1842_; 
v_head_1767_ = lean_ctor_get(v_x_1758_, 0);
v_tail_1768_ = lean_ctor_get(v_x_1758_, 1);
v_isSharedCheck_1842_ = !lean_is_exclusive(v_x_1758_);
if (v_isSharedCheck_1842_ == 0)
{
v___x_1770_ = v_x_1758_;
v_isShared_1771_ = v_isSharedCheck_1842_;
goto v_resetjp_1769_;
}
else
{
lean_inc(v_tail_1768_);
lean_inc(v_head_1767_);
lean_dec(v_x_1758_);
v___x_1770_ = lean_box(0);
v_isShared_1771_ = v_isSharedCheck_1842_;
goto v_resetjp_1769_;
}
v_resetjp_1769_:
{
lean_object* v___y_1773_; lean_object* v_fst_1787_; lean_object* v_fst_1788_; lean_object* v_snd_1789_; lean_object* v_toInductionSubgoal_1790_; lean_object* v_snd_1791_; lean_object* v_variablesKept_1792_; lean_object* v_neqs_1793_; lean_object* v_mvarId_1794_; lean_object* v_fields_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; 
v_fst_1787_ = lean_ctor_get(v_head_1767_, 0);
v_fst_1788_ = lean_ctor_get(v_fst_1787_, 0);
lean_inc(v_fst_1788_);
v_snd_1789_ = lean_ctor_get(v_fst_1787_, 1);
v_toInductionSubgoal_1790_ = lean_ctor_get(v_snd_1789_, 0);
lean_inc_ref(v_toInductionSubgoal_1790_);
v_snd_1791_ = lean_ctor_get(v_head_1767_, 1);
lean_inc(v_snd_1791_);
lean_dec(v_head_1767_);
v_variablesKept_1792_ = lean_ctor_get(v_fst_1788_, 0);
lean_inc(v_variablesKept_1792_);
v_neqs_1793_ = lean_ctor_get(v_fst_1788_, 1);
lean_inc(v_neqs_1793_);
lean_dec(v_fst_1788_);
v_mvarId_1794_ = lean_ctor_get(v_toInductionSubgoal_1790_, 0);
lean_inc(v_mvarId_1794_);
v_fields_1795_ = lean_ctor_get(v_toInductionSubgoal_1790_, 1);
lean_inc_ref(v_fields_1795_);
lean_dec_ref(v_toInductionSubgoal_1790_);
v___x_1796_ = lean_array_get_size(v_a_1757_);
v___x_1797_ = lean_unsigned_to_nat(1u);
v___x_1798_ = lean_nat_sub(v___x_1796_, v___x_1797_);
v___x_1799_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select(v_snd_1791_, v___x_1798_, v_mvarId_1794_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
if (lean_obj_tag(v___x_1799_) == 0)
{
lean_object* v_a_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; 
v_a_1800_ = lean_ctor_get(v___x_1799_, 0);
lean_inc(v_a_1800_);
lean_dec_ref_known(v___x_1799_, 1);
lean_inc_ref(v_fields_1795_);
v___x_1801_ = lean_array_to_list(v_fields_1795_);
lean_inc(v_variablesKept_1792_);
v___x_1802_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_variablesKept_1792_, v___x_1801_);
v___x_1803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1));
v___x_1804_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_MkIff_toCases_spec__0(v___x_1802_, v___x_1803_);
if (lean_obj_tag(v_neqs_1793_) == 0)
{
lean_object* v___x_1805_; lean_object* v___x_1806_; 
v___x_1805_ = lp_mathlib_Mathlib_Tactic_MkIff_List_init___redArg(v___x_1804_);
v___x_1806_ = lp_mathlib_Lean_MVarId_existsi(v_a_1800_, v___x_1805_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
if (lean_obj_tag(v___x_1806_) == 0)
{
lean_object* v_a_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; 
v_a_1807_ = lean_ctor_get(v___x_1806_, 0);
lean_inc(v_a_1807_);
lean_dec_ref_known(v___x_1806_, 1);
v___x_1808_ = l_Lean_instInhabitedExpr;
v___x_1809_ = l_List_lengthTR___redArg(v_variablesKept_1792_);
lean_dec(v_variablesKept_1792_);
v___x_1810_ = lean_nat_sub(v___x_1809_, v___x_1797_);
lean_dec(v___x_1809_);
v___x_1811_ = lean_array_get(v___x_1808_, v_fields_1795_, v___x_1810_);
lean_dec(v___x_1810_);
lean_dec_ref(v_fields_1795_);
v___x_1812_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(v_a_1807_, v___x_1811_, v___y_1761_);
v___y_1773_ = v___x_1812_;
goto v___jp_1772_;
}
else
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
lean_dec_ref(v_fields_1795_);
lean_dec(v_variablesKept_1792_);
lean_del_object(v___x_1770_);
lean_dec(v_tail_1768_);
lean_dec(v_x_1759_);
v_a_1813_ = lean_ctor_get(v___x_1806_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1806_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1806_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1806_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
return v___x_1818_;
}
}
}
}
else
{
lean_object* v_val_1821_; lean_object* v___x_1822_; 
lean_dec_ref(v_fields_1795_);
lean_dec(v_variablesKept_1792_);
v_val_1821_ = lean_ctor_get(v_neqs_1793_, 0);
lean_inc(v_val_1821_);
lean_dec_ref_known(v_neqs_1793_, 1);
v___x_1822_ = lp_mathlib_Lean_MVarId_existsi(v_a_1800_, v___x_1804_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
if (lean_obj_tag(v___x_1822_) == 0)
{
lean_object* v_a_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; 
v_a_1823_ = lean_ctor_get(v___x_1822_, 0);
lean_inc(v_a_1823_);
lean_dec_ref_known(v___x_1822_, 1);
v___x_1824_ = lean_nat_sub(v_val_1821_, v___x_1797_);
lean_dec(v_val_1821_);
v___x_1825_ = lp_mathlib_Mathlib_Tactic_MkIff_splitThenConstructor(v_a_1823_, v___x_1824_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
v___y_1773_ = v___x_1825_;
goto v___jp_1772_;
}
else
{
lean_object* v_a_1826_; lean_object* v___x_1828_; uint8_t v_isShared_1829_; uint8_t v_isSharedCheck_1833_; 
lean_dec(v_val_1821_);
lean_del_object(v___x_1770_);
lean_dec(v_tail_1768_);
lean_dec(v_x_1759_);
v_a_1826_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_1833_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_1833_ == 0)
{
v___x_1828_ = v___x_1822_;
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
else
{
lean_inc(v_a_1826_);
lean_dec(v___x_1822_);
v___x_1828_ = lean_box(0);
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
v_resetjp_1827_:
{
lean_object* v___x_1831_; 
if (v_isShared_1829_ == 0)
{
v___x_1831_ = v___x_1828_;
goto v_reusejp_1830_;
}
else
{
lean_object* v_reuseFailAlloc_1832_; 
v_reuseFailAlloc_1832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1832_, 0, v_a_1826_);
v___x_1831_ = v_reuseFailAlloc_1832_;
goto v_reusejp_1830_;
}
v_reusejp_1830_:
{
return v___x_1831_;
}
}
}
}
}
else
{
lean_object* v_a_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_1841_; 
lean_dec_ref(v_fields_1795_);
lean_dec(v_neqs_1793_);
lean_dec(v_variablesKept_1792_);
lean_del_object(v___x_1770_);
lean_dec(v_tail_1768_);
lean_dec(v_x_1759_);
v_a_1834_ = lean_ctor_get(v___x_1799_, 0);
v_isSharedCheck_1841_ = !lean_is_exclusive(v___x_1799_);
if (v_isSharedCheck_1841_ == 0)
{
v___x_1836_ = v___x_1799_;
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_a_1834_);
lean_dec(v___x_1799_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___x_1839_; 
if (v_isShared_1837_ == 0)
{
v___x_1839_ = v___x_1836_;
goto v_reusejp_1838_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v_a_1834_);
v___x_1839_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1838_;
}
v_reusejp_1838_:
{
return v___x_1839_;
}
}
}
v___jp_1772_:
{
if (lean_obj_tag(v___y_1773_) == 0)
{
lean_object* v_a_1774_; lean_object* v___x_1776_; 
v_a_1774_ = lean_ctor_get(v___y_1773_, 0);
lean_inc(v_a_1774_);
lean_dec_ref_known(v___y_1773_, 1);
if (v_isShared_1771_ == 0)
{
lean_ctor_set(v___x_1770_, 1, v_x_1759_);
lean_ctor_set(v___x_1770_, 0, v_a_1774_);
v___x_1776_ = v___x_1770_;
goto v_reusejp_1775_;
}
else
{
lean_object* v_reuseFailAlloc_1778_; 
v_reuseFailAlloc_1778_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1778_, 0, v_a_1774_);
lean_ctor_set(v_reuseFailAlloc_1778_, 1, v_x_1759_);
v___x_1776_ = v_reuseFailAlloc_1778_;
goto v_reusejp_1775_;
}
v_reusejp_1775_:
{
v_x_1758_ = v_tail_1768_;
v_x_1759_ = v___x_1776_;
goto _start;
}
}
else
{
lean_object* v_a_1779_; lean_object* v___x_1781_; uint8_t v_isShared_1782_; uint8_t v_isSharedCheck_1786_; 
lean_del_object(v___x_1770_);
lean_dec(v_tail_1768_);
lean_dec(v_x_1759_);
v_a_1779_ = lean_ctor_get(v___y_1773_, 0);
v_isSharedCheck_1786_ = !lean_is_exclusive(v___y_1773_);
if (v_isSharedCheck_1786_ == 0)
{
v___x_1781_ = v___y_1773_;
v_isShared_1782_ = v_isSharedCheck_1786_;
goto v_resetjp_1780_;
}
else
{
lean_inc(v_a_1779_);
lean_dec(v___y_1773_);
v___x_1781_ = lean_box(0);
v_isShared_1782_ = v_isSharedCheck_1786_;
goto v_resetjp_1780_;
}
v_resetjp_1780_:
{
lean_object* v___x_1784_; 
if (v_isShared_1782_ == 0)
{
v___x_1784_ = v___x_1781_;
goto v_reusejp_1783_;
}
else
{
lean_object* v_reuseFailAlloc_1785_; 
v_reuseFailAlloc_1785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1785_, 0, v_a_1779_);
v___x_1784_ = v_reuseFailAlloc_1785_;
goto v_reusejp_1783_;
}
v_reusejp_1783_:
{
return v___x_1784_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2___boxed(lean_object* v_a_1843_, lean_object* v_x_1844_, lean_object* v_x_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_){
_start:
{
lean_object* v_res_1851_; 
v_res_1851_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2(v_a_1843_, v_x_1844_, v_x_1845_, v___y_1846_, v___y_1847_, v___y_1848_, v___y_1849_);
lean_dec(v___y_1849_);
lean_dec_ref(v___y_1848_);
lean_dec(v___y_1847_);
lean_dec_ref(v___y_1846_);
lean_dec_ref(v_a_1843_);
return v_res_1851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toCases(lean_object* v_mvar_1854_, lean_object* v_shape_1855_, lean_object* v_a_1856_, lean_object* v_a_1857_, lean_object* v_a_1858_, lean_object* v_a_1859_){
_start:
{
uint8_t v___x_1861_; lean_object* v___x_1862_; 
v___x_1861_ = 0;
v___x_1862_ = l_Lean_Meta_intro1Core(v_mvar_1854_, v___x_1861_, v_a_1856_, v_a_1857_, v_a_1858_, v_a_1859_);
if (lean_obj_tag(v___x_1862_) == 0)
{
lean_object* v_a_1863_; lean_object* v_fst_1864_; lean_object* v_snd_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; 
v_a_1863_ = lean_ctor_get(v___x_1862_, 0);
lean_inc(v_a_1863_);
lean_dec_ref_known(v___x_1862_, 1);
v_fst_1864_ = lean_ctor_get(v_a_1863_, 0);
lean_inc(v_fst_1864_);
v_snd_1865_ = lean_ctor_get(v_a_1863_, 1);
lean_inc(v_snd_1865_);
lean_dec(v_a_1863_);
v___x_1866_ = lean_unsigned_to_nat(0u);
v___x_1867_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0));
v___x_1868_ = lean_box(0);
v___x_1869_ = l_Lean_MVarId_cases(v_snd_1865_, v_fst_1864_, v___x_1867_, v___x_1861_, v___x_1868_, v_a_1856_, v_a_1857_, v_a_1858_, v_a_1859_);
if (lean_obj_tag(v___x_1869_) == 0)
{
lean_object* v_a_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; 
v_a_1870_ = lean_ctor_get(v___x_1869_, 0);
lean_inc_n(v_a_1870_, 2);
lean_dec_ref_known(v___x_1869_, 1);
v___x_1871_ = lean_array_to_list(v_a_1870_);
v___x_1872_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_shape_1855_, v___x_1871_);
v___x_1873_ = l_List_zipIdxTR___redArg(v___x_1872_, v___x_1866_);
v___x_1874_ = lean_box(0);
v___x_1875_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toCases_spec__2(v_a_1870_, v___x_1873_, v___x_1874_, v_a_1856_, v_a_1857_, v_a_1858_, v_a_1859_);
lean_dec(v_a_1870_);
if (lean_obj_tag(v___x_1875_) == 0)
{
lean_object* v___x_1877_; uint8_t v_isShared_1878_; uint8_t v_isSharedCheck_1883_; 
v_isSharedCheck_1883_ = !lean_is_exclusive(v___x_1875_);
if (v_isSharedCheck_1883_ == 0)
{
lean_object* v_unused_1884_; 
v_unused_1884_ = lean_ctor_get(v___x_1875_, 0);
lean_dec(v_unused_1884_);
v___x_1877_ = v___x_1875_;
v_isShared_1878_ = v_isSharedCheck_1883_;
goto v_resetjp_1876_;
}
else
{
lean_dec(v___x_1875_);
v___x_1877_ = lean_box(0);
v_isShared_1878_ = v_isSharedCheck_1883_;
goto v_resetjp_1876_;
}
v_resetjp_1876_:
{
lean_object* v___x_1879_; lean_object* v___x_1881_; 
v___x_1879_ = lean_box(0);
if (v_isShared_1878_ == 0)
{
lean_ctor_set(v___x_1877_, 0, v___x_1879_);
v___x_1881_ = v___x_1877_;
goto v_reusejp_1880_;
}
else
{
lean_object* v_reuseFailAlloc_1882_; 
v_reuseFailAlloc_1882_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1882_, 0, v___x_1879_);
v___x_1881_ = v_reuseFailAlloc_1882_;
goto v_reusejp_1880_;
}
v_reusejp_1880_:
{
return v___x_1881_;
}
}
}
else
{
lean_object* v_a_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_1892_; 
v_a_1885_ = lean_ctor_get(v___x_1875_, 0);
v_isSharedCheck_1892_ = !lean_is_exclusive(v___x_1875_);
if (v_isSharedCheck_1892_ == 0)
{
v___x_1887_ = v___x_1875_;
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_a_1885_);
lean_dec(v___x_1875_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v___x_1890_; 
if (v_isShared_1888_ == 0)
{
v___x_1890_ = v___x_1887_;
goto v_reusejp_1889_;
}
else
{
lean_object* v_reuseFailAlloc_1891_; 
v_reuseFailAlloc_1891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1891_, 0, v_a_1885_);
v___x_1890_ = v_reuseFailAlloc_1891_;
goto v_reusejp_1889_;
}
v_reusejp_1889_:
{
return v___x_1890_;
}
}
}
}
else
{
lean_object* v_a_1893_; lean_object* v___x_1895_; uint8_t v_isShared_1896_; uint8_t v_isSharedCheck_1900_; 
lean_dec(v_shape_1855_);
v_a_1893_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1900_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1900_ == 0)
{
v___x_1895_ = v___x_1869_;
v_isShared_1896_ = v_isSharedCheck_1900_;
goto v_resetjp_1894_;
}
else
{
lean_inc(v_a_1893_);
lean_dec(v___x_1869_);
v___x_1895_ = lean_box(0);
v_isShared_1896_ = v_isSharedCheck_1900_;
goto v_resetjp_1894_;
}
v_resetjp_1894_:
{
lean_object* v___x_1898_; 
if (v_isShared_1896_ == 0)
{
v___x_1898_ = v___x_1895_;
goto v_reusejp_1897_;
}
else
{
lean_object* v_reuseFailAlloc_1899_; 
v_reuseFailAlloc_1899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1899_, 0, v_a_1893_);
v___x_1898_ = v_reuseFailAlloc_1899_;
goto v_reusejp_1897_;
}
v_reusejp_1897_:
{
return v___x_1898_;
}
}
}
}
else
{
lean_object* v_a_1901_; lean_object* v___x_1903_; uint8_t v_isShared_1904_; uint8_t v_isSharedCheck_1908_; 
lean_dec(v_shape_1855_);
v_a_1901_ = lean_ctor_get(v___x_1862_, 0);
v_isSharedCheck_1908_ = !lean_is_exclusive(v___x_1862_);
if (v_isSharedCheck_1908_ == 0)
{
v___x_1903_ = v___x_1862_;
v_isShared_1904_ = v_isSharedCheck_1908_;
goto v_resetjp_1902_;
}
else
{
lean_inc(v_a_1901_);
lean_dec(v___x_1862_);
v___x_1903_ = lean_box(0);
v_isShared_1904_ = v_isSharedCheck_1908_;
goto v_resetjp_1902_;
}
v_resetjp_1902_:
{
lean_object* v___x_1906_; 
if (v_isShared_1904_ == 0)
{
v___x_1906_ = v___x_1903_;
goto v_reusejp_1905_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v_a_1901_);
v___x_1906_ = v_reuseFailAlloc_1907_;
goto v_reusejp_1905_;
}
v_reusejp_1905_:
{
return v___x_1906_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toCases___boxed(lean_object* v_mvar_1909_, lean_object* v_shape_1910_, lean_object* v_a_1911_, lean_object* v_a_1912_, lean_object* v_a_1913_, lean_object* v_a_1914_, lean_object* v_a_1915_){
_start:
{
lean_object* v_res_1916_; 
v_res_1916_ = lp_mathlib_Mathlib_Tactic_MkIff_toCases(v_mvar_1909_, v_shape_1910_, v_a_1911_, v_a_1912_, v_a_1913_, v_a_1914_);
lean_dec(v_a_1914_);
lean_dec_ref(v_a_1913_);
lean_dec(v_a_1912_);
lean_dec_ref(v_a_1911_);
return v_res_1916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1(lean_object* v_mvarId_1917_, lean_object* v_val_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_){
_start:
{
lean_object* v___x_1924_; 
v___x_1924_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(v_mvarId_1917_, v_val_1918_, v___y_1920_);
return v___x_1924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___boxed(lean_object* v_mvarId_1925_, lean_object* v_val_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_){
_start:
{
lean_object* v_res_1932_; 
v_res_1932_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1(v_mvarId_1925_, v_val_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_);
lean_dec(v___y_1930_);
lean_dec_ref(v___y_1929_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1927_);
return v_res_1932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1(lean_object* v_00_u03b2_1933_, lean_object* v_x_1934_, lean_object* v_x_1935_, lean_object* v_x_1936_){
_start:
{
lean_object* v___x_1937_; 
v___x_1937_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1___redArg(v_x_1934_, v_x_1935_, v_x_1936_);
return v___x_1937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_1938_, lean_object* v_x_1939_, size_t v_x_1940_, size_t v_x_1941_, lean_object* v_x_1942_, lean_object* v_x_1943_){
_start:
{
lean_object* v___x_1944_; 
v___x_1944_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___redArg(v_x_1939_, v_x_1940_, v_x_1941_, v_x_1942_, v_x_1943_);
return v___x_1944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1945_, lean_object* v_x_1946_, lean_object* v_x_1947_, lean_object* v_x_1948_, lean_object* v_x_1949_, lean_object* v_x_1950_){
_start:
{
size_t v_x_2359__boxed_1951_; size_t v_x_2360__boxed_1952_; lean_object* v_res_1953_; 
v_x_2359__boxed_1951_ = lean_unbox_usize(v_x_1947_);
lean_dec(v_x_1947_);
v_x_2360__boxed_1952_ = lean_unbox_usize(v_x_1948_);
lean_dec(v_x_1948_);
v_res_1953_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2(v_00_u03b2_1945_, v_x_1946_, v_x_2359__boxed_1951_, v_x_2360__boxed_1952_, v_x_1949_, v_x_1950_);
return v_res_1953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1954_, lean_object* v_n_1955_, lean_object* v_k_1956_, lean_object* v_v_1957_){
_start:
{
lean_object* v___x_1958_; 
v___x_1958_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4___redArg(v_n_1955_, v_k_1956_, v_v_1957_);
return v___x_1958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_1959_, size_t v_depth_1960_, lean_object* v_keys_1961_, lean_object* v_vals_1962_, lean_object* v_heq_1963_, lean_object* v_i_1964_, lean_object* v_entries_1965_){
_start:
{
lean_object* v___x_1966_; 
v___x_1966_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___redArg(v_depth_1960_, v_keys_1961_, v_vals_1962_, v_i_1964_, v_entries_1965_);
return v___x_1966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b2_1967_, lean_object* v_depth_1968_, lean_object* v_keys_1969_, lean_object* v_vals_1970_, lean_object* v_heq_1971_, lean_object* v_i_1972_, lean_object* v_entries_1973_){
_start:
{
size_t v_depth_boxed_1974_; lean_object* v_res_1975_; 
v_depth_boxed_1974_ = lean_unbox_usize(v_depth_1968_);
lean_dec(v_depth_1968_);
v_res_1975_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__5(v_00_u03b2_1967_, v_depth_boxed_1974_, v_keys_1969_, v_vals_1970_, v_heq_1971_, v_i_1972_, v_entries_1973_);
lean_dec_ref(v_vals_1970_);
lean_dec_ref(v_keys_1969_);
return v_res_1975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5(lean_object* v_00_u03b2_1976_, lean_object* v_x_1977_, lean_object* v_x_1978_, lean_object* v_x_1979_, lean_object* v_x_1980_){
_start:
{
lean_object* v___x_1981_; 
v___x_1981_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1_spec__1_spec__2_spec__4_spec__5___redArg(v_x_1977_, v_x_1978_, v_x_1979_, v_x_1980_);
return v___x_1981_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1(void){
_start:
{
lean_object* v___x_1983_; lean_object* v___x_1984_; 
v___x_1983_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__0));
v___x_1984_ = l_Lean_stringToMessageData(v___x_1983_);
return v___x_1984_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3(void){
_start:
{
lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1986_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__2));
v___x_1987_ = l_Lean_stringToMessageData(v___x_1986_);
return v___x_1987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum(lean_object* v_n_1988_, lean_object* v_mvar_1989_, lean_object* v_h_1990_, lean_object* v_a_1991_, lean_object* v_a_1992_, lean_object* v_a_1993_, lean_object* v_a_1994_){
_start:
{
lean_object* v___y_1997_; lean_object* v___y_1998_; lean_object* v___y_1999_; lean_object* v___y_2000_; lean_object* v___y_2004_; lean_object* v___y_2005_; lean_object* v___y_2006_; lean_object* v___y_2007_; lean_object* v_zero_2010_; uint8_t v_isZero_2011_; 
v_zero_2010_ = lean_unsigned_to_nat(0u);
v_isZero_2011_ = lean_nat_dec_eq(v_n_1988_, v_zero_2010_);
if (v_isZero_2011_ == 1)
{
lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; 
v___x_2012_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2012_, 0, v_h_1990_);
lean_ctor_set(v___x_2012_, 1, v_mvar_1989_);
v___x_2013_ = lean_box(0);
v___x_2014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2014_, 0, v___x_2012_);
lean_ctor_set(v___x_2014_, 1, v___x_2013_);
v___x_2015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2015_, 0, v___x_2014_);
return v___x_2015_;
}
else
{
lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; 
v___x_2016_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0));
v___x_2017_ = lean_box(0);
v___x_2018_ = l_Lean_MVarId_cases(v_mvar_1989_, v_h_1990_, v___x_2016_, v_isZero_2011_, v___x_2017_, v_a_1991_, v_a_1992_, v_a_1993_, v_a_1994_);
if (lean_obj_tag(v___x_2018_) == 0)
{
lean_object* v_a_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; uint8_t v___x_2022_; 
v_a_2019_ = lean_ctor_get(v___x_2018_, 0);
lean_inc(v_a_2019_);
lean_dec_ref_known(v___x_2018_, 1);
v___x_2020_ = lean_array_get_size(v_a_2019_);
v___x_2021_ = lean_unsigned_to_nat(2u);
v___x_2022_ = lean_nat_dec_eq(v___x_2020_, v___x_2021_);
if (v___x_2022_ == 0)
{
lean_object* v___x_2023_; lean_object* v___x_2024_; 
lean_dec(v_a_2019_);
v___x_2023_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3, &lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__3);
v___x_2024_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2023_, v_a_1991_, v_a_1992_, v_a_1993_, v_a_1994_);
return v___x_2024_;
}
else
{
lean_object* v___x_2025_; lean_object* v_toInductionSubgoal_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2066_; 
v___x_2025_ = lean_array_fget(v_a_2019_, v_zero_2010_);
v_toInductionSubgoal_2026_ = lean_ctor_get(v___x_2025_, 0);
v_isSharedCheck_2066_ = !lean_is_exclusive(v___x_2025_);
if (v_isSharedCheck_2066_ == 0)
{
lean_object* v_unused_2067_; 
v_unused_2067_ = lean_ctor_get(v___x_2025_, 1);
lean_dec(v_unused_2067_);
v___x_2028_ = v___x_2025_;
v_isShared_2029_ = v_isSharedCheck_2066_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_toInductionSubgoal_2026_);
lean_dec(v___x_2025_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2066_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
lean_object* v_mvarId_2030_; lean_object* v_fields_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; uint8_t v___x_2034_; 
v_mvarId_2030_ = lean_ctor_get(v_toInductionSubgoal_2026_, 0);
lean_inc(v_mvarId_2030_);
v_fields_2031_ = lean_ctor_get(v_toInductionSubgoal_2026_, 1);
lean_inc_ref(v_fields_2031_);
lean_dec_ref(v_toInductionSubgoal_2026_);
v___x_2032_ = lean_array_get_size(v_fields_2031_);
v___x_2033_ = lean_unsigned_to_nat(1u);
v___x_2034_ = lean_nat_dec_eq(v___x_2032_, v___x_2033_);
if (v___x_2034_ == 0)
{
lean_dec_ref(v_fields_2031_);
lean_dec(v_mvarId_2030_);
lean_del_object(v___x_2028_);
lean_dec(v_a_2019_);
v___y_1997_ = v_a_1991_;
v___y_1998_ = v_a_1992_;
v___y_1999_ = v_a_1993_;
v___y_2000_ = v_a_1994_;
goto v___jp_1996_;
}
else
{
lean_object* v___x_2035_; 
v___x_2035_ = lean_array_fget(v_fields_2031_, v_zero_2010_);
lean_dec_ref(v_fields_2031_);
if (lean_obj_tag(v___x_2035_) == 1)
{
lean_object* v_fvarId_2036_; lean_object* v___x_2037_; lean_object* v_toInductionSubgoal_2038_; lean_object* v___x_2040_; uint8_t v_isShared_2041_; uint8_t v_isSharedCheck_2064_; 
v_fvarId_2036_ = lean_ctor_get(v___x_2035_, 0);
lean_inc(v_fvarId_2036_);
lean_dec_ref_known(v___x_2035_, 1);
v___x_2037_ = lean_array_fget(v_a_2019_, v___x_2033_);
lean_dec(v_a_2019_);
v_toInductionSubgoal_2038_ = lean_ctor_get(v___x_2037_, 0);
v_isSharedCheck_2064_ = !lean_is_exclusive(v___x_2037_);
if (v_isSharedCheck_2064_ == 0)
{
lean_object* v_unused_2065_; 
v_unused_2065_ = lean_ctor_get(v___x_2037_, 1);
lean_dec(v_unused_2065_);
v___x_2040_ = v___x_2037_;
v_isShared_2041_ = v_isSharedCheck_2064_;
goto v_resetjp_2039_;
}
else
{
lean_inc(v_toInductionSubgoal_2038_);
lean_dec(v___x_2037_);
v___x_2040_ = lean_box(0);
v_isShared_2041_ = v_isSharedCheck_2064_;
goto v_resetjp_2039_;
}
v_resetjp_2039_:
{
lean_object* v_mvarId_2042_; lean_object* v_fields_2043_; lean_object* v___x_2044_; uint8_t v___x_2045_; 
v_mvarId_2042_ = lean_ctor_get(v_toInductionSubgoal_2038_, 0);
lean_inc(v_mvarId_2042_);
v_fields_2043_ = lean_ctor_get(v_toInductionSubgoal_2038_, 1);
lean_inc_ref(v_fields_2043_);
lean_dec_ref(v_toInductionSubgoal_2038_);
v___x_2044_ = lean_array_get_size(v_fields_2043_);
v___x_2045_ = lean_nat_dec_eq(v___x_2044_, v___x_2033_);
if (v___x_2045_ == 0)
{
lean_dec_ref(v_fields_2043_);
lean_dec(v_mvarId_2042_);
lean_del_object(v___x_2040_);
lean_dec(v_fvarId_2036_);
lean_dec(v_mvarId_2030_);
lean_del_object(v___x_2028_);
v___y_2004_ = v_a_1991_;
v___y_2005_ = v_a_1992_;
v___y_2006_ = v_a_1993_;
v___y_2007_ = v_a_1994_;
goto v___jp_2003_;
}
else
{
lean_object* v___x_2046_; 
v___x_2046_ = lean_array_fget(v_fields_2043_, v_zero_2010_);
lean_dec_ref(v_fields_2043_);
if (lean_obj_tag(v___x_2046_) == 1)
{
lean_object* v_fvarId_2047_; lean_object* v_n_2048_; lean_object* v___x_2049_; 
v_fvarId_2047_ = lean_ctor_get(v___x_2046_, 0);
lean_inc(v_fvarId_2047_);
lean_dec_ref_known(v___x_2046_, 1);
v_n_2048_ = lean_nat_sub(v_n_1988_, v___x_2033_);
v___x_2049_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum(v_n_2048_, v_mvarId_2042_, v_fvarId_2047_, v_a_1991_, v_a_1992_, v_a_1993_, v_a_1994_);
lean_dec(v_n_2048_);
if (lean_obj_tag(v___x_2049_) == 0)
{
lean_object* v_a_2050_; lean_object* v___x_2052_; uint8_t v_isShared_2053_; uint8_t v_isSharedCheck_2063_; 
v_a_2050_ = lean_ctor_get(v___x_2049_, 0);
v_isSharedCheck_2063_ = !lean_is_exclusive(v___x_2049_);
if (v_isSharedCheck_2063_ == 0)
{
v___x_2052_ = v___x_2049_;
v_isShared_2053_ = v_isSharedCheck_2063_;
goto v_resetjp_2051_;
}
else
{
lean_inc(v_a_2050_);
lean_dec(v___x_2049_);
v___x_2052_ = lean_box(0);
v_isShared_2053_ = v_isSharedCheck_2063_;
goto v_resetjp_2051_;
}
v_resetjp_2051_:
{
lean_object* v___x_2055_; 
if (v_isShared_2041_ == 0)
{
lean_ctor_set(v___x_2040_, 1, v_mvarId_2030_);
lean_ctor_set(v___x_2040_, 0, v_fvarId_2036_);
v___x_2055_ = v___x_2040_;
goto v_reusejp_2054_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v_fvarId_2036_);
lean_ctor_set(v_reuseFailAlloc_2062_, 1, v_mvarId_2030_);
v___x_2055_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2054_;
}
v_reusejp_2054_:
{
lean_object* v___x_2057_; 
if (v_isShared_2029_ == 0)
{
lean_ctor_set_tag(v___x_2028_, 1);
lean_ctor_set(v___x_2028_, 1, v_a_2050_);
lean_ctor_set(v___x_2028_, 0, v___x_2055_);
v___x_2057_ = v___x_2028_;
goto v_reusejp_2056_;
}
else
{
lean_object* v_reuseFailAlloc_2061_; 
v_reuseFailAlloc_2061_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2061_, 0, v___x_2055_);
lean_ctor_set(v_reuseFailAlloc_2061_, 1, v_a_2050_);
v___x_2057_ = v_reuseFailAlloc_2061_;
goto v_reusejp_2056_;
}
v_reusejp_2056_:
{
lean_object* v___x_2059_; 
if (v_isShared_2053_ == 0)
{
lean_ctor_set(v___x_2052_, 0, v___x_2057_);
v___x_2059_ = v___x_2052_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2060_; 
v_reuseFailAlloc_2060_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2060_, 0, v___x_2057_);
v___x_2059_ = v_reuseFailAlloc_2060_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
return v___x_2059_;
}
}
}
}
}
else
{
lean_del_object(v___x_2040_);
lean_dec(v_fvarId_2036_);
lean_dec(v_mvarId_2030_);
lean_del_object(v___x_2028_);
return v___x_2049_;
}
}
else
{
lean_dec(v___x_2046_);
lean_dec(v_mvarId_2042_);
lean_del_object(v___x_2040_);
lean_dec(v_fvarId_2036_);
lean_dec(v_mvarId_2030_);
lean_del_object(v___x_2028_);
v___y_2004_ = v_a_1991_;
v___y_2005_ = v_a_1992_;
v___y_2006_ = v_a_1993_;
v___y_2007_ = v_a_1994_;
goto v___jp_2003_;
}
}
}
}
else
{
lean_dec(v___x_2035_);
lean_dec(v_mvarId_2030_);
lean_del_object(v___x_2028_);
lean_dec(v_a_2019_);
v___y_1997_ = v_a_1991_;
v___y_1998_ = v_a_1992_;
v___y_1999_ = v_a_1993_;
v___y_2000_ = v_a_1994_;
goto v___jp_1996_;
}
}
}
}
}
else
{
lean_object* v_a_2068_; lean_object* v___x_2070_; uint8_t v_isShared_2071_; uint8_t v_isSharedCheck_2075_; 
v_a_2068_ = lean_ctor_get(v___x_2018_, 0);
v_isSharedCheck_2075_ = !lean_is_exclusive(v___x_2018_);
if (v_isSharedCheck_2075_ == 0)
{
v___x_2070_ = v___x_2018_;
v_isShared_2071_ = v_isSharedCheck_2075_;
goto v_resetjp_2069_;
}
else
{
lean_inc(v_a_2068_);
lean_dec(v___x_2018_);
v___x_2070_ = lean_box(0);
v_isShared_2071_ = v_isSharedCheck_2075_;
goto v_resetjp_2069_;
}
v_resetjp_2069_:
{
lean_object* v___x_2073_; 
if (v_isShared_2071_ == 0)
{
v___x_2073_ = v___x_2070_;
goto v_reusejp_2072_;
}
else
{
lean_object* v_reuseFailAlloc_2074_; 
v_reuseFailAlloc_2074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2074_, 0, v_a_2068_);
v___x_2073_ = v_reuseFailAlloc_2074_;
goto v_reusejp_2072_;
}
v_reusejp_2072_:
{
return v___x_2073_;
}
}
}
}
v___jp_1996_:
{
lean_object* v___x_2001_; lean_object* v___x_2002_; 
v___x_2001_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1);
v___x_2002_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2001_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_);
return v___x_2002_;
}
v___jp_2003_:
{
lean_object* v___x_2008_; lean_object* v___x_2009_; 
v___x_2008_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1);
v___x_2009_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2008_, v___y_2004_, v___y_2005_, v___y_2006_, v___y_2007_);
return v___x_2009_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___boxed(lean_object* v_n_2076_, lean_object* v_mvar_2077_, lean_object* v_h_2078_, lean_object* v_a_2079_, lean_object* v_a_2080_, lean_object* v_a_2081_, lean_object* v_a_2082_, lean_object* v_a_2083_){
_start:
{
lean_object* v_res_2084_; 
v_res_2084_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum(v_n_2076_, v_mvar_2077_, v_h_2078_, v_a_2079_, v_a_2080_, v_a_2081_, v_a_2082_);
lean_dec(v_a_2082_);
lean_dec_ref(v_a_2081_);
lean_dec(v_a_2080_);
lean_dec_ref(v_a_2079_);
lean_dec(v_n_2076_);
return v_res_2084_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1(void){
_start:
{
lean_object* v___x_2086_; lean_object* v___x_2087_; 
v___x_2086_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__0));
v___x_2087_ = l_Lean_stringToMessageData(v___x_2086_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(lean_object* v_n_2088_, lean_object* v_mvar_2089_, lean_object* v_h_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_, lean_object* v_a_2093_, lean_object* v_a_2094_){
_start:
{
lean_object* v___y_2097_; lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v_zero_2103_; uint8_t v_isZero_2104_; 
v_zero_2103_ = lean_unsigned_to_nat(0u);
v_isZero_2104_ = lean_nat_dec_eq(v_n_2088_, v_zero_2103_);
if (v_isZero_2104_ == 1)
{
lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; 
v___x_2105_ = lean_box(0);
v___x_2106_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2106_, 0, v_h_2090_);
lean_ctor_set(v___x_2106_, 1, v___x_2105_);
v___x_2107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2107_, 0, v_mvar_2089_);
lean_ctor_set(v___x_2107_, 1, v___x_2106_);
v___x_2108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2108_, 0, v___x_2107_);
return v___x_2108_;
}
else
{
lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; 
v___x_2109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0));
v___x_2110_ = lean_box(0);
v___x_2111_ = l_Lean_MVarId_cases(v_mvar_2089_, v_h_2090_, v___x_2109_, v_isZero_2104_, v___x_2110_, v_a_2091_, v_a_2092_, v_a_2093_, v_a_2094_);
if (lean_obj_tag(v___x_2111_) == 0)
{
lean_object* v_a_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; uint8_t v___x_2115_; 
v_a_2112_ = lean_ctor_get(v___x_2111_, 0);
lean_inc(v_a_2112_);
lean_dec_ref_known(v___x_2111_, 1);
v___x_2113_ = lean_array_get_size(v_a_2112_);
v___x_2114_ = lean_unsigned_to_nat(1u);
v___x_2115_ = lean_nat_dec_eq(v___x_2113_, v___x_2114_);
if (v___x_2115_ == 0)
{
lean_object* v___x_2116_; lean_object* v___x_2117_; 
lean_dec(v_a_2112_);
v___x_2116_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___closed__1);
v___x_2117_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2116_, v_a_2091_, v_a_2092_, v_a_2093_, v_a_2094_);
return v___x_2117_;
}
else
{
lean_object* v___x_2118_; lean_object* v_toInductionSubgoal_2119_; lean_object* v___x_2121_; uint8_t v_isShared_2122_; uint8_t v_isSharedCheck_2154_; 
v___x_2118_ = lean_array_fget(v_a_2112_, v_zero_2103_);
lean_dec(v_a_2112_);
v_toInductionSubgoal_2119_ = lean_ctor_get(v___x_2118_, 0);
v_isSharedCheck_2154_ = !lean_is_exclusive(v___x_2118_);
if (v_isSharedCheck_2154_ == 0)
{
lean_object* v_unused_2155_; 
v_unused_2155_ = lean_ctor_get(v___x_2118_, 1);
lean_dec(v_unused_2155_);
v___x_2121_ = v___x_2118_;
v_isShared_2122_ = v_isSharedCheck_2154_;
goto v_resetjp_2120_;
}
else
{
lean_inc(v_toInductionSubgoal_2119_);
lean_dec(v___x_2118_);
v___x_2121_ = lean_box(0);
v_isShared_2122_ = v_isSharedCheck_2154_;
goto v_resetjp_2120_;
}
v_resetjp_2120_:
{
lean_object* v_mvarId_2123_; lean_object* v_fields_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; uint8_t v___x_2127_; 
v_mvarId_2123_ = lean_ctor_get(v_toInductionSubgoal_2119_, 0);
lean_inc(v_mvarId_2123_);
v_fields_2124_ = lean_ctor_get(v_toInductionSubgoal_2119_, 1);
lean_inc_ref(v_fields_2124_);
lean_dec_ref(v_toInductionSubgoal_2119_);
v___x_2125_ = lean_array_get_size(v_fields_2124_);
v___x_2126_ = lean_unsigned_to_nat(2u);
v___x_2127_ = lean_nat_dec_eq(v___x_2125_, v___x_2126_);
if (v___x_2127_ == 0)
{
lean_dec_ref(v_fields_2124_);
lean_dec(v_mvarId_2123_);
lean_del_object(v___x_2121_);
v___y_2097_ = v_a_2091_;
v___y_2098_ = v_a_2092_;
v___y_2099_ = v_a_2093_;
v___y_2100_ = v_a_2094_;
goto v___jp_2096_;
}
else
{
lean_object* v___x_2128_; 
v___x_2128_ = lean_array_fget_borrowed(v_fields_2124_, v_zero_2103_);
if (lean_obj_tag(v___x_2128_) == 1)
{
lean_object* v_fvarId_2129_; lean_object* v___x_2130_; 
v_fvarId_2129_ = lean_ctor_get(v___x_2128_, 0);
lean_inc(v_fvarId_2129_);
v___x_2130_ = lean_array_fget(v_fields_2124_, v___x_2114_);
lean_dec_ref(v_fields_2124_);
if (lean_obj_tag(v___x_2130_) == 1)
{
lean_object* v_fvarId_2131_; lean_object* v_n_2132_; lean_object* v___x_2133_; 
v_fvarId_2131_ = lean_ctor_get(v___x_2130_, 0);
lean_inc(v_fvarId_2131_);
lean_dec_ref_known(v___x_2130_, 1);
v_n_2132_ = lean_nat_sub(v_n_2088_, v___x_2114_);
v___x_2133_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v_n_2132_, v_mvarId_2123_, v_fvarId_2131_, v_a_2091_, v_a_2092_, v_a_2093_, v_a_2094_);
lean_dec(v_n_2132_);
if (lean_obj_tag(v___x_2133_) == 0)
{
lean_object* v_a_2134_; lean_object* v___x_2136_; uint8_t v_isShared_2137_; uint8_t v_isSharedCheck_2153_; 
v_a_2134_ = lean_ctor_get(v___x_2133_, 0);
v_isSharedCheck_2153_ = !lean_is_exclusive(v___x_2133_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2136_ = v___x_2133_;
v_isShared_2137_ = v_isSharedCheck_2153_;
goto v_resetjp_2135_;
}
else
{
lean_inc(v_a_2134_);
lean_dec(v___x_2133_);
v___x_2136_ = lean_box(0);
v_isShared_2137_ = v_isSharedCheck_2153_;
goto v_resetjp_2135_;
}
v_resetjp_2135_:
{
lean_object* v_fst_2138_; lean_object* v_snd_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2152_; 
v_fst_2138_ = lean_ctor_get(v_a_2134_, 0);
v_snd_2139_ = lean_ctor_get(v_a_2134_, 1);
v_isSharedCheck_2152_ = !lean_is_exclusive(v_a_2134_);
if (v_isSharedCheck_2152_ == 0)
{
v___x_2141_ = v_a_2134_;
v_isShared_2142_ = v_isSharedCheck_2152_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_snd_2139_);
lean_inc(v_fst_2138_);
lean_dec(v_a_2134_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2152_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v___x_2144_; 
if (v_isShared_2122_ == 0)
{
lean_ctor_set_tag(v___x_2121_, 1);
lean_ctor_set(v___x_2121_, 1, v_snd_2139_);
lean_ctor_set(v___x_2121_, 0, v_fvarId_2129_);
v___x_2144_ = v___x_2121_;
goto v_reusejp_2143_;
}
else
{
lean_object* v_reuseFailAlloc_2151_; 
v_reuseFailAlloc_2151_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2151_, 0, v_fvarId_2129_);
lean_ctor_set(v_reuseFailAlloc_2151_, 1, v_snd_2139_);
v___x_2144_ = v_reuseFailAlloc_2151_;
goto v_reusejp_2143_;
}
v_reusejp_2143_:
{
lean_object* v___x_2146_; 
if (v_isShared_2142_ == 0)
{
lean_ctor_set(v___x_2141_, 1, v___x_2144_);
v___x_2146_ = v___x_2141_;
goto v_reusejp_2145_;
}
else
{
lean_object* v_reuseFailAlloc_2150_; 
v_reuseFailAlloc_2150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2150_, 0, v_fst_2138_);
lean_ctor_set(v_reuseFailAlloc_2150_, 1, v___x_2144_);
v___x_2146_ = v_reuseFailAlloc_2150_;
goto v_reusejp_2145_;
}
v_reusejp_2145_:
{
lean_object* v___x_2148_; 
if (v_isShared_2137_ == 0)
{
lean_ctor_set(v___x_2136_, 0, v___x_2146_);
v___x_2148_ = v___x_2136_;
goto v_reusejp_2147_;
}
else
{
lean_object* v_reuseFailAlloc_2149_; 
v_reuseFailAlloc_2149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2149_, 0, v___x_2146_);
v___x_2148_ = v_reuseFailAlloc_2149_;
goto v_reusejp_2147_;
}
v_reusejp_2147_:
{
return v___x_2148_;
}
}
}
}
}
}
else
{
lean_dec(v_fvarId_2129_);
lean_del_object(v___x_2121_);
return v___x_2133_;
}
}
else
{
lean_dec(v___x_2130_);
lean_dec(v_fvarId_2129_);
lean_dec(v_mvarId_2123_);
lean_del_object(v___x_2121_);
v___y_2097_ = v_a_2091_;
v___y_2098_ = v_a_2092_;
v___y_2099_ = v_a_2093_;
v___y_2100_ = v_a_2094_;
goto v___jp_2096_;
}
}
else
{
lean_dec_ref(v_fields_2124_);
lean_dec(v_mvarId_2123_);
lean_del_object(v___x_2121_);
v___y_2097_ = v_a_2091_;
v___y_2098_ = v_a_2092_;
v___y_2099_ = v_a_2093_;
v___y_2100_ = v_a_2094_;
goto v___jp_2096_;
}
}
}
}
}
else
{
lean_object* v_a_2156_; lean_object* v___x_2158_; uint8_t v_isShared_2159_; uint8_t v_isSharedCheck_2163_; 
v_a_2156_ = lean_ctor_get(v___x_2111_, 0);
v_isSharedCheck_2163_ = !lean_is_exclusive(v___x_2111_);
if (v_isSharedCheck_2163_ == 0)
{
v___x_2158_ = v___x_2111_;
v_isShared_2159_ = v_isSharedCheck_2163_;
goto v_resetjp_2157_;
}
else
{
lean_inc(v_a_2156_);
lean_dec(v___x_2111_);
v___x_2158_ = lean_box(0);
v_isShared_2159_ = v_isSharedCheck_2163_;
goto v_resetjp_2157_;
}
v_resetjp_2157_:
{
lean_object* v___x_2161_; 
if (v_isShared_2159_ == 0)
{
v___x_2161_ = v___x_2158_;
goto v_reusejp_2160_;
}
else
{
lean_object* v_reuseFailAlloc_2162_; 
v_reuseFailAlloc_2162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2162_, 0, v_a_2156_);
v___x_2161_ = v_reuseFailAlloc_2162_;
goto v_reusejp_2160_;
}
v_reusejp_2160_:
{
return v___x_2161_;
}
}
}
}
v___jp_2096_:
{
lean_object* v___x_2101_; lean_object* v___x_2102_; 
v___x_2101_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum___closed__1);
v___x_2102_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2101_, v___y_2097_, v___y_2098_, v___y_2099_, v___y_2100_);
return v___x_2102_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd___boxed(lean_object* v_n_2164_, lean_object* v_mvar_2165_, lean_object* v_h_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_, lean_object* v_a_2169_, lean_object* v_a_2170_, lean_object* v_a_2171_){
_start:
{
lean_object* v_res_2172_; 
v_res_2172_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v_n_2164_, v_mvar_2165_, v_h_2166_, v_a_2167_, v_a_2168_, v_a_2169_, v_a_2170_);
lean_dec(v_a_2170_);
lean_dec_ref(v_a_2169_);
lean_dec(v_a_2168_);
lean_dec_ref(v_a_2167_);
lean_dec(v_n_2164_);
return v_res_2172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(lean_object* v_x_2173_, lean_object* v_x_2174_){
_start:
{
if (lean_obj_tag(v_x_2173_) == 0)
{
lean_object* v___x_2175_; 
lean_dec(v_x_2174_);
v___x_2175_ = lean_box(0);
return v___x_2175_;
}
else
{
lean_object* v_head_2176_; uint8_t v___x_2177_; 
v_head_2176_ = lean_ctor_get(v_x_2173_, 0);
v___x_2177_ = lean_unbox(v_head_2176_);
if (v___x_2177_ == 0)
{
lean_object* v_tail_2178_; lean_object* v___x_2180_; uint8_t v_isShared_2181_; uint8_t v_isSharedCheck_2187_; 
v_tail_2178_ = lean_ctor_get(v_x_2173_, 1);
v_isSharedCheck_2187_ = !lean_is_exclusive(v_x_2173_);
if (v_isSharedCheck_2187_ == 0)
{
lean_object* v_unused_2188_; 
v_unused_2188_ = lean_ctor_get(v_x_2173_, 0);
lean_dec(v_unused_2188_);
v___x_2180_ = v_x_2173_;
v_isShared_2181_ = v_isSharedCheck_2187_;
goto v_resetjp_2179_;
}
else
{
lean_inc(v_tail_2178_);
lean_dec(v_x_2173_);
v___x_2180_ = lean_box(0);
v_isShared_2181_ = v_isSharedCheck_2187_;
goto v_resetjp_2179_;
}
v_resetjp_2179_:
{
lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2185_; 
v___x_2182_ = lean_box(0);
v___x_2183_ = lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(v_tail_2178_, v_x_2174_);
if (v_isShared_2181_ == 0)
{
lean_ctor_set(v___x_2180_, 1, v___x_2183_);
lean_ctor_set(v___x_2180_, 0, v___x_2182_);
v___x_2185_ = v___x_2180_;
goto v_reusejp_2184_;
}
else
{
lean_object* v_reuseFailAlloc_2186_; 
v_reuseFailAlloc_2186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2186_, 0, v___x_2182_);
lean_ctor_set(v_reuseFailAlloc_2186_, 1, v___x_2183_);
v___x_2185_ = v_reuseFailAlloc_2186_;
goto v_reusejp_2184_;
}
v_reusejp_2184_:
{
return v___x_2185_;
}
}
}
else
{
if (lean_obj_tag(v_x_2174_) == 0)
{
lean_object* v___x_2189_; 
lean_dec_ref_known(v_x_2173_, 2);
v___x_2189_ = lean_box(0);
return v___x_2189_;
}
else
{
lean_object* v_tail_2190_; lean_object* v_head_2191_; lean_object* v_tail_2192_; lean_object* v___x_2194_; uint8_t v_isShared_2195_; uint8_t v_isSharedCheck_2201_; 
v_tail_2190_ = lean_ctor_get(v_x_2173_, 1);
lean_inc(v_tail_2190_);
lean_dec_ref_known(v_x_2173_, 2);
v_head_2191_ = lean_ctor_get(v_x_2174_, 0);
v_tail_2192_ = lean_ctor_get(v_x_2174_, 1);
v_isSharedCheck_2201_ = !lean_is_exclusive(v_x_2174_);
if (v_isSharedCheck_2201_ == 0)
{
v___x_2194_ = v_x_2174_;
v_isShared_2195_ = v_isSharedCheck_2201_;
goto v_resetjp_2193_;
}
else
{
lean_inc(v_tail_2192_);
lean_inc(v_head_2191_);
lean_dec(v_x_2174_);
v___x_2194_ = lean_box(0);
v_isShared_2195_ = v_isSharedCheck_2201_;
goto v_resetjp_2193_;
}
v_resetjp_2193_:
{
lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2199_; 
v___x_2196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2196_, 0, v_head_2191_);
v___x_2197_ = lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(v_tail_2190_, v_tail_2192_);
if (v_isShared_2195_ == 0)
{
lean_ctor_set(v___x_2194_, 1, v___x_2197_);
lean_ctor_set(v___x_2194_, 0, v___x_2196_);
v___x_2199_ = v___x_2194_;
goto v_reusejp_2198_;
}
else
{
lean_object* v_reuseFailAlloc_2200_; 
v_reuseFailAlloc_2200_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2200_, 0, v___x_2196_);
lean_ctor_set(v_reuseFailAlloc_2200_, 1, v___x_2197_);
v___x_2199_ = v_reuseFailAlloc_2200_;
goto v_reusejp_2198_;
}
v_reusejp_2198_:
{
return v___x_2199_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge(lean_object* v_00_u03b1_2202_, lean_object* v_x_2203_, lean_object* v_x_2204_){
_start:
{
lean_object* v___x_2205_; 
v___x_2205_ = lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(v_x_2203_, v_x_2204_);
return v___x_2205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg(lean_object* v_mvarId_2206_, lean_object* v_x_2207_, lean_object* v___y_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_){
_start:
{
lean_object* v___x_2213_; 
v___x_2213_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2206_, v_x_2207_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_);
if (lean_obj_tag(v___x_2213_) == 0)
{
lean_object* v_a_2214_; lean_object* v___x_2216_; uint8_t v_isShared_2217_; uint8_t v_isSharedCheck_2221_; 
v_a_2214_ = lean_ctor_get(v___x_2213_, 0);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2213_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2216_ = v___x_2213_;
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
else
{
lean_inc(v_a_2214_);
lean_dec(v___x_2213_);
v___x_2216_ = lean_box(0);
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
v_resetjp_2215_:
{
lean_object* v___x_2219_; 
if (v_isShared_2217_ == 0)
{
v___x_2219_ = v___x_2216_;
goto v_reusejp_2218_;
}
else
{
lean_object* v_reuseFailAlloc_2220_; 
v_reuseFailAlloc_2220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2220_, 0, v_a_2214_);
v___x_2219_ = v_reuseFailAlloc_2220_;
goto v_reusejp_2218_;
}
v_reusejp_2218_:
{
return v___x_2219_;
}
}
}
else
{
lean_object* v_a_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2229_; 
v_a_2222_ = lean_ctor_get(v___x_2213_, 0);
v_isSharedCheck_2229_ = !lean_is_exclusive(v___x_2213_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2224_ = v___x_2213_;
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_a_2222_);
lean_dec(v___x_2213_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v___x_2227_; 
if (v_isShared_2225_ == 0)
{
v___x_2227_ = v___x_2224_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_a_2222_);
v___x_2227_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
return v___x_2227_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg___boxed(lean_object* v_mvarId_2230_, lean_object* v_x_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_){
_start:
{
lean_object* v_res_2237_; 
v_res_2237_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg(v_mvarId_2230_, v_x_2231_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
lean_dec(v___y_2235_);
lean_dec_ref(v___y_2234_);
lean_dec(v___y_2233_);
lean_dec_ref(v___y_2232_);
return v_res_2237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3(lean_object* v_00_u03b1_2238_, lean_object* v_mvarId_2239_, lean_object* v_x_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_){
_start:
{
lean_object* v___x_2246_; 
v___x_2246_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg(v_mvarId_2239_, v_x_2240_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_);
return v___x_2246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___boxed(lean_object* v_00_u03b1_2247_, lean_object* v_mvarId_2248_, lean_object* v_x_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_){
_start:
{
lean_object* v_res_2255_; 
v_res_2255_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3(v_00_u03b1_2247_, v_mvarId_2248_, v_x_2249_, v___y_2250_, v___y_2251_, v___y_2252_, v___y_2253_);
lean_dec(v___y_2253_);
lean_dec_ref(v___y_2252_);
lean_dec(v___y_2251_);
lean_dec_ref(v___y_2250_);
return v_res_2255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__0(lean_object* v_a_2256_, lean_object* v_a_2257_){
_start:
{
if (lean_obj_tag(v_a_2256_) == 0)
{
lean_object* v___x_2258_; 
v___x_2258_ = l_List_reverse___redArg(v_a_2257_);
return v___x_2258_;
}
else
{
lean_object* v_head_2259_; uint8_t v___x_2260_; 
v_head_2259_ = lean_ctor_get(v_a_2256_, 0);
v___x_2260_ = lean_unbox(v_head_2259_);
if (v___x_2260_ == 0)
{
lean_object* v_tail_2261_; 
v_tail_2261_ = lean_ctor_get(v_a_2256_, 1);
lean_inc(v_tail_2261_);
lean_dec_ref_known(v_a_2256_, 2);
v_a_2256_ = v_tail_2261_;
goto _start;
}
else
{
lean_object* v_tail_2263_; lean_object* v___x_2265_; uint8_t v_isShared_2266_; uint8_t v_isSharedCheck_2271_; 
lean_inc(v_head_2259_);
v_tail_2263_ = lean_ctor_get(v_a_2256_, 1);
v_isSharedCheck_2271_ = !lean_is_exclusive(v_a_2256_);
if (v_isSharedCheck_2271_ == 0)
{
lean_object* v_unused_2272_; 
v_unused_2272_ = lean_ctor_get(v_a_2256_, 0);
lean_dec(v_unused_2272_);
v___x_2265_ = v_a_2256_;
v_isShared_2266_ = v_isSharedCheck_2271_;
goto v_resetjp_2264_;
}
else
{
lean_inc(v_tail_2263_);
lean_dec(v_a_2256_);
v___x_2265_ = lean_box(0);
v_isShared_2266_ = v_isSharedCheck_2271_;
goto v_resetjp_2264_;
}
v_resetjp_2264_:
{
lean_object* v___x_2268_; 
if (v_isShared_2266_ == 0)
{
lean_ctor_set(v___x_2265_, 1, v_a_2257_);
v___x_2268_ = v___x_2265_;
goto v_reusejp_2267_;
}
else
{
lean_object* v_reuseFailAlloc_2270_; 
v_reuseFailAlloc_2270_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2270_, 0, v_head_2259_);
lean_ctor_set(v_reuseFailAlloc_2270_, 1, v_a_2257_);
v___x_2268_ = v_reuseFailAlloc_2270_;
goto v_reusejp_2267_;
}
v_reusejp_2267_:
{
v_a_2256_ = v_tail_2263_;
v_a_2257_ = v___x_2268_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2(lean_object* v_x_2273_, lean_object* v_x_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_){
_start:
{
if (lean_obj_tag(v_x_2273_) == 0)
{
lean_object* v___x_2280_; lean_object* v___x_2281_; 
v___x_2280_ = l_List_reverse___redArg(v_x_2274_);
v___x_2281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2281_, 0, v___x_2280_);
return v___x_2281_;
}
else
{
lean_object* v_head_2282_; lean_object* v_tail_2283_; lean_object* v___x_2285_; uint8_t v_isShared_2286_; uint8_t v_isSharedCheck_2308_; 
v_head_2282_ = lean_ctor_get(v_x_2273_, 0);
v_tail_2283_ = lean_ctor_get(v_x_2273_, 1);
v_isSharedCheck_2308_ = !lean_is_exclusive(v_x_2273_);
if (v_isSharedCheck_2308_ == 0)
{
v___x_2285_ = v_x_2273_;
v_isShared_2286_ = v_isSharedCheck_2308_;
goto v_resetjp_2284_;
}
else
{
lean_inc(v_tail_2283_);
lean_inc(v_head_2282_);
lean_dec(v_x_2273_);
v___x_2285_ = lean_box(0);
v_isShared_2286_ = v_isSharedCheck_2308_;
goto v_resetjp_2284_;
}
v_resetjp_2284_:
{
lean_object* v_a_2288_; 
if (lean_obj_tag(v_head_2282_) == 0)
{
lean_object* v___x_2293_; uint8_t v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; 
v___x_2293_ = lean_box(0);
v___x_2294_ = 0;
v___x_2295_ = lean_box(0);
v___x_2296_ = l_Lean_Meta_mkFreshExprMVar(v___x_2293_, v___x_2294_, v___x_2295_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_);
if (lean_obj_tag(v___x_2296_) == 0)
{
lean_object* v_a_2297_; 
v_a_2297_ = lean_ctor_get(v___x_2296_, 0);
lean_inc(v_a_2297_);
lean_dec_ref_known(v___x_2296_, 1);
v_a_2288_ = v_a_2297_;
goto v___jp_2287_;
}
else
{
lean_object* v_a_2298_; lean_object* v___x_2300_; uint8_t v_isShared_2301_; uint8_t v_isSharedCheck_2305_; 
lean_del_object(v___x_2285_);
lean_dec(v_tail_2283_);
lean_dec(v_x_2274_);
v_a_2298_ = lean_ctor_get(v___x_2296_, 0);
v_isSharedCheck_2305_ = !lean_is_exclusive(v___x_2296_);
if (v_isSharedCheck_2305_ == 0)
{
v___x_2300_ = v___x_2296_;
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
else
{
lean_inc(v_a_2298_);
lean_dec(v___x_2296_);
v___x_2300_ = lean_box(0);
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
v_resetjp_2299_:
{
lean_object* v___x_2303_; 
if (v_isShared_2301_ == 0)
{
v___x_2303_ = v___x_2300_;
goto v_reusejp_2302_;
}
else
{
lean_object* v_reuseFailAlloc_2304_; 
v_reuseFailAlloc_2304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2304_, 0, v_a_2298_);
v___x_2303_ = v_reuseFailAlloc_2304_;
goto v_reusejp_2302_;
}
v_reusejp_2302_:
{
return v___x_2303_;
}
}
}
}
else
{
lean_object* v_val_2306_; lean_object* v___x_2307_; 
v_val_2306_ = lean_ctor_get(v_head_2282_, 0);
lean_inc(v_val_2306_);
lean_dec_ref_known(v_head_2282_, 1);
v___x_2307_ = l_Lean_mkFVar(v_val_2306_);
v_a_2288_ = v___x_2307_;
goto v___jp_2287_;
}
v___jp_2287_:
{
lean_object* v___x_2290_; 
if (v_isShared_2286_ == 0)
{
lean_ctor_set(v___x_2285_, 1, v_x_2274_);
lean_ctor_set(v___x_2285_, 0, v_a_2288_);
v___x_2290_ = v___x_2285_;
goto v_reusejp_2289_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v_a_2288_);
lean_ctor_set(v_reuseFailAlloc_2292_, 1, v_x_2274_);
v___x_2290_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2289_;
}
v_reusejp_2289_:
{
v_x_2273_ = v_tail_2283_;
v_x_2274_ = v___x_2290_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2___boxed(lean_object* v_x_2309_, lean_object* v_x_2310_, lean_object* v___y_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_){
_start:
{
lean_object* v_res_2316_; 
v_res_2316_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2(v_x_2309_, v_x_2310_, v___y_2311_, v___y_2312_, v___y_2313_, v___y_2314_);
lean_dec(v___y_2314_);
lean_dec_ref(v___y_2313_);
lean_dec(v___y_2312_);
lean_dec_ref(v___y_2311_);
return v_res_2316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__1(lean_object* v_a_2317_, lean_object* v_a_2318_){
_start:
{
if (lean_obj_tag(v_a_2317_) == 0)
{
lean_object* v___x_2319_; 
v___x_2319_ = l_List_reverse___redArg(v_a_2318_);
return v___x_2319_;
}
else
{
lean_object* v_head_2320_; lean_object* v_tail_2321_; lean_object* v___x_2323_; uint8_t v_isShared_2324_; uint8_t v_isSharedCheck_2330_; 
v_head_2320_ = lean_ctor_get(v_a_2317_, 0);
v_tail_2321_ = lean_ctor_get(v_a_2317_, 1);
v_isSharedCheck_2330_ = !lean_is_exclusive(v_a_2317_);
if (v_isSharedCheck_2330_ == 0)
{
v___x_2323_ = v_a_2317_;
v_isShared_2324_ = v_isSharedCheck_2330_;
goto v_resetjp_2322_;
}
else
{
lean_inc(v_tail_2321_);
lean_inc(v_head_2320_);
lean_dec(v_a_2317_);
v___x_2323_ = lean_box(0);
v_isShared_2324_ = v_isSharedCheck_2330_;
goto v_resetjp_2322_;
}
v_resetjp_2322_:
{
lean_object* v___x_2325_; lean_object* v___x_2327_; 
v___x_2325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2325_, 0, v_head_2320_);
if (v_isShared_2324_ == 0)
{
lean_ctor_set(v___x_2323_, 1, v_a_2318_);
lean_ctor_set(v___x_2323_, 0, v___x_2325_);
v___x_2327_ = v___x_2323_;
goto v_reusejp_2326_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v___x_2325_);
lean_ctor_set(v_reuseFailAlloc_2329_, 1, v_a_2318_);
v___x_2327_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2326_;
}
v_reusejp_2326_:
{
v_a_2317_ = v_tail_2321_;
v_a_2318_ = v___x_2327_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0(lean_object* v_gs_2333_, lean_object* v___x_2334_, lean_object* v_variablesKept_2335_, lean_object* v_fst_2336_, lean_object* v_fst_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_){
_start:
{
lean_object* v_lctx_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; 
v_lctx_2343_ = lean_ctor_get(v___y_2338_, 2);
v___x_2344_ = l_Lean_LocalContext_getFVarIds(v_lctx_2343_);
v___x_2345_ = lean_array_to_list(v___x_2344_);
v___x_2346_ = l_List_lengthTR___redArg(v_gs_2333_);
v___x_2347_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0));
lean_inc(v___x_2345_);
v___x_2348_ = l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_box(0), v___x_2345_, v___x_2345_, v___x_2346_, v___x_2347_);
v___x_2349_ = l_List_reverse___redArg(v___x_2345_);
lean_inc(v___x_2349_);
v___x_2350_ = l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_box(0), v___x_2349_, v___x_2349_, v___x_2334_, v___x_2347_);
lean_dec(v___x_2349_);
v___x_2351_ = l_List_reverse___redArg(v___x_2350_);
v___x_2352_ = lean_box(0);
v___x_2353_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__1(v___x_2348_, v___x_2352_);
v___x_2354_ = lp_mathlib_Mathlib_Tactic_MkIff_listBoolMerge___redArg(v_variablesKept_2335_, v___x_2351_);
v___x_2355_ = l_List_appendTR___redArg(v___x_2353_, v___x_2354_);
v___x_2356_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__2(v___x_2355_, v___x_2352_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_);
if (lean_obj_tag(v___x_2356_) == 0)
{
lean_object* v_a_2357_; lean_object* v___x_2358_; 
v_a_2357_ = lean_ctor_get(v___x_2356_, 0);
lean_inc(v_a_2357_);
lean_dec_ref_known(v___x_2356_, 1);
v___x_2358_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_fst_2336_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_);
if (lean_obj_tag(v___x_2358_) == 0)
{
lean_object* v_a_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; 
v_a_2359_ = lean_ctor_get(v___x_2358_, 0);
lean_inc(v_a_2359_);
lean_dec_ref_known(v___x_2358_, 1);
v___x_2360_ = lean_array_mk(v_a_2357_);
v___x_2361_ = l_Lean_mkAppN(v_a_2359_, v___x_2360_);
lean_dec_ref(v___x_2360_);
lean_inc(v___y_2341_);
lean_inc_ref(v___y_2340_);
lean_inc(v___y_2339_);
lean_inc_ref(v___y_2338_);
lean_inc_ref(v___x_2361_);
v___x_2362_ = lean_infer_type(v___x_2361_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_);
if (lean_obj_tag(v___x_2362_) == 0)
{
lean_object* v_a_2363_; lean_object* v___x_2364_; 
v_a_2363_ = lean_ctor_get(v___x_2362_, 0);
lean_inc(v_a_2363_);
lean_dec_ref_known(v___x_2362_, 1);
lean_inc(v_fst_2337_);
v___x_2364_ = l_Lean_MVarId_getType(v_fst_2337_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_);
if (lean_obj_tag(v___x_2364_) == 0)
{
lean_object* v_a_2365_; lean_object* v___x_2366_; 
v_a_2365_ = lean_ctor_get(v___x_2364_, 0);
lean_inc(v_a_2365_);
lean_dec_ref_known(v___x_2364_, 1);
v___x_2366_ = l_Lean_Meta_isExprDefEq(v_a_2363_, v_a_2365_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_);
lean_dec(v___y_2341_);
lean_dec_ref(v___y_2340_);
lean_dec_ref(v___y_2338_);
if (lean_obj_tag(v___x_2366_) == 0)
{
lean_object* v___x_2367_; 
lean_dec_ref_known(v___x_2366_, 1);
v___x_2367_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_MkIff_toCases_spec__1___redArg(v_fst_2337_, v___x_2361_, v___y_2339_);
lean_dec(v___y_2339_);
return v___x_2367_;
}
else
{
lean_object* v_a_2368_; lean_object* v___x_2370_; uint8_t v_isShared_2371_; uint8_t v_isSharedCheck_2375_; 
lean_dec_ref(v___x_2361_);
lean_dec(v___y_2339_);
lean_dec(v_fst_2337_);
v_a_2368_ = lean_ctor_get(v___x_2366_, 0);
v_isSharedCheck_2375_ = !lean_is_exclusive(v___x_2366_);
if (v_isSharedCheck_2375_ == 0)
{
v___x_2370_ = v___x_2366_;
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
else
{
lean_inc(v_a_2368_);
lean_dec(v___x_2366_);
v___x_2370_ = lean_box(0);
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
v_resetjp_2369_:
{
lean_object* v___x_2373_; 
if (v_isShared_2371_ == 0)
{
v___x_2373_ = v___x_2370_;
goto v_reusejp_2372_;
}
else
{
lean_object* v_reuseFailAlloc_2374_; 
v_reuseFailAlloc_2374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2374_, 0, v_a_2368_);
v___x_2373_ = v_reuseFailAlloc_2374_;
goto v_reusejp_2372_;
}
v_reusejp_2372_:
{
return v___x_2373_;
}
}
}
}
else
{
lean_object* v_a_2376_; lean_object* v___x_2378_; uint8_t v_isShared_2379_; uint8_t v_isSharedCheck_2383_; 
lean_dec(v_a_2363_);
lean_dec_ref(v___x_2361_);
lean_dec(v___y_2341_);
lean_dec_ref(v___y_2340_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v_fst_2337_);
v_a_2376_ = lean_ctor_get(v___x_2364_, 0);
v_isSharedCheck_2383_ = !lean_is_exclusive(v___x_2364_);
if (v_isSharedCheck_2383_ == 0)
{
v___x_2378_ = v___x_2364_;
v_isShared_2379_ = v_isSharedCheck_2383_;
goto v_resetjp_2377_;
}
else
{
lean_inc(v_a_2376_);
lean_dec(v___x_2364_);
v___x_2378_ = lean_box(0);
v_isShared_2379_ = v_isSharedCheck_2383_;
goto v_resetjp_2377_;
}
v_resetjp_2377_:
{
lean_object* v___x_2381_; 
if (v_isShared_2379_ == 0)
{
v___x_2381_ = v___x_2378_;
goto v_reusejp_2380_;
}
else
{
lean_object* v_reuseFailAlloc_2382_; 
v_reuseFailAlloc_2382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2382_, 0, v_a_2376_);
v___x_2381_ = v_reuseFailAlloc_2382_;
goto v_reusejp_2380_;
}
v_reusejp_2380_:
{
return v___x_2381_;
}
}
}
}
else
{
lean_object* v_a_2384_; lean_object* v___x_2386_; uint8_t v_isShared_2387_; uint8_t v_isSharedCheck_2391_; 
lean_dec_ref(v___x_2361_);
lean_dec(v___y_2341_);
lean_dec_ref(v___y_2340_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v_fst_2337_);
v_a_2384_ = lean_ctor_get(v___x_2362_, 0);
v_isSharedCheck_2391_ = !lean_is_exclusive(v___x_2362_);
if (v_isSharedCheck_2391_ == 0)
{
v___x_2386_ = v___x_2362_;
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
else
{
lean_inc(v_a_2384_);
lean_dec(v___x_2362_);
v___x_2386_ = lean_box(0);
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
v_resetjp_2385_:
{
lean_object* v___x_2389_; 
if (v_isShared_2387_ == 0)
{
v___x_2389_ = v___x_2386_;
goto v_reusejp_2388_;
}
else
{
lean_object* v_reuseFailAlloc_2390_; 
v_reuseFailAlloc_2390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2390_, 0, v_a_2384_);
v___x_2389_ = v_reuseFailAlloc_2390_;
goto v_reusejp_2388_;
}
v_reusejp_2388_:
{
return v___x_2389_;
}
}
}
}
else
{
lean_object* v_a_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2399_; 
lean_dec(v_a_2357_);
lean_dec(v___y_2341_);
lean_dec_ref(v___y_2340_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v_fst_2337_);
v_a_2392_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2399_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2399_ == 0)
{
v___x_2394_ = v___x_2358_;
v_isShared_2395_ = v_isSharedCheck_2399_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_a_2392_);
lean_dec(v___x_2358_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2399_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v___x_2397_; 
if (v_isShared_2395_ == 0)
{
v___x_2397_ = v___x_2394_;
goto v_reusejp_2396_;
}
else
{
lean_object* v_reuseFailAlloc_2398_; 
v_reuseFailAlloc_2398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2398_, 0, v_a_2392_);
v___x_2397_ = v_reuseFailAlloc_2398_;
goto v_reusejp_2396_;
}
v_reusejp_2396_:
{
return v___x_2397_;
}
}
}
}
else
{
lean_object* v_a_2400_; lean_object* v___x_2402_; uint8_t v_isShared_2403_; uint8_t v_isSharedCheck_2407_; 
lean_dec(v___y_2341_);
lean_dec_ref(v___y_2340_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v_fst_2337_);
lean_dec(v_fst_2336_);
v_a_2400_ = lean_ctor_get(v___x_2356_, 0);
v_isSharedCheck_2407_ = !lean_is_exclusive(v___x_2356_);
if (v_isSharedCheck_2407_ == 0)
{
v___x_2402_ = v___x_2356_;
v_isShared_2403_ = v_isSharedCheck_2407_;
goto v_resetjp_2401_;
}
else
{
lean_inc(v_a_2400_);
lean_dec(v___x_2356_);
v___x_2402_ = lean_box(0);
v_isShared_2403_ = v_isSharedCheck_2407_;
goto v_resetjp_2401_;
}
v_resetjp_2401_:
{
lean_object* v___x_2405_; 
if (v_isShared_2403_ == 0)
{
v___x_2405_ = v___x_2402_;
goto v_reusejp_2404_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v_a_2400_);
v___x_2405_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2404_;
}
v_reusejp_2404_:
{
return v___x_2405_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___boxed(lean_object* v_gs_2408_, lean_object* v___x_2409_, lean_object* v_variablesKept_2410_, lean_object* v_fst_2411_, lean_object* v_fst_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_){
_start:
{
lean_object* v_res_2418_; 
v_res_2418_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0(v_gs_2408_, v___x_2409_, v_variablesKept_2410_, v_fst_2411_, v_fst_2412_, v___y_2413_, v___y_2414_, v___y_2415_, v___y_2416_);
lean_dec(v_gs_2408_);
return v_res_2418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4(lean_object* v_x_2419_, lean_object* v_x_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_){
_start:
{
if (lean_obj_tag(v_x_2420_) == 0)
{
lean_object* v___x_2426_; 
v___x_2426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2426_, 0, v_x_2419_);
return v___x_2426_;
}
else
{
lean_object* v_tail_2427_; uint8_t v___x_2428_; lean_object* v___x_2429_; 
v_tail_2427_ = lean_ctor_get(v_x_2420_, 1);
v___x_2428_ = 0;
v___x_2429_ = l_Lean_Meta_intro1Core(v_x_2419_, v___x_2428_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
if (lean_obj_tag(v___x_2429_) == 0)
{
lean_object* v_a_2430_; lean_object* v_fst_2431_; lean_object* v_snd_2432_; lean_object* v___x_2433_; 
v_a_2430_ = lean_ctor_get(v___x_2429_, 0);
lean_inc(v_a_2430_);
lean_dec_ref_known(v___x_2429_, 1);
v_fst_2431_ = lean_ctor_get(v_a_2430_, 0);
lean_inc(v_fst_2431_);
v_snd_2432_ = lean_ctor_get(v_a_2430_, 1);
lean_inc(v_snd_2432_);
lean_dec(v_a_2430_);
v___x_2433_ = l_Lean_Meta_subst(v_snd_2432_, v_fst_2431_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
if (lean_obj_tag(v___x_2433_) == 0)
{
lean_object* v_a_2434_; 
v_a_2434_ = lean_ctor_get(v___x_2433_, 0);
lean_inc(v_a_2434_);
lean_dec_ref_known(v___x_2433_, 1);
v_x_2419_ = v_a_2434_;
v_x_2420_ = v_tail_2427_;
goto _start;
}
else
{
return v___x_2433_;
}
}
else
{
lean_object* v_a_2436_; lean_object* v___x_2438_; uint8_t v_isShared_2439_; uint8_t v_isSharedCheck_2443_; 
v_a_2436_ = lean_ctor_get(v___x_2429_, 0);
v_isSharedCheck_2443_ = !lean_is_exclusive(v___x_2429_);
if (v_isSharedCheck_2443_ == 0)
{
v___x_2438_ = v___x_2429_;
v_isShared_2439_ = v_isSharedCheck_2443_;
goto v_resetjp_2437_;
}
else
{
lean_inc(v_a_2436_);
lean_dec(v___x_2429_);
v___x_2438_ = lean_box(0);
v_isShared_2439_ = v_isSharedCheck_2443_;
goto v_resetjp_2437_;
}
v_resetjp_2437_:
{
lean_object* v___x_2441_; 
if (v_isShared_2439_ == 0)
{
v___x_2441_ = v___x_2438_;
goto v_reusejp_2440_;
}
else
{
lean_object* v_reuseFailAlloc_2442_; 
v_reuseFailAlloc_2442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2442_, 0, v_a_2436_);
v___x_2441_ = v_reuseFailAlloc_2442_;
goto v_reusejp_2440_;
}
v_reusejp_2440_:
{
return v___x_2441_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4___boxed(lean_object* v_x_2444_, lean_object* v_x_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_){
_start:
{
lean_object* v_res_2451_; 
v_res_2451_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4(v_x_2444_, v_x_2445_, v___y_2446_, v___y_2447_, v___y_2448_, v___y_2449_);
lean_dec(v___y_2449_);
lean_dec_ref(v___y_2448_);
lean_dec(v___y_2447_);
lean_dec_ref(v___y_2446_);
lean_dec(v_x_2445_);
return v_res_2451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5(lean_object* v_gs_2452_, lean_object* v_x_2453_, lean_object* v_x_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_){
_start:
{
if (lean_obj_tag(v_x_2453_) == 0)
{
lean_object* v___x_2460_; lean_object* v___x_2461_; 
lean_dec(v_gs_2452_);
v___x_2460_ = l_List_reverse___redArg(v_x_2454_);
v___x_2461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2461_, 0, v___x_2460_);
return v___x_2461_;
}
else
{
lean_object* v_head_2462_; lean_object* v_snd_2463_; lean_object* v_fst_2464_; lean_object* v_snd_2465_; lean_object* v_tail_2466_; lean_object* v___x_2468_; uint8_t v_isShared_2469_; uint8_t v_isSharedCheck_2588_; 
v_head_2462_ = lean_ctor_get(v_x_2453_, 0);
lean_inc(v_head_2462_);
v_snd_2463_ = lean_ctor_get(v_head_2462_, 1);
v_fst_2464_ = lean_ctor_get(v_snd_2463_, 0);
lean_inc(v_fst_2464_);
v_snd_2465_ = lean_ctor_get(v_snd_2463_, 1);
lean_inc(v_snd_2465_);
v_tail_2466_ = lean_ctor_get(v_x_2453_, 1);
v_isSharedCheck_2588_ = !lean_is_exclusive(v_x_2453_);
if (v_isSharedCheck_2588_ == 0)
{
lean_object* v_unused_2589_; 
v_unused_2589_ = lean_ctor_get(v_x_2453_, 0);
lean_dec(v_unused_2589_);
v___x_2468_ = v_x_2453_;
v_isShared_2469_ = v_isSharedCheck_2588_;
goto v_resetjp_2467_;
}
else
{
lean_inc(v_tail_2466_);
lean_dec(v_x_2453_);
v___x_2468_ = lean_box(0);
v_isShared_2469_ = v_isSharedCheck_2588_;
goto v_resetjp_2467_;
}
v_resetjp_2467_:
{
lean_object* v_fst_2470_; lean_object* v_fst_2471_; lean_object* v_snd_2472_; lean_object* v_variablesKept_2473_; lean_object* v_neqs_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v_fst_2479_; lean_object* v___y_2480_; lean_object* v___y_2481_; lean_object* v___y_2482_; lean_object* v___y_2483_; 
v_fst_2470_ = lean_ctor_get(v_head_2462_, 0);
lean_inc(v_fst_2470_);
lean_dec(v_head_2462_);
v_fst_2471_ = lean_ctor_get(v_fst_2464_, 0);
lean_inc(v_fst_2471_);
v_snd_2472_ = lean_ctor_get(v_fst_2464_, 1);
lean_inc(v_snd_2472_);
lean_dec(v_fst_2464_);
v_variablesKept_2473_ = lean_ctor_get(v_snd_2465_, 0);
lean_inc_n(v_variablesKept_2473_, 2);
v_neqs_2474_ = lean_ctor_get(v_snd_2465_, 1);
lean_inc(v_neqs_2474_);
lean_dec(v_snd_2465_);
v___x_2475_ = lean_box(0);
v___x_2476_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__0(v_variablesKept_2473_, v___x_2475_);
v___x_2477_ = l_List_lengthTR___redArg(v___x_2476_);
lean_dec(v___x_2476_);
if (lean_obj_tag(v_neqs_2474_) == 0)
{
lean_object* v___x_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; 
v___x_2499_ = lean_unsigned_to_nat(1u);
v___x_2500_ = lean_nat_sub(v___x_2477_, v___x_2499_);
v___x_2501_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v___x_2500_, v_snd_2472_, v_fst_2471_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
lean_dec(v___x_2500_);
if (lean_obj_tag(v___x_2501_) == 0)
{
lean_object* v_a_2502_; lean_object* v_fst_2503_; 
v_a_2502_ = lean_ctor_get(v___x_2501_, 0);
lean_inc(v_a_2502_);
lean_dec_ref_known(v___x_2501_, 1);
v_fst_2503_ = lean_ctor_get(v_a_2502_, 0);
lean_inc(v_fst_2503_);
lean_dec(v_a_2502_);
v_fst_2479_ = v_fst_2503_;
v___y_2480_ = v___y_2455_;
v___y_2481_ = v___y_2456_;
v___y_2482_ = v___y_2457_;
v___y_2483_ = v___y_2458_;
goto v___jp_2478_;
}
else
{
lean_object* v_a_2504_; lean_object* v___x_2506_; uint8_t v_isShared_2507_; uint8_t v_isSharedCheck_2511_; 
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2504_ = lean_ctor_get(v___x_2501_, 0);
v_isSharedCheck_2511_ = !lean_is_exclusive(v___x_2501_);
if (v_isSharedCheck_2511_ == 0)
{
v___x_2506_ = v___x_2501_;
v_isShared_2507_ = v_isSharedCheck_2511_;
goto v_resetjp_2505_;
}
else
{
lean_inc(v_a_2504_);
lean_dec(v___x_2501_);
v___x_2506_ = lean_box(0);
v_isShared_2507_ = v_isSharedCheck_2511_;
goto v_resetjp_2505_;
}
v_resetjp_2505_:
{
lean_object* v___x_2509_; 
if (v_isShared_2507_ == 0)
{
v___x_2509_ = v___x_2506_;
goto v_reusejp_2508_;
}
else
{
lean_object* v_reuseFailAlloc_2510_; 
v_reuseFailAlloc_2510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2510_, 0, v_a_2504_);
v___x_2509_ = v_reuseFailAlloc_2510_;
goto v_reusejp_2508_;
}
v_reusejp_2508_:
{
return v___x_2509_;
}
}
}
}
else
{
lean_object* v_val_2512_; lean_object* v___x_2513_; lean_object* v_zero_2514_; uint8_t v_isZero_2515_; 
v_val_2512_ = lean_ctor_get(v_neqs_2474_, 0);
lean_inc(v_val_2512_);
lean_dec_ref_known(v_neqs_2474_, 1);
v___x_2513_ = lean_box(0);
v_zero_2514_ = lean_unsigned_to_nat(0u);
v_isZero_2515_ = lean_nat_dec_eq(v_val_2512_, v_zero_2514_);
if (v_isZero_2515_ == 1)
{
lean_object* v___x_2516_; 
lean_dec(v_val_2512_);
v___x_2516_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v___x_2477_, v_snd_2472_, v_fst_2471_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2516_) == 0)
{
lean_object* v_a_2517_; lean_object* v_fst_2518_; lean_object* v_snd_2519_; lean_object* v___x_2520_; lean_object* v___x_2521_; 
v_a_2517_ = lean_ctor_get(v___x_2516_, 0);
lean_inc(v_a_2517_);
lean_dec_ref_known(v___x_2516_, 1);
v_fst_2518_ = lean_ctor_get(v_a_2517_, 0);
lean_inc(v_fst_2518_);
v_snd_2519_ = lean_ctor_get(v_a_2517_, 1);
lean_inc(v_snd_2519_);
lean_dec(v_a_2517_);
v___x_2520_ = l_List_getLast_x21___redArg(v___x_2513_, v_snd_2519_);
lean_dec(v_snd_2519_);
v___x_2521_ = l_Lean_MVarId_tryClear(v_fst_2518_, v___x_2520_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2521_) == 0)
{
lean_object* v_a_2522_; 
v_a_2522_ = lean_ctor_get(v___x_2521_, 0);
lean_inc(v_a_2522_);
lean_dec_ref_known(v___x_2521_, 1);
v_fst_2479_ = v_a_2522_;
v___y_2480_ = v___y_2455_;
v___y_2481_ = v___y_2456_;
v___y_2482_ = v___y_2457_;
v___y_2483_ = v___y_2458_;
goto v___jp_2478_;
}
else
{
lean_object* v_a_2523_; lean_object* v___x_2525_; uint8_t v_isShared_2526_; uint8_t v_isSharedCheck_2530_; 
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2523_ = lean_ctor_get(v___x_2521_, 0);
v_isSharedCheck_2530_ = !lean_is_exclusive(v___x_2521_);
if (v_isSharedCheck_2530_ == 0)
{
v___x_2525_ = v___x_2521_;
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
else
{
lean_inc(v_a_2523_);
lean_dec(v___x_2521_);
v___x_2525_ = lean_box(0);
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
v_resetjp_2524_:
{
lean_object* v___x_2528_; 
if (v_isShared_2526_ == 0)
{
v___x_2528_ = v___x_2525_;
goto v_reusejp_2527_;
}
else
{
lean_object* v_reuseFailAlloc_2529_; 
v_reuseFailAlloc_2529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2529_, 0, v_a_2523_);
v___x_2528_ = v_reuseFailAlloc_2529_;
goto v_reusejp_2527_;
}
v_reusejp_2527_:
{
return v___x_2528_;
}
}
}
}
else
{
lean_object* v_a_2531_; lean_object* v___x_2533_; uint8_t v_isShared_2534_; uint8_t v_isSharedCheck_2538_; 
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2531_ = lean_ctor_get(v___x_2516_, 0);
v_isSharedCheck_2538_ = !lean_is_exclusive(v___x_2516_);
if (v_isSharedCheck_2538_ == 0)
{
v___x_2533_ = v___x_2516_;
v_isShared_2534_ = v_isSharedCheck_2538_;
goto v_resetjp_2532_;
}
else
{
lean_inc(v_a_2531_);
lean_dec(v___x_2516_);
v___x_2533_ = lean_box(0);
v_isShared_2534_ = v_isSharedCheck_2538_;
goto v_resetjp_2532_;
}
v_resetjp_2532_:
{
lean_object* v___x_2536_; 
if (v_isShared_2534_ == 0)
{
v___x_2536_ = v___x_2533_;
goto v_reusejp_2535_;
}
else
{
lean_object* v_reuseFailAlloc_2537_; 
v_reuseFailAlloc_2537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2537_, 0, v_a_2531_);
v___x_2536_ = v_reuseFailAlloc_2537_;
goto v_reusejp_2535_;
}
v_reusejp_2535_:
{
return v___x_2536_;
}
}
}
}
else
{
lean_object* v___x_2539_; 
v___x_2539_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v___x_2477_, v_snd_2472_, v_fst_2471_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2539_) == 0)
{
lean_object* v_a_2540_; lean_object* v_fst_2541_; lean_object* v_snd_2542_; lean_object* v_one_2543_; lean_object* v_n_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; 
v_a_2540_ = lean_ctor_get(v___x_2539_, 0);
lean_inc(v_a_2540_);
lean_dec_ref_known(v___x_2539_, 1);
v_fst_2541_ = lean_ctor_get(v_a_2540_, 0);
lean_inc(v_fst_2541_);
v_snd_2542_ = lean_ctor_get(v_a_2540_, 1);
lean_inc(v_snd_2542_);
lean_dec(v_a_2540_);
v_one_2543_ = lean_unsigned_to_nat(1u);
v_n_2544_ = lean_nat_sub(v_val_2512_, v_one_2543_);
lean_dec(v_val_2512_);
v___x_2545_ = l_List_getLast_x21___redArg(v___x_2513_, v_snd_2542_);
lean_dec(v_snd_2542_);
v___x_2546_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesProd(v_n_2544_, v_fst_2541_, v___x_2545_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
lean_dec(v_n_2544_);
if (lean_obj_tag(v___x_2546_) == 0)
{
lean_object* v_a_2547_; lean_object* v_fst_2548_; lean_object* v_snd_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; 
v_a_2547_ = lean_ctor_get(v___x_2546_, 0);
lean_inc(v_a_2547_);
lean_dec_ref_known(v___x_2546_, 1);
v_fst_2548_ = lean_ctor_get(v_a_2547_, 0);
lean_inc(v_fst_2548_);
v_snd_2549_ = lean_ctor_get(v_a_2547_, 1);
lean_inc_n(v_snd_2549_, 2);
lean_dec(v_a_2547_);
v___x_2550_ = lean_array_mk(v_snd_2549_);
v___x_2551_ = l_Lean_MVarId_revert(v_fst_2548_, v___x_2550_, v_isZero_2515_, v_isZero_2515_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2551_) == 0)
{
lean_object* v_a_2552_; lean_object* v_snd_2553_; lean_object* v___x_2554_; 
v_a_2552_ = lean_ctor_get(v___x_2551_, 0);
lean_inc(v_a_2552_);
lean_dec_ref_known(v___x_2551_, 1);
v_snd_2553_ = lean_ctor_get(v_a_2552_, 1);
lean_inc(v_snd_2553_);
lean_dec(v_a_2552_);
v___x_2554_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_MkIff_toInductive_spec__4(v_snd_2553_, v_snd_2549_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
lean_dec(v_snd_2549_);
if (lean_obj_tag(v___x_2554_) == 0)
{
lean_object* v_a_2555_; 
v_a_2555_ = lean_ctor_get(v___x_2554_, 0);
lean_inc(v_a_2555_);
lean_dec_ref_known(v___x_2554_, 1);
v_fst_2479_ = v_a_2555_;
v___y_2480_ = v___y_2455_;
v___y_2481_ = v___y_2456_;
v___y_2482_ = v___y_2457_;
v___y_2483_ = v___y_2458_;
goto v___jp_2478_;
}
else
{
lean_object* v_a_2556_; lean_object* v___x_2558_; uint8_t v_isShared_2559_; uint8_t v_isSharedCheck_2563_; 
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2556_ = lean_ctor_get(v___x_2554_, 0);
v_isSharedCheck_2563_ = !lean_is_exclusive(v___x_2554_);
if (v_isSharedCheck_2563_ == 0)
{
v___x_2558_ = v___x_2554_;
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
else
{
lean_inc(v_a_2556_);
lean_dec(v___x_2554_);
v___x_2558_ = lean_box(0);
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
v_resetjp_2557_:
{
lean_object* v___x_2561_; 
if (v_isShared_2559_ == 0)
{
v___x_2561_ = v___x_2558_;
goto v_reusejp_2560_;
}
else
{
lean_object* v_reuseFailAlloc_2562_; 
v_reuseFailAlloc_2562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2562_, 0, v_a_2556_);
v___x_2561_ = v_reuseFailAlloc_2562_;
goto v_reusejp_2560_;
}
v_reusejp_2560_:
{
return v___x_2561_;
}
}
}
}
else
{
lean_object* v_a_2564_; lean_object* v___x_2566_; uint8_t v_isShared_2567_; uint8_t v_isSharedCheck_2571_; 
lean_dec(v_snd_2549_);
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2564_ = lean_ctor_get(v___x_2551_, 0);
v_isSharedCheck_2571_ = !lean_is_exclusive(v___x_2551_);
if (v_isSharedCheck_2571_ == 0)
{
v___x_2566_ = v___x_2551_;
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
else
{
lean_inc(v_a_2564_);
lean_dec(v___x_2551_);
v___x_2566_ = lean_box(0);
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
v_resetjp_2565_:
{
lean_object* v___x_2569_; 
if (v_isShared_2567_ == 0)
{
v___x_2569_ = v___x_2566_;
goto v_reusejp_2568_;
}
else
{
lean_object* v_reuseFailAlloc_2570_; 
v_reuseFailAlloc_2570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2570_, 0, v_a_2564_);
v___x_2569_ = v_reuseFailAlloc_2570_;
goto v_reusejp_2568_;
}
v_reusejp_2568_:
{
return v___x_2569_;
}
}
}
}
else
{
lean_object* v_a_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2579_; 
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2572_ = lean_ctor_get(v___x_2546_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2546_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2574_ = v___x_2546_;
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_a_2572_);
lean_dec(v___x_2546_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2577_; 
if (v_isShared_2575_ == 0)
{
v___x_2577_ = v___x_2574_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v_a_2572_);
v___x_2577_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
return v___x_2577_;
}
}
}
}
else
{
lean_object* v_a_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2587_; 
lean_dec(v_val_2512_);
lean_dec(v___x_2477_);
lean_dec(v_variablesKept_2473_);
lean_dec(v_fst_2470_);
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2580_ = lean_ctor_get(v___x_2539_, 0);
v_isSharedCheck_2587_ = !lean_is_exclusive(v___x_2539_);
if (v_isSharedCheck_2587_ == 0)
{
v___x_2582_ = v___x_2539_;
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_a_2580_);
lean_dec(v___x_2539_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2583_ == 0)
{
v___x_2585_ = v___x_2582_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2586_; 
v_reuseFailAlloc_2586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2586_, 0, v_a_2580_);
v___x_2585_ = v_reuseFailAlloc_2586_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
return v___x_2585_;
}
}
}
}
}
v___jp_2478_:
{
lean_object* v___f_2484_; lean_object* v___x_2485_; 
lean_inc(v_fst_2479_);
lean_inc(v_gs_2452_);
v___f_2484_ = lean_alloc_closure((void*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___boxed), 10, 5);
lean_closure_set(v___f_2484_, 0, v_gs_2452_);
lean_closure_set(v___f_2484_, 1, v___x_2477_);
lean_closure_set(v___f_2484_, 2, v_variablesKept_2473_);
lean_closure_set(v___f_2484_, 3, v_fst_2470_);
lean_closure_set(v___f_2484_, 4, v_fst_2479_);
v___x_2485_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_MkIff_toInductive_spec__3___redArg(v_fst_2479_, v___f_2484_, v___y_2480_, v___y_2481_, v___y_2482_, v___y_2483_);
if (lean_obj_tag(v___x_2485_) == 0)
{
lean_object* v_a_2486_; lean_object* v___x_2488_; 
v_a_2486_ = lean_ctor_get(v___x_2485_, 0);
lean_inc(v_a_2486_);
lean_dec_ref_known(v___x_2485_, 1);
if (v_isShared_2469_ == 0)
{
lean_ctor_set(v___x_2468_, 1, v_x_2454_);
lean_ctor_set(v___x_2468_, 0, v_a_2486_);
v___x_2488_ = v___x_2468_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2490_; 
v_reuseFailAlloc_2490_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2490_, 0, v_a_2486_);
lean_ctor_set(v_reuseFailAlloc_2490_, 1, v_x_2454_);
v___x_2488_ = v_reuseFailAlloc_2490_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
v_x_2453_ = v_tail_2466_;
v_x_2454_ = v___x_2488_;
goto _start;
}
}
else
{
lean_object* v_a_2491_; lean_object* v___x_2493_; uint8_t v_isShared_2494_; uint8_t v_isSharedCheck_2498_; 
lean_del_object(v___x_2468_);
lean_dec(v_tail_2466_);
lean_dec(v_x_2454_);
lean_dec(v_gs_2452_);
v_a_2491_ = lean_ctor_get(v___x_2485_, 0);
v_isSharedCheck_2498_ = !lean_is_exclusive(v___x_2485_);
if (v_isSharedCheck_2498_ == 0)
{
v___x_2493_ = v___x_2485_;
v_isShared_2494_ = v_isSharedCheck_2498_;
goto v_resetjp_2492_;
}
else
{
lean_inc(v_a_2491_);
lean_dec(v___x_2485_);
v___x_2493_ = lean_box(0);
v_isShared_2494_ = v_isSharedCheck_2498_;
goto v_resetjp_2492_;
}
v_resetjp_2492_:
{
lean_object* v___x_2496_; 
if (v_isShared_2494_ == 0)
{
v___x_2496_ = v___x_2493_;
goto v_reusejp_2495_;
}
else
{
lean_object* v_reuseFailAlloc_2497_; 
v_reuseFailAlloc_2497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2497_, 0, v_a_2491_);
v___x_2496_ = v_reuseFailAlloc_2497_;
goto v_reusejp_2495_;
}
v_reusejp_2495_:
{
return v___x_2496_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___boxed(lean_object* v_gs_2590_, lean_object* v_x_2591_, lean_object* v_x_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_){
_start:
{
lean_object* v_res_2598_; 
v_res_2598_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5(v_gs_2590_, v_x_2591_, v_x_2592_, v___y_2593_, v___y_2594_, v___y_2595_, v___y_2596_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2595_);
lean_dec(v___y_2594_);
lean_dec_ref(v___y_2593_);
return v_res_2598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toInductive(lean_object* v_mvar_2599_, lean_object* v_cs_2600_, lean_object* v_gs_2601_, lean_object* v_s_2602_, lean_object* v_h_2603_, lean_object* v_a_2604_, lean_object* v_a_2605_, lean_object* v_a_2606_, lean_object* v_a_2607_){
_start:
{
lean_object* v___x_2609_; lean_object* v_zero_2610_; uint8_t v_isZero_2611_; 
v___x_2609_ = l_List_lengthTR___redArg(v_s_2602_);
v_zero_2610_ = lean_unsigned_to_nat(0u);
v_isZero_2611_ = lean_nat_dec_eq(v___x_2609_, v_zero_2610_);
if (v_isZero_2611_ == 1)
{
lean_object* v___x_2612_; uint8_t v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; 
lean_dec(v___x_2609_);
lean_dec(v_s_2602_);
lean_dec(v_gs_2601_);
lean_dec(v_cs_2600_);
v___x_2612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_toCases___closed__0));
v___x_2613_ = 0;
v___x_2614_ = lean_box(0);
v___x_2615_ = l_Lean_MVarId_cases(v_mvar_2599_, v_h_2603_, v___x_2612_, v___x_2613_, v___x_2614_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_);
if (lean_obj_tag(v___x_2615_) == 0)
{
lean_object* v___x_2617_; uint8_t v_isShared_2618_; uint8_t v_isSharedCheck_2623_; 
v_isSharedCheck_2623_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2623_ == 0)
{
lean_object* v_unused_2624_; 
v_unused_2624_ = lean_ctor_get(v___x_2615_, 0);
lean_dec(v_unused_2624_);
v___x_2617_ = v___x_2615_;
v_isShared_2618_ = v_isSharedCheck_2623_;
goto v_resetjp_2616_;
}
else
{
lean_dec(v___x_2615_);
v___x_2617_ = lean_box(0);
v_isShared_2618_ = v_isSharedCheck_2623_;
goto v_resetjp_2616_;
}
v_resetjp_2616_:
{
lean_object* v___x_2619_; lean_object* v___x_2621_; 
v___x_2619_ = lean_box(0);
if (v_isShared_2618_ == 0)
{
lean_ctor_set(v___x_2617_, 0, v___x_2619_);
v___x_2621_ = v___x_2617_;
goto v_reusejp_2620_;
}
else
{
lean_object* v_reuseFailAlloc_2622_; 
v_reuseFailAlloc_2622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2622_, 0, v___x_2619_);
v___x_2621_ = v_reuseFailAlloc_2622_;
goto v_reusejp_2620_;
}
v_reusejp_2620_:
{
return v___x_2621_;
}
}
}
else
{
lean_object* v_a_2625_; lean_object* v___x_2627_; uint8_t v_isShared_2628_; uint8_t v_isSharedCheck_2632_; 
v_a_2625_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2632_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2632_ == 0)
{
v___x_2627_ = v___x_2615_;
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
else
{
lean_inc(v_a_2625_);
lean_dec(v___x_2615_);
v___x_2627_ = lean_box(0);
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
v_resetjp_2626_:
{
lean_object* v___x_2630_; 
if (v_isShared_2628_ == 0)
{
v___x_2630_ = v___x_2627_;
goto v_reusejp_2629_;
}
else
{
lean_object* v_reuseFailAlloc_2631_; 
v_reuseFailAlloc_2631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2631_, 0, v_a_2625_);
v___x_2630_ = v_reuseFailAlloc_2631_;
goto v_reusejp_2629_;
}
v_reusejp_2629_:
{
return v___x_2630_;
}
}
}
}
else
{
lean_object* v_one_2633_; lean_object* v_n_2634_; lean_object* v___x_2635_; 
v_one_2633_ = lean_unsigned_to_nat(1u);
v_n_2634_ = lean_nat_sub(v___x_2609_, v_one_2633_);
lean_dec(v___x_2609_);
v___x_2635_ = lp_mathlib_Mathlib_Tactic_MkIff_nCasesSum(v_n_2634_, v_mvar_2599_, v_h_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_);
lean_dec(v_n_2634_);
if (lean_obj_tag(v___x_2635_) == 0)
{
lean_object* v_a_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; 
v_a_2636_ = lean_ctor_get(v___x_2635_, 0);
lean_inc(v_a_2636_);
lean_dec_ref_known(v___x_2635_, 1);
v___x_2637_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_a_2636_, v_s_2602_);
v___x_2638_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_cs_2600_, v___x_2637_);
v___x_2639_ = lean_box(0);
v___x_2640_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5(v_gs_2601_, v___x_2638_, v___x_2639_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_);
if (lean_obj_tag(v___x_2640_) == 0)
{
lean_object* v___x_2642_; uint8_t v_isShared_2643_; uint8_t v_isSharedCheck_2648_; 
v_isSharedCheck_2648_ = !lean_is_exclusive(v___x_2640_);
if (v_isSharedCheck_2648_ == 0)
{
lean_object* v_unused_2649_; 
v_unused_2649_ = lean_ctor_get(v___x_2640_, 0);
lean_dec(v_unused_2649_);
v___x_2642_ = v___x_2640_;
v_isShared_2643_ = v_isSharedCheck_2648_;
goto v_resetjp_2641_;
}
else
{
lean_dec(v___x_2640_);
v___x_2642_ = lean_box(0);
v_isShared_2643_ = v_isSharedCheck_2648_;
goto v_resetjp_2641_;
}
v_resetjp_2641_:
{
lean_object* v___x_2644_; lean_object* v___x_2646_; 
v___x_2644_ = lean_box(0);
if (v_isShared_2643_ == 0)
{
lean_ctor_set(v___x_2642_, 0, v___x_2644_);
v___x_2646_ = v___x_2642_;
goto v_reusejp_2645_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2647_, 0, v___x_2644_);
v___x_2646_ = v_reuseFailAlloc_2647_;
goto v_reusejp_2645_;
}
v_reusejp_2645_:
{
return v___x_2646_;
}
}
}
else
{
lean_object* v_a_2650_; lean_object* v___x_2652_; uint8_t v_isShared_2653_; uint8_t v_isSharedCheck_2657_; 
v_a_2650_ = lean_ctor_get(v___x_2640_, 0);
v_isSharedCheck_2657_ = !lean_is_exclusive(v___x_2640_);
if (v_isSharedCheck_2657_ == 0)
{
v___x_2652_ = v___x_2640_;
v_isShared_2653_ = v_isSharedCheck_2657_;
goto v_resetjp_2651_;
}
else
{
lean_inc(v_a_2650_);
lean_dec(v___x_2640_);
v___x_2652_ = lean_box(0);
v_isShared_2653_ = v_isSharedCheck_2657_;
goto v_resetjp_2651_;
}
v_resetjp_2651_:
{
lean_object* v___x_2655_; 
if (v_isShared_2653_ == 0)
{
v___x_2655_ = v___x_2652_;
goto v_reusejp_2654_;
}
else
{
lean_object* v_reuseFailAlloc_2656_; 
v_reuseFailAlloc_2656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2656_, 0, v_a_2650_);
v___x_2655_ = v_reuseFailAlloc_2656_;
goto v_reusejp_2654_;
}
v_reusejp_2654_:
{
return v___x_2655_;
}
}
}
}
else
{
lean_object* v_a_2658_; lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2665_; 
lean_dec(v_s_2602_);
lean_dec(v_gs_2601_);
lean_dec(v_cs_2600_);
v_a_2658_ = lean_ctor_get(v___x_2635_, 0);
v_isSharedCheck_2665_ = !lean_is_exclusive(v___x_2635_);
if (v_isSharedCheck_2665_ == 0)
{
v___x_2660_ = v___x_2635_;
v_isShared_2661_ = v_isSharedCheck_2665_;
goto v_resetjp_2659_;
}
else
{
lean_inc(v_a_2658_);
lean_dec(v___x_2635_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2665_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v___x_2663_; 
if (v_isShared_2661_ == 0)
{
v___x_2663_ = v___x_2660_;
goto v_reusejp_2662_;
}
else
{
lean_object* v_reuseFailAlloc_2664_; 
v_reuseFailAlloc_2664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2664_, 0, v_a_2658_);
v___x_2663_ = v_reuseFailAlloc_2664_;
goto v_reusejp_2662_;
}
v_reusejp_2662_:
{
return v___x_2663_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_toInductive___boxed(lean_object* v_mvar_2666_, lean_object* v_cs_2667_, lean_object* v_gs_2668_, lean_object* v_s_2669_, lean_object* v_h_2670_, lean_object* v_a_2671_, lean_object* v_a_2672_, lean_object* v_a_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_){
_start:
{
lean_object* v_res_2676_; 
v_res_2676_ = lp_mathlib_Mathlib_Tactic_MkIff_toInductive(v_mvar_2666_, v_cs_2667_, v_gs_2668_, v_s_2669_, v_h_2670_, v_a_2671_, v_a_2672_, v_a_2673_, v_a_2674_);
lean_dec(v_a_2674_);
lean_dec_ref(v_a_2673_);
lean_dec(v_a_2672_);
lean_dec_ref(v_a_2671_);
return v_res_2676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg(lean_object* v_e_2677_, lean_object* v___y_2678_){
_start:
{
uint8_t v___x_2680_; 
v___x_2680_ = l_Lean_Expr_hasMVar(v_e_2677_);
if (v___x_2680_ == 0)
{
lean_object* v___x_2681_; 
v___x_2681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2681_, 0, v_e_2677_);
return v___x_2681_;
}
else
{
lean_object* v___x_2682_; lean_object* v_mctx_2683_; lean_object* v___x_2684_; lean_object* v_fst_2685_; lean_object* v_snd_2686_; lean_object* v___x_2687_; lean_object* v_cache_2688_; lean_object* v_zetaDeltaFVarIds_2689_; lean_object* v_postponed_2690_; lean_object* v_diag_2691_; lean_object* v___x_2693_; uint8_t v_isShared_2694_; uint8_t v_isSharedCheck_2700_; 
v___x_2682_ = lean_st_ref_get(v___y_2678_);
v_mctx_2683_ = lean_ctor_get(v___x_2682_, 0);
lean_inc_ref(v_mctx_2683_);
lean_dec(v___x_2682_);
v___x_2684_ = l_Lean_instantiateMVarsCore(v_mctx_2683_, v_e_2677_);
v_fst_2685_ = lean_ctor_get(v___x_2684_, 0);
lean_inc(v_fst_2685_);
v_snd_2686_ = lean_ctor_get(v___x_2684_, 1);
lean_inc(v_snd_2686_);
lean_dec_ref(v___x_2684_);
v___x_2687_ = lean_st_ref_take(v___y_2678_);
v_cache_2688_ = lean_ctor_get(v___x_2687_, 1);
v_zetaDeltaFVarIds_2689_ = lean_ctor_get(v___x_2687_, 2);
v_postponed_2690_ = lean_ctor_get(v___x_2687_, 3);
v_diag_2691_ = lean_ctor_get(v___x_2687_, 4);
v_isSharedCheck_2700_ = !lean_is_exclusive(v___x_2687_);
if (v_isSharedCheck_2700_ == 0)
{
lean_object* v_unused_2701_; 
v_unused_2701_ = lean_ctor_get(v___x_2687_, 0);
lean_dec(v_unused_2701_);
v___x_2693_ = v___x_2687_;
v_isShared_2694_ = v_isSharedCheck_2700_;
goto v_resetjp_2692_;
}
else
{
lean_inc(v_diag_2691_);
lean_inc(v_postponed_2690_);
lean_inc(v_zetaDeltaFVarIds_2689_);
lean_inc(v_cache_2688_);
lean_dec(v___x_2687_);
v___x_2693_ = lean_box(0);
v_isShared_2694_ = v_isSharedCheck_2700_;
goto v_resetjp_2692_;
}
v_resetjp_2692_:
{
lean_object* v___x_2696_; 
if (v_isShared_2694_ == 0)
{
lean_ctor_set(v___x_2693_, 0, v_snd_2686_);
v___x_2696_ = v___x_2693_;
goto v_reusejp_2695_;
}
else
{
lean_object* v_reuseFailAlloc_2699_; 
v_reuseFailAlloc_2699_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2699_, 0, v_snd_2686_);
lean_ctor_set(v_reuseFailAlloc_2699_, 1, v_cache_2688_);
lean_ctor_set(v_reuseFailAlloc_2699_, 2, v_zetaDeltaFVarIds_2689_);
lean_ctor_set(v_reuseFailAlloc_2699_, 3, v_postponed_2690_);
lean_ctor_set(v_reuseFailAlloc_2699_, 4, v_diag_2691_);
v___x_2696_ = v_reuseFailAlloc_2699_;
goto v_reusejp_2695_;
}
v_reusejp_2695_:
{
lean_object* v___x_2697_; lean_object* v___x_2698_; 
v___x_2697_ = lean_st_ref_set(v___y_2678_, v___x_2696_);
v___x_2698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2698_, 0, v_fst_2685_);
return v___x_2698_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg___boxed(lean_object* v_e_2702_, lean_object* v___y_2703_, lean_object* v___y_2704_){
_start:
{
lean_object* v_res_2705_; 
v_res_2705_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg(v_e_2702_, v___y_2703_);
lean_dec(v___y_2703_);
return v_res_2705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3(lean_object* v_e_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_){
_start:
{
lean_object* v___x_2712_; 
v___x_2712_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg(v_e_2706_, v___y_2708_);
return v___x_2712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___boxed(lean_object* v_e_2713_, lean_object* v___y_2714_, lean_object* v___y_2715_, lean_object* v___y_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_){
_start:
{
lean_object* v_res_2719_; 
v_res_2719_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3(v_e_2713_, v___y_2714_, v___y_2715_, v___y_2716_, v___y_2717_);
lean_dec(v___y_2717_);
lean_dec_ref(v___y_2716_);
lean_dec(v___y_2715_);
lean_dec_ref(v___y_2714_);
return v_res_2719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1(lean_object* v___x_2720_, lean_object* v___x_2721_, lean_object* v___x_2722_, lean_object* v_x_2723_, lean_object* v_x_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_){
_start:
{
if (lean_obj_tag(v_x_2723_) == 0)
{
lean_object* v___x_2730_; lean_object* v___x_2731_; 
lean_dec(v___x_2722_);
lean_dec(v___x_2721_);
lean_dec(v___x_2720_);
v___x_2730_ = l_List_reverse___redArg(v_x_2724_);
v___x_2731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2731_, 0, v___x_2730_);
return v___x_2731_;
}
else
{
lean_object* v_head_2732_; lean_object* v_tail_2733_; lean_object* v___x_2735_; uint8_t v_isShared_2736_; uint8_t v_isSharedCheck_2751_; 
v_head_2732_ = lean_ctor_get(v_x_2723_, 0);
v_tail_2733_ = lean_ctor_get(v_x_2723_, 1);
v_isSharedCheck_2751_ = !lean_is_exclusive(v_x_2723_);
if (v_isSharedCheck_2751_ == 0)
{
v___x_2735_ = v_x_2723_;
v_isShared_2736_ = v_isSharedCheck_2751_;
goto v_resetjp_2734_;
}
else
{
lean_inc(v_tail_2733_);
lean_inc(v_head_2732_);
lean_dec(v_x_2723_);
v___x_2735_ = lean_box(0);
v_isShared_2736_ = v_isSharedCheck_2751_;
goto v_resetjp_2734_;
}
v_resetjp_2734_:
{
lean_object* v___x_2737_; 
lean_inc(v___x_2722_);
lean_inc(v___x_2721_);
lean_inc(v___x_2720_);
v___x_2737_ = lp_mathlib_Mathlib_Tactic_MkIff_constrToProp(v___x_2720_, v___x_2721_, v___x_2722_, v_head_2732_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_);
if (lean_obj_tag(v___x_2737_) == 0)
{
lean_object* v_a_2738_; lean_object* v___x_2740_; 
v_a_2738_ = lean_ctor_get(v___x_2737_, 0);
lean_inc(v_a_2738_);
lean_dec_ref_known(v___x_2737_, 1);
if (v_isShared_2736_ == 0)
{
lean_ctor_set(v___x_2735_, 1, v_x_2724_);
lean_ctor_set(v___x_2735_, 0, v_a_2738_);
v___x_2740_ = v___x_2735_;
goto v_reusejp_2739_;
}
else
{
lean_object* v_reuseFailAlloc_2742_; 
v_reuseFailAlloc_2742_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2742_, 0, v_a_2738_);
lean_ctor_set(v_reuseFailAlloc_2742_, 1, v_x_2724_);
v___x_2740_ = v_reuseFailAlloc_2742_;
goto v_reusejp_2739_;
}
v_reusejp_2739_:
{
v_x_2723_ = v_tail_2733_;
v_x_2724_ = v___x_2740_;
goto _start;
}
}
else
{
lean_object* v_a_2743_; lean_object* v___x_2745_; uint8_t v_isShared_2746_; uint8_t v_isSharedCheck_2750_; 
lean_del_object(v___x_2735_);
lean_dec(v_tail_2733_);
lean_dec(v_x_2724_);
lean_dec(v___x_2722_);
lean_dec(v___x_2721_);
lean_dec(v___x_2720_);
v_a_2743_ = lean_ctor_get(v___x_2737_, 0);
v_isSharedCheck_2750_ = !lean_is_exclusive(v___x_2737_);
if (v_isSharedCheck_2750_ == 0)
{
v___x_2745_ = v___x_2737_;
v_isShared_2746_ = v_isSharedCheck_2750_;
goto v_resetjp_2744_;
}
else
{
lean_inc(v_a_2743_);
lean_dec(v___x_2737_);
v___x_2745_ = lean_box(0);
v_isShared_2746_ = v_isSharedCheck_2750_;
goto v_resetjp_2744_;
}
v_resetjp_2744_:
{
lean_object* v___x_2748_; 
if (v_isShared_2746_ == 0)
{
v___x_2748_ = v___x_2745_;
goto v_reusejp_2747_;
}
else
{
lean_object* v_reuseFailAlloc_2749_; 
v_reuseFailAlloc_2749_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2749_, 0, v_a_2743_);
v___x_2748_ = v_reuseFailAlloc_2749_;
goto v_reusejp_2747_;
}
v_reusejp_2747_:
{
return v___x_2748_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1___boxed(lean_object* v___x_2752_, lean_object* v___x_2753_, lean_object* v___x_2754_, lean_object* v_x_2755_, lean_object* v_x_2756_, lean_object* v___y_2757_, lean_object* v___y_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_, lean_object* v___y_2761_){
_start:
{
lean_object* v_res_2762_; 
v_res_2762_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1(v___x_2752_, v___x_2753_, v___x_2754_, v_x_2755_, v_x_2756_, v___y_2757_, v___y_2758_, v___y_2759_, v___y_2760_);
lean_dec(v___y_2760_);
lean_dec_ref(v___y_2759_);
lean_dec(v___y_2758_);
lean_dec_ref(v___y_2757_);
return v_res_2762_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3(void){
_start:
{
lean_object* v___x_2767_; lean_object* v___x_2768_; 
v___x_2767_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__2));
v___x_2768_ = l_Lean_stringToMessageData(v___x_2767_);
return v___x_2768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0(lean_object* v_ind_2769_, lean_object* v___x_2770_, lean_object* v_numParams_2771_, lean_object* v_ctors_2772_, lean_object* v___x_2773_, lean_object* v_fvars_2774_, lean_object* v_ty_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_){
_start:
{
uint8_t v___x_2833_; 
v___x_2833_ = l_Lean_Expr_isProp(v_ty_2775_);
if (v___x_2833_ == 0)
{
lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v_a_2836_; lean_object* v___x_2838_; uint8_t v_isShared_2839_; uint8_t v_isSharedCheck_2843_; 
lean_dec_ref(v_fvars_2774_);
lean_dec(v___x_2773_);
lean_dec(v_ctors_2772_);
lean_dec(v_numParams_2771_);
lean_dec(v___x_2770_);
lean_dec(v_ind_2769_);
v___x_2834_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__3);
v___x_2835_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_2834_, v___y_2776_, v___y_2777_, v___y_2778_, v___y_2779_);
v_a_2836_ = lean_ctor_get(v___x_2835_, 0);
v_isSharedCheck_2843_ = !lean_is_exclusive(v___x_2835_);
if (v_isSharedCheck_2843_ == 0)
{
v___x_2838_ = v___x_2835_;
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
else
{
lean_inc(v_a_2836_);
lean_dec(v___x_2835_);
v___x_2838_ = lean_box(0);
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
v_resetjp_2837_:
{
lean_object* v___x_2841_; 
if (v_isShared_2839_ == 0)
{
v___x_2841_ = v___x_2838_;
goto v_reusejp_2840_;
}
else
{
lean_object* v_reuseFailAlloc_2842_; 
v_reuseFailAlloc_2842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2842_, 0, v_a_2836_);
v___x_2841_ = v_reuseFailAlloc_2842_;
goto v_reusejp_2840_;
}
v_reusejp_2840_:
{
return v___x_2841_;
}
}
}
else
{
goto v___jp_2781_;
}
v___jp_2781_:
{
lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; 
lean_inc(v___x_2770_);
v___x_2782_ = l_Lean_mkConst(v_ind_2769_, v___x_2770_);
v___x_2783_ = l_Lean_mkAppN(v___x_2782_, v_fvars_2774_);
lean_inc_ref(v_fvars_2774_);
v___x_2784_ = lean_array_to_list(v_fvars_2774_);
v___x_2785_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_constrToProp___lam__1___closed__1));
lean_inc(v_numParams_2771_);
lean_inc(v___x_2784_);
v___x_2786_ = l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_box(0), v___x_2784_, v___x_2784_, v_numParams_2771_, v___x_2785_);
v___x_2787_ = l_List_drop___redArg(v_numParams_2771_, v___x_2784_);
lean_dec(v___x_2784_);
v___x_2788_ = lean_box(0);
v___x_2789_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__1(v___x_2770_, v___x_2786_, v___x_2787_, v_ctors_2772_, v___x_2788_, v___y_2776_, v___y_2777_, v___y_2778_, v___y_2779_);
if (lean_obj_tag(v___x_2789_) == 0)
{
lean_object* v_a_2790_; lean_object* v___x_2791_; lean_object* v_fst_2792_; lean_object* v_snd_2793_; lean_object* v___x_2795_; uint8_t v_isShared_2796_; uint8_t v_isSharedCheck_2824_; 
v_a_2790_ = lean_ctor_get(v___x_2789_, 0);
lean_inc(v_a_2790_);
lean_dec_ref_known(v___x_2789_, 1);
v___x_2791_ = l_List_unzipTR___redArg(v_a_2790_);
v_fst_2792_ = lean_ctor_get(v___x_2791_, 0);
v_snd_2793_ = lean_ctor_get(v___x_2791_, 1);
v_isSharedCheck_2824_ = !lean_is_exclusive(v___x_2791_);
if (v_isSharedCheck_2824_ == 0)
{
v___x_2795_ = v___x_2791_;
v_isShared_2796_ = v_isSharedCheck_2824_;
goto v_resetjp_2794_;
}
else
{
lean_inc(v_snd_2793_);
lean_inc(v_fst_2792_);
lean_dec(v___x_2791_);
v___x_2795_ = lean_box(0);
v_isShared_2796_ = v_isSharedCheck_2824_;
goto v_resetjp_2794_;
}
v_resetjp_2794_:
{
lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; uint8_t v___x_2801_; uint8_t v___x_2802_; uint8_t v___x_2803_; lean_object* v___x_2804_; 
v___x_2797_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___closed__1));
v___x_2798_ = l_Lean_mkConst(v___x_2797_, v___x_2773_);
v___x_2799_ = lp_mathlib_Mathlib_Tactic_MkIff_mkOrList(v_snd_2793_);
v___x_2800_ = l_Lean_mkAppB(v___x_2798_, v___x_2783_, v___x_2799_);
v___x_2801_ = 0;
v___x_2802_ = 1;
v___x_2803_ = 1;
v___x_2804_ = l_Lean_Meta_mkForallFVars(v_fvars_2774_, v___x_2800_, v___x_2801_, v___x_2802_, v___x_2802_, v___x_2803_, v___y_2776_, v___y_2777_, v___y_2778_, v___y_2779_);
lean_dec_ref(v_fvars_2774_);
if (lean_obj_tag(v___x_2804_) == 0)
{
lean_object* v_a_2805_; lean_object* v___x_2807_; uint8_t v_isShared_2808_; uint8_t v_isSharedCheck_2815_; 
v_a_2805_ = lean_ctor_get(v___x_2804_, 0);
v_isSharedCheck_2815_ = !lean_is_exclusive(v___x_2804_);
if (v_isSharedCheck_2815_ == 0)
{
v___x_2807_ = v___x_2804_;
v_isShared_2808_ = v_isSharedCheck_2815_;
goto v_resetjp_2806_;
}
else
{
lean_inc(v_a_2805_);
lean_dec(v___x_2804_);
v___x_2807_ = lean_box(0);
v_isShared_2808_ = v_isSharedCheck_2815_;
goto v_resetjp_2806_;
}
v_resetjp_2806_:
{
lean_object* v___x_2810_; 
if (v_isShared_2796_ == 0)
{
lean_ctor_set(v___x_2795_, 1, v_fst_2792_);
lean_ctor_set(v___x_2795_, 0, v_a_2805_);
v___x_2810_ = v___x_2795_;
goto v_reusejp_2809_;
}
else
{
lean_object* v_reuseFailAlloc_2814_; 
v_reuseFailAlloc_2814_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2814_, 0, v_a_2805_);
lean_ctor_set(v_reuseFailAlloc_2814_, 1, v_fst_2792_);
v___x_2810_ = v_reuseFailAlloc_2814_;
goto v_reusejp_2809_;
}
v_reusejp_2809_:
{
lean_object* v___x_2812_; 
if (v_isShared_2808_ == 0)
{
lean_ctor_set(v___x_2807_, 0, v___x_2810_);
v___x_2812_ = v___x_2807_;
goto v_reusejp_2811_;
}
else
{
lean_object* v_reuseFailAlloc_2813_; 
v_reuseFailAlloc_2813_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2813_, 0, v___x_2810_);
v___x_2812_ = v_reuseFailAlloc_2813_;
goto v_reusejp_2811_;
}
v_reusejp_2811_:
{
return v___x_2812_;
}
}
}
}
else
{
lean_object* v_a_2816_; lean_object* v___x_2818_; uint8_t v_isShared_2819_; uint8_t v_isSharedCheck_2823_; 
lean_del_object(v___x_2795_);
lean_dec(v_fst_2792_);
v_a_2816_ = lean_ctor_get(v___x_2804_, 0);
v_isSharedCheck_2823_ = !lean_is_exclusive(v___x_2804_);
if (v_isSharedCheck_2823_ == 0)
{
v___x_2818_ = v___x_2804_;
v_isShared_2819_ = v_isSharedCheck_2823_;
goto v_resetjp_2817_;
}
else
{
lean_inc(v_a_2816_);
lean_dec(v___x_2804_);
v___x_2818_ = lean_box(0);
v_isShared_2819_ = v_isSharedCheck_2823_;
goto v_resetjp_2817_;
}
v_resetjp_2817_:
{
lean_object* v___x_2821_; 
if (v_isShared_2819_ == 0)
{
v___x_2821_ = v___x_2818_;
goto v_reusejp_2820_;
}
else
{
lean_object* v_reuseFailAlloc_2822_; 
v_reuseFailAlloc_2822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2822_, 0, v_a_2816_);
v___x_2821_ = v_reuseFailAlloc_2822_;
goto v_reusejp_2820_;
}
v_reusejp_2820_:
{
return v___x_2821_;
}
}
}
}
}
else
{
lean_object* v_a_2825_; lean_object* v___x_2827_; uint8_t v_isShared_2828_; uint8_t v_isSharedCheck_2832_; 
lean_dec_ref(v___x_2783_);
lean_dec_ref(v_fvars_2774_);
lean_dec(v___x_2773_);
v_a_2825_ = lean_ctor_get(v___x_2789_, 0);
v_isSharedCheck_2832_ = !lean_is_exclusive(v___x_2789_);
if (v_isSharedCheck_2832_ == 0)
{
v___x_2827_ = v___x_2789_;
v_isShared_2828_ = v_isSharedCheck_2832_;
goto v_resetjp_2826_;
}
else
{
lean_inc(v_a_2825_);
lean_dec(v___x_2789_);
v___x_2827_ = lean_box(0);
v_isShared_2828_ = v_isSharedCheck_2832_;
goto v_resetjp_2826_;
}
v_resetjp_2826_:
{
lean_object* v___x_2830_; 
if (v_isShared_2828_ == 0)
{
v___x_2830_ = v___x_2827_;
goto v_reusejp_2829_;
}
else
{
lean_object* v_reuseFailAlloc_2831_; 
v_reuseFailAlloc_2831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2831_, 0, v_a_2825_);
v___x_2830_ = v_reuseFailAlloc_2831_;
goto v_reusejp_2829_;
}
v_reusejp_2829_:
{
return v___x_2830_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___boxed(lean_object* v_ind_2844_, lean_object* v___x_2845_, lean_object* v_numParams_2846_, lean_object* v_ctors_2847_, lean_object* v___x_2848_, lean_object* v_fvars_2849_, lean_object* v_ty_2850_, lean_object* v___y_2851_, lean_object* v___y_2852_, lean_object* v___y_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_){
_start:
{
lean_object* v_res_2856_; 
v_res_2856_ = lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0(v_ind_2844_, v___x_2845_, v_numParams_2846_, v_ctors_2847_, v___x_2848_, v_fvars_2849_, v_ty_2850_, v___y_2851_, v___y_2852_, v___y_2853_, v___y_2854_);
lean_dec(v___y_2854_);
lean_dec_ref(v___y_2853_);
lean_dec(v___y_2852_);
lean_dec_ref(v___y_2851_);
lean_dec_ref(v_ty_2850_);
return v_res_2856_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1(uint8_t v___x_2857_, lean_object* v_x_2858_){
_start:
{
return v___x_2857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1___boxed(lean_object* v___x_2859_, lean_object* v_x_2860_){
_start:
{
uint8_t v___x_7272__boxed_2861_; uint8_t v_res_2862_; lean_object* v_r_2863_; 
v___x_7272__boxed_2861_ = lean_unbox(v___x_2859_);
v_res_2862_ = lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__1(v___x_7272__boxed_2861_, v_x_2860_);
lean_dec(v_x_2860_);
v_r_2863_ = lean_box(v_res_2862_);
return v_r_2863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__2(lean_object* v_a_2864_, lean_object* v_a_2865_){
_start:
{
if (lean_obj_tag(v_a_2864_) == 0)
{
lean_object* v___x_2866_; 
v___x_2866_ = l_List_reverse___redArg(v_a_2865_);
return v___x_2866_;
}
else
{
lean_object* v_head_2867_; lean_object* v_tail_2868_; lean_object* v___x_2870_; uint8_t v_isShared_2871_; uint8_t v_isSharedCheck_2877_; 
v_head_2867_ = lean_ctor_get(v_a_2864_, 0);
v_tail_2868_ = lean_ctor_get(v_a_2864_, 1);
v_isSharedCheck_2877_ = !lean_is_exclusive(v_a_2864_);
if (v_isSharedCheck_2877_ == 0)
{
v___x_2870_ = v_a_2864_;
v_isShared_2871_ = v_isSharedCheck_2877_;
goto v_resetjp_2869_;
}
else
{
lean_inc(v_tail_2868_);
lean_inc(v_head_2867_);
lean_dec(v_a_2864_);
v___x_2870_ = lean_box(0);
v_isShared_2871_ = v_isSharedCheck_2877_;
goto v_resetjp_2869_;
}
v_resetjp_2869_:
{
lean_object* v___x_2872_; lean_object* v___x_2874_; 
v___x_2872_ = l_Lean_Expr_fvar___override(v_head_2867_);
if (v_isShared_2871_ == 0)
{
lean_ctor_set(v___x_2870_, 1, v_a_2865_);
lean_ctor_set(v___x_2870_, 0, v___x_2872_);
v___x_2874_ = v___x_2870_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2876_; 
v_reuseFailAlloc_2876_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2876_, 0, v___x_2872_);
lean_ctor_set(v_reuseFailAlloc_2876_, 1, v_a_2865_);
v___x_2874_ = v_reuseFailAlloc_2876_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
v_a_2864_ = v_tail_2868_;
v_a_2865_ = v___x_2874_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7(lean_object* v_constName_2878_, lean_object* v___y_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_){
_start:
{
lean_object* v___x_2884_; lean_object* v_env_2885_; uint8_t v___x_2886_; lean_object* v___x_2887_; 
v___x_2884_ = lean_st_ref_get(v___y_2882_);
v_env_2885_ = lean_ctor_get(v___x_2884_, 0);
lean_inc_ref(v_env_2885_);
lean_dec(v___x_2884_);
v___x_2886_ = 0;
lean_inc(v_constName_2878_);
v___x_2887_ = l_Lean_Environment_findConstVal_x3f(v_env_2885_, v_constName_2878_, v___x_2886_);
if (lean_obj_tag(v___x_2887_) == 0)
{
lean_object* v___x_2888_; 
v___x_2888_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0___redArg(v_constName_2878_, v___y_2879_, v___y_2880_, v___y_2881_, v___y_2882_);
return v___x_2888_;
}
else
{
lean_object* v_val_2889_; lean_object* v___x_2891_; uint8_t v_isShared_2892_; uint8_t v_isSharedCheck_2896_; 
lean_dec(v_constName_2878_);
v_val_2889_ = lean_ctor_get(v___x_2887_, 0);
v_isSharedCheck_2896_ = !lean_is_exclusive(v___x_2887_);
if (v_isSharedCheck_2896_ == 0)
{
v___x_2891_ = v___x_2887_;
v_isShared_2892_ = v_isSharedCheck_2896_;
goto v_resetjp_2890_;
}
else
{
lean_inc(v_val_2889_);
lean_dec(v___x_2887_);
v___x_2891_ = lean_box(0);
v_isShared_2892_ = v_isSharedCheck_2896_;
goto v_resetjp_2890_;
}
v_resetjp_2890_:
{
lean_object* v___x_2894_; 
if (v_isShared_2892_ == 0)
{
lean_ctor_set_tag(v___x_2891_, 0);
v___x_2894_ = v___x_2891_;
goto v_reusejp_2893_;
}
else
{
lean_object* v_reuseFailAlloc_2895_; 
v_reuseFailAlloc_2895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2895_, 0, v_val_2889_);
v___x_2894_ = v_reuseFailAlloc_2895_;
goto v_reusejp_2893_;
}
v_reusejp_2893_:
{
return v___x_2894_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7___boxed(lean_object* v_constName_2897_, lean_object* v___y_2898_, lean_object* v___y_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_){
_start:
{
lean_object* v_res_2903_; 
v_res_2903_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7(v_constName_2897_, v___y_2898_, v___y_2899_, v___y_2900_, v___y_2901_);
lean_dec(v___y_2901_);
lean_dec_ref(v___y_2900_);
lean_dec(v___y_2899_);
lean_dec_ref(v___y_2898_);
return v_res_2903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__0(lean_object* v_a_2904_, lean_object* v_a_2905_){
_start:
{
if (lean_obj_tag(v_a_2904_) == 0)
{
lean_object* v___x_2906_; 
v___x_2906_ = l_List_reverse___redArg(v_a_2905_);
return v___x_2906_;
}
else
{
lean_object* v_head_2907_; lean_object* v_tail_2908_; lean_object* v___x_2910_; uint8_t v_isShared_2911_; uint8_t v_isSharedCheck_2917_; 
v_head_2907_ = lean_ctor_get(v_a_2904_, 0);
v_tail_2908_ = lean_ctor_get(v_a_2904_, 1);
v_isSharedCheck_2917_ = !lean_is_exclusive(v_a_2904_);
if (v_isSharedCheck_2917_ == 0)
{
v___x_2910_ = v_a_2904_;
v_isShared_2911_ = v_isSharedCheck_2917_;
goto v_resetjp_2909_;
}
else
{
lean_inc(v_tail_2908_);
lean_inc(v_head_2907_);
lean_dec(v_a_2904_);
v___x_2910_ = lean_box(0);
v_isShared_2911_ = v_isSharedCheck_2917_;
goto v_resetjp_2909_;
}
v_resetjp_2909_:
{
lean_object* v___x_2912_; lean_object* v___x_2914_; 
v___x_2912_ = l_Lean_mkLevelParam(v_head_2907_);
if (v_isShared_2911_ == 0)
{
lean_ctor_set(v___x_2910_, 1, v_a_2905_);
lean_ctor_set(v___x_2910_, 0, v___x_2912_);
v___x_2914_ = v___x_2910_;
goto v_reusejp_2913_;
}
else
{
lean_object* v_reuseFailAlloc_2916_; 
v_reuseFailAlloc_2916_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2916_, 0, v___x_2912_);
lean_ctor_set(v_reuseFailAlloc_2916_, 1, v_a_2905_);
v___x_2914_ = v_reuseFailAlloc_2916_;
goto v_reusejp_2913_;
}
v_reusejp_2913_:
{
v_a_2904_ = v_tail_2908_;
v_a_2905_ = v___x_2914_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5(lean_object* v_constName_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_){
_start:
{
lean_object* v___x_2924_; 
lean_inc(v_constName_2918_);
v___x_2924_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5_spec__7(v_constName_2918_, v___y_2919_, v___y_2920_, v___y_2921_, v___y_2922_);
if (lean_obj_tag(v___x_2924_) == 0)
{
lean_object* v_a_2925_; lean_object* v___x_2927_; uint8_t v_isShared_2928_; uint8_t v_isSharedCheck_2936_; 
v_a_2925_ = lean_ctor_get(v___x_2924_, 0);
v_isSharedCheck_2936_ = !lean_is_exclusive(v___x_2924_);
if (v_isSharedCheck_2936_ == 0)
{
v___x_2927_ = v___x_2924_;
v_isShared_2928_ = v_isSharedCheck_2936_;
goto v_resetjp_2926_;
}
else
{
lean_inc(v_a_2925_);
lean_dec(v___x_2924_);
v___x_2927_ = lean_box(0);
v_isShared_2928_ = v_isSharedCheck_2936_;
goto v_resetjp_2926_;
}
v_resetjp_2926_:
{
lean_object* v_levelParams_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2934_; 
v_levelParams_2929_ = lean_ctor_get(v_a_2925_, 1);
lean_inc(v_levelParams_2929_);
lean_dec(v_a_2925_);
v___x_2930_ = lean_box(0);
v___x_2931_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__0(v_levelParams_2929_, v___x_2930_);
v___x_2932_ = l_Lean_mkConst(v_constName_2918_, v___x_2931_);
if (v_isShared_2928_ == 0)
{
lean_ctor_set(v___x_2927_, 0, v___x_2932_);
v___x_2934_ = v___x_2927_;
goto v_reusejp_2933_;
}
else
{
lean_object* v_reuseFailAlloc_2935_; 
v_reuseFailAlloc_2935_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2935_, 0, v___x_2932_);
v___x_2934_ = v_reuseFailAlloc_2935_;
goto v_reusejp_2933_;
}
v_reusejp_2933_:
{
return v___x_2934_;
}
}
}
else
{
lean_object* v_a_2937_; lean_object* v___x_2939_; uint8_t v_isShared_2940_; uint8_t v_isSharedCheck_2944_; 
lean_dec(v_constName_2918_);
v_a_2937_ = lean_ctor_get(v___x_2924_, 0);
v_isSharedCheck_2944_ = !lean_is_exclusive(v___x_2924_);
if (v_isSharedCheck_2944_ == 0)
{
v___x_2939_ = v___x_2924_;
v_isShared_2940_ = v_isSharedCheck_2944_;
goto v_resetjp_2938_;
}
else
{
lean_inc(v_a_2937_);
lean_dec(v___x_2924_);
v___x_2939_ = lean_box(0);
v_isShared_2940_ = v_isSharedCheck_2944_;
goto v_resetjp_2938_;
}
v_resetjp_2938_:
{
lean_object* v___x_2942_; 
if (v_isShared_2940_ == 0)
{
v___x_2942_ = v___x_2939_;
goto v_reusejp_2941_;
}
else
{
lean_object* v_reuseFailAlloc_2943_; 
v_reuseFailAlloc_2943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2943_, 0, v_a_2937_);
v___x_2942_ = v_reuseFailAlloc_2943_;
goto v_reusejp_2941_;
}
v_reusejp_2941_:
{
return v___x_2942_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5___boxed(lean_object* v_constName_2945_, lean_object* v___y_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_){
_start:
{
lean_object* v_res_2951_; 
v_res_2951_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5(v_constName_2945_, v___y_2946_, v___y_2947_, v___y_2948_, v___y_2949_);
lean_dec(v___y_2949_);
lean_dec_ref(v___y_2948_);
lean_dec(v___y_2947_);
lean_dec_ref(v___y_2946_);
return v_res_2951_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_2952_; 
v___x_2952_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2952_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_2953_; lean_object* v___x_2954_; 
v___x_2953_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__0);
v___x_2954_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2954_, 0, v___x_2953_);
return v___x_2954_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_2955_; lean_object* v___x_2956_; 
v___x_2955_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1);
v___x_2956_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2956_, 0, v___x_2955_);
lean_ctor_set(v___x_2956_, 1, v___x_2955_);
return v___x_2956_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_2957_; lean_object* v___x_2958_; 
v___x_2957_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__1);
v___x_2958_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2958_, 0, v___x_2957_);
lean_ctor_set(v___x_2958_, 1, v___x_2957_);
lean_ctor_set(v___x_2958_, 2, v___x_2957_);
lean_ctor_set(v___x_2958_, 3, v___x_2957_);
lean_ctor_set(v___x_2958_, 4, v___x_2957_);
lean_ctor_set(v___x_2958_, 5, v___x_2957_);
return v___x_2958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg(lean_object* v_declName_2959_, lean_object* v_declRanges_2960_, lean_object* v___y_2961_, lean_object* v___y_2962_){
_start:
{
uint8_t v___x_2964_; 
v___x_2964_ = l_Lean_Name_isAnonymous(v_declName_2959_);
if (v___x_2964_ == 0)
{
lean_object* v___x_2965_; lean_object* v_env_2966_; lean_object* v_nextMacroScope_2967_; lean_object* v_ngen_2968_; lean_object* v_auxDeclNGen_2969_; lean_object* v_traceState_2970_; lean_object* v_messages_2971_; lean_object* v_infoState_2972_; lean_object* v_snapshotTasks_2973_; lean_object* v___x_2975_; uint8_t v_isShared_2976_; uint8_t v_isSharedCheck_3001_; 
v___x_2965_ = lean_st_ref_take(v___y_2962_);
v_env_2966_ = lean_ctor_get(v___x_2965_, 0);
v_nextMacroScope_2967_ = lean_ctor_get(v___x_2965_, 1);
v_ngen_2968_ = lean_ctor_get(v___x_2965_, 2);
v_auxDeclNGen_2969_ = lean_ctor_get(v___x_2965_, 3);
v_traceState_2970_ = lean_ctor_get(v___x_2965_, 4);
v_messages_2971_ = lean_ctor_get(v___x_2965_, 6);
v_infoState_2972_ = lean_ctor_get(v___x_2965_, 7);
v_snapshotTasks_2973_ = lean_ctor_get(v___x_2965_, 8);
v_isSharedCheck_3001_ = !lean_is_exclusive(v___x_2965_);
if (v_isSharedCheck_3001_ == 0)
{
lean_object* v_unused_3002_; 
v_unused_3002_ = lean_ctor_get(v___x_2965_, 5);
lean_dec(v_unused_3002_);
v___x_2975_ = v___x_2965_;
v_isShared_2976_ = v_isSharedCheck_3001_;
goto v_resetjp_2974_;
}
else
{
lean_inc(v_snapshotTasks_2973_);
lean_inc(v_infoState_2972_);
lean_inc(v_messages_2971_);
lean_inc(v_traceState_2970_);
lean_inc(v_auxDeclNGen_2969_);
lean_inc(v_ngen_2968_);
lean_inc(v_nextMacroScope_2967_);
lean_inc(v_env_2966_);
lean_dec(v___x_2965_);
v___x_2975_ = lean_box(0);
v_isShared_2976_ = v_isSharedCheck_3001_;
goto v_resetjp_2974_;
}
v_resetjp_2974_:
{
lean_object* v___x_2977_; lean_object* v___x_2978_; lean_object* v___x_2979_; lean_object* v___x_2981_; 
v___x_2977_ = l_Lean_declRangeExt;
v___x_2978_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_2977_, v_env_2966_, v_declName_2959_, v_declRanges_2960_);
v___x_2979_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2);
if (v_isShared_2976_ == 0)
{
lean_ctor_set(v___x_2975_, 5, v___x_2979_);
lean_ctor_set(v___x_2975_, 0, v___x_2978_);
v___x_2981_ = v___x_2975_;
goto v_reusejp_2980_;
}
else
{
lean_object* v_reuseFailAlloc_3000_; 
v_reuseFailAlloc_3000_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3000_, 0, v___x_2978_);
lean_ctor_set(v_reuseFailAlloc_3000_, 1, v_nextMacroScope_2967_);
lean_ctor_set(v_reuseFailAlloc_3000_, 2, v_ngen_2968_);
lean_ctor_set(v_reuseFailAlloc_3000_, 3, v_auxDeclNGen_2969_);
lean_ctor_set(v_reuseFailAlloc_3000_, 4, v_traceState_2970_);
lean_ctor_set(v_reuseFailAlloc_3000_, 5, v___x_2979_);
lean_ctor_set(v_reuseFailAlloc_3000_, 6, v_messages_2971_);
lean_ctor_set(v_reuseFailAlloc_3000_, 7, v_infoState_2972_);
lean_ctor_set(v_reuseFailAlloc_3000_, 8, v_snapshotTasks_2973_);
v___x_2981_ = v_reuseFailAlloc_3000_;
goto v_reusejp_2980_;
}
v_reusejp_2980_:
{
lean_object* v___x_2982_; lean_object* v___x_2983_; lean_object* v_mctx_2984_; lean_object* v_zetaDeltaFVarIds_2985_; lean_object* v_postponed_2986_; lean_object* v_diag_2987_; lean_object* v___x_2989_; uint8_t v_isShared_2990_; uint8_t v_isSharedCheck_2998_; 
v___x_2982_ = lean_st_ref_set(v___y_2962_, v___x_2981_);
v___x_2983_ = lean_st_ref_take(v___y_2961_);
v_mctx_2984_ = lean_ctor_get(v___x_2983_, 0);
v_zetaDeltaFVarIds_2985_ = lean_ctor_get(v___x_2983_, 2);
v_postponed_2986_ = lean_ctor_get(v___x_2983_, 3);
v_diag_2987_ = lean_ctor_get(v___x_2983_, 4);
v_isSharedCheck_2998_ = !lean_is_exclusive(v___x_2983_);
if (v_isSharedCheck_2998_ == 0)
{
lean_object* v_unused_2999_; 
v_unused_2999_ = lean_ctor_get(v___x_2983_, 1);
lean_dec(v_unused_2999_);
v___x_2989_ = v___x_2983_;
v_isShared_2990_ = v_isSharedCheck_2998_;
goto v_resetjp_2988_;
}
else
{
lean_inc(v_diag_2987_);
lean_inc(v_postponed_2986_);
lean_inc(v_zetaDeltaFVarIds_2985_);
lean_inc(v_mctx_2984_);
lean_dec(v___x_2983_);
v___x_2989_ = lean_box(0);
v_isShared_2990_ = v_isSharedCheck_2998_;
goto v_resetjp_2988_;
}
v_resetjp_2988_:
{
lean_object* v___x_2991_; lean_object* v___x_2993_; 
v___x_2991_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3);
if (v_isShared_2990_ == 0)
{
lean_ctor_set(v___x_2989_, 1, v___x_2991_);
v___x_2993_ = v___x_2989_;
goto v_reusejp_2992_;
}
else
{
lean_object* v_reuseFailAlloc_2997_; 
v_reuseFailAlloc_2997_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2997_, 0, v_mctx_2984_);
lean_ctor_set(v_reuseFailAlloc_2997_, 1, v___x_2991_);
lean_ctor_set(v_reuseFailAlloc_2997_, 2, v_zetaDeltaFVarIds_2985_);
lean_ctor_set(v_reuseFailAlloc_2997_, 3, v_postponed_2986_);
lean_ctor_set(v_reuseFailAlloc_2997_, 4, v_diag_2987_);
v___x_2993_ = v_reuseFailAlloc_2997_;
goto v_reusejp_2992_;
}
v_reusejp_2992_:
{
lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; 
v___x_2994_ = lean_st_ref_set(v___y_2961_, v___x_2993_);
v___x_2995_ = lean_box(0);
v___x_2996_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2996_, 0, v___x_2995_);
return v___x_2996_;
}
}
}
}
}
else
{
lean_object* v___x_3003_; lean_object* v___x_3004_; 
lean_dec_ref(v_declRanges_2960_);
lean_dec(v_declName_2959_);
v___x_3003_ = lean_box(0);
v___x_3004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3004_, 0, v___x_3003_);
return v___x_3004_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___boxed(lean_object* v_declName_3005_, lean_object* v_declRanges_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_){
_start:
{
lean_object* v_res_3010_; 
v_res_3010_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg(v_declName_3005_, v_declRanges_3006_, v___y_3007_, v___y_3008_);
lean_dec(v___y_3008_);
lean_dec(v___y_3007_);
return v_res_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(lean_object* v_stx_3011_, lean_object* v___y_3012_){
_start:
{
uint8_t v___x_3014_; lean_object* v___x_3015_; 
v___x_3014_ = 0;
v___x_3015_ = l_Lean_Syntax_getRange_x3f(v_stx_3011_, v___x_3014_);
if (lean_obj_tag(v___x_3015_) == 1)
{
lean_object* v_val_3016_; lean_object* v___x_3018_; uint8_t v_isShared_3019_; uint8_t v_isSharedCheck_3028_; 
v_val_3016_ = lean_ctor_get(v___x_3015_, 0);
v_isSharedCheck_3028_ = !lean_is_exclusive(v___x_3015_);
if (v_isSharedCheck_3028_ == 0)
{
v___x_3018_ = v___x_3015_;
v_isShared_3019_ = v_isSharedCheck_3028_;
goto v_resetjp_3017_;
}
else
{
lean_inc(v_val_3016_);
lean_dec(v___x_3015_);
v___x_3018_ = lean_box(0);
v_isShared_3019_ = v_isSharedCheck_3028_;
goto v_resetjp_3017_;
}
v_resetjp_3017_:
{
lean_object* v_fileMap_3020_; lean_object* v_start_3021_; lean_object* v_stop_3022_; lean_object* v___x_3023_; lean_object* v___x_3025_; 
v_fileMap_3020_ = lean_ctor_get(v___y_3012_, 1);
v_start_3021_ = lean_ctor_get(v_val_3016_, 0);
lean_inc(v_start_3021_);
v_stop_3022_ = lean_ctor_get(v_val_3016_, 1);
lean_inc(v_stop_3022_);
lean_dec(v_val_3016_);
lean_inc_ref(v_fileMap_3020_);
v___x_3023_ = l_Lean_DeclarationRange_ofStringPositions(v_fileMap_3020_, v_start_3021_, v_stop_3022_);
lean_dec(v_stop_3022_);
lean_dec(v_start_3021_);
if (v_isShared_3019_ == 0)
{
lean_ctor_set(v___x_3018_, 0, v___x_3023_);
v___x_3025_ = v___x_3018_;
goto v_reusejp_3024_;
}
else
{
lean_object* v_reuseFailAlloc_3027_; 
v_reuseFailAlloc_3027_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3027_, 0, v___x_3023_);
v___x_3025_ = v_reuseFailAlloc_3027_;
goto v_reusejp_3024_;
}
v_reusejp_3024_:
{
lean_object* v___x_3026_; 
v___x_3026_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3026_, 0, v___x_3025_);
return v___x_3026_;
}
}
}
else
{
lean_object* v___x_3029_; lean_object* v___x_3030_; 
lean_dec(v___x_3015_);
v___x_3029_ = lean_box(0);
v___x_3030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3030_, 0, v___x_3029_);
return v___x_3030_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg___boxed(lean_object* v_stx_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_){
_start:
{
lean_object* v_res_3034_; 
v_res_3034_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(v_stx_3031_, v___y_3032_);
lean_dec_ref(v___y_3032_);
lean_dec(v_stx_3031_);
return v_res_3034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4(lean_object* v_declName_3035_, lean_object* v_rangeStx_3036_, lean_object* v_selectionRangeStx_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_){
_start:
{
lean_object* v___x_3043_; lean_object* v_a_3044_; lean_object* v___x_3046_; uint8_t v_isShared_3047_; uint8_t v_isSharedCheck_3060_; 
v___x_3043_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(v_rangeStx_3036_, v___y_3040_);
v_a_3044_ = lean_ctor_get(v___x_3043_, 0);
v_isSharedCheck_3060_ = !lean_is_exclusive(v___x_3043_);
if (v_isSharedCheck_3060_ == 0)
{
v___x_3046_ = v___x_3043_;
v_isShared_3047_ = v_isSharedCheck_3060_;
goto v_resetjp_3045_;
}
else
{
lean_inc(v_a_3044_);
lean_dec(v___x_3043_);
v___x_3046_ = lean_box(0);
v_isShared_3047_ = v_isSharedCheck_3060_;
goto v_resetjp_3045_;
}
v_resetjp_3045_:
{
if (lean_obj_tag(v_a_3044_) == 1)
{
lean_object* v_val_3048_; lean_object* v___x_3049_; lean_object* v_a_3050_; lean_object* v_a_3052_; 
lean_del_object(v___x_3046_);
v_val_3048_ = lean_ctor_get(v_a_3044_, 0);
lean_inc(v_val_3048_);
lean_dec_ref_known(v_a_3044_, 1);
v___x_3049_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(v_selectionRangeStx_3037_, v___y_3040_);
v_a_3050_ = lean_ctor_get(v___x_3049_, 0);
lean_inc(v_a_3050_);
lean_dec_ref(v___x_3049_);
if (lean_obj_tag(v_a_3050_) == 0)
{
lean_inc(v_val_3048_);
v_a_3052_ = v_val_3048_;
goto v___jp_3051_;
}
else
{
lean_object* v_val_3055_; 
v_val_3055_ = lean_ctor_get(v_a_3050_, 0);
lean_inc(v_val_3055_);
lean_dec_ref_known(v_a_3050_, 1);
v_a_3052_ = v_val_3055_;
goto v___jp_3051_;
}
v___jp_3051_:
{
lean_object* v___x_3053_; lean_object* v___x_3054_; 
v___x_3053_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3053_, 0, v_val_3048_);
lean_ctor_set(v___x_3053_, 1, v_a_3052_);
v___x_3054_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg(v_declName_3035_, v___x_3053_, v___y_3039_, v___y_3041_);
return v___x_3054_;
}
}
else
{
lean_object* v___x_3056_; lean_object* v___x_3058_; 
lean_dec(v_a_3044_);
lean_dec(v_declName_3035_);
v___x_3056_ = lean_box(0);
if (v_isShared_3047_ == 0)
{
lean_ctor_set(v___x_3046_, 0, v___x_3056_);
v___x_3058_ = v___x_3046_;
goto v_reusejp_3057_;
}
else
{
lean_object* v_reuseFailAlloc_3059_; 
v_reuseFailAlloc_3059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3059_, 0, v___x_3056_);
v___x_3058_ = v_reuseFailAlloc_3059_;
goto v_reusejp_3057_;
}
v_reusejp_3057_:
{
return v___x_3058_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4___boxed(lean_object* v_declName_3061_, lean_object* v_rangeStx_3062_, lean_object* v_selectionRangeStx_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_){
_start:
{
lean_object* v_res_3069_; 
v_res_3069_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4(v_declName_3061_, v_rangeStx_3062_, v_selectionRangeStx_3063_, v___y_3064_, v___y_3065_, v___y_3066_, v___y_3067_);
lean_dec(v___y_3067_);
lean_dec_ref(v___y_3066_);
lean_dec(v___y_3065_);
lean_dec_ref(v___y_3064_);
lean_dec(v_selectionRangeStx_3063_);
lean_dec(v_rangeStx_3062_);
return v_res_3069_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2(void){
_start:
{
lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; 
v___x_3074_ = lean_box(0);
v___x_3075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__1));
v___x_3076_ = l_Lean_mkConst(v___x_3075_, v___x_3074_);
return v___x_3076_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5(void){
_start:
{
lean_object* v___x_3082_; lean_object* v___x_3083_; 
v___x_3082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__4));
v___x_3083_ = l_Lean_stringToMessageData(v___x_3082_);
return v___x_3083_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9(void){
_start:
{
lean_object* v___x_3096_; lean_object* v___x_3097_; 
v___x_3096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__8));
v___x_3097_ = l_Lean_stringToMessageData(v___x_3096_);
return v___x_3097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl(lean_object* v_ind_3098_, lean_object* v_rel_3099_, lean_object* v_relStx_3100_, lean_object* v_a_3101_, lean_object* v_a_3102_, lean_object* v_a_3103_, lean_object* v_a_3104_){
_start:
{
lean_object* v___x_3106_; 
lean_inc(v_ind_3098_);
v___x_3106_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0(v_ind_3098_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3106_) == 0)
{
lean_object* v_a_3107_; 
v_a_3107_ = lean_ctor_get(v___x_3106_, 0);
lean_inc(v_a_3107_);
lean_dec_ref_known(v___x_3106_, 1);
if (lean_obj_tag(v_a_3107_) == 5)
{
lean_object* v_val_3108_; lean_object* v___x_3110_; uint8_t v_isShared_3111_; uint8_t v_isSharedCheck_3267_; 
v_val_3108_ = lean_ctor_get(v_a_3107_, 0);
v_isSharedCheck_3267_ = !lean_is_exclusive(v_a_3107_);
if (v_isSharedCheck_3267_ == 0)
{
v___x_3110_ = v_a_3107_;
v_isShared_3111_ = v_isSharedCheck_3267_;
goto v_resetjp_3109_;
}
else
{
lean_inc(v_val_3108_);
lean_dec(v_a_3107_);
v___x_3110_ = lean_box(0);
v_isShared_3111_ = v_isSharedCheck_3267_;
goto v_resetjp_3109_;
}
v_resetjp_3109_:
{
lean_object* v_toConstantVal_3112_; lean_object* v_numParams_3113_; lean_object* v_ctors_3114_; lean_object* v_levelParams_3115_; lean_object* v_type_3116_; lean_object* v___x_3118_; uint8_t v_isShared_3119_; uint8_t v_isSharedCheck_3265_; 
v_toConstantVal_3112_ = lean_ctor_get(v_val_3108_, 0);
lean_inc_ref(v_toConstantVal_3112_);
v_numParams_3113_ = lean_ctor_get(v_val_3108_, 1);
lean_inc(v_numParams_3113_);
v_ctors_3114_ = lean_ctor_get(v_val_3108_, 4);
lean_inc(v_ctors_3114_);
lean_dec_ref(v_val_3108_);
v_levelParams_3115_ = lean_ctor_get(v_toConstantVal_3112_, 1);
v_type_3116_ = lean_ctor_get(v_toConstantVal_3112_, 2);
v_isSharedCheck_3265_ = !lean_is_exclusive(v_toConstantVal_3112_);
if (v_isSharedCheck_3265_ == 0)
{
lean_object* v_unused_3266_; 
v_unused_3266_ = lean_ctor_get(v_toConstantVal_3112_, 0);
lean_dec(v_unused_3266_);
v___x_3118_ = v_toConstantVal_3112_;
v_isShared_3119_ = v_isSharedCheck_3265_;
goto v_resetjp_3117_;
}
else
{
lean_inc(v_type_3116_);
lean_inc(v_levelParams_3115_);
lean_dec(v_toConstantVal_3112_);
v___x_3118_ = lean_box(0);
v_isShared_3119_ = v_isSharedCheck_3265_;
goto v_resetjp_3117_;
}
v_resetjp_3117_:
{
lean_object* v___x_3120_; lean_object* v___x_3121_; lean_object* v___f_3122_; uint8_t v___x_3123_; lean_object* v___x_3124_; 
v___x_3120_ = lean_box(0);
lean_inc(v_levelParams_3115_);
v___x_3121_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__0(v_levelParams_3115_, v___x_3120_);
lean_inc(v_ctors_3114_);
lean_inc(v_numParams_3113_);
v___f_3122_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___lam__0___boxed), 12, 5);
lean_closure_set(v___f_3122_, 0, v_ind_3098_);
lean_closure_set(v___f_3122_, 1, v___x_3121_);
lean_closure_set(v___f_3122_, 2, v_numParams_3113_);
lean_closure_set(v___f_3122_, 3, v_ctors_3114_);
lean_closure_set(v___f_3122_, 4, v___x_3120_);
v___x_3123_ = 0;
v___x_3124_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Tactic_MkIff_constrToProp_spec__5___redArg(v_type_3116_, v___f_3122_, v___x_3123_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3124_) == 0)
{
lean_object* v_a_3125_; lean_object* v_fst_3126_; lean_object* v_snd_3127_; lean_object* v___x_3129_; 
v_a_3125_ = lean_ctor_get(v___x_3124_, 0);
lean_inc(v_a_3125_);
lean_dec_ref_known(v___x_3124_, 1);
v_fst_3126_ = lean_ctor_get(v_a_3125_, 0);
lean_inc_n(v_fst_3126_, 2);
v_snd_3127_ = lean_ctor_get(v_a_3125_, 1);
lean_inc(v_snd_3127_);
lean_dec(v_a_3125_);
if (v_isShared_3111_ == 0)
{
lean_ctor_set_tag(v___x_3110_, 1);
lean_ctor_set(v___x_3110_, 0, v_fst_3126_);
v___x_3129_ = v___x_3110_;
goto v_reusejp_3128_;
}
else
{
lean_object* v_reuseFailAlloc_3256_; 
v_reuseFailAlloc_3256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3256_, 0, v_fst_3126_);
v___x_3129_ = v_reuseFailAlloc_3256_;
goto v_reusejp_3128_;
}
v_reusejp_3128_:
{
uint8_t v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; 
v___x_3130_ = 0;
v___x_3131_ = lean_box(0);
v___x_3132_ = l_Lean_Meta_mkFreshExprMVar(v___x_3129_, v___x_3130_, v___x_3131_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3132_) == 0)
{
lean_object* v_a_3133_; lean_object* v___x_3134_; lean_object* v___x_3135_; 
v_a_3133_ = lean_ctor_get(v___x_3132_, 0);
lean_inc(v_a_3133_);
lean_dec_ref_known(v___x_3132_, 1);
v___x_3134_ = l_Lean_Expr_mvarId_x21(v_a_3133_);
v___x_3135_ = l_Lean_MVarId_intros(v___x_3134_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3135_) == 0)
{
lean_object* v_a_3136_; lean_object* v_fst_3137_; lean_object* v_snd_3138_; lean_object* v___x_3139_; uint8_t v___x_3140_; lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; 
v_a_3136_ = lean_ctor_get(v___x_3135_, 0);
lean_inc(v_a_3136_);
lean_dec_ref_known(v___x_3135_, 1);
v_fst_3137_ = lean_ctor_get(v_a_3136_, 0);
lean_inc(v_fst_3137_);
v_snd_3138_ = lean_ctor_get(v_a_3136_, 1);
lean_inc(v_snd_3138_);
lean_dec(v_a_3136_);
v___x_3139_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2, &lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__2);
v___x_3140_ = 1;
v___x_3141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__3));
v___x_3142_ = lean_box(0);
v___x_3143_ = l_Lean_MVarId_apply(v_snd_3138_, v___x_3139_, v___x_3141_, v___x_3142_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3143_) == 0)
{
lean_object* v_a_3144_; lean_object* v___y_3146_; lean_object* v___y_3147_; lean_object* v___y_3148_; lean_object* v___y_3149_; 
v_a_3144_ = lean_ctor_get(v___x_3143_, 0);
lean_inc(v_a_3144_);
lean_dec_ref_known(v___x_3143_, 1);
if (lean_obj_tag(v_a_3144_) == 1)
{
lean_object* v_tail_3152_; 
v_tail_3152_ = lean_ctor_get(v_a_3144_, 1);
lean_inc(v_tail_3152_);
if (lean_obj_tag(v_tail_3152_) == 1)
{
lean_object* v_tail_3153_; 
v_tail_3153_ = lean_ctor_get(v_tail_3152_, 1);
lean_inc(v_tail_3153_);
if (lean_obj_tag(v_tail_3153_) == 0)
{
lean_object* v_head_3154_; lean_object* v_head_3155_; lean_object* v___x_3157_; uint8_t v_isShared_3158_; uint8_t v_isSharedCheck_3230_; 
v_head_3154_ = lean_ctor_get(v_a_3144_, 0);
lean_inc(v_head_3154_);
lean_dec_ref_known(v_a_3144_, 2);
v_head_3155_ = lean_ctor_get(v_tail_3152_, 0);
v_isSharedCheck_3230_ = !lean_is_exclusive(v_tail_3152_);
if (v_isSharedCheck_3230_ == 0)
{
lean_object* v_unused_3231_; 
v_unused_3231_ = lean_ctor_get(v_tail_3152_, 1);
lean_dec(v_unused_3231_);
v___x_3157_ = v_tail_3152_;
v_isShared_3158_ = v_isSharedCheck_3230_;
goto v_resetjp_3156_;
}
else
{
lean_inc(v_head_3155_);
lean_dec(v_tail_3152_);
v___x_3157_ = lean_box(0);
v_isShared_3158_ = v_isSharedCheck_3230_;
goto v_resetjp_3156_;
}
v_resetjp_3156_:
{
lean_object* v___x_3159_; 
lean_inc(v_snd_3127_);
v___x_3159_ = lp_mathlib_Mathlib_Tactic_MkIff_toCases(v_head_3154_, v_snd_3127_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3159_) == 0)
{
lean_object* v___x_3160_; 
lean_dec_ref_known(v___x_3159_, 1);
v___x_3160_ = l_Lean_Meta_intro1Core(v_head_3155_, v___x_3123_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3160_) == 0)
{
lean_object* v_a_3161_; lean_object* v_fst_3162_; lean_object* v_snd_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; 
v_a_3161_ = lean_ctor_get(v___x_3160_, 0);
lean_inc(v_a_3161_);
lean_dec_ref_known(v___x_3160_, 1);
v_fst_3162_ = lean_ctor_get(v_a_3161_, 0);
lean_inc(v_fst_3162_);
v_snd_3163_ = lean_ctor_get(v_a_3161_, 1);
lean_inc(v_snd_3163_);
lean_dec(v_a_3161_);
v___x_3164_ = lean_array_to_list(v_fst_3137_);
v___x_3165_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_MkIff_toInductive_spec__5___lam__0___closed__0));
lean_inc(v___x_3164_);
v___x_3166_ = l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_box(0), v___x_3164_, v___x_3164_, v_numParams_3113_, v___x_3165_);
lean_dec(v___x_3164_);
v___x_3167_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__2(v___x_3166_, v___x_3120_);
v___x_3168_ = lp_mathlib_Mathlib_Tactic_MkIff_toInductive(v_snd_3163_, v_ctors_3114_, v___x_3167_, v_snd_3127_, v_fst_3162_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3168_) == 0)
{
lean_object* v___x_3169_; lean_object* v_a_3170_; lean_object* v___x_3172_; uint8_t v_isShared_3173_; uint8_t v_isSharedCheck_3221_; 
lean_dec_ref_known(v___x_3168_, 1);
v___x_3169_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__3___redArg(v_a_3133_, v_a_3102_);
v_a_3170_ = lean_ctor_get(v___x_3169_, 0);
v_isSharedCheck_3221_ = !lean_is_exclusive(v___x_3169_);
if (v_isSharedCheck_3221_ == 0)
{
v___x_3172_ = v___x_3169_;
v_isShared_3173_ = v_isSharedCheck_3221_;
goto v_resetjp_3171_;
}
else
{
lean_inc(v_a_3170_);
lean_dec(v___x_3169_);
v___x_3172_ = lean_box(0);
v_isShared_3173_ = v_isSharedCheck_3221_;
goto v_resetjp_3171_;
}
v_resetjp_3171_:
{
lean_object* v___x_3175_; 
lean_inc(v_rel_3099_);
if (v_isShared_3119_ == 0)
{
lean_ctor_set(v___x_3118_, 2, v_fst_3126_);
lean_ctor_set(v___x_3118_, 0, v_rel_3099_);
v___x_3175_ = v___x_3118_;
goto v_reusejp_3174_;
}
else
{
lean_object* v_reuseFailAlloc_3220_; 
v_reuseFailAlloc_3220_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3220_, 0, v_rel_3099_);
lean_ctor_set(v_reuseFailAlloc_3220_, 1, v_levelParams_3115_);
lean_ctor_set(v_reuseFailAlloc_3220_, 2, v_fst_3126_);
v___x_3175_ = v_reuseFailAlloc_3220_;
goto v_reusejp_3174_;
}
v_reusejp_3174_:
{
lean_object* v___x_3177_; 
lean_inc(v_rel_3099_);
if (v_isShared_3158_ == 0)
{
lean_ctor_set(v___x_3157_, 1, v___x_3120_);
lean_ctor_set(v___x_3157_, 0, v_rel_3099_);
v___x_3177_ = v___x_3157_;
goto v_reusejp_3176_;
}
else
{
lean_object* v_reuseFailAlloc_3219_; 
v_reuseFailAlloc_3219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3219_, 0, v_rel_3099_);
lean_ctor_set(v_reuseFailAlloc_3219_, 1, v___x_3120_);
v___x_3177_ = v_reuseFailAlloc_3219_;
goto v_reusejp_3176_;
}
v_reusejp_3176_:
{
lean_object* v___x_3178_; lean_object* v___x_3180_; 
v___x_3178_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3178_, 0, v___x_3175_);
lean_ctor_set(v___x_3178_, 1, v_a_3170_);
lean_ctor_set(v___x_3178_, 2, v___x_3177_);
if (v_isShared_3173_ == 0)
{
lean_ctor_set_tag(v___x_3172_, 2);
lean_ctor_set(v___x_3172_, 0, v___x_3178_);
v___x_3180_ = v___x_3172_;
goto v_reusejp_3179_;
}
else
{
lean_object* v_reuseFailAlloc_3218_; 
v_reuseFailAlloc_3218_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3218_, 0, v___x_3178_);
v___x_3180_ = v_reuseFailAlloc_3218_;
goto v_reusejp_3179_;
}
v_reusejp_3179_:
{
lean_object* v___x_3181_; 
v___x_3181_ = l_Lean_addDecl(v___x_3180_, v___x_3123_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3181_) == 0)
{
lean_object* v_ref_3182_; lean_object* v___x_3183_; 
lean_dec_ref_known(v___x_3181_, 1);
v_ref_3182_ = lean_ctor_get(v_a_3103_, 5);
lean_inc(v_rel_3099_);
v___x_3183_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4(v_rel_3099_, v_ref_3182_, v_relStx_3100_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3183_) == 0)
{
lean_object* v___x_3184_; 
lean_dec_ref_known(v___x_3183_, 1);
v___x_3184_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5(v_rel_3099_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3184_) == 0)
{
lean_object* v_a_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; 
v_a_3185_ = lean_ctor_get(v___x_3184_, 0);
lean_inc(v_a_3185_);
lean_dec_ref_known(v___x_3184_, 1);
v___x_3186_ = lean_box(v___x_3140_);
v___x_3187_ = lean_box(v___x_3123_);
v___x_3188_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_addTermInfo_x27___boxed), 14, 7);
lean_closure_set(v___x_3188_, 0, v_relStx_3100_);
lean_closure_set(v___x_3188_, 1, v_a_3185_);
lean_closure_set(v___x_3188_, 2, v___x_3142_);
lean_closure_set(v___x_3188_, 3, v___x_3142_);
lean_closure_set(v___x_3188_, 4, v___x_3131_);
lean_closure_set(v___x_3188_, 5, v___x_3186_);
lean_closure_set(v___x_3188_, 6, v___x_3187_);
v___x_3189_ = lean_box(1);
v___x_3190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__7));
v___x_3191_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_3191_, 0, v___x_3120_);
lean_ctor_set(v___x_3191_, 1, v___x_3189_);
lean_ctor_set(v___x_3191_, 2, v_tail_3153_);
lean_ctor_set(v___x_3191_, 3, v___x_3120_);
lean_ctor_set(v___x_3191_, 4, v___x_3120_);
lean_ctor_set(v___x_3191_, 5, v___x_3189_);
lean_ctor_set(v___x_3191_, 6, v___x_3120_);
v___x_3192_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_3188_, v___x_3190_, v___x_3191_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3192_) == 0)
{
lean_object* v_a_3193_; lean_object* v___x_3195_; uint8_t v_isShared_3196_; uint8_t v_isSharedCheck_3201_; 
v_a_3193_ = lean_ctor_get(v___x_3192_, 0);
v_isSharedCheck_3201_ = !lean_is_exclusive(v___x_3192_);
if (v_isSharedCheck_3201_ == 0)
{
v___x_3195_ = v___x_3192_;
v_isShared_3196_ = v_isSharedCheck_3201_;
goto v_resetjp_3194_;
}
else
{
lean_inc(v_a_3193_);
lean_dec(v___x_3192_);
v___x_3195_ = lean_box(0);
v_isShared_3196_ = v_isSharedCheck_3201_;
goto v_resetjp_3194_;
}
v_resetjp_3194_:
{
lean_object* v_fst_3197_; lean_object* v___x_3199_; 
v_fst_3197_ = lean_ctor_get(v_a_3193_, 0);
lean_inc(v_fst_3197_);
lean_dec(v_a_3193_);
if (v_isShared_3196_ == 0)
{
lean_ctor_set(v___x_3195_, 0, v_fst_3197_);
v___x_3199_ = v___x_3195_;
goto v_reusejp_3198_;
}
else
{
lean_object* v_reuseFailAlloc_3200_; 
v_reuseFailAlloc_3200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3200_, 0, v_fst_3197_);
v___x_3199_ = v_reuseFailAlloc_3200_;
goto v_reusejp_3198_;
}
v_reusejp_3198_:
{
return v___x_3199_;
}
}
}
else
{
lean_object* v_a_3202_; lean_object* v___x_3204_; uint8_t v_isShared_3205_; uint8_t v_isSharedCheck_3209_; 
v_a_3202_ = lean_ctor_get(v___x_3192_, 0);
v_isSharedCheck_3209_ = !lean_is_exclusive(v___x_3192_);
if (v_isSharedCheck_3209_ == 0)
{
v___x_3204_ = v___x_3192_;
v_isShared_3205_ = v_isSharedCheck_3209_;
goto v_resetjp_3203_;
}
else
{
lean_inc(v_a_3202_);
lean_dec(v___x_3192_);
v___x_3204_ = lean_box(0);
v_isShared_3205_ = v_isSharedCheck_3209_;
goto v_resetjp_3203_;
}
v_resetjp_3203_:
{
lean_object* v___x_3207_; 
if (v_isShared_3205_ == 0)
{
v___x_3207_ = v___x_3204_;
goto v_reusejp_3206_;
}
else
{
lean_object* v_reuseFailAlloc_3208_; 
v_reuseFailAlloc_3208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3208_, 0, v_a_3202_);
v___x_3207_ = v_reuseFailAlloc_3208_;
goto v_reusejp_3206_;
}
v_reusejp_3206_:
{
return v___x_3207_;
}
}
}
}
else
{
lean_object* v_a_3210_; lean_object* v___x_3212_; uint8_t v_isShared_3213_; uint8_t v_isSharedCheck_3217_; 
lean_dec(v_relStx_3100_);
v_a_3210_ = lean_ctor_get(v___x_3184_, 0);
v_isSharedCheck_3217_ = !lean_is_exclusive(v___x_3184_);
if (v_isSharedCheck_3217_ == 0)
{
v___x_3212_ = v___x_3184_;
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
else
{
lean_inc(v_a_3210_);
lean_dec(v___x_3184_);
v___x_3212_ = lean_box(0);
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
v_resetjp_3211_:
{
lean_object* v___x_3215_; 
if (v_isShared_3213_ == 0)
{
v___x_3215_ = v___x_3212_;
goto v_reusejp_3214_;
}
else
{
lean_object* v_reuseFailAlloc_3216_; 
v_reuseFailAlloc_3216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3216_, 0, v_a_3210_);
v___x_3215_ = v_reuseFailAlloc_3216_;
goto v_reusejp_3214_;
}
v_reusejp_3214_:
{
return v___x_3215_;
}
}
}
}
else
{
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
return v___x_3183_;
}
}
else
{
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
return v___x_3181_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_3157_);
lean_dec(v_a_3133_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
return v___x_3168_;
}
}
else
{
lean_object* v_a_3222_; lean_object* v___x_3224_; uint8_t v_isShared_3225_; uint8_t v_isSharedCheck_3229_; 
lean_del_object(v___x_3157_);
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v_a_3222_ = lean_ctor_get(v___x_3160_, 0);
v_isSharedCheck_3229_ = !lean_is_exclusive(v___x_3160_);
if (v_isSharedCheck_3229_ == 0)
{
v___x_3224_ = v___x_3160_;
v_isShared_3225_ = v_isSharedCheck_3229_;
goto v_resetjp_3223_;
}
else
{
lean_inc(v_a_3222_);
lean_dec(v___x_3160_);
v___x_3224_ = lean_box(0);
v_isShared_3225_ = v_isSharedCheck_3229_;
goto v_resetjp_3223_;
}
v_resetjp_3223_:
{
lean_object* v___x_3227_; 
if (v_isShared_3225_ == 0)
{
v___x_3227_ = v___x_3224_;
goto v_reusejp_3226_;
}
else
{
lean_object* v_reuseFailAlloc_3228_; 
v_reuseFailAlloc_3228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3228_, 0, v_a_3222_);
v___x_3227_ = v_reuseFailAlloc_3228_;
goto v_reusejp_3226_;
}
v_reusejp_3226_:
{
return v___x_3227_;
}
}
}
}
else
{
lean_del_object(v___x_3157_);
lean_dec(v_head_3155_);
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
return v___x_3159_;
}
}
}
else
{
lean_dec_ref_known(v_tail_3152_, 2);
lean_dec(v_tail_3153_);
lean_dec_ref_known(v_a_3144_, 2);
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v___y_3146_ = v_a_3101_;
v___y_3147_ = v_a_3102_;
v___y_3148_ = v_a_3103_;
v___y_3149_ = v_a_3104_;
goto v___jp_3145_;
}
}
else
{
lean_dec_ref_known(v_a_3144_, 2);
lean_dec(v_tail_3152_);
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v___y_3146_ = v_a_3101_;
v___y_3147_ = v_a_3102_;
v___y_3148_ = v_a_3103_;
v___y_3149_ = v_a_3104_;
goto v___jp_3145_;
}
}
else
{
lean_dec(v_a_3144_);
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v___y_3146_ = v_a_3101_;
v___y_3147_ = v_a_3102_;
v___y_3148_ = v_a_3103_;
v___y_3149_ = v_a_3104_;
goto v___jp_3145_;
}
v___jp_3145_:
{
lean_object* v___x_3150_; lean_object* v___x_3151_; 
v___x_3150_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5, &lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__5);
v___x_3151_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3150_, v___y_3146_, v___y_3147_, v___y_3148_, v___y_3149_);
return v___x_3151_;
}
}
else
{
lean_object* v_a_3232_; lean_object* v___x_3234_; uint8_t v_isShared_3235_; uint8_t v_isSharedCheck_3239_; 
lean_dec(v_fst_3137_);
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v_a_3232_ = lean_ctor_get(v___x_3143_, 0);
v_isSharedCheck_3239_ = !lean_is_exclusive(v___x_3143_);
if (v_isSharedCheck_3239_ == 0)
{
v___x_3234_ = v___x_3143_;
v_isShared_3235_ = v_isSharedCheck_3239_;
goto v_resetjp_3233_;
}
else
{
lean_inc(v_a_3232_);
lean_dec(v___x_3143_);
v___x_3234_ = lean_box(0);
v_isShared_3235_ = v_isSharedCheck_3239_;
goto v_resetjp_3233_;
}
v_resetjp_3233_:
{
lean_object* v___x_3237_; 
if (v_isShared_3235_ == 0)
{
v___x_3237_ = v___x_3234_;
goto v_reusejp_3236_;
}
else
{
lean_object* v_reuseFailAlloc_3238_; 
v_reuseFailAlloc_3238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3238_, 0, v_a_3232_);
v___x_3237_ = v_reuseFailAlloc_3238_;
goto v_reusejp_3236_;
}
v_reusejp_3236_:
{
return v___x_3237_;
}
}
}
}
else
{
lean_object* v_a_3240_; lean_object* v___x_3242_; uint8_t v_isShared_3243_; uint8_t v_isSharedCheck_3247_; 
lean_dec(v_a_3133_);
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v_a_3240_ = lean_ctor_get(v___x_3135_, 0);
v_isSharedCheck_3247_ = !lean_is_exclusive(v___x_3135_);
if (v_isSharedCheck_3247_ == 0)
{
v___x_3242_ = v___x_3135_;
v_isShared_3243_ = v_isSharedCheck_3247_;
goto v_resetjp_3241_;
}
else
{
lean_inc(v_a_3240_);
lean_dec(v___x_3135_);
v___x_3242_ = lean_box(0);
v_isShared_3243_ = v_isSharedCheck_3247_;
goto v_resetjp_3241_;
}
v_resetjp_3241_:
{
lean_object* v___x_3245_; 
if (v_isShared_3243_ == 0)
{
v___x_3245_ = v___x_3242_;
goto v_reusejp_3244_;
}
else
{
lean_object* v_reuseFailAlloc_3246_; 
v_reuseFailAlloc_3246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3246_, 0, v_a_3240_);
v___x_3245_ = v_reuseFailAlloc_3246_;
goto v_reusejp_3244_;
}
v_reusejp_3244_:
{
return v___x_3245_;
}
}
}
}
else
{
lean_object* v_a_3248_; lean_object* v___x_3250_; uint8_t v_isShared_3251_; uint8_t v_isSharedCheck_3255_; 
lean_dec(v_snd_3127_);
lean_dec(v_fst_3126_);
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v_a_3248_ = lean_ctor_get(v___x_3132_, 0);
v_isSharedCheck_3255_ = !lean_is_exclusive(v___x_3132_);
if (v_isSharedCheck_3255_ == 0)
{
v___x_3250_ = v___x_3132_;
v_isShared_3251_ = v_isSharedCheck_3255_;
goto v_resetjp_3249_;
}
else
{
lean_inc(v_a_3248_);
lean_dec(v___x_3132_);
v___x_3250_ = lean_box(0);
v_isShared_3251_ = v_isSharedCheck_3255_;
goto v_resetjp_3249_;
}
v_resetjp_3249_:
{
lean_object* v___x_3253_; 
if (v_isShared_3251_ == 0)
{
v___x_3253_ = v___x_3250_;
goto v_reusejp_3252_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v_a_3248_);
v___x_3253_ = v_reuseFailAlloc_3254_;
goto v_reusejp_3252_;
}
v_reusejp_3252_:
{
return v___x_3253_;
}
}
}
}
}
else
{
lean_object* v_a_3257_; lean_object* v___x_3259_; uint8_t v_isShared_3260_; uint8_t v_isSharedCheck_3264_; 
lean_del_object(v___x_3118_);
lean_dec(v_levelParams_3115_);
lean_dec(v_ctors_3114_);
lean_dec(v_numParams_3113_);
lean_del_object(v___x_3110_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
v_a_3257_ = lean_ctor_get(v___x_3124_, 0);
v_isSharedCheck_3264_ = !lean_is_exclusive(v___x_3124_);
if (v_isSharedCheck_3264_ == 0)
{
v___x_3259_ = v___x_3124_;
v_isShared_3260_ = v_isSharedCheck_3264_;
goto v_resetjp_3258_;
}
else
{
lean_inc(v_a_3257_);
lean_dec(v___x_3124_);
v___x_3259_ = lean_box(0);
v_isShared_3260_ = v_isSharedCheck_3264_;
goto v_resetjp_3258_;
}
v_resetjp_3258_:
{
lean_object* v___x_3262_; 
if (v_isShared_3260_ == 0)
{
v___x_3262_ = v___x_3259_;
goto v_reusejp_3261_;
}
else
{
lean_object* v_reuseFailAlloc_3263_; 
v_reuseFailAlloc_3263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3263_, 0, v_a_3257_);
v___x_3262_ = v_reuseFailAlloc_3263_;
goto v_reusejp_3261_;
}
v_reusejp_3261_:
{
return v___x_3262_;
}
}
}
}
}
}
else
{
lean_object* v___x_3268_; lean_object* v___x_3269_; 
lean_dec(v_a_3107_);
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
lean_dec(v_ind_3098_);
v___x_3268_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9, &lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___closed__9);
v___x_3269_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3268_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_);
return v___x_3269_;
}
}
else
{
lean_object* v_a_3270_; lean_object* v___x_3272_; uint8_t v_isShared_3273_; uint8_t v_isSharedCheck_3277_; 
lean_dec(v_relStx_3100_);
lean_dec(v_rel_3099_);
lean_dec(v_ind_3098_);
v_a_3270_ = lean_ctor_get(v___x_3106_, 0);
v_isSharedCheck_3277_ = !lean_is_exclusive(v___x_3106_);
if (v_isSharedCheck_3277_ == 0)
{
v___x_3272_ = v___x_3106_;
v_isShared_3273_ = v_isSharedCheck_3277_;
goto v_resetjp_3271_;
}
else
{
lean_inc(v_a_3270_);
lean_dec(v___x_3106_);
v___x_3272_ = lean_box(0);
v_isShared_3273_ = v_isSharedCheck_3277_;
goto v_resetjp_3271_;
}
v_resetjp_3271_:
{
lean_object* v___x_3275_; 
if (v_isShared_3273_ == 0)
{
v___x_3275_ = v___x_3272_;
goto v_reusejp_3274_;
}
else
{
lean_object* v_reuseFailAlloc_3276_; 
v_reuseFailAlloc_3276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3276_, 0, v_a_3270_);
v___x_3275_ = v_reuseFailAlloc_3276_;
goto v_reusejp_3274_;
}
v_reusejp_3274_:
{
return v___x_3275_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl___boxed(lean_object* v_ind_3278_, lean_object* v_rel_3279_, lean_object* v_relStx_3280_, lean_object* v_a_3281_, lean_object* v_a_3282_, lean_object* v_a_3283_, lean_object* v_a_3284_, lean_object* v_a_3285_){
_start:
{
lean_object* v_res_3286_; 
v_res_3286_ = lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl(v_ind_3278_, v_rel_3279_, v_relStx_3280_, v_a_3281_, v_a_3282_, v_a_3283_, v_a_3284_);
lean_dec(v_a_3284_);
lean_dec_ref(v_a_3283_);
lean_dec(v_a_3282_);
lean_dec_ref(v_a_3281_);
return v_res_3286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4(lean_object* v_stx_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_, lean_object* v___y_3290_, lean_object* v___y_3291_){
_start:
{
lean_object* v___x_3293_; 
v___x_3293_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___redArg(v_stx_3287_, v___y_3290_);
return v___x_3293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4___boxed(lean_object* v_stx_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_){
_start:
{
lean_object* v_res_3300_; 
v_res_3300_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__4(v_stx_3294_, v___y_3295_, v___y_3296_, v___y_3297_, v___y_3298_);
lean_dec(v___y_3298_);
lean_dec_ref(v___y_3297_);
lean_dec(v___y_3296_);
lean_dec_ref(v___y_3295_);
lean_dec(v_stx_3294_);
return v_res_3300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5(lean_object* v_declName_3301_, lean_object* v_declRanges_3302_, lean_object* v___y_3303_, lean_object* v___y_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_){
_start:
{
lean_object* v___x_3308_; 
v___x_3308_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg(v_declName_3301_, v_declRanges_3302_, v___y_3304_, v___y_3306_);
return v___x_3308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___boxed(lean_object* v_declName_3309_, lean_object* v_declRanges_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_, lean_object* v___y_3313_, lean_object* v___y_3314_, lean_object* v___y_3315_){
_start:
{
lean_object* v_res_3316_; 
v_res_3316_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5(v_declName_3309_, v_declRanges_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_);
lean_dec(v___y_3314_);
lean_dec_ref(v___y_3313_);
lean_dec(v___y_3312_);
lean_dec_ref(v___y_3311_);
return v_res_3316_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3387_; lean_object* v___x_3388_; lean_object* v___x_3389_; 
v___x_3387_ = lean_box(0);
v___x_3388_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3389_, 0, v___x_3388_);
lean_ctor_set(v___x_3389_, 1, v___x_3387_);
return v___x_3389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3391_; lean_object* v___x_3392_; 
v___x_3391_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___closed__0);
v___x_3392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3392_, 0, v___x_3391_);
return v___x_3392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg___boxed(lean_object* v___y_3393_){
_start:
{
lean_object* v_res_3394_; 
v_res_3394_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
return v_res_3394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0(lean_object* v_00_u03b1_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_){
_start:
{
lean_object* v___x_3399_; 
v___x_3399_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
return v___x_3399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___boxed(lean_object* v_00_u03b1_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_, lean_object* v___y_3403_){
_start:
{
lean_object* v_res_3404_; 
v_res_3404_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0(v_00_u03b1_3400_, v___y_3401_, v___y_3402_);
lean_dec(v___y_3402_);
lean_dec_ref(v___y_3401_);
return v_res_3404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0(lean_object* v___x_3405_, lean_object* v___x_3406_, lean_object* v___x_3407_, lean_object* v_r_3408_, lean_object* v___x_3409_, lean_object* v___y_3410_, lean_object* v___y_3411_){
_start:
{
lean_object* v___x_3413_; lean_object* v___x_3414_; 
v___x_3413_ = lean_st_mk_ref(v___x_3405_);
v___x_3414_ = lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl(v___x_3406_, v___x_3407_, v_r_3408_, v___x_3409_, v___x_3413_, v___y_3410_, v___y_3411_);
if (lean_obj_tag(v___x_3414_) == 0)
{
lean_object* v_a_3415_; lean_object* v___x_3417_; uint8_t v_isShared_3418_; uint8_t v_isSharedCheck_3423_; 
v_a_3415_ = lean_ctor_get(v___x_3414_, 0);
v_isSharedCheck_3423_ = !lean_is_exclusive(v___x_3414_);
if (v_isSharedCheck_3423_ == 0)
{
v___x_3417_ = v___x_3414_;
v_isShared_3418_ = v_isSharedCheck_3423_;
goto v_resetjp_3416_;
}
else
{
lean_inc(v_a_3415_);
lean_dec(v___x_3414_);
v___x_3417_ = lean_box(0);
v_isShared_3418_ = v_isSharedCheck_3423_;
goto v_resetjp_3416_;
}
v_resetjp_3416_:
{
lean_object* v___x_3419_; lean_object* v___x_3421_; 
v___x_3419_ = lean_st_ref_get(v___x_3413_);
lean_dec(v___x_3413_);
lean_dec(v___x_3419_);
if (v_isShared_3418_ == 0)
{
v___x_3421_ = v___x_3417_;
goto v_reusejp_3420_;
}
else
{
lean_object* v_reuseFailAlloc_3422_; 
v_reuseFailAlloc_3422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3422_, 0, v_a_3415_);
v___x_3421_ = v_reuseFailAlloc_3422_;
goto v_reusejp_3420_;
}
v_reusejp_3420_:
{
return v___x_3421_;
}
}
}
else
{
lean_dec(v___x_3413_);
return v___x_3414_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0___boxed(lean_object* v___x_3424_, lean_object* v___x_3425_, lean_object* v___x_3426_, lean_object* v_r_3427_, lean_object* v___x_3428_, lean_object* v___y_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_){
_start:
{
lean_object* v_res_3432_; 
v_res_3432_ = lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0(v___x_3424_, v___x_3425_, v___x_3426_, v_r_3427_, v___x_3428_, v___y_3429_, v___y_3430_);
lean_dec(v___y_3430_);
lean_dec_ref(v___y_3429_);
lean_dec_ref(v___x_3428_);
return v_res_3432_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0(void){
_start:
{
lean_object* v___x_3433_; 
v___x_3433_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3433_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1(void){
_start:
{
lean_object* v___x_3434_; lean_object* v___x_3435_; 
v___x_3434_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__0);
v___x_3435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3435_, 0, v___x_3434_);
return v___x_3435_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2(void){
_start:
{
lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; lean_object* v___x_3439_; 
v___x_3436_ = lean_box(1);
v___x_3437_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4);
v___x_3438_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1);
v___x_3439_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3439_, 0, v___x_3438_);
lean_ctor_set(v___x_3439_, 1, v___x_3437_);
lean_ctor_set(v___x_3439_, 2, v___x_3436_);
return v___x_3439_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4(void){
_start:
{
lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3444_; 
v___x_3442_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1);
v___x_3443_ = lean_unsigned_to_nat(0u);
v___x_3444_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3444_, 0, v___x_3443_);
lean_ctor_set(v___x_3444_, 1, v___x_3443_);
lean_ctor_set(v___x_3444_, 2, v___x_3443_);
lean_ctor_set(v___x_3444_, 3, v___x_3443_);
lean_ctor_set(v___x_3444_, 4, v___x_3442_);
lean_ctor_set(v___x_3444_, 5, v___x_3442_);
lean_ctor_set(v___x_3444_, 6, v___x_3442_);
lean_ctor_set(v___x_3444_, 7, v___x_3442_);
lean_ctor_set(v___x_3444_, 8, v___x_3442_);
lean_ctor_set(v___x_3444_, 9, v___x_3442_);
return v___x_3444_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5(void){
_start:
{
lean_object* v___x_3445_; lean_object* v___x_3446_; 
v___x_3445_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1);
v___x_3446_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_3446_, 0, v___x_3445_);
lean_ctor_set(v___x_3446_, 1, v___x_3445_);
lean_ctor_set(v___x_3446_, 2, v___x_3445_);
lean_ctor_set(v___x_3446_, 3, v___x_3445_);
lean_ctor_set(v___x_3446_, 4, v___x_3445_);
lean_ctor_set(v___x_3446_, 5, v___x_3445_);
return v___x_3446_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6(void){
_start:
{
lean_object* v___x_3447_; lean_object* v___x_3448_; 
v___x_3447_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1);
v___x_3448_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3448_, 0, v___x_3447_);
lean_ctor_set(v___x_3448_, 1, v___x_3447_);
lean_ctor_set(v___x_3448_, 2, v___x_3447_);
lean_ctor_set(v___x_3448_, 3, v___x_3447_);
lean_ctor_set(v___x_3448_, 4, v___x_3447_);
return v___x_3448_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7(void){
_start:
{
lean_object* v___x_3449_; lean_object* v___x_3450_; lean_object* v___x_3451_; lean_object* v___x_3452_; lean_object* v___x_3453_; lean_object* v___x_3454_; 
v___x_3449_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6);
v___x_3450_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__4);
v___x_3451_ = lean_box(1);
v___x_3452_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5);
v___x_3453_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__4);
v___x_3454_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3454_, 0, v___x_3453_);
lean_ctor_set(v___x_3454_, 1, v___x_3452_);
lean_ctor_set(v___x_3454_, 2, v___x_3451_);
lean_ctor_set(v___x_3454_, 3, v___x_3450_);
lean_ctor_set(v___x_3454_, 4, v___x_3449_);
return v___x_3454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1(lean_object* v_x_3455_, lean_object* v_a_3456_, lean_object* v_a_3457_){
_start:
{
lean_object* v___x_3459_; uint8_t v___x_3460_; 
v___x_3459_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductiveProp___closed__1));
lean_inc(v_x_3455_);
v___x_3460_ = l_Lean_Syntax_isOfKind(v_x_3455_, v___x_3459_);
if (v___x_3460_ == 0)
{
lean_object* v___x_3461_; 
lean_dec(v_x_3455_);
v___x_3461_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
return v___x_3461_;
}
else
{
lean_object* v___x_3462_; lean_object* v_i_3463_; lean_object* v___x_3464_; uint8_t v___x_3465_; 
v___x_3462_ = lean_unsigned_to_nat(1u);
v_i_3463_ = l_Lean_Syntax_getArg(v_x_3455_, v___x_3462_);
v___x_3464_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14));
lean_inc(v_i_3463_);
v___x_3465_ = l_Lean_Syntax_isOfKind(v_i_3463_, v___x_3464_);
if (v___x_3465_ == 0)
{
lean_object* v___x_3466_; 
lean_dec(v_i_3463_);
lean_dec(v_x_3455_);
v___x_3466_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
return v___x_3466_;
}
else
{
lean_object* v___x_3467_; lean_object* v_r_3468_; uint8_t v___x_3469_; 
v___x_3467_ = lean_unsigned_to_nat(2u);
v_r_3468_ = l_Lean_Syntax_getArg(v_x_3455_, v___x_3467_);
lean_dec(v_x_3455_);
lean_inc(v_r_3468_);
v___x_3469_ = l_Lean_Syntax_isOfKind(v_r_3468_, v___x_3464_);
if (v___x_3469_ == 0)
{
lean_object* v___x_3470_; 
lean_dec(v_r_3468_);
lean_dec(v_i_3463_);
v___x_3470_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1_spec__0___redArg();
return v___x_3470_;
}
else
{
lean_object* v___x_3471_; lean_object* v___x_3472_; lean_object* v___x_3473_; uint8_t v___x_3474_; uint8_t v___x_3475_; uint8_t v___x_3476_; uint8_t v___x_3477_; lean_object* v___x_3478_; uint64_t v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; lean_object* v___f_3487_; lean_object* v___x_3488_; 
v___x_3471_ = lean_unsigned_to_nat(0u);
v___x_3472_ = l_Lean_TSyntax_getId(v_i_3463_);
lean_dec(v_i_3463_);
v___x_3473_ = l_Lean_TSyntax_getId(v_r_3468_);
v___x_3474_ = 0;
v___x_3475_ = 1;
v___x_3476_ = 0;
v___x_3477_ = 2;
v___x_3478_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v___x_3478_, 0, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 1, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 2, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 3, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 4, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 5, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 6, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 7, v___x_3474_);
lean_ctor_set_uint8(v___x_3478_, 8, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 9, v___x_3475_);
lean_ctor_set_uint8(v___x_3478_, 10, v___x_3476_);
lean_ctor_set_uint8(v___x_3478_, 11, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 12, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 13, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 14, v___x_3477_);
lean_ctor_set_uint8(v___x_3478_, 15, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 16, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 17, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 18, v___x_3469_);
lean_ctor_set_uint8(v___x_3478_, 19, v___x_3474_);
v___x_3479_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_3478_);
v___x_3480_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_3480_, 0, v___x_3478_);
lean_ctor_set_uint64(v___x_3480_, sizeof(void*)*1, v___x_3479_);
v___x_3481_ = lean_box(1);
v___x_3482_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2);
v___x_3483_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__3));
v___x_3484_ = lean_box(0);
v___x_3485_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3485_, 0, v___x_3480_);
lean_ctor_set(v___x_3485_, 1, v___x_3481_);
lean_ctor_set(v___x_3485_, 2, v___x_3482_);
lean_ctor_set(v___x_3485_, 3, v___x_3483_);
lean_ctor_set(v___x_3485_, 4, v___x_3484_);
lean_ctor_set(v___x_3485_, 5, v___x_3471_);
lean_ctor_set(v___x_3485_, 6, v___x_3484_);
lean_ctor_set_uint8(v___x_3485_, sizeof(void*)*7, v___x_3474_);
lean_ctor_set_uint8(v___x_3485_, sizeof(void*)*7 + 1, v___x_3474_);
lean_ctor_set_uint8(v___x_3485_, sizeof(void*)*7 + 2, v___x_3474_);
lean_ctor_set_uint8(v___x_3485_, sizeof(void*)*7 + 3, v___x_3469_);
v___x_3486_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__7);
v___f_3487_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___lam__0___boxed), 8, 5);
lean_closure_set(v___f_3487_, 0, v___x_3486_);
lean_closure_set(v___f_3487_, 1, v___x_3472_);
lean_closure_set(v___f_3487_, 2, v___x_3473_);
lean_closure_set(v___f_3487_, 3, v_r_3468_);
lean_closure_set(v___f_3487_, 4, v___x_3485_);
v___x_3488_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_3487_, v_a_3456_, v_a_3457_);
return v___x_3488_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___boxed(lean_object* v_x_3489_, lean_object* v_a_3490_, lean_object* v_a_3491_, lean_object* v_a_3492_){
_start:
{
lean_object* v_res_3493_; 
v_res_3493_ = lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1(v_x_3489_, v_a_3490_, v_a_3491_);
lean_dec(v_a_3491_);
lean_dec_ref(v_a_3490_);
return v_res_3493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object* v_decl_3494_, lean_object* v_____x_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_){
_start:
{
lean_object* v_fst_3501_; lean_object* v_snd_3502_; lean_object* v___x_3503_; 
v_fst_3501_ = lean_ctor_get(v_____x_3495_, 0);
lean_inc(v_fst_3501_);
v_snd_3502_ = lean_ctor_get(v_____x_3495_, 1);
lean_inc(v_snd_3502_);
lean_dec_ref(v_____x_3495_);
v___x_3503_ = lp_mathlib_Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl(v_decl_3494_, v_fst_3501_, v_snd_3502_, v___y_3496_, v___y_3497_, v___y_3498_, v___y_3499_);
return v___x_3503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object* v_decl_3504_, lean_object* v_____x_3505_, lean_object* v___y_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_){
_start:
{
lean_object* v_res_3511_; 
v_res_3511_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(v_decl_3504_, v_____x_3505_, v___y_3506_, v___y_3507_, v___y_3508_, v___y_3509_);
lean_dec(v___y_3509_);
lean_dec_ref(v___y_3508_);
lean_dec(v___y_3507_);
lean_dec_ref(v___y_3506_);
return v_res_3511_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1(void){
_start:
{
lean_object* v___x_3513_; lean_object* v___x_3514_; 
v___x_3513_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0));
v___x_3514_ = l_Lean_stringToMessageData(v___x_3513_);
return v___x_3514_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_3516_; lean_object* v___x_3517_; 
v___x_3516_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2));
v___x_3517_ = l_Lean_stringToMessageData(v___x_3516_);
return v___x_3517_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5(void){
_start:
{
lean_object* v___x_3519_; lean_object* v___x_3520_; 
v___x_3519_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4));
v___x_3520_ = l_Lean_stringToMessageData(v___x_3519_);
return v___x_3520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_pre_3521_, lean_object* v_declName_3522_, lean_object* v_as_3523_, size_t v_sz_3524_, size_t v_i_3525_, lean_object* v_b_3526_, lean_object* v___y_3527_, lean_object* v___y_3528_, lean_object* v___y_3529_, lean_object* v___y_3530_){
_start:
{
lean_object* v_a_3533_; uint8_t v___x_3537_; 
v___x_3537_ = lean_usize_dec_lt(v_i_3525_, v_sz_3524_);
if (v___x_3537_ == 0)
{
lean_object* v___x_3538_; 
lean_dec(v_declName_3522_);
lean_dec(v_pre_3521_);
v___x_3538_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3538_, 0, v_b_3526_);
return v___x_3538_;
}
else
{
lean_object* v___x_3539_; lean_object* v_a_3540_; lean_object* v___x_3541_; uint8_t v___x_3542_; 
v___x_3539_ = lean_box(0);
v_a_3540_ = lean_array_uget_borrowed(v_as_3523_, v_i_3525_);
lean_inc(v_a_3540_);
lean_inc(v_pre_3521_);
v___x_3541_ = l_Lean_Name_append(v_pre_3521_, v_a_3540_);
v___x_3542_ = lean_name_eq(v___x_3541_, v_declName_3522_);
lean_dec(v___x_3541_);
if (v___x_3542_ == 0)
{
v_a_3533_ = v___x_3539_;
goto v___jp_3532_;
}
else
{
lean_object* v___x_3543_; uint8_t v___x_3544_; lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; lean_object* v___x_3550_; lean_object* v___x_3551_; lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3554_; lean_object* v___x_3555_; lean_object* v___x_3556_; lean_object* v___x_3557_; 
v___x_3543_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1);
v___x_3544_ = 0;
lean_inc(v_declName_3522_);
v___x_3545_ = l_Lean_MessageData_ofConstName(v_declName_3522_, v___x_3544_);
v___x_3546_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3546_, 0, v___x_3543_);
lean_ctor_set(v___x_3546_, 1, v___x_3545_);
v___x_3547_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3);
v___x_3548_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3548_, 0, v___x_3546_);
lean_ctor_set(v___x_3548_, 1, v___x_3547_);
lean_inc(v_pre_3521_);
v___x_3549_ = l_Lean_MessageData_ofName(v_pre_3521_);
v___x_3550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3550_, 0, v___x_3548_);
lean_ctor_set(v___x_3550_, 1, v___x_3549_);
v___x_3551_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5);
v___x_3552_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3552_, 0, v___x_3550_);
lean_ctor_set(v___x_3552_, 1, v___x_3551_);
lean_inc(v_a_3540_);
v___x_3553_ = l_Lean_MessageData_ofName(v_a_3540_);
v___x_3554_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3554_, 0, v___x_3552_);
lean_ctor_set(v___x_3554_, 1, v___x_3553_);
v___x_3555_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3);
v___x_3556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3556_, 0, v___x_3554_);
lean_ctor_set(v___x_3556_, 1, v___x_3555_);
v___x_3557_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3556_, v___y_3527_, v___y_3528_, v___y_3529_, v___y_3530_);
if (lean_obj_tag(v___x_3557_) == 0)
{
lean_dec_ref_known(v___x_3557_, 1);
v_a_3533_ = v___x_3539_;
goto v___jp_3532_;
}
else
{
lean_dec(v_declName_3522_);
lean_dec(v_pre_3521_);
return v___x_3557_;
}
}
}
v___jp_3532_:
{
size_t v___x_3534_; size_t v___x_3535_; 
v___x_3534_ = ((size_t)1ULL);
v___x_3535_ = lean_usize_add(v_i_3525_, v___x_3534_);
v_i_3525_ = v___x_3535_;
v_b_3526_ = v_a_3533_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object* v_pre_3558_, lean_object* v_declName_3559_, lean_object* v_as_3560_, lean_object* v_sz_3561_, lean_object* v_i_3562_, lean_object* v_b_3563_, lean_object* v___y_3564_, lean_object* v___y_3565_, lean_object* v___y_3566_, lean_object* v___y_3567_, lean_object* v___y_3568_){
_start:
{
size_t v_sz_boxed_3569_; size_t v_i_boxed_3570_; lean_object* v_res_3571_; 
v_sz_boxed_3569_ = lean_unbox_usize(v_sz_3561_);
lean_dec(v_sz_3561_);
v_i_boxed_3570_ = lean_unbox_usize(v_i_3562_);
lean_dec(v_i_3562_);
v_res_3571_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_pre_3558_, v_declName_3559_, v_as_3560_, v_sz_boxed_3569_, v_i_boxed_3570_, v_b_3563_, v___y_3564_, v___y_3565_, v___y_3566_, v___y_3567_);
lean_dec(v___y_3567_);
lean_dec_ref(v___y_3566_);
lean_dec(v___y_3565_);
lean_dec_ref(v___y_3564_);
lean_dec_ref(v_as_3560_);
return v_res_3571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_declName_3572_, lean_object* v___y_3573_, lean_object* v___y_3574_, lean_object* v___y_3575_, lean_object* v___y_3576_){
_start:
{
if (lean_obj_tag(v_declName_3572_) == 1)
{
lean_object* v_pre_3578_; lean_object* v___x_3579_; lean_object* v_env_3580_; uint8_t v___x_3581_; 
v_pre_3578_ = lean_ctor_get(v_declName_3572_, 0);
lean_inc_n(v_pre_3578_, 2);
v___x_3579_ = lean_st_ref_get(v___y_3576_);
v_env_3580_ = lean_ctor_get(v___x_3579_, 0);
lean_inc_ref(v_env_3580_);
lean_dec(v___x_3579_);
v___x_3581_ = l_Lean_isStructure(v_env_3580_, v_pre_3578_);
if (v___x_3581_ == 0)
{
lean_object* v___x_3582_; lean_object* v___x_3583_; 
lean_dec_ref_known(v_declName_3572_, 2);
lean_dec(v_pre_3578_);
v___x_3582_ = lean_box(0);
v___x_3583_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3583_, 0, v___x_3582_);
return v___x_3583_;
}
else
{
lean_object* v___x_3584_; lean_object* v_env_3585_; lean_object* v_fieldNames_3586_; lean_object* v___x_3587_; size_t v_sz_3588_; size_t v___x_3589_; lean_object* v___x_3590_; 
v___x_3584_ = lean_st_ref_get(v___y_3576_);
v_env_3585_ = lean_ctor_get(v___x_3584_, 0);
lean_inc_ref(v_env_3585_);
lean_dec(v___x_3584_);
lean_inc(v_pre_3578_);
v_fieldNames_3586_ = l_Lean_getStructureFieldsFlattened(v_env_3585_, v_pre_3578_, v___x_3581_);
v___x_3587_ = lean_box(0);
v_sz_3588_ = lean_array_size(v_fieldNames_3586_);
v___x_3589_ = ((size_t)0ULL);
v___x_3590_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_pre_3578_, v_declName_3572_, v_fieldNames_3586_, v_sz_3588_, v___x_3589_, v___x_3587_, v___y_3573_, v___y_3574_, v___y_3575_, v___y_3576_);
lean_dec_ref(v_fieldNames_3586_);
if (lean_obj_tag(v___x_3590_) == 0)
{
lean_object* v___x_3592_; uint8_t v_isShared_3593_; uint8_t v_isSharedCheck_3597_; 
v_isSharedCheck_3597_ = !lean_is_exclusive(v___x_3590_);
if (v_isSharedCheck_3597_ == 0)
{
lean_object* v_unused_3598_; 
v_unused_3598_ = lean_ctor_get(v___x_3590_, 0);
lean_dec(v_unused_3598_);
v___x_3592_ = v___x_3590_;
v_isShared_3593_ = v_isSharedCheck_3597_;
goto v_resetjp_3591_;
}
else
{
lean_dec(v___x_3590_);
v___x_3592_ = lean_box(0);
v_isShared_3593_ = v_isSharedCheck_3597_;
goto v_resetjp_3591_;
}
v_resetjp_3591_:
{
lean_object* v___x_3595_; 
if (v_isShared_3593_ == 0)
{
lean_ctor_set(v___x_3592_, 0, v___x_3587_);
v___x_3595_ = v___x_3592_;
goto v_reusejp_3594_;
}
else
{
lean_object* v_reuseFailAlloc_3596_; 
v_reuseFailAlloc_3596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3596_, 0, v___x_3587_);
v___x_3595_ = v_reuseFailAlloc_3596_;
goto v_reusejp_3594_;
}
v_reusejp_3594_:
{
return v___x_3595_;
}
}
}
else
{
return v___x_3590_;
}
}
}
else
{
lean_object* v___x_3599_; lean_object* v___x_3600_; 
lean_dec(v_declName_3572_);
v___x_3599_ = lean_box(0);
v___x_3600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3600_, 0, v___x_3599_);
return v___x_3600_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_declName_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_, lean_object* v___y_3604_, lean_object* v___y_3605_, lean_object* v___y_3606_){
_start:
{
lean_object* v_res_3607_; 
v_res_3607_ = lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0(v_declName_3601_, v___y_3602_, v___y_3603_, v___y_3604_, v___y_3605_);
lean_dec(v___y_3605_);
lean_dec_ref(v___y_3604_);
lean_dec(v___y_3603_);
lean_dec_ref(v___y_3602_);
return v_res_3607_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1(void){
_start:
{
lean_object* v___x_3609_; lean_object* v___x_3610_; 
v___x_3609_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__0));
v___x_3610_ = l_Lean_stringToMessageData(v___x_3609_);
return v___x_3610_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3(void){
_start:
{
lean_object* v___x_3612_; lean_object* v___x_3613_; 
v___x_3612_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__2));
v___x_3613_ = l_Lean_stringToMessageData(v___x_3612_);
return v___x_3613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5(lean_object* v_addInfo_3614_, lean_object* v_declName_3615_, uint8_t v___x_3616_, lean_object* v___f_3617_, uint8_t v___x_3618_, lean_object* v_env_3619_, lean_object* v___f_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_, lean_object* v___y_3623_, lean_object* v___y_3624_){
_start:
{
lean_object* v___x_3626_; 
lean_inc(v___y_3624_);
lean_inc_ref(v___y_3623_);
lean_inc(v___y_3622_);
lean_inc_ref(v___y_3621_);
lean_inc(v_declName_3615_);
v___x_3626_ = lean_apply_6(v_addInfo_3614_, v_declName_3615_, v___y_3621_, v___y_3622_, v___y_3623_, v___y_3624_, lean_box(0));
if (lean_obj_tag(v___x_3626_) == 0)
{
lean_object* v___x_3627_; 
lean_dec_ref_known(v___x_3626_, 1);
lean_inc(v_declName_3615_);
v___x_3627_ = l_Lean_privateToUserName_x3f(v_declName_3615_);
if (lean_obj_tag(v___x_3627_) == 0)
{
lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3631_; lean_object* v___x_3632_; lean_object* v___x_3633_; 
v___x_3628_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3);
v___x_3629_ = l_Lean_MessageData_ofConstName(v_declName_3615_, v___x_3616_);
v___x_3630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3630_, 0, v___x_3628_);
lean_ctor_set(v___x_3630_, 1, v___x_3629_);
v___x_3631_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1);
v___x_3632_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3632_, 0, v___x_3630_);
lean_ctor_set(v___x_3632_, 1, v___x_3631_);
v___x_3633_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3632_, v___y_3621_, v___y_3622_, v___y_3623_, v___y_3624_);
lean_dec(v___y_3624_);
lean_dec_ref(v___y_3623_);
lean_dec(v___y_3622_);
lean_dec_ref(v___y_3621_);
return v___x_3633_;
}
else
{
lean_object* v_val_3634_; lean_object* v___x_3635_; lean_object* v___x_3636_; lean_object* v___x_3637_; lean_object* v___x_3638_; lean_object* v___x_3639_; lean_object* v___x_3640_; 
lean_dec(v_declName_3615_);
v_val_3634_ = lean_ctor_get(v___x_3627_, 0);
lean_inc(v_val_3634_);
lean_dec_ref_known(v___x_3627_, 1);
v___x_3635_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__3);
v___x_3636_ = l_Lean_MessageData_ofConstName(v_val_3634_, v___x_3616_);
v___x_3637_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3637_, 0, v___x_3635_);
lean_ctor_set(v___x_3637_, 1, v___x_3636_);
v___x_3638_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1);
v___x_3639_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3639_, 0, v___x_3637_);
lean_ctor_set(v___x_3639_, 1, v___x_3638_);
v___x_3640_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3639_, v___y_3621_, v___y_3622_, v___y_3623_, v___y_3624_);
lean_dec(v___y_3624_);
lean_dec_ref(v___y_3623_);
lean_dec(v___y_3622_);
lean_dec_ref(v___y_3621_);
return v___x_3640_;
}
}
else
{
lean_dec(v___y_3624_);
lean_dec_ref(v___y_3623_);
lean_dec(v___y_3622_);
lean_dec_ref(v___y_3621_);
lean_dec(v_declName_3615_);
return v___x_3626_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___boxed(lean_object* v_addInfo_3641_, lean_object* v_declName_3642_, lean_object* v___x_3643_, lean_object* v___f_3644_, lean_object* v___x_3645_, lean_object* v_env_3646_, lean_object* v___f_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_, lean_object* v___y_3651_, lean_object* v___y_3652_){
_start:
{
uint8_t v___x_10244__boxed_3653_; uint8_t v___x_10246__boxed_3654_; lean_object* v_res_3655_; 
v___x_10244__boxed_3653_ = lean_unbox(v___x_3643_);
v___x_10246__boxed_3654_ = lean_unbox(v___x_3645_);
v_res_3655_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5(v_addInfo_3641_, v_declName_3642_, v___x_10244__boxed_3653_, v___f_3644_, v___x_10246__boxed_3654_, v_env_3646_, v___f_3647_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_);
lean_dec_ref(v___f_3647_);
lean_dec_ref(v_env_3646_);
lean_dec_ref(v___f_3644_);
return v_res_3655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg(lean_object* v_t_3656_, lean_object* v___y_3657_){
_start:
{
lean_object* v___x_3659_; lean_object* v_infoState_3660_; uint8_t v_enabled_3661_; 
v___x_3659_ = lean_st_ref_get(v___y_3657_);
v_infoState_3660_ = lean_ctor_get(v___x_3659_, 7);
lean_inc_ref(v_infoState_3660_);
lean_dec(v___x_3659_);
v_enabled_3661_ = lean_ctor_get_uint8(v_infoState_3660_, sizeof(void*)*3);
lean_dec_ref(v_infoState_3660_);
if (v_enabled_3661_ == 0)
{
lean_object* v___x_3662_; lean_object* v___x_3663_; 
lean_dec_ref(v_t_3656_);
v___x_3662_ = lean_box(0);
v___x_3663_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3663_, 0, v___x_3662_);
return v___x_3663_;
}
else
{
lean_object* v___x_3664_; lean_object* v_infoState_3665_; lean_object* v_env_3666_; lean_object* v_nextMacroScope_3667_; lean_object* v_ngen_3668_; lean_object* v_auxDeclNGen_3669_; lean_object* v_traceState_3670_; lean_object* v_cache_3671_; lean_object* v_messages_3672_; lean_object* v_snapshotTasks_3673_; lean_object* v___x_3675_; uint8_t v_isShared_3676_; uint8_t v_isSharedCheck_3695_; 
v___x_3664_ = lean_st_ref_take(v___y_3657_);
v_infoState_3665_ = lean_ctor_get(v___x_3664_, 7);
v_env_3666_ = lean_ctor_get(v___x_3664_, 0);
v_nextMacroScope_3667_ = lean_ctor_get(v___x_3664_, 1);
v_ngen_3668_ = lean_ctor_get(v___x_3664_, 2);
v_auxDeclNGen_3669_ = lean_ctor_get(v___x_3664_, 3);
v_traceState_3670_ = lean_ctor_get(v___x_3664_, 4);
v_cache_3671_ = lean_ctor_get(v___x_3664_, 5);
v_messages_3672_ = lean_ctor_get(v___x_3664_, 6);
v_snapshotTasks_3673_ = lean_ctor_get(v___x_3664_, 8);
v_isSharedCheck_3695_ = !lean_is_exclusive(v___x_3664_);
if (v_isSharedCheck_3695_ == 0)
{
v___x_3675_ = v___x_3664_;
v_isShared_3676_ = v_isSharedCheck_3695_;
goto v_resetjp_3674_;
}
else
{
lean_inc(v_snapshotTasks_3673_);
lean_inc(v_infoState_3665_);
lean_inc(v_messages_3672_);
lean_inc(v_cache_3671_);
lean_inc(v_traceState_3670_);
lean_inc(v_auxDeclNGen_3669_);
lean_inc(v_ngen_3668_);
lean_inc(v_nextMacroScope_3667_);
lean_inc(v_env_3666_);
lean_dec(v___x_3664_);
v___x_3675_ = lean_box(0);
v_isShared_3676_ = v_isSharedCheck_3695_;
goto v_resetjp_3674_;
}
v_resetjp_3674_:
{
uint8_t v_enabled_3677_; lean_object* v_assignment_3678_; lean_object* v_lazyAssignment_3679_; lean_object* v_trees_3680_; lean_object* v___x_3682_; uint8_t v_isShared_3683_; uint8_t v_isSharedCheck_3694_; 
v_enabled_3677_ = lean_ctor_get_uint8(v_infoState_3665_, sizeof(void*)*3);
v_assignment_3678_ = lean_ctor_get(v_infoState_3665_, 0);
v_lazyAssignment_3679_ = lean_ctor_get(v_infoState_3665_, 1);
v_trees_3680_ = lean_ctor_get(v_infoState_3665_, 2);
v_isSharedCheck_3694_ = !lean_is_exclusive(v_infoState_3665_);
if (v_isSharedCheck_3694_ == 0)
{
v___x_3682_ = v_infoState_3665_;
v_isShared_3683_ = v_isSharedCheck_3694_;
goto v_resetjp_3681_;
}
else
{
lean_inc(v_trees_3680_);
lean_inc(v_lazyAssignment_3679_);
lean_inc(v_assignment_3678_);
lean_dec(v_infoState_3665_);
v___x_3682_ = lean_box(0);
v_isShared_3683_ = v_isSharedCheck_3694_;
goto v_resetjp_3681_;
}
v_resetjp_3681_:
{
lean_object* v___x_3684_; lean_object* v___x_3686_; 
v___x_3684_ = l_Lean_PersistentArray_push___redArg(v_trees_3680_, v_t_3656_);
if (v_isShared_3683_ == 0)
{
lean_ctor_set(v___x_3682_, 2, v___x_3684_);
v___x_3686_ = v___x_3682_;
goto v_reusejp_3685_;
}
else
{
lean_object* v_reuseFailAlloc_3693_; 
v_reuseFailAlloc_3693_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_3693_, 0, v_assignment_3678_);
lean_ctor_set(v_reuseFailAlloc_3693_, 1, v_lazyAssignment_3679_);
lean_ctor_set(v_reuseFailAlloc_3693_, 2, v___x_3684_);
lean_ctor_set_uint8(v_reuseFailAlloc_3693_, sizeof(void*)*3, v_enabled_3677_);
v___x_3686_ = v_reuseFailAlloc_3693_;
goto v_reusejp_3685_;
}
v_reusejp_3685_:
{
lean_object* v___x_3688_; 
if (v_isShared_3676_ == 0)
{
lean_ctor_set(v___x_3675_, 7, v___x_3686_);
v___x_3688_ = v___x_3675_;
goto v_reusejp_3687_;
}
else
{
lean_object* v_reuseFailAlloc_3692_; 
v_reuseFailAlloc_3692_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3692_, 0, v_env_3666_);
lean_ctor_set(v_reuseFailAlloc_3692_, 1, v_nextMacroScope_3667_);
lean_ctor_set(v_reuseFailAlloc_3692_, 2, v_ngen_3668_);
lean_ctor_set(v_reuseFailAlloc_3692_, 3, v_auxDeclNGen_3669_);
lean_ctor_set(v_reuseFailAlloc_3692_, 4, v_traceState_3670_);
lean_ctor_set(v_reuseFailAlloc_3692_, 5, v_cache_3671_);
lean_ctor_set(v_reuseFailAlloc_3692_, 6, v_messages_3672_);
lean_ctor_set(v_reuseFailAlloc_3692_, 7, v___x_3686_);
lean_ctor_set(v_reuseFailAlloc_3692_, 8, v_snapshotTasks_3673_);
v___x_3688_ = v_reuseFailAlloc_3692_;
goto v_reusejp_3687_;
}
v_reusejp_3687_:
{
lean_object* v___x_3689_; lean_object* v___x_3690_; lean_object* v___x_3691_; 
v___x_3689_ = lean_st_ref_set(v___y_3657_, v___x_3688_);
v___x_3690_ = lean_box(0);
v___x_3691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3691_, 0, v___x_3690_);
return v___x_3691_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg___boxed(lean_object* v_t_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
lean_object* v_res_3699_; 
v_res_3699_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg(v_t_3696_, v___y_3697_);
lean_dec(v___y_3697_);
return v_res_3699_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0(void){
_start:
{
lean_object* v___x_3700_; lean_object* v___x_3701_; lean_object* v___x_3702_; 
v___x_3700_ = lean_unsigned_to_nat(32u);
v___x_3701_ = lean_mk_empty_array_with_capacity(v___x_3700_);
v___x_3702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3702_, 0, v___x_3701_);
return v___x_3702_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1(void){
_start:
{
size_t v___x_3703_; lean_object* v___x_3704_; lean_object* v___x_3705_; lean_object* v___x_3706_; lean_object* v___x_3707_; lean_object* v___x_3708_; 
v___x_3703_ = ((size_t)5ULL);
v___x_3704_ = lean_unsigned_to_nat(0u);
v___x_3705_ = lean_unsigned_to_nat(32u);
v___x_3706_ = lean_mk_empty_array_with_capacity(v___x_3705_);
v___x_3707_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__0);
v___x_3708_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3708_, 0, v___x_3707_);
lean_ctor_set(v___x_3708_, 1, v___x_3706_);
lean_ctor_set(v___x_3708_, 2, v___x_3704_);
lean_ctor_set(v___x_3708_, 3, v___x_3704_);
lean_ctor_set_usize(v___x_3708_, 4, v___x_3703_);
return v___x_3708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6(lean_object* v_t_3709_, lean_object* v___y_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_, lean_object* v___y_3713_){
_start:
{
lean_object* v___x_3715_; lean_object* v_infoState_3716_; uint8_t v_enabled_3717_; 
v___x_3715_ = lean_st_ref_get(v___y_3713_);
v_infoState_3716_ = lean_ctor_get(v___x_3715_, 7);
lean_inc_ref(v_infoState_3716_);
lean_dec(v___x_3715_);
v_enabled_3717_ = lean_ctor_get_uint8(v_infoState_3716_, sizeof(void*)*3);
lean_dec_ref(v_infoState_3716_);
if (v_enabled_3717_ == 0)
{
lean_object* v___x_3718_; lean_object* v___x_3719_; 
lean_dec_ref(v_t_3709_);
v___x_3718_ = lean_box(0);
v___x_3719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3719_, 0, v___x_3718_);
return v___x_3719_;
}
else
{
lean_object* v___x_3720_; lean_object* v___x_3721_; lean_object* v___x_3722_; 
v___x_3720_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___closed__1);
v___x_3721_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3721_, 0, v_t_3709_);
lean_ctor_set(v___x_3721_, 1, v___x_3720_);
v___x_3722_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg(v___x_3721_, v___y_3713_);
return v___x_3722_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6___boxed(lean_object* v_t_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_){
_start:
{
lean_object* v_res_3729_; 
v_res_3729_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6(v_t_3723_, v___y_3724_, v___y_3725_, v___y_3726_, v___y_3727_);
lean_dec(v___y_3727_);
lean_dec_ref(v___y_3726_);
lean_dec(v___y_3725_);
lean_dec_ref(v___y_3724_);
return v_res_3729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0(uint8_t v___x_3730_, lean_object* v_declName_3731_, lean_object* v___y_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_){
_start:
{
lean_object* v_ref_3737_; lean_object* v___x_3738_; 
v_ref_3737_ = lean_ctor_get(v___y_3734_, 5);
v___x_3738_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__5(v_declName_3731_, v___y_3732_, v___y_3733_, v___y_3734_, v___y_3735_);
if (lean_obj_tag(v___x_3738_) == 0)
{
lean_object* v_a_3739_; lean_object* v___x_3740_; lean_object* v___x_3741_; lean_object* v___x_3742_; lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___x_3748_; 
v_a_3739_ = lean_ctor_get(v___x_3738_, 0);
lean_inc(v_a_3739_);
lean_dec_ref_known(v___x_3738_, 1);
v___x_3740_ = lean_box(0);
lean_inc(v_ref_3737_);
v___x_3741_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3741_, 0, v___x_3740_);
lean_ctor_set(v___x_3741_, 1, v_ref_3737_);
v___x_3742_ = lean_unsigned_to_nat(32u);
v___x_3743_ = lean_mk_empty_array_with_capacity(v___x_3742_);
lean_dec_ref(v___x_3743_);
v___x_3744_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__2);
v___x_3745_ = lean_box(0);
v___x_3746_ = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(v___x_3746_, 0, v___x_3741_);
lean_ctor_set(v___x_3746_, 1, v___x_3744_);
lean_ctor_set(v___x_3746_, 2, v___x_3745_);
lean_ctor_set(v___x_3746_, 3, v_a_3739_);
lean_ctor_set_uint8(v___x_3746_, sizeof(void*)*4, v___x_3730_);
lean_ctor_set_uint8(v___x_3746_, sizeof(void*)*4 + 1, v___x_3730_);
v___x_3747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3747_, 0, v___x_3746_);
v___x_3748_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6(v___x_3747_, v___y_3732_, v___y_3733_, v___y_3734_, v___y_3735_);
return v___x_3748_;
}
else
{
lean_object* v_a_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3756_; 
v_a_3749_ = lean_ctor_get(v___x_3738_, 0);
v_isSharedCheck_3756_ = !lean_is_exclusive(v___x_3738_);
if (v_isSharedCheck_3756_ == 0)
{
v___x_3751_ = v___x_3738_;
v_isShared_3752_ = v_isSharedCheck_3756_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_a_3749_);
lean_dec(v___x_3738_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3756_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v___x_3754_; 
if (v_isShared_3752_ == 0)
{
v___x_3754_ = v___x_3751_;
goto v_reusejp_3753_;
}
else
{
lean_object* v_reuseFailAlloc_3755_; 
v_reuseFailAlloc_3755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3755_, 0, v_a_3749_);
v___x_3754_ = v_reuseFailAlloc_3755_;
goto v_reusejp_3753_;
}
v_reusejp_3753_:
{
return v___x_3754_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0___boxed(lean_object* v___x_3757_, lean_object* v_declName_3758_, lean_object* v___y_3759_, lean_object* v___y_3760_, lean_object* v___y_3761_, lean_object* v___y_3762_, lean_object* v___y_3763_){
_start:
{
uint8_t v___x_10436__boxed_3764_; lean_object* v_res_3765_; 
v___x_10436__boxed_3764_ = lean_unbox(v___x_3757_);
v_res_3765_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__0(v___x_10436__boxed_3764_, v_declName_3758_, v___y_3759_, v___y_3760_, v___y_3761_, v___y_3762_);
lean_dec(v___y_3762_);
lean_dec_ref(v___y_3761_);
lean_dec(v___y_3760_);
lean_dec_ref(v___y_3759_);
return v_res_3765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4(lean_object* v___f_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_){
_start:
{
lean_object* v___x_3772_; lean_object* v_env_3773_; lean_object* v___x_3774_; 
v___x_3772_ = lean_st_ref_get(v___y_3770_);
v_env_3773_ = lean_ctor_get(v___x_3772_, 0);
lean_inc_ref(v_env_3773_);
lean_dec(v___x_3772_);
v___x_3774_ = lean_apply_6(v___f_3766_, v_env_3773_, v___y_3767_, v___y_3768_, v___y_3769_, v___y_3770_, lean_box(0));
return v___x_3774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4___boxed(lean_object* v___f_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_, lean_object* v___y_3778_, lean_object* v___y_3779_, lean_object* v___y_3780_){
_start:
{
lean_object* v_res_3781_; 
v_res_3781_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4(v___f_3775_, v___y_3776_, v___y_3777_, v___y_3778_, v___y_3779_);
return v_res_3781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(lean_object* v_env_3782_, lean_object* v___y_3783_, lean_object* v___y_3784_){
_start:
{
lean_object* v___x_3786_; lean_object* v_nextMacroScope_3787_; lean_object* v_ngen_3788_; lean_object* v_auxDeclNGen_3789_; lean_object* v_traceState_3790_; lean_object* v_messages_3791_; lean_object* v_infoState_3792_; lean_object* v_snapshotTasks_3793_; lean_object* v___x_3795_; uint8_t v_isShared_3796_; uint8_t v_isSharedCheck_3819_; 
v___x_3786_ = lean_st_ref_take(v___y_3784_);
v_nextMacroScope_3787_ = lean_ctor_get(v___x_3786_, 1);
v_ngen_3788_ = lean_ctor_get(v___x_3786_, 2);
v_auxDeclNGen_3789_ = lean_ctor_get(v___x_3786_, 3);
v_traceState_3790_ = lean_ctor_get(v___x_3786_, 4);
v_messages_3791_ = lean_ctor_get(v___x_3786_, 6);
v_infoState_3792_ = lean_ctor_get(v___x_3786_, 7);
v_snapshotTasks_3793_ = lean_ctor_get(v___x_3786_, 8);
v_isSharedCheck_3819_ = !lean_is_exclusive(v___x_3786_);
if (v_isSharedCheck_3819_ == 0)
{
lean_object* v_unused_3820_; lean_object* v_unused_3821_; 
v_unused_3820_ = lean_ctor_get(v___x_3786_, 5);
lean_dec(v_unused_3820_);
v_unused_3821_ = lean_ctor_get(v___x_3786_, 0);
lean_dec(v_unused_3821_);
v___x_3795_ = v___x_3786_;
v_isShared_3796_ = v_isSharedCheck_3819_;
goto v_resetjp_3794_;
}
else
{
lean_inc(v_snapshotTasks_3793_);
lean_inc(v_infoState_3792_);
lean_inc(v_messages_3791_);
lean_inc(v_traceState_3790_);
lean_inc(v_auxDeclNGen_3789_);
lean_inc(v_ngen_3788_);
lean_inc(v_nextMacroScope_3787_);
lean_dec(v___x_3786_);
v___x_3795_ = lean_box(0);
v_isShared_3796_ = v_isSharedCheck_3819_;
goto v_resetjp_3794_;
}
v_resetjp_3794_:
{
lean_object* v___x_3797_; lean_object* v___x_3799_; 
v___x_3797_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2);
if (v_isShared_3796_ == 0)
{
lean_ctor_set(v___x_3795_, 5, v___x_3797_);
lean_ctor_set(v___x_3795_, 0, v_env_3782_);
v___x_3799_ = v___x_3795_;
goto v_reusejp_3798_;
}
else
{
lean_object* v_reuseFailAlloc_3818_; 
v_reuseFailAlloc_3818_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3818_, 0, v_env_3782_);
lean_ctor_set(v_reuseFailAlloc_3818_, 1, v_nextMacroScope_3787_);
lean_ctor_set(v_reuseFailAlloc_3818_, 2, v_ngen_3788_);
lean_ctor_set(v_reuseFailAlloc_3818_, 3, v_auxDeclNGen_3789_);
lean_ctor_set(v_reuseFailAlloc_3818_, 4, v_traceState_3790_);
lean_ctor_set(v_reuseFailAlloc_3818_, 5, v___x_3797_);
lean_ctor_set(v_reuseFailAlloc_3818_, 6, v_messages_3791_);
lean_ctor_set(v_reuseFailAlloc_3818_, 7, v_infoState_3792_);
lean_ctor_set(v_reuseFailAlloc_3818_, 8, v_snapshotTasks_3793_);
v___x_3799_ = v_reuseFailAlloc_3818_;
goto v_reusejp_3798_;
}
v_reusejp_3798_:
{
lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v_mctx_3802_; lean_object* v_zetaDeltaFVarIds_3803_; lean_object* v_postponed_3804_; lean_object* v_diag_3805_; lean_object* v___x_3807_; uint8_t v_isShared_3808_; uint8_t v_isSharedCheck_3816_; 
v___x_3800_ = lean_st_ref_set(v___y_3784_, v___x_3799_);
v___x_3801_ = lean_st_ref_take(v___y_3783_);
v_mctx_3802_ = lean_ctor_get(v___x_3801_, 0);
v_zetaDeltaFVarIds_3803_ = lean_ctor_get(v___x_3801_, 2);
v_postponed_3804_ = lean_ctor_get(v___x_3801_, 3);
v_diag_3805_ = lean_ctor_get(v___x_3801_, 4);
v_isSharedCheck_3816_ = !lean_is_exclusive(v___x_3801_);
if (v_isSharedCheck_3816_ == 0)
{
lean_object* v_unused_3817_; 
v_unused_3817_ = lean_ctor_get(v___x_3801_, 1);
lean_dec(v_unused_3817_);
v___x_3807_ = v___x_3801_;
v_isShared_3808_ = v_isSharedCheck_3816_;
goto v_resetjp_3806_;
}
else
{
lean_inc(v_diag_3805_);
lean_inc(v_postponed_3804_);
lean_inc(v_zetaDeltaFVarIds_3803_);
lean_inc(v_mctx_3802_);
lean_dec(v___x_3801_);
v___x_3807_ = lean_box(0);
v_isShared_3808_ = v_isSharedCheck_3816_;
goto v_resetjp_3806_;
}
v_resetjp_3806_:
{
lean_object* v___x_3809_; lean_object* v___x_3811_; 
v___x_3809_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3);
if (v_isShared_3808_ == 0)
{
lean_ctor_set(v___x_3807_, 1, v___x_3809_);
v___x_3811_ = v___x_3807_;
goto v_reusejp_3810_;
}
else
{
lean_object* v_reuseFailAlloc_3815_; 
v_reuseFailAlloc_3815_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3815_, 0, v_mctx_3802_);
lean_ctor_set(v_reuseFailAlloc_3815_, 1, v___x_3809_);
lean_ctor_set(v_reuseFailAlloc_3815_, 2, v_zetaDeltaFVarIds_3803_);
lean_ctor_set(v_reuseFailAlloc_3815_, 3, v_postponed_3804_);
lean_ctor_set(v_reuseFailAlloc_3815_, 4, v_diag_3805_);
v___x_3811_ = v_reuseFailAlloc_3815_;
goto v_reusejp_3810_;
}
v_reusejp_3810_:
{
lean_object* v___x_3812_; lean_object* v___x_3813_; lean_object* v___x_3814_; 
v___x_3812_ = lean_st_ref_set(v___y_3783_, v___x_3811_);
v___x_3813_ = lean_box(0);
v___x_3814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3814_, 0, v___x_3813_);
return v___x_3814_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg___boxed(lean_object* v_env_3822_, lean_object* v___y_3823_, lean_object* v___y_3824_, lean_object* v___y_3825_){
_start:
{
lean_object* v_res_3826_; 
v_res_3826_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(v_env_3822_, v___y_3823_, v___y_3824_);
lean_dec(v___y_3824_);
lean_dec(v___y_3823_);
return v_res_3826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(lean_object* v_env_3827_, lean_object* v_x_3828_, lean_object* v___y_3829_, lean_object* v___y_3830_, lean_object* v___y_3831_, lean_object* v___y_3832_){
_start:
{
lean_object* v___x_3834_; lean_object* v_env_3835_; lean_object* v_a_3837_; lean_object* v___x_3847_; lean_object* v___x_3848_; 
v___x_3834_ = lean_st_ref_get(v___y_3832_);
v_env_3835_ = lean_ctor_get(v___x_3834_, 0);
lean_inc_ref(v_env_3835_);
lean_dec(v___x_3834_);
v___x_3847_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(v_env_3827_, v___y_3830_, v___y_3832_);
lean_dec_ref(v___x_3847_);
lean_inc(v___y_3832_);
lean_inc_ref(v___y_3831_);
lean_inc(v___y_3830_);
lean_inc_ref(v___y_3829_);
v___x_3848_ = lean_apply_5(v_x_3828_, v___y_3829_, v___y_3830_, v___y_3831_, v___y_3832_, lean_box(0));
if (lean_obj_tag(v___x_3848_) == 0)
{
lean_object* v_a_3849_; lean_object* v___x_3850_; lean_object* v___x_3852_; uint8_t v_isShared_3853_; uint8_t v_isSharedCheck_3857_; 
v_a_3849_ = lean_ctor_get(v___x_3848_, 0);
lean_inc(v_a_3849_);
lean_dec_ref_known(v___x_3848_, 1);
v___x_3850_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(v_env_3835_, v___y_3830_, v___y_3832_);
v_isSharedCheck_3857_ = !lean_is_exclusive(v___x_3850_);
if (v_isSharedCheck_3857_ == 0)
{
lean_object* v_unused_3858_; 
v_unused_3858_ = lean_ctor_get(v___x_3850_, 0);
lean_dec(v_unused_3858_);
v___x_3852_ = v___x_3850_;
v_isShared_3853_ = v_isSharedCheck_3857_;
goto v_resetjp_3851_;
}
else
{
lean_dec(v___x_3850_);
v___x_3852_ = lean_box(0);
v_isShared_3853_ = v_isSharedCheck_3857_;
goto v_resetjp_3851_;
}
v_resetjp_3851_:
{
lean_object* v___x_3855_; 
if (v_isShared_3853_ == 0)
{
lean_ctor_set(v___x_3852_, 0, v_a_3849_);
v___x_3855_ = v___x_3852_;
goto v_reusejp_3854_;
}
else
{
lean_object* v_reuseFailAlloc_3856_; 
v_reuseFailAlloc_3856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3856_, 0, v_a_3849_);
v___x_3855_ = v_reuseFailAlloc_3856_;
goto v_reusejp_3854_;
}
v_reusejp_3854_:
{
return v___x_3855_;
}
}
}
else
{
lean_object* v_a_3859_; 
v_a_3859_ = lean_ctor_get(v___x_3848_, 0);
lean_inc(v_a_3859_);
lean_dec_ref_known(v___x_3848_, 1);
v_a_3837_ = v_a_3859_;
goto v___jp_3836_;
}
v___jp_3836_:
{
lean_object* v___x_3838_; lean_object* v___x_3840_; uint8_t v_isShared_3841_; uint8_t v_isSharedCheck_3845_; 
v___x_3838_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(v_env_3835_, v___y_3830_, v___y_3832_);
v_isSharedCheck_3845_ = !lean_is_exclusive(v___x_3838_);
if (v_isSharedCheck_3845_ == 0)
{
lean_object* v_unused_3846_; 
v_unused_3846_ = lean_ctor_get(v___x_3838_, 0);
lean_dec(v_unused_3846_);
v___x_3840_ = v___x_3838_;
v_isShared_3841_ = v_isSharedCheck_3845_;
goto v_resetjp_3839_;
}
else
{
lean_dec(v___x_3838_);
v___x_3840_ = lean_box(0);
v_isShared_3841_ = v_isSharedCheck_3845_;
goto v_resetjp_3839_;
}
v_resetjp_3839_:
{
lean_object* v___x_3843_; 
if (v_isShared_3841_ == 0)
{
lean_ctor_set_tag(v___x_3840_, 1);
lean_ctor_set(v___x_3840_, 0, v_a_3837_);
v___x_3843_ = v___x_3840_;
goto v_reusejp_3842_;
}
else
{
lean_object* v_reuseFailAlloc_3844_; 
v_reuseFailAlloc_3844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3844_, 0, v_a_3837_);
v___x_3843_ = v_reuseFailAlloc_3844_;
goto v_reusejp_3842_;
}
v_reusejp_3842_:
{
return v___x_3843_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg___boxed(lean_object* v_env_3860_, lean_object* v_x_3861_, lean_object* v___y_3862_, lean_object* v___y_3863_, lean_object* v___y_3864_, lean_object* v___y_3865_, lean_object* v___y_3866_){
_start:
{
lean_object* v_res_3867_; 
v_res_3867_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(v_env_3860_, v_x_3861_, v___y_3862_, v___y_3863_, v___y_3864_, v___y_3865_);
lean_dec(v___y_3865_);
lean_dec_ref(v___y_3864_);
lean_dec(v___y_3863_);
lean_dec_ref(v___y_3862_);
return v_res_3867_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1(void){
_start:
{
lean_object* v___x_3869_; lean_object* v___x_3870_; 
v___x_3869_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__0));
v___x_3870_ = l_Lean_stringToMessageData(v___x_3869_);
return v___x_3870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3(lean_object* v___f_3871_, lean_object* v_declName_3872_, uint8_t v___x_3873_, lean_object* v_env_3874_, lean_object* v_____do__lift_3875_, lean_object* v___y_3876_, lean_object* v___y_3877_, lean_object* v___y_3878_, lean_object* v___y_3879_){
_start:
{
uint8_t v___y_3882_; lean_object* v___x_3891_; uint8_t v___x_3892_; 
lean_inc(v_declName_3872_);
v___x_3891_ = l_Lean_privateToUserName(v_declName_3872_);
lean_inc_ref(v_env_3874_);
v___x_3892_ = lean_is_reserved_name(v_env_3874_, v___x_3891_);
if (v___x_3892_ == 0)
{
lean_object* v___x_3893_; uint8_t v___x_3894_; 
lean_inc(v_declName_3872_);
v___x_3893_ = l_Lean_mkPrivateName(v_____do__lift_3875_, v_declName_3872_);
v___x_3894_ = lean_is_reserved_name(v_env_3874_, v___x_3893_);
v___y_3882_ = v___x_3894_;
goto v___jp_3881_;
}
else
{
lean_dec_ref(v_env_3874_);
v___y_3882_ = v___x_3892_;
goto v___jp_3881_;
}
v___jp_3881_:
{
if (v___y_3882_ == 0)
{
lean_object* v___x_3883_; lean_object* v___x_3884_; 
lean_dec(v_declName_3872_);
v___x_3883_ = lean_box(0);
lean_inc(v___y_3879_);
lean_inc_ref(v___y_3878_);
lean_inc(v___y_3877_);
lean_inc_ref(v___y_3876_);
v___x_3884_ = lean_apply_6(v___f_3871_, v___x_3883_, v___y_3876_, v___y_3877_, v___y_3878_, v___y_3879_, lean_box(0));
return v___x_3884_;
}
else
{
lean_object* v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; 
lean_dec_ref(v___f_3871_);
v___x_3885_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3);
v___x_3886_ = l_Lean_MessageData_ofConstName(v_declName_3872_, v___x_3873_);
v___x_3887_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3887_, 0, v___x_3885_);
lean_ctor_set(v___x_3887_, 1, v___x_3886_);
v___x_3888_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___closed__1);
v___x_3889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3889_, 0, v___x_3887_);
lean_ctor_set(v___x_3889_, 1, v___x_3888_);
v___x_3890_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3889_, v___y_3876_, v___y_3877_, v___y_3878_, v___y_3879_);
return v___x_3890_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___boxed(lean_object* v___f_3895_, lean_object* v_declName_3896_, lean_object* v___x_3897_, lean_object* v_env_3898_, lean_object* v_____do__lift_3899_, lean_object* v___y_3900_, lean_object* v___y_3901_, lean_object* v___y_3902_, lean_object* v___y_3903_, lean_object* v___y_3904_){
_start:
{
uint8_t v___x_10676__boxed_3905_; lean_object* v_res_3906_; 
v___x_10676__boxed_3905_ = lean_unbox(v___x_3897_);
v_res_3906_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3(v___f_3895_, v_declName_3896_, v___x_10676__boxed_3905_, v_env_3898_, v_____do__lift_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
lean_dec(v___y_3903_);
lean_dec_ref(v___y_3902_);
lean_dec(v___y_3901_);
lean_dec_ref(v___y_3900_);
lean_dec_ref(v_____do__lift_3899_);
return v_res_3906_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1(void){
_start:
{
lean_object* v___x_3908_; lean_object* v___x_3909_; 
v___x_3908_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__0));
v___x_3909_ = l_Lean_stringToMessageData(v___x_3908_);
return v___x_3909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2(lean_object* v_env_3910_, lean_object* v_declName_3911_, lean_object* v___f_3912_, lean_object* v_addInfo_3913_, lean_object* v_____r_3914_, lean_object* v___y_3915_, lean_object* v___y_3916_, lean_object* v___y_3917_, lean_object* v___y_3918_){
_start:
{
lean_object* v___x_3920_; uint8_t v___x_3921_; uint8_t v___x_3922_; 
lean_inc(v_declName_3911_);
v___x_3920_ = l_Lean_mkPrivateName(v_env_3910_, v_declName_3911_);
v___x_3921_ = 1;
lean_inc(v___x_3920_);
v___x_3922_ = l_Lean_Environment_contains(v_env_3910_, v___x_3920_, v___x_3921_);
if (v___x_3922_ == 0)
{
lean_object* v___x_3923_; lean_object* v___x_3924_; 
lean_dec(v___x_3920_);
lean_dec_ref(v_addInfo_3913_);
lean_dec(v_declName_3911_);
v___x_3923_ = lean_box(0);
lean_inc(v___y_3918_);
lean_inc_ref(v___y_3917_);
lean_inc(v___y_3916_);
lean_inc_ref(v___y_3915_);
v___x_3924_ = lean_apply_6(v___f_3912_, v___x_3923_, v___y_3915_, v___y_3916_, v___y_3917_, v___y_3918_, lean_box(0));
return v___x_3924_;
}
else
{
lean_object* v___x_3925_; 
lean_dec_ref(v___f_3912_);
lean_inc(v___y_3918_);
lean_inc_ref(v___y_3917_);
lean_inc(v___y_3916_);
lean_inc_ref(v___y_3915_);
v___x_3925_ = lean_apply_6(v_addInfo_3913_, v___x_3920_, v___y_3915_, v___y_3916_, v___y_3917_, v___y_3918_, lean_box(0));
if (lean_obj_tag(v___x_3925_) == 0)
{
lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; 
lean_dec_ref_known(v___x_3925_, 1);
v___x_3926_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___closed__1);
v___x_3927_ = l_Lean_MessageData_ofConstName(v_declName_3911_, v___x_3921_);
v___x_3928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3928_, 0, v___x_3926_);
lean_ctor_set(v___x_3928_, 1, v___x_3927_);
v___x_3929_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1);
v___x_3930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3930_, 0, v___x_3928_);
lean_ctor_set(v___x_3930_, 1, v___x_3929_);
v___x_3931_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3930_, v___y_3915_, v___y_3916_, v___y_3917_, v___y_3918_);
return v___x_3931_;
}
else
{
lean_dec(v_declName_3911_);
return v___x_3925_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___boxed(lean_object* v_env_3932_, lean_object* v_declName_3933_, lean_object* v___f_3934_, lean_object* v_addInfo_3935_, lean_object* v_____r_3936_, lean_object* v___y_3937_, lean_object* v___y_3938_, lean_object* v___y_3939_, lean_object* v___y_3940_, lean_object* v___y_3941_){
_start:
{
lean_object* v_res_3942_; 
v_res_3942_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2(v_env_3932_, v_declName_3933_, v___f_3934_, v_addInfo_3935_, v_____r_3936_, v___y_3937_, v___y_3938_, v___y_3939_, v___y_3940_);
lean_dec(v___y_3940_);
lean_dec_ref(v___y_3939_);
lean_dec(v___y_3938_);
lean_dec_ref(v___y_3937_);
return v_res_3942_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1(void){
_start:
{
lean_object* v___x_3944_; lean_object* v___x_3945_; 
v___x_3944_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__0));
v___x_3945_ = l_Lean_stringToMessageData(v___x_3944_);
return v___x_3945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1(lean_object* v_declName_3946_, lean_object* v_env_3947_, lean_object* v_addInfo_3948_, lean_object* v_____r_3949_, lean_object* v___y_3950_, lean_object* v___y_3951_, lean_object* v___y_3952_, lean_object* v___y_3953_){
_start:
{
lean_object* v___x_3955_; 
v___x_3955_ = l_Lean_privateToUserName_x3f(v_declName_3946_);
if (lean_obj_tag(v___x_3955_) == 0)
{
lean_object* v___x_3956_; lean_object* v___x_3957_; 
lean_dec_ref(v_addInfo_3948_);
lean_dec_ref(v_env_3947_);
v___x_3956_ = lean_box(0);
v___x_3957_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3957_, 0, v___x_3956_);
return v___x_3957_;
}
else
{
lean_object* v_val_3958_; lean_object* v___x_3960_; uint8_t v_isShared_3961_; uint8_t v_isSharedCheck_3975_; 
v_val_3958_ = lean_ctor_get(v___x_3955_, 0);
v_isSharedCheck_3975_ = !lean_is_exclusive(v___x_3955_);
if (v_isSharedCheck_3975_ == 0)
{
v___x_3960_ = v___x_3955_;
v_isShared_3961_ = v_isSharedCheck_3975_;
goto v_resetjp_3959_;
}
else
{
lean_inc(v_val_3958_);
lean_dec(v___x_3955_);
v___x_3960_ = lean_box(0);
v_isShared_3961_ = v_isSharedCheck_3975_;
goto v_resetjp_3959_;
}
v_resetjp_3959_:
{
uint8_t v___x_3962_; uint8_t v___x_3963_; 
v___x_3962_ = 1;
lean_inc(v_val_3958_);
v___x_3963_ = l_Lean_Environment_contains(v_env_3947_, v_val_3958_, v___x_3962_);
if (v___x_3963_ == 0)
{
lean_object* v___x_3964_; lean_object* v___x_3966_; 
lean_dec(v_val_3958_);
lean_dec_ref(v_addInfo_3948_);
v___x_3964_ = lean_box(0);
if (v_isShared_3961_ == 0)
{
lean_ctor_set_tag(v___x_3960_, 0);
lean_ctor_set(v___x_3960_, 0, v___x_3964_);
v___x_3966_ = v___x_3960_;
goto v_reusejp_3965_;
}
else
{
lean_object* v_reuseFailAlloc_3967_; 
v_reuseFailAlloc_3967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3967_, 0, v___x_3964_);
v___x_3966_ = v_reuseFailAlloc_3967_;
goto v_reusejp_3965_;
}
v_reusejp_3965_:
{
return v___x_3966_;
}
}
else
{
lean_object* v___x_3968_; 
lean_del_object(v___x_3960_);
lean_inc(v___y_3953_);
lean_inc_ref(v___y_3952_);
lean_inc(v___y_3951_);
lean_inc_ref(v___y_3950_);
lean_inc(v_val_3958_);
v___x_3968_ = lean_apply_6(v_addInfo_3948_, v_val_3958_, v___y_3950_, v___y_3951_, v___y_3952_, v___y_3953_, lean_box(0));
if (lean_obj_tag(v___x_3968_) == 0)
{
lean_object* v___x_3969_; lean_object* v___x_3970_; lean_object* v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; 
lean_dec_ref_known(v___x_3968_, 1);
v___x_3969_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___closed__1);
v___x_3970_ = l_Lean_MessageData_ofConstName(v_val_3958_, v___x_3962_);
v___x_3971_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3971_, 0, v___x_3969_);
lean_ctor_set(v___x_3971_, 1, v___x_3970_);
v___x_3972_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___closed__1);
v___x_3973_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3973_, 0, v___x_3971_);
lean_ctor_set(v___x_3973_, 1, v___x_3972_);
v___x_3974_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_3973_, v___y_3950_, v___y_3951_, v___y_3952_, v___y_3953_);
return v___x_3974_;
}
else
{
lean_dec(v_val_3958_);
return v___x_3968_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___boxed(lean_object* v_declName_3976_, lean_object* v_env_3977_, lean_object* v_addInfo_3978_, lean_object* v_____r_3979_, lean_object* v___y_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_){
_start:
{
lean_object* v_res_3985_; 
v_res_3985_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1(v_declName_3976_, v_env_3977_, v_addInfo_3978_, v_____r_3979_, v___y_3980_, v___y_3981_, v___y_3982_, v___y_3983_);
lean_dec(v___y_3983_);
lean_dec_ref(v___y_3982_);
lean_dec(v___y_3981_);
lean_dec_ref(v___y_3980_);
return v_res_3985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3(lean_object* v_declName_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_){
_start:
{
lean_object* v___x_3995_; lean_object* v_env_3996_; uint8_t v___x_3997_; lean_object* v_addInfo_3998_; lean_object* v_env_3999_; lean_object* v___f_4000_; lean_object* v___f_4001_; lean_object* v___x_4002_; lean_object* v___f_4003_; uint8_t v___x_4004_; uint8_t v___x_4005_; 
v___x_3995_ = lean_st_ref_get(v___y_3993_);
v_env_3996_ = lean_ctor_get(v___x_3995_, 0);
lean_inc_ref(v_env_3996_);
lean_dec(v___x_3995_);
v___x_3997_ = 0;
v_addInfo_3998_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___closed__0));
v_env_3999_ = l_Lean_Environment_setExporting(v_env_3996_, v___x_3997_);
lean_inc_ref_n(v_env_3999_, 4);
lean_inc_n(v_declName_3989_, 4);
v___f_4000_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__1___boxed), 9, 3);
lean_closure_set(v___f_4000_, 0, v_declName_3989_);
lean_closure_set(v___f_4000_, 1, v_env_3999_);
lean_closure_set(v___f_4000_, 2, v_addInfo_3998_);
v___f_4001_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__2___boxed), 10, 4);
lean_closure_set(v___f_4001_, 0, v_env_3999_);
lean_closure_set(v___f_4001_, 1, v_declName_3989_);
lean_closure_set(v___f_4001_, 2, v___f_4000_);
lean_closure_set(v___f_4001_, 3, v_addInfo_3998_);
v___x_4002_ = lean_box(v___x_3997_);
lean_inc_ref(v___f_4001_);
v___f_4003_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__3___boxed), 10, 4);
lean_closure_set(v___f_4003_, 0, v___f_4001_);
lean_closure_set(v___f_4003_, 1, v_declName_3989_);
lean_closure_set(v___f_4003_, 2, v___x_4002_);
lean_closure_set(v___f_4003_, 3, v_env_3999_);
v___x_4004_ = 1;
v___x_4005_ = l_Lean_Environment_contains(v_env_3999_, v_declName_3989_, v___x_4004_);
if (v___x_4005_ == 0)
{
lean_object* v___f_4006_; lean_object* v___x_4007_; 
lean_dec_ref(v___f_4001_);
lean_dec(v_declName_3989_);
v___f_4006_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__4___boxed), 6, 1);
lean_closure_set(v___f_4006_, 0, v___f_4003_);
v___x_4007_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(v_env_3999_, v___f_4006_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_);
return v___x_4007_;
}
else
{
lean_object* v___x_4008_; lean_object* v___x_4009_; lean_object* v___f_4010_; lean_object* v___x_4011_; 
v___x_4008_ = lean_box(v___x_4004_);
v___x_4009_ = lean_box(v___x_3997_);
lean_inc_ref(v_env_3999_);
v___f_4010_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___lam__5___boxed), 12, 7);
lean_closure_set(v___f_4010_, 0, v_addInfo_3998_);
lean_closure_set(v___f_4010_, 1, v_declName_3989_);
lean_closure_set(v___f_4010_, 2, v___x_4008_);
lean_closure_set(v___f_4010_, 3, v___f_4001_);
lean_closure_set(v___f_4010_, 4, v___x_4009_);
lean_closure_set(v___f_4010_, 5, v_env_3999_);
lean_closure_set(v___f_4010_, 6, v___f_4003_);
v___x_4011_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(v_env_3999_, v___f_4010_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_);
return v___x_4011_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3___boxed(lean_object* v_declName_4012_, lean_object* v___y_4013_, lean_object* v___y_4014_, lean_object* v___y_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_){
_start:
{
lean_object* v_res_4018_; 
v_res_4018_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3(v_declName_4012_, v___y_4013_, v___y_4014_, v___y_4015_, v___y_4016_);
lean_dec(v___y_4016_);
lean_dec_ref(v___y_4015_);
lean_dec(v___y_4014_);
lean_dec_ref(v___y_4013_);
return v_res_4018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_modifiers_4019_, lean_object* v_declName_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_, lean_object* v___y_4024_){
_start:
{
lean_object* v___x_4026_; lean_object* v_env_4027_; uint8_t v_visibility_4028_; uint8_t v_isProtected_4029_; lean_object* v_declName_4031_; lean_object* v___y_4032_; lean_object* v___y_4033_; lean_object* v___y_4034_; lean_object* v___y_4035_; uint8_t v___x_4091_; 
v___x_4026_ = lean_st_ref_get(v___y_4024_);
v_env_4027_ = lean_ctor_get(v___x_4026_, 0);
lean_inc_ref(v_env_4027_);
lean_dec(v___x_4026_);
v_visibility_4028_ = lean_ctor_get_uint8(v_modifiers_4019_, sizeof(void*)*3);
v_isProtected_4029_ = lean_ctor_get_uint8(v_modifiers_4019_, sizeof(void*)*3 + 1);
v___x_4091_ = l_Lean_Elab_Visibility_isInferredPublic(v_env_4027_, v_visibility_4028_);
lean_dec_ref(v_env_4027_);
if (v___x_4091_ == 0)
{
lean_object* v___x_4092_; lean_object* v_env_4093_; lean_object* v_declName_4094_; 
v___x_4092_ = lean_st_ref_get(v___y_4024_);
v_env_4093_ = lean_ctor_get(v___x_4092_, 0);
lean_inc_ref(v_env_4093_);
lean_dec(v___x_4092_);
v_declName_4094_ = l_Lean_mkPrivateName(v_env_4093_, v_declName_4020_);
lean_dec_ref(v_env_4093_);
v_declName_4031_ = v_declName_4094_;
v___y_4032_ = v___y_4021_;
v___y_4033_ = v___y_4022_;
v___y_4034_ = v___y_4023_;
v___y_4035_ = v___y_4024_;
goto v___jp_4030_;
}
else
{
v_declName_4031_ = v_declName_4020_;
v___y_4032_ = v___y_4021_;
v___y_4033_ = v___y_4022_;
v___y_4034_ = v___y_4023_;
v___y_4035_ = v___y_4024_;
goto v___jp_4030_;
}
v___jp_4030_:
{
lean_object* v___x_4036_; 
lean_inc(v_declName_4031_);
v___x_4036_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3(v_declName_4031_, v___y_4032_, v___y_4033_, v___y_4034_, v___y_4035_);
if (lean_obj_tag(v___x_4036_) == 0)
{
lean_object* v___x_4038_; uint8_t v_isShared_4039_; uint8_t v_isSharedCheck_4081_; 
v_isSharedCheck_4081_ = !lean_is_exclusive(v___x_4036_);
if (v_isSharedCheck_4081_ == 0)
{
lean_object* v_unused_4082_; 
v_unused_4082_ = lean_ctor_get(v___x_4036_, 0);
lean_dec(v_unused_4082_);
v___x_4038_ = v___x_4036_;
v_isShared_4039_ = v_isSharedCheck_4081_;
goto v_resetjp_4037_;
}
else
{
lean_dec(v___x_4036_);
v___x_4038_ = lean_box(0);
v_isShared_4039_ = v_isSharedCheck_4081_;
goto v_resetjp_4037_;
}
v_resetjp_4037_:
{
if (v_isProtected_4029_ == 0)
{
lean_object* v___x_4041_; 
if (v_isShared_4039_ == 0)
{
lean_ctor_set(v___x_4038_, 0, v_declName_4031_);
v___x_4041_ = v___x_4038_;
goto v_reusejp_4040_;
}
else
{
lean_object* v_reuseFailAlloc_4042_; 
v_reuseFailAlloc_4042_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4042_, 0, v_declName_4031_);
v___x_4041_ = v_reuseFailAlloc_4042_;
goto v_reusejp_4040_;
}
v_reusejp_4040_:
{
return v___x_4041_;
}
}
else
{
lean_object* v___x_4043_; lean_object* v_env_4044_; lean_object* v_nextMacroScope_4045_; lean_object* v_ngen_4046_; lean_object* v_auxDeclNGen_4047_; lean_object* v_traceState_4048_; lean_object* v_messages_4049_; lean_object* v_infoState_4050_; lean_object* v_snapshotTasks_4051_; lean_object* v___x_4053_; uint8_t v_isShared_4054_; uint8_t v_isSharedCheck_4079_; 
v___x_4043_ = lean_st_ref_take(v___y_4035_);
v_env_4044_ = lean_ctor_get(v___x_4043_, 0);
v_nextMacroScope_4045_ = lean_ctor_get(v___x_4043_, 1);
v_ngen_4046_ = lean_ctor_get(v___x_4043_, 2);
v_auxDeclNGen_4047_ = lean_ctor_get(v___x_4043_, 3);
v_traceState_4048_ = lean_ctor_get(v___x_4043_, 4);
v_messages_4049_ = lean_ctor_get(v___x_4043_, 6);
v_infoState_4050_ = lean_ctor_get(v___x_4043_, 7);
v_snapshotTasks_4051_ = lean_ctor_get(v___x_4043_, 8);
v_isSharedCheck_4079_ = !lean_is_exclusive(v___x_4043_);
if (v_isSharedCheck_4079_ == 0)
{
lean_object* v_unused_4080_; 
v_unused_4080_ = lean_ctor_get(v___x_4043_, 5);
lean_dec(v_unused_4080_);
v___x_4053_ = v___x_4043_;
v_isShared_4054_ = v_isSharedCheck_4079_;
goto v_resetjp_4052_;
}
else
{
lean_inc(v_snapshotTasks_4051_);
lean_inc(v_infoState_4050_);
lean_inc(v_messages_4049_);
lean_inc(v_traceState_4048_);
lean_inc(v_auxDeclNGen_4047_);
lean_inc(v_ngen_4046_);
lean_inc(v_nextMacroScope_4045_);
lean_inc(v_env_4044_);
lean_dec(v___x_4043_);
v___x_4053_ = lean_box(0);
v_isShared_4054_ = v_isSharedCheck_4079_;
goto v_resetjp_4052_;
}
v_resetjp_4052_:
{
lean_object* v___x_4055_; lean_object* v___x_4056_; lean_object* v___x_4058_; 
lean_inc(v_declName_4031_);
v___x_4055_ = l_Lean_addProtected(v_env_4044_, v_declName_4031_);
v___x_4056_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__2);
if (v_isShared_4054_ == 0)
{
lean_ctor_set(v___x_4053_, 5, v___x_4056_);
lean_ctor_set(v___x_4053_, 0, v___x_4055_);
v___x_4058_ = v___x_4053_;
goto v_reusejp_4057_;
}
else
{
lean_object* v_reuseFailAlloc_4078_; 
v_reuseFailAlloc_4078_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4078_, 0, v___x_4055_);
lean_ctor_set(v_reuseFailAlloc_4078_, 1, v_nextMacroScope_4045_);
lean_ctor_set(v_reuseFailAlloc_4078_, 2, v_ngen_4046_);
lean_ctor_set(v_reuseFailAlloc_4078_, 3, v_auxDeclNGen_4047_);
lean_ctor_set(v_reuseFailAlloc_4078_, 4, v_traceState_4048_);
lean_ctor_set(v_reuseFailAlloc_4078_, 5, v___x_4056_);
lean_ctor_set(v_reuseFailAlloc_4078_, 6, v_messages_4049_);
lean_ctor_set(v_reuseFailAlloc_4078_, 7, v_infoState_4050_);
lean_ctor_set(v_reuseFailAlloc_4078_, 8, v_snapshotTasks_4051_);
v___x_4058_ = v_reuseFailAlloc_4078_;
goto v_reusejp_4057_;
}
v_reusejp_4057_:
{
lean_object* v___x_4059_; lean_object* v___x_4060_; lean_object* v_mctx_4061_; lean_object* v_zetaDeltaFVarIds_4062_; lean_object* v_postponed_4063_; lean_object* v_diag_4064_; lean_object* v___x_4066_; uint8_t v_isShared_4067_; uint8_t v_isSharedCheck_4076_; 
v___x_4059_ = lean_st_ref_set(v___y_4035_, v___x_4058_);
v___x_4060_ = lean_st_ref_take(v___y_4033_);
v_mctx_4061_ = lean_ctor_get(v___x_4060_, 0);
v_zetaDeltaFVarIds_4062_ = lean_ctor_get(v___x_4060_, 2);
v_postponed_4063_ = lean_ctor_get(v___x_4060_, 3);
v_diag_4064_ = lean_ctor_get(v___x_4060_, 4);
v_isSharedCheck_4076_ = !lean_is_exclusive(v___x_4060_);
if (v_isSharedCheck_4076_ == 0)
{
lean_object* v_unused_4077_; 
v_unused_4077_ = lean_ctor_get(v___x_4060_, 1);
lean_dec(v_unused_4077_);
v___x_4066_ = v___x_4060_;
v_isShared_4067_ = v_isSharedCheck_4076_;
goto v_resetjp_4065_;
}
else
{
lean_inc(v_diag_4064_);
lean_inc(v_postponed_4063_);
lean_inc(v_zetaDeltaFVarIds_4062_);
lean_inc(v_mctx_4061_);
lean_dec(v___x_4060_);
v___x_4066_ = lean_box(0);
v_isShared_4067_ = v_isSharedCheck_4076_;
goto v_resetjp_4065_;
}
v_resetjp_4065_:
{
lean_object* v___x_4068_; lean_object* v___x_4070_; 
v___x_4068_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_MkIff_mkIffOfInductivePropImpl_spec__4_spec__5___redArg___closed__3);
if (v_isShared_4067_ == 0)
{
lean_ctor_set(v___x_4066_, 1, v___x_4068_);
v___x_4070_ = v___x_4066_;
goto v_reusejp_4069_;
}
else
{
lean_object* v_reuseFailAlloc_4075_; 
v_reuseFailAlloc_4075_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4075_, 0, v_mctx_4061_);
lean_ctor_set(v_reuseFailAlloc_4075_, 1, v___x_4068_);
lean_ctor_set(v_reuseFailAlloc_4075_, 2, v_zetaDeltaFVarIds_4062_);
lean_ctor_set(v_reuseFailAlloc_4075_, 3, v_postponed_4063_);
lean_ctor_set(v_reuseFailAlloc_4075_, 4, v_diag_4064_);
v___x_4070_ = v_reuseFailAlloc_4075_;
goto v_reusejp_4069_;
}
v_reusejp_4069_:
{
lean_object* v___x_4071_; lean_object* v___x_4073_; 
v___x_4071_ = lean_st_ref_set(v___y_4033_, v___x_4070_);
if (v_isShared_4039_ == 0)
{
lean_ctor_set(v___x_4038_, 0, v_declName_4031_);
v___x_4073_ = v___x_4038_;
goto v_reusejp_4072_;
}
else
{
lean_object* v_reuseFailAlloc_4074_; 
v_reuseFailAlloc_4074_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4074_, 0, v_declName_4031_);
v___x_4073_ = v_reuseFailAlloc_4074_;
goto v_reusejp_4072_;
}
v_reusejp_4072_:
{
return v___x_4073_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_4083_; lean_object* v___x_4085_; uint8_t v_isShared_4086_; uint8_t v_isSharedCheck_4090_; 
lean_dec(v_declName_4031_);
v_a_4083_ = lean_ctor_get(v___x_4036_, 0);
v_isSharedCheck_4090_ = !lean_is_exclusive(v___x_4036_);
if (v_isSharedCheck_4090_ == 0)
{
v___x_4085_ = v___x_4036_;
v_isShared_4086_ = v_isSharedCheck_4090_;
goto v_resetjp_4084_;
}
else
{
lean_inc(v_a_4083_);
lean_dec(v___x_4036_);
v___x_4085_ = lean_box(0);
v_isShared_4086_ = v_isSharedCheck_4090_;
goto v_resetjp_4084_;
}
v_resetjp_4084_:
{
lean_object* v___x_4088_; 
if (v_isShared_4086_ == 0)
{
v___x_4088_ = v___x_4085_;
goto v_reusejp_4087_;
}
else
{
lean_object* v_reuseFailAlloc_4089_; 
v_reuseFailAlloc_4089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4089_, 0, v_a_4083_);
v___x_4088_ = v_reuseFailAlloc_4089_;
goto v_reusejp_4087_;
}
v_reusejp_4087_:
{
return v___x_4088_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_modifiers_4095_, lean_object* v_declName_4096_, lean_object* v___y_4097_, lean_object* v___y_4098_, lean_object* v___y_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_){
_start:
{
lean_object* v_res_4102_; 
v_res_4102_ = lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1(v_modifiers_4095_, v_declName_4096_, v___y_4097_, v___y_4098_, v___y_4099_, v___y_4100_);
lean_dec(v___y_4100_);
lean_dec_ref(v___y_4099_);
lean_dec(v___y_4098_);
lean_dec_ref(v___y_4097_);
lean_dec_ref(v_modifiers_4095_);
return v_res_4102_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1(void){
_start:
{
lean_object* v___x_4104_; lean_object* v___x_4105_; 
v___x_4104_ = ((lean_object*)(lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__0));
v___x_4105_ = l_Lean_stringToMessageData(v___x_4104_);
return v___x_4105_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5(void){
_start:
{
lean_object* v___x_4110_; lean_object* v___x_4111_; 
v___x_4110_ = ((lean_object*)(lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__4));
v___x_4111_ = l_Lean_stringToMessageData(v___x_4110_);
return v___x_4111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0(lean_object* v_currNamespace_4112_, lean_object* v_modifiers_4113_, lean_object* v_shortName_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_){
_start:
{
lean_object* v___y_4121_; lean_object* v___y_4122_; lean_object* v___y_4126_; lean_object* v_shortName_4127_; lean_object* v_currNamespace_4128_; lean_object* v___y_4129_; lean_object* v___y_4130_; lean_object* v___y_4131_; lean_object* v___y_4132_; lean_object* v_view_4186_; lean_object* v_name_4187_; lean_object* v_imported_4188_; lean_object* v_ctx_4189_; lean_object* v_scopes_4190_; lean_object* v___x_4192_; uint8_t v_isShared_4193_; uint8_t v_isSharedCheck_4244_; 
lean_inc(v_shortName_4114_);
v_view_4186_ = l_Lean_extractMacroScopes(v_shortName_4114_);
v_name_4187_ = lean_ctor_get(v_view_4186_, 0);
v_imported_4188_ = lean_ctor_get(v_view_4186_, 1);
v_ctx_4189_ = lean_ctor_get(v_view_4186_, 2);
v_scopes_4190_ = lean_ctor_get(v_view_4186_, 3);
v_isSharedCheck_4244_ = !lean_is_exclusive(v_view_4186_);
if (v_isSharedCheck_4244_ == 0)
{
v___x_4192_ = v_view_4186_;
v_isShared_4193_ = v_isSharedCheck_4244_;
goto v_resetjp_4191_;
}
else
{
lean_inc(v_scopes_4190_);
lean_inc(v_ctx_4189_);
lean_inc(v_imported_4188_);
lean_inc(v_name_4187_);
lean_dec(v_view_4186_);
v___x_4192_ = lean_box(0);
v_isShared_4193_ = v_isSharedCheck_4244_;
goto v_resetjp_4191_;
}
v___jp_4120_:
{
lean_object* v___x_4123_; lean_object* v___x_4124_; 
v___x_4123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4123_, 0, v___y_4122_);
lean_ctor_set(v___x_4123_, 1, v___y_4121_);
v___x_4124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4124_, 0, v___x_4123_);
return v___x_4124_;
}
v___jp_4125_:
{
lean_object* v___x_4133_; 
lean_inc(v___y_4126_);
v___x_4133_ = lp_mathlib_Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0(v___y_4126_, v___y_4129_, v___y_4130_, v___y_4131_, v___y_4132_);
if (lean_obj_tag(v___x_4133_) == 0)
{
lean_object* v___x_4134_; 
lean_dec_ref_known(v___x_4133_, 1);
v___x_4134_ = lp_mathlib_Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1(v_modifiers_4113_, v___y_4126_, v___y_4129_, v___y_4130_, v___y_4131_, v___y_4132_);
if (lean_obj_tag(v___x_4134_) == 0)
{
uint8_t v_isProtected_4135_; 
v_isProtected_4135_ = lean_ctor_get_uint8(v_modifiers_4113_, sizeof(void*)*3 + 1);
if (v_isProtected_4135_ == 0)
{
lean_object* v_a_4136_; lean_object* v___x_4138_; uint8_t v_isShared_4139_; uint8_t v_isSharedCheck_4144_; 
lean_dec(v_currNamespace_4128_);
v_a_4136_ = lean_ctor_get(v___x_4134_, 0);
v_isSharedCheck_4144_ = !lean_is_exclusive(v___x_4134_);
if (v_isSharedCheck_4144_ == 0)
{
v___x_4138_ = v___x_4134_;
v_isShared_4139_ = v_isSharedCheck_4144_;
goto v_resetjp_4137_;
}
else
{
lean_inc(v_a_4136_);
lean_dec(v___x_4134_);
v___x_4138_ = lean_box(0);
v_isShared_4139_ = v_isSharedCheck_4144_;
goto v_resetjp_4137_;
}
v_resetjp_4137_:
{
lean_object* v___x_4140_; lean_object* v___x_4142_; 
v___x_4140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4140_, 0, v_a_4136_);
lean_ctor_set(v___x_4140_, 1, v_shortName_4127_);
if (v_isShared_4139_ == 0)
{
lean_ctor_set(v___x_4138_, 0, v___x_4140_);
v___x_4142_ = v___x_4138_;
goto v_reusejp_4141_;
}
else
{
lean_object* v_reuseFailAlloc_4143_; 
v_reuseFailAlloc_4143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4143_, 0, v___x_4140_);
v___x_4142_ = v_reuseFailAlloc_4143_;
goto v_reusejp_4141_;
}
v_reusejp_4141_:
{
return v___x_4142_;
}
}
}
else
{
if (lean_obj_tag(v_currNamespace_4128_) == 1)
{
lean_object* v_a_4145_; lean_object* v___x_4147_; uint8_t v_isShared_4148_; uint8_t v_isSharedCheck_4157_; 
v_a_4145_ = lean_ctor_get(v___x_4134_, 0);
v_isSharedCheck_4157_ = !lean_is_exclusive(v___x_4134_);
if (v_isSharedCheck_4157_ == 0)
{
v___x_4147_ = v___x_4134_;
v_isShared_4148_ = v_isSharedCheck_4157_;
goto v_resetjp_4146_;
}
else
{
lean_inc(v_a_4145_);
lean_dec(v___x_4134_);
v___x_4147_ = lean_box(0);
v_isShared_4148_ = v_isSharedCheck_4157_;
goto v_resetjp_4146_;
}
v_resetjp_4146_:
{
lean_object* v_str_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; lean_object* v___x_4152_; lean_object* v___x_4153_; lean_object* v___x_4155_; 
v_str_4149_ = lean_ctor_get(v_currNamespace_4128_, 1);
lean_inc_ref(v_str_4149_);
lean_dec_ref_known(v_currNamespace_4128_, 2);
v___x_4150_ = lean_box(0);
v___x_4151_ = l_Lean_Name_str___override(v___x_4150_, v_str_4149_);
v___x_4152_ = l_Lean_Name_append(v___x_4151_, v_shortName_4127_);
v___x_4153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4153_, 0, v_a_4145_);
lean_ctor_set(v___x_4153_, 1, v___x_4152_);
if (v_isShared_4148_ == 0)
{
lean_ctor_set(v___x_4147_, 0, v___x_4153_);
v___x_4155_ = v___x_4147_;
goto v_reusejp_4154_;
}
else
{
lean_object* v_reuseFailAlloc_4156_; 
v_reuseFailAlloc_4156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4156_, 0, v___x_4153_);
v___x_4155_ = v_reuseFailAlloc_4156_;
goto v_reusejp_4154_;
}
v_reusejp_4154_:
{
return v___x_4155_;
}
}
}
else
{
lean_object* v_a_4158_; uint8_t v___x_4159_; 
lean_dec(v_currNamespace_4128_);
v_a_4158_ = lean_ctor_get(v___x_4134_, 0);
lean_inc(v_a_4158_);
lean_dec_ref_known(v___x_4134_, 1);
v___x_4159_ = l_Lean_Name_isAtomic(v_shortName_4127_);
if (v___x_4159_ == 0)
{
v___y_4121_ = v_shortName_4127_;
v___y_4122_ = v_a_4158_;
goto v___jp_4120_;
}
else
{
lean_object* v___x_4160_; lean_object* v___x_4161_; lean_object* v_a_4162_; lean_object* v___x_4164_; uint8_t v_isShared_4165_; uint8_t v_isSharedCheck_4169_; 
lean_dec(v_a_4158_);
lean_dec(v_shortName_4127_);
v___x_4160_ = lean_obj_once(&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1, &lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__1);
v___x_4161_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4160_, v___y_4129_, v___y_4130_, v___y_4131_, v___y_4132_);
v_a_4162_ = lean_ctor_get(v___x_4161_, 0);
v_isSharedCheck_4169_ = !lean_is_exclusive(v___x_4161_);
if (v_isSharedCheck_4169_ == 0)
{
v___x_4164_ = v___x_4161_;
v_isShared_4165_ = v_isSharedCheck_4169_;
goto v_resetjp_4163_;
}
else
{
lean_inc(v_a_4162_);
lean_dec(v___x_4161_);
v___x_4164_ = lean_box(0);
v_isShared_4165_ = v_isSharedCheck_4169_;
goto v_resetjp_4163_;
}
v_resetjp_4163_:
{
lean_object* v___x_4167_; 
if (v_isShared_4165_ == 0)
{
v___x_4167_ = v___x_4164_;
goto v_reusejp_4166_;
}
else
{
lean_object* v_reuseFailAlloc_4168_; 
v_reuseFailAlloc_4168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4168_, 0, v_a_4162_);
v___x_4167_ = v_reuseFailAlloc_4168_;
goto v_reusejp_4166_;
}
v_reusejp_4166_:
{
return v___x_4167_;
}
}
}
}
}
}
else
{
lean_object* v_a_4170_; lean_object* v___x_4172_; uint8_t v_isShared_4173_; uint8_t v_isSharedCheck_4177_; 
lean_dec(v_currNamespace_4128_);
lean_dec(v_shortName_4127_);
v_a_4170_ = lean_ctor_get(v___x_4134_, 0);
v_isSharedCheck_4177_ = !lean_is_exclusive(v___x_4134_);
if (v_isSharedCheck_4177_ == 0)
{
v___x_4172_ = v___x_4134_;
v_isShared_4173_ = v_isSharedCheck_4177_;
goto v_resetjp_4171_;
}
else
{
lean_inc(v_a_4170_);
lean_dec(v___x_4134_);
v___x_4172_ = lean_box(0);
v_isShared_4173_ = v_isSharedCheck_4177_;
goto v_resetjp_4171_;
}
v_resetjp_4171_:
{
lean_object* v___x_4175_; 
if (v_isShared_4173_ == 0)
{
v___x_4175_ = v___x_4172_;
goto v_reusejp_4174_;
}
else
{
lean_object* v_reuseFailAlloc_4176_; 
v_reuseFailAlloc_4176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4176_, 0, v_a_4170_);
v___x_4175_ = v_reuseFailAlloc_4176_;
goto v_reusejp_4174_;
}
v_reusejp_4174_:
{
return v___x_4175_;
}
}
}
}
else
{
lean_object* v_a_4178_; lean_object* v___x_4180_; uint8_t v_isShared_4181_; uint8_t v_isSharedCheck_4185_; 
lean_dec(v_currNamespace_4128_);
lean_dec(v_shortName_4127_);
lean_dec(v___y_4126_);
v_a_4178_ = lean_ctor_get(v___x_4133_, 0);
v_isSharedCheck_4185_ = !lean_is_exclusive(v___x_4133_);
if (v_isSharedCheck_4185_ == 0)
{
v___x_4180_ = v___x_4133_;
v_isShared_4181_ = v_isSharedCheck_4185_;
goto v_resetjp_4179_;
}
else
{
lean_inc(v_a_4178_);
lean_dec(v___x_4133_);
v___x_4180_ = lean_box(0);
v_isShared_4181_ = v_isSharedCheck_4185_;
goto v_resetjp_4179_;
}
v_resetjp_4179_:
{
lean_object* v___x_4183_; 
if (v_isShared_4181_ == 0)
{
v___x_4183_ = v___x_4180_;
goto v_reusejp_4182_;
}
else
{
lean_object* v_reuseFailAlloc_4184_; 
v_reuseFailAlloc_4184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4184_, 0, v_a_4178_);
v___x_4183_ = v_reuseFailAlloc_4184_;
goto v_reusejp_4182_;
}
v_reusejp_4182_:
{
return v___x_4183_;
}
}
}
}
v_resetjp_4191_:
{
lean_object* v___x_4194_; uint8_t v_isRootName_4195_; lean_object* v___y_4197_; lean_object* v___y_4198_; lean_object* v___y_4199_; lean_object* v___y_4200_; lean_object* v___y_4201_; lean_object* v___y_4222_; lean_object* v___y_4223_; lean_object* v___y_4224_; lean_object* v___y_4225_; uint8_t v___x_4233_; 
v___x_4194_ = ((lean_object*)(lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__3));
v_isRootName_4195_ = l_Lean_Name_isPrefixOf(v___x_4194_, v_name_4187_);
v___x_4233_ = lean_name_eq(v_name_4187_, v___x_4194_);
if (v___x_4233_ == 0)
{
v___y_4222_ = v___y_4115_;
v___y_4223_ = v___y_4116_;
v___y_4224_ = v___y_4117_;
v___y_4225_ = v___y_4118_;
goto v___jp_4221_;
}
else
{
lean_object* v___x_4234_; lean_object* v___x_4235_; lean_object* v_a_4236_; lean_object* v___x_4238_; uint8_t v_isShared_4239_; uint8_t v_isSharedCheck_4243_; 
lean_del_object(v___x_4192_);
lean_dec(v_scopes_4190_);
lean_dec(v_ctx_4189_);
lean_dec(v_imported_4188_);
lean_dec(v_name_4187_);
lean_dec(v_shortName_4114_);
lean_dec(v_currNamespace_4112_);
v___x_4234_ = lean_obj_once(&lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5, &lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___closed__5);
v___x_4235_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4234_, v___y_4115_, v___y_4116_, v___y_4117_, v___y_4118_);
v_a_4236_ = lean_ctor_get(v___x_4235_, 0);
v_isSharedCheck_4243_ = !lean_is_exclusive(v___x_4235_);
if (v_isSharedCheck_4243_ == 0)
{
v___x_4238_ = v___x_4235_;
v_isShared_4239_ = v_isSharedCheck_4243_;
goto v_resetjp_4237_;
}
else
{
lean_inc(v_a_4236_);
lean_dec(v___x_4235_);
v___x_4238_ = lean_box(0);
v_isShared_4239_ = v_isSharedCheck_4243_;
goto v_resetjp_4237_;
}
v_resetjp_4237_:
{
lean_object* v___x_4241_; 
if (v_isShared_4239_ == 0)
{
v___x_4241_ = v___x_4238_;
goto v_reusejp_4240_;
}
else
{
lean_object* v_reuseFailAlloc_4242_; 
v_reuseFailAlloc_4242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4242_, 0, v_a_4236_);
v___x_4241_ = v_reuseFailAlloc_4242_;
goto v_reusejp_4240_;
}
v_reusejp_4240_:
{
return v___x_4241_;
}
}
}
v___jp_4196_:
{
if (v_isRootName_4195_ == 0)
{
lean_dec(v_name_4187_);
v___y_4126_ = v___y_4201_;
v_shortName_4127_ = v_shortName_4114_;
v_currNamespace_4128_ = v_currNamespace_4112_;
v___y_4129_ = v___y_4200_;
v___y_4130_ = v___y_4199_;
v___y_4131_ = v___y_4197_;
v___y_4132_ = v___y_4198_;
goto v___jp_4125_;
}
else
{
lean_dec(v_shortName_4114_);
lean_dec(v_currNamespace_4112_);
if (lean_obj_tag(v_name_4187_) == 1)
{
lean_object* v_pre_4202_; lean_object* v_str_4203_; lean_object* v___x_4204_; lean_object* v_shortName_4205_; lean_object* v_currNamespace_4206_; 
v_pre_4202_ = lean_ctor_get(v_name_4187_, 0);
lean_inc(v_pre_4202_);
v_str_4203_ = lean_ctor_get(v_name_4187_, 1);
lean_inc_ref(v_str_4203_);
lean_dec_ref_known(v_name_4187_, 2);
v___x_4204_ = lean_box(0);
v_shortName_4205_ = l_Lean_Name_str___override(v___x_4204_, v_str_4203_);
v_currNamespace_4206_ = l_Lean_Name_replacePrefix(v_pre_4202_, v___x_4194_, v___x_4204_);
v___y_4126_ = v___y_4201_;
v_shortName_4127_ = v_shortName_4205_;
v_currNamespace_4128_ = v_currNamespace_4206_;
v___y_4129_ = v___y_4200_;
v___y_4130_ = v___y_4199_;
v___y_4131_ = v___y_4197_;
v___y_4132_ = v___y_4198_;
goto v___jp_4125_;
}
else
{
lean_object* v___x_4207_; lean_object* v___x_4208_; lean_object* v___x_4209_; lean_object* v___x_4210_; lean_object* v___x_4211_; lean_object* v___x_4212_; lean_object* v_a_4213_; lean_object* v___x_4215_; uint8_t v_isShared_4216_; uint8_t v_isSharedCheck_4220_; 
lean_dec(v___y_4201_);
v___x_4207_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_checkIfShadowingStructureField___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1);
v___x_4208_ = l_Lean_MessageData_ofName(v_name_4187_);
v___x_4209_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4209_, 0, v___x_4207_);
lean_ctor_set(v___x_4209_, 1, v___x_4208_);
v___x_4210_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3___redArg___closed__3);
v___x_4211_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4211_, 0, v___x_4209_);
lean_ctor_set(v___x_4211_, 1, v___x_4210_);
v___x_4212_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4211_, v___y_4200_, v___y_4199_, v___y_4197_, v___y_4198_);
v_a_4213_ = lean_ctor_get(v___x_4212_, 0);
v_isSharedCheck_4220_ = !lean_is_exclusive(v___x_4212_);
if (v_isSharedCheck_4220_ == 0)
{
v___x_4215_ = v___x_4212_;
v_isShared_4216_ = v_isSharedCheck_4220_;
goto v_resetjp_4214_;
}
else
{
lean_inc(v_a_4213_);
lean_dec(v___x_4212_);
v___x_4215_ = lean_box(0);
v_isShared_4216_ = v_isSharedCheck_4220_;
goto v_resetjp_4214_;
}
v_resetjp_4214_:
{
lean_object* v___x_4218_; 
if (v_isShared_4216_ == 0)
{
v___x_4218_ = v___x_4215_;
goto v_reusejp_4217_;
}
else
{
lean_object* v_reuseFailAlloc_4219_; 
v_reuseFailAlloc_4219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4219_, 0, v_a_4213_);
v___x_4218_ = v_reuseFailAlloc_4219_;
goto v_reusejp_4217_;
}
v_reusejp_4217_:
{
return v___x_4218_;
}
}
}
}
}
v___jp_4221_:
{
if (v_isRootName_4195_ == 0)
{
lean_object* v___x_4226_; 
lean_del_object(v___x_4192_);
lean_dec(v_scopes_4190_);
lean_dec(v_ctx_4189_);
lean_dec(v_imported_4188_);
lean_inc(v_shortName_4114_);
lean_inc(v_currNamespace_4112_);
v___x_4226_ = l_Lean_Name_append(v_currNamespace_4112_, v_shortName_4114_);
v___y_4197_ = v___y_4224_;
v___y_4198_ = v___y_4225_;
v___y_4199_ = v___y_4223_;
v___y_4200_ = v___y_4222_;
v___y_4201_ = v___x_4226_;
goto v___jp_4196_;
}
else
{
lean_object* v___x_4227_; lean_object* v___x_4228_; lean_object* v___x_4230_; 
v___x_4227_ = lean_box(0);
lean_inc(v_name_4187_);
v___x_4228_ = l_Lean_Name_replacePrefix(v_name_4187_, v___x_4194_, v___x_4227_);
if (v_isShared_4193_ == 0)
{
lean_ctor_set(v___x_4192_, 0, v___x_4228_);
v___x_4230_ = v___x_4192_;
goto v_reusejp_4229_;
}
else
{
lean_object* v_reuseFailAlloc_4232_; 
v_reuseFailAlloc_4232_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4232_, 0, v___x_4228_);
lean_ctor_set(v_reuseFailAlloc_4232_, 1, v_imported_4188_);
lean_ctor_set(v_reuseFailAlloc_4232_, 2, v_ctx_4189_);
lean_ctor_set(v_reuseFailAlloc_4232_, 3, v_scopes_4190_);
v___x_4230_ = v_reuseFailAlloc_4232_;
goto v_reusejp_4229_;
}
v_reusejp_4229_:
{
lean_object* v___x_4231_; 
v___x_4231_ = l_Lean_MacroScopesView_review(v___x_4230_);
v___y_4197_ = v___y_4224_;
v___y_4198_ = v___y_4225_;
v___y_4199_ = v___y_4223_;
v___y_4200_ = v___y_4222_;
v___y_4201_ = v___x_4231_;
goto v___jp_4196_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0___boxed(lean_object* v_currNamespace_4245_, lean_object* v_modifiers_4246_, lean_object* v_shortName_4247_, lean_object* v___y_4248_, lean_object* v___y_4249_, lean_object* v___y_4250_, lean_object* v___y_4251_, lean_object* v___y_4252_){
_start:
{
lean_object* v_res_4253_; 
v_res_4253_ = lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0(v_currNamespace_4245_, v_modifiers_4246_, v_shortName_4247_, v___y_4248_, v___y_4249_, v___y_4250_, v___y_4251_);
lean_dec(v___y_4251_);
lean_dec_ref(v___y_4250_);
lean_dec(v___y_4249_);
lean_dec_ref(v___y_4248_);
lean_dec_ref(v_modifiers_4246_);
return v_res_4253_;
}
}
static uint64_t _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4260_; uint64_t v___x_4261_; 
v___x_4260_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4261_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_4260_);
return v___x_4261_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_4262_; lean_object* v___x_4263_; lean_object* v___x_4264_; 
v___x_4262_ = lean_uint64_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4263_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4264_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_4264_, 0, v___x_4263_);
lean_ctor_set_uint64(v___x_4264_, sizeof(void*)*1, v___x_4262_);
return v___x_4264_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4266_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4267_ = l_Lean_stringToMessageData(v___x_4266_);
return v___x_4267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object* v___x_4269_, lean_object* v___x_4270_, lean_object* v___x_4271_, lean_object* v___x_4272_, lean_object* v___x_4273_, lean_object* v___x_4274_, lean_object* v_decl_4275_, lean_object* v_stx_4276_, uint8_t v_x_4277_, lean_object* v___y_4278_, lean_object* v___y_4279_){
_start:
{
uint8_t v___x_4281_; uint8_t v___x_4282_; lean_object* v___x_4283_; lean_object* v___x_4284_; lean_object* v___x_4285_; lean_object* v___x_4286_; lean_object* v___x_4287_; size_t v___x_4288_; lean_object* v___x_4289_; lean_object* v___x_4290_; lean_object* v___x_4291_; lean_object* v___x_4292_; lean_object* v___x_4293_; lean_object* v___x_4294_; lean_object* v___x_4295_; lean_object* v___x_4296_; lean_object* v___x_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; lean_object* v___y_4301_; lean_object* v___x_4311_; uint8_t v___x_4312_; 
v___x_4281_ = 1;
v___x_4282_ = 0;
v___x_4283_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4284_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__1);
v___x_4285_ = lean_unsigned_to_nat(32u);
v___x_4286_ = lean_mk_empty_array_with_capacity(v___x_4285_);
v___x_4287_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__3);
v___x_4288_ = ((size_t)5ULL);
lean_inc_n(v___x_4269_, 7);
v___x_4289_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_4289_, 0, v___x_4287_);
lean_ctor_set(v___x_4289_, 1, v___x_4286_);
lean_ctor_set(v___x_4289_, 2, v___x_4269_);
lean_ctor_set(v___x_4289_, 3, v___x_4269_);
lean_ctor_set_usize(v___x_4289_, 4, v___x_4288_);
v___x_4290_ = lean_box(1);
lean_inc_ref(v___x_4289_);
v___x_4291_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4291_, 0, v___x_4284_);
lean_ctor_set(v___x_4291_, 1, v___x_4289_);
lean_ctor_set(v___x_4291_, 2, v___x_4290_);
v___x_4292_ = lean_mk_empty_array_with_capacity(v___x_4269_);
v___x_4293_ = lean_box(0);
lean_inc_ref(v___x_4292_);
lean_inc(v___x_4270_);
v___x_4294_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_4294_, 0, v___x_4283_);
lean_ctor_set(v___x_4294_, 1, v___x_4270_);
lean_ctor_set(v___x_4294_, 2, v___x_4291_);
lean_ctor_set(v___x_4294_, 3, v___x_4292_);
lean_ctor_set(v___x_4294_, 4, v___x_4293_);
lean_ctor_set(v___x_4294_, 5, v___x_4269_);
lean_ctor_set(v___x_4294_, 6, v___x_4293_);
lean_ctor_set_uint8(v___x_4294_, sizeof(void*)*7, v___x_4282_);
lean_ctor_set_uint8(v___x_4294_, sizeof(void*)*7 + 1, v___x_4282_);
lean_ctor_set_uint8(v___x_4294_, sizeof(void*)*7 + 2, v___x_4282_);
lean_ctor_set_uint8(v___x_4294_, sizeof(void*)*7 + 3, v___x_4281_);
v___x_4295_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_4295_, 0, v___x_4269_);
lean_ctor_set(v___x_4295_, 1, v___x_4269_);
lean_ctor_set(v___x_4295_, 2, v___x_4269_);
lean_ctor_set(v___x_4295_, 3, v___x_4269_);
lean_ctor_set(v___x_4295_, 4, v___x_4284_);
lean_ctor_set(v___x_4295_, 5, v___x_4284_);
lean_ctor_set(v___x_4295_, 6, v___x_4284_);
lean_ctor_set(v___x_4295_, 7, v___x_4284_);
lean_ctor_set(v___x_4295_, 8, v___x_4284_);
lean_ctor_set(v___x_4295_, 9, v___x_4284_);
v___x_4296_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__5);
v___x_4297_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6, &lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_MkIff___aux__Mathlib__Tactic__MkIffOfInductiveProp______elabRules__Mathlib__Tactic__MkIff__mkIffOfInductiveProp__1___closed__6);
v___x_4298_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4298_, 0, v___x_4295_);
lean_ctor_set(v___x_4298_, 1, v___x_4296_);
lean_ctor_set(v___x_4298_, 2, v___x_4270_);
lean_ctor_set(v___x_4298_, 3, v___x_4289_);
lean_ctor_set(v___x_4298_, 4, v___x_4297_);
v___x_4299_ = lean_st_mk_ref(v___x_4298_);
v___x_4311_ = l_Lean_Name_mkStr4(v___x_4271_, v___x_4272_, v___x_4273_, v___x_4274_);
lean_inc(v_stx_4276_);
v___x_4312_ = l_Lean_Syntax_isOfKind(v_stx_4276_, v___x_4311_);
lean_dec(v___x_4311_);
if (v___x_4312_ == 0)
{
lean_object* v___x_4313_; lean_object* v___x_4314_; lean_object* v_a_4315_; lean_object* v___x_4317_; uint8_t v_isShared_4318_; uint8_t v_isSharedCheck_4322_; 
lean_dec_ref(v___x_4292_);
lean_dec(v_stx_4276_);
lean_dec(v_decl_4275_);
lean_dec(v___x_4269_);
v___x_4313_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4314_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4313_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec(v___x_4299_);
lean_dec_ref_known(v___x_4294_, 7);
v_a_4315_ = lean_ctor_get(v___x_4314_, 0);
v_isSharedCheck_4322_ = !lean_is_exclusive(v___x_4314_);
if (v_isSharedCheck_4322_ == 0)
{
v___x_4317_ = v___x_4314_;
v_isShared_4318_ = v_isSharedCheck_4322_;
goto v_resetjp_4316_;
}
else
{
lean_inc(v_a_4315_);
lean_dec(v___x_4314_);
v___x_4317_ = lean_box(0);
v_isShared_4318_ = v_isSharedCheck_4322_;
goto v_resetjp_4316_;
}
v_resetjp_4316_:
{
lean_object* v___x_4320_; 
if (v_isShared_4318_ == 0)
{
v___x_4320_ = v___x_4317_;
goto v_reusejp_4319_;
}
else
{
lean_object* v_reuseFailAlloc_4321_; 
v_reuseFailAlloc_4321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4321_, 0, v_a_4315_);
v___x_4320_ = v_reuseFailAlloc_4321_;
goto v_reusejp_4319_;
}
v_reusejp_4319_:
{
return v___x_4320_;
}
}
}
else
{
lean_object* v___x_4323_; lean_object* v___x_4324_; uint8_t v___x_4325_; 
v___x_4323_ = lean_unsigned_to_nat(1u);
v___x_4324_ = l_Lean_Syntax_getArg(v_stx_4276_, v___x_4323_);
lean_inc(v___x_4324_);
v___x_4325_ = l_Lean_Syntax_matchesNull(v___x_4324_, v___x_4323_);
if (v___x_4325_ == 0)
{
uint8_t v___x_4326_; 
lean_dec_ref(v___x_4292_);
v___x_4326_ = l_Lean_Syntax_matchesNull(v___x_4324_, v___x_4269_);
lean_dec(v___x_4269_);
if (v___x_4326_ == 0)
{
lean_object* v___x_4327_; lean_object* v___x_4328_; lean_object* v_a_4329_; lean_object* v___x_4331_; uint8_t v_isShared_4332_; uint8_t v_isSharedCheck_4336_; 
lean_dec(v_stx_4276_);
lean_dec(v_decl_4275_);
v___x_4327_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4328_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4327_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec(v___x_4299_);
lean_dec_ref_known(v___x_4294_, 7);
v_a_4329_ = lean_ctor_get(v___x_4328_, 0);
v_isSharedCheck_4336_ = !lean_is_exclusive(v___x_4328_);
if (v_isSharedCheck_4336_ == 0)
{
v___x_4331_ = v___x_4328_;
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
else
{
lean_inc(v_a_4329_);
lean_dec(v___x_4328_);
v___x_4331_ = lean_box(0);
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
v_resetjp_4330_:
{
lean_object* v___x_4334_; 
if (v_isShared_4332_ == 0)
{
v___x_4334_ = v___x_4331_;
goto v_reusejp_4333_;
}
else
{
lean_object* v_reuseFailAlloc_4335_; 
v_reuseFailAlloc_4335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4335_, 0, v_a_4329_);
v___x_4334_ = v_reuseFailAlloc_4335_;
goto v_reusejp_4333_;
}
v_reusejp_4333_:
{
return v___x_4334_;
}
}
}
else
{
lean_object* v___x_4337_; lean_object* v___x_4338_; lean_object* v___x_4339_; lean_object* v___x_4340_; lean_object* v___x_4341_; 
lean_inc(v_decl_4275_);
v___x_4337_ = lp_mathlib_Lean_Name_decapitalize(v_decl_4275_);
v___x_4338_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__5_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4339_ = lean_name_append_after(v___x_4337_, v___x_4338_);
v___x_4340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4340_, 0, v___x_4339_);
lean_ctor_set(v___x_4340_, 1, v_stx_4276_);
v___x_4341_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(v_decl_4275_, v___x_4340_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec_ref_known(v___x_4294_, 7);
v___y_4301_ = v___x_4341_;
goto v___jp_4300_;
}
}
else
{
lean_object* v_tgt_4342_; lean_object* v___x_4343_; uint8_t v___x_4344_; 
lean_dec(v_stx_4276_);
v_tgt_4342_ = l_Lean_Syntax_getArg(v___x_4324_, v___x_4269_);
lean_dec(v___x_4269_);
lean_dec(v___x_4324_);
v___x_4343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_MkIff_mkIff___closed__14));
lean_inc(v_tgt_4342_);
v___x_4344_ = l_Lean_Syntax_isOfKind(v_tgt_4342_, v___x_4343_);
if (v___x_4344_ == 0)
{
lean_object* v___x_4345_; lean_object* v___x_4346_; lean_object* v_a_4347_; lean_object* v___x_4349_; uint8_t v_isShared_4350_; uint8_t v_isSharedCheck_4354_; 
lean_dec(v_tgt_4342_);
lean_dec_ref(v___x_4292_);
lean_dec(v_decl_4275_);
v___x_4345_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1___closed__4_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4346_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_select_spec__0___redArg(v___x_4345_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec(v___x_4299_);
lean_dec_ref_known(v___x_4294_, 7);
v_a_4347_ = lean_ctor_get(v___x_4346_, 0);
v_isSharedCheck_4354_ = !lean_is_exclusive(v___x_4346_);
if (v_isSharedCheck_4354_ == 0)
{
v___x_4349_ = v___x_4346_;
v_isShared_4350_ = v_isSharedCheck_4354_;
goto v_resetjp_4348_;
}
else
{
lean_inc(v_a_4347_);
lean_dec(v___x_4346_);
v___x_4349_ = lean_box(0);
v_isShared_4350_ = v_isSharedCheck_4354_;
goto v_resetjp_4348_;
}
v_resetjp_4348_:
{
lean_object* v___x_4352_; 
if (v_isShared_4350_ == 0)
{
v___x_4352_ = v___x_4349_;
goto v_reusejp_4351_;
}
else
{
lean_object* v_reuseFailAlloc_4353_; 
v_reuseFailAlloc_4353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4353_, 0, v_a_4347_);
v___x_4352_ = v_reuseFailAlloc_4353_;
goto v_reusejp_4351_;
}
v_reusejp_4351_:
{
return v___x_4352_;
}
}
}
else
{
lean_object* v_currNamespace_4355_; lean_object* v___x_4356_; uint8_t v___x_4357_; uint8_t v___x_4358_; uint8_t v___x_4359_; lean_object* v___x_4360_; lean_object* v___x_4361_; lean_object* v___x_4362_; 
v_currNamespace_4355_ = lean_ctor_get(v___y_4278_, 6);
v___x_4356_ = lean_box(0);
v___x_4357_ = 0;
v___x_4358_ = 0;
v___x_4359_ = 2;
v___x_4360_ = lean_alloc_ctor(0, 3, 5);
lean_ctor_set(v___x_4360_, 0, v___x_4356_);
lean_ctor_set(v___x_4360_, 1, v___x_4293_);
lean_ctor_set(v___x_4360_, 2, v___x_4292_);
lean_ctor_set_uint8(v___x_4360_, sizeof(void*)*3, v___x_4357_);
lean_ctor_set_uint8(v___x_4360_, sizeof(void*)*3 + 1, v___x_4282_);
lean_ctor_set_uint8(v___x_4360_, sizeof(void*)*3 + 2, v___x_4358_);
lean_ctor_set_uint8(v___x_4360_, sizeof(void*)*3 + 3, v___x_4359_);
lean_ctor_set_uint8(v___x_4360_, sizeof(void*)*3 + 4, v___x_4282_);
v___x_4361_ = l_Lean_TSyntax_getId(v_tgt_4342_);
lean_inc(v_currNamespace_4355_);
v___x_4362_ = lp_mathlib_Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0(v_currNamespace_4355_, v___x_4360_, v___x_4361_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec_ref_known(v___x_4360_, 3);
if (lean_obj_tag(v___x_4362_) == 0)
{
lean_object* v_a_4363_; lean_object* v_fst_4364_; lean_object* v___x_4366_; uint8_t v_isShared_4367_; uint8_t v_isSharedCheck_4372_; 
v_a_4363_ = lean_ctor_get(v___x_4362_, 0);
lean_inc(v_a_4363_);
lean_dec_ref_known(v___x_4362_, 1);
v_fst_4364_ = lean_ctor_get(v_a_4363_, 0);
v_isSharedCheck_4372_ = !lean_is_exclusive(v_a_4363_);
if (v_isSharedCheck_4372_ == 0)
{
lean_object* v_unused_4373_; 
v_unused_4373_ = lean_ctor_get(v_a_4363_, 1);
lean_dec(v_unused_4373_);
v___x_4366_ = v_a_4363_;
v_isShared_4367_ = v_isSharedCheck_4372_;
goto v_resetjp_4365_;
}
else
{
lean_inc(v_fst_4364_);
lean_dec(v_a_4363_);
v___x_4366_ = lean_box(0);
v_isShared_4367_ = v_isSharedCheck_4372_;
goto v_resetjp_4365_;
}
v_resetjp_4365_:
{
lean_object* v___x_4369_; 
if (v_isShared_4367_ == 0)
{
lean_ctor_set(v___x_4366_, 1, v_tgt_4342_);
v___x_4369_ = v___x_4366_;
goto v_reusejp_4368_;
}
else
{
lean_object* v_reuseFailAlloc_4371_; 
v_reuseFailAlloc_4371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4371_, 0, v_fst_4364_);
lean_ctor_set(v_reuseFailAlloc_4371_, 1, v_tgt_4342_);
v___x_4369_ = v_reuseFailAlloc_4371_;
goto v_reusejp_4368_;
}
v_reusejp_4368_:
{
lean_object* v___x_4370_; 
v___x_4370_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(v_decl_4275_, v___x_4369_, v___x_4294_, v___x_4299_, v___y_4278_, v___y_4279_);
lean_dec_ref_known(v___x_4294_, 7);
v___y_4301_ = v___x_4370_;
goto v___jp_4300_;
}
}
}
else
{
lean_object* v_a_4374_; lean_object* v___x_4376_; uint8_t v_isShared_4377_; uint8_t v_isSharedCheck_4381_; 
lean_dec(v_tgt_4342_);
lean_dec(v___x_4299_);
lean_dec_ref_known(v___x_4294_, 7);
lean_dec(v_decl_4275_);
v_a_4374_ = lean_ctor_get(v___x_4362_, 0);
v_isSharedCheck_4381_ = !lean_is_exclusive(v___x_4362_);
if (v_isSharedCheck_4381_ == 0)
{
v___x_4376_ = v___x_4362_;
v_isShared_4377_ = v_isSharedCheck_4381_;
goto v_resetjp_4375_;
}
else
{
lean_inc(v_a_4374_);
lean_dec(v___x_4362_);
v___x_4376_ = lean_box(0);
v_isShared_4377_ = v_isSharedCheck_4381_;
goto v_resetjp_4375_;
}
v_resetjp_4375_:
{
lean_object* v___x_4379_; 
if (v_isShared_4377_ == 0)
{
v___x_4379_ = v___x_4376_;
goto v_reusejp_4378_;
}
else
{
lean_object* v_reuseFailAlloc_4380_; 
v_reuseFailAlloc_4380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4380_, 0, v_a_4374_);
v___x_4379_ = v_reuseFailAlloc_4380_;
goto v_reusejp_4378_;
}
v_reusejp_4378_:
{
return v___x_4379_;
}
}
}
}
}
}
v___jp_4300_:
{
if (lean_obj_tag(v___y_4301_) == 0)
{
lean_object* v_a_4302_; lean_object* v___x_4304_; uint8_t v_isShared_4305_; uint8_t v_isSharedCheck_4310_; 
v_a_4302_ = lean_ctor_get(v___y_4301_, 0);
v_isSharedCheck_4310_ = !lean_is_exclusive(v___y_4301_);
if (v_isSharedCheck_4310_ == 0)
{
v___x_4304_ = v___y_4301_;
v_isShared_4305_ = v_isSharedCheck_4310_;
goto v_resetjp_4303_;
}
else
{
lean_inc(v_a_4302_);
lean_dec(v___y_4301_);
v___x_4304_ = lean_box(0);
v_isShared_4305_ = v_isSharedCheck_4310_;
goto v_resetjp_4303_;
}
v_resetjp_4303_:
{
lean_object* v___x_4306_; lean_object* v___x_4308_; 
v___x_4306_ = lean_st_ref_get(v___x_4299_);
lean_dec(v___x_4299_);
lean_dec(v___x_4306_);
if (v_isShared_4305_ == 0)
{
v___x_4308_ = v___x_4304_;
goto v_reusejp_4307_;
}
else
{
lean_object* v_reuseFailAlloc_4309_; 
v_reuseFailAlloc_4309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4309_, 0, v_a_4302_);
v___x_4308_ = v_reuseFailAlloc_4309_;
goto v_reusejp_4307_;
}
v_reusejp_4307_:
{
return v___x_4308_;
}
}
}
else
{
lean_dec(v___x_4299_);
return v___y_4301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object* v___x_4382_, lean_object* v___x_4383_, lean_object* v___x_4384_, lean_object* v___x_4385_, lean_object* v___x_4386_, lean_object* v___x_4387_, lean_object* v_decl_4388_, lean_object* v_stx_4389_, lean_object* v_x_4390_, lean_object* v___y_4391_, lean_object* v___y_4392_, lean_object* v___y_4393_){
_start:
{
uint8_t v_x_11422__boxed_4394_; lean_object* v_res_4395_; 
v_x_11422__boxed_4394_ = lean_unbox(v_x_4390_);
v_res_4395_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(v___x_4382_, v___x_4383_, v___x_4384_, v___x_4385_, v___x_4386_, v___x_4387_, v_decl_4388_, v_stx_4389_, v_x_11422__boxed_4394_, v___y_4391_, v___y_4392_);
lean_dec(v___y_4392_);
lean_dec_ref(v___y_4391_);
return v_res_4395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3(lean_object* v_msgData_4396_, lean_object* v___y_4397_, lean_object* v___y_4398_){
_start:
{
lean_object* v___x_4400_; lean_object* v_env_4401_; lean_object* v_options_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; lean_object* v___x_4405_; lean_object* v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4409_; 
v___x_4400_ = lean_st_ref_get(v___y_4398_);
v_env_4401_ = lean_ctor_get(v___x_4400_, 0);
lean_inc_ref(v_env_4401_);
lean_dec(v___x_4400_);
v_options_4402_ = lean_ctor_get(v___y_4397_, 2);
v___x_4403_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__2);
v___x_4404_ = lean_unsigned_to_nat(32u);
v___x_4405_ = lean_mk_empty_array_with_capacity(v___x_4404_);
lean_dec_ref(v___x_4405_);
v___x_4406_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_MkIff_constrToProp_spec__0_spec__0_spec__3_spec__7_spec__8_spec__9___redArg___closed__5);
lean_inc_ref(v_options_4402_);
v___x_4407_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4407_, 0, v_env_4401_);
lean_ctor_set(v___x_4407_, 1, v___x_4403_);
lean_ctor_set(v___x_4407_, 2, v___x_4406_);
lean_ctor_set(v___x_4407_, 3, v_options_4402_);
v___x_4408_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_4408_, 0, v___x_4407_);
lean_ctor_set(v___x_4408_, 1, v_msgData_4396_);
v___x_4409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4409_, 0, v___x_4408_);
return v___x_4409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3___boxed(lean_object* v_msgData_4410_, lean_object* v___y_4411_, lean_object* v___y_4412_, lean_object* v___y_4413_){
_start:
{
lean_object* v_res_4414_; 
v_res_4414_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3(v_msgData_4410_, v___y_4411_, v___y_4412_);
lean_dec(v___y_4412_);
lean_dec_ref(v___y_4411_);
return v_res_4414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_4415_, lean_object* v___y_4416_, lean_object* v___y_4417_){
_start:
{
lean_object* v_ref_4419_; lean_object* v___x_4420_; lean_object* v_a_4421_; lean_object* v___x_4423_; uint8_t v_isShared_4424_; uint8_t v_isSharedCheck_4429_; 
v_ref_4419_ = lean_ctor_get(v___y_4416_, 5);
v___x_4420_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1_spec__3(v_msg_4415_, v___y_4416_, v___y_4417_);
v_a_4421_ = lean_ctor_get(v___x_4420_, 0);
v_isSharedCheck_4429_ = !lean_is_exclusive(v___x_4420_);
if (v_isSharedCheck_4429_ == 0)
{
v___x_4423_ = v___x_4420_;
v_isShared_4424_ = v_isSharedCheck_4429_;
goto v_resetjp_4422_;
}
else
{
lean_inc(v_a_4421_);
lean_dec(v___x_4420_);
v___x_4423_ = lean_box(0);
v_isShared_4424_ = v_isSharedCheck_4429_;
goto v_resetjp_4422_;
}
v_resetjp_4422_:
{
lean_object* v___x_4425_; lean_object* v___x_4427_; 
lean_inc(v_ref_4419_);
v___x_4425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4425_, 0, v_ref_4419_);
lean_ctor_set(v___x_4425_, 1, v_a_4421_);
if (v_isShared_4424_ == 0)
{
lean_ctor_set_tag(v___x_4423_, 1);
lean_ctor_set(v___x_4423_, 0, v___x_4425_);
v___x_4427_ = v___x_4423_;
goto v_reusejp_4426_;
}
else
{
lean_object* v_reuseFailAlloc_4428_; 
v_reuseFailAlloc_4428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4428_, 0, v___x_4425_);
v___x_4427_ = v_reuseFailAlloc_4428_;
goto v_reusejp_4426_;
}
v_reusejp_4426_:
{
return v___x_4427_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_){
_start:
{
lean_object* v_res_4434_; 
v_res_4434_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg(v_msg_4430_, v___y_4431_, v___y_4432_);
lean_dec(v___y_4432_);
lean_dec_ref(v___y_4431_);
return v_res_4434_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4436_; lean_object* v___x_4437_; 
v___x_4436_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4437_ = l_Lean_stringToMessageData(v___x_4436_);
return v___x_4437_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4439_; lean_object* v___x_4440_; 
v___x_4439_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4440_ = l_Lean_stringToMessageData(v___x_4439_);
return v___x_4440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(lean_object* v___x_4441_, lean_object* v_decl_4442_, lean_object* v___y_4443_, lean_object* v___y_4444_){
_start:
{
lean_object* v___x_4446_; lean_object* v___x_4447_; lean_object* v___x_4448_; lean_object* v___x_4449_; lean_object* v___x_4450_; lean_object* v___x_4451_; 
v___x_4446_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4447_ = l_Lean_MessageData_ofName(v___x_4441_);
v___x_4448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4448_, 0, v___x_4446_);
lean_ctor_set(v___x_4448_, 1, v___x_4447_);
v___x_4449_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_);
v___x_4450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4450_, 0, v___x_4448_);
lean_ctor_set(v___x_4450_, 1, v___x_4449_);
v___x_4451_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg(v___x_4450_, v___y_4443_, v___y_4444_);
return v___x_4451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object* v___x_4452_, lean_object* v_decl_4453_, lean_object* v___y_4454_, lean_object* v___y_4455_, lean_object* v___y_4456_){
_start:
{
lean_object* v_res_4457_; 
v_res_4457_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___lam__2_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(v___x_4452_, v_decl_4453_, v___y_4454_, v___y_4455_);
lean_dec(v___y_4455_);
lean_dec_ref(v___y_4454_);
lean_dec(v_decl_4453_);
return v_res_4457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_4537_; lean_object* v___x_4538_; 
v___x_4537_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn___closed__28_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_));
v___x_4538_ = l_Lean_registerBuiltinAttribute(v___x_4537_);
return v___x_4538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2____boxed(lean_object* v_a_4539_){
_start:
{
lean_object* v_res_4540_; 
v_res_4540_ = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_();
return v_res_4540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_4541_, lean_object* v_msg_4542_, lean_object* v___y_4543_, lean_object* v___y_4544_){
_start:
{
lean_object* v___x_4546_; 
v___x_4546_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___redArg(v_msg_4542_, v___y_4543_, v___y_4544_);
return v___x_4546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_4547_, lean_object* v_msg_4548_, lean_object* v___y_4549_, lean_object* v___y_4550_, lean_object* v___y_4551_){
_start:
{
lean_object* v_res_4552_; 
v_res_4552_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__1(v_00_u03b1_4547_, v_msg_4548_, v___y_4549_, v___y_4550_);
lean_dec(v___y_4550_);
lean_dec_ref(v___y_4549_);
return v_res_4552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7(lean_object* v_t_4553_, lean_object* v___y_4554_, lean_object* v___y_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_){
_start:
{
lean_object* v___x_4559_; 
v___x_4559_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___redArg(v_t_4553_, v___y_4557_);
return v___x_4559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7___boxed(lean_object* v_t_4560_, lean_object* v___y_4561_, lean_object* v___y_4562_, lean_object* v___y_4563_, lean_object* v___y_4564_, lean_object* v___y_4565_){
_start:
{
lean_object* v_res_4566_; 
v_res_4566_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__6_spec__7(v_t_4560_, v___y_4561_, v___y_4562_, v___y_4563_, v___y_4564_);
lean_dec(v___y_4564_);
lean_dec_ref(v___y_4563_);
lean_dec(v___y_4562_);
lean_dec_ref(v___y_4561_);
return v_res_4566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9(lean_object* v_env_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_, lean_object* v___y_4571_){
_start:
{
lean_object* v___x_4573_; 
v___x_4573_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___redArg(v_env_4567_, v___y_4569_, v___y_4571_);
return v___x_4573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9___boxed(lean_object* v_env_4574_, lean_object* v___y_4575_, lean_object* v___y_4576_, lean_object* v___y_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_){
_start:
{
lean_object* v_res_4580_; 
v_res_4580_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7_spec__9(v_env_4574_, v___y_4575_, v___y_4576_, v___y_4577_, v___y_4578_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___y_4576_);
lean_dec_ref(v___y_4575_);
return v_res_4580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7(lean_object* v_00_u03b1_4581_, lean_object* v_env_4582_, lean_object* v_x_4583_, lean_object* v___y_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_, lean_object* v___y_4587_){
_start:
{
lean_object* v___x_4589_; 
v___x_4589_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___redArg(v_env_4582_, v_x_4583_, v___y_4584_, v___y_4585_, v___y_4586_, v___y_4587_);
return v___x_4589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7___boxed(lean_object* v_00_u03b1_4590_, lean_object* v_env_4591_, lean_object* v_x_4592_, lean_object* v___y_4593_, lean_object* v___y_4594_, lean_object* v___y_4595_, lean_object* v___y_4596_, lean_object* v___y_4597_){
_start:
{
lean_object* v_res_4598_; 
v_res_4598_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Lean_Elab_applyVisibility___at___00Lean_Elab_mkDeclName___at___00__private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2__spec__0_spec__1_spec__3_spec__7(v_00_u03b1_4590_, v_env_4591_, v_x_4592_, v___y_4593_, v___y_4594_, v___y_4595_, v___y_4596_);
lean_dec(v___y_4596_);
lean_dec_ref(v___y_4595_);
lean_dec(v___y_4594_);
lean_dec_ref(v___y_4593_);
return v_res_4598_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Name(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_MkIffOfInductiveProp_0__Mathlib_Tactic_MkIff_initFn_00___x40_Mathlib_Tactic_MkIffOfInductiveProp_1665152415____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Name(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
}
#ifdef __cplusplus
}
#endif
