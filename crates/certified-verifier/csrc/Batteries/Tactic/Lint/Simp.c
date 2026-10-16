// Lean compiler output
// Module: Batteries.Tactic.Lint.Simp
// Imports: public import Init public meta import Init public meta import Lean.Meta.DiscrTree.Util public meta import Lean.Meta.Tactic.Simp.Main public meta import Batteries.Tactic.Lint.Basic public meta import Batteries.Tactic.OpenPrivate public meta import Batteries.Util.LibraryNote import all Lean.Meta.Tactic.Simp.SimpTheorems
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_throwError___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_checkBadRewrite_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Exception_getRef(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_Key_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_Meta_DiscrTree_instBEqKey_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* l_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_SimpTheorems_addSimpTheorem_spec__0_spec__1_spec__4(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_SimpTheorems_addSimpTheorem_spec__0_spec__1_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_binderInfo(lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_simpGlobalConfig;
lean_object* l_Lean_Meta_SimpTheorems_add(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_dsimp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_mkSorry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_preprocess(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Meta_forallTelescopeReducing___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_shouldPreprocess_spec__0___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_getSimpTheorems___redArg(lean_object*);
uint8_t l_Lean_PersistentHashMap_contains___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_eraseIfExists_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_getPrefix(lean_object*);
lean_object* l_Lean_Meta_getEqnsFor_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
lean_object* l_Lean_Elab_Term_setElabConfig(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_getConstInfo___at___00Lean_Meta_mkSimpEntryOfDeclToUnfold_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isRflTheorem(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_isEqnThm_x3f___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_addPPExplicitToExposeDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Meta_Simp_Context_mkDefault___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Option_register___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_initFn_00___x40_Lean_Meta_Tactic_Simp_SimpTheorems_838478111____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Option_get___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_isRflTheoremCore_spec__1(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
lean_object* l_Lean_sanitizeName(lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_checkBadRewrite_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_Meta_withNewMCtxDepth___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_mkSimpTheoremKeys_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Meta_DiscrTree_Key_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isCondition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isCondition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not an equality "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simpNF"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "respectTransparency"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(41, 157, 130, 207, 55, 248, 3, 166)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(75, 52, 118, 93, 62, 188, 55, 110)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 156, .m_capacity = 156, .m_length = 155, .m_data = "if true, the simpNF linter uses backward.isDefEq.respectTransparency when comparing expressions (catches more defeq abuse, but may produce false positives)"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lint"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(190, 253, 94, 7, 205, 252, 186, 173)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(213, 169, 135, 156, 82, 203, 112, 37)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(223, 155, 225, 59, 166, 107, 222, 143)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_linter_simpNF_respectTransparency;
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__0 = (const lean_object*)&lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__1 = (const lean_object*)&lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "backward"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "isDefEq"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__3_value),LEAN_SCALAR_PTR_LITERAL(77, 196, 98, 49, 58, 220, 29, 220)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__4_value),LEAN_SCALAR_PTR_LITERAL(36, 118, 4, 150, 194, 42, 143, 196)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 186, 50, 40, 52, 56, 153, 40)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___closed__0 = (const lean_object*)&lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__0_value)} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__1_value;
static const lean_array_object lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_self"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 148, 98, 216, 254, 239, 13, 169)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__0_value;
static const lean_array_object lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " only "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " +contextual"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "\nThe simp lemma is invalid because the value of argument\n  "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__7 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__7_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__9 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__9_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "\ncannot be inferred by `simp`."};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__11 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__11_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "simplify fails on hypothesis ("};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__13 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__13_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "):"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__15 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__15_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "\nThe simp lemma may be invalid because hypothesis "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__17 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__17_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = " simplifies from"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__19 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__19_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\nto"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__21 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__21_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "\nusing"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__23 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__23_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "\nTry to change the hypothesis to the simplified term!"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__25 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__25_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__0_value;
static const lean_closure_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "simplify fails on left-hand side:"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Left-hand side simplifies from"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "\nTry to change the left-hand side to the simplified term!"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 129, .m_capacity = 129, .m_length = 128, .m_data = "Left-hand side does not simplify, when using the simp lemma on itself.\n            \nThis usually means that it will never apply."};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = " can prove this:\n  by "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 115, .m_capacity = 115, .m_length = 114, .m_data = "\nOne of the lemmas above could be a duplicate.\nIf that's not the case try reordering lemmas or adding @[priority]."};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__12_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "simplify fails on right-hand side:"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__15_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__17_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "dsimp"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__18_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_Lint_simpNF___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "All left-hand sides of simp lemmas are in simp-normal form."};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpNF___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__1_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpNF___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 323, .m_capacity = 323, .m_length = 322, .m_data = "SOME SIMP LEMMAS ARE NOT IN SIMP-NORMAL FORM.\nPlease change the lemma to make sure their left-hand sides are in simp normal form.\nTo learn about simp normal forms, see\nhttps://leanprover-community.github.io/extras/simp.html#simp-normal-form\nand https://lean-lang.org/doc/reference/latest/The-Simplifier/Simp-Normal-Forms/."};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpNF___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF;
LEAN_EXPORT lean_object* lp_batteries_LibraryNote_simp_x2dnormal__form;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__0_value;
static const lean_ctor_object lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__0_value),((lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__0_value)}};
static const lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.Meta.DiscrTree.Basic"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__0_value;
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Lean.Meta.DiscrTree.insertKeyValue"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__1 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "invalid key sequence"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__2 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "should not be marked simp"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_Lint_simpComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "No commutativity lemma is marked simp."};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__1_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_simpComm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 72, .m_capacity = 72, .m_length = 71, .m_data = "COMMUTATIVITY LEMMA IS SIMP.\nSome commutativity lemmas are simp lemmas:"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_simpComm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isCondition(lean_object* v_h_1_, lean_object* v_a_2_, lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = l_Lean_Expr_fvarId_x21(v_h_1_);
v___x_8_ = l_Lean_FVarId_getDecl___redArg(v___x_7_, v_a_2_, v_a_4_, v_a_5_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v_a_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_22_; 
v_a_9_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_22_ == 0)
{
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_22_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_a_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_22_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
uint8_t v___x_13_; uint8_t v___x_14_; 
v___x_13_ = l_Lean_LocalDecl_binderInfo(v_a_9_);
v___x_14_ = l_Lean_BinderInfo_isInstImplicit(v___x_13_);
if (v___x_14_ == 0)
{
lean_object* v___x_15_; lean_object* v___x_16_; 
lean_del_object(v___x_11_);
v___x_15_ = l_Lean_LocalDecl_type(v_a_9_);
lean_dec(v_a_9_);
v___x_16_ = l_Lean_Meta_isProp(v___x_15_, v_a_2_, v_a_3_, v_a_4_, v_a_5_);
return v___x_16_;
}
else
{
uint8_t v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
lean_dec(v_a_9_);
v___x_17_ = 0;
v___x_18_ = lean_box(v___x_17_);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 0, v___x_18_);
v___x_20_ = v___x_11_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
else
{
lean_object* v_a_23_; lean_object* v___x_25_; uint8_t v_isShared_26_; uint8_t v_isSharedCheck_30_; 
v_a_23_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_30_ == 0)
{
v___x_25_ = v___x_8_;
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
else
{
lean_inc(v_a_23_);
lean_dec(v___x_8_);
v___x_25_ = lean_box(0);
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
v_resetjp_24_:
{
lean_object* v___x_28_; 
if (v_isShared_26_ == 0)
{
v___x_28_ = v___x_25_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v_a_23_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isCondition___boxed(lean_object* v_h_31_, lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_batteries_Batteries_Tactic_Lint_isCondition(v_h_31_, v_a_32_, v_a_33_, v_a_34_, v_a_35_);
lean_dec(v_a_35_);
lean_dec_ref(v_a_34_);
lean_dec(v_a_33_);
lean_dec_ref(v_a_32_);
lean_dec_ref(v_h_31_);
return v_res_37_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__2));
v___x_43_ = l_Lean_stringToMessageData(v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0(lean_object* v_k_44_, lean_object* v_hyps_45_, lean_object* v_eq_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; uint8_t v___x_54_; 
v___x_52_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1));
v___x_53_ = lean_unsigned_to_nat(3u);
v___x_54_ = l_Lean_Expr_isAppOfArity(v_eq_46_, v___x_52_, v___x_53_);
if (v___x_54_ == 0)
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
lean_dec_ref(v_hyps_45_);
lean_dec_ref(v_k_44_);
v___x_55_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__3);
v___x_56_ = l_Lean_MessageData_ofExpr(v_eq_46_);
v___x_57_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_55_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
v___x_58_ = l_Lean_throwError___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_checkBadRewrite_spec__0___redArg(v___x_57_, v___y_47_, v___y_48_, v___y_49_, v___y_50_);
return v___x_58_;
}
else
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_59_ = l_Lean_Expr_appFn_x21(v_eq_46_);
v___x_60_ = l_Lean_Expr_appArg_x21(v___x_59_);
lean_dec_ref(v___x_59_);
v___x_61_ = l_Lean_Expr_appArg_x21(v_eq_46_);
lean_dec_ref(v_eq_46_);
v___x_62_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_62_, 0, v_hyps_45_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
lean_ctor_set(v___x_62_, 2, v___x_61_);
lean_inc(v___y_50_);
lean_inc_ref(v___y_49_);
lean_inc(v___y_48_);
lean_inc_ref(v___y_47_);
v___x_63_ = lean_apply_6(v_k_44_, v___x_62_, v___y_47_, v___y_48_, v___y_49_, v___y_50_, lean_box(0));
return v___x_63_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___boxed(lean_object* v_k_64_, lean_object* v_hyps_65_, lean_object* v_eq_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0(v_k_64_, v_hyps_65_, v_eq_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_);
lean_dec(v___y_70_);
lean_dec_ref(v___y_69_);
lean_dec(v___y_68_);
lean_dec_ref(v___y_67_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg(lean_object* v_k_73_, size_t v_sz_74_, size_t v_i_75_, lean_object* v_bs_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
uint8_t v___x_82_; 
v___x_82_ = lean_usize_dec_lt(v_i_75_, v_sz_74_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; 
lean_dec_ref(v_k_73_);
v___x_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_83_, 0, v_bs_76_);
return v___x_83_;
}
else
{
lean_object* v_v_84_; lean_object* v_snd_85_; lean_object* v___f_86_; uint8_t v___x_87_; lean_object* v___x_88_; 
v_v_84_ = lean_array_uget_borrowed(v_bs_76_, v_i_75_);
v_snd_85_ = lean_ctor_get(v_v_84_, 1);
lean_inc_ref(v_k_73_);
v___f_86_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_86_, 0, v_k_73_);
v___x_87_ = 0;
lean_inc(v_snd_85_);
v___x_88_ = l_Lean_Meta_forallTelescopeReducing___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_shouldPreprocess_spec__0___redArg(v_snd_85_, v___f_86_, v___x_87_, v___x_87_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v_a_89_; lean_object* v___x_90_; lean_object* v_bs_x27_91_; size_t v___x_92_; size_t v___x_93_; lean_object* v___x_94_; 
v_a_89_ = lean_ctor_get(v___x_88_, 0);
lean_inc(v_a_89_);
lean_dec_ref_known(v___x_88_, 1);
v___x_90_ = lean_unsigned_to_nat(0u);
v_bs_x27_91_ = lean_array_uset(v_bs_76_, v_i_75_, v___x_90_);
v___x_92_ = ((size_t)1ULL);
v___x_93_ = lean_usize_add(v_i_75_, v___x_92_);
v___x_94_ = lean_array_uset(v_bs_x27_91_, v_i_75_, v_a_89_);
v_i_75_ = v___x_93_;
v_bs_76_ = v___x_94_;
goto _start;
}
else
{
lean_object* v_a_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_103_; 
lean_dec_ref(v_bs_76_);
lean_dec_ref(v_k_73_);
v_a_96_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_103_ == 0)
{
v___x_98_ = v___x_88_;
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_a_96_);
lean_dec(v___x_88_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___x_101_; 
if (v_isShared_99_ == 0)
{
v___x_101_ = v___x_98_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_a_96_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___boxed(lean_object* v_k_104_, lean_object* v_sz_105_, lean_object* v_i_106_, lean_object* v_bs_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
size_t v_sz_boxed_113_; size_t v_i_boxed_114_; lean_object* v_res_115_; 
v_sz_boxed_113_ = lean_unbox_usize(v_sz_105_);
lean_dec(v_sz_105_);
v_i_boxed_114_ = lean_unbox_usize(v_i_106_);
lean_dec(v_i_106_);
v_res_115_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg(v_k_104_, v_sz_boxed_113_, v_i_boxed_114_, v_bs_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg(lean_object* v_ty_116_, lean_object* v_k_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v_keyedConfig_123_; uint8_t v_trackZetaDelta_124_; lean_object* v_zetaDeltaSet_125_; lean_object* v_lctx_126_; lean_object* v_localInstances_127_; lean_object* v_defEqCtx_x3f_128_; lean_object* v_synthPendingDepth_129_; lean_object* v_customCanUnfoldPredicate_x3f_130_; uint8_t v_univApprox_131_; uint8_t v_inTypeClassResolution_132_; uint8_t v_cacheInferType_133_; uint8_t v___x_134_; uint8_t v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v_keyedConfig_123_ = lean_ctor_get(v_a_118_, 0);
v_trackZetaDelta_124_ = lean_ctor_get_uint8(v_a_118_, sizeof(void*)*7);
v_zetaDeltaSet_125_ = lean_ctor_get(v_a_118_, 1);
v_lctx_126_ = lean_ctor_get(v_a_118_, 2);
v_localInstances_127_ = lean_ctor_get(v_a_118_, 3);
v_defEqCtx_x3f_128_ = lean_ctor_get(v_a_118_, 4);
v_synthPendingDepth_129_ = lean_ctor_get(v_a_118_, 5);
v_customCanUnfoldPredicate_x3f_130_ = lean_ctor_get(v_a_118_, 6);
v_univApprox_131_ = lean_ctor_get_uint8(v_a_118_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_132_ = lean_ctor_get_uint8(v_a_118_, sizeof(void*)*7 + 2);
v_cacheInferType_133_ = lean_ctor_get_uint8(v_a_118_, sizeof(void*)*7 + 3);
v___x_134_ = 1;
v___x_135_ = 2;
lean_inc_ref(v_keyedConfig_123_);
v___x_136_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_135_, v_keyedConfig_123_);
lean_inc(v_customCanUnfoldPredicate_x3f_130_);
lean_inc(v_synthPendingDepth_129_);
lean_inc(v_defEqCtx_x3f_128_);
lean_inc_ref(v_localInstances_127_);
lean_inc_ref(v_lctx_126_);
lean_inc(v_zetaDeltaSet_125_);
v___x_137_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v_zetaDeltaSet_125_);
lean_ctor_set(v___x_137_, 2, v_lctx_126_);
lean_ctor_set(v___x_137_, 3, v_localInstances_127_);
lean_ctor_set(v___x_137_, 4, v_defEqCtx_x3f_128_);
lean_ctor_set(v___x_137_, 5, v_synthPendingDepth_129_);
lean_ctor_set(v___x_137_, 6, v_customCanUnfoldPredicate_x3f_130_);
lean_ctor_set_uint8(v___x_137_, sizeof(void*)*7, v_trackZetaDelta_124_);
lean_ctor_set_uint8(v___x_137_, sizeof(void*)*7 + 1, v_univApprox_131_);
lean_ctor_set_uint8(v___x_137_, sizeof(void*)*7 + 2, v_inTypeClassResolution_132_);
lean_ctor_set_uint8(v___x_137_, sizeof(void*)*7 + 3, v_cacheInferType_133_);
lean_inc_ref(v_ty_116_);
v___x_138_ = l_Lean_Meta_mkSorry(v_ty_116_, v___x_134_, v___x_137_, v_a_119_, v_a_120_, v_a_121_);
if (lean_obj_tag(v___x_138_) == 0)
{
lean_object* v_a_139_; uint8_t v___x_140_; lean_object* v___x_141_; 
v_a_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc(v_a_139_);
lean_dec_ref_known(v___x_138_, 1);
v___x_140_ = 0;
v___x_141_ = l___private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_preprocess(v_a_139_, v_ty_116_, v___x_140_, v___x_134_, v___x_137_, v_a_119_, v_a_120_, v_a_121_);
if (lean_obj_tag(v___x_141_) == 0)
{
lean_object* v_a_142_; lean_object* v___x_143_; size_t v_sz_144_; size_t v___x_145_; lean_object* v___x_146_; 
v_a_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc(v_a_142_);
lean_dec_ref_known(v___x_141_, 1);
v___x_143_ = lean_array_mk(v_a_142_);
v_sz_144_ = lean_array_size(v___x_143_);
v___x_145_ = ((size_t)0ULL);
v___x_146_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg(v_k_117_, v_sz_144_, v___x_145_, v___x_143_, v___x_137_, v_a_119_, v_a_120_, v_a_121_);
lean_dec_ref_known(v___x_137_, 7);
return v___x_146_;
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
lean_dec_ref_known(v___x_137_, 7);
lean_dec_ref(v_k_117_);
v_a_147_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_141_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_141_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
else
{
lean_object* v_a_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
lean_dec_ref_known(v___x_137_, 7);
lean_dec_ref(v_k_117_);
lean_dec_ref(v_ty_116_);
v_a_155_ = lean_ctor_get(v___x_138_, 0);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_138_);
if (v_isSharedCheck_162_ == 0)
{
v___x_157_ = v___x_138_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_a_155_);
lean_dec(v___x_138_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_a_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg___boxed(lean_object* v_ty_163_, lean_object* v_k_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg(v_ty_163_, v_k_164_, v_a_165_, v_a_166_, v_a_167_, v_a_168_);
lean_dec(v_a_168_);
lean_dec_ref(v_a_167_);
lean_dec(v_a_166_);
lean_dec_ref(v_a_165_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos(lean_object* v_00_u03b1_171_, lean_object* v_ty_172_, lean_object* v_k_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg(v_ty_172_, v_k_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___boxed(lean_object* v_00_u03b1_180_, lean_object* v_ty_181_, lean_object* v_k_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos(v_00_u03b1_180_, v_ty_181_, v_k_182_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
lean_dec(v_a_186_);
lean_dec_ref(v_a_185_);
lean_dec(v_a_184_);
lean_dec_ref(v_a_183_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0(lean_object* v_00_u03b1_189_, lean_object* v_k_190_, size_t v_sz_191_, size_t v_i_192_, lean_object* v_bs_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg(v_k_190_, v_sz_191_, v_i_192_, v_bs_193_, v___y_194_, v___y_195_, v___y_196_, v___y_197_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___boxed(lean_object* v_00_u03b1_200_, lean_object* v_k_201_, lean_object* v_sz_202_, lean_object* v_i_203_, lean_object* v_bs_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
size_t v_sz_boxed_210_; size_t v_i_boxed_211_; lean_object* v_res_212_; 
v_sz_boxed_210_ = lean_unbox_usize(v_sz_202_);
lean_dec(v_sz_202_);
v_i_boxed_211_ = lean_unbox_usize(v_i_203_);
lean_dec(v_i_203_);
v_res_212_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0(v_00_u03b1_200_, v_k_201_, v_sz_boxed_210_, v_i_boxed_211_, v_bs_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_237_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_));
v___x_238_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_));
v___x_239_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_));
v___x_240_ = l_Lean_Option_register___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_initFn_00___x40_Lean_Meta_Tactic_Simp_SimpTheorems_838478111____hygCtx___hyg_4__spec__0(v___x_237_, v___x_238_, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4____boxed(lean_object* v_a_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_();
return v_res_242_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0(lean_object* v_x_243_, lean_object* v_x_244_){
_start:
{
if (lean_obj_tag(v_x_243_) == 0)
{
if (lean_obj_tag(v_x_244_) == 0)
{
uint8_t v___x_245_; 
v___x_245_ = 1;
return v___x_245_;
}
else
{
uint8_t v___x_246_; 
v___x_246_ = 0;
return v___x_246_;
}
}
else
{
if (lean_obj_tag(v_x_244_) == 0)
{
uint8_t v___x_247_; 
v___x_247_ = 0;
return v___x_247_;
}
else
{
lean_object* v_val_248_; lean_object* v_val_249_; uint8_t v___x_250_; 
v_val_248_ = lean_ctor_get(v_x_243_, 0);
v_val_249_ = lean_ctor_get(v_x_244_, 0);
v___x_250_ = lean_name_eq(v_val_248_, v_val_249_);
return v___x_250_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0___boxed(lean_object* v_x_251_, lean_object* v_x_252_){
_start:
{
uint8_t v_res_253_; lean_object* v_r_254_; 
v_res_253_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0(v_x_251_, v_x_252_);
lean_dec(v_x_252_);
lean_dec(v_x_251_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1(lean_object* v_o_258_, lean_object* v_k_259_, uint8_t v_v_260_){
_start:
{
lean_object* v_map_261_; uint8_t v_hasTrace_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_276_; 
v_map_261_ = lean_ctor_get(v_o_258_, 0);
v_hasTrace_262_ = lean_ctor_get_uint8(v_o_258_, sizeof(void*)*1);
v_isSharedCheck_276_ = !lean_is_exclusive(v_o_258_);
if (v_isSharedCheck_276_ == 0)
{
v___x_264_ = v_o_258_;
v_isShared_265_ = v_isSharedCheck_276_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_map_261_);
lean_dec(v_o_258_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_276_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_266_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_266_, 0, v_v_260_);
lean_inc(v_k_259_);
v___x_267_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_259_, v___x_266_, v_map_261_);
if (v_hasTrace_262_ == 0)
{
lean_object* v___x_268_; uint8_t v___x_269_; lean_object* v___x_271_; 
v___x_268_ = ((lean_object*)(lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___closed__1));
v___x_269_ = l_Lean_Name_isPrefixOf(v___x_268_, v_k_259_);
lean_dec(v_k_259_);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v___x_267_);
v___x_271_ = v___x_264_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_267_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
lean_ctor_set_uint8(v___x_271_, sizeof(void*)*1, v___x_269_);
return v___x_271_;
}
}
else
{
lean_object* v___x_274_; 
lean_dec(v_k_259_);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v___x_267_);
v___x_274_ = v___x_264_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v___x_267_);
lean_ctor_set_uint8(v_reuseFailAlloc_275_, sizeof(void*)*1, v_hasTrace_262_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1___boxed(lean_object* v_o_277_, lean_object* v_k_278_, lean_object* v_v_279_){
_start:
{
uint8_t v_v_boxed_280_; lean_object* v_res_281_; 
v_v_boxed_280_ = lean_unbox(v_v_279_);
v_res_281_ = lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1(v_o_277_, v_k_278_, v_v_boxed_280_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2(lean_object* v_opts_282_, lean_object* v_opt_283_){
_start:
{
lean_object* v_name_284_; lean_object* v_defValue_285_; lean_object* v_map_286_; lean_object* v___x_287_; 
v_name_284_ = lean_ctor_get(v_opt_283_, 0);
v_defValue_285_ = lean_ctor_get(v_opt_283_, 1);
v_map_286_ = lean_ctor_get(v_opts_282_, 0);
v___x_287_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_286_, v_name_284_);
if (lean_obj_tag(v___x_287_) == 0)
{
lean_inc(v_defValue_285_);
return v_defValue_285_;
}
else
{
lean_object* v_val_288_; 
v_val_288_ = lean_ctor_get(v___x_287_, 0);
lean_inc(v_val_288_);
lean_dec_ref_known(v___x_287_, 1);
if (lean_obj_tag(v_val_288_) == 3)
{
lean_object* v_v_289_; 
v_v_289_ = lean_ctor_get(v_val_288_, 0);
lean_inc(v_v_289_);
lean_dec_ref_known(v_val_288_, 1);
return v_v_289_;
}
else
{
lean_dec(v_val_288_);
lean_inc(v_defValue_285_);
return v_defValue_285_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2___boxed(lean_object* v_opts_290_, lean_object* v_opt_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2(v_opts_290_, v_opt_291_);
lean_dec_ref(v_opt_291_);
lean_dec_ref(v_opts_290_);
return v_res_292_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0(void){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_293_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0, &lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__0);
v___x_295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
return v___x_295_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2(void){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1, &lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__1);
v___x_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq(lean_object* v_a_304_, lean_object* v_b_305_, uint8_t v_whnfFirst_306_, lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_){
_start:
{
lean_object* v___y_313_; lean_object* v___y_314_; uint8_t v___y_315_; lean_object* v___y_316_; lean_object* v___y_317_; lean_object* v___y_318_; lean_object* v___y_319_; lean_object* v___y_320_; lean_object* v___y_339_; lean_object* v___y_340_; lean_object* v___y_341_; uint8_t v___y_342_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; lean_object* v___y_346_; uint8_t v___y_347_; lean_object* v___y_369_; lean_object* v_b_370_; lean_object* v___y_371_; lean_object* v___y_372_; lean_object* v___y_373_; lean_object* v___y_374_; lean_object* v_a_393_; lean_object* v___y_394_; lean_object* v___y_395_; lean_object* v___y_396_; lean_object* v___y_397_; lean_object* v_keyedConfig_408_; uint8_t v_trackZetaDelta_409_; lean_object* v_zetaDeltaSet_410_; lean_object* v_lctx_411_; lean_object* v_localInstances_412_; lean_object* v_defEqCtx_x3f_413_; lean_object* v_synthPendingDepth_414_; lean_object* v_customCanUnfoldPredicate_x3f_415_; uint8_t v_univApprox_416_; uint8_t v_inTypeClassResolution_417_; uint8_t v_cacheInferType_418_; uint8_t v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
v_keyedConfig_408_ = lean_ctor_get(v_a_307_, 0);
v_trackZetaDelta_409_ = lean_ctor_get_uint8(v_a_307_, sizeof(void*)*7);
v_zetaDeltaSet_410_ = lean_ctor_get(v_a_307_, 1);
v_lctx_411_ = lean_ctor_get(v_a_307_, 2);
v_localInstances_412_ = lean_ctor_get(v_a_307_, 3);
v_defEqCtx_x3f_413_ = lean_ctor_get(v_a_307_, 4);
v_synthPendingDepth_414_ = lean_ctor_get(v_a_307_, 5);
v_customCanUnfoldPredicate_x3f_415_ = lean_ctor_get(v_a_307_, 6);
v_univApprox_416_ = lean_ctor_get_uint8(v_a_307_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_417_ = lean_ctor_get_uint8(v_a_307_, sizeof(void*)*7 + 2);
v_cacheInferType_418_ = lean_ctor_get_uint8(v_a_307_, sizeof(void*)*7 + 3);
v___x_419_ = 2;
lean_inc_ref(v_keyedConfig_408_);
v___x_420_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_419_, v_keyedConfig_408_);
lean_inc(v_customCanUnfoldPredicate_x3f_415_);
lean_inc(v_synthPendingDepth_414_);
lean_inc(v_defEqCtx_x3f_413_);
lean_inc_ref(v_localInstances_412_);
lean_inc_ref(v_lctx_411_);
lean_inc(v_zetaDeltaSet_410_);
v___x_421_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_421_, 0, v___x_420_);
lean_ctor_set(v___x_421_, 1, v_zetaDeltaSet_410_);
lean_ctor_set(v___x_421_, 2, v_lctx_411_);
lean_ctor_set(v___x_421_, 3, v_localInstances_412_);
lean_ctor_set(v___x_421_, 4, v_defEqCtx_x3f_413_);
lean_ctor_set(v___x_421_, 5, v_synthPendingDepth_414_);
lean_ctor_set(v___x_421_, 6, v_customCanUnfoldPredicate_x3f_415_);
lean_ctor_set_uint8(v___x_421_, sizeof(void*)*7, v_trackZetaDelta_409_);
lean_ctor_set_uint8(v___x_421_, sizeof(void*)*7 + 1, v_univApprox_416_);
lean_ctor_set_uint8(v___x_421_, sizeof(void*)*7 + 2, v_inTypeClassResolution_417_);
lean_ctor_set_uint8(v___x_421_, sizeof(void*)*7 + 3, v_cacheInferType_418_);
if (v_whnfFirst_306_ == 0)
{
v_a_393_ = v_a_304_;
v___y_394_ = v___x_421_;
v___y_395_ = v_a_308_;
v___y_396_ = v_a_309_;
v___y_397_ = v_a_310_;
goto v___jp_392_;
}
else
{
lean_object* v___x_422_; 
lean_inc(v_a_310_);
lean_inc_ref(v_a_309_);
lean_inc(v_a_308_);
lean_inc_ref(v___x_421_);
v___x_422_ = lean_whnf(v_a_304_, v___x_421_, v_a_308_, v_a_309_, v_a_310_);
if (lean_obj_tag(v___x_422_) == 0)
{
lean_object* v_a_423_; 
v_a_423_ = lean_ctor_get(v___x_422_, 0);
lean_inc(v_a_423_);
lean_dec_ref_known(v___x_422_, 1);
v_a_393_ = v_a_423_;
v___y_394_ = v___x_421_;
v___y_395_ = v_a_308_;
v___y_396_ = v_a_309_;
v___y_397_ = v_a_310_;
goto v___jp_392_;
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
lean_dec_ref_known(v___x_421_, 7);
lean_dec_ref(v_b_305_);
v_a_424_ = lean_ctor_get(v___x_422_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_422_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_422_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_424_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
v___jp_312_:
{
lean_object* v_fileName_321_; lean_object* v_fileMap_322_; lean_object* v_currRecDepth_323_; lean_object* v_ref_324_; lean_object* v_currNamespace_325_; lean_object* v_openDecls_326_; lean_object* v_initHeartbeats_327_; lean_object* v_maxHeartbeats_328_; lean_object* v_quotContext_329_; lean_object* v_currMacroScope_330_; lean_object* v_cancelTk_x3f_331_; uint8_t v_suppressElabErrors_332_; lean_object* v_inheritedTraceOptions_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v_fileName_321_ = lean_ctor_get(v___y_319_, 0);
v_fileMap_322_ = lean_ctor_get(v___y_319_, 1);
v_currRecDepth_323_ = lean_ctor_get(v___y_319_, 3);
v_ref_324_ = lean_ctor_get(v___y_319_, 5);
v_currNamespace_325_ = lean_ctor_get(v___y_319_, 6);
v_openDecls_326_ = lean_ctor_get(v___y_319_, 7);
v_initHeartbeats_327_ = lean_ctor_get(v___y_319_, 8);
v_maxHeartbeats_328_ = lean_ctor_get(v___y_319_, 9);
v_quotContext_329_ = lean_ctor_get(v___y_319_, 10);
v_currMacroScope_330_ = lean_ctor_get(v___y_319_, 11);
v_cancelTk_x3f_331_ = lean_ctor_get(v___y_319_, 12);
v_suppressElabErrors_332_ = lean_ctor_get_uint8(v___y_319_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_333_ = lean_ctor_get(v___y_319_, 13);
v___x_334_ = l_Lean_maxRecDepth;
v___x_335_ = lp_batteries_Lean_Option_get___at___00Batteries_Tactic_Lint_isSimpEq_spec__2(v___y_317_, v___x_334_);
lean_inc_ref(v_inheritedTraceOptions_333_);
lean_inc(v_cancelTk_x3f_331_);
lean_inc(v_currMacroScope_330_);
lean_inc(v_quotContext_329_);
lean_inc(v_maxHeartbeats_328_);
lean_inc(v_initHeartbeats_327_);
lean_inc(v_openDecls_326_);
lean_inc(v_currNamespace_325_);
lean_inc(v_ref_324_);
lean_inc(v_currRecDepth_323_);
lean_inc_ref(v_fileMap_322_);
lean_inc_ref(v_fileName_321_);
v___x_336_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_336_, 0, v_fileName_321_);
lean_ctor_set(v___x_336_, 1, v_fileMap_322_);
lean_ctor_set(v___x_336_, 2, v___y_317_);
lean_ctor_set(v___x_336_, 3, v_currRecDepth_323_);
lean_ctor_set(v___x_336_, 4, v___x_335_);
lean_ctor_set(v___x_336_, 5, v_ref_324_);
lean_ctor_set(v___x_336_, 6, v_currNamespace_325_);
lean_ctor_set(v___x_336_, 7, v_openDecls_326_);
lean_ctor_set(v___x_336_, 8, v_initHeartbeats_327_);
lean_ctor_set(v___x_336_, 9, v_maxHeartbeats_328_);
lean_ctor_set(v___x_336_, 10, v_quotContext_329_);
lean_ctor_set(v___x_336_, 11, v_currMacroScope_330_);
lean_ctor_set(v___x_336_, 12, v_cancelTk_x3f_331_);
lean_ctor_set(v___x_336_, 13, v_inheritedTraceOptions_333_);
lean_ctor_set_uint8(v___x_336_, sizeof(void*)*14, v___y_315_);
lean_ctor_set_uint8(v___x_336_, sizeof(void*)*14 + 1, v_suppressElabErrors_332_);
v___x_337_ = l_Lean_Meta_isExprDefEq(v___y_316_, v___y_313_, v___y_318_, v___y_314_, v___x_336_, v___y_320_);
lean_dec_ref_known(v___x_336_, 14);
lean_dec_ref(v___y_318_);
return v___x_337_;
}
v___jp_338_:
{
if (v___y_347_ == 0)
{
lean_object* v___x_348_; lean_object* v_env_349_; lean_object* v_nextMacroScope_350_; lean_object* v_ngen_351_; lean_object* v_auxDeclNGen_352_; lean_object* v_traceState_353_; lean_object* v_messages_354_; lean_object* v_infoState_355_; lean_object* v_snapshotTasks_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_366_; 
v___x_348_ = lean_st_ref_take(v___y_341_);
v_env_349_ = lean_ctor_get(v___x_348_, 0);
v_nextMacroScope_350_ = lean_ctor_get(v___x_348_, 1);
v_ngen_351_ = lean_ctor_get(v___x_348_, 2);
v_auxDeclNGen_352_ = lean_ctor_get(v___x_348_, 3);
v_traceState_353_ = lean_ctor_get(v___x_348_, 4);
v_messages_354_ = lean_ctor_get(v___x_348_, 6);
v_infoState_355_ = lean_ctor_get(v___x_348_, 7);
v_snapshotTasks_356_ = lean_ctor_get(v___x_348_, 8);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_366_ == 0)
{
lean_object* v_unused_367_; 
v_unused_367_ = lean_ctor_get(v___x_348_, 5);
lean_dec(v_unused_367_);
v___x_358_ = v___x_348_;
v_isShared_359_ = v_isSharedCheck_366_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_snapshotTasks_356_);
lean_inc(v_infoState_355_);
lean_inc(v_messages_354_);
lean_inc(v_traceState_353_);
lean_inc(v_auxDeclNGen_352_);
lean_inc(v_ngen_351_);
lean_inc(v_nextMacroScope_350_);
lean_inc(v_env_349_);
lean_dec(v___x_348_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_366_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_363_; 
v___x_360_ = l_Lean_Kernel_enableDiag(v_env_349_, v___y_342_);
v___x_361_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2, &lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2_once, _init_lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__2);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 5, v___x_361_);
lean_ctor_set(v___x_358_, 0, v___x_360_);
v___x_363_ = v___x_358_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v___x_360_);
lean_ctor_set(v_reuseFailAlloc_365_, 1, v_nextMacroScope_350_);
lean_ctor_set(v_reuseFailAlloc_365_, 2, v_ngen_351_);
lean_ctor_set(v_reuseFailAlloc_365_, 3, v_auxDeclNGen_352_);
lean_ctor_set(v_reuseFailAlloc_365_, 4, v_traceState_353_);
lean_ctor_set(v_reuseFailAlloc_365_, 5, v___x_361_);
lean_ctor_set(v_reuseFailAlloc_365_, 6, v_messages_354_);
lean_ctor_set(v_reuseFailAlloc_365_, 7, v_infoState_355_);
lean_ctor_set(v_reuseFailAlloc_365_, 8, v_snapshotTasks_356_);
v___x_363_ = v_reuseFailAlloc_365_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
lean_object* v___x_364_; 
v___x_364_ = lean_st_ref_set(v___y_341_, v___x_363_);
v___y_313_ = v___y_339_;
v___y_314_ = v___y_340_;
v___y_315_ = v___y_342_;
v___y_316_ = v___y_344_;
v___y_317_ = v___y_343_;
v___y_318_ = v___y_346_;
v___y_319_ = v___y_345_;
v___y_320_ = v___y_341_;
goto v___jp_312_;
}
}
}
else
{
v___y_313_ = v___y_339_;
v___y_314_ = v___y_340_;
v___y_315_ = v___y_342_;
v___y_316_ = v___y_344_;
v___y_317_ = v___y_343_;
v___y_318_ = v___y_346_;
v___y_319_ = v___y_345_;
v___y_320_ = v___y_341_;
goto v___jp_312_;
}
}
v___jp_368_:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; uint8_t v___x_379_; 
v___x_375_ = l_Lean_Expr_getAppFn(v___y_369_);
v___x_376_ = l_Lean_Expr_constName_x3f(v___x_375_);
lean_dec_ref(v___x_375_);
v___x_377_ = l_Lean_Expr_getAppFn(v_b_370_);
v___x_378_ = l_Lean_Expr_constName_x3f(v___x_377_);
lean_dec_ref(v___x_377_);
v___x_379_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0(v___x_376_, v___x_378_);
lean_dec(v___x_378_);
lean_dec(v___x_376_);
if (v___x_379_ == 0)
{
lean_object* v___x_380_; lean_object* v___x_381_; 
lean_dec_ref(v___y_371_);
lean_dec_ref(v_b_370_);
lean_dec_ref(v___y_369_);
v___x_380_ = lean_box(v___x_379_);
v___x_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
return v___x_381_;
}
else
{
lean_object* v___x_382_; lean_object* v_options_383_; lean_object* v_env_384_; lean_object* v___x_385_; uint8_t v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; uint8_t v___x_390_; uint8_t v___x_391_; 
v___x_382_ = lean_st_ref_get(v___y_374_);
v_options_383_ = lean_ctor_get(v___y_373_, 2);
v_env_384_ = lean_ctor_get(v___x_382_, 0);
lean_inc_ref(v_env_384_);
lean_dec(v___x_382_);
v___x_385_ = lp_batteries_Batteries_Tactic_Lint_linter_simpNF_respectTransparency;
v___x_386_ = l_Lean_Option_get___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_isRflTheoremCore_spec__1(v_options_383_, v___x_385_);
v___x_387_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_isSimpEq___closed__5));
lean_inc_ref(v_options_383_);
v___x_388_ = lp_batteries_Lean_Options_set___at___00Batteries_Tactic_Lint_isSimpEq_spec__1(v_options_383_, v___x_387_, v___x_386_);
v___x_389_ = l_Lean_diagnostics;
v___x_390_ = l_Lean_Option_get___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_isRflTheoremCore_spec__1(v___x_388_, v___x_389_);
v___x_391_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_384_);
lean_dec_ref(v_env_384_);
if (v___x_391_ == 0)
{
if (v___x_390_ == 0)
{
v___y_339_ = v_b_370_;
v___y_340_ = v___y_372_;
v___y_341_ = v___y_374_;
v___y_342_ = v___x_390_;
v___y_343_ = v___x_388_;
v___y_344_ = v___y_369_;
v___y_345_ = v___y_373_;
v___y_346_ = v___y_371_;
v___y_347_ = v___x_379_;
goto v___jp_338_;
}
else
{
v___y_339_ = v_b_370_;
v___y_340_ = v___y_372_;
v___y_341_ = v___y_374_;
v___y_342_ = v___x_390_;
v___y_343_ = v___x_388_;
v___y_344_ = v___y_369_;
v___y_345_ = v___y_373_;
v___y_346_ = v___y_371_;
v___y_347_ = v___x_391_;
goto v___jp_338_;
}
}
else
{
v___y_339_ = v_b_370_;
v___y_340_ = v___y_372_;
v___y_341_ = v___y_374_;
v___y_342_ = v___x_390_;
v___y_343_ = v___x_388_;
v___y_344_ = v___y_369_;
v___y_345_ = v___y_373_;
v___y_346_ = v___y_371_;
v___y_347_ = v___x_390_;
goto v___jp_338_;
}
}
}
v___jp_392_:
{
if (v_whnfFirst_306_ == 0)
{
v___y_369_ = v_a_393_;
v_b_370_ = v_b_305_;
v___y_371_ = v___y_394_;
v___y_372_ = v___y_395_;
v___y_373_ = v___y_396_;
v___y_374_ = v___y_397_;
goto v___jp_368_;
}
else
{
lean_object* v___x_398_; 
lean_inc(v___y_397_);
lean_inc_ref(v___y_396_);
lean_inc(v___y_395_);
lean_inc_ref(v___y_394_);
v___x_398_ = lean_whnf(v_b_305_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v_a_399_; 
v_a_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_a_399_);
lean_dec_ref_known(v___x_398_, 1);
v___y_369_ = v_a_393_;
v_b_370_ = v_a_399_;
v___y_371_ = v___y_394_;
v___y_372_ = v___y_395_;
v___y_373_ = v___y_396_;
v___y_374_ = v___y_397_;
goto v___jp_368_;
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec_ref(v___y_394_);
lean_dec_ref(v_a_393_);
v_a_400_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_398_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_398_);
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
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpEq___boxed(lean_object* v_a_432_, lean_object* v_b_433_, lean_object* v_whnfFirst_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
uint8_t v_whnfFirst_boxed_440_; lean_object* v_res_441_; 
v_whnfFirst_boxed_440_ = lean_unbox(v_whnfFirst_434_);
v_res_441_ = lp_batteries_Batteries_Tactic_Lint_isSimpEq(v_a_432_, v_b_433_, v_whnfFirst_boxed_440_, v_a_435_, v_a_436_, v_a_437_, v_a_438_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0(lean_object* v_k_442_, lean_object* v_i_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___x_449_; 
lean_inc(v___y_447_);
lean_inc_ref(v___y_446_);
lean_inc(v___y_445_);
lean_inc_ref(v___y_444_);
v___x_449_ = lean_apply_6(v_k_442_, v_i_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, lean_box(0));
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
if (lean_obj_tag(v_a_450_) == 0)
{
return v___x_449_;
}
else
{
lean_object* v_val_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_467_; 
lean_dec_ref_known(v___x_449_, 1);
v_val_451_ = lean_ctor_get(v_a_450_, 0);
v_isSharedCheck_467_ = !lean_is_exclusive(v_a_450_);
if (v_isSharedCheck_467_ == 0)
{
v___x_453_ = v_a_450_;
v_isShared_454_ = v_isSharedCheck_467_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_val_451_);
lean_dec(v_a_450_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_467_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v___x_455_; lean_object* v_a_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_466_; 
v___x_455_ = l_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_checkBadRewrite_spec__0_spec__0(v_val_451_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
v_a_456_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_466_ == 0)
{
v___x_458_ = v___x_455_;
v_isShared_459_ = v_isSharedCheck_466_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_a_456_);
lean_dec(v___x_455_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_466_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_461_; 
if (v_isShared_454_ == 0)
{
lean_ctor_set(v___x_453_, 0, v_a_456_);
v___x_461_ = v___x_453_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_456_);
v___x_461_ = v_reuseFailAlloc_465_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
lean_object* v___x_463_; 
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 0, v___x_461_);
v___x_463_ = v___x_458_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v___x_461_);
v___x_463_ = v_reuseFailAlloc_464_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
return v___x_463_;
}
}
}
}
}
}
else
{
return v___x_449_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0___boxed(lean_object* v_k_468_, lean_object* v_i_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v_res_475_; 
v_res_475_ = lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0(v_k_468_, v_i_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0(lean_object* v_as_476_, size_t v_i_477_, size_t v_stop_478_, lean_object* v_b_479_){
_start:
{
lean_object* v___y_481_; uint8_t v___x_485_; 
v___x_485_ = lean_usize_dec_eq(v_i_477_, v_stop_478_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; 
v___x_486_ = lean_array_uget_borrowed(v_as_476_, v_i_477_);
if (lean_obj_tag(v___x_486_) == 0)
{
v___y_481_ = v_b_479_;
goto v___jp_480_;
}
else
{
lean_object* v_val_487_; lean_object* v___x_488_; 
v_val_487_ = lean_ctor_get(v___x_486_, 0);
lean_inc(v_val_487_);
v___x_488_ = lean_array_push(v_b_479_, v_val_487_);
v___y_481_ = v___x_488_;
goto v___jp_480_;
}
}
else
{
return v_b_479_;
}
v___jp_480_:
{
size_t v___x_482_; size_t v___x_483_; 
v___x_482_ = ((size_t)1ULL);
v___x_483_ = lean_usize_add(v_i_477_, v___x_482_);
v_i_477_ = v___x_483_;
v_b_479_ = v___y_481_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0___boxed(lean_object* v_as_489_, lean_object* v_i_490_, lean_object* v_stop_491_, lean_object* v_b_492_){
_start:
{
size_t v_i_boxed_493_; size_t v_stop_boxed_494_; lean_object* v_res_495_; 
v_i_boxed_493_ = lean_unbox_usize(v_i_490_);
lean_dec(v_i_490_);
v_stop_boxed_494_ = lean_unbox_usize(v_stop_491_);
lean_dec(v_stop_491_);
v_res_495_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0(v_as_489_, v_i_boxed_493_, v_stop_boxed_494_, v_b_492_);
lean_dec_ref(v_as_489_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0(lean_object* v_as_498_, lean_object* v_start_499_, lean_object* v_stop_500_){
_start:
{
lean_object* v___x_501_; uint8_t v___x_502_; 
v___x_501_ = ((lean_object*)(lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___closed__0));
v___x_502_ = lean_nat_dec_lt(v_start_499_, v_stop_500_);
if (v___x_502_ == 0)
{
return v___x_501_;
}
else
{
lean_object* v___x_503_; uint8_t v___x_504_; 
v___x_503_ = lean_array_get_size(v_as_498_);
v___x_504_ = lean_nat_dec_le(v_stop_500_, v___x_503_);
if (v___x_504_ == 0)
{
uint8_t v___x_505_; 
v___x_505_ = lean_nat_dec_lt(v_start_499_, v___x_503_);
if (v___x_505_ == 0)
{
return v___x_501_;
}
else
{
size_t v___x_506_; size_t v___x_507_; lean_object* v___x_508_; 
v___x_506_ = lean_usize_of_nat(v_start_499_);
v___x_507_ = lean_usize_of_nat(v___x_503_);
v___x_508_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0(v_as_498_, v___x_506_, v___x_507_, v___x_501_);
return v___x_508_;
}
}
else
{
size_t v___x_509_; size_t v___x_510_; lean_object* v___x_511_; 
v___x_509_ = lean_usize_of_nat(v_start_499_);
v___x_510_ = lean_usize_of_nat(v_stop_500_);
v___x_511_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0_spec__0(v_as_498_, v___x_509_, v___x_510_, v___x_501_);
return v___x_511_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___boxed(lean_object* v_as_512_, lean_object* v_start_513_, lean_object* v_stop_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0(v_as_512_, v_start_513_, v_stop_514_);
lean_dec(v_stop_514_);
lean_dec(v_start_513_);
lean_dec_ref(v_as_512_);
return v_res_515_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0(void){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_516_ = lean_box(1);
v___x_517_ = l_Lean_MessageData_ofFormat(v___x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos(lean_object* v_ty_518_, lean_object* v_k_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_, lean_object* v_a_523_){
_start:
{
lean_object* v___f_525_; lean_object* v___x_526_; 
v___f_525_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___lam__0___boxed), 7, 1);
lean_closure_set(v___f_525_, 0, v_k_519_);
v___x_526_ = lp_batteries_Batteries_Tactic_Lint_withSimpTheoremInfos___redArg(v_ty_518_, v___f_525_, v_a_520_, v_a_521_, v_a_522_, v_a_523_);
if (lean_obj_tag(v___x_526_) == 0)
{
lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_547_; 
v_a_527_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_547_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_547_ == 0)
{
v___x_529_ = v___x_526_;
v_isShared_530_ = v_isSharedCheck_547_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_526_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_547_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; uint8_t v___x_535_; 
v___x_531_ = lean_unsigned_to_nat(0u);
v___x_532_ = lean_array_get_size(v_a_527_);
v___x_533_ = lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0(v_a_527_, v___x_531_, v___x_532_);
lean_dec(v_a_527_);
v___x_534_ = lean_array_get_size(v___x_533_);
v___x_535_ = lean_nat_dec_eq(v___x_534_, v___x_531_);
if (v___x_535_ == 0)
{
lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_541_; 
v___x_536_ = lean_array_to_list(v___x_533_);
v___x_537_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0, &lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___closed__0);
v___x_538_ = l_Lean_MessageData_joinSep(v___x_536_, v___x_537_);
v___x_539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v___x_539_);
v___x_541_ = v___x_529_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v___x_539_);
v___x_541_ = v_reuseFailAlloc_542_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
return v___x_541_;
}
}
else
{
lean_object* v___x_543_; lean_object* v___x_545_; 
lean_dec_ref(v___x_533_);
v___x_543_ = lean_box(0);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v___x_543_);
v___x_545_ = v___x_529_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v___x_543_);
v___x_545_ = v_reuseFailAlloc_546_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
return v___x_545_;
}
}
}
}
else
{
lean_object* v_a_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_555_; 
v_a_548_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_555_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_555_ == 0)
{
v___x_550_ = v___x_526_;
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_a_548_);
lean_dec(v___x_526_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_553_; 
if (v_isShared_551_ == 0)
{
v___x_553_ = v___x_550_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v_a_548_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos___boxed(lean_object* v_ty_556_, lean_object* v_k_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos(v_ty_556_, v_k_557_, v_a_558_, v_a_559_, v_a_560_, v_a_561_);
lean_dec(v_a_561_);
lean_dec_ref(v_a_560_);
lean_dec(v_a_559_);
lean_dec_ref(v_a_558_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(lean_object* v_declName_564_, lean_object* v_a_565_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = l_Lean_Meta_getSimpTheorems___redArg(v_a_565_);
if (lean_obj_tag(v___x_567_) == 0)
{
lean_object* v_a_568_; lean_object* v___x_570_; uint8_t v_isShared_571_; uint8_t v_isSharedCheck_581_; 
v_a_568_ = lean_ctor_get(v___x_567_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_567_);
if (v_isSharedCheck_581_ == 0)
{
v___x_570_ = v___x_567_;
v_isShared_571_ = v_isSharedCheck_581_;
goto v_resetjp_569_;
}
else
{
lean_inc(v_a_568_);
lean_dec(v___x_567_);
v___x_570_ = lean_box(0);
v_isShared_571_ = v_isSharedCheck_581_;
goto v_resetjp_569_;
}
v_resetjp_569_:
{
lean_object* v_lemmaNames_572_; uint8_t v___x_573_; uint8_t v___x_574_; lean_object* v___x_575_; uint8_t v___x_576_; lean_object* v___x_577_; lean_object* v___x_579_; 
v_lemmaNames_572_ = lean_ctor_get(v_a_568_, 2);
lean_inc_ref(v_lemmaNames_572_);
lean_dec(v_a_568_);
v___x_573_ = 1;
v___x_574_ = 0;
v___x_575_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_575_, 0, v_declName_564_);
lean_ctor_set_uint8(v___x_575_, sizeof(void*)*1, v___x_573_);
lean_ctor_set_uint8(v___x_575_, sizeof(void*)*1 + 1, v___x_574_);
v___x_576_ = l_Lean_PersistentHashMap_contains___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_eraseIfExists_spec__0___redArg(v_lemmaNames_572_, v___x_575_);
lean_dec_ref_known(v___x_575_, 1);
lean_dec_ref(v_lemmaNames_572_);
v___x_577_ = lean_box(v___x_576_);
if (v_isShared_571_ == 0)
{
lean_ctor_set(v___x_570_, 0, v___x_577_);
v___x_579_ = v___x_570_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v___x_577_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
else
{
lean_object* v_a_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_589_; 
lean_dec(v_declName_564_);
v_a_582_ = lean_ctor_get(v___x_567_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_567_);
if (v_isSharedCheck_589_ == 0)
{
v___x_584_ = v___x_567_;
v_isShared_585_ = v_isSharedCheck_589_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_a_582_);
lean_dec(v___x_567_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_589_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v___x_587_; 
if (v_isShared_585_ == 0)
{
v___x_587_ = v___x_584_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v_a_582_);
v___x_587_ = v_reuseFailAlloc_588_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
return v___x_587_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg___boxed(lean_object* v_declName_590_, lean_object* v_a_591_, lean_object* v_a_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(v_declName_590_, v_a_591_);
lean_dec(v_a_591_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem(lean_object* v_declName_594_, lean_object* v_a_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_){
_start:
{
lean_object* v___x_600_; 
v___x_600_ = lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(v_declName_594_, v_a_598_);
return v___x_600_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___boxed(lean_object* v_declName_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_batteries_Batteries_Tactic_Lint_isSimpTheorem(v_declName_601_, v_a_602_, v_a_603_, v_a_604_, v_a_605_);
lean_dec(v_a_605_);
lean_dec_ref(v_a_604_);
lean_dec(v_a_603_);
lean_dec_ref(v_a_602_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__0(lean_object* v_x1_608_, lean_object* v_x2_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lean_array_push(v_x1_608_, v_x2_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(lean_object* v_f_611_, lean_object* v_as_612_, size_t v_i_613_, size_t v_stop_614_, lean_object* v_b_615_){
_start:
{
uint8_t v___x_616_; 
v___x_616_ = lean_usize_dec_eq(v_i_613_, v_stop_614_);
if (v___x_616_ == 0)
{
lean_object* v___x_617_; lean_object* v___x_618_; size_t v___x_619_; size_t v___x_620_; 
v___x_617_ = lean_array_uget_borrowed(v_as_612_, v_i_613_);
lean_inc(v_f_611_);
lean_inc(v___x_617_);
v___x_618_ = lean_apply_2(v_f_611_, v_b_615_, v___x_617_);
v___x_619_ = ((size_t)1ULL);
v___x_620_ = lean_usize_add(v_i_613_, v___x_619_);
v_i_613_ = v___x_620_;
v_b_615_ = v___x_618_;
goto _start;
}
else
{
lean_dec(v_f_611_);
return v_b_615_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg___boxed(lean_object* v_f_622_, lean_object* v_as_623_, lean_object* v_i_624_, lean_object* v_stop_625_, lean_object* v_b_626_){
_start:
{
size_t v_i_boxed_627_; size_t v_stop_boxed_628_; lean_object* v_res_629_; 
v_i_boxed_627_ = lean_unbox_usize(v_i_624_);
lean_dec(v_i_624_);
v_stop_boxed_628_ = lean_unbox_usize(v_stop_625_);
lean_dec(v_stop_625_);
v_res_629_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(v_f_622_, v_as_623_, v_i_boxed_627_, v_stop_boxed_628_, v_b_626_);
lean_dec_ref(v_as_623_);
return v_res_629_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(lean_object* v_f_630_, lean_object* v_x_631_, lean_object* v_x_632_){
_start:
{
lean_object* v_vs_633_; lean_object* v_children_634_; lean_object* v___x_635_; lean_object* v_s_637_; lean_object* v___x_647_; uint8_t v___x_648_; 
v_vs_633_ = lean_ctor_get(v_x_632_, 0);
v_children_634_ = lean_ctor_get(v_x_632_, 1);
v___x_635_ = lean_unsigned_to_nat(0u);
v___x_647_ = lean_array_get_size(v_vs_633_);
v___x_648_ = lean_nat_dec_lt(v___x_635_, v___x_647_);
if (v___x_648_ == 0)
{
lean_object* v___x_649_; uint8_t v___x_650_; 
v___x_649_ = lean_array_get_size(v_children_634_);
v___x_650_ = lean_nat_dec_lt(v___x_635_, v___x_649_);
if (v___x_650_ == 0)
{
lean_dec(v_f_630_);
return v_x_631_;
}
else
{
uint8_t v___x_651_; 
v___x_651_ = lean_nat_dec_le(v___x_649_, v___x_649_);
if (v___x_651_ == 0)
{
if (v___x_650_ == 0)
{
lean_dec(v_f_630_);
return v_x_631_;
}
else
{
size_t v___x_652_; size_t v___x_653_; lean_object* v___x_654_; 
v___x_652_ = ((size_t)0ULL);
v___x_653_ = lean_usize_of_nat(v___x_649_);
v___x_654_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_630_, v_children_634_, v___x_652_, v___x_653_, v_x_631_);
return v___x_654_;
}
}
else
{
size_t v___x_655_; size_t v___x_656_; lean_object* v___x_657_; 
v___x_655_ = ((size_t)0ULL);
v___x_656_ = lean_usize_of_nat(v___x_649_);
v___x_657_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_630_, v_children_634_, v___x_655_, v___x_656_, v_x_631_);
return v___x_657_;
}
}
}
else
{
uint8_t v___x_658_; 
v___x_658_ = lean_nat_dec_le(v___x_647_, v___x_647_);
if (v___x_658_ == 0)
{
if (v___x_648_ == 0)
{
v_s_637_ = v_x_631_;
goto v___jp_636_;
}
else
{
size_t v___x_659_; size_t v___x_660_; lean_object* v___x_661_; 
v___x_659_ = ((size_t)0ULL);
v___x_660_ = lean_usize_of_nat(v___x_647_);
lean_inc(v_f_630_);
v___x_661_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(v_f_630_, v_vs_633_, v___x_659_, v___x_660_, v_x_631_);
v_s_637_ = v___x_661_;
goto v___jp_636_;
}
}
else
{
size_t v___x_662_; size_t v___x_663_; lean_object* v___x_664_; 
v___x_662_ = ((size_t)0ULL);
v___x_663_ = lean_usize_of_nat(v___x_647_);
lean_inc(v_f_630_);
v___x_664_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(v_f_630_, v_vs_633_, v___x_662_, v___x_663_, v_x_631_);
v_s_637_ = v___x_664_;
goto v___jp_636_;
}
}
v___jp_636_:
{
lean_object* v___x_638_; uint8_t v___x_639_; 
v___x_638_ = lean_array_get_size(v_children_634_);
v___x_639_ = lean_nat_dec_lt(v___x_635_, v___x_638_);
if (v___x_639_ == 0)
{
lean_dec(v_f_630_);
return v_s_637_;
}
else
{
uint8_t v___x_640_; 
v___x_640_ = lean_nat_dec_le(v___x_638_, v___x_638_);
if (v___x_640_ == 0)
{
if (v___x_639_ == 0)
{
lean_dec(v_f_630_);
return v_s_637_;
}
else
{
size_t v___x_641_; size_t v___x_642_; lean_object* v___x_643_; 
v___x_641_ = ((size_t)0ULL);
v___x_642_ = lean_usize_of_nat(v___x_638_);
v___x_643_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_630_, v_children_634_, v___x_641_, v___x_642_, v_s_637_);
return v___x_643_;
}
}
else
{
size_t v___x_644_; size_t v___x_645_; lean_object* v___x_646_; 
v___x_644_ = ((size_t)0ULL);
v___x_645_ = lean_usize_of_nat(v___x_638_);
v___x_646_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_630_, v_children_634_, v___x_644_, v___x_645_, v_s_637_);
return v___x_646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(lean_object* v_f_665_, lean_object* v_as_666_, size_t v_i_667_, size_t v_stop_668_, lean_object* v_b_669_){
_start:
{
uint8_t v___x_670_; 
v___x_670_ = lean_usize_dec_eq(v_i_667_, v_stop_668_);
if (v___x_670_ == 0)
{
lean_object* v___x_671_; lean_object* v_snd_672_; lean_object* v___x_673_; size_t v___x_674_; size_t v___x_675_; 
v___x_671_ = lean_array_uget_borrowed(v_as_666_, v_i_667_);
v_snd_672_ = lean_ctor_get(v___x_671_, 1);
lean_inc(v_f_665_);
v___x_673_ = lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(v_f_665_, v_b_669_, v_snd_672_);
v___x_674_ = ((size_t)1ULL);
v___x_675_ = lean_usize_add(v_i_667_, v___x_674_);
v_i_667_ = v___x_675_;
v_b_669_ = v___x_673_;
goto _start;
}
else
{
lean_dec(v_f_665_);
return v_b_669_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg___boxed(lean_object* v_f_677_, lean_object* v_as_678_, lean_object* v_i_679_, lean_object* v_stop_680_, lean_object* v_b_681_){
_start:
{
size_t v_i_boxed_682_; size_t v_stop_boxed_683_; lean_object* v_res_684_; 
v_i_boxed_682_ = lean_unbox_usize(v_i_679_);
lean_dec(v_i_679_);
v_stop_boxed_683_ = lean_unbox_usize(v_stop_680_);
lean_dec(v_stop_680_);
v_res_684_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_677_, v_as_678_, v_i_boxed_682_, v_stop_boxed_683_, v_b_681_);
lean_dec_ref(v_as_678_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg___boxed(lean_object* v_f_685_, lean_object* v_x_686_, lean_object* v_x_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(v_f_685_, v_x_686_, v_x_687_);
lean_dec_ref(v_x_687_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1(lean_object* v___f_689_, lean_object* v_s_690_, lean_object* v_x_691_, lean_object* v_t_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(v___f_689_, v_s_690_, v_t_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1___boxed(lean_object* v___f_694_, lean_object* v_s_695_, lean_object* v_x_696_, lean_object* v_t_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_batteries_Lean_Meta_DiscrTree_elements___redArg___lam__1(v___f_694_, v_s_695_, v_x_696_, v_t_697_);
lean_dec_ref(v_t_697_);
lean_dec(v_x_696_);
return v_res_698_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg(lean_object* v_f_699_, lean_object* v_keys_700_, lean_object* v_vals_701_, lean_object* v_i_702_, lean_object* v_acc_703_){
_start:
{
lean_object* v___x_704_; uint8_t v___x_705_; 
v___x_704_ = lean_array_get_size(v_keys_700_);
v___x_705_ = lean_nat_dec_lt(v_i_702_, v___x_704_);
if (v___x_705_ == 0)
{
lean_dec(v_i_702_);
lean_dec(v_f_699_);
return v_acc_703_;
}
else
{
lean_object* v_k_706_; lean_object* v_v_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v_k_706_ = lean_array_fget_borrowed(v_keys_700_, v_i_702_);
v_v_707_ = lean_array_fget_borrowed(v_vals_701_, v_i_702_);
lean_inc(v_f_699_);
lean_inc(v_v_707_);
lean_inc(v_k_706_);
v___x_708_ = lean_apply_3(v_f_699_, v_acc_703_, v_k_706_, v_v_707_);
v___x_709_ = lean_unsigned_to_nat(1u);
v___x_710_ = lean_nat_add(v_i_702_, v___x_709_);
lean_dec(v_i_702_);
v_i_702_ = v___x_710_;
v_acc_703_ = v___x_708_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_f_712_, lean_object* v_keys_713_, lean_object* v_vals_714_, lean_object* v_i_715_, lean_object* v_acc_716_){
_start:
{
lean_object* v_res_717_; 
v_res_717_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg(v_f_712_, v_keys_713_, v_vals_714_, v_i_715_, v_acc_716_);
lean_dec_ref(v_vals_714_);
lean_dec_ref(v_keys_713_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(lean_object* v_f_718_, lean_object* v_x_719_, lean_object* v_x_720_){
_start:
{
if (lean_obj_tag(v_x_719_) == 0)
{
lean_object* v_es_721_; lean_object* v___x_722_; lean_object* v___x_723_; uint8_t v___x_724_; 
v_es_721_ = lean_ctor_get(v_x_719_, 0);
v___x_722_ = lean_unsigned_to_nat(0u);
v___x_723_ = lean_array_get_size(v_es_721_);
v___x_724_ = lean_nat_dec_lt(v___x_722_, v___x_723_);
if (v___x_724_ == 0)
{
lean_dec(v_f_718_);
return v_x_720_;
}
else
{
uint8_t v___x_725_; 
v___x_725_ = lean_nat_dec_le(v___x_723_, v___x_723_);
if (v___x_725_ == 0)
{
if (v___x_724_ == 0)
{
lean_dec(v_f_718_);
return v_x_720_;
}
else
{
size_t v___x_726_; size_t v___x_727_; lean_object* v___x_728_; 
v___x_726_ = ((size_t)0ULL);
v___x_727_ = lean_usize_of_nat(v___x_723_);
v___x_728_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(v_f_718_, v_es_721_, v___x_726_, v___x_727_, v_x_720_);
return v___x_728_;
}
}
else
{
size_t v___x_729_; size_t v___x_730_; lean_object* v___x_731_; 
v___x_729_ = ((size_t)0ULL);
v___x_730_ = lean_usize_of_nat(v___x_723_);
v___x_731_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(v_f_718_, v_es_721_, v___x_729_, v___x_730_, v_x_720_);
return v___x_731_;
}
}
}
else
{
lean_object* v_ks_732_; lean_object* v_vs_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v_ks_732_ = lean_ctor_get(v_x_719_, 0);
v_vs_733_ = lean_ctor_get(v_x_719_, 1);
v___x_734_ = lean_unsigned_to_nat(0u);
v___x_735_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg(v_f_718_, v_ks_732_, v_vs_733_, v___x_734_, v_x_720_);
return v___x_735_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(lean_object* v_f_736_, lean_object* v_as_737_, size_t v_i_738_, size_t v_stop_739_, lean_object* v_b_740_){
_start:
{
lean_object* v___y_742_; uint8_t v___x_746_; 
v___x_746_ = lean_usize_dec_eq(v_i_738_, v_stop_739_);
if (v___x_746_ == 0)
{
lean_object* v___x_747_; 
v___x_747_ = lean_array_uget_borrowed(v_as_737_, v_i_738_);
switch(lean_obj_tag(v___x_747_))
{
case 0:
{
lean_object* v_key_748_; lean_object* v_val_749_; lean_object* v___x_750_; 
v_key_748_ = lean_ctor_get(v___x_747_, 0);
v_val_749_ = lean_ctor_get(v___x_747_, 1);
lean_inc(v_f_736_);
lean_inc(v_val_749_);
lean_inc(v_key_748_);
v___x_750_ = lean_apply_3(v_f_736_, v_b_740_, v_key_748_, v_val_749_);
v___y_742_ = v___x_750_;
goto v___jp_741_;
}
case 1:
{
lean_object* v_node_751_; lean_object* v___x_752_; 
v_node_751_ = lean_ctor_get(v___x_747_, 0);
lean_inc(v_f_736_);
v___x_752_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_736_, v_node_751_, v_b_740_);
v___y_742_ = v___x_752_;
goto v___jp_741_;
}
default: 
{
v___y_742_ = v_b_740_;
goto v___jp_741_;
}
}
}
else
{
lean_dec(v_f_736_);
return v_b_740_;
}
v___jp_741_:
{
size_t v___x_743_; size_t v___x_744_; 
v___x_743_ = ((size_t)1ULL);
v___x_744_ = lean_usize_add(v_i_738_, v___x_743_);
v_i_738_ = v___x_744_;
v_b_740_ = v___y_742_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg___boxed(lean_object* v_f_753_, lean_object* v_as_754_, lean_object* v_i_755_, lean_object* v_stop_756_, lean_object* v_b_757_){
_start:
{
size_t v_i_boxed_758_; size_t v_stop_boxed_759_; lean_object* v_res_760_; 
v_i_boxed_758_ = lean_unbox_usize(v_i_755_);
lean_dec(v_i_755_);
v_stop_boxed_759_ = lean_unbox_usize(v_stop_756_);
lean_dec(v_stop_756_);
v_res_760_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(v_f_753_, v_as_754_, v_i_boxed_758_, v_stop_boxed_759_, v_b_757_);
lean_dec_ref(v_as_754_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg___boxed(lean_object* v_f_761_, lean_object* v_x_762_, lean_object* v_x_763_){
_start:
{
lean_object* v_res_764_; 
v_res_764_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_761_, v_x_762_, v_x_763_);
lean_dec_ref(v_x_762_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg(lean_object* v_d_770_){
_start:
{
lean_object* v___f_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v___f_771_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__1));
v___x_772_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_elements___redArg___closed__2));
v___x_773_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v___f_771_, v_d_770_, v___x_772_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___redArg___boxed(lean_object* v_d_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_batteries_Lean_Meta_DiscrTree_elements___redArg(v_d_774_);
lean_dec_ref(v_d_774_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements(lean_object* v_00_u03b1_776_, lean_object* v_d_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_batteries_Lean_Meta_DiscrTree_elements___redArg(v_d_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_elements___boxed(lean_object* v_00_u03b1_779_, lean_object* v_d_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_batteries_Lean_Meta_DiscrTree_elements(v_00_u03b1_779_, v_d_780_);
lean_dec_ref(v_d_780_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0(lean_object* v_00_u03c3_782_, lean_object* v_00_u03b1_783_, lean_object* v_f_784_, lean_object* v_x_785_, lean_object* v_x_786_){
_start:
{
lean_object* v___x_787_; 
v___x_787_ = lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___redArg(v_f_784_, v_x_785_, v_x_786_);
return v___x_787_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0___boxed(lean_object* v_00_u03c3_788_, lean_object* v_00_u03b1_789_, lean_object* v_f_790_, lean_object* v_x_791_, lean_object* v_x_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_batteries_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0(v_00_u03c3_788_, v_00_u03b1_789_, v_f_790_, v_x_791_, v_x_792_);
lean_dec_ref(v_x_792_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___redArg(lean_object* v_map_794_, lean_object* v_f_795_, lean_object* v_init_796_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_795_, v_map_794_, v_init_796_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___redArg___boxed(lean_object* v_map_798_, lean_object* v_f_799_, lean_object* v_init_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___redArg(v_map_798_, v_f_799_, v_init_800_);
lean_dec_ref(v_map_798_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1(lean_object* v_00_u03c3_802_, lean_object* v_00_u03b2_803_, lean_object* v_map_804_, lean_object* v_f_805_, lean_object* v_init_806_){
_start:
{
lean_object* v___x_807_; 
v___x_807_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_805_, v_map_804_, v_init_806_);
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1___boxed(lean_object* v_00_u03c3_808_, lean_object* v_00_u03b2_809_, lean_object* v_map_810_, lean_object* v_f_811_, lean_object* v_init_812_){
_start:
{
lean_object* v_res_813_; 
v_res_813_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1(v_00_u03c3_808_, v_00_u03b2_809_, v_map_810_, v_f_811_, v_init_812_);
lean_dec_ref(v_map_810_);
return v_res_813_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0(lean_object* v_00_u03b1_814_, lean_object* v_00_u03c3_815_, lean_object* v_f_816_, lean_object* v_as_817_, size_t v_i_818_, size_t v_stop_819_, lean_object* v_b_820_){
_start:
{
lean_object* v___x_821_; 
v___x_821_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___redArg(v_f_816_, v_as_817_, v_i_818_, v_stop_819_, v_b_820_);
return v___x_821_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0___boxed(lean_object* v_00_u03b1_822_, lean_object* v_00_u03c3_823_, lean_object* v_f_824_, lean_object* v_as_825_, lean_object* v_i_826_, lean_object* v_stop_827_, lean_object* v_b_828_){
_start:
{
size_t v_i_boxed_829_; size_t v_stop_boxed_830_; lean_object* v_res_831_; 
v_i_boxed_829_ = lean_unbox_usize(v_i_826_);
lean_dec(v_i_826_);
v_stop_boxed_830_ = lean_unbox_usize(v_stop_827_);
lean_dec(v_stop_827_);
v_res_831_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__0(v_00_u03b1_822_, v_00_u03c3_823_, v_f_824_, v_as_825_, v_i_boxed_829_, v_stop_boxed_830_, v_b_828_);
lean_dec_ref(v_as_825_);
return v_res_831_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1(lean_object* v_00_u03b1_832_, lean_object* v_00_u03c3_833_, lean_object* v_f_834_, lean_object* v_as_835_, size_t v_i_836_, size_t v_stop_837_, lean_object* v_b_838_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___redArg(v_f_834_, v_as_835_, v_i_836_, v_stop_837_, v_b_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1___boxed(lean_object* v_00_u03b1_840_, lean_object* v_00_u03c3_841_, lean_object* v_f_842_, lean_object* v_as_843_, lean_object* v_i_844_, lean_object* v_stop_845_, lean_object* v_b_846_){
_start:
{
size_t v_i_boxed_847_; size_t v_stop_boxed_848_; lean_object* v_res_849_; 
v_i_boxed_847_ = lean_unbox_usize(v_i_844_);
lean_dec(v_i_844_);
v_stop_boxed_848_ = lean_unbox_usize(v_stop_845_);
lean_dec(v_stop_845_);
v_res_849_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Lean_Meta_DiscrTree_elements_spec__0_spec__1(v_00_u03b1_840_, v_00_u03c3_841_, v_f_842_, v_as_843_, v_i_boxed_847_, v_stop_boxed_848_, v_b_846_);
lean_dec_ref(v_as_843_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3(lean_object* v_00_u03c3_850_, lean_object* v_00_u03b1_851_, lean_object* v_00_u03b2_852_, lean_object* v_f_853_, lean_object* v_x_854_, lean_object* v_x_855_){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_853_, v_x_854_, v_x_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___boxed(lean_object* v_00_u03c3_857_, lean_object* v_00_u03b1_858_, lean_object* v_00_u03b2_859_, lean_object* v_f_860_, lean_object* v_x_861_, lean_object* v_x_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3(v_00_u03c3_857_, v_00_u03b1_858_, v_00_u03b2_859_, v_f_860_, v_x_861_, v_x_862_);
lean_dec_ref(v_x_861_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4(lean_object* v_00_u03b1_864_, lean_object* v_00_u03b2_865_, lean_object* v_00_u03c3_866_, lean_object* v_f_867_, lean_object* v_as_868_, size_t v_i_869_, size_t v_stop_870_, lean_object* v_b_871_){
_start:
{
lean_object* v___x_872_; 
v___x_872_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___redArg(v_f_867_, v_as_868_, v_i_869_, v_stop_870_, v_b_871_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4___boxed(lean_object* v_00_u03b1_873_, lean_object* v_00_u03b2_874_, lean_object* v_00_u03c3_875_, lean_object* v_f_876_, lean_object* v_as_877_, lean_object* v_i_878_, lean_object* v_stop_879_, lean_object* v_b_880_){
_start:
{
size_t v_i_boxed_881_; size_t v_stop_boxed_882_; lean_object* v_res_883_; 
v_i_boxed_881_ = lean_unbox_usize(v_i_878_);
lean_dec(v_i_878_);
v_stop_boxed_882_ = lean_unbox_usize(v_stop_879_);
lean_dec(v_stop_879_);
v_res_883_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__4(v_00_u03b1_873_, v_00_u03b2_874_, v_00_u03c3_875_, v_f_876_, v_as_877_, v_i_boxed_881_, v_stop_boxed_882_, v_b_880_);
lean_dec_ref(v_as_877_);
return v_res_883_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5(lean_object* v_00_u03c3_884_, lean_object* v_00_u03b1_885_, lean_object* v_00_u03b2_886_, lean_object* v_f_887_, lean_object* v_keys_888_, lean_object* v_vals_889_, lean_object* v_heq_890_, lean_object* v_i_891_, lean_object* v_acc_892_){
_start:
{
lean_object* v___x_893_; 
v___x_893_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___redArg(v_f_887_, v_keys_888_, v_vals_889_, v_i_891_, v_acc_892_);
return v___x_893_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03c3_894_, lean_object* v_00_u03b1_895_, lean_object* v_00_u03b2_896_, lean_object* v_f_897_, lean_object* v_keys_898_, lean_object* v_vals_899_, lean_object* v_heq_900_, lean_object* v_i_901_, lean_object* v_acc_902_){
_start:
{
lean_object* v_res_903_; 
v_res_903_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3_spec__5(v_00_u03c3_894_, v_00_u03b1_895_, v_00_u03b2_896_, v_f_897_, v_keys_898_, v_vals_899_, v_heq_900_, v_i_901_, v_acc_902_);
lean_dec_ref(v_vals_899_);
lean_dec_ref(v_keys_898_);
return v_res_903_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1(void){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_905_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__0));
v___x_906_ = l_Lean_stringToMessageData(v___x_905_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(lean_object* v_msg_907_, lean_object* v_k_908_, lean_object* v_a_909_, lean_object* v_a_910_, lean_object* v_a_911_, lean_object* v_a_912_){
_start:
{
lean_object* v___x_914_; 
lean_inc(v_a_912_);
lean_inc_ref(v_a_911_);
lean_inc(v_a_910_);
lean_inc_ref(v_a_909_);
v___x_914_ = lean_apply_5(v_k_908_, v_a_909_, v_a_910_, v_a_911_, v_a_912_, lean_box(0));
if (lean_obj_tag(v___x_914_) == 0)
{
lean_dec_ref(v_msg_907_);
return v___x_914_;
}
else
{
lean_object* v_a_915_; uint8_t v___y_917_; uint8_t v___x_932_; 
v_a_915_ = lean_ctor_get(v___x_914_, 0);
lean_inc(v_a_915_);
v___x_932_ = l_Lean_Exception_isInterrupt(v_a_915_);
if (v___x_932_ == 0)
{
uint8_t v___x_933_; 
lean_inc(v_a_915_);
v___x_933_ = l_Lean_Exception_isRuntime(v_a_915_);
v___y_917_ = v___x_933_;
goto v___jp_916_;
}
else
{
v___y_917_ = v___x_932_;
goto v___jp_916_;
}
v___jp_916_:
{
if (v___y_917_ == 0)
{
lean_object* v___x_919_; uint8_t v_isShared_920_; uint8_t v_isSharedCheck_930_; 
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_914_);
if (v_isSharedCheck_930_ == 0)
{
lean_object* v_unused_931_; 
v_unused_931_ = lean_ctor_get(v___x_914_, 0);
lean_dec(v_unused_931_);
v___x_919_ = v___x_914_;
v_isShared_920_ = v_isSharedCheck_930_;
goto v_resetjp_918_;
}
else
{
lean_dec(v___x_914_);
v___x_919_ = lean_box(0);
v_isShared_920_ = v_isSharedCheck_930_;
goto v_resetjp_918_;
}
v_resetjp_918_:
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_928_; 
v___x_921_ = l_Lean_Exception_getRef(v_a_915_);
v___x_922_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1, &lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1);
v___x_923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_923_, 0, v_msg_907_);
lean_ctor_set(v___x_923_, 1, v___x_922_);
v___x_924_ = l_Lean_Exception_toMessageData(v_a_915_);
v___x_925_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_925_, 0, v___x_923_);
lean_ctor_set(v___x_925_, 1, v___x_924_);
v___x_926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_926_, 0, v___x_921_);
lean_ctor_set(v___x_926_, 1, v___x_925_);
if (v_isShared_920_ == 0)
{
lean_ctor_set(v___x_919_, 0, v___x_926_);
v___x_928_ = v___x_919_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v___x_926_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
else
{
lean_dec(v_a_915_);
lean_dec_ref(v_msg_907_);
return v___x_914_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___boxed(lean_object* v_msg_934_, lean_object* v_k_935_, lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v_a_938_, lean_object* v_a_939_, lean_object* v_a_940_){
_start:
{
lean_object* v_res_941_; 
v_res_941_ = lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(v_msg_934_, v_k_935_, v_a_936_, v_a_937_, v_a_938_, v_a_939_);
lean_dec(v_a_939_);
lean_dec_ref(v_a_938_);
lean_dec(v_a_937_);
lean_dec_ref(v_a_936_);
return v_res_941_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError(lean_object* v_00_u03b1_942_, lean_object* v_msg_943_, lean_object* v_k_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(v_msg_943_, v_k_944_, v_a_945_, v_a_946_, v_a_947_, v_a_948_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_decorateError___boxed(lean_object* v_00_u03b1_951_, lean_object* v_msg_952_, lean_object* v_k_953_, lean_object* v_a_954_, lean_object* v_a_955_, lean_object* v_a_956_, lean_object* v_a_957_, lean_object* v_a_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_batteries_Batteries_Tactic_Lint_decorateError(v_00_u03b1_951_, v_msg_952_, v_k_953_, v_a_954_, v_a_955_, v_a_956_, v_a_957_);
lean_dec(v_a_957_);
lean_dec_ref(v_a_956_);
lean_dec(v_a_955_);
lean_dec_ref(v_a_954_);
return v_res_959_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0(lean_object* v_x_960_, lean_object* v_x_961_){
_start:
{
if (lean_obj_tag(v_x_960_) == 0)
{
if (lean_obj_tag(v_x_961_) == 0)
{
uint8_t v___x_962_; 
v___x_962_ = 1;
return v___x_962_;
}
else
{
uint8_t v___x_963_; 
v___x_963_ = 0;
return v___x_963_;
}
}
else
{
if (lean_obj_tag(v_x_961_) == 0)
{
uint8_t v___x_964_; 
v___x_964_ = 0;
return v___x_964_;
}
else
{
lean_object* v_val_965_; uint8_t v___x_966_; 
v_val_965_ = lean_ctor_get(v_x_960_, 0);
v___x_966_ = lean_unbox(v_val_965_);
if (v___x_966_ == 0)
{
lean_object* v_val_967_; uint8_t v___x_968_; 
v_val_967_ = lean_ctor_get(v_x_961_, 0);
v___x_968_ = lean_unbox(v_val_967_);
if (v___x_968_ == 0)
{
uint8_t v___x_969_; 
v___x_969_ = 1;
return v___x_969_;
}
else
{
uint8_t v___x_970_; 
v___x_970_ = lean_unbox(v_val_965_);
return v___x_970_;
}
}
else
{
lean_object* v_val_971_; uint8_t v___x_972_; 
v_val_971_ = lean_ctor_get(v_x_961_, 0);
v___x_972_ = lean_unbox(v_val_971_);
return v___x_972_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0___boxed(lean_object* v_x_973_, lean_object* v_x_974_){
_start:
{
uint8_t v_res_975_; lean_object* v_r_976_; 
v_res_975_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0(v_x_973_, v_x_974_);
lean_dec(v_x_974_);
lean_dec(v_x_973_);
v_r_976_ = lean_box(v_res_975_);
return v_r_976_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1(lean_object* v___x_980_, lean_object* v_as_981_, size_t v_sz_982_, size_t v_i_983_, lean_object* v_b_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_){
_start:
{
lean_object* v_a_991_; uint8_t v___x_995_; 
v___x_995_ = lean_usize_dec_lt(v_i_983_, v_sz_982_);
if (v___x_995_ == 0)
{
lean_object* v___x_996_; 
lean_dec_ref(v___x_980_);
v___x_996_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_996_, 0, v_b_984_);
return v___x_996_;
}
else
{
lean_object* v_a_997_; lean_object* v_fst_998_; 
v_a_997_ = lean_array_uget_borrowed(v_as_981_, v_i_983_);
v_fst_998_ = lean_ctor_get(v_a_997_, 0);
if (lean_obj_tag(v_fst_998_) == 0)
{
lean_object* v_declName_999_; uint8_t v_post_1000_; uint8_t v_inv_1001_; uint8_t v___y_1003_; 
v_declName_999_ = lean_ctor_get(v_fst_998_, 0);
v_post_1000_ = lean_ctor_get_uint8(v_fst_998_, sizeof(void*)*1);
v_inv_1001_ = lean_ctor_get_uint8(v_fst_998_, sizeof(void*)*1 + 1);
if (v_post_1000_ == 1)
{
if (v_inv_1001_ == 0)
{
uint8_t v___x_1016_; 
lean_inc(v_declName_999_);
lean_inc_ref(v___x_980_);
v___x_1016_ = l_Lean_Environment_contains(v___x_980_, v_declName_999_, v_post_1000_);
if (v___x_1016_ == 0)
{
v___y_1003_ = v___x_1016_;
goto v___jp_1002_;
}
else
{
lean_object* v___x_1017_; uint8_t v___x_1018_; 
v___x_1017_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___closed__1));
v___x_1018_ = lean_name_eq(v_declName_999_, v___x_1017_);
if (v___x_1018_ == 0)
{
v___y_1003_ = v___x_1016_;
goto v___jp_1002_;
}
else
{
v_a_991_ = v_b_984_;
goto v___jp_990_;
}
}
}
else
{
v_a_991_ = v_b_984_;
goto v___jp_990_;
}
}
else
{
v_a_991_ = v_b_984_;
goto v___jp_990_;
}
v___jp_1002_:
{
if (v___y_1003_ == 0)
{
v_a_991_ = v_b_984_;
goto v___jp_990_;
}
else
{
lean_object* v___x_1004_; 
lean_inc(v_declName_999_);
v___x_1004_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_declName_999_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
if (lean_obj_tag(v___x_1004_) == 0)
{
lean_object* v_a_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; 
v_a_1005_ = lean_ctor_get(v___x_1004_, 0);
lean_inc(v_a_1005_);
lean_dec_ref_known(v___x_1004_, 1);
v___x_1006_ = l_Lean_MessageData_ofExpr(v_a_1005_);
v___x_1007_ = lean_array_push(v_b_984_, v___x_1006_);
v_a_991_ = v___x_1007_;
goto v___jp_990_;
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
lean_dec_ref(v_b_984_);
lean_dec_ref(v___x_980_);
v_a_1008_ = lean_ctor_get(v___x_1004_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_1004_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_1004_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_1004_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1008_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
}
}
else
{
v_a_991_ = v_b_984_;
goto v___jp_990_;
}
}
v___jp_990_:
{
size_t v___x_992_; size_t v___x_993_; 
v___x_992_ = ((size_t)1ULL);
v___x_993_ = lean_usize_add(v_i_983_, v___x_992_);
v_i_983_ = v___x_993_;
v_b_984_ = v_a_991_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1___boxed(lean_object* v___x_1019_, lean_object* v_as_1020_, lean_object* v_sz_1021_, lean_object* v_i_1022_, lean_object* v_b_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_){
_start:
{
size_t v_sz_boxed_1029_; size_t v_i_boxed_1030_; lean_object* v_res_1031_; 
v_sz_boxed_1029_ = lean_unbox_usize(v_sz_1021_);
lean_dec(v_sz_1021_);
v_i_boxed_1030_ = lean_unbox_usize(v_i_1022_);
lean_dec(v_i_1022_);
v_res_1031_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1(v___x_1019_, v_as_1020_, v_sz_boxed_1029_, v_i_boxed_1030_, v_b_1023_, v___y_1024_, v___y_1025_, v___y_1026_, v___y_1027_);
lean_dec(v___y_1027_);
lean_dec_ref(v___y_1026_);
lean_dec(v___y_1025_);
lean_dec_ref(v___y_1024_);
lean_dec_ref(v_as_1020_);
return v_res_1031_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___lam__0(lean_object* v_ps_1032_, lean_object* v_k_1033_, lean_object* v_v_1034_){
_start:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; 
v___x_1035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1035_, 0, v_k_1033_);
lean_ctor_set(v___x_1035_, 1, v_v_1034_);
v___x_1036_ = lean_array_push(v_ps_1032_, v___x_1035_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg___lam__0(lean_object* v_f_1037_, lean_object* v_x1_1038_, lean_object* v_x2_1039_, lean_object* v_x3_1040_){
_start:
{
lean_object* v___x_1041_; 
v___x_1041_ = lean_apply_3(v_f_1037_, v_x1_1038_, v_x2_1039_, v_x3_1040_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg(lean_object* v_map_1042_, lean_object* v_f_1043_, lean_object* v_init_1044_){
_start:
{
lean_object* v___f_1045_; lean_object* v___x_1046_; 
v___f_1045_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1045_, 0, v_f_1043_);
v___x_1046_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v___f_1045_, v_map_1042_, v_init_1044_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg___boxed(lean_object* v_map_1047_, lean_object* v_f_1048_, lean_object* v_init_1049_){
_start:
{
lean_object* v_res_1050_; 
v_res_1050_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg(v_map_1047_, v_f_1048_, v_init_1049_);
lean_dec_ref(v_map_1047_);
return v_res_1050_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg(lean_object* v_m_1054_){
_start:
{
lean_object* v___f_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___f_1055_ = ((lean_object*)(lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__0));
v___x_1056_ = ((lean_object*)(lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___closed__1));
v___x_1057_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg(v_m_1054_, v___f_1055_, v___x_1056_);
return v___x_1057_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg___boxed(lean_object* v_m_1058_){
_start:
{
lean_object* v_res_1059_; 
v_res_1059_ = lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg(v_m_1058_);
lean_dec_ref(v_m_1058_);
return v_res_1059_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(lean_object* v_x1_1060_, lean_object* v_x2_1061_){
_start:
{
lean_object* v_snd_1062_; lean_object* v_snd_1063_; uint8_t v___x_1064_; 
v_snd_1062_ = lean_ctor_get(v_x1_1060_, 1);
v_snd_1063_ = lean_ctor_get(v_x2_1061_, 1);
v___x_1064_ = lean_nat_dec_lt(v_snd_1062_, v_snd_1063_);
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0___boxed(lean_object* v_x1_1065_, lean_object* v_x2_1066_){
_start:
{
uint8_t v_res_1067_; lean_object* v_r_1068_; 
v_res_1067_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(v_x1_1065_, v_x2_1066_);
lean_dec_ref(v_x2_1066_);
lean_dec_ref(v_x1_1065_);
v_r_1068_ = lean_box(v_res_1067_);
return v_r_1068_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg(lean_object* v_hi_1069_, lean_object* v_pivot_1070_, lean_object* v_as_1071_, lean_object* v_i_1072_, lean_object* v_k_1073_){
_start:
{
uint8_t v___x_1074_; 
v___x_1074_ = lean_nat_dec_lt(v_k_1073_, v_hi_1069_);
if (v___x_1074_ == 0)
{
lean_object* v___x_1075_; lean_object* v___x_1076_; 
lean_dec(v_k_1073_);
v___x_1075_ = lean_array_fswap(v_as_1071_, v_i_1072_, v_hi_1069_);
v___x_1076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1076_, 0, v_i_1072_);
lean_ctor_set(v___x_1076_, 1, v___x_1075_);
return v___x_1076_;
}
else
{
lean_object* v___x_1077_; lean_object* v_snd_1078_; lean_object* v_snd_1079_; uint8_t v___x_1080_; 
v___x_1077_ = lean_array_fget_borrowed(v_as_1071_, v_k_1073_);
v_snd_1078_ = lean_ctor_get(v___x_1077_, 1);
v_snd_1079_ = lean_ctor_get(v_pivot_1070_, 1);
v___x_1080_ = lean_nat_dec_lt(v_snd_1078_, v_snd_1079_);
if (v___x_1080_ == 0)
{
lean_object* v___x_1081_; lean_object* v___x_1082_; 
v___x_1081_ = lean_unsigned_to_nat(1u);
v___x_1082_ = lean_nat_add(v_k_1073_, v___x_1081_);
lean_dec(v_k_1073_);
v_k_1073_ = v___x_1082_;
goto _start;
}
else
{
lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1084_ = lean_array_fswap(v_as_1071_, v_i_1072_, v_k_1073_);
v___x_1085_ = lean_unsigned_to_nat(1u);
v___x_1086_ = lean_nat_add(v_i_1072_, v___x_1085_);
lean_dec(v_i_1072_);
v___x_1087_ = lean_nat_add(v_k_1073_, v___x_1085_);
lean_dec(v_k_1073_);
v_as_1071_ = v___x_1084_;
v_i_1072_ = v___x_1086_;
v_k_1073_ = v___x_1087_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg___boxed(lean_object* v_hi_1089_, lean_object* v_pivot_1090_, lean_object* v_as_1091_, lean_object* v_i_1092_, lean_object* v_k_1093_){
_start:
{
lean_object* v_res_1094_; 
v_res_1094_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg(v_hi_1089_, v_pivot_1090_, v_as_1091_, v_i_1092_, v_k_1093_);
lean_dec_ref(v_pivot_1090_);
lean_dec(v_hi_1089_);
return v_res_1094_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(lean_object* v_n_1095_, lean_object* v_as_1096_, lean_object* v_lo_1097_, lean_object* v_hi_1098_){
_start:
{
lean_object* v___y_1100_; uint8_t v___x_1110_; 
v___x_1110_ = lean_nat_dec_lt(v_lo_1097_, v_hi_1098_);
if (v___x_1110_ == 0)
{
lean_dec(v_lo_1097_);
return v_as_1096_;
}
else
{
lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v_mid_1113_; lean_object* v___y_1115_; lean_object* v___y_1121_; lean_object* v___x_1126_; lean_object* v___x_1127_; uint8_t v___x_1128_; 
v___x_1111_ = lean_nat_add(v_lo_1097_, v_hi_1098_);
v___x_1112_ = lean_unsigned_to_nat(1u);
v_mid_1113_ = lean_nat_shiftr(v___x_1111_, v___x_1112_);
lean_dec(v___x_1111_);
v___x_1126_ = lean_array_fget_borrowed(v_as_1096_, v_mid_1113_);
v___x_1127_ = lean_array_fget_borrowed(v_as_1096_, v_lo_1097_);
v___x_1128_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(v___x_1126_, v___x_1127_);
if (v___x_1128_ == 0)
{
v___y_1121_ = v_as_1096_;
goto v___jp_1120_;
}
else
{
lean_object* v___x_1129_; 
v___x_1129_ = lean_array_fswap(v_as_1096_, v_lo_1097_, v_mid_1113_);
v___y_1121_ = v___x_1129_;
goto v___jp_1120_;
}
v___jp_1114_:
{
lean_object* v___x_1116_; lean_object* v___x_1117_; uint8_t v___x_1118_; 
v___x_1116_ = lean_array_fget_borrowed(v___y_1115_, v_mid_1113_);
v___x_1117_ = lean_array_fget_borrowed(v___y_1115_, v_hi_1098_);
v___x_1118_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(v___x_1116_, v___x_1117_);
if (v___x_1118_ == 0)
{
lean_dec(v_mid_1113_);
v___y_1100_ = v___y_1115_;
goto v___jp_1099_;
}
else
{
lean_object* v___x_1119_; 
v___x_1119_ = lean_array_fswap(v___y_1115_, v_mid_1113_, v_hi_1098_);
lean_dec(v_mid_1113_);
v___y_1100_ = v___x_1119_;
goto v___jp_1099_;
}
}
v___jp_1120_:
{
lean_object* v___x_1122_; lean_object* v___x_1123_; uint8_t v___x_1124_; 
v___x_1122_ = lean_array_fget_borrowed(v___y_1121_, v_hi_1098_);
v___x_1123_ = lean_array_fget_borrowed(v___y_1121_, v_lo_1097_);
v___x_1124_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___lam__0(v___x_1122_, v___x_1123_);
if (v___x_1124_ == 0)
{
v___y_1115_ = v___y_1121_;
goto v___jp_1114_;
}
else
{
lean_object* v___x_1125_; 
v___x_1125_ = lean_array_fswap(v___y_1121_, v_lo_1097_, v_hi_1098_);
v___y_1115_ = v___x_1125_;
goto v___jp_1114_;
}
}
}
v___jp_1099_:
{
lean_object* v_pivot_1101_; lean_object* v___x_1102_; lean_object* v_fst_1103_; lean_object* v_snd_1104_; uint8_t v___x_1105_; 
v_pivot_1101_ = lean_array_fget(v___y_1100_, v_hi_1098_);
lean_inc_n(v_lo_1097_, 2);
v___x_1102_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg(v_hi_1098_, v_pivot_1101_, v___y_1100_, v_lo_1097_, v_lo_1097_);
lean_dec(v_pivot_1101_);
v_fst_1103_ = lean_ctor_get(v___x_1102_, 0);
lean_inc(v_fst_1103_);
v_snd_1104_ = lean_ctor_get(v___x_1102_, 1);
lean_inc(v_snd_1104_);
lean_dec_ref(v___x_1102_);
v___x_1105_ = lean_nat_dec_le(v_hi_1098_, v_fst_1103_);
if (v___x_1105_ == 0)
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; 
v___x_1106_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(v_n_1095_, v_snd_1104_, v_lo_1097_, v_fst_1103_);
v___x_1107_ = lean_unsigned_to_nat(1u);
v___x_1108_ = lean_nat_add(v_fst_1103_, v___x_1107_);
lean_dec(v_fst_1103_);
v_as_1096_ = v___x_1106_;
v_lo_1097_ = v___x_1108_;
goto _start;
}
else
{
lean_dec(v_fst_1103_);
lean_dec(v_lo_1097_);
return v_snd_1104_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg___boxed(lean_object* v_n_1130_, lean_object* v_as_1131_, lean_object* v_lo_1132_, lean_object* v_hi_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(v_n_1130_, v_as_1131_, v_lo_1132_, v_hi_1133_);
lean_dec(v_hi_1133_);
lean_dec(v_n_1130_);
return v_res_1134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__2(lean_object* v_a_1135_, lean_object* v_a_1136_){
_start:
{
if (lean_obj_tag(v_a_1135_) == 0)
{
lean_object* v___x_1137_; 
v___x_1137_ = l_List_reverse___redArg(v_a_1136_);
return v___x_1137_;
}
else
{
lean_object* v_head_1138_; lean_object* v_tail_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1147_; 
v_head_1138_ = lean_ctor_get(v_a_1135_, 0);
v_tail_1139_ = lean_ctor_get(v_a_1135_, 1);
v_isSharedCheck_1147_ = !lean_is_exclusive(v_a_1135_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1141_ = v_a_1135_;
v_isShared_1142_ = v_isSharedCheck_1147_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_tail_1139_);
lean_inc(v_head_1138_);
lean_dec(v_a_1135_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1147_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1144_; 
if (v_isShared_1142_ == 0)
{
lean_ctor_set(v___x_1141_, 1, v_a_1136_);
v___x_1144_ = v___x_1141_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_head_1138_);
lean_ctor_set(v_reuseFailAlloc_1146_, 1, v_a_1136_);
v___x_1144_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
v_a_1135_ = v_tail_1139_;
v_a_1136_ = v___x_1144_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1(void){
_start:
{
lean_object* v___x_1149_; lean_object* v___x_1150_; 
v___x_1149_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__0));
v___x_1150_ = l_Lean_stringToMessageData(v___x_1149_);
return v___x_1150_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6(void){
_start:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1157_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__5));
v___x_1158_ = l_Lean_stringToMessageData(v___x_1157_);
return v___x_1158_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7(void){
_start:
{
lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1159_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6, &lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__6);
v___x_1160_ = lean_unsigned_to_nat(1u);
v___x_1161_ = lean_mk_empty_array_with_capacity(v___x_1160_);
v___x_1162_ = lean_array_push(v___x_1161_, v___x_1159_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas(lean_object* v_usedSimps_1163_, lean_object* v_simpName_1164_, lean_object* v_higherOrder_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_){
_start:
{
lean_object* v___y_1172_; lean_object* v___y_1173_; lean_object* v___x_1185_; uint8_t v___x_1186_; lean_object* v___y_1188_; lean_object* v___y_1189_; lean_object* v___y_1190_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v___y_1212_; lean_object* v___y_1213_; lean_object* v___y_1216_; lean_object* v___y_1217_; lean_object* v___y_1218_; lean_object* v___y_1219_; lean_object* v___y_1220_; lean_object* v___y_1221_; lean_object* v___y_1224_; 
v___x_1185_ = lean_box(0);
v___x_1186_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0(v_higherOrder_1165_, v___x_1185_);
if (v___x_1186_ == 0)
{
lean_object* v___x_1235_; 
v___x_1235_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7, &lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__7);
v___y_1224_ = v___x_1235_;
goto v___jp_1223_;
}
else
{
lean_object* v___x_1236_; 
v___x_1236_ = ((lean_object*)(lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_checkAllSimpTheoremInfos_spec__0___closed__0));
v___y_1224_ = v___x_1236_;
goto v___jp_1223_;
}
v___jp_1171_:
{
lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; 
v___x_1174_ = l_Lean_stringToMessageData(v_simpName_1164_);
lean_inc_ref(v___y_1173_);
v___x_1175_ = l_Lean_stringToMessageData(v___y_1173_);
v___x_1176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1176_, 0, v___x_1174_);
lean_ctor_set(v___x_1176_, 1, v___x_1175_);
v___x_1177_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1, &lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__1);
v___x_1178_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1176_);
lean_ctor_set(v___x_1178_, 1, v___x_1177_);
v___x_1179_ = lean_array_to_list(v___y_1172_);
v___x_1180_ = lean_box(0);
v___x_1181_ = lp_batteries_List_mapTR_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__2(v___x_1179_, v___x_1180_);
v___x_1182_ = l_Lean_MessageData_ofList(v___x_1181_);
v___x_1183_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1178_);
lean_ctor_set(v___x_1183_, 1, v___x_1182_);
v___x_1184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1184_, 0, v___x_1183_);
return v___x_1184_;
}
v___jp_1187_:
{
size_t v_sz_1191_; size_t v___x_1192_; lean_object* v___x_1193_; 
v_sz_1191_ = lean_array_size(v___y_1190_);
v___x_1192_ = ((size_t)0ULL);
lean_inc_ref(v___y_1188_);
v___x_1193_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_formatLemmas_spec__1(v___y_1189_, v___y_1190_, v_sz_1191_, v___x_1192_, v___y_1188_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_);
lean_dec_ref(v___y_1190_);
if (lean_obj_tag(v___x_1193_) == 0)
{
lean_object* v_a_1194_; lean_object* v___x_1195_; uint8_t v___x_1196_; 
v_a_1194_ = lean_ctor_get(v___x_1193_, 0);
lean_inc(v_a_1194_);
lean_dec_ref_known(v___x_1193_, 1);
v___x_1195_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__2));
v___x_1196_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_formatLemmas_spec__0(v_higherOrder_1165_, v___x_1195_);
if (v___x_1196_ == 0)
{
lean_object* v___x_1197_; 
v___x_1197_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__3));
v___y_1172_ = v_a_1194_;
v___y_1173_ = v___x_1197_;
goto v___jp_1171_;
}
else
{
lean_object* v___x_1198_; 
v___x_1198_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__4));
v___y_1172_ = v_a_1194_;
v___y_1173_ = v___x_1198_;
goto v___jp_1171_;
}
}
else
{
lean_object* v_a_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1206_; 
lean_dec_ref(v_simpName_1164_);
v_a_1199_ = lean_ctor_get(v___x_1193_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1193_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1201_ = v___x_1193_;
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_a_1199_);
lean_dec(v___x_1193_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1204_; 
if (v_isShared_1202_ == 0)
{
v___x_1204_ = v___x_1201_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v_a_1199_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
}
v___jp_1207_:
{
lean_object* v___x_1214_; 
v___x_1214_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(v___y_1211_, v___y_1212_, v___y_1209_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec(v___y_1211_);
v___y_1188_ = v___y_1208_;
v___y_1189_ = v___y_1210_;
v___y_1190_ = v___x_1214_;
goto v___jp_1187_;
}
v___jp_1215_:
{
uint8_t v___x_1222_; 
v___x_1222_ = lean_nat_dec_le(v___y_1221_, v___y_1216_);
if (v___x_1222_ == 0)
{
lean_dec(v___y_1216_);
lean_inc(v___y_1221_);
v___y_1208_ = v___y_1217_;
v___y_1209_ = v___y_1221_;
v___y_1210_ = v___y_1218_;
v___y_1211_ = v___y_1219_;
v___y_1212_ = v___y_1220_;
v___y_1213_ = v___y_1221_;
goto v___jp_1207_;
}
else
{
v___y_1208_ = v___y_1217_;
v___y_1209_ = v___y_1221_;
v___y_1210_ = v___y_1218_;
v___y_1211_ = v___y_1219_;
v___y_1212_ = v___y_1220_;
v___y_1213_ = v___y_1216_;
goto v___jp_1207_;
}
}
v___jp_1223_:
{
lean_object* v___x_1225_; lean_object* v_env_1226_; lean_object* v_map_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1225_ = lean_st_ref_get(v_a_1169_);
v_env_1226_ = lean_ctor_get(v___x_1225_, 0);
lean_inc_ref(v_env_1226_);
lean_dec(v___x_1225_);
v_map_1227_ = lean_ctor_get(v_usedSimps_1163_, 0);
v___x_1228_ = lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg(v_map_1227_);
v___x_1229_ = lean_array_get_size(v___x_1228_);
v___x_1230_ = lean_unsigned_to_nat(0u);
v___x_1231_ = lean_nat_dec_eq(v___x_1229_, v___x_1230_);
if (v___x_1231_ == 0)
{
lean_object* v___x_1232_; lean_object* v___x_1233_; uint8_t v___x_1234_; 
v___x_1232_ = lean_unsigned_to_nat(1u);
v___x_1233_ = lean_nat_sub(v___x_1229_, v___x_1232_);
v___x_1234_ = lean_nat_dec_le(v___x_1230_, v___x_1233_);
if (v___x_1234_ == 0)
{
lean_inc(v___x_1233_);
v___y_1216_ = v___x_1233_;
v___y_1217_ = v___y_1224_;
v___y_1218_ = v_env_1226_;
v___y_1219_ = v___x_1229_;
v___y_1220_ = v___x_1228_;
v___y_1221_ = v___x_1233_;
goto v___jp_1215_;
}
else
{
v___y_1216_ = v___x_1233_;
v___y_1217_ = v___y_1224_;
v___y_1218_ = v_env_1226_;
v___y_1219_ = v___x_1229_;
v___y_1220_ = v___x_1228_;
v___y_1221_ = v___x_1230_;
goto v___jp_1215_;
}
}
else
{
v___y_1188_ = v___y_1224_;
v___y_1189_ = v_env_1226_;
v___y_1190_ = v___x_1228_;
goto v___jp_1187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLemmas___boxed(lean_object* v_usedSimps_1237_, lean_object* v_simpName_1238_, lean_object* v_higherOrder_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_, lean_object* v_a_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_batteries_Batteries_Tactic_Lint_formatLemmas(v_usedSimps_1237_, v_simpName_1238_, v_higherOrder_1239_, v_a_1240_, v_a_1241_, v_a_1242_, v_a_1243_);
lean_dec(v_a_1243_);
lean_dec_ref(v_a_1242_);
lean_dec(v_a_1241_);
lean_dec_ref(v_a_1240_);
lean_dec(v_higherOrder_1239_);
lean_dec_ref(v_usedSimps_1237_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3(lean_object* v_00_u03b2_1246_, lean_object* v_m_1247_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___redArg(v_m_1247_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3___boxed(lean_object* v_00_u03b2_1249_, lean_object* v_m_1250_){
_start:
{
lean_object* v_res_1251_; 
v_res_1251_ = lp_batteries_Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3(v_00_u03b2_1249_, v_m_1250_);
lean_dec_ref(v_m_1250_);
return v_res_1251_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4(lean_object* v_n_1252_, lean_object* v_as_1253_, lean_object* v_lo_1254_, lean_object* v_hi_1255_, lean_object* v_w_1256_, lean_object* v_hlo_1257_, lean_object* v_hhi_1258_){
_start:
{
lean_object* v___x_1259_; 
v___x_1259_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___redArg(v_n_1252_, v_as_1253_, v_lo_1254_, v_hi_1255_);
return v___x_1259_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4___boxed(lean_object* v_n_1260_, lean_object* v_as_1261_, lean_object* v_lo_1262_, lean_object* v_hi_1263_, lean_object* v_w_1264_, lean_object* v_hlo_1265_, lean_object* v_hhi_1266_){
_start:
{
lean_object* v_res_1267_; 
v_res_1267_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4(v_n_1260_, v_as_1261_, v_lo_1262_, v_hi_1263_, v_w_1264_, v_hlo_1265_, v_hhi_1266_);
lean_dec(v_hi_1263_);
lean_dec(v_n_1260_);
return v_res_1267_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3(lean_object* v_00_u03c3_1268_, lean_object* v_00_u03b2_1269_, lean_object* v_map_1270_, lean_object* v_f_1271_, lean_object* v_init_1272_){
_start:
{
lean_object* v___x_1273_; 
v___x_1273_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___redArg(v_map_1270_, v_f_1271_, v_init_1272_);
return v___x_1273_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3___boxed(lean_object* v_00_u03c3_1274_, lean_object* v_00_u03b2_1275_, lean_object* v_map_1276_, lean_object* v_f_1277_, lean_object* v_init_1278_){
_start:
{
lean_object* v_res_1279_; 
v_res_1279_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3(v_00_u03c3_1274_, v_00_u03b2_1275_, v_map_1276_, v_f_1277_, v_init_1278_);
lean_dec_ref(v_map_1276_);
return v_res_1279_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5(lean_object* v_n_1280_, lean_object* v_lo_1281_, lean_object* v_hi_1282_, lean_object* v_hhi_1283_, lean_object* v_pivot_1284_, lean_object* v_as_1285_, lean_object* v_i_1286_, lean_object* v_k_1287_, lean_object* v_ilo_1288_, lean_object* v_ik_1289_, lean_object* v_w_1290_){
_start:
{
lean_object* v___x_1291_; 
v___x_1291_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___redArg(v_hi_1282_, v_pivot_1284_, v_as_1285_, v_i_1286_, v_k_1287_);
return v___x_1291_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5___boxed(lean_object* v_n_1292_, lean_object* v_lo_1293_, lean_object* v_hi_1294_, lean_object* v_hhi_1295_, lean_object* v_pivot_1296_, lean_object* v_as_1297_, lean_object* v_i_1298_, lean_object* v_k_1299_, lean_object* v_ilo_1300_, lean_object* v_ik_1301_, lean_object* v_w_1302_){
_start:
{
lean_object* v_res_1303_; 
v_res_1303_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_formatLemmas_spec__4_spec__5(v_n_1292_, v_lo_1293_, v_hi_1294_, v_hhi_1295_, v_pivot_1296_, v_as_1297_, v_i_1298_, v_k_1299_, v_ilo_1300_, v_ik_1301_, v_w_1302_);
lean_dec_ref(v_pivot_1296_);
lean_dec(v_hi_1294_);
lean_dec(v_lo_1293_);
lean_dec(v_n_1292_);
return v_res_1303_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___redArg(lean_object* v_map_1304_, lean_object* v_f_1305_, lean_object* v_init_1306_){
_start:
{
lean_object* v___x_1307_; 
v___x_1307_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_1305_, v_map_1304_, v_init_1306_);
return v___x_1307_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___redArg___boxed(lean_object* v_map_1308_, lean_object* v_f_1309_, lean_object* v_init_1310_){
_start:
{
lean_object* v_res_1311_; 
v_res_1311_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___redArg(v_map_1308_, v_f_1309_, v_init_1310_);
lean_dec_ref(v_map_1308_);
return v_res_1311_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4(lean_object* v_00_u03c3_1312_, lean_object* v_00_u03b2_1313_, lean_object* v_map_1314_, lean_object* v_f_1315_, lean_object* v_init_1316_){
_start:
{
lean_object* v___x_1317_; 
v___x_1317_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_Meta_DiscrTree_elements_spec__1_spec__3___redArg(v_f_1315_, v_map_1314_, v_init_1316_);
return v___x_1317_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4___boxed(lean_object* v_00_u03c3_1318_, lean_object* v_00_u03b2_1319_, lean_object* v_map_1320_, lean_object* v_f_1321_, lean_object* v_init_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toArray___at___00Batteries_Tactic_Lint_formatLemmas_spec__3_spec__3_spec__4(v_00_u03c3_1318_, v_00_u03b2_1319_, v_map_1320_, v_f_1321_, v_init_1322_);
lean_dec_ref(v_map_1320_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0(lean_object* v_k_1324_, lean_object* v_b_1325_, lean_object* v_c_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_){
_start:
{
lean_object* v___x_1332_; 
lean_inc(v___y_1330_);
lean_inc_ref(v___y_1329_);
lean_inc(v___y_1328_);
lean_inc_ref(v___y_1327_);
v___x_1332_ = lean_apply_7(v_k_1324_, v_b_1325_, v_c_1326_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, lean_box(0));
return v___x_1332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0___boxed(lean_object* v_k_1333_, lean_object* v_b_1334_, lean_object* v_c_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_){
_start:
{
lean_object* v_res_1341_; 
v_res_1341_ = lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0(v_k_1333_, v_b_1334_, v_c_1335_, v___y_1336_, v___y_1337_, v___y_1338_, v___y_1339_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
lean_dec(v___y_1337_);
lean_dec_ref(v___y_1336_);
return v_res_1341_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg(lean_object* v_type_1342_, lean_object* v_k_1343_, uint8_t v_cleanupAnnotations_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_){
_start:
{
lean_object* v___f_1350_; uint8_t v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; 
v___f_1350_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1350_, 0, v_k_1343_);
v___x_1351_ = 0;
v___x_1352_ = lean_box(0);
v___x_1353_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_1351_, v___x_1352_, v_type_1342_, v___f_1350_, v_cleanupAnnotations_1344_, v___x_1351_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_);
if (lean_obj_tag(v___x_1353_) == 0)
{
lean_object* v_a_1354_; lean_object* v___x_1356_; uint8_t v_isShared_1357_; uint8_t v_isSharedCheck_1361_; 
v_a_1354_ = lean_ctor_get(v___x_1353_, 0);
v_isSharedCheck_1361_ = !lean_is_exclusive(v___x_1353_);
if (v_isSharedCheck_1361_ == 0)
{
v___x_1356_ = v___x_1353_;
v_isShared_1357_ = v_isSharedCheck_1361_;
goto v_resetjp_1355_;
}
else
{
lean_inc(v_a_1354_);
lean_dec(v___x_1353_);
v___x_1356_ = lean_box(0);
v_isShared_1357_ = v_isSharedCheck_1361_;
goto v_resetjp_1355_;
}
v_resetjp_1355_:
{
lean_object* v___x_1359_; 
if (v_isShared_1357_ == 0)
{
v___x_1359_ = v___x_1356_;
goto v_reusejp_1358_;
}
else
{
lean_object* v_reuseFailAlloc_1360_; 
v_reuseFailAlloc_1360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1360_, 0, v_a_1354_);
v___x_1359_ = v_reuseFailAlloc_1360_;
goto v_reusejp_1358_;
}
v_reusejp_1358_:
{
return v___x_1359_;
}
}
}
else
{
lean_object* v_a_1362_; lean_object* v___x_1364_; uint8_t v_isShared_1365_; uint8_t v_isSharedCheck_1369_; 
v_a_1362_ = lean_ctor_get(v___x_1353_, 0);
v_isSharedCheck_1369_ = !lean_is_exclusive(v___x_1353_);
if (v_isSharedCheck_1369_ == 0)
{
v___x_1364_ = v___x_1353_;
v_isShared_1365_ = v_isSharedCheck_1369_;
goto v_resetjp_1363_;
}
else
{
lean_inc(v_a_1362_);
lean_dec(v___x_1353_);
v___x_1364_ = lean_box(0);
v_isShared_1365_ = v_isSharedCheck_1369_;
goto v_resetjp_1363_;
}
v_resetjp_1363_:
{
lean_object* v___x_1367_; 
if (v_isShared_1365_ == 0)
{
v___x_1367_ = v___x_1364_;
goto v_reusejp_1366_;
}
else
{
lean_object* v_reuseFailAlloc_1368_; 
v_reuseFailAlloc_1368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1368_, 0, v_a_1362_);
v___x_1367_ = v_reuseFailAlloc_1368_;
goto v_reusejp_1366_;
}
v_reusejp_1366_:
{
return v___x_1367_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg___boxed(lean_object* v_type_1370_, lean_object* v_k_1371_, lean_object* v_cleanupAnnotations_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1378_; lean_object* v_res_1379_; 
v_cleanupAnnotations_boxed_1378_ = lean_unbox(v_cleanupAnnotations_1372_);
v_res_1379_ = lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg(v_type_1370_, v_k_1371_, v_cleanupAnnotations_boxed_1378_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
lean_dec(v___y_1376_);
lean_dec_ref(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec_ref(v___y_1373_);
return v_res_1379_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1(lean_object* v_00_u03b1_1380_, lean_object* v_type_1381_, lean_object* v_k_1382_, uint8_t v_cleanupAnnotations_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_){
_start:
{
lean_object* v___x_1389_; 
v___x_1389_ = lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg(v_type_1381_, v_k_1382_, v_cleanupAnnotations_1383_, v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_);
return v___x_1389_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___boxed(lean_object* v_00_u03b1_1390_, lean_object* v_type_1391_, lean_object* v_k_1392_, lean_object* v_cleanupAnnotations_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1399_; lean_object* v_res_1400_; 
v_cleanupAnnotations_boxed_1399_ = lean_unbox(v_cleanupAnnotations_1393_);
v_res_1400_ = lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1(v_00_u03b1_1390_, v_type_1391_, v_k_1392_, v_cleanupAnnotations_boxed_1399_, v___y_1394_, v___y_1395_, v___y_1396_, v___y_1397_);
lean_dec(v___y_1397_);
lean_dec_ref(v___y_1396_);
lean_dec(v___y_1395_);
lean_dec_ref(v___y_1394_);
return v_res_1400_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0(uint8_t v_a_1403_, uint8_t v_a_1404_, lean_object* v_e_1405_, lean_object* v_ctx_1406_, lean_object* v_stats_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_){
_start:
{
if (v_a_1403_ == 0)
{
lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; 
v___x_1413_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___closed__0));
v___x_1414_ = lean_box(0);
v___x_1415_ = l_Lean_Meta_simp(v_e_1405_, v_ctx_1406_, v___x_1413_, v___x_1414_, v_stats_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_);
lean_dec_ref(v_stats_1407_);
return v___x_1415_;
}
else
{
lean_object* v___x_1416_; lean_object* v___x_1417_; 
v___x_1416_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___closed__0));
v___x_1417_ = l_Lean_Meta_dsimp(v_e_1405_, v_ctx_1406_, v___x_1416_, v_stats_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_);
if (lean_obj_tag(v___x_1417_) == 0)
{
lean_object* v_a_1418_; lean_object* v___x_1420_; uint8_t v_isShared_1421_; uint8_t v_isSharedCheck_1436_; 
v_a_1418_ = lean_ctor_get(v___x_1417_, 0);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1417_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1420_ = v___x_1417_;
v_isShared_1421_ = v_isSharedCheck_1436_;
goto v_resetjp_1419_;
}
else
{
lean_inc(v_a_1418_);
lean_dec(v___x_1417_);
v___x_1420_ = lean_box(0);
v_isShared_1421_ = v_isSharedCheck_1436_;
goto v_resetjp_1419_;
}
v_resetjp_1419_:
{
lean_object* v_fst_1422_; lean_object* v_snd_1423_; lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1435_; 
v_fst_1422_ = lean_ctor_get(v_a_1418_, 0);
v_snd_1423_ = lean_ctor_get(v_a_1418_, 1);
v_isSharedCheck_1435_ = !lean_is_exclusive(v_a_1418_);
if (v_isSharedCheck_1435_ == 0)
{
v___x_1425_ = v_a_1418_;
v_isShared_1426_ = v_isSharedCheck_1435_;
goto v_resetjp_1424_;
}
else
{
lean_inc(v_snd_1423_);
lean_inc(v_fst_1422_);
lean_dec(v_a_1418_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1435_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1430_; 
v___x_1427_ = lean_box(0);
v___x_1428_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1428_, 0, v_fst_1422_);
lean_ctor_set(v___x_1428_, 1, v___x_1427_);
lean_ctor_set_uint8(v___x_1428_, sizeof(void*)*2, v_a_1404_);
if (v_isShared_1426_ == 0)
{
lean_ctor_set(v___x_1425_, 0, v___x_1428_);
v___x_1430_ = v___x_1425_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v___x_1428_);
lean_ctor_set(v_reuseFailAlloc_1434_, 1, v_snd_1423_);
v___x_1430_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
lean_object* v___x_1432_; 
if (v_isShared_1421_ == 0)
{
lean_ctor_set(v___x_1420_, 0, v___x_1430_);
v___x_1432_ = v___x_1420_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1433_; 
v_reuseFailAlloc_1433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1433_, 0, v___x_1430_);
v___x_1432_ = v_reuseFailAlloc_1433_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
return v___x_1432_;
}
}
}
}
}
else
{
lean_object* v_a_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1444_; 
v_a_1437_ = lean_ctor_get(v___x_1417_, 0);
v_isSharedCheck_1444_ = !lean_is_exclusive(v___x_1417_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1439_ = v___x_1417_;
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
else
{
lean_inc(v_a_1437_);
lean_dec(v___x_1417_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1442_; 
if (v_isShared_1440_ == 0)
{
v___x_1442_ = v___x_1439_;
goto v_reusejp_1441_;
}
else
{
lean_object* v_reuseFailAlloc_1443_; 
v_reuseFailAlloc_1443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1443_, 0, v_a_1437_);
v___x_1442_ = v_reuseFailAlloc_1443_;
goto v_reusejp_1441_;
}
v_reusejp_1441_:
{
return v___x_1442_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___boxed(lean_object* v_a_1445_, lean_object* v_a_1446_, lean_object* v_e_1447_, lean_object* v_ctx_1448_, lean_object* v_stats_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_){
_start:
{
uint8_t v_a_27596__boxed_1455_; uint8_t v_a_27597__boxed_1456_; lean_object* v_res_1457_; 
v_a_27596__boxed_1455_ = lean_unbox(v_a_1445_);
v_a_27597__boxed_1456_ = lean_unbox(v_a_1446_);
v_res_1457_ = lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0(v_a_27596__boxed_1455_, v_a_27597__boxed_1456_, v_e_1447_, v_ctx_1448_, v_stats_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
lean_dec(v___y_1451_);
lean_dec_ref(v___y_1450_);
return v_res_1457_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0(uint8_t v_a_1458_, lean_object* v___x_1459_, lean_object* v___x_1460_, lean_object* v_a_1461_, lean_object* v___x_1462_, uint8_t v_a_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_){
_start:
{
if (v_a_1458_ == 0)
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1469_ = lean_mk_empty_array_with_capacity(v___x_1459_);
v___x_1470_ = lean_box(0);
v___x_1471_ = l_Lean_Meta_simp(v___x_1460_, v_a_1461_, v___x_1469_, v___x_1470_, v___x_1462_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_);
lean_dec_ref(v___x_1462_);
return v___x_1471_;
}
else
{
lean_object* v___x_1472_; lean_object* v___x_1473_; 
v___x_1472_ = lean_mk_empty_array_with_capacity(v___x_1459_);
v___x_1473_ = l_Lean_Meta_dsimp(v___x_1460_, v_a_1461_, v___x_1472_, v___x_1462_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_);
if (lean_obj_tag(v___x_1473_) == 0)
{
lean_object* v_a_1474_; lean_object* v___x_1476_; uint8_t v_isShared_1477_; uint8_t v_isSharedCheck_1492_; 
v_a_1474_ = lean_ctor_get(v___x_1473_, 0);
v_isSharedCheck_1492_ = !lean_is_exclusive(v___x_1473_);
if (v_isSharedCheck_1492_ == 0)
{
v___x_1476_ = v___x_1473_;
v_isShared_1477_ = v_isSharedCheck_1492_;
goto v_resetjp_1475_;
}
else
{
lean_inc(v_a_1474_);
lean_dec(v___x_1473_);
v___x_1476_ = lean_box(0);
v_isShared_1477_ = v_isSharedCheck_1492_;
goto v_resetjp_1475_;
}
v_resetjp_1475_:
{
lean_object* v_fst_1478_; lean_object* v_snd_1479_; lean_object* v___x_1481_; uint8_t v_isShared_1482_; uint8_t v_isSharedCheck_1491_; 
v_fst_1478_ = lean_ctor_get(v_a_1474_, 0);
v_snd_1479_ = lean_ctor_get(v_a_1474_, 1);
v_isSharedCheck_1491_ = !lean_is_exclusive(v_a_1474_);
if (v_isSharedCheck_1491_ == 0)
{
v___x_1481_ = v_a_1474_;
v_isShared_1482_ = v_isSharedCheck_1491_;
goto v_resetjp_1480_;
}
else
{
lean_inc(v_snd_1479_);
lean_inc(v_fst_1478_);
lean_dec(v_a_1474_);
v___x_1481_ = lean_box(0);
v_isShared_1482_ = v_isSharedCheck_1491_;
goto v_resetjp_1480_;
}
v_resetjp_1480_:
{
lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1486_; 
v___x_1483_ = lean_box(0);
v___x_1484_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1484_, 0, v_fst_1478_);
lean_ctor_set(v___x_1484_, 1, v___x_1483_);
lean_ctor_set_uint8(v___x_1484_, sizeof(void*)*2, v_a_1463_);
if (v_isShared_1482_ == 0)
{
lean_ctor_set(v___x_1481_, 0, v___x_1484_);
v___x_1486_ = v___x_1481_;
goto v_reusejp_1485_;
}
else
{
lean_object* v_reuseFailAlloc_1490_; 
v_reuseFailAlloc_1490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1490_, 0, v___x_1484_);
lean_ctor_set(v_reuseFailAlloc_1490_, 1, v_snd_1479_);
v___x_1486_ = v_reuseFailAlloc_1490_;
goto v_reusejp_1485_;
}
v_reusejp_1485_:
{
lean_object* v___x_1488_; 
if (v_isShared_1477_ == 0)
{
lean_ctor_set(v___x_1476_, 0, v___x_1486_);
v___x_1488_ = v___x_1476_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v___x_1486_);
v___x_1488_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
return v___x_1488_;
}
}
}
}
}
else
{
lean_object* v_a_1493_; lean_object* v___x_1495_; uint8_t v_isShared_1496_; uint8_t v_isSharedCheck_1500_; 
v_a_1493_ = lean_ctor_get(v___x_1473_, 0);
v_isSharedCheck_1500_ = !lean_is_exclusive(v___x_1473_);
if (v_isSharedCheck_1500_ == 0)
{
v___x_1495_ = v___x_1473_;
v_isShared_1496_ = v_isSharedCheck_1500_;
goto v_resetjp_1494_;
}
else
{
lean_inc(v_a_1493_);
lean_dec(v___x_1473_);
v___x_1495_ = lean_box(0);
v_isShared_1496_ = v_isSharedCheck_1500_;
goto v_resetjp_1494_;
}
v_resetjp_1494_:
{
lean_object* v___x_1498_; 
if (v_isShared_1496_ == 0)
{
v___x_1498_ = v___x_1495_;
goto v_reusejp_1497_;
}
else
{
lean_object* v_reuseFailAlloc_1499_; 
v_reuseFailAlloc_1499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1499_, 0, v_a_1493_);
v___x_1498_ = v_reuseFailAlloc_1499_;
goto v_reusejp_1497_;
}
v_reusejp_1497_:
{
return v___x_1498_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0___boxed(lean_object* v_a_1501_, lean_object* v___x_1502_, lean_object* v___x_1503_, lean_object* v_a_1504_, lean_object* v___x_1505_, lean_object* v_a_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_){
_start:
{
uint8_t v_a_27683__boxed_1512_; uint8_t v_a_27688__boxed_1513_; lean_object* v_res_1514_; 
v_a_27683__boxed_1512_ = lean_unbox(v_a_1501_);
v_a_27688__boxed_1513_ = lean_unbox(v_a_1506_);
v_res_1514_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0(v_a_27683__boxed_1512_, v___x_1502_, v___x_1503_, v_a_1504_, v___x_1505_, v_a_27688__boxed_1513_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
lean_dec(v___y_1510_);
lean_dec_ref(v___y_1509_);
lean_dec(v___y_1508_);
lean_dec_ref(v___y_1507_);
lean_dec(v___x_1502_);
return v_res_1514_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1515_; 
v___x_1515_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1515_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1516_; lean_object* v___x_1517_; 
v___x_1516_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__0);
v___x_1517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1517_, 0, v___x_1516_);
return v___x_1517_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2(void){
_start:
{
lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; 
v___x_1518_ = lean_unsigned_to_nat(0u);
v___x_1519_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1);
v___x_1520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1520_, 0, v___x_1519_);
lean_ctor_set(v___x_1520_, 1, v___x_1518_);
return v___x_1520_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v___x_1521_ = lean_unsigned_to_nat(32u);
v___x_1522_ = lean_mk_empty_array_with_capacity(v___x_1521_);
v___x_1523_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1523_, 0, v___x_1522_);
return v___x_1523_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4(void){
_start:
{
size_t v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; 
v___x_1524_ = ((size_t)5ULL);
v___x_1525_ = lean_unsigned_to_nat(0u);
v___x_1526_ = lean_unsigned_to_nat(32u);
v___x_1527_ = lean_mk_empty_array_with_capacity(v___x_1526_);
v___x_1528_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__3);
v___x_1529_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1529_, 0, v___x_1528_);
lean_ctor_set(v___x_1529_, 1, v___x_1527_);
lean_ctor_set(v___x_1529_, 2, v___x_1525_);
lean_ctor_set(v___x_1529_, 3, v___x_1525_);
lean_ctor_set_usize(v___x_1529_, 4, v___x_1524_);
return v___x_1529_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5(void){
_start:
{
lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; 
v___x_1530_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__4);
v___x_1531_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__1);
v___x_1532_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1532_, 0, v___x_1531_);
lean_ctor_set(v___x_1532_, 1, v___x_1531_);
lean_ctor_set(v___x_1532_, 2, v___x_1531_);
lean_ctor_set(v___x_1532_, 3, v___x_1530_);
return v___x_1532_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6(void){
_start:
{
lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; 
v___x_1533_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__5);
v___x_1534_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__2);
v___x_1535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1535_, 0, v___x_1534_);
lean_ctor_set(v___x_1535_, 1, v___x_1533_);
return v___x_1535_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8(void){
_start:
{
lean_object* v___x_1537_; lean_object* v___x_1538_; 
v___x_1537_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__7));
v___x_1538_ = l_Lean_stringToMessageData(v___x_1537_);
return v___x_1538_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10(void){
_start:
{
lean_object* v___x_1540_; lean_object* v___x_1541_; 
v___x_1540_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__9));
v___x_1541_ = l_Lean_stringToMessageData(v___x_1540_);
return v___x_1541_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12(void){
_start:
{
lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___x_1543_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__11));
v___x_1544_ = l_Lean_stringToMessageData(v___x_1543_);
return v___x_1544_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14(void){
_start:
{
lean_object* v___x_1546_; lean_object* v___x_1547_; 
v___x_1546_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__13));
v___x_1547_ = l_Lean_stringToMessageData(v___x_1546_);
return v___x_1547_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16(void){
_start:
{
lean_object* v___x_1549_; lean_object* v___x_1550_; 
v___x_1549_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__15));
v___x_1550_ = l_Lean_stringToMessageData(v___x_1549_);
return v___x_1550_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18(void){
_start:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; 
v___x_1552_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__17));
v___x_1553_ = l_Lean_stringToMessageData(v___x_1552_);
return v___x_1553_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20(void){
_start:
{
lean_object* v___x_1555_; lean_object* v___x_1556_; 
v___x_1555_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__19));
v___x_1556_ = l_Lean_stringToMessageData(v___x_1555_);
return v___x_1556_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22(void){
_start:
{
lean_object* v___x_1558_; lean_object* v___x_1559_; 
v___x_1558_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__21));
v___x_1559_ = l_Lean_stringToMessageData(v___x_1558_);
return v___x_1559_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24(void){
_start:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; 
v___x_1561_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__23));
v___x_1562_ = l_Lean_stringToMessageData(v___x_1561_);
return v___x_1562_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26(void){
_start:
{
lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1564_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__25));
v___x_1565_ = l_Lean_stringToMessageData(v___x_1564_);
return v___x_1565_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3(uint8_t v___x_1566_, lean_object* v_lhs_1567_, lean_object* v_a_1568_, uint8_t v_a_1569_, uint8_t v_a_1570_, lean_object* v___y_1571_, lean_object* v_as_1572_, size_t v_sz_1573_, size_t v_i_1574_, lean_object* v_b_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_){
_start:
{
lean_object* v_a_1582_; lean_object* v___x_1586_; uint8_t v___x_1587_; 
v___x_1586_ = lean_unsigned_to_nat(0u);
v___x_1587_ = lean_usize_dec_lt(v_i_1574_, v_sz_1573_);
if (v___x_1587_ == 0)
{
lean_object* v___x_1588_; 
lean_dec_ref(v___y_1571_);
v___x_1588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1588_, 0, v_b_1575_);
return v___x_1588_;
}
else
{
lean_object* v_a_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v_a_1589_ = lean_array_uget_borrowed(v_as_1572_, v_i_1574_);
v___x_1590_ = l_Lean_Expr_fvarId_x21(v_a_1589_);
lean_inc(v___x_1590_);
v___x_1591_ = l_Lean_FVarId_getDecl___redArg(v___x_1590_, v___y_1576_, v___y_1578_, v___y_1579_);
if (lean_obj_tag(v___x_1591_) == 0)
{
lean_object* v_a_1592_; lean_object* v___x_1593_; lean_object* v_name_1595_; lean_object* v___y_1596_; lean_object* v___y_1597_; lean_object* v___y_1598_; lean_object* v___y_1599_; lean_object* v___x_1726_; uint8_t v___x_1727_; 
v_a_1592_ = lean_ctor_get(v___x_1591_, 0);
lean_inc(v_a_1592_);
lean_dec_ref_known(v___x_1591_, 1);
v___x_1593_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6);
v___x_1726_ = l_Lean_LocalDecl_userName(v_a_1592_);
v___x_1727_ = l_Lean_Name_hasMacroScopes(v___x_1726_);
if (v___x_1727_ == 0)
{
v_name_1595_ = v___x_1726_;
v___y_1596_ = v___y_1576_;
v___y_1597_ = v___y_1577_;
v___y_1598_ = v___y_1578_;
v___y_1599_ = v___y_1579_;
goto v___jp_1594_;
}
else
{
lean_object* v_options_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v_fst_1732_; 
v_options_1728_ = lean_ctor_get(v___y_1578_, 2);
v___x_1729_ = lean_box(1);
lean_inc_ref(v_options_1728_);
v___x_1730_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1730_, 0, v_options_1728_);
lean_ctor_set(v___x_1730_, 1, v___x_1729_);
lean_ctor_set(v___x_1730_, 2, v___x_1729_);
v___x_1731_ = l_Lean_sanitizeName(v___x_1726_, v___x_1730_);
v_fst_1732_ = lean_ctor_get(v___x_1731_, 0);
lean_inc(v_fst_1732_);
lean_dec_ref(v___x_1731_);
v_name_1595_ = v_fst_1732_;
v___y_1596_ = v___y_1576_;
v___y_1597_ = v___y_1577_;
v___y_1598_ = v___y_1578_;
v___y_1599_ = v___y_1579_;
goto v___jp_1594_;
}
v___jp_1594_:
{
lean_object* v___x_1600_; lean_object* v___x_1601_; 
v___x_1600_ = l_Lean_LocalDecl_type(v_a_1592_);
lean_inc_ref(v___x_1600_);
v___x_1601_ = l_Lean_Meta_isProp(v___x_1600_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
if (lean_obj_tag(v___x_1601_) == 0)
{
lean_object* v_a_1602_; uint8_t v___x_1603_; 
v_a_1602_ = lean_ctor_get(v___x_1601_, 0);
lean_inc(v_a_1602_);
lean_dec_ref_known(v___x_1601_, 1);
v___x_1603_ = lean_unbox(v_a_1602_);
lean_dec(v_a_1602_);
if (v___x_1603_ == 0)
{
uint8_t v___x_1604_; uint8_t v___x_1605_; 
v___x_1604_ = l_Lean_LocalDecl_binderInfo(v_a_1592_);
lean_dec(v_a_1592_);
v___x_1605_ = l_Lean_BinderInfo_isInstImplicit(v___x_1604_);
if (v___x_1605_ == 0)
{
if (v___x_1566_ == 0)
{
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
lean_dec(v___x_1590_);
v_a_1582_ = v_b_1575_;
goto v___jp_1581_;
}
else
{
uint8_t v___x_1606_; 
v___x_1606_ = l_Lean_Expr_containsFVar(v_lhs_1567_, v___x_1590_);
if (v___x_1606_ == 0)
{
uint8_t v___x_1607_; 
v___x_1607_ = l_Lean_Expr_containsFVar(v_a_1568_, v___x_1590_);
lean_dec(v___x_1590_);
if (v___x_1607_ == 0)
{
lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v___x_1608_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__8);
v___x_1609_ = l_Lean_MessageData_ofName(v_name_1595_);
v___x_1610_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1610_, 0, v___x_1608_);
lean_ctor_set(v___x_1610_, 1, v___x_1609_);
v___x_1611_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10);
v___x_1612_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1610_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
v___x_1613_ = l_Lean_MessageData_ofExpr(v___x_1600_);
v___x_1614_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1614_, 0, v___x_1612_);
lean_ctor_set(v___x_1614_, 1, v___x_1613_);
v___x_1615_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__12);
v___x_1616_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1616_, 0, v___x_1614_);
lean_ctor_set(v___x_1616_, 1, v___x_1615_);
v___x_1617_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1617_, 0, v_b_1575_);
lean_ctor_set(v___x_1617_, 1, v___x_1616_);
v_a_1582_ = v___x_1617_;
goto v___jp_1581_;
}
else
{
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
v_a_1582_ = v_b_1575_;
goto v___jp_1581_;
}
}
else
{
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
lean_dec(v___x_1590_);
v_a_1582_ = v_b_1575_;
goto v___jp_1581_;
}
}
}
else
{
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
lean_dec(v___x_1590_);
v_a_1582_ = v_b_1575_;
goto v___jp_1581_;
}
}
else
{
lean_object* v___x_1618_; 
lean_dec(v_a_1592_);
lean_dec(v___x_1590_);
v___x_1618_ = l_Lean_Meta_Simp_Context_mkDefault___redArg(v___y_1596_, v___y_1598_, v___y_1599_);
if (lean_obj_tag(v___x_1618_) == 0)
{
lean_object* v_a_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___f_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; 
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
lean_inc(v_a_1619_);
lean_dec_ref_known(v___x_1618_, 1);
v___x_1620_ = lean_box(v_a_1569_);
v___x_1621_ = lean_box(v_a_1570_);
lean_inc_ref_n(v___x_1600_, 2);
v___f_1622_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___lam__0___boxed), 11, 6);
lean_closure_set(v___f_1622_, 0, v___x_1620_);
lean_closure_set(v___f_1622_, 1, v___x_1586_);
lean_closure_set(v___f_1622_, 2, v___x_1600_);
lean_closure_set(v___f_1622_, 3, v_a_1619_);
lean_closure_set(v___f_1622_, 4, v___x_1593_);
lean_closure_set(v___f_1622_, 5, v___x_1621_);
v___x_1623_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__14);
v___x_1624_ = l_Lean_MessageData_ofName(v_name_1595_);
lean_inc_ref(v___x_1624_);
v___x_1625_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1625_, 0, v___x_1623_);
lean_ctor_set(v___x_1625_, 1, v___x_1624_);
v___x_1626_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__10);
v___x_1627_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1627_, 0, v___x_1625_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = l_Lean_MessageData_ofExpr(v___x_1600_);
v___x_1629_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1629_, 0, v___x_1627_);
lean_ctor_set(v___x_1629_, 1, v___x_1628_);
v___x_1630_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__16);
v___x_1631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1629_);
lean_ctor_set(v___x_1631_, 1, v___x_1630_);
v___x_1632_ = lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(v___x_1631_, v___f_1622_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
if (lean_obj_tag(v___x_1632_) == 0)
{
lean_object* v_a_1633_; lean_object* v_fst_1634_; lean_object* v_snd_1635_; lean_object* v___x_1637_; uint8_t v_isShared_1638_; uint8_t v_isSharedCheck_1701_; 
v_a_1633_ = lean_ctor_get(v___x_1632_, 0);
lean_inc(v_a_1633_);
lean_dec_ref_known(v___x_1632_, 1);
v_fst_1634_ = lean_ctor_get(v_a_1633_, 0);
v_snd_1635_ = lean_ctor_get(v_a_1633_, 1);
v_isSharedCheck_1701_ = !lean_is_exclusive(v_a_1633_);
if (v_isSharedCheck_1701_ == 0)
{
v___x_1637_ = v_a_1633_;
v_isShared_1638_ = v_isSharedCheck_1701_;
goto v_resetjp_1636_;
}
else
{
lean_inc(v_snd_1635_);
lean_inc(v_fst_1634_);
lean_dec(v_a_1633_);
v___x_1637_ = lean_box(0);
v_isShared_1638_ = v_isSharedCheck_1701_;
goto v_resetjp_1636_;
}
v_resetjp_1636_:
{
lean_object* v_expr_1639_; lean_object* v___x_1640_; 
v_expr_1639_ = lean_ctor_get(v_fst_1634_, 0);
lean_inc_ref_n(v_expr_1639_, 2);
lean_dec(v_fst_1634_);
lean_inc_ref(v___x_1600_);
v___x_1640_ = lp_batteries_Batteries_Tactic_Lint_isSimpEq(v_expr_1639_, v___x_1600_, v_a_1570_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
if (lean_obj_tag(v___x_1640_) == 0)
{
lean_object* v_a_1641_; uint8_t v___x_1642_; 
v_a_1641_ = lean_ctor_get(v___x_1640_, 0);
lean_inc(v_a_1641_);
lean_dec_ref_known(v___x_1640_, 1);
v___x_1642_ = lean_unbox(v_a_1641_);
lean_dec(v_a_1641_);
if (v___x_1642_ == 0)
{
lean_object* v___x_1643_; 
v___x_1643_ = l_Lean_Meta_addPPExplicitToExposeDiff(v_expr_1639_, v___x_1600_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
if (lean_obj_tag(v___x_1643_) == 0)
{
lean_object* v_a_1644_; lean_object* v_fst_1645_; lean_object* v_snd_1646_; lean_object* v___x_1648_; uint8_t v_isShared_1649_; uint8_t v_isSharedCheck_1684_; 
v_a_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1643_, 1);
v_fst_1645_ = lean_ctor_get(v_a_1644_, 0);
v_snd_1646_ = lean_ctor_get(v_a_1644_, 1);
v_isSharedCheck_1684_ = !lean_is_exclusive(v_a_1644_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1648_ = v_a_1644_;
v_isShared_1649_ = v_isSharedCheck_1684_;
goto v_resetjp_1647_;
}
else
{
lean_inc(v_snd_1646_);
lean_inc(v_fst_1645_);
lean_dec(v_a_1644_);
v___x_1648_ = lean_box(0);
v_isShared_1649_ = v_isSharedCheck_1684_;
goto v_resetjp_1647_;
}
v_resetjp_1647_:
{
lean_object* v_usedTheorems_1650_; lean_object* v___x_1652_; uint8_t v_isShared_1653_; uint8_t v_isSharedCheck_1682_; 
v_usedTheorems_1650_ = lean_ctor_get(v_snd_1635_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v_snd_1635_);
if (v_isSharedCheck_1682_ == 0)
{
lean_object* v_unused_1683_; 
v_unused_1683_ = lean_ctor_get(v_snd_1635_, 1);
lean_dec(v_unused_1683_);
v___x_1652_ = v_snd_1635_;
v_isShared_1653_ = v_isSharedCheck_1682_;
goto v_resetjp_1651_;
}
else
{
lean_inc(v_usedTheorems_1650_);
lean_dec(v_snd_1635_);
v___x_1652_ = lean_box(0);
v_isShared_1653_ = v_isSharedCheck_1682_;
goto v_resetjp_1651_;
}
v_resetjp_1651_:
{
lean_object* v___x_1654_; lean_object* v___x_1655_; 
v___x_1654_ = lean_box(0);
lean_inc_ref(v___y_1571_);
v___x_1655_ = lp_batteries_Batteries_Tactic_Lint_formatLemmas(v_usedTheorems_1650_, v___y_1571_, v___x_1654_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
lean_dec_ref(v_usedTheorems_1650_);
if (lean_obj_tag(v___x_1655_) == 0)
{
lean_object* v_a_1656_; lean_object* v___x_1657_; lean_object* v___x_1659_; 
v_a_1656_ = lean_ctor_get(v___x_1655_, 0);
lean_inc(v_a_1656_);
lean_dec_ref_known(v___x_1655_, 1);
v___x_1657_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__18);
if (v_isShared_1653_ == 0)
{
lean_ctor_set_tag(v___x_1652_, 7);
lean_ctor_set(v___x_1652_, 1, v___x_1624_);
lean_ctor_set(v___x_1652_, 0, v___x_1657_);
v___x_1659_ = v___x_1652_;
goto v_reusejp_1658_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v___x_1657_);
lean_ctor_set(v_reuseFailAlloc_1681_, 1, v___x_1624_);
v___x_1659_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1658_;
}
v_reusejp_1658_:
{
lean_object* v___x_1660_; lean_object* v___x_1662_; 
v___x_1660_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__20);
if (v_isShared_1649_ == 0)
{
lean_ctor_set_tag(v___x_1648_, 7);
lean_ctor_set(v___x_1648_, 1, v___x_1660_);
lean_ctor_set(v___x_1648_, 0, v___x_1659_);
v___x_1662_ = v___x_1648_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v___x_1659_);
lean_ctor_set(v_reuseFailAlloc_1680_, 1, v___x_1660_);
v___x_1662_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1666_; 
v___x_1663_ = l_Lean_MessageData_ofExpr(v_snd_1646_);
v___x_1664_ = l_Lean_indentD(v___x_1663_);
if (v_isShared_1638_ == 0)
{
lean_ctor_set_tag(v___x_1637_, 7);
lean_ctor_set(v___x_1637_, 1, v___x_1664_);
lean_ctor_set(v___x_1637_, 0, v___x_1662_);
v___x_1666_ = v___x_1637_;
goto v_reusejp_1665_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v___x_1662_);
lean_ctor_set(v_reuseFailAlloc_1679_, 1, v___x_1664_);
v___x_1666_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1665_;
}
v_reusejp_1665_:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; 
v___x_1667_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22);
v___x_1668_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1668_, 0, v___x_1666_);
lean_ctor_set(v___x_1668_, 1, v___x_1667_);
v___x_1669_ = l_Lean_MessageData_ofExpr(v_fst_1645_);
v___x_1670_ = l_Lean_indentD(v___x_1669_);
v___x_1671_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1671_, 0, v___x_1668_);
lean_ctor_set(v___x_1671_, 1, v___x_1670_);
v___x_1672_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24);
v___x_1673_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1673_, 0, v___x_1671_);
lean_ctor_set(v___x_1673_, 1, v___x_1672_);
v___x_1674_ = l_Lean_indentD(v_a_1656_);
v___x_1675_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1675_, 0, v___x_1673_);
lean_ctor_set(v___x_1675_, 1, v___x_1674_);
v___x_1676_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__26);
v___x_1677_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1677_, 0, v___x_1675_);
lean_ctor_set(v___x_1677_, 1, v___x_1676_);
v___x_1678_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1678_, 0, v_b_1575_);
lean_ctor_set(v___x_1678_, 1, v___x_1677_);
v_a_1582_ = v___x_1678_;
goto v___jp_1581_;
}
}
}
}
else
{
lean_del_object(v___x_1652_);
lean_del_object(v___x_1648_);
lean_dec(v_snd_1646_);
lean_dec(v_fst_1645_);
lean_del_object(v___x_1637_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
return v___x_1655_;
}
}
}
}
else
{
lean_object* v_a_1685_; lean_object* v___x_1687_; uint8_t v_isShared_1688_; uint8_t v_isSharedCheck_1692_; 
lean_del_object(v___x_1637_);
lean_dec(v_snd_1635_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1685_ = lean_ctor_get(v___x_1643_, 0);
v_isSharedCheck_1692_ = !lean_is_exclusive(v___x_1643_);
if (v_isSharedCheck_1692_ == 0)
{
v___x_1687_ = v___x_1643_;
v_isShared_1688_ = v_isSharedCheck_1692_;
goto v_resetjp_1686_;
}
else
{
lean_inc(v_a_1685_);
lean_dec(v___x_1643_);
v___x_1687_ = lean_box(0);
v_isShared_1688_ = v_isSharedCheck_1692_;
goto v_resetjp_1686_;
}
v_resetjp_1686_:
{
lean_object* v___x_1690_; 
if (v_isShared_1688_ == 0)
{
v___x_1690_ = v___x_1687_;
goto v_reusejp_1689_;
}
else
{
lean_object* v_reuseFailAlloc_1691_; 
v_reuseFailAlloc_1691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1691_, 0, v_a_1685_);
v___x_1690_ = v_reuseFailAlloc_1691_;
goto v_reusejp_1689_;
}
v_reusejp_1689_:
{
return v___x_1690_;
}
}
}
}
else
{
lean_dec_ref(v_expr_1639_);
lean_del_object(v___x_1637_);
lean_dec(v_snd_1635_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1600_);
v_a_1582_ = v_b_1575_;
goto v___jp_1581_;
}
}
else
{
lean_object* v_a_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1700_; 
lean_dec_ref(v_expr_1639_);
lean_del_object(v___x_1637_);
lean_dec(v_snd_1635_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1600_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1693_ = lean_ctor_get(v___x_1640_, 0);
v_isSharedCheck_1700_ = !lean_is_exclusive(v___x_1640_);
if (v_isSharedCheck_1700_ == 0)
{
v___x_1695_ = v___x_1640_;
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_a_1693_);
lean_dec(v___x_1640_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v___x_1698_; 
if (v_isShared_1696_ == 0)
{
v___x_1698_ = v___x_1695_;
goto v_reusejp_1697_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v_a_1693_);
v___x_1698_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1697_;
}
v_reusejp_1697_:
{
return v___x_1698_;
}
}
}
}
}
else
{
lean_object* v_a_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1709_; 
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1600_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1702_ = lean_ctor_get(v___x_1632_, 0);
v_isSharedCheck_1709_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_1709_ == 0)
{
v___x_1704_ = v___x_1632_;
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_a_1702_);
lean_dec(v___x_1632_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1707_; 
if (v_isShared_1705_ == 0)
{
v___x_1707_ = v___x_1704_;
goto v_reusejp_1706_;
}
else
{
lean_object* v_reuseFailAlloc_1708_; 
v_reuseFailAlloc_1708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1708_, 0, v_a_1702_);
v___x_1707_ = v_reuseFailAlloc_1708_;
goto v_reusejp_1706_;
}
v_reusejp_1706_:
{
return v___x_1707_;
}
}
}
}
else
{
lean_object* v_a_1710_; lean_object* v___x_1712_; uint8_t v_isShared_1713_; uint8_t v_isSharedCheck_1717_; 
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1710_ = lean_ctor_get(v___x_1618_, 0);
v_isSharedCheck_1717_ = !lean_is_exclusive(v___x_1618_);
if (v_isSharedCheck_1717_ == 0)
{
v___x_1712_ = v___x_1618_;
v_isShared_1713_ = v_isSharedCheck_1717_;
goto v_resetjp_1711_;
}
else
{
lean_inc(v_a_1710_);
lean_dec(v___x_1618_);
v___x_1712_ = lean_box(0);
v_isShared_1713_ = v_isSharedCheck_1717_;
goto v_resetjp_1711_;
}
v_resetjp_1711_:
{
lean_object* v___x_1715_; 
if (v_isShared_1713_ == 0)
{
v___x_1715_ = v___x_1712_;
goto v_reusejp_1714_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v_a_1710_);
v___x_1715_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1714_;
}
v_reusejp_1714_:
{
return v___x_1715_;
}
}
}
}
}
else
{
lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1725_; 
lean_dec_ref(v___x_1600_);
lean_dec(v_name_1595_);
lean_dec(v_a_1592_);
lean_dec(v___x_1590_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1718_ = lean_ctor_get(v___x_1601_, 0);
v_isSharedCheck_1725_ = !lean_is_exclusive(v___x_1601_);
if (v_isSharedCheck_1725_ == 0)
{
v___x_1720_ = v___x_1601_;
v_isShared_1721_ = v_isSharedCheck_1725_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1601_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1725_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
lean_object* v___x_1723_; 
if (v_isShared_1721_ == 0)
{
v___x_1723_ = v___x_1720_;
goto v_reusejp_1722_;
}
else
{
lean_object* v_reuseFailAlloc_1724_; 
v_reuseFailAlloc_1724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1724_, 0, v_a_1718_);
v___x_1723_ = v_reuseFailAlloc_1724_;
goto v_reusejp_1722_;
}
v_reusejp_1722_:
{
return v___x_1723_;
}
}
}
}
}
else
{
lean_object* v_a_1733_; lean_object* v___x_1735_; uint8_t v_isShared_1736_; uint8_t v_isSharedCheck_1740_; 
lean_dec(v___x_1590_);
lean_dec_ref(v_b_1575_);
lean_dec_ref(v___y_1571_);
v_a_1733_ = lean_ctor_get(v___x_1591_, 0);
v_isSharedCheck_1740_ = !lean_is_exclusive(v___x_1591_);
if (v_isSharedCheck_1740_ == 0)
{
v___x_1735_ = v___x_1591_;
v_isShared_1736_ = v_isSharedCheck_1740_;
goto v_resetjp_1734_;
}
else
{
lean_inc(v_a_1733_);
lean_dec(v___x_1591_);
v___x_1735_ = lean_box(0);
v_isShared_1736_ = v_isSharedCheck_1740_;
goto v_resetjp_1734_;
}
v_resetjp_1734_:
{
lean_object* v___x_1738_; 
if (v_isShared_1736_ == 0)
{
v___x_1738_ = v___x_1735_;
goto v_reusejp_1737_;
}
else
{
lean_object* v_reuseFailAlloc_1739_; 
v_reuseFailAlloc_1739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1739_, 0, v_a_1733_);
v___x_1738_ = v_reuseFailAlloc_1739_;
goto v_reusejp_1737_;
}
v_reusejp_1737_:
{
return v___x_1738_;
}
}
}
}
v___jp_1581_:
{
size_t v___x_1583_; size_t v___x_1584_; 
v___x_1583_ = ((size_t)1ULL);
v___x_1584_ = lean_usize_add(v_i_1574_, v___x_1583_);
v_i_1574_ = v___x_1584_;
v_b_1575_ = v_a_1582_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___boxed(lean_object* v___x_1741_, lean_object* v_lhs_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v___y_1746_, lean_object* v_as_1747_, lean_object* v_sz_1748_, lean_object* v_i_1749_, lean_object* v_b_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_){
_start:
{
uint8_t v___x_27891__boxed_1756_; uint8_t v_a_27894__boxed_1757_; uint8_t v_a_27895__boxed_1758_; size_t v_sz_boxed_1759_; size_t v_i_boxed_1760_; lean_object* v_res_1761_; 
v___x_27891__boxed_1756_ = lean_unbox(v___x_1741_);
v_a_27894__boxed_1757_ = lean_unbox(v_a_1744_);
v_a_27895__boxed_1758_ = lean_unbox(v_a_1745_);
v_sz_boxed_1759_ = lean_unbox_usize(v_sz_1748_);
lean_dec(v_sz_1748_);
v_i_boxed_1760_ = lean_unbox_usize(v_i_1749_);
lean_dec(v_i_1749_);
v_res_1761_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3(v___x_27891__boxed_1756_, v_lhs_1742_, v_a_1743_, v_a_27894__boxed_1757_, v_a_27895__boxed_1758_, v___y_1746_, v_as_1747_, v_sz_boxed_1759_, v_i_boxed_1760_, v_b_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_);
lean_dec(v___y_1754_);
lean_dec_ref(v___y_1753_);
lean_dec(v___y_1752_);
lean_dec_ref(v___y_1751_);
lean_dec_ref(v_as_1747_);
lean_dec_ref(v_a_1743_);
lean_dec_ref(v_lhs_1742_);
return v_res_1761_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0(lean_object* v_as_1762_, size_t v_i_1763_, size_t v_stop_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_){
_start:
{
uint8_t v___x_1770_; 
v___x_1770_ = lean_usize_dec_eq(v_i_1763_, v_stop_1764_);
if (v___x_1770_ == 0)
{
lean_object* v___x_1771_; lean_object* v___x_1772_; 
v___x_1771_ = lean_array_uget_borrowed(v_as_1762_, v_i_1763_);
v___x_1772_ = lp_batteries_Batteries_Tactic_Lint_isCondition(v___x_1771_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v_a_1773_; lean_object* v___x_1775_; uint8_t v_isShared_1776_; uint8_t v_isSharedCheck_1784_; 
v_a_1773_ = lean_ctor_get(v___x_1772_, 0);
v_isSharedCheck_1784_ = !lean_is_exclusive(v___x_1772_);
if (v_isSharedCheck_1784_ == 0)
{
v___x_1775_ = v___x_1772_;
v_isShared_1776_ = v_isSharedCheck_1784_;
goto v_resetjp_1774_;
}
else
{
lean_inc(v_a_1773_);
lean_dec(v___x_1772_);
v___x_1775_ = lean_box(0);
v_isShared_1776_ = v_isSharedCheck_1784_;
goto v_resetjp_1774_;
}
v_resetjp_1774_:
{
uint8_t v___x_1777_; 
v___x_1777_ = lean_unbox(v_a_1773_);
if (v___x_1777_ == 0)
{
size_t v___x_1778_; size_t v___x_1779_; 
lean_del_object(v___x_1775_);
lean_dec(v_a_1773_);
v___x_1778_ = ((size_t)1ULL);
v___x_1779_ = lean_usize_add(v_i_1763_, v___x_1778_);
v_i_1763_ = v___x_1779_;
goto _start;
}
else
{
lean_object* v___x_1782_; 
if (v_isShared_1776_ == 0)
{
v___x_1782_ = v___x_1775_;
goto v_reusejp_1781_;
}
else
{
lean_object* v_reuseFailAlloc_1783_; 
v_reuseFailAlloc_1783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1783_, 0, v_a_1773_);
v___x_1782_ = v_reuseFailAlloc_1783_;
goto v_reusejp_1781_;
}
v_reusejp_1781_:
{
return v___x_1782_;
}
}
}
}
else
{
return v___x_1772_;
}
}
else
{
uint8_t v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1785_ = 0;
v___x_1786_ = lean_box(v___x_1785_);
v___x_1787_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1787_, 0, v___x_1786_);
return v___x_1787_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0___boxed(lean_object* v_as_1788_, lean_object* v_i_1789_, lean_object* v_stop_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_){
_start:
{
size_t v_i_boxed_1796_; size_t v_stop_boxed_1797_; lean_object* v_res_1798_; 
v_i_boxed_1796_ = lean_unbox_usize(v_i_1789_);
lean_dec(v_i_1789_);
v_stop_boxed_1797_ = lean_unbox_usize(v_stop_1790_);
lean_dec(v_stop_1790_);
v_res_1798_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0(v_as_1788_, v_i_boxed_1796_, v_stop_boxed_1797_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
lean_dec(v___y_1792_);
lean_dec_ref(v___y_1791_);
lean_dec_ref(v_as_1788_);
return v_res_1798_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0(uint8_t v___x_1799_, lean_object* v_hyps_1800_, lean_object* v_x_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_){
_start:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; uint8_t v___x_1809_; 
v___x_1807_ = lean_unsigned_to_nat(0u);
v___x_1808_ = lean_array_get_size(v_hyps_1800_);
v___x_1809_ = lean_nat_dec_lt(v___x_1807_, v___x_1808_);
if (v___x_1809_ == 0)
{
lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1810_ = lean_box(v___x_1799_);
v___x_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1811_, 0, v___x_1810_);
return v___x_1811_;
}
else
{
if (v___x_1809_ == 0)
{
lean_object* v___x_1812_; lean_object* v___x_1813_; 
v___x_1812_ = lean_box(v___x_1799_);
v___x_1813_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1813_, 0, v___x_1812_);
return v___x_1813_;
}
else
{
size_t v___x_1814_; size_t v___x_1815_; lean_object* v___x_1816_; 
v___x_1814_ = ((size_t)0ULL);
v___x_1815_ = lean_usize_of_nat(v___x_1808_);
v___x_1816_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint_simpNF_spec__0(v_hyps_1800_, v___x_1814_, v___x_1815_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_);
return v___x_1816_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0___boxed(lean_object* v___x_1817_, lean_object* v_hyps_1818_, lean_object* v_x_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_){
_start:
{
uint8_t v___x_28340__boxed_1825_; lean_object* v_res_1826_; 
v___x_28340__boxed_1825_ = lean_unbox(v___x_1817_);
v_res_1826_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___lam__0(v___x_28340__boxed_1825_, v_hyps_1818_, v_x_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_);
lean_dec(v___y_1823_);
lean_dec_ref(v___y_1822_);
lean_dec(v___y_1821_);
lean_dec_ref(v___y_1820_);
lean_dec_ref(v_x_1819_);
lean_dec_ref(v_hyps_1818_);
return v_res_1826_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2(uint8_t v_a_1832_, lean_object* v_as_1833_, size_t v_sz_1834_, size_t v_i_1835_, lean_object* v_b_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_){
_start:
{
lean_object* v_a_1843_; uint8_t v___x_1847_; 
v___x_1847_ = lean_usize_dec_lt(v_i_1835_, v_sz_1834_);
if (v___x_1847_ == 0)
{
lean_object* v___x_1848_; 
v___x_1848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1848_, 0, v_b_1836_);
return v___x_1848_;
}
else
{
lean_object* v_a_1849_; lean_object* v___x_1850_; 
v_a_1849_ = lean_array_uget_borrowed(v_as_1833_, v_i_1835_);
v___x_1850_ = lp_batteries_Batteries_Tactic_Lint_isCondition(v_a_1849_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_);
if (lean_obj_tag(v___x_1850_) == 0)
{
lean_object* v_a_1851_; uint8_t v___x_1852_; 
v_a_1851_ = lean_ctor_get(v___x_1850_, 0);
lean_inc(v_a_1851_);
lean_dec_ref_known(v___x_1850_, 1);
v___x_1852_ = lean_unbox(v_a_1851_);
lean_dec(v_a_1851_);
if (v___x_1852_ == 0)
{
lean_object* v_fst_1853_; lean_object* v_snd_1854_; lean_object* v___x_1856_; uint8_t v_isShared_1857_; uint8_t v_isSharedCheck_1861_; 
v_fst_1853_ = lean_ctor_get(v_b_1836_, 0);
v_snd_1854_ = lean_ctor_get(v_b_1836_, 1);
v_isSharedCheck_1861_ = !lean_is_exclusive(v_b_1836_);
if (v_isSharedCheck_1861_ == 0)
{
v___x_1856_ = v_b_1836_;
v_isShared_1857_ = v_isSharedCheck_1861_;
goto v_resetjp_1855_;
}
else
{
lean_inc(v_snd_1854_);
lean_inc(v_fst_1853_);
lean_dec(v_b_1836_);
v___x_1856_ = lean_box(0);
v_isShared_1857_ = v_isSharedCheck_1861_;
goto v_resetjp_1855_;
}
v_resetjp_1855_:
{
lean_object* v___x_1859_; 
if (v_isShared_1857_ == 0)
{
v___x_1859_ = v___x_1856_;
goto v_reusejp_1858_;
}
else
{
lean_object* v_reuseFailAlloc_1860_; 
v_reuseFailAlloc_1860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1860_, 0, v_fst_1853_);
lean_ctor_set(v_reuseFailAlloc_1860_, 1, v_snd_1854_);
v___x_1859_ = v_reuseFailAlloc_1860_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
v_a_1843_ = v___x_1859_;
goto v___jp_1842_;
}
}
}
else
{
lean_object* v_fst_1862_; lean_object* v_snd_1863_; lean_object* v___x_1865_; uint8_t v_isShared_1866_; uint8_t v_isSharedCheck_1912_; 
v_fst_1862_ = lean_ctor_get(v_b_1836_, 0);
v_snd_1863_ = lean_ctor_get(v_b_1836_, 1);
v_isSharedCheck_1912_ = !lean_is_exclusive(v_b_1836_);
if (v_isSharedCheck_1912_ == 0)
{
v___x_1865_ = v_b_1836_;
v_isShared_1866_ = v_isSharedCheck_1912_;
goto v_resetjp_1864_;
}
else
{
lean_inc(v_snd_1863_);
lean_inc(v_fst_1862_);
lean_dec(v_b_1836_);
v___x_1865_ = lean_box(0);
v_isShared_1866_ = v_isSharedCheck_1912_;
goto v_resetjp_1864_;
}
v_resetjp_1864_:
{
uint8_t v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; 
v___x_1867_ = 0;
v___x_1868_ = l_Lean_Expr_fvarId_x21(v_a_1849_);
v___x_1869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1869_, 0, v___x_1868_);
v___x_1870_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__0));
v___x_1871_ = lean_unsigned_to_nat(1000u);
v___x_1872_ = l_Lean_Meta_simpGlobalConfig;
lean_inc(v_a_1849_);
v___x_1873_ = l_Lean_Meta_SimpTheorems_add(v_fst_1862_, v___x_1869_, v___x_1870_, v_a_1849_, v___x_1867_, v_a_1832_, v___x_1871_, v___x_1872_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_);
if (lean_obj_tag(v___x_1873_) == 0)
{
uint8_t v___x_1874_; 
v___x_1874_ = lean_unbox(v_snd_1863_);
if (v___x_1874_ == 0)
{
lean_object* v_a_1875_; lean_object* v___x_1876_; 
lean_dec(v_snd_1863_);
v_a_1875_ = lean_ctor_get(v___x_1873_, 0);
lean_inc(v_a_1875_);
lean_dec_ref_known(v___x_1873_, 1);
lean_inc(v___y_1840_);
lean_inc_ref(v___y_1839_);
lean_inc(v___y_1838_);
lean_inc_ref(v___y_1837_);
lean_inc(v_a_1849_);
v___x_1876_ = lean_infer_type(v_a_1849_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_);
if (lean_obj_tag(v___x_1876_) == 0)
{
lean_object* v_a_1877_; lean_object* v___f_1878_; lean_object* v___x_1879_; 
v_a_1877_ = lean_ctor_get(v___x_1876_, 0);
lean_inc(v_a_1877_);
lean_dec_ref_known(v___x_1876_, 1);
v___f_1878_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___closed__1));
v___x_1879_ = lp_batteries_Lean_Meta_forallTelescope___at___00Batteries_Tactic_Lint_simpNF_spec__1___redArg(v_a_1877_, v___f_1878_, v___x_1867_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_);
if (lean_obj_tag(v___x_1879_) == 0)
{
lean_object* v_a_1880_; lean_object* v___x_1882_; 
v_a_1880_ = lean_ctor_get(v___x_1879_, 0);
lean_inc(v_a_1880_);
lean_dec_ref_known(v___x_1879_, 1);
if (v_isShared_1866_ == 0)
{
lean_ctor_set(v___x_1865_, 1, v_a_1880_);
lean_ctor_set(v___x_1865_, 0, v_a_1875_);
v___x_1882_ = v___x_1865_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v_a_1875_);
lean_ctor_set(v_reuseFailAlloc_1883_, 1, v_a_1880_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
v_a_1843_ = v___x_1882_;
goto v___jp_1842_;
}
}
else
{
lean_object* v_a_1884_; lean_object* v___x_1886_; uint8_t v_isShared_1887_; uint8_t v_isSharedCheck_1891_; 
lean_dec(v_a_1875_);
lean_del_object(v___x_1865_);
v_a_1884_ = lean_ctor_get(v___x_1879_, 0);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1879_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1886_ = v___x_1879_;
v_isShared_1887_ = v_isSharedCheck_1891_;
goto v_resetjp_1885_;
}
else
{
lean_inc(v_a_1884_);
lean_dec(v___x_1879_);
v___x_1886_ = lean_box(0);
v_isShared_1887_ = v_isSharedCheck_1891_;
goto v_resetjp_1885_;
}
v_resetjp_1885_:
{
lean_object* v___x_1889_; 
if (v_isShared_1887_ == 0)
{
v___x_1889_ = v___x_1886_;
goto v_reusejp_1888_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1890_, 0, v_a_1884_);
v___x_1889_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1888_;
}
v_reusejp_1888_:
{
return v___x_1889_;
}
}
}
}
else
{
lean_object* v_a_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_1899_; 
lean_dec(v_a_1875_);
lean_del_object(v___x_1865_);
v_a_1892_ = lean_ctor_get(v___x_1876_, 0);
v_isSharedCheck_1899_ = !lean_is_exclusive(v___x_1876_);
if (v_isSharedCheck_1899_ == 0)
{
v___x_1894_ = v___x_1876_;
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_a_1892_);
lean_dec(v___x_1876_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v___x_1897_; 
if (v_isShared_1895_ == 0)
{
v___x_1897_ = v___x_1894_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v_a_1892_);
v___x_1897_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
return v___x_1897_;
}
}
}
}
else
{
lean_object* v_a_1900_; lean_object* v___x_1902_; 
v_a_1900_ = lean_ctor_get(v___x_1873_, 0);
lean_inc(v_a_1900_);
lean_dec_ref_known(v___x_1873_, 1);
if (v_isShared_1866_ == 0)
{
lean_ctor_set(v___x_1865_, 0, v_a_1900_);
v___x_1902_ = v___x_1865_;
goto v_reusejp_1901_;
}
else
{
lean_object* v_reuseFailAlloc_1903_; 
v_reuseFailAlloc_1903_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1903_, 0, v_a_1900_);
lean_ctor_set(v_reuseFailAlloc_1903_, 1, v_snd_1863_);
v___x_1902_ = v_reuseFailAlloc_1903_;
goto v_reusejp_1901_;
}
v_reusejp_1901_:
{
v_a_1843_ = v___x_1902_;
goto v___jp_1842_;
}
}
}
else
{
lean_object* v_a_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_1911_; 
lean_del_object(v___x_1865_);
lean_dec(v_snd_1863_);
v_a_1904_ = lean_ctor_get(v___x_1873_, 0);
v_isSharedCheck_1911_ = !lean_is_exclusive(v___x_1873_);
if (v_isSharedCheck_1911_ == 0)
{
v___x_1906_ = v___x_1873_;
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_a_1904_);
lean_dec(v___x_1873_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v___x_1909_; 
if (v_isShared_1907_ == 0)
{
v___x_1909_ = v___x_1906_;
goto v_reusejp_1908_;
}
else
{
lean_object* v_reuseFailAlloc_1910_; 
v_reuseFailAlloc_1910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1910_, 0, v_a_1904_);
v___x_1909_ = v_reuseFailAlloc_1910_;
goto v_reusejp_1908_;
}
v_reusejp_1908_:
{
return v___x_1909_;
}
}
}
}
}
}
else
{
lean_object* v_a_1913_; lean_object* v___x_1915_; uint8_t v_isShared_1916_; uint8_t v_isSharedCheck_1920_; 
lean_dec_ref(v_b_1836_);
v_a_1913_ = lean_ctor_get(v___x_1850_, 0);
v_isSharedCheck_1920_ = !lean_is_exclusive(v___x_1850_);
if (v_isSharedCheck_1920_ == 0)
{
v___x_1915_ = v___x_1850_;
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
else
{
lean_inc(v_a_1913_);
lean_dec(v___x_1850_);
v___x_1915_ = lean_box(0);
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
v_resetjp_1914_:
{
lean_object* v___x_1918_; 
if (v_isShared_1916_ == 0)
{
v___x_1918_ = v___x_1915_;
goto v_reusejp_1917_;
}
else
{
lean_object* v_reuseFailAlloc_1919_; 
v_reuseFailAlloc_1919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1919_, 0, v_a_1913_);
v___x_1918_ = v_reuseFailAlloc_1919_;
goto v_reusejp_1917_;
}
v_reusejp_1917_:
{
return v___x_1918_;
}
}
}
}
v___jp_1842_:
{
size_t v___x_1844_; size_t v___x_1845_; 
v___x_1844_ = ((size_t)1ULL);
v___x_1845_ = lean_usize_add(v_i_1835_, v___x_1844_);
v_i_1835_ = v___x_1845_;
v_b_1836_ = v_a_1843_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2___boxed(lean_object* v_a_1921_, lean_object* v_as_1922_, lean_object* v_sz_1923_, lean_object* v_i_1924_, lean_object* v_b_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_){
_start:
{
uint8_t v_a_28389__boxed_1931_; size_t v_sz_boxed_1932_; size_t v_i_boxed_1933_; lean_object* v_res_1934_; 
v_a_28389__boxed_1931_ = lean_unbox(v_a_1921_);
v_sz_boxed_1932_ = lean_unbox_usize(v_sz_1923_);
lean_dec(v_sz_1923_);
v_i_boxed_1933_ = lean_unbox_usize(v_i_1924_);
lean_dec(v_i_1924_);
v_res_1934_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2(v_a_28389__boxed_1931_, v_as_1922_, v_sz_boxed_1932_, v_i_boxed_1933_, v_b_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_);
lean_dec(v___y_1929_);
lean_dec_ref(v___y_1928_);
lean_dec(v___y_1927_);
lean_dec_ref(v___y_1926_);
lean_dec_ref(v_as_1922_);
return v_res_1934_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
v___x_1938_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__1));
v___x_1939_ = l_Lean_MessageData_ofFormat(v___x_1938_);
return v___x_1939_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1941_; lean_object* v___x_1942_; 
v___x_1941_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__3));
v___x_1942_ = l_Lean_stringToMessageData(v___x_1941_);
return v___x_1942_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6(void){
_start:
{
lean_object* v___x_1944_; lean_object* v___x_1945_; 
v___x_1944_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__5));
v___x_1945_ = l_Lean_stringToMessageData(v___x_1944_);
return v___x_1945_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1946_; lean_object* v___x_1947_; 
v___x_1946_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLemmas___closed__3));
v___x_1947_ = l_Lean_stringToMessageData(v___x_1946_);
return v___x_1947_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1949_; lean_object* v___x_1950_; 
v___x_1949_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__8));
v___x_1950_ = l_Lean_stringToMessageData(v___x_1949_);
return v___x_1950_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1952_; lean_object* v___x_1953_; 
v___x_1952_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__10));
v___x_1953_ = l_Lean_stringToMessageData(v___x_1952_);
return v___x_1953_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13(void){
_start:
{
lean_object* v___x_1955_; lean_object* v___x_1956_; 
v___x_1955_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__12));
v___x_1956_ = l_Lean_stringToMessageData(v___x_1955_);
return v___x_1956_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16(void){
_start:
{
lean_object* v___x_1960_; lean_object* v___x_1961_; 
v___x_1960_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__15));
v___x_1961_ = l_Lean_MessageData_ofFormat(v___x_1960_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1(uint8_t v_a_1964_, lean_object* v_declName_1965_, lean_object* v_x_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_){
_start:
{
lean_object* v_hyps_1972_; lean_object* v_lhs_1973_; lean_object* v_rhs_1974_; lean_object* v___x_1975_; 
v_hyps_1972_ = lean_ctor_get(v_x_1966_, 0);
lean_inc_ref(v_hyps_1972_);
v_lhs_1973_ = lean_ctor_get(v_x_1966_, 1);
lean_inc_ref(v_lhs_1973_);
v_rhs_1974_ = lean_ctor_get(v_x_1966_, 2);
lean_inc_ref(v_rhs_1974_);
lean_dec_ref(v_x_1966_);
v___x_1975_ = l_Lean_Meta_getSimpTheorems___redArg(v___y_1970_);
if (lean_obj_tag(v___x_1975_) == 0)
{
lean_object* v_a_1976_; uint8_t v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; size_t v_sz_1980_; size_t v___x_1981_; lean_object* v___x_1982_; 
v_a_1976_ = lean_ctor_get(v___x_1975_, 0);
lean_inc(v_a_1976_);
lean_dec_ref_known(v___x_1975_, 1);
v___x_1977_ = 0;
v___x_1978_ = lean_box(v___x_1977_);
v___x_1979_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1979_, 0, v_a_1976_);
lean_ctor_set(v___x_1979_, 1, v___x_1978_);
v_sz_1980_ = lean_array_size(v_hyps_1972_);
v___x_1981_ = ((size_t)0ULL);
v___x_1982_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__2(v_a_1964_, v_hyps_1972_, v_sz_1980_, v___x_1981_, v___x_1979_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_1982_) == 0)
{
lean_object* v_a_1983_; lean_object* v___x_1984_; 
v_a_1983_ = lean_ctor_get(v___x_1982_, 0);
lean_inc(v_a_1983_);
lean_dec_ref_known(v___x_1982_, 1);
v___x_1984_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v___y_1970_);
if (lean_obj_tag(v___x_1984_) == 0)
{
lean_object* v_a_1985_; lean_object* v_fst_1986_; lean_object* v_snd_1987_; lean_object* v___x_1989_; uint8_t v_isShared_1990_; uint8_t v_isSharedCheck_2268_; 
v_a_1985_ = lean_ctor_get(v___x_1984_, 0);
lean_inc(v_a_1985_);
lean_dec_ref_known(v___x_1984_, 1);
v_fst_1986_ = lean_ctor_get(v_a_1983_, 0);
v_snd_1987_ = lean_ctor_get(v_a_1983_, 1);
v_isSharedCheck_2268_ = !lean_is_exclusive(v_a_1983_);
if (v_isSharedCheck_2268_ == 0)
{
v___x_1989_ = v_a_1983_;
v_isShared_1990_ = v_isSharedCheck_2268_;
goto v_resetjp_1988_;
}
else
{
lean_inc(v_snd_1987_);
lean_inc(v_fst_1986_);
lean_dec(v_a_1983_);
v___x_1989_ = lean_box(0);
v_isShared_1990_ = v_isSharedCheck_2268_;
goto v_resetjp_1988_;
}
v_resetjp_1988_:
{
lean_object* v___x_1991_; lean_object* v___x_1992_; uint8_t v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; uint8_t v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; 
v___x_1991_ = lean_unsigned_to_nat(100000u);
v___x_1992_ = lean_unsigned_to_nat(2u);
v___x_1993_ = 0;
v___x_1994_ = lean_box(0);
v___x_1995_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_1995_, 0, v___x_1991_);
lean_ctor_set(v___x_1995_, 1, v___x_1992_);
lean_ctor_set(v___x_1995_, 2, v___x_1994_);
v___x_1996_ = lean_unbox(v_snd_1987_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3, v___x_1996_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 1, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 2, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 3, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 4, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 5, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 6, v___x_1993_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 7, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 8, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 9, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 10, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 11, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 12, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 13, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 14, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 15, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 16, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 17, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 18, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 19, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 20, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 21, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 22, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 23, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 24, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 25, v_a_1964_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 26, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 27, v___x_1977_);
lean_ctor_set_uint8(v___x_1995_, sizeof(void*)*3 + 28, v___x_1977_);
v___x_1997_ = lean_unsigned_to_nat(1u);
v___x_1998_ = lean_mk_empty_array_with_capacity(v___x_1997_);
v___x_1999_ = lean_array_push(v___x_1998_, v_fst_1986_);
v___x_2000_ = l_Lean_Options_empty;
v___x_2001_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_1995_, v___x_1999_, v_a_1985_, v___x_2000_, v___y_1967_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2001_) == 0)
{
lean_object* v_a_2002_; lean_object* v___x_2003_; 
v_a_2002_ = lean_ctor_get(v___x_2001_, 0);
lean_inc(v_a_2002_);
lean_dec_ref_known(v___x_2001_, 1);
lean_inc(v_declName_1965_);
v___x_2003_ = l_Lean_Meta_isRflTheorem(v_declName_1965_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2003_) == 0)
{
lean_object* v_a_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; 
v_a_2004_ = lean_ctor_get(v___x_2003_, 0);
lean_inc_n(v_a_2004_, 2);
lean_dec_ref_known(v___x_2003_, 1);
v___x_2005_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__2);
v___x_2006_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__6);
v___x_2007_ = lean_box(v_a_1964_);
lean_inc(v_a_2002_);
lean_inc_ref(v_lhs_1973_);
v___x_2008_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___boxed), 10, 5);
lean_closure_set(v___x_2008_, 0, v_a_2004_);
lean_closure_set(v___x_2008_, 1, v___x_2007_);
lean_closure_set(v___x_2008_, 2, v_lhs_1973_);
lean_closure_set(v___x_2008_, 3, v_a_2002_);
lean_closure_set(v___x_2008_, 4, v___x_2006_);
v___x_2009_ = lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(v___x_2005_, v___x_2008_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2009_) == 0)
{
lean_object* v_a_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2243_; 
v_a_2010_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2243_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2243_ == 0)
{
v___x_2012_ = v___x_2009_;
v_isShared_2013_ = v_isSharedCheck_2243_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_a_2010_);
lean_dec(v___x_2009_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2243_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v_fst_2014_; lean_object* v_snd_2015_; lean_object* v___x_2017_; uint8_t v_isShared_2018_; uint8_t v_isSharedCheck_2242_; 
v_fst_2014_ = lean_ctor_get(v_a_2010_, 0);
v_snd_2015_ = lean_ctor_get(v_a_2010_, 1);
v_isSharedCheck_2242_ = !lean_is_exclusive(v_a_2010_);
if (v_isSharedCheck_2242_ == 0)
{
v___x_2017_ = v_a_2010_;
v_isShared_2018_ = v_isSharedCheck_2242_;
goto v_resetjp_2016_;
}
else
{
lean_inc(v_snd_2015_);
lean_inc(v_fst_2014_);
lean_dec(v_a_2010_);
v___x_2017_ = lean_box(0);
v_isShared_2018_ = v_isSharedCheck_2242_;
goto v_resetjp_2016_;
}
v_resetjp_2016_:
{
lean_object* v_expr_2019_; lean_object* v_proof_x3f_2020_; lean_object* v___x_2021_; 
v_expr_2019_ = lean_ctor_get(v_fst_2014_, 0);
lean_inc_ref(v_expr_2019_);
v_proof_x3f_2020_ = lean_ctor_get(v_fst_2014_, 1);
lean_inc(v_proof_x3f_2020_);
lean_dec(v_fst_2014_);
v___x_2021_ = l_Lean_Meta_isEqnThm_x3f___redArg(v_declName_1965_, v___y_1970_);
if (lean_obj_tag(v___x_2021_) == 0)
{
lean_object* v_a_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2233_; 
v_a_2022_ = lean_ctor_get(v___x_2021_, 0);
v_isSharedCheck_2233_ = !lean_is_exclusive(v___x_2021_);
if (v_isSharedCheck_2233_ == 0)
{
v___x_2024_ = v___x_2021_;
v_isShared_2025_ = v_isSharedCheck_2233_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_a_2022_);
lean_dec(v___x_2021_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2233_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v_usedTheorems_2030_; lean_object* v___y_2032_; uint8_t v___y_2033_; uint8_t v___y_2034_; lean_object* v___y_2035_; uint8_t v___y_2182_; lean_object* v_map_2227_; lean_object* v___x_2228_; uint8_t v___x_2229_; 
v_usedTheorems_2030_ = lean_ctor_get(v_snd_2015_, 0);
lean_inc_ref(v_usedTheorems_2030_);
v_map_2227_ = lean_ctor_get(v_usedTheorems_2030_, 0);
v___x_2228_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_2228_, 0, v_declName_1965_);
lean_ctor_set_uint8(v___x_2228_, sizeof(void*)*1, v_a_1964_);
lean_ctor_set_uint8(v___x_2228_, sizeof(void*)*1 + 1, v___x_1977_);
v___x_2229_ = l_Lean_PersistentHashMap_contains___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_eraseIfExists_spec__0___redArg(v_map_2227_, v___x_2228_);
lean_dec_ref_known(v___x_2228_, 1);
if (v___x_2229_ == 0)
{
if (lean_obj_tag(v_a_2022_) == 0)
{
v___y_2182_ = v___x_2229_;
goto v___jp_2181_;
}
else
{
lean_object* v_val_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
v_val_2230_ = lean_ctor_get(v_a_2022_, 0);
lean_inc(v_val_2230_);
lean_dec_ref_known(v_a_2022_, 1);
v___x_2231_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_2231_, 0, v_val_2230_);
lean_ctor_set_uint8(v___x_2231_, sizeof(void*)*1, v_a_1964_);
lean_ctor_set_uint8(v___x_2231_, sizeof(void*)*1 + 1, v___x_1977_);
v___x_2232_ = l_Lean_PersistentHashMap_contains___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_eraseIfExists_spec__0___redArg(v_map_2227_, v___x_2231_);
lean_dec_ref_known(v___x_2231_, 1);
v___y_2182_ = v___x_2232_;
goto v___jp_2181_;
}
}
else
{
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_a_2022_);
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_dec(v_snd_2015_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_dec(v_a_2002_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
goto v___jp_2026_;
}
v___jp_2026_:
{
lean_object* v___x_2028_; 
if (v_isShared_2025_ == 0)
{
lean_ctor_set(v___x_2024_, 0, v___x_1994_);
v___x_2028_ = v___x_2024_;
goto v_reusejp_2027_;
}
else
{
lean_object* v_reuseFailAlloc_2029_; 
v_reuseFailAlloc_2029_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2029_, 0, v___x_1994_);
v___x_2028_ = v_reuseFailAlloc_2029_;
goto v_reusejp_2027_;
}
v_reusejp_2027_:
{
return v___x_2028_;
}
}
v___jp_2031_:
{
if (v___y_2034_ == 0)
{
lean_dec_ref(v___y_2032_);
lean_dec(v_proof_x3f_2020_);
if (v___y_2033_ == 0)
{
lean_object* v___x_2036_; 
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_dec_ref(v_hyps_1972_);
v___x_2036_ = l_Lean_Meta_addPPExplicitToExposeDiff(v_lhs_1973_, v_expr_2019_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2036_) == 0)
{
lean_object* v_a_2037_; lean_object* v_fst_2038_; lean_object* v_snd_2039_; lean_object* v___x_2041_; uint8_t v_isShared_2042_; uint8_t v_isSharedCheck_2083_; 
v_a_2037_ = lean_ctor_get(v___x_2036_, 0);
lean_inc(v_a_2037_);
lean_dec_ref_known(v___x_2036_, 1);
v_fst_2038_ = lean_ctor_get(v_a_2037_, 0);
v_snd_2039_ = lean_ctor_get(v_a_2037_, 1);
v_isSharedCheck_2083_ = !lean_is_exclusive(v_a_2037_);
if (v_isSharedCheck_2083_ == 0)
{
v___x_2041_ = v_a_2037_;
v_isShared_2042_ = v_isSharedCheck_2083_;
goto v_resetjp_2040_;
}
else
{
lean_inc(v_snd_2039_);
lean_inc(v_fst_2038_);
lean_dec(v_a_2037_);
v___x_2041_ = lean_box(0);
v_isShared_2042_ = v_isSharedCheck_2083_;
goto v_resetjp_2040_;
}
v_resetjp_2040_:
{
lean_object* v___x_2043_; lean_object* v___x_2044_; 
v___x_2043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2043_, 0, v_snd_1987_);
lean_inc_ref(v___y_2035_);
v___x_2044_ = lp_batteries_Batteries_Tactic_Lint_formatLemmas(v_usedTheorems_2030_, v___y_2035_, v___x_2043_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
lean_dec_ref_known(v___x_2043_, 1);
lean_dec_ref(v_usedTheorems_2030_);
if (lean_obj_tag(v___x_2044_) == 0)
{
lean_object* v_a_2045_; lean_object* v___x_2047_; uint8_t v_isShared_2048_; uint8_t v_isSharedCheck_2074_; 
v_a_2045_ = lean_ctor_get(v___x_2044_, 0);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2044_);
if (v_isSharedCheck_2074_ == 0)
{
v___x_2047_ = v___x_2044_;
v_isShared_2048_ = v_isSharedCheck_2074_;
goto v_resetjp_2046_;
}
else
{
lean_inc(v_a_2045_);
lean_dec(v___x_2044_);
v___x_2047_ = lean_box(0);
v_isShared_2048_ = v_isSharedCheck_2074_;
goto v_resetjp_2046_;
}
v_resetjp_2046_:
{
lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2053_; 
v___x_2049_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__4);
v___x_2050_ = l_Lean_MessageData_ofExpr(v_fst_2038_);
v___x_2051_ = l_Lean_indentD(v___x_2050_);
if (v_isShared_2042_ == 0)
{
lean_ctor_set_tag(v___x_2041_, 7);
lean_ctor_set(v___x_2041_, 1, v___x_2051_);
lean_ctor_set(v___x_2041_, 0, v___x_2049_);
v___x_2053_ = v___x_2041_;
goto v_reusejp_2052_;
}
else
{
lean_object* v_reuseFailAlloc_2073_; 
v_reuseFailAlloc_2073_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2073_, 0, v___x_2049_);
lean_ctor_set(v_reuseFailAlloc_2073_, 1, v___x_2051_);
v___x_2053_ = v_reuseFailAlloc_2073_;
goto v_reusejp_2052_;
}
v_reusejp_2052_:
{
lean_object* v___x_2054_; lean_object* v___x_2056_; 
v___x_2054_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__22);
if (v_isShared_2018_ == 0)
{
lean_ctor_set_tag(v___x_2017_, 7);
lean_ctor_set(v___x_2017_, 1, v___x_2054_);
lean_ctor_set(v___x_2017_, 0, v___x_2053_);
v___x_2056_ = v___x_2017_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2072_; 
v_reuseFailAlloc_2072_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2072_, 0, v___x_2053_);
lean_ctor_set(v_reuseFailAlloc_2072_, 1, v___x_2054_);
v___x_2056_ = v_reuseFailAlloc_2072_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2060_; 
v___x_2057_ = l_Lean_MessageData_ofExpr(v_snd_2039_);
v___x_2058_ = l_Lean_indentD(v___x_2057_);
if (v_isShared_1990_ == 0)
{
lean_ctor_set_tag(v___x_1989_, 7);
lean_ctor_set(v___x_1989_, 1, v___x_2058_);
lean_ctor_set(v___x_1989_, 0, v___x_2056_);
v___x_2060_ = v___x_1989_;
goto v_reusejp_2059_;
}
else
{
lean_object* v_reuseFailAlloc_2071_; 
v_reuseFailAlloc_2071_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2071_, 0, v___x_2056_);
lean_ctor_set(v_reuseFailAlloc_2071_, 1, v___x_2058_);
v___x_2060_ = v_reuseFailAlloc_2071_;
goto v_reusejp_2059_;
}
v_reusejp_2059_:
{
lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2069_; 
v___x_2061_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3___closed__24);
v___x_2062_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2060_);
lean_ctor_set(v___x_2062_, 1, v___x_2061_);
v___x_2063_ = l_Lean_indentD(v_a_2045_);
v___x_2064_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2064_, 0, v___x_2062_);
lean_ctor_set(v___x_2064_, 1, v___x_2063_);
v___x_2065_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__6);
v___x_2066_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2066_, 0, v___x_2064_);
lean_ctor_set(v___x_2066_, 1, v___x_2065_);
v___x_2067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2067_, 0, v___x_2066_);
if (v_isShared_2048_ == 0)
{
lean_ctor_set(v___x_2047_, 0, v___x_2067_);
v___x_2069_ = v___x_2047_;
goto v_reusejp_2068_;
}
else
{
lean_object* v_reuseFailAlloc_2070_; 
v_reuseFailAlloc_2070_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2070_, 0, v___x_2067_);
v___x_2069_ = v_reuseFailAlloc_2070_;
goto v_reusejp_2068_;
}
v_reusejp_2068_:
{
return v___x_2069_;
}
}
}
}
}
}
else
{
lean_object* v_a_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2082_; 
lean_del_object(v___x_2041_);
lean_dec(v_snd_2039_);
lean_dec(v_fst_2038_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_1989_);
v_a_2075_ = lean_ctor_get(v___x_2044_, 0);
v_isSharedCheck_2082_ = !lean_is_exclusive(v___x_2044_);
if (v_isSharedCheck_2082_ == 0)
{
v___x_2077_ = v___x_2044_;
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_a_2075_);
lean_dec(v___x_2044_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2080_; 
if (v_isShared_2078_ == 0)
{
v___x_2080_ = v___x_2077_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2081_; 
v_reuseFailAlloc_2081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2081_, 0, v_a_2075_);
v___x_2080_ = v_reuseFailAlloc_2081_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
return v___x_2080_;
}
}
}
}
}
else
{
lean_object* v_a_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2091_; 
lean_dec_ref(v_usedTheorems_2030_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
v_a_2084_ = lean_ctor_get(v___x_2036_, 0);
v_isSharedCheck_2091_ = !lean_is_exclusive(v___x_2036_);
if (v_isSharedCheck_2091_ == 0)
{
v___x_2086_ = v___x_2036_;
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_a_2084_);
lean_dec(v___x_2036_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v___x_2089_; 
if (v_isShared_2087_ == 0)
{
v___x_2089_ = v___x_2086_;
goto v_reusejp_2088_;
}
else
{
lean_object* v_reuseFailAlloc_2090_; 
v_reuseFailAlloc_2090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2090_, 0, v_a_2084_);
v___x_2089_ = v_reuseFailAlloc_2090_;
goto v_reusejp_2088_;
}
v_reusejp_2088_:
{
return v___x_2089_;
}
}
}
}
else
{
uint8_t v___x_2092_; 
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_snd_1987_);
v___x_2092_ = lean_expr_eqv(v_lhs_1973_, v_expr_2019_);
lean_dec_ref(v_expr_2019_);
if (v___x_2092_ == 0)
{
lean_object* v___x_2094_; 
lean_del_object(v___x_2017_);
lean_dec(v_a_2004_);
lean_del_object(v___x_1989_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 0, v___x_1994_);
v___x_2094_ = v___x_2012_;
goto v_reusejp_2093_;
}
else
{
lean_object* v_reuseFailAlloc_2095_; 
v_reuseFailAlloc_2095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2095_, 0, v___x_1994_);
v___x_2094_ = v_reuseFailAlloc_2095_;
goto v_reusejp_2093_;
}
v_reusejp_2093_:
{
return v___x_2094_;
}
}
else
{
lean_object* v___x_2096_; 
lean_del_object(v___x_2012_);
lean_inc(v___y_1970_);
lean_inc_ref(v___y_1969_);
lean_inc(v___y_1968_);
lean_inc_ref(v___y_1967_);
lean_inc_ref(v_lhs_1973_);
v___x_2096_ = lean_infer_type(v_lhs_1973_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2096_) == 0)
{
lean_object* v_a_2097_; lean_object* v___x_2098_; uint8_t v___x_2099_; lean_object* v___x_2100_; 
v_a_2097_ = lean_ctor_get(v___x_2096_, 0);
lean_inc(v_a_2097_);
lean_dec_ref_known(v___x_2096_, 1);
v___x_2098_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__7);
v___x_2099_ = lean_unbox(v_a_2004_);
lean_dec(v_a_2004_);
lean_inc_ref(v___y_2035_);
v___x_2100_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_simpNF_spec__3(v___x_2092_, v_lhs_1973_, v_a_2097_, v___x_2099_, v_a_1964_, v___y_2035_, v_hyps_1972_, v_sz_1980_, v___x_1981_, v___x_2098_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_a_2097_);
lean_dec_ref(v_lhs_1973_);
if (lean_obj_tag(v___x_2100_) == 0)
{
lean_object* v_a_2101_; lean_object* v___x_2103_; uint8_t v_isShared_2104_; uint8_t v_isSharedCheck_2117_; 
v_a_2101_ = lean_ctor_get(v___x_2100_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2100_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2103_ = v___x_2100_;
v_isShared_2104_ = v_isSharedCheck_2117_;
goto v_resetjp_2102_;
}
else
{
lean_inc(v_a_2101_);
lean_dec(v___x_2100_);
v___x_2103_ = lean_box(0);
v_isShared_2104_ = v_isSharedCheck_2117_;
goto v_resetjp_2102_;
}
v_resetjp_2102_:
{
lean_object* v___x_2105_; lean_object* v___x_2107_; 
v___x_2105_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__9);
if (v_isShared_2018_ == 0)
{
lean_ctor_set_tag(v___x_2017_, 7);
lean_ctor_set(v___x_2017_, 1, v_a_2101_);
lean_ctor_set(v___x_2017_, 0, v___x_2105_);
v___x_2107_ = v___x_2017_;
goto v_reusejp_2106_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v___x_2105_);
lean_ctor_set(v_reuseFailAlloc_2116_, 1, v_a_2101_);
v___x_2107_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2106_;
}
v_reusejp_2106_:
{
lean_object* v___x_2108_; lean_object* v___x_2110_; 
v___x_2108_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1, &lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_decorateError___redArg___closed__1);
if (v_isShared_1990_ == 0)
{
lean_ctor_set_tag(v___x_1989_, 7);
lean_ctor_set(v___x_1989_, 1, v___x_2108_);
lean_ctor_set(v___x_1989_, 0, v___x_2107_);
v___x_2110_ = v___x_1989_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2115_; 
v_reuseFailAlloc_2115_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2115_, 0, v___x_2107_);
lean_ctor_set(v_reuseFailAlloc_2115_, 1, v___x_2108_);
v___x_2110_ = v_reuseFailAlloc_2115_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
lean_object* v___x_2111_; lean_object* v___x_2113_; 
v___x_2111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2111_, 0, v___x_2110_);
if (v_isShared_2104_ == 0)
{
lean_ctor_set(v___x_2103_, 0, v___x_2111_);
v___x_2113_ = v___x_2103_;
goto v_reusejp_2112_;
}
else
{
lean_object* v_reuseFailAlloc_2114_; 
v_reuseFailAlloc_2114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2114_, 0, v___x_2111_);
v___x_2113_ = v_reuseFailAlloc_2114_;
goto v_reusejp_2112_;
}
v_reusejp_2112_:
{
return v___x_2113_;
}
}
}
}
}
else
{
lean_object* v_a_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2125_; 
lean_del_object(v___x_2017_);
lean_del_object(v___x_1989_);
v_a_2118_ = lean_ctor_get(v___x_2100_, 0);
v_isSharedCheck_2125_ = !lean_is_exclusive(v___x_2100_);
if (v_isSharedCheck_2125_ == 0)
{
v___x_2120_ = v___x_2100_;
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_a_2118_);
lean_dec(v___x_2100_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v___x_2123_; 
if (v_isShared_2121_ == 0)
{
v___x_2123_ = v___x_2120_;
goto v_reusejp_2122_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v_a_2118_);
v___x_2123_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2122_;
}
v_reusejp_2122_:
{
return v___x_2123_;
}
}
}
}
else
{
lean_object* v_a_2126_; lean_object* v___x_2128_; uint8_t v_isShared_2129_; uint8_t v_isSharedCheck_2133_; 
lean_del_object(v___x_2017_);
lean_dec(v_a_2004_);
lean_del_object(v___x_1989_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
v_a_2126_ = lean_ctor_get(v___x_2096_, 0);
v_isSharedCheck_2133_ = !lean_is_exclusive(v___x_2096_);
if (v_isSharedCheck_2133_ == 0)
{
v___x_2128_ = v___x_2096_;
v_isShared_2129_ = v_isSharedCheck_2133_;
goto v_resetjp_2127_;
}
else
{
lean_inc(v_a_2126_);
lean_dec(v___x_2096_);
v___x_2128_ = lean_box(0);
v_isShared_2129_ = v_isSharedCheck_2133_;
goto v_resetjp_2127_;
}
v_resetjp_2127_:
{
lean_object* v___x_2131_; 
if (v_isShared_2129_ == 0)
{
v___x_2131_ = v___x_2128_;
goto v_reusejp_2130_;
}
else
{
lean_object* v_reuseFailAlloc_2132_; 
v_reuseFailAlloc_2132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2132_, 0, v_a_2126_);
v___x_2131_ = v_reuseFailAlloc_2132_;
goto v_reusejp_2130_;
}
v_reusejp_2130_:
{
return v___x_2131_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_usedTheorems_2030_);
lean_dec_ref(v_expr_2019_);
lean_dec(v_a_2004_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
if (lean_obj_tag(v_proof_x3f_2020_) == 0)
{
lean_object* v___x_2135_; 
lean_dec_ref(v___y_2032_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 0, v___x_1994_);
v___x_2135_ = v___x_2012_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2136_; 
v_reuseFailAlloc_2136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2136_, 0, v___x_1994_);
v___x_2135_ = v_reuseFailAlloc_2136_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
return v___x_2135_;
}
}
else
{
lean_object* v___x_2138_; uint8_t v_isShared_2139_; uint8_t v_isSharedCheck_2179_; 
lean_del_object(v___x_2012_);
v_isSharedCheck_2179_ = !lean_is_exclusive(v_proof_x3f_2020_);
if (v_isSharedCheck_2179_ == 0)
{
lean_object* v_unused_2180_; 
v_unused_2180_ = lean_ctor_get(v_proof_x3f_2020_, 0);
lean_dec(v_unused_2180_);
v___x_2138_ = v_proof_x3f_2020_;
v_isShared_2139_ = v_isSharedCheck_2179_;
goto v_resetjp_2137_;
}
else
{
lean_dec(v_proof_x3f_2020_);
v___x_2138_ = lean_box(0);
v_isShared_2139_ = v_isSharedCheck_2179_;
goto v_resetjp_2137_;
}
v_resetjp_2137_:
{
lean_object* v_usedTheorems_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2177_; 
v_usedTheorems_2140_ = lean_ctor_get(v___y_2032_, 0);
v_isSharedCheck_2177_ = !lean_is_exclusive(v___y_2032_);
if (v_isSharedCheck_2177_ == 0)
{
lean_object* v_unused_2178_; 
v_unused_2178_ = lean_ctor_get(v___y_2032_, 1);
lean_dec(v_unused_2178_);
v___x_2142_ = v___y_2032_;
v_isShared_2143_ = v_isSharedCheck_2177_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_usedTheorems_2140_);
lean_dec(v___y_2032_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2177_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v___x_2145_; 
if (v_isShared_2139_ == 0)
{
lean_ctor_set(v___x_2138_, 0, v_snd_1987_);
v___x_2145_ = v___x_2138_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v_snd_1987_);
v___x_2145_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
lean_object* v___x_2146_; 
lean_inc_ref(v___y_2035_);
v___x_2146_ = lp_batteries_Batteries_Tactic_Lint_formatLemmas(v_usedTheorems_2140_, v___y_2035_, v___x_2145_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
lean_dec_ref(v___x_2145_);
lean_dec_ref(v_usedTheorems_2140_);
if (lean_obj_tag(v___x_2146_) == 0)
{
lean_object* v_a_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2167_; 
v_a_2147_ = lean_ctor_get(v___x_2146_, 0);
v_isSharedCheck_2167_ = !lean_is_exclusive(v___x_2146_);
if (v_isSharedCheck_2167_ == 0)
{
v___x_2149_ = v___x_2146_;
v_isShared_2150_ = v_isSharedCheck_2167_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_a_2147_);
lean_dec(v___x_2146_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2167_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2154_; 
lean_inc_ref(v___y_2035_);
v___x_2151_ = l_Lean_stringToMessageData(v___y_2035_);
v___x_2152_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__11);
if (v_isShared_2143_ == 0)
{
lean_ctor_set_tag(v___x_2142_, 7);
lean_ctor_set(v___x_2142_, 1, v___x_2152_);
lean_ctor_set(v___x_2142_, 0, v___x_2151_);
v___x_2154_ = v___x_2142_;
goto v_reusejp_2153_;
}
else
{
lean_object* v_reuseFailAlloc_2166_; 
v_reuseFailAlloc_2166_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2166_, 0, v___x_2151_);
lean_ctor_set(v_reuseFailAlloc_2166_, 1, v___x_2152_);
v___x_2154_ = v_reuseFailAlloc_2166_;
goto v_reusejp_2153_;
}
v_reusejp_2153_:
{
lean_object* v___x_2156_; 
if (v_isShared_2018_ == 0)
{
lean_ctor_set_tag(v___x_2017_, 7);
lean_ctor_set(v___x_2017_, 1, v_a_2147_);
lean_ctor_set(v___x_2017_, 0, v___x_2154_);
v___x_2156_ = v___x_2017_;
goto v_reusejp_2155_;
}
else
{
lean_object* v_reuseFailAlloc_2165_; 
v_reuseFailAlloc_2165_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2165_, 0, v___x_2154_);
lean_ctor_set(v_reuseFailAlloc_2165_, 1, v_a_2147_);
v___x_2156_ = v_reuseFailAlloc_2165_;
goto v_reusejp_2155_;
}
v_reusejp_2155_:
{
lean_object* v___x_2157_; lean_object* v___x_2159_; 
v___x_2157_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__13);
if (v_isShared_1990_ == 0)
{
lean_ctor_set_tag(v___x_1989_, 7);
lean_ctor_set(v___x_1989_, 1, v___x_2157_);
lean_ctor_set(v___x_1989_, 0, v___x_2156_);
v___x_2159_ = v___x_1989_;
goto v_reusejp_2158_;
}
else
{
lean_object* v_reuseFailAlloc_2164_; 
v_reuseFailAlloc_2164_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2164_, 0, v___x_2156_);
lean_ctor_set(v_reuseFailAlloc_2164_, 1, v___x_2157_);
v___x_2159_ = v_reuseFailAlloc_2164_;
goto v_reusejp_2158_;
}
v_reusejp_2158_:
{
lean_object* v___x_2160_; lean_object* v___x_2162_; 
v___x_2160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2160_, 0, v___x_2159_);
if (v_isShared_2150_ == 0)
{
lean_ctor_set(v___x_2149_, 0, v___x_2160_);
v___x_2162_ = v___x_2149_;
goto v_reusejp_2161_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v___x_2160_);
v___x_2162_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2161_;
}
v_reusejp_2161_:
{
return v___x_2162_;
}
}
}
}
}
}
else
{
lean_object* v_a_2168_; lean_object* v___x_2170_; uint8_t v_isShared_2171_; uint8_t v_isSharedCheck_2175_; 
lean_del_object(v___x_2142_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_1989_);
v_a_2168_ = lean_ctor_get(v___x_2146_, 0);
v_isSharedCheck_2175_ = !lean_is_exclusive(v___x_2146_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2170_ = v___x_2146_;
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
else
{
lean_inc(v_a_2168_);
lean_dec(v___x_2146_);
v___x_2170_ = lean_box(0);
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
v_resetjp_2169_:
{
lean_object* v___x_2173_; 
if (v_isShared_2171_ == 0)
{
v___x_2173_ = v___x_2170_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v_a_2168_);
v___x_2173_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
return v___x_2173_;
}
}
}
}
}
}
}
}
}
v___jp_2181_:
{
if (v___y_2182_ == 0)
{
lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; 
lean_del_object(v___x_2024_);
v___x_2183_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16, &lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__16);
v___x_2184_ = lean_box(v_a_1964_);
lean_inc(v_a_2004_);
v___x_2185_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__0___boxed), 10, 5);
lean_closure_set(v___x_2185_, 0, v_a_2004_);
lean_closure_set(v___x_2185_, 1, v___x_2184_);
lean_closure_set(v___x_2185_, 2, v_rhs_1974_);
lean_closure_set(v___x_2185_, 3, v_a_2002_);
lean_closure_set(v___x_2185_, 4, v_snd_2015_);
v___x_2186_ = lp_batteries_Batteries_Tactic_Lint_decorateError___redArg(v___x_2183_, v___x_2185_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2186_) == 0)
{
lean_object* v_a_2187_; lean_object* v_fst_2188_; lean_object* v_snd_2189_; lean_object* v_expr_2190_; lean_object* v___x_2191_; 
v_a_2187_ = lean_ctor_get(v___x_2186_, 0);
lean_inc(v_a_2187_);
lean_dec_ref_known(v___x_2186_, 1);
v_fst_2188_ = lean_ctor_get(v_a_2187_, 0);
lean_inc(v_fst_2188_);
v_snd_2189_ = lean_ctor_get(v_a_2187_, 1);
lean_inc(v_snd_2189_);
lean_dec(v_a_2187_);
v_expr_2190_ = lean_ctor_get(v_fst_2188_, 0);
lean_inc_ref(v_expr_2190_);
lean_dec(v_fst_2188_);
lean_inc_ref(v_expr_2019_);
v___x_2191_ = lp_batteries_Batteries_Tactic_Lint_isSimpEq(v_expr_2019_, v_expr_2190_, v___x_1977_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2191_) == 0)
{
lean_object* v_a_2192_; lean_object* v___x_2193_; 
v_a_2192_ = lean_ctor_get(v___x_2191_, 0);
lean_inc(v_a_2192_);
lean_dec_ref_known(v___x_2191_, 1);
lean_inc_ref(v_lhs_1973_);
lean_inc_ref(v_expr_2019_);
v___x_2193_ = lp_batteries_Batteries_Tactic_Lint_isSimpEq(v_expr_2019_, v_lhs_1973_, v_a_1964_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_);
if (lean_obj_tag(v___x_2193_) == 0)
{
uint8_t v___x_2194_; 
v___x_2194_ = lean_unbox(v_a_2004_);
if (v___x_2194_ == 0)
{
lean_object* v_a_2195_; lean_object* v___x_2196_; uint8_t v___x_2197_; uint8_t v___x_2198_; 
v_a_2195_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2195_);
lean_dec_ref_known(v___x_2193_, 1);
v___x_2196_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__17));
v___x_2197_ = lean_unbox(v_a_2195_);
lean_dec(v_a_2195_);
v___x_2198_ = lean_unbox(v_a_2192_);
lean_dec(v_a_2192_);
v___y_2032_ = v_snd_2189_;
v___y_2033_ = v___x_2197_;
v___y_2034_ = v___x_2198_;
v___y_2035_ = v___x_2196_;
goto v___jp_2031_;
}
else
{
lean_object* v_a_2199_; lean_object* v___x_2200_; uint8_t v___x_2201_; uint8_t v___x_2202_; 
v_a_2199_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2199_);
lean_dec_ref_known(v___x_2193_, 1);
v___x_2200_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___closed__18));
v___x_2201_ = lean_unbox(v_a_2199_);
lean_dec(v_a_2199_);
v___x_2202_ = lean_unbox(v_a_2192_);
lean_dec(v_a_2192_);
v___y_2032_ = v_snd_2189_;
v___y_2033_ = v___x_2201_;
v___y_2034_ = v___x_2202_;
v___y_2035_ = v___x_2200_;
goto v___jp_2031_;
}
}
else
{
lean_object* v_a_2203_; lean_object* v___x_2205_; uint8_t v_isShared_2206_; uint8_t v_isSharedCheck_2210_; 
lean_dec(v_a_2192_);
lean_dec(v_snd_2189_);
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
v_a_2203_ = lean_ctor_get(v___x_2193_, 0);
v_isSharedCheck_2210_ = !lean_is_exclusive(v___x_2193_);
if (v_isSharedCheck_2210_ == 0)
{
v___x_2205_ = v___x_2193_;
v_isShared_2206_ = v_isSharedCheck_2210_;
goto v_resetjp_2204_;
}
else
{
lean_inc(v_a_2203_);
lean_dec(v___x_2193_);
v___x_2205_ = lean_box(0);
v_isShared_2206_ = v_isSharedCheck_2210_;
goto v_resetjp_2204_;
}
v_resetjp_2204_:
{
lean_object* v___x_2208_; 
if (v_isShared_2206_ == 0)
{
v___x_2208_ = v___x_2205_;
goto v_reusejp_2207_;
}
else
{
lean_object* v_reuseFailAlloc_2209_; 
v_reuseFailAlloc_2209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2209_, 0, v_a_2203_);
v___x_2208_ = v_reuseFailAlloc_2209_;
goto v_reusejp_2207_;
}
v_reusejp_2207_:
{
return v___x_2208_;
}
}
}
}
else
{
lean_object* v_a_2211_; lean_object* v___x_2213_; uint8_t v_isShared_2214_; uint8_t v_isSharedCheck_2218_; 
lean_dec(v_snd_2189_);
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
v_a_2211_ = lean_ctor_get(v___x_2191_, 0);
v_isSharedCheck_2218_ = !lean_is_exclusive(v___x_2191_);
if (v_isSharedCheck_2218_ == 0)
{
v___x_2213_ = v___x_2191_;
v_isShared_2214_ = v_isSharedCheck_2218_;
goto v_resetjp_2212_;
}
else
{
lean_inc(v_a_2211_);
lean_dec(v___x_2191_);
v___x_2213_ = lean_box(0);
v_isShared_2214_ = v_isSharedCheck_2218_;
goto v_resetjp_2212_;
}
v_resetjp_2212_:
{
lean_object* v___x_2216_; 
if (v_isShared_2214_ == 0)
{
v___x_2216_ = v___x_2213_;
goto v_reusejp_2215_;
}
else
{
lean_object* v_reuseFailAlloc_2217_; 
v_reuseFailAlloc_2217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2217_, 0, v_a_2211_);
v___x_2216_ = v_reuseFailAlloc_2217_;
goto v_reusejp_2215_;
}
v_reusejp_2215_:
{
return v___x_2216_;
}
}
}
}
else
{
lean_object* v_a_2219_; lean_object* v___x_2221_; uint8_t v_isShared_2222_; uint8_t v_isSharedCheck_2226_; 
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
v_a_2219_ = lean_ctor_get(v___x_2186_, 0);
v_isSharedCheck_2226_ = !lean_is_exclusive(v___x_2186_);
if (v_isSharedCheck_2226_ == 0)
{
v___x_2221_ = v___x_2186_;
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
else
{
lean_inc(v_a_2219_);
lean_dec(v___x_2186_);
v___x_2221_ = lean_box(0);
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
v_resetjp_2220_:
{
lean_object* v___x_2224_; 
if (v_isShared_2222_ == 0)
{
v___x_2224_ = v___x_2221_;
goto v_reusejp_2223_;
}
else
{
lean_object* v_reuseFailAlloc_2225_; 
v_reuseFailAlloc_2225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2225_, 0, v_a_2219_);
v___x_2224_ = v_reuseFailAlloc_2225_;
goto v_reusejp_2223_;
}
v_reusejp_2223_:
{
return v___x_2224_;
}
}
}
}
else
{
lean_dec_ref(v_usedTheorems_2030_);
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_dec(v_snd_2015_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_dec(v_a_2002_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
goto v___jp_2026_;
}
}
}
}
else
{
lean_object* v_a_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2241_; 
lean_dec(v_proof_x3f_2020_);
lean_dec_ref(v_expr_2019_);
lean_del_object(v___x_2017_);
lean_dec(v_snd_2015_);
lean_del_object(v___x_2012_);
lean_dec(v_a_2004_);
lean_dec(v_a_2002_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2234_ = lean_ctor_get(v___x_2021_, 0);
v_isSharedCheck_2241_ = !lean_is_exclusive(v___x_2021_);
if (v_isSharedCheck_2241_ == 0)
{
v___x_2236_ = v___x_2021_;
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_a_2234_);
lean_dec(v___x_2021_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
v_resetjp_2235_:
{
lean_object* v___x_2239_; 
if (v_isShared_2237_ == 0)
{
v___x_2239_ = v___x_2236_;
goto v_reusejp_2238_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v_a_2234_);
v___x_2239_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2238_;
}
v_reusejp_2238_:
{
return v___x_2239_;
}
}
}
}
}
}
else
{
lean_object* v_a_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2251_; 
lean_dec(v_a_2004_);
lean_dec(v_a_2002_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2244_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2246_ = v___x_2009_;
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_a_2244_);
lean_dec(v___x_2009_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2249_; 
if (v_isShared_2247_ == 0)
{
v___x_2249_ = v___x_2246_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v_a_2244_);
v___x_2249_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
return v___x_2249_;
}
}
}
}
else
{
lean_object* v_a_2252_; lean_object* v___x_2254_; uint8_t v_isShared_2255_; uint8_t v_isSharedCheck_2259_; 
lean_dec(v_a_2002_);
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2252_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2259_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2259_ == 0)
{
v___x_2254_ = v___x_2003_;
v_isShared_2255_ = v_isSharedCheck_2259_;
goto v_resetjp_2253_;
}
else
{
lean_inc(v_a_2252_);
lean_dec(v___x_2003_);
v___x_2254_ = lean_box(0);
v_isShared_2255_ = v_isSharedCheck_2259_;
goto v_resetjp_2253_;
}
v_resetjp_2253_:
{
lean_object* v___x_2257_; 
if (v_isShared_2255_ == 0)
{
v___x_2257_ = v___x_2254_;
goto v_reusejp_2256_;
}
else
{
lean_object* v_reuseFailAlloc_2258_; 
v_reuseFailAlloc_2258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2258_, 0, v_a_2252_);
v___x_2257_ = v_reuseFailAlloc_2258_;
goto v_reusejp_2256_;
}
v_reusejp_2256_:
{
return v___x_2257_;
}
}
}
}
else
{
lean_object* v_a_2260_; lean_object* v___x_2262_; uint8_t v_isShared_2263_; uint8_t v_isSharedCheck_2267_; 
lean_del_object(v___x_1989_);
lean_dec(v_snd_1987_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2260_ = lean_ctor_get(v___x_2001_, 0);
v_isSharedCheck_2267_ = !lean_is_exclusive(v___x_2001_);
if (v_isSharedCheck_2267_ == 0)
{
v___x_2262_ = v___x_2001_;
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
else
{
lean_inc(v_a_2260_);
lean_dec(v___x_2001_);
v___x_2262_ = lean_box(0);
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
v_resetjp_2261_:
{
lean_object* v___x_2265_; 
if (v_isShared_2263_ == 0)
{
v___x_2265_ = v___x_2262_;
goto v_reusejp_2264_;
}
else
{
lean_object* v_reuseFailAlloc_2266_; 
v_reuseFailAlloc_2266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2266_, 0, v_a_2260_);
v___x_2265_ = v_reuseFailAlloc_2266_;
goto v_reusejp_2264_;
}
v_reusejp_2264_:
{
return v___x_2265_;
}
}
}
}
}
else
{
lean_object* v_a_2269_; lean_object* v___x_2271_; uint8_t v_isShared_2272_; uint8_t v_isSharedCheck_2276_; 
lean_dec(v_a_1983_);
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2269_ = lean_ctor_get(v___x_1984_, 0);
v_isSharedCheck_2276_ = !lean_is_exclusive(v___x_1984_);
if (v_isSharedCheck_2276_ == 0)
{
v___x_2271_ = v___x_1984_;
v_isShared_2272_ = v_isSharedCheck_2276_;
goto v_resetjp_2270_;
}
else
{
lean_inc(v_a_2269_);
lean_dec(v___x_1984_);
v___x_2271_ = lean_box(0);
v_isShared_2272_ = v_isSharedCheck_2276_;
goto v_resetjp_2270_;
}
v_resetjp_2270_:
{
lean_object* v___x_2274_; 
if (v_isShared_2272_ == 0)
{
v___x_2274_ = v___x_2271_;
goto v_reusejp_2273_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v_a_2269_);
v___x_2274_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2273_;
}
v_reusejp_2273_:
{
return v___x_2274_;
}
}
}
}
else
{
lean_object* v_a_2277_; lean_object* v___x_2279_; uint8_t v_isShared_2280_; uint8_t v_isSharedCheck_2284_; 
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2277_ = lean_ctor_get(v___x_1982_, 0);
v_isSharedCheck_2284_ = !lean_is_exclusive(v___x_1982_);
if (v_isSharedCheck_2284_ == 0)
{
v___x_2279_ = v___x_1982_;
v_isShared_2280_ = v_isSharedCheck_2284_;
goto v_resetjp_2278_;
}
else
{
lean_inc(v_a_2277_);
lean_dec(v___x_1982_);
v___x_2279_ = lean_box(0);
v_isShared_2280_ = v_isSharedCheck_2284_;
goto v_resetjp_2278_;
}
v_resetjp_2278_:
{
lean_object* v___x_2282_; 
if (v_isShared_2280_ == 0)
{
v___x_2282_ = v___x_2279_;
goto v_reusejp_2281_;
}
else
{
lean_object* v_reuseFailAlloc_2283_; 
v_reuseFailAlloc_2283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2283_, 0, v_a_2277_);
v___x_2282_ = v_reuseFailAlloc_2283_;
goto v_reusejp_2281_;
}
v_reusejp_2281_:
{
return v___x_2282_;
}
}
}
}
else
{
lean_object* v_a_2285_; lean_object* v___x_2287_; uint8_t v_isShared_2288_; uint8_t v_isSharedCheck_2292_; 
lean_dec_ref(v_rhs_1974_);
lean_dec_ref(v_lhs_1973_);
lean_dec_ref(v_hyps_1972_);
lean_dec(v_declName_1965_);
v_a_2285_ = lean_ctor_get(v___x_1975_, 0);
v_isSharedCheck_2292_ = !lean_is_exclusive(v___x_1975_);
if (v_isSharedCheck_2292_ == 0)
{
v___x_2287_ = v___x_1975_;
v_isShared_2288_ = v_isSharedCheck_2292_;
goto v_resetjp_2286_;
}
else
{
lean_inc(v_a_2285_);
lean_dec(v___x_1975_);
v___x_2287_ = lean_box(0);
v_isShared_2288_ = v_isSharedCheck_2292_;
goto v_resetjp_2286_;
}
v_resetjp_2286_:
{
lean_object* v___x_2290_; 
if (v_isShared_2288_ == 0)
{
v___x_2290_ = v___x_2287_;
goto v_reusejp_2289_;
}
else
{
lean_object* v_reuseFailAlloc_2291_; 
v_reuseFailAlloc_2291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2291_, 0, v_a_2285_);
v___x_2290_ = v_reuseFailAlloc_2291_;
goto v_reusejp_2289_;
}
v_reusejp_2289_:
{
return v___x_2290_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___boxed(lean_object* v_a_2293_, lean_object* v_declName_2294_, lean_object* v_x_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_){
_start:
{
uint8_t v_a_28653__boxed_2301_; lean_object* v_res_2302_; 
v_a_28653__boxed_2301_ = lean_unbox(v_a_2293_);
v_res_2302_ = lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1(v_a_28653__boxed_2301_, v_declName_2294_, v_x_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
return v_res_2302_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2(lean_object* v_declName_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_, lean_object* v___y_2307_){
_start:
{
lean_object* v___x_2309_; 
lean_inc(v_declName_2303_);
v___x_2309_ = lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(v_declName_2303_, v___y_2307_);
if (lean_obj_tag(v___x_2309_) == 0)
{
lean_object* v_a_2310_; lean_object* v___x_2312_; uint8_t v_isShared_2313_; uint8_t v_isSharedCheck_2357_; 
v_a_2310_ = lean_ctor_get(v___x_2309_, 0);
v_isSharedCheck_2357_ = !lean_is_exclusive(v___x_2309_);
if (v_isSharedCheck_2357_ == 0)
{
v___x_2312_ = v___x_2309_;
v_isShared_2313_ = v_isSharedCheck_2357_;
goto v_resetjp_2311_;
}
else
{
lean_inc(v_a_2310_);
lean_dec(v___x_2309_);
v___x_2312_ = lean_box(0);
v_isShared_2313_ = v_isSharedCheck_2357_;
goto v_resetjp_2311_;
}
v_resetjp_2311_:
{
uint8_t v___x_2314_; 
v___x_2314_ = lean_unbox(v_a_2310_);
if (v___x_2314_ == 0)
{
lean_object* v___x_2315_; lean_object* v___x_2317_; 
lean_dec(v_a_2310_);
lean_dec(v_declName_2303_);
v___x_2315_ = lean_box(0);
if (v_isShared_2313_ == 0)
{
lean_ctor_set(v___x_2312_, 0, v___x_2315_);
v___x_2317_ = v___x_2312_;
goto v_reusejp_2316_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v___x_2315_);
v___x_2317_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2316_;
}
v_reusejp_2316_:
{
return v___x_2317_;
}
}
else
{
lean_object* v___x_2319_; lean_object* v___x_2320_; 
lean_del_object(v___x_2312_);
v___x_2319_ = l_Lean_Name_getPrefix(v_declName_2303_);
v___x_2320_ = l_Lean_Meta_getEqnsFor_x3f(v___x_2319_, v___y_2304_, v___y_2305_, v___y_2306_, v___y_2307_);
if (lean_obj_tag(v___x_2320_) == 0)
{
uint8_t v_trackZetaDelta_2321_; lean_object* v_zetaDeltaSet_2322_; lean_object* v_lctx_2323_; lean_object* v_localInstances_2324_; lean_object* v_defEqCtx_x3f_2325_; lean_object* v_synthPendingDepth_2326_; lean_object* v_customCanUnfoldPredicate_x3f_2327_; uint8_t v_univApprox_2328_; uint8_t v_inTypeClassResolution_2329_; uint8_t v_cacheInferType_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; uint64_t v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; 
lean_dec_ref_known(v___x_2320_, 1);
v_trackZetaDelta_2321_ = lean_ctor_get_uint8(v___y_2304_, sizeof(void*)*7);
v_zetaDeltaSet_2322_ = lean_ctor_get(v___y_2304_, 1);
v_lctx_2323_ = lean_ctor_get(v___y_2304_, 2);
v_localInstances_2324_ = lean_ctor_get(v___y_2304_, 3);
v_defEqCtx_x3f_2325_ = lean_ctor_get(v___y_2304_, 4);
v_synthPendingDepth_2326_ = lean_ctor_get(v___y_2304_, 5);
v_customCanUnfoldPredicate_x3f_2327_ = lean_ctor_get(v___y_2304_, 6);
v_univApprox_2328_ = lean_ctor_get_uint8(v___y_2304_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2329_ = lean_ctor_get_uint8(v___y_2304_, sizeof(void*)*7 + 2);
v_cacheInferType_2330_ = lean_ctor_get_uint8(v___y_2304_, sizeof(void*)*7 + 3);
v___x_2331_ = l_Lean_Meta_Context_config(v___y_2304_);
v___x_2332_ = l_Lean_Elab_Term_setElabConfig(v___x_2331_);
v___x_2333_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_2332_);
v___x_2334_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_2334_, 0, v___x_2332_);
lean_ctor_set_uint64(v___x_2334_, sizeof(void*)*1, v___x_2333_);
lean_inc(v_customCanUnfoldPredicate_x3f_2327_);
lean_inc(v_synthPendingDepth_2326_);
lean_inc(v_defEqCtx_x3f_2325_);
lean_inc_ref(v_localInstances_2324_);
lean_inc_ref(v_lctx_2323_);
lean_inc(v_zetaDeltaSet_2322_);
v___x_2335_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2335_, 0, v___x_2334_);
lean_ctor_set(v___x_2335_, 1, v_zetaDeltaSet_2322_);
lean_ctor_set(v___x_2335_, 2, v_lctx_2323_);
lean_ctor_set(v___x_2335_, 3, v_localInstances_2324_);
lean_ctor_set(v___x_2335_, 4, v_defEqCtx_x3f_2325_);
lean_ctor_set(v___x_2335_, 5, v_synthPendingDepth_2326_);
lean_ctor_set(v___x_2335_, 6, v_customCanUnfoldPredicate_x3f_2327_);
lean_ctor_set_uint8(v___x_2335_, sizeof(void*)*7, v_trackZetaDelta_2321_);
lean_ctor_set_uint8(v___x_2335_, sizeof(void*)*7 + 1, v_univApprox_2328_);
lean_ctor_set_uint8(v___x_2335_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2329_);
lean_ctor_set_uint8(v___x_2335_, sizeof(void*)*7 + 3, v_cacheInferType_2330_);
lean_inc(v_declName_2303_);
v___x_2336_ = l_Lean_getConstInfo___at___00Lean_Meta_mkSimpEntryOfDeclToUnfold_spec__0(v_declName_2303_, v___x_2335_, v___y_2305_, v___y_2306_, v___y_2307_);
if (lean_obj_tag(v___x_2336_) == 0)
{
lean_object* v_a_2337_; lean_object* v___f_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; 
v_a_2337_ = lean_ctor_get(v___x_2336_, 0);
lean_inc(v_a_2337_);
lean_dec_ref_known(v___x_2336_, 1);
v___f_2338_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_simpNF___lam__1___boxed), 8, 2);
lean_closure_set(v___f_2338_, 0, v_a_2310_);
lean_closure_set(v___f_2338_, 1, v_declName_2303_);
v___x_2339_ = l_Lean_ConstantInfo_type(v_a_2337_);
lean_dec(v_a_2337_);
v___x_2340_ = lp_batteries_Batteries_Tactic_Lint_checkAllSimpTheoremInfos(v___x_2339_, v___f_2338_, v___x_2335_, v___y_2305_, v___y_2306_, v___y_2307_);
lean_dec_ref_known(v___x_2335_, 7);
return v___x_2340_;
}
else
{
lean_object* v_a_2341_; lean_object* v___x_2343_; uint8_t v_isShared_2344_; uint8_t v_isSharedCheck_2348_; 
lean_dec_ref_known(v___x_2335_, 7);
lean_dec(v_a_2310_);
lean_dec(v_declName_2303_);
v_a_2341_ = lean_ctor_get(v___x_2336_, 0);
v_isSharedCheck_2348_ = !lean_is_exclusive(v___x_2336_);
if (v_isSharedCheck_2348_ == 0)
{
v___x_2343_ = v___x_2336_;
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
else
{
lean_inc(v_a_2341_);
lean_dec(v___x_2336_);
v___x_2343_ = lean_box(0);
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
v_resetjp_2342_:
{
lean_object* v___x_2346_; 
if (v_isShared_2344_ == 0)
{
v___x_2346_ = v___x_2343_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2347_; 
v_reuseFailAlloc_2347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2347_, 0, v_a_2341_);
v___x_2346_ = v_reuseFailAlloc_2347_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
return v___x_2346_;
}
}
}
}
else
{
lean_object* v_a_2349_; lean_object* v___x_2351_; uint8_t v_isShared_2352_; uint8_t v_isSharedCheck_2356_; 
lean_dec(v_a_2310_);
lean_dec(v_declName_2303_);
v_a_2349_ = lean_ctor_get(v___x_2320_, 0);
v_isSharedCheck_2356_ = !lean_is_exclusive(v___x_2320_);
if (v_isSharedCheck_2356_ == 0)
{
v___x_2351_ = v___x_2320_;
v_isShared_2352_ = v_isSharedCheck_2356_;
goto v_resetjp_2350_;
}
else
{
lean_inc(v_a_2349_);
lean_dec(v___x_2320_);
v___x_2351_ = lean_box(0);
v_isShared_2352_ = v_isSharedCheck_2356_;
goto v_resetjp_2350_;
}
v_resetjp_2350_:
{
lean_object* v___x_2354_; 
if (v_isShared_2352_ == 0)
{
v___x_2354_ = v___x_2351_;
goto v_reusejp_2353_;
}
else
{
lean_object* v_reuseFailAlloc_2355_; 
v_reuseFailAlloc_2355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2355_, 0, v_a_2349_);
v___x_2354_ = v_reuseFailAlloc_2355_;
goto v_reusejp_2353_;
}
v_reusejp_2353_:
{
return v___x_2354_;
}
}
}
}
}
}
else
{
lean_object* v_a_2358_; lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2365_; 
lean_dec(v_declName_2303_);
v_a_2358_ = lean_ctor_get(v___x_2309_, 0);
v_isSharedCheck_2365_ = !lean_is_exclusive(v___x_2309_);
if (v_isSharedCheck_2365_ == 0)
{
v___x_2360_ = v___x_2309_;
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
else
{
lean_inc(v_a_2358_);
lean_dec(v___x_2309_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v___x_2363_; 
if (v_isShared_2361_ == 0)
{
v___x_2363_ = v___x_2360_;
goto v_reusejp_2362_;
}
else
{
lean_object* v_reuseFailAlloc_2364_; 
v_reuseFailAlloc_2364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2364_, 0, v_a_2358_);
v___x_2363_ = v_reuseFailAlloc_2364_;
goto v_reusejp_2362_;
}
v_reusejp_2362_:
{
return v___x_2363_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2___boxed(lean_object* v_declName_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_){
_start:
{
lean_object* v_res_2372_; 
v_res_2372_ = lp_batteries_Batteries_Tactic_Lint_simpNF___lam__2(v_declName_2366_, v___y_2367_, v___y_2368_, v___y_2369_, v___y_2370_);
lean_dec(v___y_2370_);
lean_dec_ref(v___y_2369_);
lean_dec(v___y_2368_);
lean_dec_ref(v___y_2367_);
return v_res_2372_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3(void){
_start:
{
lean_object* v___x_2377_; lean_object* v___x_2378_; 
v___x_2377_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___closed__2));
v___x_2378_ = l_Lean_MessageData_ofFormat(v___x_2377_);
return v___x_2378_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6(void){
_start:
{
lean_object* v___x_2382_; lean_object* v___x_2383_; 
v___x_2382_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___closed__5));
v___x_2383_ = l_Lean_MessageData_ofFormat(v___x_2382_);
return v___x_2383_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7(void){
_start:
{
uint8_t v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___f_2387_; lean_object* v___x_2388_; 
v___x_2384_ = 1;
v___x_2385_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6, &lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__6);
v___x_2386_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3, &lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__3);
v___f_2387_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpNF___closed__0));
v___x_2388_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_2388_, 0, v___f_2387_);
lean_ctor_set(v___x_2388_, 1, v___x_2386_);
lean_ctor_set(v___x_2388_, 2, v___x_2385_);
lean_ctor_set_uint8(v___x_2388_, sizeof(void*)*3, v___x_2384_);
return v___x_2388_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpNF(void){
_start:
{
lean_object* v___x_2389_; 
v___x_2389_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7, &lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_simpNF___closed__7);
return v___x_2389_;
}
}
static lean_object* _init_lp_batteries_LibraryNote_simp_x2dnormal__form(void){
_start:
{
lean_object* v___x_2390_; 
v___x_2390_ = lean_box(0);
return v___x_2390_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f(lean_object* v_x_2392_){
_start:
{
if (lean_obj_tag(v_x_2392_) == 5)
{
lean_object* v_fn_2393_; 
v_fn_2393_ = lean_ctor_get(v_x_2392_, 0);
if (lean_obj_tag(v_fn_2393_) == 5)
{
lean_object* v_fn_2394_; 
v_fn_2394_ = lean_ctor_get(v_fn_2393_, 0);
switch(lean_obj_tag(v_fn_2394_))
{
case 5:
{
lean_object* v_fn_2395_; 
v_fn_2395_ = lean_ctor_get(v_fn_2394_, 0);
if (lean_obj_tag(v_fn_2395_) == 4)
{
lean_object* v_declName_2396_; 
v_declName_2396_ = lean_ctor_get(v_fn_2395_, 0);
if (lean_obj_tag(v_declName_2396_) == 1)
{
lean_object* v_pre_2397_; 
v_pre_2397_ = lean_ctor_get(v_declName_2396_, 0);
if (lean_obj_tag(v_pre_2397_) == 0)
{
lean_object* v_arg_2398_; lean_object* v_arg_2399_; lean_object* v_str_2400_; lean_object* v___x_2401_; uint8_t v___x_2402_; 
v_arg_2398_ = lean_ctor_get(v_x_2392_, 1);
v_arg_2399_ = lean_ctor_get(v_fn_2393_, 1);
v_str_2400_ = lean_ctor_get(v_declName_2396_, 1);
v___x_2401_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__0));
v___x_2402_ = lean_string_dec_eq(v_str_2400_, v___x_2401_);
if (v___x_2402_ == 0)
{
lean_object* v___x_2403_; 
v___x_2403_ = lean_box(0);
return v___x_2403_;
}
else
{
lean_object* v___x_2404_; lean_object* v___x_2405_; 
lean_inc_ref(v_arg_2398_);
lean_inc_ref(v_arg_2399_);
v___x_2404_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2404_, 0, v_arg_2399_);
lean_ctor_set(v___x_2404_, 1, v_arg_2398_);
v___x_2405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2405_, 0, v___x_2404_);
return v___x_2405_;
}
}
else
{
lean_object* v___x_2406_; 
v___x_2406_ = lean_box(0);
return v___x_2406_;
}
}
else
{
lean_object* v___x_2407_; 
v___x_2407_ = lean_box(0);
return v___x_2407_;
}
}
else
{
lean_object* v___x_2408_; 
v___x_2408_ = lean_box(0);
return v___x_2408_;
}
}
case 4:
{
lean_object* v_declName_2409_; 
v_declName_2409_ = lean_ctor_get(v_fn_2394_, 0);
if (lean_obj_tag(v_declName_2409_) == 1)
{
lean_object* v_pre_2410_; 
v_pre_2410_ = lean_ctor_get(v_declName_2409_, 0);
if (lean_obj_tag(v_pre_2410_) == 0)
{
lean_object* v_arg_2411_; lean_object* v_arg_2412_; lean_object* v_str_2413_; lean_object* v___x_2414_; uint8_t v___x_2415_; 
v_arg_2411_ = lean_ctor_get(v_x_2392_, 1);
v_arg_2412_ = lean_ctor_get(v_fn_2393_, 1);
v_str_2413_ = lean_ctor_get(v_declName_2409_, 1);
v___x_2414_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___closed__0));
v___x_2415_ = lean_string_dec_eq(v_str_2413_, v___x_2414_);
if (v___x_2415_ == 0)
{
lean_object* v___x_2416_; 
v___x_2416_ = lean_box(0);
return v___x_2416_;
}
else
{
lean_object* v___x_2417_; lean_object* v___x_2418_; 
lean_inc_ref(v_arg_2411_);
lean_inc_ref(v_arg_2412_);
v___x_2417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2417_, 0, v_arg_2412_);
lean_ctor_set(v___x_2417_, 1, v_arg_2411_);
v___x_2418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2418_, 0, v___x_2417_);
return v___x_2418_;
}
}
else
{
lean_object* v___x_2419_; 
v___x_2419_ = lean_box(0);
return v___x_2419_;
}
}
else
{
lean_object* v___x_2420_; 
v___x_2420_ = lean_box(0);
return v___x_2420_;
}
}
default: 
{
lean_object* v___x_2421_; 
v___x_2421_ = lean_box(0);
return v___x_2421_;
}
}
}
else
{
lean_object* v___x_2422_; 
v___x_2422_ = lean_box(0);
return v___x_2422_;
}
}
else
{
lean_object* v___x_2423_; 
v___x_2423_ = lean_box(0);
return v___x_2423_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f___boxed(lean_object* v_x_2424_){
_start:
{
lean_object* v_res_2425_; 
v_res_2425_ = lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_Expr_eqOrIff_x3f(v_x_2424_);
lean_dec_ref(v_x_2424_);
return v_res_2425_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0(lean_object* v_snd_2426_, lean_object* v_fst_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_, lean_object* v___y_2430_, lean_object* v___y_2431_){
_start:
{
lean_object* v___x_2433_; 
v___x_2433_ = l_Lean_Meta_isExprDefEq(v_snd_2426_, v_fst_2427_, v___y_2428_, v___y_2429_, v___y_2430_, v___y_2431_);
return v___x_2433_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0___boxed(lean_object* v_snd_2434_, lean_object* v_fst_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_){
_start:
{
lean_object* v_res_2441_; 
v_res_2441_ = lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0(v_snd_2434_, v_fst_2435_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_);
lean_dec(v___y_2439_);
lean_dec_ref(v___y_2438_);
lean_dec(v___y_2437_);
lean_dec_ref(v___y_2436_);
return v_res_2441_;
}
}
static lean_object* _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0(void){
_start:
{
lean_object* v___x_2442_; 
v___x_2442_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_2442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3(lean_object* v_msg_2443_){
_start:
{
lean_object* v___x_2444_; lean_object* v___x_2445_; 
v___x_2444_ = lean_obj_once(&lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0, &lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0_once, _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3___closed__0);
v___x_2445_ = lean_panic_fn_borrowed(v___x_2444_, v_msg_2443_);
return v___x_2445_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(lean_object* v_a_2446_, lean_object* v_b_2447_){
_start:
{
lean_object* v_fst_2448_; lean_object* v_fst_2449_; uint8_t v___x_2450_; 
v_fst_2448_ = lean_ctor_get(v_a_2446_, 0);
v_fst_2449_ = lean_ctor_get(v_b_2447_, 0);
v___x_2450_ = l_Lean_Meta_DiscrTree_Key_lt(v_fst_2448_, v_fst_2449_);
return v___x_2450_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1___boxed(lean_object* v_a_2451_, lean_object* v_b_2452_){
_start:
{
uint8_t v_res_2453_; lean_object* v_r_2454_; 
v_res_2453_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v_a_2451_, v_b_2452_);
lean_dec_ref(v_b_2452_);
lean_dec_ref(v_a_2451_);
v_r_2454_ = lean_box(v_res_2453_);
return v_r_2454_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(lean_object* v_x_2455_, lean_object* v_keys_2456_, lean_object* v_v_2457_, lean_object* v_k_2458_, lean_object* v_x_2459_){
_start:
{
lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v_c_2462_; lean_object* v___x_2463_; 
v___x_2460_ = lean_unsigned_to_nat(1u);
v___x_2461_ = lean_nat_add(v_x_2455_, v___x_2460_);
v_c_2462_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_2456_, v_v_2457_, v___x_2461_);
lean_dec(v___x_2461_);
v___x_2463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2463_, 0, v_k_2458_);
lean_ctor_set(v___x_2463_, 1, v_c_2462_);
return v___x_2463_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0___boxed(lean_object* v_x_2464_, lean_object* v_keys_2465_, lean_object* v_v_2466_, lean_object* v_k_2467_, lean_object* v_x_2468_){
_start:
{
lean_object* v_res_2469_; 
v_res_2469_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(v_x_2464_, v_keys_2465_, v_v_2466_, v_k_2467_, v_x_2468_);
lean_dec_ref(v_keys_2465_);
lean_dec(v_x_2464_);
return v_res_2469_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_vs_2470_, lean_object* v_v_2471_, lean_object* v_i_2472_){
_start:
{
lean_object* v___x_2473_; uint8_t v___x_2474_; 
v___x_2473_ = lean_array_get_size(v_vs_2470_);
v___x_2474_ = lean_nat_dec_lt(v_i_2472_, v___x_2473_);
if (v___x_2474_ == 0)
{
lean_object* v___x_2475_; 
lean_dec(v_i_2472_);
v___x_2475_ = lean_array_push(v_vs_2470_, v_v_2471_);
return v___x_2475_;
}
else
{
if (v___x_2474_ == 0)
{
lean_object* v___x_2476_; lean_object* v___x_2477_; 
v___x_2476_ = lean_unsigned_to_nat(1u);
v___x_2477_ = lean_nat_add(v_i_2472_, v___x_2476_);
lean_dec(v_i_2472_);
v_i_2472_ = v___x_2477_;
goto _start;
}
else
{
lean_object* v___x_2479_; 
v___x_2479_ = lean_array_fset(v_vs_2470_, v_i_2472_, v_v_2471_);
lean_dec(v_i_2472_);
return v___x_2479_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2(lean_object* v_vs_2480_, lean_object* v_v_2481_){
_start:
{
lean_object* v___x_2482_; lean_object* v___x_2483_; 
v___x_2482_ = lean_unsigned_to_nat(0u);
v___x_2483_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2_spec__4(v_vs_2480_, v_v_2481_, v___x_2482_);
return v___x_2483_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg(lean_object* v_x_2488_, lean_object* v_keys_2489_, lean_object* v_v_2490_, lean_object* v_k_2491_, lean_object* v_as_2492_, lean_object* v_k_2493_, lean_object* v_x_2494_, lean_object* v_x_2495_){
_start:
{
lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v_mid_2498_; lean_object* v_midVal_2499_; uint8_t v___x_2500_; 
v___x_2496_ = lean_nat_add(v_x_2494_, v_x_2495_);
v___x_2497_ = lean_unsigned_to_nat(1u);
v_mid_2498_ = lean_nat_shiftr(v___x_2496_, v___x_2497_);
lean_dec(v___x_2496_);
v_midVal_2499_ = lean_array_fget(v_as_2492_, v_mid_2498_);
v___x_2500_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v_midVal_2499_, v_k_2493_);
if (v___x_2500_ == 0)
{
uint8_t v___x_2501_; 
lean_dec(v_x_2495_);
v___x_2501_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v_k_2493_, v_midVal_2499_);
if (v___x_2501_ == 0)
{
lean_object* v___x_2502_; uint8_t v___x_2503_; 
lean_dec(v_x_2494_);
v___x_2502_ = lean_array_get_size(v_as_2492_);
v___x_2503_ = lean_nat_dec_lt(v_mid_2498_, v___x_2502_);
if (v___x_2503_ == 0)
{
lean_dec(v_midVal_2499_);
lean_dec(v_mid_2498_);
lean_dec(v_k_2491_);
return v_as_2492_;
}
else
{
lean_object* v_snd_2504_; lean_object* v___x_2506_; uint8_t v_isShared_2507_; uint8_t v_isSharedCheck_2516_; 
v_snd_2504_ = lean_ctor_get(v_midVal_2499_, 1);
v_isSharedCheck_2516_ = !lean_is_exclusive(v_midVal_2499_);
if (v_isSharedCheck_2516_ == 0)
{
lean_object* v_unused_2517_; 
v_unused_2517_ = lean_ctor_get(v_midVal_2499_, 0);
lean_dec(v_unused_2517_);
v___x_2506_ = v_midVal_2499_;
v_isShared_2507_ = v_isSharedCheck_2516_;
goto v_resetjp_2505_;
}
else
{
lean_inc(v_snd_2504_);
lean_dec(v_midVal_2499_);
v___x_2506_ = lean_box(0);
v_isShared_2507_ = v_isSharedCheck_2516_;
goto v_resetjp_2505_;
}
v_resetjp_2505_:
{
lean_object* v___x_2508_; lean_object* v_xs_x27_2509_; lean_object* v___x_2510_; lean_object* v_c_2511_; lean_object* v___x_2513_; 
v___x_2508_ = lean_box(0);
v_xs_x27_2509_ = lean_array_fset(v_as_2492_, v_mid_2498_, v___x_2508_);
v___x_2510_ = lean_nat_add(v_x_2488_, v___x_2497_);
v_c_2511_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(v_keys_2489_, v_v_2490_, v___x_2510_, v_snd_2504_);
lean_dec(v___x_2510_);
if (v_isShared_2507_ == 0)
{
lean_ctor_set(v___x_2506_, 1, v_c_2511_);
lean_ctor_set(v___x_2506_, 0, v_k_2491_);
v___x_2513_ = v___x_2506_;
goto v_reusejp_2512_;
}
else
{
lean_object* v_reuseFailAlloc_2515_; 
v_reuseFailAlloc_2515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2515_, 0, v_k_2491_);
lean_ctor_set(v_reuseFailAlloc_2515_, 1, v_c_2511_);
v___x_2513_ = v_reuseFailAlloc_2515_;
goto v_reusejp_2512_;
}
v_reusejp_2512_:
{
lean_object* v___x_2514_; 
v___x_2514_ = lean_array_fset(v_xs_x27_2509_, v_mid_2498_, v___x_2513_);
lean_dec(v_mid_2498_);
return v___x_2514_;
}
}
}
}
else
{
lean_dec(v_midVal_2499_);
v_x_2495_ = v_mid_2498_;
goto _start;
}
}
else
{
uint8_t v___x_2519_; 
lean_dec(v_midVal_2499_);
v___x_2519_ = lean_nat_dec_eq(v_mid_2498_, v_x_2494_);
if (v___x_2519_ == 0)
{
lean_dec(v_x_2494_);
v_x_2494_ = v_mid_2498_;
goto _start;
}
else
{
lean_object* v___x_2521_; lean_object* v_c_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v_j_2525_; lean_object* v_as_2526_; lean_object* v___x_2527_; 
lean_dec(v_mid_2498_);
lean_dec(v_x_2495_);
v___x_2521_ = lean_nat_add(v_x_2488_, v___x_2497_);
v_c_2522_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_2489_, v_v_2490_, v___x_2521_);
lean_dec(v___x_2521_);
v___x_2523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2523_, 0, v_k_2491_);
lean_ctor_set(v___x_2523_, 1, v_c_2522_);
v___x_2524_ = lean_nat_add(v_x_2494_, v___x_2497_);
lean_dec(v_x_2494_);
v_j_2525_ = lean_array_get_size(v_as_2492_);
v_as_2526_ = lean_array_push(v_as_2492_, v___x_2523_);
v___x_2527_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_2524_, v_as_2526_, v_j_2525_);
lean_dec(v___x_2524_);
return v___x_2527_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_2528_, lean_object* v_keys_2529_, lean_object* v_v_2530_, lean_object* v_k_2531_, lean_object* v_as_2532_, lean_object* v_k_2533_){
_start:
{
lean_object* v___x_2534_; lean_object* v___x_2535_; uint8_t v___x_2536_; 
v___x_2534_ = lean_array_get_size(v_as_2532_);
v___x_2535_ = lean_unsigned_to_nat(0u);
v___x_2536_ = lean_nat_dec_eq(v___x_2534_, v___x_2535_);
if (v___x_2536_ == 0)
{
lean_object* v___x_2537_; uint8_t v___x_2538_; 
v___x_2537_ = lean_array_fget_borrowed(v_as_2532_, v___x_2535_);
v___x_2538_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v_k_2533_, v___x_2537_);
if (v___x_2538_ == 0)
{
uint8_t v___x_2539_; 
v___x_2539_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v___x_2537_, v_k_2533_);
if (v___x_2539_ == 0)
{
uint8_t v___x_2540_; 
v___x_2540_ = lean_nat_dec_lt(v___x_2535_, v___x_2534_);
if (v___x_2540_ == 0)
{
lean_dec(v_k_2531_);
return v_as_2532_;
}
else
{
lean_object* v___x_2541_; lean_object* v_xs_x27_2542_; lean_object* v___x_2543_; lean_object* v___x_2544_; 
lean_inc(v___x_2537_);
v___x_2541_ = lean_box(0);
v_xs_x27_2542_ = lean_array_fset(v_as_2532_, v___x_2535_, v___x_2541_);
v___x_2543_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v___x_2537_);
v___x_2544_ = lean_array_fset(v_xs_x27_2542_, v___x_2535_, v___x_2543_);
return v___x_2544_;
}
}
else
{
lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; uint8_t v___x_2548_; 
v___x_2545_ = lean_unsigned_to_nat(1u);
v___x_2546_ = lean_nat_sub(v___x_2534_, v___x_2545_);
v___x_2547_ = lean_array_fget_borrowed(v_as_2532_, v___x_2546_);
v___x_2548_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v___x_2547_, v_k_2533_);
if (v___x_2548_ == 0)
{
uint8_t v___x_2549_; 
v___x_2549_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__1(v_k_2533_, v___x_2547_);
if (v___x_2549_ == 0)
{
uint8_t v___x_2550_; 
v___x_2550_ = lean_nat_dec_lt(v___x_2546_, v___x_2534_);
if (v___x_2550_ == 0)
{
lean_dec(v___x_2546_);
lean_dec(v_k_2531_);
return v_as_2532_;
}
else
{
lean_object* v___x_2551_; lean_object* v_xs_x27_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; 
lean_inc(v___x_2547_);
v___x_2551_ = lean_box(0);
v_xs_x27_2552_ = lean_array_fset(v_as_2532_, v___x_2546_, v___x_2551_);
v___x_2553_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v___x_2547_);
v___x_2554_ = lean_array_fset(v_xs_x27_2552_, v___x_2546_, v___x_2553_);
lean_dec(v___x_2546_);
return v___x_2554_;
}
}
else
{
lean_object* v___x_2555_; 
v___x_2555_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v_as_2532_, v_k_2533_, v___x_2535_, v___x_2546_);
return v___x_2555_;
}
}
else
{
lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; 
lean_dec(v___x_2546_);
v___x_2556_ = lean_box(0);
v___x_2557_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v___x_2556_);
v___x_2558_ = lean_array_push(v_as_2532_, v___x_2557_);
return v___x_2558_;
}
}
}
else
{
lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v_as_2561_; lean_object* v___x_2562_; 
v___x_2559_ = lean_box(0);
v___x_2560_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v___x_2559_);
v_as_2561_ = lean_array_push(v_as_2532_, v___x_2560_);
v___x_2562_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_2535_, v_as_2561_, v___x_2534_);
return v___x_2562_;
}
}
else
{
lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; 
v___x_2563_ = lean_box(0);
v___x_2564_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__0(v_x_2528_, v_keys_2529_, v_v_2530_, v_k_2531_, v___x_2563_);
v___x_2565_ = lean_array_push(v_as_2532_, v___x_2564_);
return v___x_2565_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(lean_object* v_keys_2566_, lean_object* v_v_2567_, lean_object* v_x_2568_, lean_object* v_x_2569_){
_start:
{
lean_object* v_vs_2570_; lean_object* v_children_2571_; lean_object* v___x_2573_; uint8_t v_isShared_2574_; uint8_t v_isSharedCheck_2588_; 
v_vs_2570_ = lean_ctor_get(v_x_2569_, 0);
v_children_2571_ = lean_ctor_get(v_x_2569_, 1);
v_isSharedCheck_2588_ = !lean_is_exclusive(v_x_2569_);
if (v_isSharedCheck_2588_ == 0)
{
v___x_2573_ = v_x_2569_;
v_isShared_2574_ = v_isSharedCheck_2588_;
goto v_resetjp_2572_;
}
else
{
lean_inc(v_children_2571_);
lean_inc(v_vs_2570_);
lean_dec(v_x_2569_);
v___x_2573_ = lean_box(0);
v_isShared_2574_ = v_isSharedCheck_2588_;
goto v_resetjp_2572_;
}
v_resetjp_2572_:
{
lean_object* v___x_2575_; uint8_t v___x_2576_; 
v___x_2575_ = lean_array_get_size(v_keys_2566_);
v___x_2576_ = lean_nat_dec_lt(v_x_2568_, v___x_2575_);
if (v___x_2576_ == 0)
{
lean_object* v___x_2577_; lean_object* v___x_2579_; 
v___x_2577_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__2(v_vs_2570_, v_v_2567_);
if (v_isShared_2574_ == 0)
{
lean_ctor_set(v___x_2573_, 0, v___x_2577_);
v___x_2579_ = v___x_2573_;
goto v_reusejp_2578_;
}
else
{
lean_object* v_reuseFailAlloc_2580_; 
v_reuseFailAlloc_2580_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2580_, 0, v___x_2577_);
lean_ctor_set(v_reuseFailAlloc_2580_, 1, v_children_2571_);
v___x_2579_ = v_reuseFailAlloc_2580_;
goto v_reusejp_2578_;
}
v_reusejp_2578_:
{
return v___x_2579_;
}
}
else
{
lean_object* v_k_2581_; lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v_c_2584_; lean_object* v___x_2586_; 
v_k_2581_ = lean_array_fget_borrowed(v_keys_2566_, v_x_2568_);
v___x_2582_ = ((lean_object*)(lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___closed__1));
lean_inc_n(v_k_2581_, 2);
v___x_2583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2583_, 0, v_k_2581_);
lean_ctor_set(v___x_2583_, 1, v___x_2582_);
v_c_2584_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3(v_x_2568_, v_keys_2566_, v_v_2567_, v_k_2581_, v_children_2571_, v___x_2583_);
lean_dec_ref_known(v___x_2583_, 2);
if (v_isShared_2574_ == 0)
{
lean_ctor_set(v___x_2573_, 1, v_c_2584_);
v___x_2586_ = v___x_2573_;
goto v_reusejp_2585_;
}
else
{
lean_object* v_reuseFailAlloc_2587_; 
v_reuseFailAlloc_2587_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2587_, 0, v_vs_2570_);
lean_ctor_set(v_reuseFailAlloc_2587_, 1, v_c_2584_);
v___x_2586_ = v_reuseFailAlloc_2587_;
goto v_reusejp_2585_;
}
v_reusejp_2585_:
{
return v___x_2586_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2(lean_object* v_x_2589_, lean_object* v_keys_2590_, lean_object* v_v_2591_, lean_object* v_k_2592_, lean_object* v_x_2593_){
_start:
{
lean_object* v_snd_2594_; lean_object* v___x_2596_; uint8_t v_isShared_2597_; uint8_t v_isSharedCheck_2604_; 
v_snd_2594_ = lean_ctor_get(v_x_2593_, 1);
v_isSharedCheck_2604_ = !lean_is_exclusive(v_x_2593_);
if (v_isSharedCheck_2604_ == 0)
{
lean_object* v_unused_2605_; 
v_unused_2605_ = lean_ctor_get(v_x_2593_, 0);
lean_dec(v_unused_2605_);
v___x_2596_ = v_x_2593_;
v_isShared_2597_ = v_isSharedCheck_2604_;
goto v_resetjp_2595_;
}
else
{
lean_inc(v_snd_2594_);
lean_dec(v_x_2593_);
v___x_2596_ = lean_box(0);
v_isShared_2597_ = v_isSharedCheck_2604_;
goto v_resetjp_2595_;
}
v_resetjp_2595_:
{
lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v_c_2600_; lean_object* v___x_2602_; 
v___x_2598_ = lean_unsigned_to_nat(1u);
v___x_2599_ = lean_nat_add(v_x_2589_, v___x_2598_);
v_c_2600_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(v_keys_2590_, v_v_2591_, v___x_2599_, v_snd_2594_);
lean_dec(v___x_2599_);
if (v_isShared_2597_ == 0)
{
lean_ctor_set(v___x_2596_, 1, v_c_2600_);
lean_ctor_set(v___x_2596_, 0, v_k_2592_);
v___x_2602_ = v___x_2596_;
goto v_reusejp_2601_;
}
else
{
lean_object* v_reuseFailAlloc_2603_; 
v_reuseFailAlloc_2603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2603_, 0, v_k_2592_);
lean_ctor_set(v_reuseFailAlloc_2603_, 1, v_c_2600_);
v___x_2602_ = v_reuseFailAlloc_2603_;
goto v_reusejp_2601_;
}
v_reusejp_2601_:
{
return v___x_2602_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2___boxed(lean_object* v_x_2606_, lean_object* v_keys_2607_, lean_object* v_v_2608_, lean_object* v_k_2609_, lean_object* v_x_2610_){
_start:
{
lean_object* v_res_2611_; 
v_res_2611_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___lam__2(v_x_2606_, v_keys_2607_, v_v_2608_, v_k_2609_, v_x_2610_);
lean_dec_ref(v_keys_2607_);
lean_dec(v_x_2606_);
return v_res_2611_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1___boxed(lean_object* v_keys_2612_, lean_object* v_v_2613_, lean_object* v_x_2614_, lean_object* v_x_2615_){
_start:
{
lean_object* v_res_2616_; 
v_res_2616_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(v_keys_2612_, v_v_2613_, v_x_2614_, v_x_2615_);
lean_dec(v_x_2614_);
lean_dec_ref(v_keys_2612_);
return v_res_2616_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg___boxed(lean_object* v_x_2617_, lean_object* v_keys_2618_, lean_object* v_v_2619_, lean_object* v_k_2620_, lean_object* v_as_2621_, lean_object* v_k_2622_, lean_object* v_x_2623_, lean_object* v_x_2624_){
_start:
{
lean_object* v_res_2625_; 
v_res_2625_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg(v_x_2617_, v_keys_2618_, v_v_2619_, v_k_2620_, v_as_2621_, v_k_2622_, v_x_2623_, v_x_2624_);
lean_dec_ref(v_k_2622_);
lean_dec_ref(v_keys_2618_);
lean_dec(v_x_2617_);
return v_res_2625_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_x_2626_, lean_object* v_keys_2627_, lean_object* v_v_2628_, lean_object* v_k_2629_, lean_object* v_as_2630_, lean_object* v_k_2631_){
_start:
{
lean_object* v_res_2632_; 
v_res_2632_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3(v_x_2626_, v_keys_2627_, v_v_2628_, v_k_2629_, v_as_2630_, v_k_2631_);
lean_dec_ref(v_k_2631_);
lean_dec_ref(v_keys_2627_);
lean_dec(v_x_2626_);
return v_res_2632_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(lean_object* v_keys_2633_, lean_object* v_v_2634_, lean_object* v_x_2635_){
_start:
{
if (lean_obj_tag(v_x_2635_) == 0)
{
lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; 
v___x_2636_ = lean_unsigned_to_nat(1u);
v___x_2637_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_2633_, v_v_2634_, v___x_2636_);
v___x_2638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2638_, 0, v___x_2637_);
return v___x_2638_;
}
else
{
lean_object* v_val_2639_; lean_object* v___x_2641_; uint8_t v_isShared_2642_; uint8_t v_isSharedCheck_2648_; 
v_val_2639_ = lean_ctor_get(v_x_2635_, 0);
v_isSharedCheck_2648_ = !lean_is_exclusive(v_x_2635_);
if (v_isSharedCheck_2648_ == 0)
{
v___x_2641_ = v_x_2635_;
v_isShared_2642_ = v_isSharedCheck_2648_;
goto v_resetjp_2640_;
}
else
{
lean_inc(v_val_2639_);
lean_dec(v_x_2635_);
v___x_2641_ = lean_box(0);
v_isShared_2642_ = v_isSharedCheck_2648_;
goto v_resetjp_2640_;
}
v_resetjp_2640_:
{
lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2646_; 
v___x_2643_ = lean_unsigned_to_nat(1u);
v___x_2644_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1(v_keys_2633_, v_v_2634_, v___x_2643_, v_val_2639_);
if (v_isShared_2642_ == 0)
{
lean_ctor_set(v___x_2641_, 0, v___x_2644_);
v___x_2646_ = v___x_2641_;
goto v_reusejp_2645_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0___boxed(lean_object* v_keys_2649_, lean_object* v_v_2650_, lean_object* v_x_2651_){
_start:
{
lean_object* v_res_2652_; 
v_res_2652_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2649_, v_v_2650_, v_x_2651_);
lean_dec_ref(v_keys_2649_);
return v_res_2652_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2(lean_object* v_keys_2653_, lean_object* v_v_2654_, lean_object* v_x_2655_, size_t v_x_2656_, size_t v_x_2657_, lean_object* v_x_2658_){
_start:
{
if (lean_obj_tag(v_x_2655_) == 0)
{
lean_object* v_es_2659_; size_t v___x_2660_; size_t v___x_2661_; lean_object* v_j_2662_; lean_object* v___x_2663_; uint8_t v___x_2664_; 
v_es_2659_ = lean_ctor_get(v_x_2655_, 0);
v___x_2660_ = ((size_t)31ULL);
v___x_2661_ = lean_usize_land(v_x_2656_, v___x_2660_);
v_j_2662_ = lean_usize_to_nat(v___x_2661_);
v___x_2663_ = lean_array_get_size(v_es_2659_);
v___x_2664_ = lean_nat_dec_lt(v_j_2662_, v___x_2663_);
if (v___x_2664_ == 0)
{
lean_dec(v_j_2662_);
lean_dec(v_x_2658_);
return v_x_2655_;
}
else
{
lean_object* v___x_2666_; uint8_t v_isShared_2667_; uint8_t v_isSharedCheck_2732_; 
lean_inc_ref(v_es_2659_);
v_isSharedCheck_2732_ = !lean_is_exclusive(v_x_2655_);
if (v_isSharedCheck_2732_ == 0)
{
lean_object* v_unused_2733_; 
v_unused_2733_ = lean_ctor_get(v_x_2655_, 0);
lean_dec(v_unused_2733_);
v___x_2666_ = v_x_2655_;
v_isShared_2667_ = v_isSharedCheck_2732_;
goto v_resetjp_2665_;
}
else
{
lean_dec(v_x_2655_);
v___x_2666_ = lean_box(0);
v_isShared_2667_ = v_isSharedCheck_2732_;
goto v_resetjp_2665_;
}
v_resetjp_2665_:
{
lean_object* v_v_2668_; lean_object* v___x_2669_; lean_object* v_xs_x27_2670_; lean_object* v___y_2672_; 
v_v_2668_ = lean_array_fget(v_es_2659_, v_j_2662_);
v___x_2669_ = lean_box(0);
v_xs_x27_2670_ = lean_array_fset(v_es_2659_, v_j_2662_, v___x_2669_);
switch(lean_obj_tag(v_v_2668_))
{
case 0:
{
lean_object* v_key_2677_; lean_object* v_val_2678_; uint8_t v___x_2679_; 
v_key_2677_ = lean_ctor_get(v_v_2668_, 0);
v_val_2678_ = lean_ctor_get(v_v_2668_, 1);
v___x_2679_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_2658_, v_key_2677_);
if (v___x_2679_ == 0)
{
lean_object* v___x_2680_; lean_object* v___x_2681_; 
v___x_2680_ = lean_box(0);
v___x_2681_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2653_, v_v_2654_, v___x_2680_);
if (lean_obj_tag(v___x_2681_) == 0)
{
lean_dec(v_x_2658_);
v___y_2672_ = v_v_2668_;
goto v___jp_2671_;
}
else
{
lean_object* v_val_2682_; lean_object* v___x_2684_; uint8_t v_isShared_2685_; uint8_t v_isSharedCheck_2690_; 
lean_inc(v_val_2678_);
lean_inc(v_key_2677_);
lean_dec_ref_known(v_v_2668_, 2);
v_val_2682_ = lean_ctor_get(v___x_2681_, 0);
v_isSharedCheck_2690_ = !lean_is_exclusive(v___x_2681_);
if (v_isSharedCheck_2690_ == 0)
{
v___x_2684_ = v___x_2681_;
v_isShared_2685_ = v_isSharedCheck_2690_;
goto v_resetjp_2683_;
}
else
{
lean_inc(v_val_2682_);
lean_dec(v___x_2681_);
v___x_2684_ = lean_box(0);
v_isShared_2685_ = v_isSharedCheck_2690_;
goto v_resetjp_2683_;
}
v_resetjp_2683_:
{
lean_object* v___x_2686_; lean_object* v___x_2688_; 
v___x_2686_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2677_, v_val_2678_, v_x_2658_, v_val_2682_);
if (v_isShared_2685_ == 0)
{
lean_ctor_set(v___x_2684_, 0, v___x_2686_);
v___x_2688_ = v___x_2684_;
goto v_reusejp_2687_;
}
else
{
lean_object* v_reuseFailAlloc_2689_; 
v_reuseFailAlloc_2689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2689_, 0, v___x_2686_);
v___x_2688_ = v_reuseFailAlloc_2689_;
goto v_reusejp_2687_;
}
v_reusejp_2687_:
{
v___y_2672_ = v___x_2688_;
goto v___jp_2671_;
}
}
}
}
else
{
lean_object* v___x_2692_; uint8_t v_isShared_2693_; uint8_t v_isSharedCheck_2701_; 
lean_inc(v_val_2678_);
v_isSharedCheck_2701_ = !lean_is_exclusive(v_v_2668_);
if (v_isSharedCheck_2701_ == 0)
{
lean_object* v_unused_2702_; lean_object* v_unused_2703_; 
v_unused_2702_ = lean_ctor_get(v_v_2668_, 1);
lean_dec(v_unused_2702_);
v_unused_2703_ = lean_ctor_get(v_v_2668_, 0);
lean_dec(v_unused_2703_);
v___x_2692_ = v_v_2668_;
v_isShared_2693_ = v_isSharedCheck_2701_;
goto v_resetjp_2691_;
}
else
{
lean_dec(v_v_2668_);
v___x_2692_ = lean_box(0);
v_isShared_2693_ = v_isSharedCheck_2701_;
goto v_resetjp_2691_;
}
v_resetjp_2691_:
{
lean_object* v___x_2694_; lean_object* v___x_2695_; 
v___x_2694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2694_, 0, v_val_2678_);
v___x_2695_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2653_, v_v_2654_, v___x_2694_);
if (lean_obj_tag(v___x_2695_) == 0)
{
lean_object* v___x_2696_; 
lean_del_object(v___x_2692_);
lean_dec(v_x_2658_);
v___x_2696_ = lean_box(2);
v___y_2672_ = v___x_2696_;
goto v___jp_2671_;
}
else
{
lean_object* v_val_2697_; lean_object* v___x_2699_; 
v_val_2697_ = lean_ctor_get(v___x_2695_, 0);
lean_inc(v_val_2697_);
lean_dec_ref_known(v___x_2695_, 1);
if (v_isShared_2693_ == 0)
{
lean_ctor_set(v___x_2692_, 1, v_val_2697_);
lean_ctor_set(v___x_2692_, 0, v_x_2658_);
v___x_2699_ = v___x_2692_;
goto v_reusejp_2698_;
}
else
{
lean_object* v_reuseFailAlloc_2700_; 
v_reuseFailAlloc_2700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2700_, 0, v_x_2658_);
lean_ctor_set(v_reuseFailAlloc_2700_, 1, v_val_2697_);
v___x_2699_ = v_reuseFailAlloc_2700_;
goto v_reusejp_2698_;
}
v_reusejp_2698_:
{
v___y_2672_ = v___x_2699_;
goto v___jp_2671_;
}
}
}
}
}
case 1:
{
lean_object* v_node_2704_; lean_object* v___x_2706_; uint8_t v_isShared_2707_; uint8_t v_isSharedCheck_2727_; 
v_node_2704_ = lean_ctor_get(v_v_2668_, 0);
v_isSharedCheck_2727_ = !lean_is_exclusive(v_v_2668_);
if (v_isSharedCheck_2727_ == 0)
{
v___x_2706_ = v_v_2668_;
v_isShared_2707_ = v_isSharedCheck_2727_;
goto v_resetjp_2705_;
}
else
{
lean_inc(v_node_2704_);
lean_dec(v_v_2668_);
v___x_2706_ = lean_box(0);
v_isShared_2707_ = v_isSharedCheck_2727_;
goto v_resetjp_2705_;
}
v_resetjp_2705_:
{
size_t v___x_2708_; size_t v___x_2709_; size_t v___x_2710_; size_t v___x_2711_; lean_object* v_newNode_2712_; lean_object* v___x_2713_; 
v___x_2708_ = ((size_t)5ULL);
v___x_2709_ = lean_usize_shift_right(v_x_2656_, v___x_2708_);
v___x_2710_ = ((size_t)1ULL);
v___x_2711_ = lean_usize_add(v_x_2657_, v___x_2710_);
v_newNode_2712_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2(v_keys_2653_, v_v_2654_, v_node_2704_, v___x_2709_, v___x_2711_, v_x_2658_);
lean_inc_ref(v_newNode_2712_);
v___x_2713_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_2712_);
if (lean_obj_tag(v___x_2713_) == 0)
{
lean_object* v___x_2715_; 
if (v_isShared_2707_ == 0)
{
lean_ctor_set(v___x_2706_, 0, v_newNode_2712_);
v___x_2715_ = v___x_2706_;
goto v_reusejp_2714_;
}
else
{
lean_object* v_reuseFailAlloc_2716_; 
v_reuseFailAlloc_2716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2716_, 0, v_newNode_2712_);
v___x_2715_ = v_reuseFailAlloc_2716_;
goto v_reusejp_2714_;
}
v_reusejp_2714_:
{
v___y_2672_ = v___x_2715_;
goto v___jp_2671_;
}
}
else
{
lean_object* v_val_2717_; lean_object* v_fst_2718_; lean_object* v_snd_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2726_; 
lean_dec_ref(v_newNode_2712_);
lean_del_object(v___x_2706_);
v_val_2717_ = lean_ctor_get(v___x_2713_, 0);
lean_inc(v_val_2717_);
lean_dec_ref_known(v___x_2713_, 1);
v_fst_2718_ = lean_ctor_get(v_val_2717_, 0);
v_snd_2719_ = lean_ctor_get(v_val_2717_, 1);
v_isSharedCheck_2726_ = !lean_is_exclusive(v_val_2717_);
if (v_isSharedCheck_2726_ == 0)
{
v___x_2721_ = v_val_2717_;
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_snd_2719_);
lean_inc(v_fst_2718_);
lean_dec(v_val_2717_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
lean_object* v___x_2724_; 
if (v_isShared_2722_ == 0)
{
v___x_2724_ = v___x_2721_;
goto v_reusejp_2723_;
}
else
{
lean_object* v_reuseFailAlloc_2725_; 
v_reuseFailAlloc_2725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2725_, 0, v_fst_2718_);
lean_ctor_set(v_reuseFailAlloc_2725_, 1, v_snd_2719_);
v___x_2724_ = v_reuseFailAlloc_2725_;
goto v_reusejp_2723_;
}
v_reusejp_2723_:
{
v___y_2672_ = v___x_2724_;
goto v___jp_2671_;
}
}
}
}
}
default: 
{
lean_object* v___x_2728_; lean_object* v___x_2729_; 
v___x_2728_ = lean_box(0);
v___x_2729_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2653_, v_v_2654_, v___x_2728_);
if (lean_obj_tag(v___x_2729_) == 0)
{
lean_dec(v_x_2658_);
v___y_2672_ = v_v_2668_;
goto v___jp_2671_;
}
else
{
lean_object* v_val_2730_; lean_object* v___x_2731_; 
v_val_2730_ = lean_ctor_get(v___x_2729_, 0);
lean_inc(v_val_2730_);
lean_dec_ref_known(v___x_2729_, 1);
v___x_2731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2731_, 0, v_x_2658_);
lean_ctor_set(v___x_2731_, 1, v_val_2730_);
v___y_2672_ = v___x_2731_;
goto v___jp_2671_;
}
}
}
v___jp_2671_:
{
lean_object* v___x_2673_; lean_object* v___x_2675_; 
v___x_2673_ = lean_array_fset(v_xs_x27_2670_, v_j_2662_, v___y_2672_);
lean_dec(v_j_2662_);
if (v_isShared_2667_ == 0)
{
lean_ctor_set(v___x_2666_, 0, v___x_2673_);
v___x_2675_ = v___x_2666_;
goto v_reusejp_2674_;
}
else
{
lean_object* v_reuseFailAlloc_2676_; 
v_reuseFailAlloc_2676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2676_, 0, v___x_2673_);
v___x_2675_ = v_reuseFailAlloc_2676_;
goto v_reusejp_2674_;
}
v_reusejp_2674_:
{
return v___x_2675_;
}
}
}
}
}
else
{
lean_object* v_ks_2734_; lean_object* v_vs_2735_; lean_object* v___x_2737_; uint8_t v_isShared_2738_; uint8_t v_isSharedCheck_2768_; 
v_ks_2734_ = lean_ctor_get(v_x_2655_, 0);
v_vs_2735_ = lean_ctor_get(v_x_2655_, 1);
v_isSharedCheck_2768_ = !lean_is_exclusive(v_x_2655_);
if (v_isSharedCheck_2768_ == 0)
{
v___x_2737_ = v_x_2655_;
v_isShared_2738_ = v_isSharedCheck_2768_;
goto v_resetjp_2736_;
}
else
{
lean_inc(v_vs_2735_);
lean_inc(v_ks_2734_);
lean_dec(v_x_2655_);
v___x_2737_ = lean_box(0);
v_isShared_2738_ = v_isSharedCheck_2768_;
goto v_resetjp_2736_;
}
v_resetjp_2736_:
{
lean_object* v___x_2739_; 
v___x_2739_ = l_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_SimpTheorems_addSimpTheorem_spec__0_spec__1_spec__4(v_ks_2734_, v_x_2658_);
if (lean_obj_tag(v___x_2739_) == 0)
{
lean_object* v___x_2741_; 
if (v_isShared_2738_ == 0)
{
v___x_2741_ = v___x_2737_;
goto v_reusejp_2740_;
}
else
{
lean_object* v_reuseFailAlloc_2746_; 
v_reuseFailAlloc_2746_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2746_, 0, v_ks_2734_);
lean_ctor_set(v_reuseFailAlloc_2746_, 1, v_vs_2735_);
v___x_2741_ = v_reuseFailAlloc_2746_;
goto v_reusejp_2740_;
}
v_reusejp_2740_:
{
lean_object* v___x_2742_; lean_object* v___x_2743_; 
v___x_2742_ = lean_box(0);
v___x_2743_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2653_, v_v_2654_, v___x_2742_);
if (lean_obj_tag(v___x_2743_) == 0)
{
lean_dec(v_x_2658_);
return v___x_2741_;
}
else
{
lean_object* v_val_2744_; lean_object* v___x_2745_; 
v_val_2744_ = lean_ctor_get(v___x_2743_, 0);
lean_inc(v_val_2744_);
lean_dec_ref_known(v___x_2743_, 1);
v___x_2745_ = l_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_SimpTheorems_addSimpTheorem_spec__0_spec__1_spec__5___redArg(v___x_2741_, v_x_2656_, v_x_2657_, v_x_2658_, v_val_2744_);
return v___x_2745_;
}
}
}
else
{
lean_object* v_val_2747_; lean_object* v___x_2749_; uint8_t v_isShared_2750_; uint8_t v_isSharedCheck_2767_; 
v_val_2747_ = lean_ctor_get(v___x_2739_, 0);
v_isSharedCheck_2767_ = !lean_is_exclusive(v___x_2739_);
if (v_isSharedCheck_2767_ == 0)
{
v___x_2749_ = v___x_2739_;
v_isShared_2750_ = v_isSharedCheck_2767_;
goto v_resetjp_2748_;
}
else
{
lean_inc(v_val_2747_);
lean_dec(v___x_2739_);
v___x_2749_ = lean_box(0);
v_isShared_2750_ = v_isSharedCheck_2767_;
goto v_resetjp_2748_;
}
v_resetjp_2748_:
{
lean_object* v_v_x27_2751_; lean_object* v_keys_2752_; lean_object* v_vals_2753_; lean_object* v___x_2755_; 
v_v_x27_2751_ = lean_array_fget(v_vs_2735_, v_val_2747_);
lean_inc(v_val_2747_);
v_keys_2752_ = l_Array_eraseIdx___redArg(v_ks_2734_, v_val_2747_);
v_vals_2753_ = l_Array_eraseIdx___redArg(v_vs_2735_, v_val_2747_);
if (v_isShared_2750_ == 0)
{
lean_ctor_set(v___x_2749_, 0, v_v_x27_2751_);
v___x_2755_ = v___x_2749_;
goto v_reusejp_2754_;
}
else
{
lean_object* v_reuseFailAlloc_2766_; 
v_reuseFailAlloc_2766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2766_, 0, v_v_x27_2751_);
v___x_2755_ = v_reuseFailAlloc_2766_;
goto v_reusejp_2754_;
}
v_reusejp_2754_:
{
lean_object* v___x_2756_; 
v___x_2756_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___lam__0(v_keys_2653_, v_v_2654_, v___x_2755_);
if (lean_obj_tag(v___x_2756_) == 0)
{
lean_object* v___x_2758_; 
lean_dec(v_x_2658_);
if (v_isShared_2738_ == 0)
{
lean_ctor_set(v___x_2737_, 1, v_vals_2753_);
lean_ctor_set(v___x_2737_, 0, v_keys_2752_);
v___x_2758_ = v___x_2737_;
goto v_reusejp_2757_;
}
else
{
lean_object* v_reuseFailAlloc_2759_; 
v_reuseFailAlloc_2759_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2759_, 0, v_keys_2752_);
lean_ctor_set(v_reuseFailAlloc_2759_, 1, v_vals_2753_);
v___x_2758_ = v_reuseFailAlloc_2759_;
goto v_reusejp_2757_;
}
v_reusejp_2757_:
{
return v___x_2758_;
}
}
else
{
lean_object* v_val_2760_; lean_object* v_keys_2761_; lean_object* v_vals_2762_; lean_object* v___x_2764_; 
v_val_2760_ = lean_ctor_get(v___x_2756_, 0);
lean_inc(v_val_2760_);
lean_dec_ref_known(v___x_2756_, 1);
v_keys_2761_ = lean_array_push(v_keys_2752_, v_x_2658_);
v_vals_2762_ = lean_array_push(v_vals_2753_, v_val_2760_);
if (v_isShared_2738_ == 0)
{
lean_ctor_set(v___x_2737_, 1, v_vals_2762_);
lean_ctor_set(v___x_2737_, 0, v_keys_2761_);
v___x_2764_ = v___x_2737_;
goto v_reusejp_2763_;
}
else
{
lean_object* v_reuseFailAlloc_2765_; 
v_reuseFailAlloc_2765_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2765_, 0, v_keys_2761_);
lean_ctor_set(v_reuseFailAlloc_2765_, 1, v_vals_2762_);
v___x_2764_ = v_reuseFailAlloc_2765_;
goto v_reusejp_2763_;
}
v_reusejp_2763_:
{
return v___x_2764_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2___boxed(lean_object* v_keys_2769_, lean_object* v_v_2770_, lean_object* v_x_2771_, lean_object* v_x_2772_, lean_object* v_x_2773_, lean_object* v_x_2774_){
_start:
{
size_t v_x_6780__boxed_2775_; size_t v_x_6781__boxed_2776_; lean_object* v_res_2777_; 
v_x_6780__boxed_2775_ = lean_unbox_usize(v_x_2772_);
lean_dec(v_x_2772_);
v_x_6781__boxed_2776_ = lean_unbox_usize(v_x_2773_);
lean_dec(v_x_2773_);
v_res_2777_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2(v_keys_2769_, v_v_2770_, v_x_2771_, v_x_6780__boxed_2775_, v_x_6781__boxed_2776_, v_x_2774_);
lean_dec_ref(v_keys_2769_);
return v_res_2777_;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; 
v___x_2781_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__2));
v___x_2782_ = lean_unsigned_to_nat(23u);
v___x_2783_ = lean_unsigned_to_nat(166u);
v___x_2784_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__1));
v___x_2785_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__0));
v___x_2786_ = l_mkPanicMessageWithDecl(v___x_2785_, v___x_2784_, v___x_2783_, v___x_2782_, v___x_2781_);
return v___x_2786_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0(lean_object* v_d_2787_, lean_object* v_keys_2788_, lean_object* v_v_2789_){
_start:
{
lean_object* v___x_2790_; lean_object* v___x_2791_; uint8_t v___x_2792_; 
v___x_2790_ = lean_array_get_size(v_keys_2788_);
v___x_2791_ = lean_unsigned_to_nat(0u);
v___x_2792_ = lean_nat_dec_eq(v___x_2790_, v___x_2791_);
if (v___x_2792_ == 0)
{
lean_object* v___x_2793_; lean_object* v_k_2794_; uint64_t v___x_2795_; size_t v_h_2796_; size_t v___x_2797_; lean_object* v___x_2798_; 
v___x_2793_ = lean_box(0);
v_k_2794_ = lean_array_get_borrowed(v___x_2793_, v_keys_2788_, v___x_2791_);
v___x_2795_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_2794_);
v_h_2796_ = lean_uint64_to_usize(v___x_2795_);
v___x_2797_ = ((size_t)1ULL);
lean_inc(v_k_2794_);
v___x_2798_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__2(v_keys_2788_, v_v_2789_, v_d_2787_, v_h_2796_, v___x_2797_, v_k_2794_);
return v___x_2798_;
}
else
{
lean_object* v___x_2799_; lean_object* v___x_2800_; 
lean_dec_ref(v_d_2787_);
v___x_2799_ = lean_obj_once(&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3, &lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3_once, _init_lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___closed__3);
v___x_2800_ = lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__3(v___x_2799_);
return v___x_2800_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0___boxed(lean_object* v_d_2801_, lean_object* v_keys_2802_, lean_object* v_v_2803_){
_start:
{
lean_object* v_res_2804_; 
v_res_2804_ = lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0(v_d_2801_, v_keys_2802_, v_v_2803_);
lean_dec_ref(v_keys_2802_);
return v_res_2804_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0(lean_object* v_d_2805_, lean_object* v_e_2806_, lean_object* v_v_2807_, uint8_t v_noIndexAtArgs_2808_, lean_object* v_a_2809_, lean_object* v_a_2810_, lean_object* v_a_2811_, lean_object* v_a_2812_){
_start:
{
lean_object* v___x_2814_; 
v___x_2814_ = l_Lean_Meta_DiscrTree_mkPath(v_e_2806_, v_noIndexAtArgs_2808_, v_a_2809_, v_a_2810_, v_a_2811_, v_a_2812_);
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v_a_2815_; lean_object* v___x_2817_; uint8_t v_isShared_2818_; uint8_t v_isSharedCheck_2823_; 
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
v_isSharedCheck_2823_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2823_ == 0)
{
v___x_2817_ = v___x_2814_;
v_isShared_2818_ = v_isSharedCheck_2823_;
goto v_resetjp_2816_;
}
else
{
lean_inc(v_a_2815_);
lean_dec(v___x_2814_);
v___x_2817_ = lean_box(0);
v_isShared_2818_ = v_isSharedCheck_2823_;
goto v_resetjp_2816_;
}
v_resetjp_2816_:
{
lean_object* v___x_2819_; lean_object* v___x_2821_; 
v___x_2819_ = lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0(v_d_2805_, v_a_2815_, v_v_2807_);
lean_dec(v_a_2815_);
if (v_isShared_2818_ == 0)
{
lean_ctor_set(v___x_2817_, 0, v___x_2819_);
v___x_2821_ = v___x_2817_;
goto v_reusejp_2820_;
}
else
{
lean_object* v_reuseFailAlloc_2822_; 
v_reuseFailAlloc_2822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2822_, 0, v___x_2819_);
v___x_2821_ = v_reuseFailAlloc_2822_;
goto v_reusejp_2820_;
}
v_reusejp_2820_:
{
return v___x_2821_;
}
}
}
else
{
lean_object* v_a_2824_; lean_object* v___x_2826_; uint8_t v_isShared_2827_; uint8_t v_isSharedCheck_2831_; 
lean_dec_ref(v_d_2805_);
v_a_2824_ = lean_ctor_get(v___x_2814_, 0);
v_isSharedCheck_2831_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2826_ = v___x_2814_;
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
else
{
lean_inc(v_a_2824_);
lean_dec(v___x_2814_);
v___x_2826_ = lean_box(0);
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
v_resetjp_2825_:
{
lean_object* v___x_2829_; 
if (v_isShared_2827_ == 0)
{
v___x_2829_ = v___x_2826_;
goto v_reusejp_2828_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v_a_2824_);
v___x_2829_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2828_;
}
v_reusejp_2828_:
{
return v___x_2829_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0___boxed(lean_object* v_d_2832_, lean_object* v_e_2833_, lean_object* v_v_2834_, lean_object* v_noIndexAtArgs_2835_, lean_object* v_a_2836_, lean_object* v_a_2837_, lean_object* v_a_2838_, lean_object* v_a_2839_, lean_object* v_a_2840_){
_start:
{
uint8_t v_noIndexAtArgs_boxed_2841_; lean_object* v_res_2842_; 
v_noIndexAtArgs_boxed_2841_ = lean_unbox(v_noIndexAtArgs_2835_);
v_res_2842_ = lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0(v_d_2832_, v_e_2833_, v_v_2834_, v_noIndexAtArgs_boxed_2841_, v_a_2836_, v_a_2837_, v_a_2838_, v_a_2839_);
lean_dec(v_a_2839_);
lean_dec_ref(v_a_2838_);
lean_dec(v_a_2837_);
lean_dec_ref(v_a_2836_);
return v_res_2842_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0(void){
_start:
{
lean_object* v___x_2843_; 
v___x_2843_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_2843_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2(void){
_start:
{
lean_object* v___x_2845_; lean_object* v___x_2846_; 
v___x_2845_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__1));
v___x_2846_ = l_Lean_stringToMessageData(v___x_2845_);
return v___x_2846_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3(void){
_start:
{
lean_object* v___x_2847_; lean_object* v___x_2848_; 
v___x_2847_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2, &lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__2);
v___x_2848_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2848_, 0, v___x_2847_);
return v___x_2848_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1(lean_object* v___x_2851_, lean_object* v_x_2852_, lean_object* v_ty_x27_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_, lean_object* v___y_2856_, lean_object* v___y_2857_){
_start:
{
lean_object* v___y_2860_; lean_object* v___y_2861_; lean_object* v_fst_2862_; lean_object* v_snd_2863_; lean_object* v_fst_2961_; lean_object* v_snd_2962_; lean_object* v___x_3003_; lean_object* v___x_3004_; uint8_t v___x_3005_; 
v___x_3003_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1));
v___x_3004_ = lean_unsigned_to_nat(3u);
v___x_3005_ = l_Lean_Expr_isAppOfArity(v_ty_x27_2853_, v___x_3003_, v___x_3004_);
if (v___x_3005_ == 0)
{
lean_object* v___x_3006_; lean_object* v___x_3007_; uint8_t v___x_3008_; 
v___x_3006_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__4));
v___x_3007_ = lean_unsigned_to_nat(2u);
v___x_3008_ = l_Lean_Expr_isAppOfArity(v_ty_x27_2853_, v___x_3006_, v___x_3007_);
if (v___x_3008_ == 0)
{
lean_object* v___x_3009_; lean_object* v___x_3010_; 
lean_dec_ref(v___x_2851_);
v___x_3009_ = lean_box(0);
v___x_3010_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3010_, 0, v___x_3009_);
return v___x_3010_;
}
else
{
lean_object* v___x_3011_; lean_object* v___x_3012_; lean_object* v___x_3013_; 
v___x_3011_ = l_Lean_Expr_appFn_x21(v_ty_x27_2853_);
v___x_3012_ = l_Lean_Expr_appArg_x21(v___x_3011_);
lean_dec_ref(v___x_3011_);
v___x_3013_ = l_Lean_Expr_appArg_x21(v_ty_x27_2853_);
v_fst_2961_ = v___x_3012_;
v_snd_2962_ = v___x_3013_;
goto v___jp_2960_;
}
}
else
{
lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; 
v___x_3014_ = l_Lean_Expr_appFn_x21(v_ty_x27_2853_);
v___x_3015_ = l_Lean_Expr_appArg_x21(v___x_3014_);
lean_dec_ref(v___x_3014_);
v___x_3016_ = l_Lean_Expr_appArg_x21(v_ty_x27_2853_);
v_fst_2961_ = v___x_3015_;
v_snd_2962_ = v___x_3016_;
goto v___jp_2960_;
}
v___jp_2859_:
{
lean_object* v___x_2864_; 
lean_inc_ref(v_fst_2862_);
lean_inc_ref(v___y_2861_);
v___x_2864_ = l_Lean_Meta_isExprDefEq(v___y_2861_, v_fst_2862_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
if (lean_obj_tag(v___x_2864_) == 0)
{
lean_object* v_a_2865_; lean_object* v___x_2867_; uint8_t v_isShared_2868_; uint8_t v_isSharedCheck_2951_; 
v_a_2865_ = lean_ctor_get(v___x_2864_, 0);
v_isSharedCheck_2951_ = !lean_is_exclusive(v___x_2864_);
if (v_isSharedCheck_2951_ == 0)
{
v___x_2867_ = v___x_2864_;
v_isShared_2868_ = v_isSharedCheck_2951_;
goto v_resetjp_2866_;
}
else
{
lean_inc(v_a_2865_);
lean_dec(v___x_2864_);
v___x_2867_ = lean_box(0);
v_isShared_2868_ = v_isSharedCheck_2951_;
goto v_resetjp_2866_;
}
v_resetjp_2866_:
{
uint8_t v___x_2869_; 
v___x_2869_ = lean_unbox(v_a_2865_);
lean_dec(v_a_2865_);
if (v___x_2869_ == 0)
{
lean_object* v___x_2870_; lean_object* v___x_2872_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
lean_dec_ref(v___y_2861_);
lean_dec_ref(v___y_2860_);
v___x_2870_ = lean_box(0);
if (v_isShared_2868_ == 0)
{
lean_ctor_set(v___x_2867_, 0, v___x_2870_);
v___x_2872_ = v___x_2867_;
goto v_reusejp_2871_;
}
else
{
lean_object* v_reuseFailAlloc_2873_; 
v_reuseFailAlloc_2873_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2873_, 0, v___x_2870_);
v___x_2872_ = v_reuseFailAlloc_2873_;
goto v_reusejp_2871_;
}
v_reusejp_2871_:
{
return v___x_2872_;
}
}
else
{
lean_object* v___f_2874_; uint8_t v___x_2875_; lean_object* v___x_2876_; 
lean_del_object(v___x_2867_);
lean_inc_ref(v_fst_2862_);
v___f_2874_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_simpComm___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2874_, 0, v___y_2860_);
lean_closure_set(v___f_2874_, 1, v_fst_2862_);
v___x_2875_ = 0;
v___x_2876_ = l_Lean_Meta_withNewMCtxDepth___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_mkSimpTheoremKeys_spec__2___redArg(v___f_2874_, v___x_2875_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
if (lean_obj_tag(v___x_2876_) == 0)
{
lean_object* v_a_2877_; lean_object* v___x_2879_; uint8_t v_isShared_2880_; uint8_t v_isSharedCheck_2942_; 
v_a_2877_ = lean_ctor_get(v___x_2876_, 0);
v_isSharedCheck_2942_ = !lean_is_exclusive(v___x_2876_);
if (v_isSharedCheck_2942_ == 0)
{
v___x_2879_ = v___x_2876_;
v_isShared_2880_ = v_isSharedCheck_2942_;
goto v_resetjp_2878_;
}
else
{
lean_inc(v_a_2877_);
lean_dec(v___x_2876_);
v___x_2879_ = lean_box(0);
v_isShared_2880_ = v_isSharedCheck_2942_;
goto v_resetjp_2878_;
}
v_resetjp_2878_:
{
uint8_t v___x_2881_; 
v___x_2881_ = lean_unbox(v_a_2877_);
lean_dec(v_a_2877_);
if (v___x_2881_ == 0)
{
lean_object* v___x_2882_; lean_object* v___x_2884_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
lean_dec_ref(v___y_2861_);
v___x_2882_ = lean_box(0);
if (v_isShared_2880_ == 0)
{
lean_ctor_set(v___x_2879_, 0, v___x_2882_);
v___x_2884_ = v___x_2879_;
goto v_reusejp_2883_;
}
else
{
lean_object* v_reuseFailAlloc_2885_; 
v_reuseFailAlloc_2885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2885_, 0, v___x_2882_);
v___x_2884_ = v_reuseFailAlloc_2885_;
goto v_reusejp_2883_;
}
v_reusejp_2883_:
{
return v___x_2884_;
}
}
else
{
lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; 
lean_del_object(v___x_2879_);
v___x_2886_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0, &lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__0);
v___x_2887_ = lean_box(0);
v___x_2888_ = lp_batteries_Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0(v___x_2886_, v___y_2861_, v___x_2887_, v___x_2875_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
if (lean_obj_tag(v___x_2888_) == 0)
{
lean_object* v_a_2889_; lean_object* v___x_2890_; 
v_a_2889_ = lean_ctor_get(v___x_2888_, 0);
lean_inc(v_a_2889_);
lean_dec_ref_known(v___x_2888_, 1);
lean_inc_ref(v_fst_2862_);
v___x_2890_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v_a_2889_, v_fst_2862_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
lean_dec(v_a_2889_);
if (lean_obj_tag(v___x_2890_) == 0)
{
lean_object* v_a_2891_; lean_object* v___x_2893_; uint8_t v_isShared_2894_; uint8_t v_isSharedCheck_2925_; 
v_a_2891_ = lean_ctor_get(v___x_2890_, 0);
v_isSharedCheck_2925_ = !lean_is_exclusive(v___x_2890_);
if (v_isSharedCheck_2925_ == 0)
{
v___x_2893_ = v___x_2890_;
v_isShared_2894_ = v_isSharedCheck_2925_;
goto v_resetjp_2892_;
}
else
{
lean_inc(v_a_2891_);
lean_dec(v___x_2890_);
v___x_2893_ = lean_box(0);
v_isShared_2894_ = v_isSharedCheck_2925_;
goto v_resetjp_2892_;
}
v_resetjp_2892_:
{
lean_object* v___x_2895_; lean_object* v___x_2896_; uint8_t v___x_2897_; 
v___x_2895_ = lean_array_get_size(v_a_2891_);
lean_dec(v_a_2891_);
v___x_2896_ = lean_unsigned_to_nat(0u);
v___x_2897_ = lean_nat_dec_eq(v___x_2895_, v___x_2896_);
if (v___x_2897_ == 0)
{
lean_object* v___x_2898_; 
lean_del_object(v___x_2893_);
v___x_2898_ = l_Lean_Meta_isExprDefEq(v_fst_2862_, v_snd_2863_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
if (lean_obj_tag(v___x_2898_) == 0)
{
lean_object* v_a_2899_; lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2912_; 
v_a_2899_ = lean_ctor_get(v___x_2898_, 0);
v_isSharedCheck_2912_ = !lean_is_exclusive(v___x_2898_);
if (v_isSharedCheck_2912_ == 0)
{
v___x_2901_ = v___x_2898_;
v_isShared_2902_ = v_isSharedCheck_2912_;
goto v_resetjp_2900_;
}
else
{
lean_inc(v_a_2899_);
lean_dec(v___x_2898_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2912_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
uint8_t v___x_2903_; 
v___x_2903_ = lean_unbox(v_a_2899_);
lean_dec(v_a_2899_);
if (v___x_2903_ == 0)
{
lean_object* v___x_2904_; lean_object* v___x_2906_; 
v___x_2904_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3, &lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__3);
if (v_isShared_2902_ == 0)
{
lean_ctor_set(v___x_2901_, 0, v___x_2904_);
v___x_2906_ = v___x_2901_;
goto v_reusejp_2905_;
}
else
{
lean_object* v_reuseFailAlloc_2907_; 
v_reuseFailAlloc_2907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2907_, 0, v___x_2904_);
v___x_2906_ = v_reuseFailAlloc_2907_;
goto v_reusejp_2905_;
}
v_reusejp_2905_:
{
return v___x_2906_;
}
}
else
{
lean_object* v___x_2908_; lean_object* v___x_2910_; 
v___x_2908_ = lean_box(0);
if (v_isShared_2902_ == 0)
{
lean_ctor_set(v___x_2901_, 0, v___x_2908_);
v___x_2910_ = v___x_2901_;
goto v_reusejp_2909_;
}
else
{
lean_object* v_reuseFailAlloc_2911_; 
v_reuseFailAlloc_2911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2911_, 0, v___x_2908_);
v___x_2910_ = v_reuseFailAlloc_2911_;
goto v_reusejp_2909_;
}
v_reusejp_2909_:
{
return v___x_2910_;
}
}
}
}
else
{
lean_object* v_a_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_2920_; 
v_a_2913_ = lean_ctor_get(v___x_2898_, 0);
v_isSharedCheck_2920_ = !lean_is_exclusive(v___x_2898_);
if (v_isSharedCheck_2920_ == 0)
{
v___x_2915_ = v___x_2898_;
v_isShared_2916_ = v_isSharedCheck_2920_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_a_2913_);
lean_dec(v___x_2898_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_2920_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v___x_2918_; 
if (v_isShared_2916_ == 0)
{
v___x_2918_ = v___x_2915_;
goto v_reusejp_2917_;
}
else
{
lean_object* v_reuseFailAlloc_2919_; 
v_reuseFailAlloc_2919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2919_, 0, v_a_2913_);
v___x_2918_ = v_reuseFailAlloc_2919_;
goto v_reusejp_2917_;
}
v_reusejp_2917_:
{
return v___x_2918_;
}
}
}
}
else
{
lean_object* v___x_2921_; lean_object* v___x_2923_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
v___x_2921_ = lean_box(0);
if (v_isShared_2894_ == 0)
{
lean_ctor_set(v___x_2893_, 0, v___x_2921_);
v___x_2923_ = v___x_2893_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_2924_; 
v_reuseFailAlloc_2924_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2924_, 0, v___x_2921_);
v___x_2923_ = v_reuseFailAlloc_2924_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
return v___x_2923_;
}
}
}
}
else
{
lean_object* v_a_2926_; lean_object* v___x_2928_; uint8_t v_isShared_2929_; uint8_t v_isSharedCheck_2933_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
v_a_2926_ = lean_ctor_get(v___x_2890_, 0);
v_isSharedCheck_2933_ = !lean_is_exclusive(v___x_2890_);
if (v_isSharedCheck_2933_ == 0)
{
v___x_2928_ = v___x_2890_;
v_isShared_2929_ = v_isSharedCheck_2933_;
goto v_resetjp_2927_;
}
else
{
lean_inc(v_a_2926_);
lean_dec(v___x_2890_);
v___x_2928_ = lean_box(0);
v_isShared_2929_ = v_isSharedCheck_2933_;
goto v_resetjp_2927_;
}
v_resetjp_2927_:
{
lean_object* v___x_2931_; 
if (v_isShared_2929_ == 0)
{
v___x_2931_ = v___x_2928_;
goto v_reusejp_2930_;
}
else
{
lean_object* v_reuseFailAlloc_2932_; 
v_reuseFailAlloc_2932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2932_, 0, v_a_2926_);
v___x_2931_ = v_reuseFailAlloc_2932_;
goto v_reusejp_2930_;
}
v_reusejp_2930_:
{
return v___x_2931_;
}
}
}
}
else
{
lean_object* v_a_2934_; lean_object* v___x_2936_; uint8_t v_isShared_2937_; uint8_t v_isSharedCheck_2941_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
v_a_2934_ = lean_ctor_get(v___x_2888_, 0);
v_isSharedCheck_2941_ = !lean_is_exclusive(v___x_2888_);
if (v_isSharedCheck_2941_ == 0)
{
v___x_2936_ = v___x_2888_;
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
else
{
lean_inc(v_a_2934_);
lean_dec(v___x_2888_);
v___x_2936_ = lean_box(0);
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
v_resetjp_2935_:
{
lean_object* v___x_2939_; 
if (v_isShared_2937_ == 0)
{
v___x_2939_ = v___x_2936_;
goto v_reusejp_2938_;
}
else
{
lean_object* v_reuseFailAlloc_2940_; 
v_reuseFailAlloc_2940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2940_, 0, v_a_2934_);
v___x_2939_ = v_reuseFailAlloc_2940_;
goto v_reusejp_2938_;
}
v_reusejp_2938_:
{
return v___x_2939_;
}
}
}
}
}
}
else
{
lean_object* v_a_2943_; lean_object* v___x_2945_; uint8_t v_isShared_2946_; uint8_t v_isSharedCheck_2950_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
lean_dec_ref(v___y_2861_);
v_a_2943_ = lean_ctor_get(v___x_2876_, 0);
v_isSharedCheck_2950_ = !lean_is_exclusive(v___x_2876_);
if (v_isSharedCheck_2950_ == 0)
{
v___x_2945_ = v___x_2876_;
v_isShared_2946_ = v_isSharedCheck_2950_;
goto v_resetjp_2944_;
}
else
{
lean_inc(v_a_2943_);
lean_dec(v___x_2876_);
v___x_2945_ = lean_box(0);
v_isShared_2946_ = v_isSharedCheck_2950_;
goto v_resetjp_2944_;
}
v_resetjp_2944_:
{
lean_object* v___x_2948_; 
if (v_isShared_2946_ == 0)
{
v___x_2948_ = v___x_2945_;
goto v_reusejp_2947_;
}
else
{
lean_object* v_reuseFailAlloc_2949_; 
v_reuseFailAlloc_2949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2949_, 0, v_a_2943_);
v___x_2948_ = v_reuseFailAlloc_2949_;
goto v_reusejp_2947_;
}
v_reusejp_2947_:
{
return v___x_2948_;
}
}
}
}
}
}
else
{
lean_object* v_a_2952_; lean_object* v___x_2954_; uint8_t v_isShared_2955_; uint8_t v_isSharedCheck_2959_; 
lean_dec_ref(v_snd_2863_);
lean_dec_ref(v_fst_2862_);
lean_dec_ref(v___y_2861_);
lean_dec_ref(v___y_2860_);
v_a_2952_ = lean_ctor_get(v___x_2864_, 0);
v_isSharedCheck_2959_ = !lean_is_exclusive(v___x_2864_);
if (v_isSharedCheck_2959_ == 0)
{
v___x_2954_ = v___x_2864_;
v_isShared_2955_ = v_isSharedCheck_2959_;
goto v_resetjp_2953_;
}
else
{
lean_inc(v_a_2952_);
lean_dec(v___x_2864_);
v___x_2954_ = lean_box(0);
v_isShared_2955_ = v_isSharedCheck_2959_;
goto v_resetjp_2953_;
}
v_resetjp_2953_:
{
lean_object* v___x_2957_; 
if (v_isShared_2955_ == 0)
{
v___x_2957_ = v___x_2954_;
goto v_reusejp_2956_;
}
else
{
lean_object* v_reuseFailAlloc_2958_; 
v_reuseFailAlloc_2958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2958_, 0, v_a_2952_);
v___x_2957_ = v_reuseFailAlloc_2958_;
goto v_reusejp_2956_;
}
v_reusejp_2956_:
{
return v___x_2957_;
}
}
}
}
v___jp_2960_:
{
lean_object* v___x_2963_; lean_object* v___x_2964_; lean_object* v___x_2965_; lean_object* v___x_2966_; uint8_t v___x_2967_; 
v___x_2963_ = l_Lean_Expr_getAppFn(v_fst_2961_);
lean_dec_ref(v_fst_2961_);
v___x_2964_ = l_Lean_Expr_constName_x3f(v___x_2963_);
lean_dec_ref(v___x_2963_);
v___x_2965_ = l_Lean_Expr_getAppFn(v_snd_2962_);
v___x_2966_ = l_Lean_Expr_constName_x3f(v___x_2965_);
lean_dec_ref(v___x_2965_);
v___x_2967_ = lp_batteries_Option_instBEq_beq___at___00Batteries_Tactic_Lint_isSimpEq_spec__0(v___x_2964_, v___x_2966_);
lean_dec(v___x_2966_);
lean_dec(v___x_2964_);
if (v___x_2967_ == 0)
{
lean_object* v___x_2968_; lean_object* v___x_2969_; 
lean_dec_ref(v_snd_2962_);
lean_dec_ref(v___x_2851_);
v___x_2968_ = lean_box(0);
v___x_2969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2969_, 0, v___x_2968_);
return v___x_2969_;
}
else
{
lean_object* v___x_2970_; uint8_t v___x_2971_; lean_object* v___x_2972_; 
v___x_2970_ = lean_box(0);
v___x_2971_ = 0;
v___x_2972_ = l_Lean_Meta_forallMetaTelescopeReducing(v___x_2851_, v___x_2970_, v___x_2971_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
if (lean_obj_tag(v___x_2972_) == 0)
{
lean_object* v_a_2973_; lean_object* v___x_2975_; uint8_t v_isShared_2976_; uint8_t v_isSharedCheck_2994_; 
v_a_2973_ = lean_ctor_get(v___x_2972_, 0);
v_isSharedCheck_2994_ = !lean_is_exclusive(v___x_2972_);
if (v_isSharedCheck_2994_ == 0)
{
v___x_2975_ = v___x_2972_;
v_isShared_2976_ = v_isSharedCheck_2994_;
goto v_resetjp_2974_;
}
else
{
lean_inc(v_a_2973_);
lean_dec(v___x_2972_);
v___x_2975_ = lean_box(0);
v_isShared_2976_ = v_isSharedCheck_2994_;
goto v_resetjp_2974_;
}
v_resetjp_2974_:
{
lean_object* v_snd_2977_; lean_object* v_snd_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; uint8_t v___x_2981_; 
v_snd_2977_ = lean_ctor_get(v_a_2973_, 1);
lean_inc(v_snd_2977_);
lean_dec(v_a_2973_);
v_snd_2978_ = lean_ctor_get(v_snd_2977_, 1);
lean_inc(v_snd_2978_);
lean_dec(v_snd_2977_);
v___x_2979_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_withSimpTheoremInfos_spec__0___redArg___lam__0___closed__1));
v___x_2980_ = lean_unsigned_to_nat(3u);
v___x_2981_ = l_Lean_Expr_isAppOfArity(v_snd_2978_, v___x_2979_, v___x_2980_);
if (v___x_2981_ == 0)
{
lean_object* v___x_2982_; lean_object* v___x_2983_; uint8_t v___x_2984_; 
v___x_2982_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___closed__4));
v___x_2983_ = lean_unsigned_to_nat(2u);
v___x_2984_ = l_Lean_Expr_isAppOfArity(v_snd_2978_, v___x_2982_, v___x_2983_);
if (v___x_2984_ == 0)
{
lean_object* v___x_2986_; 
lean_dec(v_snd_2978_);
lean_dec_ref(v_snd_2962_);
if (v_isShared_2976_ == 0)
{
lean_ctor_set(v___x_2975_, 0, v___x_2970_);
v___x_2986_ = v___x_2975_;
goto v_reusejp_2985_;
}
else
{
lean_object* v_reuseFailAlloc_2987_; 
v_reuseFailAlloc_2987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2987_, 0, v___x_2970_);
v___x_2986_ = v_reuseFailAlloc_2987_;
goto v_reusejp_2985_;
}
v_reusejp_2985_:
{
return v___x_2986_;
}
}
else
{
lean_object* v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; 
lean_del_object(v___x_2975_);
v___x_2988_ = l_Lean_Expr_appFn_x21(v_snd_2978_);
v___x_2989_ = l_Lean_Expr_appArg_x21(v___x_2988_);
lean_dec_ref(v___x_2988_);
v___x_2990_ = l_Lean_Expr_appArg_x21(v_snd_2978_);
lean_dec(v_snd_2978_);
lean_inc_ref(v_snd_2962_);
v___y_2860_ = v_snd_2962_;
v___y_2861_ = v_snd_2962_;
v_fst_2862_ = v___x_2989_;
v_snd_2863_ = v___x_2990_;
goto v___jp_2859_;
}
}
else
{
lean_object* v___x_2991_; lean_object* v___x_2992_; lean_object* v___x_2993_; 
lean_del_object(v___x_2975_);
v___x_2991_ = l_Lean_Expr_appFn_x21(v_snd_2978_);
v___x_2992_ = l_Lean_Expr_appArg_x21(v___x_2991_);
lean_dec_ref(v___x_2991_);
v___x_2993_ = l_Lean_Expr_appArg_x21(v_snd_2978_);
lean_dec(v_snd_2978_);
lean_inc_ref(v_snd_2962_);
v___y_2860_ = v_snd_2962_;
v___y_2861_ = v_snd_2962_;
v_fst_2862_ = v___x_2992_;
v_snd_2863_ = v___x_2993_;
goto v___jp_2859_;
}
}
}
else
{
lean_object* v_a_2995_; lean_object* v___x_2997_; uint8_t v_isShared_2998_; uint8_t v_isSharedCheck_3002_; 
lean_dec_ref(v_snd_2962_);
v_a_2995_ = lean_ctor_get(v___x_2972_, 0);
v_isSharedCheck_3002_ = !lean_is_exclusive(v___x_2972_);
if (v_isSharedCheck_3002_ == 0)
{
v___x_2997_ = v___x_2972_;
v_isShared_2998_ = v_isSharedCheck_3002_;
goto v_resetjp_2996_;
}
else
{
lean_inc(v_a_2995_);
lean_dec(v___x_2972_);
v___x_2997_ = lean_box(0);
v_isShared_2998_ = v_isSharedCheck_3002_;
goto v_resetjp_2996_;
}
v_resetjp_2996_:
{
lean_object* v___x_3000_; 
if (v_isShared_2998_ == 0)
{
v___x_3000_ = v___x_2997_;
goto v_reusejp_2999_;
}
else
{
lean_object* v_reuseFailAlloc_3001_; 
v_reuseFailAlloc_3001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3001_, 0, v_a_2995_);
v___x_3000_ = v_reuseFailAlloc_3001_;
goto v_reusejp_2999_;
}
v_reusejp_2999_:
{
return v___x_3000_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___boxed(lean_object* v___x_3017_, lean_object* v_x_3018_, lean_object* v_ty_x27_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_){
_start:
{
lean_object* v_res_3025_; 
v_res_3025_ = lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1(v___x_3017_, v_x_3018_, v_ty_x27_3019_, v___y_3020_, v___y_3021_, v___y_3022_, v___y_3023_);
lean_dec(v___y_3023_);
lean_dec_ref(v___y_3022_);
lean_dec(v___y_3021_);
lean_dec_ref(v___y_3020_);
lean_dec_ref(v_ty_x27_3019_);
lean_dec_ref(v_x_3018_);
return v_res_3025_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2(lean_object* v_declName_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_){
_start:
{
lean_object* v___x_3032_; lean_object* v_config_3033_; uint8_t v_trackZetaDelta_3034_; lean_object* v_zetaDeltaSet_3035_; lean_object* v_lctx_3036_; lean_object* v_localInstances_3037_; lean_object* v_defEqCtx_x3f_3038_; lean_object* v_synthPendingDepth_3039_; lean_object* v_customCanUnfoldPredicate_x3f_3040_; uint8_t v_univApprox_3041_; uint8_t v_inTypeClassResolution_3042_; uint8_t v_cacheInferType_3043_; lean_object* v___x_3044_; 
v___x_3032_ = l_Lean_Meta_simpGlobalConfig;
v_config_3033_ = lean_ctor_get(v___x_3032_, 0);
v_trackZetaDelta_3034_ = lean_ctor_get_uint8(v___y_3027_, sizeof(void*)*7);
v_zetaDeltaSet_3035_ = lean_ctor_get(v___y_3027_, 1);
v_lctx_3036_ = lean_ctor_get(v___y_3027_, 2);
v_localInstances_3037_ = lean_ctor_get(v___y_3027_, 3);
v_defEqCtx_x3f_3038_ = lean_ctor_get(v___y_3027_, 4);
v_synthPendingDepth_3039_ = lean_ctor_get(v___y_3027_, 5);
v_customCanUnfoldPredicate_x3f_3040_ = lean_ctor_get(v___y_3027_, 6);
v_univApprox_3041_ = lean_ctor_get_uint8(v___y_3027_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3042_ = lean_ctor_get_uint8(v___y_3027_, sizeof(void*)*7 + 2);
v_cacheInferType_3043_ = lean_ctor_get_uint8(v___y_3027_, sizeof(void*)*7 + 3);
lean_inc(v_declName_3026_);
v___x_3044_ = lp_batteries_Batteries_Tactic_Lint_isSimpTheorem___redArg(v_declName_3026_, v___y_3030_);
if (lean_obj_tag(v___x_3044_) == 0)
{
lean_object* v_a_3045_; lean_object* v___x_3047_; uint8_t v_isShared_3048_; uint8_t v_isSharedCheck_3073_; 
v_a_3045_ = lean_ctor_get(v___x_3044_, 0);
v_isSharedCheck_3073_ = !lean_is_exclusive(v___x_3044_);
if (v_isSharedCheck_3073_ == 0)
{
v___x_3047_ = v___x_3044_;
v_isShared_3048_ = v_isSharedCheck_3073_;
goto v_resetjp_3046_;
}
else
{
lean_inc(v_a_3045_);
lean_dec(v___x_3044_);
v___x_3047_ = lean_box(0);
v_isShared_3048_ = v_isSharedCheck_3073_;
goto v_resetjp_3046_;
}
v_resetjp_3046_:
{
uint8_t v___x_3049_; 
v___x_3049_ = lean_unbox(v_a_3045_);
lean_dec(v_a_3045_);
if (v___x_3049_ == 0)
{
lean_object* v___x_3050_; lean_object* v___x_3052_; 
lean_dec(v_declName_3026_);
v___x_3050_ = lean_box(0);
if (v_isShared_3048_ == 0)
{
lean_ctor_set(v___x_3047_, 0, v___x_3050_);
v___x_3052_ = v___x_3047_;
goto v_reusejp_3051_;
}
else
{
lean_object* v_reuseFailAlloc_3053_; 
v_reuseFailAlloc_3053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3053_, 0, v___x_3050_);
v___x_3052_ = v_reuseFailAlloc_3053_;
goto v_reusejp_3051_;
}
v_reusejp_3051_:
{
return v___x_3052_;
}
}
else
{
uint64_t v___x_3054_; uint8_t v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; 
lean_del_object(v___x_3047_);
v___x_3054_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_config_3033_);
v___x_3055_ = 2;
lean_inc_ref(v_config_3033_);
v___x_3056_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_3056_, 0, v_config_3033_);
lean_ctor_set_uint64(v___x_3056_, sizeof(void*)*1, v___x_3054_);
v___x_3057_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3055_, v___x_3056_);
lean_inc(v_customCanUnfoldPredicate_x3f_3040_);
lean_inc(v_synthPendingDepth_3039_);
lean_inc(v_defEqCtx_x3f_3038_);
lean_inc_ref(v_localInstances_3037_);
lean_inc_ref(v_lctx_3036_);
lean_inc(v_zetaDeltaSet_3035_);
v___x_3058_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3058_, 0, v___x_3057_);
lean_ctor_set(v___x_3058_, 1, v_zetaDeltaSet_3035_);
lean_ctor_set(v___x_3058_, 2, v_lctx_3036_);
lean_ctor_set(v___x_3058_, 3, v_localInstances_3037_);
lean_ctor_set(v___x_3058_, 4, v_defEqCtx_x3f_3038_);
lean_ctor_set(v___x_3058_, 5, v_synthPendingDepth_3039_);
lean_ctor_set(v___x_3058_, 6, v_customCanUnfoldPredicate_x3f_3040_);
lean_ctor_set_uint8(v___x_3058_, sizeof(void*)*7, v_trackZetaDelta_3034_);
lean_ctor_set_uint8(v___x_3058_, sizeof(void*)*7 + 1, v_univApprox_3041_);
lean_ctor_set_uint8(v___x_3058_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3042_);
lean_ctor_set_uint8(v___x_3058_, sizeof(void*)*7 + 3, v_cacheInferType_3043_);
v___x_3059_ = l_Lean_getConstInfo___at___00Lean_Meta_mkSimpEntryOfDeclToUnfold_spec__0(v_declName_3026_, v___x_3058_, v___y_3028_, v___y_3029_, v___y_3030_);
if (lean_obj_tag(v___x_3059_) == 0)
{
lean_object* v_a_3060_; lean_object* v___x_3061_; lean_object* v___f_3062_; uint8_t v___x_3063_; lean_object* v___x_3064_; 
v_a_3060_ = lean_ctor_get(v___x_3059_, 0);
lean_inc(v_a_3060_);
lean_dec_ref_known(v___x_3059_, 1);
v___x_3061_ = l_Lean_ConstantInfo_type(v_a_3060_);
lean_dec(v_a_3060_);
lean_inc_ref(v___x_3061_);
v___f_3062_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_simpComm___lam__1___boxed), 8, 1);
lean_closure_set(v___f_3062_, 0, v___x_3061_);
v___x_3063_ = 0;
v___x_3064_ = l_Lean_Meta_forallTelescopeReducing___at___00__private_Lean_Meta_Tactic_Simp_SimpTheorems_0__Lean_Meta_shouldPreprocess_spec__0___redArg(v___x_3061_, v___f_3062_, v___x_3063_, v___x_3063_, v___x_3058_, v___y_3028_, v___y_3029_, v___y_3030_);
lean_dec_ref_known(v___x_3058_, 7);
return v___x_3064_;
}
else
{
lean_object* v_a_3065_; lean_object* v___x_3067_; uint8_t v_isShared_3068_; uint8_t v_isSharedCheck_3072_; 
lean_dec_ref_known(v___x_3058_, 7);
v_a_3065_ = lean_ctor_get(v___x_3059_, 0);
v_isSharedCheck_3072_ = !lean_is_exclusive(v___x_3059_);
if (v_isSharedCheck_3072_ == 0)
{
v___x_3067_ = v___x_3059_;
v_isShared_3068_ = v_isSharedCheck_3072_;
goto v_resetjp_3066_;
}
else
{
lean_inc(v_a_3065_);
lean_dec(v___x_3059_);
v___x_3067_ = lean_box(0);
v_isShared_3068_ = v_isSharedCheck_3072_;
goto v_resetjp_3066_;
}
v_resetjp_3066_:
{
lean_object* v___x_3070_; 
if (v_isShared_3068_ == 0)
{
v___x_3070_ = v___x_3067_;
goto v_reusejp_3069_;
}
else
{
lean_object* v_reuseFailAlloc_3071_; 
v_reuseFailAlloc_3071_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3071_, 0, v_a_3065_);
v___x_3070_ = v_reuseFailAlloc_3071_;
goto v_reusejp_3069_;
}
v_reusejp_3069_:
{
return v___x_3070_;
}
}
}
}
}
}
else
{
lean_object* v_a_3074_; lean_object* v___x_3076_; uint8_t v_isShared_3077_; uint8_t v_isSharedCheck_3081_; 
lean_dec(v_declName_3026_);
v_a_3074_ = lean_ctor_get(v___x_3044_, 0);
v_isSharedCheck_3081_ = !lean_is_exclusive(v___x_3044_);
if (v_isSharedCheck_3081_ == 0)
{
v___x_3076_ = v___x_3044_;
v_isShared_3077_ = v_isSharedCheck_3081_;
goto v_resetjp_3075_;
}
else
{
lean_inc(v_a_3074_);
lean_dec(v___x_3044_);
v___x_3076_ = lean_box(0);
v_isShared_3077_ = v_isSharedCheck_3081_;
goto v_resetjp_3075_;
}
v_resetjp_3075_:
{
lean_object* v___x_3079_; 
if (v_isShared_3077_ == 0)
{
v___x_3079_ = v___x_3076_;
goto v_reusejp_3078_;
}
else
{
lean_object* v_reuseFailAlloc_3080_; 
v_reuseFailAlloc_3080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3080_, 0, v_a_3074_);
v___x_3079_ = v_reuseFailAlloc_3080_;
goto v_reusejp_3078_;
}
v_reusejp_3078_:
{
return v___x_3079_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2___boxed(lean_object* v_declName_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_){
_start:
{
lean_object* v_res_3088_; 
v_res_3088_ = lp_batteries_Batteries_Tactic_Lint_simpComm___lam__2(v_declName_3082_, v___y_3083_, v___y_3084_, v___y_3085_, v___y_3086_);
lean_dec(v___y_3086_);
lean_dec_ref(v___y_3085_);
lean_dec(v___y_3084_);
lean_dec_ref(v___y_3083_);
return v_res_3088_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3(void){
_start:
{
lean_object* v___x_3093_; lean_object* v___x_3094_; 
v___x_3093_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___closed__2));
v___x_3094_ = l_Lean_MessageData_ofFormat(v___x_3093_);
return v___x_3094_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6(void){
_start:
{
lean_object* v___x_3098_; lean_object* v___x_3099_; 
v___x_3098_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___closed__5));
v___x_3099_ = l_Lean_MessageData_ofFormat(v___x_3098_);
return v___x_3099_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7(void){
_start:
{
uint8_t v___x_3100_; lean_object* v___x_3101_; lean_object* v___x_3102_; lean_object* v___f_3103_; lean_object* v___x_3104_; 
v___x_3100_ = 1;
v___x_3101_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6, &lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__6);
v___x_3102_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3, &lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__3);
v___f_3103_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_simpComm___closed__0));
v___x_3104_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_3104_, 0, v___f_3103_);
lean_ctor_set(v___x_3104_, 1, v___x_3102_);
lean_ctor_set(v___x_3104_, 2, v___x_3101_);
lean_ctor_set_uint8(v___x_3104_, sizeof(void*)*3, v___x_3100_);
return v___x_3104_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_simpComm(void){
_start:
{
lean_object* v___x_3105_; 
v___x_3105_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7, &lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_simpComm___closed__7);
return v___x_3105_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6(lean_object* v_x_3106_, lean_object* v_keys_3107_, lean_object* v_v_3108_, lean_object* v_k_3109_, lean_object* v_as_3110_, lean_object* v_k_3111_, lean_object* v_x_3112_, lean_object* v_x_3113_, lean_object* v_x_3114_, lean_object* v_x_3115_){
_start:
{
lean_object* v___x_3116_; 
v___x_3116_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___redArg(v_x_3106_, v_keys_3107_, v_v_3108_, v_k_3109_, v_as_3110_, v_k_3111_, v_x_3112_, v_x_3113_);
return v___x_3116_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6___boxed(lean_object* v_x_3117_, lean_object* v_keys_3118_, lean_object* v_v_3119_, lean_object* v_k_3120_, lean_object* v_as_3121_, lean_object* v_k_3122_, lean_object* v_x_3123_, lean_object* v_x_3124_, lean_object* v_x_3125_, lean_object* v_x_3126_){
_start:
{
lean_object* v_res_3127_; 
v_res_3127_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Lean_Meta_DiscrTree_insert___at___00Batteries_Tactic_Lint_simpComm_spec__0_spec__0_spec__1_spec__3_spec__6(v_x_3117_, v_keys_3118_, v_v_3119_, v_k_3120_, v_as_3121_, v_k_3122_, v_x_3123_, v_x_3124_, v_x_3125_, v_x_3126_);
lean_dec_ref(v_k_3122_);
lean_dec_ref(v_keys_3118_);
lean_dec(v_x_3117_);
return v_res_3127_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_SimpTheorems(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Simp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_SimpTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree_Util(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_OpenPrivate(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Lint_Simp(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_OpenPrivate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Lint_Simp_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Simp_249398888____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Tactic_Lint_linter_simpNF_respectTransparency = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Tactic_Lint_linter_simpNF_respectTransparency);
lean_dec_ref(res);
lp_batteries_Batteries_Tactic_Lint_simpNF = _init_lp_batteries_Batteries_Tactic_Lint_simpNF();
lean_mark_persistent(lp_batteries_Batteries_Tactic_Lint_simpNF);
lp_batteries_LibraryNote_simp_x2dnormal__form = _init_lp_batteries_LibraryNote_simp_x2dnormal__form();
lean_mark_persistent(lp_batteries_LibraryNote_simp_x2dnormal__form);
lp_batteries_Batteries_Tactic_Lint_simpComm = _init_lp_batteries_Batteries_Tactic_Lint_simpComm();
lean_mark_persistent(lp_batteries_Batteries_Tactic_Lint_simpComm);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree_Util(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_OpenPrivate(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_SimpTheorems(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Lint_Simp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_OpenPrivate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_SimpTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Lint_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Lint_Simp(builtin);
}
#ifdef __cplusplus
}
#endif
