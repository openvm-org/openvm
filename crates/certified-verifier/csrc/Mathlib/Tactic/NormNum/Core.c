// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Core
// Imports: public import Init public meta import Init public meta import Mathlib.Lean.Expr.Rat public import Mathlib.Tactic.Hint public import Mathlib.Tactic.NormNum.Result public meta import Mathlib.Util.Qq public import Lean.Elab.Tactic.Try public meta import Lean.Meta.Tactic.Try.Collect
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Meta_DiscrTree_Key_hash(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_instBEqKey_beq(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Meta_DiscrTree_Key_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Environment_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* l_Lean_registerScopedEnvExtensionUnsafe___redArg(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_addCore___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_isExtraRevModUseExt;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getEntries___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_ensureAttrDeclIsMeta(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_getFVars(lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_lambdaMetaTelescope(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withAutoBoundImplicit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedFileMap_default;
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Elab_InfoTree_substitute(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_decl_get_sorry_dep(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
uint8_t lp_mathlib_Lean_Expr_isExplicitNumber(lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isRawNatLit(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_dischargeGround___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_preDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_mkEqTransResultStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_postDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Level_succ___override(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Elab_Tactic_elabSimpConfigCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabSimpArgs(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocs___redArg(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_getLhs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_applySimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
uint8_t l_Lean_PersistentHashMap_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpArgs;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
static const lean_string_object lp_mathlib_norm__num___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "norm_num"};
static const lean_object* lp_mathlib_norm__num___closed__0 = (const lean_object*)&lp_mathlib_norm__num___closed__0_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_norm__num___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 111, 111, 127, 5, 137, 232, 113)}};
static const lean_object* lp_mathlib_norm__num___closed__1 = (const lean_object*)&lp_mathlib_norm__num___closed__1_value;
static const lean_string_object lp_mathlib_norm__num___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_norm__num___closed__2 = (const lean_object*)&lp_mathlib_norm__num___closed__2_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_norm__num___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_norm__num___closed__3 = (const lean_object*)&lp_mathlib_norm__num___closed__3_value;
static const lean_string_object lp_mathlib_norm__num___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "norm_num "};
static const lean_object* lp_mathlib_norm__num___closed__4 = (const lean_object*)&lp_mathlib_norm__num___closed__4_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_norm__num___closed__5 = (const lean_object*)&lp_mathlib_norm__num___closed__5_value;
static const lean_string_object lp_mathlib_norm__num___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_norm__num___closed__6 = (const lean_object*)&lp_mathlib_norm__num___closed__6_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_norm__num___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_norm__num___closed__7 = (const lean_object*)&lp_mathlib_norm__num___closed__7_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_norm__num___closed__8 = (const lean_object*)&lp_mathlib_norm__num___closed__8_value;
static const lean_string_object lp_mathlib_norm__num___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_norm__num___closed__9 = (const lean_object*)&lp_mathlib_norm__num___closed__9_value;
static const lean_string_object lp_mathlib_norm__num___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_norm__num___closed__10 = (const lean_object*)&lp_mathlib_norm__num___closed__10_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__10_value)}};
static const lean_object* lp_mathlib_norm__num___closed__11 = (const lean_object*)&lp_mathlib_norm__num___closed__11_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__8_value),((lean_object*)&lp_mathlib_norm__num___closed__9_value),((lean_object*)&lp_mathlib_norm__num___closed__11_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_norm__num___closed__12 = (const lean_object*)&lp_mathlib_norm__num___closed__12_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__3_value),((lean_object*)&lp_mathlib_norm__num___closed__5_value),((lean_object*)&lp_mathlib_norm__num___closed__12_value)}};
static const lean_object* lp_mathlib_norm__num___closed__13 = (const lean_object*)&lp_mathlib_norm__num___closed__13_value;
static const lean_ctor_object lp_mathlib_norm__num___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_norm__num___closed__13_value)}};
static const lean_object* lp_mathlib_norm__num___closed__14 = (const lean_object*)&lp_mathlib_norm__num___closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_norm__num = (const lean_object*)&lp_mathlib_norm__num___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_norm__num___closed__0_value),LEAN_SCALAR_PTR_LITERAL(129, 6, 67, 216, 78, 160, 93, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(136, 91, 19, 79, 227, 8, 8, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__9_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__9_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__9_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__10_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__9_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(4, 131, 4, 200, 195, 80, 13, 54)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__10_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__10_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__11_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__10_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(101, 255, 149, 212, 211, 158, 79, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__11_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__11_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__12_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__11_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(192, 162, 196, 144, 144, 245, 35, 126)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__12_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__12_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__14_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__12_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(108, 49, 157, 160, 229, 143, 59, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__14_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__14_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__15_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__14_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 187, 212, 46, 178, 20, 202, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__15_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__15_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__16_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__16_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__16_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__17_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__15_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__16_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 29, 94, 147, 200, 190, 48, 26)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__17_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__17_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__18_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__18_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__18_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__19_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__17_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__18_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 131, 92, 9, 194, 8, 146, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__19_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__19_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__20_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__19_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(90, 59, 244, 63, 24, 218, 139, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__20_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__20_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__21_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__20_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(47, 216, 254, 138, 151, 102, 117, 18)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__21_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__21_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__22_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__21_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(88, 113, 255, 83, 224, 139, 201, 104)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__22_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__22_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__23_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__22_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__9_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 219, 147, 165, 105, 34, 74, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__23_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__23_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__25_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__25_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__25_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__27_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__27_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__27_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3_value;
static const lean_array_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__9_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "declName"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__13_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__14_value),LEAN_SCALAR_PTR_LITERAL(113, 211, 58, 33, 138, 196, 138, 106)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "decl_name%"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "NormNumExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 88, 8, 16, 12, 47, 33, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0_value),((lean_object*)&lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2(lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.Meta.DiscrTree.Basic"};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Lean.Meta.DiscrTree.insertKeyValue"};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "invalid key sequence"};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "normNumExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(39, 135, 78, 52, 36, 72, 139, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_normNumExt;
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = " failed "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " ==> "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isFalse ("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isTrue ("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__18 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "isNat "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isNegNat "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__22 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isNNRat "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__24 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__24_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__26 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__26_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "isNegNNRat "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__27 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__27_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = ": no norm_nums apply"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "instAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__1_value),LEAN_SCALAR_PTR_LITERAL(251, 118, 25, 219, 65, 50, 236, 110)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "raw_refl"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__13_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__5_value),LEAN_SCALAR_PTR_LITERAL(163, 125, 133, 66, 231, 251, 113, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveInt___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveRat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBool___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__3_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__4_value),LEAN_SCALAR_PTR_LITERAL(147, 220, 216, 40, 239, 165, 44, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "not"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__3_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__7_value),LEAN_SCALAR_PTR_LITERAL(109, 253, 239, 29, 129, 48, 54, 85)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rec"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 96, 18, 10, 225, 62, 78, 176)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__16_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__22_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_eval(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_eraseCore(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "' does not have [norm_num] attribute"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__7_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__8_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__9_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__10_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__14_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3___boxed, .m_arity = 5, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__15_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__16_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5(uint8_t, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__1 = (const lean_object*)&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2;
static const lean_string_object lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "recording extra reverse use of current module"};
static const lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__3 = (const lean_object*)&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 67, .m_capacity = 67, .m_length = 66, .m_data = "invalid attribute 'norm_num', declaration is in an imported module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__23_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1919626320) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(40, 238, 118, 144, 169, 78, 77, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__25_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(143, 160, 245, 135, 31, 45, 95, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__27_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(119, 7, 147, 92, 223, 164, 45, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(98, 209, 193, 20, 192, 248, 125, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed, .m_arity = 8, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "adds a norm_num extension"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_norm__num___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_methods___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_methods___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_methods___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_methods___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_Simp_dischargeGround___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_methods___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_methods___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_self"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__8_value),LEAN_SCALAR_PTR_LITERAL(224, 148, 98, 216, 254, 239, 13, 169)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "iff_self"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__10_value),LEAN_SCALAR_PTR_LITERAL(79, 255, 41, 65, 134, 196, 244, 123)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "normNum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 202, 36, 226, 215, 147, 189, 233)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_norm__num___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_normNum;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "normNum1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 38, 130, 145, 101, 42, 168, 96)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "norm_num1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNum1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNum1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_normNum1;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "normNum1Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 69, 219, 46, 200, 52, 75, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_normNum1Conv = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNum1Conv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "normNumConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumConv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(126, 21, 223, 182, 164, 177, 138, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumConv___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumConv___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumConv___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_normNumConv;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "normNumCmd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 85, 210, 123, 182, 207, 49, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "#norm_num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " :"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNum___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__11_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__13_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_normNumCmd___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_normNumCmd;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "command#conv_=>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(254, 184, 104, 200, 83, 192, 77, 132)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__0 = (const lean_object*)&lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__1 = (const lean_object*)&lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__1_value;
static const lean_string_object lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__2 = (const lean_object*)&lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__2_value;
static const lean_array_object lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__3 = (const lean_object*)&lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_91_ = lean_unsigned_to_nat(3041507515u);
v___x_92_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__23_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_93_ = l_Lean_Name_num___override(v___x_92_, v___x_91_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__25_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_96_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__24_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_);
v___x_97_ = l_Lean_Name_str___override(v___x_96_, v___x_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__27_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_100_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__26_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_);
v___x_101_ = l_Lean_Name_str___override(v___x_100_, v___x_99_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_102_ = lean_unsigned_to_nat(2u);
v___x_103_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__28_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_);
v___x_104_ = l_Lean_Name_num___override(v___x_103_, v___x_102_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_106_; uint8_t v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_107_ = 0;
v___x_108_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__29_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_);
v___x_109_ = l_Lean_registerTraceClass(v___x_106_, v___x_107_, v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2____boxed(lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_();
return v_res_111_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__9));
v___x_138_ = l_Lean_mkAtom(v___x_137_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_139_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__11);
v___x_140_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_141_ = lean_array_push(v___x_140_, v___x_139_);
return v___x_141_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__16));
v___x_151_ = l_Lean_mkAtom(v___x_150_);
return v___x_151_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_152_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__17);
v___x_153_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_154_ = lean_array_push(v___x_153_, v___x_152_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__18);
v___x_156_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__15));
v___x_157_ = lean_box(2);
v___x_158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_156_);
lean_ctor_set(v___x_158_, 2, v___x_155_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__19);
v___x_160_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__12);
v___x_161_ = lean_array_push(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__20);
v___x_163_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__10));
v___x_164_ = lean_box(2);
v___x_165_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
lean_ctor_set(v___x_165_, 2, v___x_162_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22(void){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_166_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__21);
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_168_ = lean_array_push(v___x_167_, v___x_166_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_169_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__22);
v___x_170_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8));
v___x_171_ = lean_box(2);
v___x_172_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v___x_170_);
lean_ctor_set(v___x_172_, 2, v___x_169_);
return v___x_172_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_173_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__23);
v___x_174_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_175_ = lean_array_push(v___x_174_, v___x_173_);
return v___x_175_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_176_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__24);
v___x_177_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6));
v___x_178_ = lean_box(2);
v___x_179_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
lean_ctor_set(v___x_179_, 1, v___x_177_);
lean_ctor_set(v___x_179_, 2, v___x_176_);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_180_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__25);
v___x_181_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_182_ = lean_array_push(v___x_181_, v___x_180_);
return v___x_182_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27(void){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_183_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__26);
v___x_184_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3));
v___x_185_ = lean_box(2);
v___x_186_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v___x_184_);
lean_ctor_set(v___x_186_, 2, v___x_183_);
return v___x_186_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam(void){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__27);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1(lean_object* v_n_194_, lean_object* v_env_195_, lean_object* v_opts_196_){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1));
v___x_198_ = l_Lean_Environment_evalConstCheck___redArg(v_env_195_, v_opts_196_, v___x_197_, v_n_194_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___boxed(lean_object* v_n_199_, lean_object* v_env_200_, lean_object* v_opts_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1(v_n_199_, v_env_200_, v_opts_201_);
lean_dec_ref(v_opts_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg(lean_object* v_e_203_){
_start:
{
if (lean_obj_tag(v_e_203_) == 0)
{
lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_213_; 
v_a_205_ = lean_ctor_get(v_e_203_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v_e_203_);
if (v_isSharedCheck_213_ == 0)
{
v___x_207_ = v_e_203_;
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_dec(v_e_203_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_209_ = lean_mk_io_user_error(v_a_205_);
if (v_isShared_208_ == 0)
{
lean_ctor_set_tag(v___x_207_, 1);
lean_ctor_set(v___x_207_, 0, v___x_209_);
v___x_211_ = v___x_207_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_209_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
else
{
lean_object* v_a_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_221_; 
v_a_214_ = lean_ctor_get(v_e_203_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v_e_203_);
if (v_isSharedCheck_221_ == 0)
{
v___x_216_ = v_e_203_;
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_a_214_);
lean_dec(v_e_203_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_219_; 
if (v_isShared_217_ == 0)
{
lean_ctor_set_tag(v___x_216_, 0);
v___x_219_ = v___x_216_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_a_214_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg___boxed(lean_object* v_e_222_, lean_object* v_a_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg(v_e_222_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0(lean_object* v_00_u03b1_225_, lean_object* v_e_226_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg(v_e_226_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___boxed(lean_object* v_00_u03b1_229_, lean_object* v_e_230_, lean_object* v_a_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0(v_00_u03b1_229_, v_e_230_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt(lean_object* v_n_233_, lean_object* v_a_234_){
_start:
{
lean_object* v_env_236_; lean_object* v_opts_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_env_236_ = lean_ctor_get(v_a_234_, 0);
v_opts_237_ = lean_ctor_get(v_a_234_, 1);
v___x_238_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_mkNormNumExt_unsafe__1___closed__1));
lean_inc_ref(v_env_236_);
v___x_239_ = l_Lean_Environment_evalConstCheck___redArg(v_env_236_, v_opts_237_, v___x_238_, v_n_233_);
v___x_240_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Meta_NormNum_mkNormNumExt_spec__0___redArg(v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt___boxed(lean_object* v_n_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt(v_n_241_, v_a_242_);
lean_dec_ref(v_a_242_);
return v_res_244_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0(void){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_245_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__0);
v___x_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0(lean_object* v_00_u03b2_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0___closed__1);
return v___x_249_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0(void){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_250_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1(void){
_start:
{
lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_251_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__0);
v___x_252_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2(void){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_instInhabitedNormNums_default_spec__0(lean_box(0));
return v___x_253_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3(void){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_254_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2);
v___x_255_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__1);
v___x_256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v___x_254_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default(void){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3);
return v___x_257_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums(void){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default;
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v___y_259_){
_start:
{
lean_inc_ref(v___y_259_);
return v___y_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(v___y_260_);
lean_dec_ref(v___y_260_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10(lean_object* v_xs_262_, lean_object* v_v_263_, lean_object* v_i_264_){
_start:
{
lean_object* v___x_265_; uint8_t v___x_266_; 
v___x_265_ = lean_array_get_size(v_xs_262_);
v___x_266_ = lean_nat_dec_lt(v_i_264_, v___x_265_);
if (v___x_266_ == 0)
{
lean_object* v___x_267_; 
lean_dec(v_i_264_);
v___x_267_ = lean_box(0);
return v___x_267_;
}
else
{
lean_object* v___x_268_; uint8_t v___x_269_; 
v___x_268_ = lean_array_fget_borrowed(v_xs_262_, v_i_264_);
v___x_269_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v___x_268_, v_v_263_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = lean_unsigned_to_nat(1u);
v___x_271_ = lean_nat_add(v_i_264_, v___x_270_);
lean_dec(v_i_264_);
v_i_264_ = v___x_271_;
goto _start;
}
else
{
lean_object* v___x_273_; 
v___x_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_273_, 0, v_i_264_);
return v___x_273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10___boxed(lean_object* v_xs_274_, lean_object* v_v_275_, lean_object* v_i_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10(v_xs_274_, v_v_275_, v_i_276_);
lean_dec(v_v_275_);
lean_dec_ref(v_xs_274_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4(lean_object* v_xs_278_, lean_object* v_v_279_){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_280_ = lean_unsigned_to_nat(0u);
v___x_281_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__10(v_xs_278_, v_v_279_, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4___boxed(lean_object* v_xs_282_, lean_object* v_v_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4(v_xs_282_, v_v_283_);
lean_dec(v_v_283_);
lean_dec_ref(v_xs_282_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14___redArg(lean_object* v_x_285_, lean_object* v_x_286_, lean_object* v_x_287_, lean_object* v_x_288_){
_start:
{
lean_object* v_ks_289_; lean_object* v_vs_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_314_; 
v_ks_289_ = lean_ctor_get(v_x_285_, 0);
v_vs_290_ = lean_ctor_get(v_x_285_, 1);
v_isSharedCheck_314_ = !lean_is_exclusive(v_x_285_);
if (v_isSharedCheck_314_ == 0)
{
v___x_292_ = v_x_285_;
v_isShared_293_ = v_isSharedCheck_314_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_vs_290_);
lean_inc(v_ks_289_);
lean_dec(v_x_285_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_314_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = lean_array_get_size(v_ks_289_);
v___x_295_ = lean_nat_dec_lt(v_x_286_, v___x_294_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_299_; 
lean_dec(v_x_286_);
v___x_296_ = lean_array_push(v_ks_289_, v_x_287_);
v___x_297_ = lean_array_push(v_vs_290_, v_x_288_);
if (v_isShared_293_ == 0)
{
lean_ctor_set(v___x_292_, 1, v___x_297_);
lean_ctor_set(v___x_292_, 0, v___x_296_);
v___x_299_ = v___x_292_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_300_; 
v_reuseFailAlloc_300_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_300_, 0, v___x_296_);
lean_ctor_set(v_reuseFailAlloc_300_, 1, v___x_297_);
v___x_299_ = v_reuseFailAlloc_300_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
return v___x_299_;
}
}
else
{
lean_object* v_k_x27_301_; uint8_t v___x_302_; 
v_k_x27_301_ = lean_array_fget_borrowed(v_ks_289_, v_x_286_);
v___x_302_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_287_, v_k_x27_301_);
if (v___x_302_ == 0)
{
lean_object* v___x_304_; 
if (v_isShared_293_ == 0)
{
v___x_304_ = v___x_292_;
goto v_reusejp_303_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v_ks_289_);
lean_ctor_set(v_reuseFailAlloc_308_, 1, v_vs_290_);
v___x_304_ = v_reuseFailAlloc_308_;
goto v_reusejp_303_;
}
v_reusejp_303_:
{
lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_305_ = lean_unsigned_to_nat(1u);
v___x_306_ = lean_nat_add(v_x_286_, v___x_305_);
lean_dec(v_x_286_);
v_x_285_ = v___x_304_;
v_x_286_ = v___x_306_;
goto _start;
}
}
else
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_312_; 
v___x_309_ = lean_array_fset(v_ks_289_, v_x_286_, v_x_287_);
v___x_310_ = lean_array_fset(v_vs_290_, v_x_286_, v_x_288_);
lean_dec(v_x_286_);
if (v_isShared_293_ == 0)
{
lean_ctor_set(v___x_292_, 1, v___x_310_);
lean_ctor_set(v___x_292_, 0, v___x_309_);
v___x_312_ = v___x_292_;
goto v_reusejp_311_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_309_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v___x_310_);
v___x_312_ = v_reuseFailAlloc_313_;
goto v_reusejp_311_;
}
v_reusejp_311_:
{
return v___x_312_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12___redArg(lean_object* v_n_315_, lean_object* v_k_316_, lean_object* v_v_317_){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_unsigned_to_nat(0u);
v___x_319_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14___redArg(v_n_315_, v___x_318_, v_k_316_, v_v_317_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(lean_object* v_x_321_, size_t v_x_322_, size_t v_x_323_, lean_object* v_x_324_, lean_object* v_x_325_){
_start:
{
if (lean_obj_tag(v_x_321_) == 0)
{
lean_object* v_es_326_; size_t v___x_327_; size_t v___x_328_; lean_object* v_j_329_; lean_object* v___x_330_; uint8_t v___x_331_; 
v_es_326_ = lean_ctor_get(v_x_321_, 0);
v___x_327_ = ((size_t)31ULL);
v___x_328_ = lean_usize_land(v_x_322_, v___x_327_);
v_j_329_ = lean_usize_to_nat(v___x_328_);
v___x_330_ = lean_array_get_size(v_es_326_);
v___x_331_ = lean_nat_dec_lt(v_j_329_, v___x_330_);
if (v___x_331_ == 0)
{
lean_dec(v_j_329_);
lean_dec(v_x_325_);
lean_dec(v_x_324_);
return v_x_321_;
}
else
{
lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_370_; 
lean_inc_ref(v_es_326_);
v_isSharedCheck_370_ = !lean_is_exclusive(v_x_321_);
if (v_isSharedCheck_370_ == 0)
{
lean_object* v_unused_371_; 
v_unused_371_ = lean_ctor_get(v_x_321_, 0);
lean_dec(v_unused_371_);
v___x_333_ = v_x_321_;
v_isShared_334_ = v_isSharedCheck_370_;
goto v_resetjp_332_;
}
else
{
lean_dec(v_x_321_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_370_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v_v_335_; lean_object* v___x_336_; lean_object* v_xs_x27_337_; lean_object* v___y_339_; 
v_v_335_ = lean_array_fget(v_es_326_, v_j_329_);
v___x_336_ = lean_box(0);
v_xs_x27_337_ = lean_array_fset(v_es_326_, v_j_329_, v___x_336_);
switch(lean_obj_tag(v_v_335_))
{
case 0:
{
lean_object* v_key_344_; lean_object* v_val_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_355_; 
v_key_344_ = lean_ctor_get(v_v_335_, 0);
v_val_345_ = lean_ctor_get(v_v_335_, 1);
v_isSharedCheck_355_ = !lean_is_exclusive(v_v_335_);
if (v_isSharedCheck_355_ == 0)
{
v___x_347_ = v_v_335_;
v_isShared_348_ = v_isSharedCheck_355_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_val_345_);
lean_inc(v_key_344_);
lean_dec(v_v_335_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_355_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
uint8_t v___x_349_; 
v___x_349_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_324_, v_key_344_);
if (v___x_349_ == 0)
{
lean_object* v___x_350_; lean_object* v___x_351_; 
lean_del_object(v___x_347_);
v___x_350_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_344_, v_val_345_, v_x_324_, v_x_325_);
v___x_351_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
v___y_339_ = v___x_351_;
goto v___jp_338_;
}
else
{
lean_object* v___x_353_; 
lean_dec(v_val_345_);
lean_dec(v_key_344_);
if (v_isShared_348_ == 0)
{
lean_ctor_set(v___x_347_, 1, v_x_325_);
lean_ctor_set(v___x_347_, 0, v_x_324_);
v___x_353_ = v___x_347_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_x_324_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v_x_325_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
v___y_339_ = v___x_353_;
goto v___jp_338_;
}
}
}
}
case 1:
{
lean_object* v_node_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_368_; 
v_node_356_ = lean_ctor_get(v_v_335_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v_v_335_);
if (v_isSharedCheck_368_ == 0)
{
v___x_358_ = v_v_335_;
v_isShared_359_ = v_isSharedCheck_368_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_node_356_);
lean_dec(v_v_335_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_368_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
size_t v___x_360_; size_t v___x_361_; size_t v___x_362_; size_t v___x_363_; lean_object* v___x_364_; lean_object* v___x_366_; 
v___x_360_ = ((size_t)5ULL);
v___x_361_ = lean_usize_shift_right(v_x_322_, v___x_360_);
v___x_362_ = ((size_t)1ULL);
v___x_363_ = lean_usize_add(v_x_323_, v___x_362_);
v___x_364_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_node_356_, v___x_361_, v___x_363_, v_x_324_, v_x_325_);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 0, v___x_364_);
v___x_366_ = v___x_358_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v___x_364_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
v___y_339_ = v___x_366_;
goto v___jp_338_;
}
}
}
default: 
{
lean_object* v___x_369_; 
v___x_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_369_, 0, v_x_324_);
lean_ctor_set(v___x_369_, 1, v_x_325_);
v___y_339_ = v___x_369_;
goto v___jp_338_;
}
}
v___jp_338_:
{
lean_object* v___x_340_; lean_object* v___x_342_; 
v___x_340_ = lean_array_fset(v_xs_x27_337_, v_j_329_, v___y_339_);
lean_dec(v_j_329_);
if (v_isShared_334_ == 0)
{
lean_ctor_set(v___x_333_, 0, v___x_340_);
v___x_342_ = v___x_333_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_340_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
}
else
{
lean_object* v_ks_372_; lean_object* v_vs_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_393_; 
v_ks_372_ = lean_ctor_get(v_x_321_, 0);
v_vs_373_ = lean_ctor_get(v_x_321_, 1);
v_isSharedCheck_393_ = !lean_is_exclusive(v_x_321_);
if (v_isSharedCheck_393_ == 0)
{
v___x_375_ = v_x_321_;
v_isShared_376_ = v_isSharedCheck_393_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_vs_373_);
lean_inc(v_ks_372_);
lean_dec(v_x_321_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_393_;
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
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v_ks_372_);
lean_ctor_set(v_reuseFailAlloc_392_, 1, v_vs_373_);
v___x_378_ = v_reuseFailAlloc_392_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v_newNode_379_; uint8_t v___y_381_; size_t v___x_387_; uint8_t v___x_388_; 
v_newNode_379_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12___redArg(v___x_378_, v_x_324_, v_x_325_);
v___x_387_ = ((size_t)7ULL);
v___x_388_ = lean_usize_dec_le(v___x_387_, v_x_323_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; uint8_t v___x_391_; 
v___x_389_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_379_);
v___x_390_ = lean_unsigned_to_nat(4u);
v___x_391_ = lean_nat_dec_lt(v___x_389_, v___x_390_);
lean_dec(v___x_389_);
v___y_381_ = v___x_391_;
goto v___jp_380_;
}
else
{
v___y_381_ = v___x_388_;
goto v___jp_380_;
}
v___jp_380_:
{
if (v___y_381_ == 0)
{
lean_object* v_ks_382_; lean_object* v_vs_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v_ks_382_ = lean_ctor_get(v_newNode_379_, 0);
lean_inc_ref(v_ks_382_);
v_vs_383_ = lean_ctor_get(v_newNode_379_, 1);
lean_inc_ref(v_vs_383_);
lean_dec_ref(v_newNode_379_);
v___x_384_ = lean_unsigned_to_nat(0u);
v___x_385_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0);
v___x_386_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg(v_x_323_, v_ks_382_, v_vs_383_, v___x_384_, v___x_385_);
lean_dec_ref(v_vs_383_);
lean_dec_ref(v_ks_382_);
return v___x_386_;
}
else
{
return v_newNode_379_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg(size_t v_depth_394_, lean_object* v_keys_395_, lean_object* v_vals_396_, lean_object* v_i_397_, lean_object* v_entries_398_){
_start:
{
lean_object* v___x_399_; uint8_t v___x_400_; 
v___x_399_ = lean_array_get_size(v_keys_395_);
v___x_400_ = lean_nat_dec_lt(v_i_397_, v___x_399_);
if (v___x_400_ == 0)
{
lean_dec(v_i_397_);
return v_entries_398_;
}
else
{
lean_object* v_k_401_; lean_object* v_v_402_; uint64_t v___x_403_; size_t v_h_404_; size_t v___x_405_; lean_object* v___x_406_; size_t v___x_407_; size_t v___x_408_; size_t v___x_409_; size_t v_h_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v_k_401_ = lean_array_fget_borrowed(v_keys_395_, v_i_397_);
v_v_402_ = lean_array_fget_borrowed(v_vals_396_, v_i_397_);
v___x_403_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_401_);
v_h_404_ = lean_uint64_to_usize(v___x_403_);
v___x_405_ = ((size_t)5ULL);
v___x_406_ = lean_unsigned_to_nat(1u);
v___x_407_ = ((size_t)1ULL);
v___x_408_ = lean_usize_sub(v_depth_394_, v___x_407_);
v___x_409_ = lean_usize_mul(v___x_405_, v___x_408_);
v_h_410_ = lean_usize_shift_right(v_h_404_, v___x_409_);
v___x_411_ = lean_nat_add(v_i_397_, v___x_406_);
lean_dec(v_i_397_);
lean_inc(v_v_402_);
lean_inc(v_k_401_);
v___x_412_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_entries_398_, v_h_410_, v_depth_394_, v_k_401_, v_v_402_);
v_i_397_ = v___x_411_;
v_entries_398_ = v___x_412_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg___boxed(lean_object* v_depth_414_, lean_object* v_keys_415_, lean_object* v_vals_416_, lean_object* v_i_417_, lean_object* v_entries_418_){
_start:
{
size_t v_depth_boxed_419_; lean_object* v_res_420_; 
v_depth_boxed_419_ = lean_unbox_usize(v_depth_414_);
lean_dec(v_depth_414_);
v_res_420_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg(v_depth_boxed_419_, v_keys_415_, v_vals_416_, v_i_417_, v_entries_418_);
lean_dec_ref(v_vals_416_);
lean_dec_ref(v_keys_415_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_x_421_, lean_object* v_x_422_, lean_object* v_x_423_, lean_object* v_x_424_, lean_object* v_x_425_){
_start:
{
size_t v_x_2297__boxed_426_; size_t v_x_2298__boxed_427_; lean_object* v_res_428_; 
v_x_2297__boxed_426_ = lean_unbox_usize(v_x_422_);
lean_dec(v_x_422_);
v_x_2298__boxed_427_ = lean_unbox_usize(v_x_423_);
lean_dec(v_x_423_);
v_res_428_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_x_421_, v_x_2297__boxed_426_, v_x_2298__boxed_427_, v_x_424_, v_x_425_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(lean_object* v_vs_429_, lean_object* v_v_430_, lean_object* v_i_431_){
_start:
{
lean_object* v___x_432_; uint8_t v___x_433_; 
v___x_432_ = lean_array_get_size(v_vs_429_);
v___x_433_ = lean_nat_dec_lt(v_i_431_, v___x_432_);
if (v___x_433_ == 0)
{
lean_object* v___x_434_; 
lean_dec(v_i_431_);
v___x_434_ = lean_array_push(v_vs_429_, v_v_430_);
return v___x_434_;
}
else
{
lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_435_ = lean_unsigned_to_nat(1u);
v___x_436_ = lean_nat_add(v_i_431_, v___x_435_);
lean_dec(v_i_431_);
v_i_431_ = v___x_436_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_vs_438_, lean_object* v_v_439_){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(v_vs_438_, v_v_439_, v___x_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(lean_object* v_x_442_, lean_object* v_keys_443_, lean_object* v_v_444_, lean_object* v_k_445_, lean_object* v_x_446_){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v_c_449_; lean_object* v___x_450_; 
v___x_447_ = lean_unsigned_to_nat(1u);
v___x_448_ = lean_nat_add(v_x_442_, v___x_447_);
v_c_449_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_443_, v_v_444_, v___x_448_);
lean_dec(v___x_448_);
v___x_450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_450_, 0, v_k_445_);
lean_ctor_set(v___x_450_, 1, v_c_449_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0___boxed(lean_object* v_x_451_, lean_object* v_keys_452_, lean_object* v_v_453_, lean_object* v_k_454_, lean_object* v_x_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_451_, v_keys_452_, v_v_453_, v_k_454_, v_x_455_);
lean_dec_ref(v_keys_452_);
lean_dec(v_x_451_);
return v_res_456_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(lean_object* v_a_457_, lean_object* v_b_458_){
_start:
{
lean_object* v_fst_459_; lean_object* v_fst_460_; uint8_t v___x_461_; 
v_fst_459_ = lean_ctor_get(v_a_457_, 0);
v_fst_460_ = lean_ctor_get(v_b_458_, 0);
v___x_461_ = l_Lean_Meta_DiscrTree_Key_lt(v_fst_459_, v_fst_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1___boxed(lean_object* v_a_462_, lean_object* v_b_463_){
_start:
{
uint8_t v_res_464_; lean_object* v_r_465_; 
v_res_464_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_a_462_, v_b_463_);
lean_dec_ref(v_b_463_);
lean_dec_ref(v_a_462_);
v_r_465_ = lean_box(v_res_464_);
return v_r_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(lean_object* v_x_470_, lean_object* v_keys_471_, lean_object* v_v_472_, lean_object* v_k_473_, lean_object* v_as_474_, lean_object* v_k_475_, lean_object* v_x_476_, lean_object* v_x_477_){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v_mid_480_; lean_object* v_midVal_481_; uint8_t v___x_482_; 
v___x_478_ = lean_nat_add(v_x_476_, v_x_477_);
v___x_479_ = lean_unsigned_to_nat(1u);
v_mid_480_ = lean_nat_shiftr(v___x_478_, v___x_479_);
lean_dec(v___x_478_);
v_midVal_481_ = lean_array_fget(v_as_474_, v_mid_480_);
v___x_482_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_midVal_481_, v_k_475_);
if (v___x_482_ == 0)
{
uint8_t v___x_483_; 
lean_dec(v_x_477_);
v___x_483_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_475_, v_midVal_481_);
if (v___x_483_ == 0)
{
lean_object* v___x_484_; uint8_t v___x_485_; 
lean_dec(v_x_476_);
v___x_484_ = lean_array_get_size(v_as_474_);
v___x_485_ = lean_nat_dec_lt(v_mid_480_, v___x_484_);
if (v___x_485_ == 0)
{
lean_dec(v_midVal_481_);
lean_dec(v_mid_480_);
lean_dec(v_k_473_);
lean_dec_ref(v_v_472_);
return v_as_474_;
}
else
{
lean_object* v_snd_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_498_; 
v_snd_486_ = lean_ctor_get(v_midVal_481_, 1);
v_isSharedCheck_498_ = !lean_is_exclusive(v_midVal_481_);
if (v_isSharedCheck_498_ == 0)
{
lean_object* v_unused_499_; 
v_unused_499_ = lean_ctor_get(v_midVal_481_, 0);
lean_dec(v_unused_499_);
v___x_488_ = v_midVal_481_;
v_isShared_489_ = v_isSharedCheck_498_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_snd_486_);
lean_dec(v_midVal_481_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_498_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v___x_490_; lean_object* v_xs_x27_491_; lean_object* v___x_492_; lean_object* v_c_493_; lean_object* v___x_495_; 
v___x_490_ = lean_box(0);
v_xs_x27_491_ = lean_array_fset(v_as_474_, v_mid_480_, v___x_490_);
v___x_492_ = lean_nat_add(v_x_470_, v___x_479_);
v_c_493_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(v_keys_471_, v_v_472_, v___x_492_, v_snd_486_);
lean_dec(v___x_492_);
if (v_isShared_489_ == 0)
{
lean_ctor_set(v___x_488_, 1, v_c_493_);
lean_ctor_set(v___x_488_, 0, v_k_473_);
v___x_495_ = v___x_488_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_497_; 
v_reuseFailAlloc_497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_497_, 0, v_k_473_);
lean_ctor_set(v_reuseFailAlloc_497_, 1, v_c_493_);
v___x_495_ = v_reuseFailAlloc_497_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
lean_object* v___x_496_; 
v___x_496_ = lean_array_fset(v_xs_x27_491_, v_mid_480_, v___x_495_);
lean_dec(v_mid_480_);
return v___x_496_;
}
}
}
}
else
{
lean_dec(v_midVal_481_);
v_x_477_ = v_mid_480_;
goto _start;
}
}
else
{
uint8_t v___x_501_; 
lean_dec(v_midVal_481_);
v___x_501_ = lean_nat_dec_eq(v_mid_480_, v_x_476_);
if (v___x_501_ == 0)
{
lean_dec(v_x_476_);
v_x_476_ = v_mid_480_;
goto _start;
}
else
{
lean_object* v___x_503_; lean_object* v_c_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v_j_507_; lean_object* v_as_508_; lean_object* v___x_509_; 
lean_dec(v_mid_480_);
lean_dec(v_x_477_);
v___x_503_ = lean_nat_add(v_x_470_, v___x_479_);
v_c_504_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_471_, v_v_472_, v___x_503_);
lean_dec(v___x_503_);
v___x_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_505_, 0, v_k_473_);
lean_ctor_set(v___x_505_, 1, v_c_504_);
v___x_506_ = lean_nat_add(v_x_476_, v___x_479_);
lean_dec(v_x_476_);
v_j_507_ = lean_array_get_size(v_as_474_);
v_as_508_ = lean_array_push(v_as_474_, v___x_505_);
v___x_509_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_506_, v_as_508_, v_j_507_);
lean_dec(v___x_506_);
return v___x_509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object* v_x_510_, lean_object* v_keys_511_, lean_object* v_v_512_, lean_object* v_k_513_, lean_object* v_as_514_, lean_object* v_k_515_){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_516_ = lean_array_get_size(v_as_514_);
v___x_517_ = lean_unsigned_to_nat(0u);
v___x_518_ = lean_nat_dec_eq(v___x_516_, v___x_517_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; uint8_t v___x_520_; 
v___x_519_ = lean_array_fget_borrowed(v_as_514_, v___x_517_);
v___x_520_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_515_, v___x_519_);
if (v___x_520_ == 0)
{
uint8_t v___x_521_; 
v___x_521_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v___x_519_, v_k_515_);
if (v___x_521_ == 0)
{
uint8_t v___x_522_; 
v___x_522_ = lean_nat_dec_lt(v___x_517_, v___x_516_);
if (v___x_522_ == 0)
{
lean_dec(v_k_513_);
lean_dec_ref(v_v_512_);
return v_as_514_;
}
else
{
lean_object* v___x_523_; lean_object* v_xs_x27_524_; lean_object* v___x_525_; lean_object* v___x_526_; 
lean_inc(v___x_519_);
v___x_523_ = lean_box(0);
v_xs_x27_524_ = lean_array_fset(v_as_514_, v___x_517_, v___x_523_);
v___x_525_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v___x_519_);
v___x_526_ = lean_array_fset(v_xs_x27_524_, v___x_517_, v___x_525_);
return v___x_526_;
}
}
else
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; uint8_t v___x_530_; 
v___x_527_ = lean_unsigned_to_nat(1u);
v___x_528_ = lean_nat_sub(v___x_516_, v___x_527_);
v___x_529_ = lean_array_fget_borrowed(v_as_514_, v___x_528_);
v___x_530_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v___x_529_, v_k_515_);
if (v___x_530_ == 0)
{
uint8_t v___x_531_; 
v___x_531_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_515_, v___x_529_);
if (v___x_531_ == 0)
{
uint8_t v___x_532_; 
v___x_532_ = lean_nat_dec_lt(v___x_528_, v___x_516_);
if (v___x_532_ == 0)
{
lean_dec(v___x_528_);
lean_dec(v_k_513_);
lean_dec_ref(v_v_512_);
return v_as_514_;
}
else
{
lean_object* v___x_533_; lean_object* v_xs_x27_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
lean_inc(v___x_529_);
v___x_533_ = lean_box(0);
v_xs_x27_534_ = lean_array_fset(v_as_514_, v___x_528_, v___x_533_);
v___x_535_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v___x_529_);
v___x_536_ = lean_array_fset(v_xs_x27_534_, v___x_528_, v___x_535_);
lean_dec(v___x_528_);
return v___x_536_;
}
}
else
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v_as_514_, v_k_515_, v___x_517_, v___x_528_);
return v___x_537_;
}
}
else
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
lean_dec(v___x_528_);
v___x_538_ = lean_box(0);
v___x_539_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v___x_538_);
v___x_540_ = lean_array_push(v_as_514_, v___x_539_);
return v___x_540_;
}
}
}
else
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v_as_543_; lean_object* v___x_544_; 
v___x_541_ = lean_box(0);
v___x_542_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v___x_541_);
v_as_543_ = lean_array_push(v_as_514_, v___x_542_);
v___x_544_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_517_, v_as_543_, v___x_516_);
return v___x_544_;
}
}
else
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_545_ = lean_box(0);
v___x_546_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_510_, v_keys_511_, v_v_512_, v_k_513_, v___x_545_);
v___x_547_ = lean_array_push(v_as_514_, v___x_546_);
return v___x_547_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_keys_548_, lean_object* v_v_549_, lean_object* v_x_550_, lean_object* v_x_551_){
_start:
{
lean_object* v_vs_552_; lean_object* v_children_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_570_; 
v_vs_552_ = lean_ctor_get(v_x_551_, 0);
v_children_553_ = lean_ctor_get(v_x_551_, 1);
v_isSharedCheck_570_ = !lean_is_exclusive(v_x_551_);
if (v_isSharedCheck_570_ == 0)
{
v___x_555_ = v_x_551_;
v_isShared_556_ = v_isSharedCheck_570_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_children_553_);
lean_inc(v_vs_552_);
lean_dec(v_x_551_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_570_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; uint8_t v___x_558_; 
v___x_557_ = lean_array_get_size(v_keys_548_);
v___x_558_ = lean_nat_dec_lt(v_x_550_, v___x_557_);
if (v___x_558_ == 0)
{
lean_object* v___x_559_; lean_object* v___x_561_; 
v___x_559_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_vs_552_, v_v_549_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_559_);
v___x_561_ = v___x_555_;
goto v_reusejp_560_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v___x_559_);
lean_ctor_set(v_reuseFailAlloc_562_, 1, v_children_553_);
v___x_561_ = v_reuseFailAlloc_562_;
goto v_reusejp_560_;
}
v_reusejp_560_:
{
return v___x_561_;
}
}
else
{
lean_object* v_k_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v_c_566_; lean_object* v___x_568_; 
v_k_563_ = lean_array_fget_borrowed(v_keys_548_, v_x_550_);
v___x_564_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__1));
lean_inc_n(v_k_563_, 2);
v___x_565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_565_, 0, v_k_563_);
lean_ctor_set(v___x_565_, 1, v___x_564_);
v_c_566_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2(v_x_550_, v_keys_548_, v_v_549_, v_k_563_, v_children_553_, v___x_565_);
lean_dec_ref_known(v___x_565_, 2);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 1, v_c_566_);
v___x_568_ = v___x_555_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_vs_552_);
lean_ctor_set(v_reuseFailAlloc_569_, 1, v_c_566_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(lean_object* v_x_571_, lean_object* v_keys_572_, lean_object* v_v_573_, lean_object* v_k_574_, lean_object* v_x_575_){
_start:
{
lean_object* v_snd_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_586_; 
v_snd_576_ = lean_ctor_get(v_x_575_, 1);
v_isSharedCheck_586_ = !lean_is_exclusive(v_x_575_);
if (v_isSharedCheck_586_ == 0)
{
lean_object* v_unused_587_; 
v_unused_587_ = lean_ctor_get(v_x_575_, 0);
lean_dec(v_unused_587_);
v___x_578_ = v_x_575_;
v_isShared_579_ = v_isSharedCheck_586_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_snd_576_);
lean_dec(v_x_575_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_586_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v_c_582_; lean_object* v___x_584_; 
v___x_580_ = lean_unsigned_to_nat(1u);
v___x_581_ = lean_nat_add(v_x_571_, v___x_580_);
v_c_582_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(v_keys_572_, v_v_573_, v___x_581_, v_snd_576_);
lean_dec(v___x_581_);
if (v_isShared_579_ == 0)
{
lean_ctor_set(v___x_578_, 1, v_c_582_);
lean_ctor_set(v___x_578_, 0, v_k_574_);
v___x_584_ = v___x_578_;
goto v_reusejp_583_;
}
else
{
lean_object* v_reuseFailAlloc_585_; 
v_reuseFailAlloc_585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_585_, 0, v_k_574_);
lean_ctor_set(v_reuseFailAlloc_585_, 1, v_c_582_);
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
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2___boxed(lean_object* v_x_588_, lean_object* v_keys_589_, lean_object* v_v_590_, lean_object* v_k_591_, lean_object* v_x_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_588_, v_keys_589_, v_v_590_, v_k_591_, v_x_592_);
lean_dec_ref(v_keys_589_);
lean_dec(v_x_588_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_keys_594_, lean_object* v_v_595_, lean_object* v_x_596_, lean_object* v_x_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(v_keys_594_, v_v_595_, v_x_596_, v_x_597_);
lean_dec(v_x_596_);
lean_dec_ref(v_keys_594_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object* v_x_599_, lean_object* v_keys_600_, lean_object* v_v_601_, lean_object* v_k_602_, lean_object* v_as_603_, lean_object* v_k_604_, lean_object* v_x_605_, lean_object* v_x_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_x_599_, v_keys_600_, v_v_601_, v_k_602_, v_as_603_, v_k_604_, v_x_605_, v_x_606_);
lean_dec_ref(v_k_604_);
lean_dec_ref(v_keys_600_);
lean_dec(v_x_599_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object* v_x_608_, lean_object* v_keys_609_, lean_object* v_v_610_, lean_object* v_k_611_, lean_object* v_as_612_, lean_object* v_k_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2(v_x_608_, v_keys_609_, v_v_610_, v_k_611_, v_as_612_, v_k_613_);
lean_dec_ref(v_k_613_);
lean_dec_ref(v_keys_609_);
lean_dec(v_x_608_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object* v_keys_615_, lean_object* v_v_616_, lean_object* v_x_617_){
_start:
{
if (lean_obj_tag(v_x_617_) == 0)
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_618_ = lean_unsigned_to_nat(1u);
v___x_619_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_615_, v_v_616_, v___x_618_);
v___x_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
return v___x_620_;
}
else
{
lean_object* v_val_621_; lean_object* v___x_623_; uint8_t v_isShared_624_; uint8_t v_isSharedCheck_630_; 
v_val_621_ = lean_ctor_get(v_x_617_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v_x_617_);
if (v_isSharedCheck_630_ == 0)
{
v___x_623_ = v_x_617_;
v_isShared_624_ = v_isSharedCheck_630_;
goto v_resetjp_622_;
}
else
{
lean_inc(v_val_621_);
lean_dec(v_x_617_);
v___x_623_ = lean_box(0);
v_isShared_624_ = v_isSharedCheck_630_;
goto v_resetjp_622_;
}
v_resetjp_622_:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_628_; 
v___x_625_ = lean_unsigned_to_nat(1u);
v___x_626_ = lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0(v_keys_615_, v_v_616_, v___x_625_, v_val_621_);
if (v_isShared_624_ == 0)
{
lean_ctor_set(v___x_623_, 0, v___x_626_);
v___x_628_ = v___x_623_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v___x_626_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object* v_keys_631_, lean_object* v_v_632_, lean_object* v_x_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_631_, v_v_632_, v_x_633_);
lean_dec_ref(v_keys_631_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_keys_635_, lean_object* v_v_636_, lean_object* v_x_637_, size_t v_x_638_, size_t v_x_639_, lean_object* v_x_640_){
_start:
{
if (lean_obj_tag(v_x_637_) == 0)
{
lean_object* v_es_641_; size_t v___x_642_; size_t v___x_643_; lean_object* v_j_644_; lean_object* v___x_645_; uint8_t v___x_646_; 
v_es_641_ = lean_ctor_get(v_x_637_, 0);
v___x_642_ = ((size_t)31ULL);
v___x_643_ = lean_usize_land(v_x_638_, v___x_642_);
v_j_644_ = lean_usize_to_nat(v___x_643_);
v___x_645_ = lean_array_get_size(v_es_641_);
v___x_646_ = lean_nat_dec_lt(v_j_644_, v___x_645_);
if (v___x_646_ == 0)
{
lean_dec(v_j_644_);
lean_dec(v_x_640_);
lean_dec_ref(v_v_636_);
return v_x_637_;
}
else
{
lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_714_; 
lean_inc_ref(v_es_641_);
v_isSharedCheck_714_ = !lean_is_exclusive(v_x_637_);
if (v_isSharedCheck_714_ == 0)
{
lean_object* v_unused_715_; 
v_unused_715_ = lean_ctor_get(v_x_637_, 0);
lean_dec(v_unused_715_);
v___x_648_ = v_x_637_;
v_isShared_649_ = v_isSharedCheck_714_;
goto v_resetjp_647_;
}
else
{
lean_dec(v_x_637_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_714_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v_v_650_; lean_object* v___x_651_; lean_object* v_xs_x27_652_; lean_object* v___y_654_; 
v_v_650_ = lean_array_fget(v_es_641_, v_j_644_);
v___x_651_ = lean_box(0);
v_xs_x27_652_ = lean_array_fset(v_es_641_, v_j_644_, v___x_651_);
switch(lean_obj_tag(v_v_650_))
{
case 0:
{
lean_object* v_key_659_; lean_object* v_val_660_; uint8_t v___x_661_; 
v_key_659_ = lean_ctor_get(v_v_650_, 0);
v_val_660_ = lean_ctor_get(v_v_650_, 1);
v___x_661_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_640_, v_key_659_);
if (v___x_661_ == 0)
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = lean_box(0);
v___x_663_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_635_, v_v_636_, v___x_662_);
if (lean_obj_tag(v___x_663_) == 0)
{
lean_dec(v_x_640_);
v___y_654_ = v_v_650_;
goto v___jp_653_;
}
else
{
lean_object* v_val_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_672_; 
lean_inc(v_val_660_);
lean_inc(v_key_659_);
lean_dec_ref_known(v_v_650_, 2);
v_val_664_ = lean_ctor_get(v___x_663_, 0);
v_isSharedCheck_672_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_672_ == 0)
{
v___x_666_ = v___x_663_;
v_isShared_667_ = v_isSharedCheck_672_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_val_664_);
lean_dec(v___x_663_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_672_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_668_; lean_object* v___x_670_; 
v___x_668_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_659_, v_val_660_, v_x_640_, v_val_664_);
if (v_isShared_667_ == 0)
{
lean_ctor_set(v___x_666_, 0, v___x_668_);
v___x_670_ = v___x_666_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v___x_668_);
v___x_670_ = v_reuseFailAlloc_671_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
v___y_654_ = v___x_670_;
goto v___jp_653_;
}
}
}
}
else
{
lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_683_; 
lean_inc(v_val_660_);
v_isSharedCheck_683_ = !lean_is_exclusive(v_v_650_);
if (v_isSharedCheck_683_ == 0)
{
lean_object* v_unused_684_; lean_object* v_unused_685_; 
v_unused_684_ = lean_ctor_get(v_v_650_, 1);
lean_dec(v_unused_684_);
v_unused_685_ = lean_ctor_get(v_v_650_, 0);
lean_dec(v_unused_685_);
v___x_674_ = v_v_650_;
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
else
{
lean_dec(v_v_650_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_676_, 0, v_val_660_);
v___x_677_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_635_, v_v_636_, v___x_676_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v___x_678_; 
lean_del_object(v___x_674_);
lean_dec(v_x_640_);
v___x_678_ = lean_box(2);
v___y_654_ = v___x_678_;
goto v___jp_653_;
}
else
{
lean_object* v_val_679_; lean_object* v___x_681_; 
v_val_679_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_val_679_);
lean_dec_ref_known(v___x_677_, 1);
if (v_isShared_675_ == 0)
{
lean_ctor_set(v___x_674_, 1, v_val_679_);
lean_ctor_set(v___x_674_, 0, v_x_640_);
v___x_681_ = v___x_674_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v_x_640_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_val_679_);
v___x_681_ = v_reuseFailAlloc_682_;
goto v_reusejp_680_;
}
v_reusejp_680_:
{
v___y_654_ = v___x_681_;
goto v___jp_653_;
}
}
}
}
}
case 1:
{
lean_object* v_node_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_709_; 
v_node_686_ = lean_ctor_get(v_v_650_, 0);
v_isSharedCheck_709_ = !lean_is_exclusive(v_v_650_);
if (v_isSharedCheck_709_ == 0)
{
v___x_688_ = v_v_650_;
v_isShared_689_ = v_isSharedCheck_709_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_node_686_);
lean_dec(v_v_650_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_709_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
size_t v___x_690_; size_t v___x_691_; size_t v___x_692_; size_t v___x_693_; lean_object* v_newNode_694_; lean_object* v___x_695_; 
v___x_690_ = ((size_t)5ULL);
v___x_691_ = lean_usize_shift_right(v_x_638_, v___x_690_);
v___x_692_ = ((size_t)1ULL);
v___x_693_ = lean_usize_add(v_x_639_, v___x_692_);
v_newNode_694_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1(v_keys_635_, v_v_636_, v_node_686_, v___x_691_, v___x_693_, v_x_640_);
lean_inc_ref(v_newNode_694_);
v___x_695_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_694_);
if (lean_obj_tag(v___x_695_) == 0)
{
lean_object* v___x_697_; 
if (v_isShared_689_ == 0)
{
lean_ctor_set(v___x_688_, 0, v_newNode_694_);
v___x_697_ = v___x_688_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v_newNode_694_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
v___y_654_ = v___x_697_;
goto v___jp_653_;
}
}
else
{
lean_object* v_val_699_; lean_object* v_fst_700_; lean_object* v_snd_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_708_; 
lean_dec_ref(v_newNode_694_);
lean_del_object(v___x_688_);
v_val_699_ = lean_ctor_get(v___x_695_, 0);
lean_inc(v_val_699_);
lean_dec_ref_known(v___x_695_, 1);
v_fst_700_ = lean_ctor_get(v_val_699_, 0);
v_snd_701_ = lean_ctor_get(v_val_699_, 1);
v_isSharedCheck_708_ = !lean_is_exclusive(v_val_699_);
if (v_isSharedCheck_708_ == 0)
{
v___x_703_ = v_val_699_;
v_isShared_704_ = v_isSharedCheck_708_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_snd_701_);
lean_inc(v_fst_700_);
lean_dec(v_val_699_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_708_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
lean_object* v___x_706_; 
if (v_isShared_704_ == 0)
{
v___x_706_ = v___x_703_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v_fst_700_);
lean_ctor_set(v_reuseFailAlloc_707_, 1, v_snd_701_);
v___x_706_ = v_reuseFailAlloc_707_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
v___y_654_ = v___x_706_;
goto v___jp_653_;
}
}
}
}
}
default: 
{
lean_object* v___x_710_; lean_object* v___x_711_; 
v___x_710_ = lean_box(0);
v___x_711_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_635_, v_v_636_, v___x_710_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_dec(v_x_640_);
v___y_654_ = v_v_650_;
goto v___jp_653_;
}
else
{
lean_object* v_val_712_; lean_object* v___x_713_; 
v_val_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc(v_val_712_);
lean_dec_ref_known(v___x_711_, 1);
v___x_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_713_, 0, v_x_640_);
lean_ctor_set(v___x_713_, 1, v_val_712_);
v___y_654_ = v___x_713_;
goto v___jp_653_;
}
}
}
v___jp_653_:
{
lean_object* v___x_655_; lean_object* v___x_657_; 
v___x_655_ = lean_array_fset(v_xs_x27_652_, v_j_644_, v___y_654_);
lean_dec(v_j_644_);
if (v_isShared_649_ == 0)
{
lean_ctor_set(v___x_648_, 0, v___x_655_);
v___x_657_ = v___x_648_;
goto v_reusejp_656_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v___x_655_);
v___x_657_ = v_reuseFailAlloc_658_;
goto v_reusejp_656_;
}
v_reusejp_656_:
{
return v___x_657_;
}
}
}
}
}
else
{
lean_object* v_ks_716_; lean_object* v_vs_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_750_; 
v_ks_716_ = lean_ctor_get(v_x_637_, 0);
v_vs_717_ = lean_ctor_get(v_x_637_, 1);
v_isSharedCheck_750_ = !lean_is_exclusive(v_x_637_);
if (v_isSharedCheck_750_ == 0)
{
v___x_719_ = v_x_637_;
v_isShared_720_ = v_isSharedCheck_750_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_vs_717_);
lean_inc(v_ks_716_);
lean_dec(v_x_637_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_750_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_721_; 
v___x_721_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__4(v_ks_716_, v_x_640_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_object* v___x_723_; 
if (v_isShared_720_ == 0)
{
v___x_723_ = v___x_719_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v_ks_716_);
lean_ctor_set(v_reuseFailAlloc_728_, 1, v_vs_717_);
v___x_723_ = v_reuseFailAlloc_728_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = lean_box(0);
v___x_725_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_635_, v_v_636_, v___x_724_);
if (lean_obj_tag(v___x_725_) == 0)
{
lean_dec(v_x_640_);
return v___x_723_;
}
else
{
lean_object* v_val_726_; lean_object* v___x_727_; 
v_val_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_val_726_);
lean_dec_ref_known(v___x_725_, 1);
v___x_727_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v___x_723_, v_x_638_, v_x_639_, v_x_640_, v_val_726_);
return v___x_727_;
}
}
}
else
{
lean_object* v_val_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_749_; 
v_val_729_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_749_ == 0)
{
v___x_731_ = v___x_721_;
v_isShared_732_ = v_isSharedCheck_749_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_val_729_);
lean_dec(v___x_721_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_749_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v_v_x27_733_; lean_object* v_keys_734_; lean_object* v_vals_735_; lean_object* v___x_737_; 
v_v_x27_733_ = lean_array_fget(v_vs_717_, v_val_729_);
lean_inc(v_val_729_);
v_keys_734_ = l_Array_eraseIdx___redArg(v_ks_716_, v_val_729_);
v_vals_735_ = l_Array_eraseIdx___redArg(v_vs_717_, v_val_729_);
if (v_isShared_732_ == 0)
{
lean_ctor_set(v___x_731_, 0, v_v_x27_733_);
v___x_737_ = v___x_731_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_v_x27_733_);
v___x_737_ = v_reuseFailAlloc_748_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
lean_object* v___x_738_; 
v___x_738_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_635_, v_v_636_, v___x_737_);
if (lean_obj_tag(v___x_738_) == 0)
{
lean_object* v___x_740_; 
lean_dec(v_x_640_);
if (v_isShared_720_ == 0)
{
lean_ctor_set(v___x_719_, 1, v_vals_735_);
lean_ctor_set(v___x_719_, 0, v_keys_734_);
v___x_740_ = v___x_719_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v_keys_734_);
lean_ctor_set(v_reuseFailAlloc_741_, 1, v_vals_735_);
v___x_740_ = v_reuseFailAlloc_741_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
return v___x_740_;
}
}
else
{
lean_object* v_val_742_; lean_object* v_keys_743_; lean_object* v_vals_744_; lean_object* v___x_746_; 
v_val_742_ = lean_ctor_get(v___x_738_, 0);
lean_inc(v_val_742_);
lean_dec_ref_known(v___x_738_, 1);
v_keys_743_ = lean_array_push(v_keys_734_, v_x_640_);
v_vals_744_ = lean_array_push(v_vals_735_, v_val_742_);
if (v_isShared_720_ == 0)
{
lean_ctor_set(v___x_719_, 1, v_vals_744_);
lean_ctor_set(v___x_719_, 0, v_keys_743_);
v___x_746_ = v___x_719_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_keys_743_);
lean_ctor_set(v_reuseFailAlloc_747_, 1, v_vals_744_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_keys_751_, lean_object* v_v_752_, lean_object* v_x_753_, lean_object* v_x_754_, lean_object* v_x_755_, lean_object* v_x_756_){
_start:
{
size_t v_x_2716__boxed_757_; size_t v_x_2717__boxed_758_; lean_object* v_res_759_; 
v_x_2716__boxed_757_ = lean_unbox_usize(v_x_754_);
lean_dec(v_x_754_);
v_x_2717__boxed_758_ = lean_unbox_usize(v_x_755_);
lean_dec(v_x_755_);
v_res_759_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1(v_keys_751_, v_v_752_, v_x_753_, v_x_2716__boxed_757_, v_x_2717__boxed_758_, v_x_756_);
lean_dec_ref(v_keys_751_);
return v_res_759_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0(void){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2(lean_object* v_msg_761_){
_start:
{
lean_object* v___x_762_; lean_object* v___x_763_; 
v___x_762_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0, &lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2___closed__0);
v___x_763_ = lean_panic_fn_borrowed(v___x_762_, v_msg_761_);
return v___x_763_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3(void){
_start:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_767_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__2));
v___x_768_ = lean_unsigned_to_nat(23u);
v___x_769_ = lean_unsigned_to_nat(166u);
v___x_770_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__1));
v___x_771_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__0));
v___x_772_ = l_mkPanicMessageWithDecl(v___x_771_, v___x_770_, v___x_769_, v___x_768_, v___x_767_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0(lean_object* v_d_773_, lean_object* v_keys_774_, lean_object* v_v_775_){
_start:
{
lean_object* v___x_776_; lean_object* v___x_777_; uint8_t v___x_778_; 
v___x_776_ = lean_array_get_size(v_keys_774_);
v___x_777_ = lean_unsigned_to_nat(0u);
v___x_778_ = lean_nat_dec_eq(v___x_776_, v___x_777_);
if (v___x_778_ == 0)
{
lean_object* v___x_779_; lean_object* v_k_780_; uint64_t v___x_781_; size_t v_h_782_; size_t v___x_783_; lean_object* v___x_784_; 
v___x_779_ = lean_box(0);
v_k_780_ = lean_array_get_borrowed(v___x_779_, v_keys_774_, v___x_777_);
v___x_781_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_780_);
v_h_782_ = lean_uint64_to_usize(v___x_781_);
v___x_783_ = ((size_t)1ULL);
lean_inc(v_k_780_);
v___x_784_ = lp_mathlib_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1(v_keys_774_, v_v_775_, v_d_773_, v_h_782_, v___x_783_, v_k_780_);
return v___x_784_;
}
else
{
lean_object* v___x_785_; lean_object* v___x_786_; 
lean_dec_ref(v_v_775_);
lean_dec_ref(v_d_773_);
v___x_785_ = lean_obj_once(&lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3, &lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___closed__3);
v___x_786_ = lp_mathlib_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__2(v___x_785_);
return v___x_786_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0___boxed(lean_object* v_d_787_, lean_object* v_keys_788_, lean_object* v_v_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0(v_d_787_, v_keys_788_, v_v_789_);
lean_dec_ref(v_keys_788_);
return v_res_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2(lean_object* v_snd_791_, lean_object* v_as_792_, size_t v_i_793_, size_t v_stop_794_, lean_object* v_b_795_){
_start:
{
uint8_t v___x_796_; 
v___x_796_ = lean_usize_dec_eq(v_i_793_, v_stop_794_);
if (v___x_796_ == 0)
{
lean_object* v___x_797_; lean_object* v___x_798_; size_t v___x_799_; size_t v___x_800_; 
v___x_797_ = lean_array_uget_borrowed(v_as_792_, v_i_793_);
lean_inc_ref(v_snd_791_);
v___x_798_ = lp_mathlib_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0(v_b_795_, v___x_797_, v_snd_791_);
v___x_799_ = ((size_t)1ULL);
v___x_800_ = lean_usize_add(v_i_793_, v___x_799_);
v_i_793_ = v___x_800_;
v_b_795_ = v___x_798_;
goto _start;
}
else
{
lean_dec_ref(v_snd_791_);
return v_b_795_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2___boxed(lean_object* v_snd_802_, lean_object* v_as_803_, lean_object* v_i_804_, lean_object* v_stop_805_, lean_object* v_b_806_){
_start:
{
size_t v_i_boxed_807_; size_t v_stop_boxed_808_; lean_object* v_res_809_; 
v_i_boxed_807_ = lean_unbox_usize(v_i_804_);
lean_dec(v_i_804_);
v_stop_boxed_808_ = lean_unbox_usize(v_stop_805_);
lean_dec(v_stop_805_);
v_res_809_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2(v_snd_802_, v_as_803_, v_i_boxed_807_, v_stop_boxed_808_, v_b_806_);
lean_dec_ref(v_as_803_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16(lean_object* v_xs_810_, lean_object* v_v_811_, lean_object* v_i_812_){
_start:
{
lean_object* v___x_813_; uint8_t v___x_814_; 
v___x_813_ = lean_array_get_size(v_xs_810_);
v___x_814_ = lean_nat_dec_lt(v_i_812_, v___x_813_);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; 
lean_dec(v_i_812_);
v___x_815_ = lean_box(0);
return v___x_815_;
}
else
{
lean_object* v___x_816_; uint8_t v___x_817_; 
v___x_816_ = lean_array_fget_borrowed(v_xs_810_, v_i_812_);
v___x_817_ = lean_name_eq(v___x_816_, v_v_811_);
if (v___x_817_ == 0)
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = lean_unsigned_to_nat(1u);
v___x_819_ = lean_nat_add(v_i_812_, v___x_818_);
lean_dec(v_i_812_);
v_i_812_ = v___x_819_;
goto _start;
}
else
{
lean_object* v___x_821_; 
v___x_821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_821_, 0, v_i_812_);
return v___x_821_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16___boxed(lean_object* v_xs_822_, lean_object* v_v_823_, lean_object* v_i_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16(v_xs_822_, v_v_823_, v_i_824_);
lean_dec(v_v_823_);
lean_dec_ref(v_xs_822_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9(lean_object* v_xs_826_, lean_object* v_v_827_){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_828_ = lean_unsigned_to_nat(0u);
v___x_829_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9_spec__16(v_xs_826_, v_v_827_, v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9___boxed(lean_object* v_xs_830_, lean_object* v_v_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9(v_xs_830_, v_v_831_);
lean_dec(v_v_831_);
lean_dec_ref(v_xs_830_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(lean_object* v_x_833_, size_t v_x_834_, lean_object* v_x_835_){
_start:
{
if (lean_obj_tag(v_x_833_) == 0)
{
lean_object* v_es_836_; lean_object* v___x_837_; size_t v___x_838_; size_t v___x_839_; lean_object* v_j_840_; lean_object* v_entry_841_; 
v_es_836_ = lean_ctor_get(v_x_833_, 0);
v___x_837_ = lean_box(2);
v___x_838_ = ((size_t)31ULL);
v___x_839_ = lean_usize_land(v_x_834_, v___x_838_);
v_j_840_ = lean_usize_to_nat(v___x_839_);
v_entry_841_ = lean_array_get(v___x_837_, v_es_836_, v_j_840_);
switch(lean_obj_tag(v_entry_841_))
{
case 0:
{
lean_object* v_key_842_; uint8_t v___x_843_; 
v_key_842_ = lean_ctor_get(v_entry_841_, 0);
lean_inc(v_key_842_);
lean_dec_ref_known(v_entry_841_, 2);
v___x_843_ = lean_name_eq(v_x_835_, v_key_842_);
lean_dec(v_key_842_);
if (v___x_843_ == 0)
{
lean_dec(v_j_840_);
return v_x_833_;
}
else
{
lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_851_; 
lean_inc_ref(v_es_836_);
v_isSharedCheck_851_ = !lean_is_exclusive(v_x_833_);
if (v_isSharedCheck_851_ == 0)
{
lean_object* v_unused_852_; 
v_unused_852_ = lean_ctor_get(v_x_833_, 0);
lean_dec(v_unused_852_);
v___x_845_ = v_x_833_;
v_isShared_846_ = v_isSharedCheck_851_;
goto v_resetjp_844_;
}
else
{
lean_dec(v_x_833_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_851_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_847_; lean_object* v___x_849_; 
v___x_847_ = lean_array_set(v_es_836_, v_j_840_, v___x_837_);
lean_dec(v_j_840_);
if (v_isShared_846_ == 0)
{
lean_ctor_set(v___x_845_, 0, v___x_847_);
v___x_849_ = v___x_845_;
goto v_reusejp_848_;
}
else
{
lean_object* v_reuseFailAlloc_850_; 
v_reuseFailAlloc_850_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_850_, 0, v___x_847_);
v___x_849_ = v_reuseFailAlloc_850_;
goto v_reusejp_848_;
}
v_reusejp_848_:
{
return v___x_849_;
}
}
}
}
case 1:
{
lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_887_; 
lean_inc_ref(v_es_836_);
v_isSharedCheck_887_ = !lean_is_exclusive(v_x_833_);
if (v_isSharedCheck_887_ == 0)
{
lean_object* v_unused_888_; 
v_unused_888_ = lean_ctor_get(v_x_833_, 0);
lean_dec(v_unused_888_);
v___x_854_ = v_x_833_;
v_isShared_855_ = v_isSharedCheck_887_;
goto v_resetjp_853_;
}
else
{
lean_dec(v_x_833_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_887_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v_node_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_886_; 
v_node_856_ = lean_ctor_get(v_entry_841_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v_entry_841_);
if (v_isSharedCheck_886_ == 0)
{
v___x_858_ = v_entry_841_;
v_isShared_859_ = v_isSharedCheck_886_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_node_856_);
lean_dec(v_entry_841_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_886_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
size_t v___x_860_; lean_object* v_entries_861_; size_t v___x_862_; lean_object* v_newNode_863_; lean_object* v___x_864_; 
v___x_860_ = ((size_t)5ULL);
v_entries_861_ = lean_array_set(v_es_836_, v_j_840_, v___x_837_);
v___x_862_ = lean_usize_shift_right(v_x_834_, v___x_860_);
v_newNode_863_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(v_node_856_, v___x_862_, v_x_835_);
lean_inc_ref(v_newNode_863_);
v___x_864_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_863_);
if (lean_obj_tag(v___x_864_) == 0)
{
lean_object* v___x_866_; 
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 0, v_newNode_863_);
v___x_866_ = v___x_858_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_871_; 
v_reuseFailAlloc_871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_871_, 0, v_newNode_863_);
v___x_866_ = v_reuseFailAlloc_871_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
lean_object* v___x_867_; lean_object* v___x_869_; 
v___x_867_ = lean_array_set(v_entries_861_, v_j_840_, v___x_866_);
lean_dec(v_j_840_);
if (v_isShared_855_ == 0)
{
lean_ctor_set(v___x_854_, 0, v___x_867_);
v___x_869_ = v___x_854_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v___x_867_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
else
{
lean_object* v_val_872_; lean_object* v_fst_873_; lean_object* v_snd_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_885_; 
lean_dec_ref(v_newNode_863_);
lean_del_object(v___x_858_);
v_val_872_ = lean_ctor_get(v___x_864_, 0);
lean_inc(v_val_872_);
lean_dec_ref_known(v___x_864_, 1);
v_fst_873_ = lean_ctor_get(v_val_872_, 0);
v_snd_874_ = lean_ctor_get(v_val_872_, 1);
v_isSharedCheck_885_ = !lean_is_exclusive(v_val_872_);
if (v_isSharedCheck_885_ == 0)
{
v___x_876_ = v_val_872_;
v_isShared_877_ = v_isSharedCheck_885_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_snd_874_);
lean_inc(v_fst_873_);
lean_dec(v_val_872_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_885_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v___x_879_; 
if (v_isShared_877_ == 0)
{
v___x_879_ = v___x_876_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v_fst_873_);
lean_ctor_set(v_reuseFailAlloc_884_, 1, v_snd_874_);
v___x_879_ = v_reuseFailAlloc_884_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
lean_object* v___x_880_; lean_object* v___x_882_; 
v___x_880_ = lean_array_set(v_entries_861_, v_j_840_, v___x_879_);
lean_dec(v_j_840_);
if (v_isShared_855_ == 0)
{
lean_ctor_set(v___x_854_, 0, v___x_880_);
v___x_882_ = v___x_854_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v___x_880_);
v___x_882_ = v_reuseFailAlloc_883_;
goto v_reusejp_881_;
}
v_reusejp_881_:
{
return v___x_882_;
}
}
}
}
}
}
}
default: 
{
lean_dec(v_j_840_);
return v_x_833_;
}
}
}
else
{
lean_object* v_ks_889_; lean_object* v_vs_890_; lean_object* v___x_892_; uint8_t v_isShared_893_; uint8_t v_isSharedCheck_904_; 
v_ks_889_ = lean_ctor_get(v_x_833_, 0);
v_vs_890_ = lean_ctor_get(v_x_833_, 1);
v_isSharedCheck_904_ = !lean_is_exclusive(v_x_833_);
if (v_isSharedCheck_904_ == 0)
{
v___x_892_ = v_x_833_;
v_isShared_893_ = v_isSharedCheck_904_;
goto v_resetjp_891_;
}
else
{
lean_inc(v_vs_890_);
lean_inc(v_ks_889_);
lean_dec(v_x_833_);
v___x_892_ = lean_box(0);
v_isShared_893_ = v_isSharedCheck_904_;
goto v_resetjp_891_;
}
v_resetjp_891_:
{
lean_object* v___x_894_; 
v___x_894_ = lp_mathlib_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4_spec__9(v_ks_889_, v_x_835_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_object* v___x_896_; 
if (v_isShared_893_ == 0)
{
v___x_896_ = v___x_892_;
goto v_reusejp_895_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_897_, 0, v_ks_889_);
lean_ctor_set(v_reuseFailAlloc_897_, 1, v_vs_890_);
v___x_896_ = v_reuseFailAlloc_897_;
goto v_reusejp_895_;
}
v_reusejp_895_:
{
return v___x_896_;
}
}
else
{
lean_object* v_val_898_; lean_object* v_keys_x27_899_; lean_object* v_vals_x27_900_; lean_object* v___x_902_; 
v_val_898_ = lean_ctor_get(v___x_894_, 0);
lean_inc_n(v_val_898_, 2);
lean_dec_ref_known(v___x_894_, 1);
v_keys_x27_899_ = l_Array_eraseIdx___redArg(v_ks_889_, v_val_898_);
v_vals_x27_900_ = l_Array_eraseIdx___redArg(v_vs_890_, v_val_898_);
if (v_isShared_893_ == 0)
{
lean_ctor_set(v___x_892_, 1, v_vals_x27_900_);
lean_ctor_set(v___x_892_, 0, v_keys_x27_899_);
v___x_902_ = v___x_892_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v_keys_x27_899_);
lean_ctor_set(v_reuseFailAlloc_903_, 1, v_vals_x27_900_);
v___x_902_ = v_reuseFailAlloc_903_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
return v___x_902_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg___boxed(lean_object* v_x_905_, lean_object* v_x_906_, lean_object* v_x_907_){
_start:
{
size_t v_x_3019__boxed_908_; lean_object* v_res_909_; 
v_x_3019__boxed_908_ = lean_unbox_usize(v_x_906_);
lean_dec(v_x_906_);
v_res_909_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(v_x_905_, v_x_3019__boxed_908_, v_x_907_);
lean_dec(v_x_907_);
return v_res_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg(lean_object* v_x_910_, lean_object* v_x_911_){
_start:
{
uint64_t v___y_913_; 
if (lean_obj_tag(v_x_911_) == 0)
{
uint64_t v___x_916_; 
v___x_916_ = 1723ULL;
v___y_913_ = v___x_916_;
goto v___jp_912_;
}
else
{
uint64_t v_hash_917_; 
v_hash_917_ = lean_ctor_get_uint64(v_x_911_, sizeof(void*)*2);
v___y_913_ = v_hash_917_;
goto v___jp_912_;
}
v___jp_912_:
{
size_t v_h_914_; lean_object* v___x_915_; 
v_h_914_ = lean_uint64_to_usize(v___y_913_);
v___x_915_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(v_x_910_, v_h_914_, v_x_911_);
return v___x_915_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_x_918_, lean_object* v_x_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg(v_x_918_, v_x_919_);
lean_dec(v_x_919_);
return v_res_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v_x_921_, lean_object* v_x_922_){
_start:
{
lean_object* v_fst_923_; lean_object* v_tree_924_; lean_object* v_erased_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_948_; 
v_fst_923_ = lean_ctor_get(v_x_922_, 0);
lean_inc(v_fst_923_);
v_tree_924_ = lean_ctor_get(v_x_921_, 0);
v_erased_925_ = lean_ctor_get(v_x_921_, 1);
v_isSharedCheck_948_ = !lean_is_exclusive(v_x_921_);
if (v_isSharedCheck_948_ == 0)
{
v___x_927_ = v_x_921_;
v_isShared_928_ = v_isSharedCheck_948_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_erased_925_);
lean_inc(v_tree_924_);
lean_dec(v_x_921_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_948_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v_snd_929_; lean_object* v_fst_930_; lean_object* v_snd_931_; lean_object* v___y_933_; lean_object* v___x_938_; lean_object* v___x_939_; uint8_t v___x_940_; 
v_snd_929_ = lean_ctor_get(v_x_922_, 1);
lean_inc(v_snd_929_);
lean_dec_ref(v_x_922_);
v_fst_930_ = lean_ctor_get(v_fst_923_, 0);
lean_inc(v_fst_930_);
v_snd_931_ = lean_ctor_get(v_fst_923_, 1);
lean_inc(v_snd_931_);
lean_dec(v_fst_923_);
v___x_938_ = lean_unsigned_to_nat(0u);
v___x_939_ = lean_array_get_size(v_fst_930_);
v___x_940_ = lean_nat_dec_lt(v___x_938_, v___x_939_);
if (v___x_940_ == 0)
{
lean_dec(v_fst_930_);
lean_dec(v_snd_929_);
v___y_933_ = v_tree_924_;
goto v___jp_932_;
}
else
{
uint8_t v___x_941_; 
v___x_941_ = lean_nat_dec_le(v___x_939_, v___x_939_);
if (v___x_941_ == 0)
{
if (v___x_940_ == 0)
{
lean_dec(v_fst_930_);
lean_dec(v_snd_929_);
v___y_933_ = v_tree_924_;
goto v___jp_932_;
}
else
{
size_t v___x_942_; size_t v___x_943_; lean_object* v___x_944_; 
v___x_942_ = ((size_t)0ULL);
v___x_943_ = lean_usize_of_nat(v___x_939_);
v___x_944_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2(v_snd_929_, v_fst_930_, v___x_942_, v___x_943_, v_tree_924_);
lean_dec(v_fst_930_);
v___y_933_ = v___x_944_;
goto v___jp_932_;
}
}
else
{
size_t v___x_945_; size_t v___x_946_; lean_object* v___x_947_; 
v___x_945_ = ((size_t)0ULL);
v___x_946_ = lean_usize_of_nat(v___x_939_);
v___x_947_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__2(v_snd_929_, v_fst_930_, v___x_945_, v___x_946_, v_tree_924_);
lean_dec(v_fst_930_);
v___y_933_ = v___x_947_;
goto v___jp_932_;
}
}
v___jp_932_:
{
lean_object* v___x_934_; lean_object* v___x_936_; 
v___x_934_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg(v_erased_925_, v_snd_931_);
lean_dec(v_snd_931_);
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 1, v___x_934_);
lean_ctor_set(v___x_927_, 0, v___y_933_);
v___x_936_ = v___x_927_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v___y_933_);
lean_ctor_set(v_reuseFailAlloc_937_, 1, v___x_934_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v_x_949_, lean_object* v_x_950_, lean_object* v___y_951_){
_start:
{
lean_object* v_snd_953_; lean_object* v___x_954_; 
v_snd_953_ = lean_ctor_get(v_x_950_, 1);
lean_inc(v_snd_953_);
v___x_954_ = lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt(v_snd_953_, v___y_951_);
if (lean_obj_tag(v___x_954_) == 0)
{
lean_object* v_a_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_963_; 
v_a_955_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_963_ == 0)
{
v___x_957_ = v___x_954_;
v_isShared_958_ = v_isSharedCheck_963_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_a_955_);
lean_dec(v___x_954_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_963_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v___x_959_; lean_object* v___x_961_; 
v___x_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_959_, 0, v_x_950_);
lean_ctor_set(v___x_959_, 1, v_a_955_);
if (v_isShared_958_ == 0)
{
lean_ctor_set(v___x_957_, 0, v___x_959_);
v___x_961_ = v___x_957_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v___x_959_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
else
{
lean_object* v_a_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_971_; 
lean_dec_ref(v_x_950_);
v_a_964_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_971_ == 0)
{
v___x_966_ = v___x_954_;
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_a_964_);
lean_dec(v___x_954_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v___x_969_; 
if (v_isShared_967_ == 0)
{
v___x_969_ = v___x_966_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_a_964_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v_x_972_, lean_object* v_x_973_, lean_object* v___y_974_, lean_object* v___y_975_){
_start:
{
lean_object* v_res_976_; 
v_res_976_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(v_x_972_, v_x_973_, v___y_974_);
lean_dec_ref(v___y_974_);
lean_dec_ref(v_x_972_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v_x_977_){
_start:
{
lean_object* v_fst_978_; 
v_fst_978_ = lean_ctor_get(v_x_977_, 0);
lean_inc(v_fst_978_);
return v_fst_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v_x_979_){
_start:
{
lean_object* v_res_980_; 
v_res_980_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(v_x_979_);
lean_dec_ref(v_x_979_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v_x_981_, lean_object* v_a_982_){
_start:
{
lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_983_, 0, v_a_982_);
lean_inc_ref_n(v___x_983_, 2);
v___x_984_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_984_, 0, v___x_983_);
lean_ctor_set(v___x_984_, 1, v___x_983_);
lean_ctor_set(v___x_984_, 2, v___x_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v_x_985_, lean_object* v_a_986_){
_start:
{
lean_object* v_res_987_; 
v_res_987_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(v_x_985_, v_a_986_);
lean_dec_ref(v_x_985_);
return v_res_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(lean_object* v___x_988_){
_start:
{
lean_object* v___x_990_; 
v___x_990_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_990_, 0, v___x_988_);
return v___x_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v___x_991_, lean_object* v___y_992_){
_start:
{
lean_object* v_res_993_; 
v_res_993_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(v___x_991_);
return v_res_993_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1005_; lean_object* v___f_1006_; 
v___x_1005_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__3);
v___f_1006_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__5_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed), 2, 1);
lean_closure_set(v___f_1006_, 0, v___x_1005_);
return v___f_1006_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1007_; lean_object* v___f_1008_; lean_object* v___f_1009_; lean_object* v___f_1010_; lean_object* v___f_1011_; lean_object* v___f_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___f_1007_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___f_1008_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___f_1009_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___f_1010_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___f_1011_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___f_1012_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_);
v___x_1013_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__6_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_));
v___x_1014_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_1014_, 0, v___x_1013_);
lean_ctor_set(v___x_1014_, 1, v___f_1012_);
lean_ctor_set(v___x_1014_, 2, v___f_1011_);
lean_ctor_set(v___x_1014_, 3, v___f_1010_);
lean_ctor_set(v___x_1014_, 4, v___f_1009_);
lean_ctor_set(v___x_1014_, 5, v___f_1008_);
lean_ctor_set(v___x_1014_, 6, v___f_1007_);
return v___x_1014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1016_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_);
v___x_1017_ = l_Lean_registerScopedEnvExtensionUnsafe___redArg(v___x_1016_);
return v___x_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2____boxed(lean_object* v_a_1018_){
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_();
return v_res_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b2_1020_, lean_object* v_x_1021_, lean_object* v_x_1022_){
_start:
{
lean_object* v___x_1023_; 
v___x_1023_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___redArg(v_x_1021_, v_x_1022_);
return v___x_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b2_1024_, lean_object* v_x_1025_, lean_object* v_x_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_mathlib_Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1(v_00_u03b2_1024_, v_x_1025_, v_x_1026_);
lean_dec(v_x_1026_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4(lean_object* v_00_u03b2_1028_, lean_object* v_x_1029_, size_t v_x_1030_, lean_object* v_x_1031_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___redArg(v_x_1029_, v_x_1030_, v_x_1031_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4___boxed(lean_object* v_00_u03b2_1033_, lean_object* v_x_1034_, lean_object* v_x_1035_, lean_object* v_x_1036_){
_start:
{
size_t v_x_3382__boxed_1037_; lean_object* v_res_1038_; 
v_x_3382__boxed_1037_ = lean_unbox_usize(v_x_1035_);
lean_dec(v_x_1035_);
v_res_1038_ = lp_mathlib_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__1_spec__4(v_00_u03b2_1033_, v_x_1034_, v_x_3382__boxed_1037_, v_x_1036_);
lean_dec(v_x_1036_);
return v_res_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5(lean_object* v_00_u03b2_1039_, lean_object* v_x_1040_, size_t v_x_1041_, size_t v_x_1042_, lean_object* v_x_1043_, lean_object* v_x_1044_){
_start:
{
lean_object* v___x_1045_; 
v___x_1045_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_x_1040_, v_x_1041_, v_x_1042_, v_x_1043_, v_x_1044_);
return v___x_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5___boxed(lean_object* v_00_u03b2_1046_, lean_object* v_x_1047_, lean_object* v_x_1048_, lean_object* v_x_1049_, lean_object* v_x_1050_, lean_object* v_x_1051_){
_start:
{
size_t v_x_3393__boxed_1052_; size_t v_x_3394__boxed_1053_; lean_object* v_res_1054_; 
v_x_3393__boxed_1052_ = lean_unbox_usize(v_x_1048_);
lean_dec(v_x_1048_);
v_x_3394__boxed_1053_ = lean_unbox_usize(v_x_1049_);
lean_dec(v_x_1049_);
v_res_1054_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5(v_00_u03b2_1046_, v_x_1047_, v_x_3393__boxed_1052_, v_x_3394__boxed_1053_, v_x_1050_, v_x_1051_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(lean_object* v_x_1055_, lean_object* v_keys_1056_, lean_object* v_v_1057_, lean_object* v_k_1058_, lean_object* v_as_1059_, lean_object* v_k_1060_, lean_object* v_x_1061_, lean_object* v_x_1062_, lean_object* v_x_1063_, lean_object* v_x_1064_){
_start:
{
lean_object* v___x_1065_; 
v___x_1065_ = lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_x_1055_, v_keys_1056_, v_v_1057_, v_k_1058_, v_as_1059_, v_k_1060_, v_x_1061_, v_x_1062_);
return v___x_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___boxed(lean_object* v_x_1066_, lean_object* v_keys_1067_, lean_object* v_v_1068_, lean_object* v_k_1069_, lean_object* v_as_1070_, lean_object* v_k_1071_, lean_object* v_x_1072_, lean_object* v_x_1073_, lean_object* v_x_1074_, lean_object* v_x_1075_){
_start:
{
lean_object* v_res_1076_; 
v_res_1076_ = lp_mathlib___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(v_x_1066_, v_keys_1067_, v_v_1068_, v_k_1069_, v_as_1070_, v_k_1071_, v_x_1072_, v_x_1073_, v_x_1074_, v_x_1075_);
lean_dec_ref(v_k_1071_);
lean_dec_ref(v_keys_1067_);
lean_dec(v_x_1066_);
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12(lean_object* v_00_u03b2_1077_, lean_object* v_n_1078_, lean_object* v_k_1079_, lean_object* v_v_1080_){
_start:
{
lean_object* v___x_1081_; 
v___x_1081_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12___redArg(v_n_1078_, v_k_1079_, v_v_1080_);
return v___x_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13(lean_object* v_00_u03b2_1082_, size_t v_depth_1083_, lean_object* v_keys_1084_, lean_object* v_vals_1085_, lean_object* v_heq_1086_, lean_object* v_i_1087_, lean_object* v_entries_1088_){
_start:
{
lean_object* v___x_1089_; 
v___x_1089_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___redArg(v_depth_1083_, v_keys_1084_, v_vals_1085_, v_i_1087_, v_entries_1088_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13___boxed(lean_object* v_00_u03b2_1090_, lean_object* v_depth_1091_, lean_object* v_keys_1092_, lean_object* v_vals_1093_, lean_object* v_heq_1094_, lean_object* v_i_1095_, lean_object* v_entries_1096_){
_start:
{
size_t v_depth_boxed_1097_; lean_object* v_res_1098_; 
v_depth_boxed_1097_ = lean_unbox_usize(v_depth_1091_);
lean_dec(v_depth_1091_);
v_res_1098_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__13(v_00_u03b2_1090_, v_depth_boxed_1097_, v_keys_1092_, v_vals_1093_, v_heq_1094_, v_i_1095_, v_entries_1096_);
lean_dec_ref(v_vals_1093_);
lean_dec_ref(v_keys_1092_);
return v_res_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14(lean_object* v_00_u03b2_1099_, lean_object* v_x_1100_, lean_object* v_x_1101_, lean_object* v_x_1102_, lean_object* v_x_1103_){
_start:
{
lean_object* v___x_1104_; 
v___x_1104_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__12_spec__14___redArg(v_x_1100_, v_x_1101_, v_x_1102_, v_x_1103_);
return v___x_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg(lean_object* v_category_1105_, lean_object* v_opts_1106_, lean_object* v_act_1107_, lean_object* v_decl_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
lean_object* v___x_1114_; lean_object* v___x_1115_; 
lean_inc(v___y_1112_);
lean_inc_ref(v___y_1111_);
lean_inc(v___y_1110_);
lean_inc_ref(v___y_1109_);
v___x_1114_ = lean_apply_4(v_act_1107_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
v___x_1115_ = l_Lean_profileitIOUnsafe___redArg(v_category_1105_, v_opts_1106_, v___x_1114_, v_decl_1108_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg___boxed(lean_object* v_category_1116_, lean_object* v_opts_1117_, lean_object* v_act_1118_, lean_object* v_decl_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg(v_category_1116_, v_opts_1117_, v_act_1118_, v_decl_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
lean_dec_ref(v_opts_1117_);
lean_dec_ref(v_category_1116_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4(lean_object* v_00_u03b1_1126_, lean_object* v_category_1127_, lean_object* v_opts_1128_, lean_object* v_act_1129_, lean_object* v_decl_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_){
_start:
{
lean_object* v___x_1136_; 
v___x_1136_ = lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg(v_category_1127_, v_opts_1128_, v_act_1129_, v_decl_1130_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___boxed(lean_object* v_00_u03b1_1137_, lean_object* v_category_1138_, lean_object* v_opts_1139_, lean_object* v_act_1140_, lean_object* v_decl_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v_res_1147_; 
v_res_1147_ = lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4(v_00_u03b1_1137_, v_category_1138_, v_opts_1139_, v_act_1140_, v_decl_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec_ref(v_opts_1139_);
lean_dec_ref(v_category_1138_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0(lean_object* v_msgData_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_){
_start:
{
lean_object* v___x_1154_; lean_object* v_env_1155_; lean_object* v___x_1156_; lean_object* v_mctx_1157_; lean_object* v_lctx_1158_; lean_object* v_options_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1154_ = lean_st_ref_get(v___y_1152_);
v_env_1155_ = lean_ctor_get(v___x_1154_, 0);
lean_inc_ref(v_env_1155_);
lean_dec(v___x_1154_);
v___x_1156_ = lean_st_ref_get(v___y_1150_);
v_mctx_1157_ = lean_ctor_get(v___x_1156_, 0);
lean_inc_ref(v_mctx_1157_);
lean_dec(v___x_1156_);
v_lctx_1158_ = lean_ctor_get(v___y_1149_, 2);
v_options_1159_ = lean_ctor_get(v___y_1151_, 2);
lean_inc_ref(v_options_1159_);
lean_inc_ref(v_lctx_1158_);
v___x_1160_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1160_, 0, v_env_1155_);
lean_ctor_set(v___x_1160_, 1, v_mctx_1157_);
lean_ctor_set(v___x_1160_, 2, v_lctx_1158_);
lean_ctor_set(v___x_1160_, 3, v_options_1159_);
v___x_1161_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1160_);
lean_ctor_set(v___x_1161_, 1, v_msgData_1148_);
v___x_1162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1162_, 0, v___x_1161_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0___boxed(lean_object* v_msgData_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0(v_msgData_1163_, v___y_1164_, v___y_1165_, v___y_1166_, v___y_1167_);
lean_dec(v___y_1167_);
lean_dec_ref(v___y_1166_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(lean_object* v_msg_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_){
_start:
{
lean_object* v_ref_1176_; lean_object* v___x_1177_; lean_object* v_a_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1186_; 
v_ref_1176_ = lean_ctor_get(v___y_1173_, 5);
v___x_1177_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0(v_msg_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_);
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1186_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1186_ == 0)
{
v___x_1180_ = v___x_1177_;
v_isShared_1181_ = v_isSharedCheck_1186_;
goto v_resetjp_1179_;
}
else
{
lean_inc(v_a_1178_);
lean_dec(v___x_1177_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1186_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v___x_1182_; lean_object* v___x_1184_; 
lean_inc(v_ref_1176_);
v___x_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1182_, 0, v_ref_1176_);
lean_ctor_set(v___x_1182_, 1, v_a_1178_);
if (v_isShared_1181_ == 0)
{
lean_ctor_set_tag(v___x_1180_, 1);
lean_ctor_set(v___x_1180_, 0, v___x_1182_);
v___x_1184_ = v___x_1180_;
goto v_reusejp_1183_;
}
else
{
lean_object* v_reuseFailAlloc_1185_; 
v_reuseFailAlloc_1185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1185_, 0, v___x_1182_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg___boxed(lean_object* v_msg_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v_msg_1187_, v___y_1188_, v___y_1189_, v___y_1190_, v___y_1191_);
lean_dec(v___y_1191_);
lean_dec_ref(v___y_1190_);
lean_dec(v___y_1189_);
lean_dec_ref(v___y_1188_);
return v_res_1193_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg(lean_object* v_keys_1194_, lean_object* v_i_1195_, lean_object* v_k_1196_){
_start:
{
lean_object* v___x_1197_; uint8_t v___x_1198_; 
v___x_1197_ = lean_array_get_size(v_keys_1194_);
v___x_1198_ = lean_nat_dec_lt(v_i_1195_, v___x_1197_);
if (v___x_1198_ == 0)
{
lean_dec(v_i_1195_);
return v___x_1198_;
}
else
{
lean_object* v_k_x27_1199_; uint8_t v___x_1200_; 
v_k_x27_1199_ = lean_array_fget_borrowed(v_keys_1194_, v_i_1195_);
v___x_1200_ = lean_name_eq(v_k_1196_, v_k_x27_1199_);
if (v___x_1200_ == 0)
{
lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1201_ = lean_unsigned_to_nat(1u);
v___x_1202_ = lean_nat_add(v_i_1195_, v___x_1201_);
lean_dec(v_i_1195_);
v_i_1195_ = v___x_1202_;
goto _start;
}
else
{
lean_dec(v_i_1195_);
return v___x_1200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_keys_1204_, lean_object* v_i_1205_, lean_object* v_k_1206_){
_start:
{
uint8_t v_res_1207_; lean_object* v_r_1208_; 
v_res_1207_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg(v_keys_1204_, v_i_1205_, v_k_1206_);
lean_dec(v_k_1206_);
lean_dec_ref(v_keys_1204_);
v_r_1208_ = lean_box(v_res_1207_);
return v_r_1208_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg(lean_object* v_x_1209_, size_t v_x_1210_, lean_object* v_x_1211_){
_start:
{
if (lean_obj_tag(v_x_1209_) == 0)
{
lean_object* v_es_1212_; lean_object* v___x_1213_; size_t v___x_1214_; size_t v___x_1215_; lean_object* v_j_1216_; lean_object* v___x_1217_; 
v_es_1212_ = lean_ctor_get(v_x_1209_, 0);
v___x_1213_ = lean_box(2);
v___x_1214_ = ((size_t)31ULL);
v___x_1215_ = lean_usize_land(v_x_1210_, v___x_1214_);
v_j_1216_ = lean_usize_to_nat(v___x_1215_);
v___x_1217_ = lean_array_get_borrowed(v___x_1213_, v_es_1212_, v_j_1216_);
lean_dec(v_j_1216_);
switch(lean_obj_tag(v___x_1217_))
{
case 0:
{
lean_object* v_key_1218_; uint8_t v___x_1219_; 
v_key_1218_ = lean_ctor_get(v___x_1217_, 0);
v___x_1219_ = lean_name_eq(v_x_1211_, v_key_1218_);
return v___x_1219_;
}
case 1:
{
lean_object* v_node_1220_; size_t v___x_1221_; size_t v___x_1222_; 
v_node_1220_ = lean_ctor_get(v___x_1217_, 0);
v___x_1221_ = ((size_t)5ULL);
v___x_1222_ = lean_usize_shift_right(v_x_1210_, v___x_1221_);
v_x_1209_ = v_node_1220_;
v_x_1210_ = v___x_1222_;
goto _start;
}
default: 
{
uint8_t v___x_1224_; 
v___x_1224_ = 0;
return v___x_1224_;
}
}
}
else
{
lean_object* v_ks_1225_; lean_object* v___x_1226_; uint8_t v___x_1227_; 
v_ks_1225_ = lean_ctor_get(v_x_1209_, 0);
v___x_1226_ = lean_unsigned_to_nat(0u);
v___x_1227_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg(v_ks_1225_, v___x_1226_, v_x_1211_);
return v___x_1227_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg___boxed(lean_object* v_x_1228_, lean_object* v_x_1229_, lean_object* v_x_1230_){
_start:
{
size_t v_x_15351__boxed_1231_; uint8_t v_res_1232_; lean_object* v_r_1233_; 
v_x_15351__boxed_1231_ = lean_unbox_usize(v_x_1229_);
lean_dec(v_x_1229_);
v_res_1232_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg(v_x_1228_, v_x_15351__boxed_1231_, v_x_1230_);
lean_dec(v_x_1230_);
lean_dec_ref(v_x_1228_);
v_r_1233_ = lean_box(v_res_1232_);
return v_r_1233_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(lean_object* v_x_1234_, lean_object* v_x_1235_){
_start:
{
uint64_t v___y_1237_; 
if (lean_obj_tag(v_x_1235_) == 0)
{
uint64_t v___x_1240_; 
v___x_1240_ = 1723ULL;
v___y_1237_ = v___x_1240_;
goto v___jp_1236_;
}
else
{
uint64_t v_hash_1241_; 
v_hash_1241_ = lean_ctor_get_uint64(v_x_1235_, sizeof(void*)*2);
v___y_1237_ = v_hash_1241_;
goto v___jp_1236_;
}
v___jp_1236_:
{
size_t v___x_1238_; uint8_t v___x_1239_; 
v___x_1238_ = lean_uint64_to_usize(v___y_1237_);
v___x_1239_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg(v_x_1234_, v___x_1238_, v_x_1235_);
return v___x_1239_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg___boxed(lean_object* v_x_1242_, lean_object* v_x_1243_){
_start:
{
uint8_t v_res_1244_; lean_object* v_r_1245_; 
v_res_1244_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(v_x_1242_, v_x_1243_);
lean_dec(v_x_1243_);
lean_dec_ref(v_x_1242_);
v_r_1245_ = lean_box(v_res_1244_);
return v_r_1245_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1246_; double v___x_1247_; 
v___x_1246_ = lean_unsigned_to_nat(0u);
v___x_1247_ = lean_float_of_nat(v___x_1246_);
return v___x_1247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0(lean_object* v_cls_1251_, lean_object* v_msg_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_){
_start:
{
lean_object* v_ref_1258_; lean_object* v___x_1259_; lean_object* v_a_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1304_; 
v_ref_1258_ = lean_ctor_get(v___y_1255_, 5);
v___x_1259_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0_spec__0(v_msg_1252_, v___y_1253_, v___y_1254_, v___y_1255_, v___y_1256_);
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1259_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1262_ = v___x_1259_;
v_isShared_1263_ = v_isSharedCheck_1304_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_a_1260_);
lean_dec(v___x_1259_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1304_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v___x_1264_; lean_object* v_traceState_1265_; lean_object* v_env_1266_; lean_object* v_nextMacroScope_1267_; lean_object* v_ngen_1268_; lean_object* v_auxDeclNGen_1269_; lean_object* v_cache_1270_; lean_object* v_messages_1271_; lean_object* v_infoState_1272_; lean_object* v_snapshotTasks_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1303_; 
v___x_1264_ = lean_st_ref_take(v___y_1256_);
v_traceState_1265_ = lean_ctor_get(v___x_1264_, 4);
v_env_1266_ = lean_ctor_get(v___x_1264_, 0);
v_nextMacroScope_1267_ = lean_ctor_get(v___x_1264_, 1);
v_ngen_1268_ = lean_ctor_get(v___x_1264_, 2);
v_auxDeclNGen_1269_ = lean_ctor_get(v___x_1264_, 3);
v_cache_1270_ = lean_ctor_get(v___x_1264_, 5);
v_messages_1271_ = lean_ctor_get(v___x_1264_, 6);
v_infoState_1272_ = lean_ctor_get(v___x_1264_, 7);
v_snapshotTasks_1273_ = lean_ctor_get(v___x_1264_, 8);
v_isSharedCheck_1303_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1275_ = v___x_1264_;
v_isShared_1276_ = v_isSharedCheck_1303_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_snapshotTasks_1273_);
lean_inc(v_infoState_1272_);
lean_inc(v_messages_1271_);
lean_inc(v_cache_1270_);
lean_inc(v_traceState_1265_);
lean_inc(v_auxDeclNGen_1269_);
lean_inc(v_ngen_1268_);
lean_inc(v_nextMacroScope_1267_);
lean_inc(v_env_1266_);
lean_dec(v___x_1264_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1303_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
uint64_t v_tid_1277_; lean_object* v_traces_1278_; lean_object* v___x_1280_; uint8_t v_isShared_1281_; uint8_t v_isSharedCheck_1302_; 
v_tid_1277_ = lean_ctor_get_uint64(v_traceState_1265_, sizeof(void*)*1);
v_traces_1278_ = lean_ctor_get(v_traceState_1265_, 0);
v_isSharedCheck_1302_ = !lean_is_exclusive(v_traceState_1265_);
if (v_isSharedCheck_1302_ == 0)
{
v___x_1280_ = v_traceState_1265_;
v_isShared_1281_ = v_isSharedCheck_1302_;
goto v_resetjp_1279_;
}
else
{
lean_inc(v_traces_1278_);
lean_dec(v_traceState_1265_);
v___x_1280_ = lean_box(0);
v_isShared_1281_ = v_isSharedCheck_1302_;
goto v_resetjp_1279_;
}
v_resetjp_1279_:
{
lean_object* v___x_1282_; double v___x_1283_; uint8_t v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1292_; 
v___x_1282_ = lean_box(0);
v___x_1283_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0);
v___x_1284_ = 0;
v___x_1285_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__1));
v___x_1286_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1286_, 0, v_cls_1251_);
lean_ctor_set(v___x_1286_, 1, v___x_1282_);
lean_ctor_set(v___x_1286_, 2, v___x_1285_);
lean_ctor_set_float(v___x_1286_, sizeof(void*)*3, v___x_1283_);
lean_ctor_set_float(v___x_1286_, sizeof(void*)*3 + 8, v___x_1283_);
lean_ctor_set_uint8(v___x_1286_, sizeof(void*)*3 + 16, v___x_1284_);
v___x_1287_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__2));
v___x_1288_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1286_);
lean_ctor_set(v___x_1288_, 1, v_a_1260_);
lean_ctor_set(v___x_1288_, 2, v___x_1287_);
lean_inc(v_ref_1258_);
v___x_1289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1289_, 0, v_ref_1258_);
lean_ctor_set(v___x_1289_, 1, v___x_1288_);
v___x_1290_ = l_Lean_PersistentArray_push___redArg(v_traces_1278_, v___x_1289_);
if (v_isShared_1281_ == 0)
{
lean_ctor_set(v___x_1280_, 0, v___x_1290_);
v___x_1292_ = v___x_1280_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v___x_1290_);
lean_ctor_set_uint64(v_reuseFailAlloc_1301_, sizeof(void*)*1, v_tid_1277_);
v___x_1292_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
lean_object* v___x_1294_; 
if (v_isShared_1276_ == 0)
{
lean_ctor_set(v___x_1275_, 4, v___x_1292_);
v___x_1294_ = v___x_1275_;
goto v_reusejp_1293_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_env_1266_);
lean_ctor_set(v_reuseFailAlloc_1300_, 1, v_nextMacroScope_1267_);
lean_ctor_set(v_reuseFailAlloc_1300_, 2, v_ngen_1268_);
lean_ctor_set(v_reuseFailAlloc_1300_, 3, v_auxDeclNGen_1269_);
lean_ctor_set(v_reuseFailAlloc_1300_, 4, v___x_1292_);
lean_ctor_set(v_reuseFailAlloc_1300_, 5, v_cache_1270_);
lean_ctor_set(v_reuseFailAlloc_1300_, 6, v_messages_1271_);
lean_ctor_set(v_reuseFailAlloc_1300_, 7, v_infoState_1272_);
lean_ctor_set(v_reuseFailAlloc_1300_, 8, v_snapshotTasks_1273_);
v___x_1294_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1293_;
}
v_reusejp_1293_:
{
lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1298_; 
v___x_1295_ = lean_st_ref_set(v___y_1256_, v___x_1294_);
v___x_1296_ = lean_box(0);
if (v_isShared_1263_ == 0)
{
lean_ctor_set(v___x_1262_, 0, v___x_1296_);
v___x_1298_ = v___x_1262_;
goto v_reusejp_1297_;
}
else
{
lean_object* v_reuseFailAlloc_1299_; 
v_reuseFailAlloc_1299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1299_, 0, v___x_1296_);
v___x_1298_ = v_reuseFailAlloc_1299_;
goto v_reusejp_1297_;
}
v_reusejp_1297_:
{
return v___x_1298_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___boxed(lean_object* v_cls_1305_, lean_object* v_msg_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0(v_cls_1305_, v_msg_1306_, v___y_1307_, v___y_1308_, v___y_1309_, v___y_1310_);
lean_dec(v___y_1310_);
lean_dec_ref(v___y_1309_);
lean_dec(v___y_1308_);
lean_dec_ref(v___y_1307_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1(lean_object* v_a_1313_, lean_object* v_____r_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_){
_start:
{
lean_object* v___x_1320_; lean_object* v___x_1321_; 
v___x_1320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1320_, 0, v_a_1313_);
v___x_1321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
return v___x_1321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1___boxed(lean_object* v_a_1322_, lean_object* v_____r_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_){
_start:
{
lean_object* v_res_1329_; 
v_res_1329_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1(v_a_1322_, v_____r_1323_, v___y_1324_, v___y_1325_, v___y_1326_, v___y_1327_);
lean_dec(v___y_1327_);
lean_dec_ref(v___y_1326_);
lean_dec(v___y_1325_);
lean_dec_ref(v___y_1324_);
return v_res_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0(lean_object* v_a_1330_, lean_object* v_____r_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_){
_start:
{
lean_object* v___x_1337_; 
v___x_1337_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1330_, v___y_1333_, v___y_1335_);
if (lean_obj_tag(v___x_1337_) == 0)
{
lean_object* v_a_1338_; lean_object* v___x_1340_; uint8_t v_isShared_1341_; uint8_t v_isSharedCheck_1346_; 
v_a_1338_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1346_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1346_ == 0)
{
v___x_1340_ = v___x_1337_;
v_isShared_1341_ = v_isSharedCheck_1346_;
goto v_resetjp_1339_;
}
else
{
lean_inc(v_a_1338_);
lean_dec(v___x_1337_);
v___x_1340_ = lean_box(0);
v_isShared_1341_ = v_isSharedCheck_1346_;
goto v_resetjp_1339_;
}
v_resetjp_1339_:
{
lean_object* v___x_1342_; lean_object* v___x_1344_; 
v___x_1342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1342_, 0, v_a_1338_);
if (v_isShared_1341_ == 0)
{
lean_ctor_set(v___x_1340_, 0, v___x_1342_);
v___x_1344_ = v___x_1340_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v___x_1342_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
return v___x_1344_;
}
}
}
else
{
lean_object* v_a_1347_; lean_object* v___x_1349_; uint8_t v_isShared_1350_; uint8_t v_isSharedCheck_1354_; 
v_a_1347_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1354_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1354_ == 0)
{
v___x_1349_ = v___x_1337_;
v_isShared_1350_ = v_isSharedCheck_1354_;
goto v_resetjp_1348_;
}
else
{
lean_inc(v_a_1347_);
lean_dec(v___x_1337_);
v___x_1349_ = lean_box(0);
v_isShared_1350_ = v_isSharedCheck_1354_;
goto v_resetjp_1348_;
}
v_resetjp_1348_:
{
lean_object* v___x_1352_; 
if (v_isShared_1350_ == 0)
{
v___x_1352_ = v___x_1349_;
goto v_reusejp_1351_;
}
else
{
lean_object* v_reuseFailAlloc_1353_; 
v_reuseFailAlloc_1353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1353_, 0, v_a_1347_);
v___x_1352_ = v_reuseFailAlloc_1353_;
goto v_reusejp_1351_;
}
v_reusejp_1351_:
{
return v___x_1352_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0___boxed(lean_object* v_a_1355_, lean_object* v_____r_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_){
_start:
{
lean_object* v_res_1362_; 
v_res_1362_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0(v_a_1355_, v_____r_1356_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
lean_dec(v___y_1358_);
lean_dec_ref(v___y_1357_);
lean_dec_ref(v_a_1355_);
return v_res_1362_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1369_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_1370_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__2));
v___x_1371_ = l_Lean_Name_append(v___x_1370_, v___x_1369_);
return v___x_1371_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5(void){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1373_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__4));
v___x_1374_ = l_Lean_stringToMessageData(v___x_1373_);
return v___x_1374_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7(void){
_start:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; 
v___x_1376_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__6));
v___x_1377_ = l_Lean_stringToMessageData(v___x_1376_);
return v___x_1377_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9(void){
_start:
{
lean_object* v___x_1379_; lean_object* v___x_1380_; 
v___x_1379_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__8));
v___x_1380_ = l_Lean_stringToMessageData(v___x_1379_);
return v___x_1380_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11(void){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; 
v___x_1382_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__10));
v___x_1383_ = l_Lean_stringToMessageData(v___x_1382_);
return v___x_1383_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13(void){
_start:
{
lean_object* v___x_1385_; lean_object* v___x_1386_; 
v___x_1385_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__12));
v___x_1386_ = l_Lean_stringToMessageData(v___x_1385_);
return v___x_1386_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15(void){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__14));
v___x_1389_ = l_Lean_stringToMessageData(v___x_1388_);
return v___x_1389_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17(void){
_start:
{
lean_object* v___x_1391_; lean_object* v___x_1392_; 
v___x_1391_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__16));
v___x_1392_ = l_Lean_stringToMessageData(v___x_1391_);
return v___x_1392_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19(void){
_start:
{
lean_object* v___x_1394_; lean_object* v___x_1395_; 
v___x_1394_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__18));
v___x_1395_ = l_Lean_stringToMessageData(v___x_1394_);
return v___x_1395_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21(void){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__20));
v___x_1398_ = l_Lean_stringToMessageData(v___x_1397_);
return v___x_1398_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23(void){
_start:
{
lean_object* v___x_1400_; lean_object* v___x_1401_; 
v___x_1400_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__22));
v___x_1401_ = l_Lean_stringToMessageData(v___x_1400_);
return v___x_1401_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25(void){
_start:
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__24));
v___x_1404_ = l_Lean_stringToMessageData(v___x_1403_);
return v___x_1404_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28(void){
_start:
{
lean_object* v___x_1407_; lean_object* v___x_1408_; 
v___x_1407_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__27));
v___x_1408_ = l_Lean_stringToMessageData(v___x_1407_);
return v___x_1408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2(lean_object* v_a_1409_, lean_object* v_e_1410_, lean_object* v_u_1411_, lean_object* v_00_u03b1_1412_, lean_object* v___x_1413_, uint8_t v___x_1414_, uint8_t v_post_1415_, lean_object* v_as_1416_, size_t v_sz_1417_, size_t v_i_1418_, lean_object* v_b_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_){
_start:
{
lean_object* v_a_1426_; uint8_t v___x_1430_; 
v___x_1430_ = lean_usize_dec_lt(v_i_1418_, v_sz_1417_);
if (v___x_1430_ == 0)
{
lean_object* v___x_1431_; 
lean_dec_ref(v_00_u03b1_1412_);
lean_dec(v_u_1411_);
lean_dec_ref(v_e_1410_);
v___x_1431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1431_, 0, v_b_1419_);
return v___x_1431_;
}
else
{
lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v_a_1435_; lean_object* v___y_1447_; lean_object* v_a_1459_; lean_object* v___y_1461_; uint8_t v___y_1462_; lean_object* v_a_1492_; lean_object* v___y_1496_; lean_object* v___y_1500_; lean_object* v___y_1501_; lean_object* v___y_1502_; lean_object* v___y_1503_; lean_object* v___y_1510_; lean_object* v___y_1511_; lean_object* v___y_1512_; lean_object* v___y_1513_; lean_object* v___y_1514_; lean_object* v___y_1515_; lean_object* v___y_1526_; lean_object* v___y_1527_; lean_object* v___y_1528_; lean_object* v___y_1529_; lean_object* v___y_1530_; lean_object* v___y_1531_; lean_object* v___y_1542_; uint8_t v___y_1640_; 
lean_dec_ref(v_b_1419_);
v___x_1432_ = lean_box(0);
v___x_1433_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__0));
v_a_1459_ = lean_array_uget_borrowed(v_as_1416_, v_i_1418_);
if (v_post_1415_ == 0)
{
uint8_t v_pre_1644_; 
v_pre_1644_ = lean_ctor_get_uint8(v_a_1459_, sizeof(void*)*2);
v___y_1640_ = v_pre_1644_;
goto v___jp_1639_;
}
else
{
uint8_t v_post_1645_; 
v_post_1645_ = lean_ctor_get_uint8(v_a_1459_, sizeof(void*)*2 + 1);
v___y_1640_ = v_post_1645_;
goto v___jp_1639_;
}
v___jp_1434_:
{
if (lean_obj_tag(v_a_1435_) == 0)
{
lean_object* v_a_1436_; lean_object* v___x_1438_; uint8_t v_isShared_1439_; uint8_t v_isSharedCheck_1445_; 
lean_dec_ref(v_00_u03b1_1412_);
lean_dec(v_u_1411_);
lean_dec_ref(v_e_1410_);
v_a_1436_ = lean_ctor_get(v_a_1435_, 0);
v_isSharedCheck_1445_ = !lean_is_exclusive(v_a_1435_);
if (v_isSharedCheck_1445_ == 0)
{
v___x_1438_ = v_a_1435_;
v_isShared_1439_ = v_isSharedCheck_1445_;
goto v_resetjp_1437_;
}
else
{
lean_inc(v_a_1436_);
lean_dec(v_a_1435_);
v___x_1438_ = lean_box(0);
v_isShared_1439_ = v_isSharedCheck_1445_;
goto v_resetjp_1437_;
}
v_resetjp_1437_:
{
lean_object* v___x_1441_; 
if (v_isShared_1439_ == 0)
{
lean_ctor_set_tag(v___x_1438_, 1);
v___x_1441_ = v___x_1438_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1444_; 
v_reuseFailAlloc_1444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1444_, 0, v_a_1436_);
v___x_1441_ = v_reuseFailAlloc_1444_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
lean_object* v___x_1442_; lean_object* v___x_1443_; 
v___x_1442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1442_, 0, v___x_1441_);
lean_ctor_set(v___x_1442_, 1, v___x_1432_);
v___x_1443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1443_, 0, v___x_1442_);
return v___x_1443_;
}
}
}
else
{
lean_dec_ref_known(v_a_1435_, 1);
v_a_1426_ = v___x_1433_;
goto v___jp_1425_;
}
}
v___jp_1446_:
{
if (lean_obj_tag(v___y_1447_) == 0)
{
lean_object* v_a_1448_; 
v_a_1448_ = lean_ctor_get(v___y_1447_, 0);
lean_inc(v_a_1448_);
lean_dec_ref_known(v___y_1447_, 1);
v_a_1435_ = v_a_1448_;
goto v___jp_1434_;
}
else
{
lean_object* v_a_1449_; lean_object* v___x_1451_; uint8_t v_isShared_1452_; uint8_t v_isSharedCheck_1456_; 
lean_dec_ref(v_00_u03b1_1412_);
lean_dec(v_u_1411_);
lean_dec_ref(v_e_1410_);
v_a_1449_ = lean_ctor_get(v___y_1447_, 0);
v_isSharedCheck_1456_ = !lean_is_exclusive(v___y_1447_);
if (v_isSharedCheck_1456_ == 0)
{
v___x_1451_ = v___y_1447_;
v_isShared_1452_ = v_isSharedCheck_1456_;
goto v_resetjp_1450_;
}
else
{
lean_inc(v_a_1449_);
lean_dec(v___y_1447_);
v___x_1451_ = lean_box(0);
v_isShared_1452_ = v_isSharedCheck_1456_;
goto v_resetjp_1450_;
}
v_resetjp_1450_:
{
lean_object* v___x_1454_; 
if (v_isShared_1452_ == 0)
{
v___x_1454_ = v___x_1451_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1455_; 
v_reuseFailAlloc_1455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1455_, 0, v_a_1449_);
v___x_1454_ = v_reuseFailAlloc_1455_;
goto v_reusejp_1453_;
}
v_reusejp_1453_:
{
return v___x_1454_;
}
}
}
}
v___jp_1457_:
{
lean_object* v___x_1458_; 
v___x_1458_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0(v_a_1409_, v___x_1432_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
v___y_1447_ = v___x_1458_;
goto v___jp_1446_;
}
v___jp_1460_:
{
if (v___y_1462_ == 0)
{
lean_object* v_options_1463_; uint8_t v_hasTrace_1464_; 
v_options_1463_ = lean_ctor_get(v___y_1422_, 2);
v_hasTrace_1464_ = lean_ctor_get_uint8(v_options_1463_, sizeof(void*)*1);
if (v_hasTrace_1464_ == 0)
{
lean_dec_ref(v___y_1461_);
goto v___jp_1457_;
}
else
{
lean_object* v_inheritedTraceOptions_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; uint8_t v___x_1468_; 
v_inheritedTraceOptions_1465_ = lean_ctor_get(v___y_1422_, 13);
v___x_1466_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_1467_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3);
v___x_1468_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1465_, v_options_1463_, v___x_1467_);
if (v___x_1468_ == 0)
{
lean_dec_ref(v___y_1461_);
goto v___jp_1457_;
}
else
{
lean_object* v_name_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; 
v_name_1469_ = lean_ctor_get(v_a_1459_, 1);
lean_inc(v_name_1469_);
v___x_1470_ = l_Lean_MessageData_ofName(v_name_1469_);
v___x_1471_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__5);
v___x_1472_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1472_, 0, v___x_1470_);
lean_ctor_set(v___x_1472_, 1, v___x_1471_);
lean_inc_ref(v_e_1410_);
v___x_1473_ = l_Lean_MessageData_ofExpr(v_e_1410_);
v___x_1474_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1474_, 0, v___x_1472_);
lean_ctor_set(v___x_1474_, 1, v___x_1473_);
v___x_1475_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__7);
v___x_1476_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1476_, 0, v___x_1474_);
lean_ctor_set(v___x_1476_, 1, v___x_1475_);
v___x_1477_ = l_Lean_Exception_toMessageData(v___y_1461_);
v___x_1478_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1476_);
lean_ctor_set(v___x_1478_, 1, v___x_1477_);
v___x_1479_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0(v___x_1466_, v___x_1478_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
if (lean_obj_tag(v___x_1479_) == 0)
{
lean_object* v_a_1480_; lean_object* v___x_1481_; 
v_a_1480_ = lean_ctor_get(v___x_1479_, 0);
lean_inc(v_a_1480_);
lean_dec_ref_known(v___x_1479_, 1);
v___x_1481_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__0(v_a_1409_, v_a_1480_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
v___y_1447_ = v___x_1481_;
goto v___jp_1446_;
}
else
{
lean_object* v_a_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1489_; 
lean_dec_ref(v_00_u03b1_1412_);
lean_dec(v_u_1411_);
lean_dec_ref(v_e_1410_);
v_a_1482_ = lean_ctor_get(v___x_1479_, 0);
v_isSharedCheck_1489_ = !lean_is_exclusive(v___x_1479_);
if (v_isSharedCheck_1489_ == 0)
{
v___x_1484_ = v___x_1479_;
v_isShared_1485_ = v_isSharedCheck_1489_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_a_1482_);
lean_dec(v___x_1479_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1489_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v___x_1487_; 
if (v_isShared_1485_ == 0)
{
v___x_1487_ = v___x_1484_;
goto v_reusejp_1486_;
}
else
{
lean_object* v_reuseFailAlloc_1488_; 
v_reuseFailAlloc_1488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1488_, 0, v_a_1482_);
v___x_1487_ = v_reuseFailAlloc_1488_;
goto v_reusejp_1486_;
}
v_reusejp_1486_:
{
return v___x_1487_;
}
}
}
}
}
}
else
{
lean_object* v___x_1490_; 
lean_dec_ref(v_00_u03b1_1412_);
lean_dec(v_u_1411_);
lean_dec_ref(v_e_1410_);
v___x_1490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1490_, 0, v___y_1461_);
return v___x_1490_;
}
}
v___jp_1491_:
{
uint8_t v___x_1493_; 
v___x_1493_ = l_Lean_Exception_isInterrupt(v_a_1492_);
if (v___x_1493_ == 0)
{
uint8_t v___x_1494_; 
lean_inc_ref(v_a_1492_);
v___x_1494_ = l_Lean_Exception_isRuntime(v_a_1492_);
v___y_1461_ = v_a_1492_;
v___y_1462_ = v___x_1494_;
goto v___jp_1460_;
}
else
{
v___y_1461_ = v_a_1492_;
v___y_1462_ = v___x_1493_;
goto v___jp_1460_;
}
}
v___jp_1495_:
{
if (lean_obj_tag(v___y_1496_) == 0)
{
lean_object* v_a_1497_; 
v_a_1497_ = lean_ctor_get(v___y_1496_, 0);
lean_inc(v_a_1497_);
lean_dec_ref_known(v___y_1496_, 1);
v_a_1435_ = v_a_1497_;
goto v___jp_1434_;
}
else
{
lean_object* v_a_1498_; 
v_a_1498_ = lean_ctor_get(v___y_1496_, 0);
lean_inc(v_a_1498_);
lean_dec_ref_known(v___y_1496_, 1);
v_a_1492_ = v_a_1498_;
goto v___jp_1491_;
}
}
v___jp_1499_:
{
lean_object* v___x_1504_; lean_object* v___x_1505_; 
v___x_1504_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1504_, 0, v___y_1501_);
lean_ctor_set(v___x_1504_, 1, v___y_1503_);
lean_inc(v___y_1500_);
v___x_1505_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0(v___y_1500_, v___x_1504_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
if (lean_obj_tag(v___x_1505_) == 0)
{
lean_object* v_a_1506_; lean_object* v___x_1507_; 
v_a_1506_ = lean_ctor_get(v___x_1505_, 0);
lean_inc(v_a_1506_);
lean_dec_ref_known(v___x_1505_, 1);
lean_inc(v___y_1423_);
lean_inc_ref(v___y_1422_);
lean_inc(v___y_1421_);
lean_inc_ref(v___y_1420_);
v___x_1507_ = lean_apply_6(v___y_1502_, v_a_1506_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_, lean_box(0));
v___y_1496_ = v___x_1507_;
goto v___jp_1495_;
}
else
{
lean_object* v_a_1508_; 
lean_dec_ref(v___y_1502_);
v_a_1508_ = lean_ctor_get(v___x_1505_, 0);
lean_inc(v_a_1508_);
lean_dec_ref_known(v___x_1505_, 1);
v_a_1492_ = v_a_1508_;
goto v___jp_1491_;
}
}
v___jp_1509_:
{
lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1516_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1516_, 0, v___y_1515_);
v___x_1517_ = l_Lean_MessageData_ofFormat(v___x_1516_);
lean_inc_ref(v___y_1513_);
v___x_1518_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1518_, 0, v___y_1513_);
lean_ctor_set(v___x_1518_, 1, v___x_1517_);
v___x_1519_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9);
v___x_1520_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1520_, 0, v___x_1518_);
lean_ctor_set(v___x_1520_, 1, v___x_1519_);
v___x_1521_ = l_Lean_MessageData_ofExpr(v___y_1514_);
v___x_1522_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1522_, 0, v___x_1520_);
lean_ctor_set(v___x_1522_, 1, v___x_1521_);
v___x_1523_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1524_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1524_, 0, v___x_1522_);
lean_ctor_set(v___x_1524_, 1, v___x_1523_);
v___y_1500_ = v___y_1510_;
v___y_1501_ = v___y_1511_;
v___y_1502_ = v___y_1512_;
v___y_1503_ = v___x_1524_;
goto v___jp_1499_;
}
v___jp_1525_:
{
lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; 
v___x_1532_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1532_, 0, v___y_1531_);
v___x_1533_ = l_Lean_MessageData_ofFormat(v___x_1532_);
lean_inc_ref(v___y_1530_);
v___x_1534_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1534_, 0, v___y_1530_);
lean_ctor_set(v___x_1534_, 1, v___x_1533_);
v___x_1535_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9);
v___x_1536_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1536_, 0, v___x_1534_);
lean_ctor_set(v___x_1536_, 1, v___x_1535_);
v___x_1537_ = l_Lean_MessageData_ofExpr(v___y_1528_);
v___x_1538_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1538_, 0, v___x_1536_);
lean_ctor_set(v___x_1538_, 1, v___x_1537_);
v___x_1539_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1540_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1540_, 0, v___x_1538_);
lean_ctor_set(v___x_1540_, 1, v___x_1539_);
v___y_1500_ = v___y_1526_;
v___y_1501_ = v___y_1527_;
v___y_1502_ = v___y_1529_;
v___y_1503_ = v___x_1540_;
goto v___jp_1499_;
}
v___jp_1541_:
{
lean_object* v___x_1543_; 
lean_inc(v___y_1423_);
lean_inc_ref(v___y_1422_);
lean_inc(v___y_1421_);
lean_inc_ref(v___y_1420_);
v___x_1543_ = lean_apply_6(v___y_1542_, v___x_1432_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_, lean_box(0));
v___y_1496_ = v___x_1543_;
goto v___jp_1495_;
}
v___jp_1544_:
{
lean_object* v_eval_1545_; lean_object* v_name_1546_; lean_object* v_keyedConfig_1547_; uint8_t v_trackZetaDelta_1548_; lean_object* v_zetaDeltaSet_1549_; lean_object* v_lctx_1550_; lean_object* v_localInstances_1551_; lean_object* v_defEqCtx_x3f_1552_; lean_object* v_synthPendingDepth_1553_; lean_object* v_customCanUnfoldPredicate_x3f_1554_; uint8_t v_univApprox_1555_; uint8_t v_inTypeClassResolution_1556_; uint8_t v_cacheInferType_1557_; uint8_t v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; 
v_eval_1545_ = lean_ctor_get(v_a_1459_, 0);
v_name_1546_ = lean_ctor_get(v_a_1459_, 1);
v_keyedConfig_1547_ = lean_ctor_get(v___y_1420_, 0);
v_trackZetaDelta_1548_ = lean_ctor_get_uint8(v___y_1420_, sizeof(void*)*7);
v_zetaDeltaSet_1549_ = lean_ctor_get(v___y_1420_, 1);
v_lctx_1550_ = lean_ctor_get(v___y_1420_, 2);
v_localInstances_1551_ = lean_ctor_get(v___y_1420_, 3);
v_defEqCtx_x3f_1552_ = lean_ctor_get(v___y_1420_, 4);
v_synthPendingDepth_1553_ = lean_ctor_get(v___y_1420_, 5);
v_customCanUnfoldPredicate_x3f_1554_ = lean_ctor_get(v___y_1420_, 6);
v_univApprox_1555_ = lean_ctor_get_uint8(v___y_1420_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1556_ = lean_ctor_get_uint8(v___y_1420_, sizeof(void*)*7 + 2);
v_cacheInferType_1557_ = lean_ctor_get_uint8(v___y_1420_, sizeof(void*)*7 + 3);
v___x_1558_ = 3;
lean_inc_ref(v_keyedConfig_1547_);
v___x_1559_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1558_, v_keyedConfig_1547_);
lean_inc(v_customCanUnfoldPredicate_x3f_1554_);
lean_inc(v_synthPendingDepth_1553_);
lean_inc(v_defEqCtx_x3f_1552_);
lean_inc_ref(v_localInstances_1551_);
lean_inc_ref(v_lctx_1550_);
lean_inc(v_zetaDeltaSet_1549_);
v___x_1560_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1560_, 0, v___x_1559_);
lean_ctor_set(v___x_1560_, 1, v_zetaDeltaSet_1549_);
lean_ctor_set(v___x_1560_, 2, v_lctx_1550_);
lean_ctor_set(v___x_1560_, 3, v_localInstances_1551_);
lean_ctor_set(v___x_1560_, 4, v_defEqCtx_x3f_1552_);
lean_ctor_set(v___x_1560_, 5, v_synthPendingDepth_1553_);
lean_ctor_set(v___x_1560_, 6, v_customCanUnfoldPredicate_x3f_1554_);
lean_ctor_set_uint8(v___x_1560_, sizeof(void*)*7, v_trackZetaDelta_1548_);
lean_ctor_set_uint8(v___x_1560_, sizeof(void*)*7 + 1, v_univApprox_1555_);
lean_ctor_set_uint8(v___x_1560_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1556_);
lean_ctor_set_uint8(v___x_1560_, sizeof(void*)*7 + 3, v_cacheInferType_1557_);
lean_inc_ref(v_eval_1545_);
lean_inc(v___y_1423_);
lean_inc_ref(v___y_1422_);
lean_inc(v___y_1421_);
lean_inc_ref(v_e_1410_);
lean_inc_ref(v_00_u03b1_1412_);
lean_inc(v_u_1411_);
v___x_1561_ = lean_apply_8(v_eval_1545_, v_u_1411_, v_00_u03b1_1412_, v_e_1410_, v___x_1560_, v___y_1421_, v___y_1422_, v___y_1423_, lean_box(0));
if (lean_obj_tag(v___x_1561_) == 0)
{
lean_object* v_options_1562_; lean_object* v_a_1563_; lean_object* v_inheritedTraceOptions_1564_; uint8_t v_hasTrace_1565_; lean_object* v___f_1566_; 
v_options_1562_ = lean_ctor_get(v___y_1422_, 2);
v_a_1563_ = lean_ctor_get(v___x_1561_, 0);
lean_inc_n(v_a_1563_, 2);
lean_dec_ref_known(v___x_1561_, 1);
v_inheritedTraceOptions_1564_ = lean_ctor_get(v___y_1422_, 13);
v_hasTrace_1565_ = lean_ctor_get_uint8(v_options_1562_, sizeof(void*)*1);
v___f_1566_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___lam__1___boxed), 7, 1);
lean_closure_set(v___f_1566_, 0, v_a_1563_);
if (v_hasTrace_1565_ == 0)
{
lean_dec(v_a_1563_);
v___y_1542_ = v___f_1566_;
goto v___jp_1541_;
}
else
{
lean_object* v___x_1567_; lean_object* v___x_1568_; uint8_t v___x_1569_; 
v___x_1567_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_));
v___x_1568_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__3);
v___x_1569_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1564_, v_options_1562_, v___x_1568_);
if (v___x_1569_ == 0)
{
lean_dec(v_a_1563_);
v___y_1542_ = v___f_1566_;
goto v___jp_1541_;
}
else
{
lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; 
lean_inc(v_name_1546_);
v___x_1570_ = l_Lean_MessageData_ofName(v_name_1546_);
v___x_1571_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__13);
v___x_1572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1572_, 0, v___x_1570_);
lean_ctor_set(v___x_1572_, 1, v___x_1571_);
lean_inc_ref(v_e_1410_);
v___x_1573_ = l_Lean_MessageData_ofExpr(v_e_1410_);
v___x_1574_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1574_, 0, v___x_1572_);
lean_ctor_set(v___x_1574_, 1, v___x_1573_);
v___x_1575_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__15);
v___x_1576_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1574_);
lean_ctor_set(v___x_1576_, 1, v___x_1575_);
switch(lean_obj_tag(v_a_1563_))
{
case 0:
{
uint8_t v_val_1577_; 
v_val_1577_ = lean_ctor_get_uint8(v_a_1563_, sizeof(void*)*1);
if (v_val_1577_ == 0)
{
lean_object* v_proof_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; 
v_proof_1578_ = lean_ctor_get(v_a_1563_, 0);
lean_inc_ref(v_proof_1578_);
lean_dec_ref_known(v_a_1563_, 1);
v___x_1579_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__17);
v___x_1580_ = l_Lean_MessageData_ofExpr(v_proof_1578_);
v___x_1581_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1581_, 0, v___x_1579_);
lean_ctor_set(v___x_1581_, 1, v___x_1580_);
v___x_1582_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1583_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1581_);
lean_ctor_set(v___x_1583_, 1, v___x_1582_);
v___y_1500_ = v___x_1567_;
v___y_1501_ = v___x_1576_;
v___y_1502_ = v___f_1566_;
v___y_1503_ = v___x_1583_;
goto v___jp_1499_;
}
else
{
lean_object* v_proof_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; 
v_proof_1584_ = lean_ctor_get(v_a_1563_, 0);
lean_inc_ref(v_proof_1584_);
lean_dec_ref_known(v_a_1563_, 1);
v___x_1585_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__19);
v___x_1586_ = l_Lean_MessageData_ofExpr(v_proof_1584_);
v___x_1587_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1587_, 0, v___x_1585_);
lean_ctor_set(v___x_1587_, 1, v___x_1586_);
v___x_1588_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1589_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1589_, 0, v___x_1587_);
lean_ctor_set(v___x_1589_, 1, v___x_1588_);
v___y_1500_ = v___x_1567_;
v___y_1501_ = v___x_1576_;
v___y_1502_ = v___f_1566_;
v___y_1503_ = v___x_1589_;
goto v___jp_1499_;
}
}
case 1:
{
lean_object* v_lit_1590_; lean_object* v_proof_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; 
v_lit_1590_ = lean_ctor_get(v_a_1563_, 1);
lean_inc_ref(v_lit_1590_);
v_proof_1591_ = lean_ctor_get(v_a_1563_, 2);
lean_inc_ref(v_proof_1591_);
lean_dec_ref_known(v_a_1563_, 3);
v___x_1592_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__21);
v___x_1593_ = l_Lean_MessageData_ofExpr(v_lit_1590_);
v___x_1594_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1592_);
lean_ctor_set(v___x_1594_, 1, v___x_1593_);
v___x_1595_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9);
v___x_1596_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1596_, 0, v___x_1594_);
lean_ctor_set(v___x_1596_, 1, v___x_1595_);
v___x_1597_ = l_Lean_MessageData_ofExpr(v_proof_1591_);
v___x_1598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1598_, 0, v___x_1596_);
lean_ctor_set(v___x_1598_, 1, v___x_1597_);
v___x_1599_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1600_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1600_, 0, v___x_1598_);
lean_ctor_set(v___x_1600_, 1, v___x_1599_);
v___y_1500_ = v___x_1567_;
v___y_1501_ = v___x_1576_;
v___y_1502_ = v___f_1566_;
v___y_1503_ = v___x_1600_;
goto v___jp_1499_;
}
case 2:
{
lean_object* v_lit_1601_; lean_object* v_proof_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; 
v_lit_1601_ = lean_ctor_get(v_a_1563_, 1);
lean_inc_ref(v_lit_1601_);
v_proof_1602_ = lean_ctor_get(v_a_1563_, 2);
lean_inc_ref(v_proof_1602_);
lean_dec_ref_known(v_a_1563_, 3);
v___x_1603_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__23);
v___x_1604_ = l_Lean_MessageData_ofExpr(v_lit_1601_);
v___x_1605_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1605_, 0, v___x_1603_);
lean_ctor_set(v___x_1605_, 1, v___x_1604_);
v___x_1606_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__9);
v___x_1607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1607_, 0, v___x_1605_);
lean_ctor_set(v___x_1607_, 1, v___x_1606_);
v___x_1608_ = l_Lean_MessageData_ofExpr(v_proof_1602_);
v___x_1609_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1609_, 0, v___x_1607_);
lean_ctor_set(v___x_1609_, 1, v___x_1608_);
v___x_1610_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__11);
v___x_1611_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1611_, 0, v___x_1609_);
lean_ctor_set(v___x_1611_, 1, v___x_1610_);
v___y_1500_ = v___x_1567_;
v___y_1501_ = v___x_1576_;
v___y_1502_ = v___f_1566_;
v___y_1503_ = v___x_1611_;
goto v___jp_1499_;
}
case 3:
{
lean_object* v_q_1612_; lean_object* v_proof_1613_; lean_object* v_num_1614_; lean_object* v_den_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; uint8_t v___x_1618_; 
v_q_1612_ = lean_ctor_get(v_a_1563_, 1);
lean_inc_ref(v_q_1612_);
v_proof_1613_ = lean_ctor_get(v_a_1563_, 4);
lean_inc_ref(v_proof_1613_);
lean_dec_ref_known(v_a_1563_, 5);
v_num_1614_ = lean_ctor_get(v_q_1612_, 0);
lean_inc(v_num_1614_);
v_den_1615_ = lean_ctor_get(v_q_1612_, 1);
lean_inc(v_den_1615_);
lean_dec_ref(v_q_1612_);
v___x_1616_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__25);
v___x_1617_ = lean_unsigned_to_nat(1u);
v___x_1618_ = lean_nat_dec_eq(v_den_1615_, v___x_1617_);
if (v___x_1618_ == 0)
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1619_ = l_Int_repr(v_num_1614_);
lean_dec(v_num_1614_);
v___x_1620_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__26));
v___x_1621_ = lean_string_append(v___x_1619_, v___x_1620_);
v___x_1622_ = l_Nat_reprFast(v_den_1615_);
v___x_1623_ = lean_string_append(v___x_1621_, v___x_1622_);
lean_dec_ref(v___x_1622_);
v___y_1510_ = v___x_1567_;
v___y_1511_ = v___x_1576_;
v___y_1512_ = v___f_1566_;
v___y_1513_ = v___x_1616_;
v___y_1514_ = v_proof_1613_;
v___y_1515_ = v___x_1623_;
goto v___jp_1509_;
}
else
{
lean_object* v___x_1624_; 
lean_dec(v_den_1615_);
v___x_1624_ = l_Int_repr(v_num_1614_);
lean_dec(v_num_1614_);
v___y_1510_ = v___x_1567_;
v___y_1511_ = v___x_1576_;
v___y_1512_ = v___f_1566_;
v___y_1513_ = v___x_1616_;
v___y_1514_ = v_proof_1613_;
v___y_1515_ = v___x_1624_;
goto v___jp_1509_;
}
}
default: 
{
lean_object* v_q_1625_; lean_object* v_proof_1626_; lean_object* v_num_1627_; lean_object* v_den_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; uint8_t v___x_1631_; 
v_q_1625_ = lean_ctor_get(v_a_1563_, 1);
lean_inc_ref(v_q_1625_);
v_proof_1626_ = lean_ctor_get(v_a_1563_, 4);
lean_inc_ref(v_proof_1626_);
lean_dec_ref_known(v_a_1563_, 5);
v_num_1627_ = lean_ctor_get(v_q_1625_, 0);
lean_inc(v_num_1627_);
v_den_1628_ = lean_ctor_get(v_q_1625_, 1);
lean_inc(v_den_1628_);
lean_dec_ref(v_q_1625_);
v___x_1629_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__28);
v___x_1630_ = lean_unsigned_to_nat(1u);
v___x_1631_ = lean_nat_dec_eq(v_den_1628_, v___x_1630_);
if (v___x_1631_ == 0)
{
lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1632_ = l_Int_repr(v_num_1627_);
lean_dec(v_num_1627_);
v___x_1633_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__26));
v___x_1634_ = lean_string_append(v___x_1632_, v___x_1633_);
v___x_1635_ = l_Nat_reprFast(v_den_1628_);
v___x_1636_ = lean_string_append(v___x_1634_, v___x_1635_);
lean_dec_ref(v___x_1635_);
v___y_1526_ = v___x_1567_;
v___y_1527_ = v___x_1576_;
v___y_1528_ = v_proof_1626_;
v___y_1529_ = v___f_1566_;
v___y_1530_ = v___x_1629_;
v___y_1531_ = v___x_1636_;
goto v___jp_1525_;
}
else
{
lean_object* v___x_1637_; 
lean_dec(v_den_1628_);
v___x_1637_ = l_Int_repr(v_num_1627_);
lean_dec(v_num_1627_);
v___y_1526_ = v___x_1567_;
v___y_1527_ = v___x_1576_;
v___y_1528_ = v_proof_1626_;
v___y_1529_ = v___f_1566_;
v___y_1530_ = v___x_1629_;
v___y_1531_ = v___x_1637_;
goto v___jp_1525_;
}
}
}
}
}
}
else
{
lean_object* v_a_1638_; 
v_a_1638_ = lean_ctor_get(v___x_1561_, 0);
lean_inc(v_a_1638_);
lean_dec_ref_known(v___x_1561_, 1);
v_a_1492_ = v_a_1638_;
goto v___jp_1491_;
}
}
v___jp_1639_:
{
if (v___y_1640_ == 0)
{
v_a_1426_ = v___x_1433_;
goto v___jp_1425_;
}
else
{
lean_object* v_erased_1641_; lean_object* v_name_1642_; uint8_t v___x_1643_; 
v_erased_1641_ = lean_ctor_get(v___x_1413_, 1);
v_name_1642_ = lean_ctor_get(v_a_1459_, 1);
v___x_1643_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(v_erased_1641_, v_name_1642_);
if (v___x_1643_ == 0)
{
goto v___jp_1544_;
}
else
{
if (v___x_1414_ == 0)
{
v_a_1426_ = v___x_1433_;
goto v___jp_1425_;
}
else
{
goto v___jp_1544_;
}
}
}
}
}
v___jp_1425_:
{
size_t v___x_1427_; size_t v___x_1428_; 
v___x_1427_ = ((size_t)1ULL);
v___x_1428_ = lean_usize_add(v_i_1418_, v___x_1427_);
lean_inc_ref(v_a_1426_);
v_i_1418_ = v___x_1428_;
v_b_1419_ = v_a_1426_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___boxed(lean_object* v_a_1646_, lean_object* v_e_1647_, lean_object* v_u_1648_, lean_object* v_00_u03b1_1649_, lean_object* v___x_1650_, lean_object* v___x_1651_, lean_object* v_post_1652_, lean_object* v_as_1653_, lean_object* v_sz_1654_, lean_object* v_i_1655_, lean_object* v_b_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_){
_start:
{
uint8_t v___x_15724__boxed_1662_; uint8_t v_post_boxed_1663_; size_t v_sz_boxed_1664_; size_t v_i_boxed_1665_; lean_object* v_res_1666_; 
v___x_15724__boxed_1662_ = lean_unbox(v___x_1651_);
v_post_boxed_1663_ = lean_unbox(v_post_1652_);
v_sz_boxed_1664_ = lean_unbox_usize(v_sz_1654_);
lean_dec(v_sz_1654_);
v_i_boxed_1665_ = lean_unbox_usize(v_i_1655_);
lean_dec(v_i_1655_);
v_res_1666_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2(v_a_1646_, v_e_1647_, v_u_1648_, v_00_u03b1_1649_, v___x_1650_, v___x_15724__boxed_1662_, v_post_boxed_1663_, v_as_1653_, v_sz_boxed_1664_, v_i_boxed_1665_, v_b_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_);
lean_dec(v___y_1660_);
lean_dec_ref(v___y_1659_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec_ref(v_as_1653_);
lean_dec_ref(v___x_1650_);
lean_dec_ref(v_a_1646_);
return v_res_1666_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1668_; lean_object* v___x_1669_; 
v___x_1668_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__0));
v___x_1669_ = l_Lean_stringToMessageData(v___x_1668_);
return v___x_1669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0(lean_object* v_e_1670_, lean_object* v_u_1671_, lean_object* v_00_u03b1_1672_, uint8_t v___x_1673_, uint8_t v_post_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_){
_start:
{
lean_object* v___x_1680_; 
v___x_1680_ = l_Lean_Meta_saveState___redArg(v___y_1676_, v___y_1678_);
if (lean_obj_tag(v___x_1680_) == 0)
{
lean_object* v_a_1681_; lean_object* v___x_1682_; lean_object* v_env_1683_; lean_object* v___x_1684_; lean_object* v_ext_1685_; lean_object* v_toEnvExtension_1686_; lean_object* v_asyncMode_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v_tree_1690_; lean_object* v___x_1691_; 
v_a_1681_ = lean_ctor_get(v___x_1680_, 0);
lean_inc(v_a_1681_);
lean_dec_ref_known(v___x_1680_, 1);
v___x_1682_ = lean_st_ref_get(v___y_1678_);
v_env_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc_ref(v_env_1683_);
lean_dec(v___x_1682_);
v___x_1684_ = lp_mathlib_Mathlib_Meta_NormNum_normNumExt;
v_ext_1685_ = lean_ctor_get(v___x_1684_, 1);
v_toEnvExtension_1686_ = lean_ctor_get(v_ext_1685_, 0);
v_asyncMode_1687_ = lean_ctor_get(v_toEnvExtension_1686_, 2);
v___x_1688_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default;
v___x_1689_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1688_, v___x_1684_, v_env_1683_, v_asyncMode_1687_);
v_tree_1690_ = lean_ctor_get(v___x_1689_, 0);
lean_inc_ref(v_tree_1690_);
lean_inc_ref(v_e_1670_);
v___x_1691_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v_tree_1690_, v_e_1670_, v___y_1675_, v___y_1676_, v___y_1677_, v___y_1678_);
lean_dec_ref(v_tree_1690_);
if (lean_obj_tag(v___x_1691_) == 0)
{
lean_object* v_a_1692_; lean_object* v___x_1693_; size_t v_sz_1694_; size_t v___x_1695_; lean_object* v___x_1696_; 
v_a_1692_ = lean_ctor_get(v___x_1691_, 0);
lean_inc(v_a_1692_);
lean_dec_ref_known(v___x_1691_, 1);
v___x_1693_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__0));
v_sz_1694_ = lean_array_size(v_a_1692_);
v___x_1695_ = ((size_t)0ULL);
lean_inc_ref(v_e_1670_);
v___x_1696_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2(v_a_1681_, v_e_1670_, v_u_1671_, v_00_u03b1_1672_, v___x_1689_, v___x_1673_, v_post_1674_, v_a_1692_, v_sz_1694_, v___x_1695_, v___x_1693_, v___y_1675_, v___y_1676_, v___y_1677_, v___y_1678_);
lean_dec(v_a_1692_);
lean_dec(v___x_1689_);
lean_dec(v_a_1681_);
if (lean_obj_tag(v___x_1696_) == 0)
{
lean_object* v_a_1697_; lean_object* v___x_1699_; uint8_t v_isShared_1700_; uint8_t v_isSharedCheck_1717_; 
v_a_1697_ = lean_ctor_get(v___x_1696_, 0);
v_isSharedCheck_1717_ = !lean_is_exclusive(v___x_1696_);
if (v_isSharedCheck_1717_ == 0)
{
v___x_1699_ = v___x_1696_;
v_isShared_1700_ = v_isSharedCheck_1717_;
goto v_resetjp_1698_;
}
else
{
lean_inc(v_a_1697_);
lean_dec(v___x_1696_);
v___x_1699_ = lean_box(0);
v_isShared_1700_ = v_isSharedCheck_1717_;
goto v_resetjp_1698_;
}
v_resetjp_1698_:
{
lean_object* v_fst_1701_; lean_object* v___x_1703_; uint8_t v_isShared_1704_; uint8_t v_isSharedCheck_1715_; 
v_fst_1701_ = lean_ctor_get(v_a_1697_, 0);
v_isSharedCheck_1715_ = !lean_is_exclusive(v_a_1697_);
if (v_isSharedCheck_1715_ == 0)
{
lean_object* v_unused_1716_; 
v_unused_1716_ = lean_ctor_get(v_a_1697_, 1);
lean_dec(v_unused_1716_);
v___x_1703_ = v_a_1697_;
v_isShared_1704_ = v_isSharedCheck_1715_;
goto v_resetjp_1702_;
}
else
{
lean_inc(v_fst_1701_);
lean_dec(v_a_1697_);
v___x_1703_ = lean_box(0);
v_isShared_1704_ = v_isSharedCheck_1715_;
goto v_resetjp_1702_;
}
v_resetjp_1702_:
{
if (lean_obj_tag(v_fst_1701_) == 0)
{
lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1708_; 
lean_del_object(v___x_1699_);
v___x_1705_ = l_Lean_MessageData_ofExpr(v_e_1670_);
v___x_1706_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___closed__1);
if (v_isShared_1704_ == 0)
{
lean_ctor_set_tag(v___x_1703_, 7);
lean_ctor_set(v___x_1703_, 1, v___x_1706_);
lean_ctor_set(v___x_1703_, 0, v___x_1705_);
v___x_1708_ = v___x_1703_;
goto v_reusejp_1707_;
}
else
{
lean_object* v_reuseFailAlloc_1710_; 
v_reuseFailAlloc_1710_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1710_, 0, v___x_1705_);
lean_ctor_set(v_reuseFailAlloc_1710_, 1, v___x_1706_);
v___x_1708_ = v_reuseFailAlloc_1710_;
goto v_reusejp_1707_;
}
v_reusejp_1707_:
{
lean_object* v___x_1709_; 
v___x_1709_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v___x_1708_, v___y_1675_, v___y_1676_, v___y_1677_, v___y_1678_);
return v___x_1709_;
}
}
else
{
lean_object* v_val_1711_; lean_object* v___x_1713_; 
lean_del_object(v___x_1703_);
lean_dec_ref(v_e_1670_);
v_val_1711_ = lean_ctor_get(v_fst_1701_, 0);
lean_inc(v_val_1711_);
lean_dec_ref_known(v_fst_1701_, 1);
if (v_isShared_1700_ == 0)
{
lean_ctor_set(v___x_1699_, 0, v_val_1711_);
v___x_1713_ = v___x_1699_;
goto v_reusejp_1712_;
}
else
{
lean_object* v_reuseFailAlloc_1714_; 
v_reuseFailAlloc_1714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1714_, 0, v_val_1711_);
v___x_1713_ = v_reuseFailAlloc_1714_;
goto v_reusejp_1712_;
}
v_reusejp_1712_:
{
return v___x_1713_;
}
}
}
}
}
else
{
lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1725_; 
lean_dec_ref(v_e_1670_);
v_a_1718_ = lean_ctor_get(v___x_1696_, 0);
v_isSharedCheck_1725_ = !lean_is_exclusive(v___x_1696_);
if (v_isSharedCheck_1725_ == 0)
{
v___x_1720_ = v___x_1696_;
v_isShared_1721_ = v_isSharedCheck_1725_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1696_);
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
else
{
lean_object* v_a_1726_; lean_object* v___x_1728_; uint8_t v_isShared_1729_; uint8_t v_isSharedCheck_1733_; 
lean_dec(v___x_1689_);
lean_dec(v_a_1681_);
lean_dec_ref(v_00_u03b1_1672_);
lean_dec(v_u_1671_);
lean_dec_ref(v_e_1670_);
v_a_1726_ = lean_ctor_get(v___x_1691_, 0);
v_isSharedCheck_1733_ = !lean_is_exclusive(v___x_1691_);
if (v_isSharedCheck_1733_ == 0)
{
v___x_1728_ = v___x_1691_;
v_isShared_1729_ = v_isSharedCheck_1733_;
goto v_resetjp_1727_;
}
else
{
lean_inc(v_a_1726_);
lean_dec(v___x_1691_);
v___x_1728_ = lean_box(0);
v_isShared_1729_ = v_isSharedCheck_1733_;
goto v_resetjp_1727_;
}
v_resetjp_1727_:
{
lean_object* v___x_1731_; 
if (v_isShared_1729_ == 0)
{
v___x_1731_ = v___x_1728_;
goto v_reusejp_1730_;
}
else
{
lean_object* v_reuseFailAlloc_1732_; 
v_reuseFailAlloc_1732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1732_, 0, v_a_1726_);
v___x_1731_ = v_reuseFailAlloc_1732_;
goto v_reusejp_1730_;
}
v_reusejp_1730_:
{
return v___x_1731_;
}
}
}
}
else
{
lean_object* v_a_1734_; lean_object* v___x_1736_; uint8_t v_isShared_1737_; uint8_t v_isSharedCheck_1741_; 
lean_dec_ref(v_00_u03b1_1672_);
lean_dec(v_u_1671_);
lean_dec_ref(v_e_1670_);
v_a_1734_ = lean_ctor_get(v___x_1680_, 0);
v_isSharedCheck_1741_ = !lean_is_exclusive(v___x_1680_);
if (v_isSharedCheck_1741_ == 0)
{
v___x_1736_ = v___x_1680_;
v_isShared_1737_ = v_isSharedCheck_1741_;
goto v_resetjp_1735_;
}
else
{
lean_inc(v_a_1734_);
lean_dec(v___x_1680_);
v___x_1736_ = lean_box(0);
v_isShared_1737_ = v_isSharedCheck_1741_;
goto v_resetjp_1735_;
}
v_resetjp_1735_:
{
lean_object* v___x_1739_; 
if (v_isShared_1737_ == 0)
{
v___x_1739_ = v___x_1736_;
goto v_reusejp_1738_;
}
else
{
lean_object* v_reuseFailAlloc_1740_; 
v_reuseFailAlloc_1740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1740_, 0, v_a_1734_);
v___x_1739_ = v_reuseFailAlloc_1740_;
goto v_reusejp_1738_;
}
v_reusejp_1738_:
{
return v___x_1739_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___boxed(lean_object* v_e_1742_, lean_object* v_u_1743_, lean_object* v_00_u03b1_1744_, lean_object* v___x_1745_, lean_object* v_post_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_){
_start:
{
uint8_t v___x_16188__boxed_1752_; uint8_t v_post_boxed_1753_; lean_object* v_res_1754_; 
v___x_16188__boxed_1752_ = lean_unbox(v___x_1745_);
v_post_boxed_1753_ = lean_unbox(v_post_1746_);
v_res_1754_ = lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0(v_e_1742_, v_u_1743_, v_00_u03b1_1744_, v___x_16188__boxed_1752_, v_post_boxed_1753_, v___y_1747_, v___y_1748_, v___y_1749_, v___y_1750_);
lean_dec(v___y_1750_);
lean_dec_ref(v___y_1749_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
return v_res_1754_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3(void){
_start:
{
lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; 
v___x_1760_ = lean_box(0);
v___x_1761_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_derive___closed__2));
v___x_1762_ = l_Lean_Expr_const___override(v___x_1761_, v___x_1760_);
return v___x_1762_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7(void){
_start:
{
lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; 
v___x_1771_ = lean_box(0);
v___x_1772_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_derive___closed__6));
v___x_1773_ = l_Lean_Expr_const___override(v___x_1772_, v___x_1771_);
return v___x_1773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object* v_u_1774_, lean_object* v_00_u03b1_1775_, lean_object* v_e_1776_, uint8_t v_post_1777_, lean_object* v_a_1778_, lean_object* v_a_1779_, lean_object* v_a_1780_, lean_object* v_a_1781_){
_start:
{
uint8_t v___x_1783_; 
v___x_1783_ = l_Lean_Expr_isRawNatLit(v_e_1776_);
if (v___x_1783_ == 0)
{
lean_object* v_options_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___f_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; 
v_options_1784_ = lean_ctor_get(v_a_1780_, 2);
v___x_1785_ = lean_box(v___x_1783_);
v___x_1786_ = lean_box(v_post_1777_);
v___f_1787_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_derive___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1787_, 0, v_e_1776_);
lean_closure_set(v___f_1787_, 1, v_u_1774_);
lean_closure_set(v___f_1787_, 2, v_00_u03b1_1775_);
lean_closure_set(v___f_1787_, 3, v___x_1785_);
lean_closure_set(v___f_1787_, 4, v___x_1786_);
v___x_1788_ = ((lean_object*)(lp_mathlib_norm__num___closed__0));
v___x_1789_ = lean_box(0);
v___x_1790_ = lp_mathlib_Lean_profileitM___at___00Mathlib_Meta_NormNum_derive_spec__4___redArg(v___x_1788_, v_options_1784_, v___f_1787_, v___x_1789_, v_a_1778_, v_a_1779_, v_a_1780_, v_a_1781_);
return v___x_1790_;
}
else
{
lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; 
lean_dec_ref(v_00_u03b1_1775_);
lean_dec(v_u_1774_);
v___x_1791_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_derive___closed__3);
v___x_1792_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_derive___closed__7);
lean_inc_ref(v_e_1776_);
v___x_1793_ = l_Lean_Expr_app___override(v___x_1792_, v_e_1776_);
v___x_1794_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1794_, 0, v___x_1791_);
lean_ctor_set(v___x_1794_, 1, v_e_1776_);
lean_ctor_set(v___x_1794_, 2, v___x_1793_);
v___x_1795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1795_, 0, v___x_1794_);
return v___x_1795_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive___boxed(lean_object* v_u_1796_, lean_object* v_00_u03b1_1797_, lean_object* v_e_1798_, lean_object* v_post_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_, lean_object* v_a_1802_, lean_object* v_a_1803_, lean_object* v_a_1804_){
_start:
{
uint8_t v_post_boxed_1805_; lean_object* v_res_1806_; 
v_post_boxed_1805_ = lean_unbox(v_post_1799_);
v_res_1806_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1796_, v_00_u03b1_1797_, v_e_1798_, v_post_boxed_1805_, v_a_1800_, v_a_1801_, v_a_1802_, v_a_1803_);
lean_dec(v_a_1803_);
lean_dec_ref(v_a_1802_);
lean_dec(v_a_1801_);
lean_dec_ref(v_a_1800_);
return v_res_1806_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1(lean_object* v_00_u03b2_1807_, lean_object* v_x_1808_, lean_object* v_x_1809_){
_start:
{
uint8_t v___x_1810_; 
v___x_1810_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(v_x_1808_, v_x_1809_);
return v___x_1810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___boxed(lean_object* v_00_u03b2_1811_, lean_object* v_x_1812_, lean_object* v_x_1813_){
_start:
{
uint8_t v_res_1814_; lean_object* v_r_1815_; 
v_res_1814_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1(v_00_u03b2_1811_, v_x_1812_, v_x_1813_);
lean_dec(v_x_1813_);
lean_dec_ref(v_x_1812_);
v_r_1815_ = lean_box(v_res_1814_);
return v_r_1815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3(lean_object* v_00_u03b1_1816_, lean_object* v_msg_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v___x_1823_; 
v___x_1823_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v_msg_1817_, v___y_1818_, v___y_1819_, v___y_1820_, v___y_1821_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___boxed(lean_object* v_00_u03b1_1824_, lean_object* v_msg_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_){
_start:
{
lean_object* v_res_1831_; 
v_res_1831_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3(v_00_u03b1_1824_, v_msg_1825_, v___y_1826_, v___y_1827_, v___y_1828_, v___y_1829_);
lean_dec(v___y_1829_);
lean_dec_ref(v___y_1828_);
lean_dec(v___y_1827_);
lean_dec_ref(v___y_1826_);
return v_res_1831_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2(lean_object* v_00_u03b2_1832_, lean_object* v_x_1833_, size_t v_x_1834_, lean_object* v_x_1835_){
_start:
{
uint8_t v___x_1836_; 
v___x_1836_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___redArg(v_x_1833_, v_x_1834_, v_x_1835_);
return v___x_1836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1837_, lean_object* v_x_1838_, lean_object* v_x_1839_, lean_object* v_x_1840_){
_start:
{
size_t v_x_16433__boxed_1841_; uint8_t v_res_1842_; lean_object* v_r_1843_; 
v_x_16433__boxed_1841_ = lean_unbox_usize(v_x_1839_);
lean_dec(v_x_1839_);
v_res_1842_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2(v_00_u03b2_1837_, v_x_1838_, v_x_16433__boxed_1841_, v_x_1840_);
lean_dec(v_x_1840_);
lean_dec_ref(v_x_1838_);
v_r_1843_ = lean_box(v_res_1842_);
return v_r_1843_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1844_, lean_object* v_keys_1845_, lean_object* v_vals_1846_, lean_object* v_heq_1847_, lean_object* v_i_1848_, lean_object* v_k_1849_){
_start:
{
uint8_t v___x_1850_; 
v___x_1850_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___redArg(v_keys_1845_, v_i_1848_, v_k_1849_);
return v___x_1850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1851_, lean_object* v_keys_1852_, lean_object* v_vals_1853_, lean_object* v_heq_1854_, lean_object* v_i_1855_, lean_object* v_k_1856_){
_start:
{
uint8_t v_res_1857_; lean_object* v_r_1858_; 
v_res_1857_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1_spec__2_spec__4(v_00_u03b2_1851_, v_keys_1852_, v_vals_1853_, v_heq_1854_, v_i_1855_, v_k_1856_);
lean_dec(v_k_1856_);
lean_dec_ref(v_vals_1853_);
lean_dec_ref(v_keys_1852_);
v_r_1858_ = lean_box(v_res_1857_);
return v_r_1858_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3(void){
_start:
{
lean_object* v___x_1866_; lean_object* v___x_1867_; 
v___x_1866_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__2));
v___x_1867_ = l_Lean_mkAtom(v___x_1866_);
return v___x_1867_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4(void){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; 
v___x_1868_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__3);
v___x_1869_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1870_ = lean_array_push(v___x_1869_, v___x_1868_);
return v___x_1870_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7(void){
_start:
{
lean_object* v___x_1877_; lean_object* v___x_1878_; 
v___x_1877_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__5));
v___x_1878_ = l_Lean_mkAtom(v___x_1877_);
return v___x_1878_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8(void){
_start:
{
lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1879_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__7);
v___x_1880_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1881_ = lean_array_push(v___x_1880_, v___x_1879_);
return v___x_1881_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9(void){
_start:
{
lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; 
v___x_1882_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__8);
v___x_1883_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__6));
v___x_1884_ = lean_box(2);
v___x_1885_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1885_, 0, v___x_1884_);
lean_ctor_set(v___x_1885_, 1, v___x_1883_);
lean_ctor_set(v___x_1885_, 2, v___x_1882_);
return v___x_1885_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10(void){
_start:
{
lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; 
v___x_1886_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__9);
v___x_1887_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1888_ = lean_array_push(v___x_1887_, v___x_1886_);
return v___x_1888_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11(void){
_start:
{
lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; 
v___x_1889_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__10);
v___x_1890_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8));
v___x_1891_ = lean_box(2);
v___x_1892_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1892_, 0, v___x_1891_);
lean_ctor_set(v___x_1892_, 1, v___x_1890_);
lean_ctor_set(v___x_1892_, 2, v___x_1889_);
return v___x_1892_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12(void){
_start:
{
lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; 
v___x_1893_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__11);
v___x_1894_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1895_ = lean_array_push(v___x_1894_, v___x_1893_);
return v___x_1895_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13(void){
_start:
{
lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; 
v___x_1896_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__12);
v___x_1897_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6));
v___x_1898_ = lean_box(2);
v___x_1899_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1898_);
lean_ctor_set(v___x_1899_, 1, v___x_1897_);
lean_ctor_set(v___x_1899_, 2, v___x_1896_);
return v___x_1899_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14(void){
_start:
{
lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; 
v___x_1900_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__13);
v___x_1901_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1902_ = lean_array_push(v___x_1901_, v___x_1900_);
return v___x_1902_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15(void){
_start:
{
lean_object* v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; 
v___x_1903_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__14);
v___x_1904_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3));
v___x_1905_ = lean_box(2);
v___x_1906_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1906_, 0, v___x_1905_);
lean_ctor_set(v___x_1906_, 1, v___x_1904_);
lean_ctor_set(v___x_1906_, 2, v___x_1903_);
return v___x_1906_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16(void){
_start:
{
lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___x_1907_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__15);
v___x_1908_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__4);
v___x_1909_ = lean_array_push(v___x_1908_, v___x_1907_);
return v___x_1909_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17(void){
_start:
{
lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; 
v___x_1910_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__16);
v___x_1911_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__1));
v___x_1912_ = lean_box(2);
v___x_1913_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1913_, 0, v___x_1912_);
lean_ctor_set(v___x_1913_, 1, v___x_1911_);
lean_ctor_set(v___x_1913_, 2, v___x_1910_);
return v___x_1913_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18(void){
_start:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; 
v___x_1914_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__17);
v___x_1915_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1916_ = lean_array_push(v___x_1915_, v___x_1914_);
return v___x_1916_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19(void){
_start:
{
lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; 
v___x_1917_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__18);
v___x_1918_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8));
v___x_1919_ = lean_box(2);
v___x_1920_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1920_, 0, v___x_1919_);
lean_ctor_set(v___x_1920_, 1, v___x_1918_);
lean_ctor_set(v___x_1920_, 2, v___x_1917_);
return v___x_1920_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20(void){
_start:
{
lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; 
v___x_1921_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__19);
v___x_1922_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1923_ = lean_array_push(v___x_1922_, v___x_1921_);
return v___x_1923_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21(void){
_start:
{
lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; 
v___x_1924_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__20);
v___x_1925_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__6));
v___x_1926_ = lean_box(2);
v___x_1927_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1927_, 0, v___x_1926_);
lean_ctor_set(v___x_1927_, 1, v___x_1925_);
lean_ctor_set(v___x_1927_, 2, v___x_1924_);
return v___x_1927_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22(void){
_start:
{
lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; 
v___x_1928_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__21);
v___x_1929_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_1930_ = lean_array_push(v___x_1929_, v___x_1928_);
return v___x_1930_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23(void){
_start:
{
lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; 
v___x_1931_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__22);
v___x_1932_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__3));
v___x_1933_ = lean_box(2);
v___x_1934_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1934_, 0, v___x_1933_);
lean_ctor_set(v___x_1934_, 1, v___x_1932_);
lean_ctor_set(v___x_1934_, 2, v___x_1931_);
return v___x_1934_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1(void){
_start:
{
lean_object* v___x_1935_; 
v___x_1935_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23);
return v___x_1935_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1(void){
_start:
{
lean_object* v___x_1937_; lean_object* v___x_1938_; 
v___x_1937_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__0));
v___x_1938_ = l_Lean_stringToMessageData(v___x_1937_);
return v___x_1938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object* v_u_1939_, lean_object* v_00_u03b1_1940_, lean_object* v_e_1941_, lean_object* v_a_1942_, lean_object* v_a_1943_, lean_object* v_a_1944_, lean_object* v_a_1945_){
_start:
{
uint8_t v___x_1947_; lean_object* v___x_1948_; 
v___x_1947_ = 0;
v___x_1948_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1939_, v_00_u03b1_1940_, v_e_1941_, v___x_1947_, v_a_1942_, v_a_1943_, v_a_1944_, v_a_1945_);
if (lean_obj_tag(v___x_1948_) == 0)
{
lean_object* v_a_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1961_; 
v_a_1949_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1961_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1961_ == 0)
{
v___x_1951_ = v___x_1948_;
v_isShared_1952_ = v_isSharedCheck_1961_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_a_1949_);
lean_dec(v___x_1948_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1961_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
if (lean_obj_tag(v_a_1949_) == 1)
{
lean_object* v_lit_1953_; lean_object* v_proof_1954_; lean_object* v___x_1955_; lean_object* v___x_1957_; 
v_lit_1953_ = lean_ctor_get(v_a_1949_, 1);
lean_inc_ref(v_lit_1953_);
v_proof_1954_ = lean_ctor_get(v_a_1949_, 2);
lean_inc_ref(v_proof_1954_);
lean_dec_ref_known(v_a_1949_, 3);
v___x_1955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1955_, 0, v_lit_1953_);
lean_ctor_set(v___x_1955_, 1, v_proof_1954_);
if (v_isShared_1952_ == 0)
{
lean_ctor_set(v___x_1951_, 0, v___x_1955_);
v___x_1957_ = v___x_1951_;
goto v_reusejp_1956_;
}
else
{
lean_object* v_reuseFailAlloc_1958_; 
v_reuseFailAlloc_1958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1958_, 0, v___x_1955_);
v___x_1957_ = v_reuseFailAlloc_1958_;
goto v_reusejp_1956_;
}
v_reusejp_1956_:
{
return v___x_1957_;
}
}
else
{
lean_object* v___x_1959_; lean_object* v___x_1960_; 
lean_del_object(v___x_1951_);
lean_dec(v_a_1949_);
v___x_1959_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1);
v___x_1960_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v___x_1959_, v_a_1942_, v_a_1943_, v_a_1944_, v_a_1945_);
return v___x_1960_;
}
}
}
else
{
lean_object* v_a_1962_; lean_object* v___x_1964_; uint8_t v_isShared_1965_; uint8_t v_isSharedCheck_1969_; 
v_a_1962_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1969_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1969_ == 0)
{
v___x_1964_ = v___x_1948_;
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
else
{
lean_inc(v_a_1962_);
lean_dec(v___x_1948_);
v___x_1964_ = lean_box(0);
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
v_resetjp_1963_:
{
lean_object* v___x_1967_; 
if (v_isShared_1965_ == 0)
{
v___x_1967_ = v___x_1964_;
goto v_reusejp_1966_;
}
else
{
lean_object* v_reuseFailAlloc_1968_; 
v_reuseFailAlloc_1968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1968_, 0, v_a_1962_);
v___x_1967_ = v_reuseFailAlloc_1968_;
goto v_reusejp_1966_;
}
v_reusejp_1966_:
{
return v___x_1967_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___boxed(lean_object* v_u_1970_, lean_object* v_00_u03b1_1971_, lean_object* v_e_1972_, lean_object* v_a_1973_, lean_object* v_a_1974_, lean_object* v_a_1975_, lean_object* v_a_1976_, lean_object* v_a_1977_){
_start:
{
lean_object* v_res_1978_; 
v_res_1978_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v_u_1970_, v_00_u03b1_1971_, v_e_1972_, v_a_1973_, v_a_1974_, v_a_1975_, v_a_1976_);
lean_dec(v_a_1976_);
lean_dec_ref(v_a_1975_);
lean_dec(v_a_1974_);
lean_dec_ref(v_a_1973_);
return v_res_1978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat(lean_object* v_u_1979_, lean_object* v_00_u03b1_1980_, lean_object* v_e_1981_, lean_object* v___inst_1982_, lean_object* v_a_1983_, lean_object* v_a_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_){
_start:
{
lean_object* v___x_1988_; 
v___x_1988_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v_u_1979_, v_00_u03b1_1980_, v_e_1981_, v_a_1983_, v_a_1984_, v_a_1985_, v_a_1986_);
return v___x_1988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___boxed(lean_object* v_u_1989_, lean_object* v_00_u03b1_1990_, lean_object* v_e_1991_, lean_object* v___inst_1992_, lean_object* v_a_1993_, lean_object* v_a_1994_, lean_object* v_a_1995_, lean_object* v_a_1996_, lean_object* v_a_1997_){
_start:
{
lean_object* v_res_1998_; 
v_res_1998_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat(v_u_1989_, v_00_u03b1_1990_, v_e_1991_, v___inst_1992_, v_a_1993_, v_a_1994_, v_a_1995_, v_a_1996_);
lean_dec(v_a_1996_);
lean_dec_ref(v_a_1995_);
lean_dec(v_a_1994_);
lean_dec_ref(v_a_1993_);
lean_dec_ref(v___inst_1992_);
return v_res_1998_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveInt___auto__1(void){
_start:
{
lean_object* v___x_1999_; 
v___x_1999_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23);
return v___x_1999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveInt(lean_object* v_u_2000_, lean_object* v_00_u03b1_2001_, lean_object* v_e_2002_, lean_object* v___inst_2003_, lean_object* v_a_2004_, lean_object* v_a_2005_, lean_object* v_a_2006_, lean_object* v_a_2007_){
_start:
{
uint8_t v___x_2009_; lean_object* v___x_2010_; 
v___x_2009_ = 0;
lean_inc_ref(v_e_2002_);
lean_inc_ref(v_00_u03b1_2001_);
lean_inc(v_u_2000_);
v___x_2010_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_2000_, v_00_u03b1_2001_, v_e_2002_, v___x_2009_, v_a_2004_, v_a_2005_, v_a_2006_, v_a_2007_);
if (lean_obj_tag(v___x_2010_) == 0)
{
lean_object* v_a_2011_; lean_object* v___x_2013_; uint8_t v_isShared_2014_; uint8_t v_isSharedCheck_2023_; 
v_a_2011_ = lean_ctor_get(v___x_2010_, 0);
v_isSharedCheck_2023_ = !lean_is_exclusive(v___x_2010_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_2013_ = v___x_2010_;
v_isShared_2014_ = v_isSharedCheck_2023_;
goto v_resetjp_2012_;
}
else
{
lean_inc(v_a_2011_);
lean_dec(v___x_2010_);
v___x_2013_ = lean_box(0);
v_isShared_2014_ = v_isSharedCheck_2023_;
goto v_resetjp_2012_;
}
v_resetjp_2012_:
{
lean_object* v___x_2015_; 
v___x_2015_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_2000_, v_00_u03b1_2001_, v_e_2002_, v___inst_2003_, v_a_2011_);
if (lean_obj_tag(v___x_2015_) == 1)
{
lean_object* v_val_2016_; lean_object* v_snd_2017_; lean_object* v___x_2019_; 
v_val_2016_ = lean_ctor_get(v___x_2015_, 0);
lean_inc(v_val_2016_);
lean_dec_ref_known(v___x_2015_, 1);
v_snd_2017_ = lean_ctor_get(v_val_2016_, 1);
lean_inc(v_snd_2017_);
lean_dec(v_val_2016_);
if (v_isShared_2014_ == 0)
{
lean_ctor_set(v___x_2013_, 0, v_snd_2017_);
v___x_2019_ = v___x_2013_;
goto v_reusejp_2018_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v_snd_2017_);
v___x_2019_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2018_;
}
v_reusejp_2018_:
{
return v___x_2019_;
}
}
else
{
lean_object* v___x_2021_; lean_object* v___x_2022_; 
lean_dec(v___x_2015_);
lean_del_object(v___x_2013_);
v___x_2021_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1);
v___x_2022_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v___x_2021_, v_a_2004_, v_a_2005_, v_a_2006_, v_a_2007_);
return v___x_2022_;
}
}
}
else
{
lean_object* v_a_2024_; lean_object* v___x_2026_; uint8_t v_isShared_2027_; uint8_t v_isSharedCheck_2031_; 
lean_dec_ref(v___inst_2003_);
lean_dec_ref(v_e_2002_);
lean_dec_ref(v_00_u03b1_2001_);
lean_dec(v_u_2000_);
v_a_2024_ = lean_ctor_get(v___x_2010_, 0);
v_isSharedCheck_2031_ = !lean_is_exclusive(v___x_2010_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_2026_ = v___x_2010_;
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
else
{
lean_inc(v_a_2024_);
lean_dec(v___x_2010_);
v___x_2026_ = lean_box(0);
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
v_resetjp_2025_:
{
lean_object* v___x_2029_; 
if (v_isShared_2027_ == 0)
{
v___x_2029_ = v___x_2026_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2030_; 
v_reuseFailAlloc_2030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2030_, 0, v_a_2024_);
v___x_2029_ = v_reuseFailAlloc_2030_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
return v___x_2029_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveInt___boxed(lean_object* v_u_2032_, lean_object* v_00_u03b1_2033_, lean_object* v_e_2034_, lean_object* v___inst_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_, lean_object* v_a_2039_, lean_object* v_a_2040_){
_start:
{
lean_object* v_res_2041_; 
v_res_2041_ = lp_mathlib_Mathlib_Meta_NormNum_deriveInt(v_u_2032_, v_00_u03b1_2033_, v_e_2034_, v___inst_2035_, v_a_2036_, v_a_2037_, v_a_2038_, v_a_2039_);
lean_dec(v_a_2039_);
lean_dec_ref(v_a_2038_);
lean_dec(v_a_2037_);
lean_dec_ref(v_a_2036_);
return v_res_2041_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveRat___auto__1(void){
_start:
{
lean_object* v___x_2042_; 
v___x_2042_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1___closed__23);
return v___x_2042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveRat(lean_object* v_u_2043_, lean_object* v_00_u03b1_2044_, lean_object* v_e_2045_, lean_object* v___inst_2046_, lean_object* v_a_2047_, lean_object* v_a_2048_, lean_object* v_a_2049_, lean_object* v_a_2050_){
_start:
{
uint8_t v___x_2052_; lean_object* v___x_2053_; 
v___x_2052_ = 0;
lean_inc_ref(v_e_2045_);
lean_inc_ref(v_00_u03b1_2044_);
lean_inc(v_u_2043_);
v___x_2053_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_2043_, v_00_u03b1_2044_, v_e_2045_, v___x_2052_, v_a_2047_, v_a_2048_, v_a_2049_, v_a_2050_);
if (lean_obj_tag(v___x_2053_) == 0)
{
lean_object* v_a_2054_; lean_object* v___x_2056_; uint8_t v_isShared_2057_; uint8_t v_isSharedCheck_2065_; 
v_a_2054_ = lean_ctor_get(v___x_2053_, 0);
v_isSharedCheck_2065_ = !lean_is_exclusive(v___x_2053_);
if (v_isSharedCheck_2065_ == 0)
{
v___x_2056_ = v___x_2053_;
v_isShared_2057_ = v_isSharedCheck_2065_;
goto v_resetjp_2055_;
}
else
{
lean_inc(v_a_2054_);
lean_dec(v___x_2053_);
v___x_2056_ = lean_box(0);
v_isShared_2057_ = v_isSharedCheck_2065_;
goto v_resetjp_2055_;
}
v_resetjp_2055_:
{
lean_object* v___x_2058_; 
v___x_2058_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_2043_, v_00_u03b1_2044_, v_e_2045_, v___inst_2046_, v_a_2054_);
if (lean_obj_tag(v___x_2058_) == 1)
{
lean_object* v_val_2059_; lean_object* v___x_2061_; 
v_val_2059_ = lean_ctor_get(v___x_2058_, 0);
lean_inc(v_val_2059_);
lean_dec_ref_known(v___x_2058_, 1);
if (v_isShared_2057_ == 0)
{
lean_ctor_set(v___x_2056_, 0, v_val_2059_);
v___x_2061_ = v___x_2056_;
goto v_reusejp_2060_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v_val_2059_);
v___x_2061_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2060_;
}
v_reusejp_2060_:
{
return v___x_2061_;
}
}
else
{
lean_object* v___x_2063_; lean_object* v___x_2064_; 
lean_dec(v___x_2058_);
lean_del_object(v___x_2056_);
v___x_2063_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1);
v___x_2064_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v___x_2063_, v_a_2047_, v_a_2048_, v_a_2049_, v_a_2050_);
return v___x_2064_;
}
}
}
else
{
lean_object* v_a_2066_; lean_object* v___x_2068_; uint8_t v_isShared_2069_; uint8_t v_isSharedCheck_2073_; 
lean_dec_ref(v___inst_2046_);
lean_dec_ref(v_e_2045_);
lean_dec_ref(v_00_u03b1_2044_);
lean_dec(v_u_2043_);
v_a_2066_ = lean_ctor_get(v___x_2053_, 0);
v_isSharedCheck_2073_ = !lean_is_exclusive(v___x_2053_);
if (v_isSharedCheck_2073_ == 0)
{
v___x_2068_ = v___x_2053_;
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
else
{
lean_inc(v_a_2066_);
lean_dec(v___x_2053_);
v___x_2068_ = lean_box(0);
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
v_resetjp_2067_:
{
lean_object* v___x_2071_; 
if (v_isShared_2069_ == 0)
{
v___x_2071_ = v___x_2068_;
goto v_reusejp_2070_;
}
else
{
lean_object* v_reuseFailAlloc_2072_; 
v_reuseFailAlloc_2072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2072_, 0, v_a_2066_);
v___x_2071_ = v_reuseFailAlloc_2072_;
goto v_reusejp_2070_;
}
v_reusejp_2070_:
{
return v___x_2071_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveRat___boxed(lean_object* v_u_2074_, lean_object* v_00_u03b1_2075_, lean_object* v_e_2076_, lean_object* v___inst_2077_, lean_object* v_a_2078_, lean_object* v_a_2079_, lean_object* v_a_2080_, lean_object* v_a_2081_, lean_object* v_a_2082_){
_start:
{
lean_object* v_res_2083_; 
v_res_2083_ = lp_mathlib_Mathlib_Meta_NormNum_deriveRat(v_u_2074_, v_00_u03b1_2075_, v_e_2076_, v___inst_2077_, v_a_2078_, v_a_2079_, v_a_2080_, v_a_2081_);
lean_dec(v_a_2081_);
lean_dec_ref(v_a_2080_);
lean_dec(v_a_2079_);
lean_dec_ref(v_a_2078_);
return v_res_2083_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0(void){
_start:
{
lean_object* v___x_2084_; lean_object* v___x_2085_; 
v___x_2084_ = lean_box(0);
v___x_2085_ = l_Lean_Expr_sort___override(v___x_2084_);
return v___x_2085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBool(lean_object* v_p_2086_, lean_object* v_a_2087_, lean_object* v_a_2088_, lean_object* v_a_2089_, lean_object* v_a_2090_){
_start:
{
lean_object* v___x_2092_; lean_object* v___x_2093_; uint8_t v___x_2094_; lean_object* v___x_2095_; 
v___x_2092_ = lean_box(0);
v___x_2093_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0);
v___x_2094_ = 0;
v___x_2095_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_2092_, v___x_2093_, v_p_2086_, v___x_2094_, v_a_2087_, v_a_2088_, v_a_2089_, v_a_2090_);
if (lean_obj_tag(v___x_2095_) == 0)
{
lean_object* v_a_2096_; lean_object* v___x_2098_; uint8_t v_isShared_2099_; uint8_t v_isSharedCheck_2109_; 
v_a_2096_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2109_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2098_ = v___x_2095_;
v_isShared_2099_ = v_isSharedCheck_2109_;
goto v_resetjp_2097_;
}
else
{
lean_inc(v_a_2096_);
lean_dec(v___x_2095_);
v___x_2098_ = lean_box(0);
v_isShared_2099_ = v_isSharedCheck_2109_;
goto v_resetjp_2097_;
}
v_resetjp_2097_:
{
if (lean_obj_tag(v_a_2096_) == 0)
{
uint8_t v_val_2100_; lean_object* v_proof_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2105_; 
v_val_2100_ = lean_ctor_get_uint8(v_a_2096_, sizeof(void*)*1);
v_proof_2101_ = lean_ctor_get(v_a_2096_, 0);
lean_inc_ref(v_proof_2101_);
lean_dec_ref_known(v_a_2096_, 1);
v___x_2102_ = lean_box(v_val_2100_);
v___x_2103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2103_, 0, v___x_2102_);
lean_ctor_set(v___x_2103_, 1, v_proof_2101_);
if (v_isShared_2099_ == 0)
{
lean_ctor_set(v___x_2098_, 0, v___x_2103_);
v___x_2105_ = v___x_2098_;
goto v_reusejp_2104_;
}
else
{
lean_object* v_reuseFailAlloc_2106_; 
v_reuseFailAlloc_2106_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2106_, 0, v___x_2103_);
v___x_2105_ = v_reuseFailAlloc_2106_;
goto v_reusejp_2104_;
}
v_reusejp_2104_:
{
return v___x_2105_;
}
}
else
{
lean_object* v___x_2107_; lean_object* v___x_2108_; 
lean_del_object(v___x_2098_);
lean_dec(v_a_2096_);
v___x_2107_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg___closed__1);
v___x_2108_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_derive_spec__3___redArg(v___x_2107_, v_a_2087_, v_a_2088_, v_a_2089_, v_a_2090_);
return v___x_2108_;
}
}
}
else
{
lean_object* v_a_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2117_; 
v_a_2110_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2112_ = v___x_2095_;
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_a_2110_);
lean_dec(v___x_2095_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2115_; 
if (v_isShared_2113_ == 0)
{
v___x_2115_ = v___x_2112_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v_a_2110_);
v___x_2115_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
return v___x_2115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBool___boxed(lean_object* v_p_2118_, lean_object* v_a_2119_, lean_object* v_a_2120_, lean_object* v_a_2121_, lean_object* v_a_2122_, lean_object* v_a_2123_){
_start:
{
lean_object* v_res_2124_; 
v_res_2124_ = lp_mathlib_Mathlib_Meta_NormNum_deriveBool(v_p_2118_, v_a_2119_, v_a_2120_, v_a_2121_, v_a_2122_);
lean_dec(v_a_2122_);
lean_dec_ref(v_a_2121_);
lean_dec(v_a_2120_);
lean_dec_ref(v_a_2119_);
return v_res_2124_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2(void){
_start:
{
lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; 
v___x_2128_ = lean_box(0);
v___x_2129_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__1));
v___x_2130_ = l_Lean_Expr_const___override(v___x_2129_, v___x_2128_);
return v___x_2130_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6(void){
_start:
{
lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; 
v___x_2136_ = lean_box(0);
v___x_2137_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__5));
v___x_2138_ = l_Lean_Expr_const___override(v___x_2137_, v___x_2136_);
return v___x_2138_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9(void){
_start:
{
lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; 
v___x_2143_ = lean_box(0);
v___x_2144_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__8));
v___x_2145_ = l_Lean_Expr_const___override(v___x_2144_, v___x_2143_);
return v___x_2145_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13(void){
_start:
{
lean_object* v___x_2151_; lean_object* v___x_2152_; 
v___x_2151_ = lean_box(0);
v___x_2152_ = l_Lean_Level_succ___override(v___x_2151_);
return v___x_2152_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14(void){
_start:
{
lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; 
v___x_2153_ = lean_box(0);
v___x_2154_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__13);
v___x_2155_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2155_, 0, v___x_2154_);
lean_ctor_set(v___x_2155_, 1, v___x_2153_);
return v___x_2155_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15(void){
_start:
{
lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; 
v___x_2156_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__14);
v___x_2157_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__12));
v___x_2158_ = l_Lean_Expr_const___override(v___x_2157_, v___x_2156_);
return v___x_2158_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19(void){
_start:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; 
v___x_2164_ = lean_box(0);
v___x_2165_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__18));
v___x_2166_ = l_Lean_Expr_const___override(v___x_2165_, v___x_2164_);
return v___x_2166_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20(void){
_start:
{
uint8_t v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; 
v___x_2167_ = 0;
v___x_2168_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBool___closed__0);
v___x_2169_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__19);
v___x_2170_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__17));
v___x_2171_ = l_Lean_Expr_lam___override(v___x_2170_, v___x_2169_, v___x_2168_, v___x_2167_);
return v___x_2171_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21(void){
_start:
{
lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; 
v___x_2172_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__20);
v___x_2173_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__15);
v___x_2174_ = l_Lean_Expr_app___override(v___x_2173_, v___x_2172_);
return v___x_2174_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24(void){
_start:
{
lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; 
v___x_2179_ = lean_box(0);
v___x_2180_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__23));
v___x_2181_ = l_Lean_Expr_const___override(v___x_2180_, v___x_2179_);
return v___x_2181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff(lean_object* v_p_2182_, lean_object* v_p_x27_2183_, lean_object* v_hp_2184_, lean_object* v_a_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_, lean_object* v_a_2188_){
_start:
{
lean_object* v___x_2190_; 
lean_inc_ref(v_p_2182_);
v___x_2190_ = lp_mathlib_Mathlib_Meta_NormNum_deriveBool(v_p_2182_, v_a_2185_, v_a_2186_, v_a_2187_, v_a_2188_);
if (lean_obj_tag(v___x_2190_) == 0)
{
lean_object* v_a_2191_; lean_object* v___x_2193_; uint8_t v_isShared_2194_; uint8_t v_isSharedCheck_2245_; 
v_a_2191_ = lean_ctor_get(v___x_2190_, 0);
v_isSharedCheck_2245_ = !lean_is_exclusive(v___x_2190_);
if (v_isSharedCheck_2245_ == 0)
{
v___x_2193_ = v___x_2190_;
v_isShared_2194_ = v_isSharedCheck_2245_;
goto v_resetjp_2192_;
}
else
{
lean_inc(v_a_2191_);
lean_dec(v___x_2190_);
v___x_2193_ = lean_box(0);
v_isShared_2194_ = v_isSharedCheck_2245_;
goto v_resetjp_2192_;
}
v_resetjp_2192_:
{
lean_object* v_fst_2195_; uint8_t v___x_2196_; 
v_fst_2195_ = lean_ctor_get(v_a_2191_, 0);
lean_inc(v_fst_2195_);
v___x_2196_ = lean_unbox(v_fst_2195_);
if (v___x_2196_ == 0)
{
lean_object* v_snd_2197_; lean_object* v___x_2199_; uint8_t v_isShared_2200_; uint8_t v_isSharedCheck_2219_; 
v_snd_2197_ = lean_ctor_get(v_a_2191_, 1);
v_isSharedCheck_2219_ = !lean_is_exclusive(v_a_2191_);
if (v_isSharedCheck_2219_ == 0)
{
lean_object* v_unused_2220_; 
v_unused_2220_ = lean_ctor_get(v_a_2191_, 0);
lean_dec(v_unused_2220_);
v___x_2199_ = v_a_2191_;
v_isShared_2200_ = v_isSharedCheck_2219_;
goto v_resetjp_2198_;
}
else
{
lean_inc(v_snd_2197_);
lean_dec(v_a_2191_);
v___x_2199_ = lean_box(0);
v_isShared_2200_ = v_isSharedCheck_2219_;
goto v_resetjp_2198_;
}
v_resetjp_2198_:
{
lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2214_; 
v___x_2201_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2);
lean_inc_ref(v_p_x27_2183_);
v___x_2202_ = l_Lean_Expr_app___override(v___x_2201_, v_p_x27_2183_);
v___x_2203_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6);
lean_inc_ref(v_p_2182_);
v___x_2204_ = l_Lean_Expr_app___override(v___x_2201_, v_p_2182_);
v___x_2205_ = l_Lean_Expr_app___override(v___x_2203_, v___x_2204_);
v___x_2206_ = l_Lean_Expr_app___override(v___x_2205_, v___x_2202_);
v___x_2207_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__9);
v___x_2208_ = l_Lean_Expr_app___override(v___x_2207_, v_p_2182_);
v___x_2209_ = l_Lean_Expr_app___override(v___x_2208_, v_p_x27_2183_);
v___x_2210_ = l_Lean_Expr_app___override(v___x_2209_, v_hp_2184_);
v___x_2211_ = l_Lean_Expr_app___override(v___x_2206_, v___x_2210_);
v___x_2212_ = l_Lean_Expr_app___override(v___x_2211_, v_snd_2197_);
if (v_isShared_2200_ == 0)
{
lean_ctor_set(v___x_2199_, 1, v___x_2212_);
v___x_2214_ = v___x_2199_;
goto v_reusejp_2213_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_fst_2195_);
lean_ctor_set(v_reuseFailAlloc_2218_, 1, v___x_2212_);
v___x_2214_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2213_;
}
v_reusejp_2213_:
{
lean_object* v___x_2216_; 
if (v_isShared_2194_ == 0)
{
lean_ctor_set(v___x_2193_, 0, v___x_2214_);
v___x_2216_ = v___x_2193_;
goto v_reusejp_2215_;
}
else
{
lean_object* v_reuseFailAlloc_2217_; 
v_reuseFailAlloc_2217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2217_, 0, v___x_2214_);
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
lean_object* v_snd_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2243_; 
v_snd_2221_ = lean_ctor_get(v_a_2191_, 1);
v_isSharedCheck_2243_ = !lean_is_exclusive(v_a_2191_);
if (v_isSharedCheck_2243_ == 0)
{
lean_object* v_unused_2244_; 
v_unused_2244_ = lean_ctor_get(v_a_2191_, 0);
lean_dec(v_unused_2244_);
v___x_2223_ = v_a_2191_;
v_isShared_2224_ = v_isSharedCheck_2243_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_snd_2221_);
lean_dec(v_a_2191_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2243_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2238_; 
v___x_2225_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__21);
v___x_2226_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__2);
lean_inc_ref(v_p_x27_2183_);
v___x_2227_ = l_Lean_Expr_app___override(v___x_2226_, v_p_x27_2183_);
v___x_2228_ = l_Lean_Expr_app___override(v___x_2225_, v___x_2227_);
v___x_2229_ = l_Lean_Expr_app___override(v___x_2228_, v_p_x27_2183_);
v___x_2230_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__6);
v___x_2231_ = l_Lean_Expr_app___override(v___x_2230_, v_p_2182_);
v___x_2232_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___closed__24);
v___x_2233_ = l_Lean_Expr_app___override(v___x_2229_, v___x_2232_);
v___x_2234_ = l_Lean_Expr_app___override(v___x_2231_, v___x_2233_);
v___x_2235_ = l_Lean_Expr_app___override(v___x_2234_, v_hp_2184_);
v___x_2236_ = l_Lean_Expr_app___override(v___x_2235_, v_snd_2221_);
if (v_isShared_2224_ == 0)
{
lean_ctor_set(v___x_2223_, 1, v___x_2236_);
v___x_2238_ = v___x_2223_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2242_; 
v_reuseFailAlloc_2242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2242_, 0, v_fst_2195_);
lean_ctor_set(v_reuseFailAlloc_2242_, 1, v___x_2236_);
v___x_2238_ = v_reuseFailAlloc_2242_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
lean_object* v___x_2240_; 
if (v_isShared_2194_ == 0)
{
lean_ctor_set(v___x_2193_, 0, v___x_2238_);
v___x_2240_ = v___x_2193_;
goto v_reusejp_2239_;
}
else
{
lean_object* v_reuseFailAlloc_2241_; 
v_reuseFailAlloc_2241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2241_, 0, v___x_2238_);
v___x_2240_ = v_reuseFailAlloc_2241_;
goto v_reusejp_2239_;
}
v_reusejp_2239_:
{
return v___x_2240_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_hp_2184_);
lean_dec_ref(v_p_x27_2183_);
lean_dec_ref(v_p_2182_);
return v___x_2190_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff___boxed(lean_object* v_p_2246_, lean_object* v_p_x27_2247_, lean_object* v_hp_2248_, lean_object* v_a_2249_, lean_object* v_a_2250_, lean_object* v_a_2251_, lean_object* v_a_2252_, lean_object* v_a_2253_){
_start:
{
lean_object* v_res_2254_; 
v_res_2254_ = lp_mathlib_Mathlib_Meta_NormNum_deriveBoolOfIff(v_p_2246_, v_p_x27_2247_, v_hp_2248_, v_a_2249_, v_a_2250_, v_a_2251_, v_a_2252_);
lean_dec(v_a_2252_);
lean_dec_ref(v_a_2251_);
lean_dec(v_a_2250_);
lean_dec_ref(v_a_2249_);
return v_res_2254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_eval(lean_object* v_e_2255_, uint8_t v_post_2256_, lean_object* v_a_2257_, lean_object* v_a_2258_, lean_object* v_a_2259_, lean_object* v_a_2260_){
_start:
{
uint8_t v___x_2262_; 
lean_inc_ref(v_e_2255_);
v___x_2262_ = lp_mathlib_Lean_Expr_isExplicitNumber(v_e_2255_);
if (v___x_2262_ == 0)
{
lean_object* v___x_2263_; 
v___x_2263_ = lp_mathlib_Qq_inferTypeQ_x27(v_e_2255_, v_a_2257_, v_a_2258_, v_a_2259_, v_a_2260_);
if (lean_obj_tag(v___x_2263_) == 0)
{
lean_object* v_a_2264_; lean_object* v_snd_2265_; lean_object* v_fst_2266_; lean_object* v_fst_2267_; lean_object* v_snd_2268_; lean_object* v___x_2269_; 
v_a_2264_ = lean_ctor_get(v___x_2263_, 0);
lean_inc(v_a_2264_);
lean_dec_ref_known(v___x_2263_, 1);
v_snd_2265_ = lean_ctor_get(v_a_2264_, 1);
lean_inc(v_snd_2265_);
v_fst_2266_ = lean_ctor_get(v_a_2264_, 0);
lean_inc_n(v_fst_2266_, 2);
lean_dec(v_a_2264_);
v_fst_2267_ = lean_ctor_get(v_snd_2265_, 0);
lean_inc_n(v_fst_2267_, 2);
v_snd_2268_ = lean_ctor_get(v_snd_2265_, 1);
lean_inc_n(v_snd_2268_, 2);
lean_dec(v_snd_2265_);
v___x_2269_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_2266_, v_fst_2267_, v_snd_2268_, v_post_2256_, v_a_2257_, v_a_2258_, v_a_2259_, v_a_2260_);
if (lean_obj_tag(v___x_2269_) == 0)
{
lean_object* v_a_2270_; lean_object* v___x_2271_; 
v_a_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc(v_a_2270_);
lean_dec_ref_known(v___x_2269_, 1);
v___x_2271_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(v_fst_2266_, v_fst_2267_, v_snd_2268_, v_a_2270_, v_a_2257_, v_a_2258_, v_a_2259_, v_a_2260_);
return v___x_2271_;
}
else
{
lean_object* v_a_2272_; lean_object* v___x_2274_; uint8_t v_isShared_2275_; uint8_t v_isSharedCheck_2279_; 
lean_dec(v_snd_2268_);
lean_dec(v_fst_2267_);
lean_dec(v_fst_2266_);
v_a_2272_ = lean_ctor_get(v___x_2269_, 0);
v_isSharedCheck_2279_ = !lean_is_exclusive(v___x_2269_);
if (v_isSharedCheck_2279_ == 0)
{
v___x_2274_ = v___x_2269_;
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
else
{
lean_inc(v_a_2272_);
lean_dec(v___x_2269_);
v___x_2274_ = lean_box(0);
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
v_resetjp_2273_:
{
lean_object* v___x_2277_; 
if (v_isShared_2275_ == 0)
{
v___x_2277_ = v___x_2274_;
goto v_reusejp_2276_;
}
else
{
lean_object* v_reuseFailAlloc_2278_; 
v_reuseFailAlloc_2278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2278_, 0, v_a_2272_);
v___x_2277_ = v_reuseFailAlloc_2278_;
goto v_reusejp_2276_;
}
v_reusejp_2276_:
{
return v___x_2277_;
}
}
}
}
else
{
lean_object* v_a_2280_; lean_object* v___x_2282_; uint8_t v_isShared_2283_; uint8_t v_isSharedCheck_2287_; 
v_a_2280_ = lean_ctor_get(v___x_2263_, 0);
v_isSharedCheck_2287_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2287_ == 0)
{
v___x_2282_ = v___x_2263_;
v_isShared_2283_ = v_isSharedCheck_2287_;
goto v_resetjp_2281_;
}
else
{
lean_inc(v_a_2280_);
lean_dec(v___x_2263_);
v___x_2282_ = lean_box(0);
v_isShared_2283_ = v_isSharedCheck_2287_;
goto v_resetjp_2281_;
}
v_resetjp_2281_:
{
lean_object* v___x_2285_; 
if (v_isShared_2283_ == 0)
{
v___x_2285_ = v___x_2282_;
goto v_reusejp_2284_;
}
else
{
lean_object* v_reuseFailAlloc_2286_; 
v_reuseFailAlloc_2286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2286_, 0, v_a_2280_);
v___x_2285_ = v_reuseFailAlloc_2286_;
goto v_reusejp_2284_;
}
v_reusejp_2284_:
{
return v___x_2285_;
}
}
}
}
else
{
lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; 
v___x_2288_ = lean_box(0);
v___x_2289_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2289_, 0, v_e_2255_);
lean_ctor_set(v___x_2289_, 1, v___x_2288_);
lean_ctor_set_uint8(v___x_2289_, sizeof(void*)*2, v___x_2262_);
v___x_2290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2290_, 0, v___x_2289_);
return v___x_2290_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_eval___boxed(lean_object* v_e_2291_, lean_object* v_post_2292_, lean_object* v_a_2293_, lean_object* v_a_2294_, lean_object* v_a_2295_, lean_object* v_a_2296_, lean_object* v_a_2297_){
_start:
{
uint8_t v_post_boxed_2298_; lean_object* v_res_2299_; 
v_post_boxed_2298_ = lean_unbox(v_post_2292_);
v_res_2299_ = lp_mathlib_Mathlib_Meta_NormNum_eval(v_e_2291_, v_post_boxed_2298_, v_a_2293_, v_a_2294_, v_a_2295_, v_a_2296_);
lean_dec(v_a_2296_);
lean_dec_ref(v_a_2295_);
lean_dec(v_a_2294_);
lean_dec_ref(v_a_2293_);
return v_res_2299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_x_2300_, lean_object* v_x_2301_, lean_object* v_x_2302_, lean_object* v_x_2303_){
_start:
{
lean_object* v_ks_2304_; lean_object* v_vs_2305_; lean_object* v___x_2307_; uint8_t v_isShared_2308_; uint8_t v_isSharedCheck_2329_; 
v_ks_2304_ = lean_ctor_get(v_x_2300_, 0);
v_vs_2305_ = lean_ctor_get(v_x_2300_, 1);
v_isSharedCheck_2329_ = !lean_is_exclusive(v_x_2300_);
if (v_isSharedCheck_2329_ == 0)
{
v___x_2307_ = v_x_2300_;
v_isShared_2308_ = v_isSharedCheck_2329_;
goto v_resetjp_2306_;
}
else
{
lean_inc(v_vs_2305_);
lean_inc(v_ks_2304_);
lean_dec(v_x_2300_);
v___x_2307_ = lean_box(0);
v_isShared_2308_ = v_isSharedCheck_2329_;
goto v_resetjp_2306_;
}
v_resetjp_2306_:
{
lean_object* v___x_2309_; uint8_t v___x_2310_; 
v___x_2309_ = lean_array_get_size(v_ks_2304_);
v___x_2310_ = lean_nat_dec_lt(v_x_2301_, v___x_2309_);
if (v___x_2310_ == 0)
{
lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2314_; 
lean_dec(v_x_2301_);
v___x_2311_ = lean_array_push(v_ks_2304_, v_x_2302_);
v___x_2312_ = lean_array_push(v_vs_2305_, v_x_2303_);
if (v_isShared_2308_ == 0)
{
lean_ctor_set(v___x_2307_, 1, v___x_2312_);
lean_ctor_set(v___x_2307_, 0, v___x_2311_);
v___x_2314_ = v___x_2307_;
goto v_reusejp_2313_;
}
else
{
lean_object* v_reuseFailAlloc_2315_; 
v_reuseFailAlloc_2315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2315_, 0, v___x_2311_);
lean_ctor_set(v_reuseFailAlloc_2315_, 1, v___x_2312_);
v___x_2314_ = v_reuseFailAlloc_2315_;
goto v_reusejp_2313_;
}
v_reusejp_2313_:
{
return v___x_2314_;
}
}
else
{
lean_object* v_k_x27_2316_; uint8_t v___x_2317_; 
v_k_x27_2316_ = lean_array_fget_borrowed(v_ks_2304_, v_x_2301_);
v___x_2317_ = lean_name_eq(v_x_2302_, v_k_x27_2316_);
if (v___x_2317_ == 0)
{
lean_object* v___x_2319_; 
if (v_isShared_2308_ == 0)
{
v___x_2319_ = v___x_2307_;
goto v_reusejp_2318_;
}
else
{
lean_object* v_reuseFailAlloc_2323_; 
v_reuseFailAlloc_2323_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2323_, 0, v_ks_2304_);
lean_ctor_set(v_reuseFailAlloc_2323_, 1, v_vs_2305_);
v___x_2319_ = v_reuseFailAlloc_2323_;
goto v_reusejp_2318_;
}
v_reusejp_2318_:
{
lean_object* v___x_2320_; lean_object* v___x_2321_; 
v___x_2320_ = lean_unsigned_to_nat(1u);
v___x_2321_ = lean_nat_add(v_x_2301_, v___x_2320_);
lean_dec(v_x_2301_);
v_x_2300_ = v___x_2319_;
v_x_2301_ = v___x_2321_;
goto _start;
}
}
else
{
lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2327_; 
v___x_2324_ = lean_array_fset(v_ks_2304_, v_x_2301_, v_x_2302_);
v___x_2325_ = lean_array_fset(v_vs_2305_, v_x_2301_, v_x_2303_);
lean_dec(v_x_2301_);
if (v_isShared_2308_ == 0)
{
lean_ctor_set(v___x_2307_, 1, v___x_2325_);
lean_ctor_set(v___x_2307_, 0, v___x_2324_);
v___x_2327_ = v___x_2307_;
goto v_reusejp_2326_;
}
else
{
lean_object* v_reuseFailAlloc_2328_; 
v_reuseFailAlloc_2328_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2328_, 0, v___x_2324_);
lean_ctor_set(v_reuseFailAlloc_2328_, 1, v___x_2325_);
v___x_2327_ = v_reuseFailAlloc_2328_;
goto v_reusejp_2326_;
}
v_reusejp_2326_:
{
return v___x_2327_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1___redArg(lean_object* v_n_2330_, lean_object* v_k_2331_, lean_object* v_v_2332_){
_start:
{
lean_object* v___x_2333_; lean_object* v___x_2334_; 
v___x_2333_ = lean_unsigned_to_nat(0u);
v___x_2334_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2___redArg(v_n_2330_, v___x_2333_, v_k_2331_, v_v_2332_);
return v___x_2334_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2335_; 
v___x_2335_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_2335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(lean_object* v_x_2336_, size_t v_x_2337_, size_t v_x_2338_, lean_object* v_x_2339_, lean_object* v_x_2340_){
_start:
{
if (lean_obj_tag(v_x_2336_) == 0)
{
lean_object* v_es_2341_; size_t v___x_2342_; size_t v___x_2343_; lean_object* v_j_2344_; lean_object* v___x_2345_; uint8_t v___x_2346_; 
v_es_2341_ = lean_ctor_get(v_x_2336_, 0);
v___x_2342_ = ((size_t)31ULL);
v___x_2343_ = lean_usize_land(v_x_2337_, v___x_2342_);
v_j_2344_ = lean_usize_to_nat(v___x_2343_);
v___x_2345_ = lean_array_get_size(v_es_2341_);
v___x_2346_ = lean_nat_dec_lt(v_j_2344_, v___x_2345_);
if (v___x_2346_ == 0)
{
lean_dec(v_j_2344_);
lean_dec(v_x_2340_);
lean_dec(v_x_2339_);
return v_x_2336_;
}
else
{
lean_object* v___x_2348_; uint8_t v_isShared_2349_; uint8_t v_isSharedCheck_2385_; 
lean_inc_ref(v_es_2341_);
v_isSharedCheck_2385_ = !lean_is_exclusive(v_x_2336_);
if (v_isSharedCheck_2385_ == 0)
{
lean_object* v_unused_2386_; 
v_unused_2386_ = lean_ctor_get(v_x_2336_, 0);
lean_dec(v_unused_2386_);
v___x_2348_ = v_x_2336_;
v_isShared_2349_ = v_isSharedCheck_2385_;
goto v_resetjp_2347_;
}
else
{
lean_dec(v_x_2336_);
v___x_2348_ = lean_box(0);
v_isShared_2349_ = v_isSharedCheck_2385_;
goto v_resetjp_2347_;
}
v_resetjp_2347_:
{
lean_object* v_v_2350_; lean_object* v___x_2351_; lean_object* v_xs_x27_2352_; lean_object* v___y_2354_; 
v_v_2350_ = lean_array_fget(v_es_2341_, v_j_2344_);
v___x_2351_ = lean_box(0);
v_xs_x27_2352_ = lean_array_fset(v_es_2341_, v_j_2344_, v___x_2351_);
switch(lean_obj_tag(v_v_2350_))
{
case 0:
{
lean_object* v_key_2359_; lean_object* v_val_2360_; lean_object* v___x_2362_; uint8_t v_isShared_2363_; uint8_t v_isSharedCheck_2370_; 
v_key_2359_ = lean_ctor_get(v_v_2350_, 0);
v_val_2360_ = lean_ctor_get(v_v_2350_, 1);
v_isSharedCheck_2370_ = !lean_is_exclusive(v_v_2350_);
if (v_isSharedCheck_2370_ == 0)
{
v___x_2362_ = v_v_2350_;
v_isShared_2363_ = v_isSharedCheck_2370_;
goto v_resetjp_2361_;
}
else
{
lean_inc(v_val_2360_);
lean_inc(v_key_2359_);
lean_dec(v_v_2350_);
v___x_2362_ = lean_box(0);
v_isShared_2363_ = v_isSharedCheck_2370_;
goto v_resetjp_2361_;
}
v_resetjp_2361_:
{
uint8_t v___x_2364_; 
v___x_2364_ = lean_name_eq(v_x_2339_, v_key_2359_);
if (v___x_2364_ == 0)
{
lean_object* v___x_2365_; lean_object* v___x_2366_; 
lean_del_object(v___x_2362_);
v___x_2365_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2359_, v_val_2360_, v_x_2339_, v_x_2340_);
v___x_2366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2366_, 0, v___x_2365_);
v___y_2354_ = v___x_2366_;
goto v___jp_2353_;
}
else
{
lean_object* v___x_2368_; 
lean_dec(v_val_2360_);
lean_dec(v_key_2359_);
if (v_isShared_2363_ == 0)
{
lean_ctor_set(v___x_2362_, 1, v_x_2340_);
lean_ctor_set(v___x_2362_, 0, v_x_2339_);
v___x_2368_ = v___x_2362_;
goto v_reusejp_2367_;
}
else
{
lean_object* v_reuseFailAlloc_2369_; 
v_reuseFailAlloc_2369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2369_, 0, v_x_2339_);
lean_ctor_set(v_reuseFailAlloc_2369_, 1, v_x_2340_);
v___x_2368_ = v_reuseFailAlloc_2369_;
goto v_reusejp_2367_;
}
v_reusejp_2367_:
{
v___y_2354_ = v___x_2368_;
goto v___jp_2353_;
}
}
}
}
case 1:
{
lean_object* v_node_2371_; lean_object* v___x_2373_; uint8_t v_isShared_2374_; uint8_t v_isSharedCheck_2383_; 
v_node_2371_ = lean_ctor_get(v_v_2350_, 0);
v_isSharedCheck_2383_ = !lean_is_exclusive(v_v_2350_);
if (v_isSharedCheck_2383_ == 0)
{
v___x_2373_ = v_v_2350_;
v_isShared_2374_ = v_isSharedCheck_2383_;
goto v_resetjp_2372_;
}
else
{
lean_inc(v_node_2371_);
lean_dec(v_v_2350_);
v___x_2373_ = lean_box(0);
v_isShared_2374_ = v_isSharedCheck_2383_;
goto v_resetjp_2372_;
}
v_resetjp_2372_:
{
size_t v___x_2375_; size_t v___x_2376_; size_t v___x_2377_; size_t v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2381_; 
v___x_2375_ = ((size_t)5ULL);
v___x_2376_ = lean_usize_shift_right(v_x_2337_, v___x_2375_);
v___x_2377_ = ((size_t)1ULL);
v___x_2378_ = lean_usize_add(v_x_2338_, v___x_2377_);
v___x_2379_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(v_node_2371_, v___x_2376_, v___x_2378_, v_x_2339_, v_x_2340_);
if (v_isShared_2374_ == 0)
{
lean_ctor_set(v___x_2373_, 0, v___x_2379_);
v___x_2381_ = v___x_2373_;
goto v_reusejp_2380_;
}
else
{
lean_object* v_reuseFailAlloc_2382_; 
v_reuseFailAlloc_2382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2382_, 0, v___x_2379_);
v___x_2381_ = v_reuseFailAlloc_2382_;
goto v_reusejp_2380_;
}
v_reusejp_2380_:
{
v___y_2354_ = v___x_2381_;
goto v___jp_2353_;
}
}
}
default: 
{
lean_object* v___x_2384_; 
v___x_2384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2384_, 0, v_x_2339_);
lean_ctor_set(v___x_2384_, 1, v_x_2340_);
v___y_2354_ = v___x_2384_;
goto v___jp_2353_;
}
}
v___jp_2353_:
{
lean_object* v___x_2355_; lean_object* v___x_2357_; 
v___x_2355_ = lean_array_fset(v_xs_x27_2352_, v_j_2344_, v___y_2354_);
lean_dec(v_j_2344_);
if (v_isShared_2349_ == 0)
{
lean_ctor_set(v___x_2348_, 0, v___x_2355_);
v___x_2357_ = v___x_2348_;
goto v_reusejp_2356_;
}
else
{
lean_object* v_reuseFailAlloc_2358_; 
v_reuseFailAlloc_2358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2358_, 0, v___x_2355_);
v___x_2357_ = v_reuseFailAlloc_2358_;
goto v_reusejp_2356_;
}
v_reusejp_2356_:
{
return v___x_2357_;
}
}
}
}
}
else
{
lean_object* v_ks_2387_; lean_object* v_vs_2388_; lean_object* v___x_2390_; uint8_t v_isShared_2391_; uint8_t v_isSharedCheck_2408_; 
v_ks_2387_ = lean_ctor_get(v_x_2336_, 0);
v_vs_2388_ = lean_ctor_get(v_x_2336_, 1);
v_isSharedCheck_2408_ = !lean_is_exclusive(v_x_2336_);
if (v_isSharedCheck_2408_ == 0)
{
v___x_2390_ = v_x_2336_;
v_isShared_2391_ = v_isSharedCheck_2408_;
goto v_resetjp_2389_;
}
else
{
lean_inc(v_vs_2388_);
lean_inc(v_ks_2387_);
lean_dec(v_x_2336_);
v___x_2390_ = lean_box(0);
v_isShared_2391_ = v_isSharedCheck_2408_;
goto v_resetjp_2389_;
}
v_resetjp_2389_:
{
lean_object* v___x_2393_; 
if (v_isShared_2391_ == 0)
{
v___x_2393_ = v___x_2390_;
goto v_reusejp_2392_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v_ks_2387_);
lean_ctor_set(v_reuseFailAlloc_2407_, 1, v_vs_2388_);
v___x_2393_ = v_reuseFailAlloc_2407_;
goto v_reusejp_2392_;
}
v_reusejp_2392_:
{
lean_object* v_newNode_2394_; uint8_t v___y_2396_; size_t v___x_2402_; uint8_t v___x_2403_; 
v_newNode_2394_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1___redArg(v___x_2393_, v_x_2339_, v_x_2340_);
v___x_2402_ = ((size_t)7ULL);
v___x_2403_ = lean_usize_dec_le(v___x_2402_, v_x_2338_);
if (v___x_2403_ == 0)
{
lean_object* v___x_2404_; lean_object* v___x_2405_; uint8_t v___x_2406_; 
v___x_2404_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_2394_);
v___x_2405_ = lean_unsigned_to_nat(4u);
v___x_2406_ = lean_nat_dec_lt(v___x_2404_, v___x_2405_);
lean_dec(v___x_2404_);
v___y_2396_ = v___x_2406_;
goto v___jp_2395_;
}
else
{
v___y_2396_ = v___x_2403_;
goto v___jp_2395_;
}
v___jp_2395_:
{
if (v___y_2396_ == 0)
{
lean_object* v_ks_2397_; lean_object* v_vs_2398_; lean_object* v___x_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; 
v_ks_2397_ = lean_ctor_get(v_newNode_2394_, 0);
lean_inc_ref(v_ks_2397_);
v_vs_2398_ = lean_ctor_get(v_newNode_2394_, 1);
lean_inc_ref(v_vs_2398_);
lean_dec_ref(v_newNode_2394_);
v___x_2399_ = lean_unsigned_to_nat(0u);
v___x_2400_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___closed__0);
v___x_2401_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg(v_x_2338_, v_ks_2397_, v_vs_2398_, v___x_2399_, v___x_2400_);
lean_dec_ref(v_vs_2398_);
lean_dec_ref(v_ks_2397_);
return v___x_2401_;
}
else
{
return v_newNode_2394_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg(size_t v_depth_2409_, lean_object* v_keys_2410_, lean_object* v_vals_2411_, lean_object* v_i_2412_, lean_object* v_entries_2413_){
_start:
{
lean_object* v___x_2414_; uint8_t v___x_2415_; 
v___x_2414_ = lean_array_get_size(v_keys_2410_);
v___x_2415_ = lean_nat_dec_lt(v_i_2412_, v___x_2414_);
if (v___x_2415_ == 0)
{
lean_dec(v_i_2412_);
return v_entries_2413_;
}
else
{
lean_object* v_k_2416_; lean_object* v_v_2417_; uint64_t v___y_2419_; 
v_k_2416_ = lean_array_fget_borrowed(v_keys_2410_, v_i_2412_);
v_v_2417_ = lean_array_fget_borrowed(v_vals_2411_, v_i_2412_);
if (lean_obj_tag(v_k_2416_) == 0)
{
uint64_t v___x_2430_; 
v___x_2430_ = 1723ULL;
v___y_2419_ = v___x_2430_;
goto v___jp_2418_;
}
else
{
uint64_t v_hash_2431_; 
v_hash_2431_ = lean_ctor_get_uint64(v_k_2416_, sizeof(void*)*2);
v___y_2419_ = v_hash_2431_;
goto v___jp_2418_;
}
v___jp_2418_:
{
size_t v_h_2420_; size_t v___x_2421_; lean_object* v___x_2422_; size_t v___x_2423_; size_t v___x_2424_; size_t v___x_2425_; size_t v_h_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; 
v_h_2420_ = lean_uint64_to_usize(v___y_2419_);
v___x_2421_ = ((size_t)5ULL);
v___x_2422_ = lean_unsigned_to_nat(1u);
v___x_2423_ = ((size_t)1ULL);
v___x_2424_ = lean_usize_sub(v_depth_2409_, v___x_2423_);
v___x_2425_ = lean_usize_mul(v___x_2421_, v___x_2424_);
v_h_2426_ = lean_usize_shift_right(v_h_2420_, v___x_2425_);
v___x_2427_ = lean_nat_add(v_i_2412_, v___x_2422_);
lean_dec(v_i_2412_);
lean_inc(v_v_2417_);
lean_inc(v_k_2416_);
v___x_2428_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(v_entries_2413_, v_h_2426_, v_depth_2409_, v_k_2416_, v_v_2417_);
v_i_2412_ = v___x_2427_;
v_entries_2413_ = v___x_2428_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_depth_2432_, lean_object* v_keys_2433_, lean_object* v_vals_2434_, lean_object* v_i_2435_, lean_object* v_entries_2436_){
_start:
{
size_t v_depth_boxed_2437_; lean_object* v_res_2438_; 
v_depth_boxed_2437_ = lean_unbox_usize(v_depth_2432_);
lean_dec(v_depth_2432_);
v_res_2438_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg(v_depth_boxed_2437_, v_keys_2433_, v_vals_2434_, v_i_2435_, v_entries_2436_);
lean_dec_ref(v_vals_2434_);
lean_dec_ref(v_keys_2433_);
return v_res_2438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg___boxed(lean_object* v_x_2439_, lean_object* v_x_2440_, lean_object* v_x_2441_, lean_object* v_x_2442_, lean_object* v_x_2443_){
_start:
{
size_t v_x_358__boxed_2444_; size_t v_x_359__boxed_2445_; lean_object* v_res_2446_; 
v_x_358__boxed_2444_ = lean_unbox_usize(v_x_2440_);
lean_dec(v_x_2440_);
v_x_359__boxed_2445_ = lean_unbox_usize(v_x_2441_);
lean_dec(v_x_2441_);
v_res_2446_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(v_x_2439_, v_x_358__boxed_2444_, v_x_359__boxed_2445_, v_x_2442_, v_x_2443_);
return v_res_2446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0___redArg(lean_object* v_x_2447_, lean_object* v_x_2448_, lean_object* v_x_2449_){
_start:
{
uint64_t v___y_2451_; 
if (lean_obj_tag(v_x_2448_) == 0)
{
uint64_t v___x_2455_; 
v___x_2455_ = 1723ULL;
v___y_2451_ = v___x_2455_;
goto v___jp_2450_;
}
else
{
uint64_t v_hash_2456_; 
v_hash_2456_ = lean_ctor_get_uint64(v_x_2448_, sizeof(void*)*2);
v___y_2451_ = v_hash_2456_;
goto v___jp_2450_;
}
v___jp_2450_:
{
size_t v___x_2452_; size_t v___x_2453_; lean_object* v___x_2454_; 
v___x_2452_ = lean_uint64_to_usize(v___y_2451_);
v___x_2453_ = ((size_t)1ULL);
v___x_2454_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(v_x_2447_, v___x_2452_, v___x_2453_, v_x_2448_, v_x_2449_);
return v___x_2454_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_eraseCore(lean_object* v_d_2457_, lean_object* v_declName_2458_){
_start:
{
lean_object* v_tree_2459_; lean_object* v_erased_2460_; lean_object* v___x_2462_; uint8_t v_isShared_2463_; uint8_t v_isSharedCheck_2469_; 
v_tree_2459_ = lean_ctor_get(v_d_2457_, 0);
v_erased_2460_ = lean_ctor_get(v_d_2457_, 1);
v_isSharedCheck_2469_ = !lean_is_exclusive(v_d_2457_);
if (v_isSharedCheck_2469_ == 0)
{
v___x_2462_ = v_d_2457_;
v_isShared_2463_ = v_isSharedCheck_2469_;
goto v_resetjp_2461_;
}
else
{
lean_inc(v_erased_2460_);
lean_inc(v_tree_2459_);
lean_dec(v_d_2457_);
v___x_2462_ = lean_box(0);
v_isShared_2463_ = v_isSharedCheck_2469_;
goto v_resetjp_2461_;
}
v_resetjp_2461_:
{
lean_object* v___x_2464_; lean_object* v___x_2465_; lean_object* v___x_2467_; 
v___x_2464_ = lean_box(0);
v___x_2465_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0___redArg(v_erased_2460_, v_declName_2458_, v___x_2464_);
if (v_isShared_2463_ == 0)
{
lean_ctor_set(v___x_2462_, 1, v___x_2465_);
v___x_2467_ = v___x_2462_;
goto v_reusejp_2466_;
}
else
{
lean_object* v_reuseFailAlloc_2468_; 
v_reuseFailAlloc_2468_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2468_, 0, v_tree_2459_);
lean_ctor_set(v_reuseFailAlloc_2468_, 1, v___x_2465_);
v___x_2467_ = v_reuseFailAlloc_2468_;
goto v_reusejp_2466_;
}
v_reusejp_2466_:
{
return v___x_2467_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0(lean_object* v_00_u03b2_2470_, lean_object* v_x_2471_, lean_object* v_x_2472_, lean_object* v_x_2473_){
_start:
{
lean_object* v___x_2474_; 
v___x_2474_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0___redArg(v_x_2471_, v_x_2472_, v_x_2473_);
return v___x_2474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0(lean_object* v_00_u03b2_2475_, lean_object* v_x_2476_, size_t v_x_2477_, size_t v_x_2478_, lean_object* v_x_2479_, lean_object* v_x_2480_){
_start:
{
lean_object* v___x_2481_; 
v___x_2481_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___redArg(v_x_2476_, v_x_2477_, v_x_2478_, v_x_2479_, v_x_2480_);
return v___x_2481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0___boxed(lean_object* v_00_u03b2_2482_, lean_object* v_x_2483_, lean_object* v_x_2484_, lean_object* v_x_2485_, lean_object* v_x_2486_, lean_object* v_x_2487_){
_start:
{
size_t v_x_560__boxed_2488_; size_t v_x_561__boxed_2489_; lean_object* v_res_2490_; 
v_x_560__boxed_2488_ = lean_unbox_usize(v_x_2484_);
lean_dec(v_x_2484_);
v_x_561__boxed_2489_ = lean_unbox_usize(v_x_2485_);
lean_dec(v_x_2485_);
v_res_2490_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0(v_00_u03b2_2482_, v_x_2483_, v_x_560__boxed_2488_, v_x_561__boxed_2489_, v_x_2486_, v_x_2487_);
return v_res_2490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_2491_, lean_object* v_n_2492_, lean_object* v_k_2493_, lean_object* v_v_2494_){
_start:
{
lean_object* v___x_2495_; 
v___x_2495_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1___redArg(v_n_2492_, v_k_2493_, v_v_2494_);
return v___x_2495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_2496_, size_t v_depth_2497_, lean_object* v_keys_2498_, lean_object* v_vals_2499_, lean_object* v_heq_2500_, lean_object* v_i_2501_, lean_object* v_entries_2502_){
_start:
{
lean_object* v___x_2503_; 
v___x_2503_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___redArg(v_depth_2497_, v_keys_2498_, v_vals_2499_, v_i_2501_, v_entries_2502_);
return v___x_2503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_2504_, lean_object* v_depth_2505_, lean_object* v_keys_2506_, lean_object* v_vals_2507_, lean_object* v_heq_2508_, lean_object* v_i_2509_, lean_object* v_entries_2510_){
_start:
{
size_t v_depth_boxed_2511_; lean_object* v_res_2512_; 
v_depth_boxed_2511_ = lean_unbox_usize(v_depth_2505_);
lean_dec(v_depth_2505_);
v_res_2512_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__2(v_00_u03b2_2504_, v_depth_boxed_2511_, v_keys_2506_, v_vals_2507_, v_heq_2508_, v_i_2509_, v_entries_2510_);
lean_dec_ref(v_vals_2507_);
lean_dec_ref(v_keys_2506_);
return v_res_2512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_2513_, lean_object* v_x_2514_, lean_object* v_x_2515_, lean_object* v_x_2516_, lean_object* v_x_2517_){
_start:
{
lean_object* v___x_2518_; 
v___x_2518_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Meta_NormNum_NormNums_eraseCore_spec__0_spec__0_spec__1_spec__2___redArg(v_x_2514_, v_x_2515_, v_x_2516_, v_x_2517_);
return v___x_2518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__0(lean_object* v_x1_2519_, lean_object* v_x2_2520_){
_start:
{
lean_object* v___x_2521_; 
v___x_2521_ = lean_array_push(v_x1_2519_, v_x2_2520_);
return v___x_2521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__1(lean_object* v_d_2522_, lean_object* v_declName_2523_, lean_object* v_toPure_2524_, lean_object* v_____r_2525_){
_start:
{
lean_object* v___x_2526_; lean_object* v___x_2527_; 
v___x_2526_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_eraseCore(v_d_2522_, v_declName_2523_);
v___x_2527_ = lean_apply_2(v_toPure_2524_, lean_box(0), v___x_2526_);
return v___x_2527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__2(lean_object* v___f_2528_, lean_object* v_____r_2529_){
_start:
{
lean_object* v___x_2530_; 
v___x_2530_ = lean_apply_1(v___f_2528_, v_____r_2529_);
return v___x_2530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3(lean_object* v___x_2531_, lean_object* v___f_2532_, lean_object* v_s_2533_, lean_object* v_x_2534_, lean_object* v_t_2535_){
_start:
{
lean_object* v___x_2536_; 
v___x_2536_ = l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(v___x_2531_, v___f_2532_, v_s_2533_, v_t_2535_);
return v___x_2536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3___boxed(lean_object* v___x_2537_, lean_object* v___f_2538_, lean_object* v_s_2539_, lean_object* v_x_2540_, lean_object* v_t_2541_){
_start:
{
lean_object* v_res_2542_; 
v_res_2542_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__3(v___x_2537_, v___f_2538_, v_s_2539_, v_x_2540_, v_t_2541_);
lean_dec(v_x_2540_);
return v_res_2542_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4(lean_object* v_declName_2543_, lean_object* v_x_2544_){
_start:
{
lean_object* v_name_2545_; uint8_t v___x_2546_; 
v_name_2545_ = lean_ctor_get(v_x_2544_, 1);
v___x_2546_ = lean_name_eq(v_name_2545_, v_declName_2543_);
return v___x_2546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4___boxed(lean_object* v_declName_2547_, lean_object* v_x_2548_){
_start:
{
uint8_t v_res_2549_; lean_object* v_r_2550_; 
v_res_2549_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4(v_declName_2547_, v_x_2548_);
lean_dec_ref(v_x_2548_);
lean_dec(v_declName_2547_);
v_r_2550_ = lean_box(v_res_2549_);
return v_r_2550_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2(void){
_start:
{
lean_object* v___x_2553_; lean_object* v___x_2554_; 
v___x_2553_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__1));
v___x_2554_ = l_Lean_stringToMessageData(v___x_2553_);
return v___x_2554_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4(void){
_start:
{
lean_object* v___x_2556_; lean_object* v___x_2557_; 
v___x_2556_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__3));
v___x_2557_ = l_Lean_stringToMessageData(v___x_2556_);
return v___x_2557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg(lean_object* v_inst_2582_, lean_object* v_inst_2583_, lean_object* v_d_2584_, lean_object* v_declName_2585_){
_start:
{
lean_object* v_toApplicative_2586_; lean_object* v_toBind_2587_; lean_object* v_toPure_2588_; lean_object* v_tree_2589_; lean_object* v_erased_2590_; lean_object* v___f_2591_; lean_object* v___f_2592_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___f_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; uint8_t v___x_2607_; 
v_toApplicative_2586_ = lean_ctor_get(v_inst_2582_, 0);
v_toBind_2587_ = lean_ctor_get(v_inst_2582_, 1);
lean_inc(v_toBind_2587_);
v_toPure_2588_ = lean_ctor_get(v_toApplicative_2586_, 1);
v_tree_2589_ = lean_ctor_get(v_d_2584_, 0);
v_erased_2590_ = lean_ctor_get(v_d_2584_, 1);
lean_inc(v_toPure_2588_);
lean_inc(v_declName_2585_);
lean_inc_ref(v_d_2584_);
v___f_2591_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2591_, 0, v_d_2584_);
lean_closure_set(v___f_2591_, 1, v_declName_2585_);
lean_closure_set(v___f_2591_, 2, v_toPure_2588_);
v___f_2592_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__2), 2, 1);
lean_closure_set(v___f_2592_, 0, v___f_2591_);
v___x_2601_ = lean_unsigned_to_nat(0u);
v___x_2602_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0));
v___x_2603_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__14));
v___f_2604_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__15));
lean_inc_ref(v_tree_2589_);
v___x_2605_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_2603_, v___f_2604_, v_tree_2589_, v___x_2602_);
v___x_2606_ = lean_array_get_size(v___x_2605_);
v___x_2607_ = lean_nat_dec_lt(v___x_2601_, v___x_2606_);
if (v___x_2607_ == 0)
{
lean_dec(v___x_2605_);
lean_dec_ref(v_d_2584_);
goto v___jp_2593_;
}
else
{
if (v___x_2607_ == 0)
{
lean_dec(v___x_2605_);
lean_dec_ref(v_d_2584_);
goto v___jp_2593_;
}
else
{
lean_object* v___f_2608_; size_t v___x_2609_; size_t v___x_2610_; lean_object* v___x_2611_; uint8_t v___x_2612_; 
lean_inc(v_declName_2585_);
v___f_2608_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_2608_, 0, v_declName_2585_);
v___x_2609_ = ((size_t)0ULL);
v___x_2610_ = lean_usize_of_nat(v___x_2606_);
v___x_2611_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_2603_, v___f_2608_, v___x_2605_, v___x_2609_, v___x_2610_);
v___x_2612_ = lean_unbox(v___x_2611_);
lean_dec(v___x_2611_);
if (v___x_2612_ == 0)
{
lean_dec_ref(v_d_2584_);
goto v___jp_2593_;
}
else
{
lean_object* v___x_2613_; lean_object* v___x_2614_; uint8_t v___x_2615_; 
v___x_2613_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__16));
v___x_2614_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__17));
lean_inc(v_declName_2585_);
lean_inc_ref(v_erased_2590_);
v___x_2615_ = l_Lean_PersistentHashMap_contains___redArg(v___x_2613_, v___x_2614_, v_erased_2590_, v_declName_2585_);
if (v___x_2615_ == 0)
{
lean_object* v___x_2616_; lean_object* v___x_2617_; 
lean_inc(v_toPure_2588_);
lean_dec_ref(v___f_2592_);
lean_dec(v_toBind_2587_);
lean_dec_ref(v_inst_2583_);
lean_dec_ref(v_inst_2582_);
v___x_2616_ = lean_box(0);
v___x_2617_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___lam__1(v_d_2584_, v_declName_2585_, v_toPure_2588_, v___x_2616_);
return v___x_2617_;
}
else
{
lean_dec_ref(v_d_2584_);
goto v___jp_2593_;
}
}
}
}
v___jp_2593_:
{
lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; 
v___x_2594_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2);
v___x_2595_ = l_Lean_MessageData_ofName(v_declName_2585_);
v___x_2596_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2596_, 0, v___x_2594_);
lean_ctor_set(v___x_2596_, 1, v___x_2595_);
v___x_2597_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4);
v___x_2598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2598_, 0, v___x_2596_);
lean_ctor_set(v___x_2598_, 1, v___x_2597_);
v___x_2599_ = l_Lean_throwError___redArg(v_inst_2582_, v_inst_2583_, v___x_2598_);
v___x_2600_ = lean_apply_4(v_toBind_2587_, lean_box(0), lean_box(0), v___x_2599_, v___f_2592_);
return v___x_2600_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase(lean_object* v_m_2618_, lean_object* v_inst_2619_, lean_object* v_inst_2620_, lean_object* v_d_2621_, lean_object* v_declName_2622_){
_start:
{
lean_object* v___x_2623_; 
v___x_2623_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg(v_inst_2619_, v_inst_2620_, v_d_2621_, v_declName_2622_);
return v___x_2623_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; 
v___x_2624_ = lean_box(0);
v___x_2625_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2626_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2626_, 0, v___x_2625_);
lean_ctor_set(v___x_2626_, 1, v___x_2624_);
return v___x_2626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg(){
_start:
{
lean_object* v___x_2628_; lean_object* v___x_2629_; 
v___x_2628_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0);
v___x_2629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2629_, 0, v___x_2628_);
return v___x_2629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v___y_2630_){
_start:
{
lean_object* v_res_2631_; 
v_res_2631_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg();
return v_res_2631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_2632_, lean_object* v___y_2633_, lean_object* v___y_2634_){
_start:
{
lean_object* v___x_2636_; 
v___x_2636_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg();
return v___x_2636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_){
_start:
{
lean_object* v_res_2641_; 
v_res_2641_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1(v_00_u03b1_2637_, v___y_2638_, v___y_2639_);
lean_dec(v___y_2639_);
lean_dec_ref(v___y_2638_);
return v_res_2641_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_2642_; 
v___x_2642_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2642_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_2643_; lean_object* v___x_2644_; 
v___x_2643_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__0);
v___x_2644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2644_, 0, v___x_2643_);
return v___x_2644_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_2645_; lean_object* v___x_2646_; 
v___x_2645_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__1);
v___x_2646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2646_, 0, v___x_2645_);
lean_ctor_set(v___x_2646_, 1, v___x_2645_);
return v___x_2646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg(lean_object* v_ext_2647_, lean_object* v_b_2648_, uint8_t v_kind_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_){
_start:
{
lean_object* v_currNamespace_2653_; lean_object* v___x_2654_; lean_object* v_env_2655_; lean_object* v_nextMacroScope_2656_; lean_object* v_ngen_2657_; lean_object* v_auxDeclNGen_2658_; lean_object* v_traceState_2659_; lean_object* v_messages_2660_; lean_object* v_infoState_2661_; lean_object* v_snapshotTasks_2662_; lean_object* v___x_2664_; uint8_t v_isShared_2665_; uint8_t v_isSharedCheck_2674_; 
v_currNamespace_2653_ = lean_ctor_get(v___y_2650_, 6);
v___x_2654_ = lean_st_ref_take(v___y_2651_);
v_env_2655_ = lean_ctor_get(v___x_2654_, 0);
v_nextMacroScope_2656_ = lean_ctor_get(v___x_2654_, 1);
v_ngen_2657_ = lean_ctor_get(v___x_2654_, 2);
v_auxDeclNGen_2658_ = lean_ctor_get(v___x_2654_, 3);
v_traceState_2659_ = lean_ctor_get(v___x_2654_, 4);
v_messages_2660_ = lean_ctor_get(v___x_2654_, 6);
v_infoState_2661_ = lean_ctor_get(v___x_2654_, 7);
v_snapshotTasks_2662_ = lean_ctor_get(v___x_2654_, 8);
v_isSharedCheck_2674_ = !lean_is_exclusive(v___x_2654_);
if (v_isSharedCheck_2674_ == 0)
{
lean_object* v_unused_2675_; 
v_unused_2675_ = lean_ctor_get(v___x_2654_, 5);
lean_dec(v_unused_2675_);
v___x_2664_ = v___x_2654_;
v_isShared_2665_ = v_isSharedCheck_2674_;
goto v_resetjp_2663_;
}
else
{
lean_inc(v_snapshotTasks_2662_);
lean_inc(v_infoState_2661_);
lean_inc(v_messages_2660_);
lean_inc(v_traceState_2659_);
lean_inc(v_auxDeclNGen_2658_);
lean_inc(v_ngen_2657_);
lean_inc(v_nextMacroScope_2656_);
lean_inc(v_env_2655_);
lean_dec(v___x_2654_);
v___x_2664_ = lean_box(0);
v_isShared_2665_ = v_isSharedCheck_2674_;
goto v_resetjp_2663_;
}
v_resetjp_2663_:
{
lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2669_; 
lean_inc(v_currNamespace_2653_);
v___x_2666_ = l_Lean_ScopedEnvExtension_addCore___redArg(v_env_2655_, v_ext_2647_, v_b_2648_, v_kind_2649_, v_currNamespace_2653_);
v___x_2667_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2);
if (v_isShared_2665_ == 0)
{
lean_ctor_set(v___x_2664_, 5, v___x_2667_);
lean_ctor_set(v___x_2664_, 0, v___x_2666_);
v___x_2669_ = v___x_2664_;
goto v_reusejp_2668_;
}
else
{
lean_object* v_reuseFailAlloc_2673_; 
v_reuseFailAlloc_2673_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2673_, 0, v___x_2666_);
lean_ctor_set(v_reuseFailAlloc_2673_, 1, v_nextMacroScope_2656_);
lean_ctor_set(v_reuseFailAlloc_2673_, 2, v_ngen_2657_);
lean_ctor_set(v_reuseFailAlloc_2673_, 3, v_auxDeclNGen_2658_);
lean_ctor_set(v_reuseFailAlloc_2673_, 4, v_traceState_2659_);
lean_ctor_set(v_reuseFailAlloc_2673_, 5, v___x_2667_);
lean_ctor_set(v_reuseFailAlloc_2673_, 6, v_messages_2660_);
lean_ctor_set(v_reuseFailAlloc_2673_, 7, v_infoState_2661_);
lean_ctor_set(v_reuseFailAlloc_2673_, 8, v_snapshotTasks_2662_);
v___x_2669_ = v_reuseFailAlloc_2673_;
goto v_reusejp_2668_;
}
v_reusejp_2668_:
{
lean_object* v___x_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; 
v___x_2670_ = lean_st_ref_set(v___y_2651_, v___x_2669_);
v___x_2671_ = lean_box(0);
v___x_2672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2672_, 0, v___x_2671_);
return v___x_2672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___boxed(lean_object* v_ext_2676_, lean_object* v_b_2677_, lean_object* v_kind_2678_, lean_object* v___y_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_){
_start:
{
uint8_t v_kind_boxed_2682_; lean_object* v_res_2683_; 
v_kind_boxed_2682_ = lean_unbox(v_kind_2678_);
v_res_2683_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg(v_ext_2676_, v_b_2677_, v_kind_boxed_2682_, v___y_2679_, v___y_2680_);
lean_dec(v___y_2680_);
lean_dec_ref(v___y_2679_);
return v_res_2683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3(lean_object* v_00_u03b1_2684_, lean_object* v_00_u03b2_2685_, lean_object* v_00_u03c3_2686_, lean_object* v_ext_2687_, lean_object* v_b_2688_, uint8_t v_kind_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_){
_start:
{
lean_object* v___x_2693_; 
v___x_2693_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg(v_ext_2687_, v_b_2688_, v_kind_2689_, v___y_2690_, v___y_2691_);
return v___x_2693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___boxed(lean_object* v_00_u03b1_2694_, lean_object* v_00_u03b2_2695_, lean_object* v_00_u03c3_2696_, lean_object* v_ext_2697_, lean_object* v_b_2698_, lean_object* v_kind_2699_, lean_object* v___y_2700_, lean_object* v___y_2701_, lean_object* v___y_2702_){
_start:
{
uint8_t v_kind_boxed_2703_; lean_object* v_res_2704_; 
v_kind_boxed_2703_ = lean_unbox(v_kind_2699_);
v_res_2704_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3(v_00_u03b1_2694_, v_00_u03b2_2695_, v_00_u03c3_2696_, v_ext_2697_, v_b_2698_, v_kind_boxed_2703_, v___y_2700_, v___y_2701_);
lean_dec(v___y_2701_);
lean_dec_ref(v___y_2700_);
return v_res_2704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object* v_a_2705_, lean_object* v_x_2706_){
_start:
{
lean_inc_ref(v_a_2705_);
return v_a_2705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object* v_a_2707_, lean_object* v_x_2708_){
_start:
{
lean_object* v_res_2709_; 
v_res_2709_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(v_a_2707_, v_x_2708_);
lean_dec_ref(v_x_2708_);
lean_dec_ref(v_a_2707_);
return v_res_2709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(lean_object* v_f_2710_, lean_object* v_as_2711_, size_t v_i_2712_, size_t v_stop_2713_, lean_object* v_b_2714_){
_start:
{
uint8_t v___x_2715_; 
v___x_2715_ = lean_usize_dec_eq(v_i_2712_, v_stop_2713_);
if (v___x_2715_ == 0)
{
lean_object* v___x_2716_; lean_object* v___x_2717_; size_t v___x_2718_; size_t v___x_2719_; 
v___x_2716_ = lean_array_uget_borrowed(v_as_2711_, v_i_2712_);
lean_inc(v_f_2710_);
lean_inc(v___x_2716_);
v___x_2717_ = lean_apply_2(v_f_2710_, v_b_2714_, v___x_2716_);
v___x_2718_ = ((size_t)1ULL);
v___x_2719_ = lean_usize_add(v_i_2712_, v___x_2718_);
v_i_2712_ = v___x_2719_;
v_b_2714_ = v___x_2717_;
goto _start;
}
else
{
lean_dec(v_f_2710_);
return v_b_2714_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_f_2721_, lean_object* v_as_2722_, lean_object* v_i_2723_, lean_object* v_stop_2724_, lean_object* v_b_2725_){
_start:
{
size_t v_i_boxed_2726_; size_t v_stop_boxed_2727_; lean_object* v_res_2728_; 
v_i_boxed_2726_ = lean_unbox_usize(v_i_2723_);
lean_dec(v_i_2723_);
v_stop_boxed_2727_ = lean_unbox_usize(v_stop_2724_);
lean_dec(v_stop_2724_);
v_res_2728_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(v_f_2721_, v_as_2722_, v_i_boxed_2726_, v_stop_boxed_2727_, v_b_2725_);
lean_dec_ref(v_as_2722_);
return v_res_2728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_f_2729_, lean_object* v_x_2730_, lean_object* v_x_2731_){
_start:
{
lean_object* v_vs_2732_; lean_object* v_children_2733_; lean_object* v___x_2734_; lean_object* v_s_2736_; lean_object* v___x_2746_; uint8_t v___x_2747_; 
v_vs_2732_ = lean_ctor_get(v_x_2731_, 0);
v_children_2733_ = lean_ctor_get(v_x_2731_, 1);
v___x_2734_ = lean_unsigned_to_nat(0u);
v___x_2746_ = lean_array_get_size(v_vs_2732_);
v___x_2747_ = lean_nat_dec_lt(v___x_2734_, v___x_2746_);
if (v___x_2747_ == 0)
{
lean_object* v___x_2748_; uint8_t v___x_2749_; 
v___x_2748_ = lean_array_get_size(v_children_2733_);
v___x_2749_ = lean_nat_dec_lt(v___x_2734_, v___x_2748_);
if (v___x_2749_ == 0)
{
lean_dec(v_f_2729_);
return v_x_2730_;
}
else
{
uint8_t v___x_2750_; 
v___x_2750_ = lean_nat_dec_le(v___x_2748_, v___x_2748_);
if (v___x_2750_ == 0)
{
if (v___x_2749_ == 0)
{
lean_dec(v_f_2729_);
return v_x_2730_;
}
else
{
size_t v___x_2751_; size_t v___x_2752_; lean_object* v___x_2753_; 
v___x_2751_ = ((size_t)0ULL);
v___x_2752_ = lean_usize_of_nat(v___x_2748_);
v___x_2753_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_2729_, v_children_2733_, v___x_2751_, v___x_2752_, v_x_2730_);
return v___x_2753_;
}
}
else
{
size_t v___x_2754_; size_t v___x_2755_; lean_object* v___x_2756_; 
v___x_2754_ = ((size_t)0ULL);
v___x_2755_ = lean_usize_of_nat(v___x_2748_);
v___x_2756_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_2729_, v_children_2733_, v___x_2754_, v___x_2755_, v_x_2730_);
return v___x_2756_;
}
}
}
else
{
uint8_t v___x_2757_; 
v___x_2757_ = lean_nat_dec_le(v___x_2746_, v___x_2746_);
if (v___x_2757_ == 0)
{
if (v___x_2747_ == 0)
{
v_s_2736_ = v_x_2730_;
goto v___jp_2735_;
}
else
{
size_t v___x_2758_; size_t v___x_2759_; lean_object* v___x_2760_; 
v___x_2758_ = ((size_t)0ULL);
v___x_2759_ = lean_usize_of_nat(v___x_2746_);
lean_inc(v_f_2729_);
v___x_2760_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(v_f_2729_, v_vs_2732_, v___x_2758_, v___x_2759_, v_x_2730_);
v_s_2736_ = v___x_2760_;
goto v___jp_2735_;
}
}
else
{
size_t v___x_2761_; size_t v___x_2762_; lean_object* v___x_2763_; 
v___x_2761_ = ((size_t)0ULL);
v___x_2762_ = lean_usize_of_nat(v___x_2746_);
lean_inc(v_f_2729_);
v___x_2763_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(v_f_2729_, v_vs_2732_, v___x_2761_, v___x_2762_, v_x_2730_);
v_s_2736_ = v___x_2763_;
goto v___jp_2735_;
}
}
v___jp_2735_:
{
lean_object* v___x_2737_; uint8_t v___x_2738_; 
v___x_2737_ = lean_array_get_size(v_children_2733_);
v___x_2738_ = lean_nat_dec_lt(v___x_2734_, v___x_2737_);
if (v___x_2738_ == 0)
{
lean_dec(v_f_2729_);
return v_s_2736_;
}
else
{
uint8_t v___x_2739_; 
v___x_2739_ = lean_nat_dec_le(v___x_2737_, v___x_2737_);
if (v___x_2739_ == 0)
{
if (v___x_2738_ == 0)
{
lean_dec(v_f_2729_);
return v_s_2736_;
}
else
{
size_t v___x_2740_; size_t v___x_2741_; lean_object* v___x_2742_; 
v___x_2740_ = ((size_t)0ULL);
v___x_2741_ = lean_usize_of_nat(v___x_2737_);
v___x_2742_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_2729_, v_children_2733_, v___x_2740_, v___x_2741_, v_s_2736_);
return v___x_2742_;
}
}
else
{
size_t v___x_2743_; size_t v___x_2744_; lean_object* v___x_2745_; 
v___x_2743_ = ((size_t)0ULL);
v___x_2744_ = lean_usize_of_nat(v___x_2737_);
v___x_2745_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_2729_, v_children_2733_, v___x_2743_, v___x_2744_, v_s_2736_);
return v___x_2745_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(lean_object* v_f_2764_, lean_object* v_as_2765_, size_t v_i_2766_, size_t v_stop_2767_, lean_object* v_b_2768_){
_start:
{
uint8_t v___x_2769_; 
v___x_2769_ = lean_usize_dec_eq(v_i_2766_, v_stop_2767_);
if (v___x_2769_ == 0)
{
lean_object* v___x_2770_; lean_object* v_snd_2771_; lean_object* v___x_2772_; size_t v___x_2773_; size_t v___x_2774_; 
v___x_2770_ = lean_array_uget_borrowed(v_as_2765_, v_i_2766_);
v_snd_2771_ = lean_ctor_get(v___x_2770_, 1);
lean_inc(v_f_2764_);
v___x_2772_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(v_f_2764_, v_b_2768_, v_snd_2771_);
v___x_2773_ = ((size_t)1ULL);
v___x_2774_ = lean_usize_add(v_i_2766_, v___x_2773_);
v_i_2766_ = v___x_2774_;
v_b_2768_ = v___x_2772_;
goto _start;
}
else
{
lean_dec(v_f_2764_);
return v_b_2768_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_f_2776_, lean_object* v_as_2777_, lean_object* v_i_2778_, lean_object* v_stop_2779_, lean_object* v_b_2780_){
_start:
{
size_t v_i_boxed_2781_; size_t v_stop_boxed_2782_; lean_object* v_res_2783_; 
v_i_boxed_2781_ = lean_unbox_usize(v_i_2778_);
lean_dec(v_i_2778_);
v_stop_boxed_2782_ = lean_unbox_usize(v_stop_2779_);
lean_dec(v_stop_2779_);
v_res_2783_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_2776_, v_as_2777_, v_i_boxed_2781_, v_stop_boxed_2782_, v_b_2780_);
lean_dec_ref(v_as_2777_);
return v_res_2783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_f_2784_, lean_object* v_x_2785_, lean_object* v_x_2786_){
_start:
{
lean_object* v_res_2787_; 
v_res_2787_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(v_f_2784_, v_x_2785_, v_x_2786_);
lean_dec_ref(v_x_2786_);
return v_res_2787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1(lean_object* v___f_2788_, lean_object* v_s_2789_, lean_object* v_x_2790_, lean_object* v_t_2791_){
_start:
{
lean_object* v___x_2792_; 
v___x_2792_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(v___f_2788_, v_s_2789_, v_t_2791_);
return v___x_2792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1___boxed(lean_object* v___f_2793_, lean_object* v_s_2794_, lean_object* v_x_2795_, lean_object* v_t_2796_){
_start:
{
lean_object* v_res_2797_; 
v_res_2797_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___lam__1(v___f_2793_, v_s_2794_, v_x_2795_, v_t_2796_);
lean_dec_ref(v_t_2796_);
lean_dec(v_x_2795_);
return v_res_2797_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2(lean_object* v_declName_2798_, lean_object* v_as_2799_, size_t v_i_2800_, size_t v_stop_2801_){
_start:
{
uint8_t v___x_2802_; 
v___x_2802_ = lean_usize_dec_eq(v_i_2800_, v_stop_2801_);
if (v___x_2802_ == 0)
{
lean_object* v___x_2803_; lean_object* v_name_2804_; uint8_t v___x_2805_; 
v___x_2803_ = lean_array_uget_borrowed(v_as_2799_, v_i_2800_);
v_name_2804_ = lean_ctor_get(v___x_2803_, 1);
v___x_2805_ = lean_name_eq(v_name_2804_, v_declName_2798_);
if (v___x_2805_ == 0)
{
size_t v___x_2806_; size_t v___x_2807_; 
v___x_2806_ = ((size_t)1ULL);
v___x_2807_ = lean_usize_add(v_i_2800_, v___x_2806_);
v_i_2800_ = v___x_2807_;
goto _start;
}
else
{
return v___x_2805_;
}
}
else
{
uint8_t v___x_2809_; 
v___x_2809_ = 0;
return v___x_2809_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object* v_declName_2810_, lean_object* v_as_2811_, lean_object* v_i_2812_, lean_object* v_stop_2813_){
_start:
{
size_t v_i_boxed_2814_; size_t v_stop_boxed_2815_; uint8_t v_res_2816_; lean_object* v_r_2817_; 
v_i_boxed_2814_ = lean_unbox_usize(v_i_2812_);
lean_dec(v_i_2812_);
v_stop_boxed_2815_ = lean_unbox_usize(v_stop_2813_);
lean_dec(v_stop_2813_);
v_res_2816_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2(v_declName_2810_, v_as_2811_, v_i_boxed_2814_, v_stop_boxed_2815_);
lean_dec_ref(v_as_2811_);
lean_dec(v_declName_2810_);
v_r_2817_ = lean_box(v_res_2816_);
return v_r_2817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg(lean_object* v_f_2818_, lean_object* v_keys_2819_, lean_object* v_vals_2820_, lean_object* v_i_2821_, lean_object* v_acc_2822_){
_start:
{
lean_object* v___x_2823_; uint8_t v___x_2824_; 
v___x_2823_ = lean_array_get_size(v_keys_2819_);
v___x_2824_ = lean_nat_dec_lt(v_i_2821_, v___x_2823_);
if (v___x_2824_ == 0)
{
lean_dec(v_i_2821_);
lean_dec(v_f_2818_);
return v_acc_2822_;
}
else
{
lean_object* v_k_2825_; lean_object* v_v_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; 
v_k_2825_ = lean_array_fget_borrowed(v_keys_2819_, v_i_2821_);
v_v_2826_ = lean_array_fget_borrowed(v_vals_2820_, v_i_2821_);
lean_inc(v_f_2818_);
lean_inc(v_v_2826_);
lean_inc(v_k_2825_);
v___x_2827_ = lean_apply_3(v_f_2818_, v_acc_2822_, v_k_2825_, v_v_2826_);
v___x_2828_ = lean_unsigned_to_nat(1u);
v___x_2829_ = lean_nat_add(v_i_2821_, v___x_2828_);
lean_dec(v_i_2821_);
v_i_2821_ = v___x_2829_;
v_acc_2822_ = v___x_2827_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg___boxed(lean_object* v_f_2831_, lean_object* v_keys_2832_, lean_object* v_vals_2833_, lean_object* v_i_2834_, lean_object* v_acc_2835_){
_start:
{
lean_object* v_res_2836_; 
v_res_2836_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg(v_f_2831_, v_keys_2832_, v_vals_2833_, v_i_2834_, v_acc_2835_);
lean_dec_ref(v_vals_2833_);
lean_dec_ref(v_keys_2832_);
return v_res_2836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(lean_object* v_f_2837_, lean_object* v_x_2838_, lean_object* v_x_2839_){
_start:
{
if (lean_obj_tag(v_x_2838_) == 0)
{
lean_object* v_es_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; uint8_t v___x_2843_; 
v_es_2840_ = lean_ctor_get(v_x_2838_, 0);
v___x_2841_ = lean_unsigned_to_nat(0u);
v___x_2842_ = lean_array_get_size(v_es_2840_);
v___x_2843_ = lean_nat_dec_lt(v___x_2841_, v___x_2842_);
if (v___x_2843_ == 0)
{
lean_dec(v_f_2837_);
return v_x_2839_;
}
else
{
uint8_t v___x_2844_; 
v___x_2844_ = lean_nat_dec_le(v___x_2842_, v___x_2842_);
if (v___x_2844_ == 0)
{
if (v___x_2843_ == 0)
{
lean_dec(v_f_2837_);
return v_x_2839_;
}
else
{
size_t v___x_2845_; size_t v___x_2846_; lean_object* v___x_2847_; 
v___x_2845_ = ((size_t)0ULL);
v___x_2846_ = lean_usize_of_nat(v___x_2842_);
v___x_2847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(v_f_2837_, v_es_2840_, v___x_2845_, v___x_2846_, v_x_2839_);
return v___x_2847_;
}
}
else
{
size_t v___x_2848_; size_t v___x_2849_; lean_object* v___x_2850_; 
v___x_2848_ = ((size_t)0ULL);
v___x_2849_ = lean_usize_of_nat(v___x_2842_);
v___x_2850_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(v_f_2837_, v_es_2840_, v___x_2848_, v___x_2849_, v_x_2839_);
return v___x_2850_;
}
}
}
else
{
lean_object* v_ks_2851_; lean_object* v_vs_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; 
v_ks_2851_ = lean_ctor_get(v_x_2838_, 0);
v_vs_2852_ = lean_ctor_get(v_x_2838_, 1);
v___x_2853_ = lean_unsigned_to_nat(0u);
v___x_2854_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg(v_f_2837_, v_ks_2851_, v_vs_2852_, v___x_2853_, v_x_2839_);
return v___x_2854_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(lean_object* v_f_2855_, lean_object* v_as_2856_, size_t v_i_2857_, size_t v_stop_2858_, lean_object* v_b_2859_){
_start:
{
lean_object* v___y_2861_; uint8_t v___x_2865_; 
v___x_2865_ = lean_usize_dec_eq(v_i_2857_, v_stop_2858_);
if (v___x_2865_ == 0)
{
lean_object* v___x_2866_; 
v___x_2866_ = lean_array_uget_borrowed(v_as_2856_, v_i_2857_);
switch(lean_obj_tag(v___x_2866_))
{
case 0:
{
lean_object* v_key_2867_; lean_object* v_val_2868_; lean_object* v___x_2869_; 
v_key_2867_ = lean_ctor_get(v___x_2866_, 0);
v_val_2868_ = lean_ctor_get(v___x_2866_, 1);
lean_inc(v_f_2855_);
lean_inc(v_val_2868_);
lean_inc(v_key_2867_);
v___x_2869_ = lean_apply_3(v_f_2855_, v_b_2859_, v_key_2867_, v_val_2868_);
v___y_2861_ = v___x_2869_;
goto v___jp_2860_;
}
case 1:
{
lean_object* v_node_2870_; lean_object* v___x_2871_; 
v_node_2870_ = lean_ctor_get(v___x_2866_, 0);
lean_inc(v_f_2855_);
v___x_2871_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v_f_2855_, v_node_2870_, v_b_2859_);
v___y_2861_ = v___x_2871_;
goto v___jp_2860_;
}
default: 
{
v___y_2861_ = v_b_2859_;
goto v___jp_2860_;
}
}
}
else
{
lean_dec(v_f_2855_);
return v_b_2859_;
}
v___jp_2860_:
{
size_t v___x_2862_; size_t v___x_2863_; 
v___x_2862_ = ((size_t)1ULL);
v___x_2863_ = lean_usize_add(v_i_2857_, v___x_2862_);
v_i_2857_ = v___x_2863_;
v_b_2859_ = v___y_2861_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg___boxed(lean_object* v_f_2872_, lean_object* v_as_2873_, lean_object* v_i_2874_, lean_object* v_stop_2875_, lean_object* v_b_2876_){
_start:
{
size_t v_i_boxed_2877_; size_t v_stop_boxed_2878_; lean_object* v_res_2879_; 
v_i_boxed_2877_ = lean_unbox_usize(v_i_2874_);
lean_dec(v_i_2874_);
v_stop_boxed_2878_ = lean_unbox_usize(v_stop_2875_);
lean_dec(v_stop_2875_);
v_res_2879_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(v_f_2872_, v_as_2873_, v_i_boxed_2877_, v_stop_boxed_2878_, v_b_2876_);
lean_dec_ref(v_as_2873_);
return v_res_2879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg___boxed(lean_object* v_f_2880_, lean_object* v_x_2881_, lean_object* v_x_2882_){
_start:
{
lean_object* v_res_2883_; 
v_res_2883_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v_f_2880_, v_x_2881_, v_x_2882_);
lean_dec_ref(v_x_2881_);
return v_res_2883_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0(void){
_start:
{
lean_object* v___x_2884_; 
v___x_2884_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2884_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1(void){
_start:
{
lean_object* v___x_2885_; lean_object* v___x_2886_; 
v___x_2885_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__0);
v___x_2886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2886_, 0, v___x_2885_);
return v___x_2886_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2(void){
_start:
{
lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; 
v___x_2887_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1);
v___x_2888_ = lean_unsigned_to_nat(0u);
v___x_2889_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2889_, 0, v___x_2888_);
lean_ctor_set(v___x_2889_, 1, v___x_2888_);
lean_ctor_set(v___x_2889_, 2, v___x_2888_);
lean_ctor_set(v___x_2889_, 3, v___x_2888_);
lean_ctor_set(v___x_2889_, 4, v___x_2887_);
lean_ctor_set(v___x_2889_, 5, v___x_2887_);
lean_ctor_set(v___x_2889_, 6, v___x_2887_);
lean_ctor_set(v___x_2889_, 7, v___x_2887_);
lean_ctor_set(v___x_2889_, 8, v___x_2887_);
lean_ctor_set(v___x_2889_, 9, v___x_2887_);
return v___x_2889_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3(void){
_start:
{
lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; 
v___x_2890_ = lean_unsigned_to_nat(32u);
v___x_2891_ = lean_mk_empty_array_with_capacity(v___x_2890_);
v___x_2892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2892_, 0, v___x_2891_);
return v___x_2892_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4(void){
_start:
{
size_t v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; 
v___x_2893_ = ((size_t)5ULL);
v___x_2894_ = lean_unsigned_to_nat(0u);
v___x_2895_ = lean_unsigned_to_nat(32u);
v___x_2896_ = lean_mk_empty_array_with_capacity(v___x_2895_);
v___x_2897_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3);
v___x_2898_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2898_, 0, v___x_2897_);
lean_ctor_set(v___x_2898_, 1, v___x_2896_);
lean_ctor_set(v___x_2898_, 2, v___x_2894_);
lean_ctor_set(v___x_2898_, 3, v___x_2894_);
lean_ctor_set_usize(v___x_2898_, 4, v___x_2893_);
return v___x_2898_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5(void){
_start:
{
lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2902_; 
v___x_2899_ = lean_box(1);
v___x_2900_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__4);
v___x_2901_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__1);
v___x_2902_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2902_, 0, v___x_2901_);
lean_ctor_set(v___x_2902_, 1, v___x_2900_);
lean_ctor_set(v___x_2902_, 2, v___x_2899_);
return v___x_2902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12(lean_object* v_msgData_2903_, lean_object* v___y_2904_, lean_object* v___y_2905_){
_start:
{
lean_object* v___x_2907_; lean_object* v_env_2908_; lean_object* v_options_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; lean_object* v___x_2912_; lean_object* v___x_2913_; lean_object* v___x_2914_; 
v___x_2907_ = lean_st_ref_get(v___y_2905_);
v_env_2908_ = lean_ctor_get(v___x_2907_, 0);
lean_inc_ref(v_env_2908_);
lean_dec(v___x_2907_);
v_options_2909_ = lean_ctor_get(v___y_2904_, 2);
v___x_2910_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__2);
v___x_2911_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__5);
lean_inc_ref(v_options_2909_);
v___x_2912_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2912_, 0, v_env_2908_);
lean_ctor_set(v___x_2912_, 1, v___x_2910_);
lean_ctor_set(v___x_2912_, 2, v___x_2911_);
lean_ctor_set(v___x_2912_, 3, v_options_2909_);
v___x_2913_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2913_, 0, v___x_2912_);
lean_ctor_set(v___x_2913_, 1, v_msgData_2903_);
v___x_2914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2914_, 0, v___x_2913_);
return v___x_2914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___boxed(lean_object* v_msgData_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_){
_start:
{
lean_object* v_res_2919_; 
v_res_2919_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12(v_msgData_2915_, v___y_2916_, v___y_2917_);
lean_dec(v___y_2917_);
lean_dec_ref(v___y_2916_);
return v_res_2919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(lean_object* v_msg_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_){
_start:
{
lean_object* v_ref_2924_; lean_object* v___x_2925_; lean_object* v_a_2926_; lean_object* v___x_2928_; uint8_t v_isShared_2929_; uint8_t v_isSharedCheck_2934_; 
v_ref_2924_ = lean_ctor_get(v___y_2921_, 5);
v___x_2925_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12(v_msg_2920_, v___y_2921_, v___y_2922_);
v_a_2926_ = lean_ctor_get(v___x_2925_, 0);
v_isSharedCheck_2934_ = !lean_is_exclusive(v___x_2925_);
if (v_isSharedCheck_2934_ == 0)
{
v___x_2928_ = v___x_2925_;
v_isShared_2929_ = v_isSharedCheck_2934_;
goto v_resetjp_2927_;
}
else
{
lean_inc(v_a_2926_);
lean_dec(v___x_2925_);
v___x_2928_ = lean_box(0);
v_isShared_2929_ = v_isSharedCheck_2934_;
goto v_resetjp_2927_;
}
v_resetjp_2927_:
{
lean_object* v___x_2930_; lean_object* v___x_2932_; 
lean_inc(v_ref_2924_);
v___x_2930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2930_, 0, v_ref_2924_);
lean_ctor_set(v___x_2930_, 1, v_a_2926_);
if (v_isShared_2929_ == 0)
{
lean_ctor_set_tag(v___x_2928_, 1);
lean_ctor_set(v___x_2928_, 0, v___x_2930_);
v___x_2932_ = v___x_2928_;
goto v_reusejp_2931_;
}
else
{
lean_object* v_reuseFailAlloc_2933_; 
v_reuseFailAlloc_2933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2933_, 0, v___x_2930_);
v___x_2932_ = v_reuseFailAlloc_2933_;
goto v_reusejp_2931_;
}
v_reusejp_2931_:
{
return v___x_2932_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg___boxed(lean_object* v_msg_2935_, lean_object* v___y_2936_, lean_object* v___y_2937_, lean_object* v___y_2938_){
_start:
{
lean_object* v_res_2939_; 
v_res_2939_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(v_msg_2935_, v___y_2936_, v___y_2937_);
lean_dec(v___y_2937_);
lean_dec_ref(v___y_2936_);
return v_res_2939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0(lean_object* v_d_2942_, lean_object* v_declName_2943_, lean_object* v___y_2944_, lean_object* v___y_2945_){
_start:
{
lean_object* v_tree_2965_; lean_object* v_erased_2966_; lean_object* v___f_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; uint8_t v___x_2972_; 
v_tree_2965_ = lean_ctor_get(v_d_2942_, 0);
v_erased_2966_ = lean_ctor_get(v_d_2942_, 1);
v___f_2967_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___closed__0));
v___x_2968_ = lean_unsigned_to_nat(0u);
v___x_2969_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2__spec__0_spec__0___closed__0));
v___x_2970_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v___f_2967_, v_tree_2965_, v___x_2969_);
v___x_2971_ = lean_array_get_size(v___x_2970_);
v___x_2972_ = lean_nat_dec_lt(v___x_2968_, v___x_2971_);
if (v___x_2972_ == 0)
{
lean_dec(v___x_2970_);
lean_dec_ref(v_d_2942_);
goto v___jp_2950_;
}
else
{
if (v___x_2972_ == 0)
{
lean_dec(v___x_2970_);
lean_dec_ref(v_d_2942_);
goto v___jp_2950_;
}
else
{
size_t v___x_2973_; size_t v___x_2974_; uint8_t v___x_2975_; 
v___x_2973_ = ((size_t)0ULL);
v___x_2974_ = lean_usize_of_nat(v___x_2971_);
v___x_2975_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__2(v_declName_2943_, v___x_2970_, v___x_2973_, v___x_2974_);
lean_dec(v___x_2970_);
if (v___x_2975_ == 0)
{
lean_dec_ref(v_d_2942_);
goto v___jp_2950_;
}
else
{
uint8_t v___x_2976_; 
v___x_2976_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Meta_NormNum_derive_spec__1___redArg(v_erased_2966_, v_declName_2943_);
if (v___x_2976_ == 0)
{
goto v___jp_2947_;
}
else
{
lean_dec_ref(v_d_2942_);
goto v___jp_2950_;
}
}
}
}
v___jp_2947_:
{
lean_object* v___x_2948_; lean_object* v___x_2949_; 
v___x_2948_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_eraseCore(v_d_2942_, v_declName_2943_);
v___x_2949_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2949_, 0, v___x_2948_);
return v___x_2949_;
}
v___jp_2950_:
{
lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2953_; lean_object* v___x_2954_; lean_object* v___x_2955_; lean_object* v___x_2956_; lean_object* v_a_2957_; lean_object* v___x_2959_; uint8_t v_isShared_2960_; uint8_t v_isSharedCheck_2964_; 
v___x_2951_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__2);
v___x_2952_ = l_Lean_MessageData_ofName(v_declName_2943_);
v___x_2953_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2953_, 0, v___x_2951_);
lean_ctor_set(v___x_2953_, 1, v___x_2952_);
v___x_2954_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___redArg___closed__4);
v___x_2955_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2955_, 0, v___x_2953_);
lean_ctor_set(v___x_2955_, 1, v___x_2954_);
v___x_2956_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(v___x_2955_, v___y_2944_, v___y_2945_);
v_a_2957_ = lean_ctor_get(v___x_2956_, 0);
v_isSharedCheck_2964_ = !lean_is_exclusive(v___x_2956_);
if (v_isSharedCheck_2964_ == 0)
{
v___x_2959_ = v___x_2956_;
v_isShared_2960_ = v_isSharedCheck_2964_;
goto v_resetjp_2958_;
}
else
{
lean_inc(v_a_2957_);
lean_dec(v___x_2956_);
v___x_2959_ = lean_box(0);
v_isShared_2960_ = v_isSharedCheck_2964_;
goto v_resetjp_2958_;
}
v_resetjp_2958_:
{
lean_object* v___x_2962_; 
if (v_isShared_2960_ == 0)
{
v___x_2962_ = v___x_2959_;
goto v_reusejp_2961_;
}
else
{
lean_object* v_reuseFailAlloc_2963_; 
v_reuseFailAlloc_2963_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2963_, 0, v_a_2957_);
v___x_2962_ = v_reuseFailAlloc_2963_;
goto v_reusejp_2961_;
}
v_reusejp_2961_:
{
return v___x_2962_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0___boxed(lean_object* v_d_2977_, lean_object* v_declName_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_){
_start:
{
lean_object* v_res_2982_; 
v_res_2982_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0(v_d_2977_, v_declName_2978_, v___y_2979_, v___y_2980_);
lean_dec(v___y_2980_);
lean_dec_ref(v___y_2979_);
return v_res_2982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object* v___x_2983_, lean_object* v_declName_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_){
_start:
{
lean_object* v___x_2988_; lean_object* v_env_2989_; lean_object* v___x_2990_; lean_object* v_ext_2991_; lean_object* v_toEnvExtension_2992_; lean_object* v_asyncMode_2993_; lean_object* v___x_2994_; lean_object* v___x_2995_; 
v___x_2988_ = lean_st_ref_get(v___y_2986_);
v_env_2989_ = lean_ctor_get(v___x_2988_, 0);
lean_inc_ref(v_env_2989_);
lean_dec(v___x_2988_);
v___x_2990_ = lp_mathlib_Mathlib_Meta_NormNum_normNumExt;
v_ext_2991_ = lean_ctor_get(v___x_2990_, 1);
v_toEnvExtension_2992_ = lean_ctor_get(v_ext_2991_, 0);
v_asyncMode_2993_ = lean_ctor_get(v_toEnvExtension_2992_, 2);
v___x_2994_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_2983_, v___x_2990_, v_env_2989_, v_asyncMode_2993_);
v___x_2995_ = lp_mathlib_Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0(v___x_2994_, v_declName_2984_, v___y_2985_, v___y_2986_);
if (lean_obj_tag(v___x_2995_) == 0)
{
lean_object* v_a_2996_; lean_object* v___x_2998_; uint8_t v_isShared_2999_; uint8_t v_isSharedCheck_3025_; 
v_a_2996_ = lean_ctor_get(v___x_2995_, 0);
v_isSharedCheck_3025_ = !lean_is_exclusive(v___x_2995_);
if (v_isSharedCheck_3025_ == 0)
{
v___x_2998_ = v___x_2995_;
v_isShared_2999_ = v_isSharedCheck_3025_;
goto v_resetjp_2997_;
}
else
{
lean_inc(v_a_2996_);
lean_dec(v___x_2995_);
v___x_2998_ = lean_box(0);
v_isShared_2999_ = v_isSharedCheck_3025_;
goto v_resetjp_2997_;
}
v_resetjp_2997_:
{
lean_object* v___x_3000_; lean_object* v_env_3001_; lean_object* v_nextMacroScope_3002_; lean_object* v_ngen_3003_; lean_object* v_auxDeclNGen_3004_; lean_object* v_traceState_3005_; lean_object* v_messages_3006_; lean_object* v_infoState_3007_; lean_object* v_snapshotTasks_3008_; lean_object* v___x_3010_; uint8_t v_isShared_3011_; uint8_t v_isSharedCheck_3023_; 
v___x_3000_ = lean_st_ref_take(v___y_2986_);
v_env_3001_ = lean_ctor_get(v___x_3000_, 0);
v_nextMacroScope_3002_ = lean_ctor_get(v___x_3000_, 1);
v_ngen_3003_ = lean_ctor_get(v___x_3000_, 2);
v_auxDeclNGen_3004_ = lean_ctor_get(v___x_3000_, 3);
v_traceState_3005_ = lean_ctor_get(v___x_3000_, 4);
v_messages_3006_ = lean_ctor_get(v___x_3000_, 6);
v_infoState_3007_ = lean_ctor_get(v___x_3000_, 7);
v_snapshotTasks_3008_ = lean_ctor_get(v___x_3000_, 8);
v_isSharedCheck_3023_ = !lean_is_exclusive(v___x_3000_);
if (v_isSharedCheck_3023_ == 0)
{
lean_object* v_unused_3024_; 
v_unused_3024_ = lean_ctor_get(v___x_3000_, 5);
lean_dec(v_unused_3024_);
v___x_3010_ = v___x_3000_;
v_isShared_3011_ = v_isSharedCheck_3023_;
goto v_resetjp_3009_;
}
else
{
lean_inc(v_snapshotTasks_3008_);
lean_inc(v_infoState_3007_);
lean_inc(v_messages_3006_);
lean_inc(v_traceState_3005_);
lean_inc(v_auxDeclNGen_3004_);
lean_inc(v_ngen_3003_);
lean_inc(v_nextMacroScope_3002_);
lean_inc(v_env_3001_);
lean_dec(v___x_3000_);
v___x_3010_ = lean_box(0);
v_isShared_3011_ = v_isSharedCheck_3023_;
goto v_resetjp_3009_;
}
v_resetjp_3009_:
{
lean_object* v___f_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3016_; 
v___f_3012_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed), 2, 1);
lean_closure_set(v___f_3012_, 0, v_a_2996_);
v___x_3013_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v___x_2990_, v_env_3001_, v___f_3012_);
v___x_3014_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2);
if (v_isShared_3011_ == 0)
{
lean_ctor_set(v___x_3010_, 5, v___x_3014_);
lean_ctor_set(v___x_3010_, 0, v___x_3013_);
v___x_3016_ = v___x_3010_;
goto v_reusejp_3015_;
}
else
{
lean_object* v_reuseFailAlloc_3022_; 
v_reuseFailAlloc_3022_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3022_, 0, v___x_3013_);
lean_ctor_set(v_reuseFailAlloc_3022_, 1, v_nextMacroScope_3002_);
lean_ctor_set(v_reuseFailAlloc_3022_, 2, v_ngen_3003_);
lean_ctor_set(v_reuseFailAlloc_3022_, 3, v_auxDeclNGen_3004_);
lean_ctor_set(v_reuseFailAlloc_3022_, 4, v_traceState_3005_);
lean_ctor_set(v_reuseFailAlloc_3022_, 5, v___x_3014_);
lean_ctor_set(v_reuseFailAlloc_3022_, 6, v_messages_3006_);
lean_ctor_set(v_reuseFailAlloc_3022_, 7, v_infoState_3007_);
lean_ctor_set(v_reuseFailAlloc_3022_, 8, v_snapshotTasks_3008_);
v___x_3016_ = v_reuseFailAlloc_3022_;
goto v_reusejp_3015_;
}
v_reusejp_3015_:
{
lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3020_; 
v___x_3017_ = lean_st_ref_set(v___y_2986_, v___x_3016_);
v___x_3018_ = lean_box(0);
if (v_isShared_2999_ == 0)
{
lean_ctor_set(v___x_2998_, 0, v___x_3018_);
v___x_3020_ = v___x_2998_;
goto v_reusejp_3019_;
}
else
{
lean_object* v_reuseFailAlloc_3021_; 
v_reuseFailAlloc_3021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3021_, 0, v___x_3018_);
v___x_3020_ = v_reuseFailAlloc_3021_;
goto v_reusejp_3019_;
}
v_reusejp_3019_:
{
return v___x_3020_;
}
}
}
}
}
else
{
lean_object* v_a_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3033_; 
v_a_3026_ = lean_ctor_get(v___x_2995_, 0);
v_isSharedCheck_3033_ = !lean_is_exclusive(v___x_2995_);
if (v_isSharedCheck_3033_ == 0)
{
v___x_3028_ = v___x_2995_;
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_a_3026_);
lean_dec(v___x_2995_);
v___x_3028_ = lean_box(0);
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
v_resetjp_3027_:
{
lean_object* v___x_3031_; 
if (v_isShared_3029_ == 0)
{
v___x_3031_ = v___x_3028_;
goto v_reusejp_3030_;
}
else
{
lean_object* v_reuseFailAlloc_3032_; 
v_reuseFailAlloc_3032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3032_, 0, v_a_3026_);
v___x_3031_ = v_reuseFailAlloc_3032_;
goto v_reusejp_3030_;
}
v_reusejp_3030_:
{
return v___x_3031_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object* v___x_3034_, lean_object* v_declName_3035_, lean_object* v___y_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_){
_start:
{
lean_object* v_res_3039_; 
v_res_3039_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(v___x_3034_, v_declName_3035_, v___y_3036_, v___y_3037_);
lean_dec(v___y_3037_);
lean_dec_ref(v___y_3036_);
lean_dec_ref(v___x_3034_);
return v_res_3039_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0(void){
_start:
{
lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; 
v___x_3040_ = lean_unsigned_to_nat(32u);
v___x_3041_ = lean_mk_empty_array_with_capacity(v___x_3040_);
v___x_3042_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3042_, 0, v___x_3041_);
return v___x_3042_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1(void){
_start:
{
size_t v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; 
v___x_3043_ = ((size_t)5ULL);
v___x_3044_ = lean_unsigned_to_nat(0u);
v___x_3045_ = lean_unsigned_to_nat(32u);
v___x_3046_ = lean_mk_empty_array_with_capacity(v___x_3045_);
v___x_3047_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__0);
v___x_3048_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3048_, 0, v___x_3047_);
lean_ctor_set(v___x_3048_, 1, v___x_3046_);
lean_ctor_set(v___x_3048_, 2, v___x_3044_);
lean_ctor_set(v___x_3048_, 3, v___x_3044_);
lean_ctor_set_usize(v___x_3048_, 4, v___x_3043_);
return v___x_3048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg(lean_object* v___y_3049_){
_start:
{
lean_object* v___x_3051_; lean_object* v_infoState_3052_; lean_object* v_trees_3053_; lean_object* v___x_3054_; lean_object* v_infoState_3055_; lean_object* v_env_3056_; lean_object* v_nextMacroScope_3057_; lean_object* v_ngen_3058_; lean_object* v_auxDeclNGen_3059_; lean_object* v_traceState_3060_; lean_object* v_cache_3061_; lean_object* v_messages_3062_; lean_object* v_snapshotTasks_3063_; lean_object* v___x_3065_; uint8_t v_isShared_3066_; uint8_t v_isSharedCheck_3084_; 
v___x_3051_ = lean_st_ref_get(v___y_3049_);
v_infoState_3052_ = lean_ctor_get(v___x_3051_, 7);
lean_inc_ref(v_infoState_3052_);
lean_dec(v___x_3051_);
v_trees_3053_ = lean_ctor_get(v_infoState_3052_, 2);
lean_inc_ref(v_trees_3053_);
lean_dec_ref(v_infoState_3052_);
v___x_3054_ = lean_st_ref_take(v___y_3049_);
v_infoState_3055_ = lean_ctor_get(v___x_3054_, 7);
v_env_3056_ = lean_ctor_get(v___x_3054_, 0);
v_nextMacroScope_3057_ = lean_ctor_get(v___x_3054_, 1);
v_ngen_3058_ = lean_ctor_get(v___x_3054_, 2);
v_auxDeclNGen_3059_ = lean_ctor_get(v___x_3054_, 3);
v_traceState_3060_ = lean_ctor_get(v___x_3054_, 4);
v_cache_3061_ = lean_ctor_get(v___x_3054_, 5);
v_messages_3062_ = lean_ctor_get(v___x_3054_, 6);
v_snapshotTasks_3063_ = lean_ctor_get(v___x_3054_, 8);
v_isSharedCheck_3084_ = !lean_is_exclusive(v___x_3054_);
if (v_isSharedCheck_3084_ == 0)
{
v___x_3065_ = v___x_3054_;
v_isShared_3066_ = v_isSharedCheck_3084_;
goto v_resetjp_3064_;
}
else
{
lean_inc(v_snapshotTasks_3063_);
lean_inc(v_infoState_3055_);
lean_inc(v_messages_3062_);
lean_inc(v_cache_3061_);
lean_inc(v_traceState_3060_);
lean_inc(v_auxDeclNGen_3059_);
lean_inc(v_ngen_3058_);
lean_inc(v_nextMacroScope_3057_);
lean_inc(v_env_3056_);
lean_dec(v___x_3054_);
v___x_3065_ = lean_box(0);
v_isShared_3066_ = v_isSharedCheck_3084_;
goto v_resetjp_3064_;
}
v_resetjp_3064_:
{
uint8_t v_enabled_3067_; lean_object* v_assignment_3068_; lean_object* v_lazyAssignment_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3082_; 
v_enabled_3067_ = lean_ctor_get_uint8(v_infoState_3055_, sizeof(void*)*3);
v_assignment_3068_ = lean_ctor_get(v_infoState_3055_, 0);
v_lazyAssignment_3069_ = lean_ctor_get(v_infoState_3055_, 1);
v_isSharedCheck_3082_ = !lean_is_exclusive(v_infoState_3055_);
if (v_isSharedCheck_3082_ == 0)
{
lean_object* v_unused_3083_; 
v_unused_3083_ = lean_ctor_get(v_infoState_3055_, 2);
lean_dec(v_unused_3083_);
v___x_3071_ = v_infoState_3055_;
v_isShared_3072_ = v_isSharedCheck_3082_;
goto v_resetjp_3070_;
}
else
{
lean_inc(v_lazyAssignment_3069_);
lean_inc(v_assignment_3068_);
lean_dec(v_infoState_3055_);
v___x_3071_ = lean_box(0);
v_isShared_3072_ = v_isSharedCheck_3082_;
goto v_resetjp_3070_;
}
v_resetjp_3070_:
{
lean_object* v___x_3073_; lean_object* v___x_3075_; 
v___x_3073_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___closed__1);
if (v_isShared_3072_ == 0)
{
lean_ctor_set(v___x_3071_, 2, v___x_3073_);
v___x_3075_ = v___x_3071_;
goto v_reusejp_3074_;
}
else
{
lean_object* v_reuseFailAlloc_3081_; 
v_reuseFailAlloc_3081_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_3081_, 0, v_assignment_3068_);
lean_ctor_set(v_reuseFailAlloc_3081_, 1, v_lazyAssignment_3069_);
lean_ctor_set(v_reuseFailAlloc_3081_, 2, v___x_3073_);
lean_ctor_set_uint8(v_reuseFailAlloc_3081_, sizeof(void*)*3, v_enabled_3067_);
v___x_3075_ = v_reuseFailAlloc_3081_;
goto v_reusejp_3074_;
}
v_reusejp_3074_:
{
lean_object* v___x_3077_; 
if (v_isShared_3066_ == 0)
{
lean_ctor_set(v___x_3065_, 7, v___x_3075_);
v___x_3077_ = v___x_3065_;
goto v_reusejp_3076_;
}
else
{
lean_object* v_reuseFailAlloc_3080_; 
v_reuseFailAlloc_3080_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3080_, 0, v_env_3056_);
lean_ctor_set(v_reuseFailAlloc_3080_, 1, v_nextMacroScope_3057_);
lean_ctor_set(v_reuseFailAlloc_3080_, 2, v_ngen_3058_);
lean_ctor_set(v_reuseFailAlloc_3080_, 3, v_auxDeclNGen_3059_);
lean_ctor_set(v_reuseFailAlloc_3080_, 4, v_traceState_3060_);
lean_ctor_set(v_reuseFailAlloc_3080_, 5, v_cache_3061_);
lean_ctor_set(v_reuseFailAlloc_3080_, 6, v_messages_3062_);
lean_ctor_set(v_reuseFailAlloc_3080_, 7, v___x_3075_);
lean_ctor_set(v_reuseFailAlloc_3080_, 8, v_snapshotTasks_3063_);
v___x_3077_ = v_reuseFailAlloc_3080_;
goto v_reusejp_3076_;
}
v_reusejp_3076_:
{
lean_object* v___x_3078_; lean_object* v___x_3079_; 
v___x_3078_ = lean_st_ref_set(v___y_3049_, v___x_3077_);
v___x_3079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3079_, 0, v_trees_3053_);
return v___x_3079_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg___boxed(lean_object* v___y_3085_, lean_object* v___y_3086_){
_start:
{
lean_object* v_res_3087_; 
v_res_3087_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg(v___y_3085_);
lean_dec(v___y_3085_);
return v_res_3087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21(lean_object* v___x_3088_, lean_object* v_ctx_x3f_3089_, size_t v_sz_3090_, size_t v_i_3091_, lean_object* v_bs_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_){
_start:
{
uint8_t v___x_3100_; 
v___x_3100_ = lean_usize_dec_lt(v_i_3091_, v_sz_3090_);
if (v___x_3100_ == 0)
{
lean_object* v___x_3101_; 
lean_dec_ref(v_ctx_x3f_3089_);
v___x_3101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3101_, 0, v_bs_3092_);
return v___x_3101_;
}
else
{
lean_object* v_assignment_3102_; lean_object* v___x_3103_; 
v_assignment_3102_ = lean_ctor_get(v___x_3088_, 0);
lean_inc_ref(v_ctx_x3f_3089_);
lean_inc(v___y_3098_);
lean_inc_ref(v___y_3097_);
lean_inc(v___y_3096_);
lean_inc_ref(v___y_3095_);
lean_inc(v___y_3094_);
lean_inc_ref(v___y_3093_);
v___x_3103_ = lean_apply_7(v_ctx_x3f_3089_, v___y_3093_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, lean_box(0));
if (lean_obj_tag(v___x_3103_) == 0)
{
lean_object* v_a_3104_; lean_object* v_v_3105_; lean_object* v___x_3106_; lean_object* v_bs_x27_3107_; lean_object* v_a_3109_; lean_object* v_tree_3114_; 
v_a_3104_ = lean_ctor_get(v___x_3103_, 0);
lean_inc(v_a_3104_);
lean_dec_ref_known(v___x_3103_, 1);
v_v_3105_ = lean_array_uget(v_bs_3092_, v_i_3091_);
v___x_3106_ = lean_unsigned_to_nat(0u);
v_bs_x27_3107_ = lean_array_uset(v_bs_3092_, v_i_3091_, v___x_3106_);
v_tree_3114_ = l_Lean_Elab_InfoTree_substitute(v_v_3105_, v_assignment_3102_);
if (lean_obj_tag(v_a_3104_) == 0)
{
v_a_3109_ = v_tree_3114_;
goto v___jp_3108_;
}
else
{
lean_object* v_val_3115_; lean_object* v___x_3116_; 
v_val_3115_ = lean_ctor_get(v_a_3104_, 0);
lean_inc(v_val_3115_);
lean_dec_ref_known(v_a_3104_, 1);
v___x_3116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3116_, 0, v_val_3115_);
lean_ctor_set(v___x_3116_, 1, v_tree_3114_);
v_a_3109_ = v___x_3116_;
goto v___jp_3108_;
}
v___jp_3108_:
{
size_t v___x_3110_; size_t v___x_3111_; lean_object* v___x_3112_; 
v___x_3110_ = ((size_t)1ULL);
v___x_3111_ = lean_usize_add(v_i_3091_, v___x_3110_);
v___x_3112_ = lean_array_uset(v_bs_x27_3107_, v_i_3091_, v_a_3109_);
v_i_3091_ = v___x_3111_;
v_bs_3092_ = v___x_3112_;
goto _start;
}
}
else
{
lean_object* v_a_3117_; lean_object* v___x_3119_; uint8_t v_isShared_3120_; uint8_t v_isSharedCheck_3124_; 
lean_dec_ref(v_bs_3092_);
lean_dec_ref(v_ctx_x3f_3089_);
v_a_3117_ = lean_ctor_get(v___x_3103_, 0);
v_isSharedCheck_3124_ = !lean_is_exclusive(v___x_3103_);
if (v_isSharedCheck_3124_ == 0)
{
v___x_3119_ = v___x_3103_;
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
else
{
lean_inc(v_a_3117_);
lean_dec(v___x_3103_);
v___x_3119_ = lean_box(0);
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
v_resetjp_3118_:
{
lean_object* v___x_3122_; 
if (v_isShared_3120_ == 0)
{
v___x_3122_ = v___x_3119_;
goto v_reusejp_3121_;
}
else
{
lean_object* v_reuseFailAlloc_3123_; 
v_reuseFailAlloc_3123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3123_, 0, v_a_3117_);
v___x_3122_ = v_reuseFailAlloc_3123_;
goto v_reusejp_3121_;
}
v_reusejp_3121_:
{
return v___x_3122_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21___boxed(lean_object* v___x_3125_, lean_object* v_ctx_x3f_3126_, lean_object* v_sz_3127_, lean_object* v_i_3128_, lean_object* v_bs_3129_, lean_object* v___y_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_){
_start:
{
size_t v_sz_boxed_3137_; size_t v_i_boxed_3138_; lean_object* v_res_3139_; 
v_sz_boxed_3137_ = lean_unbox_usize(v_sz_3127_);
lean_dec(v_sz_3127_);
v_i_boxed_3138_ = lean_unbox_usize(v_i_3128_);
lean_dec(v_i_3128_);
v_res_3139_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21(v___x_3125_, v_ctx_x3f_3126_, v_sz_boxed_3137_, v_i_boxed_3138_, v_bs_3129_, v___y_3130_, v___y_3131_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_);
lean_dec(v___y_3135_);
lean_dec_ref(v___y_3134_);
lean_dec(v___y_3133_);
lean_dec_ref(v___y_3132_);
lean_dec(v___y_3131_);
lean_dec_ref(v___y_3130_);
lean_dec_ref(v___x_3125_);
return v_res_3139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20(lean_object* v___x_3140_, lean_object* v_ctx_x3f_3141_, lean_object* v_x_3142_, lean_object* v___y_3143_, lean_object* v___y_3144_, lean_object* v___y_3145_, lean_object* v___y_3146_, lean_object* v___y_3147_, lean_object* v___y_3148_){
_start:
{
if (lean_obj_tag(v_x_3142_) == 0)
{
lean_object* v_cs_3150_; lean_object* v___x_3152_; uint8_t v_isShared_3153_; uint8_t v_isSharedCheck_3176_; 
v_cs_3150_ = lean_ctor_get(v_x_3142_, 0);
v_isSharedCheck_3176_ = !lean_is_exclusive(v_x_3142_);
if (v_isSharedCheck_3176_ == 0)
{
v___x_3152_ = v_x_3142_;
v_isShared_3153_ = v_isSharedCheck_3176_;
goto v_resetjp_3151_;
}
else
{
lean_inc(v_cs_3150_);
lean_dec(v_x_3142_);
v___x_3152_ = lean_box(0);
v_isShared_3153_ = v_isSharedCheck_3176_;
goto v_resetjp_3151_;
}
v_resetjp_3151_:
{
size_t v_sz_3154_; size_t v___x_3155_; lean_object* v___x_3156_; 
v_sz_3154_ = lean_array_size(v_cs_3150_);
v___x_3155_ = ((size_t)0ULL);
v___x_3156_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22(v___x_3140_, v_ctx_x3f_3141_, v_sz_3154_, v___x_3155_, v_cs_3150_, v___y_3143_, v___y_3144_, v___y_3145_, v___y_3146_, v___y_3147_, v___y_3148_);
if (lean_obj_tag(v___x_3156_) == 0)
{
lean_object* v_a_3157_; lean_object* v___x_3159_; uint8_t v_isShared_3160_; uint8_t v_isSharedCheck_3167_; 
v_a_3157_ = lean_ctor_get(v___x_3156_, 0);
v_isSharedCheck_3167_ = !lean_is_exclusive(v___x_3156_);
if (v_isSharedCheck_3167_ == 0)
{
v___x_3159_ = v___x_3156_;
v_isShared_3160_ = v_isSharedCheck_3167_;
goto v_resetjp_3158_;
}
else
{
lean_inc(v_a_3157_);
lean_dec(v___x_3156_);
v___x_3159_ = lean_box(0);
v_isShared_3160_ = v_isSharedCheck_3167_;
goto v_resetjp_3158_;
}
v_resetjp_3158_:
{
lean_object* v___x_3162_; 
if (v_isShared_3153_ == 0)
{
lean_ctor_set(v___x_3152_, 0, v_a_3157_);
v___x_3162_ = v___x_3152_;
goto v_reusejp_3161_;
}
else
{
lean_object* v_reuseFailAlloc_3166_; 
v_reuseFailAlloc_3166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3166_, 0, v_a_3157_);
v___x_3162_ = v_reuseFailAlloc_3166_;
goto v_reusejp_3161_;
}
v_reusejp_3161_:
{
lean_object* v___x_3164_; 
if (v_isShared_3160_ == 0)
{
lean_ctor_set(v___x_3159_, 0, v___x_3162_);
v___x_3164_ = v___x_3159_;
goto v_reusejp_3163_;
}
else
{
lean_object* v_reuseFailAlloc_3165_; 
v_reuseFailAlloc_3165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3165_, 0, v___x_3162_);
v___x_3164_ = v_reuseFailAlloc_3165_;
goto v_reusejp_3163_;
}
v_reusejp_3163_:
{
return v___x_3164_;
}
}
}
}
else
{
lean_object* v_a_3168_; lean_object* v___x_3170_; uint8_t v_isShared_3171_; uint8_t v_isSharedCheck_3175_; 
lean_del_object(v___x_3152_);
v_a_3168_ = lean_ctor_get(v___x_3156_, 0);
v_isSharedCheck_3175_ = !lean_is_exclusive(v___x_3156_);
if (v_isSharedCheck_3175_ == 0)
{
v___x_3170_ = v___x_3156_;
v_isShared_3171_ = v_isSharedCheck_3175_;
goto v_resetjp_3169_;
}
else
{
lean_inc(v_a_3168_);
lean_dec(v___x_3156_);
v___x_3170_ = lean_box(0);
v_isShared_3171_ = v_isSharedCheck_3175_;
goto v_resetjp_3169_;
}
v_resetjp_3169_:
{
lean_object* v___x_3173_; 
if (v_isShared_3171_ == 0)
{
v___x_3173_ = v___x_3170_;
goto v_reusejp_3172_;
}
else
{
lean_object* v_reuseFailAlloc_3174_; 
v_reuseFailAlloc_3174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3174_, 0, v_a_3168_);
v___x_3173_ = v_reuseFailAlloc_3174_;
goto v_reusejp_3172_;
}
v_reusejp_3172_:
{
return v___x_3173_;
}
}
}
}
}
else
{
lean_object* v_vs_3177_; lean_object* v___x_3179_; uint8_t v_isShared_3180_; uint8_t v_isSharedCheck_3203_; 
v_vs_3177_ = lean_ctor_get(v_x_3142_, 0);
v_isSharedCheck_3203_ = !lean_is_exclusive(v_x_3142_);
if (v_isSharedCheck_3203_ == 0)
{
v___x_3179_ = v_x_3142_;
v_isShared_3180_ = v_isSharedCheck_3203_;
goto v_resetjp_3178_;
}
else
{
lean_inc(v_vs_3177_);
lean_dec(v_x_3142_);
v___x_3179_ = lean_box(0);
v_isShared_3180_ = v_isSharedCheck_3203_;
goto v_resetjp_3178_;
}
v_resetjp_3178_:
{
size_t v_sz_3181_; size_t v___x_3182_; lean_object* v___x_3183_; 
v_sz_3181_ = lean_array_size(v_vs_3177_);
v___x_3182_ = ((size_t)0ULL);
v___x_3183_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21(v___x_3140_, v_ctx_x3f_3141_, v_sz_3181_, v___x_3182_, v_vs_3177_, v___y_3143_, v___y_3144_, v___y_3145_, v___y_3146_, v___y_3147_, v___y_3148_);
if (lean_obj_tag(v___x_3183_) == 0)
{
lean_object* v_a_3184_; lean_object* v___x_3186_; uint8_t v_isShared_3187_; uint8_t v_isSharedCheck_3194_; 
v_a_3184_ = lean_ctor_get(v___x_3183_, 0);
v_isSharedCheck_3194_ = !lean_is_exclusive(v___x_3183_);
if (v_isSharedCheck_3194_ == 0)
{
v___x_3186_ = v___x_3183_;
v_isShared_3187_ = v_isSharedCheck_3194_;
goto v_resetjp_3185_;
}
else
{
lean_inc(v_a_3184_);
lean_dec(v___x_3183_);
v___x_3186_ = lean_box(0);
v_isShared_3187_ = v_isSharedCheck_3194_;
goto v_resetjp_3185_;
}
v_resetjp_3185_:
{
lean_object* v___x_3189_; 
if (v_isShared_3180_ == 0)
{
lean_ctor_set(v___x_3179_, 0, v_a_3184_);
v___x_3189_ = v___x_3179_;
goto v_reusejp_3188_;
}
else
{
lean_object* v_reuseFailAlloc_3193_; 
v_reuseFailAlloc_3193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3193_, 0, v_a_3184_);
v___x_3189_ = v_reuseFailAlloc_3193_;
goto v_reusejp_3188_;
}
v_reusejp_3188_:
{
lean_object* v___x_3191_; 
if (v_isShared_3187_ == 0)
{
lean_ctor_set(v___x_3186_, 0, v___x_3189_);
v___x_3191_ = v___x_3186_;
goto v_reusejp_3190_;
}
else
{
lean_object* v_reuseFailAlloc_3192_; 
v_reuseFailAlloc_3192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3192_, 0, v___x_3189_);
v___x_3191_ = v_reuseFailAlloc_3192_;
goto v_reusejp_3190_;
}
v_reusejp_3190_:
{
return v___x_3191_;
}
}
}
}
else
{
lean_object* v_a_3195_; lean_object* v___x_3197_; uint8_t v_isShared_3198_; uint8_t v_isSharedCheck_3202_; 
lean_del_object(v___x_3179_);
v_a_3195_ = lean_ctor_get(v___x_3183_, 0);
v_isSharedCheck_3202_ = !lean_is_exclusive(v___x_3183_);
if (v_isSharedCheck_3202_ == 0)
{
v___x_3197_ = v___x_3183_;
v_isShared_3198_ = v_isSharedCheck_3202_;
goto v_resetjp_3196_;
}
else
{
lean_inc(v_a_3195_);
lean_dec(v___x_3183_);
v___x_3197_ = lean_box(0);
v_isShared_3198_ = v_isSharedCheck_3202_;
goto v_resetjp_3196_;
}
v_resetjp_3196_:
{
lean_object* v___x_3200_; 
if (v_isShared_3198_ == 0)
{
v___x_3200_ = v___x_3197_;
goto v_reusejp_3199_;
}
else
{
lean_object* v_reuseFailAlloc_3201_; 
v_reuseFailAlloc_3201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3201_, 0, v_a_3195_);
v___x_3200_ = v_reuseFailAlloc_3201_;
goto v_reusejp_3199_;
}
v_reusejp_3199_:
{
return v___x_3200_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22(lean_object* v___x_3204_, lean_object* v_ctx_x3f_3205_, size_t v_sz_3206_, size_t v_i_3207_, lean_object* v_bs_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_, lean_object* v___y_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_){
_start:
{
uint8_t v___x_3216_; 
v___x_3216_ = lean_usize_dec_lt(v_i_3207_, v_sz_3206_);
if (v___x_3216_ == 0)
{
lean_object* v___x_3217_; 
lean_dec_ref(v_ctx_x3f_3205_);
v___x_3217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3217_, 0, v_bs_3208_);
return v___x_3217_;
}
else
{
lean_object* v_v_3218_; lean_object* v___x_3219_; 
v_v_3218_ = lean_array_uget_borrowed(v_bs_3208_, v_i_3207_);
lean_inc(v_v_3218_);
lean_inc_ref(v_ctx_x3f_3205_);
v___x_3219_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20(v___x_3204_, v_ctx_x3f_3205_, v_v_3218_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_);
if (lean_obj_tag(v___x_3219_) == 0)
{
lean_object* v_a_3220_; lean_object* v___x_3221_; lean_object* v_bs_x27_3222_; size_t v___x_3223_; size_t v___x_3224_; lean_object* v___x_3225_; 
v_a_3220_ = lean_ctor_get(v___x_3219_, 0);
lean_inc(v_a_3220_);
lean_dec_ref_known(v___x_3219_, 1);
v___x_3221_ = lean_unsigned_to_nat(0u);
v_bs_x27_3222_ = lean_array_uset(v_bs_3208_, v_i_3207_, v___x_3221_);
v___x_3223_ = ((size_t)1ULL);
v___x_3224_ = lean_usize_add(v_i_3207_, v___x_3223_);
v___x_3225_ = lean_array_uset(v_bs_x27_3222_, v_i_3207_, v_a_3220_);
v_i_3207_ = v___x_3224_;
v_bs_3208_ = v___x_3225_;
goto _start;
}
else
{
lean_object* v_a_3227_; lean_object* v___x_3229_; uint8_t v_isShared_3230_; uint8_t v_isSharedCheck_3234_; 
lean_dec_ref(v_bs_3208_);
lean_dec_ref(v_ctx_x3f_3205_);
v_a_3227_ = lean_ctor_get(v___x_3219_, 0);
v_isSharedCheck_3234_ = !lean_is_exclusive(v___x_3219_);
if (v_isSharedCheck_3234_ == 0)
{
v___x_3229_ = v___x_3219_;
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
else
{
lean_inc(v_a_3227_);
lean_dec(v___x_3219_);
v___x_3229_ = lean_box(0);
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
v_resetjp_3228_:
{
lean_object* v___x_3232_; 
if (v_isShared_3230_ == 0)
{
v___x_3232_ = v___x_3229_;
goto v_reusejp_3231_;
}
else
{
lean_object* v_reuseFailAlloc_3233_; 
v_reuseFailAlloc_3233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3233_, 0, v_a_3227_);
v___x_3232_ = v_reuseFailAlloc_3233_;
goto v_reusejp_3231_;
}
v_reusejp_3231_:
{
return v___x_3232_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22___boxed(lean_object* v___x_3235_, lean_object* v_ctx_x3f_3236_, lean_object* v_sz_3237_, lean_object* v_i_3238_, lean_object* v_bs_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_, lean_object* v___y_3244_, lean_object* v___y_3245_, lean_object* v___y_3246_){
_start:
{
size_t v_sz_boxed_3247_; size_t v_i_boxed_3248_; lean_object* v_res_3249_; 
v_sz_boxed_3247_ = lean_unbox_usize(v_sz_3237_);
lean_dec(v_sz_3237_);
v_i_boxed_3248_ = lean_unbox_usize(v_i_3238_);
lean_dec(v_i_3238_);
v_res_3249_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20_spec__22(v___x_3235_, v_ctx_x3f_3236_, v_sz_boxed_3247_, v_i_boxed_3248_, v_bs_3239_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_, v___y_3244_, v___y_3245_);
lean_dec(v___y_3245_);
lean_dec_ref(v___y_3244_);
lean_dec(v___y_3243_);
lean_dec_ref(v___y_3242_);
lean_dec(v___y_3241_);
lean_dec_ref(v___y_3240_);
lean_dec_ref(v___x_3235_);
return v_res_3249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20___boxed(lean_object* v___x_3250_, lean_object* v_ctx_x3f_3251_, lean_object* v_x_3252_, lean_object* v___y_3253_, lean_object* v___y_3254_, lean_object* v___y_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_){
_start:
{
lean_object* v_res_3260_; 
v_res_3260_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20(v___x_3250_, v_ctx_x3f_3251_, v_x_3252_, v___y_3253_, v___y_3254_, v___y_3255_, v___y_3256_, v___y_3257_, v___y_3258_);
lean_dec(v___y_3258_);
lean_dec_ref(v___y_3257_);
lean_dec(v___y_3256_);
lean_dec_ref(v___y_3255_);
lean_dec(v___y_3254_);
lean_dec_ref(v___y_3253_);
lean_dec_ref(v___x_3250_);
return v_res_3260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13(lean_object* v___x_3261_, lean_object* v_ctx_x3f_3262_, lean_object* v_t_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_, lean_object* v___y_3266_, lean_object* v___y_3267_, lean_object* v___y_3268_, lean_object* v___y_3269_){
_start:
{
lean_object* v_root_3271_; lean_object* v_tail_3272_; lean_object* v_size_3273_; size_t v_shift_3274_; lean_object* v_tailOff_3275_; lean_object* v___x_3277_; uint8_t v_isShared_3278_; uint8_t v_isSharedCheck_3311_; 
v_root_3271_ = lean_ctor_get(v_t_3263_, 0);
v_tail_3272_ = lean_ctor_get(v_t_3263_, 1);
v_size_3273_ = lean_ctor_get(v_t_3263_, 2);
v_shift_3274_ = lean_ctor_get_usize(v_t_3263_, 4);
v_tailOff_3275_ = lean_ctor_get(v_t_3263_, 3);
v_isSharedCheck_3311_ = !lean_is_exclusive(v_t_3263_);
if (v_isSharedCheck_3311_ == 0)
{
v___x_3277_ = v_t_3263_;
v_isShared_3278_ = v_isSharedCheck_3311_;
goto v_resetjp_3276_;
}
else
{
lean_inc(v_tailOff_3275_);
lean_inc(v_size_3273_);
lean_inc(v_tail_3272_);
lean_inc(v_root_3271_);
lean_dec(v_t_3263_);
v___x_3277_ = lean_box(0);
v_isShared_3278_ = v_isSharedCheck_3311_;
goto v_resetjp_3276_;
}
v_resetjp_3276_:
{
lean_object* v___x_3279_; 
lean_inc_ref(v_ctx_x3f_3262_);
v___x_3279_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__20(v___x_3261_, v_ctx_x3f_3262_, v_root_3271_, v___y_3264_, v___y_3265_, v___y_3266_, v___y_3267_, v___y_3268_, v___y_3269_);
if (lean_obj_tag(v___x_3279_) == 0)
{
lean_object* v_a_3280_; size_t v_sz_3281_; size_t v___x_3282_; lean_object* v___x_3283_; 
v_a_3280_ = lean_ctor_get(v___x_3279_, 0);
lean_inc(v_a_3280_);
lean_dec_ref_known(v___x_3279_, 1);
v_sz_3281_ = lean_array_size(v_tail_3272_);
v___x_3282_ = ((size_t)0ULL);
v___x_3283_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13_spec__21(v___x_3261_, v_ctx_x3f_3262_, v_sz_3281_, v___x_3282_, v_tail_3272_, v___y_3264_, v___y_3265_, v___y_3266_, v___y_3267_, v___y_3268_, v___y_3269_);
if (lean_obj_tag(v___x_3283_) == 0)
{
lean_object* v_a_3284_; lean_object* v___x_3286_; uint8_t v_isShared_3287_; uint8_t v_isSharedCheck_3294_; 
v_a_3284_ = lean_ctor_get(v___x_3283_, 0);
v_isSharedCheck_3294_ = !lean_is_exclusive(v___x_3283_);
if (v_isSharedCheck_3294_ == 0)
{
v___x_3286_ = v___x_3283_;
v_isShared_3287_ = v_isSharedCheck_3294_;
goto v_resetjp_3285_;
}
else
{
lean_inc(v_a_3284_);
lean_dec(v___x_3283_);
v___x_3286_ = lean_box(0);
v_isShared_3287_ = v_isSharedCheck_3294_;
goto v_resetjp_3285_;
}
v_resetjp_3285_:
{
lean_object* v___x_3289_; 
if (v_isShared_3278_ == 0)
{
lean_ctor_set(v___x_3277_, 1, v_a_3284_);
lean_ctor_set(v___x_3277_, 0, v_a_3280_);
v___x_3289_ = v___x_3277_;
goto v_reusejp_3288_;
}
else
{
lean_object* v_reuseFailAlloc_3293_; 
v_reuseFailAlloc_3293_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v_reuseFailAlloc_3293_, 0, v_a_3280_);
lean_ctor_set(v_reuseFailAlloc_3293_, 1, v_a_3284_);
lean_ctor_set(v_reuseFailAlloc_3293_, 2, v_size_3273_);
lean_ctor_set(v_reuseFailAlloc_3293_, 3, v_tailOff_3275_);
lean_ctor_set_usize(v_reuseFailAlloc_3293_, 4, v_shift_3274_);
v___x_3289_ = v_reuseFailAlloc_3293_;
goto v_reusejp_3288_;
}
v_reusejp_3288_:
{
lean_object* v___x_3291_; 
if (v_isShared_3287_ == 0)
{
lean_ctor_set(v___x_3286_, 0, v___x_3289_);
v___x_3291_ = v___x_3286_;
goto v_reusejp_3290_;
}
else
{
lean_object* v_reuseFailAlloc_3292_; 
v_reuseFailAlloc_3292_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3292_, 0, v___x_3289_);
v___x_3291_ = v_reuseFailAlloc_3292_;
goto v_reusejp_3290_;
}
v_reusejp_3290_:
{
return v___x_3291_;
}
}
}
}
else
{
lean_object* v_a_3295_; lean_object* v___x_3297_; uint8_t v_isShared_3298_; uint8_t v_isSharedCheck_3302_; 
lean_dec(v_a_3280_);
lean_del_object(v___x_3277_);
lean_dec(v_tailOff_3275_);
lean_dec(v_size_3273_);
v_a_3295_ = lean_ctor_get(v___x_3283_, 0);
v_isSharedCheck_3302_ = !lean_is_exclusive(v___x_3283_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3297_ = v___x_3283_;
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
else
{
lean_inc(v_a_3295_);
lean_dec(v___x_3283_);
v___x_3297_ = lean_box(0);
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
v_resetjp_3296_:
{
lean_object* v___x_3300_; 
if (v_isShared_3298_ == 0)
{
v___x_3300_ = v___x_3297_;
goto v_reusejp_3299_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v_a_3295_);
v___x_3300_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3299_;
}
v_reusejp_3299_:
{
return v___x_3300_;
}
}
}
}
else
{
lean_object* v_a_3303_; lean_object* v___x_3305_; uint8_t v_isShared_3306_; uint8_t v_isSharedCheck_3310_; 
lean_del_object(v___x_3277_);
lean_dec(v_tailOff_3275_);
lean_dec(v_size_3273_);
lean_dec_ref(v_tail_3272_);
lean_dec_ref(v_ctx_x3f_3262_);
v_a_3303_ = lean_ctor_get(v___x_3279_, 0);
v_isSharedCheck_3310_ = !lean_is_exclusive(v___x_3279_);
if (v_isSharedCheck_3310_ == 0)
{
v___x_3305_ = v___x_3279_;
v_isShared_3306_ = v_isSharedCheck_3310_;
goto v_resetjp_3304_;
}
else
{
lean_inc(v_a_3303_);
lean_dec(v___x_3279_);
v___x_3305_ = lean_box(0);
v_isShared_3306_ = v_isSharedCheck_3310_;
goto v_resetjp_3304_;
}
v_resetjp_3304_:
{
lean_object* v___x_3308_; 
if (v_isShared_3306_ == 0)
{
v___x_3308_ = v___x_3305_;
goto v_reusejp_3307_;
}
else
{
lean_object* v_reuseFailAlloc_3309_; 
v_reuseFailAlloc_3309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3309_, 0, v_a_3303_);
v___x_3308_ = v_reuseFailAlloc_3309_;
goto v_reusejp_3307_;
}
v_reusejp_3307_:
{
return v___x_3308_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13___boxed(lean_object* v___x_3312_, lean_object* v_ctx_x3f_3313_, lean_object* v_t_3314_, lean_object* v___y_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_, lean_object* v___y_3319_, lean_object* v___y_3320_, lean_object* v___y_3321_){
_start:
{
lean_object* v_res_3322_; 
v_res_3322_ = lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13(v___x_3312_, v_ctx_x3f_3313_, v_t_3314_, v___y_3315_, v___y_3316_, v___y_3317_, v___y_3318_, v___y_3319_, v___y_3320_);
lean_dec(v___y_3320_);
lean_dec_ref(v___y_3319_);
lean_dec(v___y_3318_);
lean_dec_ref(v___y_3317_);
lean_dec(v___y_3316_);
lean_dec_ref(v___y_3315_);
lean_dec_ref(v___x_3312_);
return v_res_3322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0(lean_object* v___y_3323_, lean_object* v_ctx_x3f_3324_, lean_object* v___y_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v_a_3330_, lean_object* v_a_x3f_3331_){
_start:
{
lean_object* v___x_3333_; lean_object* v_infoState_3334_; lean_object* v_trees_3335_; lean_object* v___x_3336_; 
v___x_3333_ = lean_st_ref_get(v___y_3323_);
v_infoState_3334_ = lean_ctor_get(v___x_3333_, 7);
lean_inc_ref(v_infoState_3334_);
lean_dec(v___x_3333_);
v_trees_3335_ = lean_ctor_get(v_infoState_3334_, 2);
lean_inc_ref(v_trees_3335_);
v___x_3336_ = lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__13(v_infoState_3334_, v_ctx_x3f_3324_, v_trees_3335_, v___y_3325_, v___y_3326_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3323_);
lean_dec_ref(v_infoState_3334_);
if (lean_obj_tag(v___x_3336_) == 0)
{
lean_object* v_a_3337_; lean_object* v___x_3339_; uint8_t v_isShared_3340_; uint8_t v_isSharedCheck_3375_; 
v_a_3337_ = lean_ctor_get(v___x_3336_, 0);
v_isSharedCheck_3375_ = !lean_is_exclusive(v___x_3336_);
if (v_isSharedCheck_3375_ == 0)
{
v___x_3339_ = v___x_3336_;
v_isShared_3340_ = v_isSharedCheck_3375_;
goto v_resetjp_3338_;
}
else
{
lean_inc(v_a_3337_);
lean_dec(v___x_3336_);
v___x_3339_ = lean_box(0);
v_isShared_3340_ = v_isSharedCheck_3375_;
goto v_resetjp_3338_;
}
v_resetjp_3338_:
{
lean_object* v___x_3341_; lean_object* v_infoState_3342_; lean_object* v_env_3343_; lean_object* v_nextMacroScope_3344_; lean_object* v_ngen_3345_; lean_object* v_auxDeclNGen_3346_; lean_object* v_traceState_3347_; lean_object* v_cache_3348_; lean_object* v_messages_3349_; lean_object* v_snapshotTasks_3350_; lean_object* v___x_3352_; uint8_t v_isShared_3353_; uint8_t v_isSharedCheck_3374_; 
v___x_3341_ = lean_st_ref_take(v___y_3323_);
v_infoState_3342_ = lean_ctor_get(v___x_3341_, 7);
v_env_3343_ = lean_ctor_get(v___x_3341_, 0);
v_nextMacroScope_3344_ = lean_ctor_get(v___x_3341_, 1);
v_ngen_3345_ = lean_ctor_get(v___x_3341_, 2);
v_auxDeclNGen_3346_ = lean_ctor_get(v___x_3341_, 3);
v_traceState_3347_ = lean_ctor_get(v___x_3341_, 4);
v_cache_3348_ = lean_ctor_get(v___x_3341_, 5);
v_messages_3349_ = lean_ctor_get(v___x_3341_, 6);
v_snapshotTasks_3350_ = lean_ctor_get(v___x_3341_, 8);
v_isSharedCheck_3374_ = !lean_is_exclusive(v___x_3341_);
if (v_isSharedCheck_3374_ == 0)
{
v___x_3352_ = v___x_3341_;
v_isShared_3353_ = v_isSharedCheck_3374_;
goto v_resetjp_3351_;
}
else
{
lean_inc(v_snapshotTasks_3350_);
lean_inc(v_infoState_3342_);
lean_inc(v_messages_3349_);
lean_inc(v_cache_3348_);
lean_inc(v_traceState_3347_);
lean_inc(v_auxDeclNGen_3346_);
lean_inc(v_ngen_3345_);
lean_inc(v_nextMacroScope_3344_);
lean_inc(v_env_3343_);
lean_dec(v___x_3341_);
v___x_3352_ = lean_box(0);
v_isShared_3353_ = v_isSharedCheck_3374_;
goto v_resetjp_3351_;
}
v_resetjp_3351_:
{
uint8_t v_enabled_3354_; lean_object* v_assignment_3355_; lean_object* v_lazyAssignment_3356_; lean_object* v___x_3358_; uint8_t v_isShared_3359_; uint8_t v_isSharedCheck_3372_; 
v_enabled_3354_ = lean_ctor_get_uint8(v_infoState_3342_, sizeof(void*)*3);
v_assignment_3355_ = lean_ctor_get(v_infoState_3342_, 0);
v_lazyAssignment_3356_ = lean_ctor_get(v_infoState_3342_, 1);
v_isSharedCheck_3372_ = !lean_is_exclusive(v_infoState_3342_);
if (v_isSharedCheck_3372_ == 0)
{
lean_object* v_unused_3373_; 
v_unused_3373_ = lean_ctor_get(v_infoState_3342_, 2);
lean_dec(v_unused_3373_);
v___x_3358_ = v_infoState_3342_;
v_isShared_3359_ = v_isSharedCheck_3372_;
goto v_resetjp_3357_;
}
else
{
lean_inc(v_lazyAssignment_3356_);
lean_inc(v_assignment_3355_);
lean_dec(v_infoState_3342_);
v___x_3358_ = lean_box(0);
v_isShared_3359_ = v_isSharedCheck_3372_;
goto v_resetjp_3357_;
}
v_resetjp_3357_:
{
lean_object* v___x_3360_; lean_object* v___x_3362_; 
v___x_3360_ = l_Lean_PersistentArray_append___redArg(v_a_3330_, v_a_3337_);
lean_dec(v_a_3337_);
if (v_isShared_3359_ == 0)
{
lean_ctor_set(v___x_3358_, 2, v___x_3360_);
v___x_3362_ = v___x_3358_;
goto v_reusejp_3361_;
}
else
{
lean_object* v_reuseFailAlloc_3371_; 
v_reuseFailAlloc_3371_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_3371_, 0, v_assignment_3355_);
lean_ctor_set(v_reuseFailAlloc_3371_, 1, v_lazyAssignment_3356_);
lean_ctor_set(v_reuseFailAlloc_3371_, 2, v___x_3360_);
lean_ctor_set_uint8(v_reuseFailAlloc_3371_, sizeof(void*)*3, v_enabled_3354_);
v___x_3362_ = v_reuseFailAlloc_3371_;
goto v_reusejp_3361_;
}
v_reusejp_3361_:
{
lean_object* v___x_3364_; 
if (v_isShared_3353_ == 0)
{
lean_ctor_set(v___x_3352_, 7, v___x_3362_);
v___x_3364_ = v___x_3352_;
goto v_reusejp_3363_;
}
else
{
lean_object* v_reuseFailAlloc_3370_; 
v_reuseFailAlloc_3370_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3370_, 0, v_env_3343_);
lean_ctor_set(v_reuseFailAlloc_3370_, 1, v_nextMacroScope_3344_);
lean_ctor_set(v_reuseFailAlloc_3370_, 2, v_ngen_3345_);
lean_ctor_set(v_reuseFailAlloc_3370_, 3, v_auxDeclNGen_3346_);
lean_ctor_set(v_reuseFailAlloc_3370_, 4, v_traceState_3347_);
lean_ctor_set(v_reuseFailAlloc_3370_, 5, v_cache_3348_);
lean_ctor_set(v_reuseFailAlloc_3370_, 6, v_messages_3349_);
lean_ctor_set(v_reuseFailAlloc_3370_, 7, v___x_3362_);
lean_ctor_set(v_reuseFailAlloc_3370_, 8, v_snapshotTasks_3350_);
v___x_3364_ = v_reuseFailAlloc_3370_;
goto v_reusejp_3363_;
}
v_reusejp_3363_:
{
lean_object* v___x_3365_; lean_object* v___x_3366_; lean_object* v___x_3368_; 
v___x_3365_ = lean_st_ref_set(v___y_3323_, v___x_3364_);
v___x_3366_ = lean_box(0);
if (v_isShared_3340_ == 0)
{
lean_ctor_set(v___x_3339_, 0, v___x_3366_);
v___x_3368_ = v___x_3339_;
goto v_reusejp_3367_;
}
else
{
lean_object* v_reuseFailAlloc_3369_; 
v_reuseFailAlloc_3369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3369_, 0, v___x_3366_);
v___x_3368_ = v_reuseFailAlloc_3369_;
goto v_reusejp_3367_;
}
v_reusejp_3367_:
{
return v___x_3368_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3376_; lean_object* v___x_3378_; uint8_t v_isShared_3379_; uint8_t v_isSharedCheck_3383_; 
lean_dec_ref(v_a_3330_);
v_a_3376_ = lean_ctor_get(v___x_3336_, 0);
v_isSharedCheck_3383_ = !lean_is_exclusive(v___x_3336_);
if (v_isSharedCheck_3383_ == 0)
{
v___x_3378_ = v___x_3336_;
v_isShared_3379_ = v_isSharedCheck_3383_;
goto v_resetjp_3377_;
}
else
{
lean_inc(v_a_3376_);
lean_dec(v___x_3336_);
v___x_3378_ = lean_box(0);
v_isShared_3379_ = v_isSharedCheck_3383_;
goto v_resetjp_3377_;
}
v_resetjp_3377_:
{
lean_object* v___x_3381_; 
if (v_isShared_3379_ == 0)
{
v___x_3381_ = v___x_3378_;
goto v_reusejp_3380_;
}
else
{
lean_object* v_reuseFailAlloc_3382_; 
v_reuseFailAlloc_3382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3382_, 0, v_a_3376_);
v___x_3381_ = v_reuseFailAlloc_3382_;
goto v_reusejp_3380_;
}
v_reusejp_3380_:
{
return v___x_3381_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0___boxed(lean_object* v___y_3384_, lean_object* v_ctx_x3f_3385_, lean_object* v___y_3386_, lean_object* v___y_3387_, lean_object* v___y_3388_, lean_object* v___y_3389_, lean_object* v___y_3390_, lean_object* v_a_3391_, lean_object* v_a_x3f_3392_, lean_object* v___y_3393_){
_start:
{
lean_object* v_res_3394_; 
v_res_3394_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0(v___y_3384_, v_ctx_x3f_3385_, v___y_3386_, v___y_3387_, v___y_3388_, v___y_3389_, v___y_3390_, v_a_3391_, v_a_x3f_3392_);
lean_dec(v_a_x3f_3392_);
lean_dec_ref(v___y_3390_);
lean_dec(v___y_3389_);
lean_dec_ref(v___y_3388_);
lean_dec(v___y_3387_);
lean_dec_ref(v___y_3386_);
lean_dec(v___y_3384_);
return v_res_3394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg(lean_object* v_x_3395_, lean_object* v_ctx_x3f_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_){
_start:
{
lean_object* v___x_3404_; lean_object* v_infoState_3405_; uint8_t v_enabled_3406_; 
v___x_3404_ = lean_st_ref_get(v___y_3402_);
v_infoState_3405_ = lean_ctor_get(v___x_3404_, 7);
lean_inc_ref(v_infoState_3405_);
lean_dec(v___x_3404_);
v_enabled_3406_ = lean_ctor_get_uint8(v_infoState_3405_, sizeof(void*)*3);
lean_dec_ref(v_infoState_3405_);
if (v_enabled_3406_ == 0)
{
lean_object* v___x_3407_; 
lean_dec_ref(v_ctx_x3f_3396_);
lean_inc(v___y_3402_);
lean_inc_ref(v___y_3401_);
lean_inc(v___y_3400_);
lean_inc_ref(v___y_3399_);
lean_inc(v___y_3398_);
lean_inc_ref(v___y_3397_);
v___x_3407_ = lean_apply_7(v_x_3395_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v___y_3402_, lean_box(0));
return v___x_3407_;
}
else
{
lean_object* v___x_3408_; lean_object* v_a_3409_; lean_object* v_r_3410_; 
v___x_3408_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg(v___y_3402_);
v_a_3409_ = lean_ctor_get(v___x_3408_, 0);
lean_inc(v_a_3409_);
lean_dec_ref(v___x_3408_);
lean_inc(v___y_3402_);
lean_inc_ref(v___y_3401_);
lean_inc(v___y_3400_);
lean_inc_ref(v___y_3399_);
lean_inc(v___y_3398_);
lean_inc_ref(v___y_3397_);
v_r_3410_ = lean_apply_7(v_x_3395_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v___y_3402_, lean_box(0));
if (lean_obj_tag(v_r_3410_) == 0)
{
lean_object* v_a_3411_; lean_object* v___x_3413_; uint8_t v_isShared_3414_; uint8_t v_isSharedCheck_3435_; 
v_a_3411_ = lean_ctor_get(v_r_3410_, 0);
v_isSharedCheck_3435_ = !lean_is_exclusive(v_r_3410_);
if (v_isSharedCheck_3435_ == 0)
{
v___x_3413_ = v_r_3410_;
v_isShared_3414_ = v_isSharedCheck_3435_;
goto v_resetjp_3412_;
}
else
{
lean_inc(v_a_3411_);
lean_dec(v_r_3410_);
v___x_3413_ = lean_box(0);
v_isShared_3414_ = v_isSharedCheck_3435_;
goto v_resetjp_3412_;
}
v_resetjp_3412_:
{
lean_object* v___x_3416_; 
lean_inc(v_a_3411_);
if (v_isShared_3414_ == 0)
{
lean_ctor_set_tag(v___x_3413_, 1);
v___x_3416_ = v___x_3413_;
goto v_reusejp_3415_;
}
else
{
lean_object* v_reuseFailAlloc_3434_; 
v_reuseFailAlloc_3434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3434_, 0, v_a_3411_);
v___x_3416_ = v_reuseFailAlloc_3434_;
goto v_reusejp_3415_;
}
v_reusejp_3415_:
{
lean_object* v___x_3417_; 
v___x_3417_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0(v___y_3402_, v_ctx_x3f_3396_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v_a_3409_, v___x_3416_);
lean_dec_ref(v___x_3416_);
if (lean_obj_tag(v___x_3417_) == 0)
{
lean_object* v___x_3419_; uint8_t v_isShared_3420_; uint8_t v_isSharedCheck_3424_; 
v_isSharedCheck_3424_ = !lean_is_exclusive(v___x_3417_);
if (v_isSharedCheck_3424_ == 0)
{
lean_object* v_unused_3425_; 
v_unused_3425_ = lean_ctor_get(v___x_3417_, 0);
lean_dec(v_unused_3425_);
v___x_3419_ = v___x_3417_;
v_isShared_3420_ = v_isSharedCheck_3424_;
goto v_resetjp_3418_;
}
else
{
lean_dec(v___x_3417_);
v___x_3419_ = lean_box(0);
v_isShared_3420_ = v_isSharedCheck_3424_;
goto v_resetjp_3418_;
}
v_resetjp_3418_:
{
lean_object* v___x_3422_; 
if (v_isShared_3420_ == 0)
{
lean_ctor_set(v___x_3419_, 0, v_a_3411_);
v___x_3422_ = v___x_3419_;
goto v_reusejp_3421_;
}
else
{
lean_object* v_reuseFailAlloc_3423_; 
v_reuseFailAlloc_3423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3423_, 0, v_a_3411_);
v___x_3422_ = v_reuseFailAlloc_3423_;
goto v_reusejp_3421_;
}
v_reusejp_3421_:
{
return v___x_3422_;
}
}
}
else
{
lean_object* v_a_3426_; lean_object* v___x_3428_; uint8_t v_isShared_3429_; uint8_t v_isSharedCheck_3433_; 
lean_dec(v_a_3411_);
v_a_3426_ = lean_ctor_get(v___x_3417_, 0);
v_isSharedCheck_3433_ = !lean_is_exclusive(v___x_3417_);
if (v_isSharedCheck_3433_ == 0)
{
v___x_3428_ = v___x_3417_;
v_isShared_3429_ = v_isSharedCheck_3433_;
goto v_resetjp_3427_;
}
else
{
lean_inc(v_a_3426_);
lean_dec(v___x_3417_);
v___x_3428_ = lean_box(0);
v_isShared_3429_ = v_isSharedCheck_3433_;
goto v_resetjp_3427_;
}
v_resetjp_3427_:
{
lean_object* v___x_3431_; 
if (v_isShared_3429_ == 0)
{
v___x_3431_ = v___x_3428_;
goto v_reusejp_3430_;
}
else
{
lean_object* v_reuseFailAlloc_3432_; 
v_reuseFailAlloc_3432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3432_, 0, v_a_3426_);
v___x_3431_ = v_reuseFailAlloc_3432_;
goto v_reusejp_3430_;
}
v_reusejp_3430_:
{
return v___x_3431_;
}
}
}
}
}
}
else
{
lean_object* v_a_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; 
v_a_3436_ = lean_ctor_get(v_r_3410_, 0);
lean_inc(v_a_3436_);
lean_dec_ref_known(v_r_3410_, 1);
v___x_3437_ = lean_box(0);
v___x_3438_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___lam__0(v___y_3402_, v_ctx_x3f_3396_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v_a_3409_, v___x_3437_);
if (lean_obj_tag(v___x_3438_) == 0)
{
lean_object* v___x_3440_; uint8_t v_isShared_3441_; uint8_t v_isSharedCheck_3445_; 
v_isSharedCheck_3445_ = !lean_is_exclusive(v___x_3438_);
if (v_isSharedCheck_3445_ == 0)
{
lean_object* v_unused_3446_; 
v_unused_3446_ = lean_ctor_get(v___x_3438_, 0);
lean_dec(v_unused_3446_);
v___x_3440_ = v___x_3438_;
v_isShared_3441_ = v_isSharedCheck_3445_;
goto v_resetjp_3439_;
}
else
{
lean_dec(v___x_3438_);
v___x_3440_ = lean_box(0);
v_isShared_3441_ = v_isSharedCheck_3445_;
goto v_resetjp_3439_;
}
v_resetjp_3439_:
{
lean_object* v___x_3443_; 
if (v_isShared_3441_ == 0)
{
lean_ctor_set_tag(v___x_3440_, 1);
lean_ctor_set(v___x_3440_, 0, v_a_3436_);
v___x_3443_ = v___x_3440_;
goto v_reusejp_3442_;
}
else
{
lean_object* v_reuseFailAlloc_3444_; 
v_reuseFailAlloc_3444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3444_, 0, v_a_3436_);
v___x_3443_ = v_reuseFailAlloc_3444_;
goto v_reusejp_3442_;
}
v_reusejp_3442_:
{
return v___x_3443_;
}
}
}
else
{
lean_object* v_a_3447_; lean_object* v___x_3449_; uint8_t v_isShared_3450_; uint8_t v_isSharedCheck_3454_; 
lean_dec(v_a_3436_);
v_a_3447_ = lean_ctor_get(v___x_3438_, 0);
v_isSharedCheck_3454_ = !lean_is_exclusive(v___x_3438_);
if (v_isSharedCheck_3454_ == 0)
{
v___x_3449_ = v___x_3438_;
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
else
{
lean_inc(v_a_3447_);
lean_dec(v___x_3438_);
v___x_3449_ = lean_box(0);
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
v_resetjp_3448_:
{
lean_object* v___x_3452_; 
if (v_isShared_3450_ == 0)
{
v___x_3452_ = v___x_3449_;
goto v_reusejp_3451_;
}
else
{
lean_object* v_reuseFailAlloc_3453_; 
v_reuseFailAlloc_3453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3453_, 0, v_a_3447_);
v___x_3452_ = v_reuseFailAlloc_3453_;
goto v_reusejp_3451_;
}
v_reusejp_3451_:
{
return v___x_3452_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg___boxed(lean_object* v_x_3455_, lean_object* v_ctx_x3f_3456_, lean_object* v___y_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_, lean_object* v___y_3461_, lean_object* v___y_3462_, lean_object* v___y_3463_){
_start:
{
lean_object* v_res_3464_; 
v_res_3464_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg(v_x_3455_, v_ctx_x3f_3456_, v___y_3457_, v___y_3458_, v___y_3459_, v___y_3460_, v___y_3461_, v___y_3462_);
lean_dec(v___y_3462_);
lean_dec_ref(v___y_3461_);
lean_dec(v___y_3460_);
lean_dec_ref(v___y_3459_);
lean_dec(v___y_3458_);
lean_dec_ref(v___y_3457_);
return v_res_3464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg(lean_object* v___y_3465_, lean_object* v___y_3466_, lean_object* v___y_3467_){
_start:
{
lean_object* v___x_3469_; lean_object* v_env_3470_; lean_object* v___x_3471_; lean_object* v_mctx_3472_; lean_object* v_options_3473_; lean_object* v_currNamespace_3474_; lean_object* v_openDecls_3475_; lean_object* v___x_3476_; lean_object* v_ngen_3477_; lean_object* v___x_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; 
v___x_3469_ = lean_st_ref_get(v___y_3467_);
v_env_3470_ = lean_ctor_get(v___x_3469_, 0);
lean_inc_ref(v_env_3470_);
lean_dec(v___x_3469_);
v___x_3471_ = lean_st_ref_get(v___y_3465_);
v_mctx_3472_ = lean_ctor_get(v___x_3471_, 0);
lean_inc_ref(v_mctx_3472_);
lean_dec(v___x_3471_);
v_options_3473_ = lean_ctor_get(v___y_3466_, 2);
v_currNamespace_3474_ = lean_ctor_get(v___y_3466_, 6);
v_openDecls_3475_ = lean_ctor_get(v___y_3466_, 7);
v___x_3476_ = lean_st_ref_get(v___y_3467_);
v_ngen_3477_ = lean_ctor_get(v___x_3476_, 2);
lean_inc_ref(v_ngen_3477_);
lean_dec(v___x_3476_);
v___x_3478_ = lean_box(0);
v___x_3479_ = l_Lean_instInhabitedFileMap_default;
lean_inc(v_openDecls_3475_);
lean_inc(v_currNamespace_3474_);
lean_inc_ref(v_options_3473_);
v___x_3480_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_3480_, 0, v_env_3470_);
lean_ctor_set(v___x_3480_, 1, v___x_3478_);
lean_ctor_set(v___x_3480_, 2, v___x_3479_);
lean_ctor_set(v___x_3480_, 3, v_mctx_3472_);
lean_ctor_set(v___x_3480_, 4, v_options_3473_);
lean_ctor_set(v___x_3480_, 5, v_currNamespace_3474_);
lean_ctor_set(v___x_3480_, 6, v_openDecls_3475_);
lean_ctor_set(v___x_3480_, 7, v_ngen_3477_);
v___x_3481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3481_, 0, v___x_3480_);
return v___x_3481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg___boxed(lean_object* v___y_3482_, lean_object* v___y_3483_, lean_object* v___y_3484_, lean_object* v___y_3485_){
_start:
{
lean_object* v_res_3486_; 
v_res_3486_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg(v___y_3482_, v___y_3483_, v___y_3484_);
lean_dec(v___y_3484_);
lean_dec_ref(v___y_3483_);
lean_dec(v___y_3482_);
return v_res_3486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5(lean_object* v___y_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_){
_start:
{
lean_object* v___x_3494_; lean_object* v_a_3495_; lean_object* v___x_3497_; uint8_t v_isShared_3498_; uint8_t v_isSharedCheck_3519_; 
v___x_3494_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg(v___y_3490_, v___y_3491_, v___y_3492_);
v_a_3495_ = lean_ctor_get(v___x_3494_, 0);
v_isSharedCheck_3519_ = !lean_is_exclusive(v___x_3494_);
if (v_isSharedCheck_3519_ == 0)
{
v___x_3497_ = v___x_3494_;
v_isShared_3498_ = v_isSharedCheck_3519_;
goto v_resetjp_3496_;
}
else
{
lean_inc(v_a_3495_);
lean_dec(v___x_3494_);
v___x_3497_ = lean_box(0);
v_isShared_3498_ = v_isSharedCheck_3519_;
goto v_resetjp_3496_;
}
v_resetjp_3496_:
{
lean_object* v_fileMap_3499_; lean_object* v_env_3500_; lean_object* v_mctx_3501_; lean_object* v_options_3502_; lean_object* v_currNamespace_3503_; lean_object* v_openDecls_3504_; lean_object* v_ngen_3505_; lean_object* v___x_3507_; uint8_t v_isShared_3508_; uint8_t v_isSharedCheck_3516_; 
v_fileMap_3499_ = lean_ctor_get(v___y_3491_, 1);
v_env_3500_ = lean_ctor_get(v_a_3495_, 0);
v_mctx_3501_ = lean_ctor_get(v_a_3495_, 3);
v_options_3502_ = lean_ctor_get(v_a_3495_, 4);
v_currNamespace_3503_ = lean_ctor_get(v_a_3495_, 5);
v_openDecls_3504_ = lean_ctor_get(v_a_3495_, 6);
v_ngen_3505_ = lean_ctor_get(v_a_3495_, 7);
v_isSharedCheck_3516_ = !lean_is_exclusive(v_a_3495_);
if (v_isSharedCheck_3516_ == 0)
{
lean_object* v_unused_3517_; lean_object* v_unused_3518_; 
v_unused_3517_ = lean_ctor_get(v_a_3495_, 2);
lean_dec(v_unused_3517_);
v_unused_3518_ = lean_ctor_get(v_a_3495_, 1);
lean_dec(v_unused_3518_);
v___x_3507_ = v_a_3495_;
v_isShared_3508_ = v_isSharedCheck_3516_;
goto v_resetjp_3506_;
}
else
{
lean_inc(v_ngen_3505_);
lean_inc(v_openDecls_3504_);
lean_inc(v_currNamespace_3503_);
lean_inc(v_options_3502_);
lean_inc(v_mctx_3501_);
lean_inc(v_env_3500_);
lean_dec(v_a_3495_);
v___x_3507_ = lean_box(0);
v_isShared_3508_ = v_isSharedCheck_3516_;
goto v_resetjp_3506_;
}
v_resetjp_3506_:
{
lean_object* v___x_3509_; lean_object* v___x_3511_; 
v___x_3509_ = lean_box(0);
lean_inc_ref(v_fileMap_3499_);
if (v_isShared_3508_ == 0)
{
lean_ctor_set(v___x_3507_, 2, v_fileMap_3499_);
lean_ctor_set(v___x_3507_, 1, v___x_3509_);
v___x_3511_ = v___x_3507_;
goto v_reusejp_3510_;
}
else
{
lean_object* v_reuseFailAlloc_3515_; 
v_reuseFailAlloc_3515_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_3515_, 0, v_env_3500_);
lean_ctor_set(v_reuseFailAlloc_3515_, 1, v___x_3509_);
lean_ctor_set(v_reuseFailAlloc_3515_, 2, v_fileMap_3499_);
lean_ctor_set(v_reuseFailAlloc_3515_, 3, v_mctx_3501_);
lean_ctor_set(v_reuseFailAlloc_3515_, 4, v_options_3502_);
lean_ctor_set(v_reuseFailAlloc_3515_, 5, v_currNamespace_3503_);
lean_ctor_set(v_reuseFailAlloc_3515_, 6, v_openDecls_3504_);
lean_ctor_set(v_reuseFailAlloc_3515_, 7, v_ngen_3505_);
v___x_3511_ = v_reuseFailAlloc_3515_;
goto v_reusejp_3510_;
}
v_reusejp_3510_:
{
lean_object* v___x_3513_; 
if (v_isShared_3498_ == 0)
{
lean_ctor_set(v___x_3497_, 0, v___x_3511_);
v___x_3513_ = v___x_3497_;
goto v_reusejp_3512_;
}
else
{
lean_object* v_reuseFailAlloc_3514_; 
v_reuseFailAlloc_3514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3514_, 0, v___x_3511_);
v___x_3513_ = v_reuseFailAlloc_3514_;
goto v_reusejp_3512_;
}
v_reusejp_3512_:
{
return v___x_3513_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5___boxed(lean_object* v___y_3520_, lean_object* v___y_3521_, lean_object* v___y_3522_, lean_object* v___y_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_){
_start:
{
lean_object* v_res_3527_; 
v_res_3527_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5(v___y_3520_, v___y_3521_, v___y_3522_, v___y_3523_, v___y_3524_, v___y_3525_);
lean_dec(v___y_3525_);
lean_dec_ref(v___y_3524_);
lean_dec(v___y_3523_);
lean_dec_ref(v___y_3522_);
lean_dec(v___y_3521_);
lean_dec_ref(v___y_3520_);
return v_res_3527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0(lean_object* v___y_3528_, lean_object* v___y_3529_, lean_object* v___y_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_, lean_object* v___y_3533_){
_start:
{
lean_object* v___x_3535_; lean_object* v_a_3536_; lean_object* v___x_3538_; uint8_t v_isShared_3539_; uint8_t v_isSharedCheck_3545_; 
v___x_3535_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5(v___y_3528_, v___y_3529_, v___y_3530_, v___y_3531_, v___y_3532_, v___y_3533_);
v_a_3536_ = lean_ctor_get(v___x_3535_, 0);
v_isSharedCheck_3545_ = !lean_is_exclusive(v___x_3535_);
if (v_isSharedCheck_3545_ == 0)
{
v___x_3538_ = v___x_3535_;
v_isShared_3539_ = v_isSharedCheck_3545_;
goto v_resetjp_3537_;
}
else
{
lean_inc(v_a_3536_);
lean_dec(v___x_3535_);
v___x_3538_ = lean_box(0);
v_isShared_3539_ = v_isSharedCheck_3545_;
goto v_resetjp_3537_;
}
v_resetjp_3537_:
{
lean_object* v___x_3540_; lean_object* v___x_3541_; lean_object* v___x_3543_; 
v___x_3540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3540_, 0, v_a_3536_);
v___x_3541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3541_, 0, v___x_3540_);
if (v_isShared_3539_ == 0)
{
lean_ctor_set(v___x_3538_, 0, v___x_3541_);
v___x_3543_ = v___x_3538_;
goto v_reusejp_3542_;
}
else
{
lean_object* v_reuseFailAlloc_3544_; 
v_reuseFailAlloc_3544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3544_, 0, v___x_3541_);
v___x_3543_ = v_reuseFailAlloc_3544_;
goto v_reusejp_3542_;
}
v_reusejp_3542_:
{
return v___x_3543_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0___boxed(lean_object* v___y_3546_, lean_object* v___y_3547_, lean_object* v___y_3548_, lean_object* v___y_3549_, lean_object* v___y_3550_, lean_object* v___y_3551_, lean_object* v___y_3552_){
_start:
{
lean_object* v_res_3553_; 
v_res_3553_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___lam__0(v___y_3546_, v___y_3547_, v___y_3548_, v___y_3549_, v___y_3550_, v___y_3551_);
lean_dec(v___y_3551_);
lean_dec_ref(v___y_3550_);
lean_dec(v___y_3549_);
lean_dec_ref(v___y_3548_);
lean_dec(v___y_3547_);
lean_dec_ref(v___y_3546_);
return v_res_3553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg(lean_object* v_x_3555_, lean_object* v___y_3556_, lean_object* v___y_3557_, lean_object* v___y_3558_, lean_object* v___y_3559_, lean_object* v___y_3560_, lean_object* v___y_3561_){
_start:
{
lean_object* v___f_3563_; lean_object* v___x_3564_; 
v___f_3563_ = ((lean_object*)(lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___closed__0));
v___x_3564_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg(v_x_3555_, v___f_3563_, v___y_3556_, v___y_3557_, v___y_3558_, v___y_3559_, v___y_3560_, v___y_3561_);
return v___x_3564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_x_3565_, lean_object* v___y_3566_, lean_object* v___y_3567_, lean_object* v___y_3568_, lean_object* v___y_3569_, lean_object* v___y_3570_, lean_object* v___y_3571_, lean_object* v___y_3572_){
_start:
{
lean_object* v_res_3573_; 
v_res_3573_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg(v_x_3565_, v___y_3566_, v___y_3567_, v___y_3568_, v___y_3569_, v___y_3570_, v___y_3571_);
lean_dec(v___y_3571_);
lean_dec_ref(v___y_3570_);
lean_dec(v___y_3569_);
lean_dec_ref(v___y_3568_);
lean_dec(v___y_3567_);
lean_dec_ref(v___y_3566_);
return v_res_3573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_3574_, lean_object* v_x_3575_, lean_object* v___y_3576_, lean_object* v___y_3577_, lean_object* v___y_3578_, lean_object* v___y_3579_, lean_object* v___y_3580_, lean_object* v___y_3581_){
_start:
{
lean_object* v___x_3583_; 
v___x_3583_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___redArg(v_x_3575_, v___y_3576_, v___y_3577_, v___y_3578_, v___y_3579_, v___y_3580_, v___y_3581_);
return v___x_3583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_3584_, lean_object* v_x_3585_, lean_object* v___y_3586_, lean_object* v___y_3587_, lean_object* v___y_3588_, lean_object* v___y_3589_, lean_object* v___y_3590_, lean_object* v___y_3591_, lean_object* v___y_3592_){
_start:
{
lean_object* v_res_3593_; 
v_res_3593_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2(v_00_u03b1_3584_, v_x_3585_, v___y_3586_, v___y_3587_, v___y_3588_, v___y_3589_, v___y_3590_, v___y_3591_);
lean_dec(v___y_3591_);
lean_dec_ref(v___y_3590_);
lean_dec(v___y_3589_);
lean_dec_ref(v___y_3588_);
lean_dec(v___y_3587_);
lean_dec_ref(v___y_3586_);
return v_res_3593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1(uint8_t v___x_3594_, lean_object* v_v_3595_, lean_object* v___x_3596_, uint8_t v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_, lean_object* v___y_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_){
_start:
{
lean_object* v_declName_x3f_3605_; lean_object* v_macroStack_3606_; uint8_t v_mayPostpone_3607_; uint8_t v_errToSorry_3608_; lean_object* v_autoBoundImplicitContext_3609_; lean_object* v_autoBoundImplicitForbidden_3610_; lean_object* v_sectionVars_3611_; lean_object* v_sectionFVars_3612_; uint8_t v_implicitLambda_3613_; uint8_t v_heedElabAsElim_3614_; uint8_t v_isNoncomputableSection_3615_; uint8_t v_isMetaSection_3616_; uint8_t v_inPattern_3617_; lean_object* v_tacSnap_x3f_3618_; uint8_t v_saveRecAppSyntax_3619_; uint8_t v_holesAsSyntheticOpaque_3620_; uint8_t v_checkDeprecated_3621_; lean_object* v_fixedTermElabs_3622_; lean_object* v___x_3624_; uint8_t v_isShared_3625_; uint8_t v_isSharedCheck_3656_; 
v_declName_x3f_3605_ = lean_ctor_get(v___y_3598_, 0);
v_macroStack_3606_ = lean_ctor_get(v___y_3598_, 1);
v_mayPostpone_3607_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8);
v_errToSorry_3608_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 1);
v_autoBoundImplicitContext_3609_ = lean_ctor_get(v___y_3598_, 2);
v_autoBoundImplicitForbidden_3610_ = lean_ctor_get(v___y_3598_, 3);
v_sectionVars_3611_ = lean_ctor_get(v___y_3598_, 4);
v_sectionFVars_3612_ = lean_ctor_get(v___y_3598_, 5);
v_implicitLambda_3613_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 2);
v_heedElabAsElim_3614_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_3615_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 4);
v_isMetaSection_3616_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 5);
v_inPattern_3617_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_3618_ = lean_ctor_get(v___y_3598_, 6);
v_saveRecAppSyntax_3619_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_3620_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 9);
v_checkDeprecated_3621_ = lean_ctor_get_uint8(v___y_3598_, sizeof(void*)*8 + 10);
v_fixedTermElabs_3622_ = lean_ctor_get(v___y_3598_, 7);
v_isSharedCheck_3656_ = !lean_is_exclusive(v___y_3598_);
if (v_isSharedCheck_3656_ == 0)
{
v___x_3624_ = v___y_3598_;
v_isShared_3625_ = v_isSharedCheck_3656_;
goto v_resetjp_3623_;
}
else
{
lean_inc(v_fixedTermElabs_3622_);
lean_inc(v_tacSnap_x3f_3618_);
lean_inc(v_sectionFVars_3612_);
lean_inc(v_sectionVars_3611_);
lean_inc(v_autoBoundImplicitForbidden_3610_);
lean_inc(v_autoBoundImplicitContext_3609_);
lean_inc(v_macroStack_3606_);
lean_inc(v_declName_x3f_3605_);
lean_dec(v___y_3598_);
v___x_3624_ = lean_box(0);
v_isShared_3625_ = v_isSharedCheck_3656_;
goto v_resetjp_3623_;
}
v_resetjp_3623_:
{
lean_object* v___x_3627_; 
if (v_isShared_3625_ == 0)
{
v___x_3627_ = v___x_3624_;
goto v_reusejp_3626_;
}
else
{
lean_object* v_reuseFailAlloc_3655_; 
v_reuseFailAlloc_3655_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v_reuseFailAlloc_3655_, 0, v_declName_x3f_3605_);
lean_ctor_set(v_reuseFailAlloc_3655_, 1, v_macroStack_3606_);
lean_ctor_set(v_reuseFailAlloc_3655_, 2, v_autoBoundImplicitContext_3609_);
lean_ctor_set(v_reuseFailAlloc_3655_, 3, v_autoBoundImplicitForbidden_3610_);
lean_ctor_set(v_reuseFailAlloc_3655_, 4, v_sectionVars_3611_);
lean_ctor_set(v_reuseFailAlloc_3655_, 5, v_sectionFVars_3612_);
lean_ctor_set(v_reuseFailAlloc_3655_, 6, v_tacSnap_x3f_3618_);
lean_ctor_set(v_reuseFailAlloc_3655_, 7, v_fixedTermElabs_3622_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8, v_mayPostpone_3607_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 1, v_errToSorry_3608_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 2, v_implicitLambda_3613_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 3, v_heedElabAsElim_3614_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 4, v_isNoncomputableSection_3615_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 5, v_isMetaSection_3616_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 7, v_inPattern_3617_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_3619_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_3620_);
lean_ctor_set_uint8(v_reuseFailAlloc_3655_, sizeof(void*)*8 + 10, v_checkDeprecated_3621_);
v___x_3627_ = v_reuseFailAlloc_3655_;
goto v_reusejp_3626_;
}
v_reusejp_3626_:
{
lean_object* v___x_3628_; 
lean_ctor_set_uint8(v___x_3627_, sizeof(void*)*8 + 6, v___x_3594_);
v___x_3628_ = l_Lean_Elab_Term_elabTerm(v_v_3595_, v___x_3596_, v___x_3594_, v___x_3594_, v___x_3627_, v___y_3599_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_);
lean_dec_ref(v___x_3627_);
if (lean_obj_tag(v___x_3628_) == 0)
{
lean_object* v_a_3629_; lean_object* v_lctx_3630_; lean_object* v___x_3631_; uint8_t v___x_3632_; lean_object* v___x_3633_; 
v_a_3629_ = lean_ctor_get(v___x_3628_, 0);
lean_inc(v_a_3629_);
lean_dec_ref_known(v___x_3628_, 1);
v_lctx_3630_ = lean_ctor_get(v___y_3600_, 2);
v___x_3631_ = l_Lean_LocalContext_getFVars(v_lctx_3630_);
v___x_3632_ = 1;
v___x_3633_ = l_Lean_Meta_mkLambdaFVars(v___x_3631_, v_a_3629_, v___y_3597_, v___x_3594_, v___y_3597_, v___x_3594_, v___x_3632_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_);
lean_dec_ref(v___x_3631_);
if (lean_obj_tag(v___x_3633_) == 0)
{
lean_object* v_a_3634_; lean_object* v___x_3635_; lean_object* v___x_3636_; 
v_a_3634_ = lean_ctor_get(v___x_3633_, 0);
lean_inc(v_a_3634_);
lean_dec_ref_known(v___x_3633_, 1);
v___x_3635_ = lean_box(0);
v___x_3636_ = l_Lean_Meta_lambdaMetaTelescope(v_a_3634_, v___x_3635_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_);
lean_dec(v_a_3634_);
if (lean_obj_tag(v___x_3636_) == 0)
{
lean_object* v_a_3637_; lean_object* v___x_3639_; uint8_t v_isShared_3640_; uint8_t v_isSharedCheck_3646_; 
v_a_3637_ = lean_ctor_get(v___x_3636_, 0);
v_isSharedCheck_3646_ = !lean_is_exclusive(v___x_3636_);
if (v_isSharedCheck_3646_ == 0)
{
v___x_3639_ = v___x_3636_;
v_isShared_3640_ = v_isSharedCheck_3646_;
goto v_resetjp_3638_;
}
else
{
lean_inc(v_a_3637_);
lean_dec(v___x_3636_);
v___x_3639_ = lean_box(0);
v_isShared_3640_ = v_isSharedCheck_3646_;
goto v_resetjp_3638_;
}
v_resetjp_3638_:
{
lean_object* v_snd_3641_; lean_object* v_snd_3642_; lean_object* v___x_3644_; 
v_snd_3641_ = lean_ctor_get(v_a_3637_, 1);
lean_inc(v_snd_3641_);
lean_dec(v_a_3637_);
v_snd_3642_ = lean_ctor_get(v_snd_3641_, 1);
lean_inc(v_snd_3642_);
lean_dec(v_snd_3641_);
if (v_isShared_3640_ == 0)
{
lean_ctor_set(v___x_3639_, 0, v_snd_3642_);
v___x_3644_ = v___x_3639_;
goto v_reusejp_3643_;
}
else
{
lean_object* v_reuseFailAlloc_3645_; 
v_reuseFailAlloc_3645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3645_, 0, v_snd_3642_);
v___x_3644_ = v_reuseFailAlloc_3645_;
goto v_reusejp_3643_;
}
v_reusejp_3643_:
{
return v___x_3644_;
}
}
}
else
{
lean_object* v_a_3647_; lean_object* v___x_3649_; uint8_t v_isShared_3650_; uint8_t v_isSharedCheck_3654_; 
v_a_3647_ = lean_ctor_get(v___x_3636_, 0);
v_isSharedCheck_3654_ = !lean_is_exclusive(v___x_3636_);
if (v_isSharedCheck_3654_ == 0)
{
v___x_3649_ = v___x_3636_;
v_isShared_3650_ = v_isSharedCheck_3654_;
goto v_resetjp_3648_;
}
else
{
lean_inc(v_a_3647_);
lean_dec(v___x_3636_);
v___x_3649_ = lean_box(0);
v_isShared_3650_ = v_isSharedCheck_3654_;
goto v_resetjp_3648_;
}
v_resetjp_3648_:
{
lean_object* v___x_3652_; 
if (v_isShared_3650_ == 0)
{
v___x_3652_ = v___x_3649_;
goto v_reusejp_3651_;
}
else
{
lean_object* v_reuseFailAlloc_3653_; 
v_reuseFailAlloc_3653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3653_, 0, v_a_3647_);
v___x_3652_ = v_reuseFailAlloc_3653_;
goto v_reusejp_3651_;
}
v_reusejp_3651_:
{
return v___x_3652_;
}
}
}
}
else
{
return v___x_3633_;
}
}
else
{
return v___x_3628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1___boxed(lean_object* v___x_3657_, lean_object* v_v_3658_, lean_object* v___x_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_, lean_object* v___y_3663_, lean_object* v___y_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_){
_start:
{
uint8_t v___x_20553__boxed_3668_; uint8_t v___y_20555__boxed_3669_; lean_object* v_res_3670_; 
v___x_20553__boxed_3668_ = lean_unbox(v___x_3657_);
v___y_20555__boxed_3669_ = lean_unbox(v___y_3660_);
v_res_3670_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1(v___x_20553__boxed_3668_, v_v_3658_, v___x_3659_, v___y_20555__boxed_3669_, v___y_3661_, v___y_3662_, v___y_3663_, v___y_3664_, v___y_3665_, v___y_3666_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
lean_dec(v___y_3664_);
lean_dec_ref(v___y_3663_);
lean_dec(v___y_3662_);
return v_res_3670_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0(uint8_t v___y_3671_, lean_object* v_x_3672_){
_start:
{
return v___y_3671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0___boxed(lean_object* v___y_3673_, lean_object* v_x_3674_){
_start:
{
uint8_t v___y_20652__boxed_3675_; uint8_t v_res_3676_; lean_object* v_r_3677_; 
v___y_20652__boxed_3675_ = lean_unbox(v___y_3673_);
v_res_3676_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0(v___y_20652__boxed_3675_, v_x_3674_);
lean_dec(v_x_3674_);
v_r_3677_ = lean_box(v_res_3676_);
return v_r_3677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5(uint8_t v___x_3683_, uint8_t v___y_3684_, size_t v_sz_3685_, size_t v_i_3686_, lean_object* v_bs_3687_, lean_object* v___y_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_){
_start:
{
uint8_t v___x_3693_; 
v___x_3693_ = lean_usize_dec_lt(v_i_3686_, v_sz_3685_);
if (v___x_3693_ == 0)
{
lean_object* v___x_3694_; 
v___x_3694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3694_, 0, v_bs_3687_);
return v___x_3694_;
}
else
{
lean_object* v___x_3695_; lean_object* v___f_3696_; lean_object* v___x_3697_; lean_object* v_v_3698_; lean_object* v___x_3699_; lean_object* v___x_3700_; lean_object* v___x_3701_; lean_object* v___f_3702_; lean_object* v___x_3703_; lean_object* v___x_3704_; lean_object* v___x_3705_; lean_object* v___x_3706_; lean_object* v___x_3707_; lean_object* v___x_3708_; lean_object* v___x_3709_; lean_object* v___x_3710_; 
v___x_3695_ = lean_box(v___y_3684_);
v___f_3696_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3696_, 0, v___x_3695_);
v___x_3697_ = lean_box(0);
v_v_3698_ = lean_array_uget_borrowed(v_bs_3687_, v_i_3686_);
v___x_3699_ = lean_box(0);
v___x_3700_ = lean_box(v___x_3683_);
v___x_3701_ = lean_box(v___y_3684_);
lean_inc(v_v_3698_);
v___f_3702_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___lam__1___boxed), 11, 4);
lean_closure_set(v___f_3702_, 0, v___x_3700_);
lean_closure_set(v___f_3702_, 1, v_v_3698_);
lean_closure_set(v___f_3702_, 2, v___x_3699_);
lean_closure_set(v___f_3702_, 3, v___x_3701_);
v___x_3703_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_withAutoBoundImplicit___boxed), 9, 2);
lean_closure_set(v___x_3703_, 0, lean_box(0));
lean_closure_set(v___x_3703_, 1, v___f_3702_);
v___x_3704_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2___boxed), 9, 2);
lean_closure_set(v___x_3704_, 0, lean_box(0));
lean_closure_set(v___x_3704_, 1, v___x_3703_);
v___x_3705_ = lean_box(1);
v___x_3706_ = lean_unsigned_to_nat(0u);
v___x_3707_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__0));
v___x_3708_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_3708_, 0, v___x_3699_);
lean_ctor_set(v___x_3708_, 1, v___x_3697_);
lean_ctor_set(v___x_3708_, 2, v___x_3699_);
lean_ctor_set(v___x_3708_, 3, v___f_3696_);
lean_ctor_set(v___x_3708_, 4, v___x_3705_);
lean_ctor_set(v___x_3708_, 5, v___x_3705_);
lean_ctor_set(v___x_3708_, 6, v___x_3699_);
lean_ctor_set(v___x_3708_, 7, v___x_3707_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8, v___x_3683_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 1, v___x_3683_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 2, v___x_3683_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 3, v___x_3683_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 4, v___y_3684_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 5, v___y_3684_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 6, v___y_3684_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 7, v___y_3684_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 8, v___x_3683_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 9, v___y_3684_);
lean_ctor_set_uint8(v___x_3708_, sizeof(void*)*8 + 10, v___x_3683_);
v___x_3709_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___closed__1));
v___x_3710_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_3704_, v___x_3708_, v___x_3709_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3710_) == 0)
{
lean_object* v_a_3711_; lean_object* v_fst_3712_; lean_object* v___x_3713_; 
v_a_3711_ = lean_ctor_get(v___x_3710_, 0);
lean_inc(v_a_3711_);
lean_dec_ref_known(v___x_3710_, 1);
v_fst_3712_ = lean_ctor_get(v_a_3711_, 0);
lean_inc(v_fst_3712_);
lean_dec(v_a_3711_);
v___x_3713_ = l_Lean_Meta_DiscrTree_mkPath(v_fst_3712_, v___y_3684_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3713_) == 0)
{
lean_object* v_a_3714_; lean_object* v_bs_x27_3715_; size_t v___x_3716_; size_t v___x_3717_; lean_object* v___x_3718_; 
v_a_3714_ = lean_ctor_get(v___x_3713_, 0);
lean_inc(v_a_3714_);
lean_dec_ref_known(v___x_3713_, 1);
v_bs_x27_3715_ = lean_array_uset(v_bs_3687_, v_i_3686_, v___x_3706_);
v___x_3716_ = ((size_t)1ULL);
v___x_3717_ = lean_usize_add(v_i_3686_, v___x_3716_);
v___x_3718_ = lean_array_uset(v_bs_x27_3715_, v_i_3686_, v_a_3714_);
v_i_3686_ = v___x_3717_;
v_bs_3687_ = v___x_3718_;
goto _start;
}
else
{
lean_object* v_a_3720_; lean_object* v___x_3722_; uint8_t v_isShared_3723_; uint8_t v_isSharedCheck_3727_; 
lean_dec_ref(v_bs_3687_);
v_a_3720_ = lean_ctor_get(v___x_3713_, 0);
v_isSharedCheck_3727_ = !lean_is_exclusive(v___x_3713_);
if (v_isSharedCheck_3727_ == 0)
{
v___x_3722_ = v___x_3713_;
v_isShared_3723_ = v_isSharedCheck_3727_;
goto v_resetjp_3721_;
}
else
{
lean_inc(v_a_3720_);
lean_dec(v___x_3713_);
v___x_3722_ = lean_box(0);
v_isShared_3723_ = v_isSharedCheck_3727_;
goto v_resetjp_3721_;
}
v_resetjp_3721_:
{
lean_object* v___x_3725_; 
if (v_isShared_3723_ == 0)
{
v___x_3725_ = v___x_3722_;
goto v_reusejp_3724_;
}
else
{
lean_object* v_reuseFailAlloc_3726_; 
v_reuseFailAlloc_3726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3726_, 0, v_a_3720_);
v___x_3725_ = v_reuseFailAlloc_3726_;
goto v_reusejp_3724_;
}
v_reusejp_3724_:
{
return v___x_3725_;
}
}
}
}
else
{
lean_object* v_a_3728_; lean_object* v___x_3730_; uint8_t v_isShared_3731_; uint8_t v_isSharedCheck_3735_; 
lean_dec_ref(v_bs_3687_);
v_a_3728_ = lean_ctor_get(v___x_3710_, 0);
v_isSharedCheck_3735_ = !lean_is_exclusive(v___x_3710_);
if (v_isSharedCheck_3735_ == 0)
{
v___x_3730_ = v___x_3710_;
v_isShared_3731_ = v_isSharedCheck_3735_;
goto v_resetjp_3729_;
}
else
{
lean_inc(v_a_3728_);
lean_dec(v___x_3710_);
v___x_3730_ = lean_box(0);
v_isShared_3731_ = v_isSharedCheck_3735_;
goto v_resetjp_3729_;
}
v_resetjp_3729_:
{
lean_object* v___x_3733_; 
if (v_isShared_3731_ == 0)
{
v___x_3733_ = v___x_3730_;
goto v_reusejp_3732_;
}
else
{
lean_object* v_reuseFailAlloc_3734_; 
v_reuseFailAlloc_3734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3734_, 0, v_a_3728_);
v___x_3733_ = v_reuseFailAlloc_3734_;
goto v_reusejp_3732_;
}
v_reusejp_3732_:
{
return v___x_3733_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5___boxed(lean_object* v___x_3736_, lean_object* v___y_3737_, lean_object* v_sz_3738_, lean_object* v_i_3739_, lean_object* v_bs_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_, lean_object* v___y_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_){
_start:
{
uint8_t v___x_20678__boxed_3746_; uint8_t v___y_20679__boxed_3747_; size_t v_sz_boxed_3748_; size_t v_i_boxed_3749_; lean_object* v_res_3750_; 
v___x_20678__boxed_3746_ = lean_unbox(v___x_3736_);
v___y_20679__boxed_3747_ = lean_unbox(v___y_3737_);
v_sz_boxed_3748_ = lean_unbox_usize(v_sz_3738_);
lean_dec(v_sz_3738_);
v_i_boxed_3749_ = lean_unbox_usize(v_i_3739_);
lean_dec(v_i_3739_);
v_res_3750_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5(v___x_20678__boxed_3746_, v___y_20679__boxed_3747_, v_sz_boxed_3748_, v_i_boxed_3749_, v_bs_3740_, v___y_3741_, v___y_3742_, v___y_3743_, v___y_3744_);
lean_dec(v___y_3744_);
lean_dec_ref(v___y_3743_);
lean_dec(v___y_3742_);
lean_dec_ref(v___y_3741_);
return v_res_3750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9(lean_object* v_cls_3751_, lean_object* v_msg_3752_, lean_object* v___y_3753_, lean_object* v___y_3754_){
_start:
{
lean_object* v_ref_3756_; lean_object* v___x_3757_; lean_object* v_a_3758_; lean_object* v___x_3760_; uint8_t v_isShared_3761_; uint8_t v_isSharedCheck_3802_; 
v_ref_3756_ = lean_ctor_get(v___y_3753_, 5);
v___x_3757_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12(v_msg_3752_, v___y_3753_, v___y_3754_);
v_a_3758_ = lean_ctor_get(v___x_3757_, 0);
v_isSharedCheck_3802_ = !lean_is_exclusive(v___x_3757_);
if (v_isSharedCheck_3802_ == 0)
{
v___x_3760_ = v___x_3757_;
v_isShared_3761_ = v_isSharedCheck_3802_;
goto v_resetjp_3759_;
}
else
{
lean_inc(v_a_3758_);
lean_dec(v___x_3757_);
v___x_3760_ = lean_box(0);
v_isShared_3761_ = v_isSharedCheck_3802_;
goto v_resetjp_3759_;
}
v_resetjp_3759_:
{
lean_object* v___x_3762_; lean_object* v_traceState_3763_; lean_object* v_env_3764_; lean_object* v_nextMacroScope_3765_; lean_object* v_ngen_3766_; lean_object* v_auxDeclNGen_3767_; lean_object* v_cache_3768_; lean_object* v_messages_3769_; lean_object* v_infoState_3770_; lean_object* v_snapshotTasks_3771_; lean_object* v___x_3773_; uint8_t v_isShared_3774_; uint8_t v_isSharedCheck_3801_; 
v___x_3762_ = lean_st_ref_take(v___y_3754_);
v_traceState_3763_ = lean_ctor_get(v___x_3762_, 4);
v_env_3764_ = lean_ctor_get(v___x_3762_, 0);
v_nextMacroScope_3765_ = lean_ctor_get(v___x_3762_, 1);
v_ngen_3766_ = lean_ctor_get(v___x_3762_, 2);
v_auxDeclNGen_3767_ = lean_ctor_get(v___x_3762_, 3);
v_cache_3768_ = lean_ctor_get(v___x_3762_, 5);
v_messages_3769_ = lean_ctor_get(v___x_3762_, 6);
v_infoState_3770_ = lean_ctor_get(v___x_3762_, 7);
v_snapshotTasks_3771_ = lean_ctor_get(v___x_3762_, 8);
v_isSharedCheck_3801_ = !lean_is_exclusive(v___x_3762_);
if (v_isSharedCheck_3801_ == 0)
{
v___x_3773_ = v___x_3762_;
v_isShared_3774_ = v_isSharedCheck_3801_;
goto v_resetjp_3772_;
}
else
{
lean_inc(v_snapshotTasks_3771_);
lean_inc(v_infoState_3770_);
lean_inc(v_messages_3769_);
lean_inc(v_cache_3768_);
lean_inc(v_traceState_3763_);
lean_inc(v_auxDeclNGen_3767_);
lean_inc(v_ngen_3766_);
lean_inc(v_nextMacroScope_3765_);
lean_inc(v_env_3764_);
lean_dec(v___x_3762_);
v___x_3773_ = lean_box(0);
v_isShared_3774_ = v_isSharedCheck_3801_;
goto v_resetjp_3772_;
}
v_resetjp_3772_:
{
uint64_t v_tid_3775_; lean_object* v_traces_3776_; lean_object* v___x_3778_; uint8_t v_isShared_3779_; uint8_t v_isSharedCheck_3800_; 
v_tid_3775_ = lean_ctor_get_uint64(v_traceState_3763_, sizeof(void*)*1);
v_traces_3776_ = lean_ctor_get(v_traceState_3763_, 0);
v_isSharedCheck_3800_ = !lean_is_exclusive(v_traceState_3763_);
if (v_isSharedCheck_3800_ == 0)
{
v___x_3778_ = v_traceState_3763_;
v_isShared_3779_ = v_isSharedCheck_3800_;
goto v_resetjp_3777_;
}
else
{
lean_inc(v_traces_3776_);
lean_dec(v_traceState_3763_);
v___x_3778_ = lean_box(0);
v_isShared_3779_ = v_isSharedCheck_3800_;
goto v_resetjp_3777_;
}
v_resetjp_3777_:
{
lean_object* v___x_3780_; double v___x_3781_; uint8_t v___x_3782_; lean_object* v___x_3783_; lean_object* v___x_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___x_3790_; 
v___x_3780_ = lean_box(0);
v___x_3781_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__0);
v___x_3782_ = 0;
v___x_3783_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__1));
v___x_3784_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3784_, 0, v_cls_3751_);
lean_ctor_set(v___x_3784_, 1, v___x_3780_);
lean_ctor_set(v___x_3784_, 2, v___x_3783_);
lean_ctor_set_float(v___x_3784_, sizeof(void*)*3, v___x_3781_);
lean_ctor_set_float(v___x_3784_, sizeof(void*)*3 + 8, v___x_3781_);
lean_ctor_set_uint8(v___x_3784_, sizeof(void*)*3 + 16, v___x_3782_);
v___x_3785_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_NormNum_derive_spec__0___closed__2));
v___x_3786_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3786_, 0, v___x_3784_);
lean_ctor_set(v___x_3786_, 1, v_a_3758_);
lean_ctor_set(v___x_3786_, 2, v___x_3785_);
lean_inc(v_ref_3756_);
v___x_3787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3787_, 0, v_ref_3756_);
lean_ctor_set(v___x_3787_, 1, v___x_3786_);
v___x_3788_ = l_Lean_PersistentArray_push___redArg(v_traces_3776_, v___x_3787_);
if (v_isShared_3779_ == 0)
{
lean_ctor_set(v___x_3778_, 0, v___x_3788_);
v___x_3790_ = v___x_3778_;
goto v_reusejp_3789_;
}
else
{
lean_object* v_reuseFailAlloc_3799_; 
v_reuseFailAlloc_3799_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3799_, 0, v___x_3788_);
lean_ctor_set_uint64(v_reuseFailAlloc_3799_, sizeof(void*)*1, v_tid_3775_);
v___x_3790_ = v_reuseFailAlloc_3799_;
goto v_reusejp_3789_;
}
v_reusejp_3789_:
{
lean_object* v___x_3792_; 
if (v_isShared_3774_ == 0)
{
lean_ctor_set(v___x_3773_, 4, v___x_3790_);
v___x_3792_ = v___x_3773_;
goto v_reusejp_3791_;
}
else
{
lean_object* v_reuseFailAlloc_3798_; 
v_reuseFailAlloc_3798_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3798_, 0, v_env_3764_);
lean_ctor_set(v_reuseFailAlloc_3798_, 1, v_nextMacroScope_3765_);
lean_ctor_set(v_reuseFailAlloc_3798_, 2, v_ngen_3766_);
lean_ctor_set(v_reuseFailAlloc_3798_, 3, v_auxDeclNGen_3767_);
lean_ctor_set(v_reuseFailAlloc_3798_, 4, v___x_3790_);
lean_ctor_set(v_reuseFailAlloc_3798_, 5, v_cache_3768_);
lean_ctor_set(v_reuseFailAlloc_3798_, 6, v_messages_3769_);
lean_ctor_set(v_reuseFailAlloc_3798_, 7, v_infoState_3770_);
lean_ctor_set(v_reuseFailAlloc_3798_, 8, v_snapshotTasks_3771_);
v___x_3792_ = v_reuseFailAlloc_3798_;
goto v_reusejp_3791_;
}
v_reusejp_3791_:
{
lean_object* v___x_3793_; lean_object* v___x_3794_; lean_object* v___x_3796_; 
v___x_3793_ = lean_st_ref_set(v___y_3754_, v___x_3792_);
v___x_3794_ = lean_box(0);
if (v_isShared_3761_ == 0)
{
lean_ctor_set(v___x_3760_, 0, v___x_3794_);
v___x_3796_ = v___x_3760_;
goto v_reusejp_3795_;
}
else
{
lean_object* v_reuseFailAlloc_3797_; 
v_reuseFailAlloc_3797_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3797_, 0, v___x_3794_);
v___x_3796_ = v_reuseFailAlloc_3797_;
goto v_reusejp_3795_;
}
v_reusejp_3795_:
{
return v___x_3796_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9___boxed(lean_object* v_cls_3803_, lean_object* v_msg_3804_, lean_object* v___y_3805_, lean_object* v___y_3806_, lean_object* v___y_3807_){
_start:
{
lean_object* v_res_3808_; 
v_res_3808_ = lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9(v_cls_3803_, v_msg_3804_, v___y_3805_, v___y_3806_);
lean_dec(v___y_3806_);
lean_dec_ref(v___y_3805_);
return v_res_3808_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2(void){
_start:
{
lean_object* v_cls_3812_; lean_object* v___x_3813_; lean_object* v___x_3814_; 
v_cls_3812_ = ((lean_object*)(lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__1));
v___x_3813_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_NormNum_derive_spec__2___closed__2));
v___x_3814_ = l_Lean_Name_append(v___x_3813_, v_cls_3812_);
return v___x_3814_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4(void){
_start:
{
lean_object* v___x_3816_; lean_object* v___x_3817_; 
v___x_3816_ = ((lean_object*)(lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__3));
v___x_3817_ = l_Lean_stringToMessageData(v___x_3816_);
return v___x_3817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4(lean_object* v___y_3818_, lean_object* v___y_3819_){
_start:
{
lean_object* v___x_3821_; lean_object* v_env_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___y_3826_; lean_object* v___x_3851_; lean_object* v___x_3852_; uint8_t v___x_3853_; 
v___x_3821_ = lean_st_ref_get(v___y_3819_);
v_env_3822_ = lean_ctor_get(v___x_3821_, 0);
lean_inc_ref(v_env_3822_);
lean_dec(v___x_3821_);
v___x_3823_ = lean_box(0);
v___x_3824_ = l___private_Lean_ExtraModUses_0__Lean_isExtraRevModUseExt;
v___x_3851_ = lean_box(1);
v___x_3852_ = l_Lean_SimplePersistentEnvExtension_getEntries___redArg(v___x_3823_, v___x_3824_, v_env_3822_, v___x_3851_);
v___x_3853_ = l_List_isEmpty___redArg(v___x_3852_);
lean_dec(v___x_3852_);
if (v___x_3853_ == 0)
{
lean_object* v___x_3854_; 
v___x_3854_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3854_, 0, v___x_3823_);
return v___x_3854_;
}
else
{
lean_object* v_options_3855_; uint8_t v_hasTrace_3856_; 
v_options_3855_ = lean_ctor_get(v___y_3818_, 2);
v_hasTrace_3856_ = lean_ctor_get_uint8(v_options_3855_, sizeof(void*)*1);
if (v_hasTrace_3856_ == 0)
{
v___y_3826_ = v___y_3819_;
goto v___jp_3825_;
}
else
{
lean_object* v_inheritedTraceOptions_3857_; lean_object* v_cls_3858_; lean_object* v___x_3859_; uint8_t v___x_3860_; 
v_inheritedTraceOptions_3857_ = lean_ctor_get(v___y_3818_, 13);
v_cls_3858_ = ((lean_object*)(lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__1));
v___x_3859_ = lean_obj_once(&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2, &lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2_once, _init_lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__2);
v___x_3860_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3857_, v_options_3855_, v___x_3859_);
if (v___x_3860_ == 0)
{
v___y_3826_ = v___y_3819_;
goto v___jp_3825_;
}
else
{
lean_object* v___x_3861_; lean_object* v___x_3862_; 
v___x_3861_ = lean_obj_once(&lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4, &lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4_once, _init_lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___closed__4);
v___x_3862_ = lp_mathlib_Lean_addTrace___at___00Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4_spec__9(v_cls_3858_, v___x_3861_, v___y_3818_, v___y_3819_);
if (lean_obj_tag(v___x_3862_) == 0)
{
lean_dec_ref_known(v___x_3862_, 1);
v___y_3826_ = v___y_3819_;
goto v___jp_3825_;
}
else
{
return v___x_3862_;
}
}
}
}
v___jp_3825_:
{
lean_object* v___x_3827_; lean_object* v_toEnvExtension_3828_; lean_object* v_env_3829_; lean_object* v_nextMacroScope_3830_; lean_object* v_ngen_3831_; lean_object* v_auxDeclNGen_3832_; lean_object* v_traceState_3833_; lean_object* v_messages_3834_; lean_object* v_infoState_3835_; lean_object* v_snapshotTasks_3836_; lean_object* v___x_3838_; uint8_t v_isShared_3839_; uint8_t v_isSharedCheck_3849_; 
v___x_3827_ = lean_st_ref_take(v___y_3826_);
v_toEnvExtension_3828_ = lean_ctor_get(v___x_3824_, 0);
v_env_3829_ = lean_ctor_get(v___x_3827_, 0);
v_nextMacroScope_3830_ = lean_ctor_get(v___x_3827_, 1);
v_ngen_3831_ = lean_ctor_get(v___x_3827_, 2);
v_auxDeclNGen_3832_ = lean_ctor_get(v___x_3827_, 3);
v_traceState_3833_ = lean_ctor_get(v___x_3827_, 4);
v_messages_3834_ = lean_ctor_get(v___x_3827_, 6);
v_infoState_3835_ = lean_ctor_get(v___x_3827_, 7);
v_snapshotTasks_3836_ = lean_ctor_get(v___x_3827_, 8);
v_isSharedCheck_3849_ = !lean_is_exclusive(v___x_3827_);
if (v_isSharedCheck_3849_ == 0)
{
lean_object* v_unused_3850_; 
v_unused_3850_ = lean_ctor_get(v___x_3827_, 5);
lean_dec(v_unused_3850_);
v___x_3838_ = v___x_3827_;
v_isShared_3839_ = v_isSharedCheck_3849_;
goto v_resetjp_3837_;
}
else
{
lean_inc(v_snapshotTasks_3836_);
lean_inc(v_infoState_3835_);
lean_inc(v_messages_3834_);
lean_inc(v_traceState_3833_);
lean_inc(v_auxDeclNGen_3832_);
lean_inc(v_ngen_3831_);
lean_inc(v_nextMacroScope_3830_);
lean_inc(v_env_3829_);
lean_dec(v___x_3827_);
v___x_3838_ = lean_box(0);
v_isShared_3839_ = v_isSharedCheck_3849_;
goto v_resetjp_3837_;
}
v_resetjp_3837_:
{
lean_object* v_asyncMode_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v___x_3845_; 
v_asyncMode_3840_ = lean_ctor_get(v_toEnvExtension_3828_, 2);
v___x_3841_ = lean_box(0);
v___x_3842_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_3824_, v_env_3829_, v___x_3823_, v_asyncMode_3840_, v___x_3841_);
v___x_3843_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg___closed__2);
if (v_isShared_3839_ == 0)
{
lean_ctor_set(v___x_3838_, 5, v___x_3843_);
lean_ctor_set(v___x_3838_, 0, v___x_3842_);
v___x_3845_ = v___x_3838_;
goto v_reusejp_3844_;
}
else
{
lean_object* v_reuseFailAlloc_3848_; 
v_reuseFailAlloc_3848_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3848_, 0, v___x_3842_);
lean_ctor_set(v_reuseFailAlloc_3848_, 1, v_nextMacroScope_3830_);
lean_ctor_set(v_reuseFailAlloc_3848_, 2, v_ngen_3831_);
lean_ctor_set(v_reuseFailAlloc_3848_, 3, v_auxDeclNGen_3832_);
lean_ctor_set(v_reuseFailAlloc_3848_, 4, v_traceState_3833_);
lean_ctor_set(v_reuseFailAlloc_3848_, 5, v___x_3843_);
lean_ctor_set(v_reuseFailAlloc_3848_, 6, v_messages_3834_);
lean_ctor_set(v_reuseFailAlloc_3848_, 7, v_infoState_3835_);
lean_ctor_set(v_reuseFailAlloc_3848_, 8, v_snapshotTasks_3836_);
v___x_3845_ = v_reuseFailAlloc_3848_;
goto v_reusejp_3844_;
}
v_reusejp_3844_:
{
lean_object* v___x_3846_; lean_object* v___x_3847_; 
v___x_3846_ = lean_st_ref_set(v___y_3826_, v___x_3845_);
v___x_3847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3847_, 0, v___x_3823_);
return v___x_3847_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4___boxed(lean_object* v___y_3863_, lean_object* v___y_3864_, lean_object* v___y_3865_){
_start:
{
lean_object* v_res_3866_; 
v_res_3866_ = lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4(v___y_3863_, v___y_3864_);
lean_dec(v___y_3864_);
lean_dec_ref(v___y_3863_);
return v_res_3866_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3867_; 
v___x_3867_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3867_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3868_; lean_object* v___x_3869_; 
v___x_3868_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3869_, 0, v___x_3868_);
return v___x_3869_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3870_; lean_object* v___x_3871_; 
v___x_3870_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3871_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_3871_, 0, v___x_3870_);
lean_ctor_set(v___x_3871_, 1, v___x_3870_);
lean_ctor_set(v___x_3871_, 2, v___x_3870_);
lean_ctor_set(v___x_3871_, 3, v___x_3870_);
lean_ctor_set(v___x_3871_, 4, v___x_3870_);
lean_ctor_set(v___x_3871_, 5, v___x_3870_);
return v___x_3871_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3872_; lean_object* v___x_3873_; 
v___x_3872_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3873_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3873_, 0, v___x_3872_);
lean_ctor_set(v___x_3873_, 1, v___x_3872_);
lean_ctor_set(v___x_3873_, 2, v___x_3872_);
lean_ctor_set(v___x_3873_, 3, v___x_3872_);
lean_ctor_set(v___x_3873_, 4, v___x_3872_);
return v___x_3873_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3875_; lean_object* v___x_3876_; 
v___x_3875_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__4_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_));
v___x_3876_ = l_Lean_stringToMessageData(v___x_3875_);
return v___x_3876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(lean_object* v___x_3877_, lean_object* v___x_3878_, lean_object* v_declName_3879_, lean_object* v_stx_3880_, uint8_t v_kind_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_){
_start:
{
lean_object* v___y_3886_; lean_object* v___y_3887_; lean_object* v___y_3888_; lean_object* v_a_3889_; uint8_t v___x_3895_; 
lean_inc(v_stx_3880_);
v___x_3895_ = l_Lean_Syntax_isOfKind(v_stx_3880_, v___x_3877_);
if (v___x_3895_ == 0)
{
lean_object* v___x_3896_; 
lean_dec(v_stx_3880_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
lean_dec(v___x_3877_);
v___x_3896_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg();
return v___x_3896_;
}
else
{
lean_object* v___x_3897_; lean_object* v___x_3898_; 
v___x_3897_ = lean_st_ref_get(v___y_3883_);
lean_inc(v_declName_3879_);
v___x_3898_ = l_Lean_ensureAttrDeclIsMeta(v___x_3877_, v_declName_3879_, v_kind_3881_, v___y_3882_, v___y_3883_);
if (lean_obj_tag(v___x_3898_) == 0)
{
lean_object* v___x_3900_; uint8_t v_isShared_3901_; uint8_t v_isSharedCheck_3979_; 
v_isSharedCheck_3979_ = !lean_is_exclusive(v___x_3898_);
if (v_isSharedCheck_3979_ == 0)
{
lean_object* v_unused_3980_; 
v_unused_3980_ = lean_ctor_get(v___x_3898_, 0);
lean_dec(v_unused_3980_);
v___x_3900_ = v___x_3898_;
v_isShared_3901_ = v_isSharedCheck_3979_;
goto v_resetjp_3899_;
}
else
{
lean_dec(v___x_3898_);
v___x_3900_ = lean_box(0);
v_isShared_3901_ = v_isSharedCheck_3979_;
goto v_resetjp_3899_;
}
v_resetjp_3899_:
{
lean_object* v_env_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v_es_3905_; lean_object* v___y_3907_; lean_object* v___y_3908_; uint8_t v___y_3909_; lean_object* v___y_3967_; lean_object* v___y_3968_; lean_object* v___x_3978_; 
v_env_3902_ = lean_ctor_get(v___x_3897_, 0);
lean_inc_ref(v_env_3902_);
lean_dec(v___x_3897_);
v___x_3903_ = lean_unsigned_to_nat(1u);
v___x_3904_ = l_Lean_Syntax_getArg(v_stx_3880_, v___x_3903_);
lean_dec(v_stx_3880_);
v_es_3905_ = l_Lean_Syntax_getArgs(v___x_3904_);
lean_dec(v___x_3904_);
v___x_3978_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_3902_, v_declName_3879_);
if (lean_obj_tag(v___x_3978_) == 0)
{
if (v___x_3895_ == 0)
{
lean_dec_ref(v_es_3905_);
lean_dec_ref(v_env_3902_);
lean_del_object(v___x_3900_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
goto v___jp_3975_;
}
else
{
v___y_3967_ = v___y_3882_;
v___y_3968_ = v___y_3883_;
goto v___jp_3966_;
}
}
else
{
lean_dec_ref_known(v___x_3978_, 1);
lean_dec_ref(v_es_3905_);
lean_dec_ref(v_env_3902_);
lean_del_object(v___x_3900_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
goto v___jp_3975_;
}
v___jp_3906_:
{
lean_object* v___x_3910_; lean_object* v_env_3911_; lean_object* v_options_3912_; lean_object* v_ref_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; 
v___x_3910_ = lean_st_ref_get(v___y_3908_);
v_env_3911_ = lean_ctor_get(v___x_3910_, 0);
lean_inc_ref(v_env_3911_);
lean_dec(v___x_3910_);
v_options_3912_ = lean_ctor_get(v___y_3907_, 2);
v_ref_3913_ = lean_ctor_get(v___y_3907_, 5);
lean_inc_ref(v_options_3912_);
v___x_3914_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3914_, 0, v_env_3911_);
lean_ctor_set(v___x_3914_, 1, v_options_3912_);
lean_inc(v_declName_3879_);
v___x_3915_ = lp_mathlib_Mathlib_Meta_NormNum_mkNormNumExt(v_declName_3879_, v___x_3914_);
lean_dec_ref_known(v___x_3914_, 2);
if (lean_obj_tag(v___x_3915_) == 0)
{
lean_object* v_a_3916_; uint8_t v___x_3917_; uint8_t v___x_3918_; uint8_t v___x_3919_; lean_object* v___x_3920_; uint64_t v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; size_t v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; lean_object* v___x_3938_; lean_object* v___x_3939_; size_t v_sz_3940_; size_t v___x_3941_; lean_object* v___x_3942_; 
v_a_3916_ = lean_ctor_get(v___x_3915_, 0);
lean_inc(v_a_3916_);
lean_dec_ref_known(v___x_3915_, 1);
v___x_3917_ = 1;
v___x_3918_ = 0;
v___x_3919_ = 2;
v___x_3920_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v___x_3920_, 0, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 1, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 2, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 3, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 4, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 5, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 6, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 7, v___y_3909_);
lean_ctor_set_uint8(v___x_3920_, 8, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 9, v___x_3917_);
lean_ctor_set_uint8(v___x_3920_, 10, v___x_3918_);
lean_ctor_set_uint8(v___x_3920_, 11, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 12, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 13, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 14, v___x_3919_);
lean_ctor_set_uint8(v___x_3920_, 15, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 16, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 17, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 18, v___x_3895_);
lean_ctor_set_uint8(v___x_3920_, 19, v___y_3909_);
v___x_3921_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_3920_);
v___x_3922_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_3922_, 0, v___x_3920_);
lean_ctor_set_uint64(v___x_3922_, sizeof(void*)*1, v___x_3921_);
v___x_3923_ = lean_box(1);
v___x_3924_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3925_ = lean_unsigned_to_nat(32u);
v___x_3926_ = lean_mk_empty_array_with_capacity(v___x_3925_);
v___x_3927_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6_spec__12___closed__3);
v___x_3928_ = ((size_t)5ULL);
lean_inc_n(v___x_3878_, 6);
v___x_3929_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3929_, 0, v___x_3927_);
lean_ctor_set(v___x_3929_, 1, v___x_3926_);
lean_ctor_set(v___x_3929_, 2, v___x_3878_);
lean_ctor_set(v___x_3929_, 3, v___x_3878_);
lean_ctor_set_usize(v___x_3929_, 4, v___x_3928_);
lean_inc_ref(v___x_3929_);
v___x_3930_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3930_, 0, v___x_3924_);
lean_ctor_set(v___x_3930_, 1, v___x_3929_);
lean_ctor_set(v___x_3930_, 2, v___x_3923_);
v___x_3931_ = lean_mk_empty_array_with_capacity(v___x_3878_);
v___x_3932_ = lean_box(0);
v___x_3933_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3933_, 0, v___x_3922_);
lean_ctor_set(v___x_3933_, 1, v___x_3923_);
lean_ctor_set(v___x_3933_, 2, v___x_3930_);
lean_ctor_set(v___x_3933_, 3, v___x_3931_);
lean_ctor_set(v___x_3933_, 4, v___x_3932_);
lean_ctor_set(v___x_3933_, 5, v___x_3878_);
lean_ctor_set(v___x_3933_, 6, v___x_3932_);
lean_ctor_set_uint8(v___x_3933_, sizeof(void*)*7, v___y_3909_);
lean_ctor_set_uint8(v___x_3933_, sizeof(void*)*7 + 1, v___y_3909_);
lean_ctor_set_uint8(v___x_3933_, sizeof(void*)*7 + 2, v___y_3909_);
lean_ctor_set_uint8(v___x_3933_, sizeof(void*)*7 + 3, v___x_3895_);
v___x_3934_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3934_, 0, v___x_3878_);
lean_ctor_set(v___x_3934_, 1, v___x_3878_);
lean_ctor_set(v___x_3934_, 2, v___x_3878_);
lean_ctor_set(v___x_3934_, 3, v___x_3878_);
lean_ctor_set(v___x_3934_, 4, v___x_3924_);
lean_ctor_set(v___x_3934_, 5, v___x_3924_);
lean_ctor_set(v___x_3934_, 6, v___x_3924_);
lean_ctor_set(v___x_3934_, 7, v___x_3924_);
lean_ctor_set(v___x_3934_, 8, v___x_3924_);
lean_ctor_set(v___x_3934_, 9, v___x_3924_);
v___x_3935_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3936_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3937_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3937_, 0, v___x_3934_);
lean_ctor_set(v___x_3937_, 1, v___x_3935_);
lean_ctor_set(v___x_3937_, 2, v___x_3923_);
lean_ctor_set(v___x_3937_, 3, v___x_3929_);
lean_ctor_set(v___x_3937_, 4, v___x_3936_);
v___x_3938_ = lean_st_mk_ref(v___x_3937_);
v___x_3939_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_es_3905_);
lean_dec_ref(v_es_3905_);
v_sz_3940_ = lean_array_size(v___x_3939_);
v___x_3941_ = ((size_t)0ULL);
v___x_3942_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__5(v___x_3895_, v___y_3909_, v_sz_3940_, v___x_3941_, v___x_3939_, v___x_3933_, v___x_3938_, v___y_3907_, v___y_3908_);
lean_dec_ref_known(v___x_3933_, 7);
if (lean_obj_tag(v___x_3942_) == 0)
{
lean_object* v_a_3943_; lean_object* v___x_3944_; 
v_a_3943_ = lean_ctor_get(v___x_3942_, 0);
lean_inc(v_a_3943_);
lean_dec_ref_known(v___x_3942_, 1);
v___x_3944_ = lean_st_ref_get(v___x_3938_);
lean_dec(v___x_3938_);
lean_dec(v___x_3944_);
v___y_3886_ = v___y_3907_;
v___y_3887_ = v_a_3916_;
v___y_3888_ = v___y_3908_;
v_a_3889_ = v_a_3943_;
goto v___jp_3885_;
}
else
{
lean_dec(v___x_3938_);
if (lean_obj_tag(v___x_3942_) == 0)
{
lean_object* v_a_3945_; 
v_a_3945_ = lean_ctor_get(v___x_3942_, 0);
lean_inc(v_a_3945_);
lean_dec_ref_known(v___x_3942_, 1);
v___y_3886_ = v___y_3907_;
v___y_3887_ = v_a_3916_;
v___y_3888_ = v___y_3908_;
v_a_3889_ = v_a_3945_;
goto v___jp_3885_;
}
else
{
lean_object* v_a_3946_; lean_object* v___x_3948_; uint8_t v_isShared_3949_; uint8_t v_isSharedCheck_3953_; 
lean_dec(v_a_3916_);
lean_dec(v_declName_3879_);
v_a_3946_ = lean_ctor_get(v___x_3942_, 0);
v_isSharedCheck_3953_ = !lean_is_exclusive(v___x_3942_);
if (v_isSharedCheck_3953_ == 0)
{
v___x_3948_ = v___x_3942_;
v_isShared_3949_ = v_isSharedCheck_3953_;
goto v_resetjp_3947_;
}
else
{
lean_inc(v_a_3946_);
lean_dec(v___x_3942_);
v___x_3948_ = lean_box(0);
v_isShared_3949_ = v_isSharedCheck_3953_;
goto v_resetjp_3947_;
}
v_resetjp_3947_:
{
lean_object* v___x_3951_; 
if (v_isShared_3949_ == 0)
{
v___x_3951_ = v___x_3948_;
goto v_reusejp_3950_;
}
else
{
lean_object* v_reuseFailAlloc_3952_; 
v_reuseFailAlloc_3952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3952_, 0, v_a_3946_);
v___x_3951_ = v_reuseFailAlloc_3952_;
goto v_reusejp_3950_;
}
v_reusejp_3950_:
{
return v___x_3951_;
}
}
}
}
}
else
{
lean_object* v_a_3954_; lean_object* v___x_3956_; uint8_t v_isShared_3957_; uint8_t v_isSharedCheck_3965_; 
lean_dec_ref(v_es_3905_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
v_a_3954_ = lean_ctor_get(v___x_3915_, 0);
v_isSharedCheck_3965_ = !lean_is_exclusive(v___x_3915_);
if (v_isSharedCheck_3965_ == 0)
{
v___x_3956_ = v___x_3915_;
v_isShared_3957_ = v_isSharedCheck_3965_;
goto v_resetjp_3955_;
}
else
{
lean_inc(v_a_3954_);
lean_dec(v___x_3915_);
v___x_3956_ = lean_box(0);
v_isShared_3957_ = v_isSharedCheck_3965_;
goto v_resetjp_3955_;
}
v_resetjp_3955_:
{
lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___x_3961_; lean_object* v___x_3963_; 
v___x_3958_ = lean_io_error_to_string(v_a_3954_);
v___x_3959_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3959_, 0, v___x_3958_);
v___x_3960_ = l_Lean_MessageData_ofFormat(v___x_3959_);
lean_inc(v_ref_3913_);
v___x_3961_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3961_, 0, v_ref_3913_);
lean_ctor_set(v___x_3961_, 1, v___x_3960_);
if (v_isShared_3957_ == 0)
{
lean_ctor_set(v___x_3956_, 0, v___x_3961_);
v___x_3963_ = v___x_3956_;
goto v_reusejp_3962_;
}
else
{
lean_object* v_reuseFailAlloc_3964_; 
v_reuseFailAlloc_3964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3964_, 0, v___x_3961_);
v___x_3963_ = v_reuseFailAlloc_3964_;
goto v_reusejp_3962_;
}
v_reusejp_3962_:
{
return v___x_3963_;
}
}
}
}
v___jp_3966_:
{
lean_object* v___x_3969_; 
lean_inc(v_declName_3879_);
v___x_3969_ = lean_decl_get_sorry_dep(v_env_3902_, v_declName_3879_);
if (lean_obj_tag(v___x_3969_) == 0)
{
uint8_t v___x_3970_; 
lean_del_object(v___x_3900_);
v___x_3970_ = 0;
v___y_3907_ = v___y_3967_;
v___y_3908_ = v___y_3968_;
v___y_3909_ = v___x_3970_;
goto v___jp_3906_;
}
else
{
lean_dec_ref_known(v___x_3969_, 1);
if (v___x_3895_ == 0)
{
lean_del_object(v___x_3900_);
v___y_3907_ = v___y_3967_;
v___y_3908_ = v___y_3968_;
v___y_3909_ = v___x_3895_;
goto v___jp_3906_;
}
else
{
lean_object* v___x_3971_; lean_object* v___x_3973_; 
lean_dec_ref(v_es_3905_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
v___x_3971_ = lean_box(0);
if (v_isShared_3901_ == 0)
{
lean_ctor_set(v___x_3900_, 0, v___x_3971_);
v___x_3973_ = v___x_3900_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_3974_; 
v_reuseFailAlloc_3974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3974_, 0, v___x_3971_);
v___x_3973_ = v_reuseFailAlloc_3974_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
return v___x_3973_;
}
}
}
}
v___jp_3975_:
{
lean_object* v___x_3976_; lean_object* v___x_3977_; 
v___x_3976_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_3977_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(v___x_3976_, v___y_3882_, v___y_3883_);
return v___x_3977_;
}
}
}
else
{
lean_dec(v___x_3897_);
lean_dec(v_stx_3880_);
lean_dec(v_declName_3879_);
lean_dec(v___x_3878_);
return v___x_3898_;
}
}
v___jp_3885_:
{
lean_object* v___x_3890_; lean_object* v___x_3891_; lean_object* v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; 
v___x_3890_ = lp_mathlib_Mathlib_Meta_NormNum_normNumExt;
v___x_3891_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3891_, 0, v_a_3889_);
lean_ctor_set(v___x_3891_, 1, v_declName_3879_);
v___x_3892_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3892_, 0, v___x_3891_);
lean_ctor_set(v___x_3892_, 1, v___y_3887_);
v___x_3893_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__3___redArg(v___x_3890_, v___x_3892_, v_kind_3881_, v___y_3886_, v___y_3888_);
lean_dec_ref(v___x_3893_);
v___x_3894_ = lp_mathlib_Lean_recordExtraRevUseOfCurrentModule___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__4(v___y_3886_, v___y_3888_);
return v___x_3894_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object* v___x_3981_, lean_object* v___x_3982_, lean_object* v_declName_3983_, lean_object* v_stx_3984_, lean_object* v_kind_3985_, lean_object* v___y_3986_, lean_object* v___y_3987_, lean_object* v___y_3988_){
_start:
{
uint8_t v_kind_boxed_3989_; lean_object* v_res_3990_; 
v_kind_boxed_3989_ = lean_unbox(v_kind_3985_);
v_res_3990_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__2_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(v___x_3981_, v___x_3982_, v_declName_3983_, v_stx_3984_, v_kind_boxed_3989_, v___y_3986_, v___y_3987_);
lean_dec(v___y_3987_);
lean_dec_ref(v___y_3986_);
return v_res_3990_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_3991_; lean_object* v___f_3992_; 
v___x_3991_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default;
v___f_3992_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___lam__1_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed), 5, 1);
lean_closure_set(v___f_3992_, 0, v___x_3991_);
return v___f_3992_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_4014_; lean_object* v___f_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; 
v___f_4014_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__0_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___f_4015_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__5_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_));
v___x_4016_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__7_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_));
v___x_4017_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4017_, 0, v___x_4016_);
lean_ctor_set(v___x_4017_, 1, v___f_4015_);
lean_ctor_set(v___x_4017_, 2, v___f_4014_);
return v___x_4017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_4019_; lean_object* v___x_4020_; 
v___x_4019_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn___closed__8_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_);
v___x_4020_ = l_Lean_registerBuiltinAttribute(v___x_4019_);
return v___x_4020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2____boxed(lean_object* v_a_4021_){
_start:
{
lean_object* v_res_4022_; 
v_res_4022_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_();
return v_res_4022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6(lean_object* v_00_u03b1_4023_, lean_object* v_msg_4024_, lean_object* v___y_4025_, lean_object* v___y_4026_){
_start:
{
lean_object* v___x_4028_; 
v___x_4028_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___redArg(v_msg_4024_, v___y_4025_, v___y_4026_);
return v___x_4028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6___boxed(lean_object* v_00_u03b1_4029_, lean_object* v_msg_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_){
_start:
{
lean_object* v_res_4034_; 
v_res_4034_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__6(v_00_u03b1_4029_, v_msg_4030_, v___y_4031_, v___y_4032_);
lean_dec(v___y_4032_);
lean_dec_ref(v___y_4031_);
return v_res_4034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03c3_4035_, lean_object* v_00_u03b1_4036_, lean_object* v_f_4037_, lean_object* v_x_4038_, lean_object* v_x_4039_){
_start:
{
lean_object* v___x_4040_; 
v___x_4040_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___redArg(v_f_4037_, v_x_4038_, v_x_4039_);
return v___x_4040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03c3_4041_, lean_object* v_00_u03b1_4042_, lean_object* v_f_4043_, lean_object* v_x_4044_, lean_object* v_x_4045_){
_start:
{
lean_object* v_res_4046_; 
v_res_4046_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0(v_00_u03c3_4041_, v_00_u03b1_4042_, v_f_4043_, v_x_4044_, v_x_4045_);
lean_dec_ref(v_x_4045_);
return v_res_4046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___redArg(lean_object* v_map_4047_, lean_object* v_f_4048_, lean_object* v_init_4049_){
_start:
{
lean_object* v___x_4050_; 
v___x_4050_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v_f_4048_, v_map_4047_, v_init_4049_);
return v___x_4050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___redArg___boxed(lean_object* v_map_4051_, lean_object* v_f_4052_, lean_object* v_init_4053_){
_start:
{
lean_object* v_res_4054_; 
v_res_4054_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___redArg(v_map_4051_, v_f_4052_, v_init_4053_);
lean_dec_ref(v_map_4051_);
return v_res_4054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_00_u03c3_4055_, lean_object* v_00_u03b2_4056_, lean_object* v_map_4057_, lean_object* v_f_4058_, lean_object* v_init_4059_){
_start:
{
lean_object* v___x_4060_; 
v___x_4060_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v_f_4058_, v_map_4057_, v_init_4059_);
return v___x_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_00_u03c3_4061_, lean_object* v_00_u03b2_4062_, lean_object* v_map_4063_, lean_object* v_f_4064_, lean_object* v_init_4065_){
_start:
{
lean_object* v_res_4066_; 
v_res_4066_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1(v_00_u03c3_4061_, v_00_u03b2_4062_, v_map_4063_, v_f_4064_, v_init_4065_);
lean_dec_ref(v_map_4063_);
return v_res_4066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10(lean_object* v___y_4067_, lean_object* v___y_4068_, lean_object* v___y_4069_, lean_object* v___y_4070_, lean_object* v___y_4071_, lean_object* v___y_4072_){
_start:
{
lean_object* v___x_4074_; 
v___x_4074_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___redArg(v___y_4070_, v___y_4071_, v___y_4072_);
return v___x_4074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10___boxed(lean_object* v___y_4075_, lean_object* v___y_4076_, lean_object* v___y_4077_, lean_object* v___y_4078_, lean_object* v___y_4079_, lean_object* v___y_4080_, lean_object* v___y_4081_){
_start:
{
lean_object* v_res_4082_; 
v_res_4082_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__5_spec__10(v___y_4075_, v___y_4076_, v___y_4077_, v___y_4078_, v___y_4079_, v___y_4080_);
lean_dec(v___y_4080_);
lean_dec_ref(v___y_4079_);
lean_dec(v___y_4078_);
lean_dec_ref(v___y_4077_);
lean_dec(v___y_4076_);
lean_dec_ref(v___y_4075_);
return v_res_4082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12(lean_object* v___y_4083_, lean_object* v___y_4084_, lean_object* v___y_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_){
_start:
{
lean_object* v___x_4090_; 
v___x_4090_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___redArg(v___y_4088_);
return v___x_4090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12___boxed(lean_object* v___y_4091_, lean_object* v___y_4092_, lean_object* v___y_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_, lean_object* v___y_4097_){
_start:
{
lean_object* v_res_4098_; 
v_res_4098_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6_spec__12(v___y_4091_, v___y_4092_, v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_);
lean_dec(v___y_4096_);
lean_dec_ref(v___y_4095_);
lean_dec(v___y_4094_);
lean_dec_ref(v___y_4093_);
lean_dec(v___y_4092_);
lean_dec_ref(v___y_4091_);
return v_res_4098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6(lean_object* v_00_u03b1_4099_, lean_object* v_x_4100_, lean_object* v_ctx_x3f_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_, lean_object* v___y_4107_){
_start:
{
lean_object* v___x_4109_; 
v___x_4109_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___redArg(v_x_4100_, v_ctx_x3f_4101_, v___y_4102_, v___y_4103_, v___y_4104_, v___y_4105_, v___y_4106_, v___y_4107_);
return v___x_4109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6___boxed(lean_object* v_00_u03b1_4110_, lean_object* v_x_4111_, lean_object* v_ctx_x3f_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_, lean_object* v___y_4119_){
_start:
{
lean_object* v_res_4120_; 
v_res_4120_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__2_spec__6(v_00_u03b1_4110_, v_x_4111_, v_ctx_x3f_4112_, v___y_4113_, v___y_4114_, v___y_4115_, v___y_4116_, v___y_4117_, v___y_4118_);
lean_dec(v___y_4118_);
lean_dec_ref(v___y_4117_);
lean_dec(v___y_4116_);
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4114_);
lean_dec_ref(v___y_4113_);
return v_res_4120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3(lean_object* v_00_u03b1_4121_, lean_object* v_00_u03c3_4122_, lean_object* v_f_4123_, lean_object* v_as_4124_, size_t v_i_4125_, size_t v_stop_4126_, lean_object* v_b_4127_){
_start:
{
lean_object* v___x_4128_; 
v___x_4128_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___redArg(v_f_4123_, v_as_4124_, v_i_4125_, v_stop_4126_, v_b_4127_);
return v___x_4128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b1_4129_, lean_object* v_00_u03c3_4130_, lean_object* v_f_4131_, lean_object* v_as_4132_, lean_object* v_i_4133_, lean_object* v_stop_4134_, lean_object* v_b_4135_){
_start:
{
size_t v_i_boxed_4136_; size_t v_stop_boxed_4137_; lean_object* v_res_4138_; 
v_i_boxed_4136_ = lean_unbox_usize(v_i_4133_);
lean_dec(v_i_4133_);
v_stop_boxed_4137_ = lean_unbox_usize(v_stop_4134_);
lean_dec(v_stop_4134_);
v_res_4138_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__3(v_00_u03b1_4129_, v_00_u03c3_4130_, v_f_4131_, v_as_4132_, v_i_boxed_4136_, v_stop_boxed_4137_, v_b_4135_);
lean_dec_ref(v_as_4132_);
return v_res_4138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4(lean_object* v_00_u03b1_4139_, lean_object* v_00_u03c3_4140_, lean_object* v_f_4141_, lean_object* v_as_4142_, size_t v_i_4143_, size_t v_stop_4144_, lean_object* v_b_4145_){
_start:
{
lean_object* v___x_4146_; 
v___x_4146_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___redArg(v_f_4141_, v_as_4142_, v_i_4143_, v_stop_4144_, v_b_4145_);
return v___x_4146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4___boxed(lean_object* v_00_u03b1_4147_, lean_object* v_00_u03c3_4148_, lean_object* v_f_4149_, lean_object* v_as_4150_, lean_object* v_i_4151_, lean_object* v_stop_4152_, lean_object* v_b_4153_){
_start:
{
size_t v_i_boxed_4154_; size_t v_stop_boxed_4155_; lean_object* v_res_4156_; 
v_i_boxed_4154_ = lean_unbox_usize(v_i_4151_);
lean_dec(v_i_4151_);
v_stop_boxed_4155_ = lean_unbox_usize(v_stop_4152_);
lean_dec(v_stop_4152_);
v_res_4156_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__0_spec__4(v_00_u03b1_4147_, v_00_u03c3_4148_, v_f_4149_, v_as_4150_, v_i_boxed_4154_, v_stop_boxed_4155_, v_b_4153_);
lean_dec_ref(v_as_4150_);
return v_res_4156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6(lean_object* v_00_u03c3_4157_, lean_object* v_00_u03b1_4158_, lean_object* v_00_u03b2_4159_, lean_object* v_f_4160_, lean_object* v_x_4161_, lean_object* v_x_4162_){
_start:
{
lean_object* v___x_4163_; 
v___x_4163_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___redArg(v_f_4160_, v_x_4161_, v_x_4162_);
return v___x_4163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6___boxed(lean_object* v_00_u03c3_4164_, lean_object* v_00_u03b1_4165_, lean_object* v_00_u03b2_4166_, lean_object* v_f_4167_, lean_object* v_x_4168_, lean_object* v_x_4169_){
_start:
{
lean_object* v_res_4170_; 
v_res_4170_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6(v_00_u03c3_4164_, v_00_u03b1_4165_, v_00_u03b2_4166_, v_f_4167_, v_x_4168_, v_x_4169_);
lean_dec_ref(v_x_4168_);
return v_res_4170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12(lean_object* v_00_u03b1_4171_, lean_object* v_00_u03b2_4172_, lean_object* v_00_u03c3_4173_, lean_object* v_f_4174_, lean_object* v_as_4175_, size_t v_i_4176_, size_t v_stop_4177_, lean_object* v_b_4178_){
_start:
{
lean_object* v___x_4179_; 
v___x_4179_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___redArg(v_f_4174_, v_as_4175_, v_i_4176_, v_stop_4177_, v_b_4178_);
return v___x_4179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12___boxed(lean_object* v_00_u03b1_4180_, lean_object* v_00_u03b2_4181_, lean_object* v_00_u03c3_4182_, lean_object* v_f_4183_, lean_object* v_as_4184_, lean_object* v_i_4185_, lean_object* v_stop_4186_, lean_object* v_b_4187_){
_start:
{
size_t v_i_boxed_4188_; size_t v_stop_boxed_4189_; lean_object* v_res_4190_; 
v_i_boxed_4188_ = lean_unbox_usize(v_i_4185_);
lean_dec(v_i_4185_);
v_stop_boxed_4189_ = lean_unbox_usize(v_stop_4186_);
lean_dec(v_stop_4186_);
v_res_4190_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__12(v_00_u03b1_4180_, v_00_u03b2_4181_, v_00_u03c3_4182_, v_f_4183_, v_as_4184_, v_i_boxed_4188_, v_stop_boxed_4189_, v_b_4187_);
lean_dec_ref(v_as_4184_);
return v_res_4190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13(lean_object* v_00_u03c3_4191_, lean_object* v_00_u03b1_4192_, lean_object* v_00_u03b2_4193_, lean_object* v_f_4194_, lean_object* v_keys_4195_, lean_object* v_vals_4196_, lean_object* v_heq_4197_, lean_object* v_i_4198_, lean_object* v_acc_4199_){
_start:
{
lean_object* v___x_4200_; 
v___x_4200_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___redArg(v_f_4194_, v_keys_4195_, v_vals_4196_, v_i_4198_, v_acc_4199_);
return v___x_4200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13___boxed(lean_object* v_00_u03c3_4201_, lean_object* v_00_u03b1_4202_, lean_object* v_00_u03b2_4203_, lean_object* v_f_4204_, lean_object* v_keys_4205_, lean_object* v_vals_4206_, lean_object* v_heq_4207_, lean_object* v_i_4208_, lean_object* v_acc_4209_){
_start:
{
lean_object* v_res_4210_; 
v_res_4210_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Mathlib_Meta_NormNum_NormNums_erase___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__0_spec__1_spec__6_spec__13(v_00_u03c3_4201_, v_00_u03b1_4202_, v_00_u03b2_4203_, v_f_4204_, v_keys_4205_, v_vals_4206_, v_heq_4207_, v_i_4208_, v_acc_4209_);
lean_dec_ref(v_vals_4206_);
lean_dec_ref(v_keys_4205_);
return v_res_4210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(uint8_t v_post_4213_, lean_object* v_e_4214_, lean_object* v_a_4215_, lean_object* v_a_4216_, lean_object* v_a_4217_, lean_object* v_a_4218_){
_start:
{
lean_object* v___x_4220_; 
v___x_4220_ = lp_mathlib_Mathlib_Meta_NormNum_eval(v_e_4214_, v_post_4213_, v_a_4215_, v_a_4216_, v_a_4217_, v_a_4218_);
if (lean_obj_tag(v___x_4220_) == 0)
{
lean_object* v_a_4221_; lean_object* v___x_4223_; uint8_t v_isShared_4224_; uint8_t v_isSharedCheck_4229_; 
v_a_4221_ = lean_ctor_get(v___x_4220_, 0);
v_isSharedCheck_4229_ = !lean_is_exclusive(v___x_4220_);
if (v_isSharedCheck_4229_ == 0)
{
v___x_4223_ = v___x_4220_;
v_isShared_4224_ = v_isSharedCheck_4229_;
goto v_resetjp_4222_;
}
else
{
lean_inc(v_a_4221_);
lean_dec(v___x_4220_);
v___x_4223_ = lean_box(0);
v_isShared_4224_ = v_isSharedCheck_4229_;
goto v_resetjp_4222_;
}
v_resetjp_4222_:
{
lean_object* v___x_4225_; lean_object* v___x_4227_; 
v___x_4225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4225_, 0, v_a_4221_);
if (v_isShared_4224_ == 0)
{
lean_ctor_set(v___x_4223_, 0, v___x_4225_);
v___x_4227_ = v___x_4223_;
goto v_reusejp_4226_;
}
else
{
lean_object* v_reuseFailAlloc_4228_; 
v_reuseFailAlloc_4228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4228_, 0, v___x_4225_);
v___x_4227_ = v_reuseFailAlloc_4228_;
goto v_reusejp_4226_;
}
v_reusejp_4226_:
{
return v___x_4227_;
}
}
}
else
{
lean_object* v_a_4230_; lean_object* v___x_4232_; uint8_t v_isShared_4233_; uint8_t v_isSharedCheck_4245_; 
v_a_4230_ = lean_ctor_get(v___x_4220_, 0);
v_isSharedCheck_4245_ = !lean_is_exclusive(v___x_4220_);
if (v_isSharedCheck_4245_ == 0)
{
v___x_4232_ = v___x_4220_;
v_isShared_4233_ = v_isSharedCheck_4245_;
goto v_resetjp_4231_;
}
else
{
lean_inc(v_a_4230_);
lean_dec(v___x_4220_);
v___x_4232_ = lean_box(0);
v_isShared_4233_ = v_isSharedCheck_4245_;
goto v_resetjp_4231_;
}
v_resetjp_4231_:
{
uint8_t v___y_4235_; uint8_t v___x_4243_; 
v___x_4243_ = l_Lean_Exception_isInterrupt(v_a_4230_);
if (v___x_4243_ == 0)
{
uint8_t v___x_4244_; 
lean_inc(v_a_4230_);
v___x_4244_ = l_Lean_Exception_isRuntime(v_a_4230_);
v___y_4235_ = v___x_4244_;
goto v___jp_4234_;
}
else
{
v___y_4235_ = v___x_4243_;
goto v___jp_4234_;
}
v___jp_4234_:
{
if (v___y_4235_ == 0)
{
lean_object* v___x_4236_; lean_object* v___x_4238_; 
lean_dec(v_a_4230_);
v___x_4236_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___closed__0));
if (v_isShared_4233_ == 0)
{
lean_ctor_set_tag(v___x_4232_, 0);
lean_ctor_set(v___x_4232_, 0, v___x_4236_);
v___x_4238_ = v___x_4232_;
goto v_reusejp_4237_;
}
else
{
lean_object* v_reuseFailAlloc_4239_; 
v_reuseFailAlloc_4239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4239_, 0, v___x_4236_);
v___x_4238_ = v_reuseFailAlloc_4239_;
goto v_reusejp_4237_;
}
v_reusejp_4237_:
{
return v___x_4238_;
}
}
else
{
lean_object* v___x_4241_; 
if (v_isShared_4233_ == 0)
{
v___x_4241_ = v___x_4232_;
goto v_reusejp_4240_;
}
else
{
lean_object* v_reuseFailAlloc_4242_; 
v_reuseFailAlloc_4242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4242_, 0, v_a_4230_);
v___x_4241_ = v_reuseFailAlloc_4242_;
goto v_reusejp_4240_;
}
v_reusejp_4240_:
{
return v___x_4241_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg___boxed(lean_object* v_post_4246_, lean_object* v_e_4247_, lean_object* v_a_4248_, lean_object* v_a_4249_, lean_object* v_a_4250_, lean_object* v_a_4251_, lean_object* v_a_4252_){
_start:
{
uint8_t v_post_boxed_4253_; lean_object* v_res_4254_; 
v_post_boxed_4253_ = lean_unbox(v_post_4246_);
v_res_4254_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(v_post_boxed_4253_, v_e_4247_, v_a_4248_, v_a_4249_, v_a_4250_, v_a_4251_);
lean_dec(v_a_4251_);
lean_dec_ref(v_a_4250_);
lean_dec(v_a_4249_);
lean_dec_ref(v_a_4248_);
return v_res_4254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum(uint8_t v_post_4255_, lean_object* v_e_4256_, lean_object* v_a_4257_, lean_object* v_a_4258_, lean_object* v_a_4259_, lean_object* v_a_4260_, lean_object* v_a_4261_, lean_object* v_a_4262_, lean_object* v_a_4263_){
_start:
{
lean_object* v___x_4265_; 
v___x_4265_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(v_post_4255_, v_e_4256_, v_a_4260_, v_a_4261_, v_a_4262_, v_a_4263_);
return v___x_4265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___boxed(lean_object* v_post_4266_, lean_object* v_e_4267_, lean_object* v_a_4268_, lean_object* v_a_4269_, lean_object* v_a_4270_, lean_object* v_a_4271_, lean_object* v_a_4272_, lean_object* v_a_4273_, lean_object* v_a_4274_, lean_object* v_a_4275_){
_start:
{
uint8_t v_post_boxed_4276_; lean_object* v_res_4277_; 
v_post_boxed_4276_ = lean_unbox(v_post_4266_);
v_res_4277_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum(v_post_boxed_4276_, v_e_4267_, v_a_4268_, v_a_4269_, v_a_4270_, v_a_4271_, v_a_4272_, v_a_4273_, v_a_4274_);
lean_dec(v_a_4274_);
lean_dec_ref(v_a_4273_);
lean_dec(v_a_4272_);
lean_dec_ref(v_a_4271_);
lean_dec(v_a_4270_);
lean_dec_ref(v_a_4269_);
lean_dec(v_a_4268_);
return v_res_4277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0(lean_object* v_x_4280_, lean_object* v___y_4281_, lean_object* v___y_4282_, lean_object* v___y_4283_, lean_object* v___y_4284_, lean_object* v___y_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_){
_start:
{
lean_object* v___x_4289_; lean_object* v___x_4290_; 
v___x_4289_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___closed__0));
v___x_4290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4290_, 0, v___x_4289_);
return v___x_4290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0___boxed(lean_object* v_x_4291_, lean_object* v___y_4292_, lean_object* v___y_4293_, lean_object* v___y_4294_, lean_object* v___y_4295_, lean_object* v___y_4296_, lean_object* v___y_4297_, lean_object* v___y_4298_, lean_object* v___y_4299_){
_start:
{
lean_object* v_res_4300_; 
v_res_4300_ = lp_mathlib_Mathlib_Meta_NormNum_methods___lam__0(v_x_4291_, v___y_4292_, v___y_4293_, v___y_4294_, v___y_4295_, v___y_4296_, v___y_4297_, v___y_4298_);
lean_dec(v___y_4298_);
lean_dec_ref(v___y_4297_);
lean_dec(v___y_4296_);
lean_dec_ref(v___y_4295_);
lean_dec(v___y_4294_);
lean_dec_ref(v___y_4293_);
lean_dec(v___y_4292_);
lean_dec_ref(v_x_4291_);
return v_res_4300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1(lean_object* v_e_4301_, lean_object* v___y_4302_, lean_object* v___y_4303_, lean_object* v___y_4304_, lean_object* v___y_4305_, lean_object* v___y_4306_, lean_object* v___y_4307_, lean_object* v___y_4308_){
_start:
{
lean_object* v___x_4310_; lean_object* v___x_4311_; 
v___x_4310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4310_, 0, v_e_4301_);
v___x_4311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4311_, 0, v___x_4310_);
return v___x_4311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1___boxed(lean_object* v_e_4312_, lean_object* v___y_4313_, lean_object* v___y_4314_, lean_object* v___y_4315_, lean_object* v___y_4316_, lean_object* v___y_4317_, lean_object* v___y_4318_, lean_object* v___y_4319_, lean_object* v___y_4320_){
_start:
{
lean_object* v_res_4321_; 
v_res_4321_ = lp_mathlib_Mathlib_Meta_NormNum_methods___lam__1(v_e_4312_, v___y_4313_, v___y_4314_, v___y_4315_, v___y_4316_, v___y_4317_, v___y_4318_, v___y_4319_);
lean_dec(v___y_4319_);
lean_dec_ref(v___y_4318_);
lean_dec(v___y_4317_);
lean_dec_ref(v___y_4316_);
lean_dec(v___y_4315_);
lean_dec_ref(v___y_4314_);
lean_dec(v___y_4313_);
return v_res_4321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2(lean_object* v_x_4322_, lean_object* v___y_4323_, lean_object* v___y_4324_, lean_object* v___y_4325_, lean_object* v___y_4326_, lean_object* v___y_4327_, lean_object* v___y_4328_, lean_object* v___y_4329_, lean_object* v___y_4330_){
_start:
{
uint8_t v___x_4332_; lean_object* v___x_4333_; 
v___x_4332_ = 0;
v___x_4333_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(v___x_4332_, v___y_4323_, v___y_4327_, v___y_4328_, v___y_4329_, v___y_4330_);
return v___x_4333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2___boxed(lean_object* v_x_4334_, lean_object* v___y_4335_, lean_object* v___y_4336_, lean_object* v___y_4337_, lean_object* v___y_4338_, lean_object* v___y_4339_, lean_object* v___y_4340_, lean_object* v___y_4341_, lean_object* v___y_4342_, lean_object* v___y_4343_){
_start:
{
lean_object* v_res_4344_; 
v_res_4344_ = lp_mathlib_Mathlib_Meta_NormNum_methods___lam__2(v_x_4334_, v___y_4335_, v___y_4336_, v___y_4337_, v___y_4338_, v___y_4339_, v___y_4340_, v___y_4341_, v___y_4342_);
lean_dec(v___y_4342_);
lean_dec_ref(v___y_4341_);
lean_dec(v___y_4340_);
lean_dec_ref(v___y_4339_);
lean_dec(v___y_4338_);
lean_dec_ref(v___y_4337_);
lean_dec(v___y_4336_);
return v_res_4344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3(lean_object* v_simprocs_4345_, lean_object* v___f_4346_, lean_object* v___y_4347_, lean_object* v___y_4348_, lean_object* v___y_4349_, lean_object* v___y_4350_, lean_object* v___y_4351_, lean_object* v___y_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_){
_start:
{
lean_object* v___x_4356_; 
lean_inc_ref(v___y_4347_);
v___x_4356_ = l_Lean_Meta_Simp_preDefault(v_simprocs_4345_, v___y_4347_, v___y_4348_, v___y_4349_, v___y_4350_, v___y_4351_, v___y_4352_, v___y_4353_, v___y_4354_);
if (lean_obj_tag(v___x_4356_) == 0)
{
lean_object* v_a_4357_; 
v_a_4357_ = lean_ctor_get(v___x_4356_, 0);
lean_inc(v_a_4357_);
if (lean_obj_tag(v_a_4357_) == 2)
{
lean_object* v_e_x3f_4358_; lean_object* v___x_4359_; 
lean_dec_ref_known(v___x_4356_, 1);
v_e_x3f_4358_ = lean_ctor_get(v_a_4357_, 0);
lean_inc(v_e_x3f_4358_);
lean_dec_ref_known(v_a_4357_, 1);
v___x_4359_ = lean_box(0);
if (lean_obj_tag(v_e_x3f_4358_) == 0)
{
lean_object* v___x_4360_; 
v___x_4360_ = lean_apply_10(v___f_4346_, v___x_4359_, v___y_4347_, v___y_4348_, v___y_4349_, v___y_4350_, v___y_4351_, v___y_4352_, v___y_4353_, v___y_4354_, lean_box(0));
return v___x_4360_;
}
else
{
lean_object* v_val_4361_; lean_object* v_expr_4362_; lean_object* v___x_4363_; 
lean_dec_ref(v___y_4347_);
v_val_4361_ = lean_ctor_get(v_e_x3f_4358_, 0);
lean_inc(v_val_4361_);
lean_dec_ref_known(v_e_x3f_4358_, 1);
v_expr_4362_ = lean_ctor_get(v_val_4361_, 0);
lean_inc(v___y_4354_);
lean_inc_ref(v___y_4353_);
lean_inc(v___y_4352_);
lean_inc_ref(v___y_4351_);
lean_inc_ref(v_expr_4362_);
v___x_4363_ = lean_apply_10(v___f_4346_, v___x_4359_, v_expr_4362_, v___y_4348_, v___y_4349_, v___y_4350_, v___y_4351_, v___y_4352_, v___y_4353_, v___y_4354_, lean_box(0));
if (lean_obj_tag(v___x_4363_) == 0)
{
lean_object* v_a_4364_; lean_object* v___x_4365_; 
v_a_4364_ = lean_ctor_get(v___x_4363_, 0);
lean_inc(v_a_4364_);
lean_dec_ref_known(v___x_4363_, 1);
v___x_4365_ = l_Lean_Meta_Simp_mkEqTransResultStep(v_val_4361_, v_a_4364_, v___y_4351_, v___y_4352_, v___y_4353_, v___y_4354_);
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
lean_dec(v___y_4352_);
lean_dec_ref(v___y_4351_);
return v___x_4365_;
}
else
{
lean_dec(v_val_4361_);
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
lean_dec(v___y_4352_);
lean_dec_ref(v___y_4351_);
return v___x_4363_;
}
}
}
else
{
lean_dec(v_a_4357_);
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
lean_dec(v___y_4352_);
lean_dec_ref(v___y_4351_);
lean_dec(v___y_4350_);
lean_dec_ref(v___y_4349_);
lean_dec(v___y_4348_);
lean_dec_ref(v___y_4347_);
lean_dec_ref(v___f_4346_);
return v___x_4356_;
}
}
else
{
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
lean_dec(v___y_4352_);
lean_dec_ref(v___y_4351_);
lean_dec(v___y_4350_);
lean_dec_ref(v___y_4349_);
lean_dec(v___y_4348_);
lean_dec_ref(v___y_4347_);
lean_dec_ref(v___f_4346_);
return v___x_4356_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3___boxed(lean_object* v_simprocs_4366_, lean_object* v___f_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_, lean_object* v___y_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_){
_start:
{
lean_object* v_res_4377_; 
v_res_4377_ = lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3(v_simprocs_4366_, v___f_4367_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_);
return v_res_4377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6(lean_object* v_simprocs_4378_, uint8_t v___x_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_, lean_object* v___y_4383_, lean_object* v___y_4384_, lean_object* v___y_4385_, lean_object* v___y_4386_, lean_object* v___y_4387_){
_start:
{
lean_object* v___x_4389_; 
lean_inc_ref(v___y_4380_);
v___x_4389_ = l_Lean_Meta_Simp_postDefault(v_simprocs_4378_, v___y_4380_, v___y_4381_, v___y_4382_, v___y_4383_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_);
if (lean_obj_tag(v___x_4389_) == 0)
{
lean_object* v_a_4390_; 
v_a_4390_ = lean_ctor_get(v___x_4389_, 0);
lean_inc(v_a_4390_);
if (lean_obj_tag(v_a_4390_) == 2)
{
lean_object* v_e_x3f_4391_; 
lean_dec_ref_known(v___x_4389_, 1);
v_e_x3f_4391_ = lean_ctor_get(v_a_4390_, 0);
lean_inc(v_e_x3f_4391_);
lean_dec_ref_known(v_a_4390_, 1);
if (lean_obj_tag(v_e_x3f_4391_) == 0)
{
lean_object* v___x_4392_; 
v___x_4392_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(v___x_4379_, v___y_4380_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_);
return v___x_4392_;
}
else
{
lean_object* v_val_4393_; lean_object* v_expr_4394_; lean_object* v___x_4395_; 
lean_dec_ref(v___y_4380_);
v_val_4393_ = lean_ctor_get(v_e_x3f_4391_, 0);
lean_inc(v_val_4393_);
lean_dec_ref_known(v_e_x3f_4391_, 1);
v_expr_4394_ = lean_ctor_get(v_val_4393_, 0);
lean_inc_ref(v_expr_4394_);
v___x_4395_ = lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___redArg(v___x_4379_, v_expr_4394_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_);
if (lean_obj_tag(v___x_4395_) == 0)
{
lean_object* v_a_4396_; lean_object* v___x_4397_; 
v_a_4396_ = lean_ctor_get(v___x_4395_, 0);
lean_inc(v_a_4396_);
lean_dec_ref_known(v___x_4395_, 1);
v___x_4397_ = l_Lean_Meta_Simp_mkEqTransResultStep(v_val_4393_, v_a_4396_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_);
return v___x_4397_;
}
else
{
lean_dec(v_val_4393_);
return v___x_4395_;
}
}
}
else
{
lean_dec(v_a_4390_);
lean_dec_ref(v___y_4380_);
return v___x_4389_;
}
}
else
{
lean_dec_ref(v___y_4380_);
return v___x_4389_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6___boxed(lean_object* v_simprocs_4398_, lean_object* v___x_4399_, lean_object* v___y_4400_, lean_object* v___y_4401_, lean_object* v___y_4402_, lean_object* v___y_4403_, lean_object* v___y_4404_, lean_object* v___y_4405_, lean_object* v___y_4406_, lean_object* v___y_4407_, lean_object* v___y_4408_){
_start:
{
uint8_t v___x_1430__boxed_4409_; lean_object* v_res_4410_; 
v___x_1430__boxed_4409_ = lean_unbox(v___x_4399_);
v_res_4410_ = lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6(v_simprocs_4398_, v___x_1430__boxed_4409_, v___y_4400_, v___y_4401_, v___y_4402_, v___y_4403_, v___y_4404_, v___y_4405_, v___y_4406_, v___y_4407_);
lean_dec(v___y_4407_);
lean_dec_ref(v___y_4406_);
lean_dec(v___y_4405_);
lean_dec_ref(v___y_4404_);
lean_dec(v___y_4403_);
lean_dec_ref(v___y_4402_);
lean_dec(v___y_4401_);
lean_dec_ref(v_simprocs_4398_);
return v_res_4410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods(lean_object* v_simprocs_4418_, uint8_t v_useSimp_4419_){
_start:
{
uint8_t v___x_4420_; 
v___x_4420_ = 1;
if (v_useSimp_4419_ == 0)
{
lean_object* v___f_4421_; lean_object* v___f_4422_; lean_object* v___x_4423_; lean_object* v___x_4424_; lean_object* v___x_4425_; lean_object* v___x_4426_; lean_object* v___x_4427_; 
lean_dec_ref(v_simprocs_4418_);
v___f_4421_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__0));
v___f_4422_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__1));
v___x_4423_ = lean_box(v_useSimp_4419_);
v___x_4424_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_tryNormNum___boxed), 10, 1);
lean_closure_set(v___x_4424_, 0, v___x_4423_);
v___x_4425_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__2));
v___x_4426_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__3));
v___x_4427_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_4427_, 0, v___x_4424_);
lean_ctor_set(v___x_4427_, 1, v___x_4425_);
lean_ctor_set(v___x_4427_, 2, v___f_4421_);
lean_ctor_set(v___x_4427_, 3, v___f_4422_);
lean_ctor_set(v___x_4427_, 4, v___x_4426_);
lean_ctor_set_uint8(v___x_4427_, sizeof(void*)*5, v___x_4420_);
return v___x_4427_;
}
else
{
lean_object* v___f_4428_; lean_object* v___f_4429_; lean_object* v___f_4430_; lean_object* v___f_4431_; lean_object* v___x_4432_; lean_object* v___f_4433_; lean_object* v___x_4434_; lean_object* v___x_4435_; 
v___f_4428_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__4));
lean_inc_ref(v_simprocs_4418_);
v___f_4429_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_methods___lam__3___boxed), 11, 2);
lean_closure_set(v___f_4429_, 0, v_simprocs_4418_);
lean_closure_set(v___f_4429_, 1, v___f_4428_);
v___f_4430_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__0));
v___f_4431_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__1));
v___x_4432_ = lean_box(v___x_4420_);
v___f_4433_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_methods___lam__6___boxed), 11, 2);
lean_closure_set(v___f_4433_, 0, v_simprocs_4418_);
lean_closure_set(v___f_4433_, 1, v___x_4432_);
v___x_4434_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_methods___closed__3));
v___x_4435_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_4435_, 0, v___f_4429_);
lean_ctor_set(v___x_4435_, 1, v___f_4433_);
lean_ctor_set(v___x_4435_, 2, v___f_4430_);
lean_ctor_set(v___x_4435_, 3, v___f_4431_);
lean_ctor_set(v___x_4435_, 4, v___x_4434_);
lean_ctor_set_uint8(v___x_4435_, sizeof(void*)*5, v___x_4420_);
return v___x_4435_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_methods___boxed(lean_object* v_simprocs_4436_, lean_object* v_useSimp_4437_){
_start:
{
uint8_t v_useSimp_boxed_4438_; lean_object* v_res_4439_; 
v_useSimp_boxed_4438_ = lean_unbox(v_useSimp_4437_);
v_res_4439_ = lp_mathlib_Mathlib_Meta_NormNum_methods(v_simprocs_4436_, v_useSimp_boxed_4438_);
return v_res_4439_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0(void){
_start:
{
lean_object* v___x_4440_; 
v___x_4440_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4440_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1(void){
_start:
{
lean_object* v___x_4441_; lean_object* v___x_4442_; 
v___x_4441_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__0);
v___x_4442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4442_, 0, v___x_4441_);
return v___x_4442_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2(void){
_start:
{
lean_object* v___x_4443_; lean_object* v___x_4444_; lean_object* v___x_4445_; 
v___x_4443_ = lean_unsigned_to_nat(0u);
v___x_4444_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1);
v___x_4445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4445_, 0, v___x_4444_);
lean_ctor_set(v___x_4445_, 1, v___x_4443_);
return v___x_4445_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3(void){
_start:
{
lean_object* v___x_4446_; lean_object* v___x_4447_; lean_object* v___x_4448_; 
v___x_4446_ = lean_unsigned_to_nat(32u);
v___x_4447_ = lean_mk_empty_array_with_capacity(v___x_4446_);
v___x_4448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4448_, 0, v___x_4447_);
return v___x_4448_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4(void){
_start:
{
size_t v___x_4449_; lean_object* v___x_4450_; lean_object* v___x_4451_; lean_object* v___x_4452_; lean_object* v___x_4453_; lean_object* v___x_4454_; 
v___x_4449_ = ((size_t)5ULL);
v___x_4450_ = lean_unsigned_to_nat(0u);
v___x_4451_ = lean_unsigned_to_nat(32u);
v___x_4452_ = lean_mk_empty_array_with_capacity(v___x_4451_);
v___x_4453_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__3);
v___x_4454_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_4454_, 0, v___x_4453_);
lean_ctor_set(v___x_4454_, 1, v___x_4452_);
lean_ctor_set(v___x_4454_, 2, v___x_4450_);
lean_ctor_set(v___x_4454_, 3, v___x_4450_);
lean_ctor_set_usize(v___x_4454_, 4, v___x_4449_);
return v___x_4454_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5(void){
_start:
{
lean_object* v___x_4455_; lean_object* v___x_4456_; lean_object* v___x_4457_; 
v___x_4455_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__4);
v___x_4456_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__1);
v___x_4457_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4457_, 0, v___x_4456_);
lean_ctor_set(v___x_4457_, 1, v___x_4456_);
lean_ctor_set(v___x_4457_, 2, v___x_4456_);
lean_ctor_set(v___x_4457_, 3, v___x_4455_);
return v___x_4457_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6(void){
_start:
{
lean_object* v___x_4458_; lean_object* v___x_4459_; lean_object* v___x_4460_; 
v___x_4458_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__5);
v___x_4459_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__2);
v___x_4460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4460_, 0, v___x_4459_);
lean_ctor_set(v___x_4460_, 1, v___x_4458_);
return v___x_4460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(lean_object* v_ctx_4461_, lean_object* v_simprocs_4462_, uint8_t v_useSimp_4463_, lean_object* v_e_4464_, lean_object* v_a_4465_, lean_object* v_a_4466_, lean_object* v_a_4467_, lean_object* v_a_4468_){
_start:
{
lean_object* v___x_4470_; lean_object* v___x_4471_; lean_object* v___x_4472_; 
v___x_4470_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___closed__6);
v___x_4471_ = lp_mathlib_Mathlib_Meta_NormNum_methods(v_simprocs_4462_, v_useSimp_4463_);
v___x_4472_ = l_Lean_Meta_Simp_main(v_e_4464_, v_ctx_4461_, v___x_4470_, v___x_4471_, v_a_4465_, v_a_4466_, v_a_4467_, v_a_4468_);
if (lean_obj_tag(v___x_4472_) == 0)
{
lean_object* v_a_4473_; lean_object* v___x_4475_; uint8_t v_isShared_4476_; uint8_t v_isSharedCheck_4481_; 
v_a_4473_ = lean_ctor_get(v___x_4472_, 0);
v_isSharedCheck_4481_ = !lean_is_exclusive(v___x_4472_);
if (v_isSharedCheck_4481_ == 0)
{
v___x_4475_ = v___x_4472_;
v_isShared_4476_ = v_isSharedCheck_4481_;
goto v_resetjp_4474_;
}
else
{
lean_inc(v_a_4473_);
lean_dec(v___x_4472_);
v___x_4475_ = lean_box(0);
v_isShared_4476_ = v_isSharedCheck_4481_;
goto v_resetjp_4474_;
}
v_resetjp_4474_:
{
lean_object* v_fst_4477_; lean_object* v___x_4479_; 
v_fst_4477_ = lean_ctor_get(v_a_4473_, 0);
lean_inc(v_fst_4477_);
lean_dec(v_a_4473_);
if (v_isShared_4476_ == 0)
{
lean_ctor_set(v___x_4475_, 0, v_fst_4477_);
v___x_4479_ = v___x_4475_;
goto v_reusejp_4478_;
}
else
{
lean_object* v_reuseFailAlloc_4480_; 
v_reuseFailAlloc_4480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4480_, 0, v_fst_4477_);
v___x_4479_ = v_reuseFailAlloc_4480_;
goto v_reusejp_4478_;
}
v_reusejp_4478_:
{
return v___x_4479_;
}
}
}
else
{
lean_object* v_a_4482_; lean_object* v___x_4484_; uint8_t v_isShared_4485_; uint8_t v_isSharedCheck_4489_; 
v_a_4482_ = lean_ctor_get(v___x_4472_, 0);
v_isSharedCheck_4489_ = !lean_is_exclusive(v___x_4472_);
if (v_isSharedCheck_4489_ == 0)
{
v___x_4484_ = v___x_4472_;
v_isShared_4485_ = v_isSharedCheck_4489_;
goto v_resetjp_4483_;
}
else
{
lean_inc(v_a_4482_);
lean_dec(v___x_4472_);
v___x_4484_ = lean_box(0);
v_isShared_4485_ = v_isSharedCheck_4489_;
goto v_resetjp_4483_;
}
v_resetjp_4483_:
{
lean_object* v___x_4487_; 
if (v_isShared_4485_ == 0)
{
v___x_4487_ = v___x_4484_;
goto v_reusejp_4486_;
}
else
{
lean_object* v_reuseFailAlloc_4488_; 
v_reuseFailAlloc_4488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4488_, 0, v_a_4482_);
v___x_4487_ = v_reuseFailAlloc_4488_;
goto v_reusejp_4486_;
}
v_reusejp_4486_:
{
return v___x_4487_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp___boxed(lean_object* v_ctx_4490_, lean_object* v_simprocs_4491_, lean_object* v_useSimp_4492_, lean_object* v_e_4493_, lean_object* v_a_4494_, lean_object* v_a_4495_, lean_object* v_a_4496_, lean_object* v_a_4497_, lean_object* v_a_4498_){
_start:
{
uint8_t v_useSimp_boxed_4499_; lean_object* v_res_4500_; 
v_useSimp_boxed_4499_ = lean_unbox(v_useSimp_4492_);
v_res_4500_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_ctx_4490_, v_simprocs_4491_, v_useSimp_boxed_4499_, v_e_4493_, v_a_4494_, v_a_4495_, v_a_4496_, v_a_4497_);
lean_dec(v_a_4497_);
lean_dec_ref(v_a_4496_);
lean_dec(v_a_4495_);
lean_dec_ref(v_a_4494_);
return v_res_4500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg(lean_object* v_simprocs_4501_, uint8_t v_useSimp_4502_, lean_object* v_e_4503_, lean_object* v_a_4504_, lean_object* v_a_4505_, lean_object* v_a_4506_, lean_object* v_a_4507_, lean_object* v_a_4508_){
_start:
{
lean_object* v___x_4510_; 
lean_inc_ref(v_a_4504_);
v___x_4510_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_a_4504_, v_simprocs_4501_, v_useSimp_4502_, v_e_4503_, v_a_4505_, v_a_4506_, v_a_4507_, v_a_4508_);
if (lean_obj_tag(v___x_4510_) == 0)
{
lean_object* v_a_4511_; lean_object* v___x_4512_; 
v_a_4511_ = lean_ctor_get(v___x_4510_, 0);
lean_inc(v_a_4511_);
lean_dec_ref_known(v___x_4510_, 1);
v___x_4512_ = lp_mathlib_Lean_Meta_Simp_Result_ofTrue(v_a_4511_, v_a_4505_, v_a_4506_, v_a_4507_, v_a_4508_);
return v___x_4512_;
}
else
{
lean_object* v_a_4513_; lean_object* v___x_4515_; uint8_t v_isShared_4516_; uint8_t v_isSharedCheck_4520_; 
v_a_4513_ = lean_ctor_get(v___x_4510_, 0);
v_isSharedCheck_4520_ = !lean_is_exclusive(v___x_4510_);
if (v_isSharedCheck_4520_ == 0)
{
v___x_4515_ = v___x_4510_;
v_isShared_4516_ = v_isSharedCheck_4520_;
goto v_resetjp_4514_;
}
else
{
lean_inc(v_a_4513_);
lean_dec(v___x_4510_);
v___x_4515_ = lean_box(0);
v_isShared_4516_ = v_isSharedCheck_4520_;
goto v_resetjp_4514_;
}
v_resetjp_4514_:
{
lean_object* v___x_4518_; 
if (v_isShared_4516_ == 0)
{
v___x_4518_ = v___x_4515_;
goto v_reusejp_4517_;
}
else
{
lean_object* v_reuseFailAlloc_4519_; 
v_reuseFailAlloc_4519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4519_, 0, v_a_4513_);
v___x_4518_ = v_reuseFailAlloc_4519_;
goto v_reusejp_4517_;
}
v_reusejp_4517_:
{
return v___x_4518_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg___boxed(lean_object* v_simprocs_4521_, lean_object* v_useSimp_4522_, lean_object* v_e_4523_, lean_object* v_a_4524_, lean_object* v_a_4525_, lean_object* v_a_4526_, lean_object* v_a_4527_, lean_object* v_a_4528_, lean_object* v_a_4529_){
_start:
{
uint8_t v_useSimp_boxed_4530_; lean_object* v_res_4531_; 
v_useSimp_boxed_4530_ = lean_unbox(v_useSimp_4522_);
v_res_4531_ = lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg(v_simprocs_4521_, v_useSimp_boxed_4530_, v_e_4523_, v_a_4524_, v_a_4525_, v_a_4526_, v_a_4527_, v_a_4528_);
lean_dec(v_a_4528_);
lean_dec_ref(v_a_4527_);
lean_dec(v_a_4526_);
lean_dec_ref(v_a_4525_);
lean_dec_ref(v_a_4524_);
return v_res_4531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge(lean_object* v_simprocs_4532_, uint8_t v_useSimp_4533_, lean_object* v_e_4534_, lean_object* v_a_4535_, lean_object* v_a_4536_, lean_object* v_a_4537_, lean_object* v_a_4538_, lean_object* v_a_4539_, lean_object* v_a_4540_, lean_object* v_a_4541_){
_start:
{
lean_object* v___x_4543_; 
v___x_4543_ = lp_mathlib_Mathlib_Meta_NormNum_discharge___redArg(v_simprocs_4532_, v_useSimp_4533_, v_e_4534_, v_a_4536_, v_a_4538_, v_a_4539_, v_a_4540_, v_a_4541_);
return v___x_4543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed(lean_object* v_simprocs_4544_, lean_object* v_useSimp_4545_, lean_object* v_e_4546_, lean_object* v_a_4547_, lean_object* v_a_4548_, lean_object* v_a_4549_, lean_object* v_a_4550_, lean_object* v_a_4551_, lean_object* v_a_4552_, lean_object* v_a_4553_, lean_object* v_a_4554_){
_start:
{
uint8_t v_useSimp_boxed_4555_; lean_object* v_res_4556_; 
v_useSimp_boxed_4555_ = lean_unbox(v_useSimp_4545_);
v_res_4556_ = lp_mathlib_Mathlib_Meta_NormNum_discharge(v_simprocs_4544_, v_useSimp_boxed_4555_, v_e_4546_, v_a_4547_, v_a_4548_, v_a_4549_, v_a_4550_, v_a_4551_, v_a_4552_, v_a_4553_);
lean_dec(v_a_4553_);
lean_dec_ref(v_a_4552_);
lean_dec(v_a_4551_);
lean_dec_ref(v_a_4550_);
lean_dec(v_a_4549_);
lean_dec_ref(v_a_4548_);
lean_dec(v_a_4547_);
return v_res_4556_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0(void){
_start:
{
lean_object* v___x_4557_; 
v___x_4557_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4557_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1(void){
_start:
{
lean_object* v___x_4558_; lean_object* v___x_4559_; 
v___x_4558_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__0);
v___x_4559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4559_, 0, v___x_4558_);
return v___x_4559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0(lean_object* v_00_u03b2_4560_){
_start:
{
lean_object* v___x_4561_; 
v___x_4561_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0___closed__1);
return v___x_4561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg(lean_object* v_x_4562_, lean_object* v_x_4563_, lean_object* v___y_4564_, lean_object* v___y_4565_, lean_object* v___y_4566_, lean_object* v___y_4567_){
_start:
{
if (lean_obj_tag(v_x_4563_) == 0)
{
lean_object* v___x_4569_; 
v___x_4569_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4569_, 0, v_x_4562_);
return v___x_4569_;
}
else
{
lean_object* v_head_4570_; lean_object* v_tail_4571_; uint8_t v___x_4572_; uint8_t v___x_4573_; lean_object* v___x_4574_; lean_object* v___x_4575_; 
v_head_4570_ = lean_ctor_get(v_x_4563_, 0);
lean_inc(v_head_4570_);
v_tail_4571_ = lean_ctor_get(v_x_4563_, 1);
lean_inc(v_tail_4571_);
lean_dec_ref_known(v_x_4563_, 2);
v___x_4572_ = 1;
v___x_4573_ = 0;
v___x_4574_ = lean_unsigned_to_nat(1000u);
v___x_4575_ = l_Lean_Meta_SimpTheorems_addConst(v_x_4562_, v_head_4570_, v___x_4572_, v___x_4573_, v___x_4574_, v___y_4564_, v___y_4565_, v___y_4566_, v___y_4567_);
if (lean_obj_tag(v___x_4575_) == 0)
{
lean_object* v_a_4576_; 
v_a_4576_ = lean_ctor_get(v___x_4575_, 0);
lean_inc(v_a_4576_);
lean_dec_ref_known(v___x_4575_, 1);
v_x_4562_ = v_a_4576_;
v_x_4563_ = v_tail_4571_;
goto _start;
}
else
{
lean_dec(v_tail_4571_);
return v___x_4575_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg___boxed(lean_object* v_x_4578_, lean_object* v_x_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_, lean_object* v___y_4582_, lean_object* v___y_4583_, lean_object* v___y_4584_){
_start:
{
lean_object* v_res_4585_; 
v_res_4585_ = lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg(v_x_4578_, v_x_4579_, v___y_4580_, v___y_4581_, v___y_4582_, v___y_4583_);
lean_dec(v___y_4583_);
lean_dec_ref(v___y_4582_);
lean_dec(v___y_4581_);
lean_dec_ref(v___y_4580_);
return v_res_4585_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1(void){
_start:
{
lean_object* v___x_4593_; 
v___x_4593_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_4593_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2(void){
_start:
{
lean_object* v___x_4594_; lean_object* v___x_4595_; lean_object* v___x_4596_; 
v___x_4594_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2);
v___x_4595_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__1);
v___x_4596_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4596_, 0, v___x_4595_);
lean_ctor_set(v___x_4596_, 1, v___x_4595_);
lean_ctor_set(v___x_4596_, 2, v___x_4594_);
lean_ctor_set(v___x_4596_, 3, v___x_4594_);
return v___x_4596_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3(void){
_start:
{
lean_object* v___x_4597_; 
v___x_4597_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_4597_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4(void){
_start:
{
lean_object* v___x_4598_; 
v___x_4598_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Meta_NormNum_getSimpContext_spec__0(lean_box(0));
return v___x_4598_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5(void){
_start:
{
lean_object* v___x_4599_; 
v___x_4599_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4599_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6(void){
_start:
{
lean_object* v___x_4600_; lean_object* v___x_4601_; 
v___x_4600_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__5);
v___x_4601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4601_, 0, v___x_4600_);
return v___x_4601_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7(void){
_start:
{
lean_object* v___x_4602_; lean_object* v___x_4603_; lean_object* v___x_4604_; lean_object* v___x_4605_; lean_object* v___x_4606_; 
v___x_4602_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__6);
v___x_4603_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default___closed__2);
v___x_4604_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__4);
v___x_4605_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__3);
v___x_4606_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_4606_, 0, v___x_4605_);
lean_ctor_set(v___x_4606_, 1, v___x_4605_);
lean_ctor_set(v___x_4606_, 2, v___x_4604_);
lean_ctor_set(v___x_4606_, 3, v___x_4603_);
lean_ctor_set(v___x_4606_, 4, v___x_4604_);
lean_ctor_set(v___x_4606_, 5, v___x_4602_);
return v___x_4606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(lean_object* v_cfg_4619_, lean_object* v_args_4620_, uint8_t v_simpOnly_4621_, lean_object* v_a_4622_, lean_object* v_a_4623_, lean_object* v_a_4624_, lean_object* v_a_4625_, lean_object* v_a_4626_, lean_object* v_a_4627_, lean_object* v_a_4628_, lean_object* v_a_4629_){
_start:
{
uint8_t v___x_4631_; lean_object* v___x_4632_; lean_object* v___x_4633_; lean_object* v___x_4634_; 
v___x_4631_ = 0;
v___x_4632_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__0));
v___x_4633_ = l_Lean_Options_empty;
v___x_4634_ = l_Lean_Elab_Tactic_elabSimpConfigCore___redArg(v_cfg_4619_, v___x_4632_, v___x_4633_, v_a_4622_, v_a_4628_, v_a_4629_);
if (lean_obj_tag(v___x_4634_) == 0)
{
lean_object* v_a_4635_; lean_object* v_config_4636_; lean_object* v_userConfig_4637_; lean_object* v___x_4639_; uint8_t v_isShared_4640_; uint8_t v_isSharedCheck_4744_; 
v_a_4635_ = lean_ctor_get(v___x_4634_, 0);
lean_inc(v_a_4635_);
lean_dec_ref_known(v___x_4634_, 1);
v_config_4636_ = lean_ctor_get(v_a_4635_, 0);
v_userConfig_4637_ = lean_ctor_get(v_a_4635_, 1);
v_isSharedCheck_4744_ = !lean_is_exclusive(v_a_4635_);
if (v_isSharedCheck_4744_ == 0)
{
v___x_4639_ = v_a_4635_;
v_isShared_4640_ = v_isSharedCheck_4744_;
goto v_resetjp_4638_;
}
else
{
lean_inc(v_userConfig_4637_);
lean_inc(v_config_4636_);
lean_dec(v_a_4635_);
v___x_4639_ = lean_box(0);
v_isShared_4640_ = v_isSharedCheck_4744_;
goto v_resetjp_4638_;
}
v_resetjp_4638_:
{
lean_object* v___y_4642_; lean_object* v_simprocs_4643_; lean_object* v___y_4644_; lean_object* v___y_4645_; lean_object* v___y_4646_; lean_object* v___y_4647_; lean_object* v___y_4648_; lean_object* v___y_4649_; lean_object* v___y_4650_; lean_object* v___y_4651_; lean_object* v_simpTheorems_4702_; lean_object* v___y_4703_; lean_object* v___y_4704_; lean_object* v___y_4705_; lean_object* v___y_4706_; lean_object* v___y_4707_; lean_object* v___y_4708_; lean_object* v___y_4709_; lean_object* v___y_4710_; 
if (v_simpOnly_4621_ == 0)
{
lean_object* v___x_4722_; 
v___x_4722_ = l_Lean_Meta_getSimpTheorems___redArg(v_a_4629_);
if (lean_obj_tag(v___x_4722_) == 0)
{
lean_object* v_a_4723_; 
v_a_4723_ = lean_ctor_get(v___x_4722_, 0);
lean_inc(v_a_4723_);
lean_dec_ref_known(v___x_4722_, 1);
v_simpTheorems_4702_ = v_a_4723_;
v___y_4703_ = v_a_4622_;
v___y_4704_ = v_a_4623_;
v___y_4705_ = v_a_4624_;
v___y_4706_ = v_a_4625_;
v___y_4707_ = v_a_4626_;
v___y_4708_ = v_a_4627_;
v___y_4709_ = v_a_4628_;
v___y_4710_ = v_a_4629_;
goto v___jp_4701_;
}
else
{
lean_object* v_a_4724_; lean_object* v___x_4726_; uint8_t v_isShared_4727_; uint8_t v_isSharedCheck_4731_; 
lean_del_object(v___x_4639_);
lean_dec_ref(v_userConfig_4637_);
lean_dec_ref(v_config_4636_);
v_a_4724_ = lean_ctor_get(v___x_4722_, 0);
v_isSharedCheck_4731_ = !lean_is_exclusive(v___x_4722_);
if (v_isSharedCheck_4731_ == 0)
{
v___x_4726_ = v___x_4722_;
v_isShared_4727_ = v_isSharedCheck_4731_;
goto v_resetjp_4725_;
}
else
{
lean_inc(v_a_4724_);
lean_dec(v___x_4722_);
v___x_4726_ = lean_box(0);
v_isShared_4727_ = v_isSharedCheck_4731_;
goto v_resetjp_4725_;
}
v_resetjp_4725_:
{
lean_object* v___x_4729_; 
if (v_isShared_4727_ == 0)
{
v___x_4729_ = v___x_4726_;
goto v_reusejp_4728_;
}
else
{
lean_object* v_reuseFailAlloc_4730_; 
v_reuseFailAlloc_4730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4730_, 0, v_a_4724_);
v___x_4729_ = v_reuseFailAlloc_4730_;
goto v_reusejp_4728_;
}
v_reusejp_4728_:
{
return v___x_4729_;
}
}
}
}
else
{
lean_object* v___x_4732_; lean_object* v___x_4733_; lean_object* v___x_4734_; 
v___x_4732_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__7);
v___x_4733_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__13));
v___x_4734_ = lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg(v___x_4732_, v___x_4733_, v_a_4626_, v_a_4627_, v_a_4628_, v_a_4629_);
if (lean_obj_tag(v___x_4734_) == 0)
{
lean_object* v_a_4735_; 
v_a_4735_ = lean_ctor_get(v___x_4734_, 0);
lean_inc(v_a_4735_);
lean_dec_ref_known(v___x_4734_, 1);
v_simpTheorems_4702_ = v_a_4735_;
v___y_4703_ = v_a_4622_;
v___y_4704_ = v_a_4623_;
v___y_4705_ = v_a_4624_;
v___y_4706_ = v_a_4625_;
v___y_4707_ = v_a_4626_;
v___y_4708_ = v_a_4627_;
v___y_4709_ = v_a_4628_;
v___y_4710_ = v_a_4629_;
goto v___jp_4701_;
}
else
{
lean_object* v_a_4736_; lean_object* v___x_4738_; uint8_t v_isShared_4739_; uint8_t v_isSharedCheck_4743_; 
lean_del_object(v___x_4639_);
lean_dec_ref(v_userConfig_4637_);
lean_dec_ref(v_config_4636_);
v_a_4736_ = lean_ctor_get(v___x_4734_, 0);
v_isSharedCheck_4743_ = !lean_is_exclusive(v___x_4734_);
if (v_isSharedCheck_4743_ == 0)
{
v___x_4738_ = v___x_4734_;
v_isShared_4739_ = v_isSharedCheck_4743_;
goto v_resetjp_4737_;
}
else
{
lean_inc(v_a_4736_);
lean_dec(v___x_4734_);
v___x_4738_ = lean_box(0);
v_isShared_4739_ = v_isSharedCheck_4743_;
goto v_resetjp_4737_;
}
v_resetjp_4737_:
{
lean_object* v___x_4741_; 
if (v_isShared_4739_ == 0)
{
v___x_4741_ = v___x_4738_;
goto v_reusejp_4740_;
}
else
{
lean_object* v_reuseFailAlloc_4742_; 
v_reuseFailAlloc_4742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4742_, 0, v_a_4736_);
v___x_4741_ = v_reuseFailAlloc_4742_;
goto v_reusejp_4740_;
}
v_reusejp_4740_:
{
return v___x_4741_;
}
}
}
}
v___jp_4641_:
{
lean_object* v___x_4652_; 
v___x_4652_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v___y_4651_);
if (lean_obj_tag(v___x_4652_) == 0)
{
lean_object* v_a_4653_; lean_object* v___x_4654_; lean_object* v___x_4655_; lean_object* v___x_4656_; lean_object* v___x_4657_; 
v_a_4653_ = lean_ctor_get(v___x_4652_, 0);
lean_inc(v_a_4653_);
lean_dec_ref_known(v___x_4652_, 1);
v___x_4654_ = lean_unsigned_to_nat(1u);
v___x_4655_ = lean_mk_empty_array_with_capacity(v___x_4654_);
lean_inc_ref(v___x_4655_);
v___x_4656_ = lean_array_push(v___x_4655_, v___y_4642_);
v___x_4657_ = l_Lean_Meta_Simp_mkContext___redArg(v_config_4636_, v___x_4656_, v_a_4653_, v_userConfig_4637_, v___y_4648_, v___y_4650_, v___y_4651_);
if (lean_obj_tag(v___x_4657_) == 0)
{
lean_object* v_a_4658_; lean_object* v___x_4659_; lean_object* v___x_4660_; lean_object* v___x_4661_; uint8_t v___x_4662_; lean_object* v___x_4663_; 
v_a_4658_ = lean_ctor_get(v___x_4657_, 0);
lean_inc(v_a_4658_);
lean_dec_ref_known(v___x_4657_, 1);
v___x_4659_ = lean_unsigned_to_nat(0u);
v___x_4660_ = l_Lean_Syntax_getArg(v_args_4620_, v___x_4659_);
v___x_4661_ = lean_array_push(v___x_4655_, v_simprocs_4643_);
v___x_4662_ = 0;
v___x_4663_ = l_Lean_Elab_Tactic_elabSimpArgs(v___x_4660_, v_a_4658_, v___x_4661_, v___x_4631_, v___x_4662_, v___x_4631_, v___y_4644_, v___y_4645_, v___y_4646_, v___y_4647_, v___y_4648_, v___y_4649_, v___y_4650_, v___y_4651_);
if (lean_obj_tag(v___x_4663_) == 0)
{
lean_object* v_a_4664_; lean_object* v___x_4666_; uint8_t v_isShared_4667_; uint8_t v_isSharedCheck_4676_; 
v_a_4664_ = lean_ctor_get(v___x_4663_, 0);
v_isSharedCheck_4676_ = !lean_is_exclusive(v___x_4663_);
if (v_isSharedCheck_4676_ == 0)
{
v___x_4666_ = v___x_4663_;
v_isShared_4667_ = v_isSharedCheck_4676_;
goto v_resetjp_4665_;
}
else
{
lean_inc(v_a_4664_);
lean_dec(v___x_4663_);
v___x_4666_ = lean_box(0);
v_isShared_4667_ = v_isSharedCheck_4676_;
goto v_resetjp_4665_;
}
v_resetjp_4665_:
{
lean_object* v_ctx_4668_; lean_object* v_simprocs_4669_; lean_object* v___x_4671_; 
v_ctx_4668_ = lean_ctor_get(v_a_4664_, 0);
lean_inc_ref(v_ctx_4668_);
v_simprocs_4669_ = lean_ctor_get(v_a_4664_, 1);
lean_inc_ref(v_simprocs_4669_);
lean_dec(v_a_4664_);
if (v_isShared_4640_ == 0)
{
lean_ctor_set(v___x_4639_, 1, v_simprocs_4669_);
lean_ctor_set(v___x_4639_, 0, v_ctx_4668_);
v___x_4671_ = v___x_4639_;
goto v_reusejp_4670_;
}
else
{
lean_object* v_reuseFailAlloc_4675_; 
v_reuseFailAlloc_4675_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4675_, 0, v_ctx_4668_);
lean_ctor_set(v_reuseFailAlloc_4675_, 1, v_simprocs_4669_);
v___x_4671_ = v_reuseFailAlloc_4675_;
goto v_reusejp_4670_;
}
v_reusejp_4670_:
{
lean_object* v___x_4673_; 
if (v_isShared_4667_ == 0)
{
lean_ctor_set(v___x_4666_, 0, v___x_4671_);
v___x_4673_ = v___x_4666_;
goto v_reusejp_4672_;
}
else
{
lean_object* v_reuseFailAlloc_4674_; 
v_reuseFailAlloc_4674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4674_, 0, v___x_4671_);
v___x_4673_ = v_reuseFailAlloc_4674_;
goto v_reusejp_4672_;
}
v_reusejp_4672_:
{
return v___x_4673_;
}
}
}
}
else
{
lean_object* v_a_4677_; lean_object* v___x_4679_; uint8_t v_isShared_4680_; uint8_t v_isSharedCheck_4684_; 
lean_del_object(v___x_4639_);
v_a_4677_ = lean_ctor_get(v___x_4663_, 0);
v_isSharedCheck_4684_ = !lean_is_exclusive(v___x_4663_);
if (v_isSharedCheck_4684_ == 0)
{
v___x_4679_ = v___x_4663_;
v_isShared_4680_ = v_isSharedCheck_4684_;
goto v_resetjp_4678_;
}
else
{
lean_inc(v_a_4677_);
lean_dec(v___x_4663_);
v___x_4679_ = lean_box(0);
v_isShared_4680_ = v_isSharedCheck_4684_;
goto v_resetjp_4678_;
}
v_resetjp_4678_:
{
lean_object* v___x_4682_; 
if (v_isShared_4680_ == 0)
{
v___x_4682_ = v___x_4679_;
goto v_reusejp_4681_;
}
else
{
lean_object* v_reuseFailAlloc_4683_; 
v_reuseFailAlloc_4683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4683_, 0, v_a_4677_);
v___x_4682_ = v_reuseFailAlloc_4683_;
goto v_reusejp_4681_;
}
v_reusejp_4681_:
{
return v___x_4682_;
}
}
}
}
else
{
lean_object* v_a_4685_; lean_object* v___x_4687_; uint8_t v_isShared_4688_; uint8_t v_isSharedCheck_4692_; 
lean_dec_ref(v___x_4655_);
lean_dec_ref(v_simprocs_4643_);
lean_del_object(v___x_4639_);
v_a_4685_ = lean_ctor_get(v___x_4657_, 0);
v_isSharedCheck_4692_ = !lean_is_exclusive(v___x_4657_);
if (v_isSharedCheck_4692_ == 0)
{
v___x_4687_ = v___x_4657_;
v_isShared_4688_ = v_isSharedCheck_4692_;
goto v_resetjp_4686_;
}
else
{
lean_inc(v_a_4685_);
lean_dec(v___x_4657_);
v___x_4687_ = lean_box(0);
v_isShared_4688_ = v_isSharedCheck_4692_;
goto v_resetjp_4686_;
}
v_resetjp_4686_:
{
lean_object* v___x_4690_; 
if (v_isShared_4688_ == 0)
{
v___x_4690_ = v___x_4687_;
goto v_reusejp_4689_;
}
else
{
lean_object* v_reuseFailAlloc_4691_; 
v_reuseFailAlloc_4691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4691_, 0, v_a_4685_);
v___x_4690_ = v_reuseFailAlloc_4691_;
goto v_reusejp_4689_;
}
v_reusejp_4689_:
{
return v___x_4690_;
}
}
}
}
else
{
lean_object* v_a_4693_; lean_object* v___x_4695_; uint8_t v_isShared_4696_; uint8_t v_isSharedCheck_4700_; 
lean_dec_ref(v_simprocs_4643_);
lean_dec_ref(v___y_4642_);
lean_del_object(v___x_4639_);
lean_dec_ref(v_userConfig_4637_);
lean_dec_ref(v_config_4636_);
v_a_4693_ = lean_ctor_get(v___x_4652_, 0);
v_isSharedCheck_4700_ = !lean_is_exclusive(v___x_4652_);
if (v_isSharedCheck_4700_ == 0)
{
v___x_4695_ = v___x_4652_;
v_isShared_4696_ = v_isSharedCheck_4700_;
goto v_resetjp_4694_;
}
else
{
lean_inc(v_a_4693_);
lean_dec(v___x_4652_);
v___x_4695_ = lean_box(0);
v_isShared_4696_ = v_isSharedCheck_4700_;
goto v_resetjp_4694_;
}
v_resetjp_4694_:
{
lean_object* v___x_4698_; 
if (v_isShared_4696_ == 0)
{
v___x_4698_ = v___x_4695_;
goto v_reusejp_4697_;
}
else
{
lean_object* v_reuseFailAlloc_4699_; 
v_reuseFailAlloc_4699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4699_, 0, v_a_4693_);
v___x_4698_ = v_reuseFailAlloc_4699_;
goto v_reusejp_4697_;
}
v_reusejp_4697_:
{
return v___x_4698_;
}
}
}
}
v___jp_4701_:
{
if (v_simpOnly_4621_ == 0)
{
lean_object* v___x_4711_; 
v___x_4711_ = l_Lean_Meta_Simp_getSimprocs___redArg(v___y_4710_);
if (lean_obj_tag(v___x_4711_) == 0)
{
lean_object* v_a_4712_; 
v_a_4712_ = lean_ctor_get(v___x_4711_, 0);
lean_inc(v_a_4712_);
lean_dec_ref_known(v___x_4711_, 1);
v___y_4642_ = v_simpTheorems_4702_;
v_simprocs_4643_ = v_a_4712_;
v___y_4644_ = v___y_4703_;
v___y_4645_ = v___y_4704_;
v___y_4646_ = v___y_4705_;
v___y_4647_ = v___y_4706_;
v___y_4648_ = v___y_4707_;
v___y_4649_ = v___y_4708_;
v___y_4650_ = v___y_4709_;
v___y_4651_ = v___y_4710_;
goto v___jp_4641_;
}
else
{
lean_object* v_a_4713_; lean_object* v___x_4715_; uint8_t v_isShared_4716_; uint8_t v_isSharedCheck_4720_; 
lean_dec_ref(v_simpTheorems_4702_);
lean_del_object(v___x_4639_);
lean_dec_ref(v_userConfig_4637_);
lean_dec_ref(v_config_4636_);
v_a_4713_ = lean_ctor_get(v___x_4711_, 0);
v_isSharedCheck_4720_ = !lean_is_exclusive(v___x_4711_);
if (v_isSharedCheck_4720_ == 0)
{
v___x_4715_ = v___x_4711_;
v_isShared_4716_ = v_isSharedCheck_4720_;
goto v_resetjp_4714_;
}
else
{
lean_inc(v_a_4713_);
lean_dec(v___x_4711_);
v___x_4715_ = lean_box(0);
v_isShared_4716_ = v_isSharedCheck_4720_;
goto v_resetjp_4714_;
}
v_resetjp_4714_:
{
lean_object* v___x_4718_; 
if (v_isShared_4716_ == 0)
{
v___x_4718_ = v___x_4715_;
goto v_reusejp_4717_;
}
else
{
lean_object* v_reuseFailAlloc_4719_; 
v_reuseFailAlloc_4719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4719_, 0, v_a_4713_);
v___x_4718_ = v_reuseFailAlloc_4719_;
goto v_reusejp_4717_;
}
v_reusejp_4717_:
{
return v___x_4718_;
}
}
}
}
else
{
lean_object* v___x_4721_; 
v___x_4721_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___closed__2);
v___y_4642_ = v_simpTheorems_4702_;
v_simprocs_4643_ = v___x_4721_;
v___y_4644_ = v___y_4703_;
v___y_4645_ = v___y_4704_;
v___y_4646_ = v___y_4705_;
v___y_4647_ = v___y_4706_;
v___y_4648_ = v___y_4707_;
v___y_4649_ = v___y_4708_;
v___y_4650_ = v___y_4709_;
v___y_4651_ = v___y_4710_;
goto v___jp_4641_;
}
}
}
}
else
{
lean_object* v_a_4745_; lean_object* v___x_4747_; uint8_t v_isShared_4748_; uint8_t v_isSharedCheck_4752_; 
v_a_4745_ = lean_ctor_get(v___x_4634_, 0);
v_isSharedCheck_4752_ = !lean_is_exclusive(v___x_4634_);
if (v_isSharedCheck_4752_ == 0)
{
v___x_4747_ = v___x_4634_;
v_isShared_4748_ = v_isSharedCheck_4752_;
goto v_resetjp_4746_;
}
else
{
lean_inc(v_a_4745_);
lean_dec(v___x_4634_);
v___x_4747_ = lean_box(0);
v_isShared_4748_ = v_isSharedCheck_4752_;
goto v_resetjp_4746_;
}
v_resetjp_4746_:
{
lean_object* v___x_4750_; 
if (v_isShared_4748_ == 0)
{
v___x_4750_ = v___x_4747_;
goto v_reusejp_4749_;
}
else
{
lean_object* v_reuseFailAlloc_4751_; 
v_reuseFailAlloc_4751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4751_, 0, v_a_4745_);
v___x_4750_ = v_reuseFailAlloc_4751_;
goto v_reusejp_4749_;
}
v_reusejp_4749_:
{
return v___x_4750_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_getSimpContext___boxed(lean_object* v_cfg_4753_, lean_object* v_args_4754_, lean_object* v_simpOnly_4755_, lean_object* v_a_4756_, lean_object* v_a_4757_, lean_object* v_a_4758_, lean_object* v_a_4759_, lean_object* v_a_4760_, lean_object* v_a_4761_, lean_object* v_a_4762_, lean_object* v_a_4763_, lean_object* v_a_4764_){
_start:
{
uint8_t v_simpOnly_boxed_4765_; lean_object* v_res_4766_; 
v_simpOnly_boxed_4765_ = lean_unbox(v_simpOnly_4755_);
v_res_4766_ = lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(v_cfg_4753_, v_args_4754_, v_simpOnly_boxed_4765_, v_a_4756_, v_a_4757_, v_a_4758_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, v_a_4763_);
lean_dec(v_a_4763_);
lean_dec_ref(v_a_4762_);
lean_dec(v_a_4761_);
lean_dec_ref(v_a_4760_);
lean_dec(v_a_4759_);
lean_dec_ref(v_a_4758_);
lean_dec(v_a_4757_);
lean_dec_ref(v_a_4756_);
lean_dec(v_args_4754_);
return v_res_4766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1(lean_object* v_x_4767_, lean_object* v_x_4768_, lean_object* v___y_4769_, lean_object* v___y_4770_, lean_object* v___y_4771_, lean_object* v___y_4772_, lean_object* v___y_4773_, lean_object* v___y_4774_, lean_object* v___y_4775_, lean_object* v___y_4776_){
_start:
{
lean_object* v___x_4778_; 
v___x_4778_ = lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___redArg(v_x_4767_, v_x_4768_, v___y_4773_, v___y_4774_, v___y_4775_, v___y_4776_);
return v___x_4778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1___boxed(lean_object* v_x_4779_, lean_object* v_x_4780_, lean_object* v___y_4781_, lean_object* v___y_4782_, lean_object* v___y_4783_, lean_object* v___y_4784_, lean_object* v___y_4785_, lean_object* v___y_4786_, lean_object* v___y_4787_, lean_object* v___y_4788_, lean_object* v___y_4789_){
_start:
{
lean_object* v_res_4790_; 
v_res_4790_ = lp_mathlib_List_foldlM___at___00Mathlib_Meta_NormNum_getSimpContext_spec__1(v_x_4779_, v_x_4780_, v___y_4781_, v___y_4782_, v___y_4783_, v___y_4784_, v___y_4785_, v___y_4786_, v___y_4787_, v___y_4788_);
lean_dec(v___y_4788_);
lean_dec_ref(v___y_4787_);
lean_dec(v___y_4786_);
lean_dec_ref(v___y_4785_);
lean_dec(v___y_4784_);
lean_dec_ref(v___y_4783_);
lean_dec(v___y_4782_);
lean_dec_ref(v___y_4781_);
return v_res_4790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0(lean_object* v_snd_4791_, uint8_t v_useSimp_4792_, lean_object* v_e_4793_, lean_object* v_ctx_4794_, lean_object* v___y_4795_, lean_object* v___y_4796_, lean_object* v___y_4797_, lean_object* v___y_4798_){
_start:
{
lean_object* v___x_4800_; 
v___x_4800_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_ctx_4794_, v_snd_4791_, v_useSimp_4792_, v_e_4793_, v___y_4795_, v___y_4796_, v___y_4797_, v___y_4798_);
return v___x_4800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0___boxed(lean_object* v_snd_4801_, lean_object* v_useSimp_4802_, lean_object* v_e_4803_, lean_object* v_ctx_4804_, lean_object* v___y_4805_, lean_object* v___y_4806_, lean_object* v___y_4807_, lean_object* v___y_4808_, lean_object* v___y_4809_){
_start:
{
uint8_t v_useSimp_boxed_4810_; lean_object* v_res_4811_; 
v_useSimp_boxed_4810_ = lean_unbox(v_useSimp_4802_);
v_res_4811_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0(v_snd_4801_, v_useSimp_boxed_4810_, v_e_4803_, v_ctx_4804_, v___y_4805_, v___y_4806_, v___y_4807_, v___y_4808_);
lean_dec(v___y_4808_);
lean_dec_ref(v___y_4807_);
lean_dec(v___y_4806_);
lean_dec_ref(v___y_4805_);
return v_res_4811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1(lean_object* v_cfg_4812_, lean_object* v_args_4813_, uint8_t v___y_4814_, uint8_t v_useSimp_4815_, lean_object* v_loc_4816_, lean_object* v___y_4817_, lean_object* v___y_4818_, lean_object* v___y_4819_, lean_object* v___y_4820_, lean_object* v___y_4821_, lean_object* v___y_4822_, lean_object* v___y_4823_, lean_object* v___y_4824_){
_start:
{
lean_object* v___x_4826_; 
v___x_4826_ = lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(v_cfg_4812_, v_args_4813_, v___y_4814_, v___y_4817_, v___y_4818_, v___y_4819_, v___y_4820_, v___y_4821_, v___y_4822_, v___y_4823_, v___y_4824_);
if (lean_obj_tag(v___x_4826_) == 0)
{
lean_object* v_a_4827_; lean_object* v_fst_4828_; lean_object* v_snd_4829_; lean_object* v___x_4830_; lean_object* v___f_4831_; lean_object* v___x_4832_; lean_object* v___x_4833_; uint8_t v___x_4834_; uint8_t v___x_4835_; lean_object* v___x_4836_; 
v_a_4827_ = lean_ctor_get(v___x_4826_, 0);
lean_inc(v_a_4827_);
lean_dec_ref_known(v___x_4826_, 1);
v_fst_4828_ = lean_ctor_get(v_a_4827_, 0);
lean_inc(v_fst_4828_);
v_snd_4829_ = lean_ctor_get(v_a_4827_, 1);
lean_inc(v_snd_4829_);
lean_dec(v_a_4827_);
v___x_4830_ = lean_box(v_useSimp_4815_);
v___f_4831_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__0___boxed), 9, 2);
lean_closure_set(v___f_4831_, 0, v_snd_4829_);
lean_closure_set(v___f_4831_, 1, v___x_4830_);
v___x_4832_ = l_Lean_Elab_Tactic_expandOptLocation(v_loc_4816_);
v___x_4833_ = ((lean_object*)(lp_mathlib_norm__num___closed__0));
v___x_4834_ = 0;
v___x_4835_ = 1;
v___x_4836_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_4831_, v___x_4833_, v___x_4832_, v___x_4834_, v___x_4835_, v_fst_4828_, v___y_4817_, v___y_4818_, v___y_4819_, v___y_4820_, v___y_4821_, v___y_4822_, v___y_4823_, v___y_4824_);
lean_dec(v___x_4832_);
return v___x_4836_;
}
else
{
lean_object* v_a_4837_; lean_object* v___x_4839_; uint8_t v_isShared_4840_; uint8_t v_isSharedCheck_4844_; 
v_a_4837_ = lean_ctor_get(v___x_4826_, 0);
v_isSharedCheck_4844_ = !lean_is_exclusive(v___x_4826_);
if (v_isSharedCheck_4844_ == 0)
{
v___x_4839_ = v___x_4826_;
v_isShared_4840_ = v_isSharedCheck_4844_;
goto v_resetjp_4838_;
}
else
{
lean_inc(v_a_4837_);
lean_dec(v___x_4826_);
v___x_4839_ = lean_box(0);
v_isShared_4840_ = v_isSharedCheck_4844_;
goto v_resetjp_4838_;
}
v_resetjp_4838_:
{
lean_object* v___x_4842_; 
if (v_isShared_4840_ == 0)
{
v___x_4842_ = v___x_4839_;
goto v_reusejp_4841_;
}
else
{
lean_object* v_reuseFailAlloc_4843_; 
v_reuseFailAlloc_4843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4843_, 0, v_a_4837_);
v___x_4842_ = v_reuseFailAlloc_4843_;
goto v_reusejp_4841_;
}
v_reusejp_4841_:
{
return v___x_4842_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1___boxed(lean_object* v_cfg_4845_, lean_object* v_args_4846_, lean_object* v___y_4847_, lean_object* v_useSimp_4848_, lean_object* v_loc_4849_, lean_object* v___y_4850_, lean_object* v___y_4851_, lean_object* v___y_4852_, lean_object* v___y_4853_, lean_object* v___y_4854_, lean_object* v___y_4855_, lean_object* v___y_4856_, lean_object* v___y_4857_, lean_object* v___y_4858_){
_start:
{
uint8_t v___y_231__boxed_4859_; uint8_t v_useSimp_boxed_4860_; lean_object* v_res_4861_; 
v___y_231__boxed_4859_ = lean_unbox(v___y_4847_);
v_useSimp_boxed_4860_ = lean_unbox(v_useSimp_4848_);
v_res_4861_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1(v_cfg_4845_, v_args_4846_, v___y_231__boxed_4859_, v_useSimp_boxed_4860_, v_loc_4849_, v___y_4850_, v___y_4851_, v___y_4852_, v___y_4853_, v___y_4854_, v___y_4855_, v___y_4856_, v___y_4857_);
lean_dec(v___y_4857_);
lean_dec_ref(v___y_4856_);
lean_dec(v___y_4855_);
lean_dec_ref(v___y_4854_);
lean_dec(v___y_4853_);
lean_dec_ref(v___y_4852_);
lean_dec(v___y_4851_);
lean_dec_ref(v___y_4850_);
lean_dec(v_loc_4849_);
lean_dec(v_args_4846_);
return v_res_4861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(lean_object* v_cfg_4862_, lean_object* v_args_4863_, lean_object* v_loc_4864_, uint8_t v_simpOnly_4865_, uint8_t v_useSimp_4866_, lean_object* v_a_4867_, lean_object* v_a_4868_, lean_object* v_a_4869_, lean_object* v_a_4870_, lean_object* v_a_4871_, lean_object* v_a_4872_, lean_object* v_a_4873_, lean_object* v_a_4874_){
_start:
{
uint8_t v___y_4877_; 
if (v_useSimp_4866_ == 0)
{
uint8_t v___x_4882_; 
v___x_4882_ = 1;
v___y_4877_ = v___x_4882_;
goto v___jp_4876_;
}
else
{
v___y_4877_ = v_simpOnly_4865_;
goto v___jp_4876_;
}
v___jp_4876_:
{
lean_object* v___x_4878_; lean_object* v___x_4879_; lean_object* v___f_4880_; lean_object* v___x_4881_; 
v___x_4878_ = lean_box(v___y_4877_);
v___x_4879_ = lean_box(v_useSimp_4866_);
v___f_4880_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___lam__1___boxed), 14, 5);
lean_closure_set(v___f_4880_, 0, v_cfg_4862_);
lean_closure_set(v___f_4880_, 1, v_args_4863_);
lean_closure_set(v___f_4880_, 2, v___x_4878_);
lean_closure_set(v___f_4880_, 3, v___x_4879_);
lean_closure_set(v___f_4880_, 4, v_loc_4864_);
v___x_4881_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4880_, v_a_4867_, v_a_4868_, v_a_4869_, v_a_4870_, v_a_4871_, v_a_4872_, v_a_4873_, v_a_4874_);
return v___x_4881_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_elabNormNum___boxed(lean_object* v_cfg_4883_, lean_object* v_args_4884_, lean_object* v_loc_4885_, lean_object* v_simpOnly_4886_, lean_object* v_useSimp_4887_, lean_object* v_a_4888_, lean_object* v_a_4889_, lean_object* v_a_4890_, lean_object* v_a_4891_, lean_object* v_a_4892_, lean_object* v_a_4893_, lean_object* v_a_4894_, lean_object* v_a_4895_, lean_object* v_a_4896_){
_start:
{
uint8_t v_simpOnly_boxed_4897_; uint8_t v_useSimp_boxed_4898_; lean_object* v_res_4899_; 
v_simpOnly_boxed_4897_ = lean_unbox(v_simpOnly_4886_);
v_useSimp_boxed_4898_ = lean_unbox(v_useSimp_4887_);
v_res_4899_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(v_cfg_4883_, v_args_4884_, v_loc_4885_, v_simpOnly_boxed_4897_, v_useSimp_boxed_4898_, v_a_4888_, v_a_4889_, v_a_4890_, v_a_4891_, v_a_4892_, v_a_4893_, v_a_4894_, v_a_4895_);
lean_dec(v_a_4895_);
lean_dec_ref(v_a_4894_);
lean_dec(v_a_4893_);
lean_dec_ref(v_a_4892_);
lean_dec(v_a_4891_);
lean_dec_ref(v_a_4890_);
lean_dec(v_a_4889_);
lean_dec_ref(v_a_4888_);
return v_res_4899_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__3(void){
_start:
{
lean_object* v___x_4908_; lean_object* v___x_4909_; lean_object* v___x_4910_; lean_object* v___x_4911_; 
v___x_4908_ = l_Lean_Parser_Tactic_optConfig;
v___x_4909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__2));
v___x_4910_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_4911_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4911_, 0, v___x_4910_);
lean_ctor_set(v___x_4911_, 1, v___x_4909_);
lean_ctor_set(v___x_4911_, 2, v___x_4908_);
return v___x_4911_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__9(void){
_start:
{
lean_object* v___x_4922_; lean_object* v___x_4923_; lean_object* v___x_4924_; lean_object* v___x_4925_; 
v___x_4922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__8));
v___x_4923_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__3, &lp_mathlib_Mathlib_Tactic_normNum___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__3);
v___x_4924_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_4925_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4925_, 0, v___x_4924_);
lean_ctor_set(v___x_4925_, 1, v___x_4923_);
lean_ctor_set(v___x_4925_, 2, v___x_4922_);
return v___x_4925_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__10(void){
_start:
{
lean_object* v___x_4926_; lean_object* v___x_4927_; lean_object* v___x_4928_; 
v___x_4926_ = l_Lean_Parser_Tactic_simpArgs;
v___x_4927_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__5));
v___x_4928_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4928_, 0, v___x_4927_);
lean_ctor_set(v___x_4928_, 1, v___x_4926_);
return v___x_4928_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__11(void){
_start:
{
lean_object* v___x_4929_; lean_object* v___x_4930_; lean_object* v___x_4931_; lean_object* v___x_4932_; 
v___x_4929_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__10, &lp_mathlib_Mathlib_Tactic_normNum___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__10);
v___x_4930_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__9, &lp_mathlib_Mathlib_Tactic_normNum___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__9);
v___x_4931_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_4932_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4932_, 0, v___x_4931_);
lean_ctor_set(v___x_4932_, 1, v___x_4930_);
lean_ctor_set(v___x_4932_, 2, v___x_4929_);
return v___x_4932_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__12(void){
_start:
{
lean_object* v___x_4933_; lean_object* v___x_4934_; lean_object* v___x_4935_; 
v___x_4933_ = l_Lean_Parser_Tactic_location;
v___x_4934_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__5));
v___x_4935_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4935_, 0, v___x_4934_);
lean_ctor_set(v___x_4935_, 1, v___x_4933_);
return v___x_4935_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__13(void){
_start:
{
lean_object* v___x_4936_; lean_object* v___x_4937_; lean_object* v___x_4938_; lean_object* v___x_4939_; 
v___x_4936_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__12, &lp_mathlib_Mathlib_Tactic_normNum___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__12);
v___x_4937_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__11, &lp_mathlib_Mathlib_Tactic_normNum___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__11);
v___x_4938_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_4939_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4939_, 0, v___x_4938_);
lean_ctor_set(v___x_4939_, 1, v___x_4937_);
lean_ctor_set(v___x_4939_, 2, v___x_4936_);
return v___x_4939_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum___closed__14(void){
_start:
{
lean_object* v___x_4940_; lean_object* v___x_4941_; lean_object* v___x_4942_; lean_object* v___x_4943_; 
v___x_4940_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__13, &lp_mathlib_Mathlib_Tactic_normNum___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__13);
v___x_4941_ = lean_unsigned_to_nat(1022u);
v___x_4942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__1));
v___x_4943_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4943_, 0, v___x_4942_);
lean_ctor_set(v___x_4943_, 1, v___x_4941_);
lean_ctor_set(v___x_4943_, 2, v___x_4940_);
return v___x_4943_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum(void){
_start:
{
lean_object* v___x_4944_; 
v___x_4944_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__14, &lp_mathlib_Mathlib_Tactic_normNum___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__14);
return v___x_4944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg(){
_start:
{
lean_object* v___x_4946_; lean_object* v___x_4947_; 
v___x_4946_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2__spec__1___redArg___closed__0);
v___x_4947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4947_, 0, v___x_4946_);
return v___x_4947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg___boxed(lean_object* v___y_4948_){
_start:
{
lean_object* v_res_4949_; 
v_res_4949_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg();
return v_res_4949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0(lean_object* v_00_u03b1_4950_, lean_object* v___y_4951_, lean_object* v___y_4952_, lean_object* v___y_4953_, lean_object* v___y_4954_, lean_object* v___y_4955_, lean_object* v___y_4956_, lean_object* v___y_4957_, lean_object* v___y_4958_){
_start:
{
lean_object* v___x_4960_; 
v___x_4960_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg();
return v___x_4960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___boxed(lean_object* v_00_u03b1_4961_, lean_object* v___y_4962_, lean_object* v___y_4963_, lean_object* v___y_4964_, lean_object* v___y_4965_, lean_object* v___y_4966_, lean_object* v___y_4967_, lean_object* v___y_4968_, lean_object* v___y_4969_, lean_object* v___y_4970_){
_start:
{
lean_object* v_res_4971_; 
v_res_4971_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0(v_00_u03b1_4961_, v___y_4962_, v___y_4963_, v___y_4964_, v___y_4965_, v___y_4966_, v___y_4967_, v___y_4968_, v___y_4969_);
lean_dec(v___y_4969_);
lean_dec_ref(v___y_4968_);
lean_dec(v___y_4967_);
lean_dec_ref(v___y_4966_);
lean_dec(v___y_4965_);
lean_dec_ref(v___y_4964_);
lean_dec(v___y_4963_);
lean_dec_ref(v___y_4962_);
return v_res_4971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1(lean_object* v_x_4972_, lean_object* v_a_4973_, lean_object* v_a_4974_, lean_object* v_a_4975_, lean_object* v_a_4976_, lean_object* v_a_4977_, lean_object* v_a_4978_, lean_object* v_a_4979_, lean_object* v_a_4980_){
_start:
{
lean_object* v___x_4982_; uint8_t v___x_4983_; 
v___x_4982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__1));
lean_inc(v_x_4972_);
v___x_4983_ = l_Lean_Syntax_isOfKind(v_x_4972_, v___x_4982_);
if (v___x_4983_ == 0)
{
lean_object* v___x_4984_; 
lean_dec(v_x_4972_);
v___x_4984_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg();
return v___x_4984_;
}
else
{
lean_object* v___x_4985_; lean_object* v___x_4986_; lean_object* v___x_4987_; lean_object* v___x_4988_; lean_object* v___x_4989_; lean_object* v___x_4990_; lean_object* v___x_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; 
v___x_4985_ = lean_unsigned_to_nat(1u);
v___x_4986_ = l_Lean_Syntax_getArg(v_x_4972_, v___x_4985_);
v___x_4987_ = lean_unsigned_to_nat(2u);
v___x_4988_ = l_Lean_Syntax_getArg(v_x_4972_, v___x_4987_);
v___x_4989_ = lean_unsigned_to_nat(3u);
v___x_4990_ = l_Lean_Syntax_getArg(v_x_4972_, v___x_4989_);
v___x_4991_ = lean_unsigned_to_nat(4u);
v___x_4992_ = l_Lean_Syntax_getArg(v_x_4972_, v___x_4991_);
lean_dec(v_x_4972_);
v___x_4993_ = l_Lean_Syntax_getOptional_x3f(v___x_4988_);
lean_dec(v___x_4988_);
if (lean_obj_tag(v___x_4993_) == 0)
{
uint8_t v___x_4994_; lean_object* v___x_4995_; 
v___x_4994_ = 0;
v___x_4995_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(v___x_4986_, v___x_4990_, v___x_4992_, v___x_4994_, v___x_4983_, v_a_4973_, v_a_4974_, v_a_4975_, v_a_4976_, v_a_4977_, v_a_4978_, v_a_4979_, v_a_4980_);
return v___x_4995_;
}
else
{
lean_object* v___x_4996_; 
lean_dec_ref_known(v___x_4993_, 1);
v___x_4996_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(v___x_4986_, v___x_4990_, v___x_4992_, v___x_4983_, v___x_4983_, v_a_4973_, v_a_4974_, v_a_4975_, v_a_4976_, v_a_4977_, v_a_4978_, v_a_4979_, v_a_4980_);
return v___x_4996_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1___boxed(lean_object* v_x_4997_, lean_object* v_a_4998_, lean_object* v_a_4999_, lean_object* v_a_5000_, lean_object* v_a_5001_, lean_object* v_a_5002_, lean_object* v_a_5003_, lean_object* v_a_5004_, lean_object* v_a_5005_, lean_object* v_a_5006_){
_start:
{
lean_object* v_res_5007_; 
v_res_5007_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1(v_x_4997_, v_a_4998_, v_a_4999_, v_a_5000_, v_a_5001_, v_a_5002_, v_a_5003_, v_a_5004_, v_a_5005_);
lean_dec(v_a_5005_);
lean_dec_ref(v_a_5004_);
lean_dec(v_a_5003_);
lean_dec_ref(v_a_5002_);
lean_dec(v_a_5001_);
lean_dec_ref(v_a_5000_);
lean_dec(v_a_4999_);
lean_dec_ref(v_a_4998_);
return v_res_5007_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum1___closed__4(void){
_start:
{
lean_object* v___x_5017_; lean_object* v___x_5018_; lean_object* v___x_5019_; lean_object* v___x_5020_; 
v___x_5017_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__12, &lp_mathlib_Mathlib_Tactic_normNum___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__12);
v___x_5018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum1___closed__3));
v___x_5019_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5020_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5020_, 0, v___x_5019_);
lean_ctor_set(v___x_5020_, 1, v___x_5018_);
lean_ctor_set(v___x_5020_, 2, v___x_5017_);
return v___x_5020_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum1___closed__5(void){
_start:
{
lean_object* v___x_5021_; lean_object* v___x_5022_; lean_object* v___x_5023_; lean_object* v___x_5024_; 
v___x_5021_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum1___closed__4, &lp_mathlib_Mathlib_Tactic_normNum1___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_normNum1___closed__4);
v___x_5022_ = lean_unsigned_to_nat(1022u);
v___x_5023_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum1___closed__1));
v___x_5024_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_5024_, 0, v___x_5023_);
lean_ctor_set(v___x_5024_, 1, v___x_5022_);
lean_ctor_set(v___x_5024_, 2, v___x_5021_);
return v___x_5024_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNum1(void){
_start:
{
lean_object* v___x_5025_; 
v___x_5025_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum1___closed__5, &lp_mathlib_Mathlib_Tactic_normNum1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_normNum1___closed__5);
return v___x_5025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1(lean_object* v_x_5030_, lean_object* v_a_5031_, lean_object* v_a_5032_, lean_object* v_a_5033_, lean_object* v_a_5034_, lean_object* v_a_5035_, lean_object* v_a_5036_, lean_object* v_a_5037_, lean_object* v_a_5038_){
_start:
{
lean_object* v___x_5040_; uint8_t v___x_5041_; 
v___x_5040_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum1___closed__1));
lean_inc(v_x_5030_);
v___x_5041_ = l_Lean_Syntax_isOfKind(v_x_5030_, v___x_5040_);
if (v___x_5041_ == 0)
{
lean_object* v___x_5042_; 
lean_dec(v_x_5030_);
v___x_5042_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum__1_spec__0___redArg();
return v___x_5042_;
}
else
{
lean_object* v___x_5043_; lean_object* v___x_5044_; lean_object* v___x_5045_; uint8_t v___x_5046_; lean_object* v___x_5047_; 
v___x_5043_ = lean_unsigned_to_nat(1u);
v___x_5044_ = l_Lean_Syntax_getArg(v_x_5030_, v___x_5043_);
lean_dec(v_x_5030_);
v___x_5045_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___closed__0));
v___x_5046_ = 0;
v___x_5047_ = lp_mathlib_Mathlib_Meta_NormNum_elabNormNum(v___x_5045_, v___x_5045_, v___x_5044_, v___x_5041_, v___x_5046_, v_a_5031_, v_a_5032_, v_a_5033_, v_a_5034_, v_a_5035_, v_a_5036_, v_a_5037_, v_a_5038_);
return v___x_5047_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1___boxed(lean_object* v_x_5048_, lean_object* v_a_5049_, lean_object* v_a_5050_, lean_object* v_a_5051_, lean_object* v_a_5052_, lean_object* v_a_5053_, lean_object* v_a_5054_, lean_object* v_a_5055_, lean_object* v_a_5056_, lean_object* v_a_5057_){
_start:
{
lean_object* v_res_5058_; 
v_res_5058_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______elabRules__Mathlib__Tactic__normNum1__1(v_x_5048_, v_a_5049_, v_a_5050_, v_a_5051_, v_a_5052_, v_a_5053_, v_a_5054_, v_a_5055_, v_a_5056_);
lean_dec(v_a_5056_);
lean_dec_ref(v_a_5055_);
lean_dec(v_a_5054_);
lean_dec_ref(v_a_5053_);
lean_dec(v_a_5052_);
lean_dec_ref(v_a_5051_);
lean_dec(v_a_5050_);
lean_dec_ref(v_a_5049_);
return v_res_5058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(lean_object* v_e_5069_, lean_object* v___y_5070_){
_start:
{
uint8_t v___x_5072_; 
v___x_5072_ = l_Lean_Expr_hasMVar(v_e_5069_);
if (v___x_5072_ == 0)
{
lean_object* v___x_5073_; 
v___x_5073_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5073_, 0, v_e_5069_);
return v___x_5073_;
}
else
{
lean_object* v___x_5074_; lean_object* v_mctx_5075_; lean_object* v___x_5076_; lean_object* v_fst_5077_; lean_object* v_snd_5078_; lean_object* v___x_5079_; lean_object* v_cache_5080_; lean_object* v_zetaDeltaFVarIds_5081_; lean_object* v_postponed_5082_; lean_object* v_diag_5083_; lean_object* v___x_5085_; uint8_t v_isShared_5086_; uint8_t v_isSharedCheck_5092_; 
v___x_5074_ = lean_st_ref_get(v___y_5070_);
v_mctx_5075_ = lean_ctor_get(v___x_5074_, 0);
lean_inc_ref(v_mctx_5075_);
lean_dec(v___x_5074_);
v___x_5076_ = l_Lean_instantiateMVarsCore(v_mctx_5075_, v_e_5069_);
v_fst_5077_ = lean_ctor_get(v___x_5076_, 0);
lean_inc(v_fst_5077_);
v_snd_5078_ = lean_ctor_get(v___x_5076_, 1);
lean_inc(v_snd_5078_);
lean_dec_ref(v___x_5076_);
v___x_5079_ = lean_st_ref_take(v___y_5070_);
v_cache_5080_ = lean_ctor_get(v___x_5079_, 1);
v_zetaDeltaFVarIds_5081_ = lean_ctor_get(v___x_5079_, 2);
v_postponed_5082_ = lean_ctor_get(v___x_5079_, 3);
v_diag_5083_ = lean_ctor_get(v___x_5079_, 4);
v_isSharedCheck_5092_ = !lean_is_exclusive(v___x_5079_);
if (v_isSharedCheck_5092_ == 0)
{
lean_object* v_unused_5093_; 
v_unused_5093_ = lean_ctor_get(v___x_5079_, 0);
lean_dec(v_unused_5093_);
v___x_5085_ = v___x_5079_;
v_isShared_5086_ = v_isSharedCheck_5092_;
goto v_resetjp_5084_;
}
else
{
lean_inc(v_diag_5083_);
lean_inc(v_postponed_5082_);
lean_inc(v_zetaDeltaFVarIds_5081_);
lean_inc(v_cache_5080_);
lean_dec(v___x_5079_);
v___x_5085_ = lean_box(0);
v_isShared_5086_ = v_isSharedCheck_5092_;
goto v_resetjp_5084_;
}
v_resetjp_5084_:
{
lean_object* v___x_5088_; 
if (v_isShared_5086_ == 0)
{
lean_ctor_set(v___x_5085_, 0, v_snd_5078_);
v___x_5088_ = v___x_5085_;
goto v_reusejp_5087_;
}
else
{
lean_object* v_reuseFailAlloc_5091_; 
v_reuseFailAlloc_5091_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5091_, 0, v_snd_5078_);
lean_ctor_set(v_reuseFailAlloc_5091_, 1, v_cache_5080_);
lean_ctor_set(v_reuseFailAlloc_5091_, 2, v_zetaDeltaFVarIds_5081_);
lean_ctor_set(v_reuseFailAlloc_5091_, 3, v_postponed_5082_);
lean_ctor_set(v_reuseFailAlloc_5091_, 4, v_diag_5083_);
v___x_5088_ = v_reuseFailAlloc_5091_;
goto v_reusejp_5087_;
}
v_reusejp_5087_:
{
lean_object* v___x_5089_; lean_object* v___x_5090_; 
v___x_5089_ = lean_st_ref_set(v___y_5070_, v___x_5088_);
v___x_5090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5090_, 0, v_fst_5077_);
return v___x_5090_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg___boxed(lean_object* v_e_5094_, lean_object* v___y_5095_, lean_object* v___y_5096_){
_start:
{
lean_object* v_res_5097_; 
v_res_5097_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(v_e_5094_, v___y_5095_);
lean_dec(v___y_5095_);
return v_res_5097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0(lean_object* v_e_5098_, lean_object* v___y_5099_, lean_object* v___y_5100_, lean_object* v___y_5101_, lean_object* v___y_5102_, lean_object* v___y_5103_, lean_object* v___y_5104_, lean_object* v___y_5105_, lean_object* v___y_5106_){
_start:
{
lean_object* v___x_5108_; 
v___x_5108_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(v_e_5098_, v___y_5104_);
return v___x_5108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___boxed(lean_object* v_e_5109_, lean_object* v___y_5110_, lean_object* v___y_5111_, lean_object* v___y_5112_, lean_object* v___y_5113_, lean_object* v___y_5114_, lean_object* v___y_5115_, lean_object* v___y_5116_, lean_object* v___y_5117_, lean_object* v___y_5118_){
_start:
{
lean_object* v_res_5119_; 
v_res_5119_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0(v_e_5109_, v___y_5110_, v___y_5111_, v___y_5112_, v___y_5113_, v___y_5114_, v___y_5115_, v___y_5116_, v___y_5117_);
lean_dec(v___y_5117_);
lean_dec_ref(v___y_5116_);
lean_dec(v___y_5115_);
lean_dec_ref(v___y_5114_);
lean_dec(v___y_5113_);
lean_dec_ref(v___y_5112_);
lean_dec(v___y_5111_);
lean_dec_ref(v___y_5110_);
return v_res_5119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0(lean_object* v___x_5120_, uint8_t v___x_5121_, lean_object* v___y_5122_, lean_object* v___y_5123_, lean_object* v___y_5124_, lean_object* v___y_5125_, lean_object* v___y_5126_, lean_object* v___y_5127_, lean_object* v___y_5128_, lean_object* v___y_5129_){
_start:
{
lean_object* v___x_5131_; 
lean_inc(v___x_5120_);
v___x_5131_ = lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(v___x_5120_, v___x_5120_, v___x_5121_, v___y_5122_, v___y_5123_, v___y_5124_, v___y_5125_, v___y_5126_, v___y_5127_, v___y_5128_, v___y_5129_);
lean_dec(v___x_5120_);
if (lean_obj_tag(v___x_5131_) == 0)
{
lean_object* v_a_5132_; lean_object* v_fst_5133_; lean_object* v_snd_5134_; lean_object* v___x_5135_; 
v_a_5132_ = lean_ctor_get(v___x_5131_, 0);
lean_inc(v_a_5132_);
lean_dec_ref_known(v___x_5131_, 1);
v_fst_5133_ = lean_ctor_get(v_a_5132_, 0);
lean_inc(v_fst_5133_);
v_snd_5134_ = lean_ctor_get(v_a_5132_, 1);
lean_inc(v_snd_5134_);
lean_dec(v_a_5132_);
v___x_5135_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_5123_, v___y_5126_, v___y_5127_, v___y_5128_, v___y_5129_);
if (lean_obj_tag(v___x_5135_) == 0)
{
lean_object* v_a_5136_; lean_object* v___x_5137_; lean_object* v_a_5138_; uint8_t v___x_5139_; lean_object* v___x_5140_; 
v_a_5136_ = lean_ctor_get(v___x_5135_, 0);
lean_inc(v_a_5136_);
lean_dec_ref_known(v___x_5135_, 1);
v___x_5137_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(v_a_5136_, v___y_5127_);
v_a_5138_ = lean_ctor_get(v___x_5137_, 0);
lean_inc(v_a_5138_);
lean_dec_ref(v___x_5137_);
v___x_5139_ = 0;
v___x_5140_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_fst_5133_, v_snd_5134_, v___x_5139_, v_a_5138_, v___y_5126_, v___y_5127_, v___y_5128_, v___y_5129_);
if (lean_obj_tag(v___x_5140_) == 0)
{
lean_object* v_a_5141_; lean_object* v___x_5142_; 
v_a_5141_ = lean_ctor_get(v___x_5140_, 0);
lean_inc(v_a_5141_);
lean_dec_ref_known(v___x_5140_, 1);
v___x_5142_ = l_Lean_Elab_Tactic_Conv_applySimpResult(v_a_5141_, v___y_5122_, v___y_5123_, v___y_5124_, v___y_5125_, v___y_5126_, v___y_5127_, v___y_5128_, v___y_5129_);
return v___x_5142_;
}
else
{
lean_object* v_a_5143_; lean_object* v___x_5145_; uint8_t v_isShared_5146_; uint8_t v_isSharedCheck_5150_; 
v_a_5143_ = lean_ctor_get(v___x_5140_, 0);
v_isSharedCheck_5150_ = !lean_is_exclusive(v___x_5140_);
if (v_isSharedCheck_5150_ == 0)
{
v___x_5145_ = v___x_5140_;
v_isShared_5146_ = v_isSharedCheck_5150_;
goto v_resetjp_5144_;
}
else
{
lean_inc(v_a_5143_);
lean_dec(v___x_5140_);
v___x_5145_ = lean_box(0);
v_isShared_5146_ = v_isSharedCheck_5150_;
goto v_resetjp_5144_;
}
v_resetjp_5144_:
{
lean_object* v___x_5148_; 
if (v_isShared_5146_ == 0)
{
v___x_5148_ = v___x_5145_;
goto v_reusejp_5147_;
}
else
{
lean_object* v_reuseFailAlloc_5149_; 
v_reuseFailAlloc_5149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5149_, 0, v_a_5143_);
v___x_5148_ = v_reuseFailAlloc_5149_;
goto v_reusejp_5147_;
}
v_reusejp_5147_:
{
return v___x_5148_;
}
}
}
}
else
{
lean_object* v_a_5151_; lean_object* v___x_5153_; uint8_t v_isShared_5154_; uint8_t v_isSharedCheck_5158_; 
lean_dec(v_snd_5134_);
lean_dec(v_fst_5133_);
v_a_5151_ = lean_ctor_get(v___x_5135_, 0);
v_isSharedCheck_5158_ = !lean_is_exclusive(v___x_5135_);
if (v_isSharedCheck_5158_ == 0)
{
v___x_5153_ = v___x_5135_;
v_isShared_5154_ = v_isSharedCheck_5158_;
goto v_resetjp_5152_;
}
else
{
lean_inc(v_a_5151_);
lean_dec(v___x_5135_);
v___x_5153_ = lean_box(0);
v_isShared_5154_ = v_isSharedCheck_5158_;
goto v_resetjp_5152_;
}
v_resetjp_5152_:
{
lean_object* v___x_5156_; 
if (v_isShared_5154_ == 0)
{
v___x_5156_ = v___x_5153_;
goto v_reusejp_5155_;
}
else
{
lean_object* v_reuseFailAlloc_5157_; 
v_reuseFailAlloc_5157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5157_, 0, v_a_5151_);
v___x_5156_ = v_reuseFailAlloc_5157_;
goto v_reusejp_5155_;
}
v_reusejp_5155_:
{
return v___x_5156_;
}
}
}
}
else
{
lean_object* v_a_5159_; lean_object* v___x_5161_; uint8_t v_isShared_5162_; uint8_t v_isSharedCheck_5166_; 
v_a_5159_ = lean_ctor_get(v___x_5131_, 0);
v_isSharedCheck_5166_ = !lean_is_exclusive(v___x_5131_);
if (v_isSharedCheck_5166_ == 0)
{
v___x_5161_ = v___x_5131_;
v_isShared_5162_ = v_isSharedCheck_5166_;
goto v_resetjp_5160_;
}
else
{
lean_inc(v_a_5159_);
lean_dec(v___x_5131_);
v___x_5161_ = lean_box(0);
v_isShared_5162_ = v_isSharedCheck_5166_;
goto v_resetjp_5160_;
}
v_resetjp_5160_:
{
lean_object* v___x_5164_; 
if (v_isShared_5162_ == 0)
{
v___x_5164_ = v___x_5161_;
goto v_reusejp_5163_;
}
else
{
lean_object* v_reuseFailAlloc_5165_; 
v_reuseFailAlloc_5165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5165_, 0, v_a_5159_);
v___x_5164_ = v_reuseFailAlloc_5165_;
goto v_reusejp_5163_;
}
v_reusejp_5163_:
{
return v___x_5164_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0___boxed(lean_object* v___x_5167_, lean_object* v___x_5168_, lean_object* v___y_5169_, lean_object* v___y_5170_, lean_object* v___y_5171_, lean_object* v___y_5172_, lean_object* v___y_5173_, lean_object* v___y_5174_, lean_object* v___y_5175_, lean_object* v___y_5176_, lean_object* v___y_5177_){
_start:
{
uint8_t v___x_1299__boxed_5178_; lean_object* v_res_5179_; 
v___x_1299__boxed_5178_ = lean_unbox(v___x_5168_);
v_res_5179_ = lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___lam__0(v___x_5167_, v___x_1299__boxed_5178_, v___y_5169_, v___y_5170_, v___y_5171_, v___y_5172_, v___y_5173_, v___y_5174_, v___y_5175_, v___y_5176_);
lean_dec(v___y_5176_);
lean_dec_ref(v___y_5175_);
lean_dec(v___y_5174_);
lean_dec_ref(v___y_5173_);
lean_dec(v___y_5172_);
lean_dec_ref(v___y_5171_);
lean_dec(v___y_5170_);
lean_dec_ref(v___y_5169_);
return v_res_5179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg(lean_object* v_a_5184_, lean_object* v_a_5185_, lean_object* v_a_5186_, lean_object* v_a_5187_, lean_object* v_a_5188_, lean_object* v_a_5189_, lean_object* v_a_5190_, lean_object* v_a_5191_){
_start:
{
lean_object* v___f_5193_; lean_object* v___x_5194_; 
v___f_5193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___closed__0));
v___x_5194_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5193_, v_a_5184_, v_a_5185_, v_a_5186_, v_a_5187_, v_a_5188_, v_a_5189_, v_a_5190_, v_a_5191_);
return v___x_5194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg___boxed(lean_object* v_a_5195_, lean_object* v_a_5196_, lean_object* v_a_5197_, lean_object* v_a_5198_, lean_object* v_a_5199_, lean_object* v_a_5200_, lean_object* v_a_5201_, lean_object* v_a_5202_, lean_object* v_a_5203_){
_start:
{
lean_object* v_res_5204_; 
v_res_5204_ = lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg(v_a_5195_, v_a_5196_, v_a_5197_, v_a_5198_, v_a_5199_, v_a_5200_, v_a_5201_, v_a_5202_);
lean_dec(v_a_5202_);
lean_dec_ref(v_a_5201_);
lean_dec(v_a_5200_);
lean_dec_ref(v_a_5199_);
lean_dec(v_a_5198_);
lean_dec_ref(v_a_5197_);
lean_dec(v_a_5196_);
lean_dec_ref(v_a_5195_);
return v_res_5204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv(lean_object* v_x_5205_, lean_object* v_a_5206_, lean_object* v_a_5207_, lean_object* v_a_5208_, lean_object* v_a_5209_, lean_object* v_a_5210_, lean_object* v_a_5211_, lean_object* v_a_5212_, lean_object* v_a_5213_){
_start:
{
lean_object* v___x_5215_; 
v___x_5215_ = lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___redArg(v_a_5206_, v_a_5207_, v_a_5208_, v_a_5209_, v_a_5210_, v_a_5211_, v_a_5212_, v_a_5213_);
return v___x_5215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNum1Conv___boxed(lean_object* v_x_5216_, lean_object* v_a_5217_, lean_object* v_a_5218_, lean_object* v_a_5219_, lean_object* v_a_5220_, lean_object* v_a_5221_, lean_object* v_a_5222_, lean_object* v_a_5223_, lean_object* v_a_5224_, lean_object* v_a_5225_){
_start:
{
lean_object* v_res_5226_; 
v_res_5226_ = lp_mathlib_Mathlib_Tactic_elabNormNum1Conv(v_x_5216_, v_a_5217_, v_a_5218_, v_a_5219_, v_a_5220_, v_a_5221_, v_a_5222_, v_a_5223_, v_a_5224_);
lean_dec(v_a_5224_);
lean_dec_ref(v_a_5223_);
lean_dec(v_a_5222_);
lean_dec_ref(v_a_5221_);
lean_dec(v_a_5220_);
lean_dec_ref(v_a_5219_);
lean_dec(v_a_5218_);
lean_dec_ref(v_a_5217_);
lean_dec(v_x_5216_);
return v_res_5226_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumConv___closed__2(void){
_start:
{
lean_object* v___x_5232_; lean_object* v___x_5233_; lean_object* v___x_5234_; lean_object* v___x_5235_; 
v___x_5232_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__11, &lp_mathlib_Mathlib_Tactic_normNum___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__11);
v___x_5233_ = lean_unsigned_to_nat(1022u);
v___x_5234_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumConv___closed__1));
v___x_5235_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_5235_, 0, v___x_5234_);
lean_ctor_set(v___x_5235_, 1, v___x_5233_);
lean_ctor_set(v___x_5235_, 2, v___x_5232_);
return v___x_5235_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumConv(void){
_start:
{
lean_object* v___x_5236_; 
v___x_5236_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumConv___closed__2, &lp_mathlib_Mathlib_Tactic_normNumConv___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_normNumConv___closed__2);
return v___x_5236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0(lean_object* v___x_5237_, lean_object* v___x_5238_, uint8_t v___y_5239_, lean_object* v___y_5240_, lean_object* v___y_5241_, lean_object* v___y_5242_, lean_object* v___y_5243_, lean_object* v___y_5244_, lean_object* v___y_5245_, lean_object* v___y_5246_, lean_object* v___y_5247_){
_start:
{
lean_object* v___x_5249_; 
v___x_5249_ = lp_mathlib_Mathlib_Meta_NormNum_getSimpContext(v___x_5237_, v___x_5238_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_, v___y_5243_, v___y_5244_, v___y_5245_, v___y_5246_, v___y_5247_);
if (lean_obj_tag(v___x_5249_) == 0)
{
lean_object* v_a_5250_; lean_object* v_fst_5251_; lean_object* v_snd_5252_; lean_object* v___x_5253_; 
v_a_5250_ = lean_ctor_get(v___x_5249_, 0);
lean_inc(v_a_5250_);
lean_dec_ref_known(v___x_5249_, 1);
v_fst_5251_ = lean_ctor_get(v_a_5250_, 0);
lean_inc(v_fst_5251_);
v_snd_5252_ = lean_ctor_get(v_a_5250_, 1);
lean_inc(v_snd_5252_);
lean_dec(v_a_5250_);
v___x_5253_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_5241_, v___y_5244_, v___y_5245_, v___y_5246_, v___y_5247_);
if (lean_obj_tag(v___x_5253_) == 0)
{
lean_object* v_a_5254_; lean_object* v___x_5255_; lean_object* v_a_5256_; uint8_t v___x_5257_; lean_object* v___x_5258_; 
v_a_5254_ = lean_ctor_get(v___x_5253_, 0);
lean_inc(v_a_5254_);
lean_dec_ref_known(v___x_5253_, 1);
v___x_5255_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_elabNormNum1Conv_spec__0___redArg(v_a_5254_, v___y_5245_);
v_a_5256_ = lean_ctor_get(v___x_5255_, 0);
lean_inc(v_a_5256_);
lean_dec_ref(v___x_5255_);
v___x_5257_ = 1;
v___x_5258_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_fst_5251_, v_snd_5252_, v___x_5257_, v_a_5256_, v___y_5244_, v___y_5245_, v___y_5246_, v___y_5247_);
if (lean_obj_tag(v___x_5258_) == 0)
{
lean_object* v_a_5259_; lean_object* v___x_5260_; 
v_a_5259_ = lean_ctor_get(v___x_5258_, 0);
lean_inc(v_a_5259_);
lean_dec_ref_known(v___x_5258_, 1);
v___x_5260_ = l_Lean_Elab_Tactic_Conv_applySimpResult(v_a_5259_, v___y_5240_, v___y_5241_, v___y_5242_, v___y_5243_, v___y_5244_, v___y_5245_, v___y_5246_, v___y_5247_);
return v___x_5260_;
}
else
{
lean_object* v_a_5261_; lean_object* v___x_5263_; uint8_t v_isShared_5264_; uint8_t v_isSharedCheck_5268_; 
v_a_5261_ = lean_ctor_get(v___x_5258_, 0);
v_isSharedCheck_5268_ = !lean_is_exclusive(v___x_5258_);
if (v_isSharedCheck_5268_ == 0)
{
v___x_5263_ = v___x_5258_;
v_isShared_5264_ = v_isSharedCheck_5268_;
goto v_resetjp_5262_;
}
else
{
lean_inc(v_a_5261_);
lean_dec(v___x_5258_);
v___x_5263_ = lean_box(0);
v_isShared_5264_ = v_isSharedCheck_5268_;
goto v_resetjp_5262_;
}
v_resetjp_5262_:
{
lean_object* v___x_5266_; 
if (v_isShared_5264_ == 0)
{
v___x_5266_ = v___x_5263_;
goto v_reusejp_5265_;
}
else
{
lean_object* v_reuseFailAlloc_5267_; 
v_reuseFailAlloc_5267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5267_, 0, v_a_5261_);
v___x_5266_ = v_reuseFailAlloc_5267_;
goto v_reusejp_5265_;
}
v_reusejp_5265_:
{
return v___x_5266_;
}
}
}
}
else
{
lean_object* v_a_5269_; lean_object* v___x_5271_; uint8_t v_isShared_5272_; uint8_t v_isSharedCheck_5276_; 
lean_dec(v_snd_5252_);
lean_dec(v_fst_5251_);
v_a_5269_ = lean_ctor_get(v___x_5253_, 0);
v_isSharedCheck_5276_ = !lean_is_exclusive(v___x_5253_);
if (v_isSharedCheck_5276_ == 0)
{
v___x_5271_ = v___x_5253_;
v_isShared_5272_ = v_isSharedCheck_5276_;
goto v_resetjp_5270_;
}
else
{
lean_inc(v_a_5269_);
lean_dec(v___x_5253_);
v___x_5271_ = lean_box(0);
v_isShared_5272_ = v_isSharedCheck_5276_;
goto v_resetjp_5270_;
}
v_resetjp_5270_:
{
lean_object* v___x_5274_; 
if (v_isShared_5272_ == 0)
{
v___x_5274_ = v___x_5271_;
goto v_reusejp_5273_;
}
else
{
lean_object* v_reuseFailAlloc_5275_; 
v_reuseFailAlloc_5275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5275_, 0, v_a_5269_);
v___x_5274_ = v_reuseFailAlloc_5275_;
goto v_reusejp_5273_;
}
v_reusejp_5273_:
{
return v___x_5274_;
}
}
}
}
else
{
lean_object* v_a_5277_; lean_object* v___x_5279_; uint8_t v_isShared_5280_; uint8_t v_isSharedCheck_5284_; 
v_a_5277_ = lean_ctor_get(v___x_5249_, 0);
v_isSharedCheck_5284_ = !lean_is_exclusive(v___x_5249_);
if (v_isSharedCheck_5284_ == 0)
{
v___x_5279_ = v___x_5249_;
v_isShared_5280_ = v_isSharedCheck_5284_;
goto v_resetjp_5278_;
}
else
{
lean_inc(v_a_5277_);
lean_dec(v___x_5249_);
v___x_5279_ = lean_box(0);
v_isShared_5280_ = v_isSharedCheck_5284_;
goto v_resetjp_5278_;
}
v_resetjp_5278_:
{
lean_object* v___x_5282_; 
if (v_isShared_5280_ == 0)
{
v___x_5282_ = v___x_5279_;
goto v_reusejp_5281_;
}
else
{
lean_object* v_reuseFailAlloc_5283_; 
v_reuseFailAlloc_5283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5283_, 0, v_a_5277_);
v___x_5282_ = v_reuseFailAlloc_5283_;
goto v_reusejp_5281_;
}
v_reusejp_5281_:
{
return v___x_5282_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0___boxed(lean_object* v___x_5285_, lean_object* v___x_5286_, lean_object* v___y_5287_, lean_object* v___y_5288_, lean_object* v___y_5289_, lean_object* v___y_5290_, lean_object* v___y_5291_, lean_object* v___y_5292_, lean_object* v___y_5293_, lean_object* v___y_5294_, lean_object* v___y_5295_, lean_object* v___y_5296_){
_start:
{
uint8_t v___y_550__boxed_5297_; lean_object* v_res_5298_; 
v___y_550__boxed_5297_ = lean_unbox(v___y_5287_);
v_res_5298_ = lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0(v___x_5285_, v___x_5286_, v___y_550__boxed_5297_, v___y_5288_, v___y_5289_, v___y_5290_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_);
lean_dec(v___y_5295_);
lean_dec_ref(v___y_5294_);
lean_dec(v___y_5293_);
lean_dec_ref(v___y_5292_);
lean_dec(v___y_5291_);
lean_dec_ref(v___y_5290_);
lean_dec(v___y_5289_);
lean_dec_ref(v___y_5288_);
lean_dec(v___x_5286_);
return v_res_5298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv(lean_object* v_stx_5299_, lean_object* v_a_5300_, lean_object* v_a_5301_, lean_object* v_a_5302_, lean_object* v_a_5303_, lean_object* v_a_5304_, lean_object* v_a_5305_, lean_object* v_a_5306_, lean_object* v_a_5307_){
_start:
{
lean_object* v___x_5309_; lean_object* v___x_5310_; lean_object* v___x_5311_; lean_object* v___x_5312_; uint8_t v___y_5314_; lean_object* v___x_5318_; lean_object* v___x_5319_; uint8_t v___x_5320_; 
v___x_5309_ = lean_unsigned_to_nat(1u);
v___x_5310_ = l_Lean_Syntax_getArg(v_stx_5299_, v___x_5309_);
v___x_5311_ = lean_unsigned_to_nat(3u);
v___x_5312_ = l_Lean_Syntax_getArg(v_stx_5299_, v___x_5311_);
v___x_5318_ = lean_unsigned_to_nat(2u);
v___x_5319_ = l_Lean_Syntax_getArg(v_stx_5299_, v___x_5318_);
v___x_5320_ = l_Lean_Syntax_isNone(v___x_5319_);
lean_dec(v___x_5319_);
if (v___x_5320_ == 0)
{
uint8_t v___x_5321_; 
v___x_5321_ = 1;
v___y_5314_ = v___x_5321_;
goto v___jp_5313_;
}
else
{
uint8_t v___x_5322_; 
v___x_5322_ = 0;
v___y_5314_ = v___x_5322_;
goto v___jp_5313_;
}
v___jp_5313_:
{
lean_object* v___x_5315_; lean_object* v___f_5316_; lean_object* v___x_5317_; 
v___x_5315_ = lean_box(v___y_5314_);
v___f_5316_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_elabNormNumConv___lam__0___boxed), 12, 3);
lean_closure_set(v___f_5316_, 0, v___x_5310_);
lean_closure_set(v___f_5316_, 1, v___x_5312_);
lean_closure_set(v___f_5316_, 2, v___x_5315_);
v___x_5317_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5316_, v_a_5300_, v_a_5301_, v_a_5302_, v_a_5303_, v_a_5304_, v_a_5305_, v_a_5306_, v_a_5307_);
return v___x_5317_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabNormNumConv___boxed(lean_object* v_stx_5323_, lean_object* v_a_5324_, lean_object* v_a_5325_, lean_object* v_a_5326_, lean_object* v_a_5327_, lean_object* v_a_5328_, lean_object* v_a_5329_, lean_object* v_a_5330_, lean_object* v_a_5331_, lean_object* v_a_5332_){
_start:
{
lean_object* v_res_5333_; 
v_res_5333_ = lp_mathlib_Mathlib_Tactic_elabNormNumConv(v_stx_5323_, v_a_5324_, v_a_5325_, v_a_5326_, v_a_5327_, v_a_5328_, v_a_5329_, v_a_5330_, v_a_5331_);
lean_dec(v_a_5331_);
lean_dec_ref(v_a_5330_);
lean_dec(v_a_5329_);
lean_dec_ref(v_a_5328_);
lean_dec(v_a_5327_);
lean_dec_ref(v_a_5326_);
lean_dec(v_a_5325_);
lean_dec_ref(v_a_5324_);
lean_dec(v_stx_5323_);
return v_res_5333_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4(void){
_start:
{
lean_object* v___x_5342_; lean_object* v___x_5343_; lean_object* v___x_5344_; lean_object* v___x_5345_; 
v___x_5342_ = l_Lean_Parser_Tactic_optConfig;
v___x_5343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumCmd___closed__3));
v___x_5344_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5345_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5345_, 0, v___x_5344_);
lean_ctor_set(v___x_5345_, 1, v___x_5343_);
lean_ctor_set(v___x_5345_, 2, v___x_5342_);
return v___x_5345_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5(void){
_start:
{
lean_object* v___x_5346_; lean_object* v___x_5347_; lean_object* v___x_5348_; lean_object* v___x_5349_; 
v___x_5346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNum___closed__8));
v___x_5347_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__4);
v___x_5348_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5349_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5349_, 0, v___x_5348_);
lean_ctor_set(v___x_5349_, 1, v___x_5347_);
lean_ctor_set(v___x_5349_, 2, v___x_5346_);
return v___x_5349_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6(void){
_start:
{
lean_object* v___x_5350_; lean_object* v___x_5351_; lean_object* v___x_5352_; lean_object* v___x_5353_; 
v___x_5350_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNum___closed__10, &lp_mathlib_Mathlib_Tactic_normNum___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_normNum___closed__10);
v___x_5351_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__5);
v___x_5352_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5353_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5353_, 0, v___x_5352_);
lean_ctor_set(v___x_5353_, 1, v___x_5351_);
lean_ctor_set(v___x_5353_, 2, v___x_5350_);
return v___x_5353_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10(void){
_start:
{
lean_object* v___x_5360_; lean_object* v___x_5361_; lean_object* v___x_5362_; lean_object* v___x_5363_; 
v___x_5360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumCmd___closed__9));
v___x_5361_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__6);
v___x_5362_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5363_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5363_, 0, v___x_5362_);
lean_ctor_set(v___x_5363_, 1, v___x_5361_);
lean_ctor_set(v___x_5363_, 2, v___x_5360_);
return v___x_5363_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17(void){
_start:
{
lean_object* v___x_5375_; lean_object* v___x_5376_; lean_object* v___x_5377_; lean_object* v___x_5378_; 
v___x_5375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumCmd___closed__16));
v___x_5376_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__10);
v___x_5377_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5378_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5378_, 0, v___x_5377_);
lean_ctor_set(v___x_5378_, 1, v___x_5376_);
lean_ctor_set(v___x_5378_, 2, v___x_5375_);
return v___x_5378_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18(void){
_start:
{
lean_object* v___x_5379_; lean_object* v___x_5380_; lean_object* v___x_5381_; lean_object* v___x_5382_; 
v___x_5379_ = ((lean_object*)(lp_mathlib_norm__num___closed__8));
v___x_5380_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__17);
v___x_5381_ = ((lean_object*)(lp_mathlib_norm__num___closed__3));
v___x_5382_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5382_, 0, v___x_5381_);
lean_ctor_set(v___x_5382_, 1, v___x_5380_);
lean_ctor_set(v___x_5382_, 2, v___x_5379_);
return v___x_5382_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19(void){
_start:
{
lean_object* v___x_5383_; lean_object* v___x_5384_; lean_object* v___x_5385_; lean_object* v___x_5386_; 
v___x_5383_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__18);
v___x_5384_ = lean_unsigned_to_nat(1022u);
v___x_5385_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1));
v___x_5386_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_5386_, 0, v___x_5385_);
lean_ctor_set(v___x_5386_, 1, v___x_5384_);
lean_ctor_set(v___x_5386_, 2, v___x_5383_);
return v___x_5386_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_normNumCmd(void){
_start:
{
lean_object* v___x_5387_; 
v___x_5387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19, &lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_normNumCmd___closed__19);
return v___x_5387_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5(void){
_start:
{
lean_object* v___x_5397_; 
v___x_5397_ = l_Array_mkArray0(lean_box(0));
return v___x_5397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1(lean_object* v_x_5399_, lean_object* v_a_5400_, lean_object* v_a_5401_){
_start:
{
lean_object* v___x_5402_; uint8_t v___x_5403_; 
v___x_5402_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumCmd___closed__1));
lean_inc(v_x_5399_);
v___x_5403_ = l_Lean_Syntax_isOfKind(v_x_5399_, v___x_5402_);
if (v___x_5403_ == 0)
{
lean_object* v___x_5404_; lean_object* v___x_5405_; 
lean_dec(v_x_5399_);
v___x_5404_ = lean_box(1);
v___x_5405_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5405_, 0, v___x_5404_);
lean_ctor_set(v___x_5405_, 1, v_a_5401_);
return v___x_5405_;
}
else
{
lean_object* v___x_5406_; lean_object* v___x_5407_; lean_object* v___x_5408_; lean_object* v___x_5409_; lean_object* v___x_5410_; lean_object* v___x_5411_; lean_object* v___x_5412_; lean_object* v___x_5413_; lean_object* v___y_5415_; lean_object* v___y_5416_; lean_object* v___y_5417_; lean_object* v___y_5418_; lean_object* v___y_5419_; lean_object* v___y_5420_; lean_object* v___y_5421_; lean_object* v___y_5422_; lean_object* v___y_5423_; lean_object* v___y_5432_; lean_object* v___y_5433_; lean_object* v___y_5434_; lean_object* v___y_5435_; lean_object* v___y_5436_; lean_object* v___y_5437_; lean_object* v___y_5438_; lean_object* v___y_5439_; lean_object* v___y_5440_; lean_object* v___y_5448_; lean_object* v___y_5449_; lean_object* v___y_5468_; lean_object* v___x_5479_; 
v___x_5406_ = lean_unsigned_to_nat(1u);
v___x_5407_ = l_Lean_Syntax_getArg(v_x_5399_, v___x_5406_);
v___x_5408_ = lean_unsigned_to_nat(2u);
v___x_5409_ = l_Lean_Syntax_getArg(v_x_5399_, v___x_5408_);
v___x_5410_ = lean_unsigned_to_nat(3u);
v___x_5411_ = l_Lean_Syntax_getArg(v_x_5399_, v___x_5410_);
v___x_5412_ = lean_unsigned_to_nat(6u);
v___x_5413_ = l_Lean_Syntax_getArg(v_x_5399_, v___x_5412_);
lean_dec(v_x_5399_);
v___x_5479_ = l_Lean_Syntax_getOptional_x3f(v___x_5411_);
lean_dec(v___x_5411_);
if (lean_obj_tag(v___x_5479_) == 0)
{
lean_object* v___x_5480_; 
v___x_5480_ = lean_box(0);
v___y_5468_ = v___x_5480_;
goto v___jp_5467_;
}
else
{
lean_object* v_val_5481_; lean_object* v___x_5483_; uint8_t v_isShared_5484_; uint8_t v_isSharedCheck_5488_; 
v_val_5481_ = lean_ctor_get(v___x_5479_, 0);
v_isSharedCheck_5488_ = !lean_is_exclusive(v___x_5479_);
if (v_isSharedCheck_5488_ == 0)
{
v___x_5483_ = v___x_5479_;
v_isShared_5484_ = v_isSharedCheck_5488_;
goto v_resetjp_5482_;
}
else
{
lean_inc(v_val_5481_);
lean_dec(v___x_5479_);
v___x_5483_ = lean_box(0);
v_isShared_5484_ = v_isSharedCheck_5488_;
goto v_resetjp_5482_;
}
v_resetjp_5482_:
{
lean_object* v___x_5486_; 
if (v_isShared_5484_ == 0)
{
v___x_5486_ = v___x_5483_;
goto v_reusejp_5485_;
}
else
{
lean_object* v_reuseFailAlloc_5487_; 
v_reuseFailAlloc_5487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5487_, 0, v_val_5481_);
v___x_5486_ = v_reuseFailAlloc_5487_;
goto v_reusejp_5485_;
}
v_reusejp_5485_:
{
v___y_5468_ = v___x_5486_;
goto v___jp_5467_;
}
}
}
v___jp_5414_:
{
lean_object* v___x_5424_; lean_object* v___x_5425_; lean_object* v___x_5426_; lean_object* v___x_5427_; lean_object* v___x_5428_; lean_object* v___x_5429_; lean_object* v___x_5430_; 
lean_inc_ref(v___y_5422_);
v___x_5424_ = l_Array_append___redArg(v___y_5422_, v___y_5423_);
lean_dec_ref(v___y_5423_);
lean_inc(v___y_5416_);
lean_inc_n(v___y_5417_, 3);
v___x_5425_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5425_, 0, v___y_5417_);
lean_ctor_set(v___x_5425_, 1, v___y_5416_);
lean_ctor_set(v___x_5425_, 2, v___x_5424_);
lean_inc(v___y_5419_);
v___x_5426_ = l_Lean_Syntax_node4(v___y_5417_, v___y_5419_, v___y_5418_, v___x_5407_, v___y_5421_, v___x_5425_);
v___x_5427_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__0));
v___x_5428_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5428_, 0, v___y_5417_);
lean_ctor_set(v___x_5428_, 1, v___x_5427_);
lean_inc(v___y_5415_);
v___x_5429_ = l_Lean_Syntax_node4(v___y_5417_, v___y_5415_, v___y_5420_, v___x_5426_, v___x_5428_, v___x_5413_);
v___x_5430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5430_, 0, v___x_5429_);
lean_ctor_set(v___x_5430_, 1, v_a_5401_);
return v___x_5430_;
}
v___jp_5431_:
{
lean_object* v___x_5441_; lean_object* v___x_5442_; 
lean_inc_ref(v___y_5439_);
v___x_5441_ = l_Array_append___redArg(v___y_5439_, v___y_5440_);
lean_dec_ref(v___y_5440_);
lean_inc(v___y_5433_);
lean_inc(v___y_5434_);
v___x_5442_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5442_, 0, v___y_5434_);
lean_ctor_set(v___x_5442_, 1, v___y_5433_);
lean_ctor_set(v___x_5442_, 2, v___x_5441_);
if (lean_obj_tag(v___y_5438_) == 0)
{
lean_object* v___x_5443_; 
v___x_5443_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___y_5415_ = v___y_5432_;
v___y_5416_ = v___y_5433_;
v___y_5417_ = v___y_5434_;
v___y_5418_ = v___y_5436_;
v___y_5419_ = v___y_5435_;
v___y_5420_ = v___y_5437_;
v___y_5421_ = v___x_5442_;
v___y_5422_ = v___y_5439_;
v___y_5423_ = v___x_5443_;
goto v___jp_5414_;
}
else
{
lean_object* v_val_5444_; lean_object* v___x_5445_; lean_object* v___x_5446_; 
v_val_5444_ = lean_ctor_get(v___y_5438_, 0);
lean_inc(v_val_5444_);
lean_dec_ref_known(v___y_5438_, 1);
v___x_5445_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___x_5446_ = lean_array_push(v___x_5445_, v_val_5444_);
v___y_5415_ = v___y_5432_;
v___y_5416_ = v___y_5433_;
v___y_5417_ = v___y_5434_;
v___y_5418_ = v___y_5436_;
v___y_5419_ = v___y_5435_;
v___y_5420_ = v___y_5437_;
v___y_5421_ = v___x_5442_;
v___y_5422_ = v___y_5439_;
v___y_5423_ = v___x_5446_;
goto v___jp_5414_;
}
}
v___jp_5447_:
{
lean_object* v_ref_5450_; uint8_t v___x_5451_; lean_object* v___x_5452_; lean_object* v___x_5453_; lean_object* v___x_5454_; lean_object* v___x_5455_; lean_object* v___x_5456_; lean_object* v___x_5457_; lean_object* v___x_5458_; lean_object* v___x_5459_; lean_object* v___x_5460_; 
v_ref_5450_ = lean_ctor_get(v_a_5400_, 5);
v___x_5451_ = 0;
v___x_5452_ = l_Lean_SourceInfo_fromRef(v_ref_5450_, v___x_5451_);
v___x_5453_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__3));
v___x_5454_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__4));
lean_inc_n(v___x_5452_, 2);
v___x_5455_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5455_, 0, v___x_5452_);
lean_ctor_set(v___x_5455_, 1, v___x_5454_);
v___x_5456_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_normNumConv___closed__1));
v___x_5457_ = ((lean_object*)(lp_mathlib_norm__num___closed__0));
v___x_5458_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5458_, 0, v___x_5452_);
lean_ctor_set(v___x_5458_, 1, v___x_5457_);
v___x_5459_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__8));
v___x_5460_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__5);
if (lean_obj_tag(v___y_5449_) == 1)
{
lean_object* v_val_5461_; lean_object* v___x_5462_; lean_object* v___x_5463_; lean_object* v___x_5464_; lean_object* v___x_5465_; 
v_val_5461_ = lean_ctor_get(v___y_5449_, 0);
lean_inc(v_val_5461_);
lean_dec_ref_known(v___y_5449_, 1);
v___x_5462_ = l_Lean_SourceInfo_fromRef(v_val_5461_, v___x_5403_);
lean_dec(v_val_5461_);
v___x_5463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___closed__6));
v___x_5464_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5464_, 0, v___x_5462_);
lean_ctor_set(v___x_5464_, 1, v___x_5463_);
v___x_5465_ = l_Array_mkArray1___redArg(v___x_5464_);
v___y_5432_ = v___x_5453_;
v___y_5433_ = v___x_5459_;
v___y_5434_ = v___x_5452_;
v___y_5435_ = v___x_5456_;
v___y_5436_ = v___x_5458_;
v___y_5437_ = v___x_5455_;
v___y_5438_ = v___y_5448_;
v___y_5439_ = v___x_5460_;
v___y_5440_ = v___x_5465_;
goto v___jp_5431_;
}
else
{
lean_object* v___x_5466_; 
lean_dec(v___y_5449_);
v___x_5466_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam___closed__4));
v___y_5432_ = v___x_5453_;
v___y_5433_ = v___x_5459_;
v___y_5434_ = v___x_5452_;
v___y_5435_ = v___x_5456_;
v___y_5436_ = v___x_5458_;
v___y_5437_ = v___x_5455_;
v___y_5438_ = v___y_5448_;
v___y_5439_ = v___x_5460_;
v___y_5440_ = v___x_5466_;
goto v___jp_5431_;
}
}
v___jp_5467_:
{
lean_object* v___x_5469_; 
v___x_5469_ = l_Lean_Syntax_getOptional_x3f(v___x_5409_);
lean_dec(v___x_5409_);
if (lean_obj_tag(v___x_5469_) == 0)
{
lean_object* v___x_5470_; 
v___x_5470_ = lean_box(0);
v___y_5448_ = v___y_5468_;
v___y_5449_ = v___x_5470_;
goto v___jp_5447_;
}
else
{
lean_object* v_val_5471_; lean_object* v___x_5473_; uint8_t v_isShared_5474_; uint8_t v_isSharedCheck_5478_; 
v_val_5471_ = lean_ctor_get(v___x_5469_, 0);
v_isSharedCheck_5478_ = !lean_is_exclusive(v___x_5469_);
if (v_isSharedCheck_5478_ == 0)
{
v___x_5473_ = v___x_5469_;
v_isShared_5474_ = v_isSharedCheck_5478_;
goto v_resetjp_5472_;
}
else
{
lean_inc(v_val_5471_);
lean_dec(v___x_5469_);
v___x_5473_ = lean_box(0);
v_isShared_5474_ = v_isSharedCheck_5478_;
goto v_resetjp_5472_;
}
v_resetjp_5472_:
{
lean_object* v___x_5476_; 
if (v_isShared_5474_ == 0)
{
v___x_5476_ = v___x_5473_;
goto v_reusejp_5475_;
}
else
{
lean_object* v_reuseFailAlloc_5477_; 
v_reuseFailAlloc_5477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5477_, 0, v_val_5471_);
v___x_5476_ = v_reuseFailAlloc_5477_;
goto v_reusejp_5475_;
}
v_reusejp_5475_:
{
v___y_5448_ = v___y_5468_;
v___y_5449_ = v___x_5476_;
goto v___jp_5447_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1___boxed(lean_object* v_x_5489_, lean_object* v_a_5490_, lean_object* v_a_5491_){
_start:
{
lean_object* v_res_5492_; 
v_res_5492_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__NormNum__Core______macroRules__Mathlib__Tactic__normNumCmd__1(v_x_5489_, v_a_5490_, v_a_5491_);
lean_dec_ref(v_a_5490_);
return v_res_5492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg(lean_object* v_a_5499_){
_start:
{
lean_object* v___x_5501_; lean_object* v_env_5502_; lean_object* v___x_5503_; lean_object* v___x_5504_; lean_object* v___x_5505_; lean_object* v___x_5506_; 
v___x_5501_ = lean_st_ref_get(v_a_5499_);
v_env_5502_ = lean_ctor_get(v___x_5501_, 0);
lean_inc_ref(v_env_5502_);
lean_dec(v___x_5501_);
v___x_5503_ = ((lean_object*)(lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__1));
v___x_5504_ = ((lean_object*)(lp_mathlib_norm__num___closed__0));
v___x_5505_ = ((lean_object*)(lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__2));
v___x_5506_ = l_Lean_Parser_runParserCategory(v_env_5502_, v___x_5503_, v___x_5504_, v___x_5505_);
if (lean_obj_tag(v___x_5506_) == 0)
{
lean_object* v___x_5508_; uint8_t v_isShared_5509_; uint8_t v_isSharedCheck_5514_; 
v_isSharedCheck_5514_ = !lean_is_exclusive(v___x_5506_);
if (v_isSharedCheck_5514_ == 0)
{
lean_object* v_unused_5515_; 
v_unused_5515_ = lean_ctor_get(v___x_5506_, 0);
lean_dec(v_unused_5515_);
v___x_5508_ = v___x_5506_;
v_isShared_5509_ = v_isSharedCheck_5514_;
goto v_resetjp_5507_;
}
else
{
lean_dec(v___x_5506_);
v___x_5508_ = lean_box(0);
v_isShared_5509_ = v_isSharedCheck_5514_;
goto v_resetjp_5507_;
}
v_resetjp_5507_:
{
lean_object* v___x_5510_; lean_object* v___x_5512_; 
v___x_5510_ = ((lean_object*)(lp_mathlib___auxTryTactic14848873054137305619___redArg___closed__3));
if (v_isShared_5509_ == 0)
{
lean_ctor_set(v___x_5508_, 0, v___x_5510_);
v___x_5512_ = v___x_5508_;
goto v_reusejp_5511_;
}
else
{
lean_object* v_reuseFailAlloc_5513_; 
v_reuseFailAlloc_5513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5513_, 0, v___x_5510_);
v___x_5512_ = v_reuseFailAlloc_5513_;
goto v_reusejp_5511_;
}
v_reusejp_5511_:
{
return v___x_5512_;
}
}
}
else
{
lean_object* v_a_5516_; lean_object* v___x_5518_; uint8_t v_isShared_5519_; uint8_t v_isSharedCheck_5526_; 
v_a_5516_ = lean_ctor_get(v___x_5506_, 0);
v_isSharedCheck_5526_ = !lean_is_exclusive(v___x_5506_);
if (v_isSharedCheck_5526_ == 0)
{
v___x_5518_ = v___x_5506_;
v_isShared_5519_ = v_isSharedCheck_5526_;
goto v_resetjp_5517_;
}
else
{
lean_inc(v_a_5516_);
lean_dec(v___x_5506_);
v___x_5518_ = lean_box(0);
v_isShared_5519_ = v_isSharedCheck_5526_;
goto v_resetjp_5517_;
}
v_resetjp_5517_:
{
lean_object* v___x_5520_; lean_object* v___x_5521_; lean_object* v___x_5522_; lean_object* v___x_5524_; 
v___x_5520_ = lean_unsigned_to_nat(1u);
v___x_5521_ = lean_mk_empty_array_with_capacity(v___x_5520_);
v___x_5522_ = lean_array_push(v___x_5521_, v_a_5516_);
if (v_isShared_5519_ == 0)
{
lean_ctor_set_tag(v___x_5518_, 0);
lean_ctor_set(v___x_5518_, 0, v___x_5522_);
v___x_5524_ = v___x_5518_;
goto v_reusejp_5523_;
}
else
{
lean_object* v_reuseFailAlloc_5525_; 
v_reuseFailAlloc_5525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5525_, 0, v___x_5522_);
v___x_5524_ = v_reuseFailAlloc_5525_;
goto v_reusejp_5523_;
}
v_reusejp_5523_:
{
return v___x_5524_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___redArg___boxed(lean_object* v_a_5527_, lean_object* v_a_5528_){
_start:
{
lean_object* v_res_5529_; 
v_res_5529_ = lp_mathlib___auxTryTactic14848873054137305619___redArg(v_a_5527_);
lean_dec(v_a_5527_);
return v_res_5529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619(lean_object* v___goal_5530_, lean_object* v___info_5531_, lean_object* v_a_5532_, lean_object* v_a_5533_, lean_object* v_a_5534_, lean_object* v_a_5535_){
_start:
{
lean_object* v___x_5537_; 
v___x_5537_ = lp_mathlib___auxTryTactic14848873054137305619___redArg(v_a_5535_);
return v___x_5537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic14848873054137305619___boxed(lean_object* v___goal_5538_, lean_object* v___info_5539_, lean_object* v_a_5540_, lean_object* v_a_5541_, lean_object* v_a_5542_, lean_object* v_a_5543_, lean_object* v_a_5544_){
_start:
{
lean_object* v_res_5545_; 
v_res_5545_ = lp_mathlib___auxTryTactic14848873054137305619(v___goal_5538_, v___info_5539_, v_a_5540_, v_a_5541_, v_a_5542_, v_a_5543_);
lean_dec(v_a_5543_);
lean_dec_ref(v_a_5542_);
lean_dec(v_a_5541_);
lean_dec_ref(v_a_5540_);
lean_dec_ref(v___info_5539_);
lean_dec(v___goal_5538_);
return v_res_5545_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Result(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Try(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Result(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Try_Collect(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Try_Collect(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_3041507515____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam = _init_lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_NormNumExt_name___autoParam);
lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default = _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums_default);
lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums = _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_instInhabitedNormNums);
res = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_525676807____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Meta_NormNum_normNumExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_normNumExt);
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_deriveNat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_deriveInt___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_deriveInt___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_deriveInt___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_deriveRat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_deriveRat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_deriveRat___auto__1);
res = lp_mathlib___private_Mathlib_Tactic_NormNum_Core_0__Mathlib_Meta_NormNum_initFn_00___x40_Mathlib_Tactic_NormNum_Core_1919626320____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_normNum = _init_lp_mathlib_Mathlib_Tactic_normNum();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_normNum);
lp_mathlib_Mathlib_Tactic_normNum1 = _init_lp_mathlib_Mathlib_Tactic_normNum1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_normNum1);
lp_mathlib_Mathlib_Tactic_normNumConv = _init_lp_mathlib_Mathlib_Tactic_normNumConv();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_normNumConv);
lp_mathlib_Mathlib_Tactic_normNumCmd = _init_lp_mathlib_Mathlib_Tactic_normNumCmd();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_normNumCmd);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Result(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Try(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Try_Collect(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Result(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Try_Collect(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
}
#ifdef __cplusplus
}
#endif
