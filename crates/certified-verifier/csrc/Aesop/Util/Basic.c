// Lean compiler output
// Module: Aesop.Util.Basic
// Imports: public import Init public meta import Init public import Aesop.Nanos public import Aesop.Util.UnorderedArraySet public import Lean.Meta.DiscrTree.Util public import Lean.Meta.Tactic.Simp.SimpTheorems public import Lean.Util.ForEachExpr public import Lean.Elab.Tactic.Basic public import Aesop.Index.DiscrTreeConfig import Lean.Meta.Tactic.TryThis
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
lean_object* l_Lean_Options_set___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_pruneSolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Expr_eqv___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_hash___boxed(lean_object*);
lean_object* l_Lean_MonadCacheT_instMonad___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ForEachExpr_visit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_value_x3f(lean_object*, uint8_t);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_Lean_PersistentArray_forIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonadControl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonadLift___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonad___aux__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_withContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isSorry(lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* lean_find_expr(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_hash___boxed(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_LocalContext_getUnusedName(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_LocalContext_addDecl(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_erase___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instBEqOrigin___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instHashableOrigin___lam__0___boxed(lean_object*);
uint8_t l_Lean_PersistentHashMap_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_IO_monoNanosNow___boxed(lean_object*);
uint8_t l_Lean_Meta_SimpTheorems_isLemma(lean_object*, lean_object*);
uint8_t l_Lean_Meta_SimpTheorems_isDeclToUnfold(lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Meta_Tactic_TryThis_getIndentAndColumn(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addSimpTheorem(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_registerDeclToUnfoldThms(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_time___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_monoNanosNow___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_time___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_time___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_PersistentHashSet_toList___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__1_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__5_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_PersistentHashSet_toArray___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__0_value;
static const lean_array_object lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isEmptyTrie___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isEmptyTrie___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isEmptyTrie(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isEmptyTrie___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_DiscrTree_Key_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_filterDiscrTree___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_addSimpEntry(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instBEqOrigin___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instHashableOrigin___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntries___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntries(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_SimpTheorems_simpEntries___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_SimpTheorems_simpEntries___closed__0 = (const lean_object*)&lp_aesop_Aesop_SimpTheorems_simpEntries___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_simpEntries(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_SimpTheorems_containsDecl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_containsDecl___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_eqv___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setThe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setThe(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7;
static const lean_array_object lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12;
static lean_once_cell_t lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2;
static const lean_ctor_object lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_runTacticMAsMetaM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runTacticMAsMetaM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSyntaxAsMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSyntaxAsMetaM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateSimpEntryPriority(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_getAppUpToDefeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_getAppUpToDefeq___closed__0 = (const lean_object*)&lp_aesop_Aesop_getAppUpToDefeq___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAppUpToDefeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAppUpToDefeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__0_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_partitionGoalsAndMVars_spec__2___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_runTacticMCapturingPostState___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_runTacticMCapturingPostState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_runTacticMCapturingPostState___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___closed__0 = (const lean_object*)&lp_aesop_Aesop_runTacticMCapturingPostState___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_runTacticMCapturingPostState___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___closed__1 = (const lean_object*)&lp_aesop_Aesop_runTacticMCapturingPostState___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_runTacticMCapturingPostState___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTacticMCapturingPostState___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___closed__2 = (const lean_object*)&lp_aesop_Aesop_runTacticMCapturingPostState___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticCapturingPostState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticCapturingPostState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSeqCapturingPostState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSeqCapturingPostState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__0 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__0_value;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__1 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__1_value;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__2 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__2_value;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__3 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__4 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__4_value;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__5 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value_aux_2),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__6 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__6_value;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__7 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__8 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_runTacticsCapturingPostState___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__9;
static const lean_string_object lp_aesop_Aesop_runTacticsCapturingPostState___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___closed__10 = (const lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticsCapturingPostState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0;
static const lean_string_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "withUnfoldingAll"};
static const lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__1 = (const lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_runTacticsCapturingPostState___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value_aux_2),((lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__1_value),LEAN_SCALAR_PTR_LITERAL(38, 182, 19, 172, 53, 51, 56, 135)}};
static const lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2 = (const lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2_value;
static const lean_string_object lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "with_unfolding_all"};
static const lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__3 = (const lean_object*)&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySyntax(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySyntax___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "  "};
static const lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Replace aesop with the proof it found"};
static const lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__2_value;
static const lean_string_object lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Try this:\n"};
static const lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_elabPattern_adjustCtx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabPattern(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabPattern___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "smallErrorMessages"};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(76, 2, 170, 51, 146, 185, 13, 169)}};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "(aesop) Print smaller error messages. Used for testing."};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(198, 13, 209, 97, 69, 79, 102, 158)}};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_smallErrorMessages;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_tacticsToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_tacticsToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_tacticsToMessageData___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_tacticsToMessageData___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_tacticsToMessageData___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticsToMessageData(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnusedNames(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnusedNames___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Aesop_Name_ofComponents_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Name_ofComponents(lean_object*);
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "analyze"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 114, 68, 229, 251, 70, 44, 204)}};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "proofs"};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(16, 182, 23, 133, 244, 85, 246, 31)}};
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_withPPAnalyze___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_withPPAnalyze___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_withPPAnalyze___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_runInMetaState___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_runInMetaState___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runInMetaState___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runInMetaState___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_runInMetaState___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_saveState___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runInMetaState___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_runInMetaState___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_lBoolOr(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_lBoolOr___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__3_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArrayLex___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArrayLex___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArrayLex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArrayLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArraySizeThenLex___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArraySizeThenLex___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArraySizeThenLex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArraySizeThenLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclStateRefT_x27__aesop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclStateRefT_x27__aesop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__0 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__0_value;
static const lean_string_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "debug"};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__1 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 231, 110, 25, 65, 121, 140, 223)}};
static const lean_ctor_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__1_value),LEAN_SCALAR_PTR_LITERAL(10, 160, 28, 169, 110, 1, 38, 58)}};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__2 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__2_value;
static const lean_string_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__3 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__4 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_isDefEqReducibleRigid___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__5;
static lean_once_cell_t lp_aesop_Aesop_isDefEqReducibleRigid___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_isDefEqReducibleRigid___closed__6;
static const lean_string_object lp_aesop_Aesop_isDefEqReducibleRigid___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≟ "};
static const lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__7 = (const lean_object*)&lp_aesop_Aesop_isDefEqReducibleRigid___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_isDefEqReducibleRigid___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__0(lean_object* v_start_1_, lean_object* v_a_2_, lean_object* v_toPure_3_, lean_object* v_stop_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_5_ = lean_nat_sub(v_stop_4_, v_start_1_);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v_a_2_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
v___x_7_ = lean_apply_2(v_toPure_3_, lean_box(0), v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__0___boxed(lean_object* v_start_8_, lean_object* v_a_9_, lean_object* v_toPure_10_, lean_object* v_stop_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_aesop_Aesop_time___redArg___lam__0(v_start_8_, v_a_9_, v_toPure_10_, v_stop_11_);
lean_dec(v_stop_11_);
lean_dec(v_start_8_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__1(lean_object* v_start_13_, lean_object* v_toPure_14_, lean_object* v_toBind_15_, lean_object* v___x_16_, lean_object* v_a_17_){
_start:
{
lean_object* v___f_18_; lean_object* v___x_19_; 
v___f_18_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_18_, 0, v_start_13_);
lean_closure_set(v___f_18_, 1, v_a_17_);
lean_closure_set(v___f_18_, 2, v_toPure_14_);
v___x_19_ = lean_apply_4(v_toBind_15_, lean_box(0), lean_box(0), v___x_16_, v___f_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg___lam__2(lean_object* v_toPure_20_, lean_object* v_toBind_21_, lean_object* v___x_22_, lean_object* v_x_23_, lean_object* v_start_24_){
_start:
{
lean_object* v___f_25_; lean_object* v___x_26_; 
lean_inc(v_toBind_21_);
v___f_25_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time___redArg___lam__1), 5, 4);
lean_closure_set(v___f_25_, 0, v_start_24_);
lean_closure_set(v___f_25_, 1, v_toPure_20_);
lean_closure_set(v___f_25_, 2, v_toBind_21_);
lean_closure_set(v___f_25_, 3, v___x_22_);
v___x_26_ = lean_apply_4(v_toBind_21_, lean_box(0), lean_box(0), v_x_23_, v___f_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time___redArg(lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_toApplicative_31_; lean_object* v_toBind_32_; lean_object* v_toPure_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___f_36_; lean_object* v___x_37_; 
v_toApplicative_31_ = lean_ctor_get(v_inst_28_, 0);
lean_inc_ref(v_toApplicative_31_);
v_toBind_32_ = lean_ctor_get(v_inst_28_, 1);
lean_inc_n(v_toBind_32_, 2);
lean_dec_ref(v_inst_28_);
v_toPure_33_ = lean_ctor_get(v_toApplicative_31_, 1);
lean_inc(v_toPure_33_);
lean_dec_ref(v_toApplicative_31_);
v___x_34_ = ((lean_object*)(lp_aesop_Aesop_time___redArg___closed__0));
v___x_35_ = lean_apply_2(v_inst_29_, lean_box(0), v___x_34_);
lean_inc(v___x_35_);
v___f_36_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time___redArg___lam__2), 5, 4);
lean_closure_set(v___f_36_, 0, v_toPure_33_);
lean_closure_set(v___f_36_, 1, v_toBind_32_);
lean_closure_set(v___f_36_, 2, v___x_35_);
lean_closure_set(v___f_36_, 3, v_x_30_);
v___x_37_ = lean_apply_4(v_toBind_32_, lean_box(0), lean_box(0), v___x_35_, v___f_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time(lean_object* v_m_38_, lean_object* v_00_u03b1_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_x_42_){
_start:
{
lean_object* v_toApplicative_43_; lean_object* v_toBind_44_; lean_object* v_toPure_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___f_48_; lean_object* v___x_49_; 
v_toApplicative_43_ = lean_ctor_get(v_inst_40_, 0);
lean_inc_ref(v_toApplicative_43_);
v_toBind_44_ = lean_ctor_get(v_inst_40_, 1);
lean_inc_n(v_toBind_44_, 2);
lean_dec_ref(v_inst_40_);
v_toPure_45_ = lean_ctor_get(v_toApplicative_43_, 1);
lean_inc(v_toPure_45_);
lean_dec_ref(v_toApplicative_43_);
v___x_46_ = ((lean_object*)(lp_aesop_Aesop_time___redArg___closed__0));
v___x_47_ = lean_apply_2(v_inst_41_, lean_box(0), v___x_46_);
lean_inc(v___x_47_);
v___f_48_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time___redArg___lam__2), 5, 4);
lean_closure_set(v___f_48_, 0, v_toPure_45_);
lean_closure_set(v___f_48_, 1, v_toBind_44_);
lean_closure_set(v___f_48_, 2, v___x_47_);
lean_closure_set(v___f_48_, 3, v_x_42_);
v___x_49_ = lean_apply_4(v_toBind_44_, lean_box(0), lean_box(0), v___x_47_, v___f_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__0(lean_object* v_start_50_, lean_object* v_toPure_51_, lean_object* v_stop_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = lean_nat_sub(v_stop_52_, v_start_50_);
v___x_54_ = lean_apply_2(v_toPure_51_, lean_box(0), v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__0___boxed(lean_object* v_start_55_, lean_object* v_toPure_56_, lean_object* v_stop_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_aesop_Aesop_time_x27___redArg___lam__0(v_start_55_, v_toPure_56_, v_stop_57_);
lean_dec(v_stop_57_);
lean_dec(v_start_55_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__1(lean_object* v_toBind_59_, lean_object* v___x_60_, lean_object* v___f_61_, lean_object* v_____r_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_apply_4(v_toBind_59_, lean_box(0), lean_box(0), v___x_60_, v___f_61_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg___lam__2(lean_object* v_toPure_64_, lean_object* v_toBind_65_, lean_object* v___x_66_, lean_object* v_x_67_, lean_object* v_start_68_){
_start:
{
lean_object* v___f_69_; lean_object* v___f_70_; lean_object* v___x_71_; 
v___f_69_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_69_, 0, v_start_68_);
lean_closure_set(v___f_69_, 1, v_toPure_64_);
lean_inc(v_toBind_65_);
v___f_70_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time_x27___redArg___lam__1), 4, 3);
lean_closure_set(v___f_70_, 0, v_toBind_65_);
lean_closure_set(v___f_70_, 1, v___x_66_);
lean_closure_set(v___f_70_, 2, v___f_69_);
v___x_71_ = lean_apply_4(v_toBind_65_, lean_box(0), lean_box(0), v_x_67_, v___f_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27___redArg(lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_x_74_){
_start:
{
lean_object* v_toApplicative_75_; lean_object* v_toBind_76_; lean_object* v_toPure_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___f_80_; lean_object* v___x_81_; 
v_toApplicative_75_ = lean_ctor_get(v_inst_72_, 0);
lean_inc_ref(v_toApplicative_75_);
v_toBind_76_ = lean_ctor_get(v_inst_72_, 1);
lean_inc_n(v_toBind_76_, 2);
lean_dec_ref(v_inst_72_);
v_toPure_77_ = lean_ctor_get(v_toApplicative_75_, 1);
lean_inc(v_toPure_77_);
lean_dec_ref(v_toApplicative_75_);
v___x_78_ = ((lean_object*)(lp_aesop_Aesop_time___redArg___closed__0));
v___x_79_ = lean_apply_2(v_inst_73_, lean_box(0), v___x_78_);
lean_inc(v___x_79_);
v___f_80_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_80_, 0, v_toPure_77_);
lean_closure_set(v___f_80_, 1, v_toBind_76_);
lean_closure_set(v___f_80_, 2, v___x_79_);
lean_closure_set(v___f_80_, 3, v_x_74_);
v___x_81_ = lean_apply_4(v_toBind_76_, lean_box(0), lean_box(0), v___x_79_, v___f_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_time_x27(lean_object* v_m_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_x_85_){
_start:
{
lean_object* v_toApplicative_86_; lean_object* v_toBind_87_; lean_object* v_toPure_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___f_91_; lean_object* v___x_92_; 
v_toApplicative_86_ = lean_ctor_get(v_inst_83_, 0);
lean_inc_ref(v_toApplicative_86_);
v_toBind_87_ = lean_ctor_get(v_inst_83_, 1);
lean_inc_n(v_toBind_87_, 2);
lean_dec_ref(v_inst_83_);
v_toPure_88_ = lean_ctor_get(v_toApplicative_86_, 1);
lean_inc(v_toPure_88_);
lean_dec_ref(v_toApplicative_86_);
v___x_89_ = ((lean_object*)(lp_aesop_Aesop_time___redArg___closed__0));
v___x_90_ = lean_apply_2(v_inst_84_, lean_box(0), v___x_89_);
lean_inc(v___x_90_);
v___f_91_ = lean_alloc_closure((void*)(lp_aesop_Aesop_time_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_91_, 0, v_toPure_88_);
lean_closure_set(v___f_91_, 1, v_toBind_87_);
lean_closure_set(v___f_91_, 2, v___x_90_);
lean_closure_set(v___f_91_, 3, v_x_85_);
v___x_92_ = lean_apply_4(v_toBind_87_, lean_box(0), lean_box(0), v___x_90_, v___f_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg___lam__0(lean_object* v_d_93_, lean_object* v_a_94_, lean_object* v_x_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_96_, 0, v_a_94_);
lean_ctor_set(v___x_96_, 1, v_d_93_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___redArg(lean_object* v_s_117_){
_start:
{
lean_object* v___f_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___f_118_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__0));
v___x_119_ = lean_box(0);
v___x_120_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_121_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_120_, v___f_118_, v_s_117_, v___x_119_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList(lean_object* v_00_u03b1_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_s_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___f_126_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__0));
v___x_127_ = lean_box(0);
v___x_128_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_129_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_128_, v___f_126_, v_s_125_, v___x_127_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toList___boxed(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_s_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_aesop_Aesop_PersistentHashSet_toList(v_00_u03b1_130_, v_inst_131_, v_inst_132_, v_s_133_);
lean_dec_ref(v_inst_132_);
lean_dec_ref(v_inst_131_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg___lam__0(lean_object* v_d_135_, lean_object* v_a_136_, lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lean_array_push(v_d_135_, v_a_136_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___redArg(lean_object* v_s_142_){
_start:
{
lean_object* v___f_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___f_143_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__0));
v___x_144_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1));
v___x_145_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_146_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_145_, v___f_143_, v_s_142_, v___x_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray(lean_object* v_00_u03b1_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_s_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___f_151_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__0));
v___x_152_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1));
v___x_153_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_154_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_153_, v___f_151_, v_s_150_, v___x_152_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toArray___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_s_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop_Aesop_PersistentHashSet_toArray(v_00_u03b1_155_, v_inst_156_, v_inst_157_, v_s_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_156_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___lam__0(lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_d_162_, lean_object* v_a_163_, lean_object* v_x_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_box(0);
v___x_166_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v_inst_160_, v_inst_161_, v_d_162_, v_a_163_, v___x_165_);
return v___x_166_;
}
}
static lean_object* _init_lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_167_ = lean_box(0);
v___x_168_ = lean_unsigned_to_nat(16u);
v___x_169_ = lean_mk_array(v___x_168_, v___x_167_);
return v___x_169_;
}
}
static lean_object* _init_lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_170_ = lean_obj_once(&lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0, &lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0_once, _init_lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__0);
v___x_171_ = lean_unsigned_to_nat(0u);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v___x_170_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg(lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_s_175_){
_start:
{
lean_object* v___f_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___f_176_ = lean_alloc_closure((void*)(lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___lam__0), 5, 2);
lean_closure_set(v___f_176_, 0, v_inst_173_);
lean_closure_set(v___f_176_, 1, v_inst_174_);
v___x_177_ = lean_obj_once(&lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1, &lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1_once, _init_lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg___closed__1);
v___x_178_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_179_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_178_, v___f_176_, v_s_175_, v___x_177_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_toHashSet(lean_object* v_00_u03b1_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_s_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_aesop_Aesop_PersistentHashSet_toHashSet___redArg(v_inst_181_, v_inst_182_, v_s_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter___redArg___lam__0(lean_object* v_p_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_d_188_, lean_object* v_a_189_, lean_object* v_x_190_){
_start:
{
lean_object* v___x_191_; uint8_t v___x_192_; 
lean_inc(v_a_189_);
v___x_191_ = lean_apply_1(v_p_185_, v_a_189_);
v___x_192_ = lean_unbox(v___x_191_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; 
v___x_193_ = l_Lean_PersistentHashMap_erase___redArg(v_inst_186_, v_inst_187_, v_d_188_, v_a_189_);
return v___x_193_;
}
else
{
lean_dec(v_a_189_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
return v_d_188_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter___redArg(lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_p_196_, lean_object* v_s_197_){
_start:
{
lean_object* v___f_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___f_198_ = lean_alloc_closure((void*)(lp_aesop_Aesop_PersistentHashSet_filter___redArg___lam__0), 6, 3);
lean_closure_set(v___f_198_, 0, v_p_196_);
lean_closure_set(v___f_198_, 1, v_inst_194_);
lean_closure_set(v___f_198_, 2, v_inst_195_);
v___x_199_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
lean_inc_ref(v_s_197_);
v___x_200_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_199_, v___f_198_, v_s_197_, v_s_197_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PersistentHashSet_filter(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_p_204_, lean_object* v_s_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_aesop_Aesop_PersistentHashSet_filter___redArg(v_inst_202_, v_inst_203_, v_p_204_, v_s_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(lean_object* v_x_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = l_Lean_Meta_saveState___redArg(v___y_209_, v___y_211_);
if (lean_obj_tag(v___x_213_) == 0)
{
lean_object* v_a_214_; lean_object* v_r_215_; 
v_a_214_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_a_214_);
lean_dec_ref_known(v___x_213_, 1);
lean_inc(v___y_211_);
lean_inc_ref(v___y_210_);
lean_inc(v___y_209_);
lean_inc_ref(v___y_208_);
v_r_215_ = lean_apply_5(v_x_207_, v___y_208_, v___y_209_, v___y_210_, v___y_211_, lean_box(0));
if (lean_obj_tag(v_r_215_) == 0)
{
lean_object* v_a_216_; lean_object* v___x_217_; 
v_a_216_ = lean_ctor_get(v_r_215_, 0);
lean_inc(v_a_216_);
lean_dec_ref_known(v_r_215_, 1);
v___x_217_ = l_Lean_Meta_SavedState_restore___redArg(v_a_214_, v___y_209_, v___y_211_);
lean_dec(v_a_214_);
if (lean_obj_tag(v___x_217_) == 0)
{
lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_224_ == 0)
{
lean_object* v_unused_225_; 
v_unused_225_ = lean_ctor_get(v___x_217_, 0);
lean_dec(v_unused_225_);
v___x_219_ = v___x_217_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_dec(v___x_217_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
lean_ctor_set(v___x_219_, 0, v_a_216_);
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_216_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
else
{
lean_object* v_a_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_233_; 
lean_dec(v_a_216_);
v_a_226_ = lean_ctor_get(v___x_217_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_233_ == 0)
{
v___x_228_ = v___x_217_;
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_a_226_);
lean_dec(v___x_217_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_231_; 
if (v_isShared_229_ == 0)
{
v___x_231_ = v___x_228_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_a_226_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
else
{
lean_object* v_a_234_; lean_object* v___x_235_; 
v_a_234_ = lean_ctor_get(v_r_215_, 0);
lean_inc(v_a_234_);
lean_dec_ref_known(v_r_215_, 1);
v___x_235_ = l_Lean_Meta_SavedState_restore___redArg(v_a_214_, v___y_209_, v___y_211_);
lean_dec(v_a_214_);
if (lean_obj_tag(v___x_235_) == 0)
{
lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_242_ == 0)
{
lean_object* v_unused_243_; 
v_unused_243_ = lean_ctor_get(v___x_235_, 0);
lean_dec(v_unused_243_);
v___x_237_ = v___x_235_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_dec(v___x_235_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
lean_ctor_set_tag(v___x_237_, 1);
lean_ctor_set(v___x_237_, 0, v_a_234_);
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_234_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
else
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_251_; 
lean_dec(v_a_234_);
v_a_244_ = lean_ctor_get(v___x_235_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_251_ == 0)
{
v___x_246_ = v___x_235_;
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v___x_235_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_249_; 
if (v_isShared_247_ == 0)
{
v___x_249_ = v___x_246_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_a_244_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec_ref(v_x_207_);
v_a_252_ = lean_ctor_get(v___x_213_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_213_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_213_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_213_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg___boxed(lean_object* v_x_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(v_x_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0(lean_object* v_00_u03b1_267_, lean_object* v_x_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(v_x_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___boxed(lean_object* v_00_u03b1_275_, lean_object* v_x_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0(v_00_u03b1_275_, v_x_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_);
lean_dec(v___y_280_);
lean_dec_ref(v___y_279_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0(lean_object* v_type_283_, uint8_t v___x_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = l_Lean_Meta_forallMetaTelescope(v_type_283_, v___x_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v_a_291_; lean_object* v_snd_292_; lean_object* v_snd_293_; lean_object* v___x_294_; 
v_a_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_a_291_);
lean_dec_ref_known(v___x_290_, 1);
v_snd_292_ = lean_ctor_get(v_a_291_, 1);
lean_inc(v_snd_292_);
lean_dec(v_a_291_);
v_snd_293_ = lean_ctor_get(v_snd_292_, 1);
lean_inc(v_snd_293_);
lean_dec(v_snd_292_);
v___x_294_ = lp_aesop_Aesop_mkDiscrTreePath(v_snd_293_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
return v___x_294_;
}
else
{
lean_object* v_a_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_302_; 
v_a_295_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_302_ == 0)
{
v___x_297_ = v___x_290_;
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
else
{
lean_inc(v_a_295_);
lean_dec(v___x_290_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v___x_300_; 
if (v_isShared_298_ == 0)
{
v___x_300_ = v___x_297_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_a_295_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0___boxed(lean_object* v_type_303_, lean_object* v___x_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_){
_start:
{
uint8_t v___x_1075__boxed_310_; lean_object* v_res_311_; 
v___x_1075__boxed_310_ = lean_unbox(v___x_304_);
v_res_311_ = lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0(v_type_303_, v___x_1075__boxed_310_, v___y_305_, v___y_306_, v___y_307_, v___y_308_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys(lean_object* v_type_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_){
_start:
{
uint8_t v___x_318_; lean_object* v___x_319_; lean_object* v___f_320_; lean_object* v___x_321_; 
v___x_318_ = 0;
v___x_319_ = lean_box(v___x_318_);
v___f_320_ = lean_alloc_closure((void*)(lp_aesop_Aesop_getConclusionDiscrTreeKeys___lam__0___boxed), 7, 2);
lean_closure_set(v___f_320_, 0, v_type_312_);
lean_closure_set(v___f_320_, 1, v___x_319_);
v___x_321_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(v___f_320_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys___boxed(lean_object* v_type_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_aesop_Aesop_getConclusionDiscrTreeKeys(v_type_322_, v_a_323_, v_a_324_, v_a_325_, v_a_326_);
lean_dec(v_a_326_);
lean_dec_ref(v_a_325_);
lean_dec(v_a_324_);
lean_dec_ref(v_a_323_);
return v_res_328_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_isEmptyTrie___redArg(lean_object* v_x_329_){
_start:
{
lean_object* v_vs_330_; lean_object* v_children_331_; lean_object* v___x_332_; lean_object* v___x_333_; uint8_t v___x_334_; 
v_vs_330_ = lean_ctor_get(v_x_329_, 0);
v_children_331_ = lean_ctor_get(v_x_329_, 1);
v___x_332_ = lean_array_get_size(v_vs_330_);
v___x_333_ = lean_unsigned_to_nat(0u);
v___x_334_ = lean_nat_dec_eq(v___x_332_, v___x_333_);
if (v___x_334_ == 0)
{
return v___x_334_;
}
else
{
lean_object* v___x_335_; uint8_t v___x_336_; 
v___x_335_ = lean_array_get_size(v_children_331_);
v___x_336_ = lean_nat_dec_eq(v___x_335_, v___x_333_);
return v___x_336_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isEmptyTrie___redArg___boxed(lean_object* v_x_337_){
_start:
{
uint8_t v_res_338_; lean_object* v_r_339_; 
v_res_338_ = lp_aesop_Aesop_isEmptyTrie___redArg(v_x_337_);
lean_dec_ref(v_x_337_);
v_r_339_ = lean_box(v_res_338_);
return v_r_339_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_isEmptyTrie(lean_object* v_00_u03b1_340_, lean_object* v_x_341_){
_start:
{
uint8_t v___x_342_; 
v___x_342_ = lp_aesop_Aesop_isEmptyTrie___redArg(v_x_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isEmptyTrie___boxed(lean_object* v_00_u03b1_343_, lean_object* v_x_344_){
_start:
{
uint8_t v_res_345_; lean_object* v_r_346_; 
v_res_345_ = lp_aesop_Aesop_isEmptyTrie(v_00_u03b1_343_, v_x_344_);
lean_dec_ref(v_x_344_);
v_r_346_ = lean_box(v_res_345_);
return v_r_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__0(lean_object* v_x1_347_, lean_object* v_x2_348_){
_start:
{
lean_object* v_snd_349_; uint8_t v___x_350_; 
v_snd_349_ = lean_ctor_get(v_x2_348_, 1);
v___x_350_ = lp_aesop_Aesop_isEmptyTrie___redArg(v_snd_349_);
if (v___x_350_ == 0)
{
lean_object* v___x_351_; 
v___x_351_ = lean_array_push(v_x1_347_, v_x2_348_);
return v___x_351_;
}
else
{
lean_dec_ref(v_x2_348_);
return v_x1_347_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4(lean_object* v_f_352_, lean_object* v_snd_353_, lean_object* v_v_354_, lean_object* v_toBind_355_, lean_object* v___f_356_, lean_object* v_fst_357_, lean_object* v_toPure_358_, uint8_t v_____do__lift_359_){
_start:
{
if (v_____do__lift_359_ == 0)
{
lean_object* v___x_360_; lean_object* v___x_361_; 
lean_dec(v_toPure_358_);
lean_dec_ref(v_fst_357_);
v___x_360_ = lean_apply_2(v_f_352_, v_snd_353_, v_v_354_);
v___x_361_ = lean_apply_4(v_toBind_355_, lean_box(0), lean_box(0), v___x_360_, v___f_356_);
return v___x_361_;
}
else
{
lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
lean_dec(v___f_356_);
lean_dec(v_toBind_355_);
lean_dec(v_f_352_);
v___x_362_ = lean_array_push(v_fst_357_, v_v_354_);
v___x_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v_snd_353_);
v___x_364_ = lean_apply_2(v_toPure_358_, lean_box(0), v___x_363_);
return v___x_364_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4___boxed(lean_object* v_f_365_, lean_object* v_snd_366_, lean_object* v_v_367_, lean_object* v_toBind_368_, lean_object* v___f_369_, lean_object* v_fst_370_, lean_object* v_toPure_371_, lean_object* v_____do__lift_372_){
_start:
{
uint8_t v_____do__lift_598__boxed_373_; lean_object* v_res_374_; 
v_____do__lift_598__boxed_373_ = lean_unbox(v_____do__lift_372_);
v_res_374_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4(v_f_365_, v_snd_366_, v_v_367_, v_toBind_368_, v___f_369_, v_fst_370_, v_toPure_371_, v_____do__lift_598__boxed_373_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__3(lean_object* v_fst_375_, lean_object* v_toPure_376_, lean_object* v_____do__lift_377_){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_378_, 0, v_fst_375_);
lean_ctor_set(v___x_378_, 1, v_____do__lift_377_);
v___x_379_ = lean_apply_2(v_toPure_376_, lean_box(0), v___x_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__5(lean_object* v_toPure_380_, lean_object* v_f_381_, lean_object* v_toBind_382_, lean_object* v_p_383_, lean_object* v_x_384_, lean_object* v_v_385_){
_start:
{
lean_object* v_fst_386_; lean_object* v_snd_387_; lean_object* v___f_388_; lean_object* v___f_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v_fst_386_ = lean_ctor_get(v_x_384_, 0);
lean_inc_n(v_fst_386_, 2);
v_snd_387_ = lean_ctor_get(v_x_384_, 1);
lean_inc(v_snd_387_);
lean_dec_ref(v_x_384_);
lean_inc(v_toPure_380_);
v___f_388_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__3), 3, 2);
lean_closure_set(v___f_388_, 0, v_fst_386_);
lean_closure_set(v___f_388_, 1, v_toPure_380_);
lean_inc(v_toBind_382_);
lean_inc(v_v_385_);
v___f_389_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__4___boxed), 8, 7);
lean_closure_set(v___f_389_, 0, v_f_381_);
lean_closure_set(v___f_389_, 1, v_snd_387_);
lean_closure_set(v___f_389_, 2, v_v_385_);
lean_closure_set(v___f_389_, 3, v_toBind_382_);
lean_closure_set(v___f_389_, 4, v___f_388_);
lean_closure_set(v___f_389_, 5, v_fst_386_);
lean_closure_set(v___f_389_, 6, v_toPure_380_);
v___x_390_ = lean_apply_1(v_p_383_, v_v_385_);
v___x_391_ = lean_apply_4(v_toBind_382_, lean_box(0), lean_box(0), v___x_390_, v___f_389_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1(lean_object* v_fst_392_, lean_object* v_toPure_393_, lean_object* v___x_394_, lean_object* v___f_395_, lean_object* v_____x_396_){
_start:
{
lean_object* v_fst_397_; lean_object* v_snd_398_; lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_420_; 
v_fst_397_ = lean_ctor_get(v_____x_396_, 0);
v_snd_398_ = lean_ctor_get(v_____x_396_, 1);
v_isSharedCheck_420_ = !lean_is_exclusive(v_____x_396_);
if (v_isSharedCheck_420_ == 0)
{
v___x_400_ = v_____x_396_;
v_isShared_401_ = v_isSharedCheck_420_;
goto v_resetjp_399_;
}
else
{
lean_inc(v_snd_398_);
lean_inc(v_fst_397_);
lean_dec(v_____x_396_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_420_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___y_403_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; uint8_t v___x_412_; 
v___x_409_ = lean_array_get_size(v_fst_397_);
v___x_410_ = lean_mk_empty_array_with_capacity(v___x_394_);
v___x_411_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_412_ = lean_nat_dec_lt(v___x_394_, v___x_409_);
if (v___x_412_ == 0)
{
lean_dec(v_fst_397_);
lean_dec_ref(v___f_395_);
v___y_403_ = v___x_410_;
goto v___jp_402_;
}
else
{
uint8_t v___x_413_; 
v___x_413_ = lean_nat_dec_le(v___x_409_, v___x_409_);
if (v___x_413_ == 0)
{
if (v___x_412_ == 0)
{
lean_dec(v_fst_397_);
lean_dec_ref(v___f_395_);
v___y_403_ = v___x_410_;
goto v___jp_402_;
}
else
{
size_t v___x_414_; size_t v___x_415_; lean_object* v___x_416_; 
v___x_414_ = ((size_t)0ULL);
v___x_415_ = lean_usize_of_nat(v___x_409_);
v___x_416_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_411_, v___f_395_, v_fst_397_, v___x_414_, v___x_415_, v___x_410_);
v___y_403_ = v___x_416_;
goto v___jp_402_;
}
}
else
{
size_t v___x_417_; size_t v___x_418_; lean_object* v___x_419_; 
v___x_417_ = ((size_t)0ULL);
v___x_418_ = lean_usize_of_nat(v___x_409_);
v___x_419_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_411_, v___f_395_, v_fst_397_, v___x_417_, v___x_418_, v___x_410_);
v___y_403_ = v___x_419_;
goto v___jp_402_;
}
}
v___jp_402_:
{
lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_404_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_404_, 0, v_fst_392_);
lean_ctor_set(v___x_404_, 1, v___y_403_);
if (v_isShared_401_ == 0)
{
lean_ctor_set(v___x_400_, 0, v___x_404_);
v___x_406_ = v___x_400_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v___x_404_);
lean_ctor_set(v_reuseFailAlloc_408_, 1, v_snd_398_);
v___x_406_ = v_reuseFailAlloc_408_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
lean_object* v___x_407_; 
v___x_407_ = lean_apply_2(v_toPure_393_, lean_box(0), v___x_406_);
return v___x_407_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1___boxed(lean_object* v_fst_421_, lean_object* v_toPure_422_, lean_object* v___x_423_, lean_object* v___f_424_, lean_object* v_____x_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1(v_fst_421_, v_toPure_422_, v___x_423_, v___f_424_, v_____x_425_);
lean_dec(v___x_423_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0(lean_object* v_i_427_, lean_object* v_fst_428_, lean_object* v_children_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_f_432_, lean_object* v_p_433_, lean_object* v_____x_434_){
_start:
{
lean_object* v_fst_435_; lean_object* v_snd_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_447_; 
v_fst_435_ = lean_ctor_get(v_____x_434_, 0);
v_snd_436_ = lean_ctor_get(v_____x_434_, 1);
v_isSharedCheck_447_ = !lean_is_exclusive(v_____x_434_);
if (v_isSharedCheck_447_ == 0)
{
v___x_438_ = v_____x_434_;
v_isShared_439_ = v_isSharedCheck_447_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_snd_436_);
lean_inc(v_fst_435_);
lean_dec(v_____x_434_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_447_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_443_; 
v___x_440_ = lean_unsigned_to_nat(1u);
v___x_441_ = lean_nat_add(v_i_427_, v___x_440_);
if (v_isShared_439_ == 0)
{
lean_ctor_set(v___x_438_, 1, v_fst_435_);
lean_ctor_set(v___x_438_, 0, v_fst_428_);
v___x_443_ = v___x_438_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_fst_428_);
lean_ctor_set(v_reuseFailAlloc_446_, 1, v_fst_435_);
v___x_443_ = v_reuseFailAlloc_446_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_444_ = lean_array_fset(v_children_429_, v_i_427_, v___x_443_);
v___x_445_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg(v_inst_430_, v_inst_431_, v_f_432_, v_p_433_, v_snd_436_, v___x_441_, v___x_444_);
return v___x_445_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0___boxed(lean_object* v_i_448_, lean_object* v_fst_449_, lean_object* v_children_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_f_453_, lean_object* v_p_454_, lean_object* v_____x_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0(v_i_448_, v_fst_449_, v_children_450_, v_inst_451_, v_inst_452_, v_f_453_, v_p_454_, v_____x_455_);
lean_dec(v_i_448_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg(lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_f_460_, lean_object* v_p_461_, lean_object* v_init_462_, lean_object* v_x_463_){
_start:
{
lean_object* v_toApplicative_464_; lean_object* v_toBind_465_; lean_object* v_toPure_466_; lean_object* v_vs_467_; lean_object* v_children_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_495_; 
v_toApplicative_464_ = lean_ctor_get(v_inst_458_, 0);
v_toBind_465_ = lean_ctor_get(v_inst_458_, 1);
lean_inc(v_toBind_465_);
v_toPure_466_ = lean_ctor_get(v_toApplicative_464_, 1);
v_vs_467_ = lean_ctor_get(v_x_463_, 0);
v_children_468_ = lean_ctor_get(v_x_463_, 1);
v_isSharedCheck_495_ = !lean_is_exclusive(v_x_463_);
if (v_isSharedCheck_495_ == 0)
{
v___x_470_ = v_x_463_;
v_isShared_471_ = v_isSharedCheck_495_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_children_468_);
lean_inc(v_vs_467_);
lean_dec(v_x_463_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_495_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v___f_472_; lean_object* v___x_473_; lean_object* v___f_474_; lean_object* v___x_475_; lean_object* v___x_477_; 
v___f_472_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___closed__0));
v___x_473_ = lean_unsigned_to_nat(0u);
lean_inc(v_toBind_465_);
lean_inc(v_p_461_);
lean_inc(v_f_460_);
lean_inc_ref(v_inst_458_);
lean_inc(v_toPure_466_);
v___f_474_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__2), 10, 9);
lean_closure_set(v___f_474_, 0, v_toPure_466_);
lean_closure_set(v___f_474_, 1, v___x_473_);
lean_closure_set(v___f_474_, 2, v___f_472_);
lean_closure_set(v___f_474_, 3, v_inst_458_);
lean_closure_set(v___f_474_, 4, v_inst_459_);
lean_closure_set(v___f_474_, 5, v_f_460_);
lean_closure_set(v___f_474_, 6, v_p_461_);
lean_closure_set(v___f_474_, 7, v_children_468_);
lean_closure_set(v___f_474_, 8, v_toBind_465_);
v___x_475_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1));
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 1, v_init_462_);
lean_ctor_set(v___x_470_, 0, v___x_475_);
v___x_477_ = v___x_470_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v___x_475_);
lean_ctor_set(v_reuseFailAlloc_494_, 1, v_init_462_);
v___x_477_ = v_reuseFailAlloc_494_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_478_; uint8_t v___x_479_; 
v___x_478_ = lean_array_get_size(v_vs_467_);
v___x_479_ = lean_nat_dec_lt(v___x_473_, v___x_478_);
if (v___x_479_ == 0)
{
lean_object* v___x_480_; lean_object* v___x_481_; 
lean_inc(v_toPure_466_);
lean_dec_ref(v_vs_467_);
lean_dec(v_p_461_);
lean_dec(v_f_460_);
lean_dec_ref(v_inst_458_);
v___x_480_ = lean_apply_2(v_toPure_466_, lean_box(0), v___x_477_);
v___x_481_ = lean_apply_4(v_toBind_465_, lean_box(0), lean_box(0), v___x_480_, v___f_474_);
return v___x_481_;
}
else
{
lean_object* v___f_482_; uint8_t v___x_483_; 
lean_inc(v_toBind_465_);
lean_inc(v_toPure_466_);
v___f_482_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__5), 6, 4);
lean_closure_set(v___f_482_, 0, v_toPure_466_);
lean_closure_set(v___f_482_, 1, v_f_460_);
lean_closure_set(v___f_482_, 2, v_toBind_465_);
lean_closure_set(v___f_482_, 3, v_p_461_);
v___x_483_ = lean_nat_dec_le(v___x_478_, v___x_478_);
if (v___x_483_ == 0)
{
if (v___x_479_ == 0)
{
lean_object* v___x_484_; lean_object* v___x_485_; 
lean_inc(v_toPure_466_);
lean_dec_ref(v___f_482_);
lean_dec_ref(v_vs_467_);
lean_dec_ref(v_inst_458_);
v___x_484_ = lean_apply_2(v_toPure_466_, lean_box(0), v___x_477_);
v___x_485_ = lean_apply_4(v_toBind_465_, lean_box(0), lean_box(0), v___x_484_, v___f_474_);
return v___x_485_;
}
else
{
size_t v___x_486_; size_t v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_486_ = ((size_t)0ULL);
v___x_487_ = lean_usize_of_nat(v___x_478_);
v___x_488_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_458_, v___f_482_, v_vs_467_, v___x_486_, v___x_487_, v___x_477_);
v___x_489_ = lean_apply_4(v_toBind_465_, lean_box(0), lean_box(0), v___x_488_, v___f_474_);
return v___x_489_;
}
}
else
{
size_t v___x_490_; size_t v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_490_ = ((size_t)0ULL);
v___x_491_ = lean_usize_of_nat(v___x_478_);
v___x_492_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_458_, v___f_482_, v_vs_467_, v___x_490_, v___x_491_, v___x_477_);
v___x_493_ = lean_apply_4(v_toBind_465_, lean_box(0), lean_box(0), v___x_492_, v___f_474_);
return v___x_493_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg(lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_f_498_, lean_object* v_p_499_, lean_object* v_acc_500_, lean_object* v_i_501_, lean_object* v_children_502_){
_start:
{
lean_object* v___x_503_; uint8_t v___x_504_; 
v___x_503_ = lean_array_get_size(v_children_502_);
v___x_504_ = lean_nat_dec_lt(v_i_501_, v___x_503_);
if (v___x_504_ == 0)
{
lean_object* v_toApplicative_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_514_; 
lean_dec(v_i_501_);
lean_dec(v_p_499_);
lean_dec(v_f_498_);
lean_dec(v_inst_497_);
v_toApplicative_505_ = lean_ctor_get(v_inst_496_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v_inst_496_);
if (v_isSharedCheck_514_ == 0)
{
lean_object* v_unused_515_; 
v_unused_515_ = lean_ctor_get(v_inst_496_, 1);
lean_dec(v_unused_515_);
v___x_507_ = v_inst_496_;
v_isShared_508_ = v_isSharedCheck_514_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_toApplicative_505_);
lean_dec(v_inst_496_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_514_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v_toPure_509_; lean_object* v___x_511_; 
v_toPure_509_ = lean_ctor_get(v_toApplicative_505_, 1);
lean_inc(v_toPure_509_);
lean_dec_ref(v_toApplicative_505_);
if (v_isShared_508_ == 0)
{
lean_ctor_set(v___x_507_, 1, v_acc_500_);
lean_ctor_set(v___x_507_, 0, v_children_502_);
v___x_511_ = v___x_507_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_children_502_);
lean_ctor_set(v_reuseFailAlloc_513_, 1, v_acc_500_);
v___x_511_ = v_reuseFailAlloc_513_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
lean_object* v___x_512_; 
v___x_512_ = lean_apply_2(v_toPure_509_, lean_box(0), v___x_511_);
return v___x_512_;
}
}
}
else
{
lean_object* v___x_516_; lean_object* v_fst_517_; lean_object* v_snd_518_; lean_object* v_toBind_519_; lean_object* v___f_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v___x_516_ = lean_array_fget_borrowed(v_children_502_, v_i_501_);
v_fst_517_ = lean_ctor_get(v___x_516_, 0);
lean_inc(v_fst_517_);
v_snd_518_ = lean_ctor_get(v___x_516_, 1);
lean_inc(v_snd_518_);
v_toBind_519_ = lean_ctor_get(v_inst_496_, 1);
lean_inc(v_toBind_519_);
lean_inc(v_p_499_);
lean_inc(v_f_498_);
lean_inc(v_inst_497_);
lean_inc_ref(v_inst_496_);
v___f_520_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_520_, 0, v_i_501_);
lean_closure_set(v___f_520_, 1, v_fst_517_);
lean_closure_set(v___f_520_, 2, v_children_502_);
lean_closure_set(v___f_520_, 3, v_inst_496_);
lean_closure_set(v___f_520_, 4, v_inst_497_);
lean_closure_set(v___f_520_, 5, v_f_498_);
lean_closure_set(v___f_520_, 6, v_p_499_);
v___x_521_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg(v_inst_496_, v_inst_497_, v_f_498_, v_p_499_, v_acc_500_, v_snd_518_);
v___x_522_ = lean_apply_4(v_toBind_519_, lean_box(0), lean_box(0), v___x_521_, v___f_520_);
return v___x_522_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__2(lean_object* v_toPure_523_, lean_object* v___x_524_, lean_object* v___f_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_f_528_, lean_object* v_p_529_, lean_object* v_children_530_, lean_object* v_toBind_531_, lean_object* v_____x_532_){
_start:
{
lean_object* v_fst_533_; lean_object* v_snd_534_; lean_object* v___f_535_; lean_object* v___x_536_; lean_object* v___x_537_; 
v_fst_533_ = lean_ctor_get(v_____x_532_, 0);
lean_inc(v_fst_533_);
v_snd_534_ = lean_ctor_get(v_____x_532_, 1);
lean_inc(v_snd_534_);
lean_dec_ref(v_____x_532_);
lean_inc(v___x_524_);
v___f_535_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_535_, 0, v_fst_533_);
lean_closure_set(v___f_535_, 1, v_toPure_523_);
lean_closure_set(v___f_535_, 2, v___x_524_);
lean_closure_set(v___f_535_, 3, v___f_525_);
v___x_536_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg(v_inst_526_, v_inst_527_, v_f_528_, v_p_529_, v_snd_534_, v___x_524_, v_children_530_);
v___x_537_ = lean_apply_4(v_toBind_531_, lean_box(0), lean_box(0), v___x_536_, v___f_535_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go(lean_object* v_m_538_, lean_object* v_00_u03c3_539_, lean_object* v_00_u03b1_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_f_543_, lean_object* v_p_544_, lean_object* v_acc_545_, lean_object* v_i_546_, lean_object* v_children_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM_go___redArg(v_inst_541_, v_inst_542_, v_f_543_, v_p_544_, v_acc_545_, v_i_546_, v_children_547_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM(lean_object* v_m_549_, lean_object* v_00_u03c3_550_, lean_object* v_00_u03b1_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_f_554_, lean_object* v_p_555_, lean_object* v_init_556_, lean_object* v_x_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg(v_inst_552_, v_inst_553_, v_f_554_, v_p_555_, v_init_556_, v_x_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__0(lean_object* v_toPure_559_, lean_object* v___x_560_, lean_object* v___x_561_, lean_object* v_fst_562_, lean_object* v_key_563_, lean_object* v_____x_564_){
_start:
{
lean_object* v_fst_565_; lean_object* v_snd_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_578_; 
v_fst_565_ = lean_ctor_get(v_____x_564_, 0);
v_snd_566_ = lean_ctor_get(v_____x_564_, 1);
v_isSharedCheck_578_ = !lean_is_exclusive(v_____x_564_);
if (v_isSharedCheck_578_ == 0)
{
v___x_568_ = v_____x_564_;
v_isShared_569_ = v_isSharedCheck_578_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_snd_566_);
lean_inc(v_fst_565_);
lean_dec(v_____x_564_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_578_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___y_571_; uint8_t v___x_576_; 
v___x_576_ = lp_aesop_Aesop_isEmptyTrie___redArg(v_fst_565_);
if (v___x_576_ == 0)
{
lean_object* v___x_577_; 
v___x_577_ = l_Lean_PersistentHashMap_insert___redArg(v___x_560_, v___x_561_, v_fst_562_, v_key_563_, v_fst_565_);
v___y_571_ = v___x_577_;
goto v___jp_570_;
}
else
{
lean_dec(v_fst_565_);
lean_dec(v_key_563_);
lean_dec_ref(v___x_561_);
lean_dec_ref(v___x_560_);
v___y_571_ = v_fst_562_;
goto v___jp_570_;
}
v___jp_570_:
{
lean_object* v___x_573_; 
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 0, v___y_571_);
v___x_573_ = v___x_568_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v___y_571_);
lean_ctor_set(v_reuseFailAlloc_575_, 1, v_snd_566_);
v___x_573_ = v_reuseFailAlloc_575_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
lean_object* v___x_574_; 
v___x_574_ = lean_apply_2(v_toPure_559_, lean_box(0), v___x_573_);
return v___x_574_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__1(lean_object* v_toPure_579_, lean_object* v___x_580_, lean_object* v___x_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_f_584_, lean_object* v_p_585_, lean_object* v_toBind_586_, lean_object* v_x_587_, lean_object* v_key_588_, lean_object* v_t_589_){
_start:
{
lean_object* v_fst_590_; lean_object* v_snd_591_; lean_object* v___f_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v_fst_590_ = lean_ctor_get(v_x_587_, 0);
lean_inc(v_fst_590_);
v_snd_591_ = lean_ctor_get(v_x_587_, 1);
lean_inc(v_snd_591_);
lean_dec_ref(v_x_587_);
v___f_592_ = lean_alloc_closure((void*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__0), 6, 5);
lean_closure_set(v___f_592_, 0, v_toPure_579_);
lean_closure_set(v___f_592_, 1, v___x_580_);
lean_closure_set(v___f_592_, 2, v___x_581_);
lean_closure_set(v___f_592_, 3, v_fst_590_);
lean_closure_set(v___f_592_, 4, v_key_588_);
v___x_593_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_filterTrieM___redArg(v_inst_582_, v_inst_583_, v_f_584_, v_p_585_, v_snd_591_, v_t_589_);
v___x_594_ = lean_apply_4(v_toBind_586_, lean_box(0), lean_box(0), v___x_593_, v___f_592_);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__2(lean_object* v_toPure_595_, lean_object* v_____x_596_){
_start:
{
lean_object* v_fst_597_; lean_object* v_snd_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_606_; 
v_fst_597_ = lean_ctor_get(v_____x_596_, 0);
v_snd_598_ = lean_ctor_get(v_____x_596_, 1);
v_isSharedCheck_606_ = !lean_is_exclusive(v_____x_596_);
if (v_isSharedCheck_606_ == 0)
{
v___x_600_ = v_____x_596_;
v_isShared_601_ = v_isSharedCheck_606_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_snd_598_);
lean_inc(v_fst_597_);
lean_dec(v_____x_596_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_606_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_fst_597_);
lean_ctor_set(v_reuseFailAlloc_605_, 1, v_snd_598_);
v___x_603_ = v_reuseFailAlloc_605_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
lean_object* v___x_604_; 
v___x_604_ = lean_apply_2(v_toPure_595_, lean_box(0), v___x_603_);
return v___x_604_;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2(void){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = ((lean_object*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__1));
v___x_610_ = ((lean_object*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__0));
v___x_611_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_610_, v___x_609_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM___redArg(lean_object* v_inst_612_, lean_object* v_inst_613_, lean_object* v_p_614_, lean_object* v_f_615_, lean_object* v_init_616_, lean_object* v_t_617_){
_start:
{
lean_object* v_toApplicative_618_; lean_object* v_toBind_619_; lean_object* v_toPure_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___f_623_; lean_object* v___f_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v_toApplicative_618_ = lean_ctor_get(v_inst_612_, 0);
v_toBind_619_ = lean_ctor_get(v_inst_612_, 1);
lean_inc_n(v_toBind_619_, 2);
v_toPure_620_ = lean_ctor_get(v_toApplicative_618_, 1);
v___x_621_ = ((lean_object*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__0));
v___x_622_ = ((lean_object*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__1));
lean_inc_ref(v_inst_612_);
lean_inc_n(v_toPure_620_, 2);
v___f_623_ = lean_alloc_closure((void*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__1), 11, 8);
lean_closure_set(v___f_623_, 0, v_toPure_620_);
lean_closure_set(v___f_623_, 1, v___x_621_);
lean_closure_set(v___f_623_, 2, v___x_622_);
lean_closure_set(v___f_623_, 3, v_inst_612_);
lean_closure_set(v___f_623_, 4, v_inst_613_);
lean_closure_set(v___f_623_, 5, v_f_615_);
lean_closure_set(v___f_623_, 6, v_p_614_);
lean_closure_set(v___f_623_, 7, v_toBind_619_);
v___f_624_ = lean_alloc_closure((void*)(lp_aesop_Aesop_filterDiscrTreeM___redArg___lam__2), 2, 1);
lean_closure_set(v___f_624_, 0, v_toPure_620_);
v___x_625_ = lean_obj_once(&lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2, &lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2_once, _init_lp_aesop_Aesop_filterDiscrTreeM___redArg___closed__2);
v___x_626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_625_);
lean_ctor_set(v___x_626_, 1, v_init_616_);
v___x_627_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_612_, v___f_623_, v_t_617_, v___x_626_);
v___x_628_ = lean_apply_4(v_toBind_619_, lean_box(0), lean_box(0), v___x_627_, v___f_624_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTreeM(lean_object* v_m_629_, lean_object* v_00_u03c3_630_, lean_object* v_00_u03b1_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_p_634_, lean_object* v_f_635_, lean_object* v_init_636_, lean_object* v_t_637_){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lp_aesop_Aesop_filterDiscrTreeM___redArg(v_inst_632_, v_inst_633_, v_p_634_, v_f_635_, v_init_636_, v_t_637_);
return v___x_638_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_filterDiscrTree___redArg___lam__0(lean_object* v_p_639_, lean_object* v_a_640_){
_start:
{
lean_object* v___x_641_; uint8_t v___x_642_; 
v___x_641_ = lean_apply_1(v_p_639_, v_a_640_);
v___x_642_ = lean_unbox(v___x_641_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg___lam__0___boxed(lean_object* v_p_643_, lean_object* v_a_644_){
_start:
{
uint8_t v_res_645_; lean_object* v_r_646_; 
v_res_645_ = lp_aesop_Aesop_filterDiscrTree___redArg___lam__0(v_p_643_, v_a_644_);
v_r_646_ = lean_box(v_res_645_);
return v_r_646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg___lam__1(lean_object* v_f_647_, lean_object* v_s_648_, lean_object* v_a_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lean_apply_2(v_f_647_, v_s_648_, v_a_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree___redArg(lean_object* v_inst_651_, lean_object* v_p_652_, lean_object* v_f_653_, lean_object* v_init_654_, lean_object* v_t_655_){
_start:
{
lean_object* v___f_656_; lean_object* v___f_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
v___f_656_ = lean_alloc_closure((void*)(lp_aesop_Aesop_filterDiscrTree___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_656_, 0, v_p_652_);
v___f_657_ = lean_alloc_closure((void*)(lp_aesop_Aesop_filterDiscrTree___redArg___lam__1), 3, 1);
lean_closure_set(v___f_657_, 0, v_f_653_);
v___x_658_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toList___redArg___closed__10));
v___x_659_ = lp_aesop_Aesop_filterDiscrTreeM___redArg(v___x_658_, v_inst_651_, v___f_656_, v___f_657_, v_init_654_, v_t_655_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_filterDiscrTree(lean_object* v_00_u03c3_660_, lean_object* v_00_u03b1_661_, lean_object* v_inst_662_, lean_object* v_p_663_, lean_object* v_f_664_, lean_object* v_init_665_, lean_object* v_t_666_){
_start:
{
lean_object* v___x_667_; 
v___x_667_ = lp_aesop_Aesop_filterDiscrTree___redArg(v_inst_662_, v_p_663_, v_f_664_, v_init_665_, v_t_666_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6___redArg(lean_object* v_x_668_, lean_object* v_x_669_, lean_object* v_x_670_, lean_object* v_x_671_){
_start:
{
lean_object* v_ks_672_; lean_object* v_vs_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_697_; 
v_ks_672_ = lean_ctor_get(v_x_668_, 0);
v_vs_673_ = lean_ctor_get(v_x_668_, 1);
v_isSharedCheck_697_ = !lean_is_exclusive(v_x_668_);
if (v_isSharedCheck_697_ == 0)
{
v___x_675_ = v_x_668_;
v_isShared_676_ = v_isSharedCheck_697_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_vs_673_);
lean_inc(v_ks_672_);
lean_dec(v_x_668_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_697_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_677_; uint8_t v___x_678_; 
v___x_677_ = lean_array_get_size(v_ks_672_);
v___x_678_ = lean_nat_dec_lt(v_x_669_, v___x_677_);
if (v___x_678_ == 0)
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_682_; 
lean_dec(v_x_669_);
v___x_679_ = lean_array_push(v_ks_672_, v_x_670_);
v___x_680_ = lean_array_push(v_vs_673_, v_x_671_);
if (v_isShared_676_ == 0)
{
lean_ctor_set(v___x_675_, 1, v___x_680_);
lean_ctor_set(v___x_675_, 0, v___x_679_);
v___x_682_ = v___x_675_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v___x_679_);
lean_ctor_set(v_reuseFailAlloc_683_, 1, v___x_680_);
v___x_682_ = v_reuseFailAlloc_683_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
return v___x_682_;
}
}
else
{
lean_object* v_k_x27_684_; uint8_t v___x_685_; 
v_k_x27_684_ = lean_array_fget_borrowed(v_ks_672_, v_x_669_);
v___x_685_ = lean_name_eq(v_x_670_, v_k_x27_684_);
if (v___x_685_ == 0)
{
lean_object* v___x_687_; 
if (v_isShared_676_ == 0)
{
v___x_687_ = v___x_675_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_ks_672_);
lean_ctor_set(v_reuseFailAlloc_691_, 1, v_vs_673_);
v___x_687_ = v_reuseFailAlloc_691_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
lean_object* v___x_688_; lean_object* v___x_689_; 
v___x_688_ = lean_unsigned_to_nat(1u);
v___x_689_ = lean_nat_add(v_x_669_, v___x_688_);
lean_dec(v_x_669_);
v_x_668_ = v___x_687_;
v_x_669_ = v___x_689_;
goto _start;
}
}
else
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_695_; 
v___x_692_ = lean_array_fset(v_ks_672_, v_x_669_, v_x_670_);
v___x_693_ = lean_array_fset(v_vs_673_, v_x_669_, v_x_671_);
lean_dec(v_x_669_);
if (v_isShared_676_ == 0)
{
lean_ctor_set(v___x_675_, 1, v___x_693_);
lean_ctor_set(v___x_675_, 0, v___x_692_);
v___x_695_ = v___x_675_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v___x_692_);
lean_ctor_set(v_reuseFailAlloc_696_, 1, v___x_693_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4___redArg(lean_object* v_n_698_, lean_object* v_k_699_, lean_object* v_v_700_){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_701_ = lean_unsigned_to_nat(0u);
v___x_702_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6___redArg(v_n_698_, v___x_701_, v_k_699_, v_v_700_);
return v___x_702_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(lean_object* v_x_704_, size_t v_x_705_, size_t v_x_706_, lean_object* v_x_707_, lean_object* v_x_708_){
_start:
{
if (lean_obj_tag(v_x_704_) == 0)
{
lean_object* v_es_709_; size_t v___x_710_; size_t v___x_711_; lean_object* v_j_712_; lean_object* v___x_713_; uint8_t v___x_714_; 
v_es_709_ = lean_ctor_get(v_x_704_, 0);
v___x_710_ = ((size_t)31ULL);
v___x_711_ = lean_usize_land(v_x_705_, v___x_710_);
v_j_712_ = lean_usize_to_nat(v___x_711_);
v___x_713_ = lean_array_get_size(v_es_709_);
v___x_714_ = lean_nat_dec_lt(v_j_712_, v___x_713_);
if (v___x_714_ == 0)
{
lean_dec(v_j_712_);
lean_dec(v_x_708_);
lean_dec(v_x_707_);
return v_x_704_;
}
else
{
lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_753_; 
lean_inc_ref(v_es_709_);
v_isSharedCheck_753_ = !lean_is_exclusive(v_x_704_);
if (v_isSharedCheck_753_ == 0)
{
lean_object* v_unused_754_; 
v_unused_754_ = lean_ctor_get(v_x_704_, 0);
lean_dec(v_unused_754_);
v___x_716_ = v_x_704_;
v_isShared_717_ = v_isSharedCheck_753_;
goto v_resetjp_715_;
}
else
{
lean_dec(v_x_704_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_753_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v_v_718_; lean_object* v___x_719_; lean_object* v_xs_x27_720_; lean_object* v___y_722_; 
v_v_718_ = lean_array_fget(v_es_709_, v_j_712_);
v___x_719_ = lean_box(0);
v_xs_x27_720_ = lean_array_fset(v_es_709_, v_j_712_, v___x_719_);
switch(lean_obj_tag(v_v_718_))
{
case 0:
{
lean_object* v_key_727_; lean_object* v_val_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_738_; 
v_key_727_ = lean_ctor_get(v_v_718_, 0);
v_val_728_ = lean_ctor_get(v_v_718_, 1);
v_isSharedCheck_738_ = !lean_is_exclusive(v_v_718_);
if (v_isSharedCheck_738_ == 0)
{
v___x_730_ = v_v_718_;
v_isShared_731_ = v_isSharedCheck_738_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_val_728_);
lean_inc(v_key_727_);
lean_dec(v_v_718_);
v___x_730_ = lean_box(0);
v_isShared_731_ = v_isSharedCheck_738_;
goto v_resetjp_729_;
}
v_resetjp_729_:
{
uint8_t v___x_732_; 
v___x_732_ = lean_name_eq(v_x_707_, v_key_727_);
if (v___x_732_ == 0)
{
lean_object* v___x_733_; lean_object* v___x_734_; 
lean_del_object(v___x_730_);
v___x_733_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_727_, v_val_728_, v_x_707_, v_x_708_);
v___x_734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
v___y_722_ = v___x_734_;
goto v___jp_721_;
}
else
{
lean_object* v___x_736_; 
lean_dec(v_val_728_);
lean_dec(v_key_727_);
if (v_isShared_731_ == 0)
{
lean_ctor_set(v___x_730_, 1, v_x_708_);
lean_ctor_set(v___x_730_, 0, v_x_707_);
v___x_736_ = v___x_730_;
goto v_reusejp_735_;
}
else
{
lean_object* v_reuseFailAlloc_737_; 
v_reuseFailAlloc_737_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_737_, 0, v_x_707_);
lean_ctor_set(v_reuseFailAlloc_737_, 1, v_x_708_);
v___x_736_ = v_reuseFailAlloc_737_;
goto v_reusejp_735_;
}
v_reusejp_735_:
{
v___y_722_ = v___x_736_;
goto v___jp_721_;
}
}
}
}
case 1:
{
lean_object* v_node_739_; lean_object* v___x_741_; uint8_t v_isShared_742_; uint8_t v_isSharedCheck_751_; 
v_node_739_ = lean_ctor_get(v_v_718_, 0);
v_isSharedCheck_751_ = !lean_is_exclusive(v_v_718_);
if (v_isSharedCheck_751_ == 0)
{
v___x_741_ = v_v_718_;
v_isShared_742_ = v_isSharedCheck_751_;
goto v_resetjp_740_;
}
else
{
lean_inc(v_node_739_);
lean_dec(v_v_718_);
v___x_741_ = lean_box(0);
v_isShared_742_ = v_isSharedCheck_751_;
goto v_resetjp_740_;
}
v_resetjp_740_:
{
size_t v___x_743_; size_t v___x_744_; size_t v___x_745_; size_t v___x_746_; lean_object* v___x_747_; lean_object* v___x_749_; 
v___x_743_ = ((size_t)5ULL);
v___x_744_ = lean_usize_shift_right(v_x_705_, v___x_743_);
v___x_745_ = ((size_t)1ULL);
v___x_746_ = lean_usize_add(v_x_706_, v___x_745_);
v___x_747_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(v_node_739_, v___x_744_, v___x_746_, v_x_707_, v_x_708_);
if (v_isShared_742_ == 0)
{
lean_ctor_set(v___x_741_, 0, v___x_747_);
v___x_749_ = v___x_741_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v___x_747_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
v___y_722_ = v___x_749_;
goto v___jp_721_;
}
}
}
default: 
{
lean_object* v___x_752_; 
v___x_752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_752_, 0, v_x_707_);
lean_ctor_set(v___x_752_, 1, v_x_708_);
v___y_722_ = v___x_752_;
goto v___jp_721_;
}
}
v___jp_721_:
{
lean_object* v___x_723_; lean_object* v___x_725_; 
v___x_723_ = lean_array_fset(v_xs_x27_720_, v_j_712_, v___y_722_);
lean_dec(v_j_712_);
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 0, v___x_723_);
v___x_725_ = v___x_716_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v___x_723_);
v___x_725_ = v_reuseFailAlloc_726_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
return v___x_725_;
}
}
}
}
}
else
{
lean_object* v_ks_755_; lean_object* v_vs_756_; lean_object* v___x_758_; uint8_t v_isShared_759_; uint8_t v_isSharedCheck_776_; 
v_ks_755_ = lean_ctor_get(v_x_704_, 0);
v_vs_756_ = lean_ctor_get(v_x_704_, 1);
v_isSharedCheck_776_ = !lean_is_exclusive(v_x_704_);
if (v_isSharedCheck_776_ == 0)
{
v___x_758_ = v_x_704_;
v_isShared_759_ = v_isSharedCheck_776_;
goto v_resetjp_757_;
}
else
{
lean_inc(v_vs_756_);
lean_inc(v_ks_755_);
lean_dec(v_x_704_);
v___x_758_ = lean_box(0);
v_isShared_759_ = v_isSharedCheck_776_;
goto v_resetjp_757_;
}
v_resetjp_757_:
{
lean_object* v___x_761_; 
if (v_isShared_759_ == 0)
{
v___x_761_ = v___x_758_;
goto v_reusejp_760_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_ks_755_);
lean_ctor_set(v_reuseFailAlloc_775_, 1, v_vs_756_);
v___x_761_ = v_reuseFailAlloc_775_;
goto v_reusejp_760_;
}
v_reusejp_760_:
{
lean_object* v_newNode_762_; uint8_t v___y_764_; size_t v___x_770_; uint8_t v___x_771_; 
v_newNode_762_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4___redArg(v___x_761_, v_x_707_, v_x_708_);
v___x_770_ = ((size_t)7ULL);
v___x_771_ = lean_usize_dec_le(v___x_770_, v_x_706_);
if (v___x_771_ == 0)
{
lean_object* v___x_772_; lean_object* v___x_773_; uint8_t v___x_774_; 
v___x_772_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_762_);
v___x_773_ = lean_unsigned_to_nat(4u);
v___x_774_ = lean_nat_dec_lt(v___x_772_, v___x_773_);
lean_dec(v___x_772_);
v___y_764_ = v___x_774_;
goto v___jp_763_;
}
else
{
v___y_764_ = v___x_771_;
goto v___jp_763_;
}
v___jp_763_:
{
if (v___y_764_ == 0)
{
lean_object* v_ks_765_; lean_object* v_vs_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; 
v_ks_765_ = lean_ctor_get(v_newNode_762_, 0);
lean_inc_ref(v_ks_765_);
v_vs_766_ = lean_ctor_get(v_newNode_762_, 1);
lean_inc_ref(v_vs_766_);
lean_dec_ref(v_newNode_762_);
v___x_767_ = lean_unsigned_to_nat(0u);
v___x_768_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___closed__0);
v___x_769_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg(v_x_706_, v_ks_765_, v_vs_766_, v___x_767_, v___x_768_);
lean_dec_ref(v_vs_766_);
lean_dec_ref(v_ks_765_);
return v___x_769_;
}
else
{
return v_newNode_762_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg(size_t v_depth_777_, lean_object* v_keys_778_, lean_object* v_vals_779_, lean_object* v_i_780_, lean_object* v_entries_781_){
_start:
{
lean_object* v___x_782_; uint8_t v___x_783_; 
v___x_782_ = lean_array_get_size(v_keys_778_);
v___x_783_ = lean_nat_dec_lt(v_i_780_, v___x_782_);
if (v___x_783_ == 0)
{
lean_dec(v_i_780_);
return v_entries_781_;
}
else
{
lean_object* v_k_784_; lean_object* v_v_785_; uint64_t v___y_787_; 
v_k_784_ = lean_array_fget_borrowed(v_keys_778_, v_i_780_);
v_v_785_ = lean_array_fget_borrowed(v_vals_779_, v_i_780_);
if (lean_obj_tag(v_k_784_) == 0)
{
uint64_t v___x_798_; 
v___x_798_ = 1723ULL;
v___y_787_ = v___x_798_;
goto v___jp_786_;
}
else
{
uint64_t v_hash_799_; 
v_hash_799_ = lean_ctor_get_uint64(v_k_784_, sizeof(void*)*2);
v___y_787_ = v_hash_799_;
goto v___jp_786_;
}
v___jp_786_:
{
size_t v_h_788_; size_t v___x_789_; lean_object* v___x_790_; size_t v___x_791_; size_t v___x_792_; size_t v___x_793_; size_t v_h_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v_h_788_ = lean_uint64_to_usize(v___y_787_);
v___x_789_ = ((size_t)5ULL);
v___x_790_ = lean_unsigned_to_nat(1u);
v___x_791_ = ((size_t)1ULL);
v___x_792_ = lean_usize_sub(v_depth_777_, v___x_791_);
v___x_793_ = lean_usize_mul(v___x_789_, v___x_792_);
v_h_794_ = lean_usize_shift_right(v_h_788_, v___x_793_);
v___x_795_ = lean_nat_add(v_i_780_, v___x_790_);
lean_dec(v_i_780_);
lean_inc(v_v_785_);
lean_inc(v_k_784_);
v___x_796_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(v_entries_781_, v_h_794_, v_depth_777_, v_k_784_, v_v_785_);
v_i_780_ = v___x_795_;
v_entries_781_ = v___x_796_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_depth_800_, lean_object* v_keys_801_, lean_object* v_vals_802_, lean_object* v_i_803_, lean_object* v_entries_804_){
_start:
{
size_t v_depth_boxed_805_; lean_object* v_res_806_; 
v_depth_boxed_805_ = lean_unbox_usize(v_depth_800_);
lean_dec(v_depth_800_);
v_res_806_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg(v_depth_boxed_805_, v_keys_801_, v_vals_802_, v_i_803_, v_entries_804_);
lean_dec_ref(v_vals_802_);
lean_dec_ref(v_keys_801_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg___boxed(lean_object* v_x_807_, lean_object* v_x_808_, lean_object* v_x_809_, lean_object* v_x_810_, lean_object* v_x_811_){
_start:
{
size_t v_x_662__boxed_812_; size_t v_x_663__boxed_813_; lean_object* v_res_814_; 
v_x_662__boxed_812_ = lean_unbox_usize(v_x_808_);
lean_dec(v_x_808_);
v_x_663__boxed_813_ = lean_unbox_usize(v_x_809_);
lean_dec(v_x_809_);
v_res_814_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(v_x_807_, v_x_662__boxed_812_, v_x_663__boxed_813_, v_x_810_, v_x_811_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1___redArg(lean_object* v_x_815_, lean_object* v_x_816_, lean_object* v_x_817_){
_start:
{
uint64_t v___y_819_; 
if (lean_obj_tag(v_x_816_) == 0)
{
uint64_t v___x_823_; 
v___x_823_ = 1723ULL;
v___y_819_ = v___x_823_;
goto v___jp_818_;
}
else
{
uint64_t v_hash_824_; 
v_hash_824_ = lean_ctor_get_uint64(v_x_816_, sizeof(void*)*2);
v___y_819_ = v_hash_824_;
goto v___jp_818_;
}
v___jp_818_:
{
size_t v___x_820_; size_t v___x_821_; lean_object* v___x_822_; 
v___x_820_ = lean_uint64_to_usize(v___y_819_);
v___x_821_ = ((size_t)1ULL);
v___x_822_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(v_x_815_, v___x_820_, v___x_821_, v_x_816_, v_x_817_);
return v___x_822_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3(lean_object* v_xs_825_, lean_object* v_v_826_, lean_object* v_i_827_){
_start:
{
uint8_t v___y_833_; lean_object* v___x_835_; uint8_t v___x_836_; 
v___x_835_ = lean_array_get_size(v_xs_825_);
v___x_836_ = lean_nat_dec_lt(v_i_827_, v___x_835_);
if (v___x_836_ == 0)
{
lean_object* v___x_837_; 
lean_dec(v_i_827_);
v___x_837_ = lean_box(0);
return v___x_837_;
}
else
{
lean_object* v___x_838_; 
v___x_838_ = lean_array_fget_borrowed(v_xs_825_, v_i_827_);
if (lean_obj_tag(v___x_838_) == 0)
{
if (lean_obj_tag(v_v_826_) == 0)
{
lean_object* v_declName_839_; uint8_t v_inv_840_; lean_object* v_declName_841_; uint8_t v_inv_842_; uint8_t v___x_843_; 
v_declName_839_ = lean_ctor_get(v___x_838_, 0);
v_inv_840_ = lean_ctor_get_uint8(v___x_838_, sizeof(void*)*1 + 1);
v_declName_841_ = lean_ctor_get(v_v_826_, 0);
v_inv_842_ = lean_ctor_get_uint8(v_v_826_, sizeof(void*)*1 + 1);
v___x_843_ = lean_name_eq(v_declName_839_, v_declName_841_);
if (v___x_843_ == 0)
{
v___y_833_ = v___x_843_;
goto v___jp_832_;
}
else
{
if (v_inv_840_ == 0)
{
if (v_inv_842_ == 0)
{
v___y_833_ = v___x_843_;
goto v___jp_832_;
}
else
{
goto v___jp_828_;
}
}
else
{
v___y_833_ = v_inv_842_;
goto v___jp_832_;
}
}
}
else
{
goto v___jp_828_;
}
}
else
{
if (lean_obj_tag(v_v_826_) == 0)
{
goto v___jp_828_;
}
else
{
lean_object* v___x_844_; lean_object* v___x_845_; uint8_t v___x_846_; 
v___x_844_ = l_Lean_Meta_Origin_key(v___x_838_);
v___x_845_ = l_Lean_Meta_Origin_key(v_v_826_);
v___x_846_ = lean_name_eq(v___x_844_, v___x_845_);
lean_dec(v___x_845_);
lean_dec(v___x_844_);
v___y_833_ = v___x_846_;
goto v___jp_832_;
}
}
}
v___jp_828_:
{
lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_829_ = lean_unsigned_to_nat(1u);
v___x_830_ = lean_nat_add(v_i_827_, v___x_829_);
lean_dec(v_i_827_);
v_i_827_ = v___x_830_;
goto _start;
}
v___jp_832_:
{
if (v___y_833_ == 0)
{
goto v___jp_828_;
}
else
{
lean_object* v___x_834_; 
v___x_834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_834_, 0, v_i_827_);
return v___x_834_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_xs_847_, lean_object* v_v_848_, lean_object* v_i_849_){
_start:
{
lean_object* v_res_850_; 
v_res_850_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3(v_xs_847_, v_v_848_, v_i_849_);
lean_dec_ref(v_v_848_);
lean_dec_ref(v_xs_847_);
return v_res_850_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1(lean_object* v_xs_851_, lean_object* v_v_852_){
_start:
{
lean_object* v___x_853_; lean_object* v___x_854_; 
v___x_853_ = lean_unsigned_to_nat(0u);
v___x_854_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1_spec__3(v_xs_851_, v_v_852_, v___x_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1___boxed(lean_object* v_xs_855_, lean_object* v_v_856_){
_start:
{
lean_object* v_res_857_; 
v_res_857_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1(v_xs_855_, v_v_856_);
lean_dec_ref(v_v_856_);
lean_dec_ref(v_xs_855_);
return v_res_857_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(lean_object* v_x_858_, size_t v_x_859_, lean_object* v_x_860_){
_start:
{
if (lean_obj_tag(v_x_858_) == 0)
{
lean_object* v_es_861_; lean_object* v___x_862_; size_t v___x_863_; size_t v___x_864_; lean_object* v_j_865_; uint8_t v___y_867_; lean_object* v_entry_877_; 
v_es_861_ = lean_ctor_get(v_x_858_, 0);
v___x_862_ = lean_box(2);
v___x_863_ = ((size_t)31ULL);
v___x_864_ = lean_usize_land(v_x_859_, v___x_863_);
v_j_865_ = lean_usize_to_nat(v___x_864_);
v_entry_877_ = lean_array_get(v___x_862_, v_es_861_, v_j_865_);
switch(lean_obj_tag(v_entry_877_))
{
case 0:
{
if (lean_obj_tag(v_x_860_) == 0)
{
lean_object* v_key_878_; 
v_key_878_ = lean_ctor_get(v_entry_877_, 0);
lean_inc(v_key_878_);
lean_dec_ref_known(v_entry_877_, 2);
if (lean_obj_tag(v_key_878_) == 0)
{
lean_object* v_declName_879_; uint8_t v_inv_880_; lean_object* v_declName_881_; uint8_t v_inv_882_; uint8_t v___x_883_; 
v_declName_879_ = lean_ctor_get(v_x_860_, 0);
v_inv_880_ = lean_ctor_get_uint8(v_x_860_, sizeof(void*)*1 + 1);
v_declName_881_ = lean_ctor_get(v_key_878_, 0);
lean_inc(v_declName_881_);
v_inv_882_ = lean_ctor_get_uint8(v_key_878_, sizeof(void*)*1 + 1);
lean_dec_ref_known(v_key_878_, 1);
v___x_883_ = lean_name_eq(v_declName_879_, v_declName_881_);
lean_dec(v_declName_881_);
if (v___x_883_ == 0)
{
v___y_867_ = v___x_883_;
goto v___jp_866_;
}
else
{
if (v_inv_880_ == 0)
{
if (v_inv_882_ == 0)
{
v___y_867_ = v___x_883_;
goto v___jp_866_;
}
else
{
lean_dec(v_j_865_);
return v_x_858_;
}
}
else
{
v___y_867_ = v_inv_882_;
goto v___jp_866_;
}
}
}
else
{
lean_dec(v_key_878_);
lean_dec(v_j_865_);
return v_x_858_;
}
}
else
{
lean_object* v_key_884_; 
v_key_884_ = lean_ctor_get(v_entry_877_, 0);
lean_inc(v_key_884_);
lean_dec_ref_known(v_entry_877_, 2);
if (lean_obj_tag(v_key_884_) == 0)
{
lean_dec_ref_known(v_key_884_, 1);
lean_dec(v_j_865_);
return v_x_858_;
}
else
{
lean_object* v___x_885_; lean_object* v___x_886_; uint8_t v___x_887_; 
v___x_885_ = l_Lean_Meta_Origin_key(v_x_860_);
v___x_886_ = l_Lean_Meta_Origin_key(v_key_884_);
lean_dec(v_key_884_);
v___x_887_ = lean_name_eq(v___x_885_, v___x_886_);
lean_dec(v___x_886_);
lean_dec(v___x_885_);
v___y_867_ = v___x_887_;
goto v___jp_866_;
}
}
}
case 1:
{
lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_922_; 
lean_inc_ref(v_es_861_);
v_isSharedCheck_922_ = !lean_is_exclusive(v_x_858_);
if (v_isSharedCheck_922_ == 0)
{
lean_object* v_unused_923_; 
v_unused_923_ = lean_ctor_get(v_x_858_, 0);
lean_dec(v_unused_923_);
v___x_889_ = v_x_858_;
v_isShared_890_ = v_isSharedCheck_922_;
goto v_resetjp_888_;
}
else
{
lean_dec(v_x_858_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_922_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v_node_891_; lean_object* v___x_893_; uint8_t v_isShared_894_; uint8_t v_isSharedCheck_921_; 
v_node_891_ = lean_ctor_get(v_entry_877_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v_entry_877_);
if (v_isSharedCheck_921_ == 0)
{
v___x_893_ = v_entry_877_;
v_isShared_894_ = v_isSharedCheck_921_;
goto v_resetjp_892_;
}
else
{
lean_inc(v_node_891_);
lean_dec(v_entry_877_);
v___x_893_ = lean_box(0);
v_isShared_894_ = v_isSharedCheck_921_;
goto v_resetjp_892_;
}
v_resetjp_892_:
{
size_t v___x_895_; lean_object* v_entries_896_; size_t v___x_897_; lean_object* v_newNode_898_; lean_object* v___x_899_; 
v___x_895_ = ((size_t)5ULL);
v_entries_896_ = lean_array_set(v_es_861_, v_j_865_, v___x_862_);
v___x_897_ = lean_usize_shift_right(v_x_859_, v___x_895_);
v_newNode_898_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(v_node_891_, v___x_897_, v_x_860_);
lean_inc_ref(v_newNode_898_);
v___x_899_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_898_);
if (lean_obj_tag(v___x_899_) == 0)
{
lean_object* v___x_901_; 
if (v_isShared_894_ == 0)
{
lean_ctor_set(v___x_893_, 0, v_newNode_898_);
v___x_901_ = v___x_893_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_newNode_898_);
v___x_901_ = v_reuseFailAlloc_906_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
lean_object* v___x_902_; lean_object* v___x_904_; 
v___x_902_ = lean_array_set(v_entries_896_, v_j_865_, v___x_901_);
lean_dec(v_j_865_);
if (v_isShared_890_ == 0)
{
lean_ctor_set(v___x_889_, 0, v___x_902_);
v___x_904_ = v___x_889_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v___x_902_);
v___x_904_ = v_reuseFailAlloc_905_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
return v___x_904_;
}
}
}
else
{
lean_object* v_val_907_; lean_object* v_fst_908_; lean_object* v_snd_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_920_; 
lean_dec_ref(v_newNode_898_);
lean_del_object(v___x_893_);
v_val_907_ = lean_ctor_get(v___x_899_, 0);
lean_inc(v_val_907_);
lean_dec_ref_known(v___x_899_, 1);
v_fst_908_ = lean_ctor_get(v_val_907_, 0);
v_snd_909_ = lean_ctor_get(v_val_907_, 1);
v_isSharedCheck_920_ = !lean_is_exclusive(v_val_907_);
if (v_isSharedCheck_920_ == 0)
{
v___x_911_ = v_val_907_;
v_isShared_912_ = v_isSharedCheck_920_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_snd_909_);
lean_inc(v_fst_908_);
lean_dec(v_val_907_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_920_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v___x_914_; 
if (v_isShared_912_ == 0)
{
v___x_914_ = v___x_911_;
goto v_reusejp_913_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v_fst_908_);
lean_ctor_set(v_reuseFailAlloc_919_, 1, v_snd_909_);
v___x_914_ = v_reuseFailAlloc_919_;
goto v_reusejp_913_;
}
v_reusejp_913_:
{
lean_object* v___x_915_; lean_object* v___x_917_; 
v___x_915_ = lean_array_set(v_entries_896_, v_j_865_, v___x_914_);
lean_dec(v_j_865_);
if (v_isShared_890_ == 0)
{
lean_ctor_set(v___x_889_, 0, v___x_915_);
v___x_917_ = v___x_889_;
goto v_reusejp_916_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v___x_915_);
v___x_917_ = v_reuseFailAlloc_918_;
goto v_reusejp_916_;
}
v_reusejp_916_:
{
return v___x_917_;
}
}
}
}
}
}
}
default: 
{
lean_dec(v_j_865_);
return v_x_858_;
}
}
v___jp_866_:
{
if (v___y_867_ == 0)
{
lean_dec(v_j_865_);
return v_x_858_;
}
else
{
lean_object* v___x_869_; uint8_t v_isShared_870_; uint8_t v_isSharedCheck_875_; 
lean_inc_ref(v_es_861_);
v_isSharedCheck_875_ = !lean_is_exclusive(v_x_858_);
if (v_isSharedCheck_875_ == 0)
{
lean_object* v_unused_876_; 
v_unused_876_ = lean_ctor_get(v_x_858_, 0);
lean_dec(v_unused_876_);
v___x_869_ = v_x_858_;
v_isShared_870_ = v_isSharedCheck_875_;
goto v_resetjp_868_;
}
else
{
lean_dec(v_x_858_);
v___x_869_ = lean_box(0);
v_isShared_870_ = v_isSharedCheck_875_;
goto v_resetjp_868_;
}
v_resetjp_868_:
{
lean_object* v___x_871_; lean_object* v___x_873_; 
v___x_871_ = lean_array_set(v_es_861_, v_j_865_, v___x_862_);
lean_dec(v_j_865_);
if (v_isShared_870_ == 0)
{
lean_ctor_set(v___x_869_, 0, v___x_871_);
v___x_873_ = v___x_869_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_871_);
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
}
else
{
lean_object* v_ks_924_; lean_object* v_vs_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_939_; 
v_ks_924_ = lean_ctor_get(v_x_858_, 0);
v_vs_925_ = lean_ctor_get(v_x_858_, 1);
v_isSharedCheck_939_ = !lean_is_exclusive(v_x_858_);
if (v_isSharedCheck_939_ == 0)
{
v___x_927_ = v_x_858_;
v_isShared_928_ = v_isSharedCheck_939_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_vs_925_);
lean_inc(v_ks_924_);
lean_dec(v_x_858_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_939_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v___x_929_; 
v___x_929_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0_spec__1(v_ks_924_, v_x_860_);
if (lean_obj_tag(v___x_929_) == 0)
{
lean_object* v___x_931_; 
if (v_isShared_928_ == 0)
{
v___x_931_ = v___x_927_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v_ks_924_);
lean_ctor_set(v_reuseFailAlloc_932_, 1, v_vs_925_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
else
{
lean_object* v_val_933_; lean_object* v_keys_x27_934_; lean_object* v_vals_x27_935_; lean_object* v___x_937_; 
v_val_933_ = lean_ctor_get(v___x_929_, 0);
lean_inc_n(v_val_933_, 2);
lean_dec_ref_known(v___x_929_, 1);
v_keys_x27_934_ = l_Array_eraseIdx___redArg(v_ks_924_, v_val_933_);
v_vals_x27_935_ = l_Array_eraseIdx___redArg(v_vs_925_, v_val_933_);
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 1, v_vals_x27_935_);
lean_ctor_set(v___x_927_, 0, v_keys_x27_934_);
v___x_937_ = v___x_927_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v_keys_x27_934_);
lean_ctor_set(v_reuseFailAlloc_938_, 1, v_vals_x27_935_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg___boxed(lean_object* v_x_940_, lean_object* v_x_941_, lean_object* v_x_942_){
_start:
{
size_t v_x_884__boxed_943_; lean_object* v_res_944_; 
v_x_884__boxed_943_ = lean_unbox_usize(v_x_941_);
lean_dec(v_x_941_);
v_res_944_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(v_x_940_, v_x_884__boxed_943_, v_x_942_);
lean_dec_ref(v_x_942_);
return v_res_944_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg(lean_object* v_x_945_, lean_object* v_x_946_){
_start:
{
uint64_t v___y_948_; uint64_t v___y_952_; uint64_t v___y_956_; 
if (lean_obj_tag(v_x_946_) == 0)
{
uint8_t v_inv_959_; 
v_inv_959_ = lean_ctor_get_uint8(v_x_946_, sizeof(void*)*1 + 1);
if (v_inv_959_ == 0)
{
lean_object* v_declName_960_; 
v_declName_960_ = lean_ctor_get(v_x_946_, 0);
if (lean_obj_tag(v_declName_960_) == 0)
{
uint64_t v___x_961_; 
v___x_961_ = 1723ULL;
v___y_952_ = v___x_961_;
goto v___jp_951_;
}
else
{
uint64_t v_hash_962_; 
v_hash_962_ = lean_ctor_get_uint64(v_declName_960_, sizeof(void*)*2);
v___y_952_ = v_hash_962_;
goto v___jp_951_;
}
}
else
{
lean_object* v_declName_963_; 
v_declName_963_ = lean_ctor_get(v_x_946_, 0);
if (lean_obj_tag(v_declName_963_) == 0)
{
uint64_t v___x_964_; 
v___x_964_ = 1723ULL;
v___y_956_ = v___x_964_;
goto v___jp_955_;
}
else
{
uint64_t v_hash_965_; 
v_hash_965_ = lean_ctor_get_uint64(v_declName_963_, sizeof(void*)*2);
v___y_956_ = v_hash_965_;
goto v___jp_955_;
}
}
}
else
{
lean_object* v___x_966_; 
v___x_966_ = l_Lean_Meta_Origin_key(v_x_946_);
if (lean_obj_tag(v___x_966_) == 0)
{
uint64_t v___x_967_; 
v___x_967_ = 1723ULL;
v___y_948_ = v___x_967_;
goto v___jp_947_;
}
else
{
uint64_t v_hash_968_; 
v_hash_968_ = lean_ctor_get_uint64(v___x_966_, sizeof(void*)*2);
lean_dec(v___x_966_);
v___y_948_ = v_hash_968_;
goto v___jp_947_;
}
}
v___jp_947_:
{
size_t v_h_949_; lean_object* v___x_950_; 
v_h_949_ = lean_uint64_to_usize(v___y_948_);
v___x_950_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(v_x_945_, v_h_949_, v_x_946_);
return v___x_950_;
}
v___jp_951_:
{
uint64_t v___x_953_; uint64_t v___x_954_; 
v___x_953_ = 13ULL;
v___x_954_ = lean_uint64_mix_hash(v___y_952_, v___x_953_);
v___y_948_ = v___x_954_;
goto v___jp_947_;
}
v___jp_955_:
{
uint64_t v___x_957_; uint64_t v___x_958_; 
v___x_957_ = 11ULL;
v___x_958_ = lean_uint64_mix_hash(v___y_956_, v___x_957_);
v___y_948_ = v___x_958_;
goto v___jp_947_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg___boxed(lean_object* v_x_969_, lean_object* v_x_970_){
_start:
{
lean_object* v_res_971_; 
v_res_971_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg(v_x_969_, v_x_970_);
lean_dec_ref(v_x_970_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_addSimpEntry(lean_object* v_s_972_, lean_object* v_x_973_){
_start:
{
switch(lean_obj_tag(v_x_973_))
{
case 0:
{
lean_object* v_a_974_; lean_object* v___x_975_; lean_object* v_pre_976_; lean_object* v_post_977_; lean_object* v_lemmaNames_978_; lean_object* v_toUnfold_979_; lean_object* v_toUnfoldThms_980_; lean_object* v_erased_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_990_; 
v_a_974_ = lean_ctor_get(v_x_973_, 0);
lean_inc_ref_n(v_a_974_, 2);
lean_dec_ref_known(v_x_973_, 1);
lean_inc_ref(v_s_972_);
v___x_975_ = l_Lean_Meta_SimpTheorems_addSimpTheorem(v_s_972_, v_a_974_);
v_pre_976_ = lean_ctor_get(v___x_975_, 0);
lean_inc_ref(v_pre_976_);
v_post_977_ = lean_ctor_get(v___x_975_, 1);
lean_inc_ref(v_post_977_);
v_lemmaNames_978_ = lean_ctor_get(v___x_975_, 2);
lean_inc_ref(v_lemmaNames_978_);
v_toUnfold_979_ = lean_ctor_get(v___x_975_, 3);
lean_inc_ref(v_toUnfold_979_);
v_toUnfoldThms_980_ = lean_ctor_get(v___x_975_, 5);
lean_inc_ref(v_toUnfoldThms_980_);
lean_dec_ref(v___x_975_);
v_erased_981_ = lean_ctor_get(v_s_972_, 4);
v_isSharedCheck_990_ = !lean_is_exclusive(v_s_972_);
if (v_isSharedCheck_990_ == 0)
{
lean_object* v_unused_991_; lean_object* v_unused_992_; lean_object* v_unused_993_; lean_object* v_unused_994_; lean_object* v_unused_995_; 
v_unused_991_ = lean_ctor_get(v_s_972_, 5);
lean_dec(v_unused_991_);
v_unused_992_ = lean_ctor_get(v_s_972_, 3);
lean_dec(v_unused_992_);
v_unused_993_ = lean_ctor_get(v_s_972_, 2);
lean_dec(v_unused_993_);
v_unused_994_ = lean_ctor_get(v_s_972_, 1);
lean_dec(v_unused_994_);
v_unused_995_ = lean_ctor_get(v_s_972_, 0);
lean_dec(v_unused_995_);
v___x_983_ = v_s_972_;
v_isShared_984_ = v_isSharedCheck_990_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_erased_981_);
lean_dec(v_s_972_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_990_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v_origin_985_; lean_object* v___x_986_; lean_object* v___x_988_; 
v_origin_985_ = lean_ctor_get(v_a_974_, 4);
lean_inc_ref(v_origin_985_);
lean_dec_ref(v_a_974_);
v___x_986_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg(v_erased_981_, v_origin_985_);
lean_dec_ref(v_origin_985_);
if (v_isShared_984_ == 0)
{
lean_ctor_set(v___x_983_, 5, v_toUnfoldThms_980_);
lean_ctor_set(v___x_983_, 4, v___x_986_);
lean_ctor_set(v___x_983_, 3, v_toUnfold_979_);
lean_ctor_set(v___x_983_, 2, v_lemmaNames_978_);
lean_ctor_set(v___x_983_, 1, v_post_977_);
lean_ctor_set(v___x_983_, 0, v_pre_976_);
v___x_988_ = v___x_983_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v_pre_976_);
lean_ctor_set(v_reuseFailAlloc_989_, 1, v_post_977_);
lean_ctor_set(v_reuseFailAlloc_989_, 2, v_lemmaNames_978_);
lean_ctor_set(v_reuseFailAlloc_989_, 3, v_toUnfold_979_);
lean_ctor_set(v_reuseFailAlloc_989_, 4, v___x_986_);
lean_ctor_set(v_reuseFailAlloc_989_, 5, v_toUnfoldThms_980_);
v___x_988_ = v_reuseFailAlloc_989_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
return v___x_988_;
}
}
}
case 1:
{
lean_object* v_a_996_; lean_object* v_pre_997_; lean_object* v_post_998_; lean_object* v_lemmaNames_999_; lean_object* v_toUnfold_1000_; lean_object* v_erased_1001_; lean_object* v_toUnfoldThms_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1011_; 
v_a_996_ = lean_ctor_get(v_x_973_, 0);
lean_inc(v_a_996_);
lean_dec_ref_known(v_x_973_, 1);
v_pre_997_ = lean_ctor_get(v_s_972_, 0);
v_post_998_ = lean_ctor_get(v_s_972_, 1);
v_lemmaNames_999_ = lean_ctor_get(v_s_972_, 2);
v_toUnfold_1000_ = lean_ctor_get(v_s_972_, 3);
v_erased_1001_ = lean_ctor_get(v_s_972_, 4);
v_toUnfoldThms_1002_ = lean_ctor_get(v_s_972_, 5);
v_isSharedCheck_1011_ = !lean_is_exclusive(v_s_972_);
if (v_isSharedCheck_1011_ == 0)
{
v___x_1004_ = v_s_972_;
v_isShared_1005_ = v_isSharedCheck_1011_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_toUnfoldThms_1002_);
lean_inc(v_erased_1001_);
lean_inc(v_toUnfold_1000_);
lean_inc(v_lemmaNames_999_);
lean_inc(v_post_998_);
lean_inc(v_pre_997_);
lean_dec(v_s_972_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1011_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1009_; 
v___x_1006_ = lean_box(0);
v___x_1007_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1___redArg(v_toUnfold_1000_, v_a_996_, v___x_1006_);
if (v_isShared_1005_ == 0)
{
lean_ctor_set(v___x_1004_, 3, v___x_1007_);
v___x_1009_ = v___x_1004_;
goto v_reusejp_1008_;
}
else
{
lean_object* v_reuseFailAlloc_1010_; 
v_reuseFailAlloc_1010_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1010_, 0, v_pre_997_);
lean_ctor_set(v_reuseFailAlloc_1010_, 1, v_post_998_);
lean_ctor_set(v_reuseFailAlloc_1010_, 2, v_lemmaNames_999_);
lean_ctor_set(v_reuseFailAlloc_1010_, 3, v___x_1007_);
lean_ctor_set(v_reuseFailAlloc_1010_, 4, v_erased_1001_);
lean_ctor_set(v_reuseFailAlloc_1010_, 5, v_toUnfoldThms_1002_);
v___x_1009_ = v_reuseFailAlloc_1010_;
goto v_reusejp_1008_;
}
v_reusejp_1008_:
{
return v___x_1009_;
}
}
}
default: 
{
lean_object* v_a_1012_; lean_object* v_a_1013_; lean_object* v___x_1014_; 
v_a_1012_ = lean_ctor_get(v_x_973_, 0);
lean_inc(v_a_1012_);
v_a_1013_ = lean_ctor_get(v_x_973_, 1);
lean_inc_ref(v_a_1013_);
lean_dec_ref_known(v_x_973_, 2);
v___x_1014_ = l_Lean_Meta_SimpTheorems_registerDeclToUnfoldThms(v_s_972_, v_a_1012_, v_a_1013_);
return v___x_1014_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0(lean_object* v_00_u03b2_1015_, lean_object* v_x_1016_, lean_object* v_x_1017_){
_start:
{
lean_object* v___x_1018_; 
v___x_1018_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___redArg(v_x_1016_, v_x_1017_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0___boxed(lean_object* v_00_u03b2_1019_, lean_object* v_x_1020_, lean_object* v_x_1021_){
_start:
{
lean_object* v_res_1022_; 
v_res_1022_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0(v_00_u03b2_1019_, v_x_1020_, v_x_1021_);
lean_dec_ref(v_x_1021_);
return v_res_1022_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1(lean_object* v_00_u03b2_1023_, lean_object* v_x_1024_, lean_object* v_x_1025_, lean_object* v_x_1026_){
_start:
{
lean_object* v___x_1027_; 
v___x_1027_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1___redArg(v_x_1024_, v_x_1025_, v_x_1026_);
return v___x_1027_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0(lean_object* v_00_u03b2_1028_, lean_object* v_x_1029_, size_t v_x_1030_, lean_object* v_x_1031_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___redArg(v_x_1029_, v_x_1030_, v_x_1031_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1033_, lean_object* v_x_1034_, lean_object* v_x_1035_, lean_object* v_x_1036_){
_start:
{
size_t v_x_1159__boxed_1037_; lean_object* v_res_1038_; 
v_x_1159__boxed_1037_ = lean_unbox_usize(v_x_1035_);
lean_dec(v_x_1035_);
v_res_1038_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_SimpTheorems_addSimpEntry_spec__0_spec__0(v_00_u03b2_1033_, v_x_1034_, v_x_1159__boxed_1037_, v_x_1036_);
lean_dec_ref(v_x_1036_);
return v_res_1038_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2(lean_object* v_00_u03b2_1039_, lean_object* v_x_1040_, size_t v_x_1041_, size_t v_x_1042_, lean_object* v_x_1043_, lean_object* v_x_1044_){
_start:
{
lean_object* v___x_1045_; 
v___x_1045_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___redArg(v_x_1040_, v_x_1041_, v_x_1042_, v_x_1043_, v_x_1044_);
return v___x_1045_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1046_, lean_object* v_x_1047_, lean_object* v_x_1048_, lean_object* v_x_1049_, lean_object* v_x_1050_, lean_object* v_x_1051_){
_start:
{
size_t v_x_1170__boxed_1052_; size_t v_x_1171__boxed_1053_; lean_object* v_res_1054_; 
v_x_1170__boxed_1052_ = lean_unbox_usize(v_x_1048_);
lean_dec(v_x_1048_);
v_x_1171__boxed_1053_ = lean_unbox_usize(v_x_1049_);
lean_dec(v_x_1049_);
v_res_1054_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2(v_00_u03b2_1046_, v_x_1047_, v_x_1170__boxed_1052_, v_x_1171__boxed_1053_, v_x_1050_, v_x_1051_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1055_, lean_object* v_n_1056_, lean_object* v_k_1057_, lean_object* v_v_1058_){
_start:
{
lean_object* v___x_1059_; 
v___x_1059_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4___redArg(v_n_1056_, v_k_1057_, v_v_1058_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_1060_, size_t v_depth_1061_, lean_object* v_keys_1062_, lean_object* v_vals_1063_, lean_object* v_heq_1064_, lean_object* v_i_1065_, lean_object* v_entries_1066_){
_start:
{
lean_object* v___x_1067_; 
v___x_1067_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___redArg(v_depth_1061_, v_keys_1062_, v_vals_1063_, v_i_1065_, v_entries_1066_);
return v___x_1067_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b2_1068_, lean_object* v_depth_1069_, lean_object* v_keys_1070_, lean_object* v_vals_1071_, lean_object* v_heq_1072_, lean_object* v_i_1073_, lean_object* v_entries_1074_){
_start:
{
size_t v_depth_boxed_1075_; lean_object* v_res_1076_; 
v_depth_boxed_1075_ = lean_unbox_usize(v_depth_1069_);
lean_dec(v_depth_1069_);
v_res_1076_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__5(v_00_u03b2_1068_, v_depth_boxed_1075_, v_keys_1070_, v_vals_1071_, v_heq_1072_, v_i_1073_, v_entries_1074_);
lean_dec_ref(v_vals_1071_);
lean_dec_ref(v_keys_1070_);
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6(lean_object* v_00_u03b2_1077_, lean_object* v_x_1078_, lean_object* v_x_1079_, lean_object* v_x_1080_, lean_object* v_x_1081_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_SimpTheorems_addSimpEntry_spec__1_spec__2_spec__4_spec__6___redArg(v_x_1078_, v_x_1079_, v_x_1080_, v_x_1081_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg(lean_object* v_inst_1085_, lean_object* v_f_1086_, lean_object* v_thms_1087_, lean_object* v_s_1088_, lean_object* v_thm_1089_){
_start:
{
lean_object* v_erased_1090_; lean_object* v_origin_1091_; lean_object* v___f_1092_; lean_object* v___f_1093_; uint8_t v___x_1094_; 
v_erased_1090_ = lean_ctor_get(v_thms_1087_, 4);
lean_inc_ref(v_erased_1090_);
lean_dec_ref(v_thms_1087_);
v_origin_1091_ = lean_ctor_get(v_thm_1089_, 4);
v___f_1092_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__0));
v___f_1093_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__1));
lean_inc_ref(v_origin_1091_);
v___x_1094_ = l_Lean_PersistentHashMap_contains___redArg(v___f_1092_, v___f_1093_, v_erased_1090_, v_origin_1091_);
if (v___x_1094_ == 0)
{
lean_object* v___x_1095_; lean_object* v___x_1096_; 
lean_dec_ref(v_inst_1085_);
v___x_1095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1095_, 0, v_thm_1089_);
v___x_1096_ = lean_apply_2(v_f_1086_, v_s_1088_, v___x_1095_);
return v___x_1096_;
}
else
{
lean_object* v_toApplicative_1097_; lean_object* v_toPure_1098_; lean_object* v___x_1099_; 
lean_dec_ref(v_thm_1089_);
lean_dec(v_f_1086_);
v_toApplicative_1097_ = lean_ctor_get(v_inst_1085_, 0);
lean_inc_ref(v_toApplicative_1097_);
lean_dec_ref(v_inst_1085_);
v_toPure_1098_ = lean_ctor_get(v_toApplicative_1097_, 1);
lean_inc(v_toPure_1098_);
lean_dec_ref(v_toApplicative_1097_);
v___x_1099_ = lean_apply_2(v_toPure_1098_, lean_box(0), v_s_1088_);
return v___x_1099_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem(lean_object* v_m_1100_, lean_object* v_00_u03c3_1101_, lean_object* v_inst_1102_, lean_object* v_f_1103_, lean_object* v_thms_1104_, lean_object* v_s_1105_, lean_object* v_thm_1106_){
_start:
{
lean_object* v_erased_1107_; lean_object* v_origin_1108_; lean_object* v___f_1109_; lean_object* v___f_1110_; uint8_t v___x_1111_; 
v_erased_1107_ = lean_ctor_get(v_thms_1104_, 4);
lean_inc_ref(v_erased_1107_);
lean_dec_ref(v_thms_1104_);
v_origin_1108_ = lean_ctor_get(v_thm_1106_, 4);
v___f_1109_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__0));
v___f_1110_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem___redArg___closed__1));
lean_inc_ref(v_origin_1108_);
v___x_1111_ = l_Lean_PersistentHashMap_contains___redArg(v___f_1109_, v___f_1110_, v_erased_1107_, v_origin_1108_);
if (v___x_1111_ == 0)
{
lean_object* v___x_1112_; lean_object* v___x_1113_; 
lean_dec_ref(v_inst_1102_);
v___x_1112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1112_, 0, v_thm_1106_);
v___x_1113_ = lean_apply_2(v_f_1103_, v_s_1105_, v___x_1112_);
return v___x_1113_;
}
else
{
lean_object* v_toApplicative_1114_; lean_object* v_toPure_1115_; lean_object* v___x_1116_; 
lean_dec_ref(v_thm_1106_);
lean_dec(v_f_1103_);
v_toApplicative_1114_ = lean_ctor_get(v_inst_1102_, 0);
lean_inc_ref(v_toApplicative_1114_);
lean_dec_ref(v_inst_1102_);
v_toPure_1115_ = lean_ctor_get(v_toApplicative_1114_, 1);
lean_inc(v_toPure_1115_);
lean_dec_ref(v_toApplicative_1114_);
v___x_1116_ = lean_apply_2(v_toPure_1115_, lean_box(0), v_s_1105_);
return v___x_1116_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__0(lean_object* v_f_1117_, lean_object* v_s_1118_, lean_object* v_n_1119_, lean_object* v_thms_1120_){
_start:
{
lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1121_, 0, v_n_1119_);
lean_ctor_set(v___x_1121_, 1, v_thms_1120_);
v___x_1122_ = lean_apply_2(v_f_1117_, v_s_1118_, v___x_1121_);
return v___x_1122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__1(lean_object* v_inst_1123_, lean_object* v___f_1124_, lean_object* v_toUnfoldThms_1125_, lean_object* v_s_1126_){
_start:
{
lean_object* v___x_1127_; 
v___x_1127_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1123_, v___f_1124_, v_toUnfoldThms_1125_, v_s_1126_);
return v___x_1127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__2(lean_object* v_f_1128_, lean_object* v_d_1129_, lean_object* v_a_1130_, lean_object* v_x_1131_){
_start:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; 
v___x_1132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1132_, 0, v_a_1130_);
v___x_1133_ = lean_apply_2(v_f_1128_, v_d_1129_, v___x_1132_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__3(lean_object* v_inst_1134_, lean_object* v___f_1135_, lean_object* v_toUnfold_1136_, lean_object* v_toBind_1137_, lean_object* v___f_1138_, lean_object* v_s_1139_){
_start:
{
lean_object* v___x_1140_; lean_object* v___x_1141_; 
v___x_1140_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1134_, v___f_1135_, v_toUnfold_1136_, v_s_1139_);
v___x_1141_ = lean_apply_4(v_toBind_1137_, lean_box(0), lean_box(0), v___x_1140_, v___f_1138_);
return v___x_1141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4(lean_object* v_inst_1142_, lean_object* v___x_1143_, lean_object* v_s_1144_, lean_object* v_x_1145_, lean_object* v_t_1146_){
_start:
{
lean_object* v___x_1147_; 
v___x_1147_ = l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(v_inst_1142_, v___x_1143_, v_s_1144_, v_t_1146_);
return v___x_1147_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4___boxed(lean_object* v_inst_1148_, lean_object* v___x_1149_, lean_object* v_s_1150_, lean_object* v_x_1151_, lean_object* v_t_1152_){
_start:
{
lean_object* v_res_1153_; 
v_res_1153_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4(v_inst_1148_, v___x_1149_, v_s_1150_, v_x_1151_, v_t_1152_);
lean_dec(v_x_1151_);
return v_res_1153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__6(lean_object* v_inst_1154_, lean_object* v___f_1155_, lean_object* v_post_1156_, lean_object* v_toBind_1157_, lean_object* v___f_1158_, lean_object* v_s_1159_){
_start:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; 
v___x_1160_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1154_, v___f_1155_, v_post_1156_, v_s_1159_);
v___x_1161_ = lean_apply_4(v_toBind_1157_, lean_box(0), lean_box(0), v___x_1160_, v___f_1158_);
return v___x_1161_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg(lean_object* v_inst_1162_, lean_object* v_f_1163_, lean_object* v_init_1164_, lean_object* v_thms_1165_){
_start:
{
lean_object* v_toBind_1166_; lean_object* v_pre_1167_; lean_object* v_post_1168_; lean_object* v_toUnfold_1169_; lean_object* v_toUnfoldThms_1170_; lean_object* v___f_1171_; lean_object* v___f_1172_; lean_object* v___f_1173_; lean_object* v___f_1174_; lean_object* v___x_1175_; lean_object* v___f_1176_; lean_object* v___f_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; 
v_toBind_1166_ = lean_ctor_get(v_inst_1162_, 1);
lean_inc_n(v_toBind_1166_, 3);
v_pre_1167_ = lean_ctor_get(v_thms_1165_, 0);
lean_inc_ref(v_pre_1167_);
v_post_1168_ = lean_ctor_get(v_thms_1165_, 1);
lean_inc_ref(v_post_1168_);
v_toUnfold_1169_ = lean_ctor_get(v_thms_1165_, 3);
v_toUnfoldThms_1170_ = lean_ctor_get(v_thms_1165_, 5);
lean_inc_n(v_f_1163_, 2);
v___f_1171_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1171_, 0, v_f_1163_);
lean_inc_ref(v_toUnfoldThms_1170_);
lean_inc_ref_n(v_inst_1162_, 5);
v___f_1172_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_1172_, 0, v_inst_1162_);
lean_closure_set(v___f_1172_, 1, v___f_1171_);
lean_closure_set(v___f_1172_, 2, v_toUnfoldThms_1170_);
v___f_1173_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__2), 4, 1);
lean_closure_set(v___f_1173_, 0, v_f_1163_);
lean_inc_ref(v_toUnfold_1169_);
v___f_1174_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__3), 6, 5);
lean_closure_set(v___f_1174_, 0, v_inst_1162_);
lean_closure_set(v___f_1174_, 1, v___f_1173_);
lean_closure_set(v___f_1174_, 2, v_toUnfold_1169_);
lean_closure_set(v___f_1174_, 3, v_toBind_1166_);
lean_closure_set(v___f_1174_, 4, v___f_1172_);
v___x_1175_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_SimpTheorems_foldSimpEntriesM_processTheorem), 7, 5);
lean_closure_set(v___x_1175_, 0, lean_box(0));
lean_closure_set(v___x_1175_, 1, lean_box(0));
lean_closure_set(v___x_1175_, 2, v_inst_1162_);
lean_closure_set(v___x_1175_, 3, v_f_1163_);
lean_closure_set(v___x_1175_, 4, v_thms_1165_);
v___f_1176_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__4___boxed), 5, 2);
lean_closure_set(v___f_1176_, 0, v_inst_1162_);
lean_closure_set(v___f_1176_, 1, v___x_1175_);
lean_inc_ref(v___f_1176_);
v___f_1177_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__6), 6, 5);
lean_closure_set(v___f_1177_, 0, v_inst_1162_);
lean_closure_set(v___f_1177_, 1, v___f_1176_);
lean_closure_set(v___f_1177_, 2, v_post_1168_);
lean_closure_set(v___f_1177_, 3, v_toBind_1166_);
lean_closure_set(v___f_1177_, 4, v___f_1174_);
v___x_1178_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1162_, v___f_1176_, v_pre_1167_, v_init_1164_);
v___x_1179_ = lean_apply_4(v_toBind_1166_, lean_box(0), lean_box(0), v___x_1178_, v___f_1177_);
return v___x_1179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM(lean_object* v_m_1180_, lean_object* v_00_u03c3_1181_, lean_object* v_inst_1182_, lean_object* v_f_1183_, lean_object* v_init_1184_, lean_object* v_thms_1185_){
_start:
{
lean_object* v___x_1186_; 
v___x_1186_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg(v_inst_1182_, v_f_1183_, v_init_1184_, v_thms_1185_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(lean_object* v_f_1187_, lean_object* v_as_1188_, size_t v_i_1189_, size_t v_stop_1190_, lean_object* v_b_1191_){
_start:
{
uint8_t v___x_1192_; 
v___x_1192_ = lean_usize_dec_eq(v_i_1189_, v_stop_1190_);
if (v___x_1192_ == 0)
{
lean_object* v___x_1193_; lean_object* v___x_1194_; size_t v___x_1195_; size_t v___x_1196_; 
v___x_1193_ = lean_array_uget_borrowed(v_as_1188_, v_i_1189_);
lean_inc(v_f_1187_);
lean_inc(v___x_1193_);
v___x_1194_ = lean_apply_2(v_f_1187_, v_b_1191_, v___x_1193_);
v___x_1195_ = ((size_t)1ULL);
v___x_1196_ = lean_usize_add(v_i_1189_, v___x_1195_);
v_i_1189_ = v___x_1196_;
v_b_1191_ = v___x_1194_;
goto _start;
}
else
{
lean_dec(v_f_1187_);
return v_b_1191_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_f_1198_, lean_object* v_as_1199_, lean_object* v_i_1200_, lean_object* v_stop_1201_, lean_object* v_b_1202_){
_start:
{
size_t v_i_boxed_1203_; size_t v_stop_boxed_1204_; lean_object* v_res_1205_; 
v_i_boxed_1203_ = lean_unbox_usize(v_i_1200_);
lean_dec(v_i_1200_);
v_stop_boxed_1204_ = lean_unbox_usize(v_stop_1201_);
lean_dec(v_stop_1201_);
v_res_1205_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(v_f_1198_, v_as_1199_, v_i_boxed_1203_, v_stop_boxed_1204_, v_b_1202_);
lean_dec_ref(v_as_1199_);
return v_res_1205_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(lean_object* v_f_1206_, lean_object* v_x_1207_, lean_object* v_x_1208_){
_start:
{
lean_object* v_vs_1209_; lean_object* v_children_1210_; lean_object* v___x_1211_; lean_object* v_s_1213_; lean_object* v___x_1223_; uint8_t v___x_1224_; 
v_vs_1209_ = lean_ctor_get(v_x_1208_, 0);
v_children_1210_ = lean_ctor_get(v_x_1208_, 1);
v___x_1211_ = lean_unsigned_to_nat(0u);
v___x_1223_ = lean_array_get_size(v_vs_1209_);
v___x_1224_ = lean_nat_dec_lt(v___x_1211_, v___x_1223_);
if (v___x_1224_ == 0)
{
lean_object* v___x_1225_; uint8_t v___x_1226_; 
v___x_1225_ = lean_array_get_size(v_children_1210_);
v___x_1226_ = lean_nat_dec_lt(v___x_1211_, v___x_1225_);
if (v___x_1226_ == 0)
{
lean_dec(v_f_1206_);
return v_x_1207_;
}
else
{
uint8_t v___x_1227_; 
v___x_1227_ = lean_nat_dec_le(v___x_1225_, v___x_1225_);
if (v___x_1227_ == 0)
{
if (v___x_1226_ == 0)
{
lean_dec(v_f_1206_);
return v_x_1207_;
}
else
{
size_t v___x_1228_; size_t v___x_1229_; lean_object* v___x_1230_; 
v___x_1228_ = ((size_t)0ULL);
v___x_1229_ = lean_usize_of_nat(v___x_1225_);
v___x_1230_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1206_, v_children_1210_, v___x_1228_, v___x_1229_, v_x_1207_);
return v___x_1230_;
}
}
else
{
size_t v___x_1231_; size_t v___x_1232_; lean_object* v___x_1233_; 
v___x_1231_ = ((size_t)0ULL);
v___x_1232_ = lean_usize_of_nat(v___x_1225_);
v___x_1233_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1206_, v_children_1210_, v___x_1231_, v___x_1232_, v_x_1207_);
return v___x_1233_;
}
}
}
else
{
uint8_t v___x_1234_; 
v___x_1234_ = lean_nat_dec_le(v___x_1223_, v___x_1223_);
if (v___x_1234_ == 0)
{
if (v___x_1224_ == 0)
{
v_s_1213_ = v_x_1207_;
goto v___jp_1212_;
}
else
{
size_t v___x_1235_; size_t v___x_1236_; lean_object* v___x_1237_; 
v___x_1235_ = ((size_t)0ULL);
v___x_1236_ = lean_usize_of_nat(v___x_1223_);
lean_inc(v_f_1206_);
v___x_1237_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(v_f_1206_, v_vs_1209_, v___x_1235_, v___x_1236_, v_x_1207_);
v_s_1213_ = v___x_1237_;
goto v___jp_1212_;
}
}
else
{
size_t v___x_1238_; size_t v___x_1239_; lean_object* v___x_1240_; 
v___x_1238_ = ((size_t)0ULL);
v___x_1239_ = lean_usize_of_nat(v___x_1223_);
lean_inc(v_f_1206_);
v___x_1240_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(v_f_1206_, v_vs_1209_, v___x_1238_, v___x_1239_, v_x_1207_);
v_s_1213_ = v___x_1240_;
goto v___jp_1212_;
}
}
v___jp_1212_:
{
lean_object* v___x_1214_; uint8_t v___x_1215_; 
v___x_1214_ = lean_array_get_size(v_children_1210_);
v___x_1215_ = lean_nat_dec_lt(v___x_1211_, v___x_1214_);
if (v___x_1215_ == 0)
{
lean_dec(v_f_1206_);
return v_s_1213_;
}
else
{
uint8_t v___x_1216_; 
v___x_1216_ = lean_nat_dec_le(v___x_1214_, v___x_1214_);
if (v___x_1216_ == 0)
{
if (v___x_1215_ == 0)
{
lean_dec(v_f_1206_);
return v_s_1213_;
}
else
{
size_t v___x_1217_; size_t v___x_1218_; lean_object* v___x_1219_; 
v___x_1217_ = ((size_t)0ULL);
v___x_1218_ = lean_usize_of_nat(v___x_1214_);
v___x_1219_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1206_, v_children_1210_, v___x_1217_, v___x_1218_, v_s_1213_);
return v___x_1219_;
}
}
else
{
size_t v___x_1220_; size_t v___x_1221_; lean_object* v___x_1222_; 
v___x_1220_ = ((size_t)0ULL);
v___x_1221_ = lean_usize_of_nat(v___x_1214_);
v___x_1222_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1206_, v_children_1210_, v___x_1220_, v___x_1221_, v_s_1213_);
return v___x_1222_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(lean_object* v_f_1241_, lean_object* v_as_1242_, size_t v_i_1243_, size_t v_stop_1244_, lean_object* v_b_1245_){
_start:
{
uint8_t v___x_1246_; 
v___x_1246_ = lean_usize_dec_eq(v_i_1243_, v_stop_1244_);
if (v___x_1246_ == 0)
{
lean_object* v___x_1247_; lean_object* v_snd_1248_; lean_object* v___x_1249_; size_t v___x_1250_; size_t v___x_1251_; 
v___x_1247_ = lean_array_uget_borrowed(v_as_1242_, v_i_1243_);
v_snd_1248_ = lean_ctor_get(v___x_1247_, 1);
lean_inc(v_f_1241_);
v___x_1249_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(v_f_1241_, v_b_1245_, v_snd_1248_);
v___x_1250_ = ((size_t)1ULL);
v___x_1251_ = lean_usize_add(v_i_1243_, v___x_1250_);
v_i_1243_ = v___x_1251_;
v_b_1245_ = v___x_1249_;
goto _start;
}
else
{
lean_dec(v_f_1241_);
return v_b_1245_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_f_1253_, lean_object* v_as_1254_, lean_object* v_i_1255_, lean_object* v_stop_1256_, lean_object* v_b_1257_){
_start:
{
size_t v_i_boxed_1258_; size_t v_stop_boxed_1259_; lean_object* v_res_1260_; 
v_i_boxed_1258_ = lean_unbox_usize(v_i_1255_);
lean_dec(v_i_1255_);
v_stop_boxed_1259_ = lean_unbox_usize(v_stop_1256_);
lean_dec(v_stop_1256_);
v_res_1260_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1253_, v_as_1254_, v_i_boxed_1258_, v_stop_boxed_1259_, v_b_1257_);
lean_dec_ref(v_as_1254_);
return v_res_1260_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg___boxed(lean_object* v_f_1261_, lean_object* v_x_1262_, lean_object* v_x_1263_){
_start:
{
lean_object* v_res_1264_; 
v_res_1264_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(v_f_1261_, v_x_1262_, v_x_1263_);
lean_dec_ref(v_x_1263_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0(lean_object* v___f_1265_, lean_object* v_s_1266_, lean_object* v_x_1267_, lean_object* v_t_1268_){
_start:
{
lean_object* v___x_1269_; 
v___x_1269_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(v___f_1265_, v_s_1266_, v_t_1268_);
return v___x_1269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0___boxed(lean_object* v___f_1270_, lean_object* v_s_1271_, lean_object* v_x_1272_, lean_object* v_t_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0(v___f_1270_, v_s_1271_, v_x_1272_, v_t_1273_);
lean_dec_ref(v_t_1273_);
lean_dec(v_x_1272_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg(lean_object* v_f_1275_, lean_object* v_keys_1276_, lean_object* v_vals_1277_, lean_object* v_i_1278_, lean_object* v_acc_1279_){
_start:
{
lean_object* v___x_1280_; uint8_t v___x_1281_; 
v___x_1280_ = lean_array_get_size(v_keys_1276_);
v___x_1281_ = lean_nat_dec_lt(v_i_1278_, v___x_1280_);
if (v___x_1281_ == 0)
{
lean_dec(v_i_1278_);
lean_dec(v_f_1275_);
return v_acc_1279_;
}
else
{
lean_object* v_k_1282_; lean_object* v_v_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; 
v_k_1282_ = lean_array_fget_borrowed(v_keys_1276_, v_i_1278_);
v_v_1283_ = lean_array_fget_borrowed(v_vals_1277_, v_i_1278_);
lean_inc(v_f_1275_);
lean_inc(v_v_1283_);
lean_inc(v_k_1282_);
v___x_1284_ = lean_apply_3(v_f_1275_, v_acc_1279_, v_k_1282_, v_v_1283_);
v___x_1285_ = lean_unsigned_to_nat(1u);
v___x_1286_ = lean_nat_add(v_i_1278_, v___x_1285_);
lean_dec(v_i_1278_);
v_i_1278_ = v___x_1286_;
v_acc_1279_ = v___x_1284_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg___boxed(lean_object* v_f_1288_, lean_object* v_keys_1289_, lean_object* v_vals_1290_, lean_object* v_i_1291_, lean_object* v_acc_1292_){
_start:
{
lean_object* v_res_1293_; 
v_res_1293_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg(v_f_1288_, v_keys_1289_, v_vals_1290_, v_i_1291_, v_acc_1292_);
lean_dec_ref(v_vals_1290_);
lean_dec_ref(v_keys_1289_);
return v_res_1293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(lean_object* v_f_1294_, lean_object* v_x_1295_, lean_object* v_x_1296_){
_start:
{
if (lean_obj_tag(v_x_1295_) == 0)
{
lean_object* v_es_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; uint8_t v___x_1300_; 
v_es_1297_ = lean_ctor_get(v_x_1295_, 0);
v___x_1298_ = lean_unsigned_to_nat(0u);
v___x_1299_ = lean_array_get_size(v_es_1297_);
v___x_1300_ = lean_nat_dec_lt(v___x_1298_, v___x_1299_);
if (v___x_1300_ == 0)
{
lean_dec(v_f_1294_);
return v_x_1296_;
}
else
{
uint8_t v___x_1301_; 
v___x_1301_ = lean_nat_dec_le(v___x_1299_, v___x_1299_);
if (v___x_1301_ == 0)
{
if (v___x_1300_ == 0)
{
lean_dec(v_f_1294_);
return v_x_1296_;
}
else
{
size_t v___x_1302_; size_t v___x_1303_; lean_object* v___x_1304_; 
v___x_1302_ = ((size_t)0ULL);
v___x_1303_ = lean_usize_of_nat(v___x_1299_);
v___x_1304_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(v_f_1294_, v_es_1297_, v___x_1302_, v___x_1303_, v_x_1296_);
return v___x_1304_;
}
}
else
{
size_t v___x_1305_; size_t v___x_1306_; lean_object* v___x_1307_; 
v___x_1305_ = ((size_t)0ULL);
v___x_1306_ = lean_usize_of_nat(v___x_1299_);
v___x_1307_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(v_f_1294_, v_es_1297_, v___x_1305_, v___x_1306_, v_x_1296_);
return v___x_1307_;
}
}
}
else
{
lean_object* v_ks_1308_; lean_object* v_vs_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; 
v_ks_1308_ = lean_ctor_get(v_x_1295_, 0);
v_vs_1309_ = lean_ctor_get(v_x_1295_, 1);
v___x_1310_ = lean_unsigned_to_nat(0u);
v___x_1311_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg(v_f_1294_, v_ks_1308_, v_vs_1309_, v___x_1310_, v_x_1296_);
return v___x_1311_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(lean_object* v_f_1312_, lean_object* v_as_1313_, size_t v_i_1314_, size_t v_stop_1315_, lean_object* v_b_1316_){
_start:
{
lean_object* v___y_1318_; uint8_t v___x_1322_; 
v___x_1322_ = lean_usize_dec_eq(v_i_1314_, v_stop_1315_);
if (v___x_1322_ == 0)
{
lean_object* v___x_1323_; 
v___x_1323_ = lean_array_uget_borrowed(v_as_1313_, v_i_1314_);
switch(lean_obj_tag(v___x_1323_))
{
case 0:
{
lean_object* v_key_1324_; lean_object* v_val_1325_; lean_object* v___x_1326_; 
v_key_1324_ = lean_ctor_get(v___x_1323_, 0);
v_val_1325_ = lean_ctor_get(v___x_1323_, 1);
lean_inc(v_f_1312_);
lean_inc(v_val_1325_);
lean_inc(v_key_1324_);
v___x_1326_ = lean_apply_3(v_f_1312_, v_b_1316_, v_key_1324_, v_val_1325_);
v___y_1318_ = v___x_1326_;
goto v___jp_1317_;
}
case 1:
{
lean_object* v_node_1327_; lean_object* v___x_1328_; 
v_node_1327_ = lean_ctor_get(v___x_1323_, 0);
lean_inc(v_f_1312_);
v___x_1328_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1312_, v_node_1327_, v_b_1316_);
v___y_1318_ = v___x_1328_;
goto v___jp_1317_;
}
default: 
{
v___y_1318_ = v_b_1316_;
goto v___jp_1317_;
}
}
}
else
{
lean_dec(v_f_1312_);
return v_b_1316_;
}
v___jp_1317_:
{
size_t v___x_1319_; size_t v___x_1320_; 
v___x_1319_ = ((size_t)1ULL);
v___x_1320_ = lean_usize_add(v_i_1314_, v___x_1319_);
v_i_1314_ = v___x_1320_;
v_b_1316_ = v___y_1318_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg___boxed(lean_object* v_f_1329_, lean_object* v_as_1330_, lean_object* v_i_1331_, lean_object* v_stop_1332_, lean_object* v_b_1333_){
_start:
{
size_t v_i_boxed_1334_; size_t v_stop_boxed_1335_; lean_object* v_res_1336_; 
v_i_boxed_1334_ = lean_unbox_usize(v_i_1331_);
lean_dec(v_i_1331_);
v_stop_boxed_1335_ = lean_unbox_usize(v_stop_1332_);
lean_dec(v_stop_1332_);
v_res_1336_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(v_f_1329_, v_as_1330_, v_i_boxed_1334_, v_stop_boxed_1335_, v_b_1333_);
lean_dec_ref(v_as_1330_);
return v_res_1336_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg___boxed(lean_object* v_f_1337_, lean_object* v_x_1338_, lean_object* v_x_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1337_, v_x_1338_, v_x_1339_);
lean_dec_ref(v_x_1338_);
return v_res_1340_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_keys_1341_, lean_object* v_i_1342_, lean_object* v_k_1343_){
_start:
{
uint8_t v___y_1349_; lean_object* v___x_1350_; uint8_t v___x_1351_; 
v___x_1350_ = lean_array_get_size(v_keys_1341_);
v___x_1351_ = lean_nat_dec_lt(v_i_1342_, v___x_1350_);
if (v___x_1351_ == 0)
{
lean_dec(v_i_1342_);
return v___x_1351_;
}
else
{
lean_object* v_k_x27_1352_; 
v_k_x27_1352_ = lean_array_fget_borrowed(v_keys_1341_, v_i_1342_);
if (lean_obj_tag(v_k_1343_) == 0)
{
if (lean_obj_tag(v_k_x27_1352_) == 0)
{
lean_object* v_declName_1353_; uint8_t v_inv_1354_; lean_object* v_declName_1355_; uint8_t v_inv_1356_; uint8_t v___x_1357_; 
v_declName_1353_ = lean_ctor_get(v_k_1343_, 0);
v_inv_1354_ = lean_ctor_get_uint8(v_k_1343_, sizeof(void*)*1 + 1);
v_declName_1355_ = lean_ctor_get(v_k_x27_1352_, 0);
v_inv_1356_ = lean_ctor_get_uint8(v_k_x27_1352_, sizeof(void*)*1 + 1);
v___x_1357_ = lean_name_eq(v_declName_1353_, v_declName_1355_);
if (v___x_1357_ == 0)
{
v___y_1349_ = v___x_1357_;
goto v___jp_1348_;
}
else
{
if (v_inv_1354_ == 0)
{
if (v_inv_1356_ == 0)
{
v___y_1349_ = v___x_1357_;
goto v___jp_1348_;
}
else
{
goto v___jp_1344_;
}
}
else
{
v___y_1349_ = v_inv_1356_;
goto v___jp_1348_;
}
}
}
else
{
goto v___jp_1344_;
}
}
else
{
if (lean_obj_tag(v_k_x27_1352_) == 0)
{
goto v___jp_1344_;
}
else
{
lean_object* v___x_1358_; lean_object* v___x_1359_; uint8_t v___x_1360_; 
v___x_1358_ = l_Lean_Meta_Origin_key(v_k_1343_);
v___x_1359_ = l_Lean_Meta_Origin_key(v_k_x27_1352_);
v___x_1360_ = lean_name_eq(v___x_1358_, v___x_1359_);
lean_dec(v___x_1359_);
lean_dec(v___x_1358_);
v___y_1349_ = v___x_1360_;
goto v___jp_1348_;
}
}
}
v___jp_1344_:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; 
v___x_1345_ = lean_unsigned_to_nat(1u);
v___x_1346_ = lean_nat_add(v_i_1342_, v___x_1345_);
lean_dec(v_i_1342_);
v_i_1342_ = v___x_1346_;
goto _start;
}
v___jp_1348_:
{
if (v___y_1349_ == 0)
{
goto v___jp_1344_;
}
else
{
lean_dec(v_i_1342_);
return v___y_1349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_keys_1361_, lean_object* v_i_1362_, lean_object* v_k_1363_){
_start:
{
uint8_t v_res_1364_; lean_object* v_r_1365_; 
v_res_1364_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg(v_keys_1361_, v_i_1362_, v_k_1363_);
lean_dec_ref(v_k_1363_);
lean_dec_ref(v_keys_1361_);
v_r_1365_ = lean_box(v_res_1364_);
return v_r_1365_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg(lean_object* v_x_1366_, size_t v_x_1367_, lean_object* v_x_1368_){
_start:
{
if (lean_obj_tag(v_x_1366_) == 0)
{
lean_object* v_es_1369_; lean_object* v___x_1370_; size_t v___x_1371_; size_t v___x_1372_; lean_object* v_j_1373_; lean_object* v___x_1374_; 
v_es_1369_ = lean_ctor_get(v_x_1366_, 0);
v___x_1370_ = lean_box(2);
v___x_1371_ = ((size_t)31ULL);
v___x_1372_ = lean_usize_land(v_x_1367_, v___x_1371_);
v_j_1373_ = lean_usize_to_nat(v___x_1372_);
v___x_1374_ = lean_array_get_borrowed(v___x_1370_, v_es_1369_, v_j_1373_);
lean_dec(v_j_1373_);
switch(lean_obj_tag(v___x_1374_))
{
case 0:
{
if (lean_obj_tag(v_x_1368_) == 0)
{
lean_object* v_key_1375_; 
v_key_1375_ = lean_ctor_get(v___x_1374_, 0);
if (lean_obj_tag(v_key_1375_) == 0)
{
lean_object* v_declName_1376_; uint8_t v_inv_1377_; lean_object* v_declName_1378_; uint8_t v_inv_1379_; uint8_t v___x_1380_; 
v_declName_1376_ = lean_ctor_get(v_x_1368_, 0);
v_inv_1377_ = lean_ctor_get_uint8(v_x_1368_, sizeof(void*)*1 + 1);
v_declName_1378_ = lean_ctor_get(v_key_1375_, 0);
v_inv_1379_ = lean_ctor_get_uint8(v_key_1375_, sizeof(void*)*1 + 1);
v___x_1380_ = lean_name_eq(v_declName_1376_, v_declName_1378_);
if (v___x_1380_ == 0)
{
return v___x_1380_;
}
else
{
if (v_inv_1377_ == 0)
{
if (v_inv_1379_ == 0)
{
return v___x_1380_;
}
else
{
return v_inv_1377_;
}
}
else
{
return v_inv_1379_;
}
}
}
else
{
uint8_t v___x_1381_; 
v___x_1381_ = 0;
return v___x_1381_;
}
}
else
{
lean_object* v_key_1382_; 
v_key_1382_ = lean_ctor_get(v___x_1374_, 0);
if (lean_obj_tag(v_key_1382_) == 0)
{
uint8_t v___x_1383_; 
v___x_1383_ = 0;
return v___x_1383_;
}
else
{
lean_object* v___x_1384_; lean_object* v___x_1385_; uint8_t v___x_1386_; 
v___x_1384_ = l_Lean_Meta_Origin_key(v_x_1368_);
v___x_1385_ = l_Lean_Meta_Origin_key(v_key_1382_);
v___x_1386_ = lean_name_eq(v___x_1384_, v___x_1385_);
lean_dec(v___x_1385_);
lean_dec(v___x_1384_);
return v___x_1386_;
}
}
}
case 1:
{
lean_object* v_node_1387_; size_t v___x_1388_; size_t v___x_1389_; 
v_node_1387_ = lean_ctor_get(v___x_1374_, 0);
v___x_1388_ = ((size_t)5ULL);
v___x_1389_ = lean_usize_shift_right(v_x_1367_, v___x_1388_);
v_x_1366_ = v_node_1387_;
v_x_1367_ = v___x_1389_;
goto _start;
}
default: 
{
uint8_t v___x_1391_; 
v___x_1391_ = 0;
return v___x_1391_;
}
}
}
else
{
lean_object* v_ks_1392_; lean_object* v___x_1393_; uint8_t v___x_1394_; 
v_ks_1392_ = lean_ctor_get(v_x_1366_, 0);
v___x_1393_ = lean_unsigned_to_nat(0u);
v___x_1394_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg(v_ks_1392_, v___x_1393_, v_x_1368_);
return v___x_1394_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_1395_, lean_object* v_x_1396_, lean_object* v_x_1397_){
_start:
{
size_t v_x_1682__boxed_1398_; uint8_t v_res_1399_; lean_object* v_r_1400_; 
v_x_1682__boxed_1398_ = lean_unbox_usize(v_x_1396_);
lean_dec(v_x_1396_);
v_res_1399_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg(v_x_1395_, v_x_1682__boxed_1398_, v_x_1397_);
lean_dec_ref(v_x_1397_);
lean_dec_ref(v_x_1395_);
v_r_1400_ = lean_box(v_res_1399_);
return v_r_1400_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(lean_object* v_x_1401_, lean_object* v_x_1402_){
_start:
{
uint64_t v___y_1404_; uint64_t v___y_1408_; uint64_t v___y_1412_; 
if (lean_obj_tag(v_x_1402_) == 0)
{
uint8_t v_inv_1415_; 
v_inv_1415_ = lean_ctor_get_uint8(v_x_1402_, sizeof(void*)*1 + 1);
if (v_inv_1415_ == 0)
{
lean_object* v_declName_1416_; 
v_declName_1416_ = lean_ctor_get(v_x_1402_, 0);
if (lean_obj_tag(v_declName_1416_) == 0)
{
uint64_t v___x_1417_; 
v___x_1417_ = 1723ULL;
v___y_1408_ = v___x_1417_;
goto v___jp_1407_;
}
else
{
uint64_t v_hash_1418_; 
v_hash_1418_ = lean_ctor_get_uint64(v_declName_1416_, sizeof(void*)*2);
v___y_1408_ = v_hash_1418_;
goto v___jp_1407_;
}
}
else
{
lean_object* v_declName_1419_; 
v_declName_1419_ = lean_ctor_get(v_x_1402_, 0);
if (lean_obj_tag(v_declName_1419_) == 0)
{
uint64_t v___x_1420_; 
v___x_1420_ = 1723ULL;
v___y_1412_ = v___x_1420_;
goto v___jp_1411_;
}
else
{
uint64_t v_hash_1421_; 
v_hash_1421_ = lean_ctor_get_uint64(v_declName_1419_, sizeof(void*)*2);
v___y_1412_ = v_hash_1421_;
goto v___jp_1411_;
}
}
}
else
{
lean_object* v___x_1422_; 
v___x_1422_ = l_Lean_Meta_Origin_key(v_x_1402_);
if (lean_obj_tag(v___x_1422_) == 0)
{
uint64_t v___x_1423_; 
v___x_1423_ = 1723ULL;
v___y_1404_ = v___x_1423_;
goto v___jp_1403_;
}
else
{
uint64_t v_hash_1424_; 
v_hash_1424_ = lean_ctor_get_uint64(v___x_1422_, sizeof(void*)*2);
lean_dec(v___x_1422_);
v___y_1404_ = v_hash_1424_;
goto v___jp_1403_;
}
}
v___jp_1403_:
{
size_t v___x_1405_; uint8_t v___x_1406_; 
v___x_1405_ = lean_uint64_to_usize(v___y_1404_);
v___x_1406_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg(v_x_1401_, v___x_1405_, v_x_1402_);
return v___x_1406_;
}
v___jp_1407_:
{
uint64_t v___x_1409_; uint64_t v___x_1410_; 
v___x_1409_ = 13ULL;
v___x_1410_ = lean_uint64_mix_hash(v___y_1408_, v___x_1409_);
v___y_1404_ = v___x_1410_;
goto v___jp_1403_;
}
v___jp_1411_:
{
uint64_t v___x_1413_; uint64_t v___x_1414_; 
v___x_1413_ = 11ULL;
v___x_1414_ = lean_uint64_mix_hash(v___y_1412_, v___x_1413_);
v___y_1404_ = v___x_1414_;
goto v___jp_1403_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg___boxed(lean_object* v_x_1425_, lean_object* v_x_1426_){
_start:
{
uint8_t v_res_1427_; lean_object* v_r_1428_; 
v_res_1427_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(v_x_1425_, v_x_1426_);
lean_dec_ref(v_x_1426_);
lean_dec_ref(v_x_1425_);
v_r_1428_ = lean_box(v_res_1427_);
return v_r_1428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2(lean_object* v_erased_1429_, lean_object* v_f_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_){
_start:
{
lean_object* v_origin_1433_; uint8_t v___x_1434_; 
v_origin_1433_ = lean_ctor_get(v___y_1432_, 4);
v___x_1434_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(v_erased_1429_, v_origin_1433_);
if (v___x_1434_ == 0)
{
lean_object* v___x_1435_; lean_object* v___x_1436_; 
v___x_1435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1435_, 0, v___y_1432_);
v___x_1436_ = lean_apply_2(v_f_1430_, v___y_1431_, v___x_1435_);
return v___x_1436_;
}
else
{
lean_dec_ref(v___y_1432_);
lean_dec(v_f_1430_);
return v___y_1431_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2___boxed(lean_object* v_erased_1437_, lean_object* v_f_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_){
_start:
{
lean_object* v_res_1441_; 
v_res_1441_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2(v_erased_1437_, v_f_1438_, v___y_1439_, v___y_1440_);
lean_dec_ref(v_erased_1437_);
return v_res_1441_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg(lean_object* v_f_1442_, lean_object* v_init_1443_, lean_object* v_thms_1444_){
_start:
{
lean_object* v_pre_1445_; lean_object* v_post_1446_; lean_object* v_toUnfold_1447_; lean_object* v_erased_1448_; lean_object* v_toUnfoldThms_1449_; lean_object* v___f_1450_; lean_object* v___f_1451_; lean_object* v___f_1452_; lean_object* v___f_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; 
v_pre_1445_ = lean_ctor_get(v_thms_1444_, 0);
lean_inc_ref(v_pre_1445_);
v_post_1446_ = lean_ctor_get(v_thms_1444_, 1);
lean_inc_ref(v_post_1446_);
v_toUnfold_1447_ = lean_ctor_get(v_thms_1444_, 3);
lean_inc_ref(v_toUnfold_1447_);
v_erased_1448_ = lean_ctor_get(v_thms_1444_, 4);
lean_inc_ref(v_erased_1448_);
v_toUnfoldThms_1449_ = lean_ctor_get(v_thms_1444_, 5);
lean_inc_ref(v_toUnfoldThms_1449_);
lean_dec_ref(v_thms_1444_);
lean_inc_n(v_f_1442_, 2);
v___f_1450_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__2), 4, 1);
lean_closure_set(v___f_1450_, 0, v_f_1442_);
v___f_1451_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1451_, 0, v_f_1442_);
v___f_1452_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_1452_, 0, v_erased_1448_);
lean_closure_set(v___f_1452_, 1, v_f_1442_);
v___f_1453_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1453_, 0, v___f_1452_);
lean_inc_ref(v___f_1453_);
v___x_1454_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1453_, v_pre_1445_, v_init_1443_);
lean_dec_ref(v_pre_1445_);
v___x_1455_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1453_, v_post_1446_, v___x_1454_);
lean_dec_ref(v_post_1446_);
v___x_1456_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1450_, v_toUnfold_1447_, v___x_1455_);
lean_dec_ref(v_toUnfold_1447_);
v___x_1457_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1451_, v_toUnfoldThms_1449_, v___x_1456_);
lean_dec_ref(v_toUnfoldThms_1449_);
return v___x_1457_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntries___redArg(lean_object* v_f_1458_, lean_object* v_init_1459_, lean_object* v_thms_1460_){
_start:
{
lean_object* v___x_1461_; 
v___x_1461_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg(v_f_1458_, v_init_1459_, v_thms_1460_);
return v___x_1461_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntries(lean_object* v_00_u03c3_1462_, lean_object* v_f_1463_, lean_object* v_init_1464_, lean_object* v_thms_1465_){
_start:
{
lean_object* v___x_1466_; 
v___x_1466_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg(v_f_1463_, v_init_1464_, v_thms_1465_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0(lean_object* v_00_u03c3_1467_, lean_object* v_f_1468_, lean_object* v_init_1469_, lean_object* v_thms_1470_){
_start:
{
lean_object* v___x_1471_; 
v___x_1471_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___redArg(v_f_1468_, v_init_1469_, v_thms_1470_);
return v___x_1471_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0(lean_object* v_00_u03b2_1472_, lean_object* v_x_1473_, lean_object* v_x_1474_){
_start:
{
uint8_t v___x_1475_; 
v___x_1475_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(v_x_1473_, v_x_1474_);
return v___x_1475_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1476_, lean_object* v_x_1477_, lean_object* v_x_1478_){
_start:
{
uint8_t v_res_1479_; lean_object* v_r_1480_; 
v_res_1479_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0(v_00_u03b2_1476_, v_x_1477_, v_x_1478_);
lean_dec_ref(v_x_1478_);
lean_dec_ref(v_x_1477_);
v_r_1480_ = lean_box(v_res_1479_);
return v_r_1480_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1(lean_object* v_00_u03c3_1481_, lean_object* v_00_u03b1_1482_, lean_object* v_f_1483_, lean_object* v_x_1484_, lean_object* v_x_1485_){
_start:
{
lean_object* v___x_1486_; 
v___x_1486_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(v_f_1483_, v_x_1484_, v_x_1485_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___boxed(lean_object* v_00_u03c3_1487_, lean_object* v_00_u03b1_1488_, lean_object* v_f_1489_, lean_object* v_x_1490_, lean_object* v_x_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1(v_00_u03c3_1487_, v_00_u03b1_1488_, v_f_1489_, v_x_1490_, v_x_1491_);
lean_dec_ref(v_x_1491_);
return v_res_1492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___redArg(lean_object* v_map_1493_, lean_object* v_f_1494_, lean_object* v_init_1495_){
_start:
{
lean_object* v___x_1496_; 
v___x_1496_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1494_, v_map_1493_, v_init_1495_);
return v___x_1496_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___redArg___boxed(lean_object* v_map_1497_, lean_object* v_f_1498_, lean_object* v_init_1499_){
_start:
{
lean_object* v_res_1500_; 
v_res_1500_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___redArg(v_map_1497_, v_f_1498_, v_init_1499_);
lean_dec_ref(v_map_1497_);
return v_res_1500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2(lean_object* v_00_u03c3_1501_, lean_object* v_00_u03b2_1502_, lean_object* v_map_1503_, lean_object* v_f_1504_, lean_object* v_init_1505_){
_start:
{
lean_object* v___x_1506_; 
v___x_1506_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1504_, v_map_1503_, v_init_1505_);
return v___x_1506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2___boxed(lean_object* v_00_u03c3_1507_, lean_object* v_00_u03b2_1508_, lean_object* v_map_1509_, lean_object* v_f_1510_, lean_object* v_init_1511_){
_start:
{
lean_object* v_res_1512_; 
v_res_1512_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2(v_00_u03c3_1507_, v_00_u03b2_1508_, v_map_1509_, v_f_1510_, v_init_1511_);
lean_dec_ref(v_map_1509_);
return v_res_1512_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___redArg(lean_object* v_map_1513_, lean_object* v_f_1514_, lean_object* v_init_1515_){
_start:
{
lean_object* v___x_1516_; 
v___x_1516_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1514_, v_map_1513_, v_init_1515_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___redArg___boxed(lean_object* v_map_1517_, lean_object* v_f_1518_, lean_object* v_init_1519_){
_start:
{
lean_object* v_res_1520_; 
v_res_1520_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___redArg(v_map_1517_, v_f_1518_, v_init_1519_);
lean_dec_ref(v_map_1517_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3(lean_object* v_00_u03c3_1521_, lean_object* v_00_u03b2_1522_, lean_object* v_map_1523_, lean_object* v_f_1524_, lean_object* v_init_1525_){
_start:
{
lean_object* v___x_1526_; 
v___x_1526_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1524_, v_map_1523_, v_init_1525_);
return v___x_1526_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3___boxed(lean_object* v_00_u03c3_1527_, lean_object* v_00_u03b2_1528_, lean_object* v_map_1529_, lean_object* v_f_1530_, lean_object* v_init_1531_){
_start:
{
lean_object* v_res_1532_; 
v_res_1532_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__3(v_00_u03c3_1527_, v_00_u03b2_1528_, v_map_1529_, v_f_1530_, v_init_1531_);
lean_dec_ref(v_map_1529_);
return v_res_1532_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1533_, lean_object* v_x_1534_, size_t v_x_1535_, lean_object* v_x_1536_){
_start:
{
uint8_t v___x_1537_; 
v___x_1537_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___redArg(v_x_1534_, v_x_1535_, v_x_1536_);
return v___x_1537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1538_, lean_object* v_x_1539_, lean_object* v_x_1540_, lean_object* v_x_1541_){
_start:
{
size_t v_x_1848__boxed_1542_; uint8_t v_res_1543_; lean_object* v_r_1544_; 
v_x_1848__boxed_1542_ = lean_unbox_usize(v_x_1540_);
lean_dec(v_x_1540_);
v_res_1543_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1(v_00_u03b2_1538_, v_x_1539_, v_x_1848__boxed_1542_, v_x_1541_);
lean_dec_ref(v_x_1541_);
lean_dec_ref(v_x_1539_);
v_r_1544_ = lean_box(v_res_1543_);
return v_r_1544_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_1545_, lean_object* v_00_u03c3_1546_, lean_object* v_f_1547_, lean_object* v_as_1548_, size_t v_i_1549_, size_t v_stop_1550_, lean_object* v_b_1551_){
_start:
{
lean_object* v___x_1552_; 
v___x_1552_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___redArg(v_f_1547_, v_as_1548_, v_i_1549_, v_stop_1550_, v_b_1551_);
return v___x_1552_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1553_, lean_object* v_00_u03c3_1554_, lean_object* v_f_1555_, lean_object* v_as_1556_, lean_object* v_i_1557_, lean_object* v_stop_1558_, lean_object* v_b_1559_){
_start:
{
size_t v_i_boxed_1560_; size_t v_stop_boxed_1561_; lean_object* v_res_1562_; 
v_i_boxed_1560_ = lean_unbox_usize(v_i_1557_);
lean_dec(v_i_1557_);
v_stop_boxed_1561_ = lean_unbox_usize(v_stop_1558_);
lean_dec(v_stop_1558_);
v_res_1562_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__3(v_00_u03b1_1553_, v_00_u03c3_1554_, v_f_1555_, v_as_1556_, v_i_boxed_1560_, v_stop_boxed_1561_, v_b_1559_);
lean_dec_ref(v_as_1556_);
return v_res_1562_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4(lean_object* v_00_u03b1_1563_, lean_object* v_00_u03c3_1564_, lean_object* v_f_1565_, lean_object* v_as_1566_, size_t v_i_1567_, size_t v_stop_1568_, lean_object* v_b_1569_){
_start:
{
lean_object* v___x_1570_; 
v___x_1570_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___redArg(v_f_1565_, v_as_1566_, v_i_1567_, v_stop_1568_, v_b_1569_);
return v___x_1570_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1571_, lean_object* v_00_u03c3_1572_, lean_object* v_f_1573_, lean_object* v_as_1574_, lean_object* v_i_1575_, lean_object* v_stop_1576_, lean_object* v_b_1577_){
_start:
{
size_t v_i_boxed_1578_; size_t v_stop_boxed_1579_; lean_object* v_res_1580_; 
v_i_boxed_1578_ = lean_unbox_usize(v_i_1575_);
lean_dec(v_i_1575_);
v_stop_boxed_1579_ = lean_unbox_usize(v_stop_1576_);
lean_dec(v_stop_1576_);
v_res_1580_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1_spec__4(v_00_u03b1_1571_, v_00_u03c3_1572_, v_f_1573_, v_as_1574_, v_i_boxed_1578_, v_stop_boxed_1579_, v_b_1577_);
lean_dec_ref(v_as_1574_);
return v_res_1580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6(lean_object* v_00_u03c3_1581_, lean_object* v_00_u03b1_1582_, lean_object* v_00_u03b2_1583_, lean_object* v_f_1584_, lean_object* v_x_1585_, lean_object* v_x_1586_){
_start:
{
lean_object* v___x_1587_; 
v___x_1587_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v_f_1584_, v_x_1585_, v_x_1586_);
return v___x_1587_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___boxed(lean_object* v_00_u03c3_1588_, lean_object* v_00_u03b1_1589_, lean_object* v_00_u03b2_1590_, lean_object* v_f_1591_, lean_object* v_x_1592_, lean_object* v_x_1593_){
_start:
{
lean_object* v_res_1594_; 
v_res_1594_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6(v_00_u03c3_1588_, v_00_u03b1_1589_, v_00_u03b2_1590_, v_f_1591_, v_x_1592_, v_x_1593_);
lean_dec_ref(v_x_1592_);
return v_res_1594_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1595_, lean_object* v_keys_1596_, lean_object* v_vals_1597_, lean_object* v_heq_1598_, lean_object* v_i_1599_, lean_object* v_k_1600_){
_start:
{
uint8_t v___x_1601_; 
v___x_1601_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___redArg(v_keys_1596_, v_i_1599_, v_k_1600_);
return v___x_1601_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1602_, lean_object* v_keys_1603_, lean_object* v_vals_1604_, lean_object* v_heq_1605_, lean_object* v_i_1606_, lean_object* v_k_1607_){
_start:
{
uint8_t v_res_1608_; lean_object* v_r_1609_; 
v_res_1608_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0_spec__1_spec__2(v_00_u03b2_1602_, v_keys_1603_, v_vals_1604_, v_heq_1605_, v_i_1606_, v_k_1607_);
lean_dec_ref(v_k_1607_);
lean_dec_ref(v_vals_1604_);
lean_dec_ref(v_keys_1603_);
v_r_1609_ = lean_box(v_res_1608_);
return v_r_1609_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8(lean_object* v_00_u03b1_1610_, lean_object* v_00_u03b2_1611_, lean_object* v_00_u03c3_1612_, lean_object* v_f_1613_, lean_object* v_as_1614_, size_t v_i_1615_, size_t v_stop_1616_, lean_object* v_b_1617_){
_start:
{
lean_object* v___x_1618_; 
v___x_1618_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___redArg(v_f_1613_, v_as_1614_, v_i_1615_, v_stop_1616_, v_b_1617_);
return v___x_1618_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8___boxed(lean_object* v_00_u03b1_1619_, lean_object* v_00_u03b2_1620_, lean_object* v_00_u03c3_1621_, lean_object* v_f_1622_, lean_object* v_as_1623_, lean_object* v_i_1624_, lean_object* v_stop_1625_, lean_object* v_b_1626_){
_start:
{
size_t v_i_boxed_1627_; size_t v_stop_boxed_1628_; lean_object* v_res_1629_; 
v_i_boxed_1627_ = lean_unbox_usize(v_i_1624_);
lean_dec(v_i_1624_);
v_stop_boxed_1628_ = lean_unbox_usize(v_stop_1625_);
lean_dec(v_stop_1625_);
v_res_1629_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__8(v_00_u03b1_1619_, v_00_u03b2_1620_, v_00_u03c3_1621_, v_f_1622_, v_as_1623_, v_i_boxed_1627_, v_stop_boxed_1628_, v_b_1626_);
lean_dec_ref(v_as_1623_);
return v_res_1629_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9(lean_object* v_00_u03c3_1630_, lean_object* v_00_u03b1_1631_, lean_object* v_00_u03b2_1632_, lean_object* v_f_1633_, lean_object* v_keys_1634_, lean_object* v_vals_1635_, lean_object* v_heq_1636_, lean_object* v_i_1637_, lean_object* v_acc_1638_){
_start:
{
lean_object* v___x_1639_; 
v___x_1639_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___redArg(v_f_1633_, v_keys_1634_, v_vals_1635_, v_i_1637_, v_acc_1638_);
return v___x_1639_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9___boxed(lean_object* v_00_u03c3_1640_, lean_object* v_00_u03b1_1641_, lean_object* v_00_u03b2_1642_, lean_object* v_f_1643_, lean_object* v_keys_1644_, lean_object* v_vals_1645_, lean_object* v_heq_1646_, lean_object* v_i_1647_, lean_object* v_acc_1648_){
_start:
{
lean_object* v_res_1649_; 
v_res_1649_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6_spec__9(v_00_u03c3_1640_, v_00_u03b1_1641_, v_00_u03b2_1642_, v_f_1643_, v_keys_1644_, v_vals_1645_, v_heq_1646_, v_i_1647_, v_acc_1648_);
lean_dec_ref(v_vals_1645_);
lean_dec_ref(v_keys_1644_);
return v_res_1649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__0(lean_object* v_d_1650_, lean_object* v_a_1651_, lean_object* v_x_1652_){
_start:
{
lean_object* v___x_1653_; lean_object* v___x_1654_; 
v___x_1653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1653_, 0, v_a_1651_);
v___x_1654_ = lean_array_push(v_d_1650_, v___x_1653_);
return v___x_1654_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__1(lean_object* v_s_1655_, lean_object* v_n_1656_, lean_object* v_thms_1657_){
_start:
{
lean_object* v___x_1658_; lean_object* v___x_1659_; 
v___x_1658_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1658_, 0, v_n_1656_);
lean_ctor_set(v___x_1658_, 1, v_thms_1657_);
v___x_1659_ = lean_array_push(v_s_1655_, v___x_1658_);
return v___x_1659_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2(lean_object* v_erased_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_){
_start:
{
lean_object* v_origin_1663_; uint8_t v___x_1664_; 
v_origin_1663_ = lean_ctor_get(v___y_1662_, 4);
v___x_1664_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__0___redArg(v_erased_1660_, v_origin_1663_);
if (v___x_1664_ == 0)
{
lean_object* v___x_1665_; lean_object* v___x_1666_; 
v___x_1665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1665_, 0, v___y_1662_);
v___x_1666_ = lean_array_push(v___y_1661_, v___x_1665_);
return v___x_1666_;
}
else
{
lean_dec_ref(v___y_1662_);
return v___y_1661_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2___boxed(lean_object* v_erased_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_){
_start:
{
lean_object* v_res_1670_; 
v_res_1670_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2(v_erased_1667_, v___y_1668_, v___y_1669_);
lean_dec_ref(v_erased_1667_);
return v_res_1670_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3(lean_object* v___f_1671_, lean_object* v_s_1672_, lean_object* v_x_1673_, lean_object* v_t_1674_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__1___redArg(v___f_1671_, v_s_1672_, v_t_1674_);
return v___x_1675_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3___boxed(lean_object* v___f_1676_, lean_object* v_s_1677_, lean_object* v_x_1678_, lean_object* v_t_1679_){
_start:
{
lean_object* v_res_1680_; 
v_res_1680_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3(v___f_1676_, v_s_1677_, v_x_1678_, v_t_1679_);
lean_dec_ref(v_t_1679_);
lean_dec(v_x_1678_);
return v_res_1680_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0(lean_object* v_init_1683_, lean_object* v_thms_1684_){
_start:
{
lean_object* v_pre_1685_; lean_object* v_post_1686_; lean_object* v_toUnfold_1687_; lean_object* v_erased_1688_; lean_object* v_toUnfoldThms_1689_; lean_object* v___f_1690_; lean_object* v___f_1691_; lean_object* v___f_1692_; lean_object* v___f_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; 
v_pre_1685_ = lean_ctor_get(v_thms_1684_, 0);
lean_inc_ref(v_pre_1685_);
v_post_1686_ = lean_ctor_get(v_thms_1684_, 1);
lean_inc_ref(v_post_1686_);
v_toUnfold_1687_ = lean_ctor_get(v_thms_1684_, 3);
lean_inc_ref(v_toUnfold_1687_);
v_erased_1688_ = lean_ctor_get(v_thms_1684_, 4);
lean_inc_ref(v_erased_1688_);
v_toUnfoldThms_1689_ = lean_ctor_get(v_thms_1684_, 5);
lean_inc_ref(v_toUnfoldThms_1689_);
lean_dec_ref(v_thms_1684_);
v___f_1690_ = ((lean_object*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__0));
v___f_1691_ = ((lean_object*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___closed__1));
v___f_1692_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__2___boxed), 3, 1);
lean_closure_set(v___f_1692_, 0, v_erased_1688_);
v___f_1693_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0___lam__3___boxed), 4, 1);
lean_closure_set(v___f_1693_, 0, v___f_1692_);
lean_inc_ref(v___f_1693_);
v___x_1694_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1693_, v_pre_1685_, v_init_1683_);
lean_dec_ref(v_pre_1685_);
v___x_1695_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1693_, v_post_1686_, v___x_1694_);
lean_dec_ref(v_post_1686_);
v___x_1696_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1690_, v_toUnfold_1687_, v___x_1695_);
lean_dec_ref(v_toUnfold_1687_);
v___x_1697_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0_spec__2_spec__6___redArg(v___f_1691_, v_toUnfoldThms_1689_, v___x_1696_);
lean_dec_ref(v_toUnfoldThms_1689_);
return v___x_1697_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_simpEntries(lean_object* v_thms_1700_){
_start:
{
lean_object* v___x_1701_; lean_object* v___x_1702_; 
v___x_1701_ = ((lean_object*)(lp_aesop_Aesop_SimpTheorems_simpEntries___closed__0));
v___x_1702_ = lp_aesop_Aesop_SimpTheorems_foldSimpEntriesM___at___00Aesop_SimpTheorems_foldSimpEntries_spec__0___at___00Aesop_SimpTheorems_simpEntries_spec__0(v___x_1701_, v_thms_1700_);
return v___x_1702_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_1703_, lean_object* v_i_1704_, lean_object* v_k_1705_){
_start:
{
lean_object* v___x_1706_; uint8_t v___x_1707_; 
v___x_1706_ = lean_array_get_size(v_keys_1703_);
v___x_1707_ = lean_nat_dec_lt(v_i_1704_, v___x_1706_);
if (v___x_1707_ == 0)
{
lean_dec(v_i_1704_);
return v___x_1707_;
}
else
{
lean_object* v_k_x27_1708_; uint8_t v___x_1709_; 
v_k_x27_1708_ = lean_array_fget_borrowed(v_keys_1703_, v_i_1704_);
v___x_1709_ = lean_name_eq(v_k_1705_, v_k_x27_1708_);
if (v___x_1709_ == 0)
{
lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1710_ = lean_unsigned_to_nat(1u);
v___x_1711_ = lean_nat_add(v_i_1704_, v___x_1710_);
lean_dec(v_i_1704_);
v_i_1704_ = v___x_1711_;
goto _start;
}
else
{
lean_dec(v_i_1704_);
return v___x_1709_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_1713_, lean_object* v_i_1714_, lean_object* v_k_1715_){
_start:
{
uint8_t v_res_1716_; lean_object* v_r_1717_; 
v_res_1716_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg(v_keys_1713_, v_i_1714_, v_k_1715_);
lean_dec(v_k_1715_);
lean_dec_ref(v_keys_1713_);
v_r_1717_ = lean_box(v_res_1716_);
return v_r_1717_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg(lean_object* v_x_1718_, size_t v_x_1719_, lean_object* v_x_1720_){
_start:
{
if (lean_obj_tag(v_x_1718_) == 0)
{
lean_object* v_es_1721_; lean_object* v___x_1722_; size_t v___x_1723_; size_t v___x_1724_; lean_object* v_j_1725_; lean_object* v___x_1726_; 
v_es_1721_ = lean_ctor_get(v_x_1718_, 0);
v___x_1722_ = lean_box(2);
v___x_1723_ = ((size_t)31ULL);
v___x_1724_ = lean_usize_land(v_x_1719_, v___x_1723_);
v_j_1725_ = lean_usize_to_nat(v___x_1724_);
v___x_1726_ = lean_array_get_borrowed(v___x_1722_, v_es_1721_, v_j_1725_);
lean_dec(v_j_1725_);
switch(lean_obj_tag(v___x_1726_))
{
case 0:
{
lean_object* v_key_1727_; uint8_t v___x_1728_; 
v_key_1727_ = lean_ctor_get(v___x_1726_, 0);
v___x_1728_ = lean_name_eq(v_x_1720_, v_key_1727_);
return v___x_1728_;
}
case 1:
{
lean_object* v_node_1729_; size_t v___x_1730_; size_t v___x_1731_; 
v_node_1729_ = lean_ctor_get(v___x_1726_, 0);
v___x_1730_ = ((size_t)5ULL);
v___x_1731_ = lean_usize_shift_right(v_x_1719_, v___x_1730_);
v_x_1718_ = v_node_1729_;
v_x_1719_ = v___x_1731_;
goto _start;
}
default: 
{
uint8_t v___x_1733_; 
v___x_1733_ = 0;
return v___x_1733_;
}
}
}
else
{
lean_object* v_ks_1734_; lean_object* v___x_1735_; uint8_t v___x_1736_; 
v_ks_1734_ = lean_ctor_get(v_x_1718_, 0);
v___x_1735_ = lean_unsigned_to_nat(0u);
v___x_1736_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg(v_ks_1734_, v___x_1735_, v_x_1720_);
return v___x_1736_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg___boxed(lean_object* v_x_1737_, lean_object* v_x_1738_, lean_object* v_x_1739_){
_start:
{
size_t v_x_163__boxed_1740_; uint8_t v_res_1741_; lean_object* v_r_1742_; 
v_x_163__boxed_1740_ = lean_unbox_usize(v_x_1738_);
lean_dec(v_x_1738_);
v_res_1741_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg(v_x_1737_, v_x_163__boxed_1740_, v_x_1739_);
lean_dec(v_x_1739_);
lean_dec_ref(v_x_1737_);
v_r_1742_ = lean_box(v_res_1741_);
return v_r_1742_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg(lean_object* v_x_1743_, lean_object* v_x_1744_){
_start:
{
uint64_t v___y_1746_; 
if (lean_obj_tag(v_x_1744_) == 0)
{
uint64_t v___x_1749_; 
v___x_1749_ = 1723ULL;
v___y_1746_ = v___x_1749_;
goto v___jp_1745_;
}
else
{
uint64_t v_hash_1750_; 
v_hash_1750_ = lean_ctor_get_uint64(v_x_1744_, sizeof(void*)*2);
v___y_1746_ = v_hash_1750_;
goto v___jp_1745_;
}
v___jp_1745_:
{
size_t v___x_1747_; uint8_t v___x_1748_; 
v___x_1747_ = lean_uint64_to_usize(v___y_1746_);
v___x_1748_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg(v_x_1743_, v___x_1747_, v_x_1744_);
return v___x_1748_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg___boxed(lean_object* v_x_1751_, lean_object* v_x_1752_){
_start:
{
uint8_t v_res_1753_; lean_object* v_r_1754_; 
v_res_1753_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg(v_x_1751_, v_x_1752_);
lean_dec(v_x_1752_);
lean_dec_ref(v_x_1751_);
v_r_1754_ = lean_box(v_res_1753_);
return v_r_1754_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_SimpTheorems_containsDecl(lean_object* v_thms_1755_, lean_object* v_decl_1756_){
_start:
{
uint8_t v___y_1758_; uint8_t v___x_1761_; uint8_t v___x_1762_; lean_object* v___x_1763_; uint8_t v___x_1764_; 
v___x_1761_ = 1;
v___x_1762_ = 0;
lean_inc(v_decl_1756_);
v___x_1763_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1763_, 0, v_decl_1756_);
lean_ctor_set_uint8(v___x_1763_, sizeof(void*)*1, v___x_1761_);
lean_ctor_set_uint8(v___x_1763_, sizeof(void*)*1 + 1, v___x_1762_);
v___x_1764_ = l_Lean_Meta_SimpTheorems_isLemma(v_thms_1755_, v___x_1763_);
lean_dec_ref_known(v___x_1763_, 1);
if (v___x_1764_ == 0)
{
uint8_t v___x_1765_; 
v___x_1765_ = l_Lean_Meta_SimpTheorems_isDeclToUnfold(v_thms_1755_, v_decl_1756_);
v___y_1758_ = v___x_1765_;
goto v___jp_1757_;
}
else
{
v___y_1758_ = v___x_1764_;
goto v___jp_1757_;
}
v___jp_1757_:
{
if (v___y_1758_ == 0)
{
lean_object* v_toUnfoldThms_1759_; uint8_t v___x_1760_; 
v_toUnfoldThms_1759_ = lean_ctor_get(v_thms_1755_, 5);
v___x_1760_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg(v_toUnfoldThms_1759_, v_decl_1756_);
lean_dec(v_decl_1756_);
return v___x_1760_;
}
else
{
lean_dec(v_decl_1756_);
return v___y_1758_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpTheorems_containsDecl___boxed(lean_object* v_thms_1766_, lean_object* v_decl_1767_){
_start:
{
uint8_t v_res_1768_; lean_object* v_r_1769_; 
v_res_1768_ = lp_aesop_Aesop_SimpTheorems_containsDecl(v_thms_1766_, v_decl_1767_);
lean_dec_ref(v_thms_1766_);
v_r_1769_ = lean_box(v_res_1768_);
return v_r_1769_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0(lean_object* v_00_u03b2_1770_, lean_object* v_x_1771_, lean_object* v_x_1772_){
_start:
{
uint8_t v___x_1773_; 
v___x_1773_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___redArg(v_x_1771_, v_x_1772_);
return v___x_1773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0___boxed(lean_object* v_00_u03b2_1774_, lean_object* v_x_1775_, lean_object* v_x_1776_){
_start:
{
uint8_t v_res_1777_; lean_object* v_r_1778_; 
v_res_1777_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0(v_00_u03b2_1774_, v_x_1775_, v_x_1776_);
lean_dec(v_x_1776_);
lean_dec_ref(v_x_1775_);
v_r_1778_ = lean_box(v_res_1777_);
return v_r_1778_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0(lean_object* v_00_u03b2_1779_, lean_object* v_x_1780_, size_t v_x_1781_, lean_object* v_x_1782_){
_start:
{
uint8_t v___x_1783_; 
v___x_1783_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___redArg(v_x_1780_, v_x_1781_, v_x_1782_);
return v___x_1783_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1784_, lean_object* v_x_1785_, lean_object* v_x_1786_, lean_object* v_x_1787_){
_start:
{
size_t v_x_244__boxed_1788_; uint8_t v_res_1789_; lean_object* v_r_1790_; 
v_x_244__boxed_1788_ = lean_unbox_usize(v_x_1786_);
lean_dec(v_x_1786_);
v_res_1789_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0(v_00_u03b2_1784_, v_x_1785_, v_x_244__boxed_1788_, v_x_1787_);
lean_dec(v_x_1787_);
lean_dec_ref(v_x_1785_);
v_r_1790_ = lean_box(v_res_1789_);
return v_r_1790_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1791_, lean_object* v_keys_1792_, lean_object* v_vals_1793_, lean_object* v_heq_1794_, lean_object* v_i_1795_, lean_object* v_k_1796_){
_start:
{
uint8_t v___x_1797_; 
v___x_1797_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___redArg(v_keys_1792_, v_i_1795_, v_k_1796_);
return v___x_1797_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1798_, lean_object* v_keys_1799_, lean_object* v_vals_1800_, lean_object* v_heq_1801_, lean_object* v_i_1802_, lean_object* v_k_1803_){
_start:
{
uint8_t v_res_1804_; lean_object* v_r_1805_; 
v_res_1804_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Aesop_SimpTheorems_containsDecl_spec__0_spec__0_spec__1(v_00_u03b2_1798_, v_keys_1799_, v_vals_1800_, v_heq_1801_, v_i_1802_, v_k_1803_);
lean_dec(v_k_1803_);
lean_dec_ref(v_vals_1800_);
lean_dec_ref(v_keys_1799_);
v_r_1805_ = lean_box(v_res_1804_);
return v_r_1805_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0(lean_object* v_ldecl_1806_, uint8_t v___x_1807_, lean_object* v_inst_1808_, lean_object* v_inst_1809_, lean_object* v_g_1810_, lean_object* v_a_1811_, lean_object* v_toApplicative_1812_, lean_object* v_a_1813_){
_start:
{
lean_object* v___x_1814_; 
v___x_1814_ = l_Lean_LocalDecl_value_x3f(v_ldecl_1806_, v___x_1807_);
if (lean_obj_tag(v___x_1814_) == 1)
{
lean_object* v_val_1815_; lean_object* v___x_1816_; 
lean_dec_ref(v_toApplicative_1812_);
v_val_1815_ = lean_ctor_get(v___x_1814_, 0);
lean_inc(v_val_1815_);
lean_dec_ref_known(v___x_1814_, 1);
v___x_1816_ = l_Lean_ForEachExpr_visit___redArg(v_inst_1808_, v_inst_1809_, v_g_1810_, v_val_1815_, v_a_1811_);
return v___x_1816_;
}
else
{
lean_object* v_toPure_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; 
lean_dec(v___x_1814_);
lean_dec(v_g_1810_);
lean_dec_ref(v_inst_1809_);
lean_dec(v_inst_1808_);
v_toPure_1817_ = lean_ctor_get(v_toApplicative_1812_, 1);
lean_inc(v_toPure_1817_);
lean_dec_ref(v_toApplicative_1812_);
v___x_1818_ = lean_box(0);
v___x_1819_ = lean_apply_2(v_toPure_1817_, lean_box(0), v___x_1818_);
return v___x_1819_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0___boxed(lean_object* v_ldecl_1820_, lean_object* v___x_1821_, lean_object* v_inst_1822_, lean_object* v_inst_1823_, lean_object* v_g_1824_, lean_object* v_a_1825_, lean_object* v_toApplicative_1826_, lean_object* v_a_1827_){
_start:
{
uint8_t v___x_362__boxed_1828_; lean_object* v_res_1829_; 
v___x_362__boxed_1828_ = lean_unbox(v___x_1821_);
v_res_1829_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0(v_ldecl_1820_, v___x_362__boxed_1828_, v_inst_1822_, v_inst_1823_, v_g_1824_, v_a_1825_, v_toApplicative_1826_, v_a_1827_);
lean_dec(v_a_1825_);
lean_dec_ref(v_ldecl_1820_);
return v_res_1829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1(lean_object* v_ldecl_1830_, lean_object* v_inst_1831_, lean_object* v_inst_1832_, lean_object* v_g_1833_, lean_object* v_a_1834_, lean_object* v_toBind_1835_, lean_object* v___f_1836_, lean_object* v_a_1837_){
_start:
{
lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; 
v___x_1838_ = l_Lean_LocalDecl_type(v_ldecl_1830_);
v___x_1839_ = l_Lean_ForEachExpr_visit___redArg(v_inst_1831_, v_inst_1832_, v_g_1833_, v___x_1838_, v_a_1834_);
v___x_1840_ = lean_apply_4(v_toBind_1835_, lean_box(0), lean_box(0), v___x_1839_, v___f_1836_);
return v___x_1840_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1___boxed(lean_object* v_ldecl_1841_, lean_object* v_inst_1842_, lean_object* v_inst_1843_, lean_object* v_g_1844_, lean_object* v_a_1845_, lean_object* v_toBind_1846_, lean_object* v___f_1847_, lean_object* v_a_1848_){
_start:
{
lean_object* v_res_1849_; 
v_res_1849_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1(v_ldecl_1841_, v_inst_1842_, v_inst_1843_, v_g_1844_, v_a_1845_, v_toBind_1846_, v___f_1847_, v_a_1848_);
lean_dec(v_a_1845_);
lean_dec_ref(v_ldecl_1841_);
return v_res_1849_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg(lean_object* v_inst_1850_, lean_object* v_inst_1851_, lean_object* v_ldecl_1852_, lean_object* v_g_1853_, lean_object* v_a_1854_){
_start:
{
uint8_t v___x_1855_; 
v___x_1855_ = l_Lean_LocalDecl_isImplementationDetail(v_ldecl_1852_);
if (v___x_1855_ == 0)
{
lean_object* v_toApplicative_1856_; lean_object* v_toBind_1857_; lean_object* v___x_1858_; lean_object* v___f_1859_; lean_object* v___f_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; 
v_toApplicative_1856_ = lean_ctor_get(v_inst_1851_, 0);
v_toBind_1857_ = lean_ctor_get(v_inst_1851_, 1);
lean_inc_n(v_toBind_1857_, 2);
v___x_1858_ = lean_box(v___x_1855_);
lean_inc_ref(v_toApplicative_1856_);
lean_inc_n(v_a_1854_, 2);
lean_inc_n(v_g_1853_, 2);
lean_inc_ref_n(v_inst_1851_, 2);
lean_inc_n(v_inst_1850_, 2);
lean_inc_ref_n(v_ldecl_1852_, 2);
v___f_1859_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_1859_, 0, v_ldecl_1852_);
lean_closure_set(v___f_1859_, 1, v___x_1858_);
lean_closure_set(v___f_1859_, 2, v_inst_1850_);
lean_closure_set(v___f_1859_, 3, v_inst_1851_);
lean_closure_set(v___f_1859_, 4, v_g_1853_);
lean_closure_set(v___f_1859_, 5, v_a_1854_);
lean_closure_set(v___f_1859_, 6, v_toApplicative_1856_);
v___f_1860_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDeclCore___redArg___lam__1___boxed), 8, 7);
lean_closure_set(v___f_1860_, 0, v_ldecl_1852_);
lean_closure_set(v___f_1860_, 1, v_inst_1850_);
lean_closure_set(v___f_1860_, 2, v_inst_1851_);
lean_closure_set(v___f_1860_, 3, v_g_1853_);
lean_closure_set(v___f_1860_, 4, v_a_1854_);
lean_closure_set(v___f_1860_, 5, v_toBind_1857_);
lean_closure_set(v___f_1860_, 6, v___f_1859_);
v___x_1861_ = l_Lean_LocalDecl_toExpr(v_ldecl_1852_);
v___x_1862_ = l_Lean_ForEachExpr_visit___redArg(v_inst_1850_, v_inst_1851_, v_g_1853_, v___x_1861_, v_a_1854_);
v___x_1863_ = lean_apply_4(v_toBind_1857_, lean_box(0), lean_box(0), v___x_1862_, v___f_1860_);
return v___x_1863_;
}
else
{
lean_object* v_toApplicative_1864_; lean_object* v_toPure_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; 
lean_dec(v_g_1853_);
lean_dec_ref(v_ldecl_1852_);
lean_dec(v_inst_1850_);
v_toApplicative_1864_ = lean_ctor_get(v_inst_1851_, 0);
lean_inc_ref(v_toApplicative_1864_);
lean_dec_ref(v_inst_1851_);
v_toPure_1865_ = lean_ctor_get(v_toApplicative_1864_, 1);
lean_inc(v_toPure_1865_);
lean_dec_ref(v_toApplicative_1864_);
v___x_1866_ = lean_box(0);
v___x_1867_ = lean_apply_2(v_toPure_1865_, lean_box(0), v___x_1866_);
return v___x_1867_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___redArg___boxed(lean_object* v_inst_1868_, lean_object* v_inst_1869_, lean_object* v_ldecl_1870_, lean_object* v_g_1871_, lean_object* v_a_1872_){
_start:
{
lean_object* v_res_1873_; 
v_res_1873_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_1868_, v_inst_1869_, v_ldecl_1870_, v_g_1871_, v_a_1872_);
lean_dec(v_a_1872_);
return v_res_1873_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore(lean_object* v_00_u03c9_1874_, lean_object* v_m_1875_, lean_object* v_inst_1876_, lean_object* v_inst_1877_, lean_object* v_inst_1878_, lean_object* v_ldecl_1879_, lean_object* v_g_1880_, lean_object* v_a_1881_){
_start:
{
lean_object* v___x_1882_; 
v___x_1882_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_1877_, v_inst_1878_, v_ldecl_1879_, v_g_1880_, v_a_1881_);
return v___x_1882_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDeclCore___boxed(lean_object* v_00_u03c9_1883_, lean_object* v_m_1884_, lean_object* v_inst_1885_, lean_object* v_inst_1886_, lean_object* v_inst_1887_, lean_object* v_ldecl_1888_, lean_object* v_g_1889_, lean_object* v_a_1890_){
_start:
{
lean_object* v_res_1891_; 
v_res_1891_ = lp_aesop_Aesop_forEachExprInLDeclCore(v_00_u03c9_1883_, v_m_1884_, v_inst_1885_, v_inst_1886_, v_inst_1887_, v_ldecl_1888_, v_g_1889_, v_a_1890_);
lean_dec(v_a_1890_);
return v_res_1891_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0(lean_object* v_toPure_1892_, lean_object* v_____x_1893_){
_start:
{
lean_object* v_fst_1894_; lean_object* v___x_1895_; 
v_fst_1894_ = lean_ctor_get(v_____x_1893_, 0);
lean_inc(v_fst_1894_);
lean_dec_ref(v_____x_1893_);
v___x_1895_ = lean_apply_2(v_toPure_1892_, lean_box(0), v_fst_1894_);
return v___x_1895_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__1(lean_object* v_a_1896_, lean_object* v_toPure_1897_, lean_object* v_s_1898_){
_start:
{
lean_object* v___x_1899_; lean_object* v___x_1900_; 
v___x_1899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1899_, 0, v_a_1896_);
lean_ctor_set(v___x_1899_, 1, v_s_1898_);
v___x_1900_ = lean_apply_2(v_toPure_1897_, lean_box(0), v___x_1899_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2(lean_object* v_toPure_1901_, lean_object* v_ref_1902_, lean_object* v_inst_1903_, lean_object* v_toBind_1904_, lean_object* v_a_1905_){
_start:
{
lean_object* v___f_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___f_1906_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1906_, 0, v_a_1905_);
lean_closure_set(v___f_1906_, 1, v_toPure_1901_);
v___x_1907_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_1907_, 0, lean_box(0));
lean_closure_set(v___x_1907_, 1, lean_box(0));
lean_closure_set(v___x_1907_, 2, v_ref_1902_);
v___x_1908_ = lean_apply_2(v_inst_1903_, lean_box(0), v___x_1907_);
v___x_1909_ = lean_apply_4(v_toBind_1904_, lean_box(0), lean_box(0), v___x_1908_, v___f_1906_);
return v___x_1909_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__3(lean_object* v_toPure_1910_, lean_object* v_inst_1911_, lean_object* v_toBind_1912_, lean_object* v_inst_1913_, lean_object* v_ldecl_1914_, lean_object* v_g_1915_, lean_object* v_ref_1916_){
_start:
{
lean_object* v___f_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; 
lean_inc(v_toBind_1912_);
lean_inc(v_inst_1911_);
lean_inc(v_ref_1916_);
v___f_1917_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_1917_, 0, v_toPure_1910_);
lean_closure_set(v___f_1917_, 1, v_ref_1916_);
lean_closure_set(v___f_1917_, 2, v_inst_1911_);
lean_closure_set(v___f_1917_, 3, v_toBind_1912_);
v___x_1918_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_1911_, v_inst_1913_, v_ldecl_1914_, v_g_1915_, v_ref_1916_);
lean_dec(v_ref_1916_);
v___x_1919_ = lean_apply_4(v_toBind_1912_, lean_box(0), lean_box(0), v___x_1918_, v___f_1917_);
return v___x_1919_;
}
}
static lean_object* _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0(void){
_start:
{
lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; 
v___x_1920_ = lean_box(0);
v___x_1921_ = lean_unsigned_to_nat(16u);
v___x_1922_ = lean_mk_array(v___x_1921_, v___x_1920_);
return v___x_1922_;
}
}
static lean_object* _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1(void){
_start:
{
lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; 
v___x_1923_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__0);
v___x_1924_ = lean_unsigned_to_nat(0u);
v___x_1925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1925_, 0, v___x_1924_);
lean_ctor_set(v___x_1925_, 1, v___x_1923_);
return v___x_1925_;
}
}
static lean_object* _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2(void){
_start:
{
lean_object* v___x_1926_; lean_object* v___x_1927_; 
v___x_1926_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__1);
v___x_1927_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_1927_, 0, lean_box(0));
lean_closure_set(v___x_1927_, 1, lean_box(0));
lean_closure_set(v___x_1927_, 2, v___x_1926_);
return v___x_1927_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27___redArg(lean_object* v_inst_1928_, lean_object* v_inst_1929_, lean_object* v_ldecl_1930_, lean_object* v_g_1931_){
_start:
{
lean_object* v_toApplicative_1932_; lean_object* v_toBind_1933_; lean_object* v_toPure_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___f_1937_; lean_object* v___f_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; 
v_toApplicative_1932_ = lean_ctor_get(v_inst_1929_, 0);
v_toBind_1933_ = lean_ctor_get(v_inst_1929_, 1);
lean_inc_n(v_toBind_1933_, 3);
v_toPure_1934_ = lean_ctor_get(v_toApplicative_1932_, 1);
lean_inc_n(v_toPure_1934_, 2);
v___x_1935_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
lean_inc(v_inst_1928_);
v___x_1936_ = lean_apply_2(v_inst_1928_, lean_box(0), v___x_1935_);
v___f_1937_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1937_, 0, v_toPure_1934_);
v___f_1938_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__3), 7, 6);
lean_closure_set(v___f_1938_, 0, v_toPure_1934_);
lean_closure_set(v___f_1938_, 1, v_inst_1928_);
lean_closure_set(v___f_1938_, 2, v_toBind_1933_);
lean_closure_set(v___f_1938_, 3, v_inst_1929_);
lean_closure_set(v___f_1938_, 4, v_ldecl_1930_);
lean_closure_set(v___f_1938_, 5, v_g_1931_);
v___x_1939_ = lean_apply_4(v_toBind_1933_, lean_box(0), lean_box(0), v___x_1936_, v___f_1938_);
v___x_1940_ = lean_apply_4(v_toBind_1933_, lean_box(0), lean_box(0), v___x_1939_, v___f_1937_);
return v___x_1940_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl_x27(lean_object* v_00_u03c9_1941_, lean_object* v_m_1942_, lean_object* v_inst_1943_, lean_object* v_inst_1944_, lean_object* v_inst_1945_, lean_object* v_ldecl_1946_, lean_object* v_g_1947_){
_start:
{
lean_object* v_toApplicative_1948_; lean_object* v_toBind_1949_; lean_object* v_toPure_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___f_1953_; lean_object* v___f_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; 
v_toApplicative_1948_ = lean_ctor_get(v_inst_1945_, 0);
v_toBind_1949_ = lean_ctor_get(v_inst_1945_, 1);
lean_inc_n(v_toBind_1949_, 3);
v_toPure_1950_ = lean_ctor_get(v_toApplicative_1948_, 1);
lean_inc_n(v_toPure_1950_, 2);
v___x_1951_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
lean_inc(v_inst_1944_);
v___x_1952_ = lean_apply_2(v_inst_1944_, lean_box(0), v___x_1951_);
v___f_1953_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1953_, 0, v_toPure_1950_);
v___f_1954_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__3), 7, 6);
lean_closure_set(v___f_1954_, 0, v_toPure_1950_);
lean_closure_set(v___f_1954_, 1, v_inst_1944_);
lean_closure_set(v___f_1954_, 2, v_toBind_1949_);
lean_closure_set(v___f_1954_, 3, v_inst_1945_);
lean_closure_set(v___f_1954_, 4, v_ldecl_1946_);
lean_closure_set(v___f_1954_, 5, v_g_1947_);
v___x_1955_ = lean_apply_4(v_toBind_1949_, lean_box(0), lean_box(0), v___x_1952_, v___f_1954_);
v___x_1956_ = lean_apply_4(v_toBind_1949_, lean_box(0), lean_box(0), v___x_1955_, v___f_1953_);
return v___x_1956_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1(lean_object* v_toPure_1957_, lean_object* v_____r_1958_){
_start:
{
uint8_t v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; 
v___x_1959_ = 1;
v___x_1960_ = lean_box(v___x_1959_);
v___x_1961_ = lean_apply_2(v_toPure_1957_, lean_box(0), v___x_1960_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0(lean_object* v_g_1962_, lean_object* v_toBind_1963_, lean_object* v___f_1964_, lean_object* v_e_1965_){
_start:
{
lean_object* v___x_1966_; lean_object* v___x_1967_; 
v___x_1966_ = lean_apply_1(v_g_1962_, v_e_1965_);
v___x_1967_ = lean_apply_4(v_toBind_1963_, lean_box(0), lean_box(0), v___x_1966_, v___f_1964_);
return v___x_1967_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__4(lean_object* v_toPure_1968_, lean_object* v_inst_1969_, lean_object* v_toBind_1970_, lean_object* v_inst_1971_, lean_object* v_ldecl_1972_, lean_object* v___f_1973_, lean_object* v_ref_1974_){
_start:
{
lean_object* v___f_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; 
lean_inc(v_toBind_1970_);
lean_inc(v_inst_1969_);
lean_inc(v_ref_1974_);
v___f_1975_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_1975_, 0, v_toPure_1968_);
lean_closure_set(v___f_1975_, 1, v_ref_1974_);
lean_closure_set(v___f_1975_, 2, v_inst_1969_);
lean_closure_set(v___f_1975_, 3, v_toBind_1970_);
v___x_1976_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_1969_, v_inst_1971_, v_ldecl_1972_, v___f_1973_, v_ref_1974_);
lean_dec(v_ref_1974_);
v___x_1977_ = lean_apply_4(v_toBind_1970_, lean_box(0), lean_box(0), v___x_1976_, v___f_1975_);
return v___x_1977_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl___redArg(lean_object* v_inst_1978_, lean_object* v_inst_1979_, lean_object* v_ldecl_1980_, lean_object* v_g_1981_){
_start:
{
lean_object* v_toApplicative_1982_; lean_object* v_toBind_1983_; lean_object* v_toPure_1984_; lean_object* v___f_1985_; lean_object* v___f_1986_; lean_object* v___f_1987_; lean_object* v___f_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; 
v_toApplicative_1982_ = lean_ctor_get(v_inst_1979_, 0);
v_toBind_1983_ = lean_ctor_get(v_inst_1979_, 1);
lean_inc_n(v_toBind_1983_, 4);
v_toPure_1984_ = lean_ctor_get(v_toApplicative_1982_, 1);
lean_inc_n(v_toPure_1984_, 3);
v___f_1985_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1985_, 0, v_toPure_1984_);
v___f_1986_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1986_, 0, v_toPure_1984_);
v___f_1987_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1987_, 0, v_g_1981_);
lean_closure_set(v___f_1987_, 1, v_toBind_1983_);
lean_closure_set(v___f_1987_, 2, v___f_1986_);
lean_inc(v_inst_1978_);
v___f_1988_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__4), 7, 6);
lean_closure_set(v___f_1988_, 0, v_toPure_1984_);
lean_closure_set(v___f_1988_, 1, v_inst_1978_);
lean_closure_set(v___f_1988_, 2, v_toBind_1983_);
lean_closure_set(v___f_1988_, 3, v_inst_1979_);
lean_closure_set(v___f_1988_, 4, v_ldecl_1980_);
lean_closure_set(v___f_1988_, 5, v___f_1987_);
v___x_1989_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_1990_ = lean_apply_2(v_inst_1978_, lean_box(0), v___x_1989_);
v___x_1991_ = lean_apply_4(v_toBind_1983_, lean_box(0), lean_box(0), v___x_1990_, v___f_1988_);
v___x_1992_ = lean_apply_4(v_toBind_1983_, lean_box(0), lean_box(0), v___x_1991_, v___f_1985_);
return v___x_1992_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLDecl(lean_object* v_00_u03c9_1993_, lean_object* v_m_1994_, lean_object* v_inst_1995_, lean_object* v_inst_1996_, lean_object* v_inst_1997_, lean_object* v_ldecl_1998_, lean_object* v_g_1999_){
_start:
{
lean_object* v_toApplicative_2000_; lean_object* v_toBind_2001_; lean_object* v_toPure_2002_; lean_object* v___f_2003_; lean_object* v___f_2004_; lean_object* v___f_2005_; lean_object* v___f_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; 
v_toApplicative_2000_ = lean_ctor_get(v_inst_1997_, 0);
v_toBind_2001_ = lean_ctor_get(v_inst_1997_, 1);
lean_inc_n(v_toBind_2001_, 4);
v_toPure_2002_ = lean_ctor_get(v_toApplicative_2000_, 1);
lean_inc_n(v_toPure_2002_, 3);
v___f_2003_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2003_, 0, v_toPure_2002_);
v___f_2004_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2004_, 0, v_toPure_2002_);
v___f_2005_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2005_, 0, v_g_1999_);
lean_closure_set(v___f_2005_, 1, v_toBind_2001_);
lean_closure_set(v___f_2005_, 2, v___f_2004_);
lean_inc(v_inst_1996_);
v___f_2006_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__4), 7, 6);
lean_closure_set(v___f_2006_, 0, v_toPure_2002_);
lean_closure_set(v___f_2006_, 1, v_inst_1996_);
lean_closure_set(v___f_2006_, 2, v_toBind_2001_);
lean_closure_set(v___f_2006_, 3, v_inst_1997_);
lean_closure_set(v___f_2006_, 4, v_ldecl_1998_);
lean_closure_set(v___f_2006_, 5, v___f_2005_);
v___x_2007_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2008_ = lean_apply_2(v_inst_1996_, lean_box(0), v___x_2007_);
v___x_2009_ = lean_apply_4(v_toBind_2001_, lean_box(0), lean_box(0), v___x_2008_, v___f_2006_);
v___x_2010_ = lean_apply_4(v_toBind_2001_, lean_box(0), lean_box(0), v___x_2009_, v___f_2003_);
return v___x_2010_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0(lean_object* v_toApplicative_2011_, lean_object* v_a_2012_){
_start:
{
lean_object* v_toPure_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; 
v_toPure_2013_ = lean_ctor_get(v_toApplicative_2011_, 1);
lean_inc(v_toPure_2013_);
lean_dec_ref(v_toApplicative_2011_);
v___x_2014_ = lean_box(0);
v___x_2015_ = lean_apply_2(v_toPure_2013_, lean_box(0), v___x_2014_);
return v___x_2015_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__1(lean_object* v_toApplicative_2016_, lean_object* v___x_2017_, lean_object* v_a_2018_){
_start:
{
lean_object* v_toPure_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; 
v_toPure_2019_ = lean_ctor_get(v_toApplicative_2016_, 1);
lean_inc(v_toPure_2019_);
lean_dec_ref(v_toApplicative_2016_);
v___x_2020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2020_, 0, v___x_2017_);
v___x_2021_ = lean_apply_2(v_toPure_2019_, lean_box(0), v___x_2020_);
return v___x_2021_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2(lean_object* v_toApplicative_2022_, lean_object* v_inst_2023_, lean_object* v_inst_2024_, lean_object* v_g_2025_, lean_object* v_toBind_2026_, lean_object* v___f_2027_, lean_object* v_d_x3f_2028_, lean_object* v_b_2029_, lean_object* v___y_2030_){
_start:
{
if (lean_obj_tag(v_d_x3f_2028_) == 0)
{
lean_object* v_toPure_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; 
lean_dec(v___f_2027_);
lean_dec(v_toBind_2026_);
lean_dec(v_g_2025_);
lean_dec_ref(v_inst_2024_);
lean_dec(v_inst_2023_);
v_toPure_2031_ = lean_ctor_get(v_toApplicative_2022_, 1);
lean_inc(v_toPure_2031_);
lean_dec_ref(v_toApplicative_2022_);
v___x_2032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2032_, 0, v_b_2029_);
v___x_2033_ = lean_apply_2(v_toPure_2031_, lean_box(0), v___x_2032_);
return v___x_2033_;
}
else
{
lean_object* v_val_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; 
lean_dec_ref(v_toApplicative_2022_);
v_val_2034_ = lean_ctor_get(v_d_x3f_2028_, 0);
lean_inc(v_val_2034_);
lean_dec_ref_known(v_d_x3f_2028_, 1);
v___x_2035_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_2023_, v_inst_2024_, v_val_2034_, v_g_2025_, v___y_2030_);
v___x_2036_ = lean_apply_4(v_toBind_2026_, lean_box(0), lean_box(0), v___x_2035_, v___f_2027_);
return v___x_2036_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2___boxed(lean_object* v_toApplicative_2037_, lean_object* v_inst_2038_, lean_object* v_inst_2039_, lean_object* v_g_2040_, lean_object* v_toBind_2041_, lean_object* v___f_2042_, lean_object* v_d_x3f_2043_, lean_object* v_b_2044_, lean_object* v___y_2045_){
_start:
{
lean_object* v_res_2046_; 
v_res_2046_ = lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2(v_toApplicative_2037_, v_inst_2038_, v_inst_2039_, v_g_2040_, v_toBind_2041_, v___f_2042_, v_d_x3f_2043_, v_b_2044_, v___y_2045_);
lean_dec(v___y_2045_);
return v_res_2046_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg(lean_object* v_inst_2049_, lean_object* v_inst_2050_, lean_object* v_inst_2051_, lean_object* v_lctx_2052_, lean_object* v_g_2053_, lean_object* v_a_2054_){
_start:
{
lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v_decls_2058_; lean_object* v_toApplicative_2059_; lean_object* v_toBind_2060_; lean_object* v___f_2061_; lean_object* v___x_2062_; lean_object* v___f_2063_; lean_object* v___f_2064_; lean_object* v___x_163__overap_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; 
v___x_2055_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2056_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref(v_inst_2051_);
v___x_2057_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2049_, v___x_2055_, v___x_2056_, v_inst_2051_);
v_decls_2058_ = lean_ctor_get(v_lctx_2052_, 1);
v_toApplicative_2059_ = lean_ctor_get(v_inst_2051_, 0);
lean_inc_ref_n(v_toApplicative_2059_, 3);
v_toBind_2060_ = lean_ctor_get(v_inst_2051_, 1);
lean_inc_n(v_toBind_2060_, 2);
v___f_2061_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2061_, 0, v_toApplicative_2059_);
v___x_2062_ = lean_box(0);
v___f_2063_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2063_, 0, v_toApplicative_2059_);
lean_closure_set(v___f_2063_, 1, v___x_2062_);
v___f_2064_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2___boxed), 9, 6);
lean_closure_set(v___f_2064_, 0, v_toApplicative_2059_);
lean_closure_set(v___f_2064_, 1, v_inst_2050_);
lean_closure_set(v___f_2064_, 2, v_inst_2051_);
lean_closure_set(v___f_2064_, 3, v_g_2053_);
lean_closure_set(v___f_2064_, 4, v_toBind_2060_);
lean_closure_set(v___f_2064_, 5, v___f_2063_);
v___x_163__overap_2065_ = l_Lean_PersistentArray_forIn___redArg(v___x_2057_, v_decls_2058_, v___x_2062_, v___f_2064_);
lean_inc(v_a_2054_);
v___x_2066_ = lean_apply_1(v___x_163__overap_2065_, v_a_2054_);
v___x_2067_ = lean_apply_4(v_toBind_2060_, lean_box(0), lean_box(0), v___x_2066_, v___f_2061_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___redArg___boxed(lean_object* v_inst_2068_, lean_object* v_inst_2069_, lean_object* v_inst_2070_, lean_object* v_lctx_2071_, lean_object* v_g_2072_, lean_object* v_a_2073_){
_start:
{
lean_object* v_res_2074_; 
v_res_2074_ = lp_aesop_Aesop_forEachExprInLCtxCore___redArg(v_inst_2068_, v_inst_2069_, v_inst_2070_, v_lctx_2071_, v_g_2072_, v_a_2073_);
lean_dec(v_a_2073_);
lean_dec_ref(v_lctx_2071_);
return v_res_2074_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore(lean_object* v_00_u03c9_2075_, lean_object* v_m_2076_, lean_object* v_inst_2077_, lean_object* v_inst_2078_, lean_object* v_inst_2079_, lean_object* v_lctx_2080_, lean_object* v_g_2081_, lean_object* v_a_2082_){
_start:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v_decls_2086_; lean_object* v_toApplicative_2087_; lean_object* v_toBind_2088_; lean_object* v___f_2089_; lean_object* v___x_2090_; lean_object* v___f_2091_; lean_object* v___f_2092_; lean_object* v___x_203__overap_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; 
v___x_2083_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2084_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref(v_inst_2079_);
v___x_2085_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2077_, v___x_2083_, v___x_2084_, v_inst_2079_);
v_decls_2086_ = lean_ctor_get(v_lctx_2080_, 1);
v_toApplicative_2087_ = lean_ctor_get(v_inst_2079_, 0);
lean_inc_ref_n(v_toApplicative_2087_, 3);
v_toBind_2088_ = lean_ctor_get(v_inst_2079_, 1);
lean_inc_n(v_toBind_2088_, 2);
v___f_2089_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2089_, 0, v_toApplicative_2087_);
v___x_2090_ = lean_box(0);
v___f_2091_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2091_, 0, v_toApplicative_2087_);
lean_closure_set(v___f_2091_, 1, v___x_2090_);
v___f_2092_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2___boxed), 9, 6);
lean_closure_set(v___f_2092_, 0, v_toApplicative_2087_);
lean_closure_set(v___f_2092_, 1, v_inst_2078_);
lean_closure_set(v___f_2092_, 2, v_inst_2079_);
lean_closure_set(v___f_2092_, 3, v_g_2081_);
lean_closure_set(v___f_2092_, 4, v_toBind_2088_);
lean_closure_set(v___f_2092_, 5, v___f_2091_);
v___x_203__overap_2093_ = l_Lean_PersistentArray_forIn___redArg(v___x_2085_, v_decls_2086_, v___x_2090_, v___f_2092_);
lean_inc(v_a_2082_);
v___x_2094_ = lean_apply_1(v___x_203__overap_2093_, v_a_2082_);
v___x_2095_ = lean_apply_4(v_toBind_2088_, lean_box(0), lean_box(0), v___x_2094_, v___f_2089_);
return v___x_2095_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtxCore___boxed(lean_object* v_00_u03c9_2096_, lean_object* v_m_2097_, lean_object* v_inst_2098_, lean_object* v_inst_2099_, lean_object* v_inst_2100_, lean_object* v_lctx_2101_, lean_object* v_g_2102_, lean_object* v_a_2103_){
_start:
{
lean_object* v_res_2104_; 
v_res_2104_ = lp_aesop_Aesop_forEachExprInLCtxCore(v_00_u03c9_2096_, v_m_2097_, v_inst_2098_, v_inst_2099_, v_inst_2100_, v_lctx_2101_, v_g_2102_, v_a_2103_);
lean_dec(v_a_2103_);
lean_dec_ref(v_lctx_2101_);
return v_res_2104_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4(lean_object* v___x_2105_, lean_object* v_toPure_2106_, lean_object* v_a_2107_){
_start:
{
lean_object* v___x_2108_; lean_object* v___x_2109_; 
v___x_2108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2108_, 0, v___x_2105_);
v___x_2109_ = lean_apply_2(v_toPure_2106_, lean_box(0), v___x_2108_);
return v___x_2109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0(lean_object* v_toPure_2110_, lean_object* v_inst_2111_, lean_object* v_inst_2112_, lean_object* v_g_2113_, lean_object* v_toBind_2114_, lean_object* v___f_2115_, lean_object* v_d_x3f_2116_, lean_object* v_b_2117_, lean_object* v___y_2118_){
_start:
{
if (lean_obj_tag(v_d_x3f_2116_) == 0)
{
lean_object* v___x_2119_; lean_object* v___x_2120_; 
lean_dec(v___f_2115_);
lean_dec(v_toBind_2114_);
lean_dec(v_g_2113_);
lean_dec_ref(v_inst_2112_);
lean_dec(v_inst_2111_);
v___x_2119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2119_, 0, v_b_2117_);
v___x_2120_ = lean_apply_2(v_toPure_2110_, lean_box(0), v___x_2119_);
return v___x_2120_;
}
else
{
lean_object* v_val_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; 
lean_dec(v_toPure_2110_);
v_val_2121_ = lean_ctor_get(v_d_x3f_2116_, 0);
lean_inc(v_val_2121_);
lean_dec_ref_known(v_d_x3f_2116_, 1);
v___x_2122_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_2111_, v_inst_2112_, v_val_2121_, v_g_2113_, v___y_2118_);
v___x_2123_ = lean_apply_4(v_toBind_2114_, lean_box(0), lean_box(0), v___x_2122_, v___f_2115_);
return v___x_2123_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0___boxed(lean_object* v_toPure_2124_, lean_object* v_inst_2125_, lean_object* v_inst_2126_, lean_object* v_g_2127_, lean_object* v_toBind_2128_, lean_object* v___f_2129_, lean_object* v_d_x3f_2130_, lean_object* v_b_2131_, lean_object* v___y_2132_){
_start:
{
lean_object* v_res_2133_; 
v_res_2133_ = lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0(v_toPure_2124_, v_inst_2125_, v_inst_2126_, v_g_2127_, v_toBind_2128_, v___f_2129_, v_d_x3f_2130_, v_b_2131_, v___y_2132_);
lean_dec(v___y_2132_);
return v_res_2133_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1(lean_object* v_inst_2134_, lean_object* v___x_2135_, lean_object* v___x_2136_, lean_object* v_inst_2137_, lean_object* v_lctx_2138_, lean_object* v_toPure_2139_, lean_object* v_inst_2140_, lean_object* v_toBind_2141_, lean_object* v_g_2142_, lean_object* v___f_2143_, lean_object* v_ref_2144_){
_start:
{
lean_object* v___x_2145_; lean_object* v_decls_2146_; lean_object* v___f_2147_; lean_object* v___x_2148_; lean_object* v___f_2149_; lean_object* v___f_2150_; lean_object* v___x_180__overap_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; 
lean_inc_ref(v_inst_2137_);
v___x_2145_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2134_, v___x_2135_, v___x_2136_, v_inst_2137_);
v_decls_2146_ = lean_ctor_get(v_lctx_2138_, 1);
lean_inc_n(v_toBind_2141_, 3);
lean_inc(v_inst_2140_);
lean_inc(v_ref_2144_);
lean_inc_n(v_toPure_2139_, 2);
v___f_2147_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2147_, 0, v_toPure_2139_);
lean_closure_set(v___f_2147_, 1, v_ref_2144_);
lean_closure_set(v___f_2147_, 2, v_inst_2140_);
lean_closure_set(v___f_2147_, 3, v_toBind_2141_);
v___x_2148_ = lean_box(0);
v___f_2149_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4), 3, 2);
lean_closure_set(v___f_2149_, 0, v___x_2148_);
lean_closure_set(v___f_2149_, 1, v_toPure_2139_);
v___f_2150_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0___boxed), 9, 6);
lean_closure_set(v___f_2150_, 0, v_toPure_2139_);
lean_closure_set(v___f_2150_, 1, v_inst_2140_);
lean_closure_set(v___f_2150_, 2, v_inst_2137_);
lean_closure_set(v___f_2150_, 3, v_g_2142_);
lean_closure_set(v___f_2150_, 4, v_toBind_2141_);
lean_closure_set(v___f_2150_, 5, v___f_2149_);
v___x_180__overap_2151_ = l_Lean_PersistentArray_forIn___redArg(v___x_2145_, v_decls_2146_, v___x_2148_, v___f_2150_);
v___x_2152_ = lean_apply_1(v___x_180__overap_2151_, v_ref_2144_);
v___x_2153_ = lean_apply_4(v_toBind_2141_, lean_box(0), lean_box(0), v___x_2152_, v___f_2143_);
v___x_2154_ = lean_apply_4(v_toBind_2141_, lean_box(0), lean_box(0), v___x_2153_, v___f_2147_);
return v___x_2154_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1___boxed(lean_object* v_inst_2155_, lean_object* v___x_2156_, lean_object* v___x_2157_, lean_object* v_inst_2158_, lean_object* v_lctx_2159_, lean_object* v_toPure_2160_, lean_object* v_inst_2161_, lean_object* v_toBind_2162_, lean_object* v_g_2163_, lean_object* v___f_2164_, lean_object* v_ref_2165_){
_start:
{
lean_object* v_res_2166_; 
v_res_2166_ = lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1(v_inst_2155_, v___x_2156_, v___x_2157_, v_inst_2158_, v_lctx_2159_, v_toPure_2160_, v_inst_2161_, v_toBind_2162_, v_g_2163_, v___f_2164_, v_ref_2165_);
lean_dec_ref(v_lctx_2159_);
return v_res_2166_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__2(lean_object* v_toApplicative_2167_, lean_object* v_inst_2168_, lean_object* v_inst_2169_, lean_object* v___x_2170_, lean_object* v___x_2171_, lean_object* v_inst_2172_, lean_object* v_toBind_2173_, lean_object* v_g_2174_, lean_object* v___f_2175_, lean_object* v_____do__lift_2176_){
_start:
{
lean_object* v_lctx_2177_; lean_object* v_toPure_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___f_2181_; lean_object* v___f_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; 
v_lctx_2177_ = lean_ctor_get(v_____do__lift_2176_, 1);
lean_inc_ref(v_lctx_2177_);
lean_dec_ref(v_____do__lift_2176_);
v_toPure_2178_ = lean_ctor_get(v_toApplicative_2167_, 1);
lean_inc_n(v_toPure_2178_, 2);
lean_dec_ref(v_toApplicative_2167_);
v___x_2179_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
lean_inc(v_inst_2168_);
v___x_2180_ = lean_apply_2(v_inst_2168_, lean_box(0), v___x_2179_);
v___f_2181_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2181_, 0, v_toPure_2178_);
lean_inc_n(v_toBind_2173_, 2);
v___f_2182_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__1___boxed), 11, 10);
lean_closure_set(v___f_2182_, 0, v_inst_2169_);
lean_closure_set(v___f_2182_, 1, v___x_2170_);
lean_closure_set(v___f_2182_, 2, v___x_2171_);
lean_closure_set(v___f_2182_, 3, v_inst_2172_);
lean_closure_set(v___f_2182_, 4, v_lctx_2177_);
lean_closure_set(v___f_2182_, 5, v_toPure_2178_);
lean_closure_set(v___f_2182_, 6, v_inst_2168_);
lean_closure_set(v___f_2182_, 7, v_toBind_2173_);
lean_closure_set(v___f_2182_, 8, v_g_2174_);
lean_closure_set(v___f_2182_, 9, v___f_2175_);
v___x_2183_ = lean_apply_4(v_toBind_2173_, lean_box(0), lean_box(0), v___x_2180_, v___f_2182_);
v___x_2184_ = lean_apply_4(v_toBind_2173_, lean_box(0), lean_box(0), v___x_2183_, v___f_2181_);
return v___x_2184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27___redArg(lean_object* v_inst_2185_, lean_object* v_inst_2186_, lean_object* v_inst_2187_, lean_object* v_inst_2188_, lean_object* v_inst_2189_, lean_object* v_mvarId_2190_, lean_object* v_g_2191_){
_start:
{
lean_object* v_toApplicative_2192_; lean_object* v_toBind_2193_; lean_object* v___f_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___f_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; 
v_toApplicative_2192_ = lean_ctor_get(v_inst_2187_, 0);
v_toBind_2193_ = lean_ctor_get(v_inst_2187_, 1);
lean_inc_ref_n(v_toApplicative_2192_, 2);
v___f_2194_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2194_, 0, v_toApplicative_2192_);
v___x_2195_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2196_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_n(v_toBind_2193_, 2);
lean_inc_ref(v_inst_2187_);
v___f_2197_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__2), 10, 9);
lean_closure_set(v___f_2197_, 0, v_toApplicative_2192_);
lean_closure_set(v___f_2197_, 1, v_inst_2186_);
lean_closure_set(v___f_2197_, 2, v_inst_2185_);
lean_closure_set(v___f_2197_, 3, v___x_2195_);
lean_closure_set(v___f_2197_, 4, v___x_2196_);
lean_closure_set(v___f_2197_, 5, v_inst_2187_);
lean_closure_set(v___f_2197_, 6, v_toBind_2193_);
lean_closure_set(v___f_2197_, 7, v_g_2191_);
lean_closure_set(v___f_2197_, 8, v___f_2194_);
lean_inc(v_mvarId_2190_);
v___x_2198_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2198_, 0, v_mvarId_2190_);
v___x_2199_ = lean_apply_2(v_inst_2189_, lean_box(0), v___x_2198_);
v___x_2200_ = lean_apply_4(v_toBind_2193_, lean_box(0), lean_box(0), v___x_2199_, v___f_2197_);
v___x_2201_ = l_Lean_MVarId_withContext___redArg(v_inst_2188_, v_inst_2187_, v_mvarId_2190_, v___x_2200_);
return v___x_2201_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx_x27(lean_object* v_00_u03c9_2202_, lean_object* v_m_2203_, lean_object* v_inst_2204_, lean_object* v_inst_2205_, lean_object* v_inst_2206_, lean_object* v_inst_2207_, lean_object* v_inst_2208_, lean_object* v_mvarId_2209_, lean_object* v_g_2210_){
_start:
{
lean_object* v_toApplicative_2211_; lean_object* v_toBind_2212_; lean_object* v___f_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___f_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; 
v_toApplicative_2211_ = lean_ctor_get(v_inst_2206_, 0);
v_toBind_2212_ = lean_ctor_get(v_inst_2206_, 1);
lean_inc_ref_n(v_toApplicative_2211_, 2);
v___f_2213_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2213_, 0, v_toApplicative_2211_);
v___x_2214_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2215_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_n(v_toBind_2212_, 2);
lean_inc_ref(v_inst_2206_);
v___f_2216_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__2), 10, 9);
lean_closure_set(v___f_2216_, 0, v_toApplicative_2211_);
lean_closure_set(v___f_2216_, 1, v_inst_2205_);
lean_closure_set(v___f_2216_, 2, v_inst_2204_);
lean_closure_set(v___f_2216_, 3, v___x_2214_);
lean_closure_set(v___f_2216_, 4, v___x_2215_);
lean_closure_set(v___f_2216_, 5, v_inst_2206_);
lean_closure_set(v___f_2216_, 6, v_toBind_2212_);
lean_closure_set(v___f_2216_, 7, v_g_2210_);
lean_closure_set(v___f_2216_, 8, v___f_2213_);
lean_inc(v_mvarId_2209_);
v___x_2217_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2217_, 0, v_mvarId_2209_);
v___x_2218_ = lean_apply_2(v_inst_2208_, lean_box(0), v___x_2217_);
v___x_2219_ = lean_apply_4(v_toBind_2212_, lean_box(0), lean_box(0), v___x_2218_, v___f_2216_);
v___x_2220_ = l_Lean_MVarId_withContext___redArg(v_inst_2207_, v_inst_2206_, v_mvarId_2209_, v___x_2219_);
return v___x_2220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0(lean_object* v_toPure_2221_, lean_object* v_a_2222_){
_start:
{
lean_object* v___x_2223_; lean_object* v___x_2224_; 
v___x_2223_ = lean_box(0);
v___x_2224_ = lean_apply_2(v_toPure_2221_, lean_box(0), v___x_2223_);
return v___x_2224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7(lean_object* v_toPure_2225_, lean_object* v_inst_2226_, lean_object* v_inst_2227_, lean_object* v___f_2228_, lean_object* v_toBind_2229_, lean_object* v___f_2230_, lean_object* v_d_x3f_2231_, lean_object* v_b_2232_, lean_object* v___y_2233_){
_start:
{
if (lean_obj_tag(v_d_x3f_2231_) == 0)
{
lean_object* v___x_2234_; lean_object* v___x_2235_; 
lean_dec(v___f_2230_);
lean_dec(v_toBind_2229_);
lean_dec(v___f_2228_);
lean_dec_ref(v_inst_2227_);
lean_dec(v_inst_2226_);
v___x_2234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2234_, 0, v_b_2232_);
v___x_2235_ = lean_apply_2(v_toPure_2225_, lean_box(0), v___x_2234_);
return v___x_2235_;
}
else
{
lean_object* v_val_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; 
lean_dec(v_toPure_2225_);
v_val_2236_ = lean_ctor_get(v_d_x3f_2231_, 0);
lean_inc(v_val_2236_);
lean_dec_ref_known(v_d_x3f_2231_, 1);
v___x_2237_ = lp_aesop_Aesop_forEachExprInLDeclCore___redArg(v_inst_2226_, v_inst_2227_, v_val_2236_, v___f_2228_, v___y_2233_);
v___x_2238_ = lean_apply_4(v_toBind_2229_, lean_box(0), lean_box(0), v___x_2237_, v___f_2230_);
return v___x_2238_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7___boxed(lean_object* v_toPure_2239_, lean_object* v_inst_2240_, lean_object* v_inst_2241_, lean_object* v___f_2242_, lean_object* v_toBind_2243_, lean_object* v___f_2244_, lean_object* v_d_x3f_2245_, lean_object* v_b_2246_, lean_object* v___y_2247_){
_start:
{
lean_object* v_res_2248_; 
v_res_2248_ = lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7(v_toPure_2239_, v_inst_2240_, v_inst_2241_, v___f_2242_, v_toBind_2243_, v___f_2244_, v_d_x3f_2245_, v_b_2246_, v___y_2247_);
lean_dec(v___y_2247_);
return v_res_2248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1(lean_object* v_inst_2249_, lean_object* v___x_2250_, lean_object* v___x_2251_, lean_object* v_inst_2252_, lean_object* v_lctx_2253_, lean_object* v_toPure_2254_, lean_object* v_inst_2255_, lean_object* v_toBind_2256_, lean_object* v___f_2257_, lean_object* v___f_2258_, lean_object* v_ref_2259_){
_start:
{
lean_object* v___x_2260_; lean_object* v_decls_2261_; lean_object* v___f_2262_; lean_object* v___x_2263_; lean_object* v___f_2264_; lean_object* v___f_2265_; lean_object* v___x_194__overap_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; 
lean_inc_ref(v_inst_2252_);
v___x_2260_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2249_, v___x_2250_, v___x_2251_, v_inst_2252_);
v_decls_2261_ = lean_ctor_get(v_lctx_2253_, 1);
lean_inc_n(v_toBind_2256_, 3);
lean_inc(v_inst_2255_);
lean_inc(v_ref_2259_);
lean_inc_n(v_toPure_2254_, 2);
v___f_2262_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2262_, 0, v_toPure_2254_);
lean_closure_set(v___f_2262_, 1, v_ref_2259_);
lean_closure_set(v___f_2262_, 2, v_inst_2255_);
lean_closure_set(v___f_2262_, 3, v_toBind_2256_);
v___x_2263_ = lean_box(0);
v___f_2264_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4), 3, 2);
lean_closure_set(v___f_2264_, 0, v___x_2263_);
lean_closure_set(v___f_2264_, 1, v_toPure_2254_);
v___f_2265_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7___boxed), 9, 6);
lean_closure_set(v___f_2265_, 0, v_toPure_2254_);
lean_closure_set(v___f_2265_, 1, v_inst_2255_);
lean_closure_set(v___f_2265_, 2, v_inst_2252_);
lean_closure_set(v___f_2265_, 3, v___f_2257_);
lean_closure_set(v___f_2265_, 4, v_toBind_2256_);
lean_closure_set(v___f_2265_, 5, v___f_2264_);
v___x_194__overap_2266_ = l_Lean_PersistentArray_forIn___redArg(v___x_2260_, v_decls_2261_, v___x_2263_, v___f_2265_);
v___x_2267_ = lean_apply_1(v___x_194__overap_2266_, v_ref_2259_);
v___x_2268_ = lean_apply_4(v_toBind_2256_, lean_box(0), lean_box(0), v___x_2267_, v___f_2258_);
v___x_2269_ = lean_apply_4(v_toBind_2256_, lean_box(0), lean_box(0), v___x_2268_, v___f_2262_);
return v___x_2269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1___boxed(lean_object* v_inst_2270_, lean_object* v___x_2271_, lean_object* v___x_2272_, lean_object* v_inst_2273_, lean_object* v_lctx_2274_, lean_object* v_toPure_2275_, lean_object* v_inst_2276_, lean_object* v_toBind_2277_, lean_object* v___f_2278_, lean_object* v___f_2279_, lean_object* v_ref_2280_){
_start:
{
lean_object* v_res_2281_; 
v_res_2281_ = lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1(v_inst_2270_, v___x_2271_, v___x_2272_, v_inst_2273_, v_lctx_2274_, v_toPure_2275_, v_inst_2276_, v_toBind_2277_, v___f_2278_, v___f_2279_, v_ref_2280_);
lean_dec_ref(v_lctx_2274_);
return v_res_2281_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__2(lean_object* v_inst_2282_, lean_object* v___x_2283_, lean_object* v___x_2284_, lean_object* v_inst_2285_, lean_object* v_toPure_2286_, lean_object* v_inst_2287_, lean_object* v_toBind_2288_, lean_object* v___f_2289_, lean_object* v___f_2290_, lean_object* v___f_2291_, lean_object* v_____do__lift_2292_){
_start:
{
lean_object* v_lctx_2293_; lean_object* v___f_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; 
v_lctx_2293_ = lean_ctor_get(v_____do__lift_2292_, 1);
lean_inc_ref(v_lctx_2293_);
lean_dec_ref(v_____do__lift_2292_);
lean_inc_n(v_toBind_2288_, 2);
lean_inc(v_inst_2287_);
v___f_2294_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__1___boxed), 11, 10);
lean_closure_set(v___f_2294_, 0, v_inst_2282_);
lean_closure_set(v___f_2294_, 1, v___x_2283_);
lean_closure_set(v___f_2294_, 2, v___x_2284_);
lean_closure_set(v___f_2294_, 3, v_inst_2285_);
lean_closure_set(v___f_2294_, 4, v_lctx_2293_);
lean_closure_set(v___f_2294_, 5, v_toPure_2286_);
lean_closure_set(v___f_2294_, 6, v_inst_2287_);
lean_closure_set(v___f_2294_, 7, v_toBind_2288_);
lean_closure_set(v___f_2294_, 8, v___f_2289_);
lean_closure_set(v___f_2294_, 9, v___f_2290_);
v___x_2295_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2296_ = lean_apply_2(v_inst_2287_, lean_box(0), v___x_2295_);
v___x_2297_ = lean_apply_4(v_toBind_2288_, lean_box(0), lean_box(0), v___x_2296_, v___f_2294_);
v___x_2298_ = lean_apply_4(v_toBind_2288_, lean_box(0), lean_box(0), v___x_2297_, v___f_2291_);
return v___x_2298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx___redArg(lean_object* v_inst_2299_, lean_object* v_inst_2300_, lean_object* v_inst_2301_, lean_object* v_inst_2302_, lean_object* v_inst_2303_, lean_object* v_mvarId_2304_, lean_object* v_g_2305_){
_start:
{
lean_object* v_toApplicative_2306_; lean_object* v_toBind_2307_; lean_object* v_toPure_2308_; lean_object* v___f_2309_; lean_object* v___f_2310_; lean_object* v___f_2311_; lean_object* v___f_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___f_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; 
v_toApplicative_2306_ = lean_ctor_get(v_inst_2301_, 0);
v_toBind_2307_ = lean_ctor_get(v_inst_2301_, 1);
v_toPure_2308_ = lean_ctor_get(v_toApplicative_2306_, 1);
lean_inc_n(v_toPure_2308_, 4);
v___f_2309_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2309_, 0, v_toPure_2308_);
v___f_2310_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2310_, 0, v_toPure_2308_);
lean_inc_n(v_toBind_2307_, 3);
v___f_2311_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2311_, 0, v_g_2305_);
lean_closure_set(v___f_2311_, 1, v_toBind_2307_);
lean_closure_set(v___f_2311_, 2, v___f_2310_);
v___f_2312_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2312_, 0, v_toPure_2308_);
v___x_2313_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2314_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref(v_inst_2301_);
v___f_2315_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__2), 11, 10);
lean_closure_set(v___f_2315_, 0, v_inst_2299_);
lean_closure_set(v___f_2315_, 1, v___x_2313_);
lean_closure_set(v___f_2315_, 2, v___x_2314_);
lean_closure_set(v___f_2315_, 3, v_inst_2301_);
lean_closure_set(v___f_2315_, 4, v_toPure_2308_);
lean_closure_set(v___f_2315_, 5, v_inst_2300_);
lean_closure_set(v___f_2315_, 6, v_toBind_2307_);
lean_closure_set(v___f_2315_, 7, v___f_2311_);
lean_closure_set(v___f_2315_, 8, v___f_2309_);
lean_closure_set(v___f_2315_, 9, v___f_2312_);
lean_inc(v_mvarId_2304_);
v___x_2316_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2316_, 0, v_mvarId_2304_);
v___x_2317_ = lean_apply_2(v_inst_2303_, lean_box(0), v___x_2316_);
v___x_2318_ = lean_apply_4(v_toBind_2307_, lean_box(0), lean_box(0), v___x_2317_, v___f_2315_);
v___x_2319_ = l_Lean_MVarId_withContext___redArg(v_inst_2302_, v_inst_2301_, v_mvarId_2304_, v___x_2318_);
return v___x_2319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInLCtx(lean_object* v_00_u03c9_2320_, lean_object* v_m_2321_, lean_object* v_inst_2322_, lean_object* v_inst_2323_, lean_object* v_inst_2324_, lean_object* v_inst_2325_, lean_object* v_inst_2326_, lean_object* v_mvarId_2327_, lean_object* v_g_2328_){
_start:
{
lean_object* v_toApplicative_2329_; lean_object* v_toBind_2330_; lean_object* v_toPure_2331_; lean_object* v___f_2332_; lean_object* v___f_2333_; lean_object* v___f_2334_; lean_object* v___f_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___f_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; 
v_toApplicative_2329_ = lean_ctor_get(v_inst_2324_, 0);
v_toBind_2330_ = lean_ctor_get(v_inst_2324_, 1);
v_toPure_2331_ = lean_ctor_get(v_toApplicative_2329_, 1);
lean_inc_n(v_toPure_2331_, 4);
v___f_2332_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2332_, 0, v_toPure_2331_);
v___f_2333_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2333_, 0, v_toPure_2331_);
lean_inc_n(v_toBind_2330_, 3);
v___f_2334_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2334_, 0, v_g_2328_);
lean_closure_set(v___f_2334_, 1, v_toBind_2330_);
lean_closure_set(v___f_2334_, 2, v___f_2333_);
v___f_2335_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2335_, 0, v_toPure_2331_);
v___x_2336_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2337_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref(v_inst_2324_);
v___f_2338_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__2), 11, 10);
lean_closure_set(v___f_2338_, 0, v_inst_2322_);
lean_closure_set(v___f_2338_, 1, v___x_2336_);
lean_closure_set(v___f_2338_, 2, v___x_2337_);
lean_closure_set(v___f_2338_, 3, v_inst_2324_);
lean_closure_set(v___f_2338_, 4, v_toPure_2331_);
lean_closure_set(v___f_2338_, 5, v_inst_2323_);
lean_closure_set(v___f_2338_, 6, v_toBind_2330_);
lean_closure_set(v___f_2338_, 7, v___f_2334_);
lean_closure_set(v___f_2338_, 8, v___f_2332_);
lean_closure_set(v___f_2338_, 9, v___f_2335_);
lean_inc(v_mvarId_2327_);
v___x_2339_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2339_, 0, v_mvarId_2327_);
v___x_2340_ = lean_apply_2(v_inst_2326_, lean_box(0), v___x_2339_);
v___x_2341_ = lean_apply_4(v_toBind_2330_, lean_box(0), lean_box(0), v___x_2340_, v___f_2338_);
v___x_2342_ = l_Lean_MVarId_withContext___redArg(v_inst_2325_, v_inst_2324_, v_mvarId_2327_, v___x_2341_);
return v___x_2342_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__0(lean_object* v_inst_2343_, lean_object* v_a_2344_){
_start:
{
lean_object* v_toApplicative_2345_; lean_object* v_toPure_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; 
v_toApplicative_2345_ = lean_ctor_get(v_inst_2343_, 0);
lean_inc_ref(v_toApplicative_2345_);
lean_dec_ref(v_inst_2343_);
v_toPure_2346_ = lean_ctor_get(v_toApplicative_2345_, 1);
lean_inc(v_toPure_2346_);
lean_dec_ref(v_toApplicative_2345_);
v___x_2347_ = lean_box(0);
v___x_2348_ = lean_apply_2(v_toPure_2346_, lean_box(0), v___x_2347_);
return v___x_2348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1(lean_object* v_inst_2349_, lean_object* v_inst_2350_, lean_object* v_g_2351_, lean_object* v___y_2352_, lean_object* v_a_2353_){
_start:
{
lean_object* v___x_2354_; 
v___x_2354_ = l_Lean_ForEachExpr_visit___redArg(v_inst_2349_, v_inst_2350_, v_g_2351_, v_a_2353_, v___y_2352_);
return v___x_2354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1___boxed(lean_object* v_inst_2355_, lean_object* v_inst_2356_, lean_object* v_g_2357_, lean_object* v___y_2358_, lean_object* v_a_2359_){
_start:
{
lean_object* v_res_2360_; 
v_res_2360_ = lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1(v_inst_2355_, v_inst_2356_, v_g_2357_, v___y_2358_, v_a_2359_);
lean_dec(v___y_2358_);
return v_res_2360_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__2(lean_object* v_mvarId_2361_, lean_object* v_inst_2362_, lean_object* v_toBind_2363_, lean_object* v___f_2364_, lean_object* v_a_2365_){
_start:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; 
v___x_2366_ = lean_alloc_closure((void*)(l_Lean_MVarId_getType___boxed), 6, 1);
lean_closure_set(v___x_2366_, 0, v_mvarId_2361_);
v___x_2367_ = lean_apply_2(v_inst_2362_, lean_box(0), v___x_2366_);
v___x_2368_ = lean_apply_4(v_toBind_2363_, lean_box(0), lean_box(0), v___x_2367_, v___f_2364_);
return v___x_2368_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5(lean_object* v_inst_2369_, lean_object* v_inst_2370_, lean_object* v_inst_2371_, lean_object* v_g_2372_, lean_object* v_mvarId_2373_, lean_object* v_inst_2374_, lean_object* v___f_2375_, lean_object* v_____do__lift_2376_, lean_object* v___y_2377_){
_start:
{
lean_object* v_lctx_2378_; lean_object* v_toApplicative_2379_; lean_object* v_toBind_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v_decls_2384_; lean_object* v___f_2385_; lean_object* v___f_2386_; lean_object* v___x_2387_; lean_object* v___f_2388_; lean_object* v___f_2389_; lean_object* v___x_232__overap_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; 
v_lctx_2378_ = lean_ctor_get(v_____do__lift_2376_, 1);
v_toApplicative_2379_ = lean_ctor_get(v_inst_2369_, 0);
lean_inc_ref_n(v_toApplicative_2379_, 2);
v_toBind_2380_ = lean_ctor_get(v_inst_2369_, 1);
lean_inc_n(v_toBind_2380_, 4);
v___x_2381_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2382_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref_n(v_inst_2369_, 2);
v___x_2383_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2370_, v___x_2381_, v___x_2382_, v_inst_2369_);
v_decls_2384_ = lean_ctor_get(v_lctx_2378_, 1);
lean_inc_n(v___y_2377_, 2);
lean_inc(v_g_2372_);
lean_inc(v_inst_2371_);
v___f_2385_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_2385_, 0, v_inst_2371_);
lean_closure_set(v___f_2385_, 1, v_inst_2369_);
lean_closure_set(v___f_2385_, 2, v_g_2372_);
lean_closure_set(v___f_2385_, 3, v___y_2377_);
v___f_2386_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2386_, 0, v_mvarId_2373_);
lean_closure_set(v___f_2386_, 1, v_inst_2374_);
lean_closure_set(v___f_2386_, 2, v_toBind_2380_);
lean_closure_set(v___f_2386_, 3, v___f_2385_);
v___x_2387_ = lean_box(0);
v___f_2388_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2388_, 0, v_toApplicative_2379_);
lean_closure_set(v___f_2388_, 1, v___x_2387_);
v___f_2389_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___lam__2___boxed), 9, 6);
lean_closure_set(v___f_2389_, 0, v_toApplicative_2379_);
lean_closure_set(v___f_2389_, 1, v_inst_2371_);
lean_closure_set(v___f_2389_, 2, v_inst_2369_);
lean_closure_set(v___f_2389_, 3, v_g_2372_);
lean_closure_set(v___f_2389_, 4, v_toBind_2380_);
lean_closure_set(v___f_2389_, 5, v___f_2388_);
v___x_232__overap_2390_ = l_Lean_PersistentArray_forIn___redArg(v___x_2383_, v_decls_2384_, v___x_2387_, v___f_2389_);
v___x_2391_ = lean_apply_1(v___x_232__overap_2390_, v___y_2377_);
v___x_2392_ = lean_apply_4(v_toBind_2380_, lean_box(0), lean_box(0), v___x_2391_, v___f_2375_);
v___x_2393_ = lean_apply_4(v_toBind_2380_, lean_box(0), lean_box(0), v___x_2392_, v___f_2386_);
return v___x_2393_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5___boxed(lean_object* v_inst_2394_, lean_object* v_inst_2395_, lean_object* v_inst_2396_, lean_object* v_g_2397_, lean_object* v_mvarId_2398_, lean_object* v_inst_2399_, lean_object* v___f_2400_, lean_object* v_____do__lift_2401_, lean_object* v___y_2402_){
_start:
{
lean_object* v_res_2403_; 
v_res_2403_ = lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5(v_inst_2394_, v_inst_2395_, v_inst_2396_, v_g_2397_, v_mvarId_2398_, v_inst_2399_, v___f_2400_, v_____do__lift_2401_, v___y_2402_);
lean_dec(v___y_2402_);
lean_dec_ref(v_____do__lift_2401_);
return v_res_2403_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg(lean_object* v_inst_2404_, lean_object* v_inst_2405_, lean_object* v_inst_2406_, lean_object* v_inst_2407_, lean_object* v_inst_2408_, lean_object* v_mvarId_2409_, lean_object* v_g_2410_, lean_object* v_a_2411_){
_start:
{
lean_object* v___f_2412_; lean_object* v___f_2413_; lean_object* v___x_2414_; lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___f_2417_; lean_object* v___f_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_31__overap_2425_; lean_object* v___x_2426_; 
lean_inc_ref_n(v_inst_2406_, 3);
v___f_2412_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2412_, 0, v_inst_2406_);
lean_inc(v_inst_2408_);
lean_inc_n(v_mvarId_2409_, 2);
v___f_2413_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5___boxed), 9, 7);
lean_closure_set(v___f_2413_, 0, v_inst_2406_);
lean_closure_set(v___f_2413_, 1, v_inst_2404_);
lean_closure_set(v___f_2413_, 2, v_inst_2405_);
lean_closure_set(v___f_2413_, 3, v_g_2410_);
lean_closure_set(v___f_2413_, 4, v_mvarId_2409_);
lean_closure_set(v___f_2413_, 5, v_inst_2408_);
lean_closure_set(v___f_2413_, 6, v___f_2412_);
v___x_2414_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2415_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___x_2416_ = l_Lean_MonadCacheT_instMonadControl___redArg(v_inst_2404_, v___x_2414_, v___x_2415_);
lean_inc_ref(v_inst_2407_);
lean_inc_ref(v___x_2416_);
v___f_2417_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2417_, 0, v___x_2416_);
lean_closure_set(v___f_2417_, 1, v_inst_2407_);
v___f_2418_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2418_, 0, v___x_2416_);
lean_closure_set(v___f_2418_, 1, v_inst_2407_);
v___x_2419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___f_2417_);
lean_ctor_set(v___x_2419_, 1, v___f_2418_);
v___x_2420_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2404_, v___x_2414_, v___x_2415_, v_inst_2406_);
v___x_2421_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2421_, 0, v_mvarId_2409_);
v___x_2422_ = lean_apply_2(v_inst_2408_, lean_box(0), v___x_2421_);
v___x_2423_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonadLift___aux__1___boxed), 10, 9);
lean_closure_set(v___x_2423_, 0, lean_box(0));
lean_closure_set(v___x_2423_, 1, lean_box(0));
lean_closure_set(v___x_2423_, 2, lean_box(0));
lean_closure_set(v___x_2423_, 3, lean_box(0));
lean_closure_set(v___x_2423_, 4, v_inst_2404_);
lean_closure_set(v___x_2423_, 5, v___x_2414_);
lean_closure_set(v___x_2423_, 6, v___x_2415_);
lean_closure_set(v___x_2423_, 7, lean_box(0));
lean_closure_set(v___x_2423_, 8, v___x_2422_);
v___x_2424_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonad___aux__13___boxed), 13, 12);
lean_closure_set(v___x_2424_, 0, lean_box(0));
lean_closure_set(v___x_2424_, 1, lean_box(0));
lean_closure_set(v___x_2424_, 2, lean_box(0));
lean_closure_set(v___x_2424_, 3, lean_box(0));
lean_closure_set(v___x_2424_, 4, v_inst_2404_);
lean_closure_set(v___x_2424_, 5, v___x_2414_);
lean_closure_set(v___x_2424_, 6, v___x_2415_);
lean_closure_set(v___x_2424_, 7, v_inst_2406_);
lean_closure_set(v___x_2424_, 8, lean_box(0));
lean_closure_set(v___x_2424_, 9, lean_box(0));
lean_closure_set(v___x_2424_, 10, v___x_2423_);
lean_closure_set(v___x_2424_, 11, v___f_2413_);
v___x_31__overap_2425_ = l_Lean_MVarId_withContext___redArg(v___x_2419_, v___x_2420_, v_mvarId_2409_, v___x_2424_);
lean_inc(v_a_2411_);
v___x_2426_ = lean_apply_1(v___x_31__overap_2425_, v_a_2411_);
return v___x_2426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___redArg___boxed(lean_object* v_inst_2427_, lean_object* v_inst_2428_, lean_object* v_inst_2429_, lean_object* v_inst_2430_, lean_object* v_inst_2431_, lean_object* v_mvarId_2432_, lean_object* v_g_2433_, lean_object* v_a_2434_){
_start:
{
lean_object* v_res_2435_; 
v_res_2435_ = lp_aesop_Aesop_forEachExprInGoalCore___redArg(v_inst_2427_, v_inst_2428_, v_inst_2429_, v_inst_2430_, v_inst_2431_, v_mvarId_2432_, v_g_2433_, v_a_2434_);
lean_dec(v_a_2434_);
return v_res_2435_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore(lean_object* v_00_u03c9_2436_, lean_object* v_m_2437_, lean_object* v_inst_2438_, lean_object* v_inst_2439_, lean_object* v_inst_2440_, lean_object* v_inst_2441_, lean_object* v_inst_2442_, lean_object* v_mvarId_2443_, lean_object* v_g_2444_, lean_object* v_a_2445_){
_start:
{
lean_object* v___f_2446_; lean_object* v___f_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___f_2451_; lean_object* v___f_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_192__overap_2459_; lean_object* v___x_2460_; 
lean_inc_ref_n(v_inst_2440_, 3);
v___f_2446_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2446_, 0, v_inst_2440_);
lean_inc(v_inst_2442_);
lean_inc_n(v_mvarId_2443_, 2);
v___f_2447_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__5___boxed), 9, 7);
lean_closure_set(v___f_2447_, 0, v_inst_2440_);
lean_closure_set(v___f_2447_, 1, v_inst_2438_);
lean_closure_set(v___f_2447_, 2, v_inst_2439_);
lean_closure_set(v___f_2447_, 3, v_g_2444_);
lean_closure_set(v___f_2447_, 4, v_mvarId_2443_);
lean_closure_set(v___f_2447_, 5, v_inst_2442_);
lean_closure_set(v___f_2447_, 6, v___f_2446_);
v___x_2448_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2449_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___x_2450_ = l_Lean_MonadCacheT_instMonadControl___redArg(v_inst_2438_, v___x_2448_, v___x_2449_);
lean_inc_ref(v_inst_2441_);
lean_inc_ref(v___x_2450_);
v___f_2451_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2451_, 0, v___x_2450_);
lean_closure_set(v___f_2451_, 1, v_inst_2441_);
v___f_2452_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2452_, 0, v___x_2450_);
lean_closure_set(v___f_2452_, 1, v_inst_2441_);
v___x_2453_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2453_, 0, v___f_2451_);
lean_ctor_set(v___x_2453_, 1, v___f_2452_);
v___x_2454_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2438_, v___x_2448_, v___x_2449_, v_inst_2440_);
v___x_2455_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2455_, 0, v_mvarId_2443_);
v___x_2456_ = lean_apply_2(v_inst_2442_, lean_box(0), v___x_2455_);
v___x_2457_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonadLift___aux__1___boxed), 10, 9);
lean_closure_set(v___x_2457_, 0, lean_box(0));
lean_closure_set(v___x_2457_, 1, lean_box(0));
lean_closure_set(v___x_2457_, 2, lean_box(0));
lean_closure_set(v___x_2457_, 3, lean_box(0));
lean_closure_set(v___x_2457_, 4, v_inst_2438_);
lean_closure_set(v___x_2457_, 5, v___x_2448_);
lean_closure_set(v___x_2457_, 6, v___x_2449_);
lean_closure_set(v___x_2457_, 7, lean_box(0));
lean_closure_set(v___x_2457_, 8, v___x_2456_);
v___x_2458_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonad___aux__13___boxed), 13, 12);
lean_closure_set(v___x_2458_, 0, lean_box(0));
lean_closure_set(v___x_2458_, 1, lean_box(0));
lean_closure_set(v___x_2458_, 2, lean_box(0));
lean_closure_set(v___x_2458_, 3, lean_box(0));
lean_closure_set(v___x_2458_, 4, v_inst_2438_);
lean_closure_set(v___x_2458_, 5, v___x_2448_);
lean_closure_set(v___x_2458_, 6, v___x_2449_);
lean_closure_set(v___x_2458_, 7, v_inst_2440_);
lean_closure_set(v___x_2458_, 8, lean_box(0));
lean_closure_set(v___x_2458_, 9, lean_box(0));
lean_closure_set(v___x_2458_, 10, v___x_2457_);
lean_closure_set(v___x_2458_, 11, v___f_2447_);
v___x_192__overap_2459_ = l_Lean_MVarId_withContext___redArg(v___x_2453_, v___x_2454_, v_mvarId_2443_, v___x_2458_);
lean_inc(v_a_2445_);
v___x_2460_ = lean_apply_1(v___x_192__overap_2459_, v_a_2445_);
return v___x_2460_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoalCore___boxed(lean_object* v_00_u03c9_2461_, lean_object* v_m_2462_, lean_object* v_inst_2463_, lean_object* v_inst_2464_, lean_object* v_inst_2465_, lean_object* v_inst_2466_, lean_object* v_inst_2467_, lean_object* v_mvarId_2468_, lean_object* v_g_2469_, lean_object* v_a_2470_){
_start:
{
lean_object* v_res_2471_; 
v_res_2471_ = lp_aesop_Aesop_forEachExprInGoalCore(v_00_u03c9_2461_, v_m_2462_, v_inst_2463_, v_inst_2464_, v_inst_2465_, v_inst_2466_, v_inst_2467_, v_mvarId_2468_, v_g_2469_, v_a_2470_);
lean_dec(v_a_2470_);
return v_res_2471_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5(lean_object* v_inst_2472_, lean_object* v_inst_2473_, lean_object* v_inst_2474_, lean_object* v_g_2475_, lean_object* v_mvarId_2476_, lean_object* v_inst_2477_, lean_object* v_toBind_2478_, lean_object* v_toPure_2479_, lean_object* v___f_2480_, lean_object* v_____do__lift_2481_, lean_object* v___y_2482_){
_start:
{
lean_object* v_lctx_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v_decls_2487_; lean_object* v___f_2488_; lean_object* v___f_2489_; lean_object* v___x_2490_; lean_object* v___f_2491_; lean_object* v___f_2492_; lean_object* v___x_215__overap_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; 
v_lctx_2483_ = lean_ctor_get(v_____do__lift_2481_, 1);
v___x_2484_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2485_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref_n(v_inst_2473_, 2);
v___x_2486_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2472_, v___x_2484_, v___x_2485_, v_inst_2473_);
v_decls_2487_ = lean_ctor_get(v_lctx_2483_, 1);
lean_inc_n(v___y_2482_, 2);
lean_inc(v_g_2475_);
lean_inc(v_inst_2474_);
v___f_2488_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_2488_, 0, v_inst_2474_);
lean_closure_set(v___f_2488_, 1, v_inst_2473_);
lean_closure_set(v___f_2488_, 2, v_g_2475_);
lean_closure_set(v___f_2488_, 3, v___y_2482_);
lean_inc_n(v_toBind_2478_, 3);
v___f_2489_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2489_, 0, v_mvarId_2476_);
lean_closure_set(v___f_2489_, 1, v_inst_2477_);
lean_closure_set(v___f_2489_, 2, v_toBind_2478_);
lean_closure_set(v___f_2489_, 3, v___f_2488_);
v___x_2490_ = lean_box(0);
lean_inc(v_toPure_2479_);
v___f_2491_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4), 3, 2);
lean_closure_set(v___f_2491_, 0, v___x_2490_);
lean_closure_set(v___f_2491_, 1, v_toPure_2479_);
v___f_2492_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__0___boxed), 9, 6);
lean_closure_set(v___f_2492_, 0, v_toPure_2479_);
lean_closure_set(v___f_2492_, 1, v_inst_2474_);
lean_closure_set(v___f_2492_, 2, v_inst_2473_);
lean_closure_set(v___f_2492_, 3, v_g_2475_);
lean_closure_set(v___f_2492_, 4, v_toBind_2478_);
lean_closure_set(v___f_2492_, 5, v___f_2491_);
v___x_215__overap_2493_ = l_Lean_PersistentArray_forIn___redArg(v___x_2486_, v_decls_2487_, v___x_2490_, v___f_2492_);
v___x_2494_ = lean_apply_1(v___x_215__overap_2493_, v___y_2482_);
v___x_2495_ = lean_apply_4(v_toBind_2478_, lean_box(0), lean_box(0), v___x_2494_, v___f_2480_);
v___x_2496_ = lean_apply_4(v_toBind_2478_, lean_box(0), lean_box(0), v___x_2495_, v___f_2489_);
return v___x_2496_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5___boxed(lean_object* v_inst_2497_, lean_object* v_inst_2498_, lean_object* v_inst_2499_, lean_object* v_g_2500_, lean_object* v_mvarId_2501_, lean_object* v_inst_2502_, lean_object* v_toBind_2503_, lean_object* v_toPure_2504_, lean_object* v___f_2505_, lean_object* v_____do__lift_2506_, lean_object* v___y_2507_){
_start:
{
lean_object* v_res_2508_; 
v_res_2508_ = lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5(v_inst_2497_, v_inst_2498_, v_inst_2499_, v_g_2500_, v_mvarId_2501_, v_inst_2502_, v_toBind_2503_, v_toPure_2504_, v___f_2505_, v_____do__lift_2506_, v___y_2507_);
lean_dec(v___y_2507_);
lean_dec_ref(v_____do__lift_2506_);
return v_res_2508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3(lean_object* v_toPure_2509_, lean_object* v_inst_2510_, lean_object* v_toBind_2511_, lean_object* v_inst_2512_, lean_object* v___x_2513_, lean_object* v___x_2514_, lean_object* v_inst_2515_, lean_object* v_inst_2516_, lean_object* v_mvarId_2517_, lean_object* v_inst_2518_, lean_object* v___f_2519_, lean_object* v_ref_2520_){
_start:
{
lean_object* v___f_2521_; lean_object* v___x_2522_; lean_object* v___f_2523_; lean_object* v___f_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_247__overap_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; 
lean_inc(v_toBind_2511_);
lean_inc(v_ref_2520_);
v___f_2521_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2521_, 0, v_toPure_2509_);
lean_closure_set(v___f_2521_, 1, v_ref_2520_);
lean_closure_set(v___f_2521_, 2, v_inst_2510_);
lean_closure_set(v___f_2521_, 3, v_toBind_2511_);
lean_inc_ref_n(v___x_2514_, 3);
lean_inc_ref_n(v___x_2513_, 3);
v___x_2522_ = l_Lean_MonadCacheT_instMonadControl___redArg(v_inst_2512_, v___x_2513_, v___x_2514_);
lean_inc_ref(v_inst_2515_);
lean_inc_ref(v___x_2522_);
v___f_2523_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2523_, 0, v___x_2522_);
lean_closure_set(v___f_2523_, 1, v_inst_2515_);
v___f_2524_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2524_, 0, v___x_2522_);
lean_closure_set(v___f_2524_, 1, v_inst_2515_);
v___x_2525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2525_, 0, v___f_2523_);
lean_ctor_set(v___x_2525_, 1, v___f_2524_);
lean_inc_ref(v_inst_2516_);
v___x_2526_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2512_, v___x_2513_, v___x_2514_, v_inst_2516_);
lean_inc(v_mvarId_2517_);
v___x_2527_ = lean_alloc_closure((void*)(l_Lean_MVarId_getDecl___boxed), 6, 1);
lean_closure_set(v___x_2527_, 0, v_mvarId_2517_);
v___x_2528_ = lean_apply_2(v_inst_2518_, lean_box(0), v___x_2527_);
v___x_2529_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonadLift___aux__1___boxed), 10, 9);
lean_closure_set(v___x_2529_, 0, lean_box(0));
lean_closure_set(v___x_2529_, 1, lean_box(0));
lean_closure_set(v___x_2529_, 2, lean_box(0));
lean_closure_set(v___x_2529_, 3, lean_box(0));
lean_closure_set(v___x_2529_, 4, v_inst_2512_);
lean_closure_set(v___x_2529_, 5, v___x_2513_);
lean_closure_set(v___x_2529_, 6, v___x_2514_);
lean_closure_set(v___x_2529_, 7, lean_box(0));
lean_closure_set(v___x_2529_, 8, v___x_2528_);
v___x_2530_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonad___aux__13___boxed), 13, 12);
lean_closure_set(v___x_2530_, 0, lean_box(0));
lean_closure_set(v___x_2530_, 1, lean_box(0));
lean_closure_set(v___x_2530_, 2, lean_box(0));
lean_closure_set(v___x_2530_, 3, lean_box(0));
lean_closure_set(v___x_2530_, 4, v_inst_2512_);
lean_closure_set(v___x_2530_, 5, v___x_2513_);
lean_closure_set(v___x_2530_, 6, v___x_2514_);
lean_closure_set(v___x_2530_, 7, v_inst_2516_);
lean_closure_set(v___x_2530_, 8, lean_box(0));
lean_closure_set(v___x_2530_, 9, lean_box(0));
lean_closure_set(v___x_2530_, 10, v___x_2529_);
lean_closure_set(v___x_2530_, 11, v___f_2519_);
v___x_247__overap_2531_ = l_Lean_MVarId_withContext___redArg(v___x_2525_, v___x_2526_, v_mvarId_2517_, v___x_2530_);
v___x_2532_ = lean_apply_1(v___x_247__overap_2531_, v_ref_2520_);
v___x_2533_ = lean_apply_4(v_toBind_2511_, lean_box(0), lean_box(0), v___x_2532_, v___f_2521_);
return v___x_2533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27___redArg(lean_object* v_inst_2534_, lean_object* v_inst_2535_, lean_object* v_inst_2536_, lean_object* v_inst_2537_, lean_object* v_inst_2538_, lean_object* v_mvarId_2539_, lean_object* v_g_2540_){
_start:
{
lean_object* v_toApplicative_2541_; lean_object* v_toBind_2542_; lean_object* v_toPure_2543_; lean_object* v___f_2544_; lean_object* v___f_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___f_2550_; lean_object* v___f_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; 
v_toApplicative_2541_ = lean_ctor_get(v_inst_2536_, 0);
v_toBind_2542_ = lean_ctor_get(v_inst_2536_, 1);
lean_inc_n(v_toBind_2542_, 4);
v_toPure_2543_ = lean_ctor_get(v_toApplicative_2541_, 1);
lean_inc_n(v_toPure_2543_, 4);
v___f_2544_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2544_, 0, v_toPure_2543_);
lean_inc(v_inst_2538_);
lean_inc(v_mvarId_2539_);
lean_inc_n(v_inst_2535_, 2);
lean_inc_ref(v_inst_2536_);
v___f_2545_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5___boxed), 11, 9);
lean_closure_set(v___f_2545_, 0, v_inst_2534_);
lean_closure_set(v___f_2545_, 1, v_inst_2536_);
lean_closure_set(v___f_2545_, 2, v_inst_2535_);
lean_closure_set(v___f_2545_, 3, v_g_2540_);
lean_closure_set(v___f_2545_, 4, v_mvarId_2539_);
lean_closure_set(v___f_2545_, 5, v_inst_2538_);
lean_closure_set(v___f_2545_, 6, v_toBind_2542_);
lean_closure_set(v___f_2545_, 7, v_toPure_2543_);
lean_closure_set(v___f_2545_, 8, v___f_2544_);
v___x_2546_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2547_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___x_2548_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2549_ = lean_apply_2(v_inst_2535_, lean_box(0), v___x_2548_);
v___f_2550_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2550_, 0, v_toPure_2543_);
v___f_2551_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3), 12, 11);
lean_closure_set(v___f_2551_, 0, v_toPure_2543_);
lean_closure_set(v___f_2551_, 1, v_inst_2535_);
lean_closure_set(v___f_2551_, 2, v_toBind_2542_);
lean_closure_set(v___f_2551_, 3, v_inst_2534_);
lean_closure_set(v___f_2551_, 4, v___x_2546_);
lean_closure_set(v___f_2551_, 5, v___x_2547_);
lean_closure_set(v___f_2551_, 6, v_inst_2537_);
lean_closure_set(v___f_2551_, 7, v_inst_2536_);
lean_closure_set(v___f_2551_, 8, v_mvarId_2539_);
lean_closure_set(v___f_2551_, 9, v_inst_2538_);
lean_closure_set(v___f_2551_, 10, v___f_2545_);
v___x_2552_ = lean_apply_4(v_toBind_2542_, lean_box(0), lean_box(0), v___x_2549_, v___f_2551_);
v___x_2553_ = lean_apply_4(v_toBind_2542_, lean_box(0), lean_box(0), v___x_2552_, v___f_2550_);
return v___x_2553_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal_x27(lean_object* v_00_u03c9_2554_, lean_object* v_m_2555_, lean_object* v_inst_2556_, lean_object* v_inst_2557_, lean_object* v_inst_2558_, lean_object* v_inst_2559_, lean_object* v_inst_2560_, lean_object* v_mvarId_2561_, lean_object* v_g_2562_){
_start:
{
lean_object* v_toApplicative_2563_; lean_object* v_toBind_2564_; lean_object* v_toPure_2565_; lean_object* v___f_2566_; lean_object* v___f_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___f_2572_; lean_object* v___f_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; 
v_toApplicative_2563_ = lean_ctor_get(v_inst_2558_, 0);
v_toBind_2564_ = lean_ctor_get(v_inst_2558_, 1);
lean_inc_n(v_toBind_2564_, 4);
v_toPure_2565_ = lean_ctor_get(v_toApplicative_2563_, 1);
lean_inc_n(v_toPure_2565_, 4);
v___f_2566_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2566_, 0, v_toPure_2565_);
lean_inc(v_inst_2560_);
lean_inc(v_mvarId_2561_);
lean_inc_n(v_inst_2557_, 2);
lean_inc_ref(v_inst_2558_);
v___f_2567_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__5___boxed), 11, 9);
lean_closure_set(v___f_2567_, 0, v_inst_2556_);
lean_closure_set(v___f_2567_, 1, v_inst_2558_);
lean_closure_set(v___f_2567_, 2, v_inst_2557_);
lean_closure_set(v___f_2567_, 3, v_g_2562_);
lean_closure_set(v___f_2567_, 4, v_mvarId_2561_);
lean_closure_set(v___f_2567_, 5, v_inst_2560_);
lean_closure_set(v___f_2567_, 6, v_toBind_2564_);
lean_closure_set(v___f_2567_, 7, v_toPure_2565_);
lean_closure_set(v___f_2567_, 8, v___f_2566_);
v___x_2568_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2569_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___x_2570_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2571_ = lean_apply_2(v_inst_2557_, lean_box(0), v___x_2570_);
v___f_2572_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2572_, 0, v_toPure_2565_);
v___f_2573_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3), 12, 11);
lean_closure_set(v___f_2573_, 0, v_toPure_2565_);
lean_closure_set(v___f_2573_, 1, v_inst_2557_);
lean_closure_set(v___f_2573_, 2, v_toBind_2564_);
lean_closure_set(v___f_2573_, 3, v_inst_2556_);
lean_closure_set(v___f_2573_, 4, v___x_2568_);
lean_closure_set(v___f_2573_, 5, v___x_2569_);
lean_closure_set(v___f_2573_, 6, v_inst_2559_);
lean_closure_set(v___f_2573_, 7, v_inst_2558_);
lean_closure_set(v___f_2573_, 8, v_mvarId_2561_);
lean_closure_set(v___f_2573_, 9, v_inst_2560_);
lean_closure_set(v___f_2573_, 10, v___f_2567_);
v___x_2574_ = lean_apply_4(v_toBind_2564_, lean_box(0), lean_box(0), v___x_2571_, v___f_2573_);
v___x_2575_ = lean_apply_4(v_toBind_2564_, lean_box(0), lean_box(0), v___x_2574_, v___f_2572_);
return v___x_2575_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3(lean_object* v_inst_2576_, lean_object* v_inst_2577_, lean_object* v___f_2578_, lean_object* v___y_2579_, lean_object* v_a_2580_){
_start:
{
lean_object* v___x_2581_; 
v___x_2581_ = l_Lean_ForEachExpr_visit___redArg(v_inst_2576_, v_inst_2577_, v___f_2578_, v_a_2580_, v___y_2579_);
return v___x_2581_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3___boxed(lean_object* v_inst_2582_, lean_object* v_inst_2583_, lean_object* v___f_2584_, lean_object* v___y_2585_, lean_object* v_a_2586_){
_start:
{
lean_object* v_res_2587_; 
v_res_2587_ = lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3(v_inst_2582_, v_inst_2583_, v___f_2584_, v___y_2585_, v_a_2586_);
lean_dec(v___y_2585_);
return v_res_2587_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4(lean_object* v_inst_2588_, lean_object* v_inst_2589_, lean_object* v_inst_2590_, lean_object* v___f_2591_, lean_object* v_mvarId_2592_, lean_object* v_inst_2593_, lean_object* v_toBind_2594_, lean_object* v_toPure_2595_, lean_object* v___f_2596_, lean_object* v_____do__lift_2597_, lean_object* v___y_2598_){
_start:
{
lean_object* v_lctx_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v_decls_2603_; lean_object* v___f_2604_; lean_object* v___f_2605_; lean_object* v___x_2606_; lean_object* v___f_2607_; lean_object* v___f_2608_; lean_object* v___x_239__overap_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; 
v_lctx_2599_ = lean_ctor_get(v_____do__lift_2597_, 1);
v___x_2600_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2601_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
lean_inc_ref_n(v_inst_2589_, 2);
v___x_2602_ = l_Lean_MonadCacheT_instMonad___redArg(v_inst_2588_, v___x_2600_, v___x_2601_, v_inst_2589_);
v_decls_2603_ = lean_ctor_get(v_lctx_2599_, 1);
lean_inc_n(v___y_2598_, 2);
lean_inc(v___f_2591_);
lean_inc(v_inst_2590_);
v___f_2604_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal___redArg___lam__3___boxed), 5, 4);
lean_closure_set(v___f_2604_, 0, v_inst_2590_);
lean_closure_set(v___f_2604_, 1, v_inst_2589_);
lean_closure_set(v___f_2604_, 2, v___f_2591_);
lean_closure_set(v___f_2604_, 3, v___y_2598_);
lean_inc_n(v_toBind_2594_, 3);
v___f_2605_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoalCore___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2605_, 0, v_mvarId_2592_);
lean_closure_set(v___f_2605_, 1, v_inst_2593_);
lean_closure_set(v___f_2605_, 2, v_toBind_2594_);
lean_closure_set(v___f_2605_, 3, v___f_2604_);
v___x_2606_ = lean_box(0);
lean_inc(v_toPure_2595_);
v___f_2607_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx_x27___redArg___lam__4), 3, 2);
lean_closure_set(v___f_2607_, 0, v___x_2606_);
lean_closure_set(v___f_2607_, 1, v_toPure_2595_);
v___f_2608_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__7___boxed), 9, 6);
lean_closure_set(v___f_2608_, 0, v_toPure_2595_);
lean_closure_set(v___f_2608_, 1, v_inst_2590_);
lean_closure_set(v___f_2608_, 2, v_inst_2589_);
lean_closure_set(v___f_2608_, 3, v___f_2591_);
lean_closure_set(v___f_2608_, 4, v_toBind_2594_);
lean_closure_set(v___f_2608_, 5, v___f_2607_);
v___x_239__overap_2609_ = l_Lean_PersistentArray_forIn___redArg(v___x_2602_, v_decls_2603_, v___x_2606_, v___f_2608_);
v___x_2610_ = lean_apply_1(v___x_239__overap_2609_, v___y_2598_);
v___x_2611_ = lean_apply_4(v_toBind_2594_, lean_box(0), lean_box(0), v___x_2610_, v___f_2596_);
v___x_2612_ = lean_apply_4(v_toBind_2594_, lean_box(0), lean_box(0), v___x_2611_, v___f_2605_);
return v___x_2612_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4___boxed(lean_object* v_inst_2613_, lean_object* v_inst_2614_, lean_object* v_inst_2615_, lean_object* v___f_2616_, lean_object* v_mvarId_2617_, lean_object* v_inst_2618_, lean_object* v_toBind_2619_, lean_object* v_toPure_2620_, lean_object* v___f_2621_, lean_object* v_____do__lift_2622_, lean_object* v___y_2623_){
_start:
{
lean_object* v_res_2624_; 
v_res_2624_ = lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4(v_inst_2613_, v_inst_2614_, v_inst_2615_, v___f_2616_, v_mvarId_2617_, v_inst_2618_, v_toBind_2619_, v_toPure_2620_, v___f_2621_, v_____do__lift_2622_, v___y_2623_);
lean_dec(v___y_2623_);
lean_dec_ref(v_____do__lift_2622_);
return v_res_2624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal___redArg(lean_object* v_inst_2625_, lean_object* v_inst_2626_, lean_object* v_inst_2627_, lean_object* v_inst_2628_, lean_object* v_inst_2629_, lean_object* v_mvarId_2630_, lean_object* v_g_2631_){
_start:
{
lean_object* v_toApplicative_2632_; lean_object* v_toBind_2633_; lean_object* v_toPure_2634_; lean_object* v___f_2635_; lean_object* v___f_2636_; lean_object* v___f_2637_; lean_object* v___f_2638_; lean_object* v___f_2639_; lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___f_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; 
v_toApplicative_2632_ = lean_ctor_get(v_inst_2627_, 0);
v_toBind_2633_ = lean_ctor_get(v_inst_2627_, 1);
lean_inc_n(v_toBind_2633_, 5);
v_toPure_2634_ = lean_ctor_get(v_toApplicative_2632_, 1);
lean_inc_n(v_toPure_2634_, 5);
v___f_2635_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2635_, 0, v_toPure_2634_);
v___f_2636_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2636_, 0, v_toPure_2634_);
v___f_2637_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2637_, 0, v_g_2631_);
lean_closure_set(v___f_2637_, 1, v_toBind_2633_);
lean_closure_set(v___f_2637_, 2, v___f_2636_);
lean_inc(v_inst_2629_);
lean_inc(v_mvarId_2630_);
lean_inc_n(v_inst_2626_, 2);
lean_inc_ref(v_inst_2627_);
v___f_2638_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4___boxed), 11, 9);
lean_closure_set(v___f_2638_, 0, v_inst_2625_);
lean_closure_set(v___f_2638_, 1, v_inst_2627_);
lean_closure_set(v___f_2638_, 2, v_inst_2626_);
lean_closure_set(v___f_2638_, 3, v___f_2637_);
lean_closure_set(v___f_2638_, 4, v_mvarId_2630_);
lean_closure_set(v___f_2638_, 5, v_inst_2629_);
lean_closure_set(v___f_2638_, 6, v_toBind_2633_);
lean_closure_set(v___f_2638_, 7, v_toPure_2634_);
lean_closure_set(v___f_2638_, 8, v___f_2635_);
v___f_2639_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2639_, 0, v_toPure_2634_);
v___x_2640_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2641_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___f_2642_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3), 12, 11);
lean_closure_set(v___f_2642_, 0, v_toPure_2634_);
lean_closure_set(v___f_2642_, 1, v_inst_2626_);
lean_closure_set(v___f_2642_, 2, v_toBind_2633_);
lean_closure_set(v___f_2642_, 3, v_inst_2625_);
lean_closure_set(v___f_2642_, 4, v___x_2640_);
lean_closure_set(v___f_2642_, 5, v___x_2641_);
lean_closure_set(v___f_2642_, 6, v_inst_2628_);
lean_closure_set(v___f_2642_, 7, v_inst_2627_);
lean_closure_set(v___f_2642_, 8, v_mvarId_2630_);
lean_closure_set(v___f_2642_, 9, v_inst_2629_);
lean_closure_set(v___f_2642_, 10, v___f_2638_);
v___x_2643_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2644_ = lean_apply_2(v_inst_2626_, lean_box(0), v___x_2643_);
v___x_2645_ = lean_apply_4(v_toBind_2633_, lean_box(0), lean_box(0), v___x_2644_, v___f_2642_);
v___x_2646_ = lean_apply_4(v_toBind_2633_, lean_box(0), lean_box(0), v___x_2645_, v___f_2639_);
return v___x_2646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forEachExprInGoal(lean_object* v_00_u03c9_2647_, lean_object* v_m_2648_, lean_object* v_inst_2649_, lean_object* v_inst_2650_, lean_object* v_inst_2651_, lean_object* v_inst_2652_, lean_object* v_inst_2653_, lean_object* v_mvarId_2654_, lean_object* v_g_2655_){
_start:
{
lean_object* v_toApplicative_2656_; lean_object* v_toBind_2657_; lean_object* v_toPure_2658_; lean_object* v___f_2659_; lean_object* v___f_2660_; lean_object* v___f_2661_; lean_object* v___f_2662_; lean_object* v___f_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___f_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; 
v_toApplicative_2656_ = lean_ctor_get(v_inst_2651_, 0);
v_toBind_2657_ = lean_ctor_get(v_inst_2651_, 1);
lean_inc_n(v_toBind_2657_, 5);
v_toPure_2658_ = lean_ctor_get(v_toApplicative_2656_, 1);
lean_inc_n(v_toPure_2658_, 5);
v___f_2659_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLCtx___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2659_, 0, v_toPure_2658_);
v___f_2660_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2660_, 0, v_toPure_2658_);
v___f_2661_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2661_, 0, v_g_2655_);
lean_closure_set(v___f_2661_, 1, v_toBind_2657_);
lean_closure_set(v___f_2661_, 2, v___f_2660_);
lean_inc(v_inst_2653_);
lean_inc(v_mvarId_2654_);
lean_inc_n(v_inst_2650_, 2);
lean_inc_ref(v_inst_2651_);
v___f_2662_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal___redArg___lam__4___boxed), 11, 9);
lean_closure_set(v___f_2662_, 0, v_inst_2649_);
lean_closure_set(v___f_2662_, 1, v_inst_2651_);
lean_closure_set(v___f_2662_, 2, v_inst_2650_);
lean_closure_set(v___f_2662_, 3, v___f_2661_);
lean_closure_set(v___f_2662_, 4, v_mvarId_2654_);
lean_closure_set(v___f_2662_, 5, v_inst_2653_);
lean_closure_set(v___f_2662_, 6, v_toBind_2657_);
lean_closure_set(v___f_2662_, 7, v_toPure_2658_);
lean_closure_set(v___f_2662_, 8, v___f_2659_);
v___f_2663_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2663_, 0, v_toPure_2658_);
v___x_2664_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__0));
v___x_2665_ = ((lean_object*)(lp_aesop_Aesop_forEachExprInLCtxCore___redArg___closed__1));
v___f_2666_ = lean_alloc_closure((void*)(lp_aesop_Aesop_forEachExprInGoal_x27___redArg___lam__3), 12, 11);
lean_closure_set(v___f_2666_, 0, v_toPure_2658_);
lean_closure_set(v___f_2666_, 1, v_inst_2650_);
lean_closure_set(v___f_2666_, 2, v_toBind_2657_);
lean_closure_set(v___f_2666_, 3, v_inst_2649_);
lean_closure_set(v___f_2666_, 4, v___x_2664_);
lean_closure_set(v___f_2666_, 5, v___x_2665_);
lean_closure_set(v___f_2666_, 6, v_inst_2652_);
lean_closure_set(v___f_2666_, 7, v_inst_2651_);
lean_closure_set(v___f_2666_, 8, v_mvarId_2654_);
lean_closure_set(v___f_2666_, 9, v_inst_2653_);
lean_closure_set(v___f_2666_, 10, v___f_2662_);
v___x_2667_ = lean_obj_once(&lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2, &lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_forEachExprInLDecl_x27___redArg___closed__2);
v___x_2668_ = lean_apply_2(v_inst_2650_, lean_box(0), v___x_2667_);
v___x_2669_ = lean_apply_4(v_toBind_2657_, lean_box(0), lean_box(0), v___x_2668_, v___f_2666_);
v___x_2670_ = lean_apply_4(v_toBind_2657_, lean_box(0), lean_box(0), v___x_2669_, v___f_2663_);
return v___x_2670_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setThe___redArg(lean_object* v_inst_2671_, lean_object* v_s_2672_){
_start:
{
lean_object* v_set_2673_; lean_object* v___x_2674_; 
v_set_2673_ = lean_ctor_get(v_inst_2671_, 1);
lean_inc(v_set_2673_);
lean_dec_ref(v_inst_2671_);
v___x_2674_ = lean_apply_1(v_set_2673_, v_s_2672_);
return v___x_2674_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setThe(lean_object* v_00_u03c3_2675_, lean_object* v_m_2676_, lean_object* v_inst_2677_, lean_object* v_s_2678_){
_start:
{
lean_object* v_set_2679_; lean_object* v___x_2680_; 
v_set_2679_ = lean_ctor_get(v_inst_2677_, 1);
lean_inc(v_set_2679_);
lean_dec_ref(v_inst_2677_);
v___x_2680_ = lean_apply_1(v_set_2679_, v_s_2678_);
return v___x_2680_;
}
}
static uint64_t _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1(void){
_start:
{
lean_object* v___x_2687_; uint64_t v___x_2688_; 
v___x_2687_ = ((lean_object*)(lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__0));
v___x_2688_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_2687_);
return v___x_2688_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2(void){
_start:
{
uint64_t v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; 
v___x_2689_ = lean_uint64_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__1);
v___x_2690_ = ((lean_object*)(lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__0));
v___x_2691_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_2691_, 0, v___x_2690_);
lean_ctor_set_uint64(v___x_2691_, sizeof(void*)*1, v___x_2689_);
return v___x_2691_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3(void){
_start:
{
lean_object* v___x_2692_; 
v___x_2692_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2692_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4(void){
_start:
{
lean_object* v___x_2693_; lean_object* v___x_2694_; 
v___x_2693_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__3);
v___x_2694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2694_, 0, v___x_2693_);
return v___x_2694_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5(void){
_start:
{
lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; 
v___x_2695_ = lean_unsigned_to_nat(32u);
v___x_2696_ = lean_mk_empty_array_with_capacity(v___x_2695_);
v___x_2697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2697_, 0, v___x_2696_);
return v___x_2697_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6(void){
_start:
{
size_t v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; 
v___x_2698_ = ((size_t)5ULL);
v___x_2699_ = lean_unsigned_to_nat(0u);
v___x_2700_ = lean_unsigned_to_nat(32u);
v___x_2701_ = lean_mk_empty_array_with_capacity(v___x_2700_);
v___x_2702_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__5);
v___x_2703_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2703_, 0, v___x_2702_);
lean_ctor_set(v___x_2703_, 1, v___x_2701_);
lean_ctor_set(v___x_2703_, 2, v___x_2699_);
lean_ctor_set(v___x_2703_, 3, v___x_2699_);
lean_ctor_set_usize(v___x_2703_, 4, v___x_2698_);
return v___x_2703_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7(void){
_start:
{
lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; 
v___x_2704_ = lean_box(1);
v___x_2705_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6);
v___x_2706_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4);
v___x_2707_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2707_, 0, v___x_2706_);
lean_ctor_set(v___x_2707_, 1, v___x_2705_);
lean_ctor_set(v___x_2707_, 2, v___x_2704_);
return v___x_2707_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9(void){
_start:
{
uint8_t v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; uint8_t v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; 
v___x_2710_ = 1;
v___x_2711_ = lean_unsigned_to_nat(0u);
v___x_2712_ = lean_box(0);
v___x_2713_ = ((lean_object*)(lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__8));
v___x_2714_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7);
v___x_2715_ = lean_box(1);
v___x_2716_ = 0;
v___x_2717_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2);
v___x_2718_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2718_, 0, v___x_2717_);
lean_ctor_set(v___x_2718_, 1, v___x_2715_);
lean_ctor_set(v___x_2718_, 2, v___x_2714_);
lean_ctor_set(v___x_2718_, 3, v___x_2713_);
lean_ctor_set(v___x_2718_, 4, v___x_2712_);
lean_ctor_set(v___x_2718_, 5, v___x_2711_);
lean_ctor_set(v___x_2718_, 6, v___x_2712_);
lean_ctor_set_uint8(v___x_2718_, sizeof(void*)*7, v___x_2716_);
lean_ctor_set_uint8(v___x_2718_, sizeof(void*)*7 + 1, v___x_2716_);
lean_ctor_set_uint8(v___x_2718_, sizeof(void*)*7 + 2, v___x_2716_);
lean_ctor_set_uint8(v___x_2718_, sizeof(void*)*7 + 3, v___x_2710_);
return v___x_2718_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10(void){
_start:
{
lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; 
v___x_2719_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4);
v___x_2720_ = lean_unsigned_to_nat(0u);
v___x_2721_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2721_, 0, v___x_2720_);
lean_ctor_set(v___x_2721_, 1, v___x_2720_);
lean_ctor_set(v___x_2721_, 2, v___x_2720_);
lean_ctor_set(v___x_2721_, 3, v___x_2720_);
lean_ctor_set(v___x_2721_, 4, v___x_2719_);
lean_ctor_set(v___x_2721_, 5, v___x_2719_);
lean_ctor_set(v___x_2721_, 6, v___x_2719_);
lean_ctor_set(v___x_2721_, 7, v___x_2719_);
lean_ctor_set(v___x_2721_, 8, v___x_2719_);
lean_ctor_set(v___x_2721_, 9, v___x_2719_);
return v___x_2721_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11(void){
_start:
{
lean_object* v___x_2722_; lean_object* v___x_2723_; 
v___x_2722_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4);
v___x_2723_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2723_, 0, v___x_2722_);
lean_ctor_set(v___x_2723_, 1, v___x_2722_);
lean_ctor_set(v___x_2723_, 2, v___x_2722_);
lean_ctor_set(v___x_2723_, 3, v___x_2722_);
lean_ctor_set(v___x_2723_, 4, v___x_2722_);
lean_ctor_set(v___x_2723_, 5, v___x_2722_);
return v___x_2723_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12(void){
_start:
{
lean_object* v___x_2724_; lean_object* v___x_2725_; 
v___x_2724_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__4);
v___x_2725_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2725_, 0, v___x_2724_);
lean_ctor_set(v___x_2725_, 1, v___x_2724_);
lean_ctor_set(v___x_2725_, 2, v___x_2724_);
lean_ctor_set(v___x_2725_, 3, v___x_2724_);
lean_ctor_set(v___x_2725_, 4, v___x_2724_);
return v___x_2725_;
}
}
static lean_object* _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13(void){
_start:
{
lean_object* v___x_2726_; lean_object* v___x_2727_; lean_object* v___x_2728_; lean_object* v___x_2729_; lean_object* v___x_2730_; lean_object* v___x_2731_; 
v___x_2726_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__12);
v___x_2727_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__6);
v___x_2728_ = lean_box(1);
v___x_2729_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__11);
v___x_2730_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__10);
v___x_2731_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2731_, 0, v___x_2730_);
lean_ctor_set(v___x_2731_, 1, v___x_2729_);
lean_ctor_set(v___x_2731_, 2, v___x_2728_);
lean_ctor_set(v___x_2731_, 3, v___x_2727_);
lean_ctor_set(v___x_2731_, 4, v___x_2726_);
return v___x_2731_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg(lean_object* v_x_2732_, lean_object* v_a_2733_, lean_object* v_a_2734_){
_start:
{
lean_object* v___x_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; 
v___x_2736_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9);
v___x_2737_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13);
v___x_2738_ = lean_st_mk_ref(v___x_2737_);
lean_inc(v_a_2734_);
lean_inc_ref(v_a_2733_);
lean_inc(v___x_2738_);
v___x_2739_ = lean_apply_5(v_x_2732_, v___x_2736_, v___x_2738_, v_a_2733_, v_a_2734_, lean_box(0));
if (lean_obj_tag(v___x_2739_) == 0)
{
lean_object* v_a_2740_; lean_object* v___x_2742_; uint8_t v_isShared_2743_; uint8_t v_isSharedCheck_2748_; 
v_a_2740_ = lean_ctor_get(v___x_2739_, 0);
v_isSharedCheck_2748_ = !lean_is_exclusive(v___x_2739_);
if (v_isSharedCheck_2748_ == 0)
{
v___x_2742_ = v___x_2739_;
v_isShared_2743_ = v_isSharedCheck_2748_;
goto v_resetjp_2741_;
}
else
{
lean_inc(v_a_2740_);
lean_dec(v___x_2739_);
v___x_2742_ = lean_box(0);
v_isShared_2743_ = v_isSharedCheck_2748_;
goto v_resetjp_2741_;
}
v_resetjp_2741_:
{
lean_object* v___x_2744_; lean_object* v___x_2746_; 
v___x_2744_ = lean_st_ref_get(v___x_2738_);
lean_dec(v___x_2738_);
lean_dec(v___x_2744_);
if (v_isShared_2743_ == 0)
{
v___x_2746_ = v___x_2742_;
goto v_reusejp_2745_;
}
else
{
lean_object* v_reuseFailAlloc_2747_; 
v_reuseFailAlloc_2747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2747_, 0, v_a_2740_);
v___x_2746_ = v_reuseFailAlloc_2747_;
goto v_reusejp_2745_;
}
v_reusejp_2745_:
{
return v___x_2746_;
}
}
}
else
{
lean_dec(v___x_2738_);
return v___x_2739_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___redArg___boxed(lean_object* v_x_2749_, lean_object* v_a_2750_, lean_object* v_a_2751_, lean_object* v_a_2752_){
_start:
{
lean_object* v_res_2753_; 
v_res_2753_ = lp_aesop_Aesop_runMetaMAsCoreM___redArg(v_x_2749_, v_a_2750_, v_a_2751_);
lean_dec(v_a_2751_);
lean_dec_ref(v_a_2750_);
return v_res_2753_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM(lean_object* v_00_u03b1_2754_, lean_object* v_x_2755_, lean_object* v_a_2756_, lean_object* v_a_2757_){
_start:
{
lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; 
v___x_2759_ = lean_unsigned_to_nat(32u);
v___x_2760_ = lean_mk_empty_array_with_capacity(v___x_2759_);
lean_dec_ref(v___x_2760_);
v___x_2761_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__9);
v___x_2762_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13);
v___x_2763_ = lean_st_mk_ref(v___x_2762_);
lean_inc(v_a_2757_);
lean_inc_ref(v_a_2756_);
lean_inc(v___x_2763_);
v___x_2764_ = lean_apply_5(v_x_2755_, v___x_2761_, v___x_2763_, v_a_2756_, v_a_2757_, lean_box(0));
if (lean_obj_tag(v___x_2764_) == 0)
{
lean_object* v_a_2765_; lean_object* v___x_2767_; uint8_t v_isShared_2768_; uint8_t v_isSharedCheck_2773_; 
v_a_2765_ = lean_ctor_get(v___x_2764_, 0);
v_isSharedCheck_2773_ = !lean_is_exclusive(v___x_2764_);
if (v_isSharedCheck_2773_ == 0)
{
v___x_2767_ = v___x_2764_;
v_isShared_2768_ = v_isSharedCheck_2773_;
goto v_resetjp_2766_;
}
else
{
lean_inc(v_a_2765_);
lean_dec(v___x_2764_);
v___x_2767_ = lean_box(0);
v_isShared_2768_ = v_isSharedCheck_2773_;
goto v_resetjp_2766_;
}
v_resetjp_2766_:
{
lean_object* v___x_2769_; lean_object* v___x_2771_; 
v___x_2769_ = lean_st_ref_get(v___x_2763_);
lean_dec(v___x_2763_);
lean_dec(v___x_2769_);
if (v_isShared_2768_ == 0)
{
v___x_2771_ = v___x_2767_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2772_; 
v_reuseFailAlloc_2772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2772_, 0, v_a_2765_);
v___x_2771_ = v_reuseFailAlloc_2772_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
return v___x_2771_;
}
}
}
else
{
lean_dec(v___x_2763_);
return v___x_2764_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runMetaMAsCoreM___boxed(lean_object* v_00_u03b1_2774_, lean_object* v_x_2775_, lean_object* v_a_2776_, lean_object* v_a_2777_, lean_object* v_a_2778_){
_start:
{
lean_object* v_res_2779_; 
v_res_2779_ = lp_aesop_Aesop_runMetaMAsCoreM(v_00_u03b1_2774_, v_x_2775_, v_a_2776_, v_a_2777_);
lean_dec(v_a_2777_);
lean_dec_ref(v_a_2776_);
return v_res_2779_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0(lean_object* v_x_2780_){
_start:
{
uint8_t v___x_2781_; 
v___x_2781_ = 0;
return v___x_2781_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0___boxed(lean_object* v_x_2782_){
_start:
{
uint8_t v_res_2783_; lean_object* v_r_2784_; 
v_res_2783_ = lp_aesop_Aesop_runTermElabMAsCoreM___redArg___lam__0(v_x_2782_);
lean_dec(v_x_2782_);
v_r_2784_ = lean_box(v_res_2783_);
return v_r_2784_;
}
}
static lean_object* _init_lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2(void){
_start:
{
uint8_t v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; uint8_t v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; 
v___x_2794_ = 1;
v___x_2795_ = lean_unsigned_to_nat(0u);
v___x_2796_ = lean_box(0);
v___x_2797_ = ((lean_object*)(lp_aesop_Aesop_PersistentHashSet_toArray___redArg___closed__1));
v___x_2798_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__7);
v___x_2799_ = lean_box(1);
v___x_2800_ = 0;
v___x_2801_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__2);
v___x_2802_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2802_, 0, v___x_2801_);
lean_ctor_set(v___x_2802_, 1, v___x_2799_);
lean_ctor_set(v___x_2802_, 2, v___x_2798_);
lean_ctor_set(v___x_2802_, 3, v___x_2797_);
lean_ctor_set(v___x_2802_, 4, v___x_2796_);
lean_ctor_set(v___x_2802_, 5, v___x_2795_);
lean_ctor_set(v___x_2802_, 6, v___x_2796_);
lean_ctor_set_uint8(v___x_2802_, sizeof(void*)*7, v___x_2800_);
lean_ctor_set_uint8(v___x_2802_, sizeof(void*)*7 + 1, v___x_2800_);
lean_ctor_set_uint8(v___x_2802_, sizeof(void*)*7 + 2, v___x_2800_);
lean_ctor_set_uint8(v___x_2802_, sizeof(void*)*7 + 3, v___x_2794_);
return v___x_2802_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg(lean_object* v_x_2806_, lean_object* v_a_2807_, lean_object* v_a_2808_){
_start:
{
lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; 
v___x_2810_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1));
v___x_2811_ = lean_obj_once(&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2, &lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2_once, _init_lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2);
v___x_2812_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13);
v___x_2813_ = lean_st_mk_ref(v___x_2812_);
v___x_2814_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3));
v___x_2815_ = l_Lean_Elab_Term_TermElabM_run___redArg(v_x_2806_, v___x_2810_, v___x_2814_, v___x_2811_, v___x_2813_, v_a_2807_, v_a_2808_);
if (lean_obj_tag(v___x_2815_) == 0)
{
lean_object* v_a_2816_; lean_object* v___x_2818_; uint8_t v_isShared_2819_; uint8_t v_isSharedCheck_2825_; 
v_a_2816_ = lean_ctor_get(v___x_2815_, 0);
v_isSharedCheck_2825_ = !lean_is_exclusive(v___x_2815_);
if (v_isSharedCheck_2825_ == 0)
{
v___x_2818_ = v___x_2815_;
v_isShared_2819_ = v_isSharedCheck_2825_;
goto v_resetjp_2817_;
}
else
{
lean_inc(v_a_2816_);
lean_dec(v___x_2815_);
v___x_2818_ = lean_box(0);
v_isShared_2819_ = v_isSharedCheck_2825_;
goto v_resetjp_2817_;
}
v_resetjp_2817_:
{
lean_object* v___x_2820_; lean_object* v_fst_2821_; lean_object* v___x_2823_; 
v___x_2820_ = lean_st_ref_get(v___x_2813_);
lean_dec(v___x_2813_);
lean_dec(v___x_2820_);
v_fst_2821_ = lean_ctor_get(v_a_2816_, 0);
lean_inc(v_fst_2821_);
lean_dec(v_a_2816_);
if (v_isShared_2819_ == 0)
{
lean_ctor_set(v___x_2818_, 0, v_fst_2821_);
v___x_2823_ = v___x_2818_;
goto v_reusejp_2822_;
}
else
{
lean_object* v_reuseFailAlloc_2824_; 
v_reuseFailAlloc_2824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2824_, 0, v_fst_2821_);
v___x_2823_ = v_reuseFailAlloc_2824_;
goto v_reusejp_2822_;
}
v_reusejp_2822_:
{
return v___x_2823_;
}
}
}
else
{
lean_object* v_a_2826_; lean_object* v___x_2828_; uint8_t v_isShared_2829_; uint8_t v_isSharedCheck_2833_; 
lean_dec(v___x_2813_);
v_a_2826_ = lean_ctor_get(v___x_2815_, 0);
v_isSharedCheck_2833_ = !lean_is_exclusive(v___x_2815_);
if (v_isSharedCheck_2833_ == 0)
{
v___x_2828_ = v___x_2815_;
v_isShared_2829_ = v_isSharedCheck_2833_;
goto v_resetjp_2827_;
}
else
{
lean_inc(v_a_2826_);
lean_dec(v___x_2815_);
v___x_2828_ = lean_box(0);
v_isShared_2829_ = v_isSharedCheck_2833_;
goto v_resetjp_2827_;
}
v_resetjp_2827_:
{
lean_object* v___x_2831_; 
if (v_isShared_2829_ == 0)
{
v___x_2831_ = v___x_2828_;
goto v_reusejp_2830_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v_a_2826_);
v___x_2831_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2830_;
}
v_reusejp_2830_:
{
return v___x_2831_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___redArg___boxed(lean_object* v_x_2834_, lean_object* v_a_2835_, lean_object* v_a_2836_, lean_object* v_a_2837_){
_start:
{
lean_object* v_res_2838_; 
v_res_2838_ = lp_aesop_Aesop_runTermElabMAsCoreM___redArg(v_x_2834_, v_a_2835_, v_a_2836_);
lean_dec(v_a_2836_);
lean_dec_ref(v_a_2835_);
return v_res_2838_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM(lean_object* v_00_u03b1_2839_, lean_object* v_x_2840_, lean_object* v_a_2841_, lean_object* v_a_2842_){
_start:
{
lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; 
v___x_2844_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1));
v___x_2845_ = lean_unsigned_to_nat(32u);
v___x_2846_ = lean_mk_empty_array_with_capacity(v___x_2845_);
lean_dec_ref(v___x_2846_);
v___x_2847_ = lean_obj_once(&lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2, &lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2_once, _init_lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__2);
v___x_2848_ = lean_obj_once(&lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13, &lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13_once, _init_lp_aesop_Aesop_runMetaMAsCoreM___redArg___closed__13);
v___x_2849_ = lean_st_mk_ref(v___x_2848_);
v___x_2850_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3));
v___x_2851_ = l_Lean_Elab_Term_TermElabM_run___redArg(v_x_2840_, v___x_2844_, v___x_2850_, v___x_2847_, v___x_2849_, v_a_2841_, v_a_2842_);
if (lean_obj_tag(v___x_2851_) == 0)
{
lean_object* v_a_2852_; lean_object* v___x_2854_; uint8_t v_isShared_2855_; uint8_t v_isSharedCheck_2861_; 
v_a_2852_ = lean_ctor_get(v___x_2851_, 0);
v_isSharedCheck_2861_ = !lean_is_exclusive(v___x_2851_);
if (v_isSharedCheck_2861_ == 0)
{
v___x_2854_ = v___x_2851_;
v_isShared_2855_ = v_isSharedCheck_2861_;
goto v_resetjp_2853_;
}
else
{
lean_inc(v_a_2852_);
lean_dec(v___x_2851_);
v___x_2854_ = lean_box(0);
v_isShared_2855_ = v_isSharedCheck_2861_;
goto v_resetjp_2853_;
}
v_resetjp_2853_:
{
lean_object* v___x_2856_; lean_object* v_fst_2857_; lean_object* v___x_2859_; 
v___x_2856_ = lean_st_ref_get(v___x_2849_);
lean_dec(v___x_2849_);
lean_dec(v___x_2856_);
v_fst_2857_ = lean_ctor_get(v_a_2852_, 0);
lean_inc(v_fst_2857_);
lean_dec(v_a_2852_);
if (v_isShared_2855_ == 0)
{
lean_ctor_set(v___x_2854_, 0, v_fst_2857_);
v___x_2859_ = v___x_2854_;
goto v_reusejp_2858_;
}
else
{
lean_object* v_reuseFailAlloc_2860_; 
v_reuseFailAlloc_2860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2860_, 0, v_fst_2857_);
v___x_2859_ = v_reuseFailAlloc_2860_;
goto v_reusejp_2858_;
}
v_reusejp_2858_:
{
return v___x_2859_;
}
}
}
else
{
lean_object* v_a_2862_; lean_object* v___x_2864_; uint8_t v_isShared_2865_; uint8_t v_isSharedCheck_2869_; 
lean_dec(v___x_2849_);
v_a_2862_ = lean_ctor_get(v___x_2851_, 0);
v_isSharedCheck_2869_ = !lean_is_exclusive(v___x_2851_);
if (v_isSharedCheck_2869_ == 0)
{
v___x_2864_ = v___x_2851_;
v_isShared_2865_ = v_isSharedCheck_2869_;
goto v_resetjp_2863_;
}
else
{
lean_inc(v_a_2862_);
lean_dec(v___x_2851_);
v___x_2864_ = lean_box(0);
v_isShared_2865_ = v_isSharedCheck_2869_;
goto v_resetjp_2863_;
}
v_resetjp_2863_:
{
lean_object* v___x_2867_; 
if (v_isShared_2865_ == 0)
{
v___x_2867_ = v___x_2864_;
goto v_reusejp_2866_;
}
else
{
lean_object* v_reuseFailAlloc_2868_; 
v_reuseFailAlloc_2868_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2868_, 0, v_a_2862_);
v___x_2867_ = v_reuseFailAlloc_2868_;
goto v_reusejp_2866_;
}
v_reusejp_2866_:
{
return v___x_2867_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTermElabMAsCoreM___boxed(lean_object* v_00_u03b1_2870_, lean_object* v_x_2871_, lean_object* v_a_2872_, lean_object* v_a_2873_, lean_object* v_a_2874_){
_start:
{
lean_object* v_res_2875_; 
v_res_2875_ = lp_aesop_Aesop_runTermElabMAsCoreM(v_00_u03b1_2870_, v_x_2871_, v_a_2872_, v_a_2873_);
lean_dec(v_a_2873_);
lean_dec_ref(v_a_2872_);
return v_res_2875_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1(lean_object* v_goals_2876_, lean_object* v_x_2877_, lean_object* v___x_2878_, lean_object* v___y_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_){
_start:
{
lean_object* v___x_2886_; lean_object* v___x_2887_; 
v___x_2886_ = lean_st_mk_ref(v_goals_2876_);
lean_inc(v___x_2886_);
v___x_2887_ = lean_apply_9(v_x_2877_, v___x_2878_, v___x_2886_, v___y_2879_, v___y_2880_, v___y_2881_, v___y_2882_, v___y_2883_, v___y_2884_, lean_box(0));
if (lean_obj_tag(v___x_2887_) == 0)
{
lean_object* v_a_2888_; lean_object* v___x_2890_; uint8_t v_isShared_2891_; uint8_t v_isSharedCheck_2897_; 
v_a_2888_ = lean_ctor_get(v___x_2887_, 0);
v_isSharedCheck_2897_ = !lean_is_exclusive(v___x_2887_);
if (v_isSharedCheck_2897_ == 0)
{
v___x_2890_ = v___x_2887_;
v_isShared_2891_ = v_isSharedCheck_2897_;
goto v_resetjp_2889_;
}
else
{
lean_inc(v_a_2888_);
lean_dec(v___x_2887_);
v___x_2890_ = lean_box(0);
v_isShared_2891_ = v_isSharedCheck_2897_;
goto v_resetjp_2889_;
}
v_resetjp_2889_:
{
lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2895_; 
v___x_2892_ = lean_st_ref_get(v___x_2886_);
lean_dec(v___x_2886_);
v___x_2893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2893_, 0, v_a_2888_);
lean_ctor_set(v___x_2893_, 1, v___x_2892_);
if (v_isShared_2891_ == 0)
{
lean_ctor_set(v___x_2890_, 0, v___x_2893_);
v___x_2895_ = v___x_2890_;
goto v_reusejp_2894_;
}
else
{
lean_object* v_reuseFailAlloc_2896_; 
v_reuseFailAlloc_2896_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2896_, 0, v___x_2893_);
v___x_2895_ = v_reuseFailAlloc_2896_;
goto v_reusejp_2894_;
}
v_reusejp_2894_:
{
return v___x_2895_;
}
}
}
else
{
lean_object* v_a_2898_; lean_object* v___x_2900_; uint8_t v_isShared_2901_; uint8_t v_isSharedCheck_2905_; 
lean_dec(v___x_2886_);
v_a_2898_ = lean_ctor_get(v___x_2887_, 0);
v_isSharedCheck_2905_ = !lean_is_exclusive(v___x_2887_);
if (v_isSharedCheck_2905_ == 0)
{
v___x_2900_ = v___x_2887_;
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
else
{
lean_inc(v_a_2898_);
lean_dec(v___x_2887_);
v___x_2900_ = lean_box(0);
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
v_resetjp_2899_:
{
lean_object* v___x_2903_; 
if (v_isShared_2901_ == 0)
{
v___x_2903_ = v___x_2900_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2904_; 
v_reuseFailAlloc_2904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2904_, 0, v_a_2898_);
v___x_2903_ = v_reuseFailAlloc_2904_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
return v___x_2903_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1___boxed(lean_object* v_goals_2906_, lean_object* v_x_2907_, lean_object* v___x_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_, lean_object* v___y_2914_, lean_object* v___y_2915_){
_start:
{
lean_object* v_res_2916_; 
v_res_2916_ = lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1(v_goals_2906_, v_x_2907_, v___x_2908_, v___y_2909_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_, v___y_2914_);
return v_res_2916_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg(lean_object* v_x_2920_, lean_object* v_goals_2921_, lean_object* v_a_2922_, lean_object* v_a_2923_, lean_object* v_a_2924_, lean_object* v_a_2925_){
_start:
{
lean_object* v___x_2927_; lean_object* v___f_2928_; lean_object* v___x_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; 
v___x_2927_ = ((lean_object*)(lp_aesop_Aesop_runTacticMAsMetaM___redArg___closed__0));
v___f_2928_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticMAsMetaM___redArg___lam__1___boxed), 10, 3);
lean_closure_set(v___f_2928_, 0, v_goals_2921_);
lean_closure_set(v___f_2928_, 1, v_x_2920_);
lean_closure_set(v___f_2928_, 2, v___x_2927_);
v___x_2929_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__1));
v___x_2930_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3));
v___x_2931_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_2928_, v___x_2929_, v___x_2930_, v_a_2922_, v_a_2923_, v_a_2924_, v_a_2925_);
if (lean_obj_tag(v___x_2931_) == 0)
{
lean_object* v_a_2932_; lean_object* v___x_2934_; uint8_t v_isShared_2935_; uint8_t v_isSharedCheck_2949_; 
v_a_2932_ = lean_ctor_get(v___x_2931_, 0);
v_isSharedCheck_2949_ = !lean_is_exclusive(v___x_2931_);
if (v_isSharedCheck_2949_ == 0)
{
v___x_2934_ = v___x_2931_;
v_isShared_2935_ = v_isSharedCheck_2949_;
goto v_resetjp_2933_;
}
else
{
lean_inc(v_a_2932_);
lean_dec(v___x_2931_);
v___x_2934_ = lean_box(0);
v_isShared_2935_ = v_isSharedCheck_2949_;
goto v_resetjp_2933_;
}
v_resetjp_2933_:
{
lean_object* v_fst_2936_; lean_object* v_fst_2937_; lean_object* v_snd_2938_; lean_object* v___x_2940_; uint8_t v_isShared_2941_; uint8_t v_isSharedCheck_2948_; 
v_fst_2936_ = lean_ctor_get(v_a_2932_, 0);
lean_inc(v_fst_2936_);
lean_dec(v_a_2932_);
v_fst_2937_ = lean_ctor_get(v_fst_2936_, 0);
v_snd_2938_ = lean_ctor_get(v_fst_2936_, 1);
v_isSharedCheck_2948_ = !lean_is_exclusive(v_fst_2936_);
if (v_isSharedCheck_2948_ == 0)
{
v___x_2940_ = v_fst_2936_;
v_isShared_2941_ = v_isSharedCheck_2948_;
goto v_resetjp_2939_;
}
else
{
lean_inc(v_snd_2938_);
lean_inc(v_fst_2937_);
lean_dec(v_fst_2936_);
v___x_2940_ = lean_box(0);
v_isShared_2941_ = v_isSharedCheck_2948_;
goto v_resetjp_2939_;
}
v_resetjp_2939_:
{
lean_object* v___x_2943_; 
if (v_isShared_2941_ == 0)
{
v___x_2943_ = v___x_2940_;
goto v_reusejp_2942_;
}
else
{
lean_object* v_reuseFailAlloc_2947_; 
v_reuseFailAlloc_2947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2947_, 0, v_fst_2937_);
lean_ctor_set(v_reuseFailAlloc_2947_, 1, v_snd_2938_);
v___x_2943_ = v_reuseFailAlloc_2947_;
goto v_reusejp_2942_;
}
v_reusejp_2942_:
{
lean_object* v___x_2945_; 
if (v_isShared_2935_ == 0)
{
lean_ctor_set(v___x_2934_, 0, v___x_2943_);
v___x_2945_ = v___x_2934_;
goto v_reusejp_2944_;
}
else
{
lean_object* v_reuseFailAlloc_2946_; 
v_reuseFailAlloc_2946_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2946_, 0, v___x_2943_);
v___x_2945_ = v_reuseFailAlloc_2946_;
goto v_reusejp_2944_;
}
v_reusejp_2944_:
{
return v___x_2945_;
}
}
}
}
}
else
{
lean_object* v_a_2950_; lean_object* v___x_2952_; uint8_t v_isShared_2953_; uint8_t v_isSharedCheck_2957_; 
v_a_2950_ = lean_ctor_get(v___x_2931_, 0);
v_isSharedCheck_2957_ = !lean_is_exclusive(v___x_2931_);
if (v_isSharedCheck_2957_ == 0)
{
v___x_2952_ = v___x_2931_;
v_isShared_2953_ = v_isSharedCheck_2957_;
goto v_resetjp_2951_;
}
else
{
lean_inc(v_a_2950_);
lean_dec(v___x_2931_);
v___x_2952_ = lean_box(0);
v_isShared_2953_ = v_isSharedCheck_2957_;
goto v_resetjp_2951_;
}
v_resetjp_2951_:
{
lean_object* v___x_2955_; 
if (v_isShared_2953_ == 0)
{
v___x_2955_ = v___x_2952_;
goto v_reusejp_2954_;
}
else
{
lean_object* v_reuseFailAlloc_2956_; 
v_reuseFailAlloc_2956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2956_, 0, v_a_2950_);
v___x_2955_ = v_reuseFailAlloc_2956_;
goto v_reusejp_2954_;
}
v_reusejp_2954_:
{
return v___x_2955_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___redArg___boxed(lean_object* v_x_2958_, lean_object* v_goals_2959_, lean_object* v_a_2960_, lean_object* v_a_2961_, lean_object* v_a_2962_, lean_object* v_a_2963_, lean_object* v_a_2964_){
_start:
{
lean_object* v_res_2965_; 
v_res_2965_ = lp_aesop_Aesop_runTacticMAsMetaM___redArg(v_x_2958_, v_goals_2959_, v_a_2960_, v_a_2961_, v_a_2962_, v_a_2963_);
lean_dec(v_a_2963_);
lean_dec_ref(v_a_2962_);
lean_dec(v_a_2961_);
lean_dec_ref(v_a_2960_);
return v_res_2965_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM(lean_object* v_00_u03b1_2966_, lean_object* v_x_2967_, lean_object* v_goals_2968_, lean_object* v_a_2969_, lean_object* v_a_2970_, lean_object* v_a_2971_, lean_object* v_a_2972_){
_start:
{
lean_object* v___x_2974_; 
v___x_2974_ = lp_aesop_Aesop_runTacticMAsMetaM___redArg(v_x_2967_, v_goals_2968_, v_a_2969_, v_a_2970_, v_a_2971_, v_a_2972_);
return v___x_2974_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsMetaM___boxed(lean_object* v_00_u03b1_2975_, lean_object* v_x_2976_, lean_object* v_goals_2977_, lean_object* v_a_2978_, lean_object* v_a_2979_, lean_object* v_a_2980_, lean_object* v_a_2981_, lean_object* v_a_2982_){
_start:
{
lean_object* v_res_2983_; 
v_res_2983_ = lp_aesop_Aesop_runTacticMAsMetaM(v_00_u03b1_2975_, v_x_2976_, v_goals_2977_, v_a_2978_, v_a_2979_, v_a_2980_, v_a_2981_);
lean_dec(v_a_2981_);
lean_dec_ref(v_a_2980_);
lean_dec(v_a_2979_);
lean_dec_ref(v_a_2978_);
return v_res_2983_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSyntaxAsMetaM(lean_object* v_stx_2984_, lean_object* v_goals_2985_, lean_object* v_a_2986_, lean_object* v_a_2987_, lean_object* v_a_2988_, lean_object* v_a_2989_){
_start:
{
lean_object* v___x_2991_; lean_object* v___x_2992_; 
v___x_2991_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_2991_, 0, v_stx_2984_);
v___x_2992_ = lp_aesop_Aesop_runTacticMAsMetaM___redArg(v___x_2991_, v_goals_2985_, v_a_2986_, v_a_2987_, v_a_2988_, v_a_2989_);
if (lean_obj_tag(v___x_2992_) == 0)
{
lean_object* v_a_2993_; lean_object* v___x_2995_; uint8_t v_isShared_2996_; uint8_t v_isSharedCheck_3001_; 
v_a_2993_ = lean_ctor_get(v___x_2992_, 0);
v_isSharedCheck_3001_ = !lean_is_exclusive(v___x_2992_);
if (v_isSharedCheck_3001_ == 0)
{
v___x_2995_ = v___x_2992_;
v_isShared_2996_ = v_isSharedCheck_3001_;
goto v_resetjp_2994_;
}
else
{
lean_inc(v_a_2993_);
lean_dec(v___x_2992_);
v___x_2995_ = lean_box(0);
v_isShared_2996_ = v_isSharedCheck_3001_;
goto v_resetjp_2994_;
}
v_resetjp_2994_:
{
lean_object* v_snd_2997_; lean_object* v___x_2999_; 
v_snd_2997_ = lean_ctor_get(v_a_2993_, 1);
lean_inc(v_snd_2997_);
lean_dec(v_a_2993_);
if (v_isShared_2996_ == 0)
{
lean_ctor_set(v___x_2995_, 0, v_snd_2997_);
v___x_2999_ = v___x_2995_;
goto v_reusejp_2998_;
}
else
{
lean_object* v_reuseFailAlloc_3000_; 
v_reuseFailAlloc_3000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3000_, 0, v_snd_2997_);
v___x_2999_ = v_reuseFailAlloc_3000_;
goto v_reusejp_2998_;
}
v_reusejp_2998_:
{
return v___x_2999_;
}
}
}
else
{
lean_object* v_a_3002_; lean_object* v___x_3004_; uint8_t v_isShared_3005_; uint8_t v_isSharedCheck_3009_; 
v_a_3002_ = lean_ctor_get(v___x_2992_, 0);
v_isSharedCheck_3009_ = !lean_is_exclusive(v___x_2992_);
if (v_isSharedCheck_3009_ == 0)
{
v___x_3004_ = v___x_2992_;
v_isShared_3005_ = v_isSharedCheck_3009_;
goto v_resetjp_3003_;
}
else
{
lean_inc(v_a_3002_);
lean_dec(v___x_2992_);
v___x_3004_ = lean_box(0);
v_isShared_3005_ = v_isSharedCheck_3009_;
goto v_resetjp_3003_;
}
v_resetjp_3003_:
{
lean_object* v___x_3007_; 
if (v_isShared_3005_ == 0)
{
v___x_3007_ = v___x_3004_;
goto v_reusejp_3006_;
}
else
{
lean_object* v_reuseFailAlloc_3008_; 
v_reuseFailAlloc_3008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3008_, 0, v_a_3002_);
v___x_3007_ = v_reuseFailAlloc_3008_;
goto v_reusejp_3006_;
}
v_reusejp_3006_:
{
return v___x_3007_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSyntaxAsMetaM___boxed(lean_object* v_stx_3010_, lean_object* v_goals_3011_, lean_object* v_a_3012_, lean_object* v_a_3013_, lean_object* v_a_3014_, lean_object* v_a_3015_, lean_object* v_a_3016_){
_start:
{
lean_object* v_res_3017_; 
v_res_3017_ = lp_aesop_Aesop_runTacticSyntaxAsMetaM(v_stx_3010_, v_goals_3011_, v_a_3012_, v_a_3013_, v_a_3014_, v_a_3015_);
lean_dec(v_a_3015_);
lean_dec_ref(v_a_3014_);
lean_dec(v_a_3013_);
lean_dec_ref(v_a_3012_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_updateSimpEntryPriority(lean_object* v_priority_3018_, lean_object* v_e_3019_){
_start:
{
if (lean_obj_tag(v_e_3019_) == 0)
{
lean_object* v_a_3020_; lean_object* v___x_3022_; uint8_t v_isShared_3023_; uint8_t v_isSharedCheck_3043_; 
v_a_3020_ = lean_ctor_get(v_e_3019_, 0);
v_isSharedCheck_3043_ = !lean_is_exclusive(v_e_3019_);
if (v_isSharedCheck_3043_ == 0)
{
v___x_3022_ = v_e_3019_;
v_isShared_3023_ = v_isSharedCheck_3043_;
goto v_resetjp_3021_;
}
else
{
lean_inc(v_a_3020_);
lean_dec(v_e_3019_);
v___x_3022_ = lean_box(0);
v_isShared_3023_ = v_isSharedCheck_3043_;
goto v_resetjp_3021_;
}
v_resetjp_3021_:
{
lean_object* v_keys_3024_; lean_object* v_levelParams_3025_; lean_object* v_proof_3026_; uint8_t v_post_3027_; uint8_t v_perm_3028_; lean_object* v_origin_3029_; uint8_t v_rfl_3030_; uint8_t v_backwardRfl_3031_; lean_object* v___x_3033_; uint8_t v_isShared_3034_; uint8_t v_isSharedCheck_3041_; 
v_keys_3024_ = lean_ctor_get(v_a_3020_, 0);
v_levelParams_3025_ = lean_ctor_get(v_a_3020_, 1);
v_proof_3026_ = lean_ctor_get(v_a_3020_, 2);
v_post_3027_ = lean_ctor_get_uint8(v_a_3020_, sizeof(void*)*5);
v_perm_3028_ = lean_ctor_get_uint8(v_a_3020_, sizeof(void*)*5 + 1);
v_origin_3029_ = lean_ctor_get(v_a_3020_, 4);
v_rfl_3030_ = lean_ctor_get_uint8(v_a_3020_, sizeof(void*)*5 + 2);
v_backwardRfl_3031_ = lean_ctor_get_uint8(v_a_3020_, sizeof(void*)*5 + 3);
v_isSharedCheck_3041_ = !lean_is_exclusive(v_a_3020_);
if (v_isSharedCheck_3041_ == 0)
{
lean_object* v_unused_3042_; 
v_unused_3042_ = lean_ctor_get(v_a_3020_, 3);
lean_dec(v_unused_3042_);
v___x_3033_ = v_a_3020_;
v_isShared_3034_ = v_isSharedCheck_3041_;
goto v_resetjp_3032_;
}
else
{
lean_inc(v_origin_3029_);
lean_inc(v_proof_3026_);
lean_inc(v_levelParams_3025_);
lean_inc(v_keys_3024_);
lean_dec(v_a_3020_);
v___x_3033_ = lean_box(0);
v_isShared_3034_ = v_isSharedCheck_3041_;
goto v_resetjp_3032_;
}
v_resetjp_3032_:
{
lean_object* v___x_3036_; 
if (v_isShared_3034_ == 0)
{
lean_ctor_set(v___x_3033_, 3, v_priority_3018_);
v___x_3036_ = v___x_3033_;
goto v_reusejp_3035_;
}
else
{
lean_object* v_reuseFailAlloc_3040_; 
v_reuseFailAlloc_3040_ = lean_alloc_ctor(0, 5, 4);
lean_ctor_set(v_reuseFailAlloc_3040_, 0, v_keys_3024_);
lean_ctor_set(v_reuseFailAlloc_3040_, 1, v_levelParams_3025_);
lean_ctor_set(v_reuseFailAlloc_3040_, 2, v_proof_3026_);
lean_ctor_set(v_reuseFailAlloc_3040_, 3, v_priority_3018_);
lean_ctor_set(v_reuseFailAlloc_3040_, 4, v_origin_3029_);
lean_ctor_set_uint8(v_reuseFailAlloc_3040_, sizeof(void*)*5, v_post_3027_);
lean_ctor_set_uint8(v_reuseFailAlloc_3040_, sizeof(void*)*5 + 1, v_perm_3028_);
lean_ctor_set_uint8(v_reuseFailAlloc_3040_, sizeof(void*)*5 + 2, v_rfl_3030_);
lean_ctor_set_uint8(v_reuseFailAlloc_3040_, sizeof(void*)*5 + 3, v_backwardRfl_3031_);
v___x_3036_ = v_reuseFailAlloc_3040_;
goto v_reusejp_3035_;
}
v_reusejp_3035_:
{
lean_object* v___x_3038_; 
if (v_isShared_3023_ == 0)
{
lean_ctor_set(v___x_3022_, 0, v___x_3036_);
v___x_3038_ = v___x_3022_;
goto v_reusejp_3037_;
}
else
{
lean_object* v_reuseFailAlloc_3039_; 
v_reuseFailAlloc_3039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3039_, 0, v___x_3036_);
v___x_3038_ = v_reuseFailAlloc_3039_;
goto v_reusejp_3037_;
}
v_reusejp_3037_:
{
return v___x_3038_;
}
}
}
}
}
else
{
lean_dec(v_priority_3018_);
return v_e_3019_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_3044_, lean_object* v_vals_3045_, lean_object* v_i_3046_, lean_object* v_k_3047_){
_start:
{
lean_object* v___x_3048_; uint8_t v___x_3049_; 
v___x_3048_ = lean_array_get_size(v_keys_3044_);
v___x_3049_ = lean_nat_dec_lt(v_i_3046_, v___x_3048_);
if (v___x_3049_ == 0)
{
lean_object* v___x_3050_; 
lean_dec(v_i_3046_);
v___x_3050_ = lean_box(0);
return v___x_3050_;
}
else
{
lean_object* v_k_x27_3051_; uint8_t v___x_3052_; 
v_k_x27_3051_ = lean_array_fget_borrowed(v_keys_3044_, v_i_3046_);
v___x_3052_ = l_Lean_instBEqMVarId_beq(v_k_3047_, v_k_x27_3051_);
if (v___x_3052_ == 0)
{
lean_object* v___x_3053_; lean_object* v___x_3054_; 
v___x_3053_ = lean_unsigned_to_nat(1u);
v___x_3054_ = lean_nat_add(v_i_3046_, v___x_3053_);
lean_dec(v_i_3046_);
v_i_3046_ = v___x_3054_;
goto _start;
}
else
{
lean_object* v___x_3056_; lean_object* v___x_3057_; 
v___x_3056_ = lean_array_fget_borrowed(v_vals_3045_, v_i_3046_);
lean_dec(v_i_3046_);
lean_inc(v___x_3056_);
v___x_3057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3057_, 0, v___x_3056_);
return v___x_3057_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_3058_, lean_object* v_vals_3059_, lean_object* v_i_3060_, lean_object* v_k_3061_){
_start:
{
lean_object* v_res_3062_; 
v_res_3062_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg(v_keys_3058_, v_vals_3059_, v_i_3060_, v_k_3061_);
lean_dec(v_k_3061_);
lean_dec_ref(v_vals_3059_);
lean_dec_ref(v_keys_3058_);
return v_res_3062_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg(lean_object* v_x_3063_, size_t v_x_3064_, lean_object* v_x_3065_){
_start:
{
if (lean_obj_tag(v_x_3063_) == 0)
{
lean_object* v_es_3066_; lean_object* v___x_3067_; size_t v___x_3068_; size_t v___x_3069_; lean_object* v_j_3070_; lean_object* v___x_3071_; 
v_es_3066_ = lean_ctor_get(v_x_3063_, 0);
v___x_3067_ = lean_box(2);
v___x_3068_ = ((size_t)31ULL);
v___x_3069_ = lean_usize_land(v_x_3064_, v___x_3068_);
v_j_3070_ = lean_usize_to_nat(v___x_3069_);
v___x_3071_ = lean_array_get_borrowed(v___x_3067_, v_es_3066_, v_j_3070_);
lean_dec(v_j_3070_);
switch(lean_obj_tag(v___x_3071_))
{
case 0:
{
lean_object* v_key_3072_; lean_object* v_val_3073_; uint8_t v___x_3074_; 
v_key_3072_ = lean_ctor_get(v___x_3071_, 0);
v_val_3073_ = lean_ctor_get(v___x_3071_, 1);
v___x_3074_ = l_Lean_instBEqMVarId_beq(v_x_3065_, v_key_3072_);
if (v___x_3074_ == 0)
{
lean_object* v___x_3075_; 
v___x_3075_ = lean_box(0);
return v___x_3075_;
}
else
{
lean_object* v___x_3076_; 
lean_inc(v_val_3073_);
v___x_3076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3076_, 0, v_val_3073_);
return v___x_3076_;
}
}
case 1:
{
lean_object* v_node_3077_; size_t v___x_3078_; size_t v___x_3079_; 
v_node_3077_ = lean_ctor_get(v___x_3071_, 0);
v___x_3078_ = ((size_t)5ULL);
v___x_3079_ = lean_usize_shift_right(v_x_3064_, v___x_3078_);
v_x_3063_ = v_node_3077_;
v_x_3064_ = v___x_3079_;
goto _start;
}
default: 
{
lean_object* v___x_3081_; 
v___x_3081_ = lean_box(0);
return v___x_3081_;
}
}
}
else
{
lean_object* v_ks_3082_; lean_object* v_vs_3083_; lean_object* v___x_3084_; lean_object* v___x_3085_; 
v_ks_3082_ = lean_ctor_get(v_x_3063_, 0);
v_vs_3083_ = lean_ctor_get(v_x_3063_, 1);
v___x_3084_ = lean_unsigned_to_nat(0u);
v___x_3085_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg(v_ks_3082_, v_vs_3083_, v___x_3084_, v_x_3065_);
return v___x_3085_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg___boxed(lean_object* v_x_3086_, lean_object* v_x_3087_, lean_object* v_x_3088_){
_start:
{
size_t v_x_280__boxed_3089_; lean_object* v_res_3090_; 
v_x_280__boxed_3089_ = lean_unbox_usize(v_x_3087_);
lean_dec(v_x_3087_);
v_res_3090_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg(v_x_3086_, v_x_280__boxed_3089_, v_x_3088_);
lean_dec(v_x_3088_);
lean_dec_ref(v_x_3086_);
return v_res_3090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg(lean_object* v_x_3091_, lean_object* v_x_3092_){
_start:
{
uint64_t v___x_3093_; size_t v___x_3094_; lean_object* v___x_3095_; 
v___x_3093_ = l_Lean_instHashableMVarId_hash(v_x_3092_);
v___x_3094_ = lean_uint64_to_usize(v___x_3093_);
v___x_3095_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg(v_x_3091_, v___x_3094_, v_x_3092_);
return v___x_3095_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg___boxed(lean_object* v_x_3096_, lean_object* v_x_3097_){
_start:
{
lean_object* v_res_3098_; 
v_res_3098_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg(v_x_3096_, v_x_3097_);
lean_dec(v_x_3097_);
lean_dec_ref(v_x_3096_);
return v_res_3098_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0(lean_object* v_mctx_3099_, lean_object* v_e_3100_){
_start:
{
uint8_t v___x_3101_; 
v___x_3101_ = l_Lean_Expr_isSorry(v_e_3100_);
if (v___x_3101_ == 0)
{
if (lean_obj_tag(v_e_3100_) == 2)
{
lean_object* v_mvarId_3102_; lean_object* v___x_3103_; 
v_mvarId_3102_ = lean_ctor_get(v_e_3100_, 0);
v___x_3103_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_3099_, v_mvarId_3102_);
if (lean_obj_tag(v___x_3103_) == 1)
{
lean_object* v_val_3104_; uint8_t v___x_3105_; 
v_val_3104_ = lean_ctor_get(v___x_3103_, 0);
lean_inc(v_val_3104_);
lean_dec_ref_known(v___x_3103_, 1);
v___x_3105_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(v_mctx_3099_, v_val_3104_);
lean_dec(v_val_3104_);
return v___x_3105_;
}
else
{
lean_object* v_dAssignment_3106_; lean_object* v___x_3107_; 
lean_dec(v___x_3103_);
v_dAssignment_3106_ = lean_ctor_get(v_mctx_3099_, 9);
v___x_3107_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg(v_dAssignment_3106_, v_mvarId_3102_);
if (lean_obj_tag(v___x_3107_) == 1)
{
lean_object* v_val_3108_; lean_object* v_mvarIdPending_3109_; lean_object* v___x_3110_; uint8_t v___x_3111_; 
v_val_3108_ = lean_ctor_get(v___x_3107_, 0);
lean_inc(v_val_3108_);
lean_dec_ref_known(v___x_3107_, 1);
v_mvarIdPending_3109_ = lean_ctor_get(v_val_3108_, 1);
lean_inc(v_mvarIdPending_3109_);
lean_dec(v_val_3108_);
v___x_3110_ = l_Lean_Expr_mvar___override(v_mvarIdPending_3109_);
v___x_3111_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(v_mctx_3099_, v___x_3110_);
lean_dec_ref(v___x_3110_);
return v___x_3111_;
}
else
{
lean_dec(v___x_3107_);
lean_dec_ref(v_mctx_3099_);
return v___x_3101_;
}
}
}
else
{
lean_dec_ref(v_mctx_3099_);
return v___x_3101_;
}
}
else
{
lean_dec_ref(v_mctx_3099_);
return v___x_3101_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0___boxed(lean_object* v_mctx_3112_, lean_object* v_e_3113_){
_start:
{
uint8_t v_res_3114_; lean_object* v_r_3115_; 
v_res_3114_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0(v_mctx_3112_, v_e_3113_);
lean_dec_ref(v_e_3113_);
v_r_3115_ = lean_box(v_res_3114_);
return v_r_3115_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(lean_object* v_mctx_3116_, lean_object* v_e_3117_){
_start:
{
lean_object* v___f_3118_; lean_object* v___x_3119_; 
v___f_3118_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3118_, 0, v_mctx_3116_);
v___x_3119_ = lean_find_expr(v___f_3118_, v_e_3117_);
lean_dec_ref(v___f_3118_);
if (lean_obj_tag(v___x_3119_) == 0)
{
uint8_t v___x_3120_; 
v___x_3120_ = 0;
return v___x_3120_;
}
else
{
uint8_t v___x_3121_; 
lean_dec_ref_known(v___x_3119_, 1);
v___x_3121_ = 1;
return v___x_3121_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go___boxed(lean_object* v_mctx_3122_, lean_object* v_e_3123_){
_start:
{
uint8_t v_res_3124_; lean_object* v_r_3125_; 
v_res_3124_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(v_mctx_3122_, v_e_3123_);
lean_dec_ref(v_e_3123_);
v_r_3125_ = lean_box(v_res_3124_);
return v_r_3125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0(lean_object* v_00_u03b2_3126_, lean_object* v_x_3127_, lean_object* v_x_3128_){
_start:
{
lean_object* v___x_3129_; 
v___x_3129_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___redArg(v_x_3127_, v_x_3128_);
return v___x_3129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0___boxed(lean_object* v_00_u03b2_3130_, lean_object* v_x_3131_, lean_object* v_x_3132_){
_start:
{
lean_object* v_res_3133_; 
v_res_3133_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0(v_00_u03b2_3130_, v_x_3131_, v_x_3132_);
lean_dec(v_x_3132_);
lean_dec_ref(v_x_3131_);
return v_res_3133_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0(lean_object* v_00_u03b2_3134_, lean_object* v_x_3135_, size_t v_x_3136_, lean_object* v_x_3137_){
_start:
{
lean_object* v___x_3138_; 
v___x_3138_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___redArg(v_x_3135_, v_x_3136_, v_x_3137_);
return v___x_3138_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0___boxed(lean_object* v_00_u03b2_3139_, lean_object* v_x_3140_, lean_object* v_x_3141_, lean_object* v_x_3142_){
_start:
{
size_t v_x_374__boxed_3143_; lean_object* v_res_3144_; 
v_x_374__boxed_3143_ = lean_unbox_usize(v_x_3141_);
lean_dec(v_x_3141_);
v_res_3144_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0(v_00_u03b2_3139_, v_x_3140_, v_x_374__boxed_3143_, v_x_3142_);
lean_dec(v_x_3142_);
lean_dec_ref(v_x_3140_);
return v_res_3144_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_3145_, lean_object* v_keys_3146_, lean_object* v_vals_3147_, lean_object* v_heq_3148_, lean_object* v_i_3149_, lean_object* v_k_3150_){
_start:
{
lean_object* v___x_3151_; 
v___x_3151_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___redArg(v_keys_3146_, v_vals_3147_, v_i_3149_, v_k_3150_);
return v___x_3151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_3152_, lean_object* v_keys_3153_, lean_object* v_vals_3154_, lean_object* v_heq_3155_, lean_object* v_i_3156_, lean_object* v_k_3157_){
_start:
{
lean_object* v_res_3158_; 
v_res_3158_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_hasSorry_go_spec__0_spec__0_spec__1(v_00_u03b2_3152_, v_keys_3153_, v_vals_3154_, v_heq_3155_, v_i_3156_, v_k_3157_);
lean_dec(v_k_3157_);
lean_dec_ref(v_vals_3154_);
lean_dec_ref(v_keys_3153_);
return v_res_3158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg___lam__0(lean_object* v_e_3159_, lean_object* v_toPure_3160_, lean_object* v_____do__lift_3161_){
_start:
{
uint8_t v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; 
v___x_3162_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(v_____do__lift_3161_, v_e_3159_);
v___x_3163_ = lean_box(v___x_3162_);
v___x_3164_ = lean_apply_2(v_toPure_3160_, lean_box(0), v___x_3163_);
return v___x_3164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg___lam__0___boxed(lean_object* v_e_3165_, lean_object* v_toPure_3166_, lean_object* v_____do__lift_3167_){
_start:
{
lean_object* v_res_3168_; 
v_res_3168_ = lp_aesop_Aesop_hasSorry___redArg___lam__0(v_e_3165_, v_toPure_3166_, v_____do__lift_3167_);
lean_dec_ref(v_e_3165_);
return v_res_3168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___redArg(lean_object* v_inst_3169_, lean_object* v_inst_3170_, lean_object* v_e_3171_){
_start:
{
lean_object* v_toApplicative_3172_; lean_object* v_toBind_3173_; lean_object* v_getMCtx_3174_; lean_object* v_toPure_3175_; lean_object* v___f_3176_; lean_object* v___x_3177_; 
v_toApplicative_3172_ = lean_ctor_get(v_inst_3169_, 0);
lean_inc_ref(v_toApplicative_3172_);
v_toBind_3173_ = lean_ctor_get(v_inst_3169_, 1);
lean_inc(v_toBind_3173_);
lean_dec_ref(v_inst_3169_);
v_getMCtx_3174_ = lean_ctor_get(v_inst_3170_, 0);
lean_inc(v_getMCtx_3174_);
lean_dec_ref(v_inst_3170_);
v_toPure_3175_ = lean_ctor_get(v_toApplicative_3172_, 1);
lean_inc(v_toPure_3175_);
lean_dec_ref(v_toApplicative_3172_);
v___f_3176_ = lean_alloc_closure((void*)(lp_aesop_Aesop_hasSorry___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3176_, 0, v_e_3171_);
lean_closure_set(v___f_3176_, 1, v_toPure_3175_);
v___x_3177_ = lean_apply_4(v_toBind_3173_, lean_box(0), lean_box(0), v_getMCtx_3174_, v___f_3176_);
return v___x_3177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry(lean_object* v_m_3178_, lean_object* v_inst_3179_, lean_object* v_inst_3180_, lean_object* v_e_3181_){
_start:
{
lean_object* v___x_3182_; 
v___x_3182_ = lp_aesop_Aesop_hasSorry___redArg(v_inst_3179_, v_inst_3180_, v_e_3181_);
return v___x_3182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___lam__0(lean_object* v_f_3183_, lean_object* v_e_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_){
_start:
{
lean_object* v___x_3190_; 
lean_inc(v___y_3188_);
lean_inc_ref(v___y_3187_);
lean_inc(v___y_3186_);
lean_inc_ref(v___y_3185_);
lean_inc_ref(v_f_3183_);
v___x_3190_ = lean_infer_type(v_f_3183_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
if (lean_obj_tag(v___x_3190_) == 0)
{
lean_object* v_a_3191_; uint8_t v___x_3192_; lean_object* v___x_3193_; 
v_a_3191_ = lean_ctor_get(v___x_3190_, 0);
lean_inc(v_a_3191_);
lean_dec_ref_known(v___x_3190_, 1);
v___x_3192_ = 0;
v___x_3193_ = l_Lean_Meta_forallMetaTelescope(v_a_3191_, v___x_3192_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
if (lean_obj_tag(v___x_3193_) == 0)
{
lean_object* v_a_3194_; lean_object* v_fst_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; 
v_a_3194_ = lean_ctor_get(v___x_3193_, 0);
lean_inc(v_a_3194_);
lean_dec_ref_known(v___x_3193_, 1);
v_fst_3195_ = lean_ctor_get(v_a_3194_, 0);
lean_inc(v_fst_3195_);
lean_dec(v_a_3194_);
v___x_3196_ = l_Lean_mkAppN(v_f_3183_, v_fst_3195_);
lean_dec(v_fst_3195_);
v___x_3197_ = l_Lean_Meta_isExprDefEq(v___x_3196_, v_e_3184_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
return v___x_3197_;
}
else
{
lean_object* v_a_3198_; lean_object* v___x_3200_; uint8_t v_isShared_3201_; uint8_t v_isSharedCheck_3205_; 
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec_ref(v_e_3184_);
lean_dec_ref(v_f_3183_);
v_a_3198_ = lean_ctor_get(v___x_3193_, 0);
v_isSharedCheck_3205_ = !lean_is_exclusive(v___x_3193_);
if (v_isSharedCheck_3205_ == 0)
{
v___x_3200_ = v___x_3193_;
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
else
{
lean_inc(v_a_3198_);
lean_dec(v___x_3193_);
v___x_3200_ = lean_box(0);
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
v_resetjp_3199_:
{
lean_object* v___x_3203_; 
if (v_isShared_3201_ == 0)
{
v___x_3203_ = v___x_3200_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3204_; 
v_reuseFailAlloc_3204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3204_, 0, v_a_3198_);
v___x_3203_ = v_reuseFailAlloc_3204_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
return v___x_3203_;
}
}
}
}
else
{
lean_object* v_a_3206_; lean_object* v___x_3208_; uint8_t v_isShared_3209_; uint8_t v_isSharedCheck_3213_; 
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec_ref(v_e_3184_);
lean_dec_ref(v_f_3183_);
v_a_3206_ = lean_ctor_get(v___x_3190_, 0);
v_isSharedCheck_3213_ = !lean_is_exclusive(v___x_3190_);
if (v_isSharedCheck_3213_ == 0)
{
v___x_3208_ = v___x_3190_;
v_isShared_3209_ = v_isSharedCheck_3213_;
goto v_resetjp_3207_;
}
else
{
lean_inc(v_a_3206_);
lean_dec(v___x_3190_);
v___x_3208_ = lean_box(0);
v_isShared_3209_ = v_isSharedCheck_3213_;
goto v_resetjp_3207_;
}
v_resetjp_3207_:
{
lean_object* v___x_3211_; 
if (v_isShared_3209_ == 0)
{
v___x_3211_ = v___x_3208_;
goto v_reusejp_3210_;
}
else
{
lean_object* v_reuseFailAlloc_3212_; 
v_reuseFailAlloc_3212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3212_, 0, v_a_3206_);
v___x_3211_ = v_reuseFailAlloc_3212_;
goto v_reusejp_3210_;
}
v_reusejp_3210_:
{
return v___x_3211_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___lam__0___boxed(lean_object* v_f_3214_, lean_object* v_e_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_){
_start:
{
lean_object* v_res_3221_; 
v_res_3221_ = lp_aesop_Aesop_isAppOfUpToDefeq___lam__0(v_f_3214_, v_e_3215_, v___y_3216_, v___y_3217_, v___y_3218_, v___y_3219_);
return v_res_3221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq(lean_object* v_f_3222_, lean_object* v_e_3223_, lean_object* v_a_3224_, lean_object* v_a_3225_, lean_object* v_a_3226_, lean_object* v_a_3227_){
_start:
{
lean_object* v___f_3229_; lean_object* v___x_3230_; 
v___f_3229_ = lean_alloc_closure((void*)(lp_aesop_Aesop_isAppOfUpToDefeq___lam__0___boxed), 7, 2);
lean_closure_set(v___f_3229_, 0, v_f_3222_);
lean_closure_set(v___f_3229_, 1, v_e_3223_);
v___x_3230_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(v___f_3229_, v_a_3224_, v_a_3225_, v_a_3226_, v_a_3227_);
return v___x_3230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isAppOfUpToDefeq___boxed(lean_object* v_f_3231_, lean_object* v_e_3232_, lean_object* v_a_3233_, lean_object* v_a_3234_, lean_object* v_a_3235_, lean_object* v_a_3236_, lean_object* v_a_3237_){
_start:
{
lean_object* v_res_3238_; 
v_res_3238_ = lp_aesop_Aesop_isAppOfUpToDefeq(v_f_3231_, v_e_3232_, v_a_3233_, v_a_3234_, v_a_3235_, v_a_3236_);
lean_dec(v_a_3236_);
lean_dec_ref(v_a_3235_);
lean_dec(v_a_3234_);
lean_dec_ref(v_a_3233_);
return v_res_3238_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go(lean_object* v_args_3239_, lean_object* v_e_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_, lean_object* v_a_3244_){
_start:
{
lean_object* v___x_3246_; 
lean_inc(v_a_3244_);
lean_inc_ref(v_a_3243_);
lean_inc(v_a_3242_);
lean_inc_ref(v_a_3241_);
lean_inc_ref(v_e_3240_);
v___x_3246_ = lean_whnf(v_e_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_);
if (lean_obj_tag(v___x_3246_) == 0)
{
lean_object* v_a_3247_; lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3260_; 
v_a_3247_ = lean_ctor_get(v___x_3246_, 0);
v_isSharedCheck_3260_ = !lean_is_exclusive(v___x_3246_);
if (v_isSharedCheck_3260_ == 0)
{
v___x_3249_ = v___x_3246_;
v_isShared_3250_ = v_isSharedCheck_3260_;
goto v_resetjp_3248_;
}
else
{
lean_inc(v_a_3247_);
lean_dec(v___x_3246_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3260_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
if (lean_obj_tag(v_a_3247_) == 5)
{
lean_object* v_fn_3251_; lean_object* v_arg_3252_; lean_object* v___x_3253_; 
lean_del_object(v___x_3249_);
lean_dec_ref(v_e_3240_);
v_fn_3251_ = lean_ctor_get(v_a_3247_, 0);
lean_inc_ref(v_fn_3251_);
v_arg_3252_ = lean_ctor_get(v_a_3247_, 1);
lean_inc_ref(v_arg_3252_);
lean_dec_ref_known(v_a_3247_, 2);
v___x_3253_ = lean_array_push(v_args_3239_, v_arg_3252_);
v_args_3239_ = v___x_3253_;
v_e_3240_ = v_fn_3251_;
goto _start;
}
else
{
lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3258_; 
lean_dec(v_a_3247_);
v___x_3255_ = l_Array_reverse___redArg(v_args_3239_);
v___x_3256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3256_, 0, v_e_3240_);
lean_ctor_set(v___x_3256_, 1, v___x_3255_);
if (v_isShared_3250_ == 0)
{
lean_ctor_set(v___x_3249_, 0, v___x_3256_);
v___x_3258_ = v___x_3249_;
goto v_reusejp_3257_;
}
else
{
lean_object* v_reuseFailAlloc_3259_; 
v_reuseFailAlloc_3259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3259_, 0, v___x_3256_);
v___x_3258_ = v_reuseFailAlloc_3259_;
goto v_reusejp_3257_;
}
v_reusejp_3257_:
{
return v___x_3258_;
}
}
}
}
else
{
lean_object* v_a_3261_; lean_object* v___x_3263_; uint8_t v_isShared_3264_; uint8_t v_isSharedCheck_3268_; 
lean_dec_ref(v_e_3240_);
lean_dec_ref(v_args_3239_);
v_a_3261_ = lean_ctor_get(v___x_3246_, 0);
v_isSharedCheck_3268_ = !lean_is_exclusive(v___x_3246_);
if (v_isSharedCheck_3268_ == 0)
{
v___x_3263_ = v___x_3246_;
v_isShared_3264_ = v_isSharedCheck_3268_;
goto v_resetjp_3262_;
}
else
{
lean_inc(v_a_3261_);
lean_dec(v___x_3246_);
v___x_3263_ = lean_box(0);
v_isShared_3264_ = v_isSharedCheck_3268_;
goto v_resetjp_3262_;
}
v_resetjp_3262_:
{
lean_object* v___x_3266_; 
if (v_isShared_3264_ == 0)
{
v___x_3266_ = v___x_3263_;
goto v_reusejp_3265_;
}
else
{
lean_object* v_reuseFailAlloc_3267_; 
v_reuseFailAlloc_3267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3267_, 0, v_a_3261_);
v___x_3266_ = v_reuseFailAlloc_3267_;
goto v_reusejp_3265_;
}
v_reusejp_3265_:
{
return v___x_3266_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go___boxed(lean_object* v_args_3269_, lean_object* v_e_3270_, lean_object* v_a_3271_, lean_object* v_a_3272_, lean_object* v_a_3273_, lean_object* v_a_3274_, lean_object* v_a_3275_){
_start:
{
lean_object* v_res_3276_; 
v_res_3276_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go(v_args_3269_, v_e_3270_, v_a_3271_, v_a_3272_, v_a_3273_, v_a_3274_);
lean_dec(v_a_3274_);
lean_dec_ref(v_a_3273_);
lean_dec(v_a_3272_);
lean_dec_ref(v_a_3271_);
return v_res_3276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAppUpToDefeq(lean_object* v_e_3279_, lean_object* v_a_3280_, lean_object* v_a_3281_, lean_object* v_a_3282_, lean_object* v_a_3283_){
_start:
{
lean_object* v___x_3285_; lean_object* v___x_3286_; 
v___x_3285_ = ((lean_object*)(lp_aesop_Aesop_getAppUpToDefeq___closed__0));
v___x_3286_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_getAppUpToDefeq_go(v___x_3285_, v_e_3279_, v_a_3280_, v_a_3281_, v_a_3282_, v_a_3283_);
return v___x_3286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getAppUpToDefeq___boxed(lean_object* v_e_3287_, lean_object* v_a_3288_, lean_object* v_a_3289_, lean_object* v_a_3290_, lean_object* v_a_3291_, lean_object* v_a_3292_){
_start:
{
lean_object* v_res_3293_; 
v_res_3293_ = lp_aesop_Aesop_getAppUpToDefeq(v_e_3287_, v_a_3288_, v_a_3289_, v_a_3290_, v_a_3291_);
lean_dec(v_a_3291_);
lean_dec_ref(v_a_3290_);
lean_dec(v_a_3289_);
lean_dec_ref(v_a_3288_);
return v_res_3293_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5(lean_object* v_s_3297_){
_start:
{
lean_object* v___x_3298_; lean_object* v___x_3299_; uint8_t v___x_3300_; 
v___x_3298_ = lean_array_get_size(v_s_3297_);
v___x_3299_ = lean_unsigned_to_nat(0u);
v___x_3300_ = lean_nat_dec_eq(v___x_3298_, v___x_3299_);
return v___x_3300_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5___boxed(lean_object* v_s_3301_){
_start:
{
uint8_t v_res_3302_; lean_object* v_r_3303_; 
v_res_3302_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5(v_s_3301_);
lean_dec_ref(v_s_3301_);
v_r_3303_ = lean_box(v_res_3302_);
return v_r_3303_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6(lean_object* v_x2_3304_, lean_object* v_as_3305_, size_t v_i_3306_, size_t v_stop_3307_){
_start:
{
uint8_t v___x_3308_; 
v___x_3308_ = lean_usize_dec_eq(v_i_3306_, v_stop_3307_);
if (v___x_3308_ == 0)
{
lean_object* v___x_3309_; uint8_t v___x_3310_; 
v___x_3309_ = lean_array_uget_borrowed(v_as_3305_, v_i_3306_);
v___x_3310_ = l_Lean_instBEqMVarId_beq(v___x_3309_, v_x2_3304_);
if (v___x_3310_ == 0)
{
size_t v___x_3311_; size_t v___x_3312_; 
v___x_3311_ = ((size_t)1ULL);
v___x_3312_ = lean_usize_add(v_i_3306_, v___x_3311_);
v_i_3306_ = v___x_3312_;
goto _start;
}
else
{
return v___x_3310_;
}
}
else
{
uint8_t v___x_3314_; 
v___x_3314_ = 0;
return v___x_3314_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6___boxed(lean_object* v_x2_3315_, lean_object* v_as_3316_, lean_object* v_i_3317_, lean_object* v_stop_3318_){
_start:
{
size_t v_i_boxed_3319_; size_t v_stop_boxed_3320_; uint8_t v_res_3321_; lean_object* v_r_3322_; 
v_i_boxed_3319_ = lean_unbox_usize(v_i_3317_);
lean_dec(v_i_3317_);
v_stop_boxed_3320_ = lean_unbox_usize(v_stop_3318_);
lean_dec(v_stop_3318_);
v_res_3321_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6(v_x2_3315_, v_as_3316_, v_i_boxed_3319_, v_stop_boxed_3320_);
lean_dec_ref(v_as_3316_);
lean_dec(v_x2_3315_);
v_r_3322_ = lean_box(v_res_3321_);
return v_r_3322_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(lean_object* v___x_3323_, lean_object* v_as_3324_, size_t v_i_3325_, size_t v_stop_3326_, lean_object* v_b_3327_){
_start:
{
lean_object* v___y_3329_; uint8_t v___x_3333_; 
v___x_3333_ = lean_usize_dec_eq(v_i_3325_, v_stop_3326_);
if (v___x_3333_ == 0)
{
lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___y_3337_; uint8_t v___x_3344_; 
v___x_3334_ = lean_unsigned_to_nat(0u);
v___x_3335_ = lean_array_uget_borrowed(v_as_3324_, v_i_3325_);
v___x_3344_ = lean_nat_dec_lt(v___x_3334_, v___x_3323_);
if (v___x_3344_ == 0)
{
lean_object* v___x_3345_; 
lean_inc(v___x_3335_);
v___x_3345_ = lean_array_push(v_b_3327_, v___x_3335_);
v___y_3329_ = v___x_3345_;
goto v___jp_3328_;
}
else
{
lean_object* v___x_3346_; uint8_t v___x_3347_; 
v___x_3346_ = lean_array_get_size(v_b_3327_);
v___x_3347_ = lean_nat_dec_le(v___x_3323_, v___x_3346_);
if (v___x_3347_ == 0)
{
v___y_3337_ = v___x_3346_;
goto v___jp_3336_;
}
else
{
lean_inc(v___x_3323_);
v___y_3337_ = v___x_3323_;
goto v___jp_3336_;
}
}
v___jp_3336_:
{
uint8_t v___x_3338_; 
v___x_3338_ = lean_nat_dec_lt(v___x_3334_, v___y_3337_);
if (v___x_3338_ == 0)
{
lean_object* v___x_3339_; 
lean_dec(v___y_3337_);
lean_inc(v___x_3335_);
v___x_3339_ = lean_array_push(v_b_3327_, v___x_3335_);
v___y_3329_ = v___x_3339_;
goto v___jp_3328_;
}
else
{
size_t v___x_3340_; size_t v___x_3341_; uint8_t v___x_3342_; 
v___x_3340_ = ((size_t)0ULL);
v___x_3341_ = lean_usize_of_nat(v___y_3337_);
lean_dec(v___y_3337_);
v___x_3342_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__6(v___x_3335_, v_b_3327_, v___x_3340_, v___x_3341_);
if (v___x_3342_ == 0)
{
lean_object* v___x_3343_; 
lean_inc(v___x_3335_);
v___x_3343_ = lean_array_push(v_b_3327_, v___x_3335_);
v___y_3329_ = v___x_3343_;
goto v___jp_3328_;
}
else
{
v___y_3329_ = v_b_3327_;
goto v___jp_3328_;
}
}
}
}
else
{
lean_dec(v___x_3323_);
return v_b_3327_;
}
v___jp_3328_:
{
size_t v___x_3330_; size_t v___x_3331_; 
v___x_3330_ = ((size_t)1ULL);
v___x_3331_ = lean_usize_add(v_i_3325_, v___x_3330_);
v_i_3325_ = v___x_3331_;
v_b_3327_ = v___y_3329_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7___boxed(lean_object* v___x_3348_, lean_object* v_as_3349_, lean_object* v_i_3350_, lean_object* v_stop_3351_, lean_object* v_b_3352_){
_start:
{
size_t v_i_boxed_3353_; size_t v_stop_boxed_3354_; lean_object* v_res_3355_; 
v_i_boxed_3353_ = lean_unbox_usize(v_i_3350_);
lean_dec(v_i_3350_);
v_stop_boxed_3354_ = lean_unbox_usize(v_stop_3351_);
lean_dec(v_stop_3351_);
v_res_3355_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(v___x_3348_, v_as_3349_, v_i_boxed_3353_, v_stop_boxed_3354_, v_b_3352_);
lean_dec_ref(v_as_3349_);
return v_res_3355_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3(lean_object* v_xs_3356_, lean_object* v_ys_3357_){
_start:
{
lean_object* v___x_3358_; lean_object* v___x_3359_; uint8_t v___x_3360_; 
v___x_3358_ = lean_array_get_size(v_xs_3356_);
v___x_3359_ = lean_array_get_size(v_ys_3357_);
v___x_3360_ = lean_nat_dec_lt(v___x_3358_, v___x_3359_);
if (v___x_3360_ == 0)
{
lean_object* v___x_3361_; uint8_t v___x_3362_; 
v___x_3361_ = lean_unsigned_to_nat(0u);
v___x_3362_ = lean_nat_dec_lt(v___x_3361_, v___x_3359_);
if (v___x_3362_ == 0)
{
lean_dec_ref(v_ys_3357_);
return v_xs_3356_;
}
else
{
uint8_t v___x_3363_; 
v___x_3363_ = lean_nat_dec_le(v___x_3359_, v___x_3359_);
if (v___x_3363_ == 0)
{
if (v___x_3362_ == 0)
{
lean_dec_ref(v_ys_3357_);
return v_xs_3356_;
}
else
{
size_t v___x_3364_; size_t v___x_3365_; lean_object* v___x_3366_; 
v___x_3364_ = ((size_t)0ULL);
v___x_3365_ = lean_usize_of_nat(v___x_3359_);
v___x_3366_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(v___x_3358_, v_ys_3357_, v___x_3364_, v___x_3365_, v_xs_3356_);
lean_dec_ref(v_ys_3357_);
return v___x_3366_;
}
}
else
{
size_t v___x_3367_; size_t v___x_3368_; lean_object* v___x_3369_; 
v___x_3367_ = ((size_t)0ULL);
v___x_3368_ = lean_usize_of_nat(v___x_3359_);
v___x_3369_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(v___x_3358_, v_ys_3357_, v___x_3367_, v___x_3368_, v_xs_3356_);
lean_dec_ref(v_ys_3357_);
return v___x_3369_;
}
}
}
else
{
lean_object* v___x_3370_; uint8_t v___x_3371_; 
v___x_3370_ = lean_unsigned_to_nat(0u);
v___x_3371_ = lean_nat_dec_lt(v___x_3370_, v___x_3358_);
if (v___x_3371_ == 0)
{
lean_dec_ref(v_xs_3356_);
return v_ys_3357_;
}
else
{
uint8_t v___x_3372_; 
v___x_3372_ = lean_nat_dec_le(v___x_3358_, v___x_3358_);
if (v___x_3372_ == 0)
{
if (v___x_3371_ == 0)
{
lean_dec_ref(v_xs_3356_);
return v_ys_3357_;
}
else
{
size_t v___x_3373_; size_t v___x_3374_; lean_object* v___x_3375_; 
v___x_3373_ = ((size_t)0ULL);
v___x_3374_ = lean_usize_of_nat(v___x_3358_);
v___x_3375_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(v___x_3359_, v_xs_3356_, v___x_3373_, v___x_3374_, v_ys_3357_);
lean_dec_ref(v_xs_3356_);
return v___x_3375_;
}
}
else
{
size_t v___x_3376_; size_t v___x_3377_; lean_object* v___x_3378_; 
v___x_3376_ = ((size_t)0ULL);
v___x_3377_ = lean_usize_of_nat(v___x_3358_);
v___x_3378_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3_spec__7(v___x_3359_, v_xs_3356_, v___x_3376_, v___x_3377_, v_ys_3357_);
lean_dec_ref(v_xs_3356_);
return v___x_3378_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1(lean_object* v_s_3379_, lean_object* v_t_3380_){
_start:
{
lean_object* v___x_3381_; 
v___x_3381_ = lp_aesop_Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3(v_s_3379_, v_t_3380_);
return v___x_3381_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__0(lean_object* v_x_3382_, lean_object* v_x_3383_){
_start:
{
if (lean_obj_tag(v_x_3383_) == 0)
{
return v_x_3382_;
}
else
{
lean_object* v_key_3384_; lean_object* v_tail_3385_; lean_object* v___x_3386_; 
v_key_3384_ = lean_ctor_get(v_x_3383_, 0);
lean_inc(v_key_3384_);
v_tail_3385_ = lean_ctor_get(v_x_3383_, 2);
lean_inc(v_tail_3385_);
lean_dec_ref_known(v_x_3383_, 3);
v___x_3386_ = lean_array_push(v_x_3382_, v_key_3384_);
v_x_3382_ = v___x_3386_;
v_x_3383_ = v_tail_3385_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1(lean_object* v_as_3388_, size_t v_i_3389_, size_t v_stop_3390_, lean_object* v_b_3391_){
_start:
{
uint8_t v___x_3392_; 
v___x_3392_ = lean_usize_dec_eq(v_i_3389_, v_stop_3390_);
if (v___x_3392_ == 0)
{
lean_object* v___x_3393_; lean_object* v___x_3394_; size_t v___x_3395_; size_t v___x_3396_; 
v___x_3393_ = lean_array_uget_borrowed(v_as_3388_, v_i_3389_);
lean_inc(v___x_3393_);
v___x_3394_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__0(v_b_3391_, v___x_3393_);
v___x_3395_ = ((size_t)1ULL);
v___x_3396_ = lean_usize_add(v_i_3389_, v___x_3395_);
v_i_3389_ = v___x_3396_;
v_b_3391_ = v___x_3394_;
goto _start;
}
else
{
return v_b_3391_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1___boxed(lean_object* v_as_3398_, lean_object* v_i_3399_, lean_object* v_stop_3400_, lean_object* v_b_3401_){
_start:
{
size_t v_i_boxed_3402_; size_t v_stop_boxed_3403_; lean_object* v_res_3404_; 
v_i_boxed_3402_ = lean_unbox_usize(v_i_3399_);
lean_dec(v_i_3399_);
v_stop_boxed_3403_ = lean_unbox_usize(v_stop_3400_);
lean_dec(v_stop_3400_);
v_res_3404_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1(v_as_3398_, v_i_boxed_3402_, v_stop_boxed_3403_, v_b_3401_);
lean_dec_ref(v_as_3398_);
return v_res_3404_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0(lean_object* v_xs_3405_){
_start:
{
lean_object* v_size_3406_; lean_object* v_buckets_3407_; lean_object* v___x_3408_; lean_object* v___x_3409_; lean_object* v___x_3410_; uint8_t v___x_3411_; 
v_size_3406_ = lean_ctor_get(v_xs_3405_, 0);
v_buckets_3407_ = lean_ctor_get(v_xs_3405_, 1);
v___x_3408_ = lean_mk_empty_array_with_capacity(v_size_3406_);
v___x_3409_ = lean_unsigned_to_nat(0u);
v___x_3410_ = lean_array_get_size(v_buckets_3407_);
v___x_3411_ = lean_nat_dec_lt(v___x_3409_, v___x_3410_);
if (v___x_3411_ == 0)
{
return v___x_3408_;
}
else
{
uint8_t v___x_3412_; 
v___x_3412_ = lean_nat_dec_le(v___x_3410_, v___x_3410_);
if (v___x_3412_ == 0)
{
if (v___x_3411_ == 0)
{
return v___x_3408_;
}
else
{
size_t v___x_3413_; size_t v___x_3414_; lean_object* v___x_3415_; 
v___x_3413_ = ((size_t)0ULL);
v___x_3414_ = lean_usize_of_nat(v___x_3410_);
v___x_3415_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1(v_buckets_3407_, v___x_3413_, v___x_3414_, v___x_3408_);
return v___x_3415_;
}
}
else
{
size_t v___x_3416_; size_t v___x_3417_; lean_object* v___x_3418_; 
v___x_3416_ = ((size_t)0ULL);
v___x_3417_ = lean_usize_of_nat(v___x_3410_);
v___x_3418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0_spec__1(v_buckets_3407_, v___x_3416_, v___x_3417_, v___x_3408_);
return v___x_3418_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0___boxed(lean_object* v_xs_3419_){
_start:
{
lean_object* v_res_3420_; 
v_res_3420_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0(v_xs_3419_);
lean_dec_ref(v_xs_3419_);
return v_res_3420_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg(lean_object* v_mvarId_3421_, lean_object* v_as_3422_, size_t v_sz_3423_, size_t v_i_3424_, lean_object* v_b_3425_, lean_object* v___y_3426_, lean_object* v___y_3427_, lean_object* v___y_3428_, lean_object* v___y_3429_){
_start:
{
uint8_t v___x_3431_; 
v___x_3431_ = lean_usize_dec_lt(v_i_3424_, v_sz_3423_);
if (v___x_3431_ == 0)
{
lean_object* v___x_3432_; 
lean_dec_ref(v_mvarId_3421_);
v___x_3432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3432_, 0, v_b_3425_);
return v___x_3432_;
}
else
{
lean_object* v_a_3433_; lean_object* v___x_3434_; uint8_t v___x_3435_; lean_object* v___x_3436_; 
v_a_3433_ = lean_array_uget_borrowed(v_as_3422_, v_i_3424_);
lean_inc_ref(v_mvarId_3421_);
lean_inc(v_a_3433_);
v___x_3434_ = lean_apply_1(v_mvarId_3421_, v_a_3433_);
v___x_3435_ = 0;
v___x_3436_ = l_Lean_MVarId_getMVarDependencies(v___x_3434_, v___x_3435_, v___y_3426_, v___y_3427_, v___y_3428_, v___y_3429_);
if (lean_obj_tag(v___x_3436_) == 0)
{
lean_object* v_a_3437_; lean_object* v_fst_3438_; lean_object* v_snd_3439_; lean_object* v___x_3441_; uint8_t v_isShared_3442_; uint8_t v_isSharedCheck_3453_; 
v_a_3437_ = lean_ctor_get(v___x_3436_, 0);
lean_inc(v_a_3437_);
lean_dec_ref_known(v___x_3436_, 1);
v_fst_3438_ = lean_ctor_get(v_b_3425_, 0);
v_snd_3439_ = lean_ctor_get(v_b_3425_, 1);
v_isSharedCheck_3453_ = !lean_is_exclusive(v_b_3425_);
if (v_isSharedCheck_3453_ == 0)
{
v___x_3441_ = v_b_3425_;
v_isShared_3442_ = v_isSharedCheck_3453_;
goto v_resetjp_3440_;
}
else
{
lean_inc(v_snd_3439_);
lean_inc(v_fst_3438_);
lean_dec(v_b_3425_);
v___x_3441_ = lean_box(0);
v_isShared_3442_ = v_isSharedCheck_3453_;
goto v_resetjp_3440_;
}
v_resetjp_3440_:
{
lean_object* v___x_3443_; lean_object* v___x_3444_; lean_object* v___x_3446_; 
v___x_3443_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_partitionGoalsAndMVars_spec__0(v_a_3437_);
lean_dec(v_a_3437_);
lean_inc_ref(v___x_3443_);
v___x_3444_ = lp_aesop_Array_mergeUnsortedDedup___at___00Aesop_UnorderedArraySet_merge___at___00Aesop_partitionGoalsAndMVars_spec__1_spec__3(v_snd_3439_, v___x_3443_);
lean_inc(v_a_3433_);
if (v_isShared_3442_ == 0)
{
lean_ctor_set(v___x_3441_, 1, v___x_3443_);
lean_ctor_set(v___x_3441_, 0, v_a_3433_);
v___x_3446_ = v___x_3441_;
goto v_reusejp_3445_;
}
else
{
lean_object* v_reuseFailAlloc_3452_; 
v_reuseFailAlloc_3452_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3452_, 0, v_a_3433_);
lean_ctor_set(v_reuseFailAlloc_3452_, 1, v___x_3443_);
v___x_3446_ = v_reuseFailAlloc_3452_;
goto v_reusejp_3445_;
}
v_reusejp_3445_:
{
lean_object* v___x_3447_; lean_object* v___x_3448_; size_t v___x_3449_; size_t v___x_3450_; 
v___x_3447_ = lean_array_push(v_fst_3438_, v___x_3446_);
v___x_3448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3448_, 0, v___x_3447_);
lean_ctor_set(v___x_3448_, 1, v___x_3444_);
v___x_3449_ = ((size_t)1ULL);
v___x_3450_ = lean_usize_add(v_i_3424_, v___x_3449_);
v_i_3424_ = v___x_3450_;
v_b_3425_ = v___x_3448_;
goto _start;
}
}
}
else
{
lean_object* v_a_3454_; lean_object* v___x_3456_; uint8_t v_isShared_3457_; uint8_t v_isSharedCheck_3461_; 
lean_dec_ref(v_b_3425_);
lean_dec_ref(v_mvarId_3421_);
v_a_3454_ = lean_ctor_get(v___x_3436_, 0);
v_isSharedCheck_3461_ = !lean_is_exclusive(v___x_3436_);
if (v_isSharedCheck_3461_ == 0)
{
v___x_3456_ = v___x_3436_;
v_isShared_3457_ = v_isSharedCheck_3461_;
goto v_resetjp_3455_;
}
else
{
lean_inc(v_a_3454_);
lean_dec(v___x_3436_);
v___x_3456_ = lean_box(0);
v_isShared_3457_ = v_isSharedCheck_3461_;
goto v_resetjp_3455_;
}
v_resetjp_3455_:
{
lean_object* v___x_3459_; 
if (v_isShared_3457_ == 0)
{
v___x_3459_ = v___x_3456_;
goto v_reusejp_3458_;
}
else
{
lean_object* v_reuseFailAlloc_3460_; 
v_reuseFailAlloc_3460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3460_, 0, v_a_3454_);
v___x_3459_ = v_reuseFailAlloc_3460_;
goto v_reusejp_3458_;
}
v_reusejp_3458_:
{
return v___x_3459_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg___boxed(lean_object* v_mvarId_3462_, lean_object* v_as_3463_, lean_object* v_sz_3464_, lean_object* v_i_3465_, lean_object* v_b_3466_, lean_object* v___y_3467_, lean_object* v___y_3468_, lean_object* v___y_3469_, lean_object* v___y_3470_, lean_object* v___y_3471_){
_start:
{
size_t v_sz_boxed_3472_; size_t v_i_boxed_3473_; lean_object* v_res_3474_; 
v_sz_boxed_3472_ = lean_unbox_usize(v_sz_3464_);
lean_dec(v_sz_3464_);
v_i_boxed_3473_ = lean_unbox_usize(v_i_3465_);
lean_dec(v_i_3465_);
v_res_3474_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg(v_mvarId_3462_, v_as_3463_, v_sz_boxed_3472_, v_i_boxed_3473_, v_b_3466_, v___y_3467_, v___y_3468_, v___y_3469_, v___y_3470_);
lean_dec(v___y_3470_);
lean_dec_ref(v___y_3469_);
lean_dec(v___y_3468_);
lean_dec_ref(v___y_3467_);
lean_dec_ref(v_as_3463_);
return v_res_3474_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11(lean_object* v_a_3475_, lean_object* v_as_3476_, size_t v_i_3477_, size_t v_stop_3478_){
_start:
{
uint8_t v___x_3479_; 
v___x_3479_ = lean_usize_dec_eq(v_i_3477_, v_stop_3478_);
if (v___x_3479_ == 0)
{
lean_object* v___x_3480_; uint8_t v___x_3481_; 
v___x_3480_ = lean_array_uget_borrowed(v_as_3476_, v_i_3477_);
v___x_3481_ = l_Lean_instBEqMVarId_beq(v_a_3475_, v___x_3480_);
if (v___x_3481_ == 0)
{
size_t v___x_3482_; size_t v___x_3483_; 
v___x_3482_ = ((size_t)1ULL);
v___x_3483_ = lean_usize_add(v_i_3477_, v___x_3482_);
v_i_3477_ = v___x_3483_;
goto _start;
}
else
{
return v___x_3481_;
}
}
else
{
uint8_t v___x_3485_; 
v___x_3485_ = 0;
return v___x_3485_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11___boxed(lean_object* v_a_3486_, lean_object* v_as_3487_, lean_object* v_i_3488_, lean_object* v_stop_3489_){
_start:
{
size_t v_i_boxed_3490_; size_t v_stop_boxed_3491_; uint8_t v_res_3492_; lean_object* v_r_3493_; 
v_i_boxed_3490_ = lean_unbox_usize(v_i_3488_);
lean_dec(v_i_3488_);
v_stop_boxed_3491_ = lean_unbox_usize(v_stop_3489_);
lean_dec(v_stop_3489_);
v_res_3492_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11(v_a_3486_, v_as_3487_, v_i_boxed_3490_, v_stop_boxed_3491_);
lean_dec_ref(v_as_3487_);
lean_dec(v_a_3486_);
v_r_3493_ = lean_box(v_res_3492_);
return v_r_3493_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7(lean_object* v_as_3494_, lean_object* v_a_3495_){
_start:
{
lean_object* v___x_3496_; lean_object* v___x_3497_; uint8_t v___x_3498_; 
v___x_3496_ = lean_unsigned_to_nat(0u);
v___x_3497_ = lean_array_get_size(v_as_3494_);
v___x_3498_ = lean_nat_dec_lt(v___x_3496_, v___x_3497_);
if (v___x_3498_ == 0)
{
return v___x_3498_;
}
else
{
if (v___x_3498_ == 0)
{
return v___x_3498_;
}
else
{
size_t v___x_3499_; size_t v___x_3500_; uint8_t v___x_3501_; 
v___x_3499_ = ((size_t)0ULL);
v___x_3500_ = lean_usize_of_nat(v___x_3497_);
v___x_3501_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7_spec__11(v_a_3495_, v_as_3494_, v___x_3499_, v___x_3500_);
return v___x_3501_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7___boxed(lean_object* v_as_3502_, lean_object* v_a_3503_){
_start:
{
uint8_t v_res_3504_; lean_object* v_r_3505_; 
v_res_3504_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7(v_as_3502_, v_a_3503_);
lean_dec(v_a_3503_);
lean_dec_ref(v_as_3502_);
v_r_3505_ = lean_box(v_res_3504_);
return v_r_3505_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4(lean_object* v_x_3506_, lean_object* v_s_3507_){
_start:
{
uint8_t v___x_3508_; 
v___x_3508_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7(v_s_3507_, v_x_3506_);
return v___x_3508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4___boxed(lean_object* v_x_3509_, lean_object* v_s_3510_){
_start:
{
uint8_t v_res_3511_; lean_object* v_r_3512_; 
v_res_3511_ = lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4(v_x_3509_, v_s_3510_);
lean_dec_ref(v_s_3510_);
lean_dec(v_x_3509_);
v_r_3512_ = lean_box(v_res_3511_);
return v_r_3512_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(lean_object* v_mvarId_3513_, lean_object* v___x_3514_, lean_object* v_as_3515_, size_t v_i_3516_, size_t v_stop_3517_, lean_object* v_b_3518_){
_start:
{
lean_object* v___y_3520_; uint8_t v___x_3524_; 
v___x_3524_ = lean_usize_dec_eq(v_i_3516_, v_stop_3517_);
if (v___x_3524_ == 0)
{
lean_object* v___x_3525_; lean_object* v_fst_3526_; lean_object* v___x_3527_; uint8_t v___x_3528_; 
v___x_3525_ = lean_array_uget_borrowed(v_as_3515_, v_i_3516_);
v_fst_3526_ = lean_ctor_get(v___x_3525_, 0);
lean_inc_ref(v_mvarId_3513_);
lean_inc(v_fst_3526_);
v___x_3527_ = lean_apply_1(v_mvarId_3513_, v_fst_3526_);
v___x_3528_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_partitionGoalsAndMVars_spec__4_spec__7(v___x_3514_, v___x_3527_);
lean_dec(v___x_3527_);
if (v___x_3528_ == 0)
{
lean_object* v___x_3529_; 
lean_inc(v___x_3525_);
v___x_3529_ = lean_array_push(v_b_3518_, v___x_3525_);
v___y_3520_ = v___x_3529_;
goto v___jp_3519_;
}
else
{
v___y_3520_ = v_b_3518_;
goto v___jp_3519_;
}
}
else
{
lean_dec_ref(v_mvarId_3513_);
return v_b_3518_;
}
v___jp_3519_:
{
size_t v___x_3521_; size_t v___x_3522_; 
v___x_3521_ = ((size_t)1ULL);
v___x_3522_ = lean_usize_add(v_i_3516_, v___x_3521_);
v_i_3516_ = v___x_3522_;
v_b_3518_ = v___y_3520_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg___boxed(lean_object* v_mvarId_3530_, lean_object* v___x_3531_, lean_object* v_as_3532_, lean_object* v_i_3533_, lean_object* v_stop_3534_, lean_object* v_b_3535_){
_start:
{
size_t v_i_boxed_3536_; size_t v_stop_boxed_3537_; lean_object* v_res_3538_; 
v_i_boxed_3536_ = lean_unbox_usize(v_i_3533_);
lean_dec(v_i_3533_);
v_stop_boxed_3537_ = lean_unbox_usize(v_stop_3534_);
lean_dec(v_stop_3534_);
v_res_3538_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(v_mvarId_3530_, v___x_3531_, v_as_3532_, v_i_boxed_3536_, v_stop_boxed_3537_, v_b_3535_);
lean_dec_ref(v_as_3532_);
lean_dec_ref(v___x_3531_);
return v_res_3538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg(lean_object* v_mvarId_3544_, lean_object* v_goals_3545_, lean_object* v_a_3546_, lean_object* v_a_3547_, lean_object* v_a_3548_, lean_object* v_a_3549_){
_start:
{
lean_object* v___x_3551_; lean_object* v_goalsAndMVars_3552_; lean_object* v___x_3553_; size_t v_sz_3554_; size_t v___x_3555_; lean_object* v___x_3556_; 
v___x_3551_ = lean_unsigned_to_nat(0u);
v_goalsAndMVars_3552_ = ((lean_object*)(lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__0));
v___x_3553_ = ((lean_object*)(lp_aesop_Aesop_partitionGoalsAndMVars___redArg___closed__1));
v_sz_3554_ = lean_array_size(v_goals_3545_);
v___x_3555_ = ((size_t)0ULL);
lean_inc_ref(v_mvarId_3544_);
v___x_3556_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg(v_mvarId_3544_, v_goals_3545_, v_sz_3554_, v___x_3555_, v___x_3553_, v_a_3546_, v_a_3547_, v_a_3548_, v_a_3549_);
if (lean_obj_tag(v___x_3556_) == 0)
{
lean_object* v_a_3557_; lean_object* v___x_3559_; uint8_t v_isShared_3560_; uint8_t v_isSharedCheck_3583_; 
v_a_3557_ = lean_ctor_get(v___x_3556_, 0);
v_isSharedCheck_3583_ = !lean_is_exclusive(v___x_3556_);
if (v_isSharedCheck_3583_ == 0)
{
v___x_3559_ = v___x_3556_;
v_isShared_3560_ = v_isSharedCheck_3583_;
goto v_resetjp_3558_;
}
else
{
lean_inc(v_a_3557_);
lean_dec(v___x_3556_);
v___x_3559_ = lean_box(0);
v_isShared_3560_ = v_isSharedCheck_3583_;
goto v_resetjp_3558_;
}
v_resetjp_3558_:
{
lean_object* v_fst_3561_; lean_object* v_snd_3562_; lean_object* v___x_3564_; uint8_t v_isShared_3565_; uint8_t v_isSharedCheck_3582_; 
v_fst_3561_ = lean_ctor_get(v_a_3557_, 0);
v_snd_3562_ = lean_ctor_get(v_a_3557_, 1);
v_isSharedCheck_3582_ = !lean_is_exclusive(v_a_3557_);
if (v_isSharedCheck_3582_ == 0)
{
v___x_3564_ = v_a_3557_;
v_isShared_3565_ = v_isSharedCheck_3582_;
goto v_resetjp_3563_;
}
else
{
lean_inc(v_snd_3562_);
lean_inc(v_fst_3561_);
lean_dec(v_a_3557_);
v___x_3564_ = lean_box(0);
v_isShared_3565_ = v_isSharedCheck_3582_;
goto v_resetjp_3563_;
}
v_resetjp_3563_:
{
lean_object* v___y_3567_; uint8_t v___x_3574_; 
v___x_3574_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_partitionGoalsAndMVars_spec__5(v_snd_3562_);
if (v___x_3574_ == 0)
{
lean_object* v___x_3575_; uint8_t v___x_3576_; 
v___x_3575_ = lean_array_get_size(v_fst_3561_);
v___x_3576_ = lean_nat_dec_lt(v___x_3551_, v___x_3575_);
if (v___x_3576_ == 0)
{
lean_dec(v_fst_3561_);
lean_dec_ref(v_mvarId_3544_);
v___y_3567_ = v_goalsAndMVars_3552_;
goto v___jp_3566_;
}
else
{
uint8_t v___x_3577_; 
v___x_3577_ = lean_nat_dec_le(v___x_3575_, v___x_3575_);
if (v___x_3577_ == 0)
{
if (v___x_3576_ == 0)
{
lean_dec(v_fst_3561_);
lean_dec_ref(v_mvarId_3544_);
v___y_3567_ = v_goalsAndMVars_3552_;
goto v___jp_3566_;
}
else
{
size_t v___x_3578_; lean_object* v___x_3579_; 
v___x_3578_ = lean_usize_of_nat(v___x_3575_);
v___x_3579_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(v_mvarId_3544_, v_snd_3562_, v_fst_3561_, v___x_3555_, v___x_3578_, v_goalsAndMVars_3552_);
lean_dec(v_fst_3561_);
v___y_3567_ = v___x_3579_;
goto v___jp_3566_;
}
}
else
{
size_t v___x_3580_; lean_object* v___x_3581_; 
v___x_3580_ = lean_usize_of_nat(v___x_3575_);
v___x_3581_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(v_mvarId_3544_, v_snd_3562_, v_fst_3561_, v___x_3555_, v___x_3580_, v_goalsAndMVars_3552_);
lean_dec(v_fst_3561_);
v___y_3567_ = v___x_3581_;
goto v___jp_3566_;
}
}
}
else
{
lean_dec_ref(v_mvarId_3544_);
v___y_3567_ = v_fst_3561_;
goto v___jp_3566_;
}
v___jp_3566_:
{
lean_object* v___x_3569_; 
if (v_isShared_3565_ == 0)
{
lean_ctor_set(v___x_3564_, 0, v___y_3567_);
v___x_3569_ = v___x_3564_;
goto v_reusejp_3568_;
}
else
{
lean_object* v_reuseFailAlloc_3573_; 
v_reuseFailAlloc_3573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3573_, 0, v___y_3567_);
lean_ctor_set(v_reuseFailAlloc_3573_, 1, v_snd_3562_);
v___x_3569_ = v_reuseFailAlloc_3573_;
goto v_reusejp_3568_;
}
v_reusejp_3568_:
{
lean_object* v___x_3571_; 
if (v_isShared_3560_ == 0)
{
lean_ctor_set(v___x_3559_, 0, v___x_3569_);
v___x_3571_ = v___x_3559_;
goto v_reusejp_3570_;
}
else
{
lean_object* v_reuseFailAlloc_3572_; 
v_reuseFailAlloc_3572_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3572_, 0, v___x_3569_);
v___x_3571_ = v_reuseFailAlloc_3572_;
goto v_reusejp_3570_;
}
v_reusejp_3570_:
{
return v___x_3571_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_mvarId_3544_);
return v___x_3556_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg___boxed(lean_object* v_mvarId_3584_, lean_object* v_goals_3585_, lean_object* v_a_3586_, lean_object* v_a_3587_, lean_object* v_a_3588_, lean_object* v_a_3589_, lean_object* v_a_3590_){
_start:
{
lean_object* v_res_3591_; 
v_res_3591_ = lp_aesop_Aesop_partitionGoalsAndMVars___redArg(v_mvarId_3584_, v_goals_3585_, v_a_3586_, v_a_3587_, v_a_3588_, v_a_3589_);
lean_dec(v_a_3589_);
lean_dec_ref(v_a_3588_);
lean_dec(v_a_3587_);
lean_dec_ref(v_a_3586_);
lean_dec_ref(v_goals_3585_);
return v_res_3591_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars(lean_object* v_00_u03b1_3592_, lean_object* v_mvarId_3593_, lean_object* v_goals_3594_, lean_object* v_a_3595_, lean_object* v_a_3596_, lean_object* v_a_3597_, lean_object* v_a_3598_){
_start:
{
lean_object* v___x_3600_; 
v___x_3600_ = lp_aesop_Aesop_partitionGoalsAndMVars___redArg(v_mvarId_3593_, v_goals_3594_, v_a_3595_, v_a_3596_, v_a_3597_, v_a_3598_);
return v___x_3600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___boxed(lean_object* v_00_u03b1_3601_, lean_object* v_mvarId_3602_, lean_object* v_goals_3603_, lean_object* v_a_3604_, lean_object* v_a_3605_, lean_object* v_a_3606_, lean_object* v_a_3607_, lean_object* v_a_3608_){
_start:
{
lean_object* v_res_3609_; 
v_res_3609_ = lp_aesop_Aesop_partitionGoalsAndMVars(v_00_u03b1_3601_, v_mvarId_3602_, v_goals_3603_, v_a_3604_, v_a_3605_, v_a_3606_, v_a_3607_);
lean_dec(v_a_3607_);
lean_dec_ref(v_a_3606_);
lean_dec(v_a_3605_);
lean_dec_ref(v_a_3604_);
lean_dec_ref(v_goals_3603_);
return v_res_3609_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3(lean_object* v_00_u03b1_3610_, lean_object* v_mvarId_3611_, lean_object* v_as_3612_, size_t v_sz_3613_, size_t v_i_3614_, lean_object* v_b_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_, lean_object* v___y_3618_, lean_object* v___y_3619_){
_start:
{
lean_object* v___x_3621_; 
v___x_3621_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___redArg(v_mvarId_3611_, v_as_3612_, v_sz_3613_, v_i_3614_, v_b_3615_, v___y_3616_, v___y_3617_, v___y_3618_, v___y_3619_);
return v___x_3621_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3___boxed(lean_object* v_00_u03b1_3622_, lean_object* v_mvarId_3623_, lean_object* v_as_3624_, lean_object* v_sz_3625_, lean_object* v_i_3626_, lean_object* v_b_3627_, lean_object* v___y_3628_, lean_object* v___y_3629_, lean_object* v___y_3630_, lean_object* v___y_3631_, lean_object* v___y_3632_){
_start:
{
size_t v_sz_boxed_3633_; size_t v_i_boxed_3634_; lean_object* v_res_3635_; 
v_sz_boxed_3633_ = lean_unbox_usize(v_sz_3625_);
lean_dec(v_sz_3625_);
v_i_boxed_3634_ = lean_unbox_usize(v_i_3626_);
lean_dec(v_i_3626_);
v_res_3635_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_partitionGoalsAndMVars_spec__3(v_00_u03b1_3622_, v_mvarId_3623_, v_as_3624_, v_sz_boxed_3633_, v_i_boxed_3634_, v_b_3627_, v___y_3628_, v___y_3629_, v___y_3630_, v___y_3631_);
lean_dec(v___y_3631_);
lean_dec_ref(v___y_3630_);
lean_dec(v___y_3629_);
lean_dec_ref(v___y_3628_);
lean_dec_ref(v_as_3624_);
return v_res_3635_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6(lean_object* v_00_u03b1_3636_, lean_object* v_mvarId_3637_, lean_object* v___x_3638_, lean_object* v_as_3639_, size_t v_i_3640_, size_t v_stop_3641_, lean_object* v_b_3642_){
_start:
{
lean_object* v___x_3643_; 
v___x_3643_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___redArg(v_mvarId_3637_, v___x_3638_, v_as_3639_, v_i_3640_, v_stop_3641_, v_b_3642_);
return v___x_3643_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6___boxed(lean_object* v_00_u03b1_3644_, lean_object* v_mvarId_3645_, lean_object* v___x_3646_, lean_object* v_as_3647_, lean_object* v_i_3648_, lean_object* v_stop_3649_, lean_object* v_b_3650_){
_start:
{
size_t v_i_boxed_3651_; size_t v_stop_boxed_3652_; lean_object* v_res_3653_; 
v_i_boxed_3651_ = lean_unbox_usize(v_i_3648_);
lean_dec(v_i_3648_);
v_stop_boxed_3652_ = lean_unbox_usize(v_stop_3649_);
lean_dec(v_stop_3649_);
v_res_3653_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_partitionGoalsAndMVars_spec__6(v_00_u03b1_3644_, v_mvarId_3645_, v___x_3646_, v_as_3647_, v_i_boxed_3651_, v_stop_boxed_3652_, v_b_3650_);
lean_dec_ref(v_as_3647_);
lean_dec_ref(v___x_3646_);
return v_res_3653_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_runTacticMCapturingPostState___lam__0(uint8_t v___x_3654_, lean_object* v_x_3655_){
_start:
{
return v___x_3654_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__0___boxed(lean_object* v___x_3656_, lean_object* v_x_3657_){
_start:
{
uint8_t v___x_1849__boxed_3658_; uint8_t v_res_3659_; lean_object* v_r_3660_; 
v___x_1849__boxed_3658_ = lean_unbox(v___x_3656_);
v_res_3659_ = lp_aesop_Aesop_runTacticMCapturingPostState___lam__0(v___x_1849__boxed_3658_, v_x_3657_);
lean_dec(v_x_3657_);
v_r_3660_ = lean_box(v_res_3659_);
return v_r_3660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__1(lean_object* v_preGoals_3661_, lean_object* v_preState_3662_, lean_object* v_t_3663_, lean_object* v___x_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_, lean_object* v___y_3670_){
_start:
{
lean_object* v___x_3672_; lean_object* v___x_3673_; 
v___x_3672_ = lean_st_mk_ref(v_preGoals_3661_);
v___x_3673_ = l_Lean_Meta_SavedState_restore___redArg(v_preState_3662_, v___y_3668_, v___y_3670_);
if (lean_obj_tag(v___x_3673_) == 0)
{
lean_object* v___x_3674_; 
lean_dec_ref_known(v___x_3673_, 1);
lean_inc(v___y_3670_);
lean_inc_ref(v___y_3669_);
lean_inc(v___y_3668_);
lean_inc_ref(v___y_3667_);
lean_inc(v___y_3666_);
lean_inc_ref(v___y_3665_);
lean_inc(v___x_3672_);
lean_inc_ref(v___x_3664_);
v___x_3674_ = lean_apply_9(v_t_3663_, v___x_3664_, v___x_3672_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_, v___y_3670_, lean_box(0));
if (lean_obj_tag(v___x_3674_) == 0)
{
lean_object* v___x_3675_; 
lean_dec_ref_known(v___x_3674_, 1);
v___x_3675_ = l_Lean_Elab_Tactic_pruneSolvedGoals(v___x_3664_, v___x_3672_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_, v___y_3670_);
lean_dec_ref(v___y_3669_);
lean_dec_ref(v___y_3667_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
lean_dec_ref(v___x_3664_);
if (lean_obj_tag(v___x_3675_) == 0)
{
lean_object* v___x_3676_; 
lean_dec_ref_known(v___x_3675_, 1);
v___x_3676_ = l_Lean_Meta_saveState___redArg(v___y_3668_, v___y_3670_);
lean_dec(v___y_3670_);
lean_dec(v___y_3668_);
if (lean_obj_tag(v___x_3676_) == 0)
{
lean_object* v_a_3677_; lean_object* v___x_3678_; 
v_a_3677_ = lean_ctor_get(v___x_3676_, 0);
lean_inc(v_a_3677_);
lean_dec_ref_known(v___x_3676_, 1);
v___x_3678_ = l_Lean_Elab_Tactic_getGoals___redArg(v___x_3672_);
if (lean_obj_tag(v___x_3678_) == 0)
{
lean_object* v_a_3679_; lean_object* v___x_3681_; uint8_t v_isShared_3682_; uint8_t v_isSharedCheck_3688_; 
v_a_3679_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3688_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3688_ == 0)
{
v___x_3681_ = v___x_3678_;
v_isShared_3682_ = v_isSharedCheck_3688_;
goto v_resetjp_3680_;
}
else
{
lean_inc(v_a_3679_);
lean_dec(v___x_3678_);
v___x_3681_ = lean_box(0);
v_isShared_3682_ = v_isSharedCheck_3688_;
goto v_resetjp_3680_;
}
v_resetjp_3680_:
{
lean_object* v___x_3683_; lean_object* v___x_3684_; lean_object* v___x_3686_; 
v___x_3683_ = lean_st_ref_get(v___x_3672_);
lean_dec(v___x_3672_);
lean_dec(v___x_3683_);
v___x_3684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3684_, 0, v_a_3677_);
lean_ctor_set(v___x_3684_, 1, v_a_3679_);
if (v_isShared_3682_ == 0)
{
lean_ctor_set(v___x_3681_, 0, v___x_3684_);
v___x_3686_ = v___x_3681_;
goto v_reusejp_3685_;
}
else
{
lean_object* v_reuseFailAlloc_3687_; 
v_reuseFailAlloc_3687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3687_, 0, v___x_3684_);
v___x_3686_ = v_reuseFailAlloc_3687_;
goto v_reusejp_3685_;
}
v_reusejp_3685_:
{
return v___x_3686_;
}
}
}
else
{
lean_object* v_a_3689_; lean_object* v___x_3691_; uint8_t v_isShared_3692_; uint8_t v_isSharedCheck_3696_; 
lean_dec(v_a_3677_);
lean_dec(v___x_3672_);
v_a_3689_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3696_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3696_ == 0)
{
v___x_3691_ = v___x_3678_;
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
else
{
lean_inc(v_a_3689_);
lean_dec(v___x_3678_);
v___x_3691_ = lean_box(0);
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
v_resetjp_3690_:
{
lean_object* v___x_3694_; 
if (v_isShared_3692_ == 0)
{
v___x_3694_ = v___x_3691_;
goto v_reusejp_3693_;
}
else
{
lean_object* v_reuseFailAlloc_3695_; 
v_reuseFailAlloc_3695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3695_, 0, v_a_3689_);
v___x_3694_ = v_reuseFailAlloc_3695_;
goto v_reusejp_3693_;
}
v_reusejp_3693_:
{
return v___x_3694_;
}
}
}
}
else
{
lean_object* v_a_3697_; lean_object* v___x_3699_; uint8_t v_isShared_3700_; uint8_t v_isSharedCheck_3704_; 
lean_dec(v___x_3672_);
v_a_3697_ = lean_ctor_get(v___x_3676_, 0);
v_isSharedCheck_3704_ = !lean_is_exclusive(v___x_3676_);
if (v_isSharedCheck_3704_ == 0)
{
v___x_3699_ = v___x_3676_;
v_isShared_3700_ = v_isSharedCheck_3704_;
goto v_resetjp_3698_;
}
else
{
lean_inc(v_a_3697_);
lean_dec(v___x_3676_);
v___x_3699_ = lean_box(0);
v_isShared_3700_ = v_isSharedCheck_3704_;
goto v_resetjp_3698_;
}
v_resetjp_3698_:
{
lean_object* v___x_3702_; 
if (v_isShared_3700_ == 0)
{
v___x_3702_ = v___x_3699_;
goto v_reusejp_3701_;
}
else
{
lean_object* v_reuseFailAlloc_3703_; 
v_reuseFailAlloc_3703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3703_, 0, v_a_3697_);
v___x_3702_ = v_reuseFailAlloc_3703_;
goto v_reusejp_3701_;
}
v_reusejp_3701_:
{
return v___x_3702_;
}
}
}
}
else
{
lean_object* v_a_3705_; lean_object* v___x_3707_; uint8_t v_isShared_3708_; uint8_t v_isSharedCheck_3712_; 
lean_dec(v___x_3672_);
lean_dec(v___y_3670_);
lean_dec(v___y_3668_);
v_a_3705_ = lean_ctor_get(v___x_3675_, 0);
v_isSharedCheck_3712_ = !lean_is_exclusive(v___x_3675_);
if (v_isSharedCheck_3712_ == 0)
{
v___x_3707_ = v___x_3675_;
v_isShared_3708_ = v_isSharedCheck_3712_;
goto v_resetjp_3706_;
}
else
{
lean_inc(v_a_3705_);
lean_dec(v___x_3675_);
v___x_3707_ = lean_box(0);
v_isShared_3708_ = v_isSharedCheck_3712_;
goto v_resetjp_3706_;
}
v_resetjp_3706_:
{
lean_object* v___x_3710_; 
if (v_isShared_3708_ == 0)
{
v___x_3710_ = v___x_3707_;
goto v_reusejp_3709_;
}
else
{
lean_object* v_reuseFailAlloc_3711_; 
v_reuseFailAlloc_3711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3711_, 0, v_a_3705_);
v___x_3710_ = v_reuseFailAlloc_3711_;
goto v_reusejp_3709_;
}
v_reusejp_3709_:
{
return v___x_3710_;
}
}
}
}
else
{
lean_object* v_a_3713_; lean_object* v___x_3715_; uint8_t v_isShared_3716_; uint8_t v_isSharedCheck_3720_; 
lean_dec(v___x_3672_);
lean_dec(v___y_3670_);
lean_dec_ref(v___y_3669_);
lean_dec(v___y_3668_);
lean_dec_ref(v___y_3667_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
lean_dec_ref(v___x_3664_);
v_a_3713_ = lean_ctor_get(v___x_3674_, 0);
v_isSharedCheck_3720_ = !lean_is_exclusive(v___x_3674_);
if (v_isSharedCheck_3720_ == 0)
{
v___x_3715_ = v___x_3674_;
v_isShared_3716_ = v_isSharedCheck_3720_;
goto v_resetjp_3714_;
}
else
{
lean_inc(v_a_3713_);
lean_dec(v___x_3674_);
v___x_3715_ = lean_box(0);
v_isShared_3716_ = v_isSharedCheck_3720_;
goto v_resetjp_3714_;
}
v_resetjp_3714_:
{
lean_object* v___x_3718_; 
if (v_isShared_3716_ == 0)
{
v___x_3718_ = v___x_3715_;
goto v_reusejp_3717_;
}
else
{
lean_object* v_reuseFailAlloc_3719_; 
v_reuseFailAlloc_3719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3719_, 0, v_a_3713_);
v___x_3718_ = v_reuseFailAlloc_3719_;
goto v_reusejp_3717_;
}
v_reusejp_3717_:
{
return v___x_3718_;
}
}
}
}
else
{
lean_object* v_a_3721_; lean_object* v___x_3723_; uint8_t v_isShared_3724_; uint8_t v_isSharedCheck_3728_; 
lean_dec(v___x_3672_);
lean_dec(v___y_3670_);
lean_dec_ref(v___y_3669_);
lean_dec(v___y_3668_);
lean_dec_ref(v___y_3667_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
lean_dec_ref(v___x_3664_);
lean_dec_ref(v_t_3663_);
v_a_3721_ = lean_ctor_get(v___x_3673_, 0);
v_isSharedCheck_3728_ = !lean_is_exclusive(v___x_3673_);
if (v_isSharedCheck_3728_ == 0)
{
v___x_3723_ = v___x_3673_;
v_isShared_3724_ = v_isSharedCheck_3728_;
goto v_resetjp_3722_;
}
else
{
lean_inc(v_a_3721_);
lean_dec(v___x_3673_);
v___x_3723_ = lean_box(0);
v_isShared_3724_ = v_isSharedCheck_3728_;
goto v_resetjp_3722_;
}
v_resetjp_3722_:
{
lean_object* v___x_3726_; 
if (v_isShared_3724_ == 0)
{
v___x_3726_ = v___x_3723_;
goto v_reusejp_3725_;
}
else
{
lean_object* v_reuseFailAlloc_3727_; 
v_reuseFailAlloc_3727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3727_, 0, v_a_3721_);
v___x_3726_ = v_reuseFailAlloc_3727_;
goto v_reusejp_3725_;
}
v_reusejp_3725_:
{
return v___x_3726_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__1___boxed(lean_object* v_preGoals_3729_, lean_object* v_preState_3730_, lean_object* v_t_3731_, lean_object* v___x_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_, lean_object* v___y_3737_, lean_object* v___y_3738_, lean_object* v___y_3739_){
_start:
{
lean_object* v_res_3740_; 
v_res_3740_ = lp_aesop_Aesop_runTacticMCapturingPostState___lam__1(v_preGoals_3729_, v_preState_3730_, v_t_3731_, v___x_3732_, v___y_3733_, v___y_3734_, v___y_3735_, v___y_3736_, v___y_3737_, v___y_3738_);
lean_dec_ref(v_preState_3730_);
return v_res_3740_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__2(lean_object* v___f_3741_, lean_object* v___x_3742_, lean_object* v___x_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_, lean_object* v___y_3746_, lean_object* v___y_3747_){
_start:
{
lean_object* v___x_3749_; 
v___x_3749_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_3741_, v___x_3742_, v___x_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_);
if (lean_obj_tag(v___x_3749_) == 0)
{
lean_object* v_a_3750_; lean_object* v___x_3752_; uint8_t v_isShared_3753_; uint8_t v_isSharedCheck_3758_; 
v_a_3750_ = lean_ctor_get(v___x_3749_, 0);
v_isSharedCheck_3758_ = !lean_is_exclusive(v___x_3749_);
if (v_isSharedCheck_3758_ == 0)
{
v___x_3752_ = v___x_3749_;
v_isShared_3753_ = v_isSharedCheck_3758_;
goto v_resetjp_3751_;
}
else
{
lean_inc(v_a_3750_);
lean_dec(v___x_3749_);
v___x_3752_ = lean_box(0);
v_isShared_3753_ = v_isSharedCheck_3758_;
goto v_resetjp_3751_;
}
v_resetjp_3751_:
{
lean_object* v_fst_3754_; lean_object* v___x_3756_; 
v_fst_3754_ = lean_ctor_get(v_a_3750_, 0);
lean_inc(v_fst_3754_);
lean_dec(v_a_3750_);
if (v_isShared_3753_ == 0)
{
lean_ctor_set(v___x_3752_, 0, v_fst_3754_);
v___x_3756_ = v___x_3752_;
goto v_reusejp_3755_;
}
else
{
lean_object* v_reuseFailAlloc_3757_; 
v_reuseFailAlloc_3757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3757_, 0, v_fst_3754_);
v___x_3756_ = v_reuseFailAlloc_3757_;
goto v_reusejp_3755_;
}
v_reusejp_3755_:
{
return v___x_3756_;
}
}
}
else
{
lean_object* v_a_3759_; lean_object* v___x_3761_; uint8_t v_isShared_3762_; uint8_t v_isSharedCheck_3766_; 
v_a_3759_ = lean_ctor_get(v___x_3749_, 0);
v_isSharedCheck_3766_ = !lean_is_exclusive(v___x_3749_);
if (v_isSharedCheck_3766_ == 0)
{
v___x_3761_ = v___x_3749_;
v_isShared_3762_ = v_isSharedCheck_3766_;
goto v_resetjp_3760_;
}
else
{
lean_inc(v_a_3759_);
lean_dec(v___x_3749_);
v___x_3761_ = lean_box(0);
v_isShared_3762_ = v_isSharedCheck_3766_;
goto v_resetjp_3760_;
}
v_resetjp_3760_:
{
lean_object* v___x_3764_; 
if (v_isShared_3762_ == 0)
{
v___x_3764_ = v___x_3761_;
goto v_reusejp_3763_;
}
else
{
lean_object* v_reuseFailAlloc_3765_; 
v_reuseFailAlloc_3765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3765_, 0, v_a_3759_);
v___x_3764_ = v_reuseFailAlloc_3765_;
goto v_reusejp_3763_;
}
v_reusejp_3763_:
{
return v___x_3764_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___lam__2___boxed(lean_object* v___f_3767_, lean_object* v___x_3768_, lean_object* v___x_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_){
_start:
{
lean_object* v_res_3775_; 
v_res_3775_ = lp_aesop_Aesop_runTacticMCapturingPostState___lam__2(v___f_3767_, v___x_3768_, v___x_3769_, v___y_3770_, v___y_3771_, v___y_3772_, v___y_3773_);
lean_dec(v___y_3773_);
lean_dec_ref(v___y_3772_);
lean_dec(v___y_3771_);
lean_dec_ref(v___y_3770_);
return v_res_3775_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState(lean_object* v_t_3790_, lean_object* v_preState_3791_, lean_object* v_preGoals_3792_, lean_object* v_a_3793_, lean_object* v_a_3794_, lean_object* v_a_3795_, lean_object* v_a_3796_){
_start:
{
lean_object* v___x_3798_; lean_object* v___f_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v___f_3802_; lean_object* v___x_3803_; 
v___x_3798_ = ((lean_object*)(lp_aesop_Aesop_runTacticMCapturingPostState___closed__1));
v___f_3799_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticMCapturingPostState___lam__1___boxed), 11, 4);
lean_closure_set(v___f_3799_, 0, v_preGoals_3792_);
lean_closure_set(v___f_3799_, 1, v_preState_3791_);
lean_closure_set(v___f_3799_, 2, v_t_3790_);
lean_closure_set(v___f_3799_, 3, v___x_3798_);
v___x_3800_ = ((lean_object*)(lp_aesop_Aesop_runTacticMCapturingPostState___closed__2));
v___x_3801_ = ((lean_object*)(lp_aesop_Aesop_runTermElabMAsCoreM___redArg___closed__3));
v___f_3802_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticMCapturingPostState___lam__2___boxed), 8, 3);
lean_closure_set(v___f_3802_, 0, v___f_3799_);
lean_closure_set(v___f_3802_, 1, v___x_3800_);
lean_closure_set(v___f_3802_, 2, v___x_3801_);
v___x_3803_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_getConclusionDiscrTreeKeys_spec__0___redArg(v___f_3802_, v_a_3793_, v_a_3794_, v_a_3795_, v_a_3796_);
return v___x_3803_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMCapturingPostState___boxed(lean_object* v_t_3804_, lean_object* v_preState_3805_, lean_object* v_preGoals_3806_, lean_object* v_a_3807_, lean_object* v_a_3808_, lean_object* v_a_3809_, lean_object* v_a_3810_, lean_object* v_a_3811_){
_start:
{
lean_object* v_res_3812_; 
v_res_3812_ = lp_aesop_Aesop_runTacticMCapturingPostState(v_t_3804_, v_preState_3805_, v_preGoals_3806_, v_a_3807_, v_a_3808_, v_a_3809_, v_a_3810_);
lean_dec(v_a_3810_);
lean_dec_ref(v_a_3809_);
lean_dec(v_a_3808_);
lean_dec_ref(v_a_3807_);
return v_res_3812_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticCapturingPostState(lean_object* v_t_3813_, lean_object* v_preState_3814_, lean_object* v_preGoals_3815_, lean_object* v_a_3816_, lean_object* v_a_3817_, lean_object* v_a_3818_, lean_object* v_a_3819_){
_start:
{
lean_object* v___x_3821_; lean_object* v___x_3822_; 
v___x_3821_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_3821_, 0, v_t_3813_);
v___x_3822_ = lp_aesop_Aesop_runTacticMCapturingPostState(v___x_3821_, v_preState_3814_, v_preGoals_3815_, v_a_3816_, v_a_3817_, v_a_3818_, v_a_3819_);
return v___x_3822_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticCapturingPostState___boxed(lean_object* v_t_3823_, lean_object* v_preState_3824_, lean_object* v_preGoals_3825_, lean_object* v_a_3826_, lean_object* v_a_3827_, lean_object* v_a_3828_, lean_object* v_a_3829_, lean_object* v_a_3830_){
_start:
{
lean_object* v_res_3831_; 
v_res_3831_ = lp_aesop_Aesop_runTacticCapturingPostState(v_t_3823_, v_preState_3824_, v_preGoals_3825_, v_a_3826_, v_a_3827_, v_a_3828_, v_a_3829_);
lean_dec(v_a_3829_);
lean_dec_ref(v_a_3828_);
lean_dec(v_a_3827_);
lean_dec_ref(v_a_3826_);
return v_res_3831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSeqCapturingPostState(lean_object* v_t_3832_, lean_object* v_preState_3833_, lean_object* v_preGoals_3834_, lean_object* v_a_3835_, lean_object* v_a_3836_, lean_object* v_a_3837_, lean_object* v_a_3838_){
_start:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; 
v___x_3840_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_3840_, 0, v_t_3832_);
v___x_3841_ = lp_aesop_Aesop_runTacticMCapturingPostState(v___x_3840_, v_preState_3833_, v_preGoals_3834_, v_a_3835_, v_a_3836_, v_a_3837_, v_a_3838_);
return v___x_3841_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticSeqCapturingPostState___boxed(lean_object* v_t_3842_, lean_object* v_preState_3843_, lean_object* v_preGoals_3844_, lean_object* v_a_3845_, lean_object* v_a_3846_, lean_object* v_a_3847_, lean_object* v_a_3848_, lean_object* v_a_3849_){
_start:
{
lean_object* v_res_3850_; 
v_res_3850_ = lp_aesop_Aesop_runTacticSeqCapturingPostState(v_t_3842_, v_preState_3843_, v_preGoals_3844_, v_a_3845_, v_a_3846_, v_a_3847_, v_a_3848_);
lean_dec(v_a_3848_);
lean_dec_ref(v_a_3847_);
lean_dec(v_a_3846_);
lean_dec_ref(v_a_3845_);
return v_res_3850_;
}
}
static lean_object* _init_lp_aesop_Aesop_runTacticsCapturingPostState___closed__9(void){
_start:
{
lean_object* v___x_3869_; 
v___x_3869_ = l_Array_mkArray0(lean_box(0));
return v___x_3869_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticsCapturingPostState(lean_object* v_ts_3871_, lean_object* v_preState_3872_, lean_object* v_preGoals_3873_, lean_object* v_a_3874_, lean_object* v_a_3875_, lean_object* v_a_3876_, lean_object* v_a_3877_){
_start:
{
lean_object* v_ref_3879_; uint8_t v___x_3880_; lean_object* v___x_3881_; lean_object* v___x_3882_; lean_object* v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; lean_object* v___x_3891_; lean_object* v___x_3892_; 
v_ref_3879_ = lean_ctor_get(v_a_3876_, 5);
v___x_3880_ = 0;
v___x_3881_ = l_Lean_SourceInfo_fromRef(v_ref_3879_, v___x_3880_);
v___x_3882_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__4));
v___x_3883_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__6));
v___x_3884_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__8));
v___x_3885_ = lean_obj_once(&lp_aesop_Aesop_runTacticsCapturingPostState___closed__9, &lp_aesop_Aesop_runTacticsCapturingPostState___closed__9_once, _init_lp_aesop_Aesop_runTacticsCapturingPostState___closed__9);
v___x_3886_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__10));
v___x_3887_ = l_Lean_Syntax_SepArray_ofElems(v___x_3886_, v_ts_3871_);
v___x_3888_ = l_Array_append___redArg(v___x_3885_, v___x_3887_);
lean_dec_ref(v___x_3887_);
lean_inc_n(v___x_3881_, 2);
v___x_3889_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3889_, 0, v___x_3881_);
lean_ctor_set(v___x_3889_, 1, v___x_3884_);
lean_ctor_set(v___x_3889_, 2, v___x_3888_);
v___x_3890_ = l_Lean_Syntax_node1(v___x_3881_, v___x_3883_, v___x_3889_);
v___x_3891_ = l_Lean_Syntax_node1(v___x_3881_, v___x_3882_, v___x_3890_);
v___x_3892_ = lp_aesop_Aesop_runTacticSeqCapturingPostState(v___x_3891_, v_preState_3872_, v_preGoals_3873_, v_a_3874_, v_a_3875_, v_a_3876_, v_a_3877_);
return v___x_3892_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticsCapturingPostState___boxed(lean_object* v_ts_3893_, lean_object* v_preState_3894_, lean_object* v_preGoals_3895_, lean_object* v_a_3896_, lean_object* v_a_3897_, lean_object* v_a_3898_, lean_object* v_a_3899_, lean_object* v_a_3900_){
_start:
{
lean_object* v_res_3901_; 
v_res_3901_ = lp_aesop_Aesop_runTacticsCapturingPostState(v_ts_3893_, v_preState_3894_, v_preGoals_3895_, v_a_3896_, v_a_3897_, v_a_3898_, v_a_3899_);
lean_dec(v_a_3899_);
lean_dec_ref(v_a_3898_);
lean_dec(v_a_3897_);
lean_dec_ref(v_a_3896_);
lean_dec_ref(v_ts_3893_);
return v_res_3901_;
}
}
static lean_object* _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0(void){
_start:
{
uint8_t v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; 
v___x_3902_ = 0;
v___x_3903_ = lean_box(0);
v___x_3904_ = l_Lean_SourceInfo_fromRef(v___x_3903_, v___x_3902_);
return v___x_3904_;
}
}
static lean_object* _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4(void){
_start:
{
lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; 
v___x_3912_ = ((lean_object*)(lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__3));
v___x_3913_ = lean_obj_once(&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0, &lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0_once, _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0);
v___x_3914_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3914_, 0, v___x_3913_);
lean_ctor_set(v___x_3914_, 1, v___x_3912_);
return v___x_3914_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax(uint8_t v_md_3915_, lean_object* v_k_3916_){
_start:
{
if (v_md_3915_ == 0)
{
lean_object* v___x_3917_; lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; 
v___x_3917_ = lean_obj_once(&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0, &lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0_once, _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0);
v___x_3918_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__4));
v___x_3919_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__6));
v___x_3920_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__8));
v___x_3921_ = ((lean_object*)(lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2));
v___x_3922_ = lean_obj_once(&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4, &lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4_once, _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4);
v___x_3923_ = l_Lean_Syntax_node2(v___x_3917_, v___x_3921_, v___x_3922_, v_k_3916_);
v___x_3924_ = l_Lean_Syntax_node1(v___x_3917_, v___x_3920_, v___x_3923_);
v___x_3925_ = l_Lean_Syntax_node1(v___x_3917_, v___x_3919_, v___x_3924_);
v___x_3926_ = l_Lean_Syntax_node1(v___x_3917_, v___x_3918_, v___x_3925_);
return v___x_3926_;
}
else
{
return v_k_3916_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySeqSyntax___boxed(lean_object* v_md_3927_, lean_object* v_k_3928_){
_start:
{
uint8_t v_md_boxed_3929_; lean_object* v_res_3930_; 
v_md_boxed_3929_ = lean_unbox(v_md_3927_);
v_res_3930_ = lp_aesop_Aesop_withAllTransparencySeqSyntax(v_md_boxed_3929_, v_k_3928_);
return v_res_3930_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySyntax(uint8_t v_md_3931_, lean_object* v_k_3932_){
_start:
{
if (v_md_3931_ == 0)
{
lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; lean_object* v___x_3938_; lean_object* v___x_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; 
v___x_3933_ = lean_obj_once(&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0, &lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0_once, _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__0);
v___x_3934_ = ((lean_object*)(lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__2));
v___x_3935_ = lean_obj_once(&lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4, &lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4_once, _init_lp_aesop_Aesop_withAllTransparencySeqSyntax___closed__4);
v___x_3936_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__4));
v___x_3937_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__6));
v___x_3938_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__8));
v___x_3939_ = l_Lean_Syntax_node1(v___x_3933_, v___x_3938_, v_k_3932_);
v___x_3940_ = l_Lean_Syntax_node1(v___x_3933_, v___x_3937_, v___x_3939_);
v___x_3941_ = l_Lean_Syntax_node1(v___x_3933_, v___x_3936_, v___x_3940_);
v___x_3942_ = l_Lean_Syntax_node2(v___x_3933_, v___x_3934_, v___x_3935_, v___x_3941_);
return v___x_3942_;
}
else
{
return v_k_3932_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAllTransparencySyntax___boxed(lean_object* v_md_3943_, lean_object* v_k_3944_){
_start:
{
uint8_t v_md_boxed_3945_; lean_object* v_res_3946_; 
v_md_boxed_3945_ = lean_unbox(v_md_3943_);
v_res_3946_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_boxed_3945_, v_k_3944_);
return v_res_3946_;
}
}
static lean_object* _init_lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_3948_; lean_object* v___x_3949_; 
v___x_3948_ = ((lean_object*)(lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0));
v___x_3949_ = lean_string_utf8_byte_size(v___x_3948_);
return v___x_3949_;
}
}
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg(lean_object* v_s_3950_){
_start:
{
lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v___x_3953_; uint8_t v___x_3954_; 
v___x_3951_ = ((lean_object*)(lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__0));
v___x_3952_ = lean_string_utf8_byte_size(v_s_3950_);
v___x_3953_ = lean_obj_once(&lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1, &lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1_once, _init_lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg___closed__1);
v___x_3954_ = lean_nat_dec_le(v___x_3953_, v___x_3952_);
if (v___x_3954_ == 0)
{
lean_object* v___x_3955_; 
lean_dec_ref(v_s_3950_);
v___x_3955_ = lean_box(0);
return v___x_3955_;
}
else
{
lean_object* v___x_3956_; uint8_t v___x_3957_; 
v___x_3956_ = lean_unsigned_to_nat(0u);
v___x_3957_ = lean_string_memcmp(v_s_3950_, v___x_3951_, v___x_3956_, v___x_3956_, v___x_3953_);
if (v___x_3957_ == 0)
{
lean_object* v___x_3958_; 
lean_dec_ref(v_s_3950_);
v___x_3958_ = lean_box(0);
return v___x_3958_;
}
else
{
lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___x_3961_; lean_object* v___x_3962_; 
lean_inc_ref(v_s_3950_);
v___x_3959_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3959_, 0, v_s_3950_);
lean_ctor_set(v___x_3959_, 1, v___x_3956_);
lean_ctor_set(v___x_3959_, 2, v___x_3952_);
v___x_3960_ = l_String_Slice_pos_x21(v___x_3959_, v___x_3953_);
lean_dec_ref_known(v___x_3959_, 3);
v___x_3961_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3961_, 0, v_s_3950_);
lean_ctor_set(v___x_3961_, 1, v___x_3960_);
lean_ctor_set(v___x_3961_, 2, v___x_3952_);
v___x_3962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3962_, 0, v___x_3961_);
return v___x_3962_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0(lean_object* v_s_3963_, lean_object* v_pat_3964_){
_start:
{
lean_object* v___x_3965_; 
v___x_3965_ = lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg(v_s_3963_);
return v___x_3965_;
}
}
LEAN_EXPORT lean_object* lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___boxed(lean_object* v_s_3966_, lean_object* v_pat_3967_){
_start:
{
lean_object* v_res_3968_; 
v_res_3968_ = lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0(v_s_3966_, v_pat_3967_);
lean_dec_ref(v_pat_3967_);
return v_res_3968_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__1(lean_object* v_a_3969_, lean_object* v_a_3970_){
_start:
{
if (lean_obj_tag(v_a_3969_) == 0)
{
lean_object* v___x_3971_; 
v___x_3971_ = l_List_reverse___redArg(v_a_3970_);
return v___x_3971_;
}
else
{
lean_object* v_head_3972_; lean_object* v_tail_3973_; lean_object* v___x_3975_; uint8_t v_isShared_3976_; uint8_t v_isSharedCheck_3986_; 
v_head_3972_ = lean_ctor_get(v_a_3969_, 0);
v_tail_3973_ = lean_ctor_get(v_a_3969_, 1);
v_isSharedCheck_3986_ = !lean_is_exclusive(v_a_3969_);
if (v_isSharedCheck_3986_ == 0)
{
v___x_3975_ = v_a_3969_;
v_isShared_3976_ = v_isSharedCheck_3986_;
goto v_resetjp_3974_;
}
else
{
lean_inc(v_tail_3973_);
lean_inc(v_head_3972_);
lean_dec(v_a_3969_);
v___x_3975_ = lean_box(0);
v_isShared_3976_ = v_isSharedCheck_3986_;
goto v_resetjp_3974_;
}
v_resetjp_3974_:
{
lean_object* v___y_3978_; lean_object* v___x_3983_; 
lean_inc(v_head_3972_);
v___x_3983_ = lp_aesop_String_dropPrefix_x3f___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__0___redArg(v_head_3972_);
if (lean_obj_tag(v___x_3983_) == 0)
{
v___y_3978_ = v_head_3972_;
goto v___jp_3977_;
}
else
{
lean_object* v_val_3984_; lean_object* v___x_3985_; 
lean_dec(v_head_3972_);
v_val_3984_ = lean_ctor_get(v___x_3983_, 0);
lean_inc(v_val_3984_);
lean_dec_ref_known(v___x_3983_, 1);
v___x_3985_ = l_String_Slice_toString(v_val_3984_);
lean_dec(v_val_3984_);
v___y_3978_ = v___x_3985_;
goto v___jp_3977_;
}
v___jp_3977_:
{
lean_object* v___x_3980_; 
if (v_isShared_3976_ == 0)
{
lean_ctor_set(v___x_3975_, 1, v_a_3970_);
lean_ctor_set(v___x_3975_, 0, v___y_3978_);
v___x_3980_ = v___x_3975_;
goto v_reusejp_3979_;
}
else
{
lean_object* v_reuseFailAlloc_3982_; 
v_reuseFailAlloc_3982_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3982_, 0, v___y_3978_);
lean_ctor_set(v_reuseFailAlloc_3982_, 1, v_a_3970_);
v___x_3980_ = v_reuseFailAlloc_3982_;
goto v_reusejp_3979_;
}
v_reusejp_3979_:
{
v_a_3969_ = v_tail_3973_;
v_a_3970_ = v___x_3980_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent(lean_object* v_s_3988_){
_start:
{
lean_object* v___x_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; 
v___x_3989_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___closed__0));
v___x_3990_ = lean_unsigned_to_nat(0u);
v___x_3991_ = lean_box(0);
v___x_3992_ = l_String_splitOnAux(v_s_3988_, v___x_3989_, v___x_3990_, v___x_3990_, v___x_3990_, v___x_3991_);
v___x_3993_ = lp_aesop_List_mapTR_loop___at___00__private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent_spec__1(v___x_3992_, v___x_3991_);
v___x_3994_ = l_String_intercalate(v___x_3989_, v___x_3993_);
return v___x_3994_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent___boxed(lean_object* v_s_3995_){
_start:
{
lean_object* v_res_3996_; 
v_res_3996_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent(v_s_3995_);
lean_dec_ref(v_s_3995_);
return v_res_3996_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0(lean_object* v_x_3998_){
_start:
{
lean_object* v___x_3999_; 
v___x_3999_ = ((lean_object*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___closed__0));
return v___x_3999_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0___boxed(lean_object* v_x_4000_){
_start:
{
lean_object* v_res_4001_; 
v_res_4001_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___lam__0(v_x_4000_);
lean_dec_ref(v_x_4000_);
return v_res_4001_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(lean_object* v_ref_4008_, lean_object* v_suggestion_4009_, lean_object* v_origSpan_x3f_4010_, lean_object* v_a_4011_, lean_object* v_a_4012_){
_start:
{
lean_object* v___x_4014_; lean_object* v___x_4015_; 
v___x_4014_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__4));
v___x_4015_ = l_Lean_PrettyPrinter_ppCategory(v___x_4014_, v_suggestion_4009_, v_a_4011_, v_a_4012_);
if (lean_obj_tag(v___x_4015_) == 0)
{
lean_object* v_a_4016_; lean_object* v___x_4018_; uint8_t v_isShared_4019_; uint8_t v_isSharedCheck_4057_; 
v_a_4016_ = lean_ctor_get(v___x_4015_, 0);
v_isSharedCheck_4057_ = !lean_is_exclusive(v___x_4015_);
if (v_isSharedCheck_4057_ == 0)
{
v___x_4018_ = v___x_4015_;
v_isShared_4019_ = v_isSharedCheck_4057_;
goto v_resetjp_4017_;
}
else
{
lean_inc(v_a_4016_);
lean_dec(v___x_4015_);
v___x_4018_ = lean_box(0);
v_isShared_4019_ = v_isSharedCheck_4057_;
goto v_resetjp_4017_;
}
v_resetjp_4017_:
{
lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v___x_4022_; lean_object* v___y_4024_; 
v___x_4020_ = l_Std_Format_defWidth;
v___x_4021_ = lean_unsigned_to_nat(0u);
lean_inc(v_a_4016_);
v___x_4022_ = l_Std_Format_pretty(v_a_4016_, v___x_4020_, v___x_4021_, v___x_4021_);
if (lean_obj_tag(v_origSpan_x3f_4010_) == 0)
{
lean_inc(v_ref_4008_);
v___y_4024_ = v_ref_4008_;
goto v___jp_4023_;
}
else
{
lean_object* v_val_4056_; 
v_val_4056_ = lean_ctor_get(v_origSpan_x3f_4010_, 0);
lean_inc(v_val_4056_);
v___y_4024_ = v_val_4056_;
goto v___jp_4023_;
}
v___jp_4023_:
{
uint8_t v___x_4025_; lean_object* v___x_4026_; 
v___x_4025_ = 0;
v___x_4026_ = l_Lean_Syntax_getRange_x3f(v___y_4024_, v___x_4025_);
lean_dec(v___y_4024_);
if (lean_obj_tag(v___x_4026_) == 1)
{
lean_object* v_val_4027_; lean_object* v___x_4029_; uint8_t v_isShared_4030_; uint8_t v_isSharedCheck_4051_; 
lean_del_object(v___x_4018_);
v_val_4027_ = lean_ctor_get(v___x_4026_, 0);
v_isSharedCheck_4051_ = !lean_is_exclusive(v___x_4026_);
if (v_isSharedCheck_4051_ == 0)
{
v___x_4029_ = v___x_4026_;
v_isShared_4030_ = v_isSharedCheck_4051_;
goto v_resetjp_4028_;
}
else
{
lean_inc(v_val_4027_);
lean_dec(v___x_4026_);
v___x_4029_ = lean_box(0);
v_isShared_4030_ = v_isSharedCheck_4051_;
goto v_resetjp_4028_;
}
v_resetjp_4028_:
{
lean_object* v_fileMap_4031_; lean_object* v___x_4032_; lean_object* v_fst_4033_; lean_object* v_snd_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4043_; 
v_fileMap_4031_ = lean_ctor_get(v_a_4011_, 1);
lean_inc_ref(v_fileMap_4031_);
v___x_4032_ = l_Lean_Meta_Tactic_TryThis_getIndentAndColumn(v_fileMap_4031_, v_val_4027_);
v_fst_4033_ = lean_ctor_get(v___x_4032_, 0);
lean_inc(v_fst_4033_);
v_snd_4034_ = lean_ctor_get(v___x_4032_, 1);
lean_inc(v_snd_4034_);
lean_dec_ref(v___x_4032_);
v___x_4035_ = l_Std_Format_pretty(v_a_4016_, v___x_4020_, v_fst_4033_, v_snd_4034_);
v___x_4036_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_addTryThisTacticSeqSuggestion_dedent(v___x_4035_);
lean_dec_ref(v___x_4035_);
v___x_4037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4037_, 0, v___x_4036_);
v___x_4038_ = ((lean_object*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__1));
v___x_4039_ = lean_box(0);
v___x_4040_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4040_, 0, v___x_4022_);
v___x_4041_ = l_Lean_MessageData_ofFormat(v___x_4040_);
if (v_isShared_4030_ == 0)
{
lean_ctor_set(v___x_4029_, 0, v___x_4041_);
v___x_4043_ = v___x_4029_;
goto v_reusejp_4042_;
}
else
{
lean_object* v_reuseFailAlloc_4050_; 
v_reuseFailAlloc_4050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4050_, 0, v___x_4041_);
v___x_4043_ = v_reuseFailAlloc_4050_;
goto v_reusejp_4042_;
}
v_reusejp_4042_:
{
lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; uint8_t v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; 
v___x_4044_ = ((lean_object*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__2));
v___x_4045_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_4045_, 0, v___x_4037_);
lean_ctor_set(v___x_4045_, 1, v___x_4038_);
lean_ctor_set(v___x_4045_, 2, v___x_4039_);
lean_ctor_set(v___x_4045_, 3, v___x_4039_);
lean_ctor_set(v___x_4045_, 4, v___x_4043_);
lean_ctor_set(v___x_4045_, 5, v___x_4044_);
v___x_4046_ = ((lean_object*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___closed__3));
v___x_4047_ = 4;
v___x_4048_ = l_Lean_MessageData_nil;
v___x_4049_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_ref_4008_, v___x_4045_, v_origSpan_x3f_4010_, v___x_4046_, v___x_4039_, v___x_4047_, v___x_4048_, v_a_4011_, v_a_4012_);
return v___x_4049_;
}
}
}
else
{
lean_object* v___x_4052_; lean_object* v___x_4054_; 
lean_dec(v___x_4026_);
lean_dec_ref(v___x_4022_);
lean_dec(v_a_4016_);
lean_dec(v_origSpan_x3f_4010_);
lean_dec(v_ref_4008_);
v___x_4052_ = lean_box(0);
if (v_isShared_4019_ == 0)
{
lean_ctor_set(v___x_4018_, 0, v___x_4052_);
v___x_4054_ = v___x_4018_;
goto v_reusejp_4053_;
}
else
{
lean_object* v_reuseFailAlloc_4055_; 
v_reuseFailAlloc_4055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4055_, 0, v___x_4052_);
v___x_4054_ = v_reuseFailAlloc_4055_;
goto v_reusejp_4053_;
}
v_reusejp_4053_:
{
return v___x_4054_;
}
}
}
}
}
else
{
lean_object* v_a_4058_; lean_object* v___x_4060_; uint8_t v_isShared_4061_; uint8_t v_isSharedCheck_4065_; 
lean_dec(v_origSpan_x3f_4010_);
lean_dec(v_ref_4008_);
v_a_4058_ = lean_ctor_get(v___x_4015_, 0);
v_isSharedCheck_4065_ = !lean_is_exclusive(v___x_4015_);
if (v_isSharedCheck_4065_ == 0)
{
v___x_4060_ = v___x_4015_;
v_isShared_4061_ = v_isSharedCheck_4065_;
goto v_resetjp_4059_;
}
else
{
lean_inc(v_a_4058_);
lean_dec(v___x_4015_);
v___x_4060_ = lean_box(0);
v_isShared_4061_ = v_isSharedCheck_4065_;
goto v_resetjp_4059_;
}
v_resetjp_4059_:
{
lean_object* v___x_4063_; 
if (v_isShared_4061_ == 0)
{
v___x_4063_ = v___x_4060_;
goto v_reusejp_4062_;
}
else
{
lean_object* v_reuseFailAlloc_4064_; 
v_reuseFailAlloc_4064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4064_, 0, v_a_4058_);
v___x_4063_ = v_reuseFailAlloc_4064_;
goto v_reusejp_4062_;
}
v_reusejp_4062_:
{
return v___x_4063_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg___boxed(lean_object* v_ref_4066_, lean_object* v_suggestion_4067_, lean_object* v_origSpan_x3f_4068_, lean_object* v_a_4069_, lean_object* v_a_4070_, lean_object* v_a_4071_){
_start:
{
lean_object* v_res_4072_; 
v_res_4072_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(v_ref_4066_, v_suggestion_4067_, v_origSpan_x3f_4068_, v_a_4069_, v_a_4070_);
lean_dec(v_a_4070_);
lean_dec_ref(v_a_4069_);
return v_res_4072_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion(lean_object* v_ref_4073_, lean_object* v_suggestion_4074_, lean_object* v_origSpan_x3f_4075_, lean_object* v_a_4076_, lean_object* v_a_4077_, lean_object* v_a_4078_, lean_object* v_a_4079_){
_start:
{
lean_object* v___x_4081_; 
v___x_4081_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(v_ref_4073_, v_suggestion_4074_, v_origSpan_x3f_4075_, v_a_4078_, v_a_4079_);
return v___x_4081_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___boxed(lean_object* v_ref_4082_, lean_object* v_suggestion_4083_, lean_object* v_origSpan_x3f_4084_, lean_object* v_a_4085_, lean_object* v_a_4086_, lean_object* v_a_4087_, lean_object* v_a_4088_, lean_object* v_a_4089_){
_start:
{
lean_object* v_res_4090_; 
v_res_4090_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion(v_ref_4082_, v_suggestion_4083_, v_origSpan_x3f_4084_, v_a_4085_, v_a_4086_, v_a_4087_, v_a_4088_);
lean_dec(v_a_4088_);
lean_dec_ref(v_a_4087_);
lean_dec(v_a_4086_);
lean_dec_ref(v_a_4085_);
return v_res_4090_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_elabPattern_adjustCtx(lean_object* v_old_4091_){
_start:
{
lean_object* v_declName_x3f_4092_; lean_object* v_macroStack_4093_; lean_object* v_autoBoundImplicitForbidden_4094_; uint8_t v_implicitLambda_4095_; uint8_t v_heedElabAsElim_4096_; uint8_t v_isMetaSection_4097_; lean_object* v_tacSnap_x3f_4098_; uint8_t v_checkDeprecated_4099_; lean_object* v_fixedTermElabs_4100_; lean_object* v___x_4102_; uint8_t v_isShared_4103_; uint8_t v_isSharedCheck_4111_; 
v_declName_x3f_4092_ = lean_ctor_get(v_old_4091_, 0);
v_macroStack_4093_ = lean_ctor_get(v_old_4091_, 1);
v_autoBoundImplicitForbidden_4094_ = lean_ctor_get(v_old_4091_, 3);
v_implicitLambda_4095_ = lean_ctor_get_uint8(v_old_4091_, sizeof(void*)*8 + 2);
v_heedElabAsElim_4096_ = lean_ctor_get_uint8(v_old_4091_, sizeof(void*)*8 + 3);
v_isMetaSection_4097_ = lean_ctor_get_uint8(v_old_4091_, sizeof(void*)*8 + 5);
v_tacSnap_x3f_4098_ = lean_ctor_get(v_old_4091_, 6);
v_checkDeprecated_4099_ = lean_ctor_get_uint8(v_old_4091_, sizeof(void*)*8 + 10);
v_fixedTermElabs_4100_ = lean_ctor_get(v_old_4091_, 7);
v_isSharedCheck_4111_ = !lean_is_exclusive(v_old_4091_);
if (v_isSharedCheck_4111_ == 0)
{
lean_object* v_unused_4112_; lean_object* v_unused_4113_; lean_object* v_unused_4114_; 
v_unused_4112_ = lean_ctor_get(v_old_4091_, 5);
lean_dec(v_unused_4112_);
v_unused_4113_ = lean_ctor_get(v_old_4091_, 4);
lean_dec(v_unused_4113_);
v_unused_4114_ = lean_ctor_get(v_old_4091_, 2);
lean_dec(v_unused_4114_);
v___x_4102_ = v_old_4091_;
v_isShared_4103_ = v_isSharedCheck_4111_;
goto v_resetjp_4101_;
}
else
{
lean_inc(v_fixedTermElabs_4100_);
lean_inc(v_tacSnap_x3f_4098_);
lean_inc(v_autoBoundImplicitForbidden_4094_);
lean_inc(v_macroStack_4093_);
lean_inc(v_declName_x3f_4092_);
lean_dec(v_old_4091_);
v___x_4102_ = lean_box(0);
v_isShared_4103_ = v_isSharedCheck_4111_;
goto v_resetjp_4101_;
}
v_resetjp_4101_:
{
uint8_t v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; uint8_t v___x_4107_; lean_object* v___x_4109_; 
v___x_4104_ = 0;
v___x_4105_ = lean_box(0);
v___x_4106_ = lean_box(1);
v___x_4107_ = 1;
if (v_isShared_4103_ == 0)
{
lean_ctor_set(v___x_4102_, 5, v___x_4106_);
lean_ctor_set(v___x_4102_, 4, v___x_4106_);
lean_ctor_set(v___x_4102_, 2, v___x_4105_);
v___x_4109_ = v___x_4102_;
goto v_reusejp_4108_;
}
else
{
lean_object* v_reuseFailAlloc_4110_; 
v_reuseFailAlloc_4110_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v_reuseFailAlloc_4110_, 0, v_declName_x3f_4092_);
lean_ctor_set(v_reuseFailAlloc_4110_, 1, v_macroStack_4093_);
lean_ctor_set(v_reuseFailAlloc_4110_, 2, v___x_4105_);
lean_ctor_set(v_reuseFailAlloc_4110_, 3, v_autoBoundImplicitForbidden_4094_);
lean_ctor_set(v_reuseFailAlloc_4110_, 4, v___x_4106_);
lean_ctor_set(v_reuseFailAlloc_4110_, 5, v___x_4106_);
lean_ctor_set(v_reuseFailAlloc_4110_, 6, v_tacSnap_x3f_4098_);
lean_ctor_set(v_reuseFailAlloc_4110_, 7, v_fixedTermElabs_4100_);
lean_ctor_set_uint8(v_reuseFailAlloc_4110_, sizeof(void*)*8 + 2, v_implicitLambda_4095_);
lean_ctor_set_uint8(v_reuseFailAlloc_4110_, sizeof(void*)*8 + 3, v_heedElabAsElim_4096_);
lean_ctor_set_uint8(v_reuseFailAlloc_4110_, sizeof(void*)*8 + 5, v_isMetaSection_4097_);
lean_ctor_set_uint8(v_reuseFailAlloc_4110_, sizeof(void*)*8 + 10, v_checkDeprecated_4099_);
v___x_4109_ = v_reuseFailAlloc_4110_;
goto v_reusejp_4108_;
}
v_reusejp_4108_:
{
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8, v___x_4104_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 1, v___x_4104_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 4, v___x_4104_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 6, v___x_4107_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 7, v___x_4107_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 8, v___x_4104_);
lean_ctor_set_uint8(v___x_4109_, sizeof(void*)*8 + 9, v___x_4104_);
return v___x_4109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabPattern(lean_object* v_stx_4115_, lean_object* v_a_4116_, lean_object* v_a_4117_, lean_object* v_a_4118_, lean_object* v_a_4119_, lean_object* v_a_4120_, lean_object* v_a_4121_){
_start:
{
lean_object* v___x_4123_; uint8_t v___x_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; lean_object* v___x_4127_; lean_object* v_fileName_4128_; lean_object* v_fileMap_4129_; lean_object* v_options_4130_; lean_object* v_currRecDepth_4131_; lean_object* v_maxRecDepth_4132_; lean_object* v_ref_4133_; lean_object* v_currNamespace_4134_; lean_object* v_openDecls_4135_; lean_object* v_initHeartbeats_4136_; lean_object* v_maxHeartbeats_4137_; lean_object* v_quotContext_4138_; lean_object* v_currMacroScope_4139_; uint8_t v_diag_4140_; lean_object* v_cancelTk_x3f_4141_; uint8_t v_suppressElabErrors_4142_; lean_object* v_inheritedTraceOptions_4143_; uint8_t v___x_4144_; lean_object* v_ref_4145_; lean_object* v___x_4146_; lean_object* v___x_4147_; lean_object* v___x_4148_; 
v___x_4123_ = lean_box(0);
v___x_4124_ = 1;
v___x_4125_ = lean_box(v___x_4124_);
v___x_4126_ = lean_box(v___x_4124_);
lean_inc(v_stx_4115_);
v___x_4127_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_4127_, 0, v_stx_4115_);
lean_closure_set(v___x_4127_, 1, v___x_4123_);
lean_closure_set(v___x_4127_, 2, v___x_4125_);
lean_closure_set(v___x_4127_, 3, v___x_4126_);
v_fileName_4128_ = lean_ctor_get(v_a_4120_, 0);
v_fileMap_4129_ = lean_ctor_get(v_a_4120_, 1);
v_options_4130_ = lean_ctor_get(v_a_4120_, 2);
v_currRecDepth_4131_ = lean_ctor_get(v_a_4120_, 3);
v_maxRecDepth_4132_ = lean_ctor_get(v_a_4120_, 4);
v_ref_4133_ = lean_ctor_get(v_a_4120_, 5);
v_currNamespace_4134_ = lean_ctor_get(v_a_4120_, 6);
v_openDecls_4135_ = lean_ctor_get(v_a_4120_, 7);
v_initHeartbeats_4136_ = lean_ctor_get(v_a_4120_, 8);
v_maxHeartbeats_4137_ = lean_ctor_get(v_a_4120_, 9);
v_quotContext_4138_ = lean_ctor_get(v_a_4120_, 10);
v_currMacroScope_4139_ = lean_ctor_get(v_a_4120_, 11);
v_diag_4140_ = lean_ctor_get_uint8(v_a_4120_, sizeof(void*)*14);
v_cancelTk_x3f_4141_ = lean_ctor_get(v_a_4120_, 12);
v_suppressElabErrors_4142_ = lean_ctor_get_uint8(v_a_4120_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4143_ = lean_ctor_get(v_a_4120_, 13);
v___x_4144_ = 1;
v_ref_4145_ = l_Lean_replaceRef(v_stx_4115_, v_ref_4133_);
lean_dec(v_stx_4115_);
lean_inc_ref(v_inheritedTraceOptions_4143_);
lean_inc(v_cancelTk_x3f_4141_);
lean_inc(v_currMacroScope_4139_);
lean_inc(v_quotContext_4138_);
lean_inc(v_maxHeartbeats_4137_);
lean_inc(v_initHeartbeats_4136_);
lean_inc(v_openDecls_4135_);
lean_inc(v_currNamespace_4134_);
lean_inc(v_maxRecDepth_4132_);
lean_inc(v_currRecDepth_4131_);
lean_inc_ref(v_options_4130_);
lean_inc_ref(v_fileMap_4129_);
lean_inc_ref(v_fileName_4128_);
v___x_4146_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4146_, 0, v_fileName_4128_);
lean_ctor_set(v___x_4146_, 1, v_fileMap_4129_);
lean_ctor_set(v___x_4146_, 2, v_options_4130_);
lean_ctor_set(v___x_4146_, 3, v_currRecDepth_4131_);
lean_ctor_set(v___x_4146_, 4, v_maxRecDepth_4132_);
lean_ctor_set(v___x_4146_, 5, v_ref_4145_);
lean_ctor_set(v___x_4146_, 6, v_currNamespace_4134_);
lean_ctor_set(v___x_4146_, 7, v_openDecls_4135_);
lean_ctor_set(v___x_4146_, 8, v_initHeartbeats_4136_);
lean_ctor_set(v___x_4146_, 9, v_maxHeartbeats_4137_);
lean_ctor_set(v___x_4146_, 10, v_quotContext_4138_);
lean_ctor_set(v___x_4146_, 11, v_currMacroScope_4139_);
lean_ctor_set(v___x_4146_, 12, v_cancelTk_x3f_4141_);
lean_ctor_set(v___x_4146_, 13, v_inheritedTraceOptions_4143_);
lean_ctor_set_uint8(v___x_4146_, sizeof(void*)*14, v_diag_4140_);
lean_ctor_set_uint8(v___x_4146_, sizeof(void*)*14 + 1, v_suppressElabErrors_4142_);
lean_inc_ref(v_a_4116_);
v___x_4147_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_elabPattern_adjustCtx(v_a_4116_);
v___x_4148_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_4127_, v___x_4144_, v___x_4147_, v_a_4117_, v_a_4118_, v_a_4119_, v___x_4146_, v_a_4121_);
lean_dec_ref_known(v___x_4146_, 14);
lean_dec_ref(v___x_4147_);
return v___x_4148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabPattern___boxed(lean_object* v_stx_4149_, lean_object* v_a_4150_, lean_object* v_a_4151_, lean_object* v_a_4152_, lean_object* v_a_4153_, lean_object* v_a_4154_, lean_object* v_a_4155_, lean_object* v_a_4156_){
_start:
{
lean_object* v_res_4157_; 
v_res_4157_ = lp_aesop_Aesop_elabPattern(v_stx_4149_, v_a_4150_, v_a_4151_, v_a_4152_, v_a_4153_, v_a_4154_, v_a_4155_);
lean_dec(v_a_4155_);
lean_dec_ref(v_a_4154_);
lean_dec(v_a_4153_);
lean_dec_ref(v_a_4152_);
lean_dec(v_a_4151_);
lean_dec_ref(v_a_4150_);
return v_res_4157_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0(lean_object* v_name_4158_, lean_object* v_decl_4159_, lean_object* v_ref_4160_){
_start:
{
lean_object* v_defValue_4162_; lean_object* v_descr_4163_; lean_object* v_deprecation_x3f_4164_; lean_object* v___x_4165_; uint8_t v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_4168_; 
v_defValue_4162_ = lean_ctor_get(v_decl_4159_, 0);
v_descr_4163_ = lean_ctor_get(v_decl_4159_, 1);
v_deprecation_x3f_4164_ = lean_ctor_get(v_decl_4159_, 2);
v___x_4165_ = lean_alloc_ctor(1, 0, 1);
v___x_4166_ = lean_unbox(v_defValue_4162_);
lean_ctor_set_uint8(v___x_4165_, 0, v___x_4166_);
lean_inc(v_deprecation_x3f_4164_);
lean_inc_ref(v_descr_4163_);
lean_inc_n(v_name_4158_, 2);
v___x_4167_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4167_, 0, v_name_4158_);
lean_ctor_set(v___x_4167_, 1, v_ref_4160_);
lean_ctor_set(v___x_4167_, 2, v___x_4165_);
lean_ctor_set(v___x_4167_, 3, v_descr_4163_);
lean_ctor_set(v___x_4167_, 4, v_deprecation_x3f_4164_);
v___x_4168_ = lean_register_option(v_name_4158_, v___x_4167_);
if (lean_obj_tag(v___x_4168_) == 0)
{
lean_object* v___x_4170_; uint8_t v_isShared_4171_; uint8_t v_isSharedCheck_4176_; 
v_isSharedCheck_4176_ = !lean_is_exclusive(v___x_4168_);
if (v_isSharedCheck_4176_ == 0)
{
lean_object* v_unused_4177_; 
v_unused_4177_ = lean_ctor_get(v___x_4168_, 0);
lean_dec(v_unused_4177_);
v___x_4170_ = v___x_4168_;
v_isShared_4171_ = v_isSharedCheck_4176_;
goto v_resetjp_4169_;
}
else
{
lean_dec(v___x_4168_);
v___x_4170_ = lean_box(0);
v_isShared_4171_ = v_isSharedCheck_4176_;
goto v_resetjp_4169_;
}
v_resetjp_4169_:
{
lean_object* v___x_4172_; lean_object* v___x_4174_; 
lean_inc(v_defValue_4162_);
v___x_4172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4172_, 0, v_name_4158_);
lean_ctor_set(v___x_4172_, 1, v_defValue_4162_);
if (v_isShared_4171_ == 0)
{
lean_ctor_set(v___x_4170_, 0, v___x_4172_);
v___x_4174_ = v___x_4170_;
goto v_reusejp_4173_;
}
else
{
lean_object* v_reuseFailAlloc_4175_; 
v_reuseFailAlloc_4175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4175_, 0, v___x_4172_);
v___x_4174_ = v_reuseFailAlloc_4175_;
goto v_reusejp_4173_;
}
v_reusejp_4173_:
{
return v___x_4174_;
}
}
}
else
{
lean_object* v_a_4178_; lean_object* v___x_4180_; uint8_t v_isShared_4181_; uint8_t v_isSharedCheck_4185_; 
lean_dec(v_name_4158_);
v_a_4178_ = lean_ctor_get(v___x_4168_, 0);
v_isSharedCheck_4185_ = !lean_is_exclusive(v___x_4168_);
if (v_isSharedCheck_4185_ == 0)
{
v___x_4180_ = v___x_4168_;
v_isShared_4181_ = v_isSharedCheck_4185_;
goto v_resetjp_4179_;
}
else
{
lean_inc(v_a_4178_);
lean_dec(v___x_4168_);
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
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_4186_, lean_object* v_decl_4187_, lean_object* v_ref_4188_, lean_object* v_a_4189_){
_start:
{
lean_object* v_res_4190_; 
v_res_4190_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0(v_name_4186_, v_decl_4187_, v_ref_4188_);
lean_dec_ref(v_decl_4187_);
return v_res_4190_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_4208_; lean_object* v___x_4209_; lean_object* v___x_4210_; lean_object* v___x_4211_; 
v___x_4208_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_));
v___x_4209_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_));
v___x_4210_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_));
v___x_4211_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4__spec__0(v___x_4208_, v___x_4209_, v___x_4210_);
return v___x_4211_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4____boxed(lean_object* v_a_4212_){
_start:
{
lean_object* v_res_4213_; 
v_res_4213_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_();
return v_res_4213_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0(size_t v_sz_4214_, size_t v_i_4215_, lean_object* v_bs_4216_){
_start:
{
uint8_t v___x_4217_; 
v___x_4217_ = lean_usize_dec_lt(v_i_4215_, v_sz_4214_);
if (v___x_4217_ == 0)
{
return v_bs_4216_;
}
else
{
lean_object* v_v_4218_; lean_object* v___x_4219_; lean_object* v_bs_x27_4220_; lean_object* v___x_4221_; size_t v___x_4222_; size_t v___x_4223_; lean_object* v___x_4224_; 
v_v_4218_ = lean_array_uget(v_bs_4216_, v_i_4215_);
v___x_4219_ = lean_unsigned_to_nat(0u);
v_bs_x27_4220_ = lean_array_uset(v_bs_4216_, v_i_4215_, v___x_4219_);
v___x_4221_ = l_Lean_MessageData_ofSyntax(v_v_4218_);
v___x_4222_ = ((size_t)1ULL);
v___x_4223_ = lean_usize_add(v_i_4215_, v___x_4222_);
v___x_4224_ = lean_array_uset(v_bs_x27_4220_, v_i_4215_, v___x_4221_);
v_i_4215_ = v___x_4223_;
v_bs_4216_ = v___x_4224_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0___boxed(lean_object* v_sz_4226_, lean_object* v_i_4227_, lean_object* v_bs_4228_){
_start:
{
size_t v_sz_boxed_4229_; size_t v_i_boxed_4230_; lean_object* v_res_4231_; 
v_sz_boxed_4229_ = lean_unbox_usize(v_sz_4226_);
lean_dec(v_sz_4226_);
v_i_boxed_4230_ = lean_unbox_usize(v_i_4227_);
lean_dec(v_i_4227_);
v_res_4231_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0(v_sz_boxed_4229_, v_i_boxed_4230_, v_bs_4228_);
return v_res_4231_;
}
}
static lean_object* _init_lp_aesop_Aesop_tacticsToMessageData___closed__1(void){
_start:
{
lean_object* v___x_4234_; lean_object* v___x_4235_; 
v___x_4234_ = ((lean_object*)(lp_aesop_Aesop_tacticsToMessageData___closed__0));
v___x_4235_ = l_Lean_MessageData_ofFormat(v___x_4234_);
return v___x_4235_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticsToMessageData(lean_object* v_ts_4236_){
_start:
{
size_t v_sz_4237_; size_t v___x_4238_; lean_object* v___x_4239_; lean_object* v___x_4240_; lean_object* v___x_4241_; lean_object* v___x_4242_; 
v_sz_4237_ = lean_array_size(v_ts_4236_);
v___x_4238_ = ((size_t)0ULL);
v___x_4239_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tacticsToMessageData_spec__0(v_sz_4237_, v___x_4238_, v_ts_4236_);
v___x_4240_ = lean_array_to_list(v___x_4239_);
v___x_4241_ = lean_obj_once(&lp_aesop_Aesop_tacticsToMessageData___closed__1, &lp_aesop_Aesop_tacticsToMessageData___closed__1_once, _init_lp_aesop_Aesop_tacticsToMessageData___closed__1);
v___x_4242_ = l_Lean_MessageData_joinSep(v___x_4240_, v___x_4241_);
return v___x_4242_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2(void){
_start:
{
lean_object* v___x_4246_; lean_object* v___x_4247_; 
v___x_4246_ = lean_box(0);
v___x_4247_ = l_Lean_Expr_sort___override(v___x_4246_);
return v___x_4247_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl(lean_object* v_name_4248_){
_start:
{
lean_object* v___x_4249_; lean_object* v___x_4250_; lean_object* v___x_4251_; uint8_t v___x_4252_; uint8_t v___x_4253_; lean_object* v___x_4254_; 
v___x_4249_ = lean_unsigned_to_nat(0u);
v___x_4250_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__1));
v___x_4251_ = lean_obj_once(&lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2, &lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2_once, _init_lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl___closed__2);
v___x_4252_ = 0;
v___x_4253_ = 0;
v___x_4254_ = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(v___x_4254_, 0, v___x_4249_);
lean_ctor_set(v___x_4254_, 1, v___x_4250_);
lean_ctor_set(v___x_4254_, 2, v_name_4248_);
lean_ctor_set(v___x_4254_, 3, v___x_4251_);
lean_ctor_set_uint8(v___x_4254_, sizeof(void*)*4, v___x_4252_);
lean_ctor_set_uint8(v___x_4254_, sizeof(void*)*4 + 1, v___x_4253_);
return v___x_4254_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go(lean_object* v_suggestions_4255_, lean_object* v_i_4256_, lean_object* v_acc_4257_, lean_object* v_lctx_4258_){
_start:
{
lean_object* v___x_4259_; uint8_t v___x_4260_; 
v___x_4259_ = lean_array_get_size(v_suggestions_4255_);
v___x_4260_ = lean_nat_dec_lt(v_i_4256_, v___x_4259_);
if (v___x_4260_ == 0)
{
lean_object* v___x_4261_; 
lean_dec(v_i_4256_);
v___x_4261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4261_, 0, v_acc_4257_);
lean_ctor_set(v___x_4261_, 1, v_lctx_4258_);
return v___x_4261_;
}
else
{
lean_object* v___x_4262_; lean_object* v_name_4263_; lean_object* v___x_4264_; lean_object* v_lctx_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; lean_object* v___x_4268_; 
v___x_4262_ = lean_array_fget_borrowed(v_suggestions_4255_, v_i_4256_);
v_name_4263_ = l_Lean_LocalContext_getUnusedName(v_lctx_4258_, v___x_4262_);
lean_inc(v_name_4263_);
v___x_4264_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_dummyLDecl(v_name_4263_);
v_lctx_4265_ = l_Lean_LocalContext_addDecl(v_lctx_4258_, v___x_4264_);
v___x_4266_ = lean_unsigned_to_nat(1u);
v___x_4267_ = lean_nat_add(v_i_4256_, v___x_4266_);
lean_dec(v_i_4256_);
v___x_4268_ = lean_array_push(v_acc_4257_, v_name_4263_);
v_i_4256_ = v___x_4267_;
v_acc_4257_ = v___x_4268_;
v_lctx_4258_ = v_lctx_4265_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go___boxed(lean_object* v_suggestions_4270_, lean_object* v_i_4271_, lean_object* v_acc_4272_, lean_object* v_lctx_4273_){
_start:
{
lean_object* v_res_4274_; 
v_res_4274_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go(v_suggestions_4270_, v_i_4271_, v_acc_4272_, v_lctx_4273_);
lean_dec_ref(v_suggestions_4270_);
return v_res_4274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnusedNames(lean_object* v_lctx_4275_, lean_object* v_suggestions_4276_){
_start:
{
lean_object* v___x_4277_; lean_object* v___x_4278_; lean_object* v___x_4279_; lean_object* v___x_4280_; 
v___x_4277_ = lean_unsigned_to_nat(0u);
v___x_4278_ = lean_array_get_size(v_suggestions_4276_);
v___x_4279_ = lean_mk_empty_array_with_capacity(v___x_4278_);
v___x_4280_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_getUnusedNames_go(v_suggestions_4276_, v___x_4277_, v___x_4279_, v_lctx_4275_);
return v___x_4280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnusedNames___boxed(lean_object* v_lctx_4281_, lean_object* v_suggestions_4282_){
_start:
{
lean_object* v_res_4283_; 
v_res_4283_ = lp_aesop_Aesop_getUnusedNames(v_lctx_4281_, v_suggestions_4282_);
lean_dec_ref(v_suggestions_4282_);
return v_res_4283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Aesop_Name_ofComponents_spec__0(lean_object* v_x_4284_, lean_object* v_x_4285_){
_start:
{
if (lean_obj_tag(v_x_4285_) == 0)
{
return v_x_4284_;
}
else
{
lean_object* v_head_4286_; 
v_head_4286_ = lean_ctor_get(v_x_4285_, 0);
switch(lean_obj_tag(v_head_4286_))
{
case 0:
{
lean_object* v_tail_4287_; 
v_tail_4287_ = lean_ctor_get(v_x_4285_, 1);
lean_inc(v_tail_4287_);
lean_dec_ref_known(v_x_4285_, 2);
v_x_4285_ = v_tail_4287_;
goto _start;
}
case 1:
{
lean_object* v_tail_4289_; lean_object* v_str_4290_; lean_object* v___x_4291_; 
lean_inc_ref(v_head_4286_);
v_tail_4289_ = lean_ctor_get(v_x_4285_, 1);
lean_inc(v_tail_4289_);
lean_dec_ref_known(v_x_4285_, 2);
v_str_4290_ = lean_ctor_get(v_head_4286_, 1);
lean_inc_ref(v_str_4290_);
lean_dec_ref_known(v_head_4286_, 2);
v___x_4291_ = l_Lean_Name_str___override(v_x_4284_, v_str_4290_);
v_x_4284_ = v___x_4291_;
v_x_4285_ = v_tail_4289_;
goto _start;
}
default: 
{
lean_object* v_tail_4293_; lean_object* v_i_4294_; lean_object* v___x_4295_; 
lean_inc_ref(v_head_4286_);
v_tail_4293_ = lean_ctor_get(v_x_4285_, 1);
lean_inc(v_tail_4293_);
lean_dec_ref_known(v_x_4285_, 2);
v_i_4294_ = lean_ctor_get(v_head_4286_, 1);
lean_inc(v_i_4294_);
lean_dec_ref_known(v_head_4286_, 2);
v___x_4295_ = l_Lean_Name_num___override(v_x_4284_, v_i_4294_);
v_x_4284_ = v___x_4295_;
v_x_4285_ = v_tail_4293_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Name_ofComponents(lean_object* v_cs_4297_){
_start:
{
lean_object* v___x_4298_; lean_object* v___x_4299_; 
v___x_4298_ = lean_box(0);
v___x_4299_ = lp_aesop_List_foldl___at___00Aesop_Name_ofComponents_spec__0(v___x_4298_, v_cs_4297_);
return v___x_4299_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___redArg___lam__0(lean_object* v_x_4309_){
_start:
{
lean_object* v___x_4310_; uint8_t v___x_4311_; lean_object* v___x_4312_; lean_object* v___x_4313_; lean_object* v___x_4314_; lean_object* v___x_4315_; lean_object* v___x_4316_; lean_object* v___x_4317_; 
v___x_4310_ = ((lean_object*)(lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__2));
v___x_4311_ = 1;
v___x_4312_ = l_Lean_KVMap_instValueBool;
v___x_4313_ = lean_box(v___x_4311_);
v___x_4314_ = l_Lean_Options_set___redArg(v___x_4312_, v_x_4309_, v___x_4310_, v___x_4313_);
v___x_4315_ = ((lean_object*)(lp_aesop_Aesop_withPPAnalyze___redArg___lam__0___closed__4));
v___x_4316_ = lean_box(v___x_4311_);
v___x_4317_ = l_Lean_Options_set___redArg(v___x_4312_, v___x_4314_, v___x_4315_, v___x_4316_);
return v___x_4317_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___redArg(lean_object* v_inst_4319_, lean_object* v_x_4320_){
_start:
{
lean_object* v___f_4321_; lean_object* v___x_4322_; 
v___f_4321_ = ((lean_object*)(lp_aesop_Aesop_withPPAnalyze___redArg___closed__0));
v___x_4322_ = lean_apply_3(v_inst_4319_, lean_box(0), v___f_4321_, v_x_4320_);
return v___x_4322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze(lean_object* v_m_4323_, lean_object* v_00_u03b1_4324_, lean_object* v_inst_4325_, lean_object* v_inst_4326_, lean_object* v_x_4327_){
_start:
{
lean_object* v___x_4328_; 
v___x_4328_ = lp_aesop_Aesop_withPPAnalyze___redArg(v_inst_4326_, v_x_4327_);
return v___x_4328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withPPAnalyze___boxed(lean_object* v_m_4329_, lean_object* v_00_u03b1_4330_, lean_object* v_inst_4331_, lean_object* v_inst_4332_, lean_object* v_x_4333_){
_start:
{
lean_object* v_res_4334_; 
v_res_4334_ = lp_aesop_Aesop_withPPAnalyze(v_m_4329_, v_00_u03b1_4330_, v_inst_4331_, v_inst_4332_, v_x_4333_);
lean_dec_ref(v_inst_4331_);
return v_res_4334_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0(lean_object* v_inst_4335_, lean_object* v_a_4336_, lean_object* v___y_4337_){
_start:
{
lean_object* v_findCached_x3f_4338_; lean_object* v___x_4339_; 
v_findCached_x3f_4338_ = lean_ctor_get(v_inst_4335_, 0);
lean_inc(v_findCached_x3f_4338_);
lean_dec_ref(v_inst_4335_);
v___x_4339_ = lean_apply_1(v_findCached_x3f_4338_, v_a_4336_);
return v___x_4339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0___boxed(lean_object* v_inst_4340_, lean_object* v_a_4341_, lean_object* v___y_4342_){
_start:
{
lean_object* v_res_4343_; 
v_res_4343_ = lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0(v_inst_4340_, v_a_4341_, v___y_4342_);
lean_dec(v___y_4342_);
return v_res_4343_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1(lean_object* v_inst_4344_, lean_object* v_a_4345_, lean_object* v_b_4346_, lean_object* v___y_4347_){
_start:
{
lean_object* v_cache_4348_; lean_object* v___x_4349_; 
v_cache_4348_ = lean_ctor_get(v_inst_4344_, 1);
lean_inc(v_cache_4348_);
lean_dec_ref(v_inst_4344_);
v___x_4349_ = lean_apply_2(v_cache_4348_, v_a_4345_, v_b_4346_);
return v___x_4349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1___boxed(lean_object* v_inst_4350_, lean_object* v_a_4351_, lean_object* v_b_4352_, lean_object* v___y_4353_){
_start:
{
lean_object* v_res_4354_; 
v_res_4354_ = lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1(v_inst_4350_, v_a_4351_, v_b_4352_, v___y_4353_);
lean_dec(v___y_4353_);
return v_res_4354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg(lean_object* v_inst_4355_){
_start:
{
lean_object* v___f_4356_; lean_object* v___f_4357_; lean_object* v___x_4358_; 
lean_inc_ref(v_inst_4355_);
v___f_4356_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_4356_, 0, v_inst_4355_);
v___f_4357_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_4357_, 0, v_inst_4355_);
v___x_4358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4358_, 0, v___f_4356_);
lean_ctor_set(v___x_4358_, 1, v___f_4357_);
return v___x_4358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop(lean_object* v_00_u03b1_4359_, lean_object* v_00_u03b2_4360_, lean_object* v_m_4361_, lean_object* v_00_u03c9_4362_, lean_object* v_00_u03c3_4363_, lean_object* v_inst_4364_){
_start:
{
lean_object* v___x_4365_; 
v___x_4365_ = lp_aesop_Aesop_instMonadCacheStateRefT_x27__aesop___redArg(v_inst_4364_);
return v___x_4365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__0(lean_object* v_x_4366_){
_start:
{
lean_object* v_fst_4367_; 
v_fst_4367_ = lean_ctor_get(v_x_4366_, 0);
lean_inc(v_fst_4367_);
return v_fst_4367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__0___boxed(lean_object* v_x_4368_){
_start:
{
lean_object* v_res_4369_; 
v_res_4369_ = lp_aesop_Aesop_runInMetaState___redArg___lam__0(v_x_4368_);
lean_dec_ref(v_x_4368_);
return v_res_4369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__1(lean_object* v_x_4370_, lean_object* v_____r_4371_){
_start:
{
lean_inc(v_x_4370_);
return v_x_4370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__1___boxed(lean_object* v_x_4372_, lean_object* v_____r_4373_){
_start:
{
lean_object* v_res_4374_; 
v_res_4374_ = lp_aesop_Aesop_runInMetaState___redArg___lam__1(v_x_4372_, v_____r_4373_);
lean_dec(v_x_4372_);
return v_res_4374_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__2(lean_object* v___x_4375_, lean_object* v_x_4376_){
_start:
{
lean_inc(v___x_4375_);
return v___x_4375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__2___boxed(lean_object* v___x_4377_, lean_object* v_x_4378_){
_start:
{
lean_object* v_res_4379_; 
v_res_4379_ = lp_aesop_Aesop_runInMetaState___redArg___lam__2(v___x_4377_, v_x_4378_);
lean_dec(v_x_4378_);
lean_dec(v___x_4377_);
return v_res_4379_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg___lam__3(lean_object* v_toFunctor_4380_, lean_object* v_s_4381_, lean_object* v_inst_4382_, lean_object* v_toBind_4383_, lean_object* v___f_4384_, lean_object* v_inst_4385_, lean_object* v___f_4386_, lean_object* v_initialState_4387_){
_start:
{
lean_object* v_map_4388_; lean_object* v___x_4389_; lean_object* v___x_4390_; lean_object* v___x_4391_; lean_object* v___x_4392_; lean_object* v___x_4393_; lean_object* v___f_4394_; lean_object* v_y_4395_; lean_object* v___x_4396_; 
v_map_4388_ = lean_ctor_get(v_toFunctor_4380_, 0);
lean_inc(v_map_4388_);
lean_dec_ref(v_toFunctor_4380_);
v___x_4389_ = lean_alloc_closure((void*)(l_Lean_Meta_SavedState_restore___boxed), 6, 1);
lean_closure_set(v___x_4389_, 0, v_s_4381_);
lean_inc(v_inst_4382_);
v___x_4390_ = lean_apply_2(v_inst_4382_, lean_box(0), v___x_4389_);
v___x_4391_ = lean_apply_4(v_toBind_4383_, lean_box(0), lean_box(0), v___x_4390_, v___f_4384_);
v___x_4392_ = lean_alloc_closure((void*)(l_Lean_Meta_SavedState_restore___boxed), 6, 1);
lean_closure_set(v___x_4392_, 0, v_initialState_4387_);
v___x_4393_ = lean_apply_2(v_inst_4382_, lean_box(0), v___x_4392_);
v___f_4394_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runInMetaState___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_4394_, 0, v___x_4393_);
v_y_4395_ = lean_apply_4(v_inst_4385_, lean_box(0), lean_box(0), v___x_4391_, v___f_4394_);
v___x_4396_ = lean_apply_4(v_map_4388_, lean_box(0), lean_box(0), v___f_4386_, v_y_4395_);
return v___x_4396_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___redArg(lean_object* v_inst_4399_, lean_object* v_inst_4400_, lean_object* v_inst_4401_, lean_object* v_s_4402_, lean_object* v_x_4403_){
_start:
{
lean_object* v_toApplicative_4404_; lean_object* v_toBind_4405_; lean_object* v_toFunctor_4406_; lean_object* v___f_4407_; lean_object* v___f_4408_; lean_object* v_this_4409_; lean_object* v___x_4410_; lean_object* v___f_4411_; lean_object* v___x_4412_; 
v_toApplicative_4404_ = lean_ctor_get(v_inst_4399_, 0);
lean_inc_ref(v_toApplicative_4404_);
v_toBind_4405_ = lean_ctor_get(v_inst_4399_, 1);
lean_inc_n(v_toBind_4405_, 2);
lean_dec_ref(v_inst_4399_);
v_toFunctor_4406_ = lean_ctor_get(v_toApplicative_4404_, 0);
lean_inc_ref(v_toFunctor_4406_);
lean_dec_ref(v_toApplicative_4404_);
v___f_4407_ = ((lean_object*)(lp_aesop_Aesop_runInMetaState___redArg___closed__0));
v___f_4408_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runInMetaState___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_4408_, 0, v_x_4403_);
v_this_4409_ = ((lean_object*)(lp_aesop_Aesop_runInMetaState___redArg___closed__1));
lean_inc(v_inst_4400_);
v___x_4410_ = lean_apply_2(v_inst_4400_, lean_box(0), v_this_4409_);
v___f_4411_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runInMetaState___redArg___lam__3), 8, 7);
lean_closure_set(v___f_4411_, 0, v_toFunctor_4406_);
lean_closure_set(v___f_4411_, 1, v_s_4402_);
lean_closure_set(v___f_4411_, 2, v_inst_4400_);
lean_closure_set(v___f_4411_, 3, v_toBind_4405_);
lean_closure_set(v___f_4411_, 4, v___f_4408_);
lean_closure_set(v___f_4411_, 5, v_inst_4401_);
lean_closure_set(v___f_4411_, 6, v___f_4407_);
v___x_4412_ = lean_apply_4(v_toBind_4405_, lean_box(0), lean_box(0), v___x_4410_, v___f_4411_);
return v___x_4412_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState(lean_object* v_m_4413_, lean_object* v_00_u03b1_4414_, lean_object* v_inst_4415_, lean_object* v_inst_4416_, lean_object* v_inst_4417_, lean_object* v_s_4418_, lean_object* v_x_4419_){
_start:
{
lean_object* v___x_4420_; 
v___x_4420_ = lp_aesop_Aesop_runInMetaState___redArg(v_inst_4415_, v_inst_4416_, v_inst_4417_, v_s_4418_, v_x_4419_);
return v___x_4420_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_lBoolOr(uint8_t v_x_4421_, uint8_t v_x_4422_){
_start:
{
switch(v_x_4421_)
{
case 0:
{
return v_x_4422_;
}
case 1:
{
return v_x_4421_;
}
default: 
{
if (v_x_4422_ == 1)
{
return v_x_4422_;
}
else
{
return v_x_4421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_lBoolOr___boxed(lean_object* v_x_4423_, lean_object* v_x_4424_){
_start:
{
uint8_t v_x_32__boxed_4425_; uint8_t v_x_33__boxed_4426_; uint8_t v_res_4427_; lean_object* v_r_4428_; 
v_x_32__boxed_4425_ = lean_unbox(v_x_4423_);
v_x_33__boxed_4426_ = lean_unbox(v_x_4424_);
v_res_4427_ = lp_aesop_Aesop_lBoolOr(v_x_32__boxed_4425_, v_x_33__boxed_4426_);
v_r_4428_ = lean_box(v_res_4427_);
return v_r_4428_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg(lean_object* v_xs_4444_, lean_object* v_ys_4445_, lean_object* v_cmp_4446_, lean_object* v_range_4447_, lean_object* v_b_4448_, lean_object* v_i_4449_){
_start:
{
lean_object* v_stop_4450_; lean_object* v_step_4451_; uint8_t v___x_4452_; 
v_stop_4450_ = lean_ctor_get(v_range_4447_, 1);
v_step_4451_ = lean_ctor_get(v_range_4447_, 2);
v___x_4452_ = lean_nat_dec_lt(v_i_4449_, v_stop_4450_);
if (v___x_4452_ == 0)
{
lean_dec(v_i_4449_);
lean_dec_ref(v_cmp_4446_);
lean_inc_ref(v_b_4448_);
return v_b_4448_;
}
else
{
lean_object* v___x_4453_; lean_object* v___x_4454_; uint8_t v___x_4455_; 
v___x_4453_ = lean_box(0);
v___x_4454_ = lean_array_get_size(v_xs_4444_);
v___x_4455_ = lean_nat_dec_lt(v_i_4449_, v___x_4454_);
if (v___x_4455_ == 0)
{
lean_object* v___x_4456_; 
lean_dec(v_i_4449_);
lean_dec_ref(v_cmp_4446_);
v___x_4456_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__1));
return v___x_4456_;
}
else
{
lean_object* v___x_4457_; uint8_t v___x_4458_; 
v___x_4457_ = lean_array_get_size(v_ys_4445_);
v___x_4458_ = lean_nat_dec_lt(v_i_4449_, v___x_4457_);
if (v___x_4458_ == 0)
{
lean_object* v___x_4459_; 
lean_dec(v_i_4449_);
lean_dec_ref(v_cmp_4446_);
v___x_4459_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__3));
return v___x_4459_;
}
else
{
lean_object* v___x_4460_; lean_object* v___x_4461_; lean_object* v___x_4462_; uint8_t v___x_4463_; 
v___x_4460_ = lean_array_fget_borrowed(v_xs_4444_, v_i_4449_);
v___x_4461_ = lean_array_fget_borrowed(v_ys_4445_, v_i_4449_);
lean_inc_ref(v_cmp_4446_);
lean_inc(v___x_4461_);
lean_inc(v___x_4460_);
v___x_4462_ = lean_apply_2(v_cmp_4446_, v___x_4460_, v___x_4461_);
v___x_4463_ = lean_unbox(v___x_4462_);
if (v___x_4463_ == 1)
{
lean_object* v___x_4464_; lean_object* v___x_4465_; 
v___x_4464_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__4));
v___x_4465_ = lean_nat_add(v_i_4449_, v_step_4451_);
lean_dec(v_i_4449_);
v_b_4448_ = v___x_4464_;
v_i_4449_ = v___x_4465_;
goto _start;
}
else
{
lean_object* v___x_4467_; lean_object* v___x_4468_; 
lean_dec(v_i_4449_);
lean_dec_ref(v_cmp_4446_);
v___x_4467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4467_, 0, v___x_4462_);
v___x_4468_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4468_, 0, v___x_4467_);
lean_ctor_set(v___x_4468_, 1, v___x_4453_);
return v___x_4468_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___boxed(lean_object* v_xs_4469_, lean_object* v_ys_4470_, lean_object* v_cmp_4471_, lean_object* v_range_4472_, lean_object* v_b_4473_, lean_object* v_i_4474_){
_start:
{
lean_object* v_res_4475_; 
v_res_4475_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg(v_xs_4469_, v_ys_4470_, v_cmp_4471_, v_range_4472_, v_b_4473_, v_i_4474_);
lean_dec_ref(v_b_4473_);
lean_dec_ref(v_range_4472_);
lean_dec_ref(v_ys_4470_);
lean_dec_ref(v_xs_4469_);
return v_res_4475_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArrayLex___redArg(lean_object* v_cmp_4476_, lean_object* v_xs_4477_, lean_object* v_ys_4478_){
_start:
{
lean_object* v___y_4480_; lean_object* v___x_4490_; lean_object* v___x_4491_; uint8_t v___x_4492_; 
v___x_4490_ = lean_array_get_size(v_xs_4477_);
v___x_4491_ = lean_array_get_size(v_ys_4478_);
v___x_4492_ = lean_nat_dec_le(v___x_4490_, v___x_4491_);
if (v___x_4492_ == 0)
{
v___y_4480_ = v___x_4490_;
goto v___jp_4479_;
}
else
{
v___y_4480_ = v___x_4491_;
goto v___jp_4479_;
}
v___jp_4479_:
{
lean_object* v___x_4481_; lean_object* v___x_4482_; lean_object* v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v_fst_4486_; 
v___x_4481_ = lean_unsigned_to_nat(0u);
v___x_4482_ = lean_unsigned_to_nat(1u);
v___x_4483_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4483_, 0, v___x_4481_);
lean_ctor_set(v___x_4483_, 1, v___y_4480_);
lean_ctor_set(v___x_4483_, 2, v___x_4482_);
v___x_4484_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg___closed__4));
v___x_4485_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg(v_xs_4477_, v_ys_4478_, v_cmp_4476_, v___x_4483_, v___x_4484_, v___x_4481_);
lean_dec_ref_known(v___x_4483_, 3);
v_fst_4486_ = lean_ctor_get(v___x_4485_, 0);
lean_inc(v_fst_4486_);
lean_dec_ref(v___x_4485_);
if (lean_obj_tag(v_fst_4486_) == 0)
{
uint8_t v___x_4487_; 
v___x_4487_ = 1;
return v___x_4487_;
}
else
{
lean_object* v_val_4488_; uint8_t v___x_4489_; 
v_val_4488_ = lean_ctor_get(v_fst_4486_, 0);
lean_inc(v_val_4488_);
lean_dec_ref_known(v_fst_4486_, 1);
v___x_4489_ = lean_unbox(v_val_4488_);
lean_dec(v_val_4488_);
return v___x_4489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArrayLex___redArg___boxed(lean_object* v_cmp_4493_, lean_object* v_xs_4494_, lean_object* v_ys_4495_){
_start:
{
uint8_t v_res_4496_; lean_object* v_r_4497_; 
v_res_4496_ = lp_aesop_Aesop_compareArrayLex___redArg(v_cmp_4493_, v_xs_4494_, v_ys_4495_);
lean_dec_ref(v_ys_4495_);
lean_dec_ref(v_xs_4494_);
v_r_4497_ = lean_box(v_res_4496_);
return v_r_4497_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArrayLex(lean_object* v_00_u03b1_4498_, lean_object* v_cmp_4499_, lean_object* v_xs_4500_, lean_object* v_ys_4501_){
_start:
{
uint8_t v___x_4502_; 
v___x_4502_ = lp_aesop_Aesop_compareArrayLex___redArg(v_cmp_4499_, v_xs_4500_, v_ys_4501_);
return v___x_4502_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArrayLex___boxed(lean_object* v_00_u03b1_4503_, lean_object* v_cmp_4504_, lean_object* v_xs_4505_, lean_object* v_ys_4506_){
_start:
{
uint8_t v_res_4507_; lean_object* v_r_4508_; 
v_res_4507_ = lp_aesop_Aesop_compareArrayLex(v_00_u03b1_4503_, v_cmp_4504_, v_xs_4505_, v_ys_4506_);
lean_dec_ref(v_ys_4506_);
lean_dec_ref(v_xs_4505_);
v_r_4508_ = lean_box(v_res_4507_);
return v_r_4508_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0(lean_object* v_00_u03b1_4509_, lean_object* v_xs_4510_, lean_object* v_ys_4511_, lean_object* v_cmp_4512_, lean_object* v_range_4513_, lean_object* v_b_4514_, lean_object* v_i_4515_, lean_object* v_hs_4516_, lean_object* v_hl_4517_){
_start:
{
lean_object* v___x_4518_; 
v___x_4518_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___redArg(v_xs_4510_, v_ys_4511_, v_cmp_4512_, v_range_4513_, v_b_4514_, v_i_4515_);
return v___x_4518_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0___boxed(lean_object* v_00_u03b1_4519_, lean_object* v_xs_4520_, lean_object* v_ys_4521_, lean_object* v_cmp_4522_, lean_object* v_range_4523_, lean_object* v_b_4524_, lean_object* v_i_4525_, lean_object* v_hs_4526_, lean_object* v_hl_4527_){
_start:
{
lean_object* v_res_4528_; 
v_res_4528_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_compareArrayLex_spec__0(v_00_u03b1_4519_, v_xs_4520_, v_ys_4521_, v_cmp_4522_, v_range_4523_, v_b_4524_, v_i_4525_, v_hs_4526_, v_hl_4527_);
lean_dec_ref(v_b_4524_);
lean_dec_ref(v_range_4523_);
lean_dec_ref(v_ys_4521_);
lean_dec_ref(v_xs_4520_);
return v_res_4528_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArraySizeThenLex___redArg(lean_object* v_cmp_4529_, lean_object* v_xs_4530_, lean_object* v_ys_4531_){
_start:
{
lean_object* v___x_4532_; lean_object* v___x_4533_; uint8_t v___x_4534_; 
v___x_4532_ = lean_array_get_size(v_xs_4530_);
v___x_4533_ = lean_array_get_size(v_ys_4531_);
v___x_4534_ = lean_nat_dec_lt(v___x_4532_, v___x_4533_);
if (v___x_4534_ == 0)
{
uint8_t v___x_4535_; 
v___x_4535_ = lean_nat_dec_eq(v___x_4532_, v___x_4533_);
if (v___x_4535_ == 0)
{
uint8_t v___x_4536_; 
lean_dec_ref(v_cmp_4529_);
v___x_4536_ = 2;
return v___x_4536_;
}
else
{
uint8_t v___x_4537_; 
v___x_4537_ = lp_aesop_Aesop_compareArrayLex___redArg(v_cmp_4529_, v_xs_4530_, v_ys_4531_);
return v___x_4537_;
}
}
else
{
uint8_t v___x_4538_; 
lean_dec_ref(v_cmp_4529_);
v___x_4538_ = 0;
return v___x_4538_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArraySizeThenLex___redArg___boxed(lean_object* v_cmp_4539_, lean_object* v_xs_4540_, lean_object* v_ys_4541_){
_start:
{
uint8_t v_res_4542_; lean_object* v_r_4543_; 
v_res_4542_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v_cmp_4539_, v_xs_4540_, v_ys_4541_);
lean_dec_ref(v_ys_4541_);
lean_dec_ref(v_xs_4540_);
v_r_4543_ = lean_box(v_res_4542_);
return v_r_4543_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_compareArraySizeThenLex(lean_object* v_00_u03b1_4544_, lean_object* v_cmp_4545_, lean_object* v_xs_4546_, lean_object* v_ys_4547_){
_start:
{
uint8_t v___x_4548_; 
v___x_4548_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v_cmp_4545_, v_xs_4546_, v_ys_4547_);
return v___x_4548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_compareArraySizeThenLex___boxed(lean_object* v_00_u03b1_4549_, lean_object* v_cmp_4550_, lean_object* v_xs_4551_, lean_object* v_ys_4552_){
_start:
{
uint8_t v_res_4553_; lean_object* v_r_4554_; 
v_res_4553_ = lp_aesop_Aesop_compareArraySizeThenLex(v_00_u03b1_4549_, v_cmp_4550_, v_xs_4551_, v_ys_4552_);
lean_dec_ref(v_ys_4552_);
lean_dec_ref(v_xs_4551_);
v_r_4554_ = lean_box(v_res_4553_);
return v_r_4554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0(lean_object* v_inst_4555_, lean_object* v___y_4556_){
_start:
{
lean_inc(v_inst_4555_);
return v_inst_4555_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0___boxed(lean_object* v_inst_4557_, lean_object* v___y_4558_){
_start:
{
lean_object* v_res_4559_; 
v_res_4559_ = lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0(v_inst_4557_, v___y_4558_);
lean_dec(v___y_4558_);
lean_dec(v_inst_4557_);
return v_res_4559_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg(lean_object* v_inst_4560_){
_start:
{
lean_object* v___f_4561_; 
v___f_4561_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_4561_, 0, v_inst_4560_);
return v___f_4561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclReaderT__aesop(lean_object* v_m_4562_, lean_object* v_00_u03c1_4563_, lean_object* v_inst_4564_){
_start:
{
lean_object* v___f_4565_; 
v___f_4565_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadParentDeclReaderT__aesop___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_4565_, 0, v_inst_4564_);
return v___f_4565_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclStateRefT_x27__aesop___redArg(lean_object* v_inst_4566_){
_start:
{
lean_object* v___x_4567_; 
v___x_4567_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_4567_, 0, lean_box(0));
lean_closure_set(v___x_4567_, 1, lean_box(0));
lean_closure_set(v___x_4567_, 2, lean_box(0));
lean_closure_set(v___x_4567_, 3, lean_box(0));
lean_closure_set(v___x_4567_, 4, v_inst_4566_);
return v___x_4567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadParentDeclStateRefT_x27__aesop(lean_object* v_m_4568_, lean_object* v_00_u03c9_4569_, lean_object* v_00_u03c3_4570_, lean_object* v_inst_4571_){
_start:
{
lean_object* v___x_4572_; 
v___x_4572_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_4572_, 0, lean_box(0));
lean_closure_set(v___x_4572_, 1, lean_box(0));
lean_closure_set(v___x_4572_, 2, lean_box(0));
lean_closure_set(v___x_4572_, 3, lean_box(0));
lean_closure_set(v___x_4572_, 4, v_inst_4571_);
return v___x_4572_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(lean_object* v_k_4573_, uint8_t v_allowLevelAssignments_4574_, lean_object* v___y_4575_, lean_object* v___y_4576_, lean_object* v___y_4577_, lean_object* v___y_4578_){
_start:
{
lean_object* v___x_4580_; 
v___x_4580_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_4574_, v_k_4573_, v___y_4575_, v___y_4576_, v___y_4577_, v___y_4578_);
if (lean_obj_tag(v___x_4580_) == 0)
{
lean_object* v_a_4581_; lean_object* v___x_4583_; uint8_t v_isShared_4584_; uint8_t v_isSharedCheck_4588_; 
v_a_4581_ = lean_ctor_get(v___x_4580_, 0);
v_isSharedCheck_4588_ = !lean_is_exclusive(v___x_4580_);
if (v_isSharedCheck_4588_ == 0)
{
v___x_4583_ = v___x_4580_;
v_isShared_4584_ = v_isSharedCheck_4588_;
goto v_resetjp_4582_;
}
else
{
lean_inc(v_a_4581_);
lean_dec(v___x_4580_);
v___x_4583_ = lean_box(0);
v_isShared_4584_ = v_isSharedCheck_4588_;
goto v_resetjp_4582_;
}
v_resetjp_4582_:
{
lean_object* v___x_4586_; 
if (v_isShared_4584_ == 0)
{
v___x_4586_ = v___x_4583_;
goto v_reusejp_4585_;
}
else
{
lean_object* v_reuseFailAlloc_4587_; 
v_reuseFailAlloc_4587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4587_, 0, v_a_4581_);
v___x_4586_ = v_reuseFailAlloc_4587_;
goto v_reusejp_4585_;
}
v_reusejp_4585_:
{
return v___x_4586_;
}
}
}
else
{
lean_object* v_a_4589_; lean_object* v___x_4591_; uint8_t v_isShared_4592_; uint8_t v_isSharedCheck_4596_; 
v_a_4589_ = lean_ctor_get(v___x_4580_, 0);
v_isSharedCheck_4596_ = !lean_is_exclusive(v___x_4580_);
if (v_isSharedCheck_4596_ == 0)
{
v___x_4591_ = v___x_4580_;
v_isShared_4592_ = v_isSharedCheck_4596_;
goto v_resetjp_4590_;
}
else
{
lean_inc(v_a_4589_);
lean_dec(v___x_4580_);
v___x_4591_ = lean_box(0);
v_isShared_4592_ = v_isSharedCheck_4596_;
goto v_resetjp_4590_;
}
v_resetjp_4590_:
{
lean_object* v___x_4594_; 
if (v_isShared_4592_ == 0)
{
v___x_4594_ = v___x_4591_;
goto v_reusejp_4593_;
}
else
{
lean_object* v_reuseFailAlloc_4595_; 
v_reuseFailAlloc_4595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4595_, 0, v_a_4589_);
v___x_4594_ = v_reuseFailAlloc_4595_;
goto v_reusejp_4593_;
}
v_reusejp_4593_:
{
return v___x_4594_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg___boxed(lean_object* v_k_4597_, lean_object* v_allowLevelAssignments_4598_, lean_object* v___y_4599_, lean_object* v___y_4600_, lean_object* v___y_4601_, lean_object* v___y_4602_, lean_object* v___y_4603_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_4604_; lean_object* v_res_4605_; 
v_allowLevelAssignments_boxed_4604_ = lean_unbox(v_allowLevelAssignments_4598_);
v_res_4605_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v_k_4597_, v_allowLevelAssignments_boxed_4604_, v___y_4599_, v___y_4600_, v___y_4601_, v___y_4602_);
lean_dec(v___y_4602_);
lean_dec_ref(v___y_4601_);
lean_dec(v___y_4600_);
lean_dec_ref(v___y_4599_);
return v_res_4605_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0(lean_object* v_00_u03b1_4606_, lean_object* v_k_4607_, uint8_t v_allowLevelAssignments_4608_, lean_object* v___y_4609_, lean_object* v___y_4610_, lean_object* v___y_4611_, lean_object* v___y_4612_){
_start:
{
lean_object* v___x_4614_; 
v___x_4614_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v_k_4607_, v_allowLevelAssignments_4608_, v___y_4609_, v___y_4610_, v___y_4611_, v___y_4612_);
return v___x_4614_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___boxed(lean_object* v_00_u03b1_4615_, lean_object* v_k_4616_, lean_object* v_allowLevelAssignments_4617_, lean_object* v___y_4618_, lean_object* v___y_4619_, lean_object* v___y_4620_, lean_object* v___y_4621_, lean_object* v___y_4622_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_4623_; lean_object* v_res_4624_; 
v_allowLevelAssignments_boxed_4623_ = lean_unbox(v_allowLevelAssignments_4617_);
v_res_4624_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0(v_00_u03b1_4615_, v_k_4616_, v_allowLevelAssignments_boxed_4623_, v___y_4618_, v___y_4619_, v___y_4620_, v___y_4621_);
lean_dec(v___y_4621_);
lean_dec_ref(v___y_4620_);
lean_dec(v___y_4619_);
lean_dec_ref(v___y_4618_);
return v_res_4624_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_4625_; lean_object* v___x_4626_; lean_object* v___x_4627_; 
v___x_4625_ = lean_unsigned_to_nat(32u);
v___x_4626_ = lean_mk_empty_array_with_capacity(v___x_4625_);
v___x_4627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4627_, 0, v___x_4626_);
return v___x_4627_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1(void){
_start:
{
size_t v___x_4628_; lean_object* v___x_4629_; lean_object* v___x_4630_; lean_object* v___x_4631_; lean_object* v___x_4632_; lean_object* v___x_4633_; 
v___x_4628_ = ((size_t)5ULL);
v___x_4629_ = lean_unsigned_to_nat(0u);
v___x_4630_ = lean_unsigned_to_nat(32u);
v___x_4631_ = lean_mk_empty_array_with_capacity(v___x_4630_);
v___x_4632_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__0);
v___x_4633_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_4633_, 0, v___x_4632_);
lean_ctor_set(v___x_4633_, 1, v___x_4631_);
lean_ctor_set(v___x_4633_, 2, v___x_4629_);
lean_ctor_set(v___x_4633_, 3, v___x_4629_);
lean_ctor_set_usize(v___x_4633_, 4, v___x_4628_);
return v___x_4633_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg(lean_object* v___y_4634_){
_start:
{
lean_object* v___x_4636_; lean_object* v_traceState_4637_; lean_object* v_traces_4638_; lean_object* v___x_4639_; lean_object* v_traceState_4640_; lean_object* v_env_4641_; lean_object* v_nextMacroScope_4642_; lean_object* v_ngen_4643_; lean_object* v_auxDeclNGen_4644_; lean_object* v_cache_4645_; lean_object* v_messages_4646_; lean_object* v_infoState_4647_; lean_object* v_snapshotTasks_4648_; lean_object* v___x_4650_; uint8_t v_isShared_4651_; uint8_t v_isSharedCheck_4667_; 
v___x_4636_ = lean_st_ref_get(v___y_4634_);
v_traceState_4637_ = lean_ctor_get(v___x_4636_, 4);
lean_inc_ref(v_traceState_4637_);
lean_dec(v___x_4636_);
v_traces_4638_ = lean_ctor_get(v_traceState_4637_, 0);
lean_inc_ref(v_traces_4638_);
lean_dec_ref(v_traceState_4637_);
v___x_4639_ = lean_st_ref_take(v___y_4634_);
v_traceState_4640_ = lean_ctor_get(v___x_4639_, 4);
v_env_4641_ = lean_ctor_get(v___x_4639_, 0);
v_nextMacroScope_4642_ = lean_ctor_get(v___x_4639_, 1);
v_ngen_4643_ = lean_ctor_get(v___x_4639_, 2);
v_auxDeclNGen_4644_ = lean_ctor_get(v___x_4639_, 3);
v_cache_4645_ = lean_ctor_get(v___x_4639_, 5);
v_messages_4646_ = lean_ctor_get(v___x_4639_, 6);
v_infoState_4647_ = lean_ctor_get(v___x_4639_, 7);
v_snapshotTasks_4648_ = lean_ctor_get(v___x_4639_, 8);
v_isSharedCheck_4667_ = !lean_is_exclusive(v___x_4639_);
if (v_isSharedCheck_4667_ == 0)
{
v___x_4650_ = v___x_4639_;
v_isShared_4651_ = v_isSharedCheck_4667_;
goto v_resetjp_4649_;
}
else
{
lean_inc(v_snapshotTasks_4648_);
lean_inc(v_infoState_4647_);
lean_inc(v_messages_4646_);
lean_inc(v_cache_4645_);
lean_inc(v_traceState_4640_);
lean_inc(v_auxDeclNGen_4644_);
lean_inc(v_ngen_4643_);
lean_inc(v_nextMacroScope_4642_);
lean_inc(v_env_4641_);
lean_dec(v___x_4639_);
v___x_4650_ = lean_box(0);
v_isShared_4651_ = v_isSharedCheck_4667_;
goto v_resetjp_4649_;
}
v_resetjp_4649_:
{
uint64_t v_tid_4652_; lean_object* v___x_4654_; uint8_t v_isShared_4655_; uint8_t v_isSharedCheck_4665_; 
v_tid_4652_ = lean_ctor_get_uint64(v_traceState_4640_, sizeof(void*)*1);
v_isSharedCheck_4665_ = !lean_is_exclusive(v_traceState_4640_);
if (v_isSharedCheck_4665_ == 0)
{
lean_object* v_unused_4666_; 
v_unused_4666_ = lean_ctor_get(v_traceState_4640_, 0);
lean_dec(v_unused_4666_);
v___x_4654_ = v_traceState_4640_;
v_isShared_4655_ = v_isSharedCheck_4665_;
goto v_resetjp_4653_;
}
else
{
lean_dec(v_traceState_4640_);
v___x_4654_ = lean_box(0);
v_isShared_4655_ = v_isSharedCheck_4665_;
goto v_resetjp_4653_;
}
v_resetjp_4653_:
{
lean_object* v___x_4656_; lean_object* v___x_4658_; 
v___x_4656_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___closed__1);
if (v_isShared_4655_ == 0)
{
lean_ctor_set(v___x_4654_, 0, v___x_4656_);
v___x_4658_ = v___x_4654_;
goto v_reusejp_4657_;
}
else
{
lean_object* v_reuseFailAlloc_4664_; 
v_reuseFailAlloc_4664_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_4664_, 0, v___x_4656_);
lean_ctor_set_uint64(v_reuseFailAlloc_4664_, sizeof(void*)*1, v_tid_4652_);
v___x_4658_ = v_reuseFailAlloc_4664_;
goto v_reusejp_4657_;
}
v_reusejp_4657_:
{
lean_object* v___x_4660_; 
if (v_isShared_4651_ == 0)
{
lean_ctor_set(v___x_4650_, 4, v___x_4658_);
v___x_4660_ = v___x_4650_;
goto v_reusejp_4659_;
}
else
{
lean_object* v_reuseFailAlloc_4663_; 
v_reuseFailAlloc_4663_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4663_, 0, v_env_4641_);
lean_ctor_set(v_reuseFailAlloc_4663_, 1, v_nextMacroScope_4642_);
lean_ctor_set(v_reuseFailAlloc_4663_, 2, v_ngen_4643_);
lean_ctor_set(v_reuseFailAlloc_4663_, 3, v_auxDeclNGen_4644_);
lean_ctor_set(v_reuseFailAlloc_4663_, 4, v___x_4658_);
lean_ctor_set(v_reuseFailAlloc_4663_, 5, v_cache_4645_);
lean_ctor_set(v_reuseFailAlloc_4663_, 6, v_messages_4646_);
lean_ctor_set(v_reuseFailAlloc_4663_, 7, v_infoState_4647_);
lean_ctor_set(v_reuseFailAlloc_4663_, 8, v_snapshotTasks_4648_);
v___x_4660_ = v_reuseFailAlloc_4663_;
goto v_reusejp_4659_;
}
v_reusejp_4659_:
{
lean_object* v___x_4661_; lean_object* v___x_4662_; 
v___x_4661_ = lean_st_ref_set(v___y_4634_, v___x_4660_);
v___x_4662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4662_, 0, v_traces_4638_);
return v___x_4662_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg___boxed(lean_object* v___y_4668_, lean_object* v___y_4669_){
_start:
{
lean_object* v_res_4670_; 
v_res_4670_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg(v___y_4668_);
lean_dec(v___y_4668_);
return v_res_4670_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1(lean_object* v___y_4671_, lean_object* v___y_4672_, lean_object* v___y_4673_, lean_object* v___y_4674_){
_start:
{
lean_object* v___x_4676_; 
v___x_4676_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg(v___y_4674_);
return v___x_4676_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___boxed(lean_object* v___y_4677_, lean_object* v___y_4678_, lean_object* v___y_4679_, lean_object* v___y_4680_, lean_object* v___y_4681_){
_start:
{
lean_object* v_res_4682_; 
v_res_4682_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1(v___y_4677_, v___y_4678_, v___y_4679_, v___y_4680_);
lean_dec(v___y_4680_);
lean_dec_ref(v___y_4679_);
lean_dec(v___y_4678_);
lean_dec_ref(v___y_4677_);
return v_res_4682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2(lean_object* v_msgData_4683_, lean_object* v___y_4684_, lean_object* v___y_4685_, lean_object* v___y_4686_, lean_object* v___y_4687_){
_start:
{
lean_object* v___x_4689_; lean_object* v_env_4690_; lean_object* v___x_4691_; lean_object* v_mctx_4692_; lean_object* v_lctx_4693_; lean_object* v_options_4694_; lean_object* v___x_4695_; lean_object* v___x_4696_; lean_object* v___x_4697_; 
v___x_4689_ = lean_st_ref_get(v___y_4687_);
v_env_4690_ = lean_ctor_get(v___x_4689_, 0);
lean_inc_ref(v_env_4690_);
lean_dec(v___x_4689_);
v___x_4691_ = lean_st_ref_get(v___y_4685_);
v_mctx_4692_ = lean_ctor_get(v___x_4691_, 0);
lean_inc_ref(v_mctx_4692_);
lean_dec(v___x_4691_);
v_lctx_4693_ = lean_ctor_get(v___y_4684_, 2);
v_options_4694_ = lean_ctor_get(v___y_4686_, 2);
lean_inc_ref(v_options_4694_);
lean_inc_ref(v_lctx_4693_);
v___x_4695_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4695_, 0, v_env_4690_);
lean_ctor_set(v___x_4695_, 1, v_mctx_4692_);
lean_ctor_set(v___x_4695_, 2, v_lctx_4693_);
lean_ctor_set(v___x_4695_, 3, v_options_4694_);
v___x_4696_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_4696_, 0, v___x_4695_);
lean_ctor_set(v___x_4696_, 1, v_msgData_4683_);
v___x_4697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4697_, 0, v___x_4696_);
return v___x_4697_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2___boxed(lean_object* v_msgData_4698_, lean_object* v___y_4699_, lean_object* v___y_4700_, lean_object* v___y_4701_, lean_object* v___y_4702_, lean_object* v___y_4703_){
_start:
{
lean_object* v_res_4704_; 
v_res_4704_ = lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2(v_msgData_4698_, v___y_4699_, v___y_4700_, v___y_4701_, v___y_4702_);
lean_dec(v___y_4702_);
lean_dec_ref(v___y_4701_);
lean_dec(v___y_4700_);
lean_dec_ref(v___y_4699_);
return v_res_4704_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(lean_object* v_opts_4705_, lean_object* v_opt_4706_){
_start:
{
lean_object* v_name_4707_; lean_object* v_defValue_4708_; lean_object* v_map_4709_; lean_object* v___x_4710_; 
v_name_4707_ = lean_ctor_get(v_opt_4706_, 0);
v_defValue_4708_ = lean_ctor_get(v_opt_4706_, 1);
v_map_4709_ = lean_ctor_get(v_opts_4705_, 0);
v___x_4710_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_4709_, v_name_4707_);
if (lean_obj_tag(v___x_4710_) == 0)
{
uint8_t v___x_4711_; 
v___x_4711_ = lean_unbox(v_defValue_4708_);
return v___x_4711_;
}
else
{
lean_object* v_val_4712_; 
v_val_4712_ = lean_ctor_get(v___x_4710_, 0);
lean_inc(v_val_4712_);
lean_dec_ref_known(v___x_4710_, 1);
if (lean_obj_tag(v_val_4712_) == 1)
{
uint8_t v_v_4713_; 
v_v_4713_ = lean_ctor_get_uint8(v_val_4712_, 0);
lean_dec_ref_known(v_val_4712_, 0);
return v_v_4713_;
}
else
{
uint8_t v___x_4714_; 
lean_dec(v_val_4712_);
v___x_4714_ = lean_unbox(v_defValue_4708_);
return v___x_4714_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3___boxed(lean_object* v_opts_4715_, lean_object* v_opt_4716_){
_start:
{
uint8_t v_res_4717_; lean_object* v_r_4718_; 
v_res_4717_ = lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(v_opts_4715_, v_opt_4716_);
lean_dec_ref(v_opt_4716_);
lean_dec_ref(v_opts_4715_);
v_r_4718_ = lean_box(v_res_4717_);
return v_r_4718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___lam__0(uint8_t v___x_4719_, lean_object* v_s_4720_, lean_object* v_t_4721_, lean_object* v___y_4722_, lean_object* v___y_4723_, lean_object* v___y_4724_, lean_object* v___y_4725_){
_start:
{
lean_object* v_keyedConfig_4727_; uint8_t v_trackZetaDelta_4728_; lean_object* v_zetaDeltaSet_4729_; lean_object* v_lctx_4730_; lean_object* v_localInstances_4731_; lean_object* v_defEqCtx_x3f_4732_; lean_object* v_synthPendingDepth_4733_; lean_object* v_customCanUnfoldPredicate_x3f_4734_; uint8_t v_univApprox_4735_; uint8_t v_inTypeClassResolution_4736_; uint8_t v_cacheInferType_4737_; lean_object* v___x_4739_; uint8_t v_isShared_4740_; uint8_t v_isSharedCheck_4746_; 
v_keyedConfig_4727_ = lean_ctor_get(v___y_4722_, 0);
v_trackZetaDelta_4728_ = lean_ctor_get_uint8(v___y_4722_, sizeof(void*)*7);
v_zetaDeltaSet_4729_ = lean_ctor_get(v___y_4722_, 1);
v_lctx_4730_ = lean_ctor_get(v___y_4722_, 2);
v_localInstances_4731_ = lean_ctor_get(v___y_4722_, 3);
v_defEqCtx_x3f_4732_ = lean_ctor_get(v___y_4722_, 4);
v_synthPendingDepth_4733_ = lean_ctor_get(v___y_4722_, 5);
v_customCanUnfoldPredicate_x3f_4734_ = lean_ctor_get(v___y_4722_, 6);
v_univApprox_4735_ = lean_ctor_get_uint8(v___y_4722_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_4736_ = lean_ctor_get_uint8(v___y_4722_, sizeof(void*)*7 + 2);
v_cacheInferType_4737_ = lean_ctor_get_uint8(v___y_4722_, sizeof(void*)*7 + 3);
v_isSharedCheck_4746_ = !lean_is_exclusive(v___y_4722_);
if (v_isSharedCheck_4746_ == 0)
{
v___x_4739_ = v___y_4722_;
v_isShared_4740_ = v_isSharedCheck_4746_;
goto v_resetjp_4738_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_4734_);
lean_inc(v_synthPendingDepth_4733_);
lean_inc(v_defEqCtx_x3f_4732_);
lean_inc(v_localInstances_4731_);
lean_inc(v_lctx_4730_);
lean_inc(v_zetaDeltaSet_4729_);
lean_inc(v_keyedConfig_4727_);
lean_dec(v___y_4722_);
v___x_4739_ = lean_box(0);
v_isShared_4740_ = v_isSharedCheck_4746_;
goto v_resetjp_4738_;
}
v_resetjp_4738_:
{
lean_object* v___x_4741_; lean_object* v___x_4743_; 
v___x_4741_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_4719_, v_keyedConfig_4727_);
if (v_isShared_4740_ == 0)
{
lean_ctor_set(v___x_4739_, 0, v___x_4741_);
v___x_4743_ = v___x_4739_;
goto v_reusejp_4742_;
}
else
{
lean_object* v_reuseFailAlloc_4745_; 
v_reuseFailAlloc_4745_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_4745_, 0, v___x_4741_);
lean_ctor_set(v_reuseFailAlloc_4745_, 1, v_zetaDeltaSet_4729_);
lean_ctor_set(v_reuseFailAlloc_4745_, 2, v_lctx_4730_);
lean_ctor_set(v_reuseFailAlloc_4745_, 3, v_localInstances_4731_);
lean_ctor_set(v_reuseFailAlloc_4745_, 4, v_defEqCtx_x3f_4732_);
lean_ctor_set(v_reuseFailAlloc_4745_, 5, v_synthPendingDepth_4733_);
lean_ctor_set(v_reuseFailAlloc_4745_, 6, v_customCanUnfoldPredicate_x3f_4734_);
lean_ctor_set_uint8(v_reuseFailAlloc_4745_, sizeof(void*)*7, v_trackZetaDelta_4728_);
lean_ctor_set_uint8(v_reuseFailAlloc_4745_, sizeof(void*)*7 + 1, v_univApprox_4735_);
lean_ctor_set_uint8(v_reuseFailAlloc_4745_, sizeof(void*)*7 + 2, v_inTypeClassResolution_4736_);
lean_ctor_set_uint8(v_reuseFailAlloc_4745_, sizeof(void*)*7 + 3, v_cacheInferType_4737_);
v___x_4743_ = v_reuseFailAlloc_4745_;
goto v_reusejp_4742_;
}
v_reusejp_4742_:
{
lean_object* v___x_4744_; 
v___x_4744_ = l_Lean_Meta_isExprDefEq(v_s_4720_, v_t_4721_, v___x_4743_, v___y_4723_, v___y_4724_, v___y_4725_);
lean_dec_ref(v___x_4743_);
return v___x_4744_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___lam__0___boxed(lean_object* v___x_4747_, lean_object* v_s_4748_, lean_object* v_t_4749_, lean_object* v___y_4750_, lean_object* v___y_4751_, lean_object* v___y_4752_, lean_object* v___y_4753_, lean_object* v___y_4754_){
_start:
{
uint8_t v___x_9447__boxed_4755_; lean_object* v_res_4756_; 
v___x_9447__boxed_4755_ = lean_unbox(v___x_4747_);
v_res_4756_ = lp_aesop_Aesop_isDefEqReducibleRigid___lam__0(v___x_9447__boxed_4755_, v_s_4748_, v_t_4749_, v___y_4750_, v___y_4751_, v___y_4752_, v___y_4753_);
lean_dec(v___y_4753_);
lean_dec_ref(v___y_4752_);
lean_dec(v___y_4751_);
return v_res_4756_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7(lean_object* v_opts_4757_, lean_object* v_opt_4758_){
_start:
{
lean_object* v_name_4759_; lean_object* v_defValue_4760_; lean_object* v_map_4761_; lean_object* v___x_4762_; 
v_name_4759_ = lean_ctor_get(v_opt_4758_, 0);
v_defValue_4760_ = lean_ctor_get(v_opt_4758_, 1);
v_map_4761_ = lean_ctor_get(v_opts_4757_, 0);
v___x_4762_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_4761_, v_name_4759_);
if (lean_obj_tag(v___x_4762_) == 0)
{
lean_inc(v_defValue_4760_);
return v_defValue_4760_;
}
else
{
lean_object* v_val_4763_; 
v_val_4763_ = lean_ctor_get(v___x_4762_, 0);
lean_inc(v_val_4763_);
lean_dec_ref_known(v___x_4762_, 1);
if (lean_obj_tag(v_val_4763_) == 3)
{
lean_object* v_v_4764_; 
v_v_4764_ = lean_ctor_get(v_val_4763_, 0);
lean_inc(v_v_4764_);
lean_dec_ref_known(v_val_4763_, 1);
return v_v_4764_;
}
else
{
lean_dec(v_val_4763_);
lean_inc(v_defValue_4760_);
return v_defValue_4760_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7___boxed(lean_object* v_opts_4765_, lean_object* v_opt_4766_){
_start:
{
lean_object* v_res_4767_; 
v_res_4767_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7(v_opts_4765_, v_opt_4766_);
lean_dec_ref(v_opt_4766_);
lean_dec_ref(v_opts_4765_);
return v_res_4767_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5(size_t v_sz_4768_, size_t v_i_4769_, lean_object* v_bs_4770_){
_start:
{
uint8_t v___x_4771_; 
v___x_4771_ = lean_usize_dec_lt(v_i_4769_, v_sz_4768_);
if (v___x_4771_ == 0)
{
return v_bs_4770_;
}
else
{
lean_object* v_v_4772_; lean_object* v_msg_4773_; lean_object* v___x_4774_; lean_object* v_bs_x27_4775_; size_t v___x_4776_; size_t v___x_4777_; lean_object* v___x_4778_; 
v_v_4772_ = lean_array_uget_borrowed(v_bs_4770_, v_i_4769_);
v_msg_4773_ = lean_ctor_get(v_v_4772_, 1);
lean_inc_ref(v_msg_4773_);
v___x_4774_ = lean_unsigned_to_nat(0u);
v_bs_x27_4775_ = lean_array_uset(v_bs_4770_, v_i_4769_, v___x_4774_);
v___x_4776_ = ((size_t)1ULL);
v___x_4777_ = lean_usize_add(v_i_4769_, v___x_4776_);
v___x_4778_ = lean_array_uset(v_bs_x27_4775_, v_i_4769_, v_msg_4773_);
v_i_4769_ = v___x_4777_;
v_bs_4770_ = v___x_4778_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5___boxed(lean_object* v_sz_4780_, lean_object* v_i_4781_, lean_object* v_bs_4782_){
_start:
{
size_t v_sz_boxed_4783_; size_t v_i_boxed_4784_; lean_object* v_res_4785_; 
v_sz_boxed_4783_ = lean_unbox_usize(v_sz_4780_);
lean_dec(v_sz_4780_);
v_i_boxed_4784_ = lean_unbox_usize(v_i_4781_);
lean_dec(v_i_4781_);
v_res_4785_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5(v_sz_boxed_4783_, v_i_boxed_4784_, v_bs_4782_);
return v_res_4785_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4(lean_object* v_oldTraces_4786_, lean_object* v_data_4787_, lean_object* v_ref_4788_, lean_object* v_msg_4789_, lean_object* v___y_4790_, lean_object* v___y_4791_, lean_object* v___y_4792_, lean_object* v___y_4793_){
_start:
{
lean_object* v_fileName_4795_; lean_object* v_fileMap_4796_; lean_object* v_options_4797_; lean_object* v_currRecDepth_4798_; lean_object* v_maxRecDepth_4799_; lean_object* v_ref_4800_; lean_object* v_currNamespace_4801_; lean_object* v_openDecls_4802_; lean_object* v_initHeartbeats_4803_; lean_object* v_maxHeartbeats_4804_; lean_object* v_quotContext_4805_; lean_object* v_currMacroScope_4806_; uint8_t v_diag_4807_; lean_object* v_cancelTk_x3f_4808_; uint8_t v_suppressElabErrors_4809_; lean_object* v_inheritedTraceOptions_4810_; lean_object* v___x_4811_; lean_object* v_traceState_4812_; lean_object* v_traces_4813_; lean_object* v_ref_4814_; lean_object* v___x_4815_; lean_object* v___x_4816_; size_t v_sz_4817_; size_t v___x_4818_; lean_object* v___x_4819_; lean_object* v_msg_4820_; lean_object* v___x_4821_; lean_object* v_a_4822_; lean_object* v___x_4824_; uint8_t v_isShared_4825_; uint8_t v_isSharedCheck_4859_; 
v_fileName_4795_ = lean_ctor_get(v___y_4792_, 0);
v_fileMap_4796_ = lean_ctor_get(v___y_4792_, 1);
v_options_4797_ = lean_ctor_get(v___y_4792_, 2);
v_currRecDepth_4798_ = lean_ctor_get(v___y_4792_, 3);
v_maxRecDepth_4799_ = lean_ctor_get(v___y_4792_, 4);
v_ref_4800_ = lean_ctor_get(v___y_4792_, 5);
v_currNamespace_4801_ = lean_ctor_get(v___y_4792_, 6);
v_openDecls_4802_ = lean_ctor_get(v___y_4792_, 7);
v_initHeartbeats_4803_ = lean_ctor_get(v___y_4792_, 8);
v_maxHeartbeats_4804_ = lean_ctor_get(v___y_4792_, 9);
v_quotContext_4805_ = lean_ctor_get(v___y_4792_, 10);
v_currMacroScope_4806_ = lean_ctor_get(v___y_4792_, 11);
v_diag_4807_ = lean_ctor_get_uint8(v___y_4792_, sizeof(void*)*14);
v_cancelTk_x3f_4808_ = lean_ctor_get(v___y_4792_, 12);
v_suppressElabErrors_4809_ = lean_ctor_get_uint8(v___y_4792_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4810_ = lean_ctor_get(v___y_4792_, 13);
v___x_4811_ = lean_st_ref_get(v___y_4793_);
v_traceState_4812_ = lean_ctor_get(v___x_4811_, 4);
lean_inc_ref(v_traceState_4812_);
lean_dec(v___x_4811_);
v_traces_4813_ = lean_ctor_get(v_traceState_4812_, 0);
lean_inc_ref(v_traces_4813_);
lean_dec_ref(v_traceState_4812_);
v_ref_4814_ = l_Lean_replaceRef(v_ref_4788_, v_ref_4800_);
lean_inc_ref(v_inheritedTraceOptions_4810_);
lean_inc(v_cancelTk_x3f_4808_);
lean_inc(v_currMacroScope_4806_);
lean_inc(v_quotContext_4805_);
lean_inc(v_maxHeartbeats_4804_);
lean_inc(v_initHeartbeats_4803_);
lean_inc(v_openDecls_4802_);
lean_inc(v_currNamespace_4801_);
lean_inc(v_maxRecDepth_4799_);
lean_inc(v_currRecDepth_4798_);
lean_inc_ref(v_options_4797_);
lean_inc_ref(v_fileMap_4796_);
lean_inc_ref(v_fileName_4795_);
v___x_4815_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4815_, 0, v_fileName_4795_);
lean_ctor_set(v___x_4815_, 1, v_fileMap_4796_);
lean_ctor_set(v___x_4815_, 2, v_options_4797_);
lean_ctor_set(v___x_4815_, 3, v_currRecDepth_4798_);
lean_ctor_set(v___x_4815_, 4, v_maxRecDepth_4799_);
lean_ctor_set(v___x_4815_, 5, v_ref_4814_);
lean_ctor_set(v___x_4815_, 6, v_currNamespace_4801_);
lean_ctor_set(v___x_4815_, 7, v_openDecls_4802_);
lean_ctor_set(v___x_4815_, 8, v_initHeartbeats_4803_);
lean_ctor_set(v___x_4815_, 9, v_maxHeartbeats_4804_);
lean_ctor_set(v___x_4815_, 10, v_quotContext_4805_);
lean_ctor_set(v___x_4815_, 11, v_currMacroScope_4806_);
lean_ctor_set(v___x_4815_, 12, v_cancelTk_x3f_4808_);
lean_ctor_set(v___x_4815_, 13, v_inheritedTraceOptions_4810_);
lean_ctor_set_uint8(v___x_4815_, sizeof(void*)*14, v_diag_4807_);
lean_ctor_set_uint8(v___x_4815_, sizeof(void*)*14 + 1, v_suppressElabErrors_4809_);
v___x_4816_ = l_Lean_PersistentArray_toArray___redArg(v_traces_4813_);
lean_dec_ref(v_traces_4813_);
v_sz_4817_ = lean_array_size(v___x_4816_);
v___x_4818_ = ((size_t)0ULL);
v___x_4819_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4_spec__5(v_sz_4817_, v___x_4818_, v___x_4816_);
v_msg_4820_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_4820_, 0, v_data_4787_);
lean_ctor_set(v_msg_4820_, 1, v_msg_4789_);
lean_ctor_set(v_msg_4820_, 2, v___x_4819_);
v___x_4821_ = lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2(v_msg_4820_, v___y_4790_, v___y_4791_, v___x_4815_, v___y_4793_);
lean_dec_ref_known(v___x_4815_, 14);
v_a_4822_ = lean_ctor_get(v___x_4821_, 0);
v_isSharedCheck_4859_ = !lean_is_exclusive(v___x_4821_);
if (v_isSharedCheck_4859_ == 0)
{
v___x_4824_ = v___x_4821_;
v_isShared_4825_ = v_isSharedCheck_4859_;
goto v_resetjp_4823_;
}
else
{
lean_inc(v_a_4822_);
lean_dec(v___x_4821_);
v___x_4824_ = lean_box(0);
v_isShared_4825_ = v_isSharedCheck_4859_;
goto v_resetjp_4823_;
}
v_resetjp_4823_:
{
lean_object* v___x_4826_; lean_object* v_traceState_4827_; lean_object* v_env_4828_; lean_object* v_nextMacroScope_4829_; lean_object* v_ngen_4830_; lean_object* v_auxDeclNGen_4831_; lean_object* v_cache_4832_; lean_object* v_messages_4833_; lean_object* v_infoState_4834_; lean_object* v_snapshotTasks_4835_; lean_object* v___x_4837_; uint8_t v_isShared_4838_; uint8_t v_isSharedCheck_4858_; 
v___x_4826_ = lean_st_ref_take(v___y_4793_);
v_traceState_4827_ = lean_ctor_get(v___x_4826_, 4);
v_env_4828_ = lean_ctor_get(v___x_4826_, 0);
v_nextMacroScope_4829_ = lean_ctor_get(v___x_4826_, 1);
v_ngen_4830_ = lean_ctor_get(v___x_4826_, 2);
v_auxDeclNGen_4831_ = lean_ctor_get(v___x_4826_, 3);
v_cache_4832_ = lean_ctor_get(v___x_4826_, 5);
v_messages_4833_ = lean_ctor_get(v___x_4826_, 6);
v_infoState_4834_ = lean_ctor_get(v___x_4826_, 7);
v_snapshotTasks_4835_ = lean_ctor_get(v___x_4826_, 8);
v_isSharedCheck_4858_ = !lean_is_exclusive(v___x_4826_);
if (v_isSharedCheck_4858_ == 0)
{
v___x_4837_ = v___x_4826_;
v_isShared_4838_ = v_isSharedCheck_4858_;
goto v_resetjp_4836_;
}
else
{
lean_inc(v_snapshotTasks_4835_);
lean_inc(v_infoState_4834_);
lean_inc(v_messages_4833_);
lean_inc(v_cache_4832_);
lean_inc(v_traceState_4827_);
lean_inc(v_auxDeclNGen_4831_);
lean_inc(v_ngen_4830_);
lean_inc(v_nextMacroScope_4829_);
lean_inc(v_env_4828_);
lean_dec(v___x_4826_);
v___x_4837_ = lean_box(0);
v_isShared_4838_ = v_isSharedCheck_4858_;
goto v_resetjp_4836_;
}
v_resetjp_4836_:
{
uint64_t v_tid_4839_; lean_object* v___x_4841_; uint8_t v_isShared_4842_; uint8_t v_isSharedCheck_4856_; 
v_tid_4839_ = lean_ctor_get_uint64(v_traceState_4827_, sizeof(void*)*1);
v_isSharedCheck_4856_ = !lean_is_exclusive(v_traceState_4827_);
if (v_isSharedCheck_4856_ == 0)
{
lean_object* v_unused_4857_; 
v_unused_4857_ = lean_ctor_get(v_traceState_4827_, 0);
lean_dec(v_unused_4857_);
v___x_4841_ = v_traceState_4827_;
v_isShared_4842_ = v_isSharedCheck_4856_;
goto v_resetjp_4840_;
}
else
{
lean_dec(v_traceState_4827_);
v___x_4841_ = lean_box(0);
v_isShared_4842_ = v_isSharedCheck_4856_;
goto v_resetjp_4840_;
}
v_resetjp_4840_:
{
lean_object* v___x_4843_; lean_object* v___x_4844_; lean_object* v___x_4846_; 
v___x_4843_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4843_, 0, v_ref_4788_);
lean_ctor_set(v___x_4843_, 1, v_a_4822_);
v___x_4844_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_4786_, v___x_4843_);
if (v_isShared_4842_ == 0)
{
lean_ctor_set(v___x_4841_, 0, v___x_4844_);
v___x_4846_ = v___x_4841_;
goto v_reusejp_4845_;
}
else
{
lean_object* v_reuseFailAlloc_4855_; 
v_reuseFailAlloc_4855_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_4855_, 0, v___x_4844_);
lean_ctor_set_uint64(v_reuseFailAlloc_4855_, sizeof(void*)*1, v_tid_4839_);
v___x_4846_ = v_reuseFailAlloc_4855_;
goto v_reusejp_4845_;
}
v_reusejp_4845_:
{
lean_object* v___x_4848_; 
if (v_isShared_4838_ == 0)
{
lean_ctor_set(v___x_4837_, 4, v___x_4846_);
v___x_4848_ = v___x_4837_;
goto v_reusejp_4847_;
}
else
{
lean_object* v_reuseFailAlloc_4854_; 
v_reuseFailAlloc_4854_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4854_, 0, v_env_4828_);
lean_ctor_set(v_reuseFailAlloc_4854_, 1, v_nextMacroScope_4829_);
lean_ctor_set(v_reuseFailAlloc_4854_, 2, v_ngen_4830_);
lean_ctor_set(v_reuseFailAlloc_4854_, 3, v_auxDeclNGen_4831_);
lean_ctor_set(v_reuseFailAlloc_4854_, 4, v___x_4846_);
lean_ctor_set(v_reuseFailAlloc_4854_, 5, v_cache_4832_);
lean_ctor_set(v_reuseFailAlloc_4854_, 6, v_messages_4833_);
lean_ctor_set(v_reuseFailAlloc_4854_, 7, v_infoState_4834_);
lean_ctor_set(v_reuseFailAlloc_4854_, 8, v_snapshotTasks_4835_);
v___x_4848_ = v_reuseFailAlloc_4854_;
goto v_reusejp_4847_;
}
v_reusejp_4847_:
{
lean_object* v___x_4849_; lean_object* v___x_4850_; lean_object* v___x_4852_; 
v___x_4849_ = lean_st_ref_set(v___y_4793_, v___x_4848_);
v___x_4850_ = lean_box(0);
if (v_isShared_4825_ == 0)
{
lean_ctor_set(v___x_4824_, 0, v___x_4850_);
v___x_4852_ = v___x_4824_;
goto v_reusejp_4851_;
}
else
{
lean_object* v_reuseFailAlloc_4853_; 
v_reuseFailAlloc_4853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4853_, 0, v___x_4850_);
v___x_4852_ = v_reuseFailAlloc_4853_;
goto v_reusejp_4851_;
}
v_reusejp_4851_:
{
return v___x_4852_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4___boxed(lean_object* v_oldTraces_4860_, lean_object* v_data_4861_, lean_object* v_ref_4862_, lean_object* v_msg_4863_, lean_object* v___y_4864_, lean_object* v___y_4865_, lean_object* v___y_4866_, lean_object* v___y_4867_, lean_object* v___y_4868_){
_start:
{
lean_object* v_res_4869_; 
v_res_4869_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4(v_oldTraces_4860_, v_data_4861_, v_ref_4862_, v_msg_4863_, v___y_4864_, v___y_4865_, v___y_4866_, v___y_4867_);
lean_dec(v___y_4867_);
lean_dec_ref(v___y_4866_);
lean_dec(v___y_4865_);
lean_dec_ref(v___y_4864_);
return v_res_4869_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(lean_object* v_x_4870_){
_start:
{
if (lean_obj_tag(v_x_4870_) == 0)
{
lean_object* v_a_4872_; lean_object* v___x_4874_; uint8_t v_isShared_4875_; uint8_t v_isSharedCheck_4879_; 
v_a_4872_ = lean_ctor_get(v_x_4870_, 0);
v_isSharedCheck_4879_ = !lean_is_exclusive(v_x_4870_);
if (v_isSharedCheck_4879_ == 0)
{
v___x_4874_ = v_x_4870_;
v_isShared_4875_ = v_isSharedCheck_4879_;
goto v_resetjp_4873_;
}
else
{
lean_inc(v_a_4872_);
lean_dec(v_x_4870_);
v___x_4874_ = lean_box(0);
v_isShared_4875_ = v_isSharedCheck_4879_;
goto v_resetjp_4873_;
}
v_resetjp_4873_:
{
lean_object* v___x_4877_; 
if (v_isShared_4875_ == 0)
{
lean_ctor_set_tag(v___x_4874_, 1);
v___x_4877_ = v___x_4874_;
goto v_reusejp_4876_;
}
else
{
lean_object* v_reuseFailAlloc_4878_; 
v_reuseFailAlloc_4878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4878_, 0, v_a_4872_);
v___x_4877_ = v_reuseFailAlloc_4878_;
goto v_reusejp_4876_;
}
v_reusejp_4876_:
{
return v___x_4877_;
}
}
}
else
{
lean_object* v_a_4880_; lean_object* v___x_4882_; uint8_t v_isShared_4883_; uint8_t v_isSharedCheck_4887_; 
v_a_4880_ = lean_ctor_get(v_x_4870_, 0);
v_isSharedCheck_4887_ = !lean_is_exclusive(v_x_4870_);
if (v_isSharedCheck_4887_ == 0)
{
v___x_4882_ = v_x_4870_;
v_isShared_4883_ = v_isSharedCheck_4887_;
goto v_resetjp_4881_;
}
else
{
lean_inc(v_a_4880_);
lean_dec(v_x_4870_);
v___x_4882_ = lean_box(0);
v_isShared_4883_ = v_isSharedCheck_4887_;
goto v_resetjp_4881_;
}
v_resetjp_4881_:
{
lean_object* v___x_4885_; 
if (v_isShared_4883_ == 0)
{
lean_ctor_set_tag(v___x_4882_, 0);
v___x_4885_ = v___x_4882_;
goto v_reusejp_4884_;
}
else
{
lean_object* v_reuseFailAlloc_4886_; 
v_reuseFailAlloc_4886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4886_, 0, v_a_4880_);
v___x_4885_ = v_reuseFailAlloc_4886_;
goto v_reusejp_4884_;
}
v_reusejp_4884_:
{
return v___x_4885_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg___boxed(lean_object* v_x_4888_, lean_object* v___y_4889_){
_start:
{
lean_object* v_res_4890_; 
v_res_4890_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(v_x_4888_);
return v_res_4890_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6(lean_object* v_e_4891_){
_start:
{
if (lean_obj_tag(v_e_4891_) == 0)
{
uint8_t v___x_4892_; 
v___x_4892_ = 2;
return v___x_4892_;
}
else
{
lean_object* v_a_4893_; uint8_t v___x_4894_; 
v_a_4893_ = lean_ctor_get(v_e_4891_, 0);
v___x_4894_ = lean_unbox(v_a_4893_);
if (v___x_4894_ == 0)
{
uint8_t v___x_4895_; 
v___x_4895_ = 1;
return v___x_4895_;
}
else
{
uint8_t v___x_4896_; 
v___x_4896_ = 0;
return v___x_4896_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6___boxed(lean_object* v_e_4897_){
_start:
{
uint8_t v_res_4898_; lean_object* v_r_4899_; 
v_res_4898_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6(v_e_4897_);
lean_dec_ref(v_e_4897_);
v_r_4899_ = lean_box(v_res_4898_);
return v_r_4899_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0(void){
_start:
{
lean_object* v___x_4900_; double v___x_4901_; 
v___x_4900_ = lean_unsigned_to_nat(0u);
v___x_4901_ = lean_float_of_nat(v___x_4900_);
return v___x_4901_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1(void){
_start:
{
lean_object* v___x_4902_; double v___x_4903_; 
v___x_4902_ = lean_unsigned_to_nat(1000u);
v___x_4903_ = lean_float_of_nat(v___x_4902_);
return v___x_4903_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4(lean_object* v_cls_4904_, uint8_t v_collapsed_4905_, lean_object* v_tag_4906_, lean_object* v_opts_4907_, uint8_t v_clsEnabled_4908_, lean_object* v_oldTraces_4909_, lean_object* v_ref_4910_, lean_object* v_msg_4911_, lean_object* v_resStartStop_4912_, lean_object* v___y_4913_, lean_object* v___y_4914_, lean_object* v___y_4915_, lean_object* v___y_4916_){
_start:
{
lean_object* v_fst_4918_; lean_object* v_snd_4919_; lean_object* v_data_4921_; lean_object* v_fst_4932_; lean_object* v_snd_4933_; lean_object* v___x_4934_; uint8_t v___x_4935_; uint8_t v___y_4946_; double v___y_4977_; 
v_fst_4918_ = lean_ctor_get(v_resStartStop_4912_, 0);
lean_inc(v_fst_4918_);
v_snd_4919_ = lean_ctor_get(v_resStartStop_4912_, 1);
lean_inc(v_snd_4919_);
lean_dec_ref(v_resStartStop_4912_);
v_fst_4932_ = lean_ctor_get(v_snd_4919_, 0);
lean_inc(v_fst_4932_);
v_snd_4933_ = lean_ctor_get(v_snd_4919_, 1);
lean_inc(v_snd_4933_);
lean_dec(v_snd_4919_);
v___x_4934_ = l_Lean_trace_profiler;
v___x_4935_ = lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(v_opts_4907_, v___x_4934_);
if (v___x_4935_ == 0)
{
v___y_4946_ = v___x_4935_;
goto v___jp_4945_;
}
else
{
lean_object* v___x_4982_; uint8_t v___x_4983_; 
v___x_4982_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4983_ = lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(v_opts_4907_, v___x_4982_);
if (v___x_4983_ == 0)
{
lean_object* v___x_4984_; lean_object* v___x_4985_; double v___x_4986_; double v___x_4987_; double v___x_4988_; 
v___x_4984_ = l_Lean_trace_profiler_threshold;
v___x_4985_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7(v_opts_4907_, v___x_4984_);
v___x_4986_ = lean_float_of_nat(v___x_4985_);
v___x_4987_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__1);
v___x_4988_ = lean_float_div(v___x_4986_, v___x_4987_);
v___y_4977_ = v___x_4988_;
goto v___jp_4976_;
}
else
{
lean_object* v___x_4989_; lean_object* v___x_4990_; double v___x_4991_; 
v___x_4989_ = l_Lean_trace_profiler_threshold;
v___x_4990_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__7(v_opts_4907_, v___x_4989_);
v___x_4991_ = lean_float_of_nat(v___x_4990_);
v___y_4977_ = v___x_4991_;
goto v___jp_4976_;
}
}
v___jp_4920_:
{
lean_object* v___x_4922_; 
v___x_4922_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__4(v_oldTraces_4909_, v_data_4921_, v_ref_4910_, v_msg_4911_, v___y_4913_, v___y_4914_, v___y_4915_, v___y_4916_);
if (lean_obj_tag(v___x_4922_) == 0)
{
lean_object* v___x_4923_; 
lean_dec_ref_known(v___x_4922_, 1);
v___x_4923_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(v_fst_4918_);
return v___x_4923_;
}
else
{
lean_object* v_a_4924_; lean_object* v___x_4926_; uint8_t v_isShared_4927_; uint8_t v_isSharedCheck_4931_; 
lean_dec(v_fst_4918_);
v_a_4924_ = lean_ctor_get(v___x_4922_, 0);
v_isSharedCheck_4931_ = !lean_is_exclusive(v___x_4922_);
if (v_isSharedCheck_4931_ == 0)
{
v___x_4926_ = v___x_4922_;
v_isShared_4927_ = v_isSharedCheck_4931_;
goto v_resetjp_4925_;
}
else
{
lean_inc(v_a_4924_);
lean_dec(v___x_4922_);
v___x_4926_ = lean_box(0);
v_isShared_4927_ = v_isSharedCheck_4931_;
goto v_resetjp_4925_;
}
v_resetjp_4925_:
{
lean_object* v___x_4929_; 
if (v_isShared_4927_ == 0)
{
v___x_4929_ = v___x_4926_;
goto v_reusejp_4928_;
}
else
{
lean_object* v_reuseFailAlloc_4930_; 
v_reuseFailAlloc_4930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4930_, 0, v_a_4924_);
v___x_4929_ = v_reuseFailAlloc_4930_;
goto v_reusejp_4928_;
}
v_reusejp_4928_:
{
return v___x_4929_;
}
}
}
}
v___jp_4936_:
{
uint8_t v_result_4937_; lean_object* v___x_4938_; lean_object* v___x_4939_; double v___x_4940_; lean_object* v_data_4941_; 
v_result_4937_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__6(v_fst_4918_);
v___x_4938_ = lean_box(v_result_4937_);
v___x_4939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4939_, 0, v___x_4938_);
v___x_4940_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___closed__0);
lean_inc_ref(v_tag_4906_);
lean_inc_ref(v___x_4939_);
lean_inc(v_cls_4904_);
v_data_4941_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4941_, 0, v_cls_4904_);
lean_ctor_set(v_data_4941_, 1, v___x_4939_);
lean_ctor_set(v_data_4941_, 2, v_tag_4906_);
lean_ctor_set_float(v_data_4941_, sizeof(void*)*3, v___x_4940_);
lean_ctor_set_float(v_data_4941_, sizeof(void*)*3 + 8, v___x_4940_);
lean_ctor_set_uint8(v_data_4941_, sizeof(void*)*3 + 16, v_collapsed_4905_);
if (v___x_4935_ == 0)
{
lean_dec_ref_known(v___x_4939_, 1);
lean_dec(v_snd_4933_);
lean_dec(v_fst_4932_);
lean_dec_ref(v_tag_4906_);
lean_dec(v_cls_4904_);
v_data_4921_ = v_data_4941_;
goto v___jp_4920_;
}
else
{
lean_object* v_data_4942_; double v___x_4943_; double v___x_4944_; 
lean_dec_ref_known(v_data_4941_, 3);
v_data_4942_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4942_, 0, v_cls_4904_);
lean_ctor_set(v_data_4942_, 1, v___x_4939_);
lean_ctor_set(v_data_4942_, 2, v_tag_4906_);
v___x_4943_ = lean_unbox_float(v_fst_4932_);
lean_dec(v_fst_4932_);
lean_ctor_set_float(v_data_4942_, sizeof(void*)*3, v___x_4943_);
v___x_4944_ = lean_unbox_float(v_snd_4933_);
lean_dec(v_snd_4933_);
lean_ctor_set_float(v_data_4942_, sizeof(void*)*3 + 8, v___x_4944_);
lean_ctor_set_uint8(v_data_4942_, sizeof(void*)*3 + 16, v_collapsed_4905_);
v_data_4921_ = v_data_4942_;
goto v___jp_4920_;
}
}
v___jp_4945_:
{
if (v_clsEnabled_4908_ == 0)
{
if (v___y_4946_ == 0)
{
lean_object* v___x_4947_; lean_object* v_traceState_4948_; lean_object* v_env_4949_; lean_object* v_nextMacroScope_4950_; lean_object* v_ngen_4951_; lean_object* v_auxDeclNGen_4952_; lean_object* v_cache_4953_; lean_object* v_messages_4954_; lean_object* v_infoState_4955_; lean_object* v_snapshotTasks_4956_; lean_object* v___x_4958_; uint8_t v_isShared_4959_; uint8_t v_isSharedCheck_4975_; 
lean_dec(v_snd_4933_);
lean_dec(v_fst_4932_);
lean_dec_ref(v_msg_4911_);
lean_dec(v_ref_4910_);
lean_dec_ref(v_tag_4906_);
lean_dec(v_cls_4904_);
v___x_4947_ = lean_st_ref_take(v___y_4916_);
v_traceState_4948_ = lean_ctor_get(v___x_4947_, 4);
v_env_4949_ = lean_ctor_get(v___x_4947_, 0);
v_nextMacroScope_4950_ = lean_ctor_get(v___x_4947_, 1);
v_ngen_4951_ = lean_ctor_get(v___x_4947_, 2);
v_auxDeclNGen_4952_ = lean_ctor_get(v___x_4947_, 3);
v_cache_4953_ = lean_ctor_get(v___x_4947_, 5);
v_messages_4954_ = lean_ctor_get(v___x_4947_, 6);
v_infoState_4955_ = lean_ctor_get(v___x_4947_, 7);
v_snapshotTasks_4956_ = lean_ctor_get(v___x_4947_, 8);
v_isSharedCheck_4975_ = !lean_is_exclusive(v___x_4947_);
if (v_isSharedCheck_4975_ == 0)
{
v___x_4958_ = v___x_4947_;
v_isShared_4959_ = v_isSharedCheck_4975_;
goto v_resetjp_4957_;
}
else
{
lean_inc(v_snapshotTasks_4956_);
lean_inc(v_infoState_4955_);
lean_inc(v_messages_4954_);
lean_inc(v_cache_4953_);
lean_inc(v_traceState_4948_);
lean_inc(v_auxDeclNGen_4952_);
lean_inc(v_ngen_4951_);
lean_inc(v_nextMacroScope_4950_);
lean_inc(v_env_4949_);
lean_dec(v___x_4947_);
v___x_4958_ = lean_box(0);
v_isShared_4959_ = v_isSharedCheck_4975_;
goto v_resetjp_4957_;
}
v_resetjp_4957_:
{
uint64_t v_tid_4960_; lean_object* v_traces_4961_; lean_object* v___x_4963_; uint8_t v_isShared_4964_; uint8_t v_isSharedCheck_4974_; 
v_tid_4960_ = lean_ctor_get_uint64(v_traceState_4948_, sizeof(void*)*1);
v_traces_4961_ = lean_ctor_get(v_traceState_4948_, 0);
v_isSharedCheck_4974_ = !lean_is_exclusive(v_traceState_4948_);
if (v_isSharedCheck_4974_ == 0)
{
v___x_4963_ = v_traceState_4948_;
v_isShared_4964_ = v_isSharedCheck_4974_;
goto v_resetjp_4962_;
}
else
{
lean_inc(v_traces_4961_);
lean_dec(v_traceState_4948_);
v___x_4963_ = lean_box(0);
v_isShared_4964_ = v_isSharedCheck_4974_;
goto v_resetjp_4962_;
}
v_resetjp_4962_:
{
lean_object* v___x_4965_; lean_object* v___x_4967_; 
v___x_4965_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_4909_, v_traces_4961_);
lean_dec_ref(v_traces_4961_);
if (v_isShared_4964_ == 0)
{
lean_ctor_set(v___x_4963_, 0, v___x_4965_);
v___x_4967_ = v___x_4963_;
goto v_reusejp_4966_;
}
else
{
lean_object* v_reuseFailAlloc_4973_; 
v_reuseFailAlloc_4973_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_4973_, 0, v___x_4965_);
lean_ctor_set_uint64(v_reuseFailAlloc_4973_, sizeof(void*)*1, v_tid_4960_);
v___x_4967_ = v_reuseFailAlloc_4973_;
goto v_reusejp_4966_;
}
v_reusejp_4966_:
{
lean_object* v___x_4969_; 
if (v_isShared_4959_ == 0)
{
lean_ctor_set(v___x_4958_, 4, v___x_4967_);
v___x_4969_ = v___x_4958_;
goto v_reusejp_4968_;
}
else
{
lean_object* v_reuseFailAlloc_4972_; 
v_reuseFailAlloc_4972_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4972_, 0, v_env_4949_);
lean_ctor_set(v_reuseFailAlloc_4972_, 1, v_nextMacroScope_4950_);
lean_ctor_set(v_reuseFailAlloc_4972_, 2, v_ngen_4951_);
lean_ctor_set(v_reuseFailAlloc_4972_, 3, v_auxDeclNGen_4952_);
lean_ctor_set(v_reuseFailAlloc_4972_, 4, v___x_4967_);
lean_ctor_set(v_reuseFailAlloc_4972_, 5, v_cache_4953_);
lean_ctor_set(v_reuseFailAlloc_4972_, 6, v_messages_4954_);
lean_ctor_set(v_reuseFailAlloc_4972_, 7, v_infoState_4955_);
lean_ctor_set(v_reuseFailAlloc_4972_, 8, v_snapshotTasks_4956_);
v___x_4969_ = v_reuseFailAlloc_4972_;
goto v_reusejp_4968_;
}
v_reusejp_4968_:
{
lean_object* v___x_4970_; lean_object* v___x_4971_; 
v___x_4970_ = lean_st_ref_set(v___y_4916_, v___x_4969_);
v___x_4971_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(v_fst_4918_);
return v___x_4971_;
}
}
}
}
}
else
{
goto v___jp_4936_;
}
}
else
{
goto v___jp_4936_;
}
}
v___jp_4976_:
{
double v___x_4978_; double v___x_4979_; double v___x_4980_; uint8_t v___x_4981_; 
v___x_4978_ = lean_unbox_float(v_snd_4933_);
v___x_4979_ = lean_unbox_float(v_fst_4932_);
v___x_4980_ = lean_float_sub(v___x_4978_, v___x_4979_);
v___x_4981_ = lean_float_decLt(v___y_4977_, v___x_4980_);
v___y_4946_ = v___x_4981_;
goto v___jp_4945_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4___boxed(lean_object* v_cls_4992_, lean_object* v_collapsed_4993_, lean_object* v_tag_4994_, lean_object* v_opts_4995_, lean_object* v_clsEnabled_4996_, lean_object* v_oldTraces_4997_, lean_object* v_ref_4998_, lean_object* v_msg_4999_, lean_object* v_resStartStop_5000_, lean_object* v___y_5001_, lean_object* v___y_5002_, lean_object* v___y_5003_, lean_object* v___y_5004_, lean_object* v___y_5005_){
_start:
{
uint8_t v_collapsed_boxed_5006_; uint8_t v_clsEnabled_boxed_5007_; lean_object* v_res_5008_; 
v_collapsed_boxed_5006_ = lean_unbox(v_collapsed_4993_);
v_clsEnabled_boxed_5007_ = lean_unbox(v_clsEnabled_4996_);
v_res_5008_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4(v_cls_4992_, v_collapsed_boxed_5006_, v_tag_4994_, v_opts_4995_, v_clsEnabled_boxed_5007_, v_oldTraces_4997_, v_ref_4998_, v_msg_4999_, v_resStartStop_5000_, v___y_5001_, v___y_5002_, v___y_5003_, v___y_5004_);
lean_dec(v___y_5004_);
lean_dec_ref(v___y_5003_);
lean_dec(v___y_5002_);
lean_dec_ref(v___y_5001_);
lean_dec_ref(v_opts_4995_);
return v_res_5008_;
}
}
static lean_object* _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__5(void){
_start:
{
lean_object* v___x_5018_; lean_object* v___x_5019_; lean_object* v___x_5020_; 
v___x_5018_ = ((lean_object*)(lp_aesop_Aesop_isDefEqReducibleRigid___closed__2));
v___x_5019_ = ((lean_object*)(lp_aesop_Aesop_isDefEqReducibleRigid___closed__4));
v___x_5020_ = l_Lean_Name_append(v___x_5019_, v___x_5018_);
return v___x_5020_;
}
}
static double _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__6(void){
_start:
{
lean_object* v___x_5021_; double v___x_5022_; 
v___x_5021_ = lean_unsigned_to_nat(1000000000u);
v___x_5022_ = lean_float_of_nat(v___x_5021_);
return v___x_5022_;
}
}
static lean_object* _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__8(void){
_start:
{
lean_object* v___x_5024_; lean_object* v___x_5025_; 
v___x_5024_ = ((lean_object*)(lp_aesop_Aesop_isDefEqReducibleRigid___closed__7));
v___x_5025_ = l_Lean_stringToMessageData(v___x_5024_);
return v___x_5025_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid(lean_object* v_s_5026_, lean_object* v_t_5027_, lean_object* v_a_5028_, lean_object* v_a_5029_, lean_object* v_a_5030_, lean_object* v_a_5031_){
_start:
{
lean_object* v_options_5033_; lean_object* v_fileName_5034_; lean_object* v_fileMap_5035_; lean_object* v_currRecDepth_5036_; lean_object* v_maxRecDepth_5037_; lean_object* v_ref_5038_; lean_object* v_currNamespace_5039_; lean_object* v_openDecls_5040_; lean_object* v_initHeartbeats_5041_; lean_object* v_maxHeartbeats_5042_; lean_object* v_quotContext_5043_; lean_object* v_currMacroScope_5044_; uint8_t v_diag_5045_; lean_object* v_cancelTk_x3f_5046_; uint8_t v_suppressElabErrors_5047_; lean_object* v_inheritedTraceOptions_5048_; uint8_t v_hasTrace_5049_; uint8_t v___x_5050_; lean_object* v___x_5051_; lean_object* v___f_5052_; uint8_t v___x_5053_; 
v_options_5033_ = lean_ctor_get(v_a_5030_, 2);
v_fileName_5034_ = lean_ctor_get(v_a_5030_, 0);
v_fileMap_5035_ = lean_ctor_get(v_a_5030_, 1);
v_currRecDepth_5036_ = lean_ctor_get(v_a_5030_, 3);
v_maxRecDepth_5037_ = lean_ctor_get(v_a_5030_, 4);
v_ref_5038_ = lean_ctor_get(v_a_5030_, 5);
v_currNamespace_5039_ = lean_ctor_get(v_a_5030_, 6);
v_openDecls_5040_ = lean_ctor_get(v_a_5030_, 7);
v_initHeartbeats_5041_ = lean_ctor_get(v_a_5030_, 8);
v_maxHeartbeats_5042_ = lean_ctor_get(v_a_5030_, 9);
v_quotContext_5043_ = lean_ctor_get(v_a_5030_, 10);
v_currMacroScope_5044_ = lean_ctor_get(v_a_5030_, 11);
v_diag_5045_ = lean_ctor_get_uint8(v_a_5030_, sizeof(void*)*14);
v_cancelTk_x3f_5046_ = lean_ctor_get(v_a_5030_, 12);
v_suppressElabErrors_5047_ = lean_ctor_get_uint8(v_a_5030_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_5048_ = lean_ctor_get(v_a_5030_, 13);
v_hasTrace_5049_ = lean_ctor_get_uint8(v_options_5033_, sizeof(void*)*1);
v___x_5050_ = 2;
v___x_5051_ = lean_box(v___x_5050_);
lean_inc_ref(v_t_5027_);
lean_inc_ref(v_s_5026_);
v___f_5052_ = lean_alloc_closure((void*)(lp_aesop_Aesop_isDefEqReducibleRigid___lam__0___boxed), 8, 3);
lean_closure_set(v___f_5052_, 0, v___x_5051_);
lean_closure_set(v___f_5052_, 1, v_s_5026_);
lean_closure_set(v___f_5052_, 2, v_t_5027_);
v___x_5053_ = 0;
if (v_hasTrace_5049_ == 0)
{
lean_object* v___x_5054_; 
lean_dec_ref(v_t_5027_);
lean_dec_ref(v_s_5026_);
v___x_5054_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v___f_5052_, v___x_5053_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
return v___x_5054_;
}
else
{
lean_object* v___x_5055_; lean_object* v___x_5056_; lean_object* v___x_5057_; uint8_t v___x_5058_; lean_object* v___y_5060_; lean_object* v___y_5061_; lean_object* v___y_5062_; lean_object* v_a_5063_; lean_object* v___y_5076_; lean_object* v___y_5077_; lean_object* v___y_5078_; lean_object* v_a_5079_; 
v___x_5055_ = ((lean_object*)(lp_aesop_Aesop_isDefEqReducibleRigid___closed__2));
v___x_5056_ = ((lean_object*)(lp_aesop_Aesop_runTacticsCapturingPostState___closed__10));
v___x_5057_ = lean_obj_once(&lp_aesop_Aesop_isDefEqReducibleRigid___closed__5, &lp_aesop_Aesop_isDefEqReducibleRigid___closed__5_once, _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__5);
v___x_5058_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_5048_, v_options_5033_, v___x_5057_);
if (v___x_5058_ == 0)
{
lean_object* v___x_5138_; uint8_t v___x_5139_; 
v___x_5138_ = l_Lean_trace_profiler;
v___x_5139_ = lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(v_options_5033_, v___x_5138_);
if (v___x_5139_ == 0)
{
lean_object* v___x_5140_; 
lean_dec_ref(v_t_5027_);
lean_dec_ref(v_s_5026_);
v___x_5140_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v___f_5052_, v___x_5053_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
return v___x_5140_;
}
else
{
goto v___jp_5088_;
}
}
else
{
goto v___jp_5088_;
}
v___jp_5059_:
{
lean_object* v___x_5064_; double v___x_5065_; double v___x_5066_; double v___x_5067_; double v___x_5068_; double v___x_5069_; lean_object* v___x_5070_; lean_object* v___x_5071_; lean_object* v___x_5072_; lean_object* v___x_5073_; lean_object* v___x_5074_; 
v___x_5064_ = lean_io_mono_nanos_now();
v___x_5065_ = lean_float_of_nat(v___y_5061_);
v___x_5066_ = lean_float_once(&lp_aesop_Aesop_isDefEqReducibleRigid___closed__6, &lp_aesop_Aesop_isDefEqReducibleRigid___closed__6_once, _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__6);
v___x_5067_ = lean_float_div(v___x_5065_, v___x_5066_);
v___x_5068_ = lean_float_of_nat(v___x_5064_);
v___x_5069_ = lean_float_div(v___x_5068_, v___x_5066_);
v___x_5070_ = lean_box_float(v___x_5067_);
v___x_5071_ = lean_box_float(v___x_5069_);
v___x_5072_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5072_, 0, v___x_5070_);
lean_ctor_set(v___x_5072_, 1, v___x_5071_);
v___x_5073_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5073_, 0, v_a_5063_);
lean_ctor_set(v___x_5073_, 1, v___x_5072_);
lean_inc(v_ref_5038_);
v___x_5074_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4(v___x_5055_, v_hasTrace_5049_, v___x_5056_, v_options_5033_, v___x_5058_, v___y_5060_, v_ref_5038_, v___y_5062_, v___x_5073_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
return v___x_5074_;
}
v___jp_5075_:
{
lean_object* v___x_5080_; double v___x_5081_; double v___x_5082_; lean_object* v___x_5083_; lean_object* v___x_5084_; lean_object* v___x_5085_; lean_object* v___x_5086_; lean_object* v___x_5087_; 
v___x_5080_ = lean_io_get_num_heartbeats();
v___x_5081_ = lean_float_of_nat(v___y_5078_);
v___x_5082_ = lean_float_of_nat(v___x_5080_);
v___x_5083_ = lean_box_float(v___x_5081_);
v___x_5084_ = lean_box_float(v___x_5082_);
v___x_5085_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5085_, 0, v___x_5083_);
lean_ctor_set(v___x_5085_, 1, v___x_5084_);
v___x_5086_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5086_, 0, v_a_5079_);
lean_ctor_set(v___x_5086_, 1, v___x_5085_);
lean_inc(v_ref_5038_);
v___x_5087_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4(v___x_5055_, v_hasTrace_5049_, v___x_5056_, v_options_5033_, v___x_5058_, v___y_5076_, v_ref_5038_, v___y_5077_, v___x_5086_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
return v___x_5087_;
}
v___jp_5088_:
{
lean_object* v___x_5089_; lean_object* v_a_5090_; lean_object* v_ref_5091_; lean_object* v___x_5092_; lean_object* v___x_5093_; lean_object* v___x_5094_; lean_object* v___x_5095_; lean_object* v___x_5096_; lean_object* v___x_5097_; lean_object* v___x_5098_; lean_object* v_a_5099_; lean_object* v___x_5100_; uint8_t v___x_5101_; 
v___x_5089_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_isDefEqReducibleRigid_spec__1___redArg(v_a_5031_);
v_a_5090_ = lean_ctor_get(v___x_5089_, 0);
lean_inc(v_a_5090_);
lean_dec_ref(v___x_5089_);
v_ref_5091_ = l_Lean_replaceRef(v_ref_5038_, v_ref_5038_);
lean_inc_ref(v_inheritedTraceOptions_5048_);
lean_inc(v_cancelTk_x3f_5046_);
lean_inc(v_currMacroScope_5044_);
lean_inc(v_quotContext_5043_);
lean_inc(v_maxHeartbeats_5042_);
lean_inc(v_initHeartbeats_5041_);
lean_inc(v_openDecls_5040_);
lean_inc(v_currNamespace_5039_);
lean_inc(v_maxRecDepth_5037_);
lean_inc(v_currRecDepth_5036_);
lean_inc_ref(v_options_5033_);
lean_inc_ref(v_fileMap_5035_);
lean_inc_ref(v_fileName_5034_);
v___x_5092_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_5092_, 0, v_fileName_5034_);
lean_ctor_set(v___x_5092_, 1, v_fileMap_5035_);
lean_ctor_set(v___x_5092_, 2, v_options_5033_);
lean_ctor_set(v___x_5092_, 3, v_currRecDepth_5036_);
lean_ctor_set(v___x_5092_, 4, v_maxRecDepth_5037_);
lean_ctor_set(v___x_5092_, 5, v_ref_5091_);
lean_ctor_set(v___x_5092_, 6, v_currNamespace_5039_);
lean_ctor_set(v___x_5092_, 7, v_openDecls_5040_);
lean_ctor_set(v___x_5092_, 8, v_initHeartbeats_5041_);
lean_ctor_set(v___x_5092_, 9, v_maxHeartbeats_5042_);
lean_ctor_set(v___x_5092_, 10, v_quotContext_5043_);
lean_ctor_set(v___x_5092_, 11, v_currMacroScope_5044_);
lean_ctor_set(v___x_5092_, 12, v_cancelTk_x3f_5046_);
lean_ctor_set(v___x_5092_, 13, v_inheritedTraceOptions_5048_);
lean_ctor_set_uint8(v___x_5092_, sizeof(void*)*14, v_diag_5045_);
lean_ctor_set_uint8(v___x_5092_, sizeof(void*)*14 + 1, v_suppressElabErrors_5047_);
v___x_5093_ = l_Lean_MessageData_ofExpr(v_s_5026_);
v___x_5094_ = lean_obj_once(&lp_aesop_Aesop_isDefEqReducibleRigid___closed__8, &lp_aesop_Aesop_isDefEqReducibleRigid___closed__8_once, _init_lp_aesop_Aesop_isDefEqReducibleRigid___closed__8);
v___x_5095_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5095_, 0, v___x_5093_);
lean_ctor_set(v___x_5095_, 1, v___x_5094_);
v___x_5096_ = l_Lean_MessageData_ofExpr(v_t_5027_);
v___x_5097_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5097_, 0, v___x_5095_);
lean_ctor_set(v___x_5097_, 1, v___x_5096_);
v___x_5098_ = lp_aesop_Lean_addMessageContextFull___at___00Aesop_isDefEqReducibleRigid_spec__2(v___x_5097_, v_a_5028_, v_a_5029_, v___x_5092_, v_a_5031_);
lean_dec_ref_known(v___x_5092_, 14);
v_a_5099_ = lean_ctor_get(v___x_5098_, 0);
lean_inc(v_a_5099_);
lean_dec_ref(v___x_5098_);
v___x_5100_ = l_Lean_trace_profiler_useHeartbeats;
v___x_5101_ = lp_aesop_Lean_Option_get___at___00Aesop_isDefEqReducibleRigid_spec__3(v_options_5033_, v___x_5100_);
if (v___x_5101_ == 0)
{
lean_object* v___x_5102_; lean_object* v___x_5103_; 
v___x_5102_ = lean_io_mono_nanos_now();
v___x_5103_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v___f_5052_, v___x_5053_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
if (lean_obj_tag(v___x_5103_) == 0)
{
lean_object* v_a_5104_; lean_object* v___x_5106_; uint8_t v_isShared_5107_; uint8_t v_isSharedCheck_5111_; 
v_a_5104_ = lean_ctor_get(v___x_5103_, 0);
v_isSharedCheck_5111_ = !lean_is_exclusive(v___x_5103_);
if (v_isSharedCheck_5111_ == 0)
{
v___x_5106_ = v___x_5103_;
v_isShared_5107_ = v_isSharedCheck_5111_;
goto v_resetjp_5105_;
}
else
{
lean_inc(v_a_5104_);
lean_dec(v___x_5103_);
v___x_5106_ = lean_box(0);
v_isShared_5107_ = v_isSharedCheck_5111_;
goto v_resetjp_5105_;
}
v_resetjp_5105_:
{
lean_object* v___x_5109_; 
if (v_isShared_5107_ == 0)
{
lean_ctor_set_tag(v___x_5106_, 1);
v___x_5109_ = v___x_5106_;
goto v_reusejp_5108_;
}
else
{
lean_object* v_reuseFailAlloc_5110_; 
v_reuseFailAlloc_5110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5110_, 0, v_a_5104_);
v___x_5109_ = v_reuseFailAlloc_5110_;
goto v_reusejp_5108_;
}
v_reusejp_5108_:
{
v___y_5060_ = v_a_5090_;
v___y_5061_ = v___x_5102_;
v___y_5062_ = v_a_5099_;
v_a_5063_ = v___x_5109_;
goto v___jp_5059_;
}
}
}
else
{
lean_object* v_a_5112_; lean_object* v___x_5114_; uint8_t v_isShared_5115_; uint8_t v_isSharedCheck_5119_; 
v_a_5112_ = lean_ctor_get(v___x_5103_, 0);
v_isSharedCheck_5119_ = !lean_is_exclusive(v___x_5103_);
if (v_isSharedCheck_5119_ == 0)
{
v___x_5114_ = v___x_5103_;
v_isShared_5115_ = v_isSharedCheck_5119_;
goto v_resetjp_5113_;
}
else
{
lean_inc(v_a_5112_);
lean_dec(v___x_5103_);
v___x_5114_ = lean_box(0);
v_isShared_5115_ = v_isSharedCheck_5119_;
goto v_resetjp_5113_;
}
v_resetjp_5113_:
{
lean_object* v___x_5117_; 
if (v_isShared_5115_ == 0)
{
lean_ctor_set_tag(v___x_5114_, 0);
v___x_5117_ = v___x_5114_;
goto v_reusejp_5116_;
}
else
{
lean_object* v_reuseFailAlloc_5118_; 
v_reuseFailAlloc_5118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5118_, 0, v_a_5112_);
v___x_5117_ = v_reuseFailAlloc_5118_;
goto v_reusejp_5116_;
}
v_reusejp_5116_:
{
v___y_5060_ = v_a_5090_;
v___y_5061_ = v___x_5102_;
v___y_5062_ = v_a_5099_;
v_a_5063_ = v___x_5117_;
goto v___jp_5059_;
}
}
}
}
else
{
lean_object* v___x_5120_; lean_object* v___x_5121_; 
v___x_5120_ = lean_io_get_num_heartbeats();
v___x_5121_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v___f_5052_, v___x_5053_, v_a_5028_, v_a_5029_, v_a_5030_, v_a_5031_);
if (lean_obj_tag(v___x_5121_) == 0)
{
lean_object* v_a_5122_; lean_object* v___x_5124_; uint8_t v_isShared_5125_; uint8_t v_isSharedCheck_5129_; 
v_a_5122_ = lean_ctor_get(v___x_5121_, 0);
v_isSharedCheck_5129_ = !lean_is_exclusive(v___x_5121_);
if (v_isSharedCheck_5129_ == 0)
{
v___x_5124_ = v___x_5121_;
v_isShared_5125_ = v_isSharedCheck_5129_;
goto v_resetjp_5123_;
}
else
{
lean_inc(v_a_5122_);
lean_dec(v___x_5121_);
v___x_5124_ = lean_box(0);
v_isShared_5125_ = v_isSharedCheck_5129_;
goto v_resetjp_5123_;
}
v_resetjp_5123_:
{
lean_object* v___x_5127_; 
if (v_isShared_5125_ == 0)
{
lean_ctor_set_tag(v___x_5124_, 1);
v___x_5127_ = v___x_5124_;
goto v_reusejp_5126_;
}
else
{
lean_object* v_reuseFailAlloc_5128_; 
v_reuseFailAlloc_5128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5128_, 0, v_a_5122_);
v___x_5127_ = v_reuseFailAlloc_5128_;
goto v_reusejp_5126_;
}
v_reusejp_5126_:
{
v___y_5076_ = v_a_5090_;
v___y_5077_ = v_a_5099_;
v___y_5078_ = v___x_5120_;
v_a_5079_ = v___x_5127_;
goto v___jp_5075_;
}
}
}
else
{
lean_object* v_a_5130_; lean_object* v___x_5132_; uint8_t v_isShared_5133_; uint8_t v_isSharedCheck_5137_; 
v_a_5130_ = lean_ctor_get(v___x_5121_, 0);
v_isSharedCheck_5137_ = !lean_is_exclusive(v___x_5121_);
if (v_isSharedCheck_5137_ == 0)
{
v___x_5132_ = v___x_5121_;
v_isShared_5133_ = v_isSharedCheck_5137_;
goto v_resetjp_5131_;
}
else
{
lean_inc(v_a_5130_);
lean_dec(v___x_5121_);
v___x_5132_ = lean_box(0);
v_isShared_5133_ = v_isSharedCheck_5137_;
goto v_resetjp_5131_;
}
v_resetjp_5131_:
{
lean_object* v___x_5135_; 
if (v_isShared_5133_ == 0)
{
lean_ctor_set_tag(v___x_5132_, 0);
v___x_5135_ = v___x_5132_;
goto v_reusejp_5134_;
}
else
{
lean_object* v_reuseFailAlloc_5136_; 
v_reuseFailAlloc_5136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5136_, 0, v_a_5130_);
v___x_5135_ = v_reuseFailAlloc_5136_;
goto v_reusejp_5134_;
}
v_reusejp_5134_:
{
v___y_5076_ = v_a_5090_;
v___y_5077_ = v_a_5099_;
v___y_5078_ = v___x_5120_;
v_a_5079_ = v___x_5135_;
goto v___jp_5075_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isDefEqReducibleRigid___boxed(lean_object* v_s_5141_, lean_object* v_t_5142_, lean_object* v_a_5143_, lean_object* v_a_5144_, lean_object* v_a_5145_, lean_object* v_a_5146_, lean_object* v_a_5147_){
_start:
{
lean_object* v_res_5148_; 
v_res_5148_ = lp_aesop_Aesop_isDefEqReducibleRigid(v_s_5141_, v_t_5142_, v_a_5143_, v_a_5144_, v_a_5145_, v_a_5146_);
lean_dec(v_a_5146_);
lean_dec_ref(v_a_5145_);
lean_dec(v_a_5144_);
lean_dec_ref(v_a_5143_);
return v_res_5148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5(lean_object* v_00_u03b1_5149_, lean_object* v_x_5150_, lean_object* v___y_5151_, lean_object* v___y_5152_, lean_object* v___y_5153_, lean_object* v___y_5154_){
_start:
{
lean_object* v___x_5156_; 
v___x_5156_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___redArg(v_x_5150_);
return v___x_5156_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5___boxed(lean_object* v_00_u03b1_5157_, lean_object* v_x_5158_, lean_object* v___y_5159_, lean_object* v___y_5160_, lean_object* v___y_5161_, lean_object* v___y_5162_, lean_object* v___y_5163_){
_start:
{
lean_object* v_res_5164_; 
v_res_5164_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00Aesop_isDefEqReducibleRigid_spec__4_spec__5(v_00_u03b1_5157_, v_x_5158_, v___y_5159_, v___y_5160_, v___y_5161_, v___y_5162_);
lean_dec(v___y_5162_);
lean_dec_ref(v___y_5161_);
lean_dec(v___y_5160_);
lean_dec_ref(v___y_5159_);
return v_res_5164_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1(lean_object* v_type_5165_, lean_object* v_as_5166_, size_t v_i_5167_, size_t v_stop_5168_, lean_object* v___y_5169_, lean_object* v___y_5170_, lean_object* v___y_5171_, lean_object* v___y_5172_){
_start:
{
uint8_t v___x_5174_; 
v___x_5174_ = lean_usize_dec_eq(v_i_5167_, v_stop_5168_);
if (v___x_5174_ == 0)
{
uint8_t v___x_5175_; uint8_t v_a_5177_; lean_object* v___x_5183_; 
v___x_5175_ = 1;
v___x_5183_ = lean_array_uget_borrowed(v_as_5166_, v_i_5167_);
if (lean_obj_tag(v___x_5183_) == 0)
{
v_a_5177_ = v___x_5174_;
goto v___jp_5176_;
}
else
{
lean_object* v_val_5184_; uint8_t v___x_5185_; 
v_val_5184_ = lean_ctor_get(v___x_5183_, 0);
v___x_5185_ = l_Lean_LocalDecl_isImplementationDetail(v_val_5184_);
if (v___x_5185_ == 0)
{
lean_object* v___x_5186_; lean_object* v___x_5187_; 
v___x_5186_ = l_Lean_LocalDecl_type(v_val_5184_);
lean_inc_ref(v_type_5165_);
v___x_5187_ = l_Lean_Meta_isExprDefEq(v_type_5165_, v___x_5186_, v___y_5169_, v___y_5170_, v___y_5171_, v___y_5172_);
if (lean_obj_tag(v___x_5187_) == 0)
{
lean_object* v_a_5188_; uint8_t v___x_5189_; 
v_a_5188_ = lean_ctor_get(v___x_5187_, 0);
lean_inc(v_a_5188_);
lean_dec_ref_known(v___x_5187_, 1);
v___x_5189_ = lean_unbox(v_a_5188_);
lean_dec(v_a_5188_);
v_a_5177_ = v___x_5189_;
goto v___jp_5176_;
}
else
{
lean_dec_ref(v_type_5165_);
return v___x_5187_;
}
}
else
{
v_a_5177_ = v___x_5174_;
goto v___jp_5176_;
}
}
v___jp_5176_:
{
if (v_a_5177_ == 0)
{
size_t v___x_5178_; size_t v___x_5179_; 
v___x_5178_ = ((size_t)1ULL);
v___x_5179_ = lean_usize_add(v_i_5167_, v___x_5178_);
v_i_5167_ = v___x_5179_;
goto _start;
}
else
{
lean_object* v___x_5181_; lean_object* v___x_5182_; 
lean_dec_ref(v_type_5165_);
v___x_5181_ = lean_box(v___x_5175_);
v___x_5182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5182_, 0, v___x_5181_);
return v___x_5182_;
}
}
}
else
{
uint8_t v___x_5190_; lean_object* v___x_5191_; lean_object* v___x_5192_; 
lean_dec_ref(v_type_5165_);
v___x_5190_ = 0;
v___x_5191_ = lean_box(v___x_5190_);
v___x_5192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5192_, 0, v___x_5191_);
return v___x_5192_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1___boxed(lean_object* v_type_5193_, lean_object* v_as_5194_, lean_object* v_i_5195_, lean_object* v_stop_5196_, lean_object* v___y_5197_, lean_object* v___y_5198_, lean_object* v___y_5199_, lean_object* v___y_5200_, lean_object* v___y_5201_){
_start:
{
size_t v_i_boxed_5202_; size_t v_stop_boxed_5203_; lean_object* v_res_5204_; 
v_i_boxed_5202_ = lean_unbox_usize(v_i_5195_);
lean_dec(v_i_5195_);
v_stop_boxed_5203_ = lean_unbox_usize(v_stop_5196_);
lean_dec(v_stop_5196_);
v_res_5204_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1(v_type_5193_, v_as_5194_, v_i_boxed_5202_, v_stop_boxed_5203_, v___y_5197_, v___y_5198_, v___y_5199_, v___y_5200_);
lean_dec(v___y_5200_);
lean_dec_ref(v___y_5199_);
lean_dec(v___y_5198_);
lean_dec_ref(v___y_5197_);
lean_dec_ref(v_as_5194_);
return v_res_5204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0(lean_object* v_type_5205_, lean_object* v_x_5206_, lean_object* v___y_5207_, lean_object* v___y_5208_, lean_object* v___y_5209_, lean_object* v___y_5210_){
_start:
{
if (lean_obj_tag(v_x_5206_) == 0)
{
lean_object* v_cs_5212_; lean_object* v___x_5214_; uint8_t v_isShared_5215_; uint8_t v_isSharedCheck_5230_; 
v_cs_5212_ = lean_ctor_get(v_x_5206_, 0);
v_isSharedCheck_5230_ = !lean_is_exclusive(v_x_5206_);
if (v_isSharedCheck_5230_ == 0)
{
v___x_5214_ = v_x_5206_;
v_isShared_5215_ = v_isSharedCheck_5230_;
goto v_resetjp_5213_;
}
else
{
lean_inc(v_cs_5212_);
lean_dec(v_x_5206_);
v___x_5214_ = lean_box(0);
v_isShared_5215_ = v_isSharedCheck_5230_;
goto v_resetjp_5213_;
}
v_resetjp_5213_:
{
lean_object* v___x_5216_; lean_object* v___x_5217_; uint8_t v___x_5218_; 
v___x_5216_ = lean_unsigned_to_nat(0u);
v___x_5217_ = lean_array_get_size(v_cs_5212_);
v___x_5218_ = lean_nat_dec_lt(v___x_5216_, v___x_5217_);
if (v___x_5218_ == 0)
{
lean_object* v___x_5219_; lean_object* v___x_5221_; 
lean_dec_ref(v_cs_5212_);
lean_dec_ref(v_type_5205_);
v___x_5219_ = lean_box(v___x_5218_);
if (v_isShared_5215_ == 0)
{
lean_ctor_set(v___x_5214_, 0, v___x_5219_);
v___x_5221_ = v___x_5214_;
goto v_reusejp_5220_;
}
else
{
lean_object* v_reuseFailAlloc_5222_; 
v_reuseFailAlloc_5222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5222_, 0, v___x_5219_);
v___x_5221_ = v_reuseFailAlloc_5222_;
goto v_reusejp_5220_;
}
v_reusejp_5220_:
{
return v___x_5221_;
}
}
else
{
if (v___x_5218_ == 0)
{
lean_object* v___x_5223_; lean_object* v___x_5225_; 
lean_dec_ref(v_cs_5212_);
lean_dec_ref(v_type_5205_);
v___x_5223_ = lean_box(v___x_5218_);
if (v_isShared_5215_ == 0)
{
lean_ctor_set(v___x_5214_, 0, v___x_5223_);
v___x_5225_ = v___x_5214_;
goto v_reusejp_5224_;
}
else
{
lean_object* v_reuseFailAlloc_5226_; 
v_reuseFailAlloc_5226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5226_, 0, v___x_5223_);
v___x_5225_ = v_reuseFailAlloc_5226_;
goto v_reusejp_5224_;
}
v_reusejp_5224_:
{
return v___x_5225_;
}
}
else
{
size_t v___x_5227_; size_t v___x_5228_; lean_object* v___x_5229_; 
lean_del_object(v___x_5214_);
v___x_5227_ = ((size_t)0ULL);
v___x_5228_ = lean_usize_of_nat(v___x_5217_);
v___x_5229_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1(v_type_5205_, v_cs_5212_, v___x_5227_, v___x_5228_, v___y_5207_, v___y_5208_, v___y_5209_, v___y_5210_);
lean_dec_ref(v_cs_5212_);
return v___x_5229_;
}
}
}
}
else
{
lean_object* v_vs_5231_; lean_object* v___x_5233_; uint8_t v_isShared_5234_; uint8_t v_isSharedCheck_5249_; 
v_vs_5231_ = lean_ctor_get(v_x_5206_, 0);
v_isSharedCheck_5249_ = !lean_is_exclusive(v_x_5206_);
if (v_isSharedCheck_5249_ == 0)
{
v___x_5233_ = v_x_5206_;
v_isShared_5234_ = v_isSharedCheck_5249_;
goto v_resetjp_5232_;
}
else
{
lean_inc(v_vs_5231_);
lean_dec(v_x_5206_);
v___x_5233_ = lean_box(0);
v_isShared_5234_ = v_isSharedCheck_5249_;
goto v_resetjp_5232_;
}
v_resetjp_5232_:
{
lean_object* v___x_5235_; lean_object* v___x_5236_; uint8_t v___x_5237_; 
v___x_5235_ = lean_unsigned_to_nat(0u);
v___x_5236_ = lean_array_get_size(v_vs_5231_);
v___x_5237_ = lean_nat_dec_lt(v___x_5235_, v___x_5236_);
if (v___x_5237_ == 0)
{
lean_object* v___x_5238_; lean_object* v___x_5240_; 
lean_dec_ref(v_vs_5231_);
lean_dec_ref(v_type_5205_);
v___x_5238_ = lean_box(v___x_5237_);
if (v_isShared_5234_ == 0)
{
lean_ctor_set_tag(v___x_5233_, 0);
lean_ctor_set(v___x_5233_, 0, v___x_5238_);
v___x_5240_ = v___x_5233_;
goto v_reusejp_5239_;
}
else
{
lean_object* v_reuseFailAlloc_5241_; 
v_reuseFailAlloc_5241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5241_, 0, v___x_5238_);
v___x_5240_ = v_reuseFailAlloc_5241_;
goto v_reusejp_5239_;
}
v_reusejp_5239_:
{
return v___x_5240_;
}
}
else
{
if (v___x_5237_ == 0)
{
lean_object* v___x_5242_; lean_object* v___x_5244_; 
lean_dec_ref(v_vs_5231_);
lean_dec_ref(v_type_5205_);
v___x_5242_ = lean_box(v___x_5237_);
if (v_isShared_5234_ == 0)
{
lean_ctor_set_tag(v___x_5233_, 0);
lean_ctor_set(v___x_5233_, 0, v___x_5242_);
v___x_5244_ = v___x_5233_;
goto v_reusejp_5243_;
}
else
{
lean_object* v_reuseFailAlloc_5245_; 
v_reuseFailAlloc_5245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5245_, 0, v___x_5242_);
v___x_5244_ = v_reuseFailAlloc_5245_;
goto v_reusejp_5243_;
}
v_reusejp_5243_:
{
return v___x_5244_;
}
}
else
{
size_t v___x_5246_; size_t v___x_5247_; lean_object* v___x_5248_; 
lean_del_object(v___x_5233_);
v___x_5246_ = ((size_t)0ULL);
v___x_5247_ = lean_usize_of_nat(v___x_5236_);
v___x_5248_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1(v_type_5205_, v_vs_5231_, v___x_5246_, v___x_5247_, v___y_5207_, v___y_5208_, v___y_5209_, v___y_5210_);
lean_dec_ref(v_vs_5231_);
return v___x_5248_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1(lean_object* v_type_5250_, lean_object* v_as_5251_, size_t v_i_5252_, size_t v_stop_5253_, lean_object* v___y_5254_, lean_object* v___y_5255_, lean_object* v___y_5256_, lean_object* v___y_5257_){
_start:
{
uint8_t v___x_5259_; 
v___x_5259_ = lean_usize_dec_eq(v_i_5252_, v_stop_5253_);
if (v___x_5259_ == 0)
{
lean_object* v___x_5260_; lean_object* v___x_5261_; 
v___x_5260_ = lean_array_uget_borrowed(v_as_5251_, v_i_5252_);
lean_inc(v___x_5260_);
lean_inc_ref(v_type_5250_);
v___x_5261_ = lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0(v_type_5250_, v___x_5260_, v___y_5254_, v___y_5255_, v___y_5256_, v___y_5257_);
if (lean_obj_tag(v___x_5261_) == 0)
{
lean_object* v_a_5262_; lean_object* v___x_5264_; uint8_t v_isShared_5265_; uint8_t v_isSharedCheck_5273_; 
v_a_5262_ = lean_ctor_get(v___x_5261_, 0);
v_isSharedCheck_5273_ = !lean_is_exclusive(v___x_5261_);
if (v_isSharedCheck_5273_ == 0)
{
v___x_5264_ = v___x_5261_;
v_isShared_5265_ = v_isSharedCheck_5273_;
goto v_resetjp_5263_;
}
else
{
lean_inc(v_a_5262_);
lean_dec(v___x_5261_);
v___x_5264_ = lean_box(0);
v_isShared_5265_ = v_isSharedCheck_5273_;
goto v_resetjp_5263_;
}
v_resetjp_5263_:
{
uint8_t v___x_5266_; 
v___x_5266_ = lean_unbox(v_a_5262_);
if (v___x_5266_ == 0)
{
size_t v___x_5267_; size_t v___x_5268_; 
lean_del_object(v___x_5264_);
lean_dec(v_a_5262_);
v___x_5267_ = ((size_t)1ULL);
v___x_5268_ = lean_usize_add(v_i_5252_, v___x_5267_);
v_i_5252_ = v___x_5268_;
goto _start;
}
else
{
lean_object* v___x_5271_; 
lean_dec_ref(v_type_5250_);
if (v_isShared_5265_ == 0)
{
v___x_5271_ = v___x_5264_;
goto v_reusejp_5270_;
}
else
{
lean_object* v_reuseFailAlloc_5272_; 
v_reuseFailAlloc_5272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5272_, 0, v_a_5262_);
v___x_5271_ = v_reuseFailAlloc_5272_;
goto v_reusejp_5270_;
}
v_reusejp_5270_:
{
return v___x_5271_;
}
}
}
}
else
{
lean_dec_ref(v_type_5250_);
return v___x_5261_;
}
}
else
{
uint8_t v___x_5274_; lean_object* v___x_5275_; lean_object* v___x_5276_; 
lean_dec_ref(v_type_5250_);
v___x_5274_ = 0;
v___x_5275_ = lean_box(v___x_5274_);
v___x_5276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5276_, 0, v___x_5275_);
return v___x_5276_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1___boxed(lean_object* v_type_5277_, lean_object* v_as_5278_, lean_object* v_i_5279_, lean_object* v_stop_5280_, lean_object* v___y_5281_, lean_object* v___y_5282_, lean_object* v___y_5283_, lean_object* v___y_5284_, lean_object* v___y_5285_){
_start:
{
size_t v_i_boxed_5286_; size_t v_stop_boxed_5287_; lean_object* v_res_5288_; 
v_i_boxed_5286_ = lean_unbox_usize(v_i_5279_);
lean_dec(v_i_5279_);
v_stop_boxed_5287_ = lean_unbox_usize(v_stop_5280_);
lean_dec(v_stop_5280_);
v_res_5288_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0_spec__1(v_type_5277_, v_as_5278_, v_i_boxed_5286_, v_stop_boxed_5287_, v___y_5281_, v___y_5282_, v___y_5283_, v___y_5284_);
lean_dec(v___y_5284_);
lean_dec_ref(v___y_5283_);
lean_dec(v___y_5282_);
lean_dec_ref(v___y_5281_);
lean_dec_ref(v_as_5278_);
return v_res_5288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0___boxed(lean_object* v_type_5289_, lean_object* v_x_5290_, lean_object* v___y_5291_, lean_object* v___y_5292_, lean_object* v___y_5293_, lean_object* v___y_5294_, lean_object* v___y_5295_){
_start:
{
lean_object* v_res_5296_; 
v_res_5296_ = lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0(v_type_5289_, v_x_5290_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_);
lean_dec(v___y_5294_);
lean_dec_ref(v___y_5293_);
lean_dec(v___y_5292_);
lean_dec_ref(v___y_5291_);
return v_res_5296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0(lean_object* v_type_5297_, lean_object* v_t_5298_, lean_object* v___y_5299_, lean_object* v___y_5300_, lean_object* v___y_5301_, lean_object* v___y_5302_){
_start:
{
lean_object* v_root_5304_; lean_object* v_tail_5305_; lean_object* v___x_5306_; 
v_root_5304_ = lean_ctor_get(v_t_5298_, 0);
lean_inc_ref(v_root_5304_);
v_tail_5305_ = lean_ctor_get(v_t_5298_, 1);
lean_inc_ref(v_tail_5305_);
lean_dec_ref(v_t_5298_);
lean_inc_ref(v_type_5297_);
v___x_5306_ = lp_aesop_Lean_PersistentArray_anyMAux___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__0(v_type_5297_, v_root_5304_, v___y_5299_, v___y_5300_, v___y_5301_, v___y_5302_);
if (lean_obj_tag(v___x_5306_) == 0)
{
lean_object* v_a_5307_; uint8_t v___x_5308_; 
v_a_5307_ = lean_ctor_get(v___x_5306_, 0);
lean_inc(v_a_5307_);
v___x_5308_ = lean_unbox(v_a_5307_);
lean_dec(v_a_5307_);
if (v___x_5308_ == 0)
{
lean_object* v___x_5309_; lean_object* v___x_5310_; uint8_t v___x_5311_; 
v___x_5309_ = lean_unsigned_to_nat(0u);
v___x_5310_ = lean_array_get_size(v_tail_5305_);
v___x_5311_ = lean_nat_dec_lt(v___x_5309_, v___x_5310_);
if (v___x_5311_ == 0)
{
lean_dec_ref(v_tail_5305_);
lean_dec_ref(v_type_5297_);
return v___x_5306_;
}
else
{
if (v___x_5311_ == 0)
{
lean_dec_ref(v_tail_5305_);
lean_dec_ref(v_type_5297_);
return v___x_5306_;
}
else
{
size_t v___x_5312_; size_t v___x_5313_; lean_object* v___x_5314_; 
lean_dec_ref_known(v___x_5306_, 1);
v___x_5312_ = ((size_t)0ULL);
v___x_5313_ = lean_usize_of_nat(v___x_5310_);
v___x_5314_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0_spec__1(v_type_5297_, v_tail_5305_, v___x_5312_, v___x_5313_, v___y_5299_, v___y_5300_, v___y_5301_, v___y_5302_);
lean_dec_ref(v_tail_5305_);
return v___x_5314_;
}
}
}
else
{
lean_dec_ref(v_tail_5305_);
lean_dec_ref(v_type_5297_);
return v___x_5306_;
}
}
else
{
lean_dec_ref(v_tail_5305_);
lean_dec_ref(v_type_5297_);
return v___x_5306_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0___boxed(lean_object* v_type_5315_, lean_object* v_t_5316_, lean_object* v___y_5317_, lean_object* v___y_5318_, lean_object* v___y_5319_, lean_object* v___y_5320_, lean_object* v___y_5321_){
_start:
{
lean_object* v_res_5322_; 
v_res_5322_ = lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0(v_type_5315_, v_t_5316_, v___y_5317_, v___y_5318_, v___y_5319_, v___y_5320_);
lean_dec(v___y_5320_);
lean_dec_ref(v___y_5319_);
lean_dec(v___y_5318_);
lean_dec_ref(v___y_5317_);
return v_res_5322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0(uint8_t v___x_5323_, lean_object* v_type_5324_, lean_object* v___y_5325_, lean_object* v___y_5326_, lean_object* v___y_5327_, lean_object* v___y_5328_){
_start:
{
lean_object* v_lctx_5330_; lean_object* v_keyedConfig_5331_; uint8_t v_trackZetaDelta_5332_; lean_object* v_zetaDeltaSet_5333_; lean_object* v_localInstances_5334_; lean_object* v_defEqCtx_x3f_5335_; lean_object* v_synthPendingDepth_5336_; lean_object* v_customCanUnfoldPredicate_x3f_5337_; uint8_t v_univApprox_5338_; uint8_t v_inTypeClassResolution_5339_; uint8_t v_cacheInferType_5340_; lean_object* v___x_5342_; uint8_t v_isShared_5343_; uint8_t v_isSharedCheck_5350_; 
v_lctx_5330_ = lean_ctor_get(v___y_5325_, 2);
v_keyedConfig_5331_ = lean_ctor_get(v___y_5325_, 0);
v_trackZetaDelta_5332_ = lean_ctor_get_uint8(v___y_5325_, sizeof(void*)*7);
v_zetaDeltaSet_5333_ = lean_ctor_get(v___y_5325_, 1);
v_localInstances_5334_ = lean_ctor_get(v___y_5325_, 3);
v_defEqCtx_x3f_5335_ = lean_ctor_get(v___y_5325_, 4);
v_synthPendingDepth_5336_ = lean_ctor_get(v___y_5325_, 5);
v_customCanUnfoldPredicate_x3f_5337_ = lean_ctor_get(v___y_5325_, 6);
v_univApprox_5338_ = lean_ctor_get_uint8(v___y_5325_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_5339_ = lean_ctor_get_uint8(v___y_5325_, sizeof(void*)*7 + 2);
v_cacheInferType_5340_ = lean_ctor_get_uint8(v___y_5325_, sizeof(void*)*7 + 3);
v_isSharedCheck_5350_ = !lean_is_exclusive(v___y_5325_);
if (v_isSharedCheck_5350_ == 0)
{
v___x_5342_ = v___y_5325_;
v_isShared_5343_ = v_isSharedCheck_5350_;
goto v_resetjp_5341_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_5337_);
lean_inc(v_synthPendingDepth_5336_);
lean_inc(v_defEqCtx_x3f_5335_);
lean_inc(v_localInstances_5334_);
lean_inc(v_lctx_5330_);
lean_inc(v_zetaDeltaSet_5333_);
lean_inc(v_keyedConfig_5331_);
lean_dec(v___y_5325_);
v___x_5342_ = lean_box(0);
v_isShared_5343_ = v_isSharedCheck_5350_;
goto v_resetjp_5341_;
}
v_resetjp_5341_:
{
lean_object* v_decls_5344_; lean_object* v___x_5345_; lean_object* v___x_5347_; 
v_decls_5344_ = lean_ctor_get(v_lctx_5330_, 1);
lean_inc_ref(v_decls_5344_);
v___x_5345_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_5323_, v_keyedConfig_5331_);
if (v_isShared_5343_ == 0)
{
lean_ctor_set(v___x_5342_, 0, v___x_5345_);
v___x_5347_ = v___x_5342_;
goto v_reusejp_5346_;
}
else
{
lean_object* v_reuseFailAlloc_5349_; 
v_reuseFailAlloc_5349_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_5349_, 0, v___x_5345_);
lean_ctor_set(v_reuseFailAlloc_5349_, 1, v_zetaDeltaSet_5333_);
lean_ctor_set(v_reuseFailAlloc_5349_, 2, v_lctx_5330_);
lean_ctor_set(v_reuseFailAlloc_5349_, 3, v_localInstances_5334_);
lean_ctor_set(v_reuseFailAlloc_5349_, 4, v_defEqCtx_x3f_5335_);
lean_ctor_set(v_reuseFailAlloc_5349_, 5, v_synthPendingDepth_5336_);
lean_ctor_set(v_reuseFailAlloc_5349_, 6, v_customCanUnfoldPredicate_x3f_5337_);
lean_ctor_set_uint8(v_reuseFailAlloc_5349_, sizeof(void*)*7, v_trackZetaDelta_5332_);
lean_ctor_set_uint8(v_reuseFailAlloc_5349_, sizeof(void*)*7 + 1, v_univApprox_5338_);
lean_ctor_set_uint8(v_reuseFailAlloc_5349_, sizeof(void*)*7 + 2, v_inTypeClassResolution_5339_);
lean_ctor_set_uint8(v_reuseFailAlloc_5349_, sizeof(void*)*7 + 3, v_cacheInferType_5340_);
v___x_5347_ = v_reuseFailAlloc_5349_;
goto v_reusejp_5346_;
}
v_reusejp_5346_:
{
lean_object* v___x_5348_; 
v___x_5348_ = lp_aesop_Lean_PersistentArray_anyM___at___00Aesop_isHypRedundantReducibleRigid_spec__0(v_type_5324_, v_decls_5344_, v___x_5347_, v___y_5326_, v___y_5327_, v___y_5328_);
lean_dec_ref(v___x_5347_);
return v___x_5348_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0___boxed(lean_object* v___x_5351_, lean_object* v_type_5352_, lean_object* v___y_5353_, lean_object* v___y_5354_, lean_object* v___y_5355_, lean_object* v___y_5356_, lean_object* v___y_5357_){
_start:
{
uint8_t v___x_2133__boxed_5358_; lean_object* v_res_5359_; 
v___x_2133__boxed_5358_ = lean_unbox(v___x_5351_);
v_res_5359_ = lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0(v___x_2133__boxed_5358_, v_type_5352_, v___y_5353_, v___y_5354_, v___y_5355_, v___y_5356_);
lean_dec(v___y_5356_);
lean_dec_ref(v___y_5355_);
lean_dec(v___y_5354_);
return v_res_5359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid(lean_object* v_type_5360_, lean_object* v_a_5361_, lean_object* v_a_5362_, lean_object* v_a_5363_, lean_object* v_a_5364_){
_start:
{
uint8_t v___x_5366_; lean_object* v___x_5367_; lean_object* v___f_5368_; uint8_t v___x_5369_; lean_object* v___x_5370_; 
v___x_5366_ = 2;
v___x_5367_ = lean_box(v___x_5366_);
v___f_5368_ = lean_alloc_closure((void*)(lp_aesop_Aesop_isHypRedundantReducibleRigid___lam__0___boxed), 7, 2);
lean_closure_set(v___f_5368_, 0, v___x_5367_);
lean_closure_set(v___f_5368_, 1, v_type_5360_);
v___x_5369_ = 0;
v___x_5370_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_isDefEqReducibleRigid_spec__0___redArg(v___f_5368_, v___x_5369_, v_a_5361_, v_a_5362_, v_a_5363_, v_a_5364_);
return v___x_5370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid___boxed(lean_object* v_type_5371_, lean_object* v_a_5372_, lean_object* v_a_5373_, lean_object* v_a_5374_, lean_object* v_a_5375_, lean_object* v_a_5376_){
_start:
{
lean_object* v_res_5377_; 
v_res_5377_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_type_5371_, v_a_5372_, v_a_5373_, v_a_5374_, v_a_5375_);
lean_dec(v_a_5375_);
lean_dec_ref(v_a_5374_);
lean_dec(v_a_5373_);
lean_dec_ref(v_a_5372_);
return v_res_5377_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Nanos(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_UnorderedArraySet(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree_Util(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_SimpTheorems(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_ForEachExpr(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Nanos(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_UnorderedArraySet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_SimpTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_ForEachExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Util_Basic_0__Aesop_initFn_00___x40_Aesop_Util_Basic_4267802711____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_smallErrorMessages = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_smallErrorMessages);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Nanos(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_UnorderedArraySet(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree_Util(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_SimpTheorems(uint8_t builtin);
lean_object* initialize_Lean_Util_ForEachExpr(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Nanos(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_UnorderedArraySet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_SimpTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_ForEachExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
