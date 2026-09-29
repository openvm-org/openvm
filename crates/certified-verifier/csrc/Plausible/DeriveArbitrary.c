// Lean compiler output
// Module: Plausible.DeriveArbitrary
// Imports: public import Init public meta import Init import Lean.Elab.Deriving.Basic import Lean.Elab.Deriving.Util import Plausible.ArbitraryFueled
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_isInductiveCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkContext(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
extern lean_object* l_Lean_instInhabitedInductiveVal_default;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInductArgNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkImplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInductiveApp___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInstImplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantInfo(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_unzip___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Pi_instInhabited___redArg___lam__0(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkLocalInstanceLetDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkLet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_mkCIdent(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isInductiveCore(lean_object*, lean_object*);
lean_object* l_Lean_Elab_registerDerivingHandler(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is not a constructor"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.MonadEnv"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4_value;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Lean.isCtor\?"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5_value;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7;
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__1 = (const lean_object*)&lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__1_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__2_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instance"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__4_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__5_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__6 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__6_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__7 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__7_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__9 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__9_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__11 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__11_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__12 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__12_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Plausible"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Arbitrary"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(87, 65, 220, 66, 159, 24, 65, 122)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2_value;
static const lean_closure_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__3 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__3_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__7 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__7_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "ArbitraryFueled"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__9 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__9_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(66, 156, 96, 211, 116, 176, 175, 121)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__12 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__12_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__14 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__14_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__16 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__16_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__17 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0;
static const lean_string_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__1 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__1_value;
static const lean_ctor_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__1_value)}};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__2 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__2_value;
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3;
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__0 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__0_value)}};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__1_value;
static lean_once_cell_t lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "doSeqItem"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__0 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__0_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value_aux_2),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(10, 94, 50, 120, 46, 251, 13, 13)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "doLetArrow"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value_aux_2),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 105, 77, 168, 26, 188, 17, 34)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "let"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__3 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__3_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value_aux_2),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doIdDecl"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__5 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__5_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value_aux_2),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 95, 84, 160, 28, 70, 78, 179)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__7 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__7_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "doExpr"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__8 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__8_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value_aux_2),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(130, 168, 60, 255, 153, 218, 88, 77)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9_value;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "arbitrary"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(87, 65, 220, 66, 159, 24, 65, 122)}};
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value_aux_1),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(33, 81, 203, 107, 28, 10, 107, 113)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "aux_arb"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__13 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__13_value;
static const lean_ctor_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(185, 189, 219, 93, 229, 167, 114, 204)}};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__14 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__14_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doReturn"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(210, 201, 30, 244, 146, 7, 54, 39)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "return"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__2 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__2_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__3 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__3_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__5 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__5_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__8 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__8_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__10 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__10_value;
static lean_once_cell_t lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__12 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__12_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__12_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__13 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__13_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__14 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__14_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value_aux_0),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__16_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__17 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__17_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Deriving"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__18 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__18_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value_aux_0),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value_aux_1),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__18_value),LEAN_SCALAR_PTR_LITERAL(230, 230, 99, 85, 138, 169, 166, 218)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__19_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__20 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__20_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value_aux_0),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__21_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__22 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__22_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__23_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__24 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__24_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__25_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__26 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__26_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__27 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__27_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28_value_aux_0),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__27_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__28_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__29 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__29_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30_value_aux_0),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__30_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__31 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__31_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__32 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__32_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__32_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__33 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__33_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__34 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__34_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__31_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__34_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__35 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__35_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__29_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__35_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__36 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__36_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__26_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__36_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__37 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__37_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__24_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__37_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__38 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__38_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__22_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__38_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__39 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__39_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__20_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__39_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__40 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__40_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__17_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__40_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__41 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__41_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__14_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__41_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__42 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__42_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__13_value),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__42_value)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "do"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__44 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__44_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__44_value),LEAN_SCALAR_PTR_LITERAL(181, 206, 135, 90, 45, 65, 187, 80)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "doSeqIndent"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__46 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__46_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__46_value),LEAN_SCALAR_PTR_LITERAL(93, 115, 138, 230, 225, 195, 43, 46)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48_value;
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__0 = (const lean_object*)&lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__0_value;
static const lean_closure_object lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__1 = (const lean_object*)&lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___closed__0 = (const lean_object*)&lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tuple"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__0 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 24, 88, 245, 200, 250, 27, 217)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__2 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__2_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_+_"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__6 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__6_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(57, 160, 89, 154, 247, 230, 95, 119)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__7 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__7_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__8 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__8_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "pure"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__9 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__9_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(182, 237, 62, 79, 212, 57, 236, 253)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__10 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__10_value;
static lean_once_cell_t lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed__const__1 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fuel"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__0_value),LEAN_SCALAR_PTR_LITERAL(175, 206, 203, 121, 129, 230, 90, 162)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__1_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "fuel'"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__2_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__2_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 85, 181, 139, 166, 22, 4)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__3_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0_value),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__4_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0_value),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__4_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__5_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__6_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__7 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__7_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__8 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__8_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__7_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__8_value),LEAN_SCALAR_PTR_LITERAL(51, 81, 163, 94, 71, 156, 90, 186)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__11 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__11_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term[_]"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__12 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__12_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__12_value),LEAN_SCALAR_PTR_LITERAL(86, 147, 168, 74, 195, 98, 232, 161)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__13 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__13_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__14 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__14_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__15 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__15_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "matchAlt"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__16 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__16_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__16_value),LEAN_SCALAR_PTR_LITERAL(178, 0, 203, 112, 215, 49, 100, 229)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__7_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__18 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__18_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Gen"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "oneOfWithDefault"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__21 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__21_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20_value),LEAN_SCALAR_PTR_LITERAL(59, 136, 107, 112, 1, 192, 26, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__21_value),LEAN_SCALAR_PTR_LITERAL(109, 229, 194, 54, 146, 117, 183, 247)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "frequency"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__24 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__24_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20_value),LEAN_SCALAR_PTR_LITERAL(59, 136, 107, 112, 1, 192, 26, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__24_value),LEAN_SCALAR_PTR_LITERAL(34, 221, 200, 239, 229, 186, 72, 51)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "match"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__27 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__27_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__27_value),LEAN_SCALAR_PTR_LITERAL(9, 208, 235, 82, 91, 230, 203, 159)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "matchDiscr"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__29 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__29_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__29_value),LEAN_SCALAR_PTR_LITERAL(99, 51, 127, 238, 206, 239, 57, 130)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__31 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__31_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "matchAlts"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__32 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__32_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__32_value),LEAN_SCALAR_PTR_LITERAL(193, 186, 26, 109, 82, 172, 197, 183)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "explicitBinder"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__34 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__34_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__34_value),LEAN_SCALAR_PTR_LITERAL(49, 119, 193, 23, 170, 93, 183, 238)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "letrec"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__36 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__36_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__36_value),LEAN_SCALAR_PTR_LITERAL(164, 19, 234, 96, 193, 73, 5, 238)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__38 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__38_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__38_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__39 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__39_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rec"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__40 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__40_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "letRecDecls"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__41 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__41_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__41_value),LEAN_SCALAR_PTR_LITERAL(103, 117, 148, 85, 88, 242, 214, 126)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "letRecDecl"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__43 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__43_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__43_value),LEAN_SCALAR_PTR_LITERAL(202, 48, 93, 231, 206, 172, 150, 190)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__45 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__45_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__45_value),LEAN_SCALAR_PTR_LITERAL(61, 47, 121, 206, 37, 68, 134, 111)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__47 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__47_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__47_value),LEAN_SCALAR_PTR_LITERAL(82, 96, 243, 36, 251, 209, 136, 237)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__49 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__49_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__49_value),LEAN_SCALAR_PTR_LITERAL(67, 92, 92, 51, 38, 250, 60, 190)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__53 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__53_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__53_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__55 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__55_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__55_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "derive Arbitrary failed, "};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__57 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__57_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = " has no non-recursive constructors"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__59 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__59_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__20_value),LEAN_SCALAR_PTR_LITERAL(59, 136, 107, 112, 1, 192, 26, 229)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "definition"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__4_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__4_value),LEAN_SCALAR_PTR_LITERAL(248, 187, 217, 228, 39, 184, 218, 135)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__6_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__7 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__7_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__7_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__9 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__9_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__9_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "arrow"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__11 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__11_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__11_value),LEAN_SCALAR_PTR_LITERAL(182, 146, 143, 73, 122, 115, 5, 207)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "→"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__13 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__13_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "partial"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__15 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__15_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 175, 198, 167, 172, 79, 14, 207)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mutual"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 205, 8, 5, 164, 77, 17, 1)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "end"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0;
static const lean_array_object lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "plausible"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__0_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "deriving"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__1_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(85, 3, 132, 224, 98, 236, 88, 42)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value_aux_1),((lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(75, 58, 57, 143, 49, 50, 130, 232)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__3_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__4_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5;
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__6_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is not an inductive type"};
static const lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__0 = (const lean_object*)&lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__0_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 74, .m_data = "Cannot derive instance of Arbitrary typeclass for indexed inductive type '"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__0 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__0_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__2 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__2_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 70, .m_capacity = 70, .m_length = 69, .m_data = "Cannot derive instance of Arbitrary typeclass for non-inductive types"};
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__0_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(lean_object* v_k_1_, lean_object* v_b_2_, lean_object* v_c_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
lean_inc(v___y_7_);
lean_inc_ref(v___y_6_);
lean_inc(v___y_5_);
lean_inc_ref(v___y_4_);
v___x_9_ = lean_apply_7(v_k_1_, v_b_2_, v_c_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, lean_box(0));
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed(lean_object* v_k_10_, lean_object* v_b_11_, lean_object* v_c_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(v_k_10_, v_b_11_, v_c_12_, v___y_13_, v___y_14_, v___y_15_, v___y_16_);
lean_dec(v___y_16_);
lean_dec_ref(v___y_15_);
lean_dec(v___y_14_);
lean_dec_ref(v___y_13_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(lean_object* v_type_19_, lean_object* v_k_20_, uint8_t v_cleanupAnnotations_21_, uint8_t v_whnfType_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
v___f_28_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_28_, 0, v_k_20_);
v___x_29_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_19_, v___f_28_, v_cleanupAnnotations_21_, v_whnfType_22_, v___y_23_, v___y_24_, v___y_25_, v___y_26_);
if (lean_obj_tag(v___x_29_) == 0)
{
lean_object* v_a_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_37_; 
v_a_30_ = lean_ctor_get(v___x_29_, 0);
v_isSharedCheck_37_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_37_ == 0)
{
v___x_32_ = v___x_29_;
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_a_30_);
lean_dec(v___x_29_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_35_; 
if (v_isShared_33_ == 0)
{
v___x_35_ = v___x_32_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_a_30_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
else
{
lean_object* v_a_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_45_; 
v_a_38_ = lean_ctor_get(v___x_29_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_45_ == 0)
{
v___x_40_ = v___x_29_;
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_a_38_);
lean_dec(v___x_29_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_43_; 
if (v_isShared_41_ == 0)
{
v___x_43_ = v___x_40_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v_a_38_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___boxed(lean_object* v_type_46_, lean_object* v_k_47_, lean_object* v_cleanupAnnotations_48_, lean_object* v_whnfType_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_55_; uint8_t v_whnfType_boxed_56_; lean_object* v_res_57_; 
v_cleanupAnnotations_boxed_55_ = lean_unbox(v_cleanupAnnotations_48_);
v_whnfType_boxed_56_ = lean_unbox(v_whnfType_49_);
v_res_57_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_46_, v_k_47_, v_cleanupAnnotations_boxed_55_, v_whnfType_boxed_56_, v___y_50_, v___y_51_, v___y_52_, v___y_53_);
lean_dec(v___y_53_);
lean_dec_ref(v___y_52_);
lean_dec(v___y_51_);
lean_dec_ref(v___y_50_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2(lean_object* v_00_u03b1_58_, lean_object* v_type_59_, lean_object* v_k_60_, uint8_t v_cleanupAnnotations_61_, uint8_t v_whnfType_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_59_, v_k_60_, v_cleanupAnnotations_61_, v_whnfType_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___boxed(lean_object* v_00_u03b1_69_, lean_object* v_type_70_, lean_object* v_k_71_, lean_object* v_cleanupAnnotations_72_, lean_object* v_whnfType_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_79_; uint8_t v_whnfType_boxed_80_; lean_object* v_res_81_; 
v_cleanupAnnotations_boxed_79_ = lean_unbox(v_cleanupAnnotations_72_);
v_whnfType_boxed_80_ = lean_unbox(v_whnfType_73_);
v_res_81_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2(v_00_u03b1_69_, v_type_70_, v_k_71_, v_cleanupAnnotations_boxed_79_, v_whnfType_boxed_80_, v___y_74_, v___y_75_, v___y_76_, v___y_77_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(lean_object* v_upperBound_85_, lean_object* v_args_86_, lean_object* v_indVal_87_, lean_object* v_a_88_, lean_object* v_b_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_a_95_; uint8_t v___x_99_; 
v___x_99_ = lean_nat_dec_lt(v_a_88_, v_upperBound_85_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; 
lean_dec(v_a_88_);
v___x_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_100_, 0, v_b_89_);
return v___x_100_;
}
else
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_101_ = lean_array_fget_borrowed(v_args_86_, v_a_88_);
v___x_102_ = l_Lean_Expr_fvarId_x21(v___x_101_);
v___x_103_ = l_Lean_FVarId_getType___redArg(v___x_102_, v___y_90_, v___y_91_, v___y_92_);
if (lean_obj_tag(v___x_103_) == 0)
{
lean_object* v_a_104_; lean_object* v_numParams_105_; uint8_t v___x_106_; 
v_a_104_ = lean_ctor_get(v___x_103_, 0);
lean_inc(v_a_104_);
lean_dec_ref_known(v___x_103_, 1);
v_numParams_105_ = lean_ctor_get(v_indVal_87_, 1);
v___x_106_ = lean_nat_dec_lt(v_a_88_, v_numParams_105_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1));
v___x_108_ = l_Lean_Core_mkFreshUserName(v___x_107_, v___y_91_, v___y_92_);
if (lean_obj_tag(v___x_108_) == 0)
{
lean_object* v_a_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_a_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_a_109_);
lean_dec_ref_known(v___x_108_, 1);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v_a_109_);
lean_ctor_set(v___x_110_, 1, v_a_104_);
v___x_111_ = lean_array_push(v_b_89_, v___x_110_);
v_a_95_ = v___x_111_;
goto v___jp_94_;
}
else
{
lean_object* v_a_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_119_; 
lean_dec(v_a_104_);
lean_dec_ref(v_b_89_);
lean_dec(v_a_88_);
v_a_112_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_119_ == 0)
{
v___x_114_ = v___x_108_;
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_a_112_);
lean_dec(v___x_108_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_117_; 
if (v_isShared_115_ == 0)
{
v___x_117_ = v___x_114_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_a_112_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
else
{
lean_dec(v_a_104_);
v_a_95_ = v_b_89_;
goto v___jp_94_;
}
}
else
{
lean_object* v_a_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_127_; 
lean_dec_ref(v_b_89_);
lean_dec(v_a_88_);
v_a_120_ = lean_ctor_get(v___x_103_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_127_ == 0)
{
v___x_122_ = v___x_103_;
v_isShared_123_ = v_isSharedCheck_127_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_a_120_);
lean_dec(v___x_103_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_127_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_125_; 
if (v_isShared_123_ == 0)
{
v___x_125_ = v___x_122_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_a_120_);
v___x_125_ = v_reuseFailAlloc_126_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
return v___x_125_;
}
}
}
}
v___jp_94_:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = lean_unsigned_to_nat(1u);
v___x_97_ = lean_nat_add(v_a_88_, v___x_96_);
lean_dec(v_a_88_);
v_a_88_ = v___x_97_;
v_b_89_ = v_a_95_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___boxed(lean_object* v_upperBound_128_, lean_object* v_args_129_, lean_object* v_indVal_130_, lean_object* v_a_131_, lean_object* v_b_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v_upperBound_128_, v_args_129_, v_indVal_130_, v_a_131_, v_b_132_, v___y_133_, v___y_134_, v___y_135_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec_ref(v___y_133_);
lean_dec_ref(v_indVal_130_);
lean_dec_ref(v_args_129_);
lean_dec(v_upperBound_128_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0(lean_object* v_indVal_140_, lean_object* v_args_141_, lean_object* v_x_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_148_ = lean_unsigned_to_nat(0u);
v___x_149_ = lean_array_get_size(v_args_141_);
v___x_150_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___closed__0));
v___x_151_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v___x_149_, v_args_141_, v_indVal_140_, v___x_148_, v___x_150_, v___y_143_, v___y_145_, v___y_146_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___boxed(lean_object* v_indVal_152_, lean_object* v_args_153_, lean_object* v_x_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0(v_indVal_152_, v_args_153_, v_x_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_);
lean_dec(v___y_158_);
lean_dec_ref(v___y_157_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
lean_dec_ref(v_x_154_);
lean_dec_ref(v_args_153_);
lean_dec_ref(v_indVal_152_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(lean_object* v_msgData_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v___x_167_; lean_object* v_env_168_; lean_object* v___x_169_; lean_object* v_mctx_170_; lean_object* v_lctx_171_; lean_object* v_options_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_167_ = lean_st_ref_get(v___y_165_);
v_env_168_ = lean_ctor_get(v___x_167_, 0);
lean_inc_ref(v_env_168_);
lean_dec(v___x_167_);
v___x_169_ = lean_st_ref_get(v___y_163_);
v_mctx_170_ = lean_ctor_get(v___x_169_, 0);
lean_inc_ref(v_mctx_170_);
lean_dec(v___x_169_);
v_lctx_171_ = lean_ctor_get(v___y_162_, 2);
v_options_172_ = lean_ctor_get(v___y_164_, 2);
lean_inc_ref(v_options_172_);
lean_inc_ref(v_lctx_171_);
v___x_173_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_173_, 0, v_env_168_);
lean_ctor_set(v___x_173_, 1, v_mctx_170_);
lean_ctor_set(v___x_173_, 2, v_lctx_171_);
lean_ctor_set(v___x_173_, 3, v_options_172_);
v___x_174_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_msgData_161_);
v___x_175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msgData_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(lean_object* v_msg_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v_ref_189_; lean_object* v___x_190_; lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_199_; 
v_ref_189_ = lean_ctor_get(v___y_186_, 5);
v___x_190_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_);
v_a_191_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_199_ == 0)
{
v___x_193_ = v___x_190_;
v_isShared_194_ = v_isSharedCheck_199_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_190_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_199_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_195_; lean_object* v___x_197_; 
lean_inc(v_ref_189_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v_ref_189_);
lean_ctor_set(v___x_195_, 1, v_a_191_);
if (v_isShared_194_ == 0)
{
lean_ctor_set_tag(v___x_193_, 1);
lean_ctor_set(v___x_193_, 0, v___x_195_);
v___x_197_ = v___x_193_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v___x_195_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg___boxed(lean_object* v_msg_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v_msg_200_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
return v_res_206_;
}
}
static lean_object* _init_lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = l_instMonadEIO(lean_box(0));
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(lean_object* v_msg_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v_toApplicative_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_281_; 
v___x_218_ = lean_obj_once(&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0, &lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0_once, _init_lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0);
v___x_219_ = l_StateRefT_x27_instMonad___redArg(v___x_218_);
v_toApplicative_220_ = lean_ctor_get(v___x_219_, 0);
v_isSharedCheck_281_ = !lean_is_exclusive(v___x_219_);
if (v_isSharedCheck_281_ == 0)
{
lean_object* v_unused_282_; 
v_unused_282_ = lean_ctor_get(v___x_219_, 1);
lean_dec(v_unused_282_);
v___x_222_ = v___x_219_;
v_isShared_223_ = v_isSharedCheck_281_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_toApplicative_220_);
lean_dec(v___x_219_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_281_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v_toFunctor_224_; lean_object* v_toSeq_225_; lean_object* v_toSeqLeft_226_; lean_object* v_toSeqRight_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_279_; 
v_toFunctor_224_ = lean_ctor_get(v_toApplicative_220_, 0);
v_toSeq_225_ = lean_ctor_get(v_toApplicative_220_, 2);
v_toSeqLeft_226_ = lean_ctor_get(v_toApplicative_220_, 3);
v_toSeqRight_227_ = lean_ctor_get(v_toApplicative_220_, 4);
v_isSharedCheck_279_ = !lean_is_exclusive(v_toApplicative_220_);
if (v_isSharedCheck_279_ == 0)
{
lean_object* v_unused_280_; 
v_unused_280_ = lean_ctor_get(v_toApplicative_220_, 1);
lean_dec(v_unused_280_);
v___x_229_ = v_toApplicative_220_;
v_isShared_230_ = v_isSharedCheck_279_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_toSeqRight_227_);
lean_inc(v_toSeqLeft_226_);
lean_inc(v_toSeq_225_);
lean_inc(v_toFunctor_224_);
lean_dec(v_toApplicative_220_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_279_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___f_231_; lean_object* v___f_232_; lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___x_235_; lean_object* v___f_236_; lean_object* v___f_237_; lean_object* v___f_238_; lean_object* v___x_240_; 
v___f_231_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1));
v___f_232_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_224_);
v___f_233_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_233_, 0, v_toFunctor_224_);
v___f_234_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_234_, 0, v_toFunctor_224_);
v___x_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_235_, 0, v___f_233_);
lean_ctor_set(v___x_235_, 1, v___f_234_);
v___f_236_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_236_, 0, v_toSeqRight_227_);
v___f_237_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_237_, 0, v_toSeqLeft_226_);
v___f_238_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_238_, 0, v_toSeq_225_);
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 4, v___f_236_);
lean_ctor_set(v___x_229_, 3, v___f_237_);
lean_ctor_set(v___x_229_, 2, v___f_238_);
lean_ctor_set(v___x_229_, 1, v___f_231_);
lean_ctor_set(v___x_229_, 0, v___x_235_);
v___x_240_ = v___x_229_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_235_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v___f_231_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v___f_238_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v___f_237_);
lean_ctor_set(v_reuseFailAlloc_278_, 4, v___f_236_);
v___x_240_ = v_reuseFailAlloc_278_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
lean_object* v___x_242_; 
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 1, v___f_232_);
lean_ctor_set(v___x_222_, 0, v___x_240_);
v___x_242_ = v___x_222_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_277_, 1, v___f_232_);
v___x_242_ = v_reuseFailAlloc_277_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
lean_object* v___x_243_; lean_object* v_toApplicative_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_275_; 
v___x_243_ = l_StateRefT_x27_instMonad___redArg(v___x_242_);
v_toApplicative_244_ = lean_ctor_get(v___x_243_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_243_);
if (v_isSharedCheck_275_ == 0)
{
lean_object* v_unused_276_; 
v_unused_276_ = lean_ctor_get(v___x_243_, 1);
lean_dec(v_unused_276_);
v___x_246_ = v___x_243_;
v_isShared_247_ = v_isSharedCheck_275_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_toApplicative_244_);
lean_dec(v___x_243_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_275_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v_toFunctor_248_; lean_object* v_toSeq_249_; lean_object* v_toSeqLeft_250_; lean_object* v_toSeqRight_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_273_; 
v_toFunctor_248_ = lean_ctor_get(v_toApplicative_244_, 0);
v_toSeq_249_ = lean_ctor_get(v_toApplicative_244_, 2);
v_toSeqLeft_250_ = lean_ctor_get(v_toApplicative_244_, 3);
v_toSeqRight_251_ = lean_ctor_get(v_toApplicative_244_, 4);
v_isSharedCheck_273_ = !lean_is_exclusive(v_toApplicative_244_);
if (v_isSharedCheck_273_ == 0)
{
lean_object* v_unused_274_; 
v_unused_274_ = lean_ctor_get(v_toApplicative_244_, 1);
lean_dec(v_unused_274_);
v___x_253_ = v_toApplicative_244_;
v_isShared_254_ = v_isSharedCheck_273_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_toSeqRight_251_);
lean_inc(v_toSeqLeft_250_);
lean_inc(v_toSeq_249_);
lean_inc(v_toFunctor_248_);
lean_dec(v_toApplicative_244_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_273_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___f_255_; lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___f_258_; lean_object* v___x_259_; lean_object* v___f_260_; lean_object* v___f_261_; lean_object* v___f_262_; lean_object* v___x_264_; 
v___f_255_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3));
v___f_256_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_248_);
v___f_257_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_257_, 0, v_toFunctor_248_);
v___f_258_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_258_, 0, v_toFunctor_248_);
v___x_259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_259_, 0, v___f_257_);
lean_ctor_set(v___x_259_, 1, v___f_258_);
v___f_260_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_260_, 0, v_toSeqRight_251_);
v___f_261_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_261_, 0, v_toSeqLeft_250_);
v___f_262_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_262_, 0, v_toSeq_249_);
if (v_isShared_254_ == 0)
{
lean_ctor_set(v___x_253_, 4, v___f_260_);
lean_ctor_set(v___x_253_, 3, v___f_261_);
lean_ctor_set(v___x_253_, 2, v___f_262_);
lean_ctor_set(v___x_253_, 1, v___f_255_);
lean_ctor_set(v___x_253_, 0, v___x_259_);
v___x_264_ = v___x_253_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_259_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v___f_255_);
lean_ctor_set(v_reuseFailAlloc_272_, 2, v___f_262_);
lean_ctor_set(v_reuseFailAlloc_272_, 3, v___f_261_);
lean_ctor_set(v_reuseFailAlloc_272_, 4, v___f_260_);
v___x_264_ = v_reuseFailAlloc_272_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
lean_object* v___x_266_; 
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 1, v___f_256_);
lean_ctor_set(v___x_246_, 0, v___x_264_);
v___x_266_ = v___x_246_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v___f_256_);
v___x_266_ = v_reuseFailAlloc_271_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_3027__overap_269_; lean_object* v___x_270_; 
v___x_267_ = lean_box(0);
v___x_268_ = l_instInhabitedOfMonad___redArg(v___x_266_, v___x_267_);
v___x_3027__overap_269_ = lean_panic_fn_borrowed(v___x_268_, v_msg_212_);
lean_dec(v___x_268_);
lean_inc(v___y_216_);
lean_inc_ref(v___y_215_);
lean_inc(v___y_214_);
lean_inc_ref(v___y_213_);
v___x_270_ = lean_apply_5(v___x_3027__overap_269_, v___y_213_, v___y_214_, v___y_215_, v___y_216_, lean_box(0));
return v___x_270_;
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
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___boxed(lean_object* v_msg_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(v_msg_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
return v_res_289_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0));
v___x_292_ = l_Lean_stringToMessageData(v___x_291_);
return v___x_292_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2));
v___x_295_ = l_Lean_stringToMessageData(v___x_294_);
return v___x_295_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7(void){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_299_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6));
v___x_300_ = lean_unsigned_to_nat(11u);
v___x_301_ = lean_unsigned_to_nat(122u);
v___x_302_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5));
v___x_303_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4));
v___x_304_ = l_mkPanicMessageWithDecl(v___x_303_, v___x_302_, v___x_301_, v___x_300_, v___x_299_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1(lean_object* v_constName_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_){
_start:
{
lean_object* v___x_319_; lean_object* v_env_320_; uint8_t v___x_321_; lean_object* v___x_322_; 
v___x_319_ = lean_st_ref_get(v___y_309_);
v_env_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc_ref(v_env_320_);
lean_dec(v___x_319_);
v___x_321_ = 0;
lean_inc(v_constName_305_);
v___x_322_ = l_Lean_Environment_findAsync_x3f(v_env_320_, v_constName_305_, v___x_321_);
if (lean_obj_tag(v___x_322_) == 1)
{
lean_object* v_val_323_; uint8_t v_kind_324_; 
v_val_323_ = lean_ctor_get(v___x_322_, 0);
lean_inc(v_val_323_);
lean_dec_ref_known(v___x_322_, 1);
v_kind_324_ = lean_ctor_get_uint8(v_val_323_, sizeof(void*)*3);
if (v_kind_324_ == 6)
{
lean_object* v___x_325_; 
v___x_325_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_323_);
if (lean_obj_tag(v___x_325_) == 6)
{
lean_object* v_val_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_333_; 
lean_dec(v_constName_305_);
v_val_326_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_333_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_333_ == 0)
{
v___x_328_ = v___x_325_;
v_isShared_329_ = v_isSharedCheck_333_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_val_326_);
lean_dec(v___x_325_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_333_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_331_; 
if (v_isShared_329_ == 0)
{
lean_ctor_set_tag(v___x_328_, 0);
v___x_331_ = v___x_328_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v_val_326_);
v___x_331_ = v_reuseFailAlloc_332_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
return v___x_331_;
}
}
}
else
{
lean_object* v___x_334_; lean_object* v___x_335_; 
lean_dec_ref(v___x_325_);
v___x_334_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7);
v___x_335_ = lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(v___x_334_, v___y_306_, v___y_307_, v___y_308_, v___y_309_);
if (lean_obj_tag(v___x_335_) == 0)
{
lean_object* v_a_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_344_; 
v_a_336_ = lean_ctor_get(v___x_335_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_335_);
if (v_isSharedCheck_344_ == 0)
{
v___x_338_ = v___x_335_;
v_isShared_339_ = v_isSharedCheck_344_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_a_336_);
lean_dec(v___x_335_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_344_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
if (lean_obj_tag(v_a_336_) == 0)
{
lean_del_object(v___x_338_);
goto v___jp_311_;
}
else
{
lean_object* v_val_340_; lean_object* v___x_342_; 
lean_dec(v_constName_305_);
v_val_340_ = lean_ctor_get(v_a_336_, 0);
lean_inc(v_val_340_);
lean_dec_ref_known(v_a_336_, 1);
if (v_isShared_339_ == 0)
{
lean_ctor_set(v___x_338_, 0, v_val_340_);
v___x_342_ = v___x_338_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_val_340_);
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
else
{
lean_object* v_a_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_352_; 
lean_dec(v_constName_305_);
v_a_345_ = lean_ctor_get(v___x_335_, 0);
v_isSharedCheck_352_ = !lean_is_exclusive(v___x_335_);
if (v_isSharedCheck_352_ == 0)
{
v___x_347_ = v___x_335_;
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_a_345_);
lean_dec(v___x_335_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_350_; 
if (v_isShared_348_ == 0)
{
v___x_350_ = v___x_347_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v_a_345_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
}
else
{
lean_dec(v_val_323_);
goto v___jp_311_;
}
}
else
{
lean_dec(v___x_322_);
goto v___jp_311_;
}
v___jp_311_:
{
lean_object* v___x_312_; uint8_t v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_312_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1);
v___x_313_ = 0;
v___x_314_ = l_Lean_MessageData_ofConstName(v_constName_305_, v___x_313_);
v___x_315_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_312_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
v___x_316_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3);
v___x_317_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_317_, 0, v___x_315_);
lean_ctor_set(v___x_317_, 1, v___x_316_);
v___x_318_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v___x_317_, v___y_306_, v___y_307_, v___y_308_, v___y_309_);
return v___x_318_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___boxed(lean_object* v_constName_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1(v_constName_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_);
lean_dec(v___y_357_);
lean_dec_ref(v___y_356_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg(lean_object* v_indVal_360_, lean_object* v_ctorName_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1(v_ctorName_361_, v_a_362_, v_a_363_, v_a_364_, v_a_365_);
if (lean_obj_tag(v___x_367_) == 0)
{
lean_object* v_a_368_; lean_object* v_toConstantVal_369_; lean_object* v_type_370_; lean_object* v___f_371_; uint8_t v___x_372_; lean_object* v___x_373_; 
v_a_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc(v_a_368_);
lean_dec_ref_known(v___x_367_, 1);
v_toConstantVal_369_ = lean_ctor_get(v_a_368_, 0);
lean_inc_ref(v_toConstantVal_369_);
lean_dec(v_a_368_);
v_type_370_ = lean_ctor_get(v_toConstantVal_369_, 2);
lean_inc_ref(v_type_370_);
lean_dec_ref(v_toConstantVal_369_);
v___f_371_ = lean_alloc_closure((void*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_371_, 0, v_indVal_360_);
v___x_372_ = 0;
v___x_373_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_370_, v___f_371_, v___x_372_, v___x_372_, v_a_362_, v_a_363_, v_a_364_, v_a_365_);
return v___x_373_;
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
lean_dec_ref(v_indVal_360_);
v_a_374_ = lean_ctor_get(v___x_367_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_367_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_367_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_367_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg___boxed(lean_object* v_indVal_382_, lean_object* v_ctorName_383_, lean_object* v_a_384_, lean_object* v_a_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg(v_indVal_382_, v_ctorName_383_, v_a_384_, v_a_385_, v_a_386_, v_a_387_);
lean_dec(v_a_387_);
lean_dec_ref(v_a_386_);
lean_dec(v_a_385_);
lean_dec_ref(v_a_384_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes(lean_object* v___header_390_, lean_object* v_indVal_391_, lean_object* v_ctorName_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg(v_indVal_391_, v_ctorName_392_, v_a_393_, v_a_394_, v_a_395_, v_a_396_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___boxed(lean_object* v___header_399_, lean_object* v_indVal_400_, lean_object* v_ctorName_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes(v___header_399_, v_indVal_400_, v_ctorName_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
lean_dec(v_a_405_);
lean_dec_ref(v_a_404_);
lean_dec(v_a_403_);
lean_dec_ref(v_a_402_);
lean_dec_ref(v___header_399_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0(lean_object* v_upperBound_408_, lean_object* v_args_409_, lean_object* v_indVal_410_, lean_object* v_inst_411_, lean_object* v_R_412_, lean_object* v_a_413_, lean_object* v_b_414_, lean_object* v_c_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v_upperBound_408_, v_args_409_, v_indVal_410_, v_a_413_, v_b_414_, v___y_416_, v___y_418_, v___y_419_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0___boxed(lean_object* v_upperBound_422_, lean_object* v_args_423_, lean_object* v_indVal_424_, lean_object* v_inst_425_, lean_object* v_R_426_, lean_object* v_a_427_, lean_object* v_b_428_, lean_object* v_c_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__0(v_upperBound_422_, v_args_423_, v_indVal_424_, v_inst_425_, v_R_426_, v_a_427_, v_b_428_, v_c_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
lean_dec(v___y_433_);
lean_dec_ref(v___y_432_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec_ref(v_indVal_424_);
lean_dec_ref(v_args_423_);
lean_dec(v_upperBound_422_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(lean_object* v_00_u03b1_436_, lean_object* v_msg_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v_msg_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___boxed(lean_object* v_00_u03b1_444_, lean_object* v_msg_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(v_00_u03b1_444_, v_msg_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg(lean_object* v_range_455_, lean_object* v_b_456_, lean_object* v_i_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_stop_461_; lean_object* v_step_462_; uint8_t v___x_463_; 
v_stop_461_ = lean_ctor_get(v_range_455_, 1);
v_step_462_ = lean_ctor_get(v_range_455_, 2);
v___x_463_ = lean_nat_dec_lt(v_i_457_, v_stop_461_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; 
lean_dec(v_i_457_);
v___x_464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_464_, 0, v_b_456_);
return v___x_464_;
}
else
{
lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_465_ = ((lean_object*)(lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___closed__1));
v___x_466_ = l_Lean_Core_mkFreshUserName(v___x_465_, v___y_458_, v___y_459_);
if (lean_obj_tag(v___x_466_) == 0)
{
lean_object* v_a_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v_a_467_ = lean_ctor_get(v___x_466_, 0);
lean_inc(v_a_467_);
lean_dec_ref_known(v___x_466_, 1);
v___x_468_ = lean_array_push(v_b_456_, v_a_467_);
v___x_469_ = lean_nat_add(v_i_457_, v_step_462_);
lean_dec(v_i_457_);
v_b_456_ = v___x_468_;
v_i_457_ = v___x_469_;
goto _start;
}
else
{
lean_object* v_a_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_478_; 
lean_dec(v_i_457_);
lean_dec_ref(v_b_456_);
v_a_471_ = lean_ctor_get(v___x_466_, 0);
v_isSharedCheck_478_ = !lean_is_exclusive(v___x_466_);
if (v_isSharedCheck_478_ == 0)
{
v___x_473_ = v___x_466_;
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_a_471_);
lean_dec(v___x_466_);
v___x_473_ = lean_box(0);
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
v_resetjp_472_:
{
lean_object* v___x_476_; 
if (v_isShared_474_ == 0)
{
v___x_476_ = v___x_473_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_a_471_);
v___x_476_ = v_reuseFailAlloc_477_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
return v___x_476_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg___boxed(lean_object* v_range_479_, lean_object* v_b_480_, lean_object* v_i_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg(v_range_479_, v_b_480_, v_i_481_, v___y_482_, v___y_483_);
lean_dec(v___y_483_);
lean_dec_ref(v___y_482_);
lean_dec_ref(v_range_479_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders(lean_object* v_className_488_, lean_object* v_arity_489_, lean_object* v_indVal_490_, lean_object* v_a_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v___x_498_; 
lean_inc_ref(v_indVal_490_);
v___x_498_ = l_Lean_Elab_Deriving_mkInductArgNames(v_indVal_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v___x_500_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc_n(v_a_499_, 2);
lean_dec_ref_known(v___x_498_, 1);
v___x_500_ = l_Lean_Elab_Deriving_mkImplicitBinders(v_a_499_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_);
if (lean_obj_tag(v___x_500_) == 0)
{
lean_object* v_a_501_; lean_object* v___x_502_; 
v_a_501_ = lean_ctor_get(v___x_500_, 0);
lean_inc(v_a_501_);
lean_dec_ref_known(v___x_500_, 1);
lean_inc(v_a_499_);
lean_inc_ref(v_indVal_490_);
v___x_502_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v_indVal_490_, v_a_499_, v_a_495_);
if (lean_obj_tag(v___x_502_) == 0)
{
lean_object* v_a_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; 
v_a_503_ = lean_ctor_get(v___x_502_, 0);
lean_inc(v_a_503_);
lean_dec_ref_known(v___x_502_, 1);
v___x_504_ = lean_unsigned_to_nat(0u);
v___x_505_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___closed__0));
v___x_506_ = lean_unsigned_to_nat(1u);
v___x_507_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_507_, 0, v___x_504_);
lean_ctor_set(v___x_507_, 1, v_arity_489_);
lean_ctor_set(v___x_507_, 2, v___x_506_);
v___x_508_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg(v___x_507_, v___x_505_, v___x_504_, v_a_495_, v_a_496_);
lean_dec_ref_known(v___x_507_, 3);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; lean_object* v___x_510_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref_known(v___x_508_, 1);
lean_inc(v_a_499_);
v___x_510_ = l_Lean_Elab_Deriving_mkInstImplicitBinders(v_className_488_, v_indVal_490_, v_a_499_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_);
if (lean_obj_tag(v___x_510_) == 0)
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_520_; 
v_a_511_ = lean_ctor_get(v___x_510_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_510_);
if (v_isSharedCheck_520_ == 0)
{
v___x_513_ = v___x_510_;
v_isShared_514_ = v_isSharedCheck_520_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_510_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_520_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_518_; 
v___x_515_ = l_Array_append___redArg(v_a_501_, v_a_511_);
lean_dec(v_a_511_);
v___x_516_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
lean_ctor_set(v___x_516_, 1, v_a_499_);
lean_ctor_set(v___x_516_, 2, v_a_509_);
lean_ctor_set(v___x_516_, 3, v_a_503_);
if (v_isShared_514_ == 0)
{
lean_ctor_set(v___x_513_, 0, v___x_516_);
v___x_518_ = v___x_513_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v___x_516_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
}
else
{
lean_object* v_a_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_528_; 
lean_dec(v_a_509_);
lean_dec(v_a_503_);
lean_dec(v_a_501_);
lean_dec(v_a_499_);
v_a_521_ = lean_ctor_get(v___x_510_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_510_);
if (v_isSharedCheck_528_ == 0)
{
v___x_523_ = v___x_510_;
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_a_521_);
lean_dec(v___x_510_);
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
else
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
lean_dec(v_a_503_);
lean_dec(v_a_501_);
lean_dec(v_a_499_);
lean_dec_ref(v_indVal_490_);
lean_dec(v_className_488_);
v_a_529_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_508_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_508_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
else
{
lean_object* v_a_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
lean_dec(v_a_501_);
lean_dec(v_a_499_);
lean_dec_ref(v_indVal_490_);
lean_dec(v_arity_489_);
lean_dec(v_className_488_);
v_a_537_ = lean_ctor_get(v___x_502_, 0);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_502_);
if (v_isSharedCheck_544_ == 0)
{
v___x_539_ = v___x_502_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_a_537_);
lean_dec(v___x_502_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v_a_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
else
{
lean_object* v_a_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_552_; 
lean_dec(v_a_499_);
lean_dec_ref(v_indVal_490_);
lean_dec(v_arity_489_);
lean_dec(v_className_488_);
v_a_545_ = lean_ctor_get(v___x_500_, 0);
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_500_);
if (v_isSharedCheck_552_ == 0)
{
v___x_547_ = v___x_500_;
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_a_545_);
lean_dec(v___x_500_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_550_; 
if (v_isShared_548_ == 0)
{
v___x_550_ = v___x_547_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v_a_545_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
else
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_560_; 
lean_dec_ref(v_indVal_490_);
lean_dec(v_arity_489_);
lean_dec(v_className_488_);
v_a_553_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_560_ == 0)
{
v___x_555_ = v___x_498_;
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_498_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_558_; 
if (v_isShared_556_ == 0)
{
v___x_558_ = v___x_555_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v_a_553_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders___boxed(lean_object* v_className_561_, lean_object* v_arity_562_, lean_object* v_indVal_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders(v_className_561_, v_arity_562_, v_indVal_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_, v_a_569_);
lean_dec(v_a_569_);
lean_dec_ref(v_a_568_);
lean_dec(v_a_567_);
lean_dec_ref(v_a_566_);
lean_dec(v_a_565_);
lean_dec_ref(v_a_564_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0(lean_object* v_range_572_, lean_object* v_b_573_, lean_object* v_i_574_, lean_object* v_hs_575_, lean_object* v_hl_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___redArg(v_range_572_, v_b_573_, v_i_574_, v___y_581_, v___y_582_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0___boxed(lean_object* v_range_585_, lean_object* v_b_586_, lean_object* v_i_587_, lean_object* v_hs_588_, lean_object* v_hl_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders_spec__0(v_range_585_, v_b_586_, v_i_587_, v_hs_588_, v_hl_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
lean_dec_ref(v_range_585_);
return v_res_597_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = l_Array_mkArray0(lean_box(0));
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1(lean_object* v___f_611_, lean_object* v___x_612_, lean_object* v___x_613_, lean_object* v___x_614_, lean_object* v___x_615_, lean_object* v___x_616_, lean_object* v___x_617_, lean_object* v_b_618_, lean_object* v_____r_619_, lean_object* v_val_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_){
_start:
{
lean_object* v___x_628_; 
lean_inc(v___y_626_);
lean_inc_ref(v___y_625_);
lean_inc(v___y_624_);
lean_inc_ref(v___y_623_);
lean_inc(v___y_622_);
lean_inc_ref(v___y_621_);
v___x_628_ = lean_apply_7(v___f_611_, v___y_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_, lean_box(0));
if (lean_obj_tag(v___x_628_) == 0)
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_673_; 
v_a_629_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_673_ == 0)
{
v___x_631_ = v___x_628_;
v_isShared_632_ = v_isSharedCheck_673_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_628_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_673_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_671_; 
v___x_633_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__0));
v___x_634_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__1));
lean_inc_ref_n(v___x_613_, 7);
lean_inc_ref_n(v___x_612_, 7);
v___x_635_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_633_, v___x_634_);
v___x_636_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__2));
v___x_637_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_633_, v___x_636_);
v___x_638_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
lean_inc(v___x_614_);
lean_inc_n(v_a_629_, 12);
v___x_639_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_639_, 0, v_a_629_);
lean_ctor_set(v___x_639_, 1, v___x_614_);
lean_ctor_set(v___x_639_, 2, v___x_638_);
lean_inc_ref_n(v___x_639_, 12);
v___x_640_ = l_Lean_Syntax_node7(v_a_629_, v___x_637_, v___x_639_, v___x_639_, v___x_639_, v___x_639_, v___x_639_, v___x_639_, v___x_639_);
v___x_641_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__4));
v___x_642_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_633_, v___x_641_);
v___x_643_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__5));
lean_inc_ref(v___x_615_);
v___x_644_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_615_, v___x_643_);
v___x_645_ = l_Lean_Syntax_node1(v_a_629_, v___x_644_, v___x_639_);
v___x_646_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_646_, 0, v_a_629_);
lean_ctor_set(v___x_646_, 1, v___x_641_);
v___x_647_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__6));
v___x_648_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_633_, v___x_647_);
v___x_649_ = l_Array_append___redArg(v___x_638_, v___x_616_);
v___x_650_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_650_, 0, v_a_629_);
lean_ctor_set(v___x_650_, 1, v___x_614_);
lean_ctor_set(v___x_650_, 2, v___x_649_);
v___x_651_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__7));
v___x_652_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_615_, v___x_651_);
v___x_653_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8));
v___x_654_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_654_, 0, v_a_629_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = l_Lean_Syntax_node2(v_a_629_, v___x_652_, v___x_654_, v___x_617_);
v___x_656_ = l_Lean_Syntax_node2(v_a_629_, v___x_648_, v___x_650_, v___x_655_);
v___x_657_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__9));
v___x_658_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_633_, v___x_657_);
v___x_659_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10));
v___x_660_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_660_, 0, v_a_629_);
lean_ctor_set(v___x_660_, 1, v___x_659_);
v___x_661_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__11));
v___x_662_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__12));
v___x_663_ = l_Lean_Name_mkStr4(v___x_612_, v___x_613_, v___x_661_, v___x_662_);
v___x_664_ = l_Lean_Syntax_node2(v_a_629_, v___x_663_, v___x_639_, v___x_639_);
v___x_665_ = l_Lean_Syntax_node4(v_a_629_, v___x_658_, v___x_660_, v_val_620_, v___x_664_, v___x_639_);
v___x_666_ = l_Lean_Syntax_node6(v_a_629_, v___x_642_, v___x_645_, v___x_646_, v___x_639_, v___x_639_, v___x_656_, v___x_665_);
v___x_667_ = l_Lean_Syntax_node2(v_a_629_, v___x_635_, v___x_640_, v___x_666_);
v___x_668_ = lean_array_push(v_b_618_, v___x_667_);
v___x_669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_669_, 0, v___x_668_);
if (v_isShared_632_ == 0)
{
lean_ctor_set(v___x_631_, 0, v___x_669_);
v___x_671_ = v___x_631_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_669_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
else
{
lean_object* v_a_674_; lean_object* v___x_676_; uint8_t v_isShared_677_; uint8_t v_isSharedCheck_681_; 
lean_dec(v_val_620_);
lean_dec_ref(v_b_618_);
lean_dec(v___x_617_);
lean_dec_ref(v___x_615_);
lean_dec(v___x_614_);
lean_dec_ref(v___x_613_);
lean_dec_ref(v___x_612_);
v_a_674_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_681_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_681_ == 0)
{
v___x_676_ = v___x_628_;
v_isShared_677_ = v_isSharedCheck_681_;
goto v_resetjp_675_;
}
else
{
lean_inc(v_a_674_);
lean_dec(v___x_628_);
v___x_676_ = lean_box(0);
v_isShared_677_ = v_isSharedCheck_681_;
goto v_resetjp_675_;
}
v_resetjp_675_:
{
lean_object* v___x_679_; 
if (v_isShared_677_ == 0)
{
v___x_679_ = v___x_676_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_680_; 
v_reuseFailAlloc_680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_680_, 0, v_a_674_);
v___x_679_ = v_reuseFailAlloc_680_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
return v___x_679_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___boxed(lean_object** _args){
lean_object* v___f_682_ = _args[0];
lean_object* v___x_683_ = _args[1];
lean_object* v___x_684_ = _args[2];
lean_object* v___x_685_ = _args[3];
lean_object* v___x_686_ = _args[4];
lean_object* v___x_687_ = _args[5];
lean_object* v___x_688_ = _args[6];
lean_object* v_b_689_ = _args[7];
lean_object* v_____r_690_ = _args[8];
lean_object* v_val_691_ = _args[9];
lean_object* v___y_692_ = _args[10];
lean_object* v___y_693_ = _args[11];
lean_object* v___y_694_ = _args[12];
lean_object* v___y_695_ = _args[13];
lean_object* v___y_696_ = _args[14];
lean_object* v___y_697_ = _args[15];
lean_object* v___y_698_ = _args[16];
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1(v___f_682_, v___x_683_, v___x_684_, v___x_685_, v___x_686_, v___x_687_, v___x_688_, v_b_689_, v_____r_690_, v_val_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_, v___y_696_, v___y_697_);
lean_dec(v___y_697_);
lean_dec_ref(v___y_696_);
lean_dec(v___y_695_);
lean_dec_ref(v___y_694_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
lean_dec_ref(v___x_687_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0(lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
lean_object* v_ref_707_; uint8_t v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v_ref_707_ = lean_ctor_get(v___y_704_, 5);
v___x_708_ = 0;
v___x_709_ = l_Lean_SourceInfo_fromRef(v_ref_707_, v___x_708_);
v___x_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0___boxed(lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0(v___y_711_, v___y_712_, v___y_713_, v___y_714_, v___y_715_, v___y_716_);
lean_dec(v___y_716_);
lean_dec_ref(v___y_715_);
lean_dec(v___y_714_);
lean_dec_ref(v___y_713_);
lean_dec(v___y_712_);
lean_dec_ref(v___y_711_);
return v_res_718_;
}
}
LEAN_EXPORT uint8_t lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0(lean_object* v_a_719_, lean_object* v_as_720_, size_t v_i_721_, size_t v_stop_722_){
_start:
{
uint8_t v___x_723_; 
v___x_723_ = lean_usize_dec_eq(v_i_721_, v_stop_722_);
if (v___x_723_ == 0)
{
lean_object* v___x_724_; uint8_t v___x_725_; 
v___x_724_ = lean_array_uget_borrowed(v_as_720_, v_i_721_);
v___x_725_ = lean_name_eq(v_a_719_, v___x_724_);
if (v___x_725_ == 0)
{
size_t v___x_726_; size_t v___x_727_; 
v___x_726_ = ((size_t)1ULL);
v___x_727_ = lean_usize_add(v_i_721_, v___x_726_);
v_i_721_ = v___x_727_;
goto _start;
}
else
{
return v___x_725_;
}
}
else
{
uint8_t v___x_729_; 
v___x_729_ = 0;
return v___x_729_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0___boxed(lean_object* v_a_730_, lean_object* v_as_731_, lean_object* v_i_732_, lean_object* v_stop_733_){
_start:
{
size_t v_i_boxed_734_; size_t v_stop_boxed_735_; uint8_t v_res_736_; lean_object* v_r_737_; 
v_i_boxed_734_ = lean_unbox_usize(v_i_732_);
lean_dec(v_i_732_);
v_stop_boxed_735_ = lean_unbox_usize(v_stop_733_);
lean_dec(v_stop_733_);
v_res_736_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0(v_a_730_, v_as_731_, v_i_boxed_734_, v_stop_boxed_735_);
lean_dec_ref(v_as_731_);
lean_dec(v_a_730_);
v_r_737_ = lean_box(v_res_736_);
return v_r_737_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0(lean_object* v_as_738_, lean_object* v_a_739_){
_start:
{
lean_object* v___x_740_; lean_object* v___x_741_; uint8_t v___x_742_; 
v___x_740_ = lean_unsigned_to_nat(0u);
v___x_741_ = lean_array_get_size(v_as_738_);
v___x_742_ = lean_nat_dec_lt(v___x_740_, v___x_741_);
if (v___x_742_ == 0)
{
return v___x_742_;
}
else
{
if (v___x_742_ == 0)
{
return v___x_742_;
}
else
{
size_t v___x_743_; size_t v___x_744_; uint8_t v___x_745_; 
v___x_743_ = ((size_t)0ULL);
v___x_744_ = lean_usize_of_nat(v___x_741_);
v___x_745_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0_spec__0(v_a_739_, v_as_738_, v___x_743_, v___x_744_);
return v___x_745_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0___boxed(lean_object* v_as_746_, lean_object* v_a_747_){
_start:
{
uint8_t v_res_748_; lean_object* v_r_749_; 
v_res_748_ = lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0(v_as_746_, v_a_747_);
lean_dec(v_a_747_);
lean_dec_ref(v_as_746_);
v_r_749_ = lean_box(v_res_748_);
return v_r_749_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11(void){
_start:
{
lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_769_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10));
v___x_770_ = l_Lean_mkCIdent(v___x_769_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg(lean_object* v_upperBound_782_, lean_object* v___x_783_, lean_object* v_typeNames_784_, lean_object* v_ctx_785_, uint8_t v_useAnonCtor_786_, lean_object* v_a_787_, lean_object* v_b_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_){
_start:
{
lean_object* v_a_797_; lean_object* v___y_802_; uint8_t v___x_821_; 
v___x_821_ = lean_nat_dec_lt(v_a_787_, v_upperBound_782_);
if (v___x_821_ == 0)
{
lean_object* v___x_822_; 
lean_dec(v_a_787_);
v___x_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_822_, 0, v_b_788_);
return v___x_822_;
}
else
{
lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v_toConstantVal_825_; lean_object* v_name_826_; uint8_t v___x_827_; 
v___x_823_ = l_Lean_instInhabitedInductiveVal_default;
v___x_824_ = lean_array_get_borrowed(v___x_823_, v___x_783_, v_a_787_);
v_toConstantVal_825_ = lean_ctor_get(v___x_824_, 0);
v_name_826_ = lean_ctor_get(v_toConstantVal_825_, 0);
v___x_827_ = lp_plausible_Array_contains___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__0(v_typeNames_784_, v_name_826_);
if (v___x_827_ == 0)
{
v_a_797_ = v_b_788_;
goto v___jp_796_;
}
else
{
lean_object* v___x_828_; 
lean_inc(v___x_824_);
v___x_828_ = l_Lean_Elab_Deriving_mkInductArgNames(v___x_824_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_828_) == 0)
{
lean_object* v_a_829_; lean_object* v___x_830_; 
v_a_829_ = lean_ctor_get(v___x_828_, 0);
lean_inc_n(v_a_829_, 2);
lean_dec_ref_known(v___x_828_, 1);
v___x_830_ = l_Lean_Elab_Deriving_mkImplicitBinders(v_a_829_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_object* v_a_831_; lean_object* v___x_832_; lean_object* v___x_833_; 
v_a_831_ = lean_ctor_get(v___x_830_, 0);
lean_inc(v_a_831_);
lean_dec_ref_known(v___x_830_, 1);
v___x_832_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2));
lean_inc(v_a_829_);
lean_inc(v___x_824_);
v___x_833_ = l_Lean_Elab_Deriving_mkInstImplicitBinders(v___x_832_, v___x_824_, v_a_829_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_833_) == 0)
{
lean_object* v_a_834_; lean_object* v___x_835_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
lean_inc(v_a_834_);
lean_dec_ref_known(v___x_833_, 1);
lean_inc(v___x_824_);
v___x_835_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v___x_824_, v_a_829_, v___y_793_);
if (lean_obj_tag(v___x_835_) == 0)
{
lean_object* v_a_836_; lean_object* v_auxFunNames_837_; lean_object* v_ref_838_; lean_object* v___f_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; uint8_t v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; 
v_a_836_ = lean_ctor_get(v___x_835_, 0);
lean_inc(v_a_836_);
lean_dec_ref_known(v___x_835_, 1);
v_auxFunNames_837_ = lean_ctor_get(v_ctx_785_, 2);
v_ref_838_ = lean_ctor_get(v___y_793_, 5);
v___f_839_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__3));
v___x_840_ = lean_box(0);
v___x_841_ = lean_array_get_borrowed(v___x_840_, v_auxFunNames_837_, v_a_787_);
v___x_842_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__4));
v___x_843_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__5));
v___x_844_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__6));
v___x_845_ = l_Array_append___redArg(v_a_831_, v_a_834_);
lean_dec(v_a_834_);
v___x_846_ = 0;
v___x_847_ = l_Lean_SourceInfo_fromRef(v_ref_838_, v___x_846_);
v___x_848_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
v___x_849_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__11);
v___x_850_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
lean_inc(v___x_847_);
v___x_851_ = l_Lean_Syntax_node1(v___x_847_, v___x_850_, v_a_836_);
v___x_852_ = l_Lean_Syntax_node2(v___x_847_, v___x_848_, v___x_849_, v___x_851_);
lean_inc(v___x_841_);
v___x_853_ = l_Lean_mkIdent(v___x_841_);
if (v_useAnonCtor_786_ == 0)
{
lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_854_ = lean_box(0);
v___x_855_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1(v___f_839_, v___x_842_, v___x_843_, v___x_850_, v___x_844_, v___x_845_, v___x_852_, v_b_788_, v___x_854_, v___x_853_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
lean_dec_ref(v___x_845_);
v___y_802_ = v___x_855_;
goto v___jp_801_;
}
else
{
lean_object* v___x_856_; 
v___x_856_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__0(v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_856_) == 0)
{
lean_object* v_a_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v_a_857_ = lean_ctor_get(v___x_856_, 0);
lean_inc_n(v_a_857_, 4);
lean_dec_ref_known(v___x_856_, 1);
v___x_858_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__15));
v___x_859_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__16));
v___x_860_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_860_, 0, v_a_857_);
lean_ctor_set(v___x_860_, 1, v___x_859_);
v___x_861_ = l_Lean_Syntax_node1(v_a_857_, v___x_850_, v___x_853_);
v___x_862_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__17));
v___x_863_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_863_, 0, v_a_857_);
lean_ctor_set(v___x_863_, 1, v___x_862_);
v___x_864_ = l_Lean_Syntax_node3(v_a_857_, v___x_858_, v___x_860_, v___x_861_, v___x_863_);
v___x_865_ = lean_box(0);
v___x_866_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1(v___f_839_, v___x_842_, v___x_843_, v___x_850_, v___x_844_, v___x_845_, v___x_852_, v_b_788_, v___x_865_, v___x_864_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
lean_dec_ref(v___x_845_);
v___y_802_ = v___x_866_;
goto v___jp_801_;
}
else
{
lean_object* v_a_867_; lean_object* v___x_869_; uint8_t v_isShared_870_; uint8_t v_isSharedCheck_874_; 
lean_dec(v___x_853_);
lean_dec(v___x_852_);
lean_dec_ref(v___x_845_);
lean_dec_ref(v_b_788_);
lean_dec(v_a_787_);
v_a_867_ = lean_ctor_get(v___x_856_, 0);
v_isSharedCheck_874_ = !lean_is_exclusive(v___x_856_);
if (v_isSharedCheck_874_ == 0)
{
v___x_869_ = v___x_856_;
v_isShared_870_ = v_isSharedCheck_874_;
goto v_resetjp_868_;
}
else
{
lean_inc(v_a_867_);
lean_dec(v___x_856_);
v___x_869_ = lean_box(0);
v_isShared_870_ = v_isSharedCheck_874_;
goto v_resetjp_868_;
}
v_resetjp_868_:
{
lean_object* v___x_872_; 
if (v_isShared_870_ == 0)
{
v___x_872_ = v___x_869_;
goto v_reusejp_871_;
}
else
{
lean_object* v_reuseFailAlloc_873_; 
v_reuseFailAlloc_873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_873_, 0, v_a_867_);
v___x_872_ = v_reuseFailAlloc_873_;
goto v_reusejp_871_;
}
v_reusejp_871_:
{
return v___x_872_;
}
}
}
}
}
else
{
lean_object* v_a_875_; lean_object* v___x_877_; uint8_t v_isShared_878_; uint8_t v_isSharedCheck_882_; 
lean_dec(v_a_834_);
lean_dec(v_a_831_);
lean_dec_ref(v_b_788_);
lean_dec(v_a_787_);
v_a_875_ = lean_ctor_get(v___x_835_, 0);
v_isSharedCheck_882_ = !lean_is_exclusive(v___x_835_);
if (v_isSharedCheck_882_ == 0)
{
v___x_877_ = v___x_835_;
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
else
{
lean_inc(v_a_875_);
lean_dec(v___x_835_);
v___x_877_ = lean_box(0);
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
v_resetjp_876_:
{
lean_object* v___x_880_; 
if (v_isShared_878_ == 0)
{
v___x_880_ = v___x_877_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v_a_875_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
}
}
else
{
lean_object* v_a_883_; lean_object* v___x_885_; uint8_t v_isShared_886_; uint8_t v_isSharedCheck_890_; 
lean_dec(v_a_831_);
lean_dec(v_a_829_);
lean_dec_ref(v_b_788_);
lean_dec(v_a_787_);
v_a_883_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_890_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_890_ == 0)
{
v___x_885_ = v___x_833_;
v_isShared_886_ = v_isSharedCheck_890_;
goto v_resetjp_884_;
}
else
{
lean_inc(v_a_883_);
lean_dec(v___x_833_);
v___x_885_ = lean_box(0);
v_isShared_886_ = v_isSharedCheck_890_;
goto v_resetjp_884_;
}
v_resetjp_884_:
{
lean_object* v___x_888_; 
if (v_isShared_886_ == 0)
{
v___x_888_ = v___x_885_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_889_; 
v_reuseFailAlloc_889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_889_, 0, v_a_883_);
v___x_888_ = v_reuseFailAlloc_889_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
return v___x_888_;
}
}
}
}
else
{
lean_dec(v_a_829_);
lean_dec_ref(v_b_788_);
lean_dec(v_a_787_);
return v___x_830_;
}
}
else
{
lean_object* v_a_891_; lean_object* v___x_893_; uint8_t v_isShared_894_; uint8_t v_isSharedCheck_898_; 
lean_dec_ref(v_b_788_);
lean_dec(v_a_787_);
v_a_891_ = lean_ctor_get(v___x_828_, 0);
v_isSharedCheck_898_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_898_ == 0)
{
v___x_893_ = v___x_828_;
v_isShared_894_ = v_isSharedCheck_898_;
goto v_resetjp_892_;
}
else
{
lean_inc(v_a_891_);
lean_dec(v___x_828_);
v___x_893_ = lean_box(0);
v_isShared_894_ = v_isSharedCheck_898_;
goto v_resetjp_892_;
}
v_resetjp_892_:
{
lean_object* v___x_896_; 
if (v_isShared_894_ == 0)
{
v___x_896_ = v___x_893_;
goto v_reusejp_895_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_897_, 0, v_a_891_);
v___x_896_ = v_reuseFailAlloc_897_;
goto v_reusejp_895_;
}
v_reusejp_895_:
{
return v___x_896_;
}
}
}
}
}
v___jp_796_:
{
lean_object* v___x_798_; lean_object* v___x_799_; 
v___x_798_ = lean_unsigned_to_nat(1u);
v___x_799_ = lean_nat_add(v_a_787_, v___x_798_);
lean_dec(v_a_787_);
v_a_787_ = v___x_799_;
v_b_788_ = v_a_797_;
goto _start;
}
v___jp_801_:
{
if (lean_obj_tag(v___y_802_) == 0)
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_812_; 
v_a_803_ = lean_ctor_get(v___y_802_, 0);
v_isSharedCheck_812_ = !lean_is_exclusive(v___y_802_);
if (v_isSharedCheck_812_ == 0)
{
v___x_805_ = v___y_802_;
v_isShared_806_ = v_isSharedCheck_812_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___y_802_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_812_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
if (lean_obj_tag(v_a_803_) == 0)
{
lean_object* v_a_807_; lean_object* v___x_809_; 
lean_dec(v_a_787_);
v_a_807_ = lean_ctor_get(v_a_803_, 0);
lean_inc(v_a_807_);
lean_dec_ref_known(v_a_803_, 1);
if (v_isShared_806_ == 0)
{
lean_ctor_set(v___x_805_, 0, v_a_807_);
v___x_809_ = v___x_805_;
goto v_reusejp_808_;
}
else
{
lean_object* v_reuseFailAlloc_810_; 
v_reuseFailAlloc_810_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_810_, 0, v_a_807_);
v___x_809_ = v_reuseFailAlloc_810_;
goto v_reusejp_808_;
}
v_reusejp_808_:
{
return v___x_809_;
}
}
else
{
lean_object* v_a_811_; 
lean_del_object(v___x_805_);
v_a_811_ = lean_ctor_get(v_a_803_, 0);
lean_inc(v_a_811_);
lean_dec_ref_known(v_a_803_, 1);
v_a_797_ = v_a_811_;
goto v___jp_796_;
}
}
}
else
{
lean_object* v_a_813_; lean_object* v___x_815_; uint8_t v_isShared_816_; uint8_t v_isSharedCheck_820_; 
lean_dec(v_a_787_);
v_a_813_ = lean_ctor_get(v___y_802_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v___y_802_);
if (v_isSharedCheck_820_ == 0)
{
v___x_815_ = v___y_802_;
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
else
{
lean_inc(v_a_813_);
lean_dec(v___y_802_);
v___x_815_ = lean_box(0);
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
v_resetjp_814_:
{
lean_object* v___x_818_; 
if (v_isShared_816_ == 0)
{
v___x_818_ = v___x_815_;
goto v_reusejp_817_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v_a_813_);
v___x_818_ = v_reuseFailAlloc_819_;
goto v_reusejp_817_;
}
v_reusejp_817_:
{
return v___x_818_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___boxed(lean_object* v_upperBound_899_, lean_object* v___x_900_, lean_object* v_typeNames_901_, lean_object* v_ctx_902_, lean_object* v_useAnonCtor_903_, lean_object* v_a_904_, lean_object* v_b_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_){
_start:
{
uint8_t v_useAnonCtor_boxed_913_; lean_object* v_res_914_; 
v_useAnonCtor_boxed_913_ = lean_unbox(v_useAnonCtor_903_);
v_res_914_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg(v_upperBound_899_, v___x_900_, v_typeNames_901_, v_ctx_902_, v_useAnonCtor_boxed_913_, v_a_904_, v_b_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_, v___y_910_, v___y_911_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec(v___y_909_);
lean_dec_ref(v___y_908_);
lean_dec(v___y_907_);
lean_dec_ref(v___y_906_);
lean_dec_ref(v_ctx_902_);
lean_dec_ref(v_typeNames_901_);
lean_dec_ref(v___x_900_);
lean_dec(v_upperBound_899_);
return v_res_914_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds(lean_object* v_ctx_917_, lean_object* v_typeNames_918_, uint8_t v_useAnonCtor_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_){
_start:
{
lean_object* v_typeInfos_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v_instances_930_; lean_object* v___x_931_; 
v_typeInfos_927_ = lean_ctor_get(v_ctx_917_, 1);
v___x_928_ = lean_unsigned_to_nat(0u);
v___x_929_ = lean_array_get_size(v_typeInfos_927_);
v_instances_930_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0));
v___x_931_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg(v___x_929_, v_typeInfos_927_, v_typeNames_918_, v_ctx_917_, v_useAnonCtor_919_, v___x_928_, v_instances_930_, v_a_920_, v_a_921_, v_a_922_, v_a_923_, v_a_924_, v_a_925_);
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___boxed(lean_object* v_ctx_932_, lean_object* v_typeNames_933_, lean_object* v_useAnonCtor_934_, lean_object* v_a_935_, lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v_a_938_, lean_object* v_a_939_, lean_object* v_a_940_, lean_object* v_a_941_){
_start:
{
uint8_t v_useAnonCtor_boxed_942_; lean_object* v_res_943_; 
v_useAnonCtor_boxed_942_ = lean_unbox(v_useAnonCtor_934_);
v_res_943_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds(v_ctx_932_, v_typeNames_933_, v_useAnonCtor_boxed_942_, v_a_935_, v_a_936_, v_a_937_, v_a_938_, v_a_939_, v_a_940_);
lean_dec(v_a_940_);
lean_dec_ref(v_a_939_);
lean_dec(v_a_938_);
lean_dec_ref(v_a_937_);
lean_dec(v_a_936_);
lean_dec_ref(v_a_935_);
lean_dec_ref(v_typeNames_933_);
lean_dec_ref(v_ctx_932_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1(lean_object* v_upperBound_944_, lean_object* v___x_945_, lean_object* v_typeNames_946_, lean_object* v_ctx_947_, uint8_t v_useAnonCtor_948_, lean_object* v_inst_949_, lean_object* v_R_950_, lean_object* v_a_951_, lean_object* v_b_952_, lean_object* v_c_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v___x_961_; 
v___x_961_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg(v_upperBound_944_, v___x_945_, v_typeNames_946_, v_ctx_947_, v_useAnonCtor_948_, v_a_951_, v_b_952_, v___y_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_, v___y_959_);
return v___x_961_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___boxed(lean_object** _args){
lean_object* v_upperBound_962_ = _args[0];
lean_object* v___x_963_ = _args[1];
lean_object* v_typeNames_964_ = _args[2];
lean_object* v_ctx_965_ = _args[3];
lean_object* v_useAnonCtor_966_ = _args[4];
lean_object* v_inst_967_ = _args[5];
lean_object* v_R_968_ = _args[6];
lean_object* v_a_969_ = _args[7];
lean_object* v_b_970_ = _args[8];
lean_object* v_c_971_ = _args[9];
lean_object* v___y_972_ = _args[10];
lean_object* v___y_973_ = _args[11];
lean_object* v___y_974_ = _args[12];
lean_object* v___y_975_ = _args[13];
lean_object* v___y_976_ = _args[14];
lean_object* v___y_977_ = _args[15];
lean_object* v___y_978_ = _args[16];
_start:
{
uint8_t v_useAnonCtor_boxed_979_; lean_object* v_res_980_; 
v_useAnonCtor_boxed_979_ = lean_unbox(v_useAnonCtor_966_);
v_res_980_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1(v_upperBound_962_, v___x_963_, v_typeNames_964_, v_ctx_965_, v_useAnonCtor_boxed_979_, v_inst_967_, v_R_968_, v_a_969_, v_b_970_, v_c_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec_ref(v_ctx_965_);
lean_dec_ref(v_typeNames_964_);
lean_dec_ref(v___x_963_);
lean_dec(v_upperBound_962_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader(lean_object* v_indVal_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_){
_start:
{
lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_989_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2));
v___x_990_ = lean_unsigned_to_nat(1u);
v___x_991_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkHeaderWithOnlyImplicitBinders(v___x_989_, v___x_990_, v_indVal_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_, v_a_986_, v_a_987_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader___boxed(lean_object* v_indVal_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_, lean_object* v_a_999_){
_start:
{
lean_object* v_res_1000_; 
v_res_1000_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader(v_indVal_992_, v_a_993_, v_a_994_, v_a_995_, v_a_996_, v_a_997_, v_a_998_);
lean_dec(v_a_998_);
lean_dec_ref(v_a_997_);
lean_dec(v_a_996_);
lean_dec_ref(v_a_995_);
lean_dec(v_a_994_);
lean_dec_ref(v_a_993_);
return v_res_1000_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_){
_start:
{
lean_object* v_ref_1008_; uint8_t v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; 
v_ref_1008_ = lean_ctor_get(v___y_1005_, 5);
v___x_1009_ = 0;
v___x_1010_ = l_Lean_SourceInfo_fromRef(v_ref_1008_, v___x_1009_);
v___x_1011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1011_, 0, v___x_1010_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0___boxed(lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_){
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v___y_1012_, v___y_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
return v_res_1019_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0(void){
_start:
{
lean_object* v___x_1020_; lean_object* v___x_1021_; 
v___x_1020_ = lean_box(1);
v___x_1021_ = l_Lean_MessageData_ofFormat(v___x_1020_);
return v___x_1021_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3(void){
_start:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1025_ = ((lean_object*)(lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__2));
v___x_1026_ = l_Lean_MessageData_ofFormat(v___x_1025_);
return v___x_1026_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13(lean_object* v_x_1027_, lean_object* v_x_1028_){
_start:
{
if (lean_obj_tag(v_x_1028_) == 0)
{
return v_x_1027_;
}
else
{
lean_object* v_head_1029_; lean_object* v_tail_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1052_; 
v_head_1029_ = lean_ctor_get(v_x_1028_, 0);
v_tail_1030_ = lean_ctor_get(v_x_1028_, 1);
v_isSharedCheck_1052_ = !lean_is_exclusive(v_x_1028_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1032_ = v_x_1028_;
v_isShared_1033_ = v_isSharedCheck_1052_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_tail_1030_);
lean_inc(v_head_1029_);
lean_dec(v_x_1028_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1052_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v_before_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1050_; 
v_before_1034_ = lean_ctor_get(v_head_1029_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v_head_1029_);
if (v_isSharedCheck_1050_ == 0)
{
lean_object* v_unused_1051_; 
v_unused_1051_ = lean_ctor_get(v_head_1029_, 1);
lean_dec(v_unused_1051_);
v___x_1036_ = v_head_1029_;
v_isShared_1037_ = v_isSharedCheck_1050_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_before_1034_);
lean_dec(v_head_1029_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1050_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v___x_1038_; lean_object* v___x_1040_; 
v___x_1038_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0);
if (v_isShared_1037_ == 0)
{
lean_ctor_set_tag(v___x_1036_, 7);
lean_ctor_set(v___x_1036_, 1, v___x_1038_);
lean_ctor_set(v___x_1036_, 0, v_x_1027_);
v___x_1040_ = v___x_1036_;
goto v_reusejp_1039_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v_x_1027_);
lean_ctor_set(v_reuseFailAlloc_1049_, 1, v___x_1038_);
v___x_1040_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1039_;
}
v_reusejp_1039_:
{
lean_object* v___x_1041_; lean_object* v___x_1043_; 
v___x_1041_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__3);
if (v_isShared_1033_ == 0)
{
lean_ctor_set_tag(v___x_1032_, 7);
lean_ctor_set(v___x_1032_, 1, v___x_1041_);
lean_ctor_set(v___x_1032_, 0, v___x_1040_);
v___x_1043_ = v___x_1032_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v___x_1040_);
lean_ctor_set(v_reuseFailAlloc_1048_, 1, v___x_1041_);
v___x_1043_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1044_ = l_Lean_MessageData_ofSyntax(v_before_1034_);
v___x_1045_ = l_Lean_indentD(v___x_1044_);
v___x_1046_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1043_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
v_x_1027_ = v___x_1046_;
v_x_1028_ = v_tail_1030_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12(lean_object* v_opts_1053_, lean_object* v_opt_1054_){
_start:
{
lean_object* v_name_1055_; lean_object* v_defValue_1056_; lean_object* v_map_1057_; lean_object* v___x_1058_; 
v_name_1055_ = lean_ctor_get(v_opt_1054_, 0);
v_defValue_1056_ = lean_ctor_get(v_opt_1054_, 1);
v_map_1057_ = lean_ctor_get(v_opts_1053_, 0);
v___x_1058_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1057_, v_name_1055_);
if (lean_obj_tag(v___x_1058_) == 0)
{
uint8_t v___x_1059_; 
v___x_1059_ = lean_unbox(v_defValue_1056_);
return v___x_1059_;
}
else
{
lean_object* v_val_1060_; 
v_val_1060_ = lean_ctor_get(v___x_1058_, 0);
lean_inc(v_val_1060_);
lean_dec_ref_known(v___x_1058_, 1);
if (lean_obj_tag(v_val_1060_) == 1)
{
uint8_t v_v_1061_; 
v_v_1061_ = lean_ctor_get_uint8(v_val_1060_, 0);
lean_dec_ref_known(v_val_1060_, 0);
return v_v_1061_;
}
else
{
uint8_t v___x_1062_; 
lean_dec(v_val_1060_);
v___x_1062_ = lean_unbox(v_defValue_1056_);
return v___x_1062_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12___boxed(lean_object* v_opts_1063_, lean_object* v_opt_1064_){
_start:
{
uint8_t v_res_1065_; lean_object* v_r_1066_; 
v_res_1065_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12(v_opts_1063_, v_opt_1064_);
lean_dec_ref(v_opt_1064_);
lean_dec_ref(v_opts_1063_);
v_r_1066_ = lean_box(v_res_1065_);
return v_r_1066_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2(void){
_start:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___x_1070_ = ((lean_object*)(lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__1));
v___x_1071_ = l_Lean_MessageData_ofFormat(v___x_1070_);
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg(lean_object* v_msgData_1072_, lean_object* v_macroStack_1073_, lean_object* v___y_1074_){
_start:
{
lean_object* v_options_1076_; lean_object* v___x_1077_; uint8_t v___x_1078_; 
v_options_1076_ = lean_ctor_get(v___y_1074_, 2);
v___x_1077_ = l_Lean_Elab_pp_macroStack;
v___x_1078_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12(v_options_1076_, v___x_1077_);
if (v___x_1078_ == 0)
{
lean_object* v___x_1079_; 
lean_dec(v_macroStack_1073_);
v___x_1079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1079_, 0, v_msgData_1072_);
return v___x_1079_;
}
else
{
if (lean_obj_tag(v_macroStack_1073_) == 0)
{
lean_object* v___x_1080_; 
v___x_1080_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1080_, 0, v_msgData_1072_);
return v___x_1080_;
}
else
{
lean_object* v_head_1081_; lean_object* v_after_1082_; lean_object* v___x_1084_; uint8_t v_isShared_1085_; uint8_t v_isSharedCheck_1097_; 
v_head_1081_ = lean_ctor_get(v_macroStack_1073_, 0);
lean_inc(v_head_1081_);
v_after_1082_ = lean_ctor_get(v_head_1081_, 1);
v_isSharedCheck_1097_ = !lean_is_exclusive(v_head_1081_);
if (v_isSharedCheck_1097_ == 0)
{
lean_object* v_unused_1098_; 
v_unused_1098_ = lean_ctor_get(v_head_1081_, 0);
lean_dec(v_unused_1098_);
v___x_1084_ = v_head_1081_;
v_isShared_1085_ = v_isSharedCheck_1097_;
goto v_resetjp_1083_;
}
else
{
lean_inc(v_after_1082_);
lean_dec(v_head_1081_);
v___x_1084_ = lean_box(0);
v_isShared_1085_ = v_isSharedCheck_1097_;
goto v_resetjp_1083_;
}
v_resetjp_1083_:
{
lean_object* v___x_1086_; lean_object* v___x_1088_; 
v___x_1086_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0);
if (v_isShared_1085_ == 0)
{
lean_ctor_set_tag(v___x_1084_, 7);
lean_ctor_set(v___x_1084_, 1, v___x_1086_);
lean_ctor_set(v___x_1084_, 0, v_msgData_1072_);
v___x_1088_ = v___x_1084_;
goto v_reusejp_1087_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v_msgData_1072_);
lean_ctor_set(v_reuseFailAlloc_1096_, 1, v___x_1086_);
v___x_1088_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1087_;
}
v_reusejp_1087_:
{
lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v_msgData_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1089_ = lean_obj_once(&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2, &lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2_once, _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2);
v___x_1090_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1090_, 0, v___x_1088_);
lean_ctor_set(v___x_1090_, 1, v___x_1089_);
v___x_1091_ = l_Lean_MessageData_ofSyntax(v_after_1082_);
v___x_1092_ = l_Lean_indentD(v___x_1091_);
v_msgData_1093_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1093_, 0, v___x_1090_);
lean_ctor_set(v_msgData_1093_, 1, v___x_1092_);
v___x_1094_ = lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13(v_msgData_1093_, v_macroStack_1073_);
v___x_1095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1095_, 0, v___x_1094_);
return v___x_1095_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___boxed(lean_object* v_msgData_1099_, lean_object* v_macroStack_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg(v_msgData_1099_, v_macroStack_1100_, v___y_1101_);
lean_dec_ref(v___y_1101_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(lean_object* v_msg_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
lean_object* v_ref_1112_; lean_object* v___x_1113_; lean_object* v_a_1114_; lean_object* v_macroStack_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v_a_1118_; lean_object* v___x_1120_; uint8_t v_isShared_1121_; uint8_t v_isSharedCheck_1126_; 
v_ref_1112_ = lean_ctor_get(v___y_1109_, 5);
v___x_1113_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_1104_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_);
v_a_1114_ = lean_ctor_get(v___x_1113_, 0);
lean_inc(v_a_1114_);
lean_dec_ref(v___x_1113_);
v_macroStack_1115_ = lean_ctor_get(v___y_1105_, 1);
v___x_1116_ = l_Lean_Elab_getBetterRef(v_ref_1112_, v_macroStack_1115_);
lean_inc(v_macroStack_1115_);
v___x_1117_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg(v_a_1114_, v_macroStack_1115_, v___y_1109_);
v_a_1118_ = lean_ctor_get(v___x_1117_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1117_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1120_ = v___x_1117_;
v_isShared_1121_ = v_isSharedCheck_1126_;
goto v_resetjp_1119_;
}
else
{
lean_inc(v_a_1118_);
lean_dec(v___x_1117_);
v___x_1120_ = lean_box(0);
v_isShared_1121_ = v_isSharedCheck_1126_;
goto v_resetjp_1119_;
}
v_resetjp_1119_:
{
lean_object* v___x_1122_; lean_object* v___x_1124_; 
v___x_1122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1116_);
lean_ctor_set(v___x_1122_, 1, v_a_1118_);
if (v_isShared_1121_ == 0)
{
lean_ctor_set_tag(v___x_1120_, 1);
lean_ctor_set(v___x_1120_, 0, v___x_1122_);
v___x_1124_ = v___x_1120_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v___x_1122_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg___boxed(lean_object* v_msg_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_){
_start:
{
lean_object* v_res_1135_; 
v_res_1135_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(v_msg_1127_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_);
lean_dec(v___y_1133_);
lean_dec_ref(v___y_1132_);
lean_dec(v___y_1131_);
lean_dec_ref(v___y_1130_);
lean_dec(v___y_1129_);
lean_dec_ref(v___y_1128_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2(size_t v_sz_1136_, size_t v_i_1137_, lean_object* v_bs_1138_){
_start:
{
uint8_t v___x_1139_; 
v___x_1139_ = lean_usize_dec_lt(v_i_1137_, v_sz_1136_);
if (v___x_1139_ == 0)
{
return v_bs_1138_;
}
else
{
lean_object* v_v_1140_; lean_object* v___x_1141_; lean_object* v_bs_x27_1142_; size_t v___x_1143_; size_t v___x_1144_; lean_object* v___x_1145_; 
v_v_1140_ = lean_array_uget(v_bs_1138_, v_i_1137_);
v___x_1141_ = lean_unsigned_to_nat(0u);
v_bs_x27_1142_ = lean_array_uset(v_bs_1138_, v_i_1137_, v___x_1141_);
v___x_1143_ = ((size_t)1ULL);
v___x_1144_ = lean_usize_add(v_i_1137_, v___x_1143_);
v___x_1145_ = lean_array_uset(v_bs_x27_1142_, v_i_1137_, v_v_1140_);
v_i_1137_ = v___x_1144_;
v_bs_1138_ = v___x_1145_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2___boxed(lean_object* v_sz_1147_, lean_object* v_i_1148_, lean_object* v_bs_1149_){
_start:
{
size_t v_sz_boxed_1150_; size_t v_i_boxed_1151_; lean_object* v_res_1152_; 
v_sz_boxed_1150_ = lean_unbox_usize(v_sz_1147_);
lean_dec(v_sz_1147_);
v_i_boxed_1151_ = lean_unbox_usize(v_i_1148_);
lean_dec(v_i_1148_);
v_res_1152_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2(v_sz_boxed_1150_, v_i_boxed_1151_, v_bs_1149_);
return v_res_1152_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3(lean_object* v___x_1159_, size_t v_sz_1160_, size_t v_i_1161_, lean_object* v_bs_1162_){
_start:
{
uint8_t v___x_1163_; 
v___x_1163_ = lean_usize_dec_lt(v_i_1161_, v_sz_1160_);
if (v___x_1163_ == 0)
{
lean_dec(v___x_1159_);
return v_bs_1162_;
}
else
{
lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v_v_1166_; lean_object* v___x_1167_; lean_object* v_bs_x27_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; size_t v___x_1172_; size_t v___x_1173_; lean_object* v___x_1174_; 
v___x_1164_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_1165_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
v_v_1166_ = lean_array_uget(v_bs_1162_, v_i_1161_);
v___x_1167_ = lean_unsigned_to_nat(0u);
v_bs_x27_1168_ = lean_array_uset(v_bs_1162_, v_i_1161_, v___x_1167_);
v___x_1169_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___closed__1));
lean_inc_n(v___x_1159_, 2);
v___x_1170_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1159_);
lean_ctor_set(v___x_1170_, 1, v___x_1164_);
lean_ctor_set(v___x_1170_, 2, v___x_1165_);
v___x_1171_ = l_Lean_Syntax_node2(v___x_1159_, v___x_1169_, v_v_1166_, v___x_1170_);
v___x_1172_ = ((size_t)1ULL);
v___x_1173_ = lean_usize_add(v_i_1161_, v___x_1172_);
v___x_1174_ = lean_array_uset(v_bs_x27_1168_, v_i_1161_, v___x_1171_);
v_i_1161_ = v___x_1173_;
v_bs_1162_ = v___x_1174_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3___boxed(lean_object* v___x_1176_, lean_object* v_sz_1177_, lean_object* v_i_1178_, lean_object* v_bs_1179_){
_start:
{
size_t v_sz_boxed_1180_; size_t v_i_boxed_1181_; lean_object* v_res_1182_; 
v_sz_boxed_1180_ = lean_unbox_usize(v_sz_1177_);
lean_dec(v_sz_1177_);
v_i_boxed_1181_ = lean_unbox_usize(v_i_1178_);
lean_dec(v_i_1178_);
v_res_1182_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3(v___x_1176_, v_sz_boxed_1180_, v_i_boxed_1181_, v_bs_1179_);
return v_res_1182_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12(void){
_start:
{
lean_object* v___x_1214_; lean_object* v___x_1215_; 
v___x_1214_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__11));
v___x_1215_ = l_Lean_mkIdent(v___x_1214_);
return v___x_1215_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15(void){
_start:
{
lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1219_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__14));
v___x_1220_ = l_Lean_mkIdent(v___x_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg(lean_object* v_targetTypeName_1221_, lean_object* v___x_1222_, lean_object* v___x_1223_, lean_object* v_as_1224_, size_t v_sz_1225_, size_t v_i_1226_, lean_object* v_b_1227_, lean_object* v___y_1228_){
_start:
{
uint8_t v___x_1230_; 
v___x_1230_ = lean_usize_dec_lt(v_i_1226_, v_sz_1225_);
if (v___x_1230_ == 0)
{
lean_object* v___x_1231_; 
lean_dec(v___x_1223_);
v___x_1231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1231_, 0, v_b_1227_);
return v___x_1231_;
}
else
{
lean_object* v_a_1232_; lean_object* v_fst_1233_; lean_object* v_snd_1234_; lean_object* v___x_1236_; uint8_t v_isShared_1237_; uint8_t v_isSharedCheck_1302_; 
v_a_1232_ = lean_array_uget(v_as_1224_, v_i_1226_);
v_fst_1233_ = lean_ctor_get(v_a_1232_, 0);
v_snd_1234_ = lean_ctor_get(v_a_1232_, 1);
v_isSharedCheck_1302_ = !lean_is_exclusive(v_a_1232_);
if (v_isSharedCheck_1302_ == 0)
{
v___x_1236_ = v_a_1232_;
v_isShared_1237_ = v_isSharedCheck_1302_;
goto v_resetjp_1235_;
}
else
{
lean_inc(v_snd_1234_);
lean_inc(v_fst_1233_);
lean_dec(v_a_1232_);
v___x_1236_ = lean_box(0);
v_isShared_1237_ = v_isSharedCheck_1302_;
goto v_resetjp_1235_;
}
v_resetjp_1235_:
{
lean_object* v_fst_1238_; lean_object* v_snd_1239_; lean_object* v___x_1241_; uint8_t v_isShared_1242_; uint8_t v_isSharedCheck_1301_; 
v_fst_1238_ = lean_ctor_get(v_b_1227_, 0);
v_snd_1239_ = lean_ctor_get(v_b_1227_, 1);
v_isSharedCheck_1301_ = !lean_is_exclusive(v_b_1227_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1241_ = v_b_1227_;
v_isShared_1242_ = v_isSharedCheck_1301_;
goto v_resetjp_1240_;
}
else
{
lean_inc(v_snd_1239_);
lean_inc(v_fst_1238_);
lean_dec(v_b_1227_);
v___x_1241_ = lean_box(0);
v_isShared_1242_ = v_isSharedCheck_1301_;
goto v_resetjp_1240_;
}
v_resetjp_1240_:
{
lean_object* v_bindExpr_1244_; uint8_t v_ctorIsRecursive_1245_; uint8_t v___x_1254_; 
v___x_1254_ = l_Lean_Expr_isAppOf(v_snd_1234_, v_targetTypeName_1221_);
lean_dec(v_snd_1234_);
if (v___x_1254_ == 0)
{
lean_object* v_ref_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1260_; 
v_ref_1255_ = lean_ctor_get(v___y_1228_, 5);
v___x_1256_ = l_Lean_SourceInfo_fromRef(v_ref_1255_, v___x_1254_);
v___x_1257_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1));
v___x_1258_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2));
lean_inc(v___x_1256_);
if (v_isShared_1237_ == 0)
{
lean_ctor_set_tag(v___x_1236_, 2);
lean_ctor_set(v___x_1236_, 1, v___x_1258_);
lean_ctor_set(v___x_1236_, 0, v___x_1256_);
v___x_1260_ = v___x_1236_;
goto v_reusejp_1259_;
}
else
{
lean_object* v_reuseFailAlloc_1275_; 
v_reuseFailAlloc_1275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1275_, 0, v___x_1256_);
lean_ctor_set(v_reuseFailAlloc_1275_, 1, v___x_1258_);
v___x_1260_ = v_reuseFailAlloc_1275_;
goto v_reusejp_1259_;
}
v_reusejp_1259_:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; uint8_t v___x_1274_; 
v___x_1261_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_1262_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
lean_inc_n(v___x_1256_, 5);
v___x_1263_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1256_);
lean_ctor_set(v___x_1263_, 1, v___x_1261_);
lean_ctor_set(v___x_1263_, 2, v___x_1262_);
v___x_1264_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4));
lean_inc_ref_n(v___x_1263_, 2);
v___x_1265_ = l_Lean_Syntax_node1(v___x_1256_, v___x_1264_, v___x_1263_);
v___x_1266_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6));
v___x_1267_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__7));
v___x_1268_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1268_, 0, v___x_1256_);
lean_ctor_set(v___x_1268_, 1, v___x_1267_);
v___x_1269_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9));
v___x_1270_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__12);
v___x_1271_ = l_Lean_Syntax_node1(v___x_1256_, v___x_1269_, v___x_1270_);
v___x_1272_ = l_Lean_Syntax_node4(v___x_1256_, v___x_1266_, v_fst_1233_, v___x_1263_, v___x_1268_, v___x_1271_);
v___x_1273_ = l_Lean_Syntax_node4(v___x_1256_, v___x_1257_, v___x_1260_, v___x_1263_, v___x_1265_, v___x_1272_);
v___x_1274_ = lean_unbox(v_snd_1239_);
lean_dec(v_snd_1239_);
v_bindExpr_1244_ = v___x_1273_;
v_ctorIsRecursive_1245_ = v___x_1274_;
goto v___jp_1243_;
}
}
else
{
lean_object* v_ref_1276_; lean_object* v___x_1277_; uint8_t v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1284_; 
lean_dec(v_snd_1239_);
v_ref_1276_ = lean_ctor_get(v___y_1228_, 5);
v___x_1277_ = lean_unsigned_to_nat(0u);
v___x_1278_ = lean_nat_dec_eq(v___x_1222_, v___x_1277_);
v___x_1279_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15);
v___x_1280_ = l_Lean_SourceInfo_fromRef(v_ref_1276_, v___x_1278_);
v___x_1281_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__1));
v___x_1282_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2));
lean_inc(v___x_1280_);
if (v_isShared_1237_ == 0)
{
lean_ctor_set_tag(v___x_1236_, 2);
lean_ctor_set(v___x_1236_, 1, v___x_1282_);
lean_ctor_set(v___x_1236_, 0, v___x_1280_);
v___x_1284_ = v___x_1236_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v___x_1280_);
lean_ctor_set(v_reuseFailAlloc_1300_, 1, v___x_1282_);
v___x_1284_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; 
v___x_1285_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_1286_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
lean_inc_n(v___x_1280_, 7);
v___x_1287_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1280_);
lean_ctor_set(v___x_1287_, 1, v___x_1285_);
lean_ctor_set(v___x_1287_, 2, v___x_1286_);
v___x_1288_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__4));
lean_inc_ref_n(v___x_1287_, 2);
v___x_1289_ = l_Lean_Syntax_node1(v___x_1280_, v___x_1288_, v___x_1287_);
v___x_1290_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__6));
v___x_1291_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__7));
v___x_1292_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1280_);
lean_ctor_set(v___x_1292_, 1, v___x_1291_);
v___x_1293_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__9));
v___x_1294_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
lean_inc(v___x_1223_);
v___x_1295_ = l_Lean_Syntax_node1(v___x_1280_, v___x_1285_, v___x_1223_);
v___x_1296_ = l_Lean_Syntax_node2(v___x_1280_, v___x_1294_, v___x_1279_, v___x_1295_);
v___x_1297_ = l_Lean_Syntax_node1(v___x_1280_, v___x_1293_, v___x_1296_);
v___x_1298_ = l_Lean_Syntax_node4(v___x_1280_, v___x_1290_, v_fst_1233_, v___x_1287_, v___x_1292_, v___x_1297_);
v___x_1299_ = l_Lean_Syntax_node4(v___x_1280_, v___x_1281_, v___x_1284_, v___x_1287_, v___x_1289_, v___x_1298_);
v_bindExpr_1244_ = v___x_1299_;
v_ctorIsRecursive_1245_ = v___x_1254_;
goto v___jp_1243_;
}
}
v___jp_1243_:
{
lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1249_; 
v___x_1246_ = lean_array_push(v_fst_1238_, v_bindExpr_1244_);
v___x_1247_ = lean_box(v_ctorIsRecursive_1245_);
if (v_isShared_1242_ == 0)
{
lean_ctor_set(v___x_1241_, 1, v___x_1247_);
lean_ctor_set(v___x_1241_, 0, v___x_1246_);
v___x_1249_ = v___x_1241_;
goto v_reusejp_1248_;
}
else
{
lean_object* v_reuseFailAlloc_1253_; 
v_reuseFailAlloc_1253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1253_, 0, v___x_1246_);
lean_ctor_set(v_reuseFailAlloc_1253_, 1, v___x_1247_);
v___x_1249_ = v_reuseFailAlloc_1253_;
goto v_reusejp_1248_;
}
v_reusejp_1248_:
{
size_t v___x_1250_; size_t v___x_1251_; 
v___x_1250_ = ((size_t)1ULL);
v___x_1251_ = lean_usize_add(v_i_1226_, v___x_1250_);
v_i_1226_ = v___x_1251_;
v_b_1227_ = v___x_1249_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___boxed(lean_object* v_targetTypeName_1303_, lean_object* v___x_1304_, lean_object* v___x_1305_, lean_object* v_as_1306_, lean_object* v_sz_1307_, lean_object* v_i_1308_, lean_object* v_b_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
size_t v_sz_boxed_1312_; size_t v_i_boxed_1313_; lean_object* v_res_1314_; 
v_sz_boxed_1312_ = lean_unbox_usize(v_sz_1307_);
lean_dec(v_sz_1307_);
v_i_boxed_1313_ = lean_unbox_usize(v_i_1308_);
lean_dec(v_i_1308_);
v_res_1314_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg(v_targetTypeName_1303_, v___x_1304_, v___x_1305_, v_as_1306_, v_sz_boxed_1312_, v_i_boxed_1313_, v_b_1309_, v___y_1310_);
lean_dec_ref(v___y_1310_);
lean_dec_ref(v_as_1306_);
lean_dec(v___x_1304_);
lean_dec(v_targetTypeName_1303_);
return v_res_1314_;
}
}
static lean_object* _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11(void){
_start:
{
lean_object* v___x_1339_; lean_object* v___x_1340_; 
v___x_1339_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__10));
v___x_1340_ = l_String_toRawSubstring_x27(v___x_1339_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0(lean_object* v___x_1436_, uint8_t v___x_1437_, lean_object* v___x_1438_, lean_object* v_targetTypeName_1439_, lean_object* v___x_1440_, lean_object* v___x_1441_, size_t v___x_1442_, lean_object* v___x_1443_, lean_object* v___x_1444_, lean_object* v_x_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_){
_start:
{
lean_object* v___x_1453_; lean_object* v___x_1454_; size_t v_sz_1455_; lean_object* v___x_1456_; 
v___x_1453_ = lean_box(v___x_1437_);
v___x_1454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1454_, 0, v___x_1436_);
lean_ctor_set(v___x_1454_, 1, v___x_1453_);
v_sz_1455_ = lean_array_size(v___x_1438_);
v___x_1456_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg(v_targetTypeName_1439_, v___x_1440_, v___x_1441_, v___x_1438_, v_sz_1455_, v___x_1442_, v___x_1454_, v___y_1450_);
if (lean_obj_tag(v___x_1456_) == 0)
{
lean_object* v_a_1457_; lean_object* v___x_1459_; uint8_t v_isShared_1460_; uint8_t v_isSharedCheck_1516_; 
v_a_1457_ = lean_ctor_get(v___x_1456_, 0);
v_isSharedCheck_1516_ = !lean_is_exclusive(v___x_1456_);
if (v_isSharedCheck_1516_ == 0)
{
v___x_1459_ = v___x_1456_;
v_isShared_1460_ = v_isSharedCheck_1516_;
goto v_resetjp_1458_;
}
else
{
lean_inc(v_a_1457_);
lean_dec(v___x_1456_);
v___x_1459_ = lean_box(0);
v_isShared_1460_ = v_isSharedCheck_1516_;
goto v_resetjp_1458_;
}
v_resetjp_1458_:
{
lean_object* v_fst_1461_; lean_object* v_snd_1462_; lean_object* v___x_1464_; uint8_t v_isShared_1465_; uint8_t v_isSharedCheck_1515_; 
v_fst_1461_ = lean_ctor_get(v_a_1457_, 0);
v_snd_1462_ = lean_ctor_get(v_a_1457_, 1);
v_isSharedCheck_1515_ = !lean_is_exclusive(v_a_1457_);
if (v_isSharedCheck_1515_ == 0)
{
v___x_1464_ = v_a_1457_;
v_isShared_1465_ = v_isSharedCheck_1515_;
goto v_resetjp_1463_;
}
else
{
lean_inc(v_snd_1462_);
lean_inc(v_fst_1461_);
lean_dec(v_a_1457_);
v___x_1464_ = lean_box(0);
v_isShared_1465_ = v_isSharedCheck_1515_;
goto v_resetjp_1463_;
}
v_resetjp_1463_:
{
lean_object* v_ref_1466_; lean_object* v_quotContext_1467_; lean_object* v_currMacroScope_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; size_t v_sz_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; size_t v_sz_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1510_; 
v_ref_1466_ = lean_ctor_get(v___y_1450_, 5);
v_quotContext_1467_ = lean_ctor_get(v___y_1450_, 10);
v_currMacroScope_1468_ = lean_ctor_get(v___y_1450_, 11);
v___x_1469_ = l_Lean_SourceInfo_fromRef(v_ref_1466_, v___x_1437_);
v___x_1470_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__1));
v___x_1471_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__2));
lean_inc_n(v___x_1469_, 15);
v___x_1472_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1472_, 0, v___x_1469_);
lean_ctor_set(v___x_1472_, 1, v___x_1471_);
v___x_1473_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_1474_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
v___x_1475_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
v_sz_1476_ = lean_array_size(v___x_1443_);
v___x_1477_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__2(v_sz_1476_, v___x_1442_, v___x_1443_);
v___x_1478_ = l_Array_append___redArg(v___x_1475_, v___x_1477_);
lean_dec_ref(v___x_1477_);
v___x_1479_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1479_, 0, v___x_1469_);
lean_ctor_set(v___x_1479_, 1, v___x_1473_);
lean_ctor_set(v___x_1479_, 2, v___x_1478_);
v___x_1480_ = l_Lean_Syntax_node2(v___x_1469_, v___x_1474_, v___x_1444_, v___x_1479_);
v___x_1481_ = l_Lean_Syntax_node1(v___x_1469_, v___x_1473_, v___x_1480_);
v___x_1482_ = l_Lean_Syntax_node2(v___x_1469_, v___x_1470_, v___x_1472_, v___x_1481_);
v___x_1483_ = lean_array_push(v_fst_1461_, v___x_1482_);
v___x_1484_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4));
v___x_1485_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6));
v___x_1486_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7));
v___x_1487_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1469_);
lean_ctor_set(v___x_1487_, 1, v___x_1486_);
v___x_1488_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9));
v___x_1489_ = lean_obj_once(&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11, &lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11_once, _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11);
v___x_1490_ = lean_box(0);
lean_inc(v_currMacroScope_1468_);
lean_inc(v_quotContext_1467_);
v___x_1491_ = l_Lean_addMacroScope(v_quotContext_1467_, v___x_1490_, v_currMacroScope_1468_);
v___x_1492_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43));
v___x_1493_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1493_, 0, v___x_1469_);
lean_ctor_set(v___x_1493_, 1, v___x_1489_);
lean_ctor_set(v___x_1493_, 2, v___x_1491_);
lean_ctor_set(v___x_1493_, 3, v___x_1492_);
v___x_1494_ = l_Lean_Syntax_node1(v___x_1469_, v___x_1488_, v___x_1493_);
v___x_1495_ = l_Lean_Syntax_node2(v___x_1469_, v___x_1485_, v___x_1487_, v___x_1494_);
v___x_1496_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__44));
v___x_1497_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__45));
v___x_1498_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1469_);
lean_ctor_set(v___x_1498_, 1, v___x_1496_);
v___x_1499_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__47));
v_sz_1500_ = lean_array_size(v___x_1483_);
v___x_1501_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__3(v___x_1469_, v_sz_1500_, v___x_1442_, v___x_1483_);
v___x_1502_ = l_Array_append___redArg(v___x_1475_, v___x_1501_);
lean_dec_ref(v___x_1501_);
v___x_1503_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1503_, 0, v___x_1469_);
lean_ctor_set(v___x_1503_, 1, v___x_1473_);
lean_ctor_set(v___x_1503_, 2, v___x_1502_);
v___x_1504_ = l_Lean_Syntax_node1(v___x_1469_, v___x_1499_, v___x_1503_);
v___x_1505_ = l_Lean_Syntax_node2(v___x_1469_, v___x_1497_, v___x_1498_, v___x_1504_);
v___x_1506_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48));
v___x_1507_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1507_, 0, v___x_1469_);
lean_ctor_set(v___x_1507_, 1, v___x_1506_);
v___x_1508_ = l_Lean_Syntax_node3(v___x_1469_, v___x_1484_, v___x_1495_, v___x_1505_, v___x_1507_);
if (v_isShared_1465_ == 0)
{
lean_ctor_set(v___x_1464_, 0, v___x_1508_);
v___x_1510_ = v___x_1464_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1514_; 
v_reuseFailAlloc_1514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1514_, 0, v___x_1508_);
lean_ctor_set(v_reuseFailAlloc_1514_, 1, v_snd_1462_);
v___x_1510_ = v_reuseFailAlloc_1514_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
lean_object* v___x_1512_; 
if (v_isShared_1460_ == 0)
{
lean_ctor_set(v___x_1459_, 0, v___x_1510_);
v___x_1512_ = v___x_1459_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v___x_1510_);
v___x_1512_ = v_reuseFailAlloc_1513_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
return v___x_1512_;
}
}
}
}
}
else
{
lean_object* v_a_1517_; lean_object* v___x_1519_; uint8_t v_isShared_1520_; uint8_t v_isSharedCheck_1524_; 
lean_dec(v___x_1444_);
lean_dec_ref(v___x_1443_);
v_a_1517_ = lean_ctor_get(v___x_1456_, 0);
v_isSharedCheck_1524_ = !lean_is_exclusive(v___x_1456_);
if (v_isSharedCheck_1524_ == 0)
{
v___x_1519_ = v___x_1456_;
v_isShared_1520_ = v_isSharedCheck_1524_;
goto v_resetjp_1518_;
}
else
{
lean_inc(v_a_1517_);
lean_dec(v___x_1456_);
v___x_1519_ = lean_box(0);
v_isShared_1520_ = v_isSharedCheck_1524_;
goto v_resetjp_1518_;
}
v_resetjp_1518_:
{
lean_object* v___x_1522_; 
if (v_isShared_1520_ == 0)
{
v___x_1522_ = v___x_1519_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1523_; 
v_reuseFailAlloc_1523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1523_, 0, v_a_1517_);
v___x_1522_ = v_reuseFailAlloc_1523_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
return v___x_1522_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_1525_ = _args[0];
lean_object* v___x_1526_ = _args[1];
lean_object* v___x_1527_ = _args[2];
lean_object* v_targetTypeName_1528_ = _args[3];
lean_object* v___x_1529_ = _args[4];
lean_object* v___x_1530_ = _args[5];
lean_object* v___x_1531_ = _args[6];
lean_object* v___x_1532_ = _args[7];
lean_object* v___x_1533_ = _args[8];
lean_object* v_x_1534_ = _args[9];
lean_object* v___y_1535_ = _args[10];
lean_object* v___y_1536_ = _args[11];
lean_object* v___y_1537_ = _args[12];
lean_object* v___y_1538_ = _args[13];
lean_object* v___y_1539_ = _args[14];
lean_object* v___y_1540_ = _args[15];
lean_object* v___y_1541_ = _args[16];
_start:
{
uint8_t v___x_49186__boxed_1542_; size_t v___x_49190__boxed_1543_; lean_object* v_res_1544_; 
v___x_49186__boxed_1542_ = lean_unbox(v___x_1526_);
v___x_49190__boxed_1543_ = lean_unbox_usize(v___x_1531_);
lean_dec(v___x_1531_);
v_res_1544_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0(v___x_1525_, v___x_49186__boxed_1542_, v___x_1527_, v_targetTypeName_1528_, v___x_1529_, v___x_1530_, v___x_49190__boxed_1543_, v___x_1532_, v___x_1533_, v_x_1534_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_, v___y_1539_, v___y_1540_);
lean_dec(v___y_1540_);
lean_dec_ref(v___y_1539_);
lean_dec(v___y_1538_);
lean_dec_ref(v___y_1537_);
lean_dec(v___y_1536_);
lean_dec_ref(v___y_1535_);
lean_dec_ref(v_x_1534_);
lean_dec(v___x_1529_);
lean_dec(v_targetTypeName_1528_);
lean_dec_ref(v___x_1527_);
return v_res_1544_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0(lean_object* v_snd_1545_, lean_object* v_x_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_){
_start:
{
lean_object* v___x_1554_; 
v___x_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1554_, 0, v_snd_1545_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0___boxed(lean_object* v_snd_1555_, lean_object* v_x_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_){
_start:
{
lean_object* v_res_1564_; 
v_res_1564_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0(v_snd_1555_, v_x_1556_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_);
lean_dec(v___y_1562_);
lean_dec_ref(v___y_1561_);
lean_dec(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1558_);
lean_dec_ref(v___y_1557_);
lean_dec_ref(v_x_1556_);
return v_res_1564_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4(size_t v_sz_1565_, size_t v_i_1566_, lean_object* v_bs_1567_){
_start:
{
uint8_t v___x_1568_; 
v___x_1568_ = lean_usize_dec_lt(v_i_1566_, v_sz_1565_);
if (v___x_1568_ == 0)
{
return v_bs_1567_;
}
else
{
lean_object* v_v_1569_; lean_object* v_fst_1570_; lean_object* v_snd_1571_; lean_object* v___x_1573_; uint8_t v_isShared_1574_; uint8_t v_isSharedCheck_1585_; 
v_v_1569_ = lean_array_uget(v_bs_1567_, v_i_1566_);
v_fst_1570_ = lean_ctor_get(v_v_1569_, 0);
v_snd_1571_ = lean_ctor_get(v_v_1569_, 1);
v_isSharedCheck_1585_ = !lean_is_exclusive(v_v_1569_);
if (v_isSharedCheck_1585_ == 0)
{
v___x_1573_ = v_v_1569_;
v_isShared_1574_ = v_isSharedCheck_1585_;
goto v_resetjp_1572_;
}
else
{
lean_inc(v_snd_1571_);
lean_inc(v_fst_1570_);
lean_dec(v_v_1569_);
v___x_1573_ = lean_box(0);
v_isShared_1574_ = v_isSharedCheck_1585_;
goto v_resetjp_1572_;
}
v_resetjp_1572_:
{
lean_object* v___x_1575_; lean_object* v_bs_x27_1576_; lean_object* v___f_1577_; lean_object* v___x_1579_; 
v___x_1575_ = lean_unsigned_to_nat(0u);
v_bs_x27_1576_ = lean_array_uset(v_bs_1567_, v_i_1566_, v___x_1575_);
v___f_1577_ = lean_alloc_closure((void*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1577_, 0, v_snd_1571_);
if (v_isShared_1574_ == 0)
{
lean_ctor_set(v___x_1573_, 1, v___f_1577_);
v___x_1579_ = v___x_1573_;
goto v_reusejp_1578_;
}
else
{
lean_object* v_reuseFailAlloc_1584_; 
v_reuseFailAlloc_1584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1584_, 0, v_fst_1570_);
lean_ctor_set(v_reuseFailAlloc_1584_, 1, v___f_1577_);
v___x_1579_ = v_reuseFailAlloc_1584_;
goto v_reusejp_1578_;
}
v_reusejp_1578_:
{
size_t v___x_1580_; size_t v___x_1581_; lean_object* v___x_1582_; 
v___x_1580_ = ((size_t)1ULL);
v___x_1581_ = lean_usize_add(v_i_1566_, v___x_1580_);
v___x_1582_ = lean_array_uset(v_bs_x27_1576_, v_i_1566_, v___x_1579_);
v_i_1566_ = v___x_1581_;
v_bs_1567_ = v___x_1582_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4___boxed(lean_object* v_sz_1586_, lean_object* v_i_1587_, lean_object* v_bs_1588_){
_start:
{
size_t v_sz_boxed_1589_; size_t v_i_boxed_1590_; lean_object* v_res_1591_; 
v_sz_boxed_1589_ = lean_unbox_usize(v_sz_1586_);
lean_dec(v_sz_1586_);
v_i_boxed_1590_ = lean_unbox_usize(v_i_1587_);
lean_dec(v_i_1587_);
v_res_1591_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4(v_sz_boxed_1589_, v_i_boxed_1590_, v_bs_1588_);
return v_res_1591_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6(size_t v_sz_1592_, size_t v_i_1593_, lean_object* v_bs_1594_){
_start:
{
uint8_t v___x_1595_; 
v___x_1595_ = lean_usize_dec_lt(v_i_1593_, v_sz_1592_);
if (v___x_1595_ == 0)
{
return v_bs_1594_;
}
else
{
lean_object* v_v_1596_; lean_object* v_fst_1597_; lean_object* v_snd_1598_; lean_object* v___x_1600_; uint8_t v_isShared_1601_; uint8_t v_isSharedCheck_1614_; 
v_v_1596_ = lean_array_uget(v_bs_1594_, v_i_1593_);
v_fst_1597_ = lean_ctor_get(v_v_1596_, 0);
v_snd_1598_ = lean_ctor_get(v_v_1596_, 1);
v_isSharedCheck_1614_ = !lean_is_exclusive(v_v_1596_);
if (v_isSharedCheck_1614_ == 0)
{
v___x_1600_ = v_v_1596_;
v_isShared_1601_ = v_isSharedCheck_1614_;
goto v_resetjp_1599_;
}
else
{
lean_inc(v_snd_1598_);
lean_inc(v_fst_1597_);
lean_dec(v_v_1596_);
v___x_1600_ = lean_box(0);
v_isShared_1601_ = v_isSharedCheck_1614_;
goto v_resetjp_1599_;
}
v_resetjp_1599_:
{
lean_object* v___x_1602_; lean_object* v_bs_x27_1603_; uint8_t v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1607_; 
v___x_1602_ = lean_unsigned_to_nat(0u);
v_bs_x27_1603_ = lean_array_uset(v_bs_1594_, v_i_1593_, v___x_1602_);
v___x_1604_ = 0;
v___x_1605_ = lean_box(v___x_1604_);
if (v_isShared_1601_ == 0)
{
lean_ctor_set(v___x_1600_, 0, v___x_1605_);
v___x_1607_ = v___x_1600_;
goto v_reusejp_1606_;
}
else
{
lean_object* v_reuseFailAlloc_1613_; 
v_reuseFailAlloc_1613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1613_, 0, v___x_1605_);
lean_ctor_set(v_reuseFailAlloc_1613_, 1, v_snd_1598_);
v___x_1607_ = v_reuseFailAlloc_1613_;
goto v_reusejp_1606_;
}
v_reusejp_1606_:
{
lean_object* v___x_1608_; size_t v___x_1609_; size_t v___x_1610_; lean_object* v___x_1611_; 
v___x_1608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1608_, 0, v_fst_1597_);
lean_ctor_set(v___x_1608_, 1, v___x_1607_);
v___x_1609_ = ((size_t)1ULL);
v___x_1610_ = lean_usize_add(v_i_1593_, v___x_1609_);
v___x_1611_ = lean_array_uset(v_bs_x27_1603_, v_i_1593_, v___x_1608_);
v_i_1593_ = v___x_1610_;
v_bs_1594_ = v___x_1611_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6___boxed(lean_object* v_sz_1615_, lean_object* v_i_1616_, lean_object* v_bs_1617_){
_start:
{
size_t v_sz_boxed_1618_; size_t v_i_boxed_1619_; lean_object* v_res_1620_; 
v_sz_boxed_1618_ = lean_unbox_usize(v_sz_1615_);
lean_dec(v_sz_1615_);
v_i_boxed_1619_ = lean_unbox_usize(v_i_1616_);
lean_dec(v_i_1616_);
v_res_1620_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6(v_sz_boxed_1618_, v_i_boxed_1619_, v_bs_1617_);
return v_res_1620_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0(lean_object* v___x_1621_, lean_object* v_a_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_, lean_object* v___y_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_){
_start:
{
lean_object* v___x_1630_; lean_object* v___x_48326__overap_1631_; lean_object* v___x_1632_; 
v___x_1630_ = l_Lean_instInhabitedExpr;
v___x_48326__overap_1631_ = l_instInhabitedOfMonad___redArg(v___x_1621_, v___x_1630_);
lean_inc(v___y_1628_);
lean_inc_ref(v___y_1627_);
lean_inc(v___y_1626_);
lean_inc_ref(v___y_1625_);
lean_inc(v___y_1624_);
lean_inc_ref(v___y_1623_);
v___x_1632_ = lean_apply_7(v___x_48326__overap_1631_, v___y_1623_, v___y_1624_, v___y_1625_, v___y_1626_, v___y_1627_, v___y_1628_, lean_box(0));
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0___boxed(lean_object* v___x_1633_, lean_object* v_a_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
lean_object* v_res_1642_; 
v_res_1642_ = lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0(v___x_1633_, v_a_1634_, v___y_1635_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_);
lean_dec(v___y_1640_);
lean_dec_ref(v___y_1639_);
lean_dec(v___y_1638_);
lean_dec_ref(v___y_1637_);
lean_dec(v___y_1636_);
lean_dec_ref(v___y_1635_);
lean_dec_ref(v_a_1634_);
return v_res_1642_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0(lean_object* v_k_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v_b_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_){
_start:
{
lean_object* v___x_1652_; 
lean_inc(v___y_1650_);
lean_inc_ref(v___y_1649_);
lean_inc(v___y_1648_);
lean_inc_ref(v___y_1647_);
lean_inc(v___y_1645_);
lean_inc_ref(v___y_1644_);
v___x_1652_ = lean_apply_8(v_k_1643_, v_b_1646_, v___y_1644_, v___y_1645_, v___y_1647_, v___y_1648_, v___y_1649_, v___y_1650_, lean_box(0));
return v___x_1652_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0___boxed(lean_object* v_k_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v_b_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_){
_start:
{
lean_object* v_res_1662_; 
v_res_1662_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0(v_k_1653_, v___y_1654_, v___y_1655_, v_b_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_);
lean_dec(v___y_1660_);
lean_dec_ref(v___y_1659_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec(v___y_1655_);
lean_dec_ref(v___y_1654_);
return v_res_1662_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg(lean_object* v_name_1663_, uint8_t v_bi_1664_, lean_object* v_type_1665_, lean_object* v_k_1666_, uint8_t v_kind_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_){
_start:
{
lean_object* v___f_1675_; lean_object* v___x_1676_; 
lean_inc(v___y_1669_);
lean_inc_ref(v___y_1668_);
v___f_1675_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_1675_, 0, v_k_1666_);
lean_closure_set(v___f_1675_, 1, v___y_1668_);
lean_closure_set(v___f_1675_, 2, v___y_1669_);
v___x_1676_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1663_, v_bi_1664_, v_type_1665_, v___f_1675_, v_kind_1667_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_);
if (lean_obj_tag(v___x_1676_) == 0)
{
return v___x_1676_;
}
else
{
lean_object* v_a_1677_; lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1684_; 
v_a_1677_ = lean_ctor_get(v___x_1676_, 0);
v_isSharedCheck_1684_ = !lean_is_exclusive(v___x_1676_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1679_ = v___x_1676_;
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
else
{
lean_inc(v_a_1677_);
lean_dec(v___x_1676_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1682_; 
if (v_isShared_1680_ == 0)
{
v___x_1682_ = v___x_1679_;
goto v_reusejp_1681_;
}
else
{
lean_object* v_reuseFailAlloc_1683_; 
v_reuseFailAlloc_1683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1683_, 0, v_a_1677_);
v___x_1682_ = v_reuseFailAlloc_1683_;
goto v_reusejp_1681_;
}
v_reusejp_1681_:
{
return v___x_1682_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg___boxed(lean_object* v_name_1685_, lean_object* v_bi_1686_, lean_object* v_type_1687_, lean_object* v_k_1688_, lean_object* v_kind_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_){
_start:
{
uint8_t v_bi_boxed_1697_; uint8_t v_kind_boxed_1698_; lean_object* v_res_1699_; 
v_bi_boxed_1697_ = lean_unbox(v_bi_1686_);
v_kind_boxed_1698_ = lean_unbox(v_kind_1689_);
v_res_1699_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg(v_name_1685_, v_bi_boxed_1697_, v_type_1687_, v_k_1688_, v_kind_boxed_1698_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_, v___y_1695_);
lean_dec(v___y_1695_);
lean_dec_ref(v___y_1694_);
lean_dec(v___y_1693_);
lean_dec_ref(v___y_1692_);
lean_dec(v___y_1691_);
lean_dec_ref(v___y_1690_);
return v_res_1699_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1___boxed(lean_object* v_acc_1702_, lean_object* v_declInfos_1703_, lean_object* v_k_1704_, lean_object* v_kind_1705_, lean_object* v_x_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_){
_start:
{
uint8_t v_kind_boxed_1714_; lean_object* v_res_1715_; 
v_kind_boxed_1714_ = lean_unbox(v_kind_1705_);
v_res_1715_ = lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1(v_acc_1702_, v_declInfos_1703_, v_k_1704_, v_kind_boxed_1714_, v_x_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
lean_dec(v___y_1712_);
lean_dec_ref(v___y_1711_);
lean_dec(v___y_1710_);
lean_dec_ref(v___y_1709_);
lean_dec(v___y_1708_);
lean_dec_ref(v___y_1707_);
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11(lean_object* v_declInfos_1716_, lean_object* v_k_1717_, uint8_t v_kind_1718_, lean_object* v_acc_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_){
_start:
{
lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v_toApplicative_1729_; lean_object* v___x_1731_; uint8_t v_isShared_1732_; uint8_t v_isSharedCheck_1846_; 
v___x_1727_ = lean_obj_once(&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0, &lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0_once, _init_lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0);
v___x_1728_ = l_StateRefT_x27_instMonad___redArg(v___x_1727_);
v_toApplicative_1729_ = lean_ctor_get(v___x_1728_, 0);
v_isSharedCheck_1846_ = !lean_is_exclusive(v___x_1728_);
if (v_isSharedCheck_1846_ == 0)
{
lean_object* v_unused_1847_; 
v_unused_1847_ = lean_ctor_get(v___x_1728_, 1);
lean_dec(v_unused_1847_);
v___x_1731_ = v___x_1728_;
v_isShared_1732_ = v_isSharedCheck_1846_;
goto v_resetjp_1730_;
}
else
{
lean_inc(v_toApplicative_1729_);
lean_dec(v___x_1728_);
v___x_1731_ = lean_box(0);
v_isShared_1732_ = v_isSharedCheck_1846_;
goto v_resetjp_1730_;
}
v_resetjp_1730_:
{
lean_object* v_toFunctor_1733_; lean_object* v_toSeq_1734_; lean_object* v_toSeqLeft_1735_; lean_object* v_toSeqRight_1736_; lean_object* v___x_1738_; uint8_t v_isShared_1739_; uint8_t v_isSharedCheck_1844_; 
v_toFunctor_1733_ = lean_ctor_get(v_toApplicative_1729_, 0);
v_toSeq_1734_ = lean_ctor_get(v_toApplicative_1729_, 2);
v_toSeqLeft_1735_ = lean_ctor_get(v_toApplicative_1729_, 3);
v_toSeqRight_1736_ = lean_ctor_get(v_toApplicative_1729_, 4);
v_isSharedCheck_1844_ = !lean_is_exclusive(v_toApplicative_1729_);
if (v_isSharedCheck_1844_ == 0)
{
lean_object* v_unused_1845_; 
v_unused_1845_ = lean_ctor_get(v_toApplicative_1729_, 1);
lean_dec(v_unused_1845_);
v___x_1738_ = v_toApplicative_1729_;
v_isShared_1739_ = v_isSharedCheck_1844_;
goto v_resetjp_1737_;
}
else
{
lean_inc(v_toSeqRight_1736_);
lean_inc(v_toSeqLeft_1735_);
lean_inc(v_toSeq_1734_);
lean_inc(v_toFunctor_1733_);
lean_dec(v_toApplicative_1729_);
v___x_1738_ = lean_box(0);
v_isShared_1739_ = v_isSharedCheck_1844_;
goto v_resetjp_1737_;
}
v_resetjp_1737_:
{
lean_object* v___f_1740_; lean_object* v___f_1741_; lean_object* v___f_1742_; lean_object* v___f_1743_; lean_object* v___x_1744_; lean_object* v___f_1745_; lean_object* v___f_1746_; lean_object* v___f_1747_; lean_object* v___x_1749_; 
v___f_1740_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1));
v___f_1741_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_1733_);
v___f_1742_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1742_, 0, v_toFunctor_1733_);
v___f_1743_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1743_, 0, v_toFunctor_1733_);
v___x_1744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1744_, 0, v___f_1742_);
lean_ctor_set(v___x_1744_, 1, v___f_1743_);
v___f_1745_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1745_, 0, v_toSeqRight_1736_);
v___f_1746_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1746_, 0, v_toSeqLeft_1735_);
v___f_1747_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1747_, 0, v_toSeq_1734_);
if (v_isShared_1739_ == 0)
{
lean_ctor_set(v___x_1738_, 4, v___f_1745_);
lean_ctor_set(v___x_1738_, 3, v___f_1746_);
lean_ctor_set(v___x_1738_, 2, v___f_1747_);
lean_ctor_set(v___x_1738_, 1, v___f_1740_);
lean_ctor_set(v___x_1738_, 0, v___x_1744_);
v___x_1749_ = v___x_1738_;
goto v_reusejp_1748_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v___x_1744_);
lean_ctor_set(v_reuseFailAlloc_1843_, 1, v___f_1740_);
lean_ctor_set(v_reuseFailAlloc_1843_, 2, v___f_1747_);
lean_ctor_set(v_reuseFailAlloc_1843_, 3, v___f_1746_);
lean_ctor_set(v_reuseFailAlloc_1843_, 4, v___f_1745_);
v___x_1749_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1748_;
}
v_reusejp_1748_:
{
lean_object* v___x_1751_; 
if (v_isShared_1732_ == 0)
{
lean_ctor_set(v___x_1731_, 1, v___f_1741_);
lean_ctor_set(v___x_1731_, 0, v___x_1749_);
v___x_1751_ = v___x_1731_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1842_; 
v_reuseFailAlloc_1842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1842_, 0, v___x_1749_);
lean_ctor_set(v_reuseFailAlloc_1842_, 1, v___f_1741_);
v___x_1751_ = v_reuseFailAlloc_1842_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
lean_object* v___x_1752_; lean_object* v_toApplicative_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1840_; 
v___x_1752_ = l_StateRefT_x27_instMonad___redArg(v___x_1751_);
v_toApplicative_1753_ = lean_ctor_get(v___x_1752_, 0);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1752_);
if (v_isSharedCheck_1840_ == 0)
{
lean_object* v_unused_1841_; 
v_unused_1841_ = lean_ctor_get(v___x_1752_, 1);
lean_dec(v_unused_1841_);
v___x_1755_ = v___x_1752_;
v_isShared_1756_ = v_isSharedCheck_1840_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_toApplicative_1753_);
lean_dec(v___x_1752_);
v___x_1755_ = lean_box(0);
v_isShared_1756_ = v_isSharedCheck_1840_;
goto v_resetjp_1754_;
}
v_resetjp_1754_:
{
lean_object* v_toFunctor_1757_; lean_object* v_toSeq_1758_; lean_object* v_toSeqLeft_1759_; lean_object* v_toSeqRight_1760_; lean_object* v___x_1762_; uint8_t v_isShared_1763_; uint8_t v_isSharedCheck_1838_; 
v_toFunctor_1757_ = lean_ctor_get(v_toApplicative_1753_, 0);
v_toSeq_1758_ = lean_ctor_get(v_toApplicative_1753_, 2);
v_toSeqLeft_1759_ = lean_ctor_get(v_toApplicative_1753_, 3);
v_toSeqRight_1760_ = lean_ctor_get(v_toApplicative_1753_, 4);
v_isSharedCheck_1838_ = !lean_is_exclusive(v_toApplicative_1753_);
if (v_isSharedCheck_1838_ == 0)
{
lean_object* v_unused_1839_; 
v_unused_1839_ = lean_ctor_get(v_toApplicative_1753_, 1);
lean_dec(v_unused_1839_);
v___x_1762_ = v_toApplicative_1753_;
v_isShared_1763_ = v_isSharedCheck_1838_;
goto v_resetjp_1761_;
}
else
{
lean_inc(v_toSeqRight_1760_);
lean_inc(v_toSeqLeft_1759_);
lean_inc(v_toSeq_1758_);
lean_inc(v_toFunctor_1757_);
lean_dec(v_toApplicative_1753_);
v___x_1762_ = lean_box(0);
v_isShared_1763_ = v_isSharedCheck_1838_;
goto v_resetjp_1761_;
}
v_resetjp_1761_:
{
lean_object* v___f_1764_; lean_object* v___f_1765_; lean_object* v___f_1766_; lean_object* v___f_1767_; lean_object* v___x_1768_; lean_object* v___f_1769_; lean_object* v___f_1770_; lean_object* v___f_1771_; lean_object* v___x_1773_; 
v___f_1764_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3));
v___f_1765_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_1757_);
v___f_1766_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1766_, 0, v_toFunctor_1757_);
v___f_1767_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1767_, 0, v_toFunctor_1757_);
v___x_1768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1768_, 0, v___f_1766_);
lean_ctor_set(v___x_1768_, 1, v___f_1767_);
v___f_1769_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1769_, 0, v_toSeqRight_1760_);
v___f_1770_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1770_, 0, v_toSeqLeft_1759_);
v___f_1771_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1771_, 0, v_toSeq_1758_);
if (v_isShared_1763_ == 0)
{
lean_ctor_set(v___x_1762_, 4, v___f_1769_);
lean_ctor_set(v___x_1762_, 3, v___f_1770_);
lean_ctor_set(v___x_1762_, 2, v___f_1771_);
lean_ctor_set(v___x_1762_, 1, v___f_1764_);
lean_ctor_set(v___x_1762_, 0, v___x_1768_);
v___x_1773_ = v___x_1762_;
goto v_reusejp_1772_;
}
else
{
lean_object* v_reuseFailAlloc_1837_; 
v_reuseFailAlloc_1837_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1837_, 0, v___x_1768_);
lean_ctor_set(v_reuseFailAlloc_1837_, 1, v___f_1764_);
lean_ctor_set(v_reuseFailAlloc_1837_, 2, v___f_1771_);
lean_ctor_set(v_reuseFailAlloc_1837_, 3, v___f_1770_);
lean_ctor_set(v_reuseFailAlloc_1837_, 4, v___f_1769_);
v___x_1773_ = v_reuseFailAlloc_1837_;
goto v_reusejp_1772_;
}
v_reusejp_1772_:
{
lean_object* v___x_1775_; 
if (v_isShared_1756_ == 0)
{
lean_ctor_set(v___x_1755_, 1, v___f_1765_);
lean_ctor_set(v___x_1755_, 0, v___x_1773_);
v___x_1775_ = v___x_1755_;
goto v_reusejp_1774_;
}
else
{
lean_object* v_reuseFailAlloc_1836_; 
v_reuseFailAlloc_1836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1836_, 0, v___x_1773_);
lean_ctor_set(v_reuseFailAlloc_1836_, 1, v___f_1765_);
v___x_1775_ = v_reuseFailAlloc_1836_;
goto v_reusejp_1774_;
}
v_reusejp_1774_:
{
lean_object* v___x_1776_; lean_object* v_toApplicative_1777_; lean_object* v___x_1779_; uint8_t v_isShared_1780_; uint8_t v_isSharedCheck_1834_; 
v___x_1776_ = l_StateRefT_x27_instMonad___redArg(v___x_1775_);
v_toApplicative_1777_ = lean_ctor_get(v___x_1776_, 0);
v_isSharedCheck_1834_ = !lean_is_exclusive(v___x_1776_);
if (v_isSharedCheck_1834_ == 0)
{
lean_object* v_unused_1835_; 
v_unused_1835_ = lean_ctor_get(v___x_1776_, 1);
lean_dec(v_unused_1835_);
v___x_1779_ = v___x_1776_;
v_isShared_1780_ = v_isSharedCheck_1834_;
goto v_resetjp_1778_;
}
else
{
lean_inc(v_toApplicative_1777_);
lean_dec(v___x_1776_);
v___x_1779_ = lean_box(0);
v_isShared_1780_ = v_isSharedCheck_1834_;
goto v_resetjp_1778_;
}
v_resetjp_1778_:
{
lean_object* v_toFunctor_1781_; lean_object* v_toSeq_1782_; lean_object* v_toSeqLeft_1783_; lean_object* v_toSeqRight_1784_; lean_object* v___x_1786_; uint8_t v_isShared_1787_; uint8_t v_isSharedCheck_1832_; 
v_toFunctor_1781_ = lean_ctor_get(v_toApplicative_1777_, 0);
v_toSeq_1782_ = lean_ctor_get(v_toApplicative_1777_, 2);
v_toSeqLeft_1783_ = lean_ctor_get(v_toApplicative_1777_, 3);
v_toSeqRight_1784_ = lean_ctor_get(v_toApplicative_1777_, 4);
v_isSharedCheck_1832_ = !lean_is_exclusive(v_toApplicative_1777_);
if (v_isSharedCheck_1832_ == 0)
{
lean_object* v_unused_1833_; 
v_unused_1833_ = lean_ctor_get(v_toApplicative_1777_, 1);
lean_dec(v_unused_1833_);
v___x_1786_ = v_toApplicative_1777_;
v_isShared_1787_ = v_isSharedCheck_1832_;
goto v_resetjp_1785_;
}
else
{
lean_inc(v_toSeqRight_1784_);
lean_inc(v_toSeqLeft_1783_);
lean_inc(v_toSeq_1782_);
lean_inc(v_toFunctor_1781_);
lean_dec(v_toApplicative_1777_);
v___x_1786_ = lean_box(0);
v_isShared_1787_ = v_isSharedCheck_1832_;
goto v_resetjp_1785_;
}
v_resetjp_1785_:
{
lean_object* v___f_1788_; lean_object* v___f_1789_; lean_object* v___f_1790_; lean_object* v___f_1791_; lean_object* v___x_1792_; lean_object* v___f_1793_; lean_object* v___f_1794_; lean_object* v___f_1795_; lean_object* v___x_1797_; 
v___f_1788_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__0));
v___f_1789_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___closed__1));
lean_inc_ref(v_toFunctor_1781_);
v___f_1790_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1790_, 0, v_toFunctor_1781_);
v___f_1791_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1791_, 0, v_toFunctor_1781_);
v___x_1792_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1792_, 0, v___f_1790_);
lean_ctor_set(v___x_1792_, 1, v___f_1791_);
v___f_1793_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1793_, 0, v_toSeqRight_1784_);
v___f_1794_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1794_, 0, v_toSeqLeft_1783_);
v___f_1795_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1795_, 0, v_toSeq_1782_);
if (v_isShared_1787_ == 0)
{
lean_ctor_set(v___x_1786_, 4, v___f_1793_);
lean_ctor_set(v___x_1786_, 3, v___f_1794_);
lean_ctor_set(v___x_1786_, 2, v___f_1795_);
lean_ctor_set(v___x_1786_, 1, v___f_1788_);
lean_ctor_set(v___x_1786_, 0, v___x_1792_);
v___x_1797_ = v___x_1786_;
goto v_reusejp_1796_;
}
else
{
lean_object* v_reuseFailAlloc_1831_; 
v_reuseFailAlloc_1831_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1831_, 0, v___x_1792_);
lean_ctor_set(v_reuseFailAlloc_1831_, 1, v___f_1788_);
lean_ctor_set(v_reuseFailAlloc_1831_, 2, v___f_1795_);
lean_ctor_set(v_reuseFailAlloc_1831_, 3, v___f_1794_);
lean_ctor_set(v_reuseFailAlloc_1831_, 4, v___f_1793_);
v___x_1797_ = v_reuseFailAlloc_1831_;
goto v_reusejp_1796_;
}
v_reusejp_1796_:
{
lean_object* v___x_1799_; 
if (v_isShared_1780_ == 0)
{
lean_ctor_set(v___x_1779_, 1, v___f_1789_);
lean_ctor_set(v___x_1779_, 0, v___x_1797_);
v___x_1799_ = v___x_1779_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1830_; 
v_reuseFailAlloc_1830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1830_, 0, v___x_1797_);
lean_ctor_set(v_reuseFailAlloc_1830_, 1, v___f_1789_);
v___x_1799_ = v_reuseFailAlloc_1830_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
lean_object* v___x_1800_; lean_object* v___x_1801_; uint8_t v___x_1802_; 
v___x_1800_ = lean_array_get_size(v_acc_1719_);
v___x_1801_ = lean_array_get_size(v_declInfos_1716_);
v___x_1802_ = lean_nat_dec_lt(v___x_1800_, v___x_1801_);
if (v___x_1802_ == 0)
{
lean_object* v___x_1803_; 
lean_dec_ref(v___x_1799_);
lean_dec_ref(v_declInfos_1716_);
lean_inc(v___y_1725_);
lean_inc_ref(v___y_1724_);
lean_inc(v___y_1723_);
lean_inc_ref(v___y_1722_);
lean_inc(v___y_1721_);
lean_inc_ref(v___y_1720_);
v___x_1803_ = lean_apply_8(v_k_1717_, v_acc_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, lean_box(0));
return v___x_1803_;
}
else
{
lean_object* v___f_1804_; lean_object* v___x_1805_; uint8_t v___x_1806_; lean_object* v___f_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v_snd_1812_; lean_object* v_fst_1813_; lean_object* v_fst_1814_; lean_object* v_snd_1815_; lean_object* v___x_1816_; 
v___f_1804_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1804_, 0, v___x_1799_);
v___x_1805_ = lean_box(0);
v___x_1806_ = 0;
v___f_1807_ = lean_alloc_closure((void*)(l_Pi_instInhabited___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1807_, 0, v___f_1804_);
v___x_1808_ = lean_box(v___x_1806_);
v___x_1809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1808_);
lean_ctor_set(v___x_1809_, 1, v___f_1807_);
v___x_1810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1810_, 0, v___x_1805_);
lean_ctor_set(v___x_1810_, 1, v___x_1809_);
v___x_1811_ = lean_array_get(v___x_1810_, v_declInfos_1716_, v___x_1800_);
lean_dec_ref_known(v___x_1810_, 2);
v_snd_1812_ = lean_ctor_get(v___x_1811_, 1);
lean_inc(v_snd_1812_);
v_fst_1813_ = lean_ctor_get(v___x_1811_, 0);
lean_inc(v_fst_1813_);
lean_dec(v___x_1811_);
v_fst_1814_ = lean_ctor_get(v_snd_1812_, 0);
lean_inc(v_fst_1814_);
v_snd_1815_ = lean_ctor_get(v_snd_1812_, 1);
lean_inc(v_snd_1815_);
lean_dec(v_snd_1812_);
lean_inc(v___y_1725_);
lean_inc_ref(v___y_1724_);
lean_inc(v___y_1723_);
lean_inc_ref(v___y_1722_);
lean_inc(v___y_1721_);
lean_inc_ref(v___y_1720_);
lean_inc_ref(v_acc_1719_);
v___x_1816_ = lean_apply_8(v_snd_1815_, v_acc_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, lean_box(0));
if (lean_obj_tag(v___x_1816_) == 0)
{
lean_object* v_a_1817_; lean_object* v___x_1818_; lean_object* v___f_1819_; uint8_t v___x_1820_; lean_object* v___x_1821_; 
v_a_1817_ = lean_ctor_get(v___x_1816_, 0);
lean_inc(v_a_1817_);
lean_dec_ref_known(v___x_1816_, 1);
v___x_1818_ = lean_box(v_kind_1718_);
v___f_1819_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1___boxed), 12, 4);
lean_closure_set(v___f_1819_, 0, v_acc_1719_);
lean_closure_set(v___f_1819_, 1, v_declInfos_1716_);
lean_closure_set(v___f_1819_, 2, v_k_1717_);
lean_closure_set(v___f_1819_, 3, v___x_1818_);
v___x_1820_ = lean_unbox(v_fst_1814_);
lean_dec(v_fst_1814_);
v___x_1821_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg(v_fst_1813_, v___x_1820_, v_a_1817_, v___f_1819_, v_kind_1718_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_);
return v___x_1821_;
}
else
{
lean_object* v_a_1822_; lean_object* v___x_1824_; uint8_t v_isShared_1825_; uint8_t v_isSharedCheck_1829_; 
lean_dec(v_fst_1814_);
lean_dec(v_fst_1813_);
lean_dec_ref(v_acc_1719_);
lean_dec_ref(v_k_1717_);
lean_dec_ref(v_declInfos_1716_);
v_a_1822_ = lean_ctor_get(v___x_1816_, 0);
v_isSharedCheck_1829_ = !lean_is_exclusive(v___x_1816_);
if (v_isSharedCheck_1829_ == 0)
{
v___x_1824_ = v___x_1816_;
v_isShared_1825_ = v_isSharedCheck_1829_;
goto v_resetjp_1823_;
}
else
{
lean_inc(v_a_1822_);
lean_dec(v___x_1816_);
v___x_1824_ = lean_box(0);
v_isShared_1825_ = v_isSharedCheck_1829_;
goto v_resetjp_1823_;
}
v_resetjp_1823_:
{
lean_object* v___x_1827_; 
if (v_isShared_1825_ == 0)
{
v___x_1827_ = v___x_1824_;
goto v_reusejp_1826_;
}
else
{
lean_object* v_reuseFailAlloc_1828_; 
v_reuseFailAlloc_1828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1828_, 0, v_a_1822_);
v___x_1827_ = v_reuseFailAlloc_1828_;
goto v_reusejp_1826_;
}
v_reusejp_1826_:
{
return v___x_1827_;
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
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___lam__1(lean_object* v_acc_1848_, lean_object* v_declInfos_1849_, lean_object* v_k_1850_, uint8_t v_kind_1851_, lean_object* v_x_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
lean_object* v___x_1860_; lean_object* v___x_1861_; 
v___x_1860_ = lean_array_push(v_acc_1848_, v_x_1852_);
v___x_1861_ = lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11(v_declInfos_1849_, v_k_1850_, v_kind_1851_, v___x_1860_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_, v___y_1857_, v___y_1858_);
return v___x_1861_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11___boxed(lean_object* v_declInfos_1862_, lean_object* v_k_1863_, lean_object* v_kind_1864_, lean_object* v_acc_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_){
_start:
{
uint8_t v_kind_boxed_1873_; lean_object* v_res_1874_; 
v_kind_boxed_1873_ = lean_unbox(v_kind_1864_);
v_res_1874_ = lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11(v_declInfos_1862_, v_k_1863_, v_kind_boxed_1873_, v_acc_1865_, v___y_1866_, v___y_1867_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
lean_dec(v___y_1871_);
lean_dec_ref(v___y_1870_);
lean_dec(v___y_1869_);
lean_dec_ref(v___y_1868_);
lean_dec(v___y_1867_);
lean_dec_ref(v___y_1866_);
return v_res_1874_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7(lean_object* v_declInfos_1877_, lean_object* v_k_1878_, uint8_t v_kind_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_, lean_object* v___y_1882_, lean_object* v___y_1883_, lean_object* v___y_1884_, lean_object* v___y_1885_){
_start:
{
lean_object* v___x_1887_; lean_object* v___x_1888_; 
v___x_1887_ = ((lean_object*)(lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___closed__0));
v___x_1888_ = lp_plausible___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11(v_declInfos_1877_, v_k_1878_, v_kind_1879_, v___x_1887_, v___y_1880_, v___y_1881_, v___y_1882_, v___y_1883_, v___y_1884_, v___y_1885_);
return v___x_1888_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7___boxed(lean_object* v_declInfos_1889_, lean_object* v_k_1890_, lean_object* v_kind_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_){
_start:
{
uint8_t v_kind_boxed_1899_; lean_object* v_res_1900_; 
v_kind_boxed_1899_ = lean_unbox(v_kind_1891_);
v_res_1900_ = lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7(v_declInfos_1889_, v_k_1890_, v_kind_boxed_1899_, v___y_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
lean_dec(v___y_1897_);
lean_dec_ref(v___y_1896_);
lean_dec(v___y_1895_);
lean_dec_ref(v___y_1894_);
lean_dec(v___y_1893_);
lean_dec_ref(v___y_1892_);
return v_res_1900_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5(lean_object* v_declInfos_1901_, lean_object* v_k_1902_, uint8_t v_kind_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
size_t v_sz_1911_; size_t v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; 
v_sz_1911_ = lean_array_size(v_declInfos_1901_);
v___x_1912_ = ((size_t)0ULL);
v___x_1913_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__6(v_sz_1911_, v___x_1912_, v_declInfos_1901_);
v___x_1914_ = lp_plausible_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7(v___x_1913_, v_k_1902_, v_kind_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_, v___y_1909_);
return v___x_1914_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5___boxed(lean_object* v_declInfos_1915_, lean_object* v_k_1916_, lean_object* v_kind_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_){
_start:
{
uint8_t v_kind_boxed_1925_; lean_object* v_res_1926_; 
v_kind_boxed_1925_ = lean_unbox(v_kind_1917_);
v_res_1926_ = lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5(v_declInfos_1915_, v_k_1916_, v_kind_boxed_1925_, v___y_1918_, v___y_1919_, v___y_1920_, v___y_1921_, v___y_1922_, v___y_1923_);
lean_dec(v___y_1923_);
lean_dec_ref(v___y_1922_);
lean_dec(v___y_1921_);
lean_dec_ref(v___y_1920_);
lean_dec(v___y_1919_);
lean_dec_ref(v___y_1918_);
return v_res_1926_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4(lean_object* v_declInfos_1927_, lean_object* v_k_1928_, uint8_t v_kind_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
size_t v_sz_1937_; size_t v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; 
v_sz_1937_ = lean_array_size(v_declInfos_1927_);
v___x_1938_ = ((size_t)0ULL);
v___x_1939_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__4(v_sz_1937_, v___x_1938_, v_declInfos_1927_);
v___x_1940_ = lp_plausible_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5(v___x_1939_, v_k_1928_, v_kind_1929_, v___y_1930_, v___y_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_);
return v___x_1940_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4___boxed(lean_object* v_declInfos_1941_, lean_object* v_k_1942_, lean_object* v_kind_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_){
_start:
{
uint8_t v_kind_boxed_1951_; lean_object* v_res_1952_; 
v_kind_boxed_1951_ = lean_unbox(v_kind_1943_);
v_res_1952_ = lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4(v_declInfos_1941_, v_k_1942_, v_kind_boxed_1951_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1946_);
lean_dec(v___y_1945_);
lean_dec_ref(v___y_1944_);
return v_res_1952_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0(size_t v_sz_1953_, size_t v_i_1954_, lean_object* v_bs_1955_){
_start:
{
uint8_t v___x_1956_; 
v___x_1956_ = lean_usize_dec_lt(v_i_1954_, v_sz_1953_);
if (v___x_1956_ == 0)
{
return v_bs_1955_;
}
else
{
lean_object* v_v_1957_; lean_object* v___x_1958_; lean_object* v_bs_x27_1959_; lean_object* v___x_1960_; size_t v___x_1961_; size_t v___x_1962_; lean_object* v___x_1963_; 
v_v_1957_ = lean_array_uget(v_bs_1955_, v_i_1954_);
v___x_1958_ = lean_unsigned_to_nat(0u);
v_bs_x27_1959_ = lean_array_uset(v_bs_1955_, v_i_1954_, v___x_1958_);
v___x_1960_ = l_Lean_mkIdent(v_v_1957_);
v___x_1961_ = ((size_t)1ULL);
v___x_1962_ = lean_usize_add(v_i_1954_, v___x_1961_);
v___x_1963_ = lean_array_uset(v_bs_x27_1959_, v_i_1954_, v___x_1960_);
v_i_1954_ = v___x_1962_;
v_bs_1955_ = v___x_1963_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0___boxed(lean_object* v_sz_1965_, lean_object* v_i_1966_, lean_object* v_bs_1967_){
_start:
{
size_t v_sz_boxed_1968_; size_t v_i_boxed_1969_; lean_object* v_res_1970_; 
v_sz_boxed_1968_ = lean_unbox_usize(v_sz_1965_);
lean_dec(v_sz_1965_);
v_i_boxed_1969_ = lean_unbox_usize(v_i_1966_);
lean_dec(v_i_1966_);
v_res_1970_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0(v_sz_boxed_1968_, v_i_boxed_1969_, v_bs_1967_);
return v_res_1970_;
}
}
static lean_object* _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11(void){
_start:
{
lean_object* v___x_1989_; lean_object* v___x_1990_; 
v___x_1989_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__10));
v___x_1990_ = l_Lean_mkIdent(v___x_1989_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg(lean_object* v_inductiveVal_1993_, lean_object* v_targetTypeName_1994_, lean_object* v___x_1995_, lean_object* v_as_x27_1996_, lean_object* v_b_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_){
_start:
{
if (lean_obj_tag(v_as_x27_1996_) == 0)
{
lean_object* v___x_2005_; 
lean_dec(v___x_1995_);
lean_dec(v_targetTypeName_1994_);
lean_dec_ref(v_inductiveVal_1993_);
v___x_2005_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2005_, 0, v_b_1997_);
return v___x_2005_;
}
else
{
lean_object* v_head_2006_; lean_object* v_tail_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; 
v_head_2006_ = lean_ctor_get(v_as_x27_1996_, 0);
v_tail_2007_ = lean_ctor_get(v_as_x27_1996_, 1);
lean_inc_n(v_head_2006_, 2);
v___x_2008_ = l_Lean_mkIdent(v_head_2006_);
lean_inc_ref(v_inductiveVal_1993_);
v___x_2009_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes___redArg(v_inductiveVal_1993_, v_head_2006_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
if (lean_obj_tag(v___x_2009_) == 0)
{
lean_object* v_a_2010_; lean_object* v___x_2011_; lean_object* v_snd_2012_; lean_object* v_fst_2013_; lean_object* v_snd_2014_; lean_object* v___x_2016_; uint8_t v_isShared_2017_; uint8_t v_isSharedCheck_2181_; 
v_a_2010_ = lean_ctor_get(v___x_2009_, 0);
lean_inc(v_a_2010_);
lean_dec_ref_known(v___x_2009_, 1);
v___x_2011_ = l_Array_unzip___redArg(v_a_2010_);
v_snd_2012_ = lean_ctor_get(v_b_1997_, 1);
lean_inc(v_snd_2012_);
v_fst_2013_ = lean_ctor_get(v___x_2011_, 0);
v_snd_2014_ = lean_ctor_get(v___x_2011_, 1);
v_isSharedCheck_2181_ = !lean_is_exclusive(v___x_2011_);
if (v_isSharedCheck_2181_ == 0)
{
v___x_2016_ = v___x_2011_;
v_isShared_2017_ = v_isSharedCheck_2181_;
goto v_resetjp_2015_;
}
else
{
lean_inc(v_snd_2014_);
lean_inc(v_fst_2013_);
lean_dec(v___x_2011_);
v___x_2016_ = lean_box(0);
v_isShared_2017_ = v_isSharedCheck_2181_;
goto v_resetjp_2015_;
}
v_resetjp_2015_:
{
lean_object* v_fst_2018_; lean_object* v___x_2020_; uint8_t v_isShared_2021_; uint8_t v_isSharedCheck_2179_; 
v_fst_2018_ = lean_ctor_get(v_b_1997_, 0);
v_isSharedCheck_2179_ = !lean_is_exclusive(v_b_1997_);
if (v_isSharedCheck_2179_ == 0)
{
lean_object* v_unused_2180_; 
v_unused_2180_ = lean_ctor_get(v_b_1997_, 1);
lean_dec(v_unused_2180_);
v___x_2020_ = v_b_1997_;
v_isShared_2021_ = v_isSharedCheck_2179_;
goto v_resetjp_2019_;
}
else
{
lean_inc(v_fst_2018_);
lean_dec(v_b_1997_);
v___x_2020_ = lean_box(0);
v_isShared_2021_ = v_isSharedCheck_2179_;
goto v_resetjp_2019_;
}
v_resetjp_2019_:
{
lean_object* v_fst_2022_; lean_object* v_snd_2023_; lean_object* v___x_2025_; uint8_t v_isShared_2026_; uint8_t v_isSharedCheck_2178_; 
v_fst_2022_ = lean_ctor_get(v_snd_2012_, 0);
v_snd_2023_ = lean_ctor_get(v_snd_2012_, 1);
v_isSharedCheck_2178_ = !lean_is_exclusive(v_snd_2012_);
if (v_isSharedCheck_2178_ == 0)
{
v___x_2025_ = v_snd_2012_;
v_isShared_2026_ = v_isSharedCheck_2178_;
goto v_resetjp_2024_;
}
else
{
lean_inc(v_snd_2023_);
lean_inc(v_fst_2022_);
lean_dec(v_snd_2012_);
v___x_2025_ = lean_box(0);
v_isShared_2026_ = v_isSharedCheck_2178_;
goto v_resetjp_2024_;
}
v_resetjp_2024_:
{
lean_object* v___x_2027_; lean_object* v___x_2028_; uint8_t v___x_2029_; 
v___x_2027_ = lean_unsigned_to_nat(0u);
v___x_2028_ = lean_array_get_size(v_a_2010_);
v___x_2029_ = lean_nat_dec_eq(v___x_2028_, v___x_2027_);
if (v___x_2029_ == 0)
{
lean_object* v___x_2030_; size_t v_sz_2031_; size_t v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___f_2037_; uint8_t v___x_2038_; lean_object* v___x_2039_; 
v___x_2030_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0));
v_sz_2031_ = lean_array_size(v_fst_2013_);
v___x_2032_ = ((size_t)0ULL);
v___x_2033_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__0(v_sz_2031_, v___x_2032_, v_fst_2013_);
v___x_2034_ = l_Array_zip___redArg(v___x_2033_, v_snd_2014_);
lean_dec(v_snd_2014_);
v___x_2035_ = lean_box(v___x_2029_);
v___x_2036_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed__const__1));
lean_inc(v___x_1995_);
lean_inc(v_targetTypeName_1994_);
v___f_2037_ = lean_alloc_closure((void*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___boxed), 17, 9);
lean_closure_set(v___f_2037_, 0, v___x_2030_);
lean_closure_set(v___f_2037_, 1, v___x_2035_);
lean_closure_set(v___f_2037_, 2, v___x_2034_);
lean_closure_set(v___f_2037_, 3, v_targetTypeName_1994_);
lean_closure_set(v___f_2037_, 4, v___x_2028_);
lean_closure_set(v___f_2037_, 5, v___x_1995_);
lean_closure_set(v___f_2037_, 6, v___x_2036_);
lean_closure_set(v___f_2037_, 7, v___x_2033_);
lean_closure_set(v___f_2037_, 8, v___x_2008_);
v___x_2038_ = 0;
v___x_2039_ = lp_plausible_Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4(v_a_2010_, v___f_2037_, v___x_2038_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
if (lean_obj_tag(v___x_2039_) == 0)
{
lean_object* v_a_2040_; lean_object* v_fst_2041_; lean_object* v_snd_2042_; lean_object* v___x_2044_; uint8_t v_isShared_2045_; uint8_t v_isSharedCheck_2125_; 
v_a_2040_ = lean_ctor_get(v___x_2039_, 0);
lean_inc(v_a_2040_);
lean_dec_ref_known(v___x_2039_, 1);
v_fst_2041_ = lean_ctor_get(v_a_2040_, 0);
v_snd_2042_ = lean_ctor_get(v_a_2040_, 1);
v_isSharedCheck_2125_ = !lean_is_exclusive(v_a_2040_);
if (v_isSharedCheck_2125_ == 0)
{
v___x_2044_ = v_a_2040_;
v_isShared_2045_ = v_isSharedCheck_2125_;
goto v_resetjp_2043_;
}
else
{
lean_inc(v_snd_2042_);
lean_inc(v_fst_2041_);
lean_dec(v_a_2040_);
v___x_2044_ = lean_box(0);
v_isShared_2045_ = v_isSharedCheck_2125_;
goto v_resetjp_2043_;
}
v_resetjp_2043_:
{
uint8_t v___x_2084_; 
v___x_2084_ = lean_unbox(v_snd_2042_);
lean_dec(v_snd_2042_);
if (v___x_2084_ == 0)
{
lean_del_object(v___x_2020_);
lean_del_object(v___x_2016_);
goto v___jp_2046_;
}
else
{
if (v___x_2029_ == 0)
{
lean_object* v_ref_2085_; lean_object* v_quotContext_2086_; lean_object* v_currMacroScope_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2119_; 
lean_del_object(v___x_2044_);
lean_del_object(v___x_2025_);
v_ref_2085_ = lean_ctor_get(v___y_2002_, 5);
v_quotContext_2086_ = lean_ctor_get(v___y_2002_, 10);
v_currMacroScope_2087_ = lean_ctor_get(v___y_2002_, 11);
v___x_2088_ = l_Lean_SourceInfo_fromRef(v_ref_2085_, v___x_2029_);
v___x_2089_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1));
v___x_2090_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6));
v___x_2091_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7));
lean_inc_n(v___x_2088_, 12);
v___x_2092_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2092_, 0, v___x_2088_);
lean_ctor_set(v___x_2092_, 1, v___x_2091_);
v___x_2093_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9));
v___x_2094_ = lean_obj_once(&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11, &lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11_once, _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11);
v___x_2095_ = lean_box(0);
lean_inc(v_currMacroScope_2087_);
lean_inc(v_quotContext_2086_);
v___x_2096_ = l_Lean_addMacroScope(v_quotContext_2086_, v___x_2095_, v_currMacroScope_2087_);
v___x_2097_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43));
v___x_2098_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2098_, 0, v___x_2088_);
lean_ctor_set(v___x_2098_, 1, v___x_2094_);
lean_ctor_set(v___x_2098_, 2, v___x_2096_);
lean_ctor_set(v___x_2098_, 3, v___x_2097_);
v___x_2099_ = l_Lean_Syntax_node1(v___x_2088_, v___x_2093_, v___x_2098_);
v___x_2100_ = l_Lean_Syntax_node2(v___x_2088_, v___x_2090_, v___x_2092_, v___x_2099_);
v___x_2101_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_2102_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__7));
v___x_2103_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__8));
v___x_2104_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2104_, 0, v___x_2088_);
lean_ctor_set(v___x_2104_, 1, v___x_2103_);
v___x_2105_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3));
v___x_2106_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4));
v___x_2107_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2107_, 0, v___x_2088_);
lean_ctor_set(v___x_2107_, 1, v___x_2106_);
v___x_2108_ = l_Lean_Syntax_node1(v___x_2088_, v___x_2105_, v___x_2107_);
lean_inc(v___x_1995_);
v___x_2109_ = l_Lean_Syntax_node3(v___x_2088_, v___x_2102_, v___x_1995_, v___x_2104_, v___x_2108_);
v___x_2110_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5));
v___x_2111_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2111_, 0, v___x_2088_);
lean_ctor_set(v___x_2111_, 1, v___x_2110_);
v___x_2112_ = l_Lean_Syntax_node1(v___x_2088_, v___x_2101_, v_fst_2041_);
v___x_2113_ = l_Lean_Syntax_node3(v___x_2088_, v___x_2101_, v___x_2109_, v___x_2111_, v___x_2112_);
v___x_2114_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48));
v___x_2115_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2115_, 0, v___x_2088_);
lean_ctor_set(v___x_2115_, 1, v___x_2114_);
v___x_2116_ = l_Lean_Syntax_node3(v___x_2088_, v___x_2089_, v___x_2100_, v___x_2113_, v___x_2115_);
v___x_2117_ = lean_array_push(v_fst_2022_, v___x_2116_);
if (v_isShared_2021_ == 0)
{
lean_ctor_set(v___x_2020_, 1, v_snd_2023_);
lean_ctor_set(v___x_2020_, 0, v___x_2117_);
v___x_2119_ = v___x_2020_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v___x_2117_);
lean_ctor_set(v_reuseFailAlloc_2124_, 1, v_snd_2023_);
v___x_2119_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
lean_object* v___x_2121_; 
if (v_isShared_2017_ == 0)
{
lean_ctor_set(v___x_2016_, 1, v___x_2119_);
lean_ctor_set(v___x_2016_, 0, v_fst_2018_);
v___x_2121_ = v___x_2016_;
goto v_reusejp_2120_;
}
else
{
lean_object* v_reuseFailAlloc_2123_; 
v_reuseFailAlloc_2123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2123_, 0, v_fst_2018_);
lean_ctor_set(v_reuseFailAlloc_2123_, 1, v___x_2119_);
v___x_2121_ = v_reuseFailAlloc_2123_;
goto v_reusejp_2120_;
}
v_reusejp_2120_:
{
v_as_x27_1996_ = v_tail_2007_;
v_b_1997_ = v___x_2121_;
goto _start;
}
}
}
else
{
lean_del_object(v___x_2020_);
lean_del_object(v___x_2016_);
goto v___jp_2046_;
}
}
v___jp_2046_:
{
lean_object* v_ref_2047_; lean_object* v_quotContext_2048_; lean_object* v_currMacroScope_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; lean_object* v___x_2078_; 
v_ref_2047_ = lean_ctor_get(v___y_2002_, 5);
v_quotContext_2048_ = lean_ctor_get(v___y_2002_, 10);
v_currMacroScope_2049_ = lean_ctor_get(v___y_2002_, 11);
v___x_2050_ = l_Lean_SourceInfo_fromRef(v_ref_2047_, v___x_2029_);
v___x_2051_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1));
v___x_2052_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6));
v___x_2053_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7));
lean_inc_n(v___x_2050_, 10);
v___x_2054_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2054_, 0, v___x_2050_);
lean_ctor_set(v___x_2054_, 1, v___x_2053_);
v___x_2055_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9));
v___x_2056_ = lean_obj_once(&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11, &lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11_once, _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11);
v___x_2057_ = lean_box(0);
lean_inc(v_currMacroScope_2049_);
lean_inc(v_quotContext_2048_);
v___x_2058_ = l_Lean_addMacroScope(v_quotContext_2048_, v___x_2057_, v_currMacroScope_2049_);
v___x_2059_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43));
v___x_2060_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2060_, 0, v___x_2050_);
lean_ctor_set(v___x_2060_, 1, v___x_2056_);
lean_ctor_set(v___x_2060_, 2, v___x_2058_);
lean_ctor_set(v___x_2060_, 3, v___x_2059_);
v___x_2061_ = l_Lean_Syntax_node1(v___x_2050_, v___x_2055_, v___x_2060_);
v___x_2062_ = l_Lean_Syntax_node2(v___x_2050_, v___x_2052_, v___x_2054_, v___x_2061_);
v___x_2063_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_2064_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3));
v___x_2065_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4));
v___x_2066_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2066_, 0, v___x_2050_);
lean_ctor_set(v___x_2066_, 1, v___x_2065_);
v___x_2067_ = l_Lean_Syntax_node1(v___x_2050_, v___x_2064_, v___x_2066_);
v___x_2068_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5));
v___x_2069_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2069_, 0, v___x_2050_);
lean_ctor_set(v___x_2069_, 1, v___x_2068_);
lean_inc(v_fst_2041_);
v___x_2070_ = l_Lean_Syntax_node1(v___x_2050_, v___x_2063_, v_fst_2041_);
v___x_2071_ = l_Lean_Syntax_node3(v___x_2050_, v___x_2063_, v___x_2067_, v___x_2069_, v___x_2070_);
v___x_2072_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48));
v___x_2073_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2073_, 0, v___x_2050_);
lean_ctor_set(v___x_2073_, 1, v___x_2072_);
v___x_2074_ = l_Lean_Syntax_node3(v___x_2050_, v___x_2051_, v___x_2062_, v___x_2071_, v___x_2073_);
v___x_2075_ = lean_array_push(v_fst_2018_, v___x_2074_);
v___x_2076_ = lean_array_push(v_snd_2023_, v_fst_2041_);
if (v_isShared_2045_ == 0)
{
lean_ctor_set(v___x_2044_, 1, v___x_2076_);
lean_ctor_set(v___x_2044_, 0, v_fst_2022_);
v___x_2078_ = v___x_2044_;
goto v_reusejp_2077_;
}
else
{
lean_object* v_reuseFailAlloc_2083_; 
v_reuseFailAlloc_2083_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2083_, 0, v_fst_2022_);
lean_ctor_set(v_reuseFailAlloc_2083_, 1, v___x_2076_);
v___x_2078_ = v_reuseFailAlloc_2083_;
goto v_reusejp_2077_;
}
v_reusejp_2077_:
{
lean_object* v___x_2080_; 
if (v_isShared_2026_ == 0)
{
lean_ctor_set(v___x_2025_, 1, v___x_2078_);
lean_ctor_set(v___x_2025_, 0, v___x_2075_);
v___x_2080_ = v___x_2025_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2082_; 
v_reuseFailAlloc_2082_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2082_, 0, v___x_2075_);
lean_ctor_set(v_reuseFailAlloc_2082_, 1, v___x_2078_);
v___x_2080_ = v_reuseFailAlloc_2082_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
v_as_x27_1996_ = v_tail_2007_;
v_b_1997_ = v___x_2080_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_2126_; lean_object* v___x_2128_; uint8_t v_isShared_2129_; uint8_t v_isSharedCheck_2133_; 
lean_del_object(v___x_2025_);
lean_dec(v_snd_2023_);
lean_dec(v_fst_2022_);
lean_del_object(v___x_2020_);
lean_dec(v_fst_2018_);
lean_del_object(v___x_2016_);
lean_dec(v___x_1995_);
lean_dec(v_targetTypeName_1994_);
lean_dec_ref(v_inductiveVal_1993_);
v_a_2126_ = lean_ctor_get(v___x_2039_, 0);
v_isSharedCheck_2133_ = !lean_is_exclusive(v___x_2039_);
if (v_isSharedCheck_2133_ == 0)
{
v___x_2128_ = v___x_2039_;
v_isShared_2129_ = v_isSharedCheck_2133_;
goto v_resetjp_2127_;
}
else
{
lean_inc(v_a_2126_);
lean_dec(v___x_2039_);
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
else
{
lean_object* v_ref_2134_; lean_object* v_quotContext_2135_; lean_object* v_currMacroScope_2136_; uint8_t v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2172_; 
lean_del_object(v___x_2016_);
lean_dec(v_snd_2014_);
lean_dec(v_fst_2013_);
lean_dec(v_a_2010_);
v_ref_2134_ = lean_ctor_get(v___y_2002_, 5);
v_quotContext_2135_ = lean_ctor_get(v___y_2002_, 10);
v_currMacroScope_2136_ = lean_ctor_get(v___y_2002_, 11);
v___x_2137_ = 0;
v___x_2138_ = l_Lean_SourceInfo_fromRef(v_ref_2134_, v___x_2137_);
v___x_2139_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__4));
v___x_2140_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__6));
v___x_2141_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7));
lean_inc_n(v___x_2138_, 13);
v___x_2142_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2142_, 0, v___x_2138_);
lean_ctor_set(v___x_2142_, 1, v___x_2141_);
v___x_2143_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__9));
v___x_2144_ = lean_obj_once(&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11, &lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11_once, _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__11);
v___x_2145_ = lean_box(0);
lean_inc(v_currMacroScope_2136_);
lean_inc(v_quotContext_2135_);
v___x_2146_ = l_Lean_addMacroScope(v_quotContext_2135_, v___x_2145_, v_currMacroScope_2136_);
v___x_2147_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__43));
v___x_2148_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2148_, 0, v___x_2138_);
lean_ctor_set(v___x_2148_, 1, v___x_2144_);
lean_ctor_set(v___x_2148_, 2, v___x_2146_);
lean_ctor_set(v___x_2148_, 3, v___x_2147_);
v___x_2149_ = l_Lean_Syntax_node1(v___x_2138_, v___x_2143_, v___x_2148_);
v___x_2150_ = l_Lean_Syntax_node2(v___x_2138_, v___x_2140_, v___x_2142_, v___x_2149_);
v___x_2151_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
v___x_2152_ = lean_obj_once(&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11, &lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11_once, _init_lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__11);
v___x_2153_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_2154_ = l_Lean_Syntax_node1(v___x_2138_, v___x_2153_, v___x_2008_);
v___x_2155_ = l_Lean_Syntax_node2(v___x_2138_, v___x_2151_, v___x_2152_, v___x_2154_);
v___x_2156_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48));
v___x_2157_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2157_, 0, v___x_2138_);
lean_ctor_set(v___x_2157_, 1, v___x_2156_);
lean_inc_ref(v___x_2157_);
lean_inc(v___x_2150_);
v___x_2158_ = l_Lean_Syntax_node3(v___x_2138_, v___x_2139_, v___x_2150_, v___x_2155_, v___x_2157_);
v___x_2159_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__1));
v___x_2160_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3));
v___x_2161_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4));
v___x_2162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2162_, 0, v___x_2138_);
lean_ctor_set(v___x_2162_, 1, v___x_2161_);
v___x_2163_ = l_Lean_Syntax_node1(v___x_2138_, v___x_2160_, v___x_2162_);
v___x_2164_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5));
v___x_2165_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2165_, 0, v___x_2138_);
lean_ctor_set(v___x_2165_, 1, v___x_2164_);
lean_inc(v___x_2158_);
v___x_2166_ = l_Lean_Syntax_node1(v___x_2138_, v___x_2153_, v___x_2158_);
v___x_2167_ = l_Lean_Syntax_node3(v___x_2138_, v___x_2153_, v___x_2163_, v___x_2165_, v___x_2166_);
v___x_2168_ = l_Lean_Syntax_node3(v___x_2138_, v___x_2159_, v___x_2150_, v___x_2167_, v___x_2157_);
v___x_2169_ = lean_array_push(v_fst_2018_, v___x_2168_);
v___x_2170_ = lean_array_push(v_snd_2023_, v___x_2158_);
if (v_isShared_2026_ == 0)
{
lean_ctor_set(v___x_2025_, 1, v___x_2170_);
v___x_2172_ = v___x_2025_;
goto v_reusejp_2171_;
}
else
{
lean_object* v_reuseFailAlloc_2177_; 
v_reuseFailAlloc_2177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2177_, 0, v_fst_2022_);
lean_ctor_set(v_reuseFailAlloc_2177_, 1, v___x_2170_);
v___x_2172_ = v_reuseFailAlloc_2177_;
goto v_reusejp_2171_;
}
v_reusejp_2171_:
{
lean_object* v___x_2174_; 
if (v_isShared_2021_ == 0)
{
lean_ctor_set(v___x_2020_, 1, v___x_2172_);
lean_ctor_set(v___x_2020_, 0, v___x_2169_);
v___x_2174_ = v___x_2020_;
goto v_reusejp_2173_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v___x_2169_);
lean_ctor_set(v_reuseFailAlloc_2176_, 1, v___x_2172_);
v___x_2174_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2173_;
}
v_reusejp_2173_:
{
v_as_x27_1996_ = v_tail_2007_;
v_b_1997_ = v___x_2174_;
goto _start;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2182_; lean_object* v___x_2184_; uint8_t v_isShared_2185_; uint8_t v_isSharedCheck_2189_; 
lean_dec(v___x_2008_);
lean_dec_ref(v_b_1997_);
lean_dec(v___x_1995_);
lean_dec(v_targetTypeName_1994_);
lean_dec_ref(v_inductiveVal_1993_);
v_a_2182_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2189_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2189_ == 0)
{
v___x_2184_ = v___x_2009_;
v_isShared_2185_ = v_isSharedCheck_2189_;
goto v_resetjp_2183_;
}
else
{
lean_inc(v_a_2182_);
lean_dec(v___x_2009_);
v___x_2184_ = lean_box(0);
v_isShared_2185_ = v_isSharedCheck_2189_;
goto v_resetjp_2183_;
}
v_resetjp_2183_:
{
lean_object* v___x_2187_; 
if (v_isShared_2185_ == 0)
{
v___x_2187_ = v___x_2184_;
goto v_reusejp_2186_;
}
else
{
lean_object* v_reuseFailAlloc_2188_; 
v_reuseFailAlloc_2188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2188_, 0, v_a_2182_);
v___x_2187_ = v_reuseFailAlloc_2188_;
goto v_reusejp_2186_;
}
v_reusejp_2186_:
{
return v___x_2187_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___boxed(lean_object* v_inductiveVal_2190_, lean_object* v_targetTypeName_2191_, lean_object* v___x_2192_, lean_object* v_as_x27_2193_, lean_object* v_b_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_){
_start:
{
lean_object* v_res_2202_; 
v_res_2202_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg(v_inductiveVal_2190_, v_targetTypeName_2191_, v___x_2192_, v_as_x27_2193_, v_b_2194_, v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_);
lean_dec(v___y_2200_);
lean_dec_ref(v___y_2199_);
lean_dec(v___y_2198_);
lean_dec_ref(v___y_2197_);
lean_dec(v___y_2196_);
lean_dec_ref(v___y_2195_);
lean_dec(v_as_x27_2193_);
return v_res_2202_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6(size_t v_sz_2203_, size_t v_i_2204_, lean_object* v_bs_2205_){
_start:
{
uint8_t v___x_2206_; 
v___x_2206_ = lean_usize_dec_lt(v_i_2204_, v_sz_2203_);
if (v___x_2206_ == 0)
{
return v_bs_2205_;
}
else
{
lean_object* v_v_2207_; lean_object* v___x_2208_; lean_object* v_bs_x27_2209_; size_t v___x_2210_; size_t v___x_2211_; lean_object* v___x_2212_; 
v_v_2207_ = lean_array_uget(v_bs_2205_, v_i_2204_);
v___x_2208_ = lean_unsigned_to_nat(0u);
v_bs_x27_2209_ = lean_array_uset(v_bs_2205_, v_i_2204_, v___x_2208_);
v___x_2210_ = ((size_t)1ULL);
v___x_2211_ = lean_usize_add(v_i_2204_, v___x_2210_);
v___x_2212_ = lean_array_uset(v_bs_x27_2209_, v_i_2204_, v_v_2207_);
v_i_2204_ = v___x_2211_;
v_bs_2205_ = v___x_2212_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6___boxed(lean_object* v_sz_2214_, lean_object* v_i_2215_, lean_object* v_bs_2216_){
_start:
{
size_t v_sz_boxed_2217_; size_t v_i_boxed_2218_; lean_object* v_res_2219_; 
v_sz_boxed_2217_ = lean_unbox_usize(v_sz_2214_);
lean_dec(v_sz_2214_);
v_i_boxed_2218_ = lean_unbox_usize(v_i_2215_);
lean_dec(v_i_2215_);
v_res_2219_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6(v_sz_boxed_2217_, v_i_boxed_2218_, v_bs_2216_);
return v_res_2219_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10(void){
_start:
{
lean_object* v___x_2237_; lean_object* v___x_2238_; 
v___x_2237_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__9));
v___x_2238_ = l_Lean_mkIdent(v___x_2237_);
return v___x_2238_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19(void){
_start:
{
lean_object* v___x_2253_; lean_object* v___x_2254_; 
v___x_2253_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__18));
v___x_2254_ = l_Lean_mkIdent(v___x_2253_);
return v___x_2254_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23(void){
_start:
{
lean_object* v___x_2261_; lean_object* v___x_2262_; 
v___x_2261_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__22));
v___x_2262_ = l_Lean_mkIdent(v___x_2261_);
return v___x_2262_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26(void){
_start:
{
lean_object* v___x_2268_; lean_object* v___x_2269_; 
v___x_2268_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__25));
v___x_2269_ = l_Lean_mkIdent(v___x_2268_);
return v___x_2269_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58(void){
_start:
{
lean_object* v___x_2358_; lean_object* v___x_2359_; 
v___x_2358_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__57));
v___x_2359_ = l_Lean_stringToMessageData(v___x_2358_);
return v___x_2359_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60(void){
_start:
{
lean_object* v___x_2361_; lean_object* v___x_2362_; 
v___x_2361_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__59));
v___x_2362_ = l_Lean_stringToMessageData(v___x_2361_);
return v___x_2362_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody(lean_object* v_header_2363_, lean_object* v_inductiveVal_2364_, lean_object* v_generatorType_2365_, lean_object* v_a_2366_, lean_object* v_a_2367_, lean_object* v_a_2368_, lean_object* v_a_2369_, lean_object* v_a_2370_, lean_object* v_a_2371_){
_start:
{
lean_object* v___x_2373_; lean_object* v___x_2374_; 
v___x_2373_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__1));
v___x_2374_ = l_Lean_Core_mkFreshUserName(v___x_2373_, v_a_2370_, v_a_2371_);
if (lean_obj_tag(v___x_2374_) == 0)
{
lean_object* v_a_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; 
v_a_2375_ = lean_ctor_get(v___x_2374_, 0);
lean_inc(v_a_2375_);
lean_dec_ref_known(v___x_2374_, 1);
v___x_2376_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__3));
v___x_2377_ = l_Lean_Core_mkFreshUserName(v___x_2376_, v_a_2370_, v_a_2371_);
if (lean_obj_tag(v___x_2377_) == 0)
{
lean_object* v_toConstantVal_2378_; lean_object* v_a_2379_; lean_object* v_ctors_2380_; lean_object* v_name_2381_; lean_object* v___x_2383_; uint8_t v_isShared_2384_; uint8_t v_isSharedCheck_2570_; 
v_toConstantVal_2378_ = lean_ctor_get(v_inductiveVal_2364_, 0);
lean_inc_ref(v_toConstantVal_2378_);
v_a_2379_ = lean_ctor_get(v___x_2377_, 0);
lean_inc(v_a_2379_);
lean_dec_ref_known(v___x_2377_, 1);
v_ctors_2380_ = lean_ctor_get(v_inductiveVal_2364_, 4);
lean_inc(v_ctors_2380_);
v_name_2381_ = lean_ctor_get(v_toConstantVal_2378_, 0);
v_isSharedCheck_2570_ = !lean_is_exclusive(v_toConstantVal_2378_);
if (v_isSharedCheck_2570_ == 0)
{
lean_object* v_unused_2571_; lean_object* v_unused_2572_; 
v_unused_2571_ = lean_ctor_get(v_toConstantVal_2378_, 2);
lean_dec(v_unused_2571_);
v_unused_2572_ = lean_ctor_get(v_toConstantVal_2378_, 1);
lean_dec(v_unused_2572_);
v___x_2383_ = v_toConstantVal_2378_;
v_isShared_2384_ = v_isSharedCheck_2570_;
goto v_resetjp_2382_;
}
else
{
lean_inc(v_name_2381_);
lean_dec(v_toConstantVal_2378_);
v___x_2383_ = lean_box(0);
v_isShared_2384_ = v_isSharedCheck_2570_;
goto v_resetjp_2382_;
}
v_resetjp_2382_:
{
lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; 
v___x_2385_ = l_Lean_mkIdent(v_a_2379_);
v___x_2386_ = lean_unsigned_to_nat(0u);
v___x_2387_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0));
v___x_2388_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__5));
lean_inc(v___x_2385_);
lean_inc(v_name_2381_);
v___x_2389_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg(v_inductiveVal_2364_, v_name_2381_, v___x_2385_, v_ctors_2380_, v___x_2388_, v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
lean_dec(v_ctors_2380_);
if (lean_obj_tag(v___x_2389_) == 0)
{
lean_object* v_a_2390_; lean_object* v_snd_2391_; lean_object* v_fst_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2561_; 
v_a_2390_ = lean_ctor_get(v___x_2389_, 0);
lean_inc(v_a_2390_);
lean_dec_ref_known(v___x_2389_, 1);
v_snd_2391_ = lean_ctor_get(v_a_2390_, 1);
v_fst_2392_ = lean_ctor_get(v_a_2390_, 0);
v_isSharedCheck_2561_ = !lean_is_exclusive(v_a_2390_);
if (v_isSharedCheck_2561_ == 0)
{
v___x_2394_ = v_a_2390_;
v_isShared_2395_ = v_isSharedCheck_2561_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_snd_2391_);
lean_inc(v_fst_2392_);
lean_dec(v_a_2390_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2561_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v_fst_2396_; lean_object* v_snd_2397_; lean_object* v___x_2399_; uint8_t v_isShared_2400_; uint8_t v_isSharedCheck_2560_; 
v_fst_2396_ = lean_ctor_get(v_snd_2391_, 0);
v_snd_2397_ = lean_ctor_get(v_snd_2391_, 1);
v_isSharedCheck_2560_ = !lean_is_exclusive(v_snd_2391_);
if (v_isSharedCheck_2560_ == 0)
{
v___x_2399_ = v_snd_2391_;
v_isShared_2400_ = v_isSharedCheck_2560_;
goto v_resetjp_2398_;
}
else
{
lean_inc(v_snd_2397_);
lean_inc(v_fst_2396_);
lean_dec(v_snd_2391_);
v___x_2399_ = lean_box(0);
v_isShared_2400_ = v_isSharedCheck_2560_;
goto v_resetjp_2398_;
}
v_resetjp_2398_:
{
lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v_a_2404_; lean_object* v___x_2551_; uint8_t v___x_2552_; 
v___x_2401_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__15);
v___x_2402_ = l_Lean_mkIdent(v_a_2375_);
v___x_2551_ = lean_array_get_size(v_snd_2397_);
v___x_2552_ = lean_nat_dec_lt(v___x_2386_, v___x_2551_);
if (v___x_2552_ == 0)
{
lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; 
lean_dec(v___x_2402_);
lean_del_object(v___x_2399_);
lean_dec(v_snd_2397_);
lean_dec(v_fst_2396_);
lean_del_object(v___x_2394_);
lean_dec(v_fst_2392_);
lean_dec(v___x_2385_);
lean_del_object(v___x_2383_);
lean_dec(v_generatorType_2365_);
v___x_2553_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__58);
v___x_2554_ = l_Lean_MessageData_ofName(v_name_2381_);
v___x_2555_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2555_, 0, v___x_2553_);
lean_ctor_set(v___x_2555_, 1, v___x_2554_);
v___x_2556_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__60);
v___x_2557_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2557_, 0, v___x_2555_);
lean_ctor_set(v___x_2557_, 1, v___x_2556_);
v___x_2558_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(v___x_2557_, v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
return v___x_2558_;
}
else
{
lean_object* v___x_2559_; 
lean_dec(v_name_2381_);
v___x_2559_ = lean_array_fget_borrowed(v_snd_2397_, v___x_2386_);
lean_inc(v___x_2559_);
v_a_2404_ = v___x_2559_;
goto v___jp_2403_;
}
v___jp_2403_:
{
lean_object* v_ref_2405_; uint8_t v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2410_; 
v_ref_2405_ = lean_ctor_get(v_a_2370_, 5);
v___x_2406_ = 0;
v___x_2407_ = l_Lean_SourceInfo_fromRef(v_ref_2405_, v___x_2406_);
v___x_2408_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__6));
lean_inc(v___x_2407_);
if (v_isShared_2400_ == 0)
{
lean_ctor_set_tag(v___x_2399_, 2);
lean_ctor_set(v___x_2399_, 1, v___x_2408_);
lean_ctor_set(v___x_2399_, 0, v___x_2407_);
v___x_2410_ = v___x_2399_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2550_; 
v_reuseFailAlloc_2550_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2550_, 0, v___x_2407_);
lean_ctor_set(v_reuseFailAlloc_2550_, 1, v___x_2408_);
v___x_2410_ = v_reuseFailAlloc_2550_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; lean_object* v___x_2414_; lean_object* v___x_2415_; lean_object* v___x_2417_; 
v___x_2411_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_2412_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__10);
lean_inc_n(v___x_2407_, 3);
v___x_2413_ = l_Lean_Syntax_node1(v___x_2407_, v___x_2411_, v___x_2412_);
v___x_2414_ = l_Lean_Syntax_node1(v___x_2407_, v___x_2411_, v___x_2413_);
v___x_2415_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__11));
if (v_isShared_2395_ == 0)
{
lean_ctor_set_tag(v___x_2394_, 2);
lean_ctor_set(v___x_2394_, 1, v___x_2415_);
lean_ctor_set(v___x_2394_, 0, v___x_2407_);
v___x_2417_ = v___x_2394_;
goto v_reusejp_2416_;
}
else
{
lean_object* v_reuseFailAlloc_2549_; 
v_reuseFailAlloc_2549_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2549_, 0, v___x_2407_);
lean_ctor_set(v_reuseFailAlloc_2549_, 1, v___x_2415_);
v___x_2417_ = v_reuseFailAlloc_2549_;
goto v_reusejp_2416_;
}
v_reusejp_2416_:
{
lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2426_; 
v___x_2418_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__13));
v___x_2419_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__14));
lean_inc_n(v___x_2407_, 2);
v___x_2420_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2420_, 0, v___x_2407_);
lean_ctor_set(v___x_2420_, 1, v___x_2419_);
v___x_2421_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
v___x_2422_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__5));
v___x_2423_ = l_Lean_Syntax_SepArray_ofElems(v___x_2422_, v_snd_2397_);
lean_dec(v_snd_2397_);
v___x_2424_ = l_Array_append___redArg(v___x_2421_, v___x_2423_);
lean_dec_ref(v___x_2423_);
if (v_isShared_2384_ == 0)
{
lean_ctor_set_tag(v___x_2383_, 1);
lean_ctor_set(v___x_2383_, 2, v___x_2424_);
lean_ctor_set(v___x_2383_, 1, v___x_2411_);
lean_ctor_set(v___x_2383_, 0, v___x_2407_);
v___x_2426_ = v___x_2383_;
goto v_reusejp_2425_;
}
else
{
lean_object* v_reuseFailAlloc_2548_; 
v_reuseFailAlloc_2548_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2548_, 0, v___x_2407_);
lean_ctor_set(v_reuseFailAlloc_2548_, 1, v___x_2411_);
lean_ctor_set(v_reuseFailAlloc_2548_, 2, v___x_2424_);
v___x_2426_ = v_reuseFailAlloc_2548_;
goto v_reusejp_2425_;
}
v_reusejp_2425_:
{
lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v_a_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v_a_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v_a_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; lean_object* v___x_2468_; lean_object* v_a_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; size_t v_sz_2490_; size_t v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v_a_2498_; lean_object* v___x_2500_; uint8_t v_isShared_2501_; uint8_t v_isSharedCheck_2547_; 
v___x_2427_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__15));
lean_inc_n(v___x_2407_, 4);
v___x_2428_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2428_, 0, v___x_2407_);
lean_ctor_set(v___x_2428_, 1, v___x_2427_);
v___x_2429_ = l_Lean_Syntax_node3(v___x_2407_, v___x_2418_, v___x_2420_, v___x_2426_, v___x_2428_);
v___x_2430_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
v_a_2431_ = lean_ctor_get(v___x_2430_, 0);
lean_inc_n(v_a_2431_, 5);
lean_dec_ref(v___x_2430_);
lean_inc(v_a_2404_);
v___x_2432_ = l_Lean_Syntax_node2(v___x_2407_, v___x_2411_, v_a_2404_, v___x_2429_);
v___x_2433_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2433_, 0, v_a_2431_);
lean_ctor_set(v___x_2433_, 1, v___x_2419_);
v___x_2434_ = l_Lean_Syntax_SepArray_ofElems(v___x_2422_, v_fst_2392_);
lean_dec(v_fst_2392_);
v___x_2435_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2435_, 0, v_a_2431_);
lean_ctor_set(v___x_2435_, 1, v___x_2422_);
v___x_2436_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
v_a_2437_ = lean_ctor_get(v___x_2436_, 0);
lean_inc_n(v_a_2437_, 11);
lean_dec_ref(v___x_2436_);
v___x_2438_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__17));
v___x_2439_ = l_Array_append___redArg(v___x_2421_, v___x_2434_);
lean_dec_ref(v___x_2434_);
v___x_2440_ = lean_array_push(v___x_2439_, v___x_2435_);
v___x_2441_ = l_Lean_Syntax_SepArray_ofElems(v___x_2422_, v_fst_2396_);
lean_dec(v_fst_2396_);
v___x_2442_ = l_Array_append___redArg(v___x_2440_, v___x_2441_);
lean_dec_ref(v___x_2441_);
v___x_2443_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2443_, 0, v_a_2431_);
lean_ctor_set(v___x_2443_, 1, v___x_2411_);
lean_ctor_set(v___x_2443_, 2, v___x_2442_);
v___x_2444_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2444_, 0, v_a_2431_);
lean_ctor_set(v___x_2444_, 1, v___x_2427_);
v___x_2445_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2445_, 0, v_a_2437_);
lean_ctor_set(v___x_2445_, 1, v___x_2408_);
v___x_2446_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__7));
v___x_2447_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__8));
v___x_2448_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2448_, 0, v_a_2437_);
lean_ctor_set(v___x_2448_, 1, v___x_2447_);
v___x_2449_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__3));
v___x_2450_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___closed__4));
v___x_2451_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2451_, 0, v_a_2437_);
lean_ctor_set(v___x_2451_, 1, v___x_2450_);
v___x_2452_ = l_Lean_Syntax_node1(v_a_2437_, v___x_2449_, v___x_2451_);
v___x_2453_ = l_Lean_Syntax_node3(v_a_2437_, v___x_2446_, v___x_2385_, v___x_2448_, v___x_2452_);
v___x_2454_ = l_Lean_Syntax_node1(v_a_2437_, v___x_2411_, v___x_2453_);
v___x_2455_ = l_Lean_Syntax_node1(v_a_2437_, v___x_2411_, v___x_2454_);
v___x_2456_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2456_, 0, v_a_2437_);
lean_ctor_set(v___x_2456_, 1, v___x_2415_);
v___x_2457_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
v_a_2458_ = lean_ctor_get(v___x_2457_, 0);
lean_inc_n(v_a_2458_, 6);
lean_dec_ref(v___x_2457_);
v___x_2459_ = l_Lean_Syntax_node3(v_a_2431_, v___x_2418_, v___x_2433_, v___x_2443_, v___x_2444_);
v___x_2460_ = l_Lean_Syntax_node2(v_a_2437_, v___x_2411_, v_a_2404_, v___x_2459_);
v___x_2461_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__7));
v___x_2462_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2462_, 0, v_a_2458_);
lean_ctor_set(v___x_2462_, 1, v___x_2461_);
lean_inc_n(v___x_2402_, 2);
v___x_2463_ = l_Lean_Syntax_node1(v_a_2458_, v___x_2411_, v___x_2402_);
v___x_2464_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8));
v___x_2465_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2465_, 0, v_a_2458_);
lean_ctor_set(v___x_2465_, 1, v___x_2464_);
v___x_2466_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19);
v___x_2467_ = l_Lean_Syntax_node2(v_a_2458_, v___x_2411_, v___x_2465_, v___x_2466_);
v___x_2468_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
v_a_2469_ = lean_ctor_get(v___x_2468_, 0);
lean_inc_n(v_a_2469_, 8);
lean_dec_ref(v___x_2468_);
v___x_2470_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
v___x_2471_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__23);
v___x_2472_ = l_Lean_Syntax_node2(v___x_2407_, v___x_2470_, v___x_2471_, v___x_2432_);
v___x_2473_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2473_, 0, v_a_2458_);
lean_ctor_set(v___x_2473_, 1, v___x_2411_);
lean_ctor_set(v___x_2473_, 2, v___x_2421_);
v___x_2474_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__26);
v___x_2475_ = l_Lean_Syntax_node2(v_a_2437_, v___x_2470_, v___x_2474_, v___x_2460_);
v___x_2476_ = l_Lean_Syntax_node4(v___x_2407_, v___x_2438_, v___x_2410_, v___x_2414_, v___x_2417_, v___x_2472_);
v___x_2477_ = lean_array_push(v___x_2387_, v___x_2476_);
v___x_2478_ = l_Lean_Syntax_node4(v_a_2437_, v___x_2438_, v___x_2445_, v___x_2455_, v___x_2456_, v___x_2475_);
v___x_2479_ = lean_array_push(v___x_2477_, v___x_2478_);
v___x_2480_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__27));
v___x_2481_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__28));
v___x_2482_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2482_, 0, v_a_2469_);
lean_ctor_set(v___x_2482_, 1, v___x_2480_);
v___x_2483_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2483_, 0, v_a_2469_);
lean_ctor_set(v___x_2483_, 1, v___x_2411_);
lean_ctor_set(v___x_2483_, 2, v___x_2421_);
v___x_2484_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__30));
lean_inc_ref_n(v___x_2483_, 2);
v___x_2485_ = l_Lean_Syntax_node2(v_a_2469_, v___x_2484_, v___x_2483_, v___x_2402_);
v___x_2486_ = l_Lean_Syntax_node1(v_a_2469_, v___x_2411_, v___x_2485_);
v___x_2487_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__31));
v___x_2488_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2488_, 0, v_a_2469_);
lean_ctor_set(v___x_2488_, 1, v___x_2487_);
v___x_2489_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__33));
v_sz_2490_ = lean_array_size(v___x_2479_);
v___x_2491_ = ((size_t)0ULL);
v___x_2492_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__6(v_sz_2490_, v___x_2491_, v___x_2479_);
v___x_2493_ = l_Array_append___redArg(v___x_2421_, v___x_2492_);
lean_dec_ref(v___x_2492_);
v___x_2494_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2494_, 0, v_a_2469_);
lean_ctor_set(v___x_2494_, 1, v___x_2411_);
lean_ctor_set(v___x_2494_, 2, v___x_2493_);
v___x_2495_ = l_Lean_Syntax_node1(v_a_2469_, v___x_2489_, v___x_2494_);
v___x_2496_ = l_Lean_Syntax_node6(v_a_2469_, v___x_2481_, v___x_2482_, v___x_2483_, v___x_2483_, v___x_2486_, v___x_2488_, v___x_2495_);
v___x_2497_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
v_a_2498_ = lean_ctor_get(v___x_2497_, 0);
v_isSharedCheck_2547_ = !lean_is_exclusive(v___x_2497_);
if (v_isSharedCheck_2547_ == 0)
{
v___x_2500_ = v___x_2497_;
v_isShared_2501_ = v_isSharedCheck_2547_;
goto v_resetjp_2499_;
}
else
{
lean_inc(v_a_2498_);
lean_dec(v___x_2497_);
v___x_2500_ = lean_box(0);
v_isShared_2501_ = v_isSharedCheck_2547_;
goto v_resetjp_2499_;
}
v_resetjp_2499_:
{
lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; lean_object* v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; lean_object* v___x_2543_; lean_object* v___x_2545_; 
v___x_2502_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__48));
lean_inc(v_a_2458_);
v___x_2503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2503_, 0, v_a_2458_);
lean_ctor_set(v___x_2503_, 1, v___x_2502_);
v___x_2504_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__35));
v___x_2505_ = l_Lean_Syntax_node5(v_a_2458_, v___x_2504_, v___x_2462_, v___x_2463_, v___x_2467_, v___x_2473_, v___x_2503_);
v___x_2506_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__37));
v___x_2507_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__39));
v___x_2508_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__2));
lean_inc_n(v_a_2498_, 22);
v___x_2509_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2509_, 0, v_a_2498_);
lean_ctor_set(v___x_2509_, 1, v___x_2508_);
v___x_2510_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__40));
v___x_2511_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2511_, 0, v_a_2498_);
lean_ctor_set(v___x_2511_, 1, v___x_2510_);
v___x_2512_ = l_Lean_Syntax_node2(v_a_2498_, v___x_2507_, v___x_2509_, v___x_2511_);
v___x_2513_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__42));
v___x_2514_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__44));
v___x_2515_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2515_, 0, v_a_2498_);
lean_ctor_set(v___x_2515_, 1, v___x_2411_);
lean_ctor_set(v___x_2515_, 2, v___x_2421_);
v___x_2516_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__46));
v___x_2517_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__48));
v___x_2518_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__50));
v___x_2519_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2518_, v___x_2401_);
v___x_2520_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2411_, v___x_2505_);
v___x_2521_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51));
v___x_2522_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2522_, 0, v_a_2498_);
lean_ctor_set(v___x_2522_, 1, v___x_2464_);
v___x_2523_ = l_Lean_Syntax_node2(v_a_2498_, v___x_2521_, v___x_2522_, v_generatorType_2365_);
v___x_2524_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2411_, v___x_2523_);
v___x_2525_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10));
v___x_2526_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2526_, 0, v_a_2498_);
lean_ctor_set(v___x_2526_, 1, v___x_2525_);
v___x_2527_ = l_Lean_Syntax_node5(v_a_2498_, v___x_2517_, v___x_2519_, v___x_2520_, v___x_2524_, v___x_2526_, v___x_2496_);
v___x_2528_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2516_, v___x_2527_);
v___x_2529_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52));
lean_inc_ref_n(v___x_2515_, 5);
v___x_2530_ = l_Lean_Syntax_node2(v_a_2498_, v___x_2529_, v___x_2515_, v___x_2515_);
v___x_2531_ = l_Lean_Syntax_node4(v_a_2498_, v___x_2514_, v___x_2515_, v___x_2515_, v___x_2528_, v___x_2530_);
v___x_2532_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2411_, v___x_2531_);
v___x_2533_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2513_, v___x_2532_);
v___x_2534_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__53));
v___x_2535_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__54));
v___x_2536_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2536_, 0, v_a_2498_);
lean_ctor_set(v___x_2536_, 1, v___x_2534_);
v___x_2537_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__56));
v___x_2538_ = l_Lean_Syntax_node1(v_a_2498_, v___x_2411_, v___x_2402_);
v___x_2539_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2539_, 0, v_a_2498_);
lean_ctor_set(v___x_2539_, 1, v___x_2415_);
lean_inc(v___x_2538_);
v___x_2540_ = l_Lean_Syntax_node2(v_a_2498_, v___x_2470_, v___x_2401_, v___x_2538_);
v___x_2541_ = l_Lean_Syntax_node4(v_a_2498_, v___x_2537_, v___x_2538_, v___x_2515_, v___x_2539_, v___x_2540_);
v___x_2542_ = l_Lean_Syntax_node2(v_a_2498_, v___x_2535_, v___x_2536_, v___x_2541_);
v___x_2543_ = l_Lean_Syntax_node4(v_a_2498_, v___x_2506_, v___x_2512_, v___x_2533_, v___x_2515_, v___x_2542_);
if (v_isShared_2501_ == 0)
{
lean_ctor_set(v___x_2500_, 0, v___x_2543_);
v___x_2545_ = v___x_2500_;
goto v_reusejp_2544_;
}
else
{
lean_object* v_reuseFailAlloc_2546_; 
v_reuseFailAlloc_2546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2546_, 0, v___x_2543_);
v___x_2545_ = v_reuseFailAlloc_2546_;
goto v_reusejp_2544_;
}
v_reusejp_2544_:
{
return v___x_2545_;
}
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
lean_object* v_a_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2569_; 
lean_dec(v___x_2385_);
lean_del_object(v___x_2383_);
lean_dec(v_name_2381_);
lean_dec(v_a_2375_);
lean_dec(v_generatorType_2365_);
v_a_2562_ = lean_ctor_get(v___x_2389_, 0);
v_isSharedCheck_2569_ = !lean_is_exclusive(v___x_2389_);
if (v_isSharedCheck_2569_ == 0)
{
v___x_2564_ = v___x_2389_;
v_isShared_2565_ = v_isSharedCheck_2569_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_a_2562_);
lean_dec(v___x_2389_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2569_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2567_; 
if (v_isShared_2565_ == 0)
{
v___x_2567_ = v___x_2564_;
goto v_reusejp_2566_;
}
else
{
lean_object* v_reuseFailAlloc_2568_; 
v_reuseFailAlloc_2568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2568_, 0, v_a_2562_);
v___x_2567_ = v_reuseFailAlloc_2568_;
goto v_reusejp_2566_;
}
v_reusejp_2566_:
{
return v___x_2567_;
}
}
}
}
}
else
{
lean_object* v_a_2573_; lean_object* v___x_2575_; uint8_t v_isShared_2576_; uint8_t v_isSharedCheck_2580_; 
lean_dec(v_a_2375_);
lean_dec(v_generatorType_2365_);
lean_dec_ref(v_inductiveVal_2364_);
v_a_2573_ = lean_ctor_get(v___x_2377_, 0);
v_isSharedCheck_2580_ = !lean_is_exclusive(v___x_2377_);
if (v_isSharedCheck_2580_ == 0)
{
v___x_2575_ = v___x_2377_;
v_isShared_2576_ = v_isSharedCheck_2580_;
goto v_resetjp_2574_;
}
else
{
lean_inc(v_a_2573_);
lean_dec(v___x_2377_);
v___x_2575_ = lean_box(0);
v_isShared_2576_ = v_isSharedCheck_2580_;
goto v_resetjp_2574_;
}
v_resetjp_2574_:
{
lean_object* v___x_2578_; 
if (v_isShared_2576_ == 0)
{
v___x_2578_ = v___x_2575_;
goto v_reusejp_2577_;
}
else
{
lean_object* v_reuseFailAlloc_2579_; 
v_reuseFailAlloc_2579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2579_, 0, v_a_2573_);
v___x_2578_ = v_reuseFailAlloc_2579_;
goto v_reusejp_2577_;
}
v_reusejp_2577_:
{
return v___x_2578_;
}
}
}
}
else
{
lean_object* v_a_2581_; lean_object* v___x_2583_; uint8_t v_isShared_2584_; uint8_t v_isSharedCheck_2588_; 
lean_dec(v_generatorType_2365_);
lean_dec_ref(v_inductiveVal_2364_);
v_a_2581_ = lean_ctor_get(v___x_2374_, 0);
v_isSharedCheck_2588_ = !lean_is_exclusive(v___x_2374_);
if (v_isSharedCheck_2588_ == 0)
{
v___x_2583_ = v___x_2374_;
v_isShared_2584_ = v_isSharedCheck_2588_;
goto v_resetjp_2582_;
}
else
{
lean_inc(v_a_2581_);
lean_dec(v___x_2374_);
v___x_2583_ = lean_box(0);
v_isShared_2584_ = v_isSharedCheck_2588_;
goto v_resetjp_2582_;
}
v_resetjp_2582_:
{
lean_object* v___x_2586_; 
if (v_isShared_2584_ == 0)
{
v___x_2586_ = v___x_2583_;
goto v_reusejp_2585_;
}
else
{
lean_object* v_reuseFailAlloc_2587_; 
v_reuseFailAlloc_2587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2587_, 0, v_a_2581_);
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
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___boxed(lean_object* v_header_2589_, lean_object* v_inductiveVal_2590_, lean_object* v_generatorType_2591_, lean_object* v_a_2592_, lean_object* v_a_2593_, lean_object* v_a_2594_, lean_object* v_a_2595_, lean_object* v_a_2596_, lean_object* v_a_2597_, lean_object* v_a_2598_){
_start:
{
lean_object* v_res_2599_; 
v_res_2599_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody(v_header_2589_, v_inductiveVal_2590_, v_generatorType_2591_, v_a_2592_, v_a_2593_, v_a_2594_, v_a_2595_, v_a_2596_, v_a_2597_);
lean_dec(v_a_2597_);
lean_dec_ref(v_a_2596_);
lean_dec(v_a_2595_);
lean_dec_ref(v_a_2594_);
lean_dec(v_a_2593_);
lean_dec_ref(v_a_2592_);
lean_dec_ref(v_header_2589_);
return v_res_2599_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1(lean_object* v_targetTypeName_2600_, lean_object* v___x_2601_, lean_object* v___x_2602_, lean_object* v_as_2603_, size_t v_sz_2604_, size_t v_i_2605_, lean_object* v_b_2606_, lean_object* v___y_2607_, lean_object* v___y_2608_, lean_object* v___y_2609_, lean_object* v___y_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_){
_start:
{
lean_object* v___x_2614_; 
v___x_2614_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg(v_targetTypeName_2600_, v___x_2601_, v___x_2602_, v_as_2603_, v_sz_2604_, v_i_2605_, v_b_2606_, v___y_2611_);
return v___x_2614_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___boxed(lean_object* v_targetTypeName_2615_, lean_object* v___x_2616_, lean_object* v___x_2617_, lean_object* v_as_2618_, lean_object* v_sz_2619_, lean_object* v_i_2620_, lean_object* v_b_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_){
_start:
{
size_t v_sz_boxed_2629_; size_t v_i_boxed_2630_; lean_object* v_res_2631_; 
v_sz_boxed_2629_ = lean_unbox_usize(v_sz_2619_);
lean_dec(v_sz_2619_);
v_i_boxed_2630_ = lean_unbox_usize(v_i_2620_);
lean_dec(v_i_2620_);
v_res_2631_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1(v_targetTypeName_2615_, v___x_2616_, v___x_2617_, v_as_2618_, v_sz_boxed_2629_, v_i_boxed_2630_, v_b_2621_, v___y_2622_, v___y_2623_, v___y_2624_, v___y_2625_, v___y_2626_, v___y_2627_);
lean_dec(v___y_2627_);
lean_dec_ref(v___y_2626_);
lean_dec(v___y_2625_);
lean_dec_ref(v___y_2624_);
lean_dec(v___y_2623_);
lean_dec_ref(v___y_2622_);
lean_dec_ref(v_as_2618_);
lean_dec(v___x_2616_);
lean_dec(v_targetTypeName_2615_);
return v_res_2631_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5(lean_object* v_header_2632_, lean_object* v_inductiveVal_2633_, lean_object* v_targetTypeName_2634_, lean_object* v___x_2635_, lean_object* v_as_2636_, lean_object* v_as_x27_2637_, lean_object* v_b_2638_, lean_object* v_a_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_){
_start:
{
lean_object* v___x_2647_; 
v___x_2647_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg(v_inductiveVal_2633_, v_targetTypeName_2634_, v___x_2635_, v_as_x27_2637_, v_b_2638_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
return v___x_2647_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___boxed(lean_object* v_header_2648_, lean_object* v_inductiveVal_2649_, lean_object* v_targetTypeName_2650_, lean_object* v___x_2651_, lean_object* v_as_2652_, lean_object* v_as_x27_2653_, lean_object* v_b_2654_, lean_object* v_a_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_, lean_object* v___y_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_){
_start:
{
lean_object* v_res_2663_; 
v_res_2663_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5(v_header_2648_, v_inductiveVal_2649_, v_targetTypeName_2650_, v___x_2651_, v_as_2652_, v_as_x27_2653_, v_b_2654_, v_a_2655_, v___y_2656_, v___y_2657_, v___y_2658_, v___y_2659_, v___y_2660_, v___y_2661_);
lean_dec(v___y_2661_);
lean_dec_ref(v___y_2660_);
lean_dec(v___y_2659_);
lean_dec_ref(v___y_2658_);
lean_dec(v___y_2657_);
lean_dec_ref(v___y_2656_);
lean_dec(v_as_x27_2653_);
lean_dec(v_as_2652_);
lean_dec_ref(v_header_2648_);
return v_res_2663_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7(lean_object* v_00_u03b1_2664_, lean_object* v_msg_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_, lean_object* v___y_2670_, lean_object* v___y_2671_){
_start:
{
lean_object* v___x_2673_; 
v___x_2673_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(v_msg_2665_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_, v___y_2670_, v___y_2671_);
return v___x_2673_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___boxed(lean_object* v_00_u03b1_2674_, lean_object* v_msg_2675_, lean_object* v___y_2676_, lean_object* v___y_2677_, lean_object* v___y_2678_, lean_object* v___y_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_){
_start:
{
lean_object* v_res_2683_; 
v_res_2683_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7(v_00_u03b1_2674_, v_msg_2675_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_, v___y_2680_, v___y_2681_);
lean_dec(v___y_2681_);
lean_dec_ref(v___y_2680_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
return v_res_2683_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9(lean_object* v_msgData_2684_, lean_object* v_macroStack_2685_, lean_object* v___y_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_){
_start:
{
lean_object* v___x_2693_; 
v___x_2693_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg(v_msgData_2684_, v_macroStack_2685_, v___y_2690_);
return v___x_2693_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___boxed(lean_object* v_msgData_2694_, lean_object* v_macroStack_2695_, lean_object* v___y_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_, lean_object* v___y_2700_, lean_object* v___y_2701_, lean_object* v___y_2702_){
_start:
{
lean_object* v_res_2703_; 
v_res_2703_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9(v_msgData_2694_, v_macroStack_2695_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_, v___y_2700_, v___y_2701_);
lean_dec(v___y_2701_);
lean_dec_ref(v___y_2700_);
lean_dec(v___y_2699_);
lean_dec_ref(v___y_2698_);
lean_dec(v___y_2697_);
lean_dec_ref(v___y_2696_);
return v_res_2703_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14(lean_object* v_00_u03b1_2704_, lean_object* v_name_2705_, uint8_t v_bi_2706_, lean_object* v_type_2707_, lean_object* v_k_2708_, uint8_t v_kind_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_, lean_object* v___y_2714_, lean_object* v___y_2715_){
_start:
{
lean_object* v___x_2717_; 
v___x_2717_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___redArg(v_name_2705_, v_bi_2706_, v_type_2707_, v_k_2708_, v_kind_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_, v___y_2714_, v___y_2715_);
return v___x_2717_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14___boxed(lean_object* v_00_u03b1_2718_, lean_object* v_name_2719_, lean_object* v_bi_2720_, lean_object* v_type_2721_, lean_object* v_k_2722_, lean_object* v_kind_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_){
_start:
{
uint8_t v_bi_boxed_2731_; uint8_t v_kind_boxed_2732_; lean_object* v_res_2733_; 
v_bi_boxed_2731_ = lean_unbox(v_bi_2720_);
v_kind_boxed_2732_ = lean_unbox(v_kind_2723_);
v_res_2733_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_withLocalDeclsDND___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__4_spec__5_spec__7_spec__11_spec__14(v_00_u03b1_2718_, v_name_2719_, v_bi_boxed_2731_, v_type_2721_, v_k_2722_, v_kind_boxed_2732_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_);
lean_dec(v___y_2729_);
lean_dec_ref(v___y_2728_);
lean_dec(v___y_2727_);
lean_dec_ref(v___y_2726_);
lean_dec(v___y_2725_);
lean_dec_ref(v___y_2724_);
return v_res_2733_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1(void){
_start:
{
lean_object* v___x_2737_; lean_object* v___x_2738_; 
v___x_2737_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__0));
v___x_2738_ = l_Lean_mkIdent(v___x_2737_);
return v___x_2738_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction(lean_object* v_ctx_2786_, lean_object* v_i_2787_, lean_object* v_a_2788_, lean_object* v_a_2789_, lean_object* v_a_2790_, lean_object* v_a_2791_, lean_object* v_a_2792_, lean_object* v_a_2793_){
_start:
{
lean_object* v_typeInfos_2795_; lean_object* v_auxFunNames_2796_; uint8_t v_usePartial_2797_; lean_object* v___x_2798_; lean_object* v_indVal_2799_; lean_object* v___x_2800_; 
v_typeInfos_2795_ = lean_ctor_get(v_ctx_2786_, 1);
v_auxFunNames_2796_ = lean_ctor_get(v_ctx_2786_, 2);
v_usePartial_2797_ = lean_ctor_get_uint8(v_ctx_2786_, sizeof(void*)*3);
v___x_2798_ = l_Lean_instInhabitedInductiveVal_default;
v_indVal_2799_ = lean_array_get_borrowed(v___x_2798_, v_typeInfos_2795_, v_i_2787_);
lean_inc(v_indVal_2799_);
v___x_2800_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryHeader(v_indVal_2799_, v_a_2788_, v_a_2789_, v_a_2790_, v_a_2791_, v_a_2792_, v_a_2793_);
if (lean_obj_tag(v___x_2800_) == 0)
{
lean_object* v_a_2801_; lean_object* v_binders_2802_; lean_object* v_argNames_2803_; lean_object* v___x_2804_; 
v_a_2801_ = lean_ctor_get(v___x_2800_, 0);
lean_inc(v_a_2801_);
lean_dec_ref_known(v___x_2800_, 1);
v_binders_2802_ = lean_ctor_get(v_a_2801_, 0);
lean_inc_ref(v_binders_2802_);
v_argNames_2803_ = lean_ctor_get(v_a_2801_, 1);
lean_inc_ref_n(v_argNames_2803_, 2);
lean_inc(v_indVal_2799_);
v___x_2804_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v_indVal_2799_, v_argNames_2803_, v_a_2792_);
if (lean_obj_tag(v___x_2804_) == 0)
{
lean_object* v_a_2805_; lean_object* v_ref_2806_; uint8_t v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; 
v_a_2805_ = lean_ctor_get(v___x_2804_, 0);
lean_inc(v_a_2805_);
lean_dec_ref_known(v___x_2804_, 1);
v_ref_2806_ = lean_ctor_get(v_a_2792_, 5);
v___x_2807_ = 0;
v___x_2808_ = l_Lean_SourceInfo_fromRef(v_ref_2806_, v___x_2807_);
v___x_2809_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__8));
v___x_2810_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__1);
v___x_2811_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
lean_inc(v___x_2808_);
v___x_2812_ = l_Lean_Syntax_node1(v___x_2808_, v___x_2811_, v_a_2805_);
v___x_2813_ = l_Lean_Syntax_node2(v___x_2808_, v___x_2809_, v___x_2810_, v___x_2812_);
lean_inc(v___x_2813_);
lean_inc(v_indVal_2799_);
v___x_2814_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody(v_a_2801_, v_indVal_2799_, v___x_2813_, v_a_2788_, v_a_2789_, v_a_2790_, v_a_2791_, v_a_2792_, v_a_2793_);
lean_dec(v_a_2801_);
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v_a_2815_; lean_object* v___x_2816_; lean_object* v_auxFunName_2817_; lean_object* v_body_2819_; lean_object* v___y_2820_; lean_object* v___y_2821_; lean_object* v___y_2822_; lean_object* v___y_2823_; lean_object* v___y_2824_; lean_object* v___y_2825_; 
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
lean_inc(v_a_2815_);
lean_dec_ref_known(v___x_2814_, 1);
v___x_2816_ = lean_box(0);
v_auxFunName_2817_ = lean_array_get_borrowed(v___x_2816_, v_auxFunNames_2796_, v_i_2787_);
if (v_usePartial_2797_ == 0)
{
lean_dec_ref(v_argNames_2803_);
v_body_2819_ = v_a_2815_;
v___y_2820_ = v_a_2788_;
v___y_2821_ = v_a_2789_;
v___y_2822_ = v_a_2790_;
v___y_2823_ = v_a_2791_;
v___y_2824_ = v_a_2792_;
v___y_2825_ = v_a_2793_;
goto v___jp_2818_;
}
else
{
lean_object* v___x_2915_; lean_object* v___x_2916_; 
v___x_2915_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__10));
v___x_2916_ = l_Lean_Elab_Deriving_mkLocalInstanceLetDecls(v_ctx_2786_, v___x_2915_, v_argNames_2803_, v_a_2788_, v_a_2789_, v_a_2790_, v_a_2791_, v_a_2792_, v_a_2793_);
if (lean_obj_tag(v___x_2916_) == 0)
{
lean_object* v_a_2917_; lean_object* v___x_2918_; 
v_a_2917_ = lean_ctor_get(v___x_2916_, 0);
lean_inc(v_a_2917_);
lean_dec_ref_known(v___x_2916_, 1);
v___x_2918_ = l_Lean_Elab_Deriving_mkLet(v_a_2917_, v_a_2815_, v_a_2788_, v_a_2789_, v_a_2790_, v_a_2791_, v_a_2792_, v_a_2793_);
lean_dec(v_a_2917_);
if (lean_obj_tag(v___x_2918_) == 0)
{
lean_object* v_a_2919_; 
v_a_2919_ = lean_ctor_get(v___x_2918_, 0);
lean_inc(v_a_2919_);
lean_dec_ref_known(v___x_2918_, 1);
v_body_2819_ = v_a_2919_;
v___y_2820_ = v_a_2788_;
v___y_2821_ = v_a_2789_;
v___y_2822_ = v_a_2790_;
v___y_2823_ = v_a_2791_;
v___y_2824_ = v_a_2792_;
v___y_2825_ = v_a_2793_;
goto v___jp_2818_;
}
else
{
lean_dec(v___x_2813_);
lean_dec_ref(v_binders_2802_);
return v___x_2918_;
}
}
else
{
lean_object* v_a_2920_; lean_object* v___x_2922_; uint8_t v_isShared_2923_; uint8_t v_isSharedCheck_2927_; 
lean_dec(v_a_2815_);
lean_dec(v___x_2813_);
lean_dec_ref(v_binders_2802_);
v_a_2920_ = lean_ctor_get(v___x_2916_, 0);
v_isSharedCheck_2927_ = !lean_is_exclusive(v___x_2916_);
if (v_isSharedCheck_2927_ == 0)
{
v___x_2922_ = v___x_2916_;
v_isShared_2923_ = v_isSharedCheck_2927_;
goto v_resetjp_2921_;
}
else
{
lean_inc(v_a_2920_);
lean_dec(v___x_2916_);
v___x_2922_ = lean_box(0);
v_isShared_2923_ = v_isSharedCheck_2927_;
goto v_resetjp_2921_;
}
v_resetjp_2921_:
{
lean_object* v___x_2925_; 
if (v_isShared_2923_ == 0)
{
v___x_2925_ = v___x_2922_;
goto v_reusejp_2924_;
}
else
{
lean_object* v_reuseFailAlloc_2926_; 
v_reuseFailAlloc_2926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2926_, 0, v_a_2920_);
v___x_2925_ = v_reuseFailAlloc_2926_;
goto v_reusejp_2924_;
}
v_reusejp_2924_:
{
return v___x_2925_;
}
}
}
}
v___jp_2818_:
{
if (v_usePartial_2797_ == 0)
{
lean_object* v___x_2826_; lean_object* v_a_2827_; lean_object* v___x_2829_; uint8_t v_isShared_2830_; uint8_t v_isSharedCheck_2867_; 
v___x_2826_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v___y_2820_, v___y_2821_, v___y_2822_, v___y_2823_, v___y_2824_, v___y_2825_);
v_a_2827_ = lean_ctor_get(v___x_2826_, 0);
v_isSharedCheck_2867_ = !lean_is_exclusive(v___x_2826_);
if (v_isSharedCheck_2867_ == 0)
{
v___x_2829_ = v___x_2826_;
v_isShared_2830_ = v_isSharedCheck_2867_;
goto v_resetjp_2828_;
}
else
{
lean_inc(v_a_2827_);
lean_dec(v___x_2826_);
v___x_2829_ = lean_box(0);
v_isShared_2830_ = v_isSharedCheck_2867_;
goto v_resetjp_2828_;
}
v_resetjp_2828_:
{
lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2865_; 
v___x_2831_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2));
v___x_2832_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3));
v___x_2833_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
lean_inc_n(v_a_2827_, 15);
v___x_2834_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2834_, 0, v_a_2827_);
lean_ctor_set(v___x_2834_, 1, v___x_2811_);
lean_ctor_set(v___x_2834_, 2, v___x_2833_);
lean_inc_ref_n(v___x_2834_, 11);
v___x_2835_ = l_Lean_Syntax_node7(v_a_2827_, v___x_2832_, v___x_2834_, v___x_2834_, v___x_2834_, v___x_2834_, v___x_2834_, v___x_2834_, v___x_2834_);
v___x_2836_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5));
v___x_2837_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__6));
v___x_2838_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2838_, 0, v_a_2827_);
lean_ctor_set(v___x_2838_, 1, v___x_2837_);
v___x_2839_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8));
lean_inc(v_auxFunName_2817_);
v___x_2840_ = l_Lean_mkIdent(v_auxFunName_2817_);
v___x_2841_ = l_Lean_Syntax_node2(v_a_2827_, v___x_2839_, v___x_2840_, v___x_2834_);
v___x_2842_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10));
v___x_2843_ = l_Array_append___redArg(v___x_2833_, v_binders_2802_);
lean_dec_ref(v_binders_2802_);
v___x_2844_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2844_, 0, v_a_2827_);
lean_ctor_set(v___x_2844_, 1, v___x_2811_);
lean_ctor_set(v___x_2844_, 2, v___x_2843_);
v___x_2845_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51));
v___x_2846_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8));
v___x_2847_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2847_, 0, v_a_2827_);
lean_ctor_set(v___x_2847_, 1, v___x_2846_);
v___x_2848_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12));
v___x_2849_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19);
v___x_2850_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__13));
v___x_2851_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2851_, 0, v_a_2827_);
lean_ctor_set(v___x_2851_, 1, v___x_2850_);
v___x_2852_ = l_Lean_Syntax_node3(v_a_2827_, v___x_2848_, v___x_2849_, v___x_2851_, v___x_2813_);
v___x_2853_ = l_Lean_Syntax_node2(v_a_2827_, v___x_2845_, v___x_2847_, v___x_2852_);
v___x_2854_ = l_Lean_Syntax_node1(v_a_2827_, v___x_2811_, v___x_2853_);
v___x_2855_ = l_Lean_Syntax_node2(v_a_2827_, v___x_2842_, v___x_2844_, v___x_2854_);
v___x_2856_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14));
v___x_2857_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10));
v___x_2858_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2858_, 0, v_a_2827_);
lean_ctor_set(v___x_2858_, 1, v___x_2857_);
v___x_2859_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52));
v___x_2860_ = l_Lean_Syntax_node2(v_a_2827_, v___x_2859_, v___x_2834_, v___x_2834_);
v___x_2861_ = l_Lean_Syntax_node4(v_a_2827_, v___x_2856_, v___x_2858_, v_body_2819_, v___x_2860_, v___x_2834_);
v___x_2862_ = l_Lean_Syntax_node5(v_a_2827_, v___x_2836_, v___x_2838_, v___x_2841_, v___x_2855_, v___x_2861_, v___x_2834_);
v___x_2863_ = l_Lean_Syntax_node2(v_a_2827_, v___x_2831_, v___x_2835_, v___x_2862_);
if (v_isShared_2830_ == 0)
{
lean_ctor_set(v___x_2829_, 0, v___x_2863_);
v___x_2865_ = v___x_2829_;
goto v_reusejp_2864_;
}
else
{
lean_object* v_reuseFailAlloc_2866_; 
v_reuseFailAlloc_2866_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2866_, 0, v___x_2863_);
v___x_2865_ = v_reuseFailAlloc_2866_;
goto v_reusejp_2864_;
}
v_reusejp_2864_:
{
return v___x_2865_;
}
}
}
else
{
lean_object* v___x_2868_; lean_object* v_a_2869_; lean_object* v___x_2871_; uint8_t v_isShared_2872_; uint8_t v_isSharedCheck_2914_; 
v___x_2868_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___lam__0(v___y_2820_, v___y_2821_, v___y_2822_, v___y_2823_, v___y_2824_, v___y_2825_);
v_a_2869_ = lean_ctor_get(v___x_2868_, 0);
v_isSharedCheck_2914_ = !lean_is_exclusive(v___x_2868_);
if (v_isSharedCheck_2914_ == 0)
{
v___x_2871_ = v___x_2868_;
v_isShared_2872_ = v_isSharedCheck_2914_;
goto v_resetjp_2870_;
}
else
{
lean_inc(v_a_2869_);
lean_dec(v___x_2868_);
v___x_2871_ = lean_box(0);
v_isShared_2872_ = v_isSharedCheck_2914_;
goto v_resetjp_2870_;
}
v_resetjp_2870_:
{
lean_object* v___x_2873_; lean_object* v___x_2874_; lean_object* v___x_2875_; lean_object* v___x_2876_; lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v___x_2879_; lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2902_; lean_object* v___x_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2912_; 
v___x_2873_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__2));
v___x_2874_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__3));
v___x_2875_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
lean_inc_n(v_a_2869_, 18);
v___x_2876_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2876_, 0, v_a_2869_);
lean_ctor_set(v___x_2876_, 1, v___x_2811_);
lean_ctor_set(v___x_2876_, 2, v___x_2875_);
v___x_2877_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__15));
v___x_2878_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__16));
v___x_2879_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2879_, 0, v_a_2869_);
lean_ctor_set(v___x_2879_, 1, v___x_2877_);
v___x_2880_ = l_Lean_Syntax_node1(v_a_2869_, v___x_2878_, v___x_2879_);
v___x_2881_ = l_Lean_Syntax_node1(v_a_2869_, v___x_2811_, v___x_2880_);
lean_inc_ref_n(v___x_2876_, 10);
v___x_2882_ = l_Lean_Syntax_node7(v_a_2869_, v___x_2874_, v___x_2876_, v___x_2876_, v___x_2876_, v___x_2876_, v___x_2876_, v___x_2876_, v___x_2881_);
v___x_2883_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__5));
v___x_2884_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__6));
v___x_2885_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2885_, 0, v_a_2869_);
lean_ctor_set(v___x_2885_, 1, v___x_2884_);
v___x_2886_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__8));
lean_inc(v_auxFunName_2817_);
v___x_2887_ = l_Lean_mkIdent(v_auxFunName_2817_);
v___x_2888_ = l_Lean_Syntax_node2(v_a_2869_, v___x_2886_, v___x_2887_, v___x_2876_);
v___x_2889_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__10));
v___x_2890_ = l_Array_append___redArg(v___x_2875_, v_binders_2802_);
lean_dec_ref(v_binders_2802_);
v___x_2891_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2891_, 0, v_a_2869_);
lean_ctor_set(v___x_2891_, 1, v___x_2811_);
lean_ctor_set(v___x_2891_, 2, v___x_2890_);
v___x_2892_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__51));
v___x_2893_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__8));
v___x_2894_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2894_, 0, v_a_2869_);
lean_ctor_set(v___x_2894_, 1, v___x_2893_);
v___x_2895_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__12));
v___x_2896_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__19);
v___x_2897_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__13));
v___x_2898_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2898_, 0, v_a_2869_);
lean_ctor_set(v___x_2898_, 1, v___x_2897_);
v___x_2899_ = l_Lean_Syntax_node3(v_a_2869_, v___x_2895_, v___x_2896_, v___x_2898_, v___x_2813_);
v___x_2900_ = l_Lean_Syntax_node2(v_a_2869_, v___x_2892_, v___x_2894_, v___x_2899_);
v___x_2901_ = l_Lean_Syntax_node1(v_a_2869_, v___x_2811_, v___x_2900_);
v___x_2902_ = l_Lean_Syntax_node2(v_a_2869_, v___x_2889_, v___x_2891_, v___x_2901_);
v___x_2903_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___closed__14));
v___x_2904_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__10));
v___x_2905_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2905_, 0, v_a_2869_);
lean_ctor_set(v___x_2905_, 1, v___x_2904_);
v___x_2906_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkBody___closed__52));
v___x_2907_ = l_Lean_Syntax_node2(v_a_2869_, v___x_2906_, v___x_2876_, v___x_2876_);
v___x_2908_ = l_Lean_Syntax_node4(v_a_2869_, v___x_2903_, v___x_2905_, v_body_2819_, v___x_2907_, v___x_2876_);
v___x_2909_ = l_Lean_Syntax_node5(v_a_2869_, v___x_2883_, v___x_2885_, v___x_2888_, v___x_2902_, v___x_2908_, v___x_2876_);
v___x_2910_ = l_Lean_Syntax_node2(v_a_2869_, v___x_2873_, v___x_2882_, v___x_2909_);
if (v_isShared_2872_ == 0)
{
lean_ctor_set(v___x_2871_, 0, v___x_2910_);
v___x_2912_ = v___x_2871_;
goto v_reusejp_2911_;
}
else
{
lean_object* v_reuseFailAlloc_2913_; 
v_reuseFailAlloc_2913_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2913_, 0, v___x_2910_);
v___x_2912_ = v_reuseFailAlloc_2913_;
goto v_reusejp_2911_;
}
v_reusejp_2911_:
{
return v___x_2912_;
}
}
}
}
}
else
{
lean_dec(v___x_2813_);
lean_dec_ref(v_argNames_2803_);
lean_dec_ref(v_binders_2802_);
return v___x_2814_;
}
}
else
{
lean_dec_ref(v_argNames_2803_);
lean_dec_ref(v_binders_2802_);
lean_dec(v_a_2801_);
return v___x_2804_;
}
}
else
{
lean_object* v_a_2928_; lean_object* v___x_2930_; uint8_t v_isShared_2931_; uint8_t v_isSharedCheck_2935_; 
v_a_2928_ = lean_ctor_get(v___x_2800_, 0);
v_isSharedCheck_2935_ = !lean_is_exclusive(v___x_2800_);
if (v_isSharedCheck_2935_ == 0)
{
v___x_2930_ = v___x_2800_;
v_isShared_2931_ = v_isSharedCheck_2935_;
goto v_resetjp_2929_;
}
else
{
lean_inc(v_a_2928_);
lean_dec(v___x_2800_);
v___x_2930_ = lean_box(0);
v_isShared_2931_ = v_isSharedCheck_2935_;
goto v_resetjp_2929_;
}
v_resetjp_2929_:
{
lean_object* v___x_2933_; 
if (v_isShared_2931_ == 0)
{
v___x_2933_ = v___x_2930_;
goto v_reusejp_2932_;
}
else
{
lean_object* v_reuseFailAlloc_2934_; 
v_reuseFailAlloc_2934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2934_, 0, v_a_2928_);
v___x_2933_ = v_reuseFailAlloc_2934_;
goto v_reusejp_2932_;
}
v_reusejp_2932_:
{
return v___x_2933_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction___boxed(lean_object* v_ctx_2936_, lean_object* v_i_2937_, lean_object* v_a_2938_, lean_object* v_a_2939_, lean_object* v_a_2940_, lean_object* v_a_2941_, lean_object* v_a_2942_, lean_object* v_a_2943_, lean_object* v_a_2944_){
_start:
{
lean_object* v_res_2945_; 
v_res_2945_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction(v_ctx_2936_, v_i_2937_, v_a_2938_, v_a_2939_, v_a_2940_, v_a_2941_, v_a_2942_, v_a_2943_);
lean_dec(v_a_2943_);
lean_dec_ref(v_a_2942_);
lean_dec(v_a_2941_);
lean_dec_ref(v_a_2940_);
lean_dec(v_a_2939_);
lean_dec_ref(v_a_2938_);
lean_dec(v_i_2937_);
lean_dec_ref(v_ctx_2936_);
return v_res_2945_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg(lean_object* v_upperBound_2946_, lean_object* v_ctx_2947_, lean_object* v_a_2948_, lean_object* v_b_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_, lean_object* v___y_2955_){
_start:
{
uint8_t v___x_2957_; 
v___x_2957_ = lean_nat_dec_lt(v_a_2948_, v_upperBound_2946_);
if (v___x_2957_ == 0)
{
lean_object* v___x_2958_; 
lean_dec(v_a_2948_);
v___x_2958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2958_, 0, v_b_2949_);
return v___x_2958_;
}
else
{
lean_object* v___x_2959_; 
v___x_2959_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkAuxFunction(v_ctx_2947_, v_a_2948_, v___y_2950_, v___y_2951_, v___y_2952_, v___y_2953_, v___y_2954_, v___y_2955_);
if (lean_obj_tag(v___x_2959_) == 0)
{
lean_object* v_a_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2963_; 
v_a_2960_ = lean_ctor_get(v___x_2959_, 0);
lean_inc(v_a_2960_);
lean_dec_ref_known(v___x_2959_, 1);
v___x_2961_ = lean_array_push(v_b_2949_, v_a_2960_);
v___x_2962_ = lean_unsigned_to_nat(1u);
v___x_2963_ = lean_nat_add(v_a_2948_, v___x_2962_);
lean_dec(v_a_2948_);
v_a_2948_ = v___x_2963_;
v_b_2949_ = v___x_2961_;
goto _start;
}
else
{
lean_object* v_a_2965_; lean_object* v___x_2967_; uint8_t v_isShared_2968_; uint8_t v_isSharedCheck_2972_; 
lean_dec_ref(v_b_2949_);
lean_dec(v_a_2948_);
v_a_2965_ = lean_ctor_get(v___x_2959_, 0);
v_isSharedCheck_2972_ = !lean_is_exclusive(v___x_2959_);
if (v_isSharedCheck_2972_ == 0)
{
v___x_2967_ = v___x_2959_;
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
else
{
lean_inc(v_a_2965_);
lean_dec(v___x_2959_);
v___x_2967_ = lean_box(0);
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
v_resetjp_2966_:
{
lean_object* v___x_2970_; 
if (v_isShared_2968_ == 0)
{
v___x_2970_ = v___x_2967_;
goto v_reusejp_2969_;
}
else
{
lean_object* v_reuseFailAlloc_2971_; 
v_reuseFailAlloc_2971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2971_, 0, v_a_2965_);
v___x_2970_ = v_reuseFailAlloc_2971_;
goto v_reusejp_2969_;
}
v_reusejp_2969_:
{
return v___x_2970_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg___boxed(lean_object* v_upperBound_2973_, lean_object* v_ctx_2974_, lean_object* v_a_2975_, lean_object* v_b_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_){
_start:
{
lean_object* v_res_2984_; 
v_res_2984_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg(v_upperBound_2973_, v_ctx_2974_, v_a_2975_, v_b_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_);
lean_dec(v___y_2982_);
lean_dec_ref(v___y_2981_);
lean_dec(v___y_2980_);
lean_dec_ref(v___y_2979_);
lean_dec(v___y_2978_);
lean_dec_ref(v___y_2977_);
lean_dec_ref(v_ctx_2974_);
lean_dec(v_upperBound_2973_);
return v_res_2984_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock(lean_object* v_ctx_2992_, lean_object* v_a_2993_, lean_object* v_a_2994_, lean_object* v_a_2995_, lean_object* v_a_2996_, lean_object* v_a_2997_, lean_object* v_a_2998_){
_start:
{
lean_object* v_typeInfos_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v_auxDefs_3003_; lean_object* v___x_3004_; 
v_typeInfos_3000_ = lean_ctor_get(v_ctx_2992_, 1);
v___x_3001_ = lean_unsigned_to_nat(0u);
v___x_3002_ = lean_array_get_size(v_typeInfos_3000_);
v_auxDefs_3003_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds___closed__0));
v___x_3004_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg(v___x_3002_, v_ctx_2992_, v___x_3001_, v_auxDefs_3003_, v_a_2993_, v_a_2994_, v_a_2995_, v_a_2996_, v_a_2997_, v_a_2998_);
if (lean_obj_tag(v___x_3004_) == 0)
{
lean_object* v_a_3005_; lean_object* v___x_3007_; uint8_t v_isShared_3008_; uint8_t v_isSharedCheck_3025_; 
v_a_3005_ = lean_ctor_get(v___x_3004_, 0);
v_isSharedCheck_3025_ = !lean_is_exclusive(v___x_3004_);
if (v_isSharedCheck_3025_ == 0)
{
v___x_3007_ = v___x_3004_;
v_isShared_3008_ = v_isSharedCheck_3025_;
goto v_resetjp_3006_;
}
else
{
lean_inc(v_a_3005_);
lean_dec(v___x_3004_);
v___x_3007_ = lean_box(0);
v_isShared_3008_ = v_isSharedCheck_3025_;
goto v_resetjp_3006_;
}
v_resetjp_3006_:
{
lean_object* v_ref_3009_; uint8_t v___x_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3023_; 
v_ref_3009_ = lean_ctor_get(v_a_2997_, 5);
v___x_3010_ = 0;
v___x_3011_ = l_Lean_SourceInfo_fromRef(v_ref_3009_, v___x_3010_);
v___x_3012_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__0));
v___x_3013_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__1));
lean_inc_n(v___x_3011_, 3);
v___x_3014_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3014_, 0, v___x_3011_);
lean_ctor_set(v___x_3014_, 1, v___x_3012_);
v___x_3015_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__13));
v___x_3016_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___lam__1___closed__3);
v___x_3017_ = l_Array_append___redArg(v___x_3016_, v_a_3005_);
lean_dec(v_a_3005_);
v___x_3018_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3018_, 0, v___x_3011_);
lean_ctor_set(v___x_3018_, 1, v___x_3015_);
lean_ctor_set(v___x_3018_, 2, v___x_3017_);
v___x_3019_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___closed__2));
v___x_3020_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3020_, 0, v___x_3011_);
lean_ctor_set(v___x_3020_, 1, v___x_3019_);
v___x_3021_ = l_Lean_Syntax_node3(v___x_3011_, v___x_3013_, v___x_3014_, v___x_3018_, v___x_3020_);
if (v_isShared_3008_ == 0)
{
lean_ctor_set(v___x_3007_, 0, v___x_3021_);
v___x_3023_ = v___x_3007_;
goto v_reusejp_3022_;
}
else
{
lean_object* v_reuseFailAlloc_3024_; 
v_reuseFailAlloc_3024_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3024_, 0, v___x_3021_);
v___x_3023_ = v_reuseFailAlloc_3024_;
goto v_reusejp_3022_;
}
v_reusejp_3022_:
{
return v___x_3023_;
}
}
}
else
{
lean_object* v_a_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3033_; 
v_a_3026_ = lean_ctor_get(v___x_3004_, 0);
v_isSharedCheck_3033_ = !lean_is_exclusive(v___x_3004_);
if (v_isSharedCheck_3033_ == 0)
{
v___x_3028_ = v___x_3004_;
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_a_3026_);
lean_dec(v___x_3004_);
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
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock___boxed(lean_object* v_ctx_3034_, lean_object* v_a_3035_, lean_object* v_a_3036_, lean_object* v_a_3037_, lean_object* v_a_3038_, lean_object* v_a_3039_, lean_object* v_a_3040_, lean_object* v_a_3041_){
_start:
{
lean_object* v_res_3042_; 
v_res_3042_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock(v_ctx_3034_, v_a_3035_, v_a_3036_, v_a_3037_, v_a_3038_, v_a_3039_, v_a_3040_);
lean_dec(v_a_3040_);
lean_dec_ref(v_a_3039_);
lean_dec(v_a_3038_);
lean_dec_ref(v_a_3037_);
lean_dec(v_a_3036_);
lean_dec_ref(v_a_3035_);
lean_dec_ref(v_ctx_3034_);
return v_res_3042_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0(lean_object* v_upperBound_3043_, lean_object* v_ctx_3044_, lean_object* v_inst_3045_, lean_object* v_R_3046_, lean_object* v_a_3047_, lean_object* v_b_3048_, lean_object* v_c_3049_, lean_object* v___y_3050_, lean_object* v___y_3051_, lean_object* v___y_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_){
_start:
{
lean_object* v___x_3057_; 
v___x_3057_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___redArg(v_upperBound_3043_, v_ctx_3044_, v_a_3047_, v_b_3048_, v___y_3050_, v___y_3051_, v___y_3052_, v___y_3053_, v___y_3054_, v___y_3055_);
return v___x_3057_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0___boxed(lean_object* v_upperBound_3058_, lean_object* v_ctx_3059_, lean_object* v_inst_3060_, lean_object* v_R_3061_, lean_object* v_a_3062_, lean_object* v_b_3063_, lean_object* v_c_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_){
_start:
{
lean_object* v_res_3072_; 
v_res_3072_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock_spec__0(v_upperBound_3058_, v_ctx_3059_, v_inst_3060_, v_R_3061_, v_a_3062_, v_b_3063_, v_c_3064_, v___y_3065_, v___y_3066_, v___y_3067_, v___y_3068_, v___y_3069_, v___y_3070_);
lean_dec(v___y_3070_);
lean_dec_ref(v___y_3069_);
lean_dec(v___y_3068_);
lean_dec_ref(v___y_3067_);
lean_dec(v___y_3066_);
lean_dec_ref(v___y_3065_);
lean_dec_ref(v_ctx_3059_);
lean_dec(v_upperBound_3058_);
return v_res_3072_;
}
}
static double _init_lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_3073_; double v___x_3074_; 
v___x_3073_ = lean_unsigned_to_nat(0u);
v___x_3074_ = lean_float_of_nat(v___x_3073_);
return v___x_3074_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg(lean_object* v_cls_3077_, lean_object* v_msg_3078_, lean_object* v___y_3079_, lean_object* v___y_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_){
_start:
{
lean_object* v_ref_3084_; lean_object* v___x_3085_; lean_object* v_a_3086_; lean_object* v___x_3088_; uint8_t v_isShared_3089_; uint8_t v_isSharedCheck_3130_; 
v_ref_3084_ = lean_ctor_get(v___y_3081_, 5);
v___x_3085_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_3078_, v___y_3079_, v___y_3080_, v___y_3081_, v___y_3082_);
v_a_3086_ = lean_ctor_get(v___x_3085_, 0);
v_isSharedCheck_3130_ = !lean_is_exclusive(v___x_3085_);
if (v_isSharedCheck_3130_ == 0)
{
v___x_3088_ = v___x_3085_;
v_isShared_3089_ = v_isSharedCheck_3130_;
goto v_resetjp_3087_;
}
else
{
lean_inc(v_a_3086_);
lean_dec(v___x_3085_);
v___x_3088_ = lean_box(0);
v_isShared_3089_ = v_isSharedCheck_3130_;
goto v_resetjp_3087_;
}
v_resetjp_3087_:
{
lean_object* v___x_3090_; lean_object* v_traceState_3091_; lean_object* v_env_3092_; lean_object* v_nextMacroScope_3093_; lean_object* v_ngen_3094_; lean_object* v_auxDeclNGen_3095_; lean_object* v_cache_3096_; lean_object* v_messages_3097_; lean_object* v_infoState_3098_; lean_object* v_snapshotTasks_3099_; lean_object* v___x_3101_; uint8_t v_isShared_3102_; uint8_t v_isSharedCheck_3129_; 
v___x_3090_ = lean_st_ref_take(v___y_3082_);
v_traceState_3091_ = lean_ctor_get(v___x_3090_, 4);
v_env_3092_ = lean_ctor_get(v___x_3090_, 0);
v_nextMacroScope_3093_ = lean_ctor_get(v___x_3090_, 1);
v_ngen_3094_ = lean_ctor_get(v___x_3090_, 2);
v_auxDeclNGen_3095_ = lean_ctor_get(v___x_3090_, 3);
v_cache_3096_ = lean_ctor_get(v___x_3090_, 5);
v_messages_3097_ = lean_ctor_get(v___x_3090_, 6);
v_infoState_3098_ = lean_ctor_get(v___x_3090_, 7);
v_snapshotTasks_3099_ = lean_ctor_get(v___x_3090_, 8);
v_isSharedCheck_3129_ = !lean_is_exclusive(v___x_3090_);
if (v_isSharedCheck_3129_ == 0)
{
v___x_3101_ = v___x_3090_;
v_isShared_3102_ = v_isSharedCheck_3129_;
goto v_resetjp_3100_;
}
else
{
lean_inc(v_snapshotTasks_3099_);
lean_inc(v_infoState_3098_);
lean_inc(v_messages_3097_);
lean_inc(v_cache_3096_);
lean_inc(v_traceState_3091_);
lean_inc(v_auxDeclNGen_3095_);
lean_inc(v_ngen_3094_);
lean_inc(v_nextMacroScope_3093_);
lean_inc(v_env_3092_);
lean_dec(v___x_3090_);
v___x_3101_ = lean_box(0);
v_isShared_3102_ = v_isSharedCheck_3129_;
goto v_resetjp_3100_;
}
v_resetjp_3100_:
{
uint64_t v_tid_3103_; lean_object* v_traces_3104_; lean_object* v___x_3106_; uint8_t v_isShared_3107_; uint8_t v_isSharedCheck_3128_; 
v_tid_3103_ = lean_ctor_get_uint64(v_traceState_3091_, sizeof(void*)*1);
v_traces_3104_ = lean_ctor_get(v_traceState_3091_, 0);
v_isSharedCheck_3128_ = !lean_is_exclusive(v_traceState_3091_);
if (v_isSharedCheck_3128_ == 0)
{
v___x_3106_ = v_traceState_3091_;
v_isShared_3107_ = v_isSharedCheck_3128_;
goto v_resetjp_3105_;
}
else
{
lean_inc(v_traces_3104_);
lean_dec(v_traceState_3091_);
v___x_3106_ = lean_box(0);
v_isShared_3107_ = v_isSharedCheck_3128_;
goto v_resetjp_3105_;
}
v_resetjp_3105_:
{
lean_object* v___x_3108_; double v___x_3109_; uint8_t v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; lean_object* v___x_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3118_; 
v___x_3108_ = lean_box(0);
v___x_3109_ = lean_float_once(&lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0, &lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0_once, _init_lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__0);
v___x_3110_ = 0;
v___x_3111_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__5___redArg___lam__0___closed__10));
v___x_3112_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3112_, 0, v_cls_3077_);
lean_ctor_set(v___x_3112_, 1, v___x_3108_);
lean_ctor_set(v___x_3112_, 2, v___x_3111_);
lean_ctor_set_float(v___x_3112_, sizeof(void*)*3, v___x_3109_);
lean_ctor_set_float(v___x_3112_, sizeof(void*)*3 + 8, v___x_3109_);
lean_ctor_set_uint8(v___x_3112_, sizeof(void*)*3 + 16, v___x_3110_);
v___x_3113_ = ((lean_object*)(lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___closed__1));
v___x_3114_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3114_, 0, v___x_3112_);
lean_ctor_set(v___x_3114_, 1, v_a_3086_);
lean_ctor_set(v___x_3114_, 2, v___x_3113_);
lean_inc(v_ref_3084_);
v___x_3115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3115_, 0, v_ref_3084_);
lean_ctor_set(v___x_3115_, 1, v___x_3114_);
v___x_3116_ = l_Lean_PersistentArray_push___redArg(v_traces_3104_, v___x_3115_);
if (v_isShared_3107_ == 0)
{
lean_ctor_set(v___x_3106_, 0, v___x_3116_);
v___x_3118_ = v___x_3106_;
goto v_reusejp_3117_;
}
else
{
lean_object* v_reuseFailAlloc_3127_; 
v_reuseFailAlloc_3127_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3127_, 0, v___x_3116_);
lean_ctor_set_uint64(v_reuseFailAlloc_3127_, sizeof(void*)*1, v_tid_3103_);
v___x_3118_ = v_reuseFailAlloc_3127_;
goto v_reusejp_3117_;
}
v_reusejp_3117_:
{
lean_object* v___x_3120_; 
if (v_isShared_3102_ == 0)
{
lean_ctor_set(v___x_3101_, 4, v___x_3118_);
v___x_3120_ = v___x_3101_;
goto v_reusejp_3119_;
}
else
{
lean_object* v_reuseFailAlloc_3126_; 
v_reuseFailAlloc_3126_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3126_, 0, v_env_3092_);
lean_ctor_set(v_reuseFailAlloc_3126_, 1, v_nextMacroScope_3093_);
lean_ctor_set(v_reuseFailAlloc_3126_, 2, v_ngen_3094_);
lean_ctor_set(v_reuseFailAlloc_3126_, 3, v_auxDeclNGen_3095_);
lean_ctor_set(v_reuseFailAlloc_3126_, 4, v___x_3118_);
lean_ctor_set(v_reuseFailAlloc_3126_, 5, v_cache_3096_);
lean_ctor_set(v_reuseFailAlloc_3126_, 6, v_messages_3097_);
lean_ctor_set(v_reuseFailAlloc_3126_, 7, v_infoState_3098_);
lean_ctor_set(v_reuseFailAlloc_3126_, 8, v_snapshotTasks_3099_);
v___x_3120_ = v_reuseFailAlloc_3126_;
goto v_reusejp_3119_;
}
v_reusejp_3119_:
{
lean_object* v___x_3121_; lean_object* v___x_3122_; lean_object* v___x_3124_; 
v___x_3121_ = lean_st_ref_set(v___y_3082_, v___x_3120_);
v___x_3122_ = lean_box(0);
if (v_isShared_3089_ == 0)
{
lean_ctor_set(v___x_3088_, 0, v___x_3122_);
v___x_3124_ = v___x_3088_;
goto v_reusejp_3123_;
}
else
{
lean_object* v_reuseFailAlloc_3125_; 
v_reuseFailAlloc_3125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3125_, 0, v___x_3122_);
v___x_3124_ = v_reuseFailAlloc_3125_;
goto v_reusejp_3123_;
}
v_reusejp_3123_:
{
return v___x_3124_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg___boxed(lean_object* v_cls_3131_, lean_object* v_msg_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
lean_object* v_res_3138_; 
v_res_3138_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg(v_cls_3131_, v_msg_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_);
lean_dec(v___y_3136_);
lean_dec_ref(v___y_3135_);
lean_dec(v___y_3134_);
lean_dec_ref(v___y_3133_);
return v_res_3138_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__0(lean_object* v_a_3139_, lean_object* v_a_3140_){
_start:
{
if (lean_obj_tag(v_a_3139_) == 0)
{
lean_object* v___x_3141_; 
v___x_3141_ = l_List_reverse___redArg(v_a_3140_);
return v___x_3141_;
}
else
{
lean_object* v_head_3142_; lean_object* v_tail_3143_; lean_object* v___x_3145_; uint8_t v_isShared_3146_; uint8_t v_isSharedCheck_3152_; 
v_head_3142_ = lean_ctor_get(v_a_3139_, 0);
v_tail_3143_ = lean_ctor_get(v_a_3139_, 1);
v_isSharedCheck_3152_ = !lean_is_exclusive(v_a_3139_);
if (v_isSharedCheck_3152_ == 0)
{
v___x_3145_ = v_a_3139_;
v_isShared_3146_ = v_isSharedCheck_3152_;
goto v_resetjp_3144_;
}
else
{
lean_inc(v_tail_3143_);
lean_inc(v_head_3142_);
lean_dec(v_a_3139_);
v___x_3145_ = lean_box(0);
v_isShared_3146_ = v_isSharedCheck_3152_;
goto v_resetjp_3144_;
}
v_resetjp_3144_:
{
lean_object* v___x_3147_; lean_object* v___x_3149_; 
v___x_3147_ = l_Lean_MessageData_ofSyntax(v_head_3142_);
if (v_isShared_3146_ == 0)
{
lean_ctor_set(v___x_3145_, 1, v_a_3140_);
lean_ctor_set(v___x_3145_, 0, v___x_3147_);
v___x_3149_ = v___x_3145_;
goto v_reusejp_3148_;
}
else
{
lean_object* v_reuseFailAlloc_3151_; 
v_reuseFailAlloc_3151_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3151_, 0, v___x_3147_);
lean_ctor_set(v_reuseFailAlloc_3151_, 1, v_a_3140_);
v___x_3149_ = v_reuseFailAlloc_3151_;
goto v_reusejp_3148_;
}
v_reusejp_3148_:
{
v_a_3139_ = v_tail_3143_;
v_a_3140_ = v___x_3149_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5(void){
_start:
{
lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; 
v___x_3162_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2));
v___x_3163_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__4));
v___x_3164_ = l_Lean_Name_append(v___x_3163_, v___x_3162_);
return v___x_3164_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7(void){
_start:
{
lean_object* v___x_3166_; lean_object* v___x_3167_; 
v___x_3166_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__6));
v___x_3167_ = l_Lean_stringToMessageData(v___x_3166_);
return v___x_3167_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd(lean_object* v_declName_3168_, lean_object* v_a_3169_, lean_object* v_a_3170_, lean_object* v_a_3171_, lean_object* v_a_3172_, lean_object* v_a_3173_, lean_object* v_a_3174_){
_start:
{
lean_object* v___x_3176_; lean_object* v___x_3177_; uint8_t v___x_3178_; lean_object* v___x_3179_; 
v___x_3176_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2));
v___x_3177_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__1___redArg___closed__10));
v___x_3178_ = 1;
lean_inc(v_declName_3168_);
v___x_3179_ = l_Lean_Elab_Deriving_mkContext(v___x_3176_, v___x_3177_, v_declName_3168_, v___x_3178_, v_a_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
if (lean_obj_tag(v___x_3179_) == 0)
{
lean_object* v_a_3180_; lean_object* v___x_3181_; 
v_a_3180_ = lean_ctor_get(v___x_3179_, 0);
lean_inc(v_a_3180_);
lean_dec_ref_known(v___x_3179_, 1);
v___x_3181_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkMutualBlock(v_a_3180_, v_a_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
if (lean_obj_tag(v___x_3181_) == 0)
{
lean_object* v_a_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; 
v_a_3182_ = lean_ctor_get(v___x_3181_, 0);
lean_inc(v_a_3182_);
lean_dec_ref_known(v___x_3181_, 1);
v___x_3183_ = lean_unsigned_to_nat(1u);
v___x_3184_ = lean_mk_empty_array_with_capacity(v___x_3183_);
lean_inc_ref(v___x_3184_);
v___x_3185_ = lean_array_push(v___x_3184_, v_declName_3168_);
v___x_3186_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds(v_a_3180_, v___x_3185_, v___x_3178_, v_a_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
lean_dec_ref(v___x_3185_);
lean_dec(v_a_3180_);
if (lean_obj_tag(v___x_3186_) == 0)
{
lean_object* v_options_3187_; lean_object* v_a_3188_; lean_object* v___x_3190_; uint8_t v_isShared_3191_; uint8_t v_isSharedCheck_3228_; 
v_options_3187_ = lean_ctor_get(v_a_3173_, 2);
v_a_3188_ = lean_ctor_get(v___x_3186_, 0);
v_isSharedCheck_3228_ = !lean_is_exclusive(v___x_3186_);
if (v_isSharedCheck_3228_ == 0)
{
v___x_3190_ = v___x_3186_;
v_isShared_3191_ = v_isSharedCheck_3228_;
goto v_resetjp_3189_;
}
else
{
lean_inc(v_a_3188_);
lean_dec(v___x_3186_);
v___x_3190_ = lean_box(0);
v_isShared_3191_ = v_isSharedCheck_3228_;
goto v_resetjp_3189_;
}
v_resetjp_3189_:
{
lean_object* v_inheritedTraceOptions_3192_; uint8_t v_hasTrace_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; 
v_inheritedTraceOptions_3192_ = lean_ctor_get(v_a_3173_, 13);
v_hasTrace_3193_ = lean_ctor_get_uint8(v_options_3187_, sizeof(void*)*1);
v___x_3194_ = lean_array_push(v___x_3184_, v_a_3182_);
v___x_3195_ = l_Array_append___redArg(v___x_3194_, v_a_3188_);
lean_dec(v_a_3188_);
if (v_hasTrace_3193_ == 0)
{
lean_object* v___x_3197_; 
if (v_isShared_3191_ == 0)
{
lean_ctor_set(v___x_3190_, 0, v___x_3195_);
v___x_3197_ = v___x_3190_;
goto v_reusejp_3196_;
}
else
{
lean_object* v_reuseFailAlloc_3198_; 
v_reuseFailAlloc_3198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3198_, 0, v___x_3195_);
v___x_3197_ = v_reuseFailAlloc_3198_;
goto v_reusejp_3196_;
}
v_reusejp_3196_:
{
return v___x_3197_;
}
}
else
{
lean_object* v___x_3199_; lean_object* v___x_3200_; uint8_t v___x_3201_; 
v___x_3199_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__2));
v___x_3200_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__5);
v___x_3201_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3192_, v_options_3187_, v___x_3200_);
if (v___x_3201_ == 0)
{
lean_object* v___x_3203_; 
if (v_isShared_3191_ == 0)
{
lean_ctor_set(v___x_3190_, 0, v___x_3195_);
v___x_3203_ = v___x_3190_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3204_; 
v_reuseFailAlloc_3204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3204_, 0, v___x_3195_);
v___x_3203_ = v_reuseFailAlloc_3204_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
return v___x_3203_;
}
}
else
{
lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; lean_object* v___x_3208_; lean_object* v___x_3209_; lean_object* v___x_3210_; lean_object* v___x_3211_; 
lean_del_object(v___x_3190_);
v___x_3205_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___closed__7);
lean_inc_ref(v___x_3195_);
v___x_3206_ = lean_array_to_list(v___x_3195_);
v___x_3207_ = lean_box(0);
v___x_3208_ = lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__0(v___x_3206_, v___x_3207_);
v___x_3209_ = l_Lean_MessageData_ofList(v___x_3208_);
v___x_3210_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3210_, 0, v___x_3205_);
lean_ctor_set(v___x_3210_, 1, v___x_3209_);
v___x_3211_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg(v___x_3199_, v___x_3210_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
if (lean_obj_tag(v___x_3211_) == 0)
{
lean_object* v___x_3213_; uint8_t v_isShared_3214_; uint8_t v_isSharedCheck_3218_; 
v_isSharedCheck_3218_ = !lean_is_exclusive(v___x_3211_);
if (v_isSharedCheck_3218_ == 0)
{
lean_object* v_unused_3219_; 
v_unused_3219_ = lean_ctor_get(v___x_3211_, 0);
lean_dec(v_unused_3219_);
v___x_3213_ = v___x_3211_;
v_isShared_3214_ = v_isSharedCheck_3218_;
goto v_resetjp_3212_;
}
else
{
lean_dec(v___x_3211_);
v___x_3213_ = lean_box(0);
v_isShared_3214_ = v_isSharedCheck_3218_;
goto v_resetjp_3212_;
}
v_resetjp_3212_:
{
lean_object* v___x_3216_; 
if (v_isShared_3214_ == 0)
{
lean_ctor_set(v___x_3213_, 0, v___x_3195_);
v___x_3216_ = v___x_3213_;
goto v_reusejp_3215_;
}
else
{
lean_object* v_reuseFailAlloc_3217_; 
v_reuseFailAlloc_3217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3217_, 0, v___x_3195_);
v___x_3216_ = v_reuseFailAlloc_3217_;
goto v_reusejp_3215_;
}
v_reusejp_3215_:
{
return v___x_3216_;
}
}
}
else
{
lean_object* v_a_3220_; lean_object* v___x_3222_; uint8_t v_isShared_3223_; uint8_t v_isSharedCheck_3227_; 
lean_dec_ref(v___x_3195_);
v_a_3220_ = lean_ctor_get(v___x_3211_, 0);
v_isSharedCheck_3227_ = !lean_is_exclusive(v___x_3211_);
if (v_isSharedCheck_3227_ == 0)
{
v___x_3222_ = v___x_3211_;
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
else
{
lean_inc(v_a_3220_);
lean_dec(v___x_3211_);
v___x_3222_ = lean_box(0);
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
v_resetjp_3221_:
{
lean_object* v___x_3225_; 
if (v_isShared_3223_ == 0)
{
v___x_3225_ = v___x_3222_;
goto v_reusejp_3224_;
}
else
{
lean_object* v_reuseFailAlloc_3226_; 
v_reuseFailAlloc_3226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3226_, 0, v_a_3220_);
v___x_3225_ = v_reuseFailAlloc_3226_;
goto v_reusejp_3224_;
}
v_reusejp_3224_:
{
return v___x_3225_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3229_; lean_object* v___x_3231_; uint8_t v_isShared_3232_; uint8_t v_isSharedCheck_3236_; 
lean_dec_ref(v___x_3184_);
lean_dec(v_a_3182_);
v_a_3229_ = lean_ctor_get(v___x_3186_, 0);
v_isSharedCheck_3236_ = !lean_is_exclusive(v___x_3186_);
if (v_isSharedCheck_3236_ == 0)
{
v___x_3231_ = v___x_3186_;
v_isShared_3232_ = v_isSharedCheck_3236_;
goto v_resetjp_3230_;
}
else
{
lean_inc(v_a_3229_);
lean_dec(v___x_3186_);
v___x_3231_ = lean_box(0);
v_isShared_3232_ = v_isSharedCheck_3236_;
goto v_resetjp_3230_;
}
v_resetjp_3230_:
{
lean_object* v___x_3234_; 
if (v_isShared_3232_ == 0)
{
v___x_3234_ = v___x_3231_;
goto v_reusejp_3233_;
}
else
{
lean_object* v_reuseFailAlloc_3235_; 
v_reuseFailAlloc_3235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3235_, 0, v_a_3229_);
v___x_3234_ = v_reuseFailAlloc_3235_;
goto v_reusejp_3233_;
}
v_reusejp_3233_:
{
return v___x_3234_;
}
}
}
}
else
{
lean_object* v_a_3237_; lean_object* v___x_3239_; uint8_t v_isShared_3240_; uint8_t v_isSharedCheck_3244_; 
lean_dec(v_a_3180_);
lean_dec(v_declName_3168_);
v_a_3237_ = lean_ctor_get(v___x_3181_, 0);
v_isSharedCheck_3244_ = !lean_is_exclusive(v___x_3181_);
if (v_isSharedCheck_3244_ == 0)
{
v___x_3239_ = v___x_3181_;
v_isShared_3240_ = v_isSharedCheck_3244_;
goto v_resetjp_3238_;
}
else
{
lean_inc(v_a_3237_);
lean_dec(v___x_3181_);
v___x_3239_ = lean_box(0);
v_isShared_3240_ = v_isSharedCheck_3244_;
goto v_resetjp_3238_;
}
v_resetjp_3238_:
{
lean_object* v___x_3242_; 
if (v_isShared_3240_ == 0)
{
v___x_3242_ = v___x_3239_;
goto v_reusejp_3241_;
}
else
{
lean_object* v_reuseFailAlloc_3243_; 
v_reuseFailAlloc_3243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3243_, 0, v_a_3237_);
v___x_3242_ = v_reuseFailAlloc_3243_;
goto v_reusejp_3241_;
}
v_reusejp_3241_:
{
return v___x_3242_;
}
}
}
}
else
{
lean_object* v_a_3245_; lean_object* v___x_3247_; uint8_t v_isShared_3248_; uint8_t v_isSharedCheck_3252_; 
lean_dec(v_declName_3168_);
v_a_3245_ = lean_ctor_get(v___x_3179_, 0);
v_isSharedCheck_3252_ = !lean_is_exclusive(v___x_3179_);
if (v_isSharedCheck_3252_ == 0)
{
v___x_3247_ = v___x_3179_;
v_isShared_3248_ = v_isSharedCheck_3252_;
goto v_resetjp_3246_;
}
else
{
lean_inc(v_a_3245_);
lean_dec(v___x_3179_);
v___x_3247_ = lean_box(0);
v_isShared_3248_ = v_isSharedCheck_3252_;
goto v_resetjp_3246_;
}
v_resetjp_3246_:
{
lean_object* v___x_3250_; 
if (v_isShared_3248_ == 0)
{
v___x_3250_ = v___x_3247_;
goto v_reusejp_3249_;
}
else
{
lean_object* v_reuseFailAlloc_3251_; 
v_reuseFailAlloc_3251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3251_, 0, v_a_3245_);
v___x_3250_ = v_reuseFailAlloc_3251_;
goto v_reusejp_3249_;
}
v_reusejp_3249_:
{
return v___x_3250_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___boxed(lean_object* v_declName_3253_, lean_object* v_a_3254_, lean_object* v_a_3255_, lean_object* v_a_3256_, lean_object* v_a_3257_, lean_object* v_a_3258_, lean_object* v_a_3259_, lean_object* v_a_3260_){
_start:
{
lean_object* v_res_3261_; 
v_res_3261_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd(v_declName_3253_, v_a_3254_, v_a_3255_, v_a_3256_, v_a_3257_, v_a_3258_, v_a_3259_);
lean_dec(v_a_3259_);
lean_dec_ref(v_a_3258_);
lean_dec(v_a_3257_);
lean_dec_ref(v_a_3256_);
lean_dec(v_a_3255_);
lean_dec_ref(v_a_3254_);
return v_res_3261_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1(lean_object* v_cls_3262_, lean_object* v_msg_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_, lean_object* v___y_3266_, lean_object* v___y_3267_, lean_object* v___y_3268_, lean_object* v___y_3269_){
_start:
{
lean_object* v___x_3271_; 
v___x_3271_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___redArg(v_cls_3262_, v_msg_3263_, v___y_3266_, v___y_3267_, v___y_3268_, v___y_3269_);
return v___x_3271_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1___boxed(lean_object* v_cls_3272_, lean_object* v_msg_3273_, lean_object* v___y_3274_, lean_object* v___y_3275_, lean_object* v___y_3276_, lean_object* v___y_3277_, lean_object* v___y_3278_, lean_object* v___y_3279_, lean_object* v___y_3280_){
_start:
{
lean_object* v_res_3281_; 
v_res_3281_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd_spec__1(v_cls_3272_, v_msg_3273_, v___y_3274_, v___y_3275_, v___y_3276_, v___y_3277_, v___y_3278_, v___y_3279_);
lean_dec(v___y_3279_);
lean_dec_ref(v___y_3278_);
lean_dec(v___y_3277_);
lean_dec_ref(v___y_3276_);
lean_dec(v___y_3275_);
lean_dec_ref(v___y_3274_);
return v_res_3281_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg(lean_object* v_declName_3282_, lean_object* v___y_3283_){
_start:
{
lean_object* v___x_3285_; lean_object* v_env_3286_; uint8_t v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; 
v___x_3285_ = lean_st_ref_get(v___y_3283_);
v_env_3286_ = lean_ctor_get(v___x_3285_, 0);
lean_inc_ref(v_env_3286_);
lean_dec(v___x_3285_);
v___x_3287_ = l_Lean_isInductiveCore(v_env_3286_, v_declName_3282_);
v___x_3288_ = lean_box(v___x_3287_);
v___x_3289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3289_, 0, v___x_3288_);
return v___x_3289_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg___boxed(lean_object* v_declName_3290_, lean_object* v___y_3291_, lean_object* v___y_3292_){
_start:
{
lean_object* v_res_3293_; 
v_res_3293_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg(v_declName_3290_, v___y_3291_);
lean_dec(v___y_3291_);
return v_res_3293_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0(lean_object* v_declName_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_){
_start:
{
lean_object* v___x_3298_; 
v___x_3298_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg(v_declName_3294_, v___y_3296_);
return v___x_3298_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___boxed(lean_object* v_declName_3299_, lean_object* v___y_3300_, lean_object* v___y_3301_, lean_object* v___y_3302_){
_start:
{
lean_object* v_res_3303_; 
v_res_3303_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0(v_declName_3299_, v___y_3300_, v___y_3301_);
lean_dec(v___y_3301_);
lean_dec_ref(v___y_3300_);
return v_res_3303_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0(uint8_t v_____do__lift_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_){
_start:
{
if (v_____do__lift_3304_ == 0)
{
uint8_t v___x_3308_; lean_object* v___x_3309_; lean_object* v___x_3310_; 
v___x_3308_ = 1;
v___x_3309_ = lean_box(v___x_3308_);
v___x_3310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3310_, 0, v___x_3309_);
return v___x_3310_;
}
else
{
uint8_t v___x_3311_; lean_object* v___x_3312_; lean_object* v___x_3313_; 
v___x_3311_ = 0;
v___x_3312_ = lean_box(v___x_3311_);
v___x_3313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3313_, 0, v___x_3312_);
return v___x_3313_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0___boxed(lean_object* v_____do__lift_3314_, lean_object* v___y_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_){
_start:
{
uint8_t v_____do__lift_5017__boxed_3318_; lean_object* v_res_3319_; 
v_____do__lift_5017__boxed_3318_ = lean_unbox(v_____do__lift_3314_);
v_res_3319_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0(v_____do__lift_5017__boxed_3318_, v___y_3315_, v___y_3316_);
lean_dec(v___y_3316_);
lean_dec_ref(v___y_3315_);
return v_res_3319_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1(void){
_start:
{
lean_object* v___x_3321_; lean_object* v___x_3322_; 
v___x_3321_ = ((lean_object*)(lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__0));
v___x_3322_ = l_Lean_stringToMessageData(v___x_3321_);
return v___x_3322_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2(lean_object* v_constName_3323_, lean_object* v___y_3324_, lean_object* v___y_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_){
_start:
{
lean_object* v___x_3331_; lean_object* v_env_3332_; lean_object* v___x_3333_; 
v___x_3331_ = lean_st_ref_get(v___y_3329_);
v_env_3332_ = lean_ctor_get(v___x_3331_, 0);
lean_inc_ref(v_env_3332_);
lean_dec(v___x_3331_);
lean_inc(v_constName_3323_);
v___x_3333_ = l_Lean_isInductiveCore_x3f(v_env_3332_, v_constName_3323_);
if (lean_obj_tag(v___x_3333_) == 0)
{
lean_object* v___x_3334_; uint8_t v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3340_; 
v___x_3334_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveArbitrary_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1);
v___x_3335_ = 0;
v___x_3336_ = l_Lean_MessageData_ofConstName(v_constName_3323_, v___x_3335_);
v___x_3337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3337_, 0, v___x_3334_);
lean_ctor_set(v___x_3337_, 1, v___x_3336_);
v___x_3338_ = lean_obj_once(&lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1, &lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1_once, _init_lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___closed__1);
v___x_3339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3339_, 0, v___x_3337_);
lean_ctor_set(v___x_3339_, 1, v___x_3338_);
v___x_3340_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7___redArg(v___x_3339_, v___y_3324_, v___y_3325_, v___y_3326_, v___y_3327_, v___y_3328_, v___y_3329_);
return v___x_3340_;
}
else
{
lean_object* v_val_3341_; lean_object* v___x_3343_; uint8_t v_isShared_3344_; uint8_t v_isSharedCheck_3348_; 
lean_dec(v_constName_3323_);
v_val_3341_ = lean_ctor_get(v___x_3333_, 0);
v_isSharedCheck_3348_ = !lean_is_exclusive(v___x_3333_);
if (v_isSharedCheck_3348_ == 0)
{
v___x_3343_ = v___x_3333_;
v_isShared_3344_ = v_isSharedCheck_3348_;
goto v_resetjp_3342_;
}
else
{
lean_inc(v_val_3341_);
lean_dec(v___x_3333_);
v___x_3343_ = lean_box(0);
v_isShared_3344_ = v_isSharedCheck_3348_;
goto v_resetjp_3342_;
}
v_resetjp_3342_:
{
lean_object* v___x_3346_; 
if (v_isShared_3344_ == 0)
{
lean_ctor_set_tag(v___x_3343_, 0);
v___x_3346_ = v___x_3343_;
goto v_reusejp_3345_;
}
else
{
lean_object* v_reuseFailAlloc_3347_; 
v_reuseFailAlloc_3347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3347_, 0, v_val_3341_);
v___x_3346_ = v_reuseFailAlloc_3347_;
goto v_reusejp_3345_;
}
v_reusejp_3345_:
{
return v___x_3346_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___boxed(lean_object* v_constName_3349_, lean_object* v___y_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_, lean_object* v___y_3355_, lean_object* v___y_3356_){
_start:
{
lean_object* v_res_3357_; 
v_res_3357_ = lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2(v_constName_3349_, v___y_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v___y_3355_);
lean_dec(v___y_3355_);
lean_dec_ref(v___y_3354_);
lean_dec(v___y_3353_);
lean_dec_ref(v___y_3352_);
lean_dec(v___y_3351_);
lean_dec_ref(v___y_3350_);
return v_res_3357_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_3358_; 
v___x_3358_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3358_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_3359_; lean_object* v___x_3360_; 
v___x_3359_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__0);
v___x_3360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3360_, 0, v___x_3359_);
return v___x_3360_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_3361_; lean_object* v___x_3362_; lean_object* v___x_3363_; 
v___x_3361_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1);
v___x_3362_ = lean_unsigned_to_nat(0u);
v___x_3363_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3363_, 0, v___x_3362_);
lean_ctor_set(v___x_3363_, 1, v___x_3362_);
lean_ctor_set(v___x_3363_, 2, v___x_3362_);
lean_ctor_set(v___x_3363_, 3, v___x_3362_);
lean_ctor_set(v___x_3363_, 4, v___x_3361_);
lean_ctor_set(v___x_3363_, 5, v___x_3361_);
lean_ctor_set(v___x_3363_, 6, v___x_3361_);
lean_ctor_set(v___x_3363_, 7, v___x_3361_);
lean_ctor_set(v___x_3363_, 8, v___x_3361_);
lean_ctor_set(v___x_3363_, 9, v___x_3361_);
return v___x_3363_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___x_3366_; 
v___x_3364_ = lean_unsigned_to_nat(32u);
v___x_3365_ = lean_mk_empty_array_with_capacity(v___x_3364_);
v___x_3366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3366_, 0, v___x_3365_);
return v___x_3366_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; lean_object* v___x_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; 
v___x_3367_ = ((size_t)5ULL);
v___x_3368_ = lean_unsigned_to_nat(0u);
v___x_3369_ = lean_unsigned_to_nat(32u);
v___x_3370_ = lean_mk_empty_array_with_capacity(v___x_3369_);
v___x_3371_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__3);
v___x_3372_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3372_, 0, v___x_3371_);
lean_ctor_set(v___x_3372_, 1, v___x_3370_);
lean_ctor_set(v___x_3372_, 2, v___x_3368_);
lean_ctor_set(v___x_3372_, 3, v___x_3368_);
lean_ctor_set_usize(v___x_3372_, 4, v___x_3367_);
return v___x_3372_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; 
v___x_3373_ = lean_box(1);
v___x_3374_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__4);
v___x_3375_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__1);
v___x_3376_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3376_, 0, v___x_3375_);
lean_ctor_set(v___x_3376_, 1, v___x_3374_);
lean_ctor_set(v___x_3376_, 2, v___x_3373_);
return v___x_3376_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg(lean_object* v_msgData_3377_, lean_object* v___y_3378_){
_start:
{
lean_object* v___x_3380_; lean_object* v_env_3381_; lean_object* v___x_3382_; lean_object* v_scopes_3383_; lean_object* v___x_3384_; lean_object* v___x_3385_; lean_object* v_opts_3386_; lean_object* v___x_3387_; lean_object* v___x_3388_; lean_object* v___x_3389_; lean_object* v___x_3390_; lean_object* v___x_3391_; 
v___x_3380_ = lean_st_ref_get(v___y_3378_);
v_env_3381_ = lean_ctor_get(v___x_3380_, 0);
lean_inc_ref(v_env_3381_);
lean_dec(v___x_3380_);
v___x_3382_ = lean_st_ref_get(v___y_3378_);
v_scopes_3383_ = lean_ctor_get(v___x_3382_, 2);
lean_inc(v_scopes_3383_);
lean_dec(v___x_3382_);
v___x_3384_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3385_ = l_List_head_x21___redArg(v___x_3384_, v_scopes_3383_);
lean_dec(v_scopes_3383_);
v_opts_3386_ = lean_ctor_get(v___x_3385_, 1);
lean_inc_ref(v_opts_3386_);
lean_dec(v___x_3385_);
v___x_3387_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__2);
v___x_3388_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___closed__5);
v___x_3389_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3389_, 0, v_env_3381_);
lean_ctor_set(v___x_3389_, 1, v___x_3387_);
lean_ctor_set(v___x_3389_, 2, v___x_3388_);
lean_ctor_set(v___x_3389_, 3, v_opts_3386_);
v___x_3390_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3390_, 0, v___x_3389_);
lean_ctor_set(v___x_3390_, 1, v_msgData_3377_);
v___x_3391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3391_, 0, v___x_3390_);
return v___x_3391_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg___boxed(lean_object* v_msgData_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_){
_start:
{
lean_object* v_res_3395_; 
v_res_3395_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg(v_msgData_3392_, v___y_3393_);
lean_dec(v___y_3393_);
return v_res_3395_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg(lean_object* v_msgData_3396_, lean_object* v_macroStack_3397_, lean_object* v___y_3398_){
_start:
{
lean_object* v___x_3400_; lean_object* v_scopes_3401_; lean_object* v___x_3402_; lean_object* v___x_3403_; lean_object* v_opts_3404_; lean_object* v___x_3405_; uint8_t v___x_3406_; 
v___x_3400_ = lean_st_ref_get(v___y_3398_);
v_scopes_3401_ = lean_ctor_get(v___x_3400_, 2);
lean_inc(v_scopes_3401_);
lean_dec(v___x_3400_);
v___x_3402_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3403_ = l_List_head_x21___redArg(v___x_3402_, v_scopes_3401_);
lean_dec(v_scopes_3401_);
v_opts_3404_ = lean_ctor_get(v___x_3403_, 1);
lean_inc_ref(v_opts_3404_);
lean_dec(v___x_3403_);
v___x_3405_ = l_Lean_Elab_pp_macroStack;
v___x_3406_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__12(v_opts_3404_, v___x_3405_);
lean_dec_ref(v_opts_3404_);
if (v___x_3406_ == 0)
{
lean_object* v___x_3407_; 
lean_dec(v_macroStack_3397_);
v___x_3407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3407_, 0, v_msgData_3396_);
return v___x_3407_;
}
else
{
if (lean_obj_tag(v_macroStack_3397_) == 0)
{
lean_object* v___x_3408_; 
v___x_3408_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3408_, 0, v_msgData_3396_);
return v___x_3408_;
}
else
{
lean_object* v_head_3409_; lean_object* v_after_3410_; lean_object* v___x_3412_; uint8_t v_isShared_3413_; uint8_t v_isSharedCheck_3425_; 
v_head_3409_ = lean_ctor_get(v_macroStack_3397_, 0);
lean_inc(v_head_3409_);
v_after_3410_ = lean_ctor_get(v_head_3409_, 1);
v_isSharedCheck_3425_ = !lean_is_exclusive(v_head_3409_);
if (v_isSharedCheck_3425_ == 0)
{
lean_object* v_unused_3426_; 
v_unused_3426_ = lean_ctor_get(v_head_3409_, 0);
lean_dec(v_unused_3426_);
v___x_3412_ = v_head_3409_;
v_isShared_3413_ = v_isSharedCheck_3425_;
goto v_resetjp_3411_;
}
else
{
lean_inc(v_after_3410_);
lean_dec(v_head_3409_);
v___x_3412_ = lean_box(0);
v_isShared_3413_ = v_isSharedCheck_3425_;
goto v_resetjp_3411_;
}
v_resetjp_3411_:
{
lean_object* v___x_3414_; lean_object* v___x_3416_; 
v___x_3414_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13___closed__0);
if (v_isShared_3413_ == 0)
{
lean_ctor_set_tag(v___x_3412_, 7);
lean_ctor_set(v___x_3412_, 1, v___x_3414_);
lean_ctor_set(v___x_3412_, 0, v_msgData_3396_);
v___x_3416_ = v___x_3412_;
goto v_reusejp_3415_;
}
else
{
lean_object* v_reuseFailAlloc_3424_; 
v_reuseFailAlloc_3424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3424_, 0, v_msgData_3396_);
lean_ctor_set(v_reuseFailAlloc_3424_, 1, v___x_3414_);
v___x_3416_ = v_reuseFailAlloc_3424_;
goto v_reusejp_3415_;
}
v_reusejp_3415_:
{
lean_object* v___x_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; lean_object* v___x_3420_; lean_object* v_msgData_3421_; lean_object* v___x_3422_; lean_object* v___x_3423_; 
v___x_3417_ = lean_obj_once(&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2, &lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2_once, _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9___redArg___closed__2);
v___x_3418_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3418_, 0, v___x_3416_);
lean_ctor_set(v___x_3418_, 1, v___x_3417_);
v___x_3419_ = l_Lean_MessageData_ofSyntax(v_after_3410_);
v___x_3420_ = l_Lean_indentD(v___x_3419_);
v_msgData_3421_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_3421_, 0, v___x_3418_);
lean_ctor_set(v_msgData_3421_, 1, v___x_3420_);
v___x_3422_ = lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkBody_spec__7_spec__9_spec__13(v_msgData_3421_, v_macroStack_3397_);
v___x_3423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3423_, 0, v___x_3422_);
return v___x_3423_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_3427_, lean_object* v_macroStack_3428_, lean_object* v___y_3429_, lean_object* v___y_3430_){
_start:
{
lean_object* v_res_3431_; 
v_res_3431_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg(v_msgData_3427_, v_macroStack_3428_, v___y_3429_);
lean_dec(v___y_3429_);
return v_res_3431_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(lean_object* v_msg_3432_, lean_object* v___y_3433_, lean_object* v___y_3434_){
_start:
{
lean_object* v___x_3436_; 
v___x_3436_ = l_Lean_Elab_Command_getRef___redArg(v___y_3433_);
if (lean_obj_tag(v___x_3436_) == 0)
{
lean_object* v_a_3437_; lean_object* v_macroStack_3438_; lean_object* v___x_3439_; lean_object* v_a_3440_; lean_object* v___x_3441_; lean_object* v___x_3442_; lean_object* v_a_3443_; lean_object* v___x_3445_; uint8_t v_isShared_3446_; uint8_t v_isSharedCheck_3451_; 
v_a_3437_ = lean_ctor_get(v___x_3436_, 0);
lean_inc(v_a_3437_);
lean_dec_ref_known(v___x_3436_, 1);
v_macroStack_3438_ = lean_ctor_get(v___y_3433_, 4);
v___x_3439_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg(v_msg_3432_, v___y_3434_);
v_a_3440_ = lean_ctor_get(v___x_3439_, 0);
lean_inc(v_a_3440_);
lean_dec_ref(v___x_3439_);
v___x_3441_ = l_Lean_Elab_getBetterRef(v_a_3437_, v_macroStack_3438_);
lean_dec(v_a_3437_);
lean_inc(v_macroStack_3438_);
v___x_3442_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg(v_a_3440_, v_macroStack_3438_, v___y_3434_);
v_a_3443_ = lean_ctor_get(v___x_3442_, 0);
v_isSharedCheck_3451_ = !lean_is_exclusive(v___x_3442_);
if (v_isSharedCheck_3451_ == 0)
{
v___x_3445_ = v___x_3442_;
v_isShared_3446_ = v_isSharedCheck_3451_;
goto v_resetjp_3444_;
}
else
{
lean_inc(v_a_3443_);
lean_dec(v___x_3442_);
v___x_3445_ = lean_box(0);
v_isShared_3446_ = v_isSharedCheck_3451_;
goto v_resetjp_3444_;
}
v_resetjp_3444_:
{
lean_object* v___x_3447_; lean_object* v___x_3449_; 
v___x_3447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3447_, 0, v___x_3441_);
lean_ctor_set(v___x_3447_, 1, v_a_3443_);
if (v_isShared_3446_ == 0)
{
lean_ctor_set_tag(v___x_3445_, 1);
lean_ctor_set(v___x_3445_, 0, v___x_3447_);
v___x_3449_ = v___x_3445_;
goto v_reusejp_3448_;
}
else
{
lean_object* v_reuseFailAlloc_3450_; 
v_reuseFailAlloc_3450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3450_, 0, v___x_3447_);
v___x_3449_ = v_reuseFailAlloc_3450_;
goto v_reusejp_3448_;
}
v_reusejp_3448_:
{
return v___x_3449_;
}
}
}
else
{
lean_object* v_a_3452_; lean_object* v___x_3454_; uint8_t v_isShared_3455_; uint8_t v_isSharedCheck_3459_; 
lean_dec_ref(v_msg_3432_);
v_a_3452_ = lean_ctor_get(v___x_3436_, 0);
v_isSharedCheck_3459_ = !lean_is_exclusive(v___x_3436_);
if (v_isSharedCheck_3459_ == 0)
{
v___x_3454_ = v___x_3436_;
v_isShared_3455_ = v_isSharedCheck_3459_;
goto v_resetjp_3453_;
}
else
{
lean_inc(v_a_3452_);
lean_dec(v___x_3436_);
v___x_3454_ = lean_box(0);
v_isShared_3455_ = v_isSharedCheck_3459_;
goto v_resetjp_3453_;
}
v_resetjp_3453_:
{
lean_object* v___x_3457_; 
if (v_isShared_3455_ == 0)
{
v___x_3457_ = v___x_3454_;
goto v_reusejp_3456_;
}
else
{
lean_object* v_reuseFailAlloc_3458_; 
v_reuseFailAlloc_3458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3458_, 0, v_a_3452_);
v___x_3457_ = v_reuseFailAlloc_3458_;
goto v_reusejp_3456_;
}
v_reusejp_3456_:
{
return v___x_3457_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg___boxed(lean_object* v_msg_3460_, lean_object* v___y_3461_, lean_object* v___y_3462_, lean_object* v___y_3463_){
_start:
{
lean_object* v_res_3464_; 
v_res_3464_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(v_msg_3460_, v___y_3461_, v___y_3462_);
lean_dec(v___y_3462_);
lean_dec_ref(v___y_3461_);
return v_res_3464_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1(void){
_start:
{
lean_object* v___x_3466_; lean_object* v___x_3467_; 
v___x_3466_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__0));
v___x_3467_ = l_Lean_stringToMessageData(v___x_3466_);
return v___x_3467_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3(void){
_start:
{
lean_object* v___x_3469_; lean_object* v___x_3470_; 
v___x_3469_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__2));
v___x_3470_ = l_Lean_stringToMessageData(v___x_3469_);
return v___x_3470_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4(lean_object* v_as_3471_, size_t v_sz_3472_, size_t v_i_3473_, lean_object* v_b_3474_, lean_object* v___y_3475_, lean_object* v___y_3476_){
_start:
{
lean_object* v_a_3479_; uint8_t v___x_3483_; 
v___x_3483_ = lean_usize_dec_lt(v_i_3473_, v_sz_3472_);
if (v___x_3483_ == 0)
{
lean_object* v___x_3484_; 
v___x_3484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3484_, 0, v_b_3474_);
return v___x_3484_;
}
else
{
lean_object* v_a_3485_; lean_object* v___x_3486_; lean_object* v___x_3487_; 
v_a_3485_ = lean_array_uget_borrowed(v_as_3471_, v_i_3473_);
lean_inc(v_a_3485_);
v___x_3486_ = lean_alloc_closure((void*)(lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__2___boxed), 8, 1);
lean_closure_set(v___x_3486_, 0, v_a_3485_);
v___x_3487_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_3486_, v___y_3475_, v___y_3476_);
if (lean_obj_tag(v___x_3487_) == 0)
{
lean_object* v_a_3488_; lean_object* v_numIndices_3489_; lean_object* v___x_3490_; lean_object* v___x_3491_; uint8_t v___x_3492_; 
v_a_3488_ = lean_ctor_get(v___x_3487_, 0);
lean_inc(v_a_3488_);
lean_dec_ref_known(v___x_3487_, 1);
v_numIndices_3489_ = lean_ctor_get(v_a_3488_, 2);
lean_inc(v_numIndices_3489_);
lean_dec(v_a_3488_);
v___x_3490_ = lean_unsigned_to_nat(0u);
v___x_3491_ = lean_box(0);
v___x_3492_ = lean_nat_dec_lt(v___x_3490_, v_numIndices_3489_);
lean_dec(v_numIndices_3489_);
if (v___x_3492_ == 0)
{
v_a_3479_ = v___x_3491_;
goto v___jp_3478_;
}
else
{
lean_object* v___x_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; lean_object* v___x_3496_; lean_object* v___x_3497_; lean_object* v___x_3498_; 
v___x_3493_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__1);
lean_inc(v_a_3485_);
v___x_3494_ = l_Lean_MessageData_ofName(v_a_3485_);
v___x_3495_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3495_, 0, v___x_3493_);
lean_ctor_set(v___x_3495_, 1, v___x_3494_);
v___x_3496_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___closed__3);
v___x_3497_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3497_, 0, v___x_3495_);
lean_ctor_set(v___x_3497_, 1, v___x_3496_);
v___x_3498_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(v___x_3497_, v___y_3475_, v___y_3476_);
if (lean_obj_tag(v___x_3498_) == 0)
{
lean_dec_ref_known(v___x_3498_, 1);
v_a_3479_ = v___x_3491_;
goto v___jp_3478_;
}
else
{
return v___x_3498_;
}
}
}
else
{
lean_object* v_a_3499_; lean_object* v___x_3501_; uint8_t v_isShared_3502_; uint8_t v_isSharedCheck_3506_; 
v_a_3499_ = lean_ctor_get(v___x_3487_, 0);
v_isSharedCheck_3506_ = !lean_is_exclusive(v___x_3487_);
if (v_isSharedCheck_3506_ == 0)
{
v___x_3501_ = v___x_3487_;
v_isShared_3502_ = v_isSharedCheck_3506_;
goto v_resetjp_3500_;
}
else
{
lean_inc(v_a_3499_);
lean_dec(v___x_3487_);
v___x_3501_ = lean_box(0);
v_isShared_3502_ = v_isSharedCheck_3506_;
goto v_resetjp_3500_;
}
v_resetjp_3500_:
{
lean_object* v___x_3504_; 
if (v_isShared_3502_ == 0)
{
v___x_3504_ = v___x_3501_;
goto v_reusejp_3503_;
}
else
{
lean_object* v_reuseFailAlloc_3505_; 
v_reuseFailAlloc_3505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3505_, 0, v_a_3499_);
v___x_3504_ = v_reuseFailAlloc_3505_;
goto v_reusejp_3503_;
}
v_reusejp_3503_:
{
return v___x_3504_;
}
}
}
}
v___jp_3478_:
{
size_t v___x_3480_; size_t v___x_3481_; 
v___x_3480_ = ((size_t)1ULL);
v___x_3481_ = lean_usize_add(v_i_3473_, v___x_3480_);
v_i_3473_ = v___x_3481_;
v_b_3474_ = v_a_3479_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4___boxed(lean_object* v_as_3507_, lean_object* v_sz_3508_, lean_object* v_i_3509_, lean_object* v_b_3510_, lean_object* v___y_3511_, lean_object* v___y_3512_, lean_object* v___y_3513_){
_start:
{
size_t v_sz_boxed_3514_; size_t v_i_boxed_3515_; lean_object* v_res_3516_; 
v_sz_boxed_3514_ = lean_unbox_usize(v_sz_3508_);
lean_dec(v_sz_3508_);
v_i_boxed_3515_ = lean_unbox_usize(v_i_3509_);
lean_dec(v_i_3509_);
v_res_3516_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4(v_as_3507_, v_sz_boxed_3514_, v_i_boxed_3515_, v_b_3510_, v___y_3511_, v___y_3512_);
lean_dec(v___y_3512_);
lean_dec_ref(v___y_3511_);
lean_dec_ref(v_as_3507_);
return v_res_3516_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1(lean_object* v_as_3517_, size_t v_i_3518_, size_t v_stop_3519_, lean_object* v_b_3520_, lean_object* v___y_3521_, lean_object* v___y_3522_){
_start:
{
uint8_t v___x_3524_; 
v___x_3524_ = lean_usize_dec_eq(v_i_3518_, v_stop_3519_);
if (v___x_3524_ == 0)
{
lean_object* v___x_3525_; lean_object* v___x_3526_; 
v___x_3525_ = lean_array_uget_borrowed(v_as_3517_, v_i_3518_);
lean_inc(v___x_3525_);
v___x_3526_ = l_Lean_Elab_Command_elabCommand(v___x_3525_, v___y_3521_, v___y_3522_);
if (lean_obj_tag(v___x_3526_) == 0)
{
lean_object* v_a_3527_; size_t v___x_3528_; size_t v___x_3529_; 
v_a_3527_ = lean_ctor_get(v___x_3526_, 0);
lean_inc(v_a_3527_);
lean_dec_ref_known(v___x_3526_, 1);
v___x_3528_ = ((size_t)1ULL);
v___x_3529_ = lean_usize_add(v_i_3518_, v___x_3528_);
v_i_3518_ = v___x_3529_;
v_b_3520_ = v_a_3527_;
goto _start;
}
else
{
return v___x_3526_;
}
}
else
{
lean_object* v___x_3531_; 
v___x_3531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3531_, 0, v_b_3520_);
return v___x_3531_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1___boxed(lean_object* v_as_3532_, lean_object* v_i_3533_, lean_object* v_stop_3534_, lean_object* v_b_3535_, lean_object* v___y_3536_, lean_object* v___y_3537_, lean_object* v___y_3538_){
_start:
{
size_t v_i_boxed_3539_; size_t v_stop_boxed_3540_; lean_object* v_res_3541_; 
v_i_boxed_3539_ = lean_unbox_usize(v_i_3533_);
lean_dec(v_i_3533_);
v_stop_boxed_3540_ = lean_unbox_usize(v_stop_3534_);
lean_dec(v_stop_3534_);
v_res_3541_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1(v_as_3532_, v_i_boxed_3539_, v_stop_boxed_3540_, v_b_3535_, v___y_3536_, v___y_3537_);
lean_dec(v___y_3537_);
lean_dec_ref(v___y_3536_);
lean_dec_ref(v_as_3532_);
return v_res_3541_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5(lean_object* v_as_3542_, size_t v_sz_3543_, size_t v_i_3544_, lean_object* v_b_3545_, lean_object* v___y_3546_, lean_object* v___y_3547_){
_start:
{
lean_object* v_a_3550_; uint8_t v___x_3554_; 
v___x_3554_ = lean_usize_dec_lt(v_i_3544_, v_sz_3543_);
if (v___x_3554_ == 0)
{
lean_object* v___x_3555_; 
v___x_3555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3555_, 0, v_b_3545_);
return v___x_3555_;
}
else
{
lean_object* v_a_3556_; lean_object* v___x_3557_; lean_object* v___x_3558_; 
v_a_3556_ = lean_array_uget_borrowed(v_as_3542_, v_i_3544_);
lean_inc(v_a_3556_);
v___x_3557_ = lean_alloc_closure((void*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmd___boxed), 8, 1);
lean_closure_set(v___x_3557_, 0, v_a_3556_);
v___x_3558_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_3557_, v___y_3546_, v___y_3547_);
if (lean_obj_tag(v___x_3558_) == 0)
{
lean_object* v_a_3559_; lean_object* v___x_3560_; lean_object* v___y_3562_; lean_object* v___x_3563_; lean_object* v___x_3564_; uint8_t v___x_3565_; 
v_a_3559_ = lean_ctor_get(v___x_3558_, 0);
lean_inc(v_a_3559_);
lean_dec_ref_known(v___x_3558_, 1);
v___x_3560_ = lean_box(0);
v___x_3563_ = lean_unsigned_to_nat(0u);
v___x_3564_ = lean_array_get_size(v_a_3559_);
v___x_3565_ = lean_nat_dec_lt(v___x_3563_, v___x_3564_);
if (v___x_3565_ == 0)
{
lean_dec(v_a_3559_);
v_a_3550_ = v___x_3560_;
goto v___jp_3549_;
}
else
{
uint8_t v___x_3566_; 
v___x_3566_ = lean_nat_dec_le(v___x_3564_, v___x_3564_);
if (v___x_3566_ == 0)
{
if (v___x_3565_ == 0)
{
lean_dec(v_a_3559_);
v_a_3550_ = v___x_3560_;
goto v___jp_3549_;
}
else
{
size_t v___x_3567_; size_t v___x_3568_; lean_object* v___x_3569_; 
v___x_3567_ = ((size_t)0ULL);
v___x_3568_ = lean_usize_of_nat(v___x_3564_);
v___x_3569_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1(v_a_3559_, v___x_3567_, v___x_3568_, v___x_3560_, v___y_3546_, v___y_3547_);
lean_dec(v_a_3559_);
v___y_3562_ = v___x_3569_;
goto v___jp_3561_;
}
}
else
{
size_t v___x_3570_; size_t v___x_3571_; lean_object* v___x_3572_; 
v___x_3570_ = ((size_t)0ULL);
v___x_3571_ = lean_usize_of_nat(v___x_3564_);
v___x_3572_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__1(v_a_3559_, v___x_3570_, v___x_3571_, v___x_3560_, v___y_3546_, v___y_3547_);
lean_dec(v_a_3559_);
v___y_3562_ = v___x_3572_;
goto v___jp_3561_;
}
}
v___jp_3561_:
{
if (lean_obj_tag(v___y_3562_) == 0)
{
lean_dec_ref_known(v___y_3562_, 1);
v_a_3550_ = v___x_3560_;
goto v___jp_3549_;
}
else
{
return v___y_3562_;
}
}
}
else
{
lean_object* v_a_3573_; lean_object* v___x_3575_; uint8_t v_isShared_3576_; uint8_t v_isSharedCheck_3580_; 
v_a_3573_ = lean_ctor_get(v___x_3558_, 0);
v_isSharedCheck_3580_ = !lean_is_exclusive(v___x_3558_);
if (v_isSharedCheck_3580_ == 0)
{
v___x_3575_ = v___x_3558_;
v_isShared_3576_ = v_isSharedCheck_3580_;
goto v_resetjp_3574_;
}
else
{
lean_inc(v_a_3573_);
lean_dec(v___x_3558_);
v___x_3575_ = lean_box(0);
v_isShared_3576_ = v_isSharedCheck_3580_;
goto v_resetjp_3574_;
}
v_resetjp_3574_:
{
lean_object* v___x_3578_; 
if (v_isShared_3576_ == 0)
{
v___x_3578_ = v___x_3575_;
goto v_reusejp_3577_;
}
else
{
lean_object* v_reuseFailAlloc_3579_; 
v_reuseFailAlloc_3579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3579_, 0, v_a_3573_);
v___x_3578_ = v_reuseFailAlloc_3579_;
goto v_reusejp_3577_;
}
v_reusejp_3577_:
{
return v___x_3578_;
}
}
}
}
v___jp_3549_:
{
size_t v___x_3551_; size_t v___x_3552_; 
v___x_3551_ = ((size_t)1ULL);
v___x_3552_ = lean_usize_add(v_i_3544_, v___x_3551_);
v_i_3544_ = v___x_3552_;
v_b_3545_ = v_a_3550_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5___boxed(lean_object* v_as_3581_, lean_object* v_sz_3582_, lean_object* v_i_3583_, lean_object* v_b_3584_, lean_object* v___y_3585_, lean_object* v___y_3586_, lean_object* v___y_3587_){
_start:
{
size_t v_sz_boxed_3588_; size_t v_i_boxed_3589_; lean_object* v_res_3590_; 
v_sz_boxed_3588_ = lean_unbox_usize(v_sz_3582_);
lean_dec(v_sz_3582_);
v_i_boxed_3589_ = lean_unbox_usize(v_i_3583_);
lean_dec(v_i_3583_);
v_res_3590_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5(v_as_3581_, v_sz_boxed_3588_, v_i_boxed_3589_, v_b_3584_, v___y_3585_, v___y_3586_);
lean_dec(v___y_3586_);
lean_dec_ref(v___y_3585_);
lean_dec_ref(v_as_3581_);
return v_res_3590_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6(lean_object* v_as_3591_, size_t v_i_3592_, size_t v_stop_3593_, lean_object* v___y_3594_, lean_object* v___y_3595_){
_start:
{
uint8_t v___x_3597_; 
v___x_3597_ = lean_usize_dec_eq(v_i_3592_, v_stop_3593_);
if (v___x_3597_ == 0)
{
uint8_t v___x_3598_; uint8_t v_a_3600_; lean_object* v___x_3606_; lean_object* v___x_3607_; 
v___x_3598_ = 1;
v___x_3606_ = lean_array_uget_borrowed(v_as_3591_, v_i_3592_);
lean_inc(v___x_3606_);
v___x_3607_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__0___redArg(v___x_3606_, v___y_3595_);
if (lean_obj_tag(v___x_3607_) == 0)
{
lean_object* v_a_3608_; lean_object* v___x_3610_; uint8_t v_isShared_3611_; uint8_t v_isSharedCheck_3617_; 
v_a_3608_ = lean_ctor_get(v___x_3607_, 0);
v_isSharedCheck_3617_ = !lean_is_exclusive(v___x_3607_);
if (v_isSharedCheck_3617_ == 0)
{
v___x_3610_ = v___x_3607_;
v_isShared_3611_ = v_isSharedCheck_3617_;
goto v_resetjp_3609_;
}
else
{
lean_inc(v_a_3608_);
lean_dec(v___x_3607_);
v___x_3610_ = lean_box(0);
v_isShared_3611_ = v_isSharedCheck_3617_;
goto v_resetjp_3609_;
}
v_resetjp_3609_:
{
uint8_t v___x_3612_; 
v___x_3612_ = lean_unbox(v_a_3608_);
lean_dec(v_a_3608_);
if (v___x_3612_ == 0)
{
lean_object* v___x_3613_; lean_object* v___x_3615_; 
v___x_3613_ = lean_box(v___x_3598_);
if (v_isShared_3611_ == 0)
{
lean_ctor_set(v___x_3610_, 0, v___x_3613_);
v___x_3615_ = v___x_3610_;
goto v_reusejp_3614_;
}
else
{
lean_object* v_reuseFailAlloc_3616_; 
v_reuseFailAlloc_3616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3616_, 0, v___x_3613_);
v___x_3615_ = v_reuseFailAlloc_3616_;
goto v_reusejp_3614_;
}
v_reusejp_3614_:
{
return v___x_3615_;
}
}
else
{
lean_del_object(v___x_3610_);
v_a_3600_ = v___x_3597_;
goto v___jp_3599_;
}
}
}
else
{
if (lean_obj_tag(v___x_3607_) == 0)
{
lean_object* v_a_3618_; uint8_t v___x_3619_; 
v_a_3618_ = lean_ctor_get(v___x_3607_, 0);
lean_inc(v_a_3618_);
lean_dec_ref_known(v___x_3607_, 1);
v___x_3619_ = lean_unbox(v_a_3618_);
lean_dec(v_a_3618_);
v_a_3600_ = v___x_3619_;
goto v___jp_3599_;
}
else
{
return v___x_3607_;
}
}
v___jp_3599_:
{
if (v_a_3600_ == 0)
{
size_t v___x_3601_; size_t v___x_3602_; 
v___x_3601_ = ((size_t)1ULL);
v___x_3602_ = lean_usize_add(v_i_3592_, v___x_3601_);
v_i_3592_ = v___x_3602_;
goto _start;
}
else
{
lean_object* v___x_3604_; lean_object* v___x_3605_; 
v___x_3604_ = lean_box(v___x_3598_);
v___x_3605_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3605_, 0, v___x_3604_);
return v___x_3605_;
}
}
}
else
{
uint8_t v___x_3620_; lean_object* v___x_3621_; lean_object* v___x_3622_; 
v___x_3620_ = 0;
v___x_3621_ = lean_box(v___x_3620_);
v___x_3622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3622_, 0, v___x_3621_);
return v___x_3622_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6___boxed(lean_object* v_as_3623_, lean_object* v_i_3624_, lean_object* v_stop_3625_, lean_object* v___y_3626_, lean_object* v___y_3627_, lean_object* v___y_3628_){
_start:
{
size_t v_i_boxed_3629_; size_t v_stop_boxed_3630_; lean_object* v_res_3631_; 
v_i_boxed_3629_ = lean_unbox_usize(v_i_3624_);
lean_dec(v_i_3624_);
v_stop_boxed_3630_ = lean_unbox_usize(v_stop_3625_);
lean_dec(v_stop_3625_);
v_res_3631_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6(v_as_3623_, v_i_boxed_3629_, v_stop_boxed_3630_, v___y_3626_, v___y_3627_);
lean_dec(v___y_3627_);
lean_dec_ref(v___y_3626_);
lean_dec_ref(v_as_3623_);
return v_res_3631_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1(void){
_start:
{
lean_object* v___x_3633_; lean_object* v___x_3634_; 
v___x_3633_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__0));
v___x_3634_ = l_Lean_stringToMessageData(v___x_3633_);
return v___x_3634_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler(lean_object* v_declNames_3635_, lean_object* v_a_3636_, lean_object* v_a_3637_){
_start:
{
lean_object* v___y_3640_; lean_object* v___y_3641_; lean_object* v___y_3674_; lean_object* v___x_3687_; lean_object* v___x_3688_; uint8_t v___x_3689_; 
v___x_3687_ = lean_unsigned_to_nat(0u);
v___x_3688_ = lean_array_get_size(v_declNames_3635_);
v___x_3689_ = lean_nat_dec_lt(v___x_3687_, v___x_3688_);
if (v___x_3689_ == 0)
{
lean_object* v___x_3690_; 
v___x_3690_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0(v___x_3689_, v_a_3636_, v_a_3637_);
v___y_3674_ = v___x_3690_;
goto v___jp_3673_;
}
else
{
if (v___x_3689_ == 0)
{
v___y_3640_ = v_a_3636_;
v___y_3641_ = v_a_3637_;
goto v___jp_3639_;
}
else
{
size_t v___x_3691_; size_t v___x_3692_; lean_object* v___x_3693_; 
v___x_3691_ = ((size_t)0ULL);
v___x_3692_ = lean_usize_of_nat(v___x_3688_);
v___x_3693_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__6(v_declNames_3635_, v___x_3691_, v___x_3692_, v_a_3636_, v_a_3637_);
if (lean_obj_tag(v___x_3693_) == 0)
{
lean_object* v_a_3694_; uint8_t v___x_3695_; lean_object* v___x_3696_; 
v_a_3694_ = lean_ctor_get(v___x_3693_, 0);
lean_inc(v_a_3694_);
lean_dec_ref_known(v___x_3693_, 1);
v___x_3695_ = lean_unbox(v_a_3694_);
lean_dec(v_a_3694_);
v___x_3696_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___lam__0(v___x_3695_, v_a_3636_, v_a_3637_);
v___y_3674_ = v___x_3696_;
goto v___jp_3673_;
}
else
{
v___y_3674_ = v___x_3693_;
goto v___jp_3673_;
}
}
}
v___jp_3639_:
{
lean_object* v___x_3642_; size_t v_sz_3643_; size_t v___x_3644_; lean_object* v___x_3645_; 
v___x_3642_ = lean_box(0);
v_sz_3643_ = lean_array_size(v_declNames_3635_);
v___x_3644_ = ((size_t)0ULL);
v___x_3645_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__4(v_declNames_3635_, v_sz_3643_, v___x_3644_, v___x_3642_, v___y_3640_, v___y_3641_);
if (lean_obj_tag(v___x_3645_) == 0)
{
lean_object* v___x_3646_; 
lean_dec_ref_known(v___x_3645_, 1);
v___x_3646_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__5(v_declNames_3635_, v_sz_3643_, v___x_3644_, v___x_3642_, v___y_3640_, v___y_3641_);
if (lean_obj_tag(v___x_3646_) == 0)
{
lean_object* v___x_3648_; uint8_t v_isShared_3649_; uint8_t v_isSharedCheck_3655_; 
v_isSharedCheck_3655_ = !lean_is_exclusive(v___x_3646_);
if (v_isSharedCheck_3655_ == 0)
{
lean_object* v_unused_3656_; 
v_unused_3656_ = lean_ctor_get(v___x_3646_, 0);
lean_dec(v_unused_3656_);
v___x_3648_ = v___x_3646_;
v_isShared_3649_ = v_isSharedCheck_3655_;
goto v_resetjp_3647_;
}
else
{
lean_dec(v___x_3646_);
v___x_3648_ = lean_box(0);
v_isShared_3649_ = v_isSharedCheck_3655_;
goto v_resetjp_3647_;
}
v_resetjp_3647_:
{
uint8_t v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3653_; 
v___x_3650_ = 1;
v___x_3651_ = lean_box(v___x_3650_);
if (v_isShared_3649_ == 0)
{
lean_ctor_set(v___x_3648_, 0, v___x_3651_);
v___x_3653_ = v___x_3648_;
goto v_reusejp_3652_;
}
else
{
lean_object* v_reuseFailAlloc_3654_; 
v_reuseFailAlloc_3654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3654_, 0, v___x_3651_);
v___x_3653_ = v_reuseFailAlloc_3654_;
goto v_reusejp_3652_;
}
v_reusejp_3652_:
{
return v___x_3653_;
}
}
}
else
{
lean_object* v_a_3657_; lean_object* v___x_3659_; uint8_t v_isShared_3660_; uint8_t v_isSharedCheck_3664_; 
v_a_3657_ = lean_ctor_get(v___x_3646_, 0);
v_isSharedCheck_3664_ = !lean_is_exclusive(v___x_3646_);
if (v_isSharedCheck_3664_ == 0)
{
v___x_3659_ = v___x_3646_;
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
else
{
lean_inc(v_a_3657_);
lean_dec(v___x_3646_);
v___x_3659_ = lean_box(0);
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
v_resetjp_3658_:
{
lean_object* v___x_3662_; 
if (v_isShared_3660_ == 0)
{
v___x_3662_ = v___x_3659_;
goto v_reusejp_3661_;
}
else
{
lean_object* v_reuseFailAlloc_3663_; 
v_reuseFailAlloc_3663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3663_, 0, v_a_3657_);
v___x_3662_ = v_reuseFailAlloc_3663_;
goto v_reusejp_3661_;
}
v_reusejp_3661_:
{
return v___x_3662_;
}
}
}
}
else
{
lean_object* v_a_3665_; lean_object* v___x_3667_; uint8_t v_isShared_3668_; uint8_t v_isSharedCheck_3672_; 
v_a_3665_ = lean_ctor_get(v___x_3645_, 0);
v_isSharedCheck_3672_ = !lean_is_exclusive(v___x_3645_);
if (v_isSharedCheck_3672_ == 0)
{
v___x_3667_ = v___x_3645_;
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
else
{
lean_inc(v_a_3665_);
lean_dec(v___x_3645_);
v___x_3667_ = lean_box(0);
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
v_resetjp_3666_:
{
lean_object* v___x_3670_; 
if (v_isShared_3668_ == 0)
{
v___x_3670_ = v___x_3667_;
goto v_reusejp_3669_;
}
else
{
lean_object* v_reuseFailAlloc_3671_; 
v_reuseFailAlloc_3671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3671_, 0, v_a_3665_);
v___x_3670_ = v_reuseFailAlloc_3671_;
goto v_reusejp_3669_;
}
v_reusejp_3669_:
{
return v___x_3670_;
}
}
}
}
v___jp_3673_:
{
if (lean_obj_tag(v___y_3674_) == 0)
{
lean_object* v_a_3675_; uint8_t v___x_3676_; 
v_a_3675_ = lean_ctor_get(v___y_3674_, 0);
lean_inc(v_a_3675_);
lean_dec_ref_known(v___y_3674_, 1);
v___x_3676_ = lean_unbox(v_a_3675_);
lean_dec(v_a_3675_);
if (v___x_3676_ == 0)
{
lean_object* v___x_3677_; lean_object* v___x_3678_; lean_object* v_a_3679_; lean_object* v___x_3681_; uint8_t v_isShared_3682_; uint8_t v_isSharedCheck_3686_; 
v___x_3677_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1, &lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1_once, _init_lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___closed__1);
v___x_3678_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(v___x_3677_, v_a_3636_, v_a_3637_);
v_a_3679_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3686_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3686_ == 0)
{
v___x_3681_ = v___x_3678_;
v_isShared_3682_ = v_isSharedCheck_3686_;
goto v_resetjp_3680_;
}
else
{
lean_inc(v_a_3679_);
lean_dec(v___x_3678_);
v___x_3681_ = lean_box(0);
v_isShared_3682_ = v_isSharedCheck_3686_;
goto v_resetjp_3680_;
}
v_resetjp_3680_:
{
lean_object* v___x_3684_; 
if (v_isShared_3682_ == 0)
{
v___x_3684_ = v___x_3681_;
goto v_reusejp_3683_;
}
else
{
lean_object* v_reuseFailAlloc_3685_; 
v_reuseFailAlloc_3685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3685_, 0, v_a_3679_);
v___x_3684_ = v_reuseFailAlloc_3685_;
goto v_reusejp_3683_;
}
v_reusejp_3683_:
{
return v___x_3684_;
}
}
}
else
{
v___y_3640_ = v_a_3636_;
v___y_3641_ = v_a_3637_;
goto v___jp_3639_;
}
}
else
{
return v___y_3674_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler___boxed(lean_object* v_declNames_3697_, lean_object* v_a_3698_, lean_object* v_a_3699_, lean_object* v_a_3700_){
_start:
{
lean_object* v_res_3701_; 
v_res_3701_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler(v_declNames_3697_, v_a_3698_, v_a_3699_);
lean_dec(v_a_3699_);
lean_dec_ref(v_a_3698_);
lean_dec_ref(v_declNames_3697_);
return v_res_3701_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3(lean_object* v_msgData_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_){
_start:
{
lean_object* v___x_3706_; 
v___x_3706_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___redArg(v_msgData_3702_, v___y_3704_);
return v___x_3706_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3___boxed(lean_object* v_msgData_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_){
_start:
{
lean_object* v_res_3711_; 
v_res_3711_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__3(v_msgData_3707_, v___y_3708_, v___y_3709_);
lean_dec(v___y_3709_);
lean_dec_ref(v___y_3708_);
return v_res_3711_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3(lean_object* v_00_u03b1_3712_, lean_object* v_msg_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_){
_start:
{
lean_object* v___x_3717_; 
v___x_3717_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___redArg(v_msg_3713_, v___y_3714_, v___y_3715_);
return v___x_3717_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3___boxed(lean_object* v_00_u03b1_3718_, lean_object* v_msg_3719_, lean_object* v___y_3720_, lean_object* v___y_3721_, lean_object* v___y_3722_){
_start:
{
lean_object* v_res_3723_; 
v_res_3723_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3(v_00_u03b1_3718_, v_msg_3719_, v___y_3720_, v___y_3721_);
lean_dec(v___y_3721_);
lean_dec_ref(v___y_3720_);
return v_res_3723_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4(lean_object* v_msgData_3724_, lean_object* v_macroStack_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_){
_start:
{
lean_object* v___x_3729_; 
v___x_3729_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___redArg(v_msgData_3724_, v_macroStack_3725_, v___y_3727_);
return v___x_3729_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4___boxed(lean_object* v_msgData_3730_, lean_object* v_macroStack_3731_, lean_object* v___y_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_){
_start:
{
lean_object* v_res_3735_; 
v_res_3735_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryInstanceHandler_spec__3_spec__4(v_msgData_3730_, v_macroStack_3731_, v___y_3732_, v___y_3733_);
lean_dec(v___y_3733_);
lean_dec_ref(v___y_3732_);
return v_res_3735_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_3738_; lean_object* v___x_3739_; lean_object* v___x_3740_; 
v___x_3738_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveArbitrary_0__Plausible_mkArbitraryFueledInstanceCmds_spec__1___redArg___closed__2));
v___x_3739_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_));
v___x_3740_ = l_Lean_Elab_registerDerivingHandler(v___x_3738_, v___x_3739_);
return v___x_3740_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2____boxed(lean_object* v_a_3741_){
_start:
{
lean_object* v_res_3742_; 
v_res_3742_ = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_();
return v_res_3742_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Deriving_Util(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_DeriveArbitrary(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Deriving_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_ArbitraryFueled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_plausible___private_Plausible_DeriveArbitrary_0__Plausible_initFn_00___x40_Plausible_DeriveArbitrary_3192332529____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_DeriveArbitrary(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Deriving_Util(uint8_t builtin);
lean_object* initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_DeriveArbitrary(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Deriving_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_ArbitraryFueled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_DeriveArbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_DeriveArbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_DeriveArbitrary(builtin);
}
#ifdef __cplusplus
}
#endif
