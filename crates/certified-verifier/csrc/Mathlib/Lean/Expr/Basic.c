// Lean compiler output
// Module: Mathlib.Lean.Expr.Basic
// Imports: public import Init public meta import Init import Mathlib.Tactic.Linter.Header public import Lean.Meta.AppBuilder public import Lean.Meta.Match.MatcherInfo public import Lean.Meta.Transform
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_ExprStructEq_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_ExprStructEq_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getRevArg_x21(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_findField_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_getProjFnForField_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPathToBaseStructure_x3f(lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getProjectionFnInfo_x3f(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConst(lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* l_Lean_Expr_getAppFn_x27(lean_object*);
lean_object* l_Lean_Expr_rawNatLit_x3f(lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
lean_object* l_Lean_Expr_constName_x21(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_isRec___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_mkSort(lean_object*);
uint8_t l_Lean_Name_isInternalDetail(lean_object*);
uint8_t l_Lean_isAuxRecursor(lean_object*, lean_object*);
uint8_t l_Lean_isNoConfusion(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_ConstantInfo_toConstantVal(lean_object*);
lean_object* l_Lean_isSubobjectField_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_componentsRev(lean_object*);
lean_object* l_List_splitAt___redArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_mkNot(lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Level_dec(lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_getStructureFields(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_mkApp(lean_object*, lean_object*);
lean_object* lean_find_expr(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkSorry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_levelParams(lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedDeclaration_default;
lean_object* l_Lean_Meta_isMatcher___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_has_loose_bvar(lean_object*, lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__0 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__0_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__1 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_BinderInfo_brackets___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__0_value),((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__1_value)}};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__2 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__2_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{{"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__3 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__3_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "}}"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__4 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_BinderInfo_brackets___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__3_value),((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__5 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__5_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__6 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__6_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__7 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_BinderInfo_brackets___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__6_value),((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__7_value)}};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__8 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__8_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__9 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__9_value;
static const lean_string_object lp_mathlib_Lean_BinderInfo_brackets___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__10 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_BinderInfo_brackets___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__9_value),((lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_BinderInfo_brackets___closed__11 = (const lean_object*)&lp_mathlib_Lean_BinderInfo_brackets___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_BinderInfo_brackets(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_BinderInfo_brackets___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_mapPrefix(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Name_fromComponents_go(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_updateLast(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Name_lastComponentAsString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_Name_lastComponentAsString___closed__0 = (const lean_object*)&lp_mathlib_Lean_Name_lastComponentAsString___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_lastComponentAsString(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_splitAt(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Lean_Name_isPrefixOf_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Name_isPrefixOf_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Name_isPrefixOf_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isPrefixOf_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isPrefixOf_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "sorryAx"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Name_isBlackListed___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Name_isBlackListed___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 190, 164, 146, 38, 179, 69, 72)}};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "noConfusionType"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inj"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_ConstantInfo_isDef(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_isDef___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_ConstantInfo_isThm(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_isThm___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateConstantVal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateName(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateType(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateLevelParams(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateAll(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateValue(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Mathlib.Lean.Expr.Basic"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0_value;
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Lean.ConstantInfo.toDeclaration!"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1_value;
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "toDeclaration for quotInfo not implemented"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__2 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3;
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "toDeclaration for inductInfo not implemented"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__4 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5;
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "toDeclaration for ctorInfo not implemented"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__6 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7;
static const lean_string_object lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "toDeclaration for recInfo not implemented"};
static const lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__8 = (const lean_object*)&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConst_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConst_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_bvarIdx_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_bvarIdx_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getAppAppsAux(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Expr_getAppApps___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_getAppApps___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getAppApps(lean_object*);
static const lean_ctor_object lp_mathlib_Lean_Expr_eraseProofs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_eraseProofs___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transform"};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___closed__0_value;
static const lean_array_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Expr_eraseProofs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Expr_eraseProofs___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Expr_eraseProofs___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_eraseProofs___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Expr_eraseProofs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Expr_eraseProofs___lam__1___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Expr_eraseProofs___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_eraseProofs___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_type_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_type_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConstP(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConstP___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConst___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getUnusedForallInstanceBinderIdxsWhere_go(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_hasInstanceBinderOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_hasInstanceBinderOf___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_letDepth(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_letDepth___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "tactic failed, resulting expression contains metavariables"};
static const lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_ofNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Lean_Expr_ofNat___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_ofNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Lean_Expr_ofNat___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ofNat___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_ofNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Lean_Expr_ofNat___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_ofNat___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Expr_ofInt___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_ofInt___closed__0;
static const lean_string_object lp_mathlib_Lean_Expr_ofInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Lean_Expr_ofInt___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_ofInt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Lean_Expr_ofInt___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ofInt___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__1_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_ofInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Lean_Expr_ofInt___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_ofInt___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Lean_Expr_numeral_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_numeral_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_numeral_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_numeral_x3f___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_numeral_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 81, 163, 94, 71, 156, 90, 186)}};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Expr_numeral_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "succ"};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_numeral_x3f___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_numeral_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(93, 165, 73, 246, 125, 40, 156, 223)}};
static const lean_object* lp_mathlib_Lean_Expr_numeral_x3f___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_numeral_x3f___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_numeral_x3f(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_zero_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_zero_x3f___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ne_x3f_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__4_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_ne_x3f_x27___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_le_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Lean_Expr_le_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_le_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Lean_Expr_le_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_le_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_le_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Lean_Expr_le_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_le_x3f___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_le_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_le_x3f___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_lt_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Lean_Expr_lt_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_lt_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Lean_Expr_lt_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_lt_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_lt_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Lean_Expr_lt_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_lt_x3f___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_lt_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_lt_x3f___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_sides_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Lean_Expr_sides_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_sides_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Lean_Expr_sides_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_sides_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_mathlib_Lean_Expr_sides_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_sides_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(67, 180, 169, 191, 74, 196, 152, 188)}};
static const lean_object* lp_mathlib_Lean_Expr_sides_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_sides_x3f___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_sides_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_sides_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isSorryAx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isSorryAx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyAppArgM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyAppArgM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyRevArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyRevArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getRevArg_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getRevArg_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getArg_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getArg_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_renameBVar(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_renameBVar___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getBinderName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getBinderName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mapForallBinderNames(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_mkDirectProjection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " doesn't have field "};
static const lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_mkDirectProjection___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_mkDirectProjection___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___closed__1;
static const lean_string_object lp_mathlib_Lean_Expr_mkDirectProjection___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = " doesn't have a structure as type"};
static const lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_mkDirectProjection___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_mkDirectProjection___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkDirectProjection(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Expr_mkProjection_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_mkProjection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_mkProjection___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_mkProjection___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_mkProjection___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_mkProjection___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_mkProjection___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_mkProjection___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__3;
static const lean_string_object lp_mathlib_Lean_Expr_mkProjection___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "No parent of "};
static const lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_mkProjection___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_mkProjection___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__5;
static const lean_string_object lp_mathlib_Lean_Expr_mkProjection___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " has field "};
static const lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_mkProjection___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_mkProjection___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_mkProjection___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkProjection(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkProjection___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "ill-formed expression, "};
static const lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1;
static const lean_string_object lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = " is the "};
static const lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3;
static const lean_string_object lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "-th projection function but "};
static const lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5;
static const lean_string_object lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " does not have enough arguments"};
static const lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "forall_not_of_not_exists"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(64, 176, 52, 188, 216, 118, 163, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_getFieldsToParents___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_getFieldsToParents___closed__0 = (const lean_object*)&lp_mathlib_Lean_getFieldsToParents___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getFieldsToParents(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_BinderInfo_brackets(uint8_t v_x_21_){
_start:
{
switch(v_x_21_)
{
case 1:
{
lean_object* v___x_22_; 
v___x_22_ = ((lean_object*)(lp_mathlib_Lean_BinderInfo_brackets___closed__2));
return v___x_22_;
}
case 2:
{
lean_object* v___x_23_; 
v___x_23_ = ((lean_object*)(lp_mathlib_Lean_BinderInfo_brackets___closed__5));
return v___x_23_;
}
case 3:
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Lean_BinderInfo_brackets___closed__8));
return v___x_24_;
}
default: 
{
lean_object* v___x_25_; 
v___x_25_ = ((lean_object*)(lp_mathlib_Lean_BinderInfo_brackets___closed__11));
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_BinderInfo_brackets___boxed(lean_object* v_x_26_){
_start:
{
uint8_t v_x_91__boxed_27_; lean_object* v_res_28_; 
v_x_91__boxed_27_ = lean_unbox(v_x_26_);
v_res_28_ = lp_mathlib_Lean_BinderInfo_brackets(v_x_91__boxed_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_mapPrefix(lean_object* v_f_29_, lean_object* v_n_30_){
_start:
{
lean_object* v___x_31_; 
lean_inc_ref(v_f_29_);
lean_inc(v_n_30_);
v___x_31_ = lean_apply_1(v_f_29_, v_n_30_);
if (lean_obj_tag(v___x_31_) == 1)
{
lean_object* v_val_32_; 
lean_dec(v_n_30_);
lean_dec_ref(v_f_29_);
v_val_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc(v_val_32_);
lean_dec_ref_known(v___x_31_, 1);
return v_val_32_;
}
else
{
lean_dec(v___x_31_);
switch(lean_obj_tag(v_n_30_))
{
case 0:
{
lean_dec_ref(v_f_29_);
return v_n_30_;
}
case 1:
{
lean_object* v_pre_33_; lean_object* v_str_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v_pre_33_ = lean_ctor_get(v_n_30_, 0);
lean_inc(v_pre_33_);
v_str_34_ = lean_ctor_get(v_n_30_, 1);
lean_inc_ref(v_str_34_);
lean_dec_ref_known(v_n_30_, 2);
v___x_35_ = lp_mathlib_Lean_Name_mapPrefix(v_f_29_, v_pre_33_);
v___x_36_ = l_Lean_Name_str___override(v___x_35_, v_str_34_);
return v___x_36_;
}
default: 
{
lean_object* v_pre_37_; lean_object* v_i_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v_pre_37_ = lean_ctor_get(v_n_30_, 0);
lean_inc(v_pre_37_);
v_i_38_ = lean_ctor_get(v_n_30_, 1);
lean_inc(v_i_38_);
lean_dec_ref_known(v_n_30_, 2);
v___x_39_ = lp_mathlib_Lean_Name_mapPrefix(v_f_29_, v_pre_37_);
v___x_40_ = l_Lean_Name_num___override(v___x_39_, v_i_38_);
return v___x_40_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Name_fromComponents_go(lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
if (lean_obj_tag(v_a_42_) == 0)
{
return v_a_41_;
}
else
{
lean_object* v_head_43_; lean_object* v_tail_44_; lean_object* v___x_45_; 
v_head_43_ = lean_ctor_get(v_a_42_, 0);
lean_inc(v_head_43_);
v_tail_44_ = lean_ctor_get(v_a_42_, 1);
lean_inc(v_tail_44_);
lean_dec_ref_known(v_a_42_, 2);
v___x_45_ = l_Lean_Name_updatePrefix(v_head_43_, v_a_41_);
v_a_41_ = v___x_45_;
v_a_42_ = v_tail_44_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_fromComponents(lean_object* v_a_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_box(0);
v___x_49_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Name_fromComponents_go(v___x_48_, v_a_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_updateLast(lean_object* v_f_50_, lean_object* v_x_51_){
_start:
{
if (lean_obj_tag(v_x_51_) == 1)
{
lean_object* v_pre_52_; lean_object* v_str_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v_pre_52_ = lean_ctor_get(v_x_51_, 0);
lean_inc(v_pre_52_);
v_str_53_ = lean_ctor_get(v_x_51_, 1);
lean_inc_ref(v_str_53_);
lean_dec_ref_known(v_x_51_, 2);
v___x_54_ = lean_apply_1(v_f_50_, v_str_53_);
v___x_55_ = l_Lean_Name_str___override(v_pre_52_, v___x_54_);
return v___x_55_;
}
else
{
lean_dec_ref(v_f_50_);
return v_x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_lastComponentAsString(lean_object* v_x_57_){
_start:
{
switch(lean_obj_tag(v_x_57_))
{
case 0:
{
lean_object* v___x_58_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Lean_Name_lastComponentAsString___closed__0));
return v___x_58_;
}
case 1:
{
lean_object* v_str_59_; 
v_str_59_ = lean_ctor_get(v_x_57_, 1);
lean_inc_ref(v_str_59_);
lean_dec_ref_known(v_x_57_, 2);
return v_str_59_;
}
default: 
{
lean_object* v_i_60_; lean_object* v___x_61_; 
v_i_60_ = lean_ctor_get(v_x_57_, 1);
lean_inc(v_i_60_);
lean_dec_ref_known(v_x_57_, 2);
v___x_61_ = l_Nat_reprFast(v_i_60_);
return v___x_61_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_splitAt(lean_object* v_nm_62_, lean_object* v_n_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v_fst_66_; lean_object* v_snd_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_78_; 
v___x_64_ = l_Lean_Name_componentsRev(v_nm_62_);
v___x_65_ = l_List_splitAt___redArg(v_n_63_, v___x_64_);
v_fst_66_ = lean_ctor_get(v___x_65_, 0);
v_snd_67_ = lean_ctor_get(v___x_65_, 1);
v_isSharedCheck_78_ = !lean_is_exclusive(v___x_65_);
if (v_isSharedCheck_78_ == 0)
{
v___x_69_ = v___x_65_;
v_isShared_70_ = v_isSharedCheck_78_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_snd_67_);
lean_inc(v_fst_66_);
lean_dec(v___x_65_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_78_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_76_; 
v___x_71_ = l_List_reverse___redArg(v_snd_67_);
v___x_72_ = lp_mathlib_Lean_Name_fromComponents(v___x_71_);
v___x_73_ = l_List_reverse___redArg(v_fst_66_);
v___x_74_ = lp_mathlib_Lean_Name_fromComponents(v___x_73_);
if (v_isShared_70_ == 0)
{
lean_ctor_set(v___x_69_, 1, v___x_74_);
lean_ctor_set(v___x_69_, 0, v___x_72_);
v___x_76_ = v___x_69_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v___x_72_);
lean_ctor_set(v_reuseFailAlloc_77_, 1, v___x_74_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isPrefixOf_x3f(lean_object* v_pre_81_, lean_object* v_nm_82_){
_start:
{
uint8_t v___x_83_; 
v___x_83_ = lean_name_eq(v_pre_81_, v_nm_82_);
if (v___x_83_ == 0)
{
switch(lean_obj_tag(v_nm_82_))
{
case 0:
{
lean_object* v___x_84_; 
v___x_84_ = lean_box(0);
return v___x_84_;
}
case 1:
{
lean_object* v_pre_85_; lean_object* v_str_86_; lean_object* v___x_87_; 
v_pre_85_ = lean_ctor_get(v_nm_82_, 0);
lean_inc(v_pre_85_);
v_str_86_ = lean_ctor_get(v_nm_82_, 1);
lean_inc_ref(v_str_86_);
lean_dec_ref_known(v_nm_82_, 2);
v___x_87_ = lp_mathlib_Lean_Name_isPrefixOf_x3f(v_pre_81_, v_pre_85_);
if (lean_obj_tag(v___x_87_) == 0)
{
lean_dec_ref(v_str_86_);
return v___x_87_;
}
else
{
lean_object* v_val_88_; lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_96_; 
v_val_88_ = lean_ctor_get(v___x_87_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_87_);
if (v_isSharedCheck_96_ == 0)
{
v___x_90_ = v___x_87_;
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
else
{
lean_inc(v_val_88_);
lean_dec(v___x_87_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_92_ = l_Lean_Name_str___override(v_val_88_, v_str_86_);
if (v_isShared_91_ == 0)
{
lean_ctor_set(v___x_90_, 0, v___x_92_);
v___x_94_ = v___x_90_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_92_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
default: 
{
lean_object* v_pre_97_; lean_object* v_i_98_; lean_object* v___x_99_; 
v_pre_97_ = lean_ctor_get(v_nm_82_, 0);
lean_inc(v_pre_97_);
v_i_98_ = lean_ctor_get(v_nm_82_, 1);
lean_inc(v_i_98_);
lean_dec_ref_known(v_nm_82_, 2);
v___x_99_ = lp_mathlib_Lean_Name_isPrefixOf_x3f(v_pre_81_, v_pre_97_);
if (lean_obj_tag(v___x_99_) == 0)
{
lean_dec(v_i_98_);
return v___x_99_;
}
else
{
lean_object* v_val_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_108_; 
v_val_100_ = lean_ctor_get(v___x_99_, 0);
v_isSharedCheck_108_ = !lean_is_exclusive(v___x_99_);
if (v_isSharedCheck_108_ == 0)
{
v___x_102_ = v___x_99_;
v_isShared_103_ = v_isSharedCheck_108_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_val_100_);
lean_dec(v___x_99_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_108_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___x_104_; lean_object* v___x_106_; 
v___x_104_ = l_Lean_Name_num___override(v_val_100_, v_i_98_);
if (v_isShared_103_ == 0)
{
lean_ctor_set(v___x_102_, 0, v___x_104_);
v___x_106_ = v___x_102_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v___x_104_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
}
}
}
else
{
lean_object* v___x_109_; 
lean_dec(v_nm_82_);
v___x_109_ = ((lean_object*)(lp_mathlib_Lean_Name_isPrefixOf_x3f___closed__0));
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isPrefixOf_x3f___boxed(lean_object* v_pre_110_, lean_object* v_nm_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_Lean_Name_isPrefixOf_x3f(v_pre_110_, v_nm_111_);
lean_dec(v_pre_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0(lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_declName_115_, lean_object* v_toPure_116_, uint8_t v_b_117_){
_start:
{
if (v_b_117_ == 0)
{
lean_object* v___x_118_; 
lean_dec(v_toPure_116_);
v___x_118_ = l_Lean_Meta_isMatcher___redArg(v_inst_113_, v_inst_114_, v_declName_115_);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; 
lean_dec(v_declName_115_);
lean_dec_ref(v_inst_114_);
lean_dec_ref(v_inst_113_);
v___x_119_ = lean_box(v_b_117_);
v___x_120_ = lean_apply_2(v_toPure_116_, lean_box(0), v___x_119_);
return v___x_120_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0___boxed(lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_declName_123_, lean_object* v_toPure_124_, lean_object* v_b_125_){
_start:
{
uint8_t v_b_boxed_126_; lean_object* v_res_127_; 
v_b_boxed_126_ = lean_unbox(v_b_125_);
v_res_127_ = lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0(v_inst_121_, v_inst_122_, v_declName_123_, v_toPure_124_, v_b_boxed_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1(lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_declName_130_, lean_object* v_toBind_131_, lean_object* v___f_132_, lean_object* v_toPure_133_, uint8_t v_b_134_){
_start:
{
if (v_b_134_ == 0)
{
lean_object* v___x_135_; lean_object* v___x_136_; 
lean_dec(v_toPure_133_);
v___x_135_ = l_Lean_isRec___redArg(v_inst_128_, v_inst_129_, v_declName_130_);
v___x_136_ = lean_apply_4(v_toBind_131_, lean_box(0), lean_box(0), v___x_135_, v___f_132_);
return v___x_136_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; 
lean_dec(v___f_132_);
lean_dec(v_toBind_131_);
lean_dec(v_declName_130_);
lean_dec_ref(v_inst_129_);
lean_dec_ref(v_inst_128_);
v___x_137_ = lean_box(v_b_134_);
v___x_138_ = lean_apply_2(v_toPure_133_, lean_box(0), v___x_137_);
return v___x_138_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1___boxed(lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_declName_141_, lean_object* v_toBind_142_, lean_object* v___f_143_, lean_object* v_toPure_144_, lean_object* v_b_145_){
_start:
{
uint8_t v_b_boxed_146_; lean_object* v_res_147_; 
v_b_boxed_146_ = lean_unbox(v_b_145_);
v_res_147_ = lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1(v_inst_139_, v_inst_140_, v_declName_141_, v_toBind_142_, v___f_143_, v_toPure_144_, v_b_boxed_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2(lean_object* v_declName_148_, lean_object* v_toPure_149_, uint8_t v___x_150_, lean_object* v_env_151_){
_start:
{
uint8_t v___x_152_; 
lean_inc(v_declName_148_);
v___x_152_ = l_Lean_Name_isInternalDetail(v_declName_148_);
if (v___x_152_ == 0)
{
uint8_t v___x_153_; 
lean_inc(v_declName_148_);
lean_inc_ref(v_env_151_);
v___x_153_ = l_Lean_isAuxRecursor(v_env_151_, v_declName_148_);
if (v___x_153_ == 0)
{
uint8_t v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = l_Lean_isNoConfusion(v_env_151_, v_declName_148_);
v___x_155_ = lean_box(v___x_154_);
v___x_156_ = lean_apply_2(v_toPure_149_, lean_box(0), v___x_155_);
return v___x_156_;
}
else
{
lean_object* v___x_157_; lean_object* v___x_158_; 
lean_dec_ref(v_env_151_);
lean_dec(v_declName_148_);
v___x_157_ = lean_box(v___x_150_);
v___x_158_ = lean_apply_2(v_toPure_149_, lean_box(0), v___x_157_);
return v___x_158_;
}
}
else
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec_ref(v_env_151_);
lean_dec(v_declName_148_);
v___x_159_ = lean_box(v___x_150_);
v___x_160_ = lean_apply_2(v_toPure_149_, lean_box(0), v___x_159_);
return v___x_160_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2___boxed(lean_object* v_declName_161_, lean_object* v_toPure_162_, lean_object* v___x_163_, lean_object* v_env_164_){
_start:
{
uint8_t v___x_466__boxed_165_; lean_object* v_res_166_; 
v___x_466__boxed_165_ = lean_unbox(v___x_163_);
v_res_166_ = lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2(v_declName_161_, v_toPure_162_, v___x_466__boxed_165_, v_env_164_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___redArg(lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_declName_174_){
_start:
{
lean_object* v_toApplicative_175_; lean_object* v_toBind_176_; lean_object* v_toPure_177_; lean_object* v___x_178_; uint8_t v___x_179_; uint8_t v___x_180_; lean_object* v___f_181_; lean_object* v___f_182_; 
v_toApplicative_175_ = lean_ctor_get(v_inst_172_, 0);
v_toBind_176_ = lean_ctor_get(v_inst_172_, 1);
lean_inc_n(v_toBind_176_, 2);
v_toPure_177_ = lean_ctor_get(v_toApplicative_175_, 1);
lean_inc_n(v_toPure_177_, 3);
v___x_178_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___redArg___closed__1));
v___x_179_ = lean_name_eq(v_declName_174_, v___x_178_);
v___x_180_ = 1;
lean_inc_n(v_declName_174_, 2);
lean_inc_ref_n(v_inst_173_, 2);
lean_inc_ref(v_inst_172_);
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Name_isBlackListed___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_181_, 0, v_inst_172_);
lean_closure_set(v___f_181_, 1, v_inst_173_);
lean_closure_set(v___f_181_, 2, v_declName_174_);
lean_closure_set(v___f_181_, 3, v_toPure_177_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Name_isBlackListed___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_182_, 0, v_inst_172_);
lean_closure_set(v___f_182_, 1, v_inst_173_);
lean_closure_set(v___f_182_, 2, v_declName_174_);
lean_closure_set(v___f_182_, 3, v_toBind_176_);
lean_closure_set(v___f_182_, 4, v___f_181_);
lean_closure_set(v___f_182_, 5, v_toPure_177_);
if (v___x_179_ == 0)
{
lean_object* v___x_183_; lean_object* v___f_184_; 
v___x_183_ = lean_box(v___x_180_);
lean_inc(v_toPure_177_);
lean_inc(v_declName_174_);
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Name_isBlackListed___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_184_, 0, v_declName_174_);
lean_closure_set(v___f_184_, 1, v_toPure_177_);
lean_closure_set(v___f_184_, 2, v___x_183_);
if (lean_obj_tag(v_declName_174_) == 1)
{
lean_object* v_str_196_; lean_object* v___x_197_; uint8_t v___x_198_; 
v_str_196_ = lean_ctor_get(v_declName_174_, 1);
v___x_197_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___redArg___closed__3));
v___x_198_ = lean_string_dec_eq(v_str_196_, v___x_197_);
if (v___x_198_ == 0)
{
goto v___jp_189_;
}
else
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec_ref_known(v_declName_174_, 2);
lean_dec_ref(v___f_184_);
lean_dec_ref(v_inst_173_);
v___x_199_ = lean_box(v___x_180_);
v___x_200_ = lean_apply_2(v_toPure_177_, lean_box(0), v___x_199_);
v___x_201_ = lean_apply_4(v_toBind_176_, lean_box(0), lean_box(0), v___x_200_, v___f_182_);
return v___x_201_;
}
}
else
{
goto v___jp_189_;
}
v___jp_185_:
{
lean_object* v_getEnv_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v_getEnv_186_ = lean_ctor_get(v_inst_173_, 0);
lean_inc(v_getEnv_186_);
lean_dec_ref(v_inst_173_);
lean_inc(v_toBind_176_);
v___x_187_ = lean_apply_4(v_toBind_176_, lean_box(0), lean_box(0), v_getEnv_186_, v___f_184_);
v___x_188_ = lean_apply_4(v_toBind_176_, lean_box(0), lean_box(0), v___x_187_, v___f_182_);
return v___x_188_;
}
v___jp_189_:
{
if (lean_obj_tag(v_declName_174_) == 1)
{
lean_object* v_str_190_; lean_object* v___x_191_; uint8_t v___x_192_; 
v_str_190_ = lean_ctor_get(v_declName_174_, 1);
lean_inc_ref(v_str_190_);
lean_dec_ref_known(v_declName_174_, 2);
v___x_191_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___redArg___closed__2));
v___x_192_ = lean_string_dec_eq(v_str_190_, v___x_191_);
lean_dec_ref(v_str_190_);
if (v___x_192_ == 0)
{
lean_dec(v_toPure_177_);
goto v___jp_185_;
}
else
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec_ref(v___f_184_);
lean_dec_ref(v_inst_173_);
v___x_193_ = lean_box(v___x_180_);
v___x_194_ = lean_apply_2(v_toPure_177_, lean_box(0), v___x_193_);
v___x_195_ = lean_apply_4(v_toBind_176_, lean_box(0), lean_box(0), v___x_194_, v___f_182_);
return v___x_195_;
}
}
else
{
lean_dec(v_toPure_177_);
lean_dec(v_declName_174_);
goto v___jp_185_;
}
}
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
lean_dec(v_declName_174_);
lean_dec_ref(v_inst_173_);
v___x_202_ = lean_box(v___x_180_);
v___x_203_ = lean_apply_2(v_toPure_177_, lean_box(0), v___x_202_);
v___x_204_ = lean_apply_4(v_toBind_176_, lean_box(0), lean_box(0), v___x_203_, v___f_182_);
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed(lean_object* v_m_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_declName_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_Lean_Name_isBlackListed___redArg(v_inst_206_, v_inst_207_, v_declName_208_);
return v___x_209_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_ConstantInfo_isDef(lean_object* v_x_210_){
_start:
{
if (lean_obj_tag(v_x_210_) == 1)
{
uint8_t v___x_211_; 
v___x_211_ = 1;
return v___x_211_;
}
else
{
uint8_t v___x_212_; 
v___x_212_ = 0;
return v___x_212_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_isDef___boxed(lean_object* v_x_213_){
_start:
{
uint8_t v_res_214_; lean_object* v_r_215_; 
v_res_214_ = lp_mathlib_Lean_ConstantInfo_isDef(v_x_213_);
lean_dec_ref(v_x_213_);
v_r_215_ = lean_box(v_res_214_);
return v_r_215_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_ConstantInfo_isThm(lean_object* v_x_216_){
_start:
{
if (lean_obj_tag(v_x_216_) == 2)
{
uint8_t v___x_217_; 
v___x_217_ = 1;
return v___x_217_;
}
else
{
uint8_t v___x_218_; 
v___x_218_ = 0;
return v___x_218_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_isThm___boxed(lean_object* v_x_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_mathlib_Lean_ConstantInfo_isThm(v_x_219_);
lean_dec_ref(v_x_219_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateConstantVal(lean_object* v_x_222_, lean_object* v_x_223_){
_start:
{
switch(lean_obj_tag(v_x_222_))
{
case 0:
{
lean_object* v_val_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_240_; 
v_val_224_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_240_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_240_ == 0)
{
v___x_226_ = v_x_222_;
v_isShared_227_ = v_isSharedCheck_240_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_val_224_);
lean_dec(v_x_222_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_240_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
uint8_t v_isUnsafe_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_238_; 
v_isUnsafe_228_ = lean_ctor_get_uint8(v_val_224_, sizeof(void*)*1);
v_isSharedCheck_238_ = !lean_is_exclusive(v_val_224_);
if (v_isSharedCheck_238_ == 0)
{
lean_object* v_unused_239_; 
v_unused_239_ = lean_ctor_get(v_val_224_, 0);
lean_dec(v_unused_239_);
v___x_230_ = v_val_224_;
v_isShared_231_ = v_isSharedCheck_238_;
goto v_resetjp_229_;
}
else
{
lean_dec(v_val_224_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_238_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v_x_223_);
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_x_223_);
lean_ctor_set_uint8(v_reuseFailAlloc_237_, sizeof(void*)*1, v_isUnsafe_228_);
v___x_233_ = v_reuseFailAlloc_237_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
lean_object* v___x_235_; 
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 0, v___x_233_);
v___x_235_ = v___x_226_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v___x_233_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
}
}
}
case 1:
{
lean_object* v_val_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_260_; 
v_val_241_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_260_ == 0)
{
v___x_243_ = v_x_222_;
v_isShared_244_ = v_isSharedCheck_260_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_val_241_);
lean_dec(v_x_222_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_260_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v_value_245_; lean_object* v_hints_246_; uint8_t v_safety_247_; lean_object* v_all_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_258_; 
v_value_245_ = lean_ctor_get(v_val_241_, 1);
v_hints_246_ = lean_ctor_get(v_val_241_, 2);
v_safety_247_ = lean_ctor_get_uint8(v_val_241_, sizeof(void*)*4);
v_all_248_ = lean_ctor_get(v_val_241_, 3);
v_isSharedCheck_258_ = !lean_is_exclusive(v_val_241_);
if (v_isSharedCheck_258_ == 0)
{
lean_object* v_unused_259_; 
v_unused_259_ = lean_ctor_get(v_val_241_, 0);
lean_dec(v_unused_259_);
v___x_250_ = v_val_241_;
v_isShared_251_ = v_isSharedCheck_258_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_all_248_);
lean_inc(v_hints_246_);
lean_inc(v_value_245_);
lean_dec(v_val_241_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_258_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 0, v_x_223_);
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_257_, 1, v_value_245_);
lean_ctor_set(v_reuseFailAlloc_257_, 2, v_hints_246_);
lean_ctor_set(v_reuseFailAlloc_257_, 3, v_all_248_);
lean_ctor_set_uint8(v_reuseFailAlloc_257_, sizeof(void*)*4, v_safety_247_);
v___x_253_ = v_reuseFailAlloc_257_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
lean_object* v___x_255_; 
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 0, v___x_253_);
v___x_255_ = v___x_243_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v___x_253_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
}
case 2:
{
lean_object* v_val_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_278_; 
v_val_261_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_278_ == 0)
{
v___x_263_ = v_x_222_;
v_isShared_264_ = v_isSharedCheck_278_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_val_261_);
lean_dec(v_x_222_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_278_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
lean_object* v_value_265_; lean_object* v_all_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_276_; 
v_value_265_ = lean_ctor_get(v_val_261_, 1);
v_all_266_ = lean_ctor_get(v_val_261_, 2);
v_isSharedCheck_276_ = !lean_is_exclusive(v_val_261_);
if (v_isSharedCheck_276_ == 0)
{
lean_object* v_unused_277_; 
v_unused_277_ = lean_ctor_get(v_val_261_, 0);
lean_dec(v_unused_277_);
v___x_268_ = v_val_261_;
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_all_266_);
lean_inc(v_value_265_);
lean_dec(v_val_261_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_271_; 
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 0, v_x_223_);
v___x_271_ = v___x_268_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_275_, 1, v_value_265_);
lean_ctor_set(v_reuseFailAlloc_275_, 2, v_all_266_);
v___x_271_ = v_reuseFailAlloc_275_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
lean_object* v___x_273_; 
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 0, v___x_271_);
v___x_273_ = v___x_263_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v___x_271_);
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
case 3:
{
lean_object* v_val_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_297_; 
v_val_279_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_297_ == 0)
{
v___x_281_ = v_x_222_;
v_isShared_282_ = v_isSharedCheck_297_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_val_279_);
lean_dec(v_x_222_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_297_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v_value_283_; uint8_t v_isUnsafe_284_; lean_object* v_all_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_295_; 
v_value_283_ = lean_ctor_get(v_val_279_, 1);
v_isUnsafe_284_ = lean_ctor_get_uint8(v_val_279_, sizeof(void*)*3);
v_all_285_ = lean_ctor_get(v_val_279_, 2);
v_isSharedCheck_295_ = !lean_is_exclusive(v_val_279_);
if (v_isSharedCheck_295_ == 0)
{
lean_object* v_unused_296_; 
v_unused_296_ = lean_ctor_get(v_val_279_, 0);
lean_dec(v_unused_296_);
v___x_287_ = v_val_279_;
v_isShared_288_ = v_isSharedCheck_295_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_all_285_);
lean_inc(v_value_283_);
lean_dec(v_val_279_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_295_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
lean_ctor_set(v___x_287_, 0, v_x_223_);
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_294_, 1, v_value_283_);
lean_ctor_set(v_reuseFailAlloc_294_, 2, v_all_285_);
lean_ctor_set_uint8(v_reuseFailAlloc_294_, sizeof(void*)*3, v_isUnsafe_284_);
v___x_290_ = v_reuseFailAlloc_294_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_292_; 
if (v_isShared_282_ == 0)
{
lean_ctor_set(v___x_281_, 0, v___x_290_);
v___x_292_ = v___x_281_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_290_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
return v___x_292_;
}
}
}
}
}
case 4:
{
lean_object* v_val_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_314_; 
v_val_298_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_314_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_314_ == 0)
{
v___x_300_ = v_x_222_;
v_isShared_301_ = v_isSharedCheck_314_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_val_298_);
lean_dec(v_x_222_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_314_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
uint8_t v_kind_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_312_; 
v_kind_302_ = lean_ctor_get_uint8(v_val_298_, sizeof(void*)*1);
v_isSharedCheck_312_ = !lean_is_exclusive(v_val_298_);
if (v_isSharedCheck_312_ == 0)
{
lean_object* v_unused_313_; 
v_unused_313_ = lean_ctor_get(v_val_298_, 0);
lean_dec(v_unused_313_);
v___x_304_ = v_val_298_;
v_isShared_305_ = v_isSharedCheck_312_;
goto v_resetjp_303_;
}
else
{
lean_dec(v_val_298_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_312_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v___x_307_; 
if (v_isShared_305_ == 0)
{
lean_ctor_set(v___x_304_, 0, v_x_223_);
v___x_307_ = v___x_304_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v_x_223_);
lean_ctor_set_uint8(v_reuseFailAlloc_311_, sizeof(void*)*1, v_kind_302_);
v___x_307_ = v_reuseFailAlloc_311_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
lean_object* v___x_309_; 
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_307_);
v___x_309_ = v___x_300_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v___x_307_);
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
case 5:
{
lean_object* v_val_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_338_; 
v_val_315_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_338_ == 0)
{
v___x_317_ = v_x_222_;
v_isShared_318_ = v_isSharedCheck_338_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_val_315_);
lean_dec(v_x_222_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_338_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v_numParams_319_; lean_object* v_numIndices_320_; lean_object* v_all_321_; lean_object* v_ctors_322_; lean_object* v_numNested_323_; uint8_t v_isRec_324_; uint8_t v_isUnsafe_325_; uint8_t v_isReflexive_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_336_; 
v_numParams_319_ = lean_ctor_get(v_val_315_, 1);
v_numIndices_320_ = lean_ctor_get(v_val_315_, 2);
v_all_321_ = lean_ctor_get(v_val_315_, 3);
v_ctors_322_ = lean_ctor_get(v_val_315_, 4);
v_numNested_323_ = lean_ctor_get(v_val_315_, 5);
v_isRec_324_ = lean_ctor_get_uint8(v_val_315_, sizeof(void*)*6);
v_isUnsafe_325_ = lean_ctor_get_uint8(v_val_315_, sizeof(void*)*6 + 1);
v_isReflexive_326_ = lean_ctor_get_uint8(v_val_315_, sizeof(void*)*6 + 2);
v_isSharedCheck_336_ = !lean_is_exclusive(v_val_315_);
if (v_isSharedCheck_336_ == 0)
{
lean_object* v_unused_337_; 
v_unused_337_ = lean_ctor_get(v_val_315_, 0);
lean_dec(v_unused_337_);
v___x_328_ = v_val_315_;
v_isShared_329_ = v_isSharedCheck_336_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_numNested_323_);
lean_inc(v_ctors_322_);
lean_inc(v_all_321_);
lean_inc(v_numIndices_320_);
lean_inc(v_numParams_319_);
lean_dec(v_val_315_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_336_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_331_; 
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 0, v_x_223_);
v___x_331_ = v___x_328_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 6, 3);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v_numParams_319_);
lean_ctor_set(v_reuseFailAlloc_335_, 2, v_numIndices_320_);
lean_ctor_set(v_reuseFailAlloc_335_, 3, v_all_321_);
lean_ctor_set(v_reuseFailAlloc_335_, 4, v_ctors_322_);
lean_ctor_set(v_reuseFailAlloc_335_, 5, v_numNested_323_);
lean_ctor_set_uint8(v_reuseFailAlloc_335_, sizeof(void*)*6, v_isRec_324_);
lean_ctor_set_uint8(v_reuseFailAlloc_335_, sizeof(void*)*6 + 1, v_isUnsafe_325_);
lean_ctor_set_uint8(v_reuseFailAlloc_335_, sizeof(void*)*6 + 2, v_isReflexive_326_);
v___x_331_ = v_reuseFailAlloc_335_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
lean_object* v___x_333_; 
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 0, v___x_331_);
v___x_333_ = v___x_317_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_331_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
}
case 6:
{
lean_object* v_val_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_359_; 
v_val_339_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_359_ == 0)
{
v___x_341_ = v_x_222_;
v_isShared_342_ = v_isSharedCheck_359_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_val_339_);
lean_dec(v_x_222_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_359_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v_induct_343_; lean_object* v_cidx_344_; lean_object* v_numParams_345_; lean_object* v_numFields_346_; uint8_t v_isUnsafe_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_357_; 
v_induct_343_ = lean_ctor_get(v_val_339_, 1);
v_cidx_344_ = lean_ctor_get(v_val_339_, 2);
v_numParams_345_ = lean_ctor_get(v_val_339_, 3);
v_numFields_346_ = lean_ctor_get(v_val_339_, 4);
v_isUnsafe_347_ = lean_ctor_get_uint8(v_val_339_, sizeof(void*)*5);
v_isSharedCheck_357_ = !lean_is_exclusive(v_val_339_);
if (v_isSharedCheck_357_ == 0)
{
lean_object* v_unused_358_; 
v_unused_358_ = lean_ctor_get(v_val_339_, 0);
lean_dec(v_unused_358_);
v___x_349_ = v_val_339_;
v_isShared_350_ = v_isSharedCheck_357_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_numFields_346_);
lean_inc(v_numParams_345_);
lean_inc(v_cidx_344_);
lean_inc(v_induct_343_);
lean_dec(v_val_339_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_357_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_352_; 
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 0, v_x_223_);
v___x_352_ = v___x_349_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_356_, 1, v_induct_343_);
lean_ctor_set(v_reuseFailAlloc_356_, 2, v_cidx_344_);
lean_ctor_set(v_reuseFailAlloc_356_, 3, v_numParams_345_);
lean_ctor_set(v_reuseFailAlloc_356_, 4, v_numFields_346_);
lean_ctor_set_uint8(v_reuseFailAlloc_356_, sizeof(void*)*5, v_isUnsafe_347_);
v___x_352_ = v_reuseFailAlloc_356_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
lean_object* v___x_354_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 0, v___x_352_);
v___x_354_ = v___x_341_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_352_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
}
default: 
{
lean_object* v_val_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_383_; 
v_val_360_ = lean_ctor_get(v_x_222_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v_x_222_);
if (v_isSharedCheck_383_ == 0)
{
v___x_362_ = v_x_222_;
v_isShared_363_ = v_isSharedCheck_383_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_val_360_);
lean_dec(v_x_222_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_383_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v_all_364_; lean_object* v_numParams_365_; lean_object* v_numIndices_366_; lean_object* v_numMotives_367_; lean_object* v_numMinors_368_; lean_object* v_rules_369_; uint8_t v_k_370_; uint8_t v_isUnsafe_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_381_; 
v_all_364_ = lean_ctor_get(v_val_360_, 1);
v_numParams_365_ = lean_ctor_get(v_val_360_, 2);
v_numIndices_366_ = lean_ctor_get(v_val_360_, 3);
v_numMotives_367_ = lean_ctor_get(v_val_360_, 4);
v_numMinors_368_ = lean_ctor_get(v_val_360_, 5);
v_rules_369_ = lean_ctor_get(v_val_360_, 6);
v_k_370_ = lean_ctor_get_uint8(v_val_360_, sizeof(void*)*7);
v_isUnsafe_371_ = lean_ctor_get_uint8(v_val_360_, sizeof(void*)*7 + 1);
v_isSharedCheck_381_ = !lean_is_exclusive(v_val_360_);
if (v_isSharedCheck_381_ == 0)
{
lean_object* v_unused_382_; 
v_unused_382_ = lean_ctor_get(v_val_360_, 0);
lean_dec(v_unused_382_);
v___x_373_ = v_val_360_;
v_isShared_374_ = v_isSharedCheck_381_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_rules_369_);
lean_inc(v_numMinors_368_);
lean_inc(v_numMotives_367_);
lean_inc(v_numIndices_366_);
lean_inc(v_numParams_365_);
lean_inc(v_all_364_);
lean_dec(v_val_360_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_381_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_376_; 
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v_x_223_);
v___x_376_ = v___x_373_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 7, 2);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_x_223_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v_all_364_);
lean_ctor_set(v_reuseFailAlloc_380_, 2, v_numParams_365_);
lean_ctor_set(v_reuseFailAlloc_380_, 3, v_numIndices_366_);
lean_ctor_set(v_reuseFailAlloc_380_, 4, v_numMotives_367_);
lean_ctor_set(v_reuseFailAlloc_380_, 5, v_numMinors_368_);
lean_ctor_set(v_reuseFailAlloc_380_, 6, v_rules_369_);
lean_ctor_set_uint8(v_reuseFailAlloc_380_, sizeof(void*)*7, v_k_370_);
lean_ctor_set_uint8(v_reuseFailAlloc_380_, sizeof(void*)*7 + 1, v_isUnsafe_371_);
v___x_376_ = v_reuseFailAlloc_380_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
lean_object* v___x_378_; 
if (v_isShared_363_ == 0)
{
lean_ctor_set(v___x_362_, 0, v___x_376_);
v___x_378_ = v___x_362_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(7, 1, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v___x_376_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateName(lean_object* v_c_384_, lean_object* v_name_385_){
_start:
{
lean_object* v___x_386_; lean_object* v_levelParams_387_; lean_object* v_type_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_396_; 
v___x_386_ = l_Lean_ConstantInfo_toConstantVal(v_c_384_);
v_levelParams_387_ = lean_ctor_get(v___x_386_, 1);
v_type_388_ = lean_ctor_get(v___x_386_, 2);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_396_ == 0)
{
lean_object* v_unused_397_; 
v_unused_397_ = lean_ctor_get(v___x_386_, 0);
lean_dec(v_unused_397_);
v___x_390_ = v___x_386_;
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_type_388_);
lean_inc(v_levelParams_387_);
lean_dec(v___x_386_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_393_; 
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 0, v_name_385_);
v___x_393_ = v___x_390_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v_name_385_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v_levelParams_387_);
lean_ctor_set(v_reuseFailAlloc_395_, 2, v_type_388_);
v___x_393_ = v_reuseFailAlloc_395_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_Lean_ConstantInfo_updateConstantVal(v_c_384_, v___x_393_);
return v___x_394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateType(lean_object* v_c_398_, lean_object* v_type_399_){
_start:
{
lean_object* v___x_400_; lean_object* v_name_401_; lean_object* v_levelParams_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_410_; 
v___x_400_ = l_Lean_ConstantInfo_toConstantVal(v_c_398_);
v_name_401_ = lean_ctor_get(v___x_400_, 0);
v_levelParams_402_ = lean_ctor_get(v___x_400_, 1);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_400_);
if (v_isSharedCheck_410_ == 0)
{
lean_object* v_unused_411_; 
v_unused_411_ = lean_ctor_get(v___x_400_, 2);
lean_dec(v_unused_411_);
v___x_404_ = v___x_400_;
v_isShared_405_ = v_isSharedCheck_410_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_levelParams_402_);
lean_inc(v_name_401_);
lean_dec(v___x_400_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_410_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 2, v_type_399_);
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_name_401_);
lean_ctor_set(v_reuseFailAlloc_409_, 1, v_levelParams_402_);
lean_ctor_set(v_reuseFailAlloc_409_, 2, v_type_399_);
v___x_407_ = v_reuseFailAlloc_409_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_Lean_ConstantInfo_updateConstantVal(v_c_398_, v___x_407_);
return v___x_408_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateLevelParams(lean_object* v_c_412_, lean_object* v_levelParams_413_){
_start:
{
lean_object* v___x_414_; lean_object* v_name_415_; lean_object* v_type_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_424_; 
v___x_414_ = l_Lean_ConstantInfo_toConstantVal(v_c_412_);
v_name_415_ = lean_ctor_get(v___x_414_, 0);
v_type_416_ = lean_ctor_get(v___x_414_, 2);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_424_ == 0)
{
lean_object* v_unused_425_; 
v_unused_425_ = lean_ctor_get(v___x_414_, 1);
lean_dec(v_unused_425_);
v___x_418_ = v___x_414_;
v_isShared_419_ = v_isSharedCheck_424_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_type_416_);
lean_inc(v_name_415_);
lean_dec(v___x_414_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_424_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v___x_421_; 
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 1, v_levelParams_413_);
v___x_421_ = v___x_418_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_name_415_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v_levelParams_413_);
lean_ctor_set(v_reuseFailAlloc_423_, 2, v_type_416_);
v___x_421_ = v_reuseFailAlloc_423_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_Lean_ConstantInfo_updateConstantVal(v_c_412_, v___x_421_);
return v___x_422_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateAll(lean_object* v_x_426_, lean_object* v_x_427_){
_start:
{
switch(lean_obj_tag(v_x_426_))
{
case 1:
{
lean_object* v_val_428_; lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_447_; 
v_val_428_ = lean_ctor_get(v_x_426_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v_x_426_);
if (v_isSharedCheck_447_ == 0)
{
v___x_430_ = v_x_426_;
v_isShared_431_ = v_isSharedCheck_447_;
goto v_resetjp_429_;
}
else
{
lean_inc(v_val_428_);
lean_dec(v_x_426_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_447_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v_toConstantVal_432_; lean_object* v_value_433_; lean_object* v_hints_434_; uint8_t v_safety_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_445_; 
v_toConstantVal_432_ = lean_ctor_get(v_val_428_, 0);
v_value_433_ = lean_ctor_get(v_val_428_, 1);
v_hints_434_ = lean_ctor_get(v_val_428_, 2);
v_safety_435_ = lean_ctor_get_uint8(v_val_428_, sizeof(void*)*4);
v_isSharedCheck_445_ = !lean_is_exclusive(v_val_428_);
if (v_isSharedCheck_445_ == 0)
{
lean_object* v_unused_446_; 
v_unused_446_ = lean_ctor_get(v_val_428_, 3);
lean_dec(v_unused_446_);
v___x_437_ = v_val_428_;
v_isShared_438_ = v_isSharedCheck_445_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_hints_434_);
lean_inc(v_value_433_);
lean_inc(v_toConstantVal_432_);
lean_dec(v_val_428_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_445_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v___x_440_; 
if (v_isShared_438_ == 0)
{
lean_ctor_set(v___x_437_, 3, v_x_427_);
v___x_440_ = v___x_437_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v_toConstantVal_432_);
lean_ctor_set(v_reuseFailAlloc_444_, 1, v_value_433_);
lean_ctor_set(v_reuseFailAlloc_444_, 2, v_hints_434_);
lean_ctor_set(v_reuseFailAlloc_444_, 3, v_x_427_);
lean_ctor_set_uint8(v_reuseFailAlloc_444_, sizeof(void*)*4, v_safety_435_);
v___x_440_ = v_reuseFailAlloc_444_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
lean_object* v___x_442_; 
if (v_isShared_431_ == 0)
{
lean_ctor_set(v___x_430_, 0, v___x_440_);
v___x_442_ = v___x_430_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_440_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
}
case 2:
{
lean_object* v_val_448_; lean_object* v___x_450_; uint8_t v_isShared_451_; uint8_t v_isSharedCheck_465_; 
v_val_448_ = lean_ctor_get(v_x_426_, 0);
v_isSharedCheck_465_ = !lean_is_exclusive(v_x_426_);
if (v_isSharedCheck_465_ == 0)
{
v___x_450_ = v_x_426_;
v_isShared_451_ = v_isSharedCheck_465_;
goto v_resetjp_449_;
}
else
{
lean_inc(v_val_448_);
lean_dec(v_x_426_);
v___x_450_ = lean_box(0);
v_isShared_451_ = v_isSharedCheck_465_;
goto v_resetjp_449_;
}
v_resetjp_449_:
{
lean_object* v_toConstantVal_452_; lean_object* v_value_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_463_; 
v_toConstantVal_452_ = lean_ctor_get(v_val_448_, 0);
v_value_453_ = lean_ctor_get(v_val_448_, 1);
v_isSharedCheck_463_ = !lean_is_exclusive(v_val_448_);
if (v_isSharedCheck_463_ == 0)
{
lean_object* v_unused_464_; 
v_unused_464_ = lean_ctor_get(v_val_448_, 2);
lean_dec(v_unused_464_);
v___x_455_ = v_val_448_;
v_isShared_456_ = v_isSharedCheck_463_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_value_453_);
lean_inc(v_toConstantVal_452_);
lean_dec(v_val_448_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_463_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
lean_ctor_set(v___x_455_, 2, v_x_427_);
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_toConstantVal_452_);
lean_ctor_set(v_reuseFailAlloc_462_, 1, v_value_453_);
lean_ctor_set(v_reuseFailAlloc_462_, 2, v_x_427_);
v___x_458_ = v_reuseFailAlloc_462_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
lean_object* v___x_460_; 
if (v_isShared_451_ == 0)
{
lean_ctor_set(v___x_450_, 0, v___x_458_);
v___x_460_ = v___x_450_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v___x_458_);
v___x_460_ = v_reuseFailAlloc_461_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
return v___x_460_;
}
}
}
}
}
case 3:
{
lean_object* v_val_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_484_; 
v_val_466_ = lean_ctor_get(v_x_426_, 0);
v_isSharedCheck_484_ = !lean_is_exclusive(v_x_426_);
if (v_isSharedCheck_484_ == 0)
{
v___x_468_ = v_x_426_;
v_isShared_469_ = v_isSharedCheck_484_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_val_466_);
lean_dec(v_x_426_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_484_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v_toConstantVal_470_; lean_object* v_value_471_; uint8_t v_isUnsafe_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_482_; 
v_toConstantVal_470_ = lean_ctor_get(v_val_466_, 0);
v_value_471_ = lean_ctor_get(v_val_466_, 1);
v_isUnsafe_472_ = lean_ctor_get_uint8(v_val_466_, sizeof(void*)*3);
v_isSharedCheck_482_ = !lean_is_exclusive(v_val_466_);
if (v_isSharedCheck_482_ == 0)
{
lean_object* v_unused_483_; 
v_unused_483_ = lean_ctor_get(v_val_466_, 2);
lean_dec(v_unused_483_);
v___x_474_ = v_val_466_;
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_value_471_);
lean_inc(v_toConstantVal_470_);
lean_dec(v_val_466_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_477_; 
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 2, v_x_427_);
v___x_477_ = v___x_474_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_toConstantVal_470_);
lean_ctor_set(v_reuseFailAlloc_481_, 1, v_value_471_);
lean_ctor_set(v_reuseFailAlloc_481_, 2, v_x_427_);
lean_ctor_set_uint8(v_reuseFailAlloc_481_, sizeof(void*)*3, v_isUnsafe_472_);
v___x_477_ = v_reuseFailAlloc_481_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_479_; 
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 0, v___x_477_);
v___x_479_ = v___x_468_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v___x_477_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
case 5:
{
lean_object* v_val_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_508_; 
v_val_485_ = lean_ctor_get(v_x_426_, 0);
v_isSharedCheck_508_ = !lean_is_exclusive(v_x_426_);
if (v_isSharedCheck_508_ == 0)
{
v___x_487_ = v_x_426_;
v_isShared_488_ = v_isSharedCheck_508_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_val_485_);
lean_dec(v_x_426_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_508_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v_toConstantVal_489_; lean_object* v_numParams_490_; lean_object* v_numIndices_491_; lean_object* v_ctors_492_; lean_object* v_numNested_493_; uint8_t v_isRec_494_; uint8_t v_isUnsafe_495_; uint8_t v_isReflexive_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_506_; 
v_toConstantVal_489_ = lean_ctor_get(v_val_485_, 0);
v_numParams_490_ = lean_ctor_get(v_val_485_, 1);
v_numIndices_491_ = lean_ctor_get(v_val_485_, 2);
v_ctors_492_ = lean_ctor_get(v_val_485_, 4);
v_numNested_493_ = lean_ctor_get(v_val_485_, 5);
v_isRec_494_ = lean_ctor_get_uint8(v_val_485_, sizeof(void*)*6);
v_isUnsafe_495_ = lean_ctor_get_uint8(v_val_485_, sizeof(void*)*6 + 1);
v_isReflexive_496_ = lean_ctor_get_uint8(v_val_485_, sizeof(void*)*6 + 2);
v_isSharedCheck_506_ = !lean_is_exclusive(v_val_485_);
if (v_isSharedCheck_506_ == 0)
{
lean_object* v_unused_507_; 
v_unused_507_ = lean_ctor_get(v_val_485_, 3);
lean_dec(v_unused_507_);
v___x_498_ = v_val_485_;
v_isShared_499_ = v_isSharedCheck_506_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_numNested_493_);
lean_inc(v_ctors_492_);
lean_inc(v_numIndices_491_);
lean_inc(v_numParams_490_);
lean_inc(v_toConstantVal_489_);
lean_dec(v_val_485_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_506_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_501_; 
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 3, v_x_427_);
v___x_501_ = v___x_498_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 6, 3);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_toConstantVal_489_);
lean_ctor_set(v_reuseFailAlloc_505_, 1, v_numParams_490_);
lean_ctor_set(v_reuseFailAlloc_505_, 2, v_numIndices_491_);
lean_ctor_set(v_reuseFailAlloc_505_, 3, v_x_427_);
lean_ctor_set(v_reuseFailAlloc_505_, 4, v_ctors_492_);
lean_ctor_set(v_reuseFailAlloc_505_, 5, v_numNested_493_);
lean_ctor_set_uint8(v_reuseFailAlloc_505_, sizeof(void*)*6, v_isRec_494_);
lean_ctor_set_uint8(v_reuseFailAlloc_505_, sizeof(void*)*6 + 1, v_isUnsafe_495_);
lean_ctor_set_uint8(v_reuseFailAlloc_505_, sizeof(void*)*6 + 2, v_isReflexive_496_);
v___x_501_ = v_reuseFailAlloc_505_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
lean_object* v___x_503_; 
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 0, v___x_501_);
v___x_503_ = v___x_487_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v___x_501_);
v___x_503_ = v_reuseFailAlloc_504_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
return v___x_503_;
}
}
}
}
}
default: 
{
lean_dec(v_x_427_);
return v_x_426_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_updateValue(lean_object* v_x_509_, lean_object* v_x_510_){
_start:
{
switch(lean_obj_tag(v_x_509_))
{
case 1:
{
lean_object* v_val_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_530_; 
v_val_511_ = lean_ctor_get(v_x_509_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v_x_509_);
if (v_isSharedCheck_530_ == 0)
{
v___x_513_ = v_x_509_;
v_isShared_514_ = v_isSharedCheck_530_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_val_511_);
lean_dec(v_x_509_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_530_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v_toConstantVal_515_; lean_object* v_hints_516_; uint8_t v_safety_517_; lean_object* v_all_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_528_; 
v_toConstantVal_515_ = lean_ctor_get(v_val_511_, 0);
v_hints_516_ = lean_ctor_get(v_val_511_, 2);
v_safety_517_ = lean_ctor_get_uint8(v_val_511_, sizeof(void*)*4);
v_all_518_ = lean_ctor_get(v_val_511_, 3);
v_isSharedCheck_528_ = !lean_is_exclusive(v_val_511_);
if (v_isSharedCheck_528_ == 0)
{
lean_object* v_unused_529_; 
v_unused_529_ = lean_ctor_get(v_val_511_, 1);
lean_dec(v_unused_529_);
v___x_520_ = v_val_511_;
v_isShared_521_ = v_isSharedCheck_528_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_all_518_);
lean_inc(v_hints_516_);
lean_inc(v_toConstantVal_515_);
lean_dec(v_val_511_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_528_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_523_; 
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 1, v_x_510_);
v___x_523_ = v___x_520_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_toConstantVal_515_);
lean_ctor_set(v_reuseFailAlloc_527_, 1, v_x_510_);
lean_ctor_set(v_reuseFailAlloc_527_, 2, v_hints_516_);
lean_ctor_set(v_reuseFailAlloc_527_, 3, v_all_518_);
lean_ctor_set_uint8(v_reuseFailAlloc_527_, sizeof(void*)*4, v_safety_517_);
v___x_523_ = v_reuseFailAlloc_527_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
lean_object* v___x_525_; 
if (v_isShared_514_ == 0)
{
lean_ctor_set(v___x_513_, 0, v___x_523_);
v___x_525_ = v___x_513_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v___x_523_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
}
case 2:
{
lean_object* v_val_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_548_; 
v_val_531_ = lean_ctor_get(v_x_509_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v_x_509_);
if (v_isSharedCheck_548_ == 0)
{
v___x_533_ = v_x_509_;
v_isShared_534_ = v_isSharedCheck_548_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_val_531_);
lean_dec(v_x_509_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_548_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v_toConstantVal_535_; lean_object* v_all_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_546_; 
v_toConstantVal_535_ = lean_ctor_get(v_val_531_, 0);
v_all_536_ = lean_ctor_get(v_val_531_, 2);
v_isSharedCheck_546_ = !lean_is_exclusive(v_val_531_);
if (v_isSharedCheck_546_ == 0)
{
lean_object* v_unused_547_; 
v_unused_547_ = lean_ctor_get(v_val_531_, 1);
lean_dec(v_unused_547_);
v___x_538_ = v_val_531_;
v_isShared_539_ = v_isSharedCheck_546_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_all_536_);
lean_inc(v_toConstantVal_535_);
lean_dec(v_val_531_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_546_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_541_; 
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 1, v_x_510_);
v___x_541_ = v___x_538_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_toConstantVal_535_);
lean_ctor_set(v_reuseFailAlloc_545_, 1, v_x_510_);
lean_ctor_set(v_reuseFailAlloc_545_, 2, v_all_536_);
v___x_541_ = v_reuseFailAlloc_545_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
lean_object* v___x_543_; 
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 0, v___x_541_);
v___x_543_ = v___x_533_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v___x_541_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
}
}
}
case 3:
{
lean_object* v_val_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_567_; 
v_val_549_ = lean_ctor_get(v_x_509_, 0);
v_isSharedCheck_567_ = !lean_is_exclusive(v_x_509_);
if (v_isSharedCheck_567_ == 0)
{
v___x_551_ = v_x_509_;
v_isShared_552_ = v_isSharedCheck_567_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_val_549_);
lean_dec(v_x_509_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_567_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v_toConstantVal_553_; uint8_t v_isUnsafe_554_; lean_object* v_all_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_565_; 
v_toConstantVal_553_ = lean_ctor_get(v_val_549_, 0);
v_isUnsafe_554_ = lean_ctor_get_uint8(v_val_549_, sizeof(void*)*3);
v_all_555_ = lean_ctor_get(v_val_549_, 2);
v_isSharedCheck_565_ = !lean_is_exclusive(v_val_549_);
if (v_isSharedCheck_565_ == 0)
{
lean_object* v_unused_566_; 
v_unused_566_ = lean_ctor_get(v_val_549_, 1);
lean_dec(v_unused_566_);
v___x_557_ = v_val_549_;
v_isShared_558_ = v_isSharedCheck_565_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_all_555_);
lean_inc(v_toConstantVal_553_);
lean_dec(v_val_549_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_565_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_560_; 
if (v_isShared_558_ == 0)
{
lean_ctor_set(v___x_557_, 1, v_x_510_);
v___x_560_ = v___x_557_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_toConstantVal_553_);
lean_ctor_set(v_reuseFailAlloc_564_, 1, v_x_510_);
lean_ctor_set(v_reuseFailAlloc_564_, 2, v_all_555_);
lean_ctor_set_uint8(v_reuseFailAlloc_564_, sizeof(void*)*3, v_isUnsafe_554_);
v___x_560_ = v_reuseFailAlloc_564_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
lean_object* v___x_562_; 
if (v_isShared_552_ == 0)
{
lean_ctor_set(v___x_551_, 0, v___x_560_);
v___x_562_ = v___x_551_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_560_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
default: 
{
lean_dec_ref(v_x_510_);
return v_x_509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(lean_object* v_msg_568_){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = l_Lean_instInhabitedDeclaration_default;
v___x_570_ = lean_panic_fn_borrowed(v___x_569_, v_msg_568_);
return v___x_570_;
}
}
static lean_object* _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3(void){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_574_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__2));
v___x_575_ = lean_unsigned_to_nat(20u);
v___x_576_ = lean_unsigned_to_nat(170u);
v___x_577_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1));
v___x_578_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0));
v___x_579_ = l_mkPanicMessageWithDecl(v___x_578_, v___x_577_, v___x_576_, v___x_575_, v___x_574_);
return v___x_579_;
}
}
static lean_object* _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5(void){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_581_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__4));
v___x_582_ = lean_unsigned_to_nat(20u);
v___x_583_ = lean_unsigned_to_nat(171u);
v___x_584_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1));
v___x_585_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0));
v___x_586_ = l_mkPanicMessageWithDecl(v___x_585_, v___x_584_, v___x_583_, v___x_582_, v___x_581_);
return v___x_586_;
}
}
static lean_object* _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_588_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__6));
v___x_589_ = lean_unsigned_to_nat(20u);
v___x_590_ = lean_unsigned_to_nat(172u);
v___x_591_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1));
v___x_592_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0));
v___x_593_ = l_mkPanicMessageWithDecl(v___x_592_, v___x_591_, v___x_590_, v___x_589_, v___x_588_);
return v___x_593_;
}
}
static lean_object* _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9(void){
_start:
{
lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
v___x_595_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__8));
v___x_596_ = lean_unsigned_to_nat(20u);
v___x_597_ = lean_unsigned_to_nat(173u);
v___x_598_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__1));
v___x_599_ = ((lean_object*)(lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__0));
v___x_600_ = l_mkPanicMessageWithDecl(v___x_599_, v___x_598_, v___x_597_, v___x_596_, v___x_595_);
return v___x_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ConstantInfo_toDeclaration_x21(lean_object* v_x_601_){
_start:
{
switch(lean_obj_tag(v_x_601_))
{
case 0:
{
lean_object* v_val_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_609_; 
v_val_602_ = lean_ctor_get(v_x_601_, 0);
v_isSharedCheck_609_ = !lean_is_exclusive(v_x_601_);
if (v_isSharedCheck_609_ == 0)
{
v___x_604_ = v_x_601_;
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_val_602_);
lean_dec(v_x_601_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
lean_object* v___x_607_; 
if (v_isShared_605_ == 0)
{
v___x_607_ = v___x_604_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_val_602_);
v___x_607_ = v_reuseFailAlloc_608_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
return v___x_607_;
}
}
}
case 1:
{
lean_object* v_val_610_; lean_object* v___x_612_; uint8_t v_isShared_613_; uint8_t v_isSharedCheck_617_; 
v_val_610_ = lean_ctor_get(v_x_601_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v_x_601_);
if (v_isSharedCheck_617_ == 0)
{
v___x_612_ = v_x_601_;
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
else
{
lean_inc(v_val_610_);
lean_dec(v_x_601_);
v___x_612_ = lean_box(0);
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
v_resetjp_611_:
{
lean_object* v___x_615_; 
if (v_isShared_613_ == 0)
{
v___x_615_ = v___x_612_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v_val_610_);
v___x_615_ = v_reuseFailAlloc_616_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
return v___x_615_;
}
}
}
case 2:
{
lean_object* v_val_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
v_val_618_ = lean_ctor_get(v_x_601_, 0);
v_isSharedCheck_625_ = !lean_is_exclusive(v_x_601_);
if (v_isSharedCheck_625_ == 0)
{
v___x_620_ = v_x_601_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_val_618_);
lean_dec(v_x_601_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_623_; 
if (v_isShared_621_ == 0)
{
v___x_623_ = v___x_620_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_val_618_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
case 3:
{
lean_object* v_val_626_; lean_object* v___x_628_; uint8_t v_isShared_629_; uint8_t v_isSharedCheck_633_; 
v_val_626_ = lean_ctor_get(v_x_601_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v_x_601_);
if (v_isSharedCheck_633_ == 0)
{
v___x_628_ = v_x_601_;
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
else
{
lean_inc(v_val_626_);
lean_dec(v_x_601_);
v___x_628_ = lean_box(0);
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
v_resetjp_627_:
{
lean_object* v___x_631_; 
if (v_isShared_629_ == 0)
{
v___x_631_ = v___x_628_;
goto v_reusejp_630_;
}
else
{
lean_object* v_reuseFailAlloc_632_; 
v_reuseFailAlloc_632_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_632_, 0, v_val_626_);
v___x_631_ = v_reuseFailAlloc_632_;
goto v_reusejp_630_;
}
v_reusejp_630_:
{
return v___x_631_;
}
}
}
case 4:
{
lean_object* v___x_634_; lean_object* v___x_635_; 
lean_dec_ref_known(v_x_601_, 1);
v___x_634_ = lean_obj_once(&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3, &lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3_once, _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__3);
v___x_635_ = lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(v___x_634_);
return v___x_635_;
}
case 5:
{
lean_object* v___x_636_; lean_object* v___x_637_; 
lean_dec_ref_known(v_x_601_, 1);
v___x_636_ = lean_obj_once(&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5, &lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5_once, _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__5);
v___x_637_ = lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(v___x_636_);
return v___x_637_;
}
case 6:
{
lean_object* v___x_638_; lean_object* v___x_639_; 
lean_dec_ref_known(v_x_601_, 1);
v___x_638_ = lean_obj_once(&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7, &lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7_once, _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__7);
v___x_639_ = lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(v___x_638_);
return v___x_639_;
}
default: 
{
lean_object* v___x_640_; lean_object* v___x_641_; 
lean_dec_ref_known(v_x_601_, 1);
v___x_640_ = lean_obj_once(&lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9, &lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9_once, _init_lp_mathlib_Lean_ConstantInfo_toDeclaration_x21___closed__9);
v___x_641_ = lp_mathlib_panic___at___00Lean_ConstantInfo_toDeclaration_x21_spec__0(v___x_640_);
return v___x_641_;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_642_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_643_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0);
v___x_644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_644_, 0, v___x_643_);
return v___x_644_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_645_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_646_ = lean_unsigned_to_nat(0u);
v___x_647_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_647_, 0, v___x_646_);
lean_ctor_set(v___x_647_, 1, v___x_646_);
lean_ctor_set(v___x_647_, 2, v___x_646_);
lean_ctor_set(v___x_647_, 3, v___x_646_);
lean_ctor_set(v___x_647_, 4, v___x_645_);
lean_ctor_set(v___x_647_, 5, v___x_645_);
lean_ctor_set(v___x_647_, 6, v___x_645_);
lean_ctor_set(v___x_647_, 7, v___x_645_);
lean_ctor_set(v___x_647_, 8, v___x_645_);
lean_ctor_set(v___x_647_, 9, v___x_645_);
return v___x_647_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_648_ = lean_unsigned_to_nat(32u);
v___x_649_ = lean_mk_empty_array_with_capacity(v___x_648_);
v___x_650_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_650_, 0, v___x_649_);
return v___x_650_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4(void){
_start:
{
size_t v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_651_ = ((size_t)5ULL);
v___x_652_ = lean_unsigned_to_nat(0u);
v___x_653_ = lean_unsigned_to_nat(32u);
v___x_654_ = lean_mk_empty_array_with_capacity(v___x_653_);
v___x_655_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3);
v___x_656_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_656_, 0, v___x_655_);
lean_ctor_set(v___x_656_, 1, v___x_654_);
lean_ctor_set(v___x_656_, 2, v___x_652_);
lean_ctor_set(v___x_656_, 3, v___x_652_);
lean_ctor_set_usize(v___x_656_, 4, v___x_651_);
return v___x_656_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_657_ = lean_box(1);
v___x_658_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4);
v___x_659_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_660_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_660_, 0, v___x_659_);
lean_ctor_set(v___x_660_, 1, v___x_658_);
lean_ctor_set(v___x_660_, 2, v___x_657_);
return v___x_660_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6));
v___x_663_ = l_Lean_stringToMessageData(v___x_662_);
return v___x_663_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9(void){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_665_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8));
v___x_666_ = l_Lean_stringToMessageData(v___x_665_);
return v___x_666_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11(void){
_start:
{
lean_object* v___x_668_; lean_object* v___x_669_; 
v___x_668_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10));
v___x_669_ = l_Lean_stringToMessageData(v___x_668_);
return v___x_669_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13(void){
_start:
{
lean_object* v___x_671_; lean_object* v___x_672_; 
v___x_671_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12));
v___x_672_ = l_Lean_stringToMessageData(v___x_671_);
return v___x_672_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15(void){
_start:
{
lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_674_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__14));
v___x_675_ = l_Lean_stringToMessageData(v___x_674_);
return v___x_675_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17(void){
_start:
{
lean_object* v___x_677_; lean_object* v___x_678_; 
v___x_677_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__16));
v___x_678_ = l_Lean_stringToMessageData(v___x_677_);
return v___x_678_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19(void){
_start:
{
lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_680_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__18));
v___x_681_ = l_Lean_stringToMessageData(v___x_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_msg_682_, lean_object* v_declHint_683_, lean_object* v___y_684_){
_start:
{
lean_object* v___x_686_; lean_object* v_env_687_; uint8_t v___x_688_; 
v___x_686_ = lean_st_ref_get(v___y_684_);
v_env_687_ = lean_ctor_get(v___x_686_, 0);
lean_inc_ref(v_env_687_);
lean_dec(v___x_686_);
v___x_688_ = l_Lean_Name_isAnonymous(v_declHint_683_);
if (v___x_688_ == 0)
{
uint8_t v_isExporting_689_; 
v_isExporting_689_ = lean_ctor_get_uint8(v_env_687_, sizeof(void*)*8);
if (v_isExporting_689_ == 0)
{
lean_object* v___x_690_; 
lean_dec_ref(v_env_687_);
lean_dec(v_declHint_683_);
v___x_690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_690_, 0, v_msg_682_);
return v___x_690_;
}
else
{
lean_object* v___x_691_; uint8_t v___x_692_; 
lean_inc_ref(v_env_687_);
v___x_691_ = l_Lean_Environment_setExporting(v_env_687_, v___x_688_);
lean_inc(v_declHint_683_);
lean_inc_ref(v___x_691_);
v___x_692_ = l_Lean_Environment_contains(v___x_691_, v_declHint_683_, v_isExporting_689_);
if (v___x_692_ == 0)
{
lean_object* v___x_693_; 
lean_dec_ref(v___x_691_);
lean_dec_ref(v_env_687_);
lean_dec(v_declHint_683_);
v___x_693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_693_, 0, v_msg_682_);
return v___x_693_;
}
else
{
lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v_c_699_; lean_object* v___x_700_; 
v___x_694_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2);
v___x_695_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5);
v___x_696_ = l_Lean_Options_empty;
v___x_697_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_697_, 0, v___x_691_);
lean_ctor_set(v___x_697_, 1, v___x_694_);
lean_ctor_set(v___x_697_, 2, v___x_695_);
lean_ctor_set(v___x_697_, 3, v___x_696_);
lean_inc(v_declHint_683_);
v___x_698_ = l_Lean_MessageData_ofConstName(v_declHint_683_, v___x_688_);
v_c_699_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_699_, 0, v___x_697_);
lean_ctor_set(v_c_699_, 1, v___x_698_);
v___x_700_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_687_, v_declHint_683_);
if (lean_obj_tag(v___x_700_) == 0)
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
lean_dec_ref(v_env_687_);
lean_dec(v_declHint_683_);
v___x_701_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7);
v___x_702_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
lean_ctor_set(v___x_702_, 1, v_c_699_);
v___x_703_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9);
v___x_704_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_704_, 0, v___x_702_);
lean_ctor_set(v___x_704_, 1, v___x_703_);
v___x_705_ = l_Lean_MessageData_note(v___x_704_);
v___x_706_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_706_, 0, v_msg_682_);
lean_ctor_set(v___x_706_, 1, v___x_705_);
v___x_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
else
{
lean_object* v_val_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_743_; 
v_val_708_ = lean_ctor_get(v___x_700_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_700_);
if (v_isSharedCheck_743_ == 0)
{
v___x_710_ = v___x_700_;
v_isShared_711_ = v_isSharedCheck_743_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_val_708_);
lean_dec(v___x_700_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_743_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v_mod_715_; uint8_t v___x_716_; 
v___x_712_ = lean_box(0);
v___x_713_ = l_Lean_Environment_header(v_env_687_);
lean_dec_ref(v_env_687_);
v___x_714_ = l_Lean_EnvironmentHeader_moduleNames(v___x_713_);
v_mod_715_ = lean_array_get(v___x_712_, v___x_714_, v_val_708_);
lean_dec(v_val_708_);
lean_dec_ref(v___x_714_);
v___x_716_ = l_Lean_isPrivateName(v_declHint_683_);
lean_dec(v_declHint_683_);
if (v___x_716_ == 0)
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_728_; 
v___x_717_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11);
v___x_718_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_717_);
lean_ctor_set(v___x_718_, 1, v_c_699_);
v___x_719_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13);
v___x_720_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_720_, 0, v___x_718_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
v___x_721_ = l_Lean_MessageData_ofName(v_mod_715_);
v___x_722_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_722_, 0, v___x_720_);
lean_ctor_set(v___x_722_, 1, v___x_721_);
v___x_723_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__15);
v___x_724_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_722_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = l_Lean_MessageData_note(v___x_724_);
v___x_726_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_726_, 0, v_msg_682_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
if (v_isShared_711_ == 0)
{
lean_ctor_set_tag(v___x_710_, 0);
lean_ctor_set(v___x_710_, 0, v___x_726_);
v___x_728_ = v___x_710_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_729_; 
v_reuseFailAlloc_729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_729_, 0, v___x_726_);
v___x_728_ = v_reuseFailAlloc_729_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
return v___x_728_;
}
}
else
{
lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_741_; 
v___x_730_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7);
v___x_731_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_731_, 0, v___x_730_);
lean_ctor_set(v___x_731_, 1, v_c_699_);
v___x_732_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__17);
v___x_733_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_733_, 0, v___x_731_);
lean_ctor_set(v___x_733_, 1, v___x_732_);
v___x_734_ = l_Lean_MessageData_ofName(v_mod_715_);
v___x_735_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_733_);
lean_ctor_set(v___x_735_, 1, v___x_734_);
v___x_736_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__19);
v___x_737_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_737_, 0, v___x_735_);
lean_ctor_set(v___x_737_, 1, v___x_736_);
v___x_738_ = l_Lean_MessageData_note(v___x_737_);
v___x_739_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_739_, 0, v_msg_682_);
lean_ctor_set(v___x_739_, 1, v___x_738_);
if (v_isShared_711_ == 0)
{
lean_ctor_set_tag(v___x_710_, 0);
lean_ctor_set(v___x_710_, 0, v___x_739_);
v___x_741_ = v___x_710_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v___x_739_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_744_; 
lean_dec_ref(v_env_687_);
lean_dec(v_declHint_683_);
v___x_744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_744_, 0, v_msg_682_);
return v___x_744_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___boxed(lean_object* v_msg_745_, lean_object* v_declHint_746_, lean_object* v___y_747_, lean_object* v___y_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_745_, v_declHint_746_, v___y_747_);
lean_dec(v___y_747_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4(lean_object* v_msg_750_, lean_object* v_declHint_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_){
_start:
{
lean_object* v___x_757_; lean_object* v_a_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_767_; 
v___x_757_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_750_, v_declHint_751_, v___y_755_);
v_a_758_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_767_ == 0)
{
v___x_760_ = v___x_757_;
v_isShared_761_ = v_isSharedCheck_767_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_a_758_);
lean_dec(v___x_757_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_767_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_765_; 
v___x_762_ = l_Lean_unknownIdentifierMessageTag;
v___x_763_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_763_, 0, v___x_762_);
lean_ctor_set(v___x_763_, 1, v_a_758_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 0, v___x_763_);
v___x_765_ = v___x_760_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v___x_763_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4___boxed(lean_object* v_msg_768_, lean_object* v_declHint_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4(v_msg_768_, v_declHint_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(lean_object* v_msgData_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v___x_782_; lean_object* v_env_783_; lean_object* v___x_784_; lean_object* v_mctx_785_; lean_object* v_lctx_786_; lean_object* v_options_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; 
v___x_782_ = lean_st_ref_get(v___y_780_);
v_env_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc_ref(v_env_783_);
lean_dec(v___x_782_);
v___x_784_ = lean_st_ref_get(v___y_778_);
v_mctx_785_ = lean_ctor_get(v___x_784_, 0);
lean_inc_ref(v_mctx_785_);
lean_dec(v___x_784_);
v_lctx_786_ = lean_ctor_get(v___y_777_, 2);
v_options_787_ = lean_ctor_get(v___y_779_, 2);
lean_inc_ref(v_options_787_);
lean_inc_ref(v_lctx_786_);
v___x_788_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_788_, 0, v_env_783_);
lean_ctor_set(v___x_788_, 1, v_mctx_785_);
lean_ctor_set(v___x_788_, 2, v_lctx_786_);
lean_ctor_set(v___x_788_, 3, v_options_787_);
v___x_789_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_788_);
lean_ctor_set(v___x_789_, 1, v_msgData_776_);
v___x_790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_790_, 0, v___x_789_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8___boxed(lean_object* v_msgData_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_){
_start:
{
lean_object* v_res_797_; 
v_res_797_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(v_msgData_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
lean_dec(v___y_795_);
lean_dec_ref(v___y_794_);
lean_dec(v___y_793_);
lean_dec_ref(v___y_792_);
return v_res_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(lean_object* v_msg_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_){
_start:
{
lean_object* v_ref_804_; lean_object* v___x_805_; lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_814_; 
v_ref_804_ = lean_ctor_get(v___y_801_, 5);
v___x_805_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(v_msg_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
v_a_806_ = lean_ctor_get(v___x_805_, 0);
v_isSharedCheck_814_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_814_ == 0)
{
v___x_808_ = v___x_805_;
v_isShared_809_ = v_isSharedCheck_814_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_805_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_814_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_810_; lean_object* v___x_812_; 
lean_inc(v_ref_804_);
v___x_810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_810_, 0, v_ref_804_);
lean_ctor_set(v___x_810_, 1, v_a_806_);
if (v_isShared_809_ == 0)
{
lean_ctor_set_tag(v___x_808_, 1);
lean_ctor_set(v___x_808_, 0, v___x_810_);
v___x_812_ = v___x_808_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v___x_810_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg___boxed(lean_object* v_msg_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v_res_821_; 
v_res_821_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_);
lean_dec(v___y_819_);
lean_dec_ref(v___y_818_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(lean_object* v_ref_822_, lean_object* v_msg_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_){
_start:
{
lean_object* v_fileName_829_; lean_object* v_fileMap_830_; lean_object* v_options_831_; lean_object* v_currRecDepth_832_; lean_object* v_maxRecDepth_833_; lean_object* v_ref_834_; lean_object* v_currNamespace_835_; lean_object* v_openDecls_836_; lean_object* v_initHeartbeats_837_; lean_object* v_maxHeartbeats_838_; lean_object* v_quotContext_839_; lean_object* v_currMacroScope_840_; uint8_t v_diag_841_; lean_object* v_cancelTk_x3f_842_; uint8_t v_suppressElabErrors_843_; lean_object* v_inheritedTraceOptions_844_; lean_object* v_ref_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v_fileName_829_ = lean_ctor_get(v___y_826_, 0);
v_fileMap_830_ = lean_ctor_get(v___y_826_, 1);
v_options_831_ = lean_ctor_get(v___y_826_, 2);
v_currRecDepth_832_ = lean_ctor_get(v___y_826_, 3);
v_maxRecDepth_833_ = lean_ctor_get(v___y_826_, 4);
v_ref_834_ = lean_ctor_get(v___y_826_, 5);
v_currNamespace_835_ = lean_ctor_get(v___y_826_, 6);
v_openDecls_836_ = lean_ctor_get(v___y_826_, 7);
v_initHeartbeats_837_ = lean_ctor_get(v___y_826_, 8);
v_maxHeartbeats_838_ = lean_ctor_get(v___y_826_, 9);
v_quotContext_839_ = lean_ctor_get(v___y_826_, 10);
v_currMacroScope_840_ = lean_ctor_get(v___y_826_, 11);
v_diag_841_ = lean_ctor_get_uint8(v___y_826_, sizeof(void*)*14);
v_cancelTk_x3f_842_ = lean_ctor_get(v___y_826_, 12);
v_suppressElabErrors_843_ = lean_ctor_get_uint8(v___y_826_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_844_ = lean_ctor_get(v___y_826_, 13);
v_ref_845_ = l_Lean_replaceRef(v_ref_822_, v_ref_834_);
lean_inc_ref(v_inheritedTraceOptions_844_);
lean_inc(v_cancelTk_x3f_842_);
lean_inc(v_currMacroScope_840_);
lean_inc(v_quotContext_839_);
lean_inc(v_maxHeartbeats_838_);
lean_inc(v_initHeartbeats_837_);
lean_inc(v_openDecls_836_);
lean_inc(v_currNamespace_835_);
lean_inc(v_maxRecDepth_833_);
lean_inc(v_currRecDepth_832_);
lean_inc_ref(v_options_831_);
lean_inc_ref(v_fileMap_830_);
lean_inc_ref(v_fileName_829_);
v___x_846_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_846_, 0, v_fileName_829_);
lean_ctor_set(v___x_846_, 1, v_fileMap_830_);
lean_ctor_set(v___x_846_, 2, v_options_831_);
lean_ctor_set(v___x_846_, 3, v_currRecDepth_832_);
lean_ctor_set(v___x_846_, 4, v_maxRecDepth_833_);
lean_ctor_set(v___x_846_, 5, v_ref_845_);
lean_ctor_set(v___x_846_, 6, v_currNamespace_835_);
lean_ctor_set(v___x_846_, 7, v_openDecls_836_);
lean_ctor_set(v___x_846_, 8, v_initHeartbeats_837_);
lean_ctor_set(v___x_846_, 9, v_maxHeartbeats_838_);
lean_ctor_set(v___x_846_, 10, v_quotContext_839_);
lean_ctor_set(v___x_846_, 11, v_currMacroScope_840_);
lean_ctor_set(v___x_846_, 12, v_cancelTk_x3f_842_);
lean_ctor_set(v___x_846_, 13, v_inheritedTraceOptions_844_);
lean_ctor_set_uint8(v___x_846_, sizeof(void*)*14, v_diag_841_);
lean_ctor_set_uint8(v___x_846_, sizeof(void*)*14 + 1, v_suppressElabErrors_843_);
v___x_847_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_823_, v___y_824_, v___y_825_, v___x_846_, v___y_827_);
lean_dec_ref_known(v___x_846_, 14);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_ref_848_, lean_object* v_msg_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_848_, v_msg_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
lean_dec(v___y_853_);
lean_dec_ref(v___y_852_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v_ref_848_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_ref_856_, lean_object* v_msg_857_, lean_object* v_declHint_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_){
_start:
{
lean_object* v___x_864_; lean_object* v_a_865_; lean_object* v___x_866_; 
v___x_864_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4(v_msg_857_, v_declHint_858_, v___y_859_, v___y_860_, v___y_861_, v___y_862_);
v_a_865_ = lean_ctor_get(v___x_864_, 0);
lean_inc(v_a_865_);
lean_dec_ref(v___x_864_);
v___x_866_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_856_, v_a_865_, v___y_859_, v___y_860_, v___y_861_, v___y_862_);
return v___x_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_ref_867_, lean_object* v_msg_868_, lean_object* v_declHint_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_867_, v_msg_868_, v_declHint_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
lean_dec(v___y_873_);
lean_dec_ref(v___y_872_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec(v_ref_867_);
return v_res_875_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_877_; lean_object* v___x_878_; 
v___x_877_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__0));
v___x_878_ = l_Lean_stringToMessageData(v___x_877_);
return v___x_878_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_880_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__2));
v___x_881_ = l_Lean_stringToMessageData(v___x_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_882_, lean_object* v_constName_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_){
_start:
{
lean_object* v___x_889_; uint8_t v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_889_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__1);
v___x_890_ = 0;
lean_inc(v_constName_883_);
v___x_891_ = l_Lean_MessageData_ofConstName(v_constName_883_, v___x_890_);
v___x_892_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_892_, 0, v___x_889_);
lean_ctor_set(v___x_892_, 1, v___x_891_);
v___x_893_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___closed__3);
v___x_894_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_894_, 0, v___x_892_);
lean_ctor_set(v___x_894_, 1, v___x_893_);
v___x_895_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_882_, v___x_894_, v_constName_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_896_, lean_object* v_constName_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_){
_start:
{
lean_object* v_res_903_; 
v_res_903_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg(v_ref_896_, v_constName_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_);
lean_dec(v___y_901_);
lean_dec_ref(v___y_900_);
lean_dec(v___y_899_);
lean_dec_ref(v___y_898_);
lean_dec(v_ref_896_);
return v_res_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg(lean_object* v_constName_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_){
_start:
{
lean_object* v_ref_910_; lean_object* v___x_911_; 
v_ref_910_ = lean_ctor_get(v___y_907_, 5);
v___x_911_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg(v_ref_910_, v_constName_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
return v___x_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg___boxed(lean_object* v_constName_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg(v_constName_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_);
lean_dec(v___y_916_);
lean_dec_ref(v___y_915_);
lean_dec(v___y_914_);
lean_dec_ref(v___y_913_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0(lean_object* v_constName_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
lean_object* v___x_925_; lean_object* v_env_926_; uint8_t v___x_927_; lean_object* v___x_928_; 
v___x_925_ = lean_st_ref_get(v___y_923_);
v_env_926_ = lean_ctor_get(v___x_925_, 0);
lean_inc_ref(v_env_926_);
lean_dec(v___x_925_);
v___x_927_ = 0;
lean_inc(v_constName_919_);
v___x_928_ = l_Lean_Environment_find_x3f(v_env_926_, v_constName_919_, v___x_927_);
if (lean_obj_tag(v___x_928_) == 0)
{
lean_object* v___x_929_; 
v___x_929_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg(v_constName_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_);
return v___x_929_;
}
else
{
lean_object* v_val_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
lean_dec(v_constName_919_);
v_val_930_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_937_ == 0)
{
v___x_932_ = v___x_928_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_val_930_);
lean_dec(v___x_928_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
lean_ctor_set_tag(v___x_932_, 0);
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_val_930_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0___boxed(lean_object* v_constName_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v_res_944_; 
v_res_944_ = lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0(v_constName_938_, v___y_939_, v___y_940_, v___y_941_, v___y_942_);
lean_dec(v___y_942_);
lean_dec_ref(v___y_941_);
lean_dec(v___y_940_);
lean_dec_ref(v___y_939_);
return v_res_944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1(lean_object* v_x_945_, lean_object* v_x_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_){
_start:
{
if (lean_obj_tag(v_x_945_) == 0)
{
lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_952_ = l_List_reverse___redArg(v_x_946_);
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
return v___x_953_;
}
else
{
lean_object* v_tail_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_972_; 
v_tail_954_ = lean_ctor_get(v_x_945_, 1);
v_isSharedCheck_972_ = !lean_is_exclusive(v_x_945_);
if (v_isSharedCheck_972_ == 0)
{
lean_object* v_unused_973_; 
v_unused_973_ = lean_ctor_get(v_x_945_, 0);
lean_dec(v_unused_973_);
v___x_956_ = v_x_945_;
v_isShared_957_ = v_isSharedCheck_972_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_tail_954_);
lean_dec(v_x_945_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_972_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_958_; 
v___x_958_ = l_Lean_Meta_mkFreshLevelMVar(v___y_947_, v___y_948_, v___y_949_, v___y_950_);
if (lean_obj_tag(v___x_958_) == 0)
{
lean_object* v_a_959_; lean_object* v___x_961_; 
v_a_959_ = lean_ctor_get(v___x_958_, 0);
lean_inc(v_a_959_);
lean_dec_ref_known(v___x_958_, 1);
if (v_isShared_957_ == 0)
{
lean_ctor_set(v___x_956_, 1, v_x_946_);
lean_ctor_set(v___x_956_, 0, v_a_959_);
v___x_961_ = v___x_956_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_963_; 
v_reuseFailAlloc_963_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_963_, 0, v_a_959_);
lean_ctor_set(v_reuseFailAlloc_963_, 1, v_x_946_);
v___x_961_ = v_reuseFailAlloc_963_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
v_x_945_ = v_tail_954_;
v_x_946_ = v___x_961_;
goto _start;
}
}
else
{
lean_object* v_a_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_971_; 
lean_del_object(v___x_956_);
lean_dec(v_tail_954_);
lean_dec(v_x_946_);
v_a_964_ = lean_ctor_get(v___x_958_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_958_);
if (v_isSharedCheck_971_ == 0)
{
v___x_966_ = v___x_958_;
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_a_964_);
lean_dec(v___x_958_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1___boxed(lean_object* v_x_974_, lean_object* v_x_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_){
_start:
{
lean_object* v_res_981_; 
v_res_981_ = lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1(v_x_974_, v_x_975_, v___y_976_, v___y_977_, v___y_978_, v___y_979_);
lean_dec(v___y_979_);
lean_dec_ref(v___y_978_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
return v_res_981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConst_x27(lean_object* v_constName_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_){
_start:
{
lean_object* v___x_988_; 
lean_inc(v_constName_982_);
v___x_988_ = lp_mathlib_Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0(v_constName_982_, v_a_983_, v_a_984_, v_a_985_, v_a_986_);
if (lean_obj_tag(v___x_988_) == 0)
{
lean_object* v_a_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; 
v_a_989_ = lean_ctor_get(v___x_988_, 0);
lean_inc(v_a_989_);
lean_dec_ref_known(v___x_988_, 1);
v___x_990_ = l_Lean_ConstantInfo_levelParams(v_a_989_);
lean_dec(v_a_989_);
v___x_991_ = lean_box(0);
v___x_992_ = lp_mathlib_List_mapM_loop___at___00Lean_mkConst_x27_spec__1(v___x_990_, v___x_991_, v_a_983_, v_a_984_, v_a_985_, v_a_986_);
if (lean_obj_tag(v___x_992_) == 0)
{
lean_object* v_a_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1001_; 
v_a_993_ = lean_ctor_get(v___x_992_, 0);
v_isSharedCheck_1001_ = !lean_is_exclusive(v___x_992_);
if (v_isSharedCheck_1001_ == 0)
{
v___x_995_ = v___x_992_;
v_isShared_996_ = v_isSharedCheck_1001_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_a_993_);
lean_dec(v___x_992_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1001_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v___x_997_; lean_object* v___x_999_; 
v___x_997_ = l_Lean_mkConst(v_constName_982_, v_a_993_);
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 0, v___x_997_);
v___x_999_ = v___x_995_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v___x_997_);
v___x_999_ = v_reuseFailAlloc_1000_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
return v___x_999_;
}
}
}
else
{
lean_object* v_a_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1009_; 
lean_dec(v_constName_982_);
v_a_1002_ = lean_ctor_get(v___x_992_, 0);
v_isSharedCheck_1009_ = !lean_is_exclusive(v___x_992_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_1004_ = v___x_992_;
v_isShared_1005_ = v_isSharedCheck_1009_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_a_1002_);
lean_dec(v___x_992_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1009_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
lean_object* v___x_1007_; 
if (v_isShared_1005_ == 0)
{
v___x_1007_ = v___x_1004_;
goto v_reusejp_1006_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v_a_1002_);
v___x_1007_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1006_;
}
v_reusejp_1006_:
{
return v___x_1007_;
}
}
}
}
else
{
lean_object* v_a_1010_; lean_object* v___x_1012_; uint8_t v_isShared_1013_; uint8_t v_isSharedCheck_1017_; 
lean_dec(v_constName_982_);
v_a_1010_ = lean_ctor_get(v___x_988_, 0);
v_isSharedCheck_1017_ = !lean_is_exclusive(v___x_988_);
if (v_isSharedCheck_1017_ == 0)
{
v___x_1012_ = v___x_988_;
v_isShared_1013_ = v_isSharedCheck_1017_;
goto v_resetjp_1011_;
}
else
{
lean_inc(v_a_1010_);
lean_dec(v___x_988_);
v___x_1012_ = lean_box(0);
v_isShared_1013_ = v_isSharedCheck_1017_;
goto v_resetjp_1011_;
}
v_resetjp_1011_:
{
lean_object* v___x_1015_; 
if (v_isShared_1013_ == 0)
{
v___x_1015_ = v___x_1012_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v_a_1010_);
v___x_1015_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
return v___x_1015_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConst_x27___boxed(lean_object* v_constName_1018_, lean_object* v_a_1019_, lean_object* v_a_1020_, lean_object* v_a_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_){
_start:
{
lean_object* v_res_1024_; 
v_res_1024_ = lp_mathlib_Lean_mkConst_x27(v_constName_1018_, v_a_1019_, v_a_1020_, v_a_1021_, v_a_1022_);
lean_dec(v_a_1022_);
lean_dec_ref(v_a_1021_);
lean_dec(v_a_1020_);
lean_dec_ref(v_a_1019_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0(lean_object* v_00_u03b1_1025_, lean_object* v_constName_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___redArg(v_constName_1026_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1033_, lean_object* v_constName_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_){
_start:
{
lean_object* v_res_1040_; 
v_res_1040_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0(v_00_u03b1_1033_, v_constName_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_);
lean_dec(v___y_1038_);
lean_dec_ref(v___y_1037_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
return v_res_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_1041_, lean_object* v_ref_1042_, lean_object* v_constName_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_){
_start:
{
lean_object* v___x_1049_; 
v___x_1049_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___redArg(v_ref_1042_, v_constName_1043_, v___y_1044_, v___y_1045_, v___y_1046_, v___y_1047_);
return v___x_1049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_1050_, lean_object* v_ref_1051_, lean_object* v_constName_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_){
_start:
{
lean_object* v_res_1058_; 
v_res_1058_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1(v_00_u03b1_1050_, v_ref_1051_, v_constName_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
lean_dec(v___y_1054_);
lean_dec_ref(v___y_1053_);
lean_dec(v_ref_1051_);
return v_res_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_1059_, lean_object* v_ref_1060_, lean_object* v_msg_1061_, lean_object* v_declHint_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_){
_start:
{
lean_object* v___x_1068_; 
v___x_1068_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_1060_, v_msg_1061_, v_declHint_1062_, v___y_1063_, v___y_1064_, v___y_1065_, v___y_1066_);
return v___x_1068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1069_, lean_object* v_ref_1070_, lean_object* v_msg_1071_, lean_object* v_declHint_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_){
_start:
{
lean_object* v_res_1078_; 
v_res_1078_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3(v_00_u03b1_1069_, v_ref_1070_, v_msg_1071_, v_declHint_1072_, v___y_1073_, v___y_1074_, v___y_1075_, v___y_1076_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
lean_dec(v___y_1074_);
lean_dec_ref(v___y_1073_);
lean_dec(v_ref_1070_);
return v_res_1078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(lean_object* v_msg_1079_, lean_object* v_declHint_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_){
_start:
{
lean_object* v___x_1086_; 
v___x_1086_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_1079_, v_declHint_1080_, v___y_1084_);
return v___x_1086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___boxed(lean_object* v_msg_1087_, lean_object* v_declHint_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
lean_object* v_res_1094_; 
v_res_1094_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(v_msg_1087_, v_declHint_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
lean_dec(v___y_1092_);
lean_dec_ref(v___y_1091_);
lean_dec(v___y_1090_);
lean_dec_ref(v___y_1089_);
return v_res_1094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5(lean_object* v_00_u03b1_1095_, lean_object* v_ref_1096_, lean_object* v_msg_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_){
_start:
{
lean_object* v___x_1103_; 
v___x_1103_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_1096_, v_msg_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03b1_1104_, lean_object* v_ref_1105_, lean_object* v_msg_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_){
_start:
{
lean_object* v_res_1112_; 
v_res_1112_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5(v_00_u03b1_1104_, v_ref_1105_, v_msg_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_);
lean_dec(v___y_1110_);
lean_dec_ref(v___y_1109_);
lean_dec(v___y_1108_);
lean_dec_ref(v___y_1107_);
lean_dec(v_ref_1105_);
return v_res_1112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(lean_object* v_00_u03b1_1113_, lean_object* v_msg_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_){
_start:
{
lean_object* v___x_1120_; 
v___x_1120_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_1114_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_);
return v___x_1120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___boxed(lean_object* v_00_u03b1_1121_, lean_object* v_msg_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_){
_start:
{
lean_object* v_res_1128_; 
v_res_1128_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(v_00_u03b1_1121_, v_msg_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
lean_dec(v___y_1124_);
lean_dec_ref(v___y_1123_);
return v_res_1128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_bvarIdx_x3f(lean_object* v_x_1129_){
_start:
{
if (lean_obj_tag(v_x_1129_) == 0)
{
lean_object* v_deBruijnIndex_1130_; lean_object* v___x_1131_; 
v_deBruijnIndex_1130_ = lean_ctor_get(v_x_1129_, 0);
lean_inc(v_deBruijnIndex_1130_);
v___x_1131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1131_, 0, v_deBruijnIndex_1130_);
return v___x_1131_;
}
else
{
lean_object* v___x_1132_; 
v___x_1132_ = lean_box(0);
return v___x_1132_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_bvarIdx_x3f___boxed(lean_object* v_x_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_mathlib_Lean_Expr_bvarIdx_x3f(v_x_1133_);
lean_dec_ref(v_x_1133_);
return v_res_1134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getAppAppsAux(lean_object* v_x_1135_, lean_object* v_x_1136_, lean_object* v_x_1137_){
_start:
{
if (lean_obj_tag(v_x_1135_) == 5)
{
lean_object* v_fn_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; 
v_fn_1138_ = lean_ctor_get(v_x_1135_, 0);
lean_inc_ref(v_fn_1138_);
v___x_1139_ = lean_array_set(v_x_1136_, v_x_1137_, v_x_1135_);
v___x_1140_ = lean_unsigned_to_nat(1u);
v___x_1141_ = lean_nat_sub(v_x_1137_, v___x_1140_);
lean_dec(v_x_1137_);
v_x_1135_ = v_fn_1138_;
v_x_1136_ = v___x_1139_;
v_x_1137_ = v___x_1141_;
goto _start;
}
else
{
lean_dec(v_x_1137_);
lean_dec_ref(v_x_1135_);
return v_x_1136_;
}
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_getAppApps___closed__0(void){
_start:
{
lean_object* v___x_1143_; lean_object* v_dummy_1144_; 
v___x_1143_ = lean_box(0);
v_dummy_1144_ = l_Lean_mkSort(v___x_1143_);
return v_dummy_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getAppApps(lean_object* v_e_1145_){
_start:
{
lean_object* v_dummy_1146_; lean_object* v_nargs_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; 
v_dummy_1146_ = lean_obj_once(&lp_mathlib_Lean_Expr_getAppApps___closed__0, &lp_mathlib_Lean_Expr_getAppApps___closed__0_once, _init_lp_mathlib_Lean_Expr_getAppApps___closed__0);
v_nargs_1147_ = l_Lean_Expr_getAppNumArgs(v_e_1145_);
lean_inc(v_nargs_1147_);
v___x_1148_ = lean_mk_array(v_nargs_1147_, v_dummy_1146_);
v___x_1149_ = lean_unsigned_to_nat(1u);
v___x_1150_ = lean_nat_sub(v_nargs_1147_, v___x_1149_);
lean_dec(v_nargs_1147_);
v___x_1151_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getAppAppsAux(v_e_1145_, v___x_1148_, v___x_1150_);
return v___x_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__0(lean_object* v_e_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v___x_1160_; 
lean_inc_ref(v_e_1154_);
v___x_1160_ = l_Lean_Meta_isProof(v_e_1154_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_);
if (lean_obj_tag(v___x_1160_) == 0)
{
lean_object* v_a_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1200_; 
v_a_1161_ = lean_ctor_get(v___x_1160_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1160_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1163_ = v___x_1160_;
v_isShared_1164_ = v_isSharedCheck_1200_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_a_1161_);
lean_dec(v___x_1160_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1200_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
uint8_t v___x_1165_; 
v___x_1165_ = lean_unbox(v_a_1161_);
if (v___x_1165_ == 0)
{
lean_object* v___x_1166_; lean_object* v___x_1168_; 
lean_dec(v_a_1161_);
lean_dec_ref(v_e_1154_);
v___x_1166_ = ((lean_object*)(lp_mathlib_Lean_Expr_eraseProofs___lam__0___closed__0));
if (v_isShared_1164_ == 0)
{
lean_ctor_set(v___x_1163_, 0, v___x_1166_);
v___x_1168_ = v___x_1163_;
goto v_reusejp_1167_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v___x_1166_);
v___x_1168_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1167_;
}
v_reusejp_1167_:
{
return v___x_1168_;
}
}
else
{
lean_object* v___x_1170_; 
lean_del_object(v___x_1163_);
lean_inc(v___y_1158_);
lean_inc_ref(v___y_1157_);
lean_inc(v___y_1156_);
lean_inc_ref(v___y_1155_);
v___x_1170_ = lean_infer_type(v_e_1154_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_);
if (lean_obj_tag(v___x_1170_) == 0)
{
lean_object* v_a_1171_; uint8_t v___x_1172_; lean_object* v___x_1173_; 
v_a_1171_ = lean_ctor_get(v___x_1170_, 0);
lean_inc(v_a_1171_);
lean_dec_ref_known(v___x_1170_, 1);
v___x_1172_ = lean_unbox(v_a_1161_);
lean_dec(v_a_1161_);
v___x_1173_ = l_Lean_Meta_mkSorry(v_a_1171_, v___x_1172_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_);
if (lean_obj_tag(v___x_1173_) == 0)
{
lean_object* v_a_1174_; lean_object* v___x_1176_; uint8_t v_isShared_1177_; uint8_t v_isSharedCheck_1183_; 
v_a_1174_ = lean_ctor_get(v___x_1173_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1173_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1176_ = v___x_1173_;
v_isShared_1177_ = v_isSharedCheck_1183_;
goto v_resetjp_1175_;
}
else
{
lean_inc(v_a_1174_);
lean_dec(v___x_1173_);
v___x_1176_ = lean_box(0);
v_isShared_1177_ = v_isSharedCheck_1183_;
goto v_resetjp_1175_;
}
v_resetjp_1175_:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1181_; 
v___x_1178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1178_, 0, v_a_1174_);
v___x_1179_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1179_, 0, v___x_1178_);
if (v_isShared_1177_ == 0)
{
lean_ctor_set(v___x_1176_, 0, v___x_1179_);
v___x_1181_ = v___x_1176_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v___x_1179_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
v_a_1184_ = lean_ctor_get(v___x_1173_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_1173_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_1173_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1173_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1189_; 
if (v_isShared_1187_ == 0)
{
v___x_1189_ = v___x_1186_;
goto v_reusejp_1188_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_a_1184_);
v___x_1189_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1188_;
}
v_reusejp_1188_:
{
return v___x_1189_;
}
}
}
}
else
{
lean_object* v_a_1192_; lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1199_; 
lean_dec(v_a_1161_);
v_a_1192_ = lean_ctor_get(v___x_1170_, 0);
v_isSharedCheck_1199_ = !lean_is_exclusive(v___x_1170_);
if (v_isSharedCheck_1199_ == 0)
{
v___x_1194_ = v___x_1170_;
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
else
{
lean_inc(v_a_1192_);
lean_dec(v___x_1170_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1197_; 
if (v_isShared_1195_ == 0)
{
v___x_1197_ = v___x_1194_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v_a_1192_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
}
}
}
else
{
lean_object* v_a_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1208_; 
lean_dec_ref(v_e_1154_);
v_a_1201_ = lean_ctor_get(v___x_1160_, 0);
v_isSharedCheck_1208_ = !lean_is_exclusive(v___x_1160_);
if (v_isSharedCheck_1208_ == 0)
{
v___x_1203_ = v___x_1160_;
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_a_1201_);
lean_dec(v___x_1160_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1206_; 
if (v_isShared_1204_ == 0)
{
v___x_1206_ = v___x_1203_;
goto v_reusejp_1205_;
}
else
{
lean_object* v_reuseFailAlloc_1207_; 
v_reuseFailAlloc_1207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1207_, 0, v_a_1201_);
v___x_1206_ = v_reuseFailAlloc_1207_;
goto v_reusejp_1205_;
}
v_reusejp_1205_:
{
return v___x_1206_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__0___boxed(lean_object* v_e_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v_res_1215_; 
v_res_1215_ = lp_mathlib_Lean_Expr_eraseProofs___lam__0(v_e_1209_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
return v_res_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__1(lean_object* v_e_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_){
_start:
{
lean_object* v___x_1222_; lean_object* v___x_1223_; 
v___x_1222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1222_, 0, v_e_1216_);
v___x_1223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1223_, 0, v___x_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___lam__1___boxed(lean_object* v_e_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_Lean_Expr_eraseProofs___lam__1(v_e_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0(lean_object* v_00_u03b1_1231_, lean_object* v_x_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; 
v___x_1238_ = lean_apply_1(v_x_1232_, lean_box(0));
v___x_1239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1239_, 0, v___x_1238_);
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0___boxed(lean_object* v_00_u03b1_1240_, lean_object* v_x_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_){
_start:
{
lean_object* v_res_1247_; 
v_res_1247_ = lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0(v_00_u03b1_1240_, v_x_1241_, v___y_1242_, v___y_1243_, v___y_1244_, v___y_1245_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1243_);
lean_dec_ref(v___y_1242_);
return v_res_1247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0(lean_object* v_00_u03b1_1248_, lean_object* v_x_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_){
_start:
{
lean_object* v___x_1255_; lean_object* v___x_1256_; 
v___x_1255_ = lean_apply_1(v_x_1249_, lean_box(0));
v___x_1256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1256_, 0, v___x_1255_);
return v___x_1256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0___boxed(lean_object* v_00_u03b1_1257_, lean_object* v_x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_){
_start:
{
lean_object* v_res_1264_; 
v_res_1264_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0(v_00_u03b1_1257_, v_x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_);
lean_dec(v___y_1262_);
lean_dec_ref(v___y_1261_);
lean_dec(v___y_1260_);
lean_dec_ref(v___y_1259_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0(lean_object* v_k_1265_, lean_object* v___y_1266_, lean_object* v_b_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_){
_start:
{
lean_object* v___x_1273_; 
lean_inc(v___y_1271_);
lean_inc_ref(v___y_1270_);
lean_inc(v___y_1269_);
lean_inc_ref(v___y_1268_);
lean_inc(v___y_1266_);
v___x_1273_ = lean_apply_7(v_k_1265_, v_b_1267_, v___y_1266_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_, lean_box(0));
return v___x_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0___boxed(lean_object* v_k_1274_, lean_object* v___y_1275_, lean_object* v_b_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_){
_start:
{
lean_object* v_res_1282_; 
v_res_1282_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0(v_k_1274_, v___y_1275_, v_b_1276_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_);
lean_dec(v___y_1280_);
lean_dec_ref(v___y_1279_);
lean_dec(v___y_1278_);
lean_dec_ref(v___y_1277_);
lean_dec(v___y_1275_);
return v_res_1282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(lean_object* v_name_1283_, uint8_t v_bi_1284_, lean_object* v_type_1285_, lean_object* v_k_1286_, uint8_t v_kind_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_){
_start:
{
lean_object* v___f_1294_; lean_object* v___x_1295_; 
lean_inc(v___y_1288_);
v___f_1294_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1294_, 0, v_k_1286_);
lean_closure_set(v___f_1294_, 1, v___y_1288_);
v___x_1295_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1283_, v_bi_1284_, v_type_1285_, v___f_1294_, v_kind_1287_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_);
if (lean_obj_tag(v___x_1295_) == 0)
{
return v___x_1295_;
}
else
{
lean_object* v_a_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1303_; 
v_a_1296_ = lean_ctor_get(v___x_1295_, 0);
v_isSharedCheck_1303_ = !lean_is_exclusive(v___x_1295_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1298_ = v___x_1295_;
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_a_1296_);
lean_dec(v___x_1295_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
lean_object* v___x_1301_; 
if (v_isShared_1299_ == 0)
{
v___x_1301_ = v___x_1298_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v_a_1296_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___boxed(lean_object* v_name_1304_, lean_object* v_bi_1305_, lean_object* v_type_1306_, lean_object* v_k_1307_, lean_object* v_kind_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_){
_start:
{
uint8_t v_bi_boxed_1315_; uint8_t v_kind_boxed_1316_; lean_object* v_res_1317_; 
v_bi_boxed_1315_ = lean_unbox(v_bi_1305_);
v_kind_boxed_1316_ = lean_unbox(v_kind_1308_);
v_res_1317_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(v_name_1304_, v_bi_boxed_1315_, v_type_1306_, v_k_1307_, v_kind_boxed_1316_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
lean_dec(v___y_1311_);
lean_dec_ref(v___y_1310_);
lean_dec(v___y_1309_);
return v_res_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2(lean_object* v___x_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v___x_1324_; 
v___x_1324_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1318_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2___boxed(lean_object* v___x_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_){
_start:
{
lean_object* v_res_1331_; 
v_res_1331_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2(v___x_1325_, v___y_1326_, v___y_1327_, v___y_1328_, v___y_1329_);
lean_dec(v___y_1329_);
lean_dec_ref(v___y_1328_);
lean_dec(v___y_1327_);
lean_dec_ref(v___y_1326_);
return v_res_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg(lean_object* v_name_1332_, lean_object* v_type_1333_, lean_object* v_val_1334_, lean_object* v_k_1335_, uint8_t v_nondep_1336_, uint8_t v_kind_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_){
_start:
{
lean_object* v___f_1344_; lean_object* v___x_1345_; 
lean_inc(v___y_1338_);
v___f_1344_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1344_, 0, v_k_1335_);
lean_closure_set(v___f_1344_, 1, v___y_1338_);
v___x_1345_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_1332_, v_type_1333_, v_val_1334_, v___f_1344_, v_nondep_1336_, v_kind_1337_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_);
if (lean_obj_tag(v___x_1345_) == 0)
{
return v___x_1345_;
}
else
{
lean_object* v_a_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1353_; 
v_a_1346_ = lean_ctor_get(v___x_1345_, 0);
v_isSharedCheck_1353_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1353_ == 0)
{
v___x_1348_ = v___x_1345_;
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_a_1346_);
lean_dec(v___x_1345_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v___x_1351_; 
if (v_isShared_1349_ == 0)
{
v___x_1351_ = v___x_1348_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1352_; 
v_reuseFailAlloc_1352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1352_, 0, v_a_1346_);
v___x_1351_ = v_reuseFailAlloc_1352_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
return v___x_1351_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg___boxed(lean_object* v_name_1354_, lean_object* v_type_1355_, lean_object* v_val_1356_, lean_object* v_k_1357_, lean_object* v_nondep_1358_, lean_object* v_kind_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_){
_start:
{
uint8_t v_nondep_boxed_1366_; uint8_t v_kind_boxed_1367_; lean_object* v_res_1368_; 
v_nondep_boxed_1366_ = lean_unbox(v_nondep_1358_);
v_kind_boxed_1367_ = lean_unbox(v_kind_1359_);
v_res_1368_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg(v_name_1354_, v_type_1355_, v_val_1356_, v_k_1357_, v_nondep_boxed_1366_, v_kind_boxed_1367_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_);
lean_dec(v___y_1364_);
lean_dec_ref(v___y_1363_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
lean_dec(v___y_1360_);
return v_res_1368_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3(void){
_start:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___x_1374_ = l_Lean_maxRecDepthErrorMessage;
v___x_1375_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1375_, 0, v___x_1374_);
return v___x_1375_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4(void){
_start:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; 
v___x_1376_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__3);
v___x_1377_ = l_Lean_MessageData_ofFormat(v___x_1376_);
return v___x_1377_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5(void){
_start:
{
lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; 
v___x_1378_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__4);
v___x_1379_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__2));
v___x_1380_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1380_, 0, v___x_1379_);
lean_ctor_set(v___x_1380_, 1, v___x_1378_);
return v___x_1380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg(lean_object* v_ref_1381_){
_start:
{
lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; 
v___x_1383_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___closed__5);
v___x_1384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1384_, 0, v_ref_1381_);
lean_ctor_set(v___x_1384_, 1, v___x_1383_);
v___x_1385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1385_, 0, v___x_1384_);
return v___x_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg___boxed(lean_object* v_ref_1386_, lean_object* v___y_1387_){
_start:
{
lean_object* v_res_1388_; 
v_res_1388_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg(v_ref_1386_);
return v_res_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg(lean_object* v_x_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
lean_object* v___y_1397_; lean_object* v_fileName_1406_; lean_object* v_fileMap_1407_; lean_object* v_options_1408_; lean_object* v_currRecDepth_1409_; lean_object* v_maxRecDepth_1410_; lean_object* v_ref_1411_; lean_object* v_currNamespace_1412_; lean_object* v_openDecls_1413_; lean_object* v_initHeartbeats_1414_; lean_object* v_maxHeartbeats_1415_; lean_object* v_quotContext_1416_; lean_object* v_currMacroScope_1417_; uint8_t v_diag_1418_; lean_object* v_cancelTk_x3f_1419_; uint8_t v_suppressElabErrors_1420_; lean_object* v_inheritedTraceOptions_1421_; lean_object* v___x_1427_; uint8_t v___x_1428_; 
v_fileName_1406_ = lean_ctor_get(v___y_1393_, 0);
v_fileMap_1407_ = lean_ctor_get(v___y_1393_, 1);
v_options_1408_ = lean_ctor_get(v___y_1393_, 2);
v_currRecDepth_1409_ = lean_ctor_get(v___y_1393_, 3);
v_maxRecDepth_1410_ = lean_ctor_get(v___y_1393_, 4);
v_ref_1411_ = lean_ctor_get(v___y_1393_, 5);
v_currNamespace_1412_ = lean_ctor_get(v___y_1393_, 6);
v_openDecls_1413_ = lean_ctor_get(v___y_1393_, 7);
v_initHeartbeats_1414_ = lean_ctor_get(v___y_1393_, 8);
v_maxHeartbeats_1415_ = lean_ctor_get(v___y_1393_, 9);
v_quotContext_1416_ = lean_ctor_get(v___y_1393_, 10);
v_currMacroScope_1417_ = lean_ctor_get(v___y_1393_, 11);
v_diag_1418_ = lean_ctor_get_uint8(v___y_1393_, sizeof(void*)*14);
v_cancelTk_x3f_1419_ = lean_ctor_get(v___y_1393_, 12);
v_suppressElabErrors_1420_ = lean_ctor_get_uint8(v___y_1393_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1421_ = lean_ctor_get(v___y_1393_, 13);
v___x_1427_ = lean_unsigned_to_nat(0u);
v___x_1428_ = lean_nat_dec_eq(v_maxRecDepth_1410_, v___x_1427_);
if (v___x_1428_ == 0)
{
uint8_t v___x_1429_; 
v___x_1429_ = lean_nat_dec_eq(v_currRecDepth_1409_, v_maxRecDepth_1410_);
if (v___x_1429_ == 0)
{
goto v___jp_1422_;
}
else
{
lean_object* v___x_1430_; 
lean_dec_ref(v_x_1389_);
lean_inc(v_ref_1411_);
v___x_1430_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg(v_ref_1411_);
v___y_1397_ = v___x_1430_;
goto v___jp_1396_;
}
}
else
{
goto v___jp_1422_;
}
v___jp_1396_:
{
if (lean_obj_tag(v___y_1397_) == 0)
{
return v___y_1397_;
}
else
{
lean_object* v_a_1398_; lean_object* v___x_1400_; uint8_t v_isShared_1401_; uint8_t v_isSharedCheck_1405_; 
v_a_1398_ = lean_ctor_get(v___y_1397_, 0);
v_isSharedCheck_1405_ = !lean_is_exclusive(v___y_1397_);
if (v_isSharedCheck_1405_ == 0)
{
v___x_1400_ = v___y_1397_;
v_isShared_1401_ = v_isSharedCheck_1405_;
goto v_resetjp_1399_;
}
else
{
lean_inc(v_a_1398_);
lean_dec(v___y_1397_);
v___x_1400_ = lean_box(0);
v_isShared_1401_ = v_isSharedCheck_1405_;
goto v_resetjp_1399_;
}
v_resetjp_1399_:
{
lean_object* v___x_1403_; 
if (v_isShared_1401_ == 0)
{
v___x_1403_ = v___x_1400_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v_a_1398_);
v___x_1403_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
return v___x_1403_;
}
}
}
}
v___jp_1422_:
{
lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; 
v___x_1423_ = lean_unsigned_to_nat(1u);
v___x_1424_ = lean_nat_add(v_currRecDepth_1409_, v___x_1423_);
lean_inc_ref(v_inheritedTraceOptions_1421_);
lean_inc(v_cancelTk_x3f_1419_);
lean_inc(v_currMacroScope_1417_);
lean_inc(v_quotContext_1416_);
lean_inc(v_maxHeartbeats_1415_);
lean_inc(v_initHeartbeats_1414_);
lean_inc(v_openDecls_1413_);
lean_inc(v_currNamespace_1412_);
lean_inc(v_ref_1411_);
lean_inc(v_maxRecDepth_1410_);
lean_inc_ref(v_options_1408_);
lean_inc_ref(v_fileMap_1407_);
lean_inc_ref(v_fileName_1406_);
v___x_1425_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1425_, 0, v_fileName_1406_);
lean_ctor_set(v___x_1425_, 1, v_fileMap_1407_);
lean_ctor_set(v___x_1425_, 2, v_options_1408_);
lean_ctor_set(v___x_1425_, 3, v___x_1424_);
lean_ctor_set(v___x_1425_, 4, v_maxRecDepth_1410_);
lean_ctor_set(v___x_1425_, 5, v_ref_1411_);
lean_ctor_set(v___x_1425_, 6, v_currNamespace_1412_);
lean_ctor_set(v___x_1425_, 7, v_openDecls_1413_);
lean_ctor_set(v___x_1425_, 8, v_initHeartbeats_1414_);
lean_ctor_set(v___x_1425_, 9, v_maxHeartbeats_1415_);
lean_ctor_set(v___x_1425_, 10, v_quotContext_1416_);
lean_ctor_set(v___x_1425_, 11, v_currMacroScope_1417_);
lean_ctor_set(v___x_1425_, 12, v_cancelTk_x3f_1419_);
lean_ctor_set(v___x_1425_, 13, v_inheritedTraceOptions_1421_);
lean_ctor_set_uint8(v___x_1425_, sizeof(void*)*14, v_diag_1418_);
lean_ctor_set_uint8(v___x_1425_, sizeof(void*)*14 + 1, v_suppressElabErrors_1420_);
lean_inc(v___y_1394_);
lean_inc(v___y_1392_);
lean_inc_ref(v___y_1391_);
lean_inc(v___y_1390_);
v___x_1426_ = lean_apply_6(v_x_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___x_1425_, v___y_1394_, lean_box(0));
v___y_1397_ = v___x_1426_;
goto v___jp_1396_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg___boxed(lean_object* v_x_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_){
_start:
{
lean_object* v_res_1438_; 
v_res_1438_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg(v_x_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec(v___y_1434_);
lean_dec_ref(v___y_1433_);
lean_dec(v___y_1432_);
return v_res_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17___redArg(lean_object* v_a_1439_, lean_object* v_b_1440_, lean_object* v_x_1441_){
_start:
{
if (lean_obj_tag(v_x_1441_) == 0)
{
lean_dec(v_b_1440_);
lean_dec_ref(v_a_1439_);
return v_x_1441_;
}
else
{
lean_object* v_key_1442_; lean_object* v_value_1443_; lean_object* v_tail_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1456_; 
v_key_1442_ = lean_ctor_get(v_x_1441_, 0);
v_value_1443_ = lean_ctor_get(v_x_1441_, 1);
v_tail_1444_ = lean_ctor_get(v_x_1441_, 2);
v_isSharedCheck_1456_ = !lean_is_exclusive(v_x_1441_);
if (v_isSharedCheck_1456_ == 0)
{
v___x_1446_ = v_x_1441_;
v_isShared_1447_ = v_isSharedCheck_1456_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_tail_1444_);
lean_inc(v_value_1443_);
lean_inc(v_key_1442_);
lean_dec(v_x_1441_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1456_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
uint8_t v___x_1448_; 
v___x_1448_ = l_Lean_ExprStructEq_beq(v_key_1442_, v_a_1439_);
if (v___x_1448_ == 0)
{
lean_object* v___x_1449_; lean_object* v___x_1451_; 
v___x_1449_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17___redArg(v_a_1439_, v_b_1440_, v_tail_1444_);
if (v_isShared_1447_ == 0)
{
lean_ctor_set(v___x_1446_, 2, v___x_1449_);
v___x_1451_ = v___x_1446_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_key_1442_);
lean_ctor_set(v_reuseFailAlloc_1452_, 1, v_value_1443_);
lean_ctor_set(v_reuseFailAlloc_1452_, 2, v___x_1449_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
else
{
lean_object* v___x_1454_; 
lean_dec(v_value_1443_);
lean_dec(v_key_1442_);
if (v_isShared_1447_ == 0)
{
lean_ctor_set(v___x_1446_, 1, v_b_1440_);
lean_ctor_set(v___x_1446_, 0, v_a_1439_);
v___x_1454_ = v___x_1446_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1455_; 
v_reuseFailAlloc_1455_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1455_, 0, v_a_1439_);
lean_ctor_set(v_reuseFailAlloc_1455_, 1, v_b_1440_);
lean_ctor_set(v_reuseFailAlloc_1455_, 2, v_tail_1444_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18___redArg(lean_object* v_x_1457_, lean_object* v_x_1458_){
_start:
{
if (lean_obj_tag(v_x_1458_) == 0)
{
return v_x_1457_;
}
else
{
lean_object* v_key_1459_; lean_object* v_value_1460_; lean_object* v_tail_1461_; lean_object* v___x_1463_; uint8_t v_isShared_1464_; uint8_t v_isSharedCheck_1484_; 
v_key_1459_ = lean_ctor_get(v_x_1458_, 0);
v_value_1460_ = lean_ctor_get(v_x_1458_, 1);
v_tail_1461_ = lean_ctor_get(v_x_1458_, 2);
v_isSharedCheck_1484_ = !lean_is_exclusive(v_x_1458_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1463_ = v_x_1458_;
v_isShared_1464_ = v_isSharedCheck_1484_;
goto v_resetjp_1462_;
}
else
{
lean_inc(v_tail_1461_);
lean_inc(v_value_1460_);
lean_inc(v_key_1459_);
lean_dec(v_x_1458_);
v___x_1463_ = lean_box(0);
v_isShared_1464_ = v_isSharedCheck_1484_;
goto v_resetjp_1462_;
}
v_resetjp_1462_:
{
lean_object* v___x_1465_; uint64_t v___x_1466_; uint64_t v___x_1467_; uint64_t v___x_1468_; uint64_t v_fold_1469_; uint64_t v___x_1470_; uint64_t v___x_1471_; uint64_t v___x_1472_; size_t v___x_1473_; size_t v___x_1474_; size_t v___x_1475_; size_t v___x_1476_; size_t v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1480_; 
v___x_1465_ = lean_array_get_size(v_x_1457_);
v___x_1466_ = l_Lean_ExprStructEq_hash(v_key_1459_);
v___x_1467_ = 32ULL;
v___x_1468_ = lean_uint64_shift_right(v___x_1466_, v___x_1467_);
v_fold_1469_ = lean_uint64_xor(v___x_1466_, v___x_1468_);
v___x_1470_ = 16ULL;
v___x_1471_ = lean_uint64_shift_right(v_fold_1469_, v___x_1470_);
v___x_1472_ = lean_uint64_xor(v_fold_1469_, v___x_1471_);
v___x_1473_ = lean_uint64_to_usize(v___x_1472_);
v___x_1474_ = lean_usize_of_nat(v___x_1465_);
v___x_1475_ = ((size_t)1ULL);
v___x_1476_ = lean_usize_sub(v___x_1474_, v___x_1475_);
v___x_1477_ = lean_usize_land(v___x_1473_, v___x_1476_);
v___x_1478_ = lean_array_uget_borrowed(v_x_1457_, v___x_1477_);
lean_inc(v___x_1478_);
if (v_isShared_1464_ == 0)
{
lean_ctor_set(v___x_1463_, 2, v___x_1478_);
v___x_1480_ = v___x_1463_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v_key_1459_);
lean_ctor_set(v_reuseFailAlloc_1483_, 1, v_value_1460_);
lean_ctor_set(v_reuseFailAlloc_1483_, 2, v___x_1478_);
v___x_1480_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
lean_object* v___x_1481_; 
v___x_1481_ = lean_array_uset(v_x_1457_, v___x_1477_, v___x_1480_);
v_x_1457_ = v___x_1481_;
v_x_1458_ = v_tail_1461_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17___redArg(lean_object* v_i_1485_, lean_object* v_source_1486_, lean_object* v_target_1487_){
_start:
{
lean_object* v___x_1488_; uint8_t v___x_1489_; 
v___x_1488_ = lean_array_get_size(v_source_1486_);
v___x_1489_ = lean_nat_dec_lt(v_i_1485_, v___x_1488_);
if (v___x_1489_ == 0)
{
lean_dec_ref(v_source_1486_);
lean_dec(v_i_1485_);
return v_target_1487_;
}
else
{
lean_object* v_es_1490_; lean_object* v___x_1491_; lean_object* v_source_1492_; lean_object* v_target_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v_es_1490_ = lean_array_fget(v_source_1486_, v_i_1485_);
v___x_1491_ = lean_box(0);
v_source_1492_ = lean_array_fset(v_source_1486_, v_i_1485_, v___x_1491_);
v_target_1493_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18___redArg(v_target_1487_, v_es_1490_);
v___x_1494_ = lean_unsigned_to_nat(1u);
v___x_1495_ = lean_nat_add(v_i_1485_, v___x_1494_);
lean_dec(v_i_1485_);
v_i_1485_ = v___x_1495_;
v_source_1486_ = v_source_1492_;
v_target_1487_ = v_target_1493_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16___redArg(lean_object* v_data_1497_){
_start:
{
lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v_nbuckets_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; 
v___x_1498_ = lean_array_get_size(v_data_1497_);
v___x_1499_ = lean_unsigned_to_nat(2u);
v_nbuckets_1500_ = lean_nat_mul(v___x_1498_, v___x_1499_);
v___x_1501_ = lean_unsigned_to_nat(0u);
v___x_1502_ = lean_box(0);
v___x_1503_ = lean_mk_array(v_nbuckets_1500_, v___x_1502_);
v___x_1504_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17___redArg(v___x_1501_, v_data_1497_, v___x_1503_);
return v___x_1504_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg(lean_object* v_a_1505_, lean_object* v_x_1506_){
_start:
{
if (lean_obj_tag(v_x_1506_) == 0)
{
uint8_t v___x_1507_; 
v___x_1507_ = 0;
return v___x_1507_;
}
else
{
lean_object* v_key_1508_; lean_object* v_tail_1509_; uint8_t v___x_1510_; 
v_key_1508_ = lean_ctor_get(v_x_1506_, 0);
v_tail_1509_ = lean_ctor_get(v_x_1506_, 2);
v___x_1510_ = l_Lean_ExprStructEq_beq(v_key_1508_, v_a_1505_);
if (v___x_1510_ == 0)
{
v_x_1506_ = v_tail_1509_;
goto _start;
}
else
{
return v___x_1510_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg___boxed(lean_object* v_a_1512_, lean_object* v_x_1513_){
_start:
{
uint8_t v_res_1514_; lean_object* v_r_1515_; 
v_res_1514_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg(v_a_1512_, v_x_1513_);
lean_dec(v_x_1513_);
lean_dec_ref(v_a_1512_);
v_r_1515_ = lean_box(v_res_1514_);
return v_r_1515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10___redArg(lean_object* v_m_1516_, lean_object* v_a_1517_, lean_object* v_b_1518_){
_start:
{
lean_object* v_size_1519_; lean_object* v_buckets_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1563_; 
v_size_1519_ = lean_ctor_get(v_m_1516_, 0);
v_buckets_1520_ = lean_ctor_get(v_m_1516_, 1);
v_isSharedCheck_1563_ = !lean_is_exclusive(v_m_1516_);
if (v_isSharedCheck_1563_ == 0)
{
v___x_1522_ = v_m_1516_;
v_isShared_1523_ = v_isSharedCheck_1563_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_buckets_1520_);
lean_inc(v_size_1519_);
lean_dec(v_m_1516_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1563_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1524_; uint64_t v___x_1525_; uint64_t v___x_1526_; uint64_t v___x_1527_; uint64_t v_fold_1528_; uint64_t v___x_1529_; uint64_t v___x_1530_; uint64_t v___x_1531_; size_t v___x_1532_; size_t v___x_1533_; size_t v___x_1534_; size_t v___x_1535_; size_t v___x_1536_; lean_object* v_bkt_1537_; uint8_t v___x_1538_; 
v___x_1524_ = lean_array_get_size(v_buckets_1520_);
v___x_1525_ = l_Lean_ExprStructEq_hash(v_a_1517_);
v___x_1526_ = 32ULL;
v___x_1527_ = lean_uint64_shift_right(v___x_1525_, v___x_1526_);
v_fold_1528_ = lean_uint64_xor(v___x_1525_, v___x_1527_);
v___x_1529_ = 16ULL;
v___x_1530_ = lean_uint64_shift_right(v_fold_1528_, v___x_1529_);
v___x_1531_ = lean_uint64_xor(v_fold_1528_, v___x_1530_);
v___x_1532_ = lean_uint64_to_usize(v___x_1531_);
v___x_1533_ = lean_usize_of_nat(v___x_1524_);
v___x_1534_ = ((size_t)1ULL);
v___x_1535_ = lean_usize_sub(v___x_1533_, v___x_1534_);
v___x_1536_ = lean_usize_land(v___x_1532_, v___x_1535_);
v_bkt_1537_ = lean_array_uget_borrowed(v_buckets_1520_, v___x_1536_);
v___x_1538_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg(v_a_1517_, v_bkt_1537_);
if (v___x_1538_ == 0)
{
lean_object* v___x_1539_; lean_object* v_size_x27_1540_; lean_object* v___x_1541_; lean_object* v_buckets_x27_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; uint8_t v___x_1548_; 
v___x_1539_ = lean_unsigned_to_nat(1u);
v_size_x27_1540_ = lean_nat_add(v_size_1519_, v___x_1539_);
lean_dec(v_size_1519_);
lean_inc(v_bkt_1537_);
v___x_1541_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1541_, 0, v_a_1517_);
lean_ctor_set(v___x_1541_, 1, v_b_1518_);
lean_ctor_set(v___x_1541_, 2, v_bkt_1537_);
v_buckets_x27_1542_ = lean_array_uset(v_buckets_1520_, v___x_1536_, v___x_1541_);
v___x_1543_ = lean_unsigned_to_nat(4u);
v___x_1544_ = lean_nat_mul(v_size_x27_1540_, v___x_1543_);
v___x_1545_ = lean_unsigned_to_nat(3u);
v___x_1546_ = lean_nat_div(v___x_1544_, v___x_1545_);
lean_dec(v___x_1544_);
v___x_1547_ = lean_array_get_size(v_buckets_x27_1542_);
v___x_1548_ = lean_nat_dec_le(v___x_1546_, v___x_1547_);
lean_dec(v___x_1546_);
if (v___x_1548_ == 0)
{
lean_object* v_val_1549_; lean_object* v___x_1551_; 
v_val_1549_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16___redArg(v_buckets_x27_1542_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v_val_1549_);
lean_ctor_set(v___x_1522_, 0, v_size_x27_1540_);
v___x_1551_ = v___x_1522_;
goto v_reusejp_1550_;
}
else
{
lean_object* v_reuseFailAlloc_1552_; 
v_reuseFailAlloc_1552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1552_, 0, v_size_x27_1540_);
lean_ctor_set(v_reuseFailAlloc_1552_, 1, v_val_1549_);
v___x_1551_ = v_reuseFailAlloc_1552_;
goto v_reusejp_1550_;
}
v_reusejp_1550_:
{
return v___x_1551_;
}
}
else
{
lean_object* v___x_1554_; 
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v_buckets_x27_1542_);
lean_ctor_set(v___x_1522_, 0, v_size_x27_1540_);
v___x_1554_ = v___x_1522_;
goto v_reusejp_1553_;
}
else
{
lean_object* v_reuseFailAlloc_1555_; 
v_reuseFailAlloc_1555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1555_, 0, v_size_x27_1540_);
lean_ctor_set(v_reuseFailAlloc_1555_, 1, v_buckets_x27_1542_);
v___x_1554_ = v_reuseFailAlloc_1555_;
goto v_reusejp_1553_;
}
v_reusejp_1553_:
{
return v___x_1554_;
}
}
}
else
{
lean_object* v___x_1556_; lean_object* v_buckets_x27_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1561_; 
lean_inc(v_bkt_1537_);
v___x_1556_ = lean_box(0);
v_buckets_x27_1557_ = lean_array_uset(v_buckets_1520_, v___x_1536_, v___x_1556_);
v___x_1558_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17___redArg(v_a_1517_, v_b_1518_, v_bkt_1537_);
v___x_1559_ = lean_array_uset(v_buckets_x27_1557_, v___x_1536_, v___x_1558_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v___x_1559_);
v___x_1561_ = v___x_1522_;
goto v_reusejp_1560_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_size_1519_);
lean_ctor_set(v_reuseFailAlloc_1562_, 1, v___x_1559_);
v___x_1561_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1560_;
}
v_reusejp_1560_:
{
return v___x_1561_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2(lean_object* v_a_1564_, lean_object* v_e_1565_, lean_object* v_a_1566_){
_start:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___x_1568_ = lean_st_ref_take(v_a_1564_);
v___x_1569_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10___redArg(v___x_1568_, v_e_1565_, v_a_1566_);
v___x_1570_ = lean_st_ref_set(v_a_1564_, v___x_1569_);
v___x_1571_ = lean_box(0);
return v___x_1571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2___boxed(lean_object* v_a_1572_, lean_object* v_e_1573_, lean_object* v_a_1574_, lean_object* v___y_1575_){
_start:
{
lean_object* v_res_1576_; 
v_res_1576_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2(v_a_1572_, v_e_1573_, v_a_1574_);
lean_dec(v_a_1572_);
return v_res_1576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg(lean_object* v_a_1577_, lean_object* v_x_1578_){
_start:
{
if (lean_obj_tag(v_x_1578_) == 0)
{
lean_object* v___x_1579_; 
v___x_1579_ = lean_box(0);
return v___x_1579_;
}
else
{
lean_object* v_key_1580_; lean_object* v_value_1581_; lean_object* v_tail_1582_; uint8_t v___x_1583_; 
v_key_1580_ = lean_ctor_get(v_x_1578_, 0);
v_value_1581_ = lean_ctor_get(v_x_1578_, 1);
v_tail_1582_ = lean_ctor_get(v_x_1578_, 2);
v___x_1583_ = l_Lean_ExprStructEq_beq(v_key_1580_, v_a_1577_);
if (v___x_1583_ == 0)
{
v_x_1578_ = v_tail_1582_;
goto _start;
}
else
{
lean_object* v___x_1585_; 
lean_inc(v_value_1581_);
v___x_1585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1585_, 0, v_value_1581_);
return v___x_1585_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object* v_a_1586_, lean_object* v_x_1587_){
_start:
{
lean_object* v_res_1588_; 
v_res_1588_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg(v_a_1586_, v_x_1587_);
lean_dec(v_x_1587_);
lean_dec_ref(v_a_1586_);
return v_res_1588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg(lean_object* v_m_1589_, lean_object* v_a_1590_){
_start:
{
lean_object* v_buckets_1591_; lean_object* v___x_1592_; uint64_t v___x_1593_; uint64_t v___x_1594_; uint64_t v___x_1595_; uint64_t v_fold_1596_; uint64_t v___x_1597_; uint64_t v___x_1598_; uint64_t v___x_1599_; size_t v___x_1600_; size_t v___x_1601_; size_t v___x_1602_; size_t v___x_1603_; size_t v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; 
v_buckets_1591_ = lean_ctor_get(v_m_1589_, 1);
v___x_1592_ = lean_array_get_size(v_buckets_1591_);
v___x_1593_ = l_Lean_ExprStructEq_hash(v_a_1590_);
v___x_1594_ = 32ULL;
v___x_1595_ = lean_uint64_shift_right(v___x_1593_, v___x_1594_);
v_fold_1596_ = lean_uint64_xor(v___x_1593_, v___x_1595_);
v___x_1597_ = 16ULL;
v___x_1598_ = lean_uint64_shift_right(v_fold_1596_, v___x_1597_);
v___x_1599_ = lean_uint64_xor(v_fold_1596_, v___x_1598_);
v___x_1600_ = lean_uint64_to_usize(v___x_1599_);
v___x_1601_ = lean_usize_of_nat(v___x_1592_);
v___x_1602_ = ((size_t)1ULL);
v___x_1603_ = lean_usize_sub(v___x_1601_, v___x_1602_);
v___x_1604_ = lean_usize_land(v___x_1600_, v___x_1603_);
v___x_1605_ = lean_array_uget_borrowed(v_buckets_1591_, v___x_1604_);
v___x_1606_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg(v_a_1590_, v___x_1605_);
return v___x_1606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_m_1607_, lean_object* v_a_1608_){
_start:
{
lean_object* v_res_1609_; 
v_res_1609_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg(v_m_1607_, v_a_1608_);
lean_dec_ref(v_a_1608_);
lean_dec_ref(v_m_1607_);
return v_res_1609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0(lean_object* v_fvars_1613_, lean_object* v_pre_1614_, lean_object* v_post_1615_, uint8_t v_usedLetOnly_1616_, uint8_t v_skipConstInApp_1617_, uint8_t v_skipInstances_1618_, lean_object* v_body_1619_, lean_object* v_x_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v___x_1627_; lean_object* v___x_1628_; 
v___x_1627_ = lean_array_push(v_fvars_1613_, v_x_1620_);
v___x_1628_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6(v_pre_1614_, v_post_1615_, v_usedLetOnly_1616_, v_skipConstInApp_1617_, v_skipInstances_1618_, v___x_1627_, v_body_1619_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
return v___x_1628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0___boxed(lean_object* v_fvars_1629_, lean_object* v_pre_1630_, lean_object* v_post_1631_, lean_object* v_usedLetOnly_1632_, lean_object* v_skipConstInApp_1633_, lean_object* v_skipInstances_1634_, lean_object* v_body_1635_, lean_object* v_x_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_){
_start:
{
uint8_t v_usedLetOnly_boxed_1643_; uint8_t v_skipConstInApp_boxed_1644_; uint8_t v_skipInstances_boxed_1645_; lean_object* v_res_1646_; 
v_usedLetOnly_boxed_1643_ = lean_unbox(v_usedLetOnly_1632_);
v_skipConstInApp_boxed_1644_ = lean_unbox(v_skipConstInApp_1633_);
v_skipInstances_boxed_1645_ = lean_unbox(v_skipInstances_1634_);
v_res_1646_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0(v_fvars_1629_, v_pre_1630_, v_post_1631_, v_usedLetOnly_boxed_1643_, v_skipConstInApp_boxed_1644_, v_skipInstances_boxed_1645_, v_body_1635_, v_x_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
lean_dec(v___y_1641_);
lean_dec_ref(v___y_1640_);
lean_dec(v___y_1639_);
lean_dec_ref(v___y_1638_);
lean_dec(v___y_1637_);
return v_res_1646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(lean_object* v_pre_1647_, lean_object* v_post_1648_, uint8_t v_usedLetOnly_1649_, uint8_t v_skipConstInApp_1650_, uint8_t v_skipInstances_1651_, lean_object* v_e_1652_, lean_object* v_a_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_){
_start:
{
lean_object* v___x_1659_; 
lean_inc_ref(v_post_1648_);
lean_inc(v___y_1657_);
lean_inc_ref(v___y_1656_);
lean_inc(v___y_1655_);
lean_inc_ref(v___y_1654_);
lean_inc_ref(v_e_1652_);
v___x_1659_ = lean_apply_6(v_post_1648_, v_e_1652_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_, lean_box(0));
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_object* v_a_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1678_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1662_ = v___x_1659_;
v_isShared_1663_ = v_isSharedCheck_1678_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_a_1660_);
lean_dec(v___x_1659_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1678_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
switch(lean_obj_tag(v_a_1660_))
{
case 0:
{
lean_object* v_e_1664_; lean_object* v___x_1666_; 
lean_dec_ref(v_e_1652_);
lean_dec_ref(v_post_1648_);
lean_dec_ref(v_pre_1647_);
v_e_1664_ = lean_ctor_get(v_a_1660_, 0);
lean_inc_ref(v_e_1664_);
lean_dec_ref_known(v_a_1660_, 1);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v_e_1664_);
v___x_1666_ = v___x_1662_;
goto v_reusejp_1665_;
}
else
{
lean_object* v_reuseFailAlloc_1667_; 
v_reuseFailAlloc_1667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1667_, 0, v_e_1664_);
v___x_1666_ = v_reuseFailAlloc_1667_;
goto v_reusejp_1665_;
}
v_reusejp_1665_:
{
return v___x_1666_;
}
}
case 1:
{
lean_object* v_e_1668_; lean_object* v___x_1669_; 
lean_del_object(v___x_1662_);
lean_dec_ref(v_e_1652_);
v_e_1668_ = lean_ctor_get(v_a_1660_, 0);
lean_inc_ref(v_e_1668_);
lean_dec_ref_known(v_a_1660_, 1);
v___x_1669_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1647_, v_post_1648_, v_usedLetOnly_1649_, v_skipConstInApp_1650_, v_skipInstances_1651_, v_e_1668_, v_a_1653_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_);
return v___x_1669_;
}
default: 
{
lean_object* v_e_x3f_1670_; 
lean_dec_ref(v_post_1648_);
lean_dec_ref(v_pre_1647_);
v_e_x3f_1670_ = lean_ctor_get(v_a_1660_, 0);
lean_inc(v_e_x3f_1670_);
lean_dec_ref_known(v_a_1660_, 1);
if (lean_obj_tag(v_e_x3f_1670_) == 0)
{
lean_object* v___x_1672_; 
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v_e_1652_);
v___x_1672_ = v___x_1662_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_e_1652_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
else
{
lean_object* v_val_1674_; lean_object* v___x_1676_; 
lean_dec_ref(v_e_1652_);
v_val_1674_ = lean_ctor_get(v_e_x3f_1670_, 0);
lean_inc(v_val_1674_);
lean_dec_ref_known(v_e_x3f_1670_, 1);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v_val_1674_);
v___x_1676_ = v___x_1662_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v_val_1674_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
}
}
}
else
{
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1686_; 
lean_dec_ref(v_e_1652_);
lean_dec_ref(v_post_1648_);
lean_dec_ref(v_pre_1647_);
v_a_1679_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1686_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1681_ = v___x_1659_;
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1659_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
lean_object* v___x_1684_; 
if (v_isShared_1682_ == 0)
{
v___x_1684_ = v___x_1681_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_a_1679_);
v___x_1684_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
return v___x_1684_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6(lean_object* v_pre_1687_, lean_object* v_post_1688_, uint8_t v_usedLetOnly_1689_, uint8_t v_skipConstInApp_1690_, uint8_t v_skipInstances_1691_, lean_object* v_fvars_1692_, lean_object* v_e_1693_, lean_object* v_a_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_){
_start:
{
if (lean_obj_tag(v_e_1693_) == 6)
{
lean_object* v_binderName_1700_; lean_object* v_binderType_1701_; lean_object* v_body_1702_; uint8_t v_binderInfo_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
v_binderName_1700_ = lean_ctor_get(v_e_1693_, 0);
lean_inc(v_binderName_1700_);
v_binderType_1701_ = lean_ctor_get(v_e_1693_, 1);
lean_inc_ref(v_binderType_1701_);
v_body_1702_ = lean_ctor_get(v_e_1693_, 2);
lean_inc_ref(v_body_1702_);
v_binderInfo_1703_ = lean_ctor_get_uint8(v_e_1693_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_1693_, 3);
v___x_1704_ = lean_expr_instantiate_rev(v_binderType_1701_, v_fvars_1692_);
lean_dec_ref(v_binderType_1701_);
lean_inc_ref(v_post_1688_);
lean_inc_ref(v_pre_1687_);
v___x_1705_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1687_, v_post_1688_, v_usedLetOnly_1689_, v_skipConstInApp_1690_, v_skipInstances_1691_, v___x_1704_, v_a_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v_a_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___f_1710_; uint8_t v___x_1711_; lean_object* v___x_1712_; 
v_a_1706_ = lean_ctor_get(v___x_1705_, 0);
lean_inc(v_a_1706_);
lean_dec_ref_known(v___x_1705_, 1);
v___x_1707_ = lean_box(v_usedLetOnly_1689_);
v___x_1708_ = lean_box(v_skipConstInApp_1690_);
v___x_1709_ = lean_box(v_skipInstances_1691_);
v___f_1710_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___lam__0___boxed), 14, 7);
lean_closure_set(v___f_1710_, 0, v_fvars_1692_);
lean_closure_set(v___f_1710_, 1, v_pre_1687_);
lean_closure_set(v___f_1710_, 2, v_post_1688_);
lean_closure_set(v___f_1710_, 3, v___x_1707_);
lean_closure_set(v___f_1710_, 4, v___x_1708_);
lean_closure_set(v___f_1710_, 5, v___x_1709_);
lean_closure_set(v___f_1710_, 6, v_body_1702_);
v___x_1711_ = 0;
v___x_1712_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(v_binderName_1700_, v_binderInfo_1703_, v_a_1706_, v___f_1710_, v___x_1711_, v_a_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
return v___x_1712_;
}
else
{
lean_dec_ref(v_body_1702_);
lean_dec(v_binderName_1700_);
lean_dec_ref(v_fvars_1692_);
lean_dec_ref(v_post_1688_);
lean_dec_ref(v_pre_1687_);
return v___x_1705_;
}
}
else
{
lean_object* v___x_1713_; lean_object* v___x_1714_; 
v___x_1713_ = lean_expr_instantiate_rev(v_e_1693_, v_fvars_1692_);
lean_dec_ref(v_e_1693_);
lean_inc_ref(v_post_1688_);
lean_inc_ref(v_pre_1687_);
v___x_1714_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1687_, v_post_1688_, v_usedLetOnly_1689_, v_skipConstInApp_1690_, v_skipInstances_1691_, v___x_1713_, v_a_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
if (lean_obj_tag(v___x_1714_) == 0)
{
lean_object* v_a_1715_; uint8_t v___x_1716_; uint8_t v___x_1717_; uint8_t v___x_1718_; lean_object* v___x_1719_; 
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___x_1714_, 1);
v___x_1716_ = 0;
v___x_1717_ = 1;
v___x_1718_ = 1;
v___x_1719_ = l_Lean_Meta_mkLambdaFVars(v_fvars_1692_, v_a_1715_, v___x_1716_, v_usedLetOnly_1689_, v___x_1716_, v___x_1717_, v___x_1718_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
lean_dec_ref(v_fvars_1692_);
if (lean_obj_tag(v___x_1719_) == 0)
{
lean_object* v_a_1720_; lean_object* v___x_1721_; 
v_a_1720_ = lean_ctor_get(v___x_1719_, 0);
lean_inc(v_a_1720_);
lean_dec_ref_known(v___x_1719_, 1);
v___x_1721_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_1687_, v_post_1688_, v_usedLetOnly_1689_, v_skipConstInApp_1690_, v_skipInstances_1691_, v_a_1720_, v_a_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
return v___x_1721_;
}
else
{
lean_dec_ref(v_post_1688_);
lean_dec_ref(v_pre_1687_);
return v___x_1719_;
}
}
else
{
lean_dec_ref(v_fvars_1692_);
lean_dec_ref(v_post_1688_);
lean_dec_ref(v_pre_1687_);
return v___x_1714_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0(lean_object* v_fvars_1722_, lean_object* v_pre_1723_, lean_object* v_post_1724_, uint8_t v_usedLetOnly_1725_, uint8_t v_skipConstInApp_1726_, uint8_t v_skipInstances_1727_, lean_object* v_body_1728_, lean_object* v_x_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_){
_start:
{
lean_object* v___x_1736_; lean_object* v___x_1737_; 
v___x_1736_ = lean_array_push(v_fvars_1722_, v_x_1729_);
v___x_1737_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7(v_pre_1723_, v_post_1724_, v_usedLetOnly_1725_, v_skipConstInApp_1726_, v_skipInstances_1727_, v___x_1736_, v_body_1728_, v___y_1730_, v___y_1731_, v___y_1732_, v___y_1733_, v___y_1734_);
return v___x_1737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0___boxed(lean_object* v_fvars_1738_, lean_object* v_pre_1739_, lean_object* v_post_1740_, lean_object* v_usedLetOnly_1741_, lean_object* v_skipConstInApp_1742_, lean_object* v_skipInstances_1743_, lean_object* v_body_1744_, lean_object* v_x_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_){
_start:
{
uint8_t v_usedLetOnly_boxed_1752_; uint8_t v_skipConstInApp_boxed_1753_; uint8_t v_skipInstances_boxed_1754_; lean_object* v_res_1755_; 
v_usedLetOnly_boxed_1752_ = lean_unbox(v_usedLetOnly_1741_);
v_skipConstInApp_boxed_1753_ = lean_unbox(v_skipConstInApp_1742_);
v_skipInstances_boxed_1754_ = lean_unbox(v_skipInstances_1743_);
v_res_1755_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0(v_fvars_1738_, v_pre_1739_, v_post_1740_, v_usedLetOnly_boxed_1752_, v_skipConstInApp_boxed_1753_, v_skipInstances_boxed_1754_, v_body_1744_, v_x_1745_, v___y_1746_, v___y_1747_, v___y_1748_, v___y_1749_, v___y_1750_);
lean_dec(v___y_1750_);
lean_dec_ref(v___y_1749_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
lean_dec(v___y_1746_);
return v_res_1755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7(lean_object* v_pre_1756_, lean_object* v_post_1757_, uint8_t v_usedLetOnly_1758_, uint8_t v_skipConstInApp_1759_, uint8_t v_skipInstances_1760_, lean_object* v_fvars_1761_, lean_object* v_e_1762_, lean_object* v_a_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_){
_start:
{
if (lean_obj_tag(v_e_1762_) == 8)
{
lean_object* v_declName_1769_; lean_object* v_type_1770_; lean_object* v_value_1771_; lean_object* v_body_1772_; uint8_t v_nondep_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; 
v_declName_1769_ = lean_ctor_get(v_e_1762_, 0);
lean_inc(v_declName_1769_);
v_type_1770_ = lean_ctor_get(v_e_1762_, 1);
lean_inc_ref(v_type_1770_);
v_value_1771_ = lean_ctor_get(v_e_1762_, 2);
lean_inc_ref(v_value_1771_);
v_body_1772_ = lean_ctor_get(v_e_1762_, 3);
lean_inc_ref(v_body_1772_);
v_nondep_1773_ = lean_ctor_get_uint8(v_e_1762_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_1762_, 4);
v___x_1774_ = lean_expr_instantiate_rev(v_type_1770_, v_fvars_1761_);
lean_dec_ref(v_type_1770_);
lean_inc_ref(v_post_1757_);
lean_inc_ref(v_pre_1756_);
v___x_1775_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1756_, v_post_1757_, v_usedLetOnly_1758_, v_skipConstInApp_1759_, v_skipInstances_1760_, v___x_1774_, v_a_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
if (lean_obj_tag(v___x_1775_) == 0)
{
lean_object* v_a_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; 
v_a_1776_ = lean_ctor_get(v___x_1775_, 0);
lean_inc(v_a_1776_);
lean_dec_ref_known(v___x_1775_, 1);
v___x_1777_ = lean_expr_instantiate_rev(v_value_1771_, v_fvars_1761_);
lean_dec_ref(v_value_1771_);
lean_inc_ref(v_post_1757_);
lean_inc_ref(v_pre_1756_);
v___x_1778_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1756_, v_post_1757_, v_usedLetOnly_1758_, v_skipConstInApp_1759_, v_skipInstances_1760_, v___x_1777_, v_a_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
if (lean_obj_tag(v___x_1778_) == 0)
{
lean_object* v_a_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___f_1783_; uint8_t v___x_1784_; lean_object* v___x_1785_; 
v_a_1779_ = lean_ctor_get(v___x_1778_, 0);
lean_inc(v_a_1779_);
lean_dec_ref_known(v___x_1778_, 1);
v___x_1780_ = lean_box(v_usedLetOnly_1758_);
v___x_1781_ = lean_box(v_skipConstInApp_1759_);
v___x_1782_ = lean_box(v_skipInstances_1760_);
v___f_1783_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___lam__0___boxed), 14, 7);
lean_closure_set(v___f_1783_, 0, v_fvars_1761_);
lean_closure_set(v___f_1783_, 1, v_pre_1756_);
lean_closure_set(v___f_1783_, 2, v_post_1757_);
lean_closure_set(v___f_1783_, 3, v___x_1780_);
lean_closure_set(v___f_1783_, 4, v___x_1781_);
lean_closure_set(v___f_1783_, 5, v___x_1782_);
lean_closure_set(v___f_1783_, 6, v_body_1772_);
v___x_1784_ = 0;
v___x_1785_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg(v_declName_1769_, v_a_1776_, v_a_1779_, v___f_1783_, v_nondep_1773_, v___x_1784_, v_a_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
return v___x_1785_;
}
else
{
lean_dec(v_a_1776_);
lean_dec_ref(v_body_1772_);
lean_dec(v_declName_1769_);
lean_dec_ref(v_fvars_1761_);
lean_dec_ref(v_post_1757_);
lean_dec_ref(v_pre_1756_);
return v___x_1778_;
}
}
else
{
lean_dec_ref(v_body_1772_);
lean_dec_ref(v_value_1771_);
lean_dec(v_declName_1769_);
lean_dec_ref(v_fvars_1761_);
lean_dec_ref(v_post_1757_);
lean_dec_ref(v_pre_1756_);
return v___x_1775_;
}
}
else
{
lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1786_ = lean_expr_instantiate_rev(v_e_1762_, v_fvars_1761_);
lean_dec_ref(v_e_1762_);
lean_inc_ref(v_post_1757_);
lean_inc_ref(v_pre_1756_);
v___x_1787_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1756_, v_post_1757_, v_usedLetOnly_1758_, v_skipConstInApp_1759_, v_skipInstances_1760_, v___x_1786_, v_a_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
if (lean_obj_tag(v___x_1787_) == 0)
{
lean_object* v_a_1788_; uint8_t v___x_1789_; uint8_t v___x_1790_; lean_object* v___x_1791_; 
v_a_1788_ = lean_ctor_get(v___x_1787_, 0);
lean_inc(v_a_1788_);
lean_dec_ref_known(v___x_1787_, 1);
v___x_1789_ = 0;
v___x_1790_ = 1;
v___x_1791_ = l_Lean_Meta_mkLetFVars(v_fvars_1761_, v_a_1788_, v_usedLetOnly_1758_, v___x_1789_, v___x_1790_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
lean_dec_ref(v_fvars_1761_);
if (lean_obj_tag(v___x_1791_) == 0)
{
lean_object* v_a_1792_; lean_object* v___x_1793_; 
v_a_1792_ = lean_ctor_get(v___x_1791_, 0);
lean_inc(v_a_1792_);
lean_dec_ref_known(v___x_1791_, 1);
v___x_1793_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_1756_, v_post_1757_, v_usedLetOnly_1758_, v_skipConstInApp_1759_, v_skipInstances_1760_, v_a_1792_, v_a_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
return v___x_1793_;
}
else
{
lean_dec_ref(v_post_1757_);
lean_dec_ref(v_pre_1756_);
return v___x_1791_;
}
}
else
{
lean_dec_ref(v_fvars_1761_);
lean_dec_ref(v_post_1757_);
lean_dec_ref(v_pre_1756_);
return v___x_1787_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1794_; lean_object* v_dummy_1795_; 
v___x_1794_ = lean_box(0);
v_dummy_1795_ = l_Lean_Expr_sort___override(v___x_1794_);
return v_dummy_1795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1(lean_object* v_pre_1796_, lean_object* v_post_1797_, uint8_t v_usedLetOnly_1798_, uint8_t v_skipConstInApp_1799_, uint8_t v_skipInstances_1800_, size_t v_sz_1801_, size_t v_i_1802_, lean_object* v_bs_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_){
_start:
{
uint8_t v___x_1810_; 
v___x_1810_ = lean_usize_dec_lt(v_i_1802_, v_sz_1801_);
if (v___x_1810_ == 0)
{
lean_object* v___x_1811_; 
lean_dec_ref(v_post_1797_);
lean_dec_ref(v_pre_1796_);
v___x_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1811_, 0, v_bs_1803_);
return v___x_1811_;
}
else
{
lean_object* v_v_1812_; lean_object* v___x_1813_; 
v_v_1812_ = lean_array_uget_borrowed(v_bs_1803_, v_i_1802_);
lean_inc(v_v_1812_);
lean_inc_ref(v_post_1797_);
lean_inc_ref(v_pre_1796_);
v___x_1813_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1796_, v_post_1797_, v_usedLetOnly_1798_, v_skipConstInApp_1799_, v_skipInstances_1800_, v_v_1812_, v___y_1804_, v___y_1805_, v___y_1806_, v___y_1807_, v___y_1808_);
if (lean_obj_tag(v___x_1813_) == 0)
{
lean_object* v_a_1814_; lean_object* v___x_1815_; lean_object* v_bs_x27_1816_; size_t v___x_1817_; size_t v___x_1818_; lean_object* v___x_1819_; 
v_a_1814_ = lean_ctor_get(v___x_1813_, 0);
lean_inc(v_a_1814_);
lean_dec_ref_known(v___x_1813_, 1);
v___x_1815_ = lean_unsigned_to_nat(0u);
v_bs_x27_1816_ = lean_array_uset(v_bs_1803_, v_i_1802_, v___x_1815_);
v___x_1817_ = ((size_t)1ULL);
v___x_1818_ = lean_usize_add(v_i_1802_, v___x_1817_);
v___x_1819_ = lean_array_uset(v_bs_x27_1816_, v_i_1802_, v_a_1814_);
v_i_1802_ = v___x_1818_;
v_bs_1803_ = v___x_1819_;
goto _start;
}
else
{
lean_object* v_a_1821_; lean_object* v___x_1823_; uint8_t v_isShared_1824_; uint8_t v_isSharedCheck_1828_; 
lean_dec_ref(v_bs_1803_);
lean_dec_ref(v_post_1797_);
lean_dec_ref(v_pre_1796_);
v_a_1821_ = lean_ctor_get(v___x_1813_, 0);
v_isSharedCheck_1828_ = !lean_is_exclusive(v___x_1813_);
if (v_isSharedCheck_1828_ == 0)
{
v___x_1823_ = v___x_1813_;
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
else
{
lean_inc(v_a_1821_);
lean_dec(v___x_1813_);
v___x_1823_ = lean_box(0);
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
v_resetjp_1822_:
{
lean_object* v___x_1826_; 
if (v_isShared_1824_ == 0)
{
v___x_1826_ = v___x_1823_;
goto v_reusejp_1825_;
}
else
{
lean_object* v_reuseFailAlloc_1827_; 
v_reuseFailAlloc_1827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1827_, 0, v_a_1821_);
v___x_1826_ = v_reuseFailAlloc_1827_;
goto v_reusejp_1825_;
}
v_reusejp_1825_:
{
return v___x_1826_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0(lean_object* v_pre_1829_, lean_object* v_post_1830_, uint8_t v_usedLetOnly_1831_, uint8_t v_skipConstInApp_1832_, uint8_t v_skipInstances_1833_, lean_object* v___x_1834_, lean_object* v___y_1835_, lean_object* v_b_1836_, lean_object* v_a_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_){
_start:
{
lean_object* v___x_1843_; 
v___x_1843_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1829_, v_post_1830_, v_usedLetOnly_1831_, v_skipConstInApp_1832_, v_skipInstances_1833_, v___x_1834_, v___y_1835_, v___y_1838_, v___y_1839_, v___y_1840_, v___y_1841_);
if (lean_obj_tag(v___x_1843_) == 0)
{
lean_object* v_a_1844_; lean_object* v___x_1846_; uint8_t v_isShared_1847_; uint8_t v_isSharedCheck_1853_; 
v_a_1844_ = lean_ctor_get(v___x_1843_, 0);
v_isSharedCheck_1853_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1853_ == 0)
{
v___x_1846_ = v___x_1843_;
v_isShared_1847_ = v_isSharedCheck_1853_;
goto v_resetjp_1845_;
}
else
{
lean_inc(v_a_1844_);
lean_dec(v___x_1843_);
v___x_1846_ = lean_box(0);
v_isShared_1847_ = v_isSharedCheck_1853_;
goto v_resetjp_1845_;
}
v_resetjp_1845_:
{
lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1851_; 
v___x_1848_ = lean_array_fset(v_b_1836_, v_a_1837_, v_a_1844_);
v___x_1849_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1849_, 0, v___x_1848_);
if (v_isShared_1847_ == 0)
{
lean_ctor_set(v___x_1846_, 0, v___x_1849_);
v___x_1851_ = v___x_1846_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v___x_1849_);
v___x_1851_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
return v___x_1851_;
}
}
}
else
{
lean_object* v_a_1854_; lean_object* v___x_1856_; uint8_t v_isShared_1857_; uint8_t v_isSharedCheck_1861_; 
lean_dec_ref(v_b_1836_);
v_a_1854_ = lean_ctor_get(v___x_1843_, 0);
v_isSharedCheck_1861_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1861_ == 0)
{
v___x_1856_ = v___x_1843_;
v_isShared_1857_ = v_isSharedCheck_1861_;
goto v_resetjp_1855_;
}
else
{
lean_inc(v_a_1854_);
lean_dec(v___x_1843_);
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
v_reuseFailAlloc_1860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1860_, 0, v_a_1854_);
v___x_1859_ = v_reuseFailAlloc_1860_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
return v___x_1859_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0___boxed(lean_object* v_pre_1862_, lean_object* v_post_1863_, lean_object* v_usedLetOnly_1864_, lean_object* v_skipConstInApp_1865_, lean_object* v_skipInstances_1866_, lean_object* v___x_1867_, lean_object* v___y_1868_, lean_object* v_b_1869_, lean_object* v_a_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_){
_start:
{
uint8_t v_usedLetOnly_boxed_1876_; uint8_t v_skipConstInApp_boxed_1877_; uint8_t v_skipInstances_boxed_1878_; lean_object* v_res_1879_; 
v_usedLetOnly_boxed_1876_ = lean_unbox(v_usedLetOnly_1864_);
v_skipConstInApp_boxed_1877_ = lean_unbox(v_skipConstInApp_1865_);
v_skipInstances_boxed_1878_ = lean_unbox(v_skipInstances_1866_);
v_res_1879_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0(v_pre_1862_, v_post_1863_, v_usedLetOnly_boxed_1876_, v_skipConstInApp_boxed_1877_, v_skipInstances_boxed_1878_, v___x_1867_, v___y_1868_, v_b_1869_, v_a_1870_, v___y_1871_, v___y_1872_, v___y_1873_, v___y_1874_);
lean_dec(v___y_1874_);
lean_dec_ref(v___y_1873_);
lean_dec(v___y_1872_);
lean_dec_ref(v___y_1871_);
lean_dec(v_a_1870_);
lean_dec(v___y_1868_);
return v_res_1879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg(lean_object* v_upperBound_1880_, lean_object* v___x_1881_, lean_object* v_pre_1882_, lean_object* v_post_1883_, uint8_t v_usedLetOnly_1884_, uint8_t v_skipConstInApp_1885_, uint8_t v_skipInstances_1886_, lean_object* v_a_1887_, lean_object* v_b_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_){
_start:
{
lean_object* v___y_1896_; uint8_t v___x_1919_; 
v___x_1919_ = lean_nat_dec_lt(v_a_1887_, v_upperBound_1880_);
if (v___x_1919_ == 0)
{
lean_object* v___x_1920_; 
lean_dec(v_a_1887_);
lean_dec_ref(v_post_1883_);
lean_dec_ref(v_pre_1882_);
v___x_1920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1920_, 0, v_b_1888_);
return v___x_1920_;
}
else
{
lean_object* v___x_1921_; lean_object* v___x_1922_; uint8_t v___x_1923_; 
v___x_1921_ = lean_array_fget_borrowed(v_b_1888_, v_a_1887_);
v___x_1922_ = lean_array_get_size(v___x_1881_);
v___x_1923_ = lean_nat_dec_lt(v_a_1887_, v___x_1922_);
if (v___x_1923_ == 0)
{
lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___f_1927_; 
lean_inc(v___x_1921_);
v___x_1924_ = lean_box(v_usedLetOnly_1884_);
v___x_1925_ = lean_box(v_skipConstInApp_1885_);
v___x_1926_ = lean_box(v_skipInstances_1886_);
lean_inc(v_a_1887_);
lean_inc(v___y_1889_);
lean_inc_ref(v_post_1883_);
lean_inc_ref(v_pre_1882_);
v___f_1927_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_1927_, 0, v_pre_1882_);
lean_closure_set(v___f_1927_, 1, v_post_1883_);
lean_closure_set(v___f_1927_, 2, v___x_1924_);
lean_closure_set(v___f_1927_, 3, v___x_1925_);
lean_closure_set(v___f_1927_, 4, v___x_1926_);
lean_closure_set(v___f_1927_, 5, v___x_1921_);
lean_closure_set(v___f_1927_, 6, v___y_1889_);
lean_closure_set(v___f_1927_, 7, v_b_1888_);
lean_closure_set(v___f_1927_, 8, v_a_1887_);
v___y_1896_ = v___f_1927_;
goto v___jp_1895_;
}
else
{
lean_object* v___x_1928_; uint8_t v_isInstance_1929_; 
v___x_1928_ = lean_array_fget_borrowed(v___x_1881_, v_a_1887_);
v_isInstance_1929_ = lean_ctor_get_uint8(v___x_1928_, sizeof(void*)*1 + 4);
if (v_isInstance_1929_ == 0)
{
lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___f_1933_; 
lean_inc(v___x_1921_);
v___x_1930_ = lean_box(v_usedLetOnly_1884_);
v___x_1931_ = lean_box(v_skipConstInApp_1885_);
v___x_1932_ = lean_box(v_skipInstances_1886_);
lean_inc(v_a_1887_);
lean_inc(v___y_1889_);
lean_inc_ref(v_post_1883_);
lean_inc_ref(v_pre_1882_);
v___f_1933_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_1933_, 0, v_pre_1882_);
lean_closure_set(v___f_1933_, 1, v_post_1883_);
lean_closure_set(v___f_1933_, 2, v___x_1930_);
lean_closure_set(v___f_1933_, 3, v___x_1931_);
lean_closure_set(v___f_1933_, 4, v___x_1932_);
lean_closure_set(v___f_1933_, 5, v___x_1921_);
lean_closure_set(v___f_1933_, 6, v___y_1889_);
lean_closure_set(v___f_1933_, 7, v_b_1888_);
lean_closure_set(v___f_1933_, 8, v_a_1887_);
v___y_1896_ = v___f_1933_;
goto v___jp_1895_;
}
else
{
lean_object* v___x_1934_; lean_object* v___f_1935_; 
v___x_1934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1934_, 0, v_b_1888_);
v___f_1935_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___lam__2___boxed), 6, 1);
lean_closure_set(v___f_1935_, 0, v___x_1934_);
v___y_1896_ = v___f_1935_;
goto v___jp_1895_;
}
}
}
v___jp_1895_:
{
lean_object* v___x_1897_; 
lean_inc(v___y_1893_);
lean_inc_ref(v___y_1892_);
lean_inc(v___y_1891_);
lean_inc_ref(v___y_1890_);
v___x_1897_ = lean_apply_5(v___y_1896_, v___y_1890_, v___y_1891_, v___y_1892_, v___y_1893_, lean_box(0));
if (lean_obj_tag(v___x_1897_) == 0)
{
lean_object* v_a_1898_; lean_object* v___x_1900_; uint8_t v_isShared_1901_; uint8_t v_isSharedCheck_1910_; 
v_a_1898_ = lean_ctor_get(v___x_1897_, 0);
v_isSharedCheck_1910_ = !lean_is_exclusive(v___x_1897_);
if (v_isSharedCheck_1910_ == 0)
{
v___x_1900_ = v___x_1897_;
v_isShared_1901_ = v_isSharedCheck_1910_;
goto v_resetjp_1899_;
}
else
{
lean_inc(v_a_1898_);
lean_dec(v___x_1897_);
v___x_1900_ = lean_box(0);
v_isShared_1901_ = v_isSharedCheck_1910_;
goto v_resetjp_1899_;
}
v_resetjp_1899_:
{
if (lean_obj_tag(v_a_1898_) == 0)
{
lean_object* v_a_1902_; lean_object* v___x_1904_; 
lean_dec(v_a_1887_);
lean_dec_ref(v_post_1883_);
lean_dec_ref(v_pre_1882_);
v_a_1902_ = lean_ctor_get(v_a_1898_, 0);
lean_inc(v_a_1902_);
lean_dec_ref_known(v_a_1898_, 1);
if (v_isShared_1901_ == 0)
{
lean_ctor_set(v___x_1900_, 0, v_a_1902_);
v___x_1904_ = v___x_1900_;
goto v_reusejp_1903_;
}
else
{
lean_object* v_reuseFailAlloc_1905_; 
v_reuseFailAlloc_1905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1905_, 0, v_a_1902_);
v___x_1904_ = v_reuseFailAlloc_1905_;
goto v_reusejp_1903_;
}
v_reusejp_1903_:
{
return v___x_1904_;
}
}
else
{
lean_object* v_a_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; 
lean_del_object(v___x_1900_);
v_a_1906_ = lean_ctor_get(v_a_1898_, 0);
lean_inc(v_a_1906_);
lean_dec_ref_known(v_a_1898_, 1);
v___x_1907_ = lean_unsigned_to_nat(1u);
v___x_1908_ = lean_nat_add(v_a_1887_, v___x_1907_);
lean_dec(v_a_1887_);
v_a_1887_ = v___x_1908_;
v_b_1888_ = v_a_1906_;
goto _start;
}
}
}
else
{
lean_object* v_a_1911_; lean_object* v___x_1913_; uint8_t v_isShared_1914_; uint8_t v_isSharedCheck_1918_; 
lean_dec(v_a_1887_);
lean_dec_ref(v_post_1883_);
lean_dec_ref(v_pre_1882_);
v_a_1911_ = lean_ctor_get(v___x_1897_, 0);
v_isSharedCheck_1918_ = !lean_is_exclusive(v___x_1897_);
if (v_isSharedCheck_1918_ == 0)
{
v___x_1913_ = v___x_1897_;
v_isShared_1914_ = v_isSharedCheck_1918_;
goto v_resetjp_1912_;
}
else
{
lean_inc(v_a_1911_);
lean_dec(v___x_1897_);
v___x_1913_ = lean_box(0);
v_isShared_1914_ = v_isSharedCheck_1918_;
goto v_resetjp_1912_;
}
v_resetjp_1912_:
{
lean_object* v___x_1916_; 
if (v_isShared_1914_ == 0)
{
v___x_1916_ = v___x_1913_;
goto v_reusejp_1915_;
}
else
{
lean_object* v_reuseFailAlloc_1917_; 
v_reuseFailAlloc_1917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1917_, 0, v_a_1911_);
v___x_1916_ = v_reuseFailAlloc_1917_;
goto v_reusejp_1915_;
}
v_reusejp_1915_:
{
return v___x_1916_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8(uint8_t v_skipInstances_1936_, lean_object* v_pre_1937_, lean_object* v_post_1938_, uint8_t v_usedLetOnly_1939_, uint8_t v_skipConstInApp_1940_, lean_object* v_x_1941_, lean_object* v_x_1942_, lean_object* v_x_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
lean_object* v_f_1951_; lean_object* v___y_1952_; lean_object* v___y_1953_; lean_object* v___y_1954_; lean_object* v___y_1955_; lean_object* v___y_1956_; 
if (lean_obj_tag(v_x_1941_) == 5)
{
lean_object* v_fn_1999_; lean_object* v_arg_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; 
v_fn_1999_ = lean_ctor_get(v_x_1941_, 0);
lean_inc_ref(v_fn_1999_);
v_arg_2000_ = lean_ctor_get(v_x_1941_, 1);
lean_inc_ref(v_arg_2000_);
lean_dec_ref_known(v_x_1941_, 2);
v___x_2001_ = lean_array_set(v_x_1942_, v_x_1943_, v_arg_2000_);
v___x_2002_ = lean_unsigned_to_nat(1u);
v___x_2003_ = lean_nat_sub(v_x_1943_, v___x_2002_);
lean_dec(v_x_1943_);
v_x_1941_ = v_fn_1999_;
v_x_1942_ = v___x_2001_;
v_x_1943_ = v___x_2003_;
goto _start;
}
else
{
lean_dec(v_x_1943_);
if (v_skipConstInApp_1940_ == 0)
{
goto v___jp_1996_;
}
else
{
uint8_t v___x_2005_; 
v___x_2005_ = l_Lean_Expr_isConst(v_x_1941_);
if (v___x_2005_ == 0)
{
goto v___jp_1996_;
}
else
{
v_f_1951_ = v_x_1941_;
v___y_1952_ = v___y_1944_;
v___y_1953_ = v___y_1945_;
v___y_1954_ = v___y_1946_;
v___y_1955_ = v___y_1947_;
v___y_1956_ = v___y_1948_;
goto v___jp_1950_;
}
}
}
v___jp_1950_:
{
if (v_skipInstances_1936_ == 0)
{
size_t v_sz_1957_; size_t v___x_1958_; lean_object* v___x_1959_; 
v_sz_1957_ = lean_array_size(v_x_1942_);
v___x_1958_ = ((size_t)0ULL);
lean_inc_ref(v_post_1938_);
lean_inc_ref(v_pre_1937_);
v___x_1959_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1(v_pre_1937_, v_post_1938_, v_usedLetOnly_1939_, v_skipConstInApp_1940_, v_skipInstances_1936_, v_sz_1957_, v___x_1958_, v_x_1942_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
if (lean_obj_tag(v___x_1959_) == 0)
{
lean_object* v_a_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; 
v_a_1960_ = lean_ctor_get(v___x_1959_, 0);
lean_inc(v_a_1960_);
lean_dec_ref_known(v___x_1959_, 1);
v___x_1961_ = l_Lean_mkAppN(v_f_1951_, v_a_1960_);
lean_dec(v_a_1960_);
v___x_1962_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_1937_, v_post_1938_, v_usedLetOnly_1939_, v_skipConstInApp_1940_, v_skipInstances_1936_, v___x_1961_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
return v___x_1962_;
}
else
{
lean_object* v_a_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1970_; 
lean_dec_ref(v_f_1951_);
lean_dec_ref(v_post_1938_);
lean_dec_ref(v_pre_1937_);
v_a_1963_ = lean_ctor_get(v___x_1959_, 0);
v_isSharedCheck_1970_ = !lean_is_exclusive(v___x_1959_);
if (v_isSharedCheck_1970_ == 0)
{
v___x_1965_ = v___x_1959_;
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_a_1963_);
lean_dec(v___x_1959_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1968_; 
if (v_isShared_1966_ == 0)
{
v___x_1968_ = v___x_1965_;
goto v_reusejp_1967_;
}
else
{
lean_object* v_reuseFailAlloc_1969_; 
v_reuseFailAlloc_1969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1969_, 0, v_a_1963_);
v___x_1968_ = v_reuseFailAlloc_1969_;
goto v_reusejp_1967_;
}
v_reusejp_1967_:
{
return v___x_1968_;
}
}
}
}
else
{
lean_object* v___x_1971_; lean_object* v___x_1972_; 
v___x_1971_ = lean_array_get_size(v_x_1942_);
lean_inc_ref(v_f_1951_);
v___x_1972_ = l_Lean_Meta_getFunInfoNArgs(v_f_1951_, v___x_1971_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
if (lean_obj_tag(v___x_1972_) == 0)
{
lean_object* v_a_1973_; lean_object* v_paramInfo_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; 
v_a_1973_ = lean_ctor_get(v___x_1972_, 0);
lean_inc(v_a_1973_);
lean_dec_ref_known(v___x_1972_, 1);
v_paramInfo_1974_ = lean_ctor_get(v_a_1973_, 0);
lean_inc_ref(v_paramInfo_1974_);
lean_dec(v_a_1973_);
v___x_1975_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_post_1938_);
lean_inc_ref(v_pre_1937_);
v___x_1976_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg(v___x_1971_, v_paramInfo_1974_, v_pre_1937_, v_post_1938_, v_usedLetOnly_1939_, v_skipConstInApp_1940_, v_skipInstances_1936_, v___x_1975_, v_x_1942_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
lean_dec_ref(v_paramInfo_1974_);
if (lean_obj_tag(v___x_1976_) == 0)
{
lean_object* v_a_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; 
v_a_1977_ = lean_ctor_get(v___x_1976_, 0);
lean_inc(v_a_1977_);
lean_dec_ref_known(v___x_1976_, 1);
v___x_1978_ = l_Lean_mkAppN(v_f_1951_, v_a_1977_);
lean_dec(v_a_1977_);
v___x_1979_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_1937_, v_post_1938_, v_usedLetOnly_1939_, v_skipConstInApp_1940_, v_skipInstances_1936_, v___x_1978_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
return v___x_1979_;
}
else
{
lean_object* v_a_1980_; lean_object* v___x_1982_; uint8_t v_isShared_1983_; uint8_t v_isSharedCheck_1987_; 
lean_dec_ref(v_f_1951_);
lean_dec_ref(v_post_1938_);
lean_dec_ref(v_pre_1937_);
v_a_1980_ = lean_ctor_get(v___x_1976_, 0);
v_isSharedCheck_1987_ = !lean_is_exclusive(v___x_1976_);
if (v_isSharedCheck_1987_ == 0)
{
v___x_1982_ = v___x_1976_;
v_isShared_1983_ = v_isSharedCheck_1987_;
goto v_resetjp_1981_;
}
else
{
lean_inc(v_a_1980_);
lean_dec(v___x_1976_);
v___x_1982_ = lean_box(0);
v_isShared_1983_ = v_isSharedCheck_1987_;
goto v_resetjp_1981_;
}
v_resetjp_1981_:
{
lean_object* v___x_1985_; 
if (v_isShared_1983_ == 0)
{
v___x_1985_ = v___x_1982_;
goto v_reusejp_1984_;
}
else
{
lean_object* v_reuseFailAlloc_1986_; 
v_reuseFailAlloc_1986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1986_, 0, v_a_1980_);
v___x_1985_ = v_reuseFailAlloc_1986_;
goto v_reusejp_1984_;
}
v_reusejp_1984_:
{
return v___x_1985_;
}
}
}
}
else
{
lean_object* v_a_1988_; lean_object* v___x_1990_; uint8_t v_isShared_1991_; uint8_t v_isSharedCheck_1995_; 
lean_dec_ref(v_f_1951_);
lean_dec_ref(v_x_1942_);
lean_dec_ref(v_post_1938_);
lean_dec_ref(v_pre_1937_);
v_a_1988_ = lean_ctor_get(v___x_1972_, 0);
v_isSharedCheck_1995_ = !lean_is_exclusive(v___x_1972_);
if (v_isSharedCheck_1995_ == 0)
{
v___x_1990_ = v___x_1972_;
v_isShared_1991_ = v_isSharedCheck_1995_;
goto v_resetjp_1989_;
}
else
{
lean_inc(v_a_1988_);
lean_dec(v___x_1972_);
v___x_1990_ = lean_box(0);
v_isShared_1991_ = v_isSharedCheck_1995_;
goto v_resetjp_1989_;
}
v_resetjp_1989_:
{
lean_object* v___x_1993_; 
if (v_isShared_1991_ == 0)
{
v___x_1993_ = v___x_1990_;
goto v_reusejp_1992_;
}
else
{
lean_object* v_reuseFailAlloc_1994_; 
v_reuseFailAlloc_1994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1994_, 0, v_a_1988_);
v___x_1993_ = v_reuseFailAlloc_1994_;
goto v_reusejp_1992_;
}
v_reusejp_1992_:
{
return v___x_1993_;
}
}
}
}
}
v___jp_1996_:
{
lean_object* v___x_1997_; 
lean_inc_ref(v_post_1938_);
lean_inc_ref(v_pre_1937_);
v___x_1997_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_1937_, v_post_1938_, v_usedLetOnly_1939_, v_skipConstInApp_1940_, v_skipInstances_1936_, v_x_1941_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
if (lean_obj_tag(v___x_1997_) == 0)
{
lean_object* v_a_1998_; 
v_a_1998_ = lean_ctor_get(v___x_1997_, 0);
lean_inc(v_a_1998_);
lean_dec_ref_known(v___x_1997_, 1);
v_f_1951_ = v_a_1998_;
v___y_1952_ = v___y_1944_;
v___y_1953_ = v___y_1945_;
v___y_1954_ = v___y_1946_;
v___y_1955_ = v___y_1947_;
v___y_1956_ = v___y_1948_;
goto v___jp_1950_;
}
else
{
lean_dec_ref(v_x_1942_);
lean_dec_ref(v_post_1938_);
lean_dec_ref(v_pre_1937_);
return v___x_1997_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1(lean_object* v___x_2006_, lean_object* v_pre_2007_, lean_object* v_e_2008_, lean_object* v_post_2009_, uint8_t v_usedLetOnly_2010_, uint8_t v_skipConstInApp_2011_, uint8_t v_skipInstances_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_){
_start:
{
lean_object* v___x_2019_; 
v___x_2019_ = l_Lean_Core_checkSystem(v___x_2006_, v___y_2016_, v___y_2017_);
if (lean_obj_tag(v___x_2019_) == 0)
{
lean_object* v___x_2020_; 
lean_dec_ref_known(v___x_2019_, 1);
lean_inc_ref(v_pre_2007_);
lean_inc(v___y_2017_);
lean_inc_ref(v___y_2016_);
lean_inc(v___y_2015_);
lean_inc_ref(v___y_2014_);
lean_inc_ref(v_e_2008_);
v___x_2020_ = lean_apply_6(v_pre_2007_, v_e_2008_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_, lean_box(0));
if (lean_obj_tag(v___x_2020_) == 0)
{
lean_object* v_a_2021_; lean_object* v___x_2023_; uint8_t v_isShared_2024_; uint8_t v_isSharedCheck_2069_; 
v_a_2021_ = lean_ctor_get(v___x_2020_, 0);
v_isSharedCheck_2069_ = !lean_is_exclusive(v___x_2020_);
if (v_isSharedCheck_2069_ == 0)
{
v___x_2023_ = v___x_2020_;
v_isShared_2024_ = v_isSharedCheck_2069_;
goto v_resetjp_2022_;
}
else
{
lean_inc(v_a_2021_);
lean_dec(v___x_2020_);
v___x_2023_ = lean_box(0);
v_isShared_2024_ = v_isSharedCheck_2069_;
goto v_resetjp_2022_;
}
v_resetjp_2022_:
{
lean_object* v___y_2026_; 
switch(lean_obj_tag(v_a_2021_))
{
case 0:
{
lean_object* v_e_2061_; lean_object* v___x_2063_; 
lean_dec_ref(v_post_2009_);
lean_dec_ref(v_e_2008_);
lean_dec_ref(v_pre_2007_);
v_e_2061_ = lean_ctor_get(v_a_2021_, 0);
lean_inc_ref(v_e_2061_);
lean_dec_ref_known(v_a_2021_, 1);
if (v_isShared_2024_ == 0)
{
lean_ctor_set(v___x_2023_, 0, v_e_2061_);
v___x_2063_ = v___x_2023_;
goto v_reusejp_2062_;
}
else
{
lean_object* v_reuseFailAlloc_2064_; 
v_reuseFailAlloc_2064_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2064_, 0, v_e_2061_);
v___x_2063_ = v_reuseFailAlloc_2064_;
goto v_reusejp_2062_;
}
v_reusejp_2062_:
{
return v___x_2063_;
}
}
case 1:
{
lean_object* v_e_2065_; lean_object* v___x_2066_; 
lean_del_object(v___x_2023_);
lean_dec_ref(v_e_2008_);
v_e_2065_ = lean_ctor_get(v_a_2021_, 0);
lean_inc_ref(v_e_2065_);
lean_dec_ref_known(v_a_2021_, 1);
v___x_2066_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v_e_2065_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2066_;
}
default: 
{
lean_object* v_e_x3f_2067_; 
lean_del_object(v___x_2023_);
v_e_x3f_2067_ = lean_ctor_get(v_a_2021_, 0);
lean_inc(v_e_x3f_2067_);
lean_dec_ref_known(v_a_2021_, 1);
if (lean_obj_tag(v_e_x3f_2067_) == 0)
{
v___y_2026_ = v_e_2008_;
goto v___jp_2025_;
}
else
{
lean_object* v_val_2068_; 
lean_dec_ref(v_e_2008_);
v_val_2068_ = lean_ctor_get(v_e_x3f_2067_, 0);
lean_inc(v_val_2068_);
lean_dec_ref_known(v_e_x3f_2067_, 1);
v___y_2026_ = v_val_2068_;
goto v___jp_2025_;
}
}
}
v___jp_2025_:
{
switch(lean_obj_tag(v___y_2026_))
{
case 7:
{
lean_object* v___x_2027_; lean_object* v___x_2028_; 
v___x_2027_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0));
v___x_2028_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___x_2027_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2028_;
}
case 6:
{
lean_object* v___x_2029_; lean_object* v___x_2030_; 
v___x_2029_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0));
v___x_2030_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___x_2029_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2030_;
}
case 8:
{
lean_object* v___x_2031_; lean_object* v___x_2032_; 
v___x_2031_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__0));
v___x_2032_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___x_2031_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2032_;
}
case 5:
{
lean_object* v_dummy_2033_; lean_object* v_nargs_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; 
v_dummy_2033_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1);
v_nargs_2034_ = l_Lean_Expr_getAppNumArgs(v___y_2026_);
lean_inc(v_nargs_2034_);
v___x_2035_ = lean_mk_array(v_nargs_2034_, v_dummy_2033_);
v___x_2036_ = lean_unsigned_to_nat(1u);
v___x_2037_ = lean_nat_sub(v_nargs_2034_, v___x_2036_);
lean_dec(v_nargs_2034_);
v___x_2038_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8(v_skipInstances_2012_, v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v___y_2026_, v___x_2035_, v___x_2037_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2038_;
}
case 10:
{
lean_object* v_data_2039_; lean_object* v_expr_2040_; lean_object* v___x_2041_; 
v_data_2039_ = lean_ctor_get(v___y_2026_, 0);
v_expr_2040_ = lean_ctor_get(v___y_2026_, 1);
lean_inc_ref(v_expr_2040_);
lean_inc_ref(v_post_2009_);
lean_inc_ref(v_pre_2007_);
v___x_2041_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v_expr_2040_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
if (lean_obj_tag(v___x_2041_) == 0)
{
lean_object* v_a_2042_; size_t v___x_2043_; size_t v___x_2044_; uint8_t v___x_2045_; 
v_a_2042_ = lean_ctor_get(v___x_2041_, 0);
lean_inc(v_a_2042_);
lean_dec_ref_known(v___x_2041_, 1);
v___x_2043_ = lean_ptr_addr(v_expr_2040_);
v___x_2044_ = lean_ptr_addr(v_a_2042_);
v___x_2045_ = lean_usize_dec_eq(v___x_2043_, v___x_2044_);
if (v___x_2045_ == 0)
{
lean_object* v___x_2046_; lean_object* v___x_2047_; 
lean_inc(v_data_2039_);
lean_dec_ref_known(v___y_2026_, 2);
v___x_2046_ = l_Lean_Expr_mdata___override(v_data_2039_, v_a_2042_);
v___x_2047_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___x_2046_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2047_;
}
else
{
lean_object* v___x_2048_; 
lean_dec(v_a_2042_);
v___x_2048_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2048_;
}
}
else
{
lean_dec_ref_known(v___y_2026_, 2);
lean_dec_ref(v_post_2009_);
lean_dec_ref(v_pre_2007_);
return v___x_2041_;
}
}
case 11:
{
lean_object* v_typeName_2049_; lean_object* v_idx_2050_; lean_object* v_struct_2051_; lean_object* v___x_2052_; 
v_typeName_2049_ = lean_ctor_get(v___y_2026_, 0);
v_idx_2050_ = lean_ctor_get(v___y_2026_, 1);
v_struct_2051_ = lean_ctor_get(v___y_2026_, 2);
lean_inc_ref(v_struct_2051_);
lean_inc_ref(v_post_2009_);
lean_inc_ref(v_pre_2007_);
v___x_2052_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v_struct_2051_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
if (lean_obj_tag(v___x_2052_) == 0)
{
lean_object* v_a_2053_; size_t v___x_2054_; size_t v___x_2055_; uint8_t v___x_2056_; 
v_a_2053_ = lean_ctor_get(v___x_2052_, 0);
lean_inc(v_a_2053_);
lean_dec_ref_known(v___x_2052_, 1);
v___x_2054_ = lean_ptr_addr(v_struct_2051_);
v___x_2055_ = lean_ptr_addr(v_a_2053_);
v___x_2056_ = lean_usize_dec_eq(v___x_2054_, v___x_2055_);
if (v___x_2056_ == 0)
{
lean_object* v___x_2057_; lean_object* v___x_2058_; 
lean_inc(v_idx_2050_);
lean_inc(v_typeName_2049_);
lean_dec_ref_known(v___y_2026_, 3);
v___x_2057_ = l_Lean_Expr_proj___override(v_typeName_2049_, v_idx_2050_, v_a_2053_);
v___x_2058_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___x_2057_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2058_;
}
else
{
lean_object* v___x_2059_; 
lean_dec(v_a_2053_);
v___x_2059_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2059_;
}
}
else
{
lean_dec_ref_known(v___y_2026_, 3);
lean_dec_ref(v_post_2009_);
lean_dec_ref(v_pre_2007_);
return v___x_2052_;
}
}
default: 
{
lean_object* v___x_2060_; 
v___x_2060_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2007_, v_post_2009_, v_usedLetOnly_2010_, v_skipConstInApp_2011_, v_skipInstances_2012_, v___y_2026_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
return v___x_2060_;
}
}
}
}
}
else
{
lean_object* v_a_2070_; lean_object* v___x_2072_; uint8_t v_isShared_2073_; uint8_t v_isSharedCheck_2077_; 
lean_dec_ref(v_post_2009_);
lean_dec_ref(v_e_2008_);
lean_dec_ref(v_pre_2007_);
v_a_2070_ = lean_ctor_get(v___x_2020_, 0);
v_isSharedCheck_2077_ = !lean_is_exclusive(v___x_2020_);
if (v_isSharedCheck_2077_ == 0)
{
v___x_2072_ = v___x_2020_;
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
else
{
lean_inc(v_a_2070_);
lean_dec(v___x_2020_);
v___x_2072_ = lean_box(0);
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
v_resetjp_2071_:
{
lean_object* v___x_2075_; 
if (v_isShared_2073_ == 0)
{
v___x_2075_ = v___x_2072_;
goto v_reusejp_2074_;
}
else
{
lean_object* v_reuseFailAlloc_2076_; 
v_reuseFailAlloc_2076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2076_, 0, v_a_2070_);
v___x_2075_ = v_reuseFailAlloc_2076_;
goto v_reusejp_2074_;
}
v_reusejp_2074_:
{
return v___x_2075_;
}
}
}
}
else
{
lean_object* v_a_2078_; lean_object* v___x_2080_; uint8_t v_isShared_2081_; uint8_t v_isSharedCheck_2085_; 
lean_dec_ref(v_post_2009_);
lean_dec_ref(v_e_2008_);
lean_dec_ref(v_pre_2007_);
v_a_2078_ = lean_ctor_get(v___x_2019_, 0);
v_isSharedCheck_2085_ = !lean_is_exclusive(v___x_2019_);
if (v_isSharedCheck_2085_ == 0)
{
v___x_2080_ = v___x_2019_;
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
else
{
lean_inc(v_a_2078_);
lean_dec(v___x_2019_);
v___x_2080_ = lean_box(0);
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
v_resetjp_2079_:
{
lean_object* v___x_2083_; 
if (v_isShared_2081_ == 0)
{
v___x_2083_ = v___x_2080_;
goto v_reusejp_2082_;
}
else
{
lean_object* v_reuseFailAlloc_2084_; 
v_reuseFailAlloc_2084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2084_, 0, v_a_2078_);
v___x_2083_ = v_reuseFailAlloc_2084_;
goto v_reusejp_2082_;
}
v_reusejp_2082_:
{
return v___x_2083_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___boxed(lean_object* v___x_2086_, lean_object* v_pre_2087_, lean_object* v_e_2088_, lean_object* v_post_2089_, lean_object* v_usedLetOnly_2090_, lean_object* v_skipConstInApp_2091_, lean_object* v_skipInstances_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_){
_start:
{
uint8_t v_usedLetOnly_boxed_2099_; uint8_t v_skipConstInApp_boxed_2100_; uint8_t v_skipInstances_boxed_2101_; lean_object* v_res_2102_; 
v_usedLetOnly_boxed_2099_ = lean_unbox(v_usedLetOnly_2090_);
v_skipConstInApp_boxed_2100_ = lean_unbox(v_skipConstInApp_2091_);
v_skipInstances_boxed_2101_ = lean_unbox(v_skipInstances_2092_);
v_res_2102_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1(v___x_2086_, v_pre_2087_, v_e_2088_, v_post_2089_, v_usedLetOnly_boxed_2099_, v_skipConstInApp_boxed_2100_, v_skipInstances_boxed_2101_, v___y_2093_, v___y_2094_, v___y_2095_, v___y_2096_, v___y_2097_);
lean_dec(v___y_2097_);
lean_dec_ref(v___y_2096_);
lean_dec(v___y_2095_);
lean_dec_ref(v___y_2094_);
lean_dec(v___y_2093_);
return v_res_2102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(lean_object* v_pre_2103_, lean_object* v_post_2104_, uint8_t v_usedLetOnly_2105_, uint8_t v_skipConstInApp_2106_, uint8_t v_skipInstances_2107_, lean_object* v_e_2108_, lean_object* v_a_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_){
_start:
{
lean_object* v___x_2115_; lean_object* v___x_2116_; 
lean_inc(v_a_2109_);
v___x_2115_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2115_, 0, lean_box(0));
lean_closure_set(v___x_2115_, 1, lean_box(0));
lean_closure_set(v___x_2115_, 2, v_a_2109_);
v___x_2116_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0(lean_box(0), v___x_2115_, v___y_2110_, v___y_2111_, v___y_2112_, v___y_2113_);
if (lean_obj_tag(v___x_2116_) == 0)
{
lean_object* v_a_2117_; lean_object* v___x_2119_; uint8_t v_isShared_2120_; uint8_t v_isSharedCheck_2151_; 
v_a_2117_ = lean_ctor_get(v___x_2116_, 0);
v_isSharedCheck_2151_ = !lean_is_exclusive(v___x_2116_);
if (v_isSharedCheck_2151_ == 0)
{
v___x_2119_ = v___x_2116_;
v_isShared_2120_ = v_isSharedCheck_2151_;
goto v_resetjp_2118_;
}
else
{
lean_inc(v_a_2117_);
lean_dec(v___x_2116_);
v___x_2119_ = lean_box(0);
v_isShared_2120_ = v_isSharedCheck_2151_;
goto v_resetjp_2118_;
}
v_resetjp_2118_:
{
lean_object* v___x_2121_; 
v___x_2121_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg(v_a_2117_, v_e_2108_);
lean_dec(v_a_2117_);
if (lean_obj_tag(v___x_2121_) == 0)
{
lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___f_2126_; lean_object* v___x_2127_; 
lean_del_object(v___x_2119_);
v___x_2122_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___closed__0));
v___x_2123_ = lean_box(v_usedLetOnly_2105_);
v___x_2124_ = lean_box(v_skipConstInApp_2106_);
v___x_2125_ = lean_box(v_skipInstances_2107_);
lean_inc_ref(v_e_2108_);
v___f_2126_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___boxed), 13, 7);
lean_closure_set(v___f_2126_, 0, v___x_2122_);
lean_closure_set(v___f_2126_, 1, v_pre_2103_);
lean_closure_set(v___f_2126_, 2, v_e_2108_);
lean_closure_set(v___f_2126_, 3, v_post_2104_);
lean_closure_set(v___f_2126_, 4, v___x_2123_);
lean_closure_set(v___f_2126_, 5, v___x_2124_);
lean_closure_set(v___f_2126_, 6, v___x_2125_);
v___x_2127_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg(v___f_2126_, v_a_2109_, v___y_2110_, v___y_2111_, v___y_2112_, v___y_2113_);
if (lean_obj_tag(v___x_2127_) == 0)
{
lean_object* v_a_2128_; lean_object* v___f_2129_; lean_object* v___x_2130_; 
v_a_2128_ = lean_ctor_get(v___x_2127_, 0);
lean_inc_n(v_a_2128_, 2);
lean_dec_ref_known(v___x_2127_, 1);
lean_inc(v_a_2109_);
v___f_2129_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__2___boxed), 4, 3);
lean_closure_set(v___f_2129_, 0, v_a_2109_);
lean_closure_set(v___f_2129_, 1, v_e_2108_);
lean_closure_set(v___f_2129_, 2, v_a_2128_);
v___x_2130_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__0(lean_box(0), v___f_2129_, v___y_2110_, v___y_2111_, v___y_2112_, v___y_2113_);
if (lean_obj_tag(v___x_2130_) == 0)
{
lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2137_; 
v_isSharedCheck_2137_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2137_ == 0)
{
lean_object* v_unused_2138_; 
v_unused_2138_ = lean_ctor_get(v___x_2130_, 0);
lean_dec(v_unused_2138_);
v___x_2132_ = v___x_2130_;
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
else
{
lean_dec(v___x_2130_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
lean_object* v___x_2135_; 
if (v_isShared_2133_ == 0)
{
lean_ctor_set(v___x_2132_, 0, v_a_2128_);
v___x_2135_ = v___x_2132_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2136_; 
v_reuseFailAlloc_2136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2136_, 0, v_a_2128_);
v___x_2135_ = v_reuseFailAlloc_2136_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
return v___x_2135_;
}
}
}
else
{
lean_object* v_a_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2146_; 
lean_dec(v_a_2128_);
v_a_2139_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2146_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2146_ == 0)
{
v___x_2141_ = v___x_2130_;
v_isShared_2142_ = v_isSharedCheck_2146_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_a_2139_);
lean_dec(v___x_2130_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2146_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v___x_2144_; 
if (v_isShared_2142_ == 0)
{
v___x_2144_ = v___x_2141_;
goto v_reusejp_2143_;
}
else
{
lean_object* v_reuseFailAlloc_2145_; 
v_reuseFailAlloc_2145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2145_, 0, v_a_2139_);
v___x_2144_ = v_reuseFailAlloc_2145_;
goto v_reusejp_2143_;
}
v_reusejp_2143_:
{
return v___x_2144_;
}
}
}
}
else
{
lean_dec_ref(v_e_2108_);
return v___x_2127_;
}
}
else
{
lean_object* v_val_2147_; lean_object* v___x_2149_; 
lean_dec_ref(v_e_2108_);
lean_dec_ref(v_post_2104_);
lean_dec_ref(v_pre_2103_);
v_val_2147_ = lean_ctor_get(v___x_2121_, 0);
lean_inc(v_val_2147_);
lean_dec_ref_known(v___x_2121_, 1);
if (v_isShared_2120_ == 0)
{
lean_ctor_set(v___x_2119_, 0, v_val_2147_);
v___x_2149_ = v___x_2119_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2150_; 
v_reuseFailAlloc_2150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2150_, 0, v_val_2147_);
v___x_2149_ = v_reuseFailAlloc_2150_;
goto v_reusejp_2148_;
}
v_reusejp_2148_:
{
return v___x_2149_;
}
}
}
}
else
{
lean_object* v_a_2152_; lean_object* v___x_2154_; uint8_t v_isShared_2155_; uint8_t v_isSharedCheck_2159_; 
lean_dec_ref(v_e_2108_);
lean_dec_ref(v_post_2104_);
lean_dec_ref(v_pre_2103_);
v_a_2152_ = lean_ctor_get(v___x_2116_, 0);
v_isSharedCheck_2159_ = !lean_is_exclusive(v___x_2116_);
if (v_isSharedCheck_2159_ == 0)
{
v___x_2154_ = v___x_2116_;
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
else
{
lean_inc(v_a_2152_);
lean_dec(v___x_2116_);
v___x_2154_ = lean_box(0);
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
v_resetjp_2153_:
{
lean_object* v___x_2157_; 
if (v_isShared_2155_ == 0)
{
v___x_2157_ = v___x_2154_;
goto v_reusejp_2156_;
}
else
{
lean_object* v_reuseFailAlloc_2158_; 
v_reuseFailAlloc_2158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2158_, 0, v_a_2152_);
v___x_2157_ = v_reuseFailAlloc_2158_;
goto v_reusejp_2156_;
}
v_reusejp_2156_:
{
return v___x_2157_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0___boxed(lean_object* v_fvars_2160_, lean_object* v_pre_2161_, lean_object* v_post_2162_, lean_object* v_usedLetOnly_2163_, lean_object* v_skipConstInApp_2164_, lean_object* v_skipInstances_2165_, lean_object* v_body_2166_, lean_object* v_x_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_){
_start:
{
uint8_t v_usedLetOnly_boxed_2174_; uint8_t v_skipConstInApp_boxed_2175_; uint8_t v_skipInstances_boxed_2176_; lean_object* v_res_2177_; 
v_usedLetOnly_boxed_2174_ = lean_unbox(v_usedLetOnly_2163_);
v_skipConstInApp_boxed_2175_ = lean_unbox(v_skipConstInApp_2164_);
v_skipInstances_boxed_2176_ = lean_unbox(v_skipInstances_2165_);
v_res_2177_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0(v_fvars_2160_, v_pre_2161_, v_post_2162_, v_usedLetOnly_boxed_2174_, v_skipConstInApp_boxed_2175_, v_skipInstances_boxed_2176_, v_body_2166_, v_x_2167_, v___y_2168_, v___y_2169_, v___y_2170_, v___y_2171_, v___y_2172_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2171_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___y_2168_);
return v_res_2177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5(lean_object* v_pre_2178_, lean_object* v_post_2179_, uint8_t v_usedLetOnly_2180_, uint8_t v_skipConstInApp_2181_, uint8_t v_skipInstances_2182_, lean_object* v_fvars_2183_, lean_object* v_e_2184_, lean_object* v_a_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_){
_start:
{
if (lean_obj_tag(v_e_2184_) == 7)
{
lean_object* v_binderName_2191_; lean_object* v_binderType_2192_; lean_object* v_body_2193_; uint8_t v_binderInfo_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; 
v_binderName_2191_ = lean_ctor_get(v_e_2184_, 0);
lean_inc(v_binderName_2191_);
v_binderType_2192_ = lean_ctor_get(v_e_2184_, 1);
lean_inc_ref(v_binderType_2192_);
v_body_2193_ = lean_ctor_get(v_e_2184_, 2);
lean_inc_ref(v_body_2193_);
v_binderInfo_2194_ = lean_ctor_get_uint8(v_e_2184_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2184_, 3);
v___x_2195_ = lean_expr_instantiate_rev(v_binderType_2192_, v_fvars_2183_);
lean_dec_ref(v_binderType_2192_);
lean_inc_ref(v_post_2179_);
lean_inc_ref(v_pre_2178_);
v___x_2196_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2178_, v_post_2179_, v_usedLetOnly_2180_, v_skipConstInApp_2181_, v_skipInstances_2182_, v___x_2195_, v_a_2185_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
if (lean_obj_tag(v___x_2196_) == 0)
{
lean_object* v_a_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___f_2201_; uint8_t v___x_2202_; lean_object* v___x_2203_; 
v_a_2197_ = lean_ctor_get(v___x_2196_, 0);
lean_inc(v_a_2197_);
lean_dec_ref_known(v___x_2196_, 1);
v___x_2198_ = lean_box(v_usedLetOnly_2180_);
v___x_2199_ = lean_box(v_skipConstInApp_2181_);
v___x_2200_ = lean_box(v_skipInstances_2182_);
v___f_2201_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0___boxed), 14, 7);
lean_closure_set(v___f_2201_, 0, v_fvars_2183_);
lean_closure_set(v___f_2201_, 1, v_pre_2178_);
lean_closure_set(v___f_2201_, 2, v_post_2179_);
lean_closure_set(v___f_2201_, 3, v___x_2198_);
lean_closure_set(v___f_2201_, 4, v___x_2199_);
lean_closure_set(v___f_2201_, 5, v___x_2200_);
lean_closure_set(v___f_2201_, 6, v_body_2193_);
v___x_2202_ = 0;
v___x_2203_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(v_binderName_2191_, v_binderInfo_2194_, v_a_2197_, v___f_2201_, v___x_2202_, v_a_2185_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
return v___x_2203_;
}
else
{
lean_dec_ref(v_body_2193_);
lean_dec(v_binderName_2191_);
lean_dec_ref(v_fvars_2183_);
lean_dec_ref(v_post_2179_);
lean_dec_ref(v_pre_2178_);
return v___x_2196_;
}
}
else
{
lean_object* v___x_2204_; lean_object* v___x_2205_; 
v___x_2204_ = lean_expr_instantiate_rev(v_e_2184_, v_fvars_2183_);
lean_dec_ref(v_e_2184_);
lean_inc_ref(v_post_2179_);
lean_inc_ref(v_pre_2178_);
v___x_2205_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2178_, v_post_2179_, v_usedLetOnly_2180_, v_skipConstInApp_2181_, v_skipInstances_2182_, v___x_2204_, v_a_2185_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
if (lean_obj_tag(v___x_2205_) == 0)
{
lean_object* v_a_2206_; uint8_t v___x_2207_; uint8_t v___x_2208_; uint8_t v___x_2209_; lean_object* v___x_2210_; 
v_a_2206_ = lean_ctor_get(v___x_2205_, 0);
lean_inc(v_a_2206_);
lean_dec_ref_known(v___x_2205_, 1);
v___x_2207_ = 0;
v___x_2208_ = 1;
v___x_2209_ = 1;
v___x_2210_ = l_Lean_Meta_mkForallFVars(v_fvars_2183_, v_a_2206_, v___x_2207_, v_usedLetOnly_2180_, v___x_2208_, v___x_2209_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
lean_dec_ref(v_fvars_2183_);
if (lean_obj_tag(v___x_2210_) == 0)
{
lean_object* v_a_2211_; lean_object* v___x_2212_; 
v_a_2211_ = lean_ctor_get(v___x_2210_, 0);
lean_inc(v_a_2211_);
lean_dec_ref_known(v___x_2210_, 1);
v___x_2212_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2178_, v_post_2179_, v_usedLetOnly_2180_, v_skipConstInApp_2181_, v_skipInstances_2182_, v_a_2211_, v_a_2185_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
return v___x_2212_;
}
else
{
lean_dec_ref(v_post_2179_);
lean_dec_ref(v_pre_2178_);
return v___x_2210_;
}
}
else
{
lean_dec_ref(v_fvars_2183_);
lean_dec_ref(v_post_2179_);
lean_dec_ref(v_pre_2178_);
return v___x_2205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___lam__0(lean_object* v_fvars_2213_, lean_object* v_pre_2214_, lean_object* v_post_2215_, uint8_t v_usedLetOnly_2216_, uint8_t v_skipConstInApp_2217_, uint8_t v_skipInstances_2218_, lean_object* v_body_2219_, lean_object* v_x_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_){
_start:
{
lean_object* v___x_2227_; lean_object* v___x_2228_; 
v___x_2227_ = lean_array_push(v_fvars_2213_, v_x_2220_);
v___x_2228_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5(v_pre_2214_, v_post_2215_, v_usedLetOnly_2216_, v_skipConstInApp_2217_, v_skipInstances_2218_, v___x_2227_, v_body_2219_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_, v___y_2225_);
return v___x_2228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2___boxed(lean_object* v_pre_2229_, lean_object* v_post_2230_, lean_object* v_usedLetOnly_2231_, lean_object* v_skipConstInApp_2232_, lean_object* v_skipInstances_2233_, lean_object* v_e_2234_, lean_object* v_a_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_){
_start:
{
uint8_t v_usedLetOnly_boxed_2241_; uint8_t v_skipConstInApp_boxed_2242_; uint8_t v_skipInstances_boxed_2243_; lean_object* v_res_2244_; 
v_usedLetOnly_boxed_2241_ = lean_unbox(v_usedLetOnly_2231_);
v_skipConstInApp_boxed_2242_ = lean_unbox(v_skipConstInApp_2232_);
v_skipInstances_boxed_2243_ = lean_unbox(v_skipInstances_2233_);
v_res_2244_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__2(v_pre_2229_, v_post_2230_, v_usedLetOnly_boxed_2241_, v_skipConstInApp_boxed_2242_, v_skipInstances_boxed_2243_, v_e_2234_, v_a_2235_, v___y_2236_, v___y_2237_, v___y_2238_, v___y_2239_);
lean_dec(v___y_2239_);
lean_dec_ref(v___y_2238_);
lean_dec(v___y_2237_);
lean_dec_ref(v___y_2236_);
lean_dec(v_a_2235_);
return v_res_2244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1___boxed(lean_object* v_pre_2245_, lean_object* v_post_2246_, lean_object* v_usedLetOnly_2247_, lean_object* v_skipConstInApp_2248_, lean_object* v_skipInstances_2249_, lean_object* v_sz_2250_, lean_object* v_i_2251_, lean_object* v_bs_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_){
_start:
{
uint8_t v_usedLetOnly_boxed_2259_; uint8_t v_skipConstInApp_boxed_2260_; uint8_t v_skipInstances_boxed_2261_; size_t v_sz_boxed_2262_; size_t v_i_boxed_2263_; lean_object* v_res_2264_; 
v_usedLetOnly_boxed_2259_ = lean_unbox(v_usedLetOnly_2247_);
v_skipConstInApp_boxed_2260_ = lean_unbox(v_skipConstInApp_2248_);
v_skipInstances_boxed_2261_ = lean_unbox(v_skipInstances_2249_);
v_sz_boxed_2262_ = lean_unbox_usize(v_sz_2250_);
lean_dec(v_sz_2250_);
v_i_boxed_2263_ = lean_unbox_usize(v_i_2251_);
lean_dec(v_i_2251_);
v_res_2264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__1(v_pre_2245_, v_post_2246_, v_usedLetOnly_boxed_2259_, v_skipConstInApp_boxed_2260_, v_skipInstances_boxed_2261_, v_sz_boxed_2262_, v_i_boxed_2263_, v_bs_2252_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
lean_dec(v___y_2255_);
lean_dec_ref(v___y_2254_);
lean_dec(v___y_2253_);
return v_res_2264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___boxed(lean_object* v_pre_2265_, lean_object* v_post_2266_, lean_object* v_usedLetOnly_2267_, lean_object* v_skipConstInApp_2268_, lean_object* v_skipInstances_2269_, lean_object* v_e_2270_, lean_object* v_a_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_){
_start:
{
uint8_t v_usedLetOnly_boxed_2277_; uint8_t v_skipConstInApp_boxed_2278_; uint8_t v_skipInstances_boxed_2279_; lean_object* v_res_2280_; 
v_usedLetOnly_boxed_2277_ = lean_unbox(v_usedLetOnly_2267_);
v_skipConstInApp_boxed_2278_ = lean_unbox(v_skipConstInApp_2268_);
v_skipInstances_boxed_2279_ = lean_unbox(v_skipInstances_2269_);
v_res_2280_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2265_, v_post_2266_, v_usedLetOnly_boxed_2277_, v_skipConstInApp_boxed_2278_, v_skipInstances_boxed_2279_, v_e_2270_, v_a_2271_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_);
lean_dec(v___y_2275_);
lean_dec_ref(v___y_2274_);
lean_dec(v___y_2273_);
lean_dec_ref(v___y_2272_);
lean_dec(v_a_2271_);
return v_res_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5___boxed(lean_object* v_pre_2281_, lean_object* v_post_2282_, lean_object* v_usedLetOnly_2283_, lean_object* v_skipConstInApp_2284_, lean_object* v_skipInstances_2285_, lean_object* v_fvars_2286_, lean_object* v_e_2287_, lean_object* v_a_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_){
_start:
{
uint8_t v_usedLetOnly_boxed_2294_; uint8_t v_skipConstInApp_boxed_2295_; uint8_t v_skipInstances_boxed_2296_; lean_object* v_res_2297_; 
v_usedLetOnly_boxed_2294_ = lean_unbox(v_usedLetOnly_2283_);
v_skipConstInApp_boxed_2295_ = lean_unbox(v_skipConstInApp_2284_);
v_skipInstances_boxed_2296_ = lean_unbox(v_skipInstances_2285_);
v_res_2297_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5(v_pre_2281_, v_post_2282_, v_usedLetOnly_boxed_2294_, v_skipConstInApp_boxed_2295_, v_skipInstances_boxed_2296_, v_fvars_2286_, v_e_2287_, v_a_2288_, v___y_2289_, v___y_2290_, v___y_2291_, v___y_2292_);
lean_dec(v___y_2292_);
lean_dec_ref(v___y_2291_);
lean_dec(v___y_2290_);
lean_dec_ref(v___y_2289_);
lean_dec(v_a_2288_);
return v_res_2297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6___boxed(lean_object* v_pre_2298_, lean_object* v_post_2299_, lean_object* v_usedLetOnly_2300_, lean_object* v_skipConstInApp_2301_, lean_object* v_skipInstances_2302_, lean_object* v_fvars_2303_, lean_object* v_e_2304_, lean_object* v_a_2305_, lean_object* v___y_2306_, lean_object* v___y_2307_, lean_object* v___y_2308_, lean_object* v___y_2309_, lean_object* v___y_2310_){
_start:
{
uint8_t v_usedLetOnly_boxed_2311_; uint8_t v_skipConstInApp_boxed_2312_; uint8_t v_skipInstances_boxed_2313_; lean_object* v_res_2314_; 
v_usedLetOnly_boxed_2311_ = lean_unbox(v_usedLetOnly_2300_);
v_skipConstInApp_boxed_2312_ = lean_unbox(v_skipConstInApp_2301_);
v_skipInstances_boxed_2313_ = lean_unbox(v_skipInstances_2302_);
v_res_2314_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__6(v_pre_2298_, v_post_2299_, v_usedLetOnly_boxed_2311_, v_skipConstInApp_boxed_2312_, v_skipInstances_boxed_2313_, v_fvars_2303_, v_e_2304_, v_a_2305_, v___y_2306_, v___y_2307_, v___y_2308_, v___y_2309_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2306_);
lean_dec(v_a_2305_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7___boxed(lean_object* v_pre_2315_, lean_object* v_post_2316_, lean_object* v_usedLetOnly_2317_, lean_object* v_skipConstInApp_2318_, lean_object* v_skipInstances_2319_, lean_object* v_fvars_2320_, lean_object* v_e_2321_, lean_object* v_a_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_, lean_object* v___y_2325_, lean_object* v___y_2326_, lean_object* v___y_2327_){
_start:
{
uint8_t v_usedLetOnly_boxed_2328_; uint8_t v_skipConstInApp_boxed_2329_; uint8_t v_skipInstances_boxed_2330_; lean_object* v_res_2331_; 
v_usedLetOnly_boxed_2328_ = lean_unbox(v_usedLetOnly_2317_);
v_skipConstInApp_boxed_2329_ = lean_unbox(v_skipConstInApp_2318_);
v_skipInstances_boxed_2330_ = lean_unbox(v_skipInstances_2319_);
v_res_2331_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7(v_pre_2315_, v_post_2316_, v_usedLetOnly_boxed_2328_, v_skipConstInApp_boxed_2329_, v_skipInstances_boxed_2330_, v_fvars_2320_, v_e_2321_, v_a_2322_, v___y_2323_, v___y_2324_, v___y_2325_, v___y_2326_);
lean_dec(v___y_2326_);
lean_dec_ref(v___y_2325_);
lean_dec(v___y_2324_);
lean_dec_ref(v___y_2323_);
lean_dec(v_a_2322_);
return v_res_2331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_upperBound_2332_, lean_object* v___x_2333_, lean_object* v_pre_2334_, lean_object* v_post_2335_, lean_object* v_usedLetOnly_2336_, lean_object* v_skipConstInApp_2337_, lean_object* v_skipInstances_2338_, lean_object* v_a_2339_, lean_object* v_b_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_){
_start:
{
uint8_t v_usedLetOnly_boxed_2347_; uint8_t v_skipConstInApp_boxed_2348_; uint8_t v_skipInstances_boxed_2349_; lean_object* v_res_2350_; 
v_usedLetOnly_boxed_2347_ = lean_unbox(v_usedLetOnly_2336_);
v_skipConstInApp_boxed_2348_ = lean_unbox(v_skipConstInApp_2337_);
v_skipInstances_boxed_2349_ = lean_unbox(v_skipInstances_2338_);
v_res_2350_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg(v_upperBound_2332_, v___x_2333_, v_pre_2334_, v_post_2335_, v_usedLetOnly_boxed_2347_, v_skipConstInApp_boxed_2348_, v_skipInstances_boxed_2349_, v_a_2339_, v_b_2340_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_);
lean_dec(v___y_2345_);
lean_dec_ref(v___y_2344_);
lean_dec(v___y_2343_);
lean_dec_ref(v___y_2342_);
lean_dec(v___y_2341_);
lean_dec_ref(v___x_2333_);
lean_dec(v_upperBound_2332_);
return v_res_2350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8___boxed(lean_object* v_skipInstances_2351_, lean_object* v_pre_2352_, lean_object* v_post_2353_, lean_object* v_usedLetOnly_2354_, lean_object* v_skipConstInApp_2355_, lean_object* v_x_2356_, lean_object* v_x_2357_, lean_object* v_x_2358_, lean_object* v___y_2359_, lean_object* v___y_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_){
_start:
{
uint8_t v_skipInstances_boxed_2365_; uint8_t v_usedLetOnly_boxed_2366_; uint8_t v_skipConstInApp_boxed_2367_; lean_object* v_res_2368_; 
v_skipInstances_boxed_2365_ = lean_unbox(v_skipInstances_2351_);
v_usedLetOnly_boxed_2366_ = lean_unbox(v_usedLetOnly_2354_);
v_skipConstInApp_boxed_2367_ = lean_unbox(v_skipConstInApp_2355_);
v_res_2368_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__8(v_skipInstances_boxed_2365_, v_pre_2352_, v_post_2353_, v_usedLetOnly_boxed_2366_, v_skipConstInApp_boxed_2367_, v_x_2356_, v_x_2357_, v_x_2358_, v___y_2359_, v___y_2360_, v___y_2361_, v___y_2362_, v___y_2363_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2362_);
lean_dec(v___y_2361_);
lean_dec_ref(v___y_2360_);
lean_dec(v___y_2359_);
return v_res_2368_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; 
v___x_2369_ = lean_box(0);
v___x_2370_ = lean_unsigned_to_nat(16u);
v___x_2371_ = lean_mk_array(v___x_2370_, v___x_2369_);
return v___x_2371_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; 
v___x_2372_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0, &lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0_once, _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__0);
v___x_2373_ = lean_unsigned_to_nat(0u);
v___x_2374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___x_2373_);
lean_ctor_set(v___x_2374_, 1, v___x_2372_);
return v___x_2374_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2(void){
_start:
{
lean_object* v___x_2375_; lean_object* v___x_2376_; 
v___x_2375_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1, &lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1_once, _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__1);
v___x_2376_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_2376_, 0, lean_box(0));
lean_closure_set(v___x_2376_, 1, lean_box(0));
lean_closure_set(v___x_2376_, 2, v___x_2375_);
return v___x_2376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0(lean_object* v_input_2377_, lean_object* v_pre_2378_, lean_object* v_post_2379_, uint8_t v_usedLetOnly_2380_, uint8_t v_skipConstInApp_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_){
_start:
{
lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v_a_2389_; uint8_t v___x_2390_; lean_object* v___x_2391_; 
v___x_2387_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2, &lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2_once, _init_lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___closed__2);
v___x_2388_ = lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0(lean_box(0), v___x_2387_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
v_a_2389_ = lean_ctor_get(v___x_2388_, 0);
lean_inc(v_a_2389_);
lean_dec_ref(v___x_2388_);
v___x_2390_ = 0;
v___x_2391_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0(v_pre_2378_, v_post_2379_, v_usedLetOnly_2380_, v_skipConstInApp_2381_, v___x_2390_, v_input_2377_, v_a_2389_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
if (lean_obj_tag(v___x_2391_) == 0)
{
lean_object* v_a_2392_; lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2396_; uint8_t v_isShared_2397_; uint8_t v_isSharedCheck_2401_; 
v_a_2392_ = lean_ctor_get(v___x_2391_, 0);
lean_inc(v_a_2392_);
lean_dec_ref_known(v___x_2391_, 1);
v___x_2393_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2393_, 0, lean_box(0));
lean_closure_set(v___x_2393_, 1, lean_box(0));
lean_closure_set(v___x_2393_, 2, v_a_2389_);
v___x_2394_ = lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___lam__0(lean_box(0), v___x_2393_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
v_isSharedCheck_2401_ = !lean_is_exclusive(v___x_2394_);
if (v_isSharedCheck_2401_ == 0)
{
lean_object* v_unused_2402_; 
v_unused_2402_ = lean_ctor_get(v___x_2394_, 0);
lean_dec(v_unused_2402_);
v___x_2396_ = v___x_2394_;
v_isShared_2397_ = v_isSharedCheck_2401_;
goto v_resetjp_2395_;
}
else
{
lean_dec(v___x_2394_);
v___x_2396_ = lean_box(0);
v_isShared_2397_ = v_isSharedCheck_2401_;
goto v_resetjp_2395_;
}
v_resetjp_2395_:
{
lean_object* v___x_2399_; 
if (v_isShared_2397_ == 0)
{
lean_ctor_set(v___x_2396_, 0, v_a_2392_);
v___x_2399_ = v___x_2396_;
goto v_reusejp_2398_;
}
else
{
lean_object* v_reuseFailAlloc_2400_; 
v_reuseFailAlloc_2400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2400_, 0, v_a_2392_);
v___x_2399_ = v_reuseFailAlloc_2400_;
goto v_reusejp_2398_;
}
v_reusejp_2398_:
{
return v___x_2399_;
}
}
}
else
{
lean_dec(v_a_2389_);
return v___x_2391_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0___boxed(lean_object* v_input_2403_, lean_object* v_pre_2404_, lean_object* v_post_2405_, lean_object* v_usedLetOnly_2406_, lean_object* v_skipConstInApp_2407_, lean_object* v___y_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_){
_start:
{
uint8_t v_usedLetOnly_boxed_2413_; uint8_t v_skipConstInApp_boxed_2414_; lean_object* v_res_2415_; 
v_usedLetOnly_boxed_2413_ = lean_unbox(v_usedLetOnly_2406_);
v_skipConstInApp_boxed_2414_ = lean_unbox(v_skipConstInApp_2407_);
v_res_2415_ = lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0(v_input_2403_, v_pre_2404_, v_post_2405_, v_usedLetOnly_boxed_2413_, v_skipConstInApp_boxed_2414_, v___y_2408_, v___y_2409_, v___y_2410_, v___y_2411_);
lean_dec(v___y_2411_);
lean_dec_ref(v___y_2410_);
lean_dec(v___y_2409_);
lean_dec_ref(v___y_2408_);
return v_res_2415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs(lean_object* v_e_2418_, lean_object* v_a_2419_, lean_object* v_a_2420_, lean_object* v_a_2421_, lean_object* v_a_2422_){
_start:
{
lean_object* v___f_2424_; lean_object* v___f_2425_; uint8_t v___x_2426_; uint8_t v___x_2427_; lean_object* v___x_2428_; 
v___f_2424_ = ((lean_object*)(lp_mathlib_Lean_Expr_eraseProofs___closed__0));
v___f_2425_ = ((lean_object*)(lp_mathlib_Lean_Expr_eraseProofs___closed__1));
v___x_2426_ = 0;
v___x_2427_ = 1;
v___x_2428_ = lp_mathlib_Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0(v_e_2418_, v___f_2424_, v___f_2425_, v___x_2426_, v___x_2427_, v_a_2419_, v_a_2420_, v_a_2421_, v_a_2422_);
return v___x_2428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_eraseProofs___boxed(lean_object* v_e_2429_, lean_object* v_a_2430_, lean_object* v_a_2431_, lean_object* v_a_2432_, lean_object* v_a_2433_, lean_object* v_a_2434_){
_start:
{
lean_object* v_res_2435_; 
v_res_2435_ = lp_mathlib_Lean_Expr_eraseProofs(v_e_2429_, v_a_2430_, v_a_2431_, v_a_2432_, v_a_2433_);
lean_dec(v_a_2433_);
lean_dec_ref(v_a_2432_);
lean_dec(v_a_2431_);
lean_dec_ref(v_a_2430_);
return v_res_2435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3(lean_object* v_upperBound_2436_, lean_object* v___x_2437_, lean_object* v_pre_2438_, lean_object* v_post_2439_, uint8_t v_usedLetOnly_2440_, uint8_t v_skipConstInApp_2441_, uint8_t v_skipInstances_2442_, lean_object* v___x_2443_, lean_object* v_inst_2444_, lean_object* v_R_2445_, lean_object* v_a_2446_, lean_object* v_b_2447_, lean_object* v_c_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_){
_start:
{
lean_object* v___x_2455_; 
v___x_2455_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___redArg(v_upperBound_2436_, v___x_2437_, v_pre_2438_, v_post_2439_, v_usedLetOnly_2440_, v_skipConstInApp_2441_, v_skipInstances_2442_, v_a_2446_, v_b_2447_, v___y_2449_, v___y_2450_, v___y_2451_, v___y_2452_, v___y_2453_);
return v___x_2455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3___boxed(lean_object** _args){
lean_object* v_upperBound_2456_ = _args[0];
lean_object* v___x_2457_ = _args[1];
lean_object* v_pre_2458_ = _args[2];
lean_object* v_post_2459_ = _args[3];
lean_object* v_usedLetOnly_2460_ = _args[4];
lean_object* v_skipConstInApp_2461_ = _args[5];
lean_object* v_skipInstances_2462_ = _args[6];
lean_object* v___x_2463_ = _args[7];
lean_object* v_inst_2464_ = _args[8];
lean_object* v_R_2465_ = _args[9];
lean_object* v_a_2466_ = _args[10];
lean_object* v_b_2467_ = _args[11];
lean_object* v_c_2468_ = _args[12];
lean_object* v___y_2469_ = _args[13];
lean_object* v___y_2470_ = _args[14];
lean_object* v___y_2471_ = _args[15];
lean_object* v___y_2472_ = _args[16];
lean_object* v___y_2473_ = _args[17];
lean_object* v___y_2474_ = _args[18];
_start:
{
uint8_t v_usedLetOnly_boxed_2475_; uint8_t v_skipConstInApp_boxed_2476_; uint8_t v_skipInstances_boxed_2477_; lean_object* v_res_2478_; 
v_usedLetOnly_boxed_2475_ = lean_unbox(v_usedLetOnly_2460_);
v_skipConstInApp_boxed_2476_ = lean_unbox(v_skipConstInApp_2461_);
v_skipInstances_boxed_2477_ = lean_unbox(v_skipInstances_2462_);
v_res_2478_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__3(v_upperBound_2456_, v___x_2457_, v_pre_2458_, v_post_2459_, v_usedLetOnly_boxed_2475_, v_skipConstInApp_boxed_2476_, v_skipInstances_boxed_2477_, v___x_2463_, v_inst_2464_, v_R_2465_, v_a_2466_, v_b_2467_, v_c_2468_, v___y_2469_, v___y_2470_, v___y_2471_, v___y_2472_, v___y_2473_);
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
lean_dec(v___y_2471_);
lean_dec_ref(v___y_2470_);
lean_dec(v___y_2469_);
lean_dec(v___x_2463_);
lean_dec_ref(v___x_2457_);
lean_dec(v_upperBound_2456_);
return v_res_2478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4(lean_object* v_00_u03b2_2479_, lean_object* v_m_2480_, lean_object* v_a_2481_){
_start:
{
lean_object* v___x_2482_; 
v___x_2482_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___redArg(v_m_2480_, v_a_2481_);
return v___x_2482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4___boxed(lean_object* v_00_u03b2_2483_, lean_object* v_m_2484_, lean_object* v_a_2485_){
_start:
{
lean_object* v_res_2486_; 
v_res_2486_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4(v_00_u03b2_2483_, v_m_2484_, v_a_2485_);
lean_dec_ref(v_a_2485_);
lean_dec_ref(v_m_2484_);
return v_res_2486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7(lean_object* v_00_u03b1_2487_, lean_object* v_name_2488_, uint8_t v_bi_2489_, lean_object* v_type_2490_, lean_object* v_k_2491_, uint8_t v_kind_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_){
_start:
{
lean_object* v___x_2499_; 
v___x_2499_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___redArg(v_name_2488_, v_bi_2489_, v_type_2490_, v_k_2491_, v_kind_2492_, v___y_2493_, v___y_2494_, v___y_2495_, v___y_2496_, v___y_2497_);
return v___x_2499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7___boxed(lean_object* v_00_u03b1_2500_, lean_object* v_name_2501_, lean_object* v_bi_2502_, lean_object* v_type_2503_, lean_object* v_k_2504_, lean_object* v_kind_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_){
_start:
{
uint8_t v_bi_boxed_2512_; uint8_t v_kind_boxed_2513_; lean_object* v_res_2514_; 
v_bi_boxed_2512_ = lean_unbox(v_bi_2502_);
v_kind_boxed_2513_ = lean_unbox(v_kind_2505_);
v_res_2514_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__5_spec__7(v_00_u03b1_2500_, v_name_2501_, v_bi_boxed_2512_, v_type_2503_, v_k_2504_, v_kind_boxed_2513_, v___y_2506_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_);
lean_dec(v___y_2510_);
lean_dec_ref(v___y_2509_);
lean_dec(v___y_2508_);
lean_dec_ref(v___y_2507_);
lean_dec(v___y_2506_);
return v_res_2514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10(lean_object* v_00_u03b1_2515_, lean_object* v_name_2516_, lean_object* v_type_2517_, lean_object* v_val_2518_, lean_object* v_k_2519_, uint8_t v_nondep_2520_, uint8_t v_kind_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_){
_start:
{
lean_object* v___x_2528_; 
v___x_2528_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___redArg(v_name_2516_, v_type_2517_, v_val_2518_, v_k_2519_, v_nondep_2520_, v_kind_2521_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_, v___y_2526_);
return v___x_2528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10___boxed(lean_object* v_00_u03b1_2529_, lean_object* v_name_2530_, lean_object* v_type_2531_, lean_object* v_val_2532_, lean_object* v_k_2533_, lean_object* v_nondep_2534_, lean_object* v_kind_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_){
_start:
{
uint8_t v_nondep_boxed_2542_; uint8_t v_kind_boxed_2543_; lean_object* v_res_2544_; 
v_nondep_boxed_2542_ = lean_unbox(v_nondep_2534_);
v_kind_boxed_2543_ = lean_unbox(v_kind_2535_);
v_res_2544_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__7_spec__10(v_00_u03b1_2529_, v_name_2530_, v_type_2531_, v_val_2532_, v_k_2533_, v_nondep_boxed_2542_, v_kind_boxed_2543_, v___y_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_);
lean_dec(v___y_2540_);
lean_dec_ref(v___y_2539_);
lean_dec(v___y_2538_);
lean_dec_ref(v___y_2537_);
lean_dec(v___y_2536_);
return v_res_2544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13(lean_object* v_00_u03b1_2545_, lean_object* v_ref_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_){
_start:
{
lean_object* v___x_2552_; 
v___x_2552_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___redArg(v_ref_2546_);
return v___x_2552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13___boxed(lean_object* v_00_u03b1_2553_, lean_object* v_ref_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_, lean_object* v___y_2559_){
_start:
{
lean_object* v_res_2560_; 
v_res_2560_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9_spec__13(v_00_u03b1_2553_, v_ref_2554_, v___y_2555_, v___y_2556_, v___y_2557_, v___y_2558_);
lean_dec(v___y_2558_);
lean_dec_ref(v___y_2557_);
lean_dec(v___y_2556_);
lean_dec_ref(v___y_2555_);
return v_res_2560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9(lean_object* v_00_u03b1_2561_, lean_object* v_x_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
lean_object* v___x_2569_; 
v___x_2569_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___redArg(v_x_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_, v___y_2567_);
return v___x_2569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9___boxed(lean_object* v_00_u03b1_2570_, lean_object* v_x_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_){
_start:
{
lean_object* v_res_2578_; 
v_res_2578_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__9(v_00_u03b1_2570_, v_x_2571_, v___y_2572_, v___y_2573_, v___y_2574_, v___y_2575_, v___y_2576_);
lean_dec(v___y_2576_);
lean_dec_ref(v___y_2575_);
lean_dec(v___y_2574_);
lean_dec_ref(v___y_2573_);
lean_dec(v___y_2572_);
return v_res_2578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10(lean_object* v_00_u03b2_2579_, lean_object* v_m_2580_, lean_object* v_a_2581_, lean_object* v_b_2582_){
_start:
{
lean_object* v___x_2583_; 
v___x_2583_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10___redArg(v_m_2580_, v_a_2581_, v_b_2582_);
return v___x_2583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5(lean_object* v_00_u03b2_2584_, lean_object* v_a_2585_, lean_object* v_x_2586_){
_start:
{
lean_object* v___x_2587_; 
v___x_2587_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___redArg(v_a_2585_, v_x_2586_);
return v___x_2587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5___boxed(lean_object* v_00_u03b2_2588_, lean_object* v_a_2589_, lean_object* v_x_2590_){
_start:
{
lean_object* v_res_2591_; 
v_res_2591_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__4_spec__5(v_00_u03b2_2588_, v_a_2589_, v_x_2590_);
lean_dec(v_x_2590_);
lean_dec_ref(v_a_2589_);
return v_res_2591_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15(lean_object* v_00_u03b2_2592_, lean_object* v_a_2593_, lean_object* v_x_2594_){
_start:
{
uint8_t v___x_2595_; 
v___x_2595_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___redArg(v_a_2593_, v_x_2594_);
return v___x_2595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15___boxed(lean_object* v_00_u03b2_2596_, lean_object* v_a_2597_, lean_object* v_x_2598_){
_start:
{
uint8_t v_res_2599_; lean_object* v_r_2600_; 
v_res_2599_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__15(v_00_u03b2_2596_, v_a_2597_, v_x_2598_);
lean_dec(v_x_2598_);
lean_dec_ref(v_a_2597_);
v_r_2600_ = lean_box(v_res_2599_);
return v_r_2600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16(lean_object* v_00_u03b2_2601_, lean_object* v_data_2602_){
_start:
{
lean_object* v___x_2603_; 
v___x_2603_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16___redArg(v_data_2602_);
return v___x_2603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17(lean_object* v_00_u03b2_2604_, lean_object* v_a_2605_, lean_object* v_b_2606_, lean_object* v_x_2607_){
_start:
{
lean_object* v___x_2608_; 
v___x_2608_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__17___redArg(v_a_2605_, v_b_2606_, v_x_2607_);
return v___x_2608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17(lean_object* v_00_u03b2_2609_, lean_object* v_i_2610_, lean_object* v_source_2611_, lean_object* v_target_2612_){
_start:
{
lean_object* v___x_2613_; 
v___x_2613_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17___redArg(v_i_2610_, v_source_2611_, v_target_2612_);
return v___x_2613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18(lean_object* v_00_u03b2_2614_, lean_object* v_x_2615_, lean_object* v_x_2616_){
_start:
{
lean_object* v___x_2617_; 
v___x_2617_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0_spec__10_spec__16_spec__17_spec__18___redArg(v_x_2615_, v_x_2616_);
return v___x_2617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_type_x3f(lean_object* v_x_2618_){
_start:
{
if (lean_obj_tag(v_x_2618_) == 3)
{
lean_object* v_u_2619_; lean_object* v___x_2620_; 
v_u_2619_ = lean_ctor_get(v_x_2618_, 0);
v___x_2620_ = l_Lean_Level_dec(v_u_2619_);
return v___x_2620_;
}
else
{
lean_object* v___x_2621_; 
v___x_2621_ = lean_box(0);
return v___x_2621_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_type_x3f___boxed(lean_object* v_x_2622_){
_start:
{
lean_object* v_res_2623_; 
v_res_2623_ = lp_mathlib_Lean_Expr_type_x3f(v_x_2622_);
lean_dec_ref(v_x_2622_);
return v_res_2623_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConstP(lean_object* v_p_2624_, lean_object* v_type_2625_){
_start:
{
lean_object* v___x_2626_; lean_object* v___x_2627_; 
v___x_2626_ = l_Lean_Expr_cleanupAnnotations(v_type_2625_);
v___x_2627_ = l_Lean_Expr_getAppFn_x27(v___x_2626_);
lean_dec_ref(v___x_2626_);
switch(lean_obj_tag(v___x_2627_))
{
case 4:
{
lean_object* v_declName_2628_; lean_object* v___x_2629_; uint8_t v___x_2630_; 
v_declName_2628_ = lean_ctor_get(v___x_2627_, 0);
lean_inc(v_declName_2628_);
lean_dec_ref_known(v___x_2627_, 2);
v___x_2629_ = lean_apply_1(v_p_2624_, v_declName_2628_);
v___x_2630_ = lean_unbox(v___x_2629_);
return v___x_2630_;
}
case 7:
{
lean_object* v_body_2631_; 
v_body_2631_ = lean_ctor_get(v___x_2627_, 2);
lean_inc_ref(v_body_2631_);
lean_dec_ref_known(v___x_2627_, 3);
v_type_2625_ = v_body_2631_;
goto _start;
}
default: 
{
uint8_t v___x_2633_; 
lean_dec_ref(v___x_2627_);
lean_dec_ref(v_p_2624_);
v___x_2633_ = 0;
return v___x_2633_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConstP___boxed(lean_object* v_p_2634_, lean_object* v_type_2635_){
_start:
{
uint8_t v_res_2636_; lean_object* v_r_2637_; 
v_res_2636_ = lp_mathlib_Lean_Expr_isAppOrForallOfConstP(v_p_2634_, v_type_2635_);
v_r_2637_ = lean_box(v_res_2636_);
return v_r_2637_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0(lean_object* v_declName_2638_, lean_object* v_x_2639_){
_start:
{
uint8_t v___x_2640_; 
v___x_2640_ = lean_name_eq(v_x_2639_, v_declName_2638_);
return v___x_2640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0___boxed(lean_object* v_declName_2641_, lean_object* v_x_2642_){
_start:
{
uint8_t v_res_2643_; lean_object* v_r_2644_; 
v_res_2643_ = lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0(v_declName_2641_, v_x_2642_);
lean_dec(v_x_2642_);
lean_dec(v_declName_2641_);
v_r_2644_ = lean_box(v_res_2643_);
return v_r_2644_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isAppOrForallOfConst(lean_object* v_declName_2645_, lean_object* v_type_2646_){
_start:
{
lean_object* v___f_2647_; uint8_t v___x_2648_; 
v___f_2647_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_isAppOrForallOfConst___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2647_, 0, v_declName_2645_);
v___x_2648_ = lp_mathlib_Lean_Expr_isAppOrForallOfConstP(v___f_2647_, v_type_2646_);
return v___x_2648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isAppOrForallOfConst___boxed(lean_object* v_declName_2649_, lean_object* v_type_2650_){
_start:
{
uint8_t v_res_2651_; lean_object* v_r_2652_; 
v_res_2651_ = lp_mathlib_Lean_Expr_isAppOrForallOfConst(v_declName_2649_, v_type_2650_);
v_r_2652_ = lean_box(v_res_2651_);
return v_r_2652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getUnusedForallInstanceBinderIdxsWhere_go(lean_object* v_p_2653_, lean_object* v_body_2654_, lean_object* v_current_2655_, lean_object* v_acc_2656_){
_start:
{
lean_object* v___x_2657_; 
v___x_2657_ = l_Lean_Expr_cleanupAnnotations(v_body_2654_);
switch(lean_obj_tag(v___x_2657_))
{
case 7:
{
lean_object* v_binderType_2658_; lean_object* v_body_2659_; uint8_t v_binderInfo_2660_; lean_object* v___x_2661_; lean_object* v___x_2662_; uint8_t v___y_2664_; uint8_t v___x_2671_; 
v_binderType_2658_ = lean_ctor_get(v___x_2657_, 1);
lean_inc_ref(v_binderType_2658_);
v_body_2659_ = lean_ctor_get(v___x_2657_, 2);
lean_inc_ref(v_body_2659_);
v_binderInfo_2660_ = lean_ctor_get_uint8(v___x_2657_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v___x_2657_, 3);
v___x_2661_ = lean_unsigned_to_nat(1u);
v___x_2662_ = lean_nat_add(v_current_2655_, v___x_2661_);
v___x_2671_ = l_Lean_BinderInfo_isInstImplicit(v_binderInfo_2660_);
if (v___x_2671_ == 0)
{
lean_dec_ref(v_binderType_2658_);
v___y_2664_ = v___x_2671_;
goto v___jp_2663_;
}
else
{
lean_object* v___x_2672_; uint8_t v___x_2673_; 
lean_inc_ref(v_p_2653_);
v___x_2672_ = lean_apply_1(v_p_2653_, v_binderType_2658_);
v___x_2673_ = lean_unbox(v___x_2672_);
v___y_2664_ = v___x_2673_;
goto v___jp_2663_;
}
v___jp_2663_:
{
if (v___y_2664_ == 0)
{
lean_dec(v_current_2655_);
v_body_2654_ = v_body_2659_;
v_current_2655_ = v___x_2662_;
goto _start;
}
else
{
lean_object* v___x_2666_; uint8_t v___x_2667_; 
v___x_2666_ = lean_unsigned_to_nat(0u);
v___x_2667_ = lean_expr_has_loose_bvar(v_body_2659_, v___x_2666_);
if (v___x_2667_ == 0)
{
lean_object* v___x_2668_; 
v___x_2668_ = lean_array_push(v_acc_2656_, v_current_2655_);
v_body_2654_ = v_body_2659_;
v_current_2655_ = v___x_2662_;
v_acc_2656_ = v___x_2668_;
goto _start;
}
else
{
lean_dec(v_current_2655_);
v_body_2654_ = v_body_2659_;
v_current_2655_ = v___x_2662_;
goto _start;
}
}
}
}
case 8:
{
lean_object* v_body_2674_; 
v_body_2674_ = lean_ctor_get(v___x_2657_, 3);
lean_inc_ref(v_body_2674_);
lean_dec_ref_known(v___x_2657_, 4);
v_body_2654_ = v_body_2674_;
goto _start;
}
default: 
{
lean_dec_ref(v___x_2657_);
lean_dec(v_current_2655_);
lean_dec_ref(v_p_2653_);
return v_acc_2656_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere(lean_object* v_p_2678_, lean_object* v_e_2679_){
_start:
{
lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; 
v___x_2680_ = lean_unsigned_to_nat(0u);
v___x_2681_ = ((lean_object*)(lp_mathlib_Lean_Expr_getUnusedForallInstanceBinderIdxsWhere___closed__0));
v___x_2682_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_getUnusedForallInstanceBinderIdxsWhere_go(v_p_2678_, v_e_2679_, v___x_2680_, v___x_2681_);
return v___x_2682_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_hasInstanceBinderOf(lean_object* v_p_2683_, lean_object* v_e_2684_){
_start:
{
lean_object* v___x_2685_; 
v___x_2685_ = l_Lean_Expr_cleanupAnnotations(v_e_2684_);
switch(lean_obj_tag(v___x_2685_))
{
case 7:
{
lean_object* v_binderType_2686_; lean_object* v_body_2687_; uint8_t v_binderInfo_2688_; uint8_t v___y_2690_; uint8_t v___x_2692_; 
v_binderType_2686_ = lean_ctor_get(v___x_2685_, 1);
lean_inc_ref(v_binderType_2686_);
v_body_2687_ = lean_ctor_get(v___x_2685_, 2);
lean_inc_ref(v_body_2687_);
v_binderInfo_2688_ = lean_ctor_get_uint8(v___x_2685_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v___x_2685_, 3);
v___x_2692_ = l_Lean_BinderInfo_isInstImplicit(v_binderInfo_2688_);
if (v___x_2692_ == 0)
{
lean_dec_ref(v_binderType_2686_);
v___y_2690_ = v___x_2692_;
goto v___jp_2689_;
}
else
{
lean_object* v___x_2693_; uint8_t v___x_2694_; 
lean_inc_ref(v_p_2683_);
v___x_2693_ = lean_apply_1(v_p_2683_, v_binderType_2686_);
v___x_2694_ = lean_unbox(v___x_2693_);
v___y_2690_ = v___x_2694_;
goto v___jp_2689_;
}
v___jp_2689_:
{
if (v___y_2690_ == 0)
{
v_e_2684_ = v_body_2687_;
goto _start;
}
else
{
lean_dec_ref(v_body_2687_);
lean_dec_ref(v_p_2683_);
return v___y_2690_;
}
}
}
case 8:
{
lean_object* v_body_2695_; 
v_body_2695_ = lean_ctor_get(v___x_2685_, 3);
lean_inc_ref(v_body_2695_);
lean_dec_ref_known(v___x_2685_, 4);
v_e_2684_ = v_body_2695_;
goto _start;
}
default: 
{
uint8_t v___x_2697_; 
lean_dec_ref(v___x_2685_);
lean_dec_ref(v_p_2683_);
v___x_2697_ = 0;
return v___x_2697_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_hasInstanceBinderOf___boxed(lean_object* v_p_2698_, lean_object* v_e_2699_){
_start:
{
uint8_t v_res_2700_; lean_object* v_r_2701_; 
v_res_2700_ = lp_mathlib_Lean_Expr_hasInstanceBinderOf(v_p_2698_, v_e_2699_);
v_r_2701_ = lean_box(v_res_2700_);
return v_r_2701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_letDepth(lean_object* v_x_2702_){
_start:
{
if (lean_obj_tag(v_x_2702_) == 8)
{
lean_object* v_body_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; 
v_body_2703_ = lean_ctor_get(v_x_2702_, 3);
v___x_2704_ = lp_mathlib_Lean_Expr_letDepth(v_body_2703_);
v___x_2705_ = lean_unsigned_to_nat(1u);
v___x_2706_ = lean_nat_add(v___x_2704_, v___x_2705_);
lean_dec(v___x_2704_);
return v___x_2706_;
}
else
{
lean_object* v___x_2707_; 
v___x_2707_ = lean_unsigned_to_nat(0u);
return v___x_2707_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_letDepth___boxed(lean_object* v_x_2708_){
_start:
{
lean_object* v_res_2709_; 
v_res_2709_ = lp_mathlib_Lean_Expr_letDepth(v_x_2708_);
lean_dec_ref(v_x_2708_);
return v_res_2709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg(lean_object* v_e_2710_, lean_object* v___y_2711_){
_start:
{
uint8_t v___x_2713_; 
v___x_2713_ = l_Lean_Expr_hasMVar(v_e_2710_);
if (v___x_2713_ == 0)
{
lean_object* v___x_2714_; 
v___x_2714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2714_, 0, v_e_2710_);
return v___x_2714_;
}
else
{
lean_object* v___x_2715_; lean_object* v_mctx_2716_; lean_object* v___x_2717_; lean_object* v_fst_2718_; lean_object* v_snd_2719_; lean_object* v___x_2720_; lean_object* v_cache_2721_; lean_object* v_zetaDeltaFVarIds_2722_; lean_object* v_postponed_2723_; lean_object* v_diag_2724_; lean_object* v___x_2726_; uint8_t v_isShared_2727_; uint8_t v_isSharedCheck_2733_; 
v___x_2715_ = lean_st_ref_get(v___y_2711_);
v_mctx_2716_ = lean_ctor_get(v___x_2715_, 0);
lean_inc_ref(v_mctx_2716_);
lean_dec(v___x_2715_);
v___x_2717_ = l_Lean_instantiateMVarsCore(v_mctx_2716_, v_e_2710_);
v_fst_2718_ = lean_ctor_get(v___x_2717_, 0);
lean_inc(v_fst_2718_);
v_snd_2719_ = lean_ctor_get(v___x_2717_, 1);
lean_inc(v_snd_2719_);
lean_dec_ref(v___x_2717_);
v___x_2720_ = lean_st_ref_take(v___y_2711_);
v_cache_2721_ = lean_ctor_get(v___x_2720_, 1);
v_zetaDeltaFVarIds_2722_ = lean_ctor_get(v___x_2720_, 2);
v_postponed_2723_ = lean_ctor_get(v___x_2720_, 3);
v_diag_2724_ = lean_ctor_get(v___x_2720_, 4);
v_isSharedCheck_2733_ = !lean_is_exclusive(v___x_2720_);
if (v_isSharedCheck_2733_ == 0)
{
lean_object* v_unused_2734_; 
v_unused_2734_ = lean_ctor_get(v___x_2720_, 0);
lean_dec(v_unused_2734_);
v___x_2726_ = v___x_2720_;
v_isShared_2727_ = v_isSharedCheck_2733_;
goto v_resetjp_2725_;
}
else
{
lean_inc(v_diag_2724_);
lean_inc(v_postponed_2723_);
lean_inc(v_zetaDeltaFVarIds_2722_);
lean_inc(v_cache_2721_);
lean_dec(v___x_2720_);
v___x_2726_ = lean_box(0);
v_isShared_2727_ = v_isSharedCheck_2733_;
goto v_resetjp_2725_;
}
v_resetjp_2725_:
{
lean_object* v___x_2729_; 
if (v_isShared_2727_ == 0)
{
lean_ctor_set(v___x_2726_, 0, v_snd_2719_);
v___x_2729_ = v___x_2726_;
goto v_reusejp_2728_;
}
else
{
lean_object* v_reuseFailAlloc_2732_; 
v_reuseFailAlloc_2732_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2732_, 0, v_snd_2719_);
lean_ctor_set(v_reuseFailAlloc_2732_, 1, v_cache_2721_);
lean_ctor_set(v_reuseFailAlloc_2732_, 2, v_zetaDeltaFVarIds_2722_);
lean_ctor_set(v_reuseFailAlloc_2732_, 3, v_postponed_2723_);
lean_ctor_set(v_reuseFailAlloc_2732_, 4, v_diag_2724_);
v___x_2729_ = v_reuseFailAlloc_2732_;
goto v_reusejp_2728_;
}
v_reusejp_2728_:
{
lean_object* v___x_2730_; lean_object* v___x_2731_; 
v___x_2730_ = lean_st_ref_set(v___y_2711_, v___x_2729_);
v___x_2731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2731_, 0, v_fst_2718_);
return v___x_2731_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg___boxed(lean_object* v_e_2735_, lean_object* v___y_2736_, lean_object* v___y_2737_){
_start:
{
lean_object* v_res_2738_; 
v_res_2738_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg(v_e_2735_, v___y_2736_);
lean_dec(v___y_2736_);
return v_res_2738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0(lean_object* v_e_2739_, lean_object* v___y_2740_, lean_object* v___y_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_){
_start:
{
lean_object* v___x_2745_; 
v___x_2745_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg(v_e_2739_, v___y_2741_);
return v___x_2745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___boxed(lean_object* v_e_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_){
_start:
{
lean_object* v_res_2752_; 
v_res_2752_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0(v_e_2746_, v___y_2747_, v___y_2748_, v___y_2749_, v___y_2750_);
lean_dec(v___y_2750_);
lean_dec_ref(v___y_2749_);
lean_dec(v___y_2748_);
lean_dec_ref(v___y_2747_);
return v_res_2752_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1(void){
_start:
{
lean_object* v___x_2754_; lean_object* v___x_2755_; 
v___x_2754_ = ((lean_object*)(lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__0));
v___x_2755_ = l_Lean_stringToMessageData(v___x_2754_);
return v___x_2755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars(lean_object* v_e_2756_, lean_object* v_a_2757_, lean_object* v_a_2758_, lean_object* v_a_2759_, lean_object* v_a_2760_){
_start:
{
lean_object* v___x_2762_; lean_object* v_a_2763_; lean_object* v___x_2765_; uint8_t v_isShared_2766_; uint8_t v_isSharedCheck_2776_; 
v___x_2762_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ensureHasNoMVars_spec__0___redArg(v_e_2756_, v_a_2758_);
v_a_2763_ = lean_ctor_get(v___x_2762_, 0);
v_isSharedCheck_2776_ = !lean_is_exclusive(v___x_2762_);
if (v_isSharedCheck_2776_ == 0)
{
v___x_2765_ = v___x_2762_;
v_isShared_2766_ = v_isSharedCheck_2776_;
goto v_resetjp_2764_;
}
else
{
lean_inc(v_a_2763_);
lean_dec(v___x_2762_);
v___x_2765_ = lean_box(0);
v_isShared_2766_ = v_isSharedCheck_2776_;
goto v_resetjp_2764_;
}
v_resetjp_2764_:
{
uint8_t v___x_2767_; 
v___x_2767_ = l_Lean_Expr_hasExprMVar(v_a_2763_);
if (v___x_2767_ == 0)
{
lean_object* v___x_2768_; lean_object* v___x_2770_; 
lean_dec(v_a_2763_);
v___x_2768_ = lean_box(0);
if (v_isShared_2766_ == 0)
{
lean_ctor_set(v___x_2765_, 0, v___x_2768_);
v___x_2770_ = v___x_2765_;
goto v_reusejp_2769_;
}
else
{
lean_object* v_reuseFailAlloc_2771_; 
v_reuseFailAlloc_2771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2771_, 0, v___x_2768_);
v___x_2770_ = v_reuseFailAlloc_2771_;
goto v_reusejp_2769_;
}
v_reusejp_2769_:
{
return v___x_2770_;
}
}
else
{
lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; 
lean_del_object(v___x_2765_);
v___x_2772_ = lean_obj_once(&lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1, &lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1_once, _init_lp_mathlib_Lean_Expr_ensureHasNoMVars___closed__1);
v___x_2773_ = l_Lean_indentExpr(v_a_2763_);
v___x_2774_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2774_, 0, v___x_2772_);
lean_ctor_set(v___x_2774_, 1, v___x_2773_);
v___x_2775_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_2774_, v_a_2757_, v_a_2758_, v_a_2759_, v_a_2760_);
return v___x_2775_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ensureHasNoMVars___boxed(lean_object* v_e_2777_, lean_object* v_a_2778_, lean_object* v_a_2779_, lean_object* v_a_2780_, lean_object* v_a_2781_, lean_object* v_a_2782_){
_start:
{
lean_object* v_res_2783_; 
v_res_2783_ = lp_mathlib_Lean_Expr_ensureHasNoMVars(v_e_2777_, v_a_2778_, v_a_2779_, v_a_2780_, v_a_2781_);
lean_dec(v_a_2781_);
lean_dec_ref(v_a_2780_);
lean_dec(v_a_2779_);
lean_dec_ref(v_a_2778_);
return v_res_2783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofNat(lean_object* v_00_u03b1_2789_, lean_object* v_n_2790_, lean_object* v_a_2791_, lean_object* v_a_2792_, lean_object* v_a_2793_, lean_object* v_a_2794_){
_start:
{
lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; 
v___x_2796_ = ((lean_object*)(lp_mathlib_Lean_Expr_ofNat___closed__2));
v___x_2797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2797_, 0, v_00_u03b1_2789_);
v___x_2798_ = l_Lean_mkRawNatLit(v_n_2790_);
v___x_2799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2799_, 0, v___x_2798_);
v___x_2800_ = lean_box(0);
v___x_2801_ = lean_unsigned_to_nat(3u);
v___x_2802_ = lean_mk_empty_array_with_capacity(v___x_2801_);
v___x_2803_ = lean_array_push(v___x_2802_, v___x_2797_);
v___x_2804_ = lean_array_push(v___x_2803_, v___x_2799_);
v___x_2805_ = lean_array_push(v___x_2804_, v___x_2800_);
v___x_2806_ = l_Lean_Meta_mkAppOptM(v___x_2796_, v___x_2805_, v_a_2791_, v_a_2792_, v_a_2793_, v_a_2794_);
return v___x_2806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofNat___boxed(lean_object* v_00_u03b1_2807_, lean_object* v_n_2808_, lean_object* v_a_2809_, lean_object* v_a_2810_, lean_object* v_a_2811_, lean_object* v_a_2812_, lean_object* v_a_2813_){
_start:
{
lean_object* v_res_2814_; 
v_res_2814_ = lp_mathlib_Lean_Expr_ofNat(v_00_u03b1_2807_, v_n_2808_, v_a_2809_, v_a_2810_, v_a_2811_, v_a_2812_);
lean_dec(v_a_2812_);
lean_dec_ref(v_a_2811_);
lean_dec(v_a_2810_);
lean_dec_ref(v_a_2809_);
return v_res_2814_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_ofInt___closed__0(void){
_start:
{
lean_object* v_natZero_2815_; lean_object* v_intZero_2816_; 
v_natZero_2815_ = lean_unsigned_to_nat(0u);
v_intZero_2816_ = lean_nat_to_int(v_natZero_2815_);
return v_intZero_2816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofInt(lean_object* v_00_u03b1_2822_, lean_object* v_x_2823_, lean_object* v_a_2824_, lean_object* v_a_2825_, lean_object* v_a_2826_, lean_object* v_a_2827_){
_start:
{
lean_object* v_intZero_2829_; uint8_t v_isNeg_2830_; 
v_intZero_2829_ = lean_obj_once(&lp_mathlib_Lean_Expr_ofInt___closed__0, &lp_mathlib_Lean_Expr_ofInt___closed__0_once, _init_lp_mathlib_Lean_Expr_ofInt___closed__0);
v_isNeg_2830_ = lean_int_dec_lt(v_x_2823_, v_intZero_2829_);
if (v_isNeg_2830_ == 0)
{
lean_object* v_a_2831_; lean_object* v___x_2832_; 
v_a_2831_ = lean_nat_abs(v_x_2823_);
v___x_2832_ = lp_mathlib_Lean_Expr_ofNat(v_00_u03b1_2822_, v_a_2831_, v_a_2824_, v_a_2825_, v_a_2826_, v_a_2827_);
return v___x_2832_;
}
else
{
lean_object* v_abs_2833_; lean_object* v_one_2834_; lean_object* v_a_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; 
v_abs_2833_ = lean_nat_abs(v_x_2823_);
v_one_2834_ = lean_unsigned_to_nat(1u);
v_a_2835_ = lean_nat_sub(v_abs_2833_, v_one_2834_);
lean_dec(v_abs_2833_);
v___x_2836_ = lean_nat_add(v_a_2835_, v_one_2834_);
lean_dec(v_a_2835_);
v___x_2837_ = lp_mathlib_Lean_Expr_ofNat(v_00_u03b1_2822_, v___x_2836_, v_a_2824_, v_a_2825_, v_a_2826_, v_a_2827_);
if (lean_obj_tag(v___x_2837_) == 0)
{
lean_object* v_a_2838_; lean_object* v___x_2839_; lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; 
v_a_2838_ = lean_ctor_get(v___x_2837_, 0);
lean_inc(v_a_2838_);
lean_dec_ref_known(v___x_2837_, 1);
v___x_2839_ = ((lean_object*)(lp_mathlib_Lean_Expr_ofInt___closed__3));
v___x_2840_ = lean_mk_empty_array_with_capacity(v_one_2834_);
v___x_2841_ = lean_array_push(v___x_2840_, v_a_2838_);
v___x_2842_ = l_Lean_Meta_mkAppM(v___x_2839_, v___x_2841_, v_a_2824_, v_a_2825_, v_a_2826_, v_a_2827_);
return v___x_2842_;
}
else
{
return v___x_2837_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ofInt___boxed(lean_object* v_00_u03b1_2843_, lean_object* v_x_2844_, lean_object* v_a_2845_, lean_object* v_a_2846_, lean_object* v_a_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_){
_start:
{
lean_object* v_res_2850_; 
v_res_2850_ = lp_mathlib_Lean_Expr_ofInt(v_00_u03b1_2843_, v_x_2844_, v_a_2845_, v_a_2846_, v_a_2847_, v_a_2848_);
lean_dec(v_a_2848_);
lean_dec_ref(v_a_2847_);
lean_dec(v_a_2846_);
lean_dec_ref(v_a_2845_);
lean_dec(v_x_2844_);
return v_res_2850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_numeral_x3f(lean_object* v_e_2862_){
_start:
{
uint8_t v___y_2864_; lean_object* v___x_2867_; 
lean_inc_ref(v_e_2862_);
v___x_2867_ = l_Lean_Expr_rawNatLit_x3f(v_e_2862_);
if (lean_obj_tag(v___x_2867_) == 1)
{
lean_dec_ref(v_e_2862_);
return v___x_2867_;
}
else
{
lean_object* v_e_2868_; lean_object* v_f_2869_; uint8_t v___x_2870_; 
lean_dec(v___x_2867_);
v_e_2868_ = l_Lean_Expr_consumeMData(v_e_2862_);
lean_dec_ref(v_e_2862_);
v_f_2869_ = l_Lean_Expr_getAppFn(v_e_2868_);
v___x_2870_ = l_Lean_Expr_isConst(v_f_2869_);
if (v___x_2870_ == 0)
{
lean_object* v___x_2871_; 
lean_dec_ref(v_f_2869_);
lean_dec_ref(v_e_2868_);
v___x_2871_ = lean_box(0);
return v___x_2871_;
}
else
{
lean_object* v_fName_2872_; uint8_t v___y_2874_; uint8_t v___y_2887_; lean_object* v___x_2905_; uint8_t v___x_2906_; 
v_fName_2872_ = l_Lean_Expr_constName_x21(v_f_2869_);
lean_dec_ref(v_f_2869_);
v___x_2905_ = ((lean_object*)(lp_mathlib_Lean_Expr_numeral_x3f___closed__5));
v___x_2906_ = lean_name_eq(v_fName_2872_, v___x_2905_);
if (v___x_2906_ == 0)
{
v___y_2887_ = v___x_2906_;
goto v___jp_2886_;
}
else
{
lean_object* v___x_2907_; lean_object* v___x_2908_; uint8_t v___x_2909_; 
v___x_2907_ = l_Lean_Expr_getAppNumArgs(v_e_2868_);
v___x_2908_ = lean_unsigned_to_nat(1u);
v___x_2909_ = lean_nat_dec_eq(v___x_2907_, v___x_2908_);
lean_dec(v___x_2907_);
v___y_2887_ = v___x_2909_;
goto v___jp_2886_;
}
v___jp_2873_:
{
if (v___y_2874_ == 0)
{
lean_object* v___x_2875_; uint8_t v___x_2876_; 
v___x_2875_ = ((lean_object*)(lp_mathlib_Lean_Expr_numeral_x3f___closed__3));
v___x_2876_ = lean_name_eq(v_fName_2872_, v___x_2875_);
lean_dec(v_fName_2872_);
if (v___x_2876_ == 0)
{
lean_dec_ref(v_e_2868_);
v___y_2864_ = v___x_2876_;
goto v___jp_2863_;
}
else
{
lean_object* v___x_2877_; lean_object* v___x_2878_; uint8_t v___x_2879_; 
v___x_2877_ = l_Lean_Expr_getAppNumArgs(v_e_2868_);
lean_dec_ref(v_e_2868_);
v___x_2878_ = lean_unsigned_to_nat(0u);
v___x_2879_ = lean_nat_dec_eq(v___x_2877_, v___x_2878_);
lean_dec(v___x_2877_);
v___y_2864_ = v___x_2879_;
goto v___jp_2863_;
}
}
else
{
lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; 
lean_dec(v_fName_2872_);
v___x_2880_ = lean_unsigned_to_nat(1u);
v___x_2881_ = l_Lean_Expr_getAppNumArgs(v_e_2868_);
v___x_2882_ = lean_nat_sub(v___x_2881_, v___x_2880_);
lean_dec(v___x_2881_);
v___x_2883_ = lean_nat_sub(v___x_2882_, v___x_2880_);
lean_dec(v___x_2882_);
v___x_2884_ = l_Lean_Expr_getRevArg_x21(v_e_2868_, v___x_2883_);
lean_dec_ref(v_e_2868_);
v_e_2862_ = v___x_2884_;
goto _start;
}
}
v___jp_2886_:
{
if (v___y_2887_ == 0)
{
lean_object* v___x_2888_; uint8_t v___x_2889_; 
v___x_2888_ = ((lean_object*)(lp_mathlib_Lean_Expr_ofNat___closed__2));
v___x_2889_ = lean_name_eq(v_fName_2872_, v___x_2888_);
if (v___x_2889_ == 0)
{
v___y_2874_ = v___x_2889_;
goto v___jp_2873_;
}
else
{
lean_object* v___x_2890_; lean_object* v___x_2891_; uint8_t v___x_2892_; 
v___x_2890_ = l_Lean_Expr_getAppNumArgs(v_e_2868_);
v___x_2891_ = lean_unsigned_to_nat(3u);
v___x_2892_ = lean_nat_dec_eq(v___x_2890_, v___x_2891_);
lean_dec(v___x_2890_);
v___y_2874_ = v___x_2892_;
goto v___jp_2873_;
}
}
else
{
lean_object* v___x_2893_; lean_object* v___x_2894_; 
lean_dec(v_fName_2872_);
v___x_2893_ = l_Lean_Expr_appArg_x21(v_e_2868_);
lean_dec_ref(v_e_2868_);
v___x_2894_ = lp_mathlib_Lean_Expr_numeral_x3f(v___x_2893_);
if (lean_obj_tag(v___x_2894_) == 0)
{
return v___x_2894_;
}
else
{
lean_object* v_val_2895_; lean_object* v___x_2897_; uint8_t v_isShared_2898_; uint8_t v_isSharedCheck_2904_; 
v_val_2895_ = lean_ctor_get(v___x_2894_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2894_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2897_ = v___x_2894_;
v_isShared_2898_ = v_isSharedCheck_2904_;
goto v_resetjp_2896_;
}
else
{
lean_inc(v_val_2895_);
lean_dec(v___x_2894_);
v___x_2897_ = lean_box(0);
v_isShared_2898_ = v_isSharedCheck_2904_;
goto v_resetjp_2896_;
}
v_resetjp_2896_:
{
lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2902_; 
v___x_2899_ = lean_unsigned_to_nat(1u);
v___x_2900_ = lean_nat_add(v_val_2895_, v___x_2899_);
lean_dec(v_val_2895_);
if (v_isShared_2898_ == 0)
{
lean_ctor_set(v___x_2897_, 0, v___x_2900_);
v___x_2902_ = v___x_2897_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v___x_2900_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
}
}
}
}
v___jp_2863_:
{
if (v___y_2864_ == 0)
{
lean_object* v___x_2865_; 
v___x_2865_ = lean_box(0);
return v___x_2865_;
}
else
{
lean_object* v___x_2866_; 
v___x_2866_ = ((lean_object*)(lp_mathlib_Lean_Expr_numeral_x3f___closed__0));
return v___x_2866_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_zero_x3f(lean_object* v_e_2910_){
_start:
{
lean_object* v___x_2911_; 
v___x_2911_ = lp_mathlib_Lean_Expr_numeral_x3f(v_e_2910_);
if (lean_obj_tag(v___x_2911_) == 1)
{
lean_object* v_val_2912_; lean_object* v___x_2913_; uint8_t v___x_2914_; 
v_val_2912_ = lean_ctor_get(v___x_2911_, 0);
lean_inc(v_val_2912_);
lean_dec_ref_known(v___x_2911_, 1);
v___x_2913_ = lean_unsigned_to_nat(0u);
v___x_2914_ = lean_nat_dec_eq(v_val_2912_, v___x_2913_);
lean_dec(v_val_2912_);
return v___x_2914_;
}
else
{
uint8_t v___x_2915_; 
lean_dec(v___x_2911_);
v___x_2915_ = 0;
return v___x_2915_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_zero_x3f___boxed(lean_object* v_e_2916_){
_start:
{
uint8_t v_res_2917_; lean_object* v_r_2918_; 
v_res_2917_ = lp_mathlib_Lean_Expr_zero_x3f(v_e_2916_);
v_r_2918_ = lean_box(v_res_2917_);
return v_r_2918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27(lean_object* v_e_2928_){
_start:
{
lean_object* v___x_2929_; lean_object* v___x_2930_; uint8_t v___x_2931_; 
v___x_2929_ = ((lean_object*)(lp_mathlib_Lean_Expr_ne_x3f_x27___closed__1));
v___x_2930_ = lean_unsigned_to_nat(3u);
v___x_2931_ = l_Lean_Expr_isAppOfArity(v_e_2928_, v___x_2929_, v___x_2930_);
if (v___x_2931_ == 0)
{
lean_object* v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; uint8_t v___x_2935_; 
v___x_2932_ = lean_box(0);
v___x_2933_ = ((lean_object*)(lp_mathlib_Lean_Expr_ne_x3f_x27___closed__3));
v___x_2934_ = lean_unsigned_to_nat(1u);
v___x_2935_ = l_Lean_Expr_isAppOfArity(v_e_2928_, v___x_2933_, v___x_2934_);
if (v___x_2935_ == 0)
{
return v___x_2932_;
}
else
{
lean_object* v___x_2936_; lean_object* v___x_2937_; uint8_t v___x_2938_; 
v___x_2936_ = l_Lean_Expr_appArg_x21(v_e_2928_);
v___x_2937_ = ((lean_object*)(lp_mathlib_Lean_Expr_ne_x3f_x27___closed__5));
v___x_2938_ = l_Lean_Expr_isAppOfArity(v___x_2936_, v___x_2937_, v___x_2930_);
if (v___x_2938_ == 0)
{
lean_dec_ref(v___x_2936_);
return v___x_2932_;
}
else
{
lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; 
v___x_2939_ = l_Lean_Expr_appFn_x21(v___x_2936_);
v___x_2940_ = l_Lean_Expr_appFn_x21(v___x_2939_);
v___x_2941_ = l_Lean_Expr_appArg_x21(v___x_2940_);
lean_dec_ref(v___x_2940_);
v___x_2942_ = l_Lean_Expr_appArg_x21(v___x_2939_);
lean_dec_ref(v___x_2939_);
v___x_2943_ = l_Lean_Expr_appArg_x21(v___x_2936_);
lean_dec_ref(v___x_2936_);
v___x_2944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2944_, 0, v___x_2942_);
lean_ctor_set(v___x_2944_, 1, v___x_2943_);
v___x_2945_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2945_, 0, v___x_2941_);
lean_ctor_set(v___x_2945_, 1, v___x_2944_);
v___x_2946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2946_, 0, v___x_2945_);
return v___x_2946_;
}
}
}
else
{
lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2953_; lean_object* v___x_2954_; 
v___x_2947_ = l_Lean_Expr_appFn_x21(v_e_2928_);
v___x_2948_ = l_Lean_Expr_appFn_x21(v___x_2947_);
v___x_2949_ = l_Lean_Expr_appArg_x21(v___x_2948_);
lean_dec_ref(v___x_2948_);
v___x_2950_ = l_Lean_Expr_appArg_x21(v___x_2947_);
lean_dec_ref(v___x_2947_);
v___x_2951_ = l_Lean_Expr_appArg_x21(v_e_2928_);
v___x_2952_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2952_, 0, v___x_2950_);
lean_ctor_set(v___x_2952_, 1, v___x_2951_);
v___x_2953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2953_, 0, v___x_2949_);
lean_ctor_set(v___x_2953_, 1, v___x_2952_);
v___x_2954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2954_, 0, v___x_2953_);
return v___x_2954_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27___boxed(lean_object* v_e_2955_){
_start:
{
lean_object* v_res_2956_; 
v_res_2956_ = lp_mathlib_Lean_Expr_ne_x3f_x27(v_e_2955_);
lean_dec_ref(v_e_2955_);
return v_res_2956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_le_x3f(lean_object* v_p_2962_){
_start:
{
lean_object* v___x_2963_; lean_object* v___x_2964_; uint8_t v___x_2965_; 
v___x_2963_ = ((lean_object*)(lp_mathlib_Lean_Expr_le_x3f___closed__2));
v___x_2964_ = lean_unsigned_to_nat(4u);
v___x_2965_ = l_Lean_Expr_isAppOfArity(v_p_2962_, v___x_2963_, v___x_2964_);
if (v___x_2965_ == 0)
{
lean_object* v___x_2966_; 
v___x_2966_ = lean_box(0);
return v___x_2966_;
}
else
{
lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___x_2973_; lean_object* v___x_2974_; lean_object* v___x_2975_; 
v___x_2967_ = l_Lean_Expr_appFn_x21(v_p_2962_);
v___x_2968_ = l_Lean_Expr_appFn_x21(v___x_2967_);
v___x_2969_ = l_Lean_Expr_appFn_x21(v___x_2968_);
lean_dec_ref(v___x_2968_);
v___x_2970_ = l_Lean_Expr_appArg_x21(v___x_2969_);
lean_dec_ref(v___x_2969_);
v___x_2971_ = l_Lean_Expr_appArg_x21(v___x_2967_);
lean_dec_ref(v___x_2967_);
v___x_2972_ = l_Lean_Expr_appArg_x21(v_p_2962_);
v___x_2973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2973_, 0, v___x_2971_);
lean_ctor_set(v___x_2973_, 1, v___x_2972_);
v___x_2974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2974_, 0, v___x_2970_);
lean_ctor_set(v___x_2974_, 1, v___x_2973_);
v___x_2975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2975_, 0, v___x_2974_);
return v___x_2975_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_le_x3f___boxed(lean_object* v_p_2976_){
_start:
{
lean_object* v_res_2977_; 
v_res_2977_ = lp_mathlib_Lean_Expr_le_x3f(v_p_2976_);
lean_dec_ref(v_p_2976_);
return v_res_2977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_lt_x3f(lean_object* v_p_2983_){
_start:
{
lean_object* v___x_2984_; lean_object* v___x_2985_; uint8_t v___x_2986_; 
v___x_2984_ = ((lean_object*)(lp_mathlib_Lean_Expr_lt_x3f___closed__2));
v___x_2985_ = lean_unsigned_to_nat(4u);
v___x_2986_ = l_Lean_Expr_isAppOfArity(v_p_2983_, v___x_2984_, v___x_2985_);
if (v___x_2986_ == 0)
{
lean_object* v___x_2987_; 
v___x_2987_ = lean_box(0);
return v___x_2987_;
}
else
{
lean_object* v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; lean_object* v___x_2992_; lean_object* v___x_2993_; lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; 
v___x_2988_ = l_Lean_Expr_appFn_x21(v_p_2983_);
v___x_2989_ = l_Lean_Expr_appFn_x21(v___x_2988_);
v___x_2990_ = l_Lean_Expr_appFn_x21(v___x_2989_);
lean_dec_ref(v___x_2989_);
v___x_2991_ = l_Lean_Expr_appArg_x21(v___x_2990_);
lean_dec_ref(v___x_2990_);
v___x_2992_ = l_Lean_Expr_appArg_x21(v___x_2988_);
lean_dec_ref(v___x_2988_);
v___x_2993_ = l_Lean_Expr_appArg_x21(v_p_2983_);
v___x_2994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2994_, 0, v___x_2992_);
lean_ctor_set(v___x_2994_, 1, v___x_2993_);
v___x_2995_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2995_, 0, v___x_2991_);
lean_ctor_set(v___x_2995_, 1, v___x_2994_);
v___x_2996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2996_, 0, v___x_2995_);
return v___x_2996_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_lt_x3f___boxed(lean_object* v_p_2997_){
_start:
{
lean_object* v_res_2998_; 
v_res_2998_ = lp_mathlib_Lean_Expr_lt_x3f(v_p_2997_);
lean_dec_ref(v_p_2997_);
return v_res_2998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_sides_x3f(lean_object* v_ty_3005_){
_start:
{
lean_object* v___x_3006_; lean_object* v___x_3007_; uint8_t v___x_3008_; 
v___x_3006_ = ((lean_object*)(lp_mathlib_Lean_Expr_sides_x3f___closed__1));
v___x_3007_ = lean_unsigned_to_nat(2u);
v___x_3008_ = l_Lean_Expr_isAppOfArity(v_ty_3005_, v___x_3006_, v___x_3007_);
if (v___x_3008_ == 0)
{
lean_object* v___x_3009_; lean_object* v___x_3010_; uint8_t v___x_3011_; 
v___x_3009_ = ((lean_object*)(lp_mathlib_Lean_Expr_ne_x3f_x27___closed__5));
v___x_3010_ = lean_unsigned_to_nat(3u);
v___x_3011_ = l_Lean_Expr_isAppOfArity(v_ty_3005_, v___x_3009_, v___x_3010_);
if (v___x_3011_ == 0)
{
lean_object* v___x_3012_; lean_object* v___x_3013_; uint8_t v___x_3014_; 
v___x_3012_ = ((lean_object*)(lp_mathlib_Lean_Expr_sides_x3f___closed__3));
v___x_3013_ = lean_unsigned_to_nat(4u);
v___x_3014_ = l_Lean_Expr_isAppOfArity(v_ty_3005_, v___x_3012_, v___x_3013_);
if (v___x_3014_ == 0)
{
lean_object* v___x_3015_; 
v___x_3015_ = lean_box(0);
return v___x_3015_;
}
else
{
lean_object* v___x_3016_; lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; 
v___x_3016_ = l_Lean_Expr_appFn_x21(v_ty_3005_);
v___x_3017_ = l_Lean_Expr_appFn_x21(v___x_3016_);
v___x_3018_ = l_Lean_Expr_appFn_x21(v___x_3017_);
v___x_3019_ = l_Lean_Expr_appArg_x21(v___x_3018_);
lean_dec_ref(v___x_3018_);
v___x_3020_ = l_Lean_Expr_appArg_x21(v___x_3017_);
lean_dec_ref(v___x_3017_);
v___x_3021_ = l_Lean_Expr_appArg_x21(v___x_3016_);
lean_dec_ref(v___x_3016_);
v___x_3022_ = l_Lean_Expr_appArg_x21(v_ty_3005_);
v___x_3023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3023_, 0, v___x_3021_);
lean_ctor_set(v___x_3023_, 1, v___x_3022_);
v___x_3024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3024_, 0, v___x_3020_);
lean_ctor_set(v___x_3024_, 1, v___x_3023_);
v___x_3025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3025_, 0, v___x_3019_);
lean_ctor_set(v___x_3025_, 1, v___x_3024_);
v___x_3026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3026_, 0, v___x_3025_);
return v___x_3026_;
}
}
else
{
lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3033_; lean_object* v___x_3034_; lean_object* v___x_3035_; 
v___x_3027_ = l_Lean_Expr_appFn_x21(v_ty_3005_);
v___x_3028_ = l_Lean_Expr_appFn_x21(v___x_3027_);
v___x_3029_ = l_Lean_Expr_appArg_x21(v___x_3028_);
lean_dec_ref(v___x_3028_);
v___x_3030_ = l_Lean_Expr_appArg_x21(v___x_3027_);
lean_dec_ref(v___x_3027_);
v___x_3031_ = l_Lean_Expr_appArg_x21(v_ty_3005_);
lean_inc_ref(v___x_3029_);
v___x_3032_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3032_, 0, v___x_3029_);
lean_ctor_set(v___x_3032_, 1, v___x_3031_);
v___x_3033_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3033_, 0, v___x_3030_);
lean_ctor_set(v___x_3033_, 1, v___x_3032_);
v___x_3034_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3034_, 0, v___x_3029_);
lean_ctor_set(v___x_3034_, 1, v___x_3033_);
v___x_3035_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3035_, 0, v___x_3034_);
return v___x_3035_;
}
}
else
{
lean_object* v___x_3036_; lean_object* v___x_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; 
v___x_3036_ = l_Lean_Expr_appFn_x21(v_ty_3005_);
v___x_3037_ = l_Lean_Expr_appArg_x21(v___x_3036_);
lean_dec_ref(v___x_3036_);
v___x_3038_ = l_Lean_Expr_appArg_x21(v_ty_3005_);
v___x_3039_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1);
v___x_3040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3040_, 0, v___x_3039_);
lean_ctor_set(v___x_3040_, 1, v___x_3038_);
v___x_3041_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3041_, 0, v___x_3037_);
lean_ctor_set(v___x_3041_, 1, v___x_3040_);
v___x_3042_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3042_, 0, v___x_3039_);
lean_ctor_set(v___x_3042_, 1, v___x_3041_);
v___x_3043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3043_, 0, v___x_3042_);
return v___x_3043_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_sides_x3f___boxed(lean_object* v_ty_3044_){
_start:
{
lean_object* v_res_3045_; 
v_res_3045_ = lp_mathlib_Lean_Expr_sides_x3f(v_ty_3044_);
lean_dec_ref(v_ty_3044_);
return v_res_3045_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isSorryAx(lean_object* v_x_3046_){
_start:
{
if (lean_obj_tag(v_x_3046_) == 5)
{
lean_object* v_fn_3047_; 
v_fn_3047_ = lean_ctor_get(v_x_3046_, 0);
if (lean_obj_tag(v_fn_3047_) == 5)
{
lean_object* v_fn_3048_; lean_object* v___x_3049_; uint8_t v___x_3050_; 
v_fn_3048_ = lean_ctor_get(v_fn_3047_, 0);
v___x_3049_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___redArg___closed__1));
v___x_3050_ = l_Lean_Expr_isConstOf(v_fn_3048_, v___x_3049_);
return v___x_3050_;
}
else
{
uint8_t v___x_3051_; 
v___x_3051_ = 0;
return v___x_3051_;
}
}
else
{
uint8_t v___x_3052_; 
v___x_3052_ = 0;
return v___x_3052_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isSorryAx___boxed(lean_object* v_x_3053_){
_start:
{
uint8_t v_res_3054_; lean_object* v_r_3055_; 
v_res_3054_ = lp_mathlib_Lean_Expr_isSorryAx(v_x_3053_);
lean_dec_ref(v_x_3053_);
v_r_3055_ = lean_box(v_res_3054_);
return v_r_3055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyAppArgM___redArg(lean_object* v_inst_3056_, lean_object* v_inst_3057_, lean_object* v_modifier_3058_, lean_object* v_x_3059_){
_start:
{
if (lean_obj_tag(v_x_3059_) == 5)
{
lean_object* v_fn_3060_; lean_object* v_arg_3061_; lean_object* v_map_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; 
lean_dec(v_inst_3057_);
v_fn_3060_ = lean_ctor_get(v_x_3059_, 0);
lean_inc_ref(v_fn_3060_);
v_arg_3061_ = lean_ctor_get(v_x_3059_, 1);
lean_inc_ref(v_arg_3061_);
lean_dec_ref_known(v_x_3059_, 2);
v_map_3062_ = lean_ctor_get(v_inst_3056_, 0);
lean_inc(v_map_3062_);
lean_dec_ref(v_inst_3056_);
v___x_3063_ = lean_alloc_closure((void*)(l_Lean_mkApp), 2, 1);
lean_closure_set(v___x_3063_, 0, v_fn_3060_);
v___x_3064_ = lean_apply_1(v_modifier_3058_, v_arg_3061_);
v___x_3065_ = lean_apply_4(v_map_3062_, lean_box(0), lean_box(0), v___x_3063_, v___x_3064_);
return v___x_3065_;
}
else
{
lean_object* v___x_3066_; 
lean_dec(v_modifier_3058_);
lean_dec_ref(v_inst_3056_);
v___x_3066_ = lean_apply_2(v_inst_3057_, lean_box(0), v_x_3059_);
return v___x_3066_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyAppArgM(lean_object* v_M_3067_, lean_object* v_inst_3068_, lean_object* v_inst_3069_, lean_object* v_modifier_3070_, lean_object* v_x_3071_){
_start:
{
lean_object* v___x_3072_; 
v___x_3072_ = lp_mathlib_Lean_Expr_modifyAppArgM___redArg(v_inst_3068_, v_inst_3069_, v_modifier_3070_, v_x_3071_);
return v___x_3072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyRevArg(lean_object* v_modifier_3073_, lean_object* v_x_3074_, lean_object* v_x_3075_){
_start:
{
lean_object* v_zero_3076_; uint8_t v_isZero_3077_; 
v_zero_3076_ = lean_unsigned_to_nat(0u);
v_isZero_3077_ = lean_nat_dec_eq(v_x_3074_, v_zero_3076_);
if (v_isZero_3077_ == 1)
{
if (lean_obj_tag(v_x_3075_) == 5)
{
lean_object* v_fn_3078_; lean_object* v_arg_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; 
v_fn_3078_ = lean_ctor_get(v_x_3075_, 0);
lean_inc_ref(v_fn_3078_);
v_arg_3079_ = lean_ctor_get(v_x_3075_, 1);
lean_inc_ref(v_arg_3079_);
lean_dec_ref_known(v_x_3075_, 2);
v___x_3080_ = lean_apply_1(v_modifier_3073_, v_arg_3079_);
v___x_3081_ = l_Lean_Expr_app___override(v_fn_3078_, v___x_3080_);
return v___x_3081_;
}
else
{
lean_dec_ref(v_modifier_3073_);
return v_x_3075_;
}
}
else
{
if (lean_obj_tag(v_x_3075_) == 5)
{
lean_object* v_fn_3082_; lean_object* v_arg_3083_; lean_object* v_one_3084_; lean_object* v_n_3085_; lean_object* v___x_3086_; lean_object* v___x_3087_; 
v_fn_3082_ = lean_ctor_get(v_x_3075_, 0);
lean_inc_ref(v_fn_3082_);
v_arg_3083_ = lean_ctor_get(v_x_3075_, 1);
lean_inc_ref(v_arg_3083_);
lean_dec_ref_known(v_x_3075_, 2);
v_one_3084_ = lean_unsigned_to_nat(1u);
v_n_3085_ = lean_nat_sub(v_x_3074_, v_one_3084_);
v___x_3086_ = lp_mathlib_Lean_Expr_modifyRevArg(v_modifier_3073_, v_n_3085_, v_fn_3082_);
lean_dec(v_n_3085_);
v___x_3087_ = l_Lean_Expr_app___override(v___x_3086_, v_arg_3083_);
return v___x_3087_;
}
else
{
lean_dec_ref(v_modifier_3073_);
return v_x_3075_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyRevArg___boxed(lean_object* v_modifier_3088_, lean_object* v_x_3089_, lean_object* v_x_3090_){
_start:
{
lean_object* v_res_3091_; 
v_res_3091_ = lp_mathlib_Lean_Expr_modifyRevArg(v_modifier_3088_, v_x_3089_, v_x_3090_);
lean_dec(v_x_3089_);
return v_res_3091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArg(lean_object* v_modifier_3092_, lean_object* v_e_3093_, lean_object* v_i_3094_, lean_object* v_n_3095_){
_start:
{
lean_object* v___x_3096_; lean_object* v___x_3097_; lean_object* v___x_3098_; lean_object* v___x_3099_; 
v___x_3096_ = lean_nat_sub(v_n_3095_, v_i_3094_);
v___x_3097_ = lean_unsigned_to_nat(1u);
v___x_3098_ = lean_nat_sub(v___x_3096_, v___x_3097_);
lean_dec(v___x_3096_);
v___x_3099_ = lp_mathlib_Lean_Expr_modifyRevArg(v_modifier_3092_, v___x_3098_, v_e_3093_);
lean_dec(v___x_3098_);
return v___x_3099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArg___boxed(lean_object* v_modifier_3100_, lean_object* v_e_3101_, lean_object* v_i_3102_, lean_object* v_n_3103_){
_start:
{
lean_object* v_res_3104_; 
v_res_3104_ = lp_mathlib_Lean_Expr_modifyArg(v_modifier_3100_, v_e_3101_, v_i_3102_, v_n_3103_);
lean_dec(v_n_3103_);
lean_dec(v_i_3102_);
return v_res_3104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___lam__0(lean_object* v_x_3105_, lean_object* v_x_3106_){
_start:
{
lean_inc_ref(v_x_3105_);
return v_x_3105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___lam__0___boxed(lean_object* v_x_3107_, lean_object* v_x_3108_){
_start:
{
lean_object* v_res_3109_; 
v_res_3109_ = lp_mathlib_Lean_Expr_setArg___lam__0(v_x_3107_, v_x_3108_);
lean_dec_ref(v_x_3108_);
lean_dec_ref(v_x_3107_);
return v_res_3109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg(lean_object* v_e_3110_, lean_object* v_i_3111_, lean_object* v_x_3112_, lean_object* v_n_3113_){
_start:
{
lean_object* v___f_3114_; lean_object* v___x_3115_; 
v___f_3114_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_setArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3114_, 0, v_x_3112_);
v___x_3115_ = lp_mathlib_Lean_Expr_modifyArg(v___f_3114_, v_e_3110_, v_i_3111_, v_n_3113_);
return v___x_3115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_setArg___boxed(lean_object* v_e_3116_, lean_object* v_i_3117_, lean_object* v_x_3118_, lean_object* v_n_3119_){
_start:
{
lean_object* v_res_3120_; 
v_res_3120_ = lp_mathlib_Lean_Expr_setArg(v_e_3116_, v_i_3117_, v_x_3118_, v_n_3119_);
lean_dec(v_n_3119_);
lean_dec(v_i_3117_);
return v_res_3120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getRevArg_x3f(lean_object* v_x_3121_, lean_object* v_x_3122_){
_start:
{
if (lean_obj_tag(v_x_3121_) == 5)
{
lean_object* v_fn_3123_; lean_object* v_arg_3124_; lean_object* v_zero_3125_; uint8_t v_isZero_3126_; 
v_fn_3123_ = lean_ctor_get(v_x_3121_, 0);
v_arg_3124_ = lean_ctor_get(v_x_3121_, 1);
v_zero_3125_ = lean_unsigned_to_nat(0u);
v_isZero_3126_ = lean_nat_dec_eq(v_x_3122_, v_zero_3125_);
if (v_isZero_3126_ == 1)
{
lean_object* v___x_3127_; 
lean_inc_ref(v_arg_3124_);
v___x_3127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3127_, 0, v_arg_3124_);
return v___x_3127_;
}
else
{
lean_object* v_one_3128_; lean_object* v_n_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; 
v_one_3128_ = lean_unsigned_to_nat(1u);
v_n_3129_ = lean_nat_sub(v_x_3122_, v_one_3128_);
v___x_3130_ = l_Lean_Expr_getRevArg_x21(v_fn_3123_, v_n_3129_);
v___x_3131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3131_, 0, v___x_3130_);
return v___x_3131_;
}
}
else
{
lean_object* v___x_3132_; 
v___x_3132_ = lean_box(0);
return v___x_3132_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getRevArg_x3f___boxed(lean_object* v_x_3133_, lean_object* v_x_3134_){
_start:
{
lean_object* v_res_3135_; 
v_res_3135_ = lp_mathlib_Lean_Expr_getRevArg_x3f(v_x_3133_, v_x_3134_);
lean_dec(v_x_3134_);
lean_dec_ref(v_x_3133_);
return v_res_3135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getArg_x3f(lean_object* v_e_3136_, lean_object* v_i_3137_, lean_object* v_n_3138_){
_start:
{
lean_object* v___x_3139_; lean_object* v___x_3140_; lean_object* v___x_3141_; lean_object* v___x_3142_; 
v___x_3139_ = lean_nat_sub(v_n_3138_, v_i_3137_);
v___x_3140_ = lean_unsigned_to_nat(1u);
v___x_3141_ = lean_nat_sub(v___x_3139_, v___x_3140_);
lean_dec(v___x_3139_);
v___x_3142_ = lp_mathlib_Lean_Expr_getRevArg_x3f(v_e_3136_, v___x_3141_);
lean_dec(v___x_3141_);
return v___x_3142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getArg_x3f___boxed(lean_object* v_e_3143_, lean_object* v_i_3144_, lean_object* v_n_3145_){
_start:
{
lean_object* v_res_3146_; 
v_res_3146_ = lp_mathlib_Lean_Expr_getArg_x3f(v_e_3143_, v_i_3144_, v_n_3145_);
lean_dec(v_n_3145_);
lean_dec(v_i_3144_);
lean_dec_ref(v_e_3143_);
return v_res_3146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0(lean_object* v_a_3147_, lean_object* v_x_3148_){
_start:
{
lean_inc_ref(v_a_3147_);
return v_a_3147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0___boxed(lean_object* v_a_3149_, lean_object* v_x_3150_){
_start:
{
lean_object* v_res_3151_; 
v_res_3151_ = lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0(v_a_3149_, v_x_3150_);
lean_dec_ref(v_x_3150_);
lean_dec_ref(v_a_3149_);
return v_res_3151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1(lean_object* v_e_3152_, lean_object* v_i_3153_, lean_object* v_n_3154_, lean_object* v_toPure_3155_, lean_object* v_a_3156_){
_start:
{
lean_object* v___f_3157_; lean_object* v___x_3158_; lean_object* v___x_3159_; 
v___f_3157_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3157_, 0, v_a_3156_);
v___x_3158_ = lp_mathlib_Lean_Expr_modifyArg(v___f_3157_, v_e_3152_, v_i_3153_, v_n_3154_);
v___x_3159_ = lean_apply_2(v_toPure_3155_, lean_box(0), v___x_3158_);
return v___x_3159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1___boxed(lean_object* v_e_3160_, lean_object* v_i_3161_, lean_object* v_n_3162_, lean_object* v_toPure_3163_, lean_object* v_a_3164_){
_start:
{
lean_object* v_res_3165_; 
v_res_3165_ = lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1(v_e_3160_, v_i_3161_, v_n_3162_, v_toPure_3163_, v_a_3164_);
lean_dec(v_n_3162_);
lean_dec(v_i_3161_);
return v_res_3165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM___redArg(lean_object* v_inst_3166_, lean_object* v_modifier_3167_, lean_object* v_e_3168_, lean_object* v_i_3169_, lean_object* v_n_3170_){
_start:
{
lean_object* v_toApplicative_3171_; lean_object* v_toBind_3172_; lean_object* v_toPure_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; 
v_toApplicative_3171_ = lean_ctor_get(v_inst_3166_, 0);
lean_inc_ref(v_toApplicative_3171_);
v_toBind_3172_ = lean_ctor_get(v_inst_3166_, 1);
lean_inc(v_toBind_3172_);
lean_dec_ref(v_inst_3166_);
v_toPure_3173_ = lean_ctor_get(v_toApplicative_3171_, 1);
lean_inc(v_toPure_3173_);
lean_dec_ref(v_toApplicative_3171_);
v___x_3174_ = l_Lean_Expr_getAppNumArgs(v_e_3168_);
v___x_3175_ = lp_mathlib_Lean_Expr_getArg_x3f(v_e_3168_, v_i_3169_, v___x_3174_);
lean_dec(v___x_3174_);
if (lean_obj_tag(v___x_3175_) == 1)
{
lean_object* v_val_3176_; lean_object* v___f_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; 
v_val_3176_ = lean_ctor_get(v___x_3175_, 0);
lean_inc(v_val_3176_);
lean_dec_ref_known(v___x_3175_, 1);
v___f_3177_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_modifyArgM___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_3177_, 0, v_e_3168_);
lean_closure_set(v___f_3177_, 1, v_i_3169_);
lean_closure_set(v___f_3177_, 2, v_n_3170_);
lean_closure_set(v___f_3177_, 3, v_toPure_3173_);
v___x_3178_ = lean_apply_1(v_modifier_3167_, v_val_3176_);
v___x_3179_ = lean_apply_4(v_toBind_3172_, lean_box(0), lean_box(0), v___x_3178_, v___f_3177_);
return v___x_3179_;
}
else
{
lean_object* v___x_3180_; 
lean_dec(v___x_3175_);
lean_dec(v_toBind_3172_);
lean_dec(v_n_3170_);
lean_dec(v_i_3169_);
lean_dec(v_modifier_3167_);
v___x_3180_ = lean_apply_2(v_toPure_3173_, lean_box(0), v_e_3168_);
return v___x_3180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_modifyArgM(lean_object* v_M_3181_, lean_object* v_inst_3182_, lean_object* v_modifier_3183_, lean_object* v_e_3184_, lean_object* v_i_3185_, lean_object* v_n_3186_){
_start:
{
lean_object* v___x_3187_; 
v___x_3187_ = lp_mathlib_Lean_Expr_modifyArgM___redArg(v_inst_3182_, v_modifier_3183_, v_e_3184_, v_i_3185_, v_n_3186_);
return v___x_3187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_renameBVar(lean_object* v_e_3188_, lean_object* v_old_3189_, lean_object* v_new_3190_){
_start:
{
switch(lean_obj_tag(v_e_3188_))
{
case 5:
{
lean_object* v_fn_3191_; lean_object* v_arg_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; 
v_fn_3191_ = lean_ctor_get(v_e_3188_, 0);
lean_inc_ref(v_fn_3191_);
v_arg_3192_ = lean_ctor_get(v_e_3188_, 1);
lean_inc_ref(v_arg_3192_);
lean_dec_ref_known(v_e_3188_, 2);
lean_inc(v_new_3190_);
v___x_3193_ = lp_mathlib_Lean_Expr_renameBVar(v_fn_3191_, v_old_3189_, v_new_3190_);
v___x_3194_ = lp_mathlib_Lean_Expr_renameBVar(v_arg_3192_, v_old_3189_, v_new_3190_);
v___x_3195_ = l_Lean_Expr_app___override(v___x_3193_, v___x_3194_);
return v___x_3195_;
}
case 6:
{
lean_object* v_binderName_3196_; lean_object* v_binderType_3197_; lean_object* v_body_3198_; uint8_t v_binderInfo_3199_; lean_object* v___y_3201_; uint8_t v___x_3205_; 
v_binderName_3196_ = lean_ctor_get(v_e_3188_, 0);
lean_inc(v_binderName_3196_);
v_binderType_3197_ = lean_ctor_get(v_e_3188_, 1);
lean_inc_ref(v_binderType_3197_);
v_body_3198_ = lean_ctor_get(v_e_3188_, 2);
lean_inc_ref(v_body_3198_);
v_binderInfo_3199_ = lean_ctor_get_uint8(v_e_3188_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_3188_, 3);
v___x_3205_ = lean_name_eq(v_binderName_3196_, v_old_3189_);
if (v___x_3205_ == 0)
{
v___y_3201_ = v_binderName_3196_;
goto v___jp_3200_;
}
else
{
lean_dec(v_binderName_3196_);
lean_inc(v_new_3190_);
v___y_3201_ = v_new_3190_;
goto v___jp_3200_;
}
v___jp_3200_:
{
lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; 
lean_inc(v_new_3190_);
v___x_3202_ = lp_mathlib_Lean_Expr_renameBVar(v_binderType_3197_, v_old_3189_, v_new_3190_);
v___x_3203_ = lp_mathlib_Lean_Expr_renameBVar(v_body_3198_, v_old_3189_, v_new_3190_);
v___x_3204_ = l_Lean_Expr_lam___override(v___y_3201_, v___x_3202_, v___x_3203_, v_binderInfo_3199_);
return v___x_3204_;
}
}
case 7:
{
lean_object* v_binderName_3206_; lean_object* v_binderType_3207_; lean_object* v_body_3208_; uint8_t v_binderInfo_3209_; lean_object* v___y_3211_; uint8_t v___x_3215_; 
v_binderName_3206_ = lean_ctor_get(v_e_3188_, 0);
lean_inc(v_binderName_3206_);
v_binderType_3207_ = lean_ctor_get(v_e_3188_, 1);
lean_inc_ref(v_binderType_3207_);
v_body_3208_ = lean_ctor_get(v_e_3188_, 2);
lean_inc_ref(v_body_3208_);
v_binderInfo_3209_ = lean_ctor_get_uint8(v_e_3188_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_3188_, 3);
v___x_3215_ = lean_name_eq(v_binderName_3206_, v_old_3189_);
if (v___x_3215_ == 0)
{
v___y_3211_ = v_binderName_3206_;
goto v___jp_3210_;
}
else
{
lean_dec(v_binderName_3206_);
lean_inc(v_new_3190_);
v___y_3211_ = v_new_3190_;
goto v___jp_3210_;
}
v___jp_3210_:
{
lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; 
lean_inc(v_new_3190_);
v___x_3212_ = lp_mathlib_Lean_Expr_renameBVar(v_binderType_3207_, v_old_3189_, v_new_3190_);
v___x_3213_ = lp_mathlib_Lean_Expr_renameBVar(v_body_3208_, v_old_3189_, v_new_3190_);
v___x_3214_ = l_Lean_Expr_forallE___override(v___y_3211_, v___x_3212_, v___x_3213_, v_binderInfo_3209_);
return v___x_3214_;
}
}
case 10:
{
lean_object* v_data_3216_; lean_object* v_expr_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; 
v_data_3216_ = lean_ctor_get(v_e_3188_, 0);
lean_inc(v_data_3216_);
v_expr_3217_ = lean_ctor_get(v_e_3188_, 1);
lean_inc_ref(v_expr_3217_);
lean_dec_ref_known(v_e_3188_, 2);
v___x_3218_ = lp_mathlib_Lean_Expr_renameBVar(v_expr_3217_, v_old_3189_, v_new_3190_);
v___x_3219_ = l_Lean_Expr_mdata___override(v_data_3216_, v___x_3218_);
return v___x_3219_;
}
default: 
{
lean_dec(v_new_3190_);
return v_e_3188_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_renameBVar___boxed(lean_object* v_e_3220_, lean_object* v_old_3221_, lean_object* v_new_3222_){
_start:
{
lean_object* v_res_3223_; 
v_res_3223_ = lp_mathlib_Lean_Expr_renameBVar(v_e_3220_, v_old_3221_, v_new_3222_);
lean_dec(v_old_3221_);
return v_res_3223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getBinderName(lean_object* v_e_3224_, lean_object* v_a_3225_, lean_object* v_a_3226_, lean_object* v_a_3227_, lean_object* v_a_3228_){
_start:
{
lean_object* v_keyedConfig_3230_; uint8_t v_trackZetaDelta_3231_; lean_object* v_zetaDeltaSet_3232_; lean_object* v_lctx_3233_; lean_object* v_localInstances_3234_; lean_object* v_defEqCtx_x3f_3235_; lean_object* v_synthPendingDepth_3236_; lean_object* v_customCanUnfoldPredicate_x3f_3237_; uint8_t v_univApprox_3238_; uint8_t v_inTypeClassResolution_3239_; uint8_t v_cacheInferType_3240_; uint8_t v___x_3241_; lean_object* v___x_3242_; lean_object* v___x_3243_; lean_object* v___x_3244_; 
v_keyedConfig_3230_ = lean_ctor_get(v_a_3225_, 0);
v_trackZetaDelta_3231_ = lean_ctor_get_uint8(v_a_3225_, sizeof(void*)*7);
v_zetaDeltaSet_3232_ = lean_ctor_get(v_a_3225_, 1);
v_lctx_3233_ = lean_ctor_get(v_a_3225_, 2);
v_localInstances_3234_ = lean_ctor_get(v_a_3225_, 3);
v_defEqCtx_x3f_3235_ = lean_ctor_get(v_a_3225_, 4);
v_synthPendingDepth_3236_ = lean_ctor_get(v_a_3225_, 5);
v_customCanUnfoldPredicate_x3f_3237_ = lean_ctor_get(v_a_3225_, 6);
v_univApprox_3238_ = lean_ctor_get_uint8(v_a_3225_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3239_ = lean_ctor_get_uint8(v_a_3225_, sizeof(void*)*7 + 2);
v_cacheInferType_3240_ = lean_ctor_get_uint8(v_a_3225_, sizeof(void*)*7 + 3);
v___x_3241_ = 2;
lean_inc_ref(v_keyedConfig_3230_);
v___x_3242_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3241_, v_keyedConfig_3230_);
lean_inc(v_customCanUnfoldPredicate_x3f_3237_);
lean_inc(v_synthPendingDepth_3236_);
lean_inc(v_defEqCtx_x3f_3235_);
lean_inc_ref(v_localInstances_3234_);
lean_inc_ref(v_lctx_3233_);
lean_inc(v_zetaDeltaSet_3232_);
v___x_3243_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3243_, 0, v___x_3242_);
lean_ctor_set(v___x_3243_, 1, v_zetaDeltaSet_3232_);
lean_ctor_set(v___x_3243_, 2, v_lctx_3233_);
lean_ctor_set(v___x_3243_, 3, v_localInstances_3234_);
lean_ctor_set(v___x_3243_, 4, v_defEqCtx_x3f_3235_);
lean_ctor_set(v___x_3243_, 5, v_synthPendingDepth_3236_);
lean_ctor_set(v___x_3243_, 6, v_customCanUnfoldPredicate_x3f_3237_);
lean_ctor_set_uint8(v___x_3243_, sizeof(void*)*7, v_trackZetaDelta_3231_);
lean_ctor_set_uint8(v___x_3243_, sizeof(void*)*7 + 1, v_univApprox_3238_);
lean_ctor_set_uint8(v___x_3243_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3239_);
lean_ctor_set_uint8(v___x_3243_, sizeof(void*)*7 + 3, v_cacheInferType_3240_);
lean_inc(v_a_3228_);
lean_inc_ref(v_a_3227_);
lean_inc(v_a_3226_);
v___x_3244_ = lean_whnf(v_e_3224_, v___x_3243_, v_a_3226_, v_a_3227_, v_a_3228_);
if (lean_obj_tag(v___x_3244_) == 0)
{
lean_object* v_a_3245_; lean_object* v___x_3247_; uint8_t v_isShared_3248_; uint8_t v_isSharedCheck_3259_; 
v_a_3245_ = lean_ctor_get(v___x_3244_, 0);
v_isSharedCheck_3259_ = !lean_is_exclusive(v___x_3244_);
if (v_isSharedCheck_3259_ == 0)
{
v___x_3247_ = v___x_3244_;
v_isShared_3248_ = v_isSharedCheck_3259_;
goto v_resetjp_3246_;
}
else
{
lean_inc(v_a_3245_);
lean_dec(v___x_3244_);
v___x_3247_ = lean_box(0);
v_isShared_3248_ = v_isSharedCheck_3259_;
goto v_resetjp_3246_;
}
v_resetjp_3246_:
{
lean_object* v_n_3250_; 
switch(lean_obj_tag(v_a_3245_))
{
case 7:
{
lean_object* v_binderName_3255_; 
v_binderName_3255_ = lean_ctor_get(v_a_3245_, 0);
lean_inc(v_binderName_3255_);
lean_dec_ref_known(v_a_3245_, 3);
v_n_3250_ = v_binderName_3255_;
goto v___jp_3249_;
}
case 6:
{
lean_object* v_binderName_3256_; 
v_binderName_3256_ = lean_ctor_get(v_a_3245_, 0);
lean_inc(v_binderName_3256_);
lean_dec_ref_known(v_a_3245_, 3);
v_n_3250_ = v_binderName_3256_;
goto v___jp_3249_;
}
default: 
{
lean_object* v___x_3257_; lean_object* v___x_3258_; 
lean_del_object(v___x_3247_);
lean_dec(v_a_3245_);
v___x_3257_ = lean_box(0);
v___x_3258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3258_, 0, v___x_3257_);
return v___x_3258_;
}
}
v___jp_3249_:
{
lean_object* v___x_3251_; lean_object* v___x_3253_; 
v___x_3251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3251_, 0, v_n_3250_);
if (v_isShared_3248_ == 0)
{
lean_ctor_set(v___x_3247_, 0, v___x_3251_);
v___x_3253_ = v___x_3247_;
goto v_reusejp_3252_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v___x_3251_);
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
else
{
lean_object* v_a_3260_; lean_object* v___x_3262_; uint8_t v_isShared_3263_; uint8_t v_isSharedCheck_3267_; 
v_a_3260_ = lean_ctor_get(v___x_3244_, 0);
v_isSharedCheck_3267_ = !lean_is_exclusive(v___x_3244_);
if (v_isSharedCheck_3267_ == 0)
{
v___x_3262_ = v___x_3244_;
v_isShared_3263_ = v_isSharedCheck_3267_;
goto v_resetjp_3261_;
}
else
{
lean_inc(v_a_3260_);
lean_dec(v___x_3244_);
v___x_3262_ = lean_box(0);
v_isShared_3263_ = v_isSharedCheck_3267_;
goto v_resetjp_3261_;
}
v_resetjp_3261_:
{
lean_object* v___x_3265_; 
if (v_isShared_3263_ == 0)
{
v___x_3265_ = v___x_3262_;
goto v_reusejp_3264_;
}
else
{
lean_object* v_reuseFailAlloc_3266_; 
v_reuseFailAlloc_3266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3266_, 0, v_a_3260_);
v___x_3265_ = v_reuseFailAlloc_3266_;
goto v_reusejp_3264_;
}
v_reusejp_3264_:
{
return v___x_3265_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_getBinderName___boxed(lean_object* v_e_3268_, lean_object* v_a_3269_, lean_object* v_a_3270_, lean_object* v_a_3271_, lean_object* v_a_3272_, lean_object* v_a_3273_){
_start:
{
lean_object* v_res_3274_; 
v_res_3274_ = lp_mathlib_Lean_Expr_getBinderName(v_e_3268_, v_a_3269_, v_a_3270_, v_a_3271_, v_a_3272_);
lean_dec(v_a_3272_);
lean_dec_ref(v_a_3271_);
lean_dec(v_a_3270_);
lean_dec_ref(v_a_3269_);
return v_res_3274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mapForallBinderNames(lean_object* v_x_3275_, lean_object* v_x_3276_){
_start:
{
if (lean_obj_tag(v_x_3275_) == 7)
{
lean_object* v_binderName_3277_; lean_object* v_binderType_3278_; lean_object* v_body_3279_; uint8_t v_binderInfo_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; lean_object* v___x_3283_; 
v_binderName_3277_ = lean_ctor_get(v_x_3275_, 0);
lean_inc(v_binderName_3277_);
v_binderType_3278_ = lean_ctor_get(v_x_3275_, 1);
lean_inc_ref(v_binderType_3278_);
v_body_3279_ = lean_ctor_get(v_x_3275_, 2);
lean_inc_ref(v_body_3279_);
v_binderInfo_3280_ = lean_ctor_get_uint8(v_x_3275_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_3275_, 3);
lean_inc_ref(v_x_3276_);
v___x_3281_ = lean_apply_1(v_x_3276_, v_binderName_3277_);
v___x_3282_ = lp_mathlib_Lean_Expr_mapForallBinderNames(v_body_3279_, v_x_3276_);
v___x_3283_ = l_Lean_Expr_forallE___override(v___x_3281_, v_binderType_3278_, v___x_3282_, v_binderInfo_3280_);
return v___x_3283_;
}
else
{
lean_dec_ref(v_x_3276_);
return v_x_3275_;
}
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__1(void){
_start:
{
lean_object* v___x_3285_; lean_object* v___x_3286_; 
v___x_3285_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkDirectProjection___closed__0));
v___x_3286_ = l_Lean_stringToMessageData(v___x_3285_);
return v___x_3286_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__3(void){
_start:
{
lean_object* v___x_3288_; lean_object* v___x_3289_; 
v___x_3288_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkDirectProjection___closed__2));
v___x_3289_ = l_Lean_stringToMessageData(v___x_3288_);
return v___x_3289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkDirectProjection(lean_object* v_e_3290_, lean_object* v_fieldName_3291_, lean_object* v_a_3292_, lean_object* v_a_3293_, lean_object* v_a_3294_, lean_object* v_a_3295_){
_start:
{
lean_object* v___x_3297_; 
lean_inc(v_a_3295_);
lean_inc_ref(v_a_3294_);
lean_inc(v_a_3293_);
lean_inc_ref(v_a_3292_);
lean_inc_ref(v_e_3290_);
v___x_3297_ = lean_infer_type(v_e_3290_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_);
if (lean_obj_tag(v___x_3297_) == 0)
{
lean_object* v_a_3298_; lean_object* v___x_3299_; 
v_a_3298_ = lean_ctor_get(v___x_3297_, 0);
lean_inc(v_a_3298_);
lean_dec_ref_known(v___x_3297_, 1);
lean_inc(v_a_3295_);
lean_inc_ref(v_a_3294_);
lean_inc(v_a_3293_);
lean_inc_ref(v_a_3292_);
v___x_3299_ = lean_whnf(v_a_3298_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_);
if (lean_obj_tag(v___x_3299_) == 0)
{
lean_object* v_a_3300_; lean_object* v___x_3302_; uint8_t v_isShared_3303_; uint8_t v_isSharedCheck_3333_; 
v_a_3300_ = lean_ctor_get(v___x_3299_, 0);
v_isSharedCheck_3333_ = !lean_is_exclusive(v___x_3299_);
if (v_isSharedCheck_3333_ == 0)
{
v___x_3302_ = v___x_3299_;
v_isShared_3303_ = v_isSharedCheck_3333_;
goto v_resetjp_3301_;
}
else
{
lean_inc(v_a_3300_);
lean_dec(v___x_3299_);
v___x_3302_ = lean_box(0);
v_isShared_3303_ = v_isSharedCheck_3333_;
goto v_resetjp_3301_;
}
v_resetjp_3301_:
{
lean_object* v___x_3304_; 
v___x_3304_ = l_Lean_Expr_getAppFn(v_a_3300_);
if (lean_obj_tag(v___x_3304_) == 4)
{
lean_object* v_declName_3305_; lean_object* v_us_3306_; lean_object* v___x_3307_; lean_object* v_env_3308_; lean_object* v___x_3309_; 
v_declName_3305_ = lean_ctor_get(v___x_3304_, 0);
lean_inc_n(v_declName_3305_, 2);
v_us_3306_ = lean_ctor_get(v___x_3304_, 1);
lean_inc(v_us_3306_);
lean_dec_ref_known(v___x_3304_, 2);
v___x_3307_ = lean_st_ref_get(v_a_3295_);
v_env_3308_ = lean_ctor_get(v___x_3307_, 0);
lean_inc_ref(v_env_3308_);
lean_dec(v___x_3307_);
lean_inc(v_fieldName_3291_);
v___x_3309_ = l_Lean_getProjFnForField_x3f(v_env_3308_, v_declName_3305_, v_fieldName_3291_);
if (lean_obj_tag(v___x_3309_) == 1)
{
lean_object* v_val_3310_; lean_object* v___x_3311_; lean_object* v_dummy_3312_; lean_object* v_nargs_3313_; lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v___x_3317_; lean_object* v___x_3318_; lean_object* v___x_3319_; lean_object* v___x_3321_; 
lean_dec(v_declName_3305_);
lean_dec(v_fieldName_3291_);
v_val_3310_ = lean_ctor_get(v___x_3309_, 0);
lean_inc(v_val_3310_);
lean_dec_ref_known(v___x_3309_, 1);
v___x_3311_ = l_Lean_Expr_const___override(v_val_3310_, v_us_3306_);
v_dummy_3312_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1);
v_nargs_3313_ = l_Lean_Expr_getAppNumArgs(v_a_3300_);
lean_inc(v_nargs_3313_);
v___x_3314_ = lean_mk_array(v_nargs_3313_, v_dummy_3312_);
v___x_3315_ = lean_unsigned_to_nat(1u);
v___x_3316_ = lean_nat_sub(v_nargs_3313_, v___x_3315_);
lean_dec(v_nargs_3313_);
v___x_3317_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_3300_, v___x_3314_, v___x_3316_);
v___x_3318_ = lean_array_push(v___x_3317_, v_e_3290_);
v___x_3319_ = l_Lean_mkAppN(v___x_3311_, v___x_3318_);
lean_dec_ref(v___x_3318_);
if (v_isShared_3303_ == 0)
{
lean_ctor_set(v___x_3302_, 0, v___x_3319_);
v___x_3321_ = v___x_3302_;
goto v_reusejp_3320_;
}
else
{
lean_object* v_reuseFailAlloc_3322_; 
v_reuseFailAlloc_3322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3322_, 0, v___x_3319_);
v___x_3321_ = v_reuseFailAlloc_3322_;
goto v_reusejp_3320_;
}
v_reusejp_3320_:
{
return v___x_3321_;
}
}
else
{
lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; 
lean_dec(v___x_3309_);
lean_dec(v_us_3306_);
lean_del_object(v___x_3302_);
lean_dec(v_a_3300_);
lean_dec_ref(v_e_3290_);
v___x_3323_ = l_Lean_MessageData_ofName(v_declName_3305_);
v___x_3324_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkDirectProjection___closed__1, &lp_mathlib_Lean_Expr_mkDirectProjection___closed__1_once, _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__1);
v___x_3325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3325_, 0, v___x_3323_);
lean_ctor_set(v___x_3325_, 1, v___x_3324_);
v___x_3326_ = l_Lean_MessageData_ofName(v_fieldName_3291_);
v___x_3327_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3327_, 0, v___x_3325_);
lean_ctor_set(v___x_3327_, 1, v___x_3326_);
v___x_3328_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3327_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_);
return v___x_3328_;
}
}
else
{
lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; 
lean_dec_ref(v___x_3304_);
lean_del_object(v___x_3302_);
lean_dec(v_a_3300_);
lean_dec(v_fieldName_3291_);
v___x_3329_ = l_Lean_MessageData_ofExpr(v_e_3290_);
v___x_3330_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkDirectProjection___closed__3, &lp_mathlib_Lean_Expr_mkDirectProjection___closed__3_once, _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__3);
v___x_3331_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3331_, 0, v___x_3329_);
lean_ctor_set(v___x_3331_, 1, v___x_3330_);
v___x_3332_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3331_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_);
return v___x_3332_;
}
}
}
else
{
lean_dec(v_fieldName_3291_);
lean_dec_ref(v_e_3290_);
return v___x_3299_;
}
}
else
{
lean_dec(v_fieldName_3291_);
lean_dec_ref(v_e_3290_);
return v___x_3297_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkDirectProjection___boxed(lean_object* v_e_3334_, lean_object* v_fieldName_3335_, lean_object* v_a_3336_, lean_object* v_a_3337_, lean_object* v_a_3338_, lean_object* v_a_3339_, lean_object* v_a_3340_){
_start:
{
lean_object* v_res_3341_; 
v_res_3341_ = lp_mathlib_Lean_Expr_mkDirectProjection(v_e_3334_, v_fieldName_3335_, v_a_3336_, v_a_3337_, v_a_3338_, v_a_3339_);
lean_dec(v_a_3339_);
lean_dec_ref(v_a_3338_);
lean_dec(v_a_3337_);
lean_dec_ref(v_a_3336_);
return v_res_3341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Expr_mkProjection_spec__1(lean_object* v_msg_3342_){
_start:
{
lean_object* v___x_3343_; lean_object* v___x_3344_; 
v___x_3343_ = lean_box(0);
v___x_3344_ = lean_panic_fn_borrowed(v___x_3343_, v_msg_3342_);
return v___x_3344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg(lean_object* v_as_x27_3345_, lean_object* v_b_3346_, lean_object* v___y_3347_, lean_object* v___y_3348_, lean_object* v___y_3349_, lean_object* v___y_3350_){
_start:
{
if (lean_obj_tag(v_as_x27_3345_) == 0)
{
lean_object* v___x_3352_; 
v___x_3352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3352_, 0, v_b_3346_);
return v___x_3352_;
}
else
{
lean_object* v_head_3353_; lean_object* v_tail_3354_; lean_object* v___x_3355_; 
v_head_3353_ = lean_ctor_get(v_as_x27_3345_, 0);
v_tail_3354_ = lean_ctor_get(v_as_x27_3345_, 1);
lean_inc(v___y_3350_);
lean_inc_ref(v___y_3349_);
lean_inc(v___y_3348_);
lean_inc_ref(v___y_3347_);
lean_inc_ref(v_b_3346_);
v___x_3355_ = lean_infer_type(v_b_3346_, v___y_3347_, v___y_3348_, v___y_3349_, v___y_3350_);
if (lean_obj_tag(v___x_3355_) == 0)
{
lean_object* v_a_3356_; lean_object* v___x_3357_; 
v_a_3356_ = lean_ctor_get(v___x_3355_, 0);
lean_inc(v_a_3356_);
lean_dec_ref_known(v___x_3355_, 1);
lean_inc(v___y_3350_);
lean_inc_ref(v___y_3349_);
lean_inc(v___y_3348_);
lean_inc_ref(v___y_3347_);
v___x_3357_ = lean_whnf(v_a_3356_, v___y_3347_, v___y_3348_, v___y_3349_, v___y_3350_);
if (lean_obj_tag(v___x_3357_) == 0)
{
lean_object* v_a_3358_; lean_object* v___x_3359_; 
v_a_3358_ = lean_ctor_get(v___x_3357_, 0);
lean_inc(v_a_3358_);
lean_dec_ref_known(v___x_3357_, 1);
v___x_3359_ = l_Lean_Expr_getAppFn(v_a_3358_);
if (lean_obj_tag(v___x_3359_) == 4)
{
lean_object* v_us_3360_; lean_object* v___x_3361_; lean_object* v_dummy_3362_; lean_object* v_nargs_3363_; lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___x_3366_; lean_object* v___x_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; 
v_us_3360_ = lean_ctor_get(v___x_3359_, 1);
lean_inc(v_us_3360_);
lean_dec_ref_known(v___x_3359_, 2);
lean_inc(v_head_3353_);
v___x_3361_ = l_Lean_Expr_const___override(v_head_3353_, v_us_3360_);
v_dummy_3362_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1);
v_nargs_3363_ = l_Lean_Expr_getAppNumArgs(v_a_3358_);
lean_inc(v_nargs_3363_);
v___x_3364_ = lean_mk_array(v_nargs_3363_, v_dummy_3362_);
v___x_3365_ = lean_unsigned_to_nat(1u);
v___x_3366_ = lean_nat_sub(v_nargs_3363_, v___x_3365_);
lean_dec(v_nargs_3363_);
v___x_3367_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_3358_, v___x_3364_, v___x_3366_);
v___x_3368_ = lean_array_push(v___x_3367_, v_b_3346_);
v___x_3369_ = l_Lean_mkAppN(v___x_3361_, v___x_3368_);
lean_dec_ref(v___x_3368_);
v_as_x27_3345_ = v_tail_3354_;
v_b_3346_ = v___x_3369_;
goto _start;
}
else
{
lean_object* v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v_a_3375_; lean_object* v___x_3377_; uint8_t v_isShared_3378_; uint8_t v_isSharedCheck_3382_; 
lean_dec_ref(v___x_3359_);
lean_dec(v_a_3358_);
v___x_3371_ = l_Lean_MessageData_ofExpr(v_b_3346_);
v___x_3372_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkDirectProjection___closed__3, &lp_mathlib_Lean_Expr_mkDirectProjection___closed__3_once, _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__3);
v___x_3373_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3373_, 0, v___x_3371_);
lean_ctor_set(v___x_3373_, 1, v___x_3372_);
v___x_3374_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3373_, v___y_3347_, v___y_3348_, v___y_3349_, v___y_3350_);
v_a_3375_ = lean_ctor_get(v___x_3374_, 0);
v_isSharedCheck_3382_ = !lean_is_exclusive(v___x_3374_);
if (v_isSharedCheck_3382_ == 0)
{
v___x_3377_ = v___x_3374_;
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
else
{
lean_inc(v_a_3375_);
lean_dec(v___x_3374_);
v___x_3377_ = lean_box(0);
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
v_resetjp_3376_:
{
lean_object* v___x_3380_; 
if (v_isShared_3378_ == 0)
{
v___x_3380_ = v___x_3377_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3381_; 
v_reuseFailAlloc_3381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3381_, 0, v_a_3375_);
v___x_3380_ = v_reuseFailAlloc_3381_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
return v___x_3380_;
}
}
}
}
else
{
lean_dec_ref(v_b_3346_);
return v___x_3357_;
}
}
else
{
lean_dec_ref(v_b_3346_);
return v___x_3355_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg___boxed(lean_object* v_as_x27_3383_, lean_object* v_b_3384_, lean_object* v___y_3385_, lean_object* v___y_3386_, lean_object* v___y_3387_, lean_object* v___y_3388_, lean_object* v___y_3389_){
_start:
{
lean_object* v_res_3390_; 
v_res_3390_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg(v_as_x27_3383_, v_b_3384_, v___y_3385_, v___y_3386_, v___y_3387_, v___y_3388_);
lean_dec(v___y_3388_);
lean_dec_ref(v___y_3387_);
lean_dec(v___y_3386_);
lean_dec_ref(v___y_3385_);
lean_dec(v_as_x27_3383_);
return v_res_3390_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_mkProjection___closed__3(void){
_start:
{
lean_object* v___x_3394_; lean_object* v___x_3395_; lean_object* v___x_3396_; lean_object* v___x_3397_; lean_object* v___x_3398_; lean_object* v___x_3399_; 
v___x_3394_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkProjection___closed__2));
v___x_3395_ = lean_unsigned_to_nat(14u);
v___x_3396_ = lean_unsigned_to_nat(22u);
v___x_3397_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkProjection___closed__1));
v___x_3398_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkProjection___closed__0));
v___x_3399_ = l_mkPanicMessageWithDecl(v___x_3398_, v___x_3397_, v___x_3396_, v___x_3395_, v___x_3394_);
return v___x_3399_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_mkProjection___closed__5(void){
_start:
{
lean_object* v___x_3401_; lean_object* v___x_3402_; 
v___x_3401_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkProjection___closed__4));
v___x_3402_ = l_Lean_stringToMessageData(v___x_3401_);
return v___x_3402_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_mkProjection___closed__7(void){
_start:
{
lean_object* v___x_3404_; lean_object* v___x_3405_; 
v___x_3404_ = ((lean_object*)(lp_mathlib_Lean_Expr_mkProjection___closed__6));
v___x_3405_ = l_Lean_stringToMessageData(v___x_3404_);
return v___x_3405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkProjection(lean_object* v_e_3406_, lean_object* v_fieldName_3407_, lean_object* v_a_3408_, lean_object* v_a_3409_, lean_object* v_a_3410_, lean_object* v_a_3411_){
_start:
{
lean_object* v___x_3413_; 
lean_inc(v_a_3411_);
lean_inc_ref(v_a_3410_);
lean_inc(v_a_3409_);
lean_inc_ref(v_a_3408_);
lean_inc_ref(v_e_3406_);
v___x_3413_ = lean_infer_type(v_e_3406_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
if (lean_obj_tag(v___x_3413_) == 0)
{
lean_object* v_a_3414_; lean_object* v___x_3415_; 
v_a_3414_ = lean_ctor_get(v___x_3413_, 0);
lean_inc(v_a_3414_);
lean_dec_ref_known(v___x_3413_, 1);
lean_inc(v_a_3411_);
lean_inc_ref(v_a_3410_);
lean_inc(v_a_3409_);
lean_inc_ref(v_a_3408_);
v___x_3415_ = lean_whnf(v_a_3414_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
if (lean_obj_tag(v___x_3415_) == 0)
{
lean_object* v_a_3416_; lean_object* v___x_3417_; 
v_a_3416_ = lean_ctor_get(v___x_3415_, 0);
lean_inc(v_a_3416_);
lean_dec_ref_known(v___x_3415_, 1);
v___x_3417_ = l_Lean_Expr_getAppFn(v_a_3416_);
lean_dec(v_a_3416_);
if (lean_obj_tag(v___x_3417_) == 4)
{
lean_object* v_declName_3418_; lean_object* v___x_3419_; lean_object* v_env_3420_; lean_object* v___x_3421_; 
v_declName_3418_ = lean_ctor_get(v___x_3417_, 0);
lean_inc_n(v_declName_3418_, 2);
lean_dec_ref_known(v___x_3417_, 2);
v___x_3419_ = lean_st_ref_get(v_a_3411_);
v_env_3420_ = lean_ctor_get(v___x_3419_, 0);
lean_inc_ref(v_env_3420_);
lean_dec(v___x_3419_);
v___x_3421_ = l_Lean_findField_x3f(v_env_3420_, v_declName_3418_, v_fieldName_3407_);
if (lean_obj_tag(v___x_3421_) == 1)
{
lean_object* v_val_3422_; lean_object* v___x_3423_; lean_object* v___y_3425_; lean_object* v_env_3429_; lean_object* v___x_3430_; 
v_val_3422_ = lean_ctor_get(v___x_3421_, 0);
lean_inc(v_val_3422_);
lean_dec_ref_known(v___x_3421_, 1);
v___x_3423_ = lean_st_ref_get(v_a_3411_);
v_env_3429_ = lean_ctor_get(v___x_3423_, 0);
lean_inc_ref(v_env_3429_);
lean_dec(v___x_3423_);
v___x_3430_ = l_Lean_getPathToBaseStructure_x3f(v_env_3429_, v_val_3422_, v_declName_3418_);
lean_dec(v_val_3422_);
if (lean_obj_tag(v___x_3430_) == 0)
{
lean_object* v___x_3431_; lean_object* v___x_3432_; 
v___x_3431_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkProjection___closed__3, &lp_mathlib_Lean_Expr_mkProjection___closed__3_once, _init_lp_mathlib_Lean_Expr_mkProjection___closed__3);
v___x_3432_ = lp_mathlib_panic___at___00Lean_Expr_mkProjection_spec__1(v___x_3431_);
v___y_3425_ = v___x_3432_;
goto v___jp_3424_;
}
else
{
lean_object* v_val_3433_; 
v_val_3433_ = lean_ctor_get(v___x_3430_, 0);
lean_inc(v_val_3433_);
lean_dec_ref_known(v___x_3430_, 1);
v___y_3425_ = v_val_3433_;
goto v___jp_3424_;
}
v___jp_3424_:
{
lean_object* v___x_3426_; 
v___x_3426_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg(v___y_3425_, v_e_3406_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
lean_dec(v___y_3425_);
if (lean_obj_tag(v___x_3426_) == 0)
{
lean_object* v_a_3427_; lean_object* v___x_3428_; 
v_a_3427_ = lean_ctor_get(v___x_3426_, 0);
lean_inc(v_a_3427_);
lean_dec_ref_known(v___x_3426_, 1);
v___x_3428_ = lp_mathlib_Lean_Expr_mkDirectProjection(v_a_3427_, v_fieldName_3407_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
return v___x_3428_;
}
else
{
lean_dec(v_fieldName_3407_);
return v___x_3426_;
}
}
}
else
{
lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___x_3441_; 
lean_dec(v___x_3421_);
lean_dec_ref(v_e_3406_);
v___x_3434_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkProjection___closed__5, &lp_mathlib_Lean_Expr_mkProjection___closed__5_once, _init_lp_mathlib_Lean_Expr_mkProjection___closed__5);
v___x_3435_ = l_Lean_MessageData_ofName(v_declName_3418_);
v___x_3436_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3436_, 0, v___x_3434_);
lean_ctor_set(v___x_3436_, 1, v___x_3435_);
v___x_3437_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkProjection___closed__7, &lp_mathlib_Lean_Expr_mkProjection___closed__7_once, _init_lp_mathlib_Lean_Expr_mkProjection___closed__7);
v___x_3438_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3438_, 0, v___x_3436_);
lean_ctor_set(v___x_3438_, 1, v___x_3437_);
v___x_3439_ = l_Lean_MessageData_ofName(v_fieldName_3407_);
v___x_3440_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3440_, 0, v___x_3438_);
lean_ctor_set(v___x_3440_, 1, v___x_3439_);
v___x_3441_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3440_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
return v___x_3441_;
}
}
else
{
lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3444_; lean_object* v___x_3445_; 
lean_dec_ref(v___x_3417_);
lean_dec(v_fieldName_3407_);
v___x_3442_ = l_Lean_MessageData_ofExpr(v_e_3406_);
v___x_3443_ = lean_obj_once(&lp_mathlib_Lean_Expr_mkDirectProjection___closed__3, &lp_mathlib_Lean_Expr_mkDirectProjection___closed__3_once, _init_lp_mathlib_Lean_Expr_mkDirectProjection___closed__3);
v___x_3444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3444_, 0, v___x_3442_);
lean_ctor_set(v___x_3444_, 1, v___x_3443_);
v___x_3445_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3444_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_);
return v___x_3445_;
}
}
else
{
lean_dec(v_fieldName_3407_);
lean_dec_ref(v_e_3406_);
return v___x_3415_;
}
}
else
{
lean_dec(v_fieldName_3407_);
lean_dec_ref(v_e_3406_);
return v___x_3413_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_mkProjection___boxed(lean_object* v_e_3446_, lean_object* v_fieldName_3447_, lean_object* v_a_3448_, lean_object* v_a_3449_, lean_object* v_a_3450_, lean_object* v_a_3451_, lean_object* v_a_3452_){
_start:
{
lean_object* v_res_3453_; 
v_res_3453_ = lp_mathlib_Lean_Expr_mkProjection(v_e_3446_, v_fieldName_3447_, v_a_3448_, v_a_3449_, v_a_3450_, v_a_3451_);
lean_dec(v_a_3451_);
lean_dec_ref(v_a_3450_);
lean_dec(v_a_3449_);
lean_dec_ref(v_a_3448_);
return v_res_3453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0(lean_object* v_as_3454_, lean_object* v_as_x27_3455_, lean_object* v_b_3456_, lean_object* v_a_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_, lean_object* v___y_3461_){
_start:
{
lean_object* v___x_3463_; 
v___x_3463_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___redArg(v_as_x27_3455_, v_b_3456_, v___y_3458_, v___y_3459_, v___y_3460_, v___y_3461_);
return v___x_3463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0___boxed(lean_object* v_as_3464_, lean_object* v_as_x27_3465_, lean_object* v_b_3466_, lean_object* v_a_3467_, lean_object* v___y_3468_, lean_object* v___y_3469_, lean_object* v___y_3470_, lean_object* v___y_3471_, lean_object* v___y_3472_){
_start:
{
lean_object* v_res_3473_; 
v_res_3473_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Expr_mkProjection_spec__0(v_as_3464_, v_as_x27_3465_, v_b_3466_, v_a_3467_, v___y_3468_, v___y_3469_, v___y_3470_, v___y_3471_);
lean_dec(v___y_3471_);
lean_dec_ref(v___y_3470_);
lean_dec(v___y_3469_);
lean_dec_ref(v___y_3468_);
lean_dec(v_as_x27_3465_);
lean_dec(v_as_3464_);
return v_res_3473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg(lean_object* v_declName_3474_, lean_object* v___y_3475_){
_start:
{
lean_object* v___x_3477_; lean_object* v_env_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; 
v___x_3477_ = lean_st_ref_get(v___y_3475_);
v_env_3478_ = lean_ctor_get(v___x_3477_, 0);
lean_inc_ref(v_env_3478_);
lean_dec(v___x_3477_);
v___x_3479_ = l_Lean_Environment_getProjectionFnInfo_x3f(v_env_3478_, v_declName_3474_);
v___x_3480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3480_, 0, v___x_3479_);
return v___x_3480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg___boxed(lean_object* v_declName_3481_, lean_object* v___y_3482_, lean_object* v___y_3483_){
_start:
{
lean_object* v_res_3484_; 
v_res_3484_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg(v_declName_3481_, v___y_3482_);
lean_dec(v___y_3482_);
return v_res_3484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0(lean_object* v_declName_3485_, lean_object* v___y_3486_, lean_object* v___y_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_){
_start:
{
lean_object* v___x_3491_; 
v___x_3491_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg(v_declName_3485_, v___y_3489_);
return v___x_3491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___boxed(lean_object* v_declName_3492_, lean_object* v___y_3493_, lean_object* v___y_3494_, lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_){
_start:
{
lean_object* v_res_3498_; 
v_res_3498_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0(v_declName_3492_, v___y_3493_, v___y_3494_, v___y_3495_, v___y_3496_);
lean_dec(v___y_3496_);
lean_dec_ref(v___y_3495_);
lean_dec(v___y_3494_);
lean_dec_ref(v___y_3493_);
return v_res_3498_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1(void){
_start:
{
lean_object* v___x_3500_; lean_object* v___x_3501_; 
v___x_3500_ = ((lean_object*)(lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__0));
v___x_3501_ = l_Lean_stringToMessageData(v___x_3500_);
return v___x_3501_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3(void){
_start:
{
lean_object* v___x_3503_; lean_object* v___x_3504_; 
v___x_3503_ = ((lean_object*)(lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__2));
v___x_3504_ = l_Lean_stringToMessageData(v___x_3503_);
return v___x_3504_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5(void){
_start:
{
lean_object* v___x_3506_; lean_object* v___x_3507_; 
v___x_3506_ = ((lean_object*)(lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__4));
v___x_3507_ = l_Lean_stringToMessageData(v___x_3506_);
return v___x_3507_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7(void){
_start:
{
lean_object* v___x_3509_; lean_object* v___x_3510_; 
v___x_3509_ = ((lean_object*)(lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__6));
v___x_3510_ = l_Lean_stringToMessageData(v___x_3509_);
return v___x_3510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f(lean_object* v_e_3511_, lean_object* v_a_3512_, lean_object* v_a_3513_, lean_object* v_a_3514_, lean_object* v_a_3515_){
_start:
{
lean_object* v___x_3517_; 
v___x_3517_ = l_Lean_Expr_getAppFn(v_e_3511_);
if (lean_obj_tag(v___x_3517_) == 4)
{
lean_object* v_declName_3518_; lean_object* v___x_3519_; lean_object* v_a_3520_; lean_object* v___x_3522_; uint8_t v_isShared_3523_; uint8_t v_isSharedCheck_3587_; 
v_declName_3518_ = lean_ctor_get(v___x_3517_, 0);
lean_inc_n(v_declName_3518_, 2);
lean_dec_ref_known(v___x_3517_, 2);
v___x_3519_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Lean_Expr_reduceProjStruct_x3f_spec__0___redArg(v_declName_3518_, v_a_3515_);
v_a_3520_ = lean_ctor_get(v___x_3519_, 0);
v_isSharedCheck_3587_ = !lean_is_exclusive(v___x_3519_);
if (v_isSharedCheck_3587_ == 0)
{
v___x_3522_ = v___x_3519_;
v_isShared_3523_ = v_isSharedCheck_3587_;
goto v_resetjp_3521_;
}
else
{
lean_inc(v_a_3520_);
lean_dec(v___x_3519_);
v___x_3522_ = lean_box(0);
v_isShared_3523_ = v_isSharedCheck_3587_;
goto v_resetjp_3521_;
}
v_resetjp_3521_:
{
if (lean_obj_tag(v_a_3520_) == 1)
{
lean_object* v_val_3524_; lean_object* v___x_3526_; uint8_t v_isShared_3527_; uint8_t v_isSharedCheck_3582_; 
v_val_3524_ = lean_ctor_get(v_a_3520_, 0);
v_isSharedCheck_3582_ = !lean_is_exclusive(v_a_3520_);
if (v_isSharedCheck_3582_ == 0)
{
v___x_3526_ = v_a_3520_;
v_isShared_3527_ = v_isSharedCheck_3582_;
goto v_resetjp_3525_;
}
else
{
lean_inc(v_val_3524_);
lean_dec(v_a_3520_);
v___x_3526_ = lean_box(0);
v_isShared_3527_ = v_isSharedCheck_3582_;
goto v_resetjp_3525_;
}
v_resetjp_3525_:
{
lean_object* v_ctorName_3528_; lean_object* v_numParams_3529_; lean_object* v_i_3530_; lean_object* v_nargs_3531_; lean_object* v_dummy_3532_; lean_object* v___x_3533_; lean_object* v___x_3534_; lean_object* v___x_3535_; lean_object* v___x_3536_; lean_object* v___x_3537_; lean_object* v___x_3538_; uint8_t v___x_3539_; 
v_ctorName_3528_ = lean_ctor_get(v_val_3524_, 0);
lean_inc(v_ctorName_3528_);
v_numParams_3529_ = lean_ctor_get(v_val_3524_, 1);
lean_inc(v_numParams_3529_);
v_i_3530_ = lean_ctor_get(v_val_3524_, 2);
lean_inc(v_i_3530_);
lean_dec(v_val_3524_);
v_nargs_3531_ = l_Lean_Expr_getAppNumArgs(v_e_3511_);
v_dummy_3532_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Lean_Expr_eraseProofs_spec__0_spec__0___lam__1___closed__1);
lean_inc(v_nargs_3531_);
v___x_3533_ = lean_mk_array(v_nargs_3531_, v_dummy_3532_);
v___x_3534_ = lean_unsigned_to_nat(1u);
v___x_3535_ = lean_nat_sub(v_nargs_3531_, v___x_3534_);
lean_dec(v_nargs_3531_);
v___x_3536_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_e_3511_, v___x_3533_, v___x_3535_);
v___x_3537_ = lean_array_get_size(v___x_3536_);
v___x_3538_ = lean_nat_add(v_numParams_3529_, v___x_3534_);
v___x_3539_ = lean_nat_dec_eq(v___x_3537_, v___x_3538_);
lean_dec(v___x_3538_);
if (v___x_3539_ == 0)
{
lean_object* v___x_3540_; lean_object* v___x_3542_; 
lean_dec_ref(v___x_3536_);
lean_dec(v_i_3530_);
lean_dec(v_numParams_3529_);
lean_dec(v_ctorName_3528_);
lean_del_object(v___x_3526_);
lean_dec(v_declName_3518_);
v___x_3540_ = lean_box(0);
if (v_isShared_3523_ == 0)
{
lean_ctor_set(v___x_3522_, 0, v___x_3540_);
v___x_3542_ = v___x_3522_;
goto v_reusejp_3541_;
}
else
{
lean_object* v_reuseFailAlloc_3543_; 
v_reuseFailAlloc_3543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3543_, 0, v___x_3540_);
v___x_3542_ = v_reuseFailAlloc_3543_;
goto v_reusejp_3541_;
}
v_reusejp_3541_:
{
return v___x_3542_;
}
}
else
{
lean_object* v___x_3544_; lean_object* v___x_3545_; uint8_t v___x_3546_; 
v___x_3544_ = lean_array_fget(v___x_3536_, v_numParams_3529_);
lean_dec_ref(v___x_3536_);
v___x_3545_ = l_Lean_Expr_getAppFn(v___x_3544_);
v___x_3546_ = l_Lean_Expr_isConstOf(v___x_3545_, v_ctorName_3528_);
lean_dec(v_ctorName_3528_);
lean_dec_ref(v___x_3545_);
if (v___x_3546_ == 0)
{
lean_object* v___x_3547_; lean_object* v___x_3549_; 
lean_dec(v___x_3544_);
lean_dec(v_i_3530_);
lean_dec(v_numParams_3529_);
lean_del_object(v___x_3526_);
lean_dec(v_declName_3518_);
v___x_3547_ = lean_box(0);
if (v_isShared_3523_ == 0)
{
lean_ctor_set(v___x_3522_, 0, v___x_3547_);
v___x_3549_ = v___x_3522_;
goto v_reusejp_3548_;
}
else
{
lean_object* v_reuseFailAlloc_3550_; 
v_reuseFailAlloc_3550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3550_, 0, v___x_3547_);
v___x_3549_ = v_reuseFailAlloc_3550_;
goto v_reusejp_3548_;
}
v_reusejp_3548_:
{
return v___x_3549_;
}
}
else
{
lean_object* v_nargs_3551_; lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3554_; lean_object* v___x_3555_; lean_object* v___x_3556_; uint8_t v___x_3557_; 
v_nargs_3551_ = l_Lean_Expr_getAppNumArgs(v___x_3544_);
lean_inc(v_nargs_3551_);
v___x_3552_ = lean_mk_array(v_nargs_3551_, v_dummy_3532_);
v___x_3553_ = lean_nat_sub(v_nargs_3551_, v___x_3534_);
lean_dec(v_nargs_3551_);
lean_inc(v___x_3544_);
v___x_3554_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v___x_3544_, v___x_3552_, v___x_3553_);
v___x_3555_ = lean_nat_add(v_numParams_3529_, v_i_3530_);
lean_dec(v_numParams_3529_);
v___x_3556_ = lean_array_get_size(v___x_3554_);
v___x_3557_ = lean_nat_dec_lt(v___x_3555_, v___x_3556_);
if (v___x_3557_ == 0)
{
lean_object* v___x_3558_; lean_object* v___x_3559_; lean_object* v___x_3560_; lean_object* v___x_3561_; lean_object* v___x_3562_; lean_object* v___x_3563_; lean_object* v___x_3564_; lean_object* v___x_3565_; lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3568_; lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3574_; 
lean_dec(v___x_3555_);
lean_dec_ref(v___x_3554_);
lean_del_object(v___x_3526_);
lean_del_object(v___x_3522_);
v___x_3558_ = lean_obj_once(&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1, &lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1_once, _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__1);
v___x_3559_ = l_Lean_MessageData_ofName(v_declName_3518_);
v___x_3560_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3560_, 0, v___x_3558_);
lean_ctor_set(v___x_3560_, 1, v___x_3559_);
v___x_3561_ = lean_obj_once(&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3, &lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3_once, _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__3);
v___x_3562_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3562_, 0, v___x_3560_);
lean_ctor_set(v___x_3562_, 1, v___x_3561_);
v___x_3563_ = lean_nat_add(v_i_3530_, v___x_3534_);
lean_dec(v_i_3530_);
v___x_3564_ = l_Nat_reprFast(v___x_3563_);
v___x_3565_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3565_, 0, v___x_3564_);
v___x_3566_ = l_Lean_MessageData_ofFormat(v___x_3565_);
v___x_3567_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3567_, 0, v___x_3562_);
lean_ctor_set(v___x_3567_, 1, v___x_3566_);
v___x_3568_ = lean_obj_once(&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5, &lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5_once, _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__5);
v___x_3569_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3569_, 0, v___x_3567_);
lean_ctor_set(v___x_3569_, 1, v___x_3568_);
v___x_3570_ = l_Lean_MessageData_ofExpr(v___x_3544_);
v___x_3571_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3571_, 0, v___x_3569_);
lean_ctor_set(v___x_3571_, 1, v___x_3570_);
v___x_3572_ = lean_obj_once(&lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7, &lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7_once, _init_lp_mathlib_Lean_Expr_reduceProjStruct_x3f___closed__7);
v___x_3573_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3573_, 0, v___x_3571_);
lean_ctor_set(v___x_3573_, 1, v___x_3572_);
v___x_3574_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3573_, v_a_3512_, v_a_3513_, v_a_3514_, v_a_3515_);
return v___x_3574_;
}
else
{
lean_object* v___x_3575_; lean_object* v___x_3577_; 
lean_dec(v___x_3544_);
lean_dec(v_i_3530_);
lean_dec(v_declName_3518_);
v___x_3575_ = lean_array_fget(v___x_3554_, v___x_3555_);
lean_dec(v___x_3555_);
lean_dec_ref(v___x_3554_);
if (v_isShared_3527_ == 0)
{
lean_ctor_set(v___x_3526_, 0, v___x_3575_);
v___x_3577_ = v___x_3526_;
goto v_reusejp_3576_;
}
else
{
lean_object* v_reuseFailAlloc_3581_; 
v_reuseFailAlloc_3581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3581_, 0, v___x_3575_);
v___x_3577_ = v_reuseFailAlloc_3581_;
goto v_reusejp_3576_;
}
v_reusejp_3576_:
{
lean_object* v___x_3579_; 
if (v_isShared_3523_ == 0)
{
lean_ctor_set(v___x_3522_, 0, v___x_3577_);
v___x_3579_ = v___x_3522_;
goto v_reusejp_3578_;
}
else
{
lean_object* v_reuseFailAlloc_3580_; 
v_reuseFailAlloc_3580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3580_, 0, v___x_3577_);
v___x_3579_ = v_reuseFailAlloc_3580_;
goto v_reusejp_3578_;
}
v_reusejp_3578_:
{
return v___x_3579_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3583_; lean_object* v___x_3585_; 
lean_dec(v_a_3520_);
lean_dec(v_declName_3518_);
lean_dec_ref(v_e_3511_);
v___x_3583_ = lean_box(0);
if (v_isShared_3523_ == 0)
{
lean_ctor_set(v___x_3522_, 0, v___x_3583_);
v___x_3585_ = v___x_3522_;
goto v_reusejp_3584_;
}
else
{
lean_object* v_reuseFailAlloc_3586_; 
v_reuseFailAlloc_3586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3586_, 0, v___x_3583_);
v___x_3585_ = v_reuseFailAlloc_3586_;
goto v_reusejp_3584_;
}
v_reusejp_3584_:
{
return v___x_3585_;
}
}
}
}
else
{
lean_object* v___x_3588_; lean_object* v___x_3589_; 
lean_dec_ref(v___x_3517_);
lean_dec_ref(v_e_3511_);
v___x_3588_ = lean_box(0);
v___x_3589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3589_, 0, v___x_3588_);
return v___x_3589_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f___boxed(lean_object* v_e_3590_, lean_object* v_a_3591_, lean_object* v_a_3592_, lean_object* v_a_3593_, lean_object* v_a_3594_, lean_object* v_a_3595_){
_start:
{
lean_object* v_res_3596_; 
v_res_3596_ = lp_mathlib_Lean_Expr_reduceProjStruct_x3f(v_e_3590_, v_a_3591_, v_a_3592_, v_a_3593_, v_a_3594_);
lean_dec(v_a_3594_);
lean_dec_ref(v_a_3593_);
lean_dec(v_a_3592_);
lean_dec_ref(v_a_3591_);
return v_res_3596_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___lam__0(lean_object* v_p_3597_, lean_object* v_x_3598_){
_start:
{
if (lean_obj_tag(v_x_3598_) == 4)
{
lean_object* v_declName_3599_; lean_object* v___x_3600_; uint8_t v___x_3601_; 
v_declName_3599_ = lean_ctor_get(v_x_3598_, 0);
lean_inc(v_declName_3599_);
lean_dec_ref_known(v_x_3598_, 2);
v___x_3600_ = lean_apply_1(v_p_3597_, v_declName_3599_);
v___x_3601_ = lean_unbox(v___x_3600_);
return v___x_3601_;
}
else
{
uint8_t v___x_3602_; 
lean_dec_ref(v_x_3598_);
lean_dec_ref(v_p_3597_);
v___x_3602_ = 0;
return v___x_3602_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___lam__0___boxed(lean_object* v_p_3603_, lean_object* v_x_3604_){
_start:
{
uint8_t v_res_3605_; lean_object* v_r_3606_; 
v_res_3605_ = lp_mathlib_Lean_Expr_containsConst___lam__0(v_p_3603_, v_x_3604_);
v_r_3606_ = lean_box(v_res_3605_);
return v_r_3606_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst(lean_object* v_e_3607_, lean_object* v_p_3608_){
_start:
{
lean_object* v___f_3609_; lean_object* v___x_3610_; 
v___f_3609_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_containsConst___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3609_, 0, v_p_3608_);
v___x_3610_ = lean_find_expr(v___f_3609_, v_e_3607_);
lean_dec_ref(v___f_3609_);
if (lean_obj_tag(v___x_3610_) == 0)
{
uint8_t v___x_3611_; 
v___x_3611_ = 0;
return v___x_3611_;
}
else
{
uint8_t v___x_3612_; 
lean_dec_ref_known(v___x_3610_, 1);
v___x_3612_ = 1;
return v___x_3612_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___boxed(lean_object* v_e_3613_, lean_object* v_p_3614_){
_start:
{
uint8_t v_res_3615_; lean_object* v_r_3616_; 
v_res_3615_ = lp_mathlib_Lean_Expr_containsConst(v_e_3613_, v_p_3614_);
lean_dec_ref(v_e_3613_);
v_r_3616_ = lean_box(v_res_3615_);
return v_r_3616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0(lean_object* v_k_3617_, lean_object* v_b_3618_, lean_object* v___y_3619_, lean_object* v___y_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_){
_start:
{
lean_object* v___x_3624_; 
lean_inc(v___y_3622_);
lean_inc_ref(v___y_3621_);
lean_inc(v___y_3620_);
lean_inc_ref(v___y_3619_);
v___x_3624_ = lean_apply_6(v_k_3617_, v_b_3618_, v___y_3619_, v___y_3620_, v___y_3621_, v___y_3622_, lean_box(0));
return v___x_3624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_3625_, lean_object* v_b_3626_, lean_object* v___y_3627_, lean_object* v___y_3628_, lean_object* v___y_3629_, lean_object* v___y_3630_, lean_object* v___y_3631_){
_start:
{
lean_object* v_res_3632_; 
v_res_3632_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0(v_k_3625_, v_b_3626_, v___y_3627_, v___y_3628_, v___y_3629_, v___y_3630_);
lean_dec(v___y_3630_);
lean_dec_ref(v___y_3629_);
lean_dec(v___y_3628_);
lean_dec_ref(v___y_3627_);
return v_res_3632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg(lean_object* v_name_3633_, uint8_t v_bi_3634_, lean_object* v_type_3635_, lean_object* v_k_3636_, uint8_t v_kind_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_){
_start:
{
lean_object* v___f_3643_; lean_object* v___x_3644_; 
v___f_3643_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3643_, 0, v_k_3636_);
v___x_3644_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_3633_, v_bi_3634_, v_type_3635_, v___f_3643_, v_kind_3637_, v___y_3638_, v___y_3639_, v___y_3640_, v___y_3641_);
if (lean_obj_tag(v___x_3644_) == 0)
{
lean_object* v_a_3645_; lean_object* v___x_3647_; uint8_t v_isShared_3648_; uint8_t v_isSharedCheck_3652_; 
v_a_3645_ = lean_ctor_get(v___x_3644_, 0);
v_isSharedCheck_3652_ = !lean_is_exclusive(v___x_3644_);
if (v_isSharedCheck_3652_ == 0)
{
v___x_3647_ = v___x_3644_;
v_isShared_3648_ = v_isSharedCheck_3652_;
goto v_resetjp_3646_;
}
else
{
lean_inc(v_a_3645_);
lean_dec(v___x_3644_);
v___x_3647_ = lean_box(0);
v_isShared_3648_ = v_isSharedCheck_3652_;
goto v_resetjp_3646_;
}
v_resetjp_3646_:
{
lean_object* v___x_3650_; 
if (v_isShared_3648_ == 0)
{
v___x_3650_ = v___x_3647_;
goto v_reusejp_3649_;
}
else
{
lean_object* v_reuseFailAlloc_3651_; 
v_reuseFailAlloc_3651_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3651_, 0, v_a_3645_);
v___x_3650_ = v_reuseFailAlloc_3651_;
goto v_reusejp_3649_;
}
v_reusejp_3649_:
{
return v___x_3650_;
}
}
}
else
{
lean_object* v_a_3653_; lean_object* v___x_3655_; uint8_t v_isShared_3656_; uint8_t v_isSharedCheck_3660_; 
v_a_3653_ = lean_ctor_get(v___x_3644_, 0);
v_isSharedCheck_3660_ = !lean_is_exclusive(v___x_3644_);
if (v_isSharedCheck_3660_ == 0)
{
v___x_3655_ = v___x_3644_;
v_isShared_3656_ = v_isSharedCheck_3660_;
goto v_resetjp_3654_;
}
else
{
lean_inc(v_a_3653_);
lean_dec(v___x_3644_);
v___x_3655_ = lean_box(0);
v_isShared_3656_ = v_isSharedCheck_3660_;
goto v_resetjp_3654_;
}
v_resetjp_3654_:
{
lean_object* v___x_3658_; 
if (v_isShared_3656_ == 0)
{
v___x_3658_ = v___x_3655_;
goto v_reusejp_3657_;
}
else
{
lean_object* v_reuseFailAlloc_3659_; 
v_reuseFailAlloc_3659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3659_, 0, v_a_3653_);
v___x_3658_ = v_reuseFailAlloc_3659_;
goto v_reusejp_3657_;
}
v_reusejp_3657_:
{
return v___x_3658_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg___boxed(lean_object* v_name_3661_, lean_object* v_bi_3662_, lean_object* v_type_3663_, lean_object* v_k_3664_, lean_object* v_kind_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_, lean_object* v___y_3670_){
_start:
{
uint8_t v_bi_boxed_3671_; uint8_t v_kind_boxed_3672_; lean_object* v_res_3673_; 
v_bi_boxed_3671_ = lean_unbox(v_bi_3662_);
v_kind_boxed_3672_ = lean_unbox(v_kind_3665_);
v_res_3673_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg(v_name_3661_, v_bi_boxed_3671_, v_type_3663_, v_k_3664_, v_kind_boxed_3672_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
lean_dec(v___y_3669_);
lean_dec_ref(v___y_3668_);
lean_dec(v___y_3667_);
lean_dec_ref(v___y_3666_);
return v_res_3673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(lean_object* v_name_3674_, lean_object* v_type_3675_, lean_object* v_k_3676_, lean_object* v___y_3677_, lean_object* v___y_3678_, lean_object* v___y_3679_, lean_object* v___y_3680_){
_start:
{
uint8_t v___x_3682_; uint8_t v___x_3683_; lean_object* v___x_3684_; 
v___x_3682_ = 0;
v___x_3683_ = 0;
v___x_3684_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg(v_name_3674_, v___x_3682_, v_type_3675_, v_k_3676_, v___x_3683_, v___y_3677_, v___y_3678_, v___y_3679_, v___y_3680_);
return v___x_3684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg___boxed(lean_object* v_name_3685_, lean_object* v_type_3686_, lean_object* v_k_3687_, lean_object* v___y_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_, lean_object* v___y_3692_){
_start:
{
lean_object* v_res_3693_; 
v_res_3693_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(v_name_3685_, v_type_3686_, v_k_3687_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_);
lean_dec(v___y_3691_);
lean_dec_ref(v___y_3690_);
lean_dec(v___y_3689_);
lean_dec_ref(v___y_3688_);
return v_res_3693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0(lean_object* v_head_3704_, lean_object* v_arg_3705_, lean_object* v_arg_3706_, lean_object* v___x_3707_, uint8_t v___x_3708_, lean_object* v___x_3709_, lean_object* v___x_3710_, lean_object* v_x_3711_, lean_object* v_hNotPx_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_, lean_object* v___y_3716_){
_start:
{
lean_object* v___x_3718_; 
lean_inc_ref(v_hNotPx_3712_);
v___x_3718_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go(v_head_3704_, v_arg_3705_, v_arg_3706_, v_hNotPx_3712_, v___y_3713_, v___y_3714_, v___y_3715_, v___y_3716_);
if (lean_obj_tag(v___x_3718_) == 0)
{
lean_object* v_a_3719_; lean_object* v_fst_3720_; lean_object* v_snd_3721_; lean_object* v___x_3723_; uint8_t v_isShared_3724_; uint8_t v_isSharedCheck_3770_; 
v_a_3719_ = lean_ctor_get(v___x_3718_, 0);
lean_inc(v_a_3719_);
lean_dec_ref_known(v___x_3718_, 1);
v_fst_3720_ = lean_ctor_get(v_a_3719_, 0);
v_snd_3721_ = lean_ctor_get(v_a_3719_, 1);
v_isSharedCheck_3770_ = !lean_is_exclusive(v_a_3719_);
if (v_isSharedCheck_3770_ == 0)
{
v___x_3723_ = v_a_3719_;
v_isShared_3724_ = v_isSharedCheck_3770_;
goto v_resetjp_3722_;
}
else
{
lean_inc(v_snd_3721_);
lean_inc(v_fst_3720_);
lean_dec(v_a_3719_);
v___x_3723_ = lean_box(0);
v_isShared_3724_ = v_isSharedCheck_3770_;
goto v_resetjp_3722_;
}
v_resetjp_3722_:
{
uint8_t v___x_3725_; uint8_t v___x_3726_; lean_object* v___x_3727_; 
v___x_3725_ = 0;
v___x_3726_ = 1;
v___x_3727_ = l_Lean_Meta_mkForallFVars(v___x_3707_, v_fst_3720_, v___x_3725_, v___x_3708_, v___x_3708_, v___x_3726_, v___y_3713_, v___y_3714_, v___y_3715_, v___y_3716_);
if (lean_obj_tag(v___x_3727_) == 0)
{
lean_object* v_a_3728_; lean_object* v___x_3729_; lean_object* v___x_3730_; 
v_a_3728_ = lean_ctor_get(v___x_3727_, 0);
lean_inc(v_a_3728_);
lean_dec_ref_known(v___x_3727_, 1);
v___x_3729_ = lean_array_push(v___x_3709_, v_hNotPx_3712_);
v___x_3730_ = l_Lean_Meta_mkLambdaFVars(v___x_3729_, v_snd_3721_, v___x_3725_, v___x_3708_, v___x_3725_, v___x_3708_, v___x_3726_, v___y_3713_, v___y_3714_, v___y_3715_, v___y_3716_);
lean_dec_ref(v___x_3729_);
if (lean_obj_tag(v___x_3730_) == 0)
{
lean_object* v_a_3731_; lean_object* v___x_3732_; lean_object* v___x_3733_; lean_object* v___x_3734_; 
v_a_3731_ = lean_ctor_get(v___x_3730_, 0);
lean_inc(v_a_3731_);
lean_dec_ref_known(v___x_3730_, 1);
v___x_3732_ = l_Lean_Expr_app___override(v___x_3710_, v_x_3711_);
v___x_3733_ = l_Lean_Expr_app___override(v_a_3731_, v___x_3732_);
v___x_3734_ = l_Lean_Meta_mkLambdaFVars(v___x_3707_, v___x_3733_, v___x_3725_, v___x_3708_, v___x_3725_, v___x_3708_, v___x_3726_, v___y_3713_, v___y_3714_, v___y_3715_, v___y_3716_);
if (lean_obj_tag(v___x_3734_) == 0)
{
lean_object* v_a_3735_; lean_object* v___x_3737_; uint8_t v_isShared_3738_; uint8_t v_isSharedCheck_3745_; 
v_a_3735_ = lean_ctor_get(v___x_3734_, 0);
v_isSharedCheck_3745_ = !lean_is_exclusive(v___x_3734_);
if (v_isSharedCheck_3745_ == 0)
{
v___x_3737_ = v___x_3734_;
v_isShared_3738_ = v_isSharedCheck_3745_;
goto v_resetjp_3736_;
}
else
{
lean_inc(v_a_3735_);
lean_dec(v___x_3734_);
v___x_3737_ = lean_box(0);
v_isShared_3738_ = v_isSharedCheck_3745_;
goto v_resetjp_3736_;
}
v_resetjp_3736_:
{
lean_object* v___x_3740_; 
if (v_isShared_3724_ == 0)
{
lean_ctor_set(v___x_3723_, 1, v_a_3735_);
lean_ctor_set(v___x_3723_, 0, v_a_3728_);
v___x_3740_ = v___x_3723_;
goto v_reusejp_3739_;
}
else
{
lean_object* v_reuseFailAlloc_3744_; 
v_reuseFailAlloc_3744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3744_, 0, v_a_3728_);
lean_ctor_set(v_reuseFailAlloc_3744_, 1, v_a_3735_);
v___x_3740_ = v_reuseFailAlloc_3744_;
goto v_reusejp_3739_;
}
v_reusejp_3739_:
{
lean_object* v___x_3742_; 
if (v_isShared_3738_ == 0)
{
lean_ctor_set(v___x_3737_, 0, v___x_3740_);
v___x_3742_ = v___x_3737_;
goto v_reusejp_3741_;
}
else
{
lean_object* v_reuseFailAlloc_3743_; 
v_reuseFailAlloc_3743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3743_, 0, v___x_3740_);
v___x_3742_ = v_reuseFailAlloc_3743_;
goto v_reusejp_3741_;
}
v_reusejp_3741_:
{
return v___x_3742_;
}
}
}
}
else
{
lean_object* v_a_3746_; lean_object* v___x_3748_; uint8_t v_isShared_3749_; uint8_t v_isSharedCheck_3753_; 
lean_dec(v_a_3728_);
lean_del_object(v___x_3723_);
v_a_3746_ = lean_ctor_get(v___x_3734_, 0);
v_isSharedCheck_3753_ = !lean_is_exclusive(v___x_3734_);
if (v_isSharedCheck_3753_ == 0)
{
v___x_3748_ = v___x_3734_;
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
else
{
lean_inc(v_a_3746_);
lean_dec(v___x_3734_);
v___x_3748_ = lean_box(0);
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
v_resetjp_3747_:
{
lean_object* v___x_3751_; 
if (v_isShared_3749_ == 0)
{
v___x_3751_ = v___x_3748_;
goto v_reusejp_3750_;
}
else
{
lean_object* v_reuseFailAlloc_3752_; 
v_reuseFailAlloc_3752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3752_, 0, v_a_3746_);
v___x_3751_ = v_reuseFailAlloc_3752_;
goto v_reusejp_3750_;
}
v_reusejp_3750_:
{
return v___x_3751_;
}
}
}
}
else
{
lean_object* v_a_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3761_; 
lean_dec(v_a_3728_);
lean_del_object(v___x_3723_);
lean_dec_ref(v_x_3711_);
lean_dec_ref(v___x_3710_);
v_a_3754_ = lean_ctor_get(v___x_3730_, 0);
v_isSharedCheck_3761_ = !lean_is_exclusive(v___x_3730_);
if (v_isSharedCheck_3761_ == 0)
{
v___x_3756_ = v___x_3730_;
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_a_3754_);
lean_dec(v___x_3730_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___x_3759_; 
if (v_isShared_3757_ == 0)
{
v___x_3759_ = v___x_3756_;
goto v_reusejp_3758_;
}
else
{
lean_object* v_reuseFailAlloc_3760_; 
v_reuseFailAlloc_3760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3760_, 0, v_a_3754_);
v___x_3759_ = v_reuseFailAlloc_3760_;
goto v_reusejp_3758_;
}
v_reusejp_3758_:
{
return v___x_3759_;
}
}
}
}
else
{
lean_object* v_a_3762_; lean_object* v___x_3764_; uint8_t v_isShared_3765_; uint8_t v_isSharedCheck_3769_; 
lean_del_object(v___x_3723_);
lean_dec(v_snd_3721_);
lean_dec_ref(v_hNotPx_3712_);
lean_dec_ref(v_x_3711_);
lean_dec_ref(v___x_3710_);
lean_dec_ref(v___x_3709_);
v_a_3762_ = lean_ctor_get(v___x_3727_, 0);
v_isSharedCheck_3769_ = !lean_is_exclusive(v___x_3727_);
if (v_isSharedCheck_3769_ == 0)
{
v___x_3764_ = v___x_3727_;
v_isShared_3765_ = v_isSharedCheck_3769_;
goto v_resetjp_3763_;
}
else
{
lean_inc(v_a_3762_);
lean_dec(v___x_3727_);
v___x_3764_ = lean_box(0);
v_isShared_3765_ = v_isSharedCheck_3769_;
goto v_resetjp_3763_;
}
v_resetjp_3763_:
{
lean_object* v___x_3767_; 
if (v_isShared_3765_ == 0)
{
v___x_3767_ = v___x_3764_;
goto v_reusejp_3766_;
}
else
{
lean_object* v_reuseFailAlloc_3768_; 
v_reuseFailAlloc_3768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3768_, 0, v_a_3762_);
v___x_3767_ = v_reuseFailAlloc_3768_;
goto v_reusejp_3766_;
}
v_reusejp_3766_:
{
return v___x_3767_;
}
}
}
}
}
else
{
lean_dec_ref(v_hNotPx_3712_);
lean_dec_ref(v_x_3711_);
lean_dec_ref(v___x_3710_);
lean_dec_ref(v___x_3709_);
return v___x_3718_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0___boxed(lean_object* v_head_3771_, lean_object* v_arg_3772_, lean_object* v_arg_3773_, lean_object* v___x_3774_, lean_object* v___x_3775_, lean_object* v___x_3776_, lean_object* v___x_3777_, lean_object* v_x_3778_, lean_object* v_hNotPx_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_, lean_object* v___y_3782_, lean_object* v___y_3783_, lean_object* v___y_3784_){
_start:
{
uint8_t v___x_1808__boxed_3785_; lean_object* v_res_3786_; 
v___x_1808__boxed_3785_ = lean_unbox(v___x_3775_);
v_res_3786_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0(v_head_3771_, v_arg_3772_, v_arg_3773_, v___x_3774_, v___x_1808__boxed_3785_, v___x_3776_, v___x_3777_, v_x_3778_, v_hNotPx_3779_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
lean_dec(v___y_3783_);
lean_dec_ref(v___y_3782_);
lean_dec(v___y_3781_);
lean_dec_ref(v___y_3780_);
lean_dec_ref(v___x_3774_);
return v_res_3786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1(lean_object* v_p_3787_, lean_object* v_lvl_3788_, lean_object* v_A_3789_, lean_object* v_hNotEx_3790_, lean_object* v_x_3791_, lean_object* v___y_3792_, lean_object* v___y_3793_, lean_object* v___y_3794_, lean_object* v___y_3795_){
_start:
{
lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; 
v___x_3797_ = lean_unsigned_to_nat(1u);
v___x_3798_ = lean_mk_empty_array_with_capacity(v___x_3797_);
lean_inc_ref(v_x_3791_);
lean_inc_ref(v___x_3798_);
v___x_3799_ = lean_array_push(v___x_3798_, v_x_3791_);
lean_inc_ref(v___x_3799_);
lean_inc_ref(v_p_3787_);
v___x_3800_ = l_Lean_Expr_beta(v_p_3787_, v___x_3799_);
lean_inc_ref(v___x_3800_);
v___x_3801_ = l_Lean_mkNot(v___x_3800_);
v___x_3802_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__1));
v___x_3803_ = lean_box(0);
v___x_3804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3804_, 0, v_lvl_3788_);
lean_ctor_set(v___x_3804_, 1, v___x_3803_);
v___x_3805_ = l_Lean_Expr_const___override(v___x_3802_, v___x_3804_);
v___x_3806_ = l_Lean_mkApp3(v___x_3805_, v_A_3789_, v_p_3787_, v_hNotEx_3790_);
if (lean_obj_tag(v___x_3800_) == 5)
{
lean_object* v_fn_3829_; 
v_fn_3829_ = lean_ctor_get(v___x_3800_, 0);
lean_inc_ref(v_fn_3829_);
if (lean_obj_tag(v_fn_3829_) == 5)
{
lean_object* v_fn_3830_; 
v_fn_3830_ = lean_ctor_get(v_fn_3829_, 0);
lean_inc_ref(v_fn_3830_);
if (lean_obj_tag(v_fn_3830_) == 4)
{
lean_object* v_declName_3831_; 
v_declName_3831_ = lean_ctor_get(v_fn_3830_, 0);
lean_inc(v_declName_3831_);
if (lean_obj_tag(v_declName_3831_) == 1)
{
lean_object* v_pre_3832_; 
v_pre_3832_ = lean_ctor_get(v_declName_3831_, 0);
if (lean_obj_tag(v_pre_3832_) == 0)
{
lean_object* v_arg_3833_; lean_object* v_arg_3834_; lean_object* v_us_3835_; lean_object* v_str_3836_; lean_object* v___x_3837_; uint8_t v___x_3838_; 
v_arg_3833_ = lean_ctor_get(v___x_3800_, 1);
lean_inc_ref(v_arg_3833_);
lean_dec_ref_known(v___x_3800_, 2);
v_arg_3834_ = lean_ctor_get(v_fn_3829_, 1);
lean_inc_ref(v_arg_3834_);
lean_dec_ref_known(v_fn_3829_, 2);
v_us_3835_ = lean_ctor_get(v_fn_3830_, 1);
lean_inc(v_us_3835_);
lean_dec_ref_known(v_fn_3830_, 2);
v_str_3836_ = lean_ctor_get(v_declName_3831_, 1);
lean_inc_ref(v_str_3836_);
lean_dec_ref_known(v_declName_3831_, 2);
v___x_3837_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__2));
v___x_3838_ = lean_string_dec_eq(v_str_3836_, v___x_3837_);
lean_dec_ref(v_str_3836_);
if (v___x_3838_ == 0)
{
lean_dec(v_us_3835_);
lean_dec_ref(v_arg_3834_);
lean_dec_ref(v_arg_3833_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
else
{
if (lean_obj_tag(v_us_3835_) == 1)
{
lean_object* v_tail_3839_; 
v_tail_3839_ = lean_ctor_get(v_us_3835_, 1);
if (lean_obj_tag(v_tail_3839_) == 0)
{
lean_object* v_head_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; 
v_head_3840_ = lean_ctor_get(v_us_3835_, 0);
lean_inc(v_head_3840_);
lean_dec_ref_known(v_us_3835_, 2);
v___x_3841_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__4));
v___x_3842_ = l_Lean_Core_mkFreshUserName(v___x_3841_, v___y_3794_, v___y_3795_);
if (lean_obj_tag(v___x_3842_) == 0)
{
lean_object* v_a_3843_; lean_object* v___x_3844_; lean_object* v___f_3845_; lean_object* v___x_3846_; 
v_a_3843_ = lean_ctor_get(v___x_3842_, 0);
lean_inc(v_a_3843_);
lean_dec_ref_known(v___x_3842_, 1);
v___x_3844_ = lean_box(v___x_3838_);
v___f_3845_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__0___boxed), 14, 8);
lean_closure_set(v___f_3845_, 0, v_head_3840_);
lean_closure_set(v___f_3845_, 1, v_arg_3834_);
lean_closure_set(v___f_3845_, 2, v_arg_3833_);
lean_closure_set(v___f_3845_, 3, v___x_3799_);
lean_closure_set(v___f_3845_, 4, v___x_3844_);
lean_closure_set(v___f_3845_, 5, v___x_3798_);
lean_closure_set(v___f_3845_, 6, v___x_3806_);
lean_closure_set(v___f_3845_, 7, v_x_3791_);
v___x_3846_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(v_a_3843_, v___x_3801_, v___f_3845_, v___y_3792_, v___y_3793_, v___y_3794_, v___y_3795_);
return v___x_3846_;
}
else
{
lean_object* v_a_3847_; lean_object* v___x_3849_; uint8_t v_isShared_3850_; uint8_t v_isSharedCheck_3854_; 
lean_dec(v_head_3840_);
lean_dec_ref(v_arg_3834_);
lean_dec_ref(v_arg_3833_);
lean_dec_ref(v___x_3806_);
lean_dec_ref(v___x_3801_);
lean_dec_ref(v___x_3799_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
v_a_3847_ = lean_ctor_get(v___x_3842_, 0);
v_isSharedCheck_3854_ = !lean_is_exclusive(v___x_3842_);
if (v_isSharedCheck_3854_ == 0)
{
v___x_3849_ = v___x_3842_;
v_isShared_3850_ = v_isSharedCheck_3854_;
goto v_resetjp_3848_;
}
else
{
lean_inc(v_a_3847_);
lean_dec(v___x_3842_);
v___x_3849_ = lean_box(0);
v_isShared_3850_ = v_isSharedCheck_3854_;
goto v_resetjp_3848_;
}
v_resetjp_3848_:
{
lean_object* v___x_3852_; 
if (v_isShared_3850_ == 0)
{
v___x_3852_ = v___x_3849_;
goto v_reusejp_3851_;
}
else
{
lean_object* v_reuseFailAlloc_3853_; 
v_reuseFailAlloc_3853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3853_, 0, v_a_3847_);
v___x_3852_ = v_reuseFailAlloc_3853_;
goto v_reusejp_3851_;
}
v_reusejp_3851_:
{
return v___x_3852_;
}
}
}
}
else
{
lean_dec_ref_known(v_us_3835_, 2);
lean_dec_ref(v_arg_3834_);
lean_dec_ref(v_arg_3833_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
else
{
lean_dec(v_us_3835_);
lean_dec_ref(v_arg_3834_);
lean_dec_ref(v_arg_3833_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
}
else
{
lean_dec_ref_known(v_declName_3831_, 2);
lean_dec_ref_known(v_fn_3830_, 2);
lean_dec_ref_known(v_fn_3829_, 2);
lean_dec_ref_known(v___x_3800_, 2);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
else
{
lean_dec(v_declName_3831_);
lean_dec_ref_known(v_fn_3830_, 2);
lean_dec_ref_known(v_fn_3829_, 2);
lean_dec_ref_known(v___x_3800_, 2);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
else
{
lean_dec_ref(v_fn_3830_);
lean_dec_ref_known(v_fn_3829_, 2);
lean_dec_ref_known(v___x_3800_, 2);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
else
{
lean_dec_ref_known(v___x_3800_, 2);
lean_dec_ref(v_fn_3829_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
}
else
{
lean_dec_ref(v___x_3800_);
lean_dec_ref(v___x_3798_);
lean_dec_ref(v_x_3791_);
goto v___jp_3807_;
}
v___jp_3807_:
{
uint8_t v___x_3808_; uint8_t v___x_3809_; uint8_t v___x_3810_; lean_object* v___x_3811_; 
v___x_3808_ = 0;
v___x_3809_ = 1;
v___x_3810_ = 1;
v___x_3811_ = l_Lean_Meta_mkForallFVars(v___x_3799_, v___x_3801_, v___x_3808_, v___x_3809_, v___x_3809_, v___x_3810_, v___y_3792_, v___y_3793_, v___y_3794_, v___y_3795_);
lean_dec_ref(v___x_3799_);
if (lean_obj_tag(v___x_3811_) == 0)
{
lean_object* v_a_3812_; lean_object* v___x_3814_; uint8_t v_isShared_3815_; uint8_t v_isSharedCheck_3820_; 
v_a_3812_ = lean_ctor_get(v___x_3811_, 0);
v_isSharedCheck_3820_ = !lean_is_exclusive(v___x_3811_);
if (v_isSharedCheck_3820_ == 0)
{
v___x_3814_ = v___x_3811_;
v_isShared_3815_ = v_isSharedCheck_3820_;
goto v_resetjp_3813_;
}
else
{
lean_inc(v_a_3812_);
lean_dec(v___x_3811_);
v___x_3814_ = lean_box(0);
v_isShared_3815_ = v_isSharedCheck_3820_;
goto v_resetjp_3813_;
}
v_resetjp_3813_:
{
lean_object* v___x_3816_; lean_object* v___x_3818_; 
v___x_3816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3816_, 0, v_a_3812_);
lean_ctor_set(v___x_3816_, 1, v___x_3806_);
if (v_isShared_3815_ == 0)
{
lean_ctor_set(v___x_3814_, 0, v___x_3816_);
v___x_3818_ = v___x_3814_;
goto v_reusejp_3817_;
}
else
{
lean_object* v_reuseFailAlloc_3819_; 
v_reuseFailAlloc_3819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3819_, 0, v___x_3816_);
v___x_3818_ = v_reuseFailAlloc_3819_;
goto v_reusejp_3817_;
}
v_reusejp_3817_:
{
return v___x_3818_;
}
}
}
else
{
lean_object* v_a_3821_; lean_object* v___x_3823_; uint8_t v_isShared_3824_; uint8_t v_isSharedCheck_3828_; 
lean_dec_ref(v___x_3806_);
v_a_3821_ = lean_ctor_get(v___x_3811_, 0);
v_isSharedCheck_3828_ = !lean_is_exclusive(v___x_3811_);
if (v_isSharedCheck_3828_ == 0)
{
v___x_3823_ = v___x_3811_;
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
else
{
lean_inc(v_a_3821_);
lean_dec(v___x_3811_);
v___x_3823_ = lean_box(0);
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
v_resetjp_3822_:
{
lean_object* v___x_3826_; 
if (v_isShared_3824_ == 0)
{
v___x_3826_ = v___x_3823_;
goto v_reusejp_3825_;
}
else
{
lean_object* v_reuseFailAlloc_3827_; 
v_reuseFailAlloc_3827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3827_, 0, v_a_3821_);
v___x_3826_ = v_reuseFailAlloc_3827_;
goto v_reusejp_3825_;
}
v_reusejp_3825_:
{
return v___x_3826_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___boxed(lean_object* v_p_3855_, lean_object* v_lvl_3856_, lean_object* v_A_3857_, lean_object* v_hNotEx_3858_, lean_object* v_x_3859_, lean_object* v___y_3860_, lean_object* v___y_3861_, lean_object* v___y_3862_, lean_object* v___y_3863_, lean_object* v___y_3864_){
_start:
{
lean_object* v_res_3865_; 
v_res_3865_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1(v_p_3855_, v_lvl_3856_, v_A_3857_, v_hNotEx_3858_, v_x_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_);
lean_dec(v___y_3863_);
lean_dec_ref(v___y_3862_);
lean_dec(v___y_3861_);
lean_dec_ref(v___y_3860_);
return v_res_3865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go(lean_object* v_lvl_3866_, lean_object* v_A_3867_, lean_object* v_p_3868_, lean_object* v_hNotEx_3869_, lean_object* v_a_3870_, lean_object* v_a_3871_, lean_object* v_a_3872_, lean_object* v_a_3873_){
_start:
{
lean_object* v___x_3875_; lean_object* v___x_3876_; 
v___x_3875_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___closed__1));
v___x_3876_ = l_Lean_Core_mkFreshUserName(v___x_3875_, v_a_3872_, v_a_3873_);
if (lean_obj_tag(v___x_3876_) == 0)
{
lean_object* v_a_3877_; lean_object* v___f_3878_; lean_object* v___x_3879_; 
v_a_3877_ = lean_ctor_get(v___x_3876_, 0);
lean_inc(v_a_3877_);
lean_dec_ref_known(v___x_3876_, 1);
lean_inc_ref(v_A_3867_);
v___f_3878_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___boxed), 10, 4);
lean_closure_set(v___f_3878_, 0, v_p_3868_);
lean_closure_set(v___f_3878_, 1, v_lvl_3866_);
lean_closure_set(v___f_3878_, 2, v_A_3867_);
lean_closure_set(v___f_3878_, 3, v_hNotEx_3869_);
v___x_3879_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(v_a_3877_, v_A_3867_, v___f_3878_, v_a_3870_, v_a_3871_, v_a_3872_, v_a_3873_);
return v___x_3879_;
}
else
{
lean_object* v_a_3880_; lean_object* v___x_3882_; uint8_t v_isShared_3883_; uint8_t v_isSharedCheck_3887_; 
lean_dec_ref(v_hNotEx_3869_);
lean_dec_ref(v_p_3868_);
lean_dec_ref(v_A_3867_);
lean_dec(v_lvl_3866_);
v_a_3880_ = lean_ctor_get(v___x_3876_, 0);
v_isSharedCheck_3887_ = !lean_is_exclusive(v___x_3876_);
if (v_isSharedCheck_3887_ == 0)
{
v___x_3882_ = v___x_3876_;
v_isShared_3883_ = v_isSharedCheck_3887_;
goto v_resetjp_3881_;
}
else
{
lean_inc(v_a_3880_);
lean_dec(v___x_3876_);
v___x_3882_ = lean_box(0);
v_isShared_3883_ = v_isSharedCheck_3887_;
goto v_resetjp_3881_;
}
v_resetjp_3881_:
{
lean_object* v___x_3885_; 
if (v_isShared_3883_ == 0)
{
v___x_3885_ = v___x_3882_;
goto v_reusejp_3884_;
}
else
{
lean_object* v_reuseFailAlloc_3886_; 
v_reuseFailAlloc_3886_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3886_, 0, v_a_3880_);
v___x_3885_ = v_reuseFailAlloc_3886_;
goto v_reusejp_3884_;
}
v_reusejp_3884_:
{
return v___x_3885_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___boxed(lean_object* v_lvl_3888_, lean_object* v_A_3889_, lean_object* v_p_3890_, lean_object* v_hNotEx_3891_, lean_object* v_a_3892_, lean_object* v_a_3893_, lean_object* v_a_3894_, lean_object* v_a_3895_, lean_object* v_a_3896_){
_start:
{
lean_object* v_res_3897_; 
v_res_3897_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go(v_lvl_3888_, v_A_3889_, v_p_3890_, v_hNotEx_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_);
lean_dec(v_a_3895_);
lean_dec_ref(v_a_3894_);
lean_dec(v_a_3893_);
lean_dec_ref(v_a_3892_);
return v_res_3897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0(lean_object* v_00_u03b1_3898_, lean_object* v_name_3899_, uint8_t v_bi_3900_, lean_object* v_type_3901_, lean_object* v_k_3902_, uint8_t v_kind_3903_, lean_object* v___y_3904_, lean_object* v___y_3905_, lean_object* v___y_3906_, lean_object* v___y_3907_){
_start:
{
lean_object* v___x_3909_; 
v___x_3909_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___redArg(v_name_3899_, v_bi_3900_, v_type_3901_, v_k_3902_, v_kind_3903_, v___y_3904_, v___y_3905_, v___y_3906_, v___y_3907_);
return v___x_3909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0___boxed(lean_object* v_00_u03b1_3910_, lean_object* v_name_3911_, lean_object* v_bi_3912_, lean_object* v_type_3913_, lean_object* v_k_3914_, lean_object* v_kind_3915_, lean_object* v___y_3916_, lean_object* v___y_3917_, lean_object* v___y_3918_, lean_object* v___y_3919_, lean_object* v___y_3920_){
_start:
{
uint8_t v_bi_boxed_3921_; uint8_t v_kind_boxed_3922_; lean_object* v_res_3923_; 
v_bi_boxed_3921_ = lean_unbox(v_bi_3912_);
v_kind_boxed_3922_ = lean_unbox(v_kind_3915_);
v_res_3923_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0_spec__0(v_00_u03b1_3910_, v_name_3911_, v_bi_boxed_3921_, v_type_3913_, v_k_3914_, v_kind_boxed_3922_, v___y_3916_, v___y_3917_, v___y_3918_, v___y_3919_);
lean_dec(v___y_3919_);
lean_dec_ref(v___y_3918_);
lean_dec(v___y_3917_);
lean_dec_ref(v___y_3916_);
return v_res_3923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0(lean_object* v_00_u03b1_3924_, lean_object* v_name_3925_, lean_object* v_type_3926_, lean_object* v_k_3927_, lean_object* v___y_3928_, lean_object* v___y_3929_, lean_object* v___y_3930_, lean_object* v___y_3931_){
_start:
{
lean_object* v___x_3933_; 
v___x_3933_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___redArg(v_name_3925_, v_type_3926_, v_k_3927_, v___y_3928_, v___y_3929_, v___y_3930_, v___y_3931_);
return v___x_3933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0___boxed(lean_object* v_00_u03b1_3934_, lean_object* v_name_3935_, lean_object* v_type_3936_, lean_object* v_k_3937_, lean_object* v___y_3938_, lean_object* v___y_3939_, lean_object* v___y_3940_, lean_object* v___y_3941_, lean_object* v___y_3942_){
_start:
{
lean_object* v_res_3943_; 
v_res_3943_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go_spec__0(v_00_u03b1_3934_, v_name_3935_, v_type_3936_, v_k_3937_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_);
lean_dec(v___y_3941_);
lean_dec_ref(v___y_3940_);
lean_dec(v___y_3939_);
lean_dec_ref(v___y_3938_);
return v_res_3943_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1(void){
_start:
{
lean_object* v___x_3945_; lean_object* v___x_3946_; 
v___x_3945_ = ((lean_object*)(lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__0));
v___x_3946_ = l_Lean_stringToMessageData(v___x_3945_);
return v___x_3946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists(lean_object* v_ex_3947_, lean_object* v_hNotEx_3948_, lean_object* v_a_3949_, lean_object* v_a_3950_, lean_object* v_a_3951_, lean_object* v_a_3952_){
_start:
{
lean_object* v___y_3955_; lean_object* v___y_3956_; lean_object* v___y_3957_; lean_object* v___y_3958_; 
if (lean_obj_tag(v_ex_3947_) == 5)
{
lean_object* v_fn_3961_; 
v_fn_3961_ = lean_ctor_get(v_ex_3947_, 0);
lean_inc_ref(v_fn_3961_);
if (lean_obj_tag(v_fn_3961_) == 5)
{
lean_object* v_fn_3962_; 
v_fn_3962_ = lean_ctor_get(v_fn_3961_, 0);
lean_inc_ref(v_fn_3962_);
if (lean_obj_tag(v_fn_3962_) == 4)
{
lean_object* v_declName_3963_; 
v_declName_3963_ = lean_ctor_get(v_fn_3962_, 0);
lean_inc(v_declName_3963_);
if (lean_obj_tag(v_declName_3963_) == 1)
{
lean_object* v_pre_3964_; 
v_pre_3964_ = lean_ctor_get(v_declName_3963_, 0);
if (lean_obj_tag(v_pre_3964_) == 0)
{
lean_object* v_arg_3965_; lean_object* v_arg_3966_; lean_object* v_us_3967_; lean_object* v_str_3968_; lean_object* v___x_3969_; uint8_t v___x_3970_; 
v_arg_3965_ = lean_ctor_get(v_ex_3947_, 1);
lean_inc_ref(v_arg_3965_);
lean_dec_ref_known(v_ex_3947_, 2);
v_arg_3966_ = lean_ctor_get(v_fn_3961_, 1);
lean_inc_ref(v_arg_3966_);
lean_dec_ref_known(v_fn_3961_, 2);
v_us_3967_ = lean_ctor_get(v_fn_3962_, 1);
lean_inc(v_us_3967_);
lean_dec_ref_known(v_fn_3962_, 2);
v_str_3968_ = lean_ctor_get(v_declName_3963_, 1);
lean_inc_ref(v_str_3968_);
lean_dec_ref_known(v_declName_3963_, 2);
v___x_3969_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go___lam__1___closed__2));
v___x_3970_ = lean_string_dec_eq(v_str_3968_, v___x_3969_);
lean_dec_ref(v_str_3968_);
if (v___x_3970_ == 0)
{
lean_dec(v_us_3967_);
lean_dec_ref(v_arg_3966_);
lean_dec_ref(v_arg_3965_);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
else
{
if (lean_obj_tag(v_us_3967_) == 1)
{
lean_object* v_tail_3971_; 
v_tail_3971_ = lean_ctor_get(v_us_3967_, 1);
if (lean_obj_tag(v_tail_3971_) == 0)
{
lean_object* v_head_3972_; lean_object* v___x_3973_; 
v_head_3972_ = lean_ctor_get(v_us_3967_, 0);
lean_inc(v_head_3972_);
lean_dec_ref_known(v_us_3967_, 2);
v___x_3973_ = lp_mathlib___private_Mathlib_Lean_Expr_Basic_0__Lean_Expr_forallNot__of__notExists_go(v_head_3972_, v_arg_3966_, v_arg_3965_, v_hNotEx_3948_, v_a_3949_, v_a_3950_, v_a_3951_, v_a_3952_);
return v___x_3973_;
}
else
{
lean_dec_ref_known(v_us_3967_, 2);
lean_dec_ref(v_arg_3966_);
lean_dec_ref(v_arg_3965_);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
else
{
lean_dec(v_us_3967_);
lean_dec_ref(v_arg_3966_);
lean_dec_ref(v_arg_3965_);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
}
else
{
lean_dec_ref_known(v_declName_3963_, 2);
lean_dec_ref_known(v_fn_3962_, 2);
lean_dec_ref_known(v_fn_3961_, 2);
lean_dec_ref_known(v_ex_3947_, 2);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
else
{
lean_dec_ref_known(v_fn_3962_, 2);
lean_dec(v_declName_3963_);
lean_dec_ref_known(v_fn_3961_, 2);
lean_dec_ref_known(v_ex_3947_, 2);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
else
{
lean_dec_ref(v_fn_3962_);
lean_dec_ref_known(v_fn_3961_, 2);
lean_dec_ref_known(v_ex_3947_, 2);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
else
{
lean_dec_ref(v_fn_3961_);
lean_dec_ref_known(v_ex_3947_, 2);
lean_dec_ref(v_hNotEx_3948_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
}
else
{
lean_dec_ref(v_hNotEx_3948_);
lean_dec_ref(v_ex_3947_);
v___y_3955_ = v_a_3949_;
v___y_3956_ = v_a_3950_;
v___y_3957_ = v_a_3951_;
v___y_3958_ = v_a_3952_;
goto v___jp_3954_;
}
v___jp_3954_:
{
lean_object* v___x_3959_; lean_object* v___x_3960_; 
v___x_3959_ = lean_obj_once(&lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1, &lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1_once, _init_lp_mathlib_Lean_Expr_forallNot__of__notExists___closed__1);
v___x_3960_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Lean_mkConst_x27_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v___x_3959_, v___y_3955_, v___y_3956_, v___y_3957_, v___y_3958_);
return v___x_3960_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_forallNot__of__notExists___boxed(lean_object* v_ex_3974_, lean_object* v_hNotEx_3975_, lean_object* v_a_3976_, lean_object* v_a_3977_, lean_object* v_a_3978_, lean_object* v_a_3979_, lean_object* v_a_3980_){
_start:
{
lean_object* v_res_3981_; 
v_res_3981_ = lp_mathlib_Lean_Expr_forallNot__of__notExists(v_ex_3974_, v_hNotEx_3975_, v_a_3976_, v_a_3977_, v_a_3978_, v_a_3979_);
lean_dec(v_a_3979_);
lean_dec_ref(v_a_3978_);
lean_dec(v_a_3977_);
lean_dec_ref(v_a_3976_);
return v_res_3981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0(lean_object* v_env_3982_, lean_object* v_structName_3983_, lean_object* v_as_3984_, size_t v_i_3985_, size_t v_stop_3986_, lean_object* v_b_3987_){
_start:
{
lean_object* v___y_3989_; uint8_t v___x_3993_; 
v___x_3993_ = lean_usize_dec_eq(v_i_3985_, v_stop_3986_);
if (v___x_3993_ == 0)
{
lean_object* v___x_3994_; lean_object* v___x_3995_; 
v___x_3994_ = lean_array_uget_borrowed(v_as_3984_, v_i_3985_);
lean_inc(v___x_3994_);
lean_inc(v_structName_3983_);
lean_inc_ref(v_env_3982_);
v___x_3995_ = l_Lean_isSubobjectField_x3f(v_env_3982_, v_structName_3983_, v___x_3994_);
if (lean_obj_tag(v___x_3995_) == 0)
{
v___y_3989_ = v_b_3987_;
goto v___jp_3988_;
}
else
{
lean_object* v___x_3996_; 
lean_dec_ref_known(v___x_3995_, 1);
lean_inc(v___x_3994_);
v___x_3996_ = lean_array_push(v_b_3987_, v___x_3994_);
v___y_3989_ = v___x_3996_;
goto v___jp_3988_;
}
}
else
{
lean_dec(v_structName_3983_);
lean_dec_ref(v_env_3982_);
return v_b_3987_;
}
v___jp_3988_:
{
size_t v___x_3990_; size_t v___x_3991_; 
v___x_3990_ = ((size_t)1ULL);
v___x_3991_ = lean_usize_add(v_i_3985_, v___x_3990_);
v_i_3985_ = v___x_3991_;
v_b_3987_ = v___y_3989_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0___boxed(lean_object* v_env_3997_, lean_object* v_structName_3998_, lean_object* v_as_3999_, lean_object* v_i_4000_, lean_object* v_stop_4001_, lean_object* v_b_4002_){
_start:
{
size_t v_i_boxed_4003_; size_t v_stop_boxed_4004_; lean_object* v_res_4005_; 
v_i_boxed_4003_ = lean_unbox_usize(v_i_4000_);
lean_dec(v_i_4000_);
v_stop_boxed_4004_ = lean_unbox_usize(v_stop_4001_);
lean_dec(v_stop_4001_);
v_res_4005_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0(v_env_3997_, v_structName_3998_, v_as_3999_, v_i_boxed_4003_, v_stop_boxed_4004_, v_b_4002_);
lean_dec_ref(v_as_3999_);
return v_res_4005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getFieldsToParents(lean_object* v_env_4008_, lean_object* v_structName_4009_){
_start:
{
lean_object* v___x_4010_; lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; uint8_t v___x_4014_; 
lean_inc(v_structName_4009_);
lean_inc_ref(v_env_4008_);
v___x_4010_ = l_Lean_getStructureFields(v_env_4008_, v_structName_4009_);
v___x_4011_ = lean_unsigned_to_nat(0u);
v___x_4012_ = lean_array_get_size(v___x_4010_);
v___x_4013_ = ((lean_object*)(lp_mathlib_Lean_getFieldsToParents___closed__0));
v___x_4014_ = lean_nat_dec_lt(v___x_4011_, v___x_4012_);
if (v___x_4014_ == 0)
{
lean_dec_ref(v___x_4010_);
lean_dec(v_structName_4009_);
lean_dec_ref(v_env_4008_);
return v___x_4013_;
}
else
{
uint8_t v___x_4015_; 
v___x_4015_ = lean_nat_dec_le(v___x_4012_, v___x_4012_);
if (v___x_4015_ == 0)
{
if (v___x_4014_ == 0)
{
lean_dec_ref(v___x_4010_);
lean_dec(v_structName_4009_);
lean_dec_ref(v_env_4008_);
return v___x_4013_;
}
else
{
size_t v___x_4016_; size_t v___x_4017_; lean_object* v___x_4018_; 
v___x_4016_ = ((size_t)0ULL);
v___x_4017_ = lean_usize_of_nat(v___x_4012_);
v___x_4018_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0(v_env_4008_, v_structName_4009_, v___x_4010_, v___x_4016_, v___x_4017_, v___x_4013_);
lean_dec_ref(v___x_4010_);
return v___x_4018_;
}
}
else
{
size_t v___x_4019_; size_t v___x_4020_; lean_object* v___x_4021_; 
v___x_4019_ = ((size_t)0ULL);
v___x_4020_ = lean_usize_of_nat(v___x_4012_);
v___x_4021_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_getFieldsToParents_spec__0(v_env_4008_, v_structName_4009_, v___x_4010_, v___x_4019_, v___x_4020_, v___x_4013_);
lean_dec_ref(v___x_4010_);
return v___x_4021_;
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_AppBuilder(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Match_MatcherInfo(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Transform(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_AppBuilder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Match_MatcherInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Transform(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Meta_AppBuilder(uint8_t builtin);
lean_object* initialize_Lean_Meta_Match_MatcherInfo(uint8_t builtin);
lean_object* initialize_Lean_Meta_Transform(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_AppBuilder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Match_MatcherInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Transform(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
