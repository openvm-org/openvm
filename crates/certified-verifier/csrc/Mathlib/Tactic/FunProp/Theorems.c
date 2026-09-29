// Lean compiler output
// Module: Mathlib.Tactic.FunProp.Theorems
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.FunProp.Decl public meta import Mathlib.Tactic.FunProp.Types public meta import Mathlib.Tactic.FunProp.FunctionData public meta import Mathlib.Lean.Meta.RefinedDiscrTree.Initialize public meta import Mathlib.Lean.Meta.RefinedDiscrTree.Lookup public import Mathlib.Lean.Meta.RefinedDiscrTree.Lookup public import Mathlib.Tactic.FunProp.Decl public import Mathlib.Tactic.FunProp.Types
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
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_balance___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunProp_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isFVar(lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerSimpleScopedEnvExtension___redArg(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_addCore___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_insert___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_MatchResult_toArray___redArg(lean_object*);
uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default;
lean_object* l_Lean_Expr_fvar___override(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_id_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_id_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_const_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_apply_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_apply_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_comp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_comp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_pi_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_pi_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremArgs.apply"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremArgs.const"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremArgs.id"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremArgs.pi"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremArgs.comp"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__11_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs___closed__0_value;
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremType.id"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremType.const"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremType.apply"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremType.comp"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Meta.FunProp.LambdaTheoremType.pi"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType___closed__0_value;
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorem_default___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheorem_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheorem_getProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FunProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "lambdaTheoremsExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(34, 54, 44, 240, 60, 34, 52, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_lambdaTheoremsExt;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Mathlib.Meta.FunProp.TheoremForm.uncurried"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Mathlib.Meta.FunProp.TheoremForm.comp"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "compositional"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm___boxed(lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem;
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionTheorem_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionTheorem_getProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "functionTheoremsExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(222, 195, 209, 221, 95, 63, 219, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_GeneralTheorem_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_GeneralTheorem_getProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "transitionTheoremsExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(207, 249, 97, 173, 194, 121, 82, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "morTheoremsExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 158, 98, 68, 99, 139, 98, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_lam_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_lam_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_function_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_function_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_mor_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_mor_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_transition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_transition_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Not a valid `fun_prop` theorem!"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "fun_prop theorem about morphism coercion has to be in fully applied form"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unrecognized theoremType `"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "function in invalid form "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "unrecognized function property `"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp_addTheorem_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fun_prop"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "attr"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__0_value),LEAN_SCALAR_PTR_LITERAL(194, 95, 140, 15, 16, 100, 236, 219)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__1_value),LEAN_SCALAR_PTR_LITERAL(186, 231, 153, 117, 210, 184, 65, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__2_value),LEAN_SCALAR_PTR_LITERAL(176, 68, 102, 141, 146, 150, 56, 47)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "lambda theorem: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "\nfunction property: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\ntype: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "function theorem: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "\nfunction name: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "\nmain arguments: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "\napplied arguments: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\nform: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " form"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "morphism theorem: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "transition theorem: "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
case 3:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
default: 
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(4u);
return v___x_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorIdx___boxed(lean_object* v_x_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorIdx(v_x_7_);
lean_dec(v_x_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(lean_object* v_t_9_, lean_object* v_k_10_){
_start:
{
if (lean_obj_tag(v_t_9_) == 3)
{
lean_object* v_fArgId_11_; lean_object* v_gArgId_12_; lean_object* v___x_13_; 
v_fArgId_11_ = lean_ctor_get(v_t_9_, 0);
lean_inc(v_fArgId_11_);
v_gArgId_12_ = lean_ctor_get(v_t_9_, 1);
lean_inc(v_gArgId_12_);
lean_dec_ref_known(v_t_9_, 2);
v___x_13_ = lean_apply_2(v_k_10_, v_fArgId_11_, v_gArgId_12_);
return v___x_13_;
}
else
{
lean_dec(v_t_9_);
return v_k_10_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim(lean_object* v_motive_14_, lean_object* v_ctorIdx_15_, lean_object* v_t_16_, lean_object* v_h_17_, lean_object* v_k_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_16_, v_k_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___boxed(lean_object* v_motive_20_, lean_object* v_ctorIdx_21_, lean_object* v_t_22_, lean_object* v_h_23_, lean_object* v_k_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim(v_motive_20_, v_ctorIdx_21_, v_t_22_, v_h_23_, v_k_24_);
lean_dec(v_ctorIdx_21_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_id_elim___redArg(lean_object* v_t_26_, lean_object* v_id_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_26_, v_id_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_id_elim(lean_object* v_motive_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_id_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_30_, v_id_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_const_elim___redArg(lean_object* v_t_34_, lean_object* v_const_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_34_, v_const_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_const_elim(lean_object* v_motive_37_, lean_object* v_t_38_, lean_object* v_h_39_, lean_object* v_const_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_38_, v_const_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_apply_elim___redArg(lean_object* v_t_42_, lean_object* v_apply_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_42_, v_apply_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_apply_elim(lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_apply_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_46_, v_apply_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_comp_elim___redArg(lean_object* v_t_50_, lean_object* v_comp_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_50_, v_comp_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_comp_elim(lean_object* v_motive_53_, lean_object* v_t_54_, lean_object* v_h_55_, lean_object* v_comp_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_54_, v_comp_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_pi_elim___redArg(lean_object* v_t_58_, lean_object* v_pi_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_58_, v_pi_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_pi_elim(lean_object* v_motive_61_, lean_object* v_t_62_, lean_object* v_h_63_, lean_object* v_pi_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_ctorElim___redArg(v_t_62_, v_pi_64_);
return v___x_65_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs_default(void){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_box(0);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_box(0);
return v___x_67_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq(lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
switch(lean_obj_tag(v_x_68_))
{
case 0:
{
if (lean_obj_tag(v_x_69_) == 0)
{
uint8_t v___x_70_; 
v___x_70_ = 1;
return v___x_70_;
}
else
{
uint8_t v___x_71_; 
v___x_71_ = 0;
return v___x_71_;
}
}
case 1:
{
if (lean_obj_tag(v_x_69_) == 1)
{
uint8_t v___x_72_; 
v___x_72_ = 1;
return v___x_72_;
}
else
{
uint8_t v___x_73_; 
v___x_73_ = 0;
return v___x_73_;
}
}
case 2:
{
if (lean_obj_tag(v_x_69_) == 2)
{
uint8_t v___x_74_; 
v___x_74_ = 1;
return v___x_74_;
}
else
{
uint8_t v___x_75_; 
v___x_75_ = 0;
return v___x_75_;
}
}
case 3:
{
if (lean_obj_tag(v_x_69_) == 3)
{
lean_object* v_fArgId_76_; lean_object* v_gArgId_77_; lean_object* v_fArgId_78_; lean_object* v_gArgId_79_; uint8_t v___x_80_; 
v_fArgId_76_ = lean_ctor_get(v_x_68_, 0);
v_gArgId_77_ = lean_ctor_get(v_x_68_, 1);
v_fArgId_78_ = lean_ctor_get(v_x_69_, 0);
v_gArgId_79_ = lean_ctor_get(v_x_69_, 1);
v___x_80_ = lean_nat_dec_eq(v_fArgId_76_, v_fArgId_78_);
if (v___x_80_ == 0)
{
return v___x_80_;
}
else
{
uint8_t v___x_81_; 
v___x_81_ = lean_nat_dec_eq(v_gArgId_77_, v_gArgId_79_);
return v___x_81_;
}
}
else
{
uint8_t v___x_82_; 
v___x_82_ = 0;
return v___x_82_;
}
}
default: 
{
if (lean_obj_tag(v_x_69_) == 4)
{
uint8_t v___x_83_; 
v___x_83_ = 1;
return v___x_83_;
}
else
{
uint8_t v___x_84_; 
v___x_84_ = 0;
return v___x_84_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq___boxed(lean_object* v_x_85_, lean_object* v_x_86_){
_start:
{
uint8_t v_res_87_; lean_object* v_r_88_; 
v_res_87_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq(v_x_85_, v_x_86_);
lean_dec(v_x_86_);
lean_dec(v_x_85_);
v_r_88_ = lean_box(v_res_87_);
return v_r_88_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = lean_unsigned_to_nat(2u);
v___x_104_ = lean_nat_to_int(v___x_103_);
return v___x_104_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = lean_unsigned_to_nat(1u);
v___x_106_ = lean_nat_to_int(v___x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr(lean_object* v_x_113_, lean_object* v_prec_114_){
_start:
{
lean_object* v___y_116_; lean_object* v___y_123_; lean_object* v___y_130_; lean_object* v___y_137_; 
switch(lean_obj_tag(v_x_113_))
{
case 0:
{
lean_object* v___x_143_; uint8_t v___x_144_; 
v___x_143_ = lean_unsigned_to_nat(1024u);
v___x_144_ = lean_nat_dec_le(v___x_143_, v_prec_114_);
if (v___x_144_ == 0)
{
lean_object* v___x_145_; 
v___x_145_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_130_ = v___x_145_;
goto v___jp_129_;
}
else
{
lean_object* v___x_146_; 
v___x_146_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_130_ = v___x_146_;
goto v___jp_129_;
}
}
case 1:
{
lean_object* v___x_147_; uint8_t v___x_148_; 
v___x_147_ = lean_unsigned_to_nat(1024u);
v___x_148_ = lean_nat_dec_le(v___x_147_, v_prec_114_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; 
v___x_149_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_123_ = v___x_149_;
goto v___jp_122_;
}
else
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_123_ = v___x_150_;
goto v___jp_122_;
}
}
case 2:
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = lean_unsigned_to_nat(1024u);
v___x_152_ = lean_nat_dec_le(v___x_151_, v_prec_114_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; 
v___x_153_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_116_ = v___x_153_;
goto v___jp_115_;
}
else
{
lean_object* v___x_154_; 
v___x_154_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_116_ = v___x_154_;
goto v___jp_115_;
}
}
case 3:
{
lean_object* v_fArgId_155_; lean_object* v_gArgId_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_181_; 
v_fArgId_155_ = lean_ctor_get(v_x_113_, 0);
v_gArgId_156_ = lean_ctor_get(v_x_113_, 1);
v_isSharedCheck_181_ = !lean_is_exclusive(v_x_113_);
if (v_isSharedCheck_181_ == 0)
{
v___x_158_ = v_x_113_;
v_isShared_159_ = v_isSharedCheck_181_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_gArgId_156_);
lean_inc(v_fArgId_155_);
lean_dec(v_x_113_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_181_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___y_161_; lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_177_ = lean_unsigned_to_nat(1024u);
v___x_178_ = lean_nat_dec_le(v___x_177_, v_prec_114_);
if (v___x_178_ == 0)
{
lean_object* v___x_179_; 
v___x_179_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_161_ = v___x_179_;
goto v___jp_160_;
}
else
{
lean_object* v___x_180_; 
v___x_180_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_161_ = v___x_180_;
goto v___jp_160_;
}
v___jp_160_:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_167_; 
v___x_162_ = lean_box(1);
v___x_163_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__12));
v___x_164_ = l_Nat_reprFast(v_fArgId_155_);
v___x_165_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
if (v_isShared_159_ == 0)
{
lean_ctor_set_tag(v___x_158_, 5);
lean_ctor_set(v___x_158_, 1, v___x_165_);
lean_ctor_set(v___x_158_, 0, v___x_163_);
v___x_167_ = v___x_158_;
goto v_reusejp_166_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_163_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v___x_165_);
v___x_167_ = v_reuseFailAlloc_176_;
goto v_reusejp_166_;
}
v_reusejp_166_:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; uint8_t v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_168_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
lean_ctor_set(v___x_168_, 1, v___x_162_);
v___x_169_ = l_Nat_reprFast(v_gArgId_156_);
v___x_170_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
v___x_171_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_168_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
lean_inc(v___y_161_);
v___x_172_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_172_, 0, v___y_161_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = 0;
v___x_174_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_174_, 0, v___x_172_);
lean_ctor_set_uint8(v___x_174_, sizeof(void*)*1, v___x_173_);
v___x_175_ = l_Repr_addAppParen(v___x_174_, v_prec_114_);
return v___x_175_;
}
}
}
}
default: 
{
lean_object* v___x_182_; uint8_t v___x_183_; 
v___x_182_ = lean_unsigned_to_nat(1024u);
v___x_183_ = lean_nat_dec_le(v___x_182_, v_prec_114_);
if (v___x_183_ == 0)
{
lean_object* v___x_184_; 
v___x_184_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_137_ = v___x_184_;
goto v___jp_136_;
}
else
{
lean_object* v___x_185_; 
v___x_185_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_137_ = v___x_185_;
goto v___jp_136_;
}
}
}
v___jp_115_:
{
lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__1));
lean_inc(v___y_116_);
v___x_118_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_118_, 0, v___y_116_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = 0;
v___x_120_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_120_, 0, v___x_118_);
lean_ctor_set_uint8(v___x_120_, sizeof(void*)*1, v___x_119_);
v___x_121_ = l_Repr_addAppParen(v___x_120_, v_prec_114_);
return v___x_121_;
}
v___jp_122_:
{
lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_124_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__3));
lean_inc(v___y_123_);
v___x_125_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_125_, 0, v___y_123_);
lean_ctor_set(v___x_125_, 1, v___x_124_);
v___x_126_ = 0;
v___x_127_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_127_, 0, v___x_125_);
lean_ctor_set_uint8(v___x_127_, sizeof(void*)*1, v___x_126_);
v___x_128_ = l_Repr_addAppParen(v___x_127_, v_prec_114_);
return v___x_128_;
}
v___jp_129_:
{
lean_object* v___x_131_; lean_object* v___x_132_; uint8_t v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_131_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__5));
lean_inc(v___y_130_);
v___x_132_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_132_, 0, v___y_130_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = 0;
v___x_134_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_134_, 0, v___x_132_);
lean_ctor_set_uint8(v___x_134_, sizeof(void*)*1, v___x_133_);
v___x_135_ = l_Repr_addAppParen(v___x_134_, v_prec_114_);
return v___x_135_;
}
v___jp_136_:
{
lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__7));
lean_inc(v___y_137_);
v___x_139_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_139_, 0, v___y_137_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = 0;
v___x_141_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_141_, 0, v___x_139_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1, v___x_140_);
v___x_142_ = l_Repr_addAppParen(v___x_141_, v_prec_114_);
return v___x_142_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___boxed(lean_object* v_x_186_, lean_object* v_prec_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr(v_x_186_, v_prec_187_);
lean_dec(v_prec_187_);
return v_res_188_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash(lean_object* v_x_191_){
_start:
{
switch(lean_obj_tag(v_x_191_))
{
case 0:
{
uint64_t v___x_192_; 
v___x_192_ = 0ULL;
return v___x_192_;
}
case 1:
{
uint64_t v___x_193_; 
v___x_193_ = 1ULL;
return v___x_193_;
}
case 2:
{
uint64_t v___x_194_; 
v___x_194_ = 2ULL;
return v___x_194_;
}
case 3:
{
lean_object* v_fArgId_195_; lean_object* v_gArgId_196_; uint64_t v___x_197_; uint64_t v___x_198_; uint64_t v___x_199_; uint64_t v___x_200_; uint64_t v___x_201_; 
v_fArgId_195_ = lean_ctor_get(v_x_191_, 0);
v_gArgId_196_ = lean_ctor_get(v_x_191_, 1);
v___x_197_ = 3ULL;
v___x_198_ = lean_uint64_of_nat(v_fArgId_195_);
v___x_199_ = lean_uint64_mix_hash(v___x_197_, v___x_198_);
v___x_200_ = lean_uint64_of_nat(v_gArgId_196_);
v___x_201_ = lean_uint64_mix_hash(v___x_199_, v___x_200_);
return v___x_201_;
}
default: 
{
uint64_t v___x_202_; 
v___x_202_ = 4ULL;
return v___x_202_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash___boxed(lean_object* v_x_203_){
_start:
{
uint64_t v_res_204_; lean_object* v_r_205_; 
v_res_204_ = lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremArgs_hash(v_x_203_);
lean_dec(v_x_203_);
v_r_205_ = lean_box_uint64(v_res_204_);
return v_r_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx(uint8_t v_x_208_){
_start:
{
switch(v_x_208_)
{
case 0:
{
lean_object* v___x_209_; 
v___x_209_ = lean_unsigned_to_nat(0u);
return v___x_209_;
}
case 1:
{
lean_object* v___x_210_; 
v___x_210_ = lean_unsigned_to_nat(1u);
return v___x_210_;
}
case 2:
{
lean_object* v___x_211_; 
v___x_211_ = lean_unsigned_to_nat(2u);
return v___x_211_;
}
case 3:
{
lean_object* v___x_212_; 
v___x_212_ = lean_unsigned_to_nat(3u);
return v___x_212_;
}
default: 
{
lean_object* v___x_213_; 
v___x_213_ = lean_unsigned_to_nat(4u);
return v___x_213_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx___boxed(lean_object* v_x_214_){
_start:
{
uint8_t v_x_boxed_215_; lean_object* v_res_216_; 
v_x_boxed_215_ = lean_unbox(v_x_214_);
v_res_216_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx(v_x_boxed_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___redArg(lean_object* v_k_217_){
_start:
{
lean_inc(v_k_217_);
return v_k_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___redArg___boxed(lean_object* v_k_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___redArg(v_k_218_);
lean_dec(v_k_218_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim(lean_object* v_motive_220_, lean_object* v_ctorIdx_221_, uint8_t v_t_222_, lean_object* v_h_223_, lean_object* v_k_224_){
_start:
{
lean_inc(v_k_224_);
return v_k_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim___boxed(lean_object* v_motive_225_, lean_object* v_ctorIdx_226_, lean_object* v_t_227_, lean_object* v_h_228_, lean_object* v_k_229_){
_start:
{
uint8_t v_t_boxed_230_; lean_object* v_res_231_; 
v_t_boxed_230_ = lean_unbox(v_t_227_);
v_res_231_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorElim(v_motive_225_, v_ctorIdx_226_, v_t_boxed_230_, v_h_228_, v_k_229_);
lean_dec(v_k_229_);
lean_dec(v_ctorIdx_226_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___redArg(lean_object* v_id_232_){
_start:
{
lean_inc(v_id_232_);
return v_id_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___redArg___boxed(lean_object* v_id_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___redArg(v_id_233_);
lean_dec(v_id_233_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim(lean_object* v_motive_235_, uint8_t v_t_236_, lean_object* v_h_237_, lean_object* v_id_238_){
_start:
{
lean_inc(v_id_238_);
return v_id_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim___boxed(lean_object* v_motive_239_, lean_object* v_t_240_, lean_object* v_h_241_, lean_object* v_id_242_){
_start:
{
uint8_t v_t_boxed_243_; lean_object* v_res_244_; 
v_t_boxed_243_ = lean_unbox(v_t_240_);
v_res_244_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_id_elim(v_motive_239_, v_t_boxed_243_, v_h_241_, v_id_242_);
lean_dec(v_id_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___redArg(lean_object* v_const_245_){
_start:
{
lean_inc(v_const_245_);
return v_const_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___redArg___boxed(lean_object* v_const_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___redArg(v_const_246_);
lean_dec(v_const_246_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim(lean_object* v_motive_248_, uint8_t v_t_249_, lean_object* v_h_250_, lean_object* v_const_251_){
_start:
{
lean_inc(v_const_251_);
return v_const_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim___boxed(lean_object* v_motive_252_, lean_object* v_t_253_, lean_object* v_h_254_, lean_object* v_const_255_){
_start:
{
uint8_t v_t_boxed_256_; lean_object* v_res_257_; 
v_t_boxed_256_ = lean_unbox(v_t_253_);
v_res_257_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_const_elim(v_motive_252_, v_t_boxed_256_, v_h_254_, v_const_255_);
lean_dec(v_const_255_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___redArg(lean_object* v_apply_258_){
_start:
{
lean_inc(v_apply_258_);
return v_apply_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___redArg___boxed(lean_object* v_apply_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___redArg(v_apply_259_);
lean_dec(v_apply_259_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim(lean_object* v_motive_261_, uint8_t v_t_262_, lean_object* v_h_263_, lean_object* v_apply_264_){
_start:
{
lean_inc(v_apply_264_);
return v_apply_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim___boxed(lean_object* v_motive_265_, lean_object* v_t_266_, lean_object* v_h_267_, lean_object* v_apply_268_){
_start:
{
uint8_t v_t_boxed_269_; lean_object* v_res_270_; 
v_t_boxed_269_ = lean_unbox(v_t_266_);
v_res_270_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_apply_elim(v_motive_265_, v_t_boxed_269_, v_h_267_, v_apply_268_);
lean_dec(v_apply_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___redArg(lean_object* v_comp_271_){
_start:
{
lean_inc(v_comp_271_);
return v_comp_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___redArg___boxed(lean_object* v_comp_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___redArg(v_comp_272_);
lean_dec(v_comp_272_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim(lean_object* v_motive_274_, uint8_t v_t_275_, lean_object* v_h_276_, lean_object* v_comp_277_){
_start:
{
lean_inc(v_comp_277_);
return v_comp_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim___boxed(lean_object* v_motive_278_, lean_object* v_t_279_, lean_object* v_h_280_, lean_object* v_comp_281_){
_start:
{
uint8_t v_t_boxed_282_; lean_object* v_res_283_; 
v_t_boxed_282_ = lean_unbox(v_t_279_);
v_res_283_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_comp_elim(v_motive_278_, v_t_boxed_282_, v_h_280_, v_comp_281_);
lean_dec(v_comp_281_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___redArg(lean_object* v_pi_284_){
_start:
{
lean_inc(v_pi_284_);
return v_pi_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___redArg___boxed(lean_object* v_pi_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___redArg(v_pi_285_);
lean_dec(v_pi_285_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim(lean_object* v_motive_287_, uint8_t v_t_288_, lean_object* v_h_289_, lean_object* v_pi_290_){
_start:
{
lean_inc(v_pi_290_);
return v_pi_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim___boxed(lean_object* v_motive_291_, lean_object* v_t_292_, lean_object* v_h_293_, lean_object* v_pi_294_){
_start:
{
uint8_t v_t_boxed_295_; lean_object* v_res_296_; 
v_t_boxed_295_ = lean_unbox(v_t_292_);
v_res_296_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_pi_elim(v_motive_291_, v_t_boxed_295_, v_h_293_, v_pi_294_);
lean_dec(v_pi_294_);
return v_res_296_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType_default(void){
_start:
{
uint8_t v___x_297_; 
v___x_297_ = 0;
return v___x_297_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType(void){
_start:
{
uint8_t v___x_298_; 
v___x_298_ = 0;
return v___x_298_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(uint8_t v_x_299_, uint8_t v_y_300_){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_301_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx(v_x_299_);
v___x_302_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremType_ctorIdx(v_y_300_);
v___x_303_ = lean_nat_dec_eq(v___x_301_, v___x_302_);
lean_dec(v___x_302_);
lean_dec(v___x_301_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq___boxed(lean_object* v_x_304_, lean_object* v_y_305_){
_start:
{
uint8_t v_x_17__boxed_306_; uint8_t v_y_18__boxed_307_; uint8_t v_res_308_; lean_object* v_r_309_; 
v_x_17__boxed_306_ = lean_unbox(v_x_304_);
v_y_18__boxed_307_ = lean_unbox(v_y_305_);
v_res_308_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(v_x_17__boxed_306_, v_y_18__boxed_307_);
v_r_309_ = lean_box(v_res_308_);
return v_r_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr(uint8_t v_x_327_, lean_object* v_prec_328_){
_start:
{
lean_object* v___y_330_; lean_object* v___y_337_; lean_object* v___y_344_; lean_object* v___y_351_; lean_object* v___y_358_; 
switch(v_x_327_)
{
case 0:
{
lean_object* v___x_364_; uint8_t v___x_365_; 
v___x_364_ = lean_unsigned_to_nat(1024u);
v___x_365_ = lean_nat_dec_le(v___x_364_, v_prec_328_);
if (v___x_365_ == 0)
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_330_ = v___x_366_;
goto v___jp_329_;
}
else
{
lean_object* v___x_367_; 
v___x_367_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_330_ = v___x_367_;
goto v___jp_329_;
}
}
case 1:
{
lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_368_ = lean_unsigned_to_nat(1024u);
v___x_369_ = lean_nat_dec_le(v___x_368_, v_prec_328_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; 
v___x_370_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_337_ = v___x_370_;
goto v___jp_336_;
}
else
{
lean_object* v___x_371_; 
v___x_371_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_337_ = v___x_371_;
goto v___jp_336_;
}
}
case 2:
{
lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_372_ = lean_unsigned_to_nat(1024u);
v___x_373_ = lean_nat_dec_le(v___x_372_, v_prec_328_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; 
v___x_374_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_344_ = v___x_374_;
goto v___jp_343_;
}
else
{
lean_object* v___x_375_; 
v___x_375_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_344_ = v___x_375_;
goto v___jp_343_;
}
}
case 3:
{
lean_object* v___x_376_; uint8_t v___x_377_; 
v___x_376_ = lean_unsigned_to_nat(1024u);
v___x_377_ = lean_nat_dec_le(v___x_376_, v_prec_328_);
if (v___x_377_ == 0)
{
lean_object* v___x_378_; 
v___x_378_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_351_ = v___x_378_;
goto v___jp_350_;
}
else
{
lean_object* v___x_379_; 
v___x_379_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_351_ = v___x_379_;
goto v___jp_350_;
}
}
default: 
{
lean_object* v___x_380_; uint8_t v___x_381_; 
v___x_380_ = lean_unsigned_to_nat(1024u);
v___x_381_ = lean_nat_dec_le(v___x_380_, v_prec_328_);
if (v___x_381_ == 0)
{
lean_object* v___x_382_; 
v___x_382_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_358_ = v___x_382_;
goto v___jp_357_;
}
else
{
lean_object* v___x_383_; 
v___x_383_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_358_ = v___x_383_;
goto v___jp_357_;
}
}
}
v___jp_329_:
{
lean_object* v___x_331_; lean_object* v___x_332_; uint8_t v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__1));
lean_inc(v___y_330_);
v___x_332_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_332_, 0, v___y_330_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v___x_333_ = 0;
v___x_334_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_334_, 0, v___x_332_);
lean_ctor_set_uint8(v___x_334_, sizeof(void*)*1, v___x_333_);
v___x_335_ = l_Repr_addAppParen(v___x_334_, v_prec_328_);
return v___x_335_;
}
v___jp_336_:
{
lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_338_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__3));
lean_inc(v___y_337_);
v___x_339_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_339_, 0, v___y_337_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
v___x_340_ = 0;
v___x_341_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_341_, 0, v___x_339_);
lean_ctor_set_uint8(v___x_341_, sizeof(void*)*1, v___x_340_);
v___x_342_ = l_Repr_addAppParen(v___x_341_, v_prec_328_);
return v___x_342_;
}
v___jp_343_:
{
lean_object* v___x_345_; lean_object* v___x_346_; uint8_t v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__5));
lean_inc(v___y_344_);
v___x_346_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_346_, 0, v___y_344_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = 0;
v___x_348_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set_uint8(v___x_348_, sizeof(void*)*1, v___x_347_);
v___x_349_ = l_Repr_addAppParen(v___x_348_, v_prec_328_);
return v___x_349_;
}
v___jp_350_:
{
lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_352_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__7));
lean_inc(v___y_351_);
v___x_353_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_353_, 0, v___y_351_);
lean_ctor_set(v___x_353_, 1, v___x_352_);
v___x_354_ = 0;
v___x_355_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_355_, 0, v___x_353_);
lean_ctor_set_uint8(v___x_355_, sizeof(void*)*1, v___x_354_);
v___x_356_ = l_Repr_addAppParen(v___x_355_, v_prec_328_);
return v___x_356_;
}
v___jp_357_:
{
lean_object* v___x_359_; lean_object* v___x_360_; uint8_t v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___closed__9));
lean_inc(v___y_358_);
v___x_360_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_360_, 0, v___y_358_);
lean_ctor_set(v___x_360_, 1, v___x_359_);
v___x_361_ = 0;
v___x_362_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_362_, 0, v___x_360_);
lean_ctor_set_uint8(v___x_362_, sizeof(void*)*1, v___x_361_);
v___x_363_ = l_Repr_addAppParen(v___x_362_, v_prec_328_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr___boxed(lean_object* v_x_384_, lean_object* v_prec_385_){
_start:
{
uint8_t v_x_285__boxed_386_; lean_object* v_res_387_; 
v_x_285__boxed_386_ = lean_unbox(v_x_384_);
v_res_387_ = lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr(v_x_285__boxed_386_, v_prec_385_);
lean_dec(v_prec_385_);
return v_res_387_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(uint8_t v_x_390_){
_start:
{
switch(v_x_390_)
{
case 0:
{
uint64_t v___x_391_; 
v___x_391_ = 0ULL;
return v___x_391_;
}
case 1:
{
uint64_t v___x_392_; 
v___x_392_ = 1ULL;
return v___x_392_;
}
case 2:
{
uint64_t v___x_393_; 
v___x_393_ = 2ULL;
return v___x_393_;
}
case 3:
{
uint64_t v___x_394_; 
v___x_394_ = 3ULL;
return v___x_394_;
}
default: 
{
uint64_t v___x_395_; 
v___x_395_ = 4ULL;
return v___x_395_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash___boxed(lean_object* v_x_396_){
_start:
{
uint8_t v_x_64__boxed_397_; uint64_t v_res_398_; lean_object* v_r_399_; 
v_x_64__boxed_397_ = lean_unbox(v_x_396_);
v_res_398_ = lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(v_x_64__boxed_397_);
v_r_399_ = lean_box_uint64(v_res_398_);
return v_r_399_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type(lean_object* v_t_402_){
_start:
{
switch(lean_obj_tag(v_t_402_))
{
case 0:
{
uint8_t v___x_403_; 
v___x_403_ = 0;
return v___x_403_;
}
case 1:
{
uint8_t v___x_404_; 
v___x_404_ = 1;
return v___x_404_;
}
case 2:
{
uint8_t v___x_405_; 
v___x_405_ = 2;
return v___x_405_;
}
case 3:
{
uint8_t v___x_406_; 
v___x_406_ = 3;
return v___x_406_;
}
default: 
{
uint8_t v___x_407_; 
v___x_407_ = 4;
return v___x_407_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type___boxed(lean_object* v_t_408_){
_start:
{
uint8_t v_res_409_; lean_object* v_r_410_; 
v_res_409_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type(v_t_408_);
lean_dec(v_t_408_);
v_r_410_ = lean_box(v_res_409_);
return v_r_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0(lean_object* v_k_411_, lean_object* v_b_412_, lean_object* v_c_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v___x_419_; 
lean_inc(v___y_417_);
lean_inc_ref(v___y_416_);
lean_inc(v___y_415_);
lean_inc_ref(v___y_414_);
v___x_419_ = lean_apply_7(v_k_411_, v_b_412_, v_c_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, lean_box(0));
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0___boxed(lean_object* v_k_420_, lean_object* v_b_421_, lean_object* v_c_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0(v_k_420_, v_b_421_, v_c_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(lean_object* v_type_429_, lean_object* v_k_430_, uint8_t v_cleanupAnnotations_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_){
_start:
{
lean_object* v___f_437_; uint8_t v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v___f_437_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_437_, 0, v_k_430_);
v___x_438_ = 0;
v___x_439_ = lean_box(0);
v___x_440_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_438_, v___x_439_, v_type_429_, v___f_437_, v_cleanupAnnotations_431_, v___x_438_, v___y_432_, v___y_433_, v___y_434_, v___y_435_);
if (lean_obj_tag(v___x_440_) == 0)
{
lean_object* v_a_441_; lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_448_; 
v_a_441_ = lean_ctor_get(v___x_440_, 0);
v_isSharedCheck_448_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_448_ == 0)
{
v___x_443_ = v___x_440_;
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
else
{
lean_inc(v_a_441_);
lean_dec(v___x_440_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_446_; 
if (v_isShared_444_ == 0)
{
v___x_446_ = v___x_443_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v_a_441_);
v___x_446_ = v_reuseFailAlloc_447_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
return v___x_446_;
}
}
}
else
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_456_; 
v_a_449_ = lean_ctor_get(v___x_440_, 0);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_456_ == 0)
{
v___x_451_ = v___x_440_;
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_440_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_454_; 
if (v_isShared_452_ == 0)
{
v___x_454_ = v___x_451_;
goto v_reusejp_453_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v_a_449_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg___boxed(lean_object* v_type_457_, lean_object* v_k_458_, lean_object* v_cleanupAnnotations_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_465_; lean_object* v_res_466_; 
v_cleanupAnnotations_boxed_465_ = lean_unbox(v_cleanupAnnotations_459_);
v_res_466_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(v_type_457_, v_k_458_, v_cleanupAnnotations_boxed_465_, v___y_460_, v___y_461_, v___y_462_, v___y_463_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0(lean_object* v_00_u03b1_467_, lean_object* v_type_468_, lean_object* v_k_469_, uint8_t v_cleanupAnnotations_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(v_type_468_, v_k_469_, v_cleanupAnnotations_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___boxed(lean_object* v_00_u03b1_477_, lean_object* v_type_478_, lean_object* v_k_479_, lean_object* v_cleanupAnnotations_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_486_; lean_object* v_res_487_; 
v_cleanupAnnotations_boxed_486_ = lean_unbox(v_cleanupAnnotations_480_);
v_res_487_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0(v_00_u03b1_477_, v_type_478_, v_k_479_, v_cleanupAnnotations_boxed_486_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0(lean_object* v_f_488_, lean_object* v_xs_489_, lean_object* v_x_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v___x_496_; lean_object* v___x_497_; uint8_t v___x_498_; uint8_t v___x_499_; uint8_t v___x_500_; lean_object* v___x_501_; 
v___x_496_ = l_Lean_mkAppN(v_f_488_, v_xs_489_);
v___x_497_ = l_Lean_Expr_headBeta(v___x_496_);
v___x_498_ = 0;
v___x_499_ = 1;
v___x_500_ = 1;
v___x_501_ = l_Lean_Meta_mkLambdaFVars(v_xs_489_, v___x_497_, v___x_498_, v___x_499_, v___x_498_, v___x_499_, v___x_500_, v___y_491_, v___y_492_, v___y_493_, v___y_494_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0___boxed(lean_object* v_f_502_, lean_object* v_xs_503_, lean_object* v_x_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0(v_f_502_, v_xs_503_, v_x_504_, v___y_505_, v___y_506_, v___y_507_, v___y_508_);
lean_dec(v___y_508_);
lean_dec_ref(v___y_507_);
lean_dec(v___y_506_);
lean_dec_ref(v___y_505_);
lean_dec_ref(v_x_504_);
lean_dec_ref(v_xs_503_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1(lean_object* v_fn_511_, lean_object* v_as_512_, lean_object* v_j_513_){
_start:
{
lean_object* v___x_514_; uint8_t v___x_515_; 
v___x_514_ = lean_array_get_size(v_as_512_);
v___x_515_ = lean_nat_dec_lt(v_j_513_, v___x_514_);
if (v___x_515_ == 0)
{
lean_object* v___x_516_; 
lean_dec(v_j_513_);
v___x_516_ = lean_box(0);
return v___x_516_;
}
else
{
lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_517_ = lean_array_fget_borrowed(v_as_512_, v_j_513_);
v___x_518_ = lean_expr_eqv(v___x_517_, v_fn_511_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = lean_unsigned_to_nat(1u);
v___x_520_ = lean_nat_add(v_j_513_, v___x_519_);
lean_dec(v_j_513_);
v_j_513_ = v___x_520_;
goto _start;
}
else
{
lean_object* v___x_522_; 
v___x_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_522_, 0, v_j_513_);
return v___x_522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1___boxed(lean_object* v_fn_523_, lean_object* v_as_524_, lean_object* v_j_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1(v_fn_523_, v_as_524_, v_j_525_);
lean_dec_ref(v_as_524_);
lean_dec_ref(v_fn_523_);
return v_res_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs(lean_object* v_f_535_, lean_object* v_ctxVars_536_, lean_object* v_a_537_, lean_object* v_a_538_, lean_object* v_a_539_, lean_object* v_a_540_){
_start:
{
lean_object* v___x_545_; 
lean_inc(v_a_540_);
lean_inc_ref(v_a_539_);
lean_inc(v_a_538_);
lean_inc_ref(v_a_537_);
lean_inc_ref(v_f_535_);
v___x_545_ = lean_infer_type(v_f_535_, v_a_537_, v_a_538_, v_a_539_, v_a_540_);
if (lean_obj_tag(v___x_545_) == 0)
{
lean_object* v_a_546_; lean_object* v___f_547_; uint8_t v___x_548_; lean_object* v___x_549_; 
v_a_546_ = lean_ctor_get(v___x_545_, 0);
lean_inc(v_a_546_);
lean_dec_ref_known(v___x_545_, 1);
v___f_547_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___lam__0___boxed), 8, 1);
lean_closure_set(v___f_547_, 0, v_f_535_);
v___x_548_ = 0;
v___x_549_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(v_a_546_, v___f_547_, v___x_548_, v_a_537_, v_a_538_, v_a_539_, v_a_540_);
if (lean_obj_tag(v___x_549_) == 0)
{
lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_624_; 
v_a_550_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_624_ == 0)
{
v___x_552_ = v___x_549_;
v_isShared_553_ = v_isSharedCheck_624_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_549_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_624_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
if (lean_obj_tag(v_a_550_) == 6)
{
lean_object* v_body_554_; uint8_t v___x_555_; 
v_body_554_ = lean_ctor_get(v_a_550_, 2);
lean_inc_ref(v_body_554_);
lean_dec_ref_known(v_a_550_, 3);
v___x_555_ = l_Lean_Expr_hasLooseBVars(v_body_554_);
if (v___x_555_ == 0)
{
lean_object* v___x_556_; lean_object* v___x_558_; 
lean_dec_ref(v_body_554_);
v___x_556_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__0));
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_556_);
v___x_558_ = v___x_552_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v___x_556_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
else
{
switch(lean_obj_tag(v_body_554_))
{
case 0:
{
lean_object* v_deBruijnIndex_560_; lean_object* v___x_561_; uint8_t v___x_562_; 
v_deBruijnIndex_560_ = lean_ctor_get(v_body_554_, 0);
lean_inc(v_deBruijnIndex_560_);
lean_dec_ref_known(v_body_554_, 1);
v___x_561_ = lean_unsigned_to_nat(0u);
v___x_562_ = lean_nat_dec_eq(v_deBruijnIndex_560_, v___x_561_);
lean_dec(v_deBruijnIndex_560_);
if (v___x_562_ == 0)
{
lean_del_object(v___x_552_);
goto v___jp_542_;
}
else
{
lean_object* v___x_563_; lean_object* v___x_565_; 
v___x_563_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__1));
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_563_);
v___x_565_ = v___x_552_;
goto v_reusejp_564_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v___x_563_);
v___x_565_ = v_reuseFailAlloc_566_;
goto v_reusejp_564_;
}
v_reusejp_564_:
{
return v___x_565_;
}
}
}
case 5:
{
lean_object* v_fn_567_; 
v_fn_567_ = lean_ctor_get(v_body_554_, 0);
lean_inc_ref(v_fn_567_);
switch(lean_obj_tag(v_fn_567_))
{
case 0:
{
lean_object* v_arg_568_; lean_object* v_deBruijnIndex_569_; lean_object* v___x_570_; uint8_t v___x_571_; 
v_arg_568_ = lean_ctor_get(v_body_554_, 1);
lean_inc_ref(v_arg_568_);
lean_dec_ref_known(v_body_554_, 2);
v_deBruijnIndex_569_ = lean_ctor_get(v_fn_567_, 0);
lean_inc(v_deBruijnIndex_569_);
lean_dec_ref_known(v_fn_567_, 1);
v___x_570_ = lean_unsigned_to_nat(0u);
v___x_571_ = lean_nat_dec_eq(v_deBruijnIndex_569_, v___x_570_);
lean_dec(v_deBruijnIndex_569_);
if (v___x_571_ == 0)
{
lean_dec_ref(v_arg_568_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
else
{
if (lean_obj_tag(v_arg_568_) == 1)
{
lean_object* v___x_572_; lean_object* v___x_574_; 
lean_dec_ref_known(v_arg_568_, 1);
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__2));
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_572_);
v___x_574_ = v___x_552_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v___x_572_);
v___x_574_ = v_reuseFailAlloc_575_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
return v___x_574_;
}
}
else
{
lean_dec_ref(v_arg_568_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
}
case 1:
{
lean_object* v_arg_576_; 
v_arg_576_ = lean_ctor_get(v_body_554_, 1);
lean_inc_ref(v_arg_576_);
lean_dec_ref_known(v_body_554_, 2);
if (lean_obj_tag(v_arg_576_) == 5)
{
lean_object* v_fn_577_; 
v_fn_577_ = lean_ctor_get(v_arg_576_, 0);
lean_inc_ref(v_fn_577_);
if (lean_obj_tag(v_fn_577_) == 1)
{
lean_object* v_arg_578_; 
v_arg_578_ = lean_ctor_get(v_arg_576_, 1);
lean_inc_ref(v_arg_578_);
lean_dec_ref_known(v_arg_576_, 2);
if (lean_obj_tag(v_arg_578_) == 0)
{
lean_object* v_deBruijnIndex_579_; lean_object* v___x_580_; uint8_t v___x_581_; 
v_deBruijnIndex_579_ = lean_ctor_get(v_arg_578_, 0);
lean_inc(v_deBruijnIndex_579_);
lean_dec_ref_known(v_arg_578_, 1);
v___x_580_ = lean_unsigned_to_nat(0u);
v___x_581_ = lean_nat_dec_eq(v_deBruijnIndex_579_, v___x_580_);
lean_dec(v_deBruijnIndex_579_);
if (v___x_581_ == 0)
{
lean_dec_ref_known(v_fn_577_, 1);
lean_dec_ref_known(v_fn_567_, 1);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
else
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1(v_fn_567_, v_ctxVars_536_, v___x_580_);
lean_dec_ref_known(v_fn_567_, 1);
if (lean_obj_tag(v___x_582_) == 1)
{
lean_object* v_val_583_; lean_object* v___x_584_; 
v_val_583_ = lean_ctor_get(v___x_582_, 0);
lean_inc(v_val_583_);
lean_dec_ref_known(v___x_582_, 1);
v___x_584_ = lp_mathlib_Array_findIdx_x3f_loop___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__1(v_fn_577_, v_ctxVars_536_, v___x_580_);
lean_dec_ref_known(v_fn_577_, 1);
if (lean_obj_tag(v___x_584_) == 1)
{
lean_object* v_val_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_596_; 
v_val_585_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_596_ == 0)
{
v___x_587_ = v___x_584_;
v_isShared_588_ = v_isSharedCheck_596_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_val_585_);
lean_dec(v___x_584_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_596_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_589_; lean_object* v___x_591_; 
v___x_589_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_589_, 0, v_val_583_);
lean_ctor_set(v___x_589_, 1, v_val_585_);
if (v_isShared_588_ == 0)
{
lean_ctor_set(v___x_587_, 0, v___x_589_);
v___x_591_ = v___x_587_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v___x_589_);
v___x_591_ = v_reuseFailAlloc_595_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
lean_object* v___x_593_; 
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_591_);
v___x_593_ = v___x_552_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v___x_591_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
}
}
else
{
lean_object* v___x_597_; lean_object* v___x_599_; 
lean_dec(v___x_584_);
lean_dec(v_val_583_);
v___x_597_ = lean_box(0);
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_597_);
v___x_599_ = v___x_552_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v___x_597_);
v___x_599_ = v_reuseFailAlloc_600_;
goto v_reusejp_598_;
}
v_reusejp_598_:
{
return v___x_599_;
}
}
}
else
{
lean_object* v___x_601_; lean_object* v___x_603_; 
lean_dec(v___x_582_);
lean_dec_ref_known(v_fn_577_, 1);
v___x_601_ = lean_box(0);
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_601_);
v___x_603_ = v___x_552_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v___x_601_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
else
{
lean_dec_ref(v_arg_578_);
lean_dec_ref_known(v_fn_577_, 1);
lean_dec_ref_known(v_fn_567_, 1);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
else
{
lean_dec_ref(v_fn_577_);
lean_dec_ref_known(v_arg_576_, 2);
lean_dec_ref_known(v_fn_567_, 1);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
else
{
lean_dec_ref_known(v_fn_567_, 1);
lean_dec_ref(v_arg_576_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
default: 
{
lean_dec_ref(v_fn_567_);
lean_dec_ref_known(v_body_554_, 2);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
}
case 6:
{
lean_object* v_body_605_; 
v_body_605_ = lean_ctor_get(v_body_554_, 2);
lean_inc_ref(v_body_605_);
lean_dec_ref_known(v_body_554_, 3);
if (lean_obj_tag(v_body_605_) == 5)
{
lean_object* v_fn_606_; 
v_fn_606_ = lean_ctor_get(v_body_605_, 0);
if (lean_obj_tag(v_fn_606_) == 5)
{
lean_object* v_fn_607_; 
v_fn_607_ = lean_ctor_get(v_fn_606_, 0);
if (lean_obj_tag(v_fn_607_) == 1)
{
lean_object* v_arg_608_; 
v_arg_608_ = lean_ctor_get(v_fn_606_, 1);
lean_inc_ref(v_arg_608_);
if (lean_obj_tag(v_arg_608_) == 0)
{
lean_object* v_arg_609_; lean_object* v_deBruijnIndex_610_; lean_object* v___x_611_; uint8_t v___x_612_; 
v_arg_609_ = lean_ctor_get(v_body_605_, 1);
lean_inc_ref(v_arg_609_);
lean_dec_ref_known(v_body_605_, 2);
v_deBruijnIndex_610_ = lean_ctor_get(v_arg_608_, 0);
lean_inc(v_deBruijnIndex_610_);
lean_dec_ref_known(v_arg_608_, 1);
v___x_611_ = lean_unsigned_to_nat(1u);
v___x_612_ = lean_nat_dec_eq(v_deBruijnIndex_610_, v___x_611_);
lean_dec(v_deBruijnIndex_610_);
if (v___x_612_ == 0)
{
lean_dec_ref(v_arg_609_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
else
{
if (lean_obj_tag(v_arg_609_) == 0)
{
lean_object* v_deBruijnIndex_613_; lean_object* v___x_614_; uint8_t v___x_615_; 
v_deBruijnIndex_613_ = lean_ctor_get(v_arg_609_, 0);
lean_inc(v_deBruijnIndex_613_);
lean_dec_ref_known(v_arg_609_, 1);
v___x_614_ = lean_unsigned_to_nat(0u);
v___x_615_ = lean_nat_dec_eq(v_deBruijnIndex_613_, v___x_614_);
lean_dec(v_deBruijnIndex_613_);
if (v___x_615_ == 0)
{
lean_del_object(v___x_552_);
goto v___jp_542_;
}
else
{
lean_object* v___x_616_; lean_object* v___x_618_; 
v___x_616_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___closed__3));
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_616_);
v___x_618_ = v___x_552_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v___x_616_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
else
{
lean_dec_ref(v_arg_609_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
}
else
{
lean_dec_ref(v_arg_608_);
lean_dec_ref_known(v_body_605_, 2);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
else
{
lean_dec_ref_known(v_body_605_, 2);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
else
{
lean_dec_ref_known(v_body_605_, 2);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
else
{
lean_dec_ref(v_body_605_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
default: 
{
lean_dec_ref(v_body_554_);
lean_del_object(v___x_552_);
goto v___jp_542_;
}
}
}
}
else
{
lean_object* v___x_620_; lean_object* v___x_622_; 
lean_dec(v_a_550_);
v___x_620_ = lean_box(0);
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_620_);
v___x_622_ = v___x_552_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v___x_620_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
}
else
{
lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_632_; 
v_a_625_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_632_ == 0)
{
v___x_627_ = v___x_549_;
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_549_);
v___x_627_ = lean_box(0);
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
v_resetjp_626_:
{
lean_object* v___x_630_; 
if (v_isShared_628_ == 0)
{
v___x_630_ = v___x_627_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_a_625_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
else
{
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
lean_dec_ref(v_f_535_);
v_a_633_ = lean_ctor_get(v___x_545_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_545_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_545_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_545_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_a_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
v___jp_542_:
{
lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_543_ = lean_box(0);
v___x_544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_544_, 0, v___x_543_);
return v___x_544_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs___boxed(lean_object* v_f_641_, lean_object* v_ctxVars_642_, lean_object* v_a_643_, lean_object* v_a_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs(v_f_641_, v_ctxVars_642_, v_a_643_, v_a_644_, v_a_645_, v_a_646_);
lean_dec(v_a_646_);
lean_dec_ref(v_a_645_);
lean_dec(v_a_644_);
lean_dec_ref(v_a_643_);
lean_dec_ref(v_ctxVars_642_);
return v_res_648_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq(lean_object* v_x_654_, lean_object* v_x_655_){
_start:
{
lean_object* v_funPropName_656_; lean_object* v_thmName_657_; lean_object* v_thmArgs_658_; lean_object* v_funPropName_659_; lean_object* v_thmName_660_; lean_object* v_thmArgs_661_; uint8_t v___x_662_; 
v_funPropName_656_ = lean_ctor_get(v_x_654_, 0);
v_thmName_657_ = lean_ctor_get(v_x_654_, 1);
v_thmArgs_658_ = lean_ctor_get(v_x_654_, 2);
v_funPropName_659_ = lean_ctor_get(v_x_655_, 0);
v_thmName_660_ = lean_ctor_get(v_x_655_, 1);
v_thmArgs_661_ = lean_ctor_get(v_x_655_, 2);
v___x_662_ = lean_name_eq(v_funPropName_656_, v_funPropName_659_);
if (v___x_662_ == 0)
{
return v___x_662_;
}
else
{
uint8_t v___x_663_; 
v___x_663_ = lean_name_eq(v_thmName_657_, v_thmName_660_);
if (v___x_663_ == 0)
{
return v___x_663_;
}
else
{
uint8_t v___x_664_; 
v___x_664_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremArgs_beq(v_thmArgs_658_, v_thmArgs_661_);
return v___x_664_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq___boxed(lean_object* v_x_665_, lean_object* v_x_666_){
_start:
{
uint8_t v_res_667_; lean_object* v_r_668_; 
v_res_667_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheorem_beq(v_x_665_, v_x_666_);
lean_dec_ref(v_x_666_);
lean_dec_ref(v_x_665_);
v_r_668_ = lean_box(v_res_667_);
return v_r_668_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0(void){
_start:
{
lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_671_ = lean_box(0);
v___x_672_ = lean_unsigned_to_nat(16u);
v___x_673_ = lean_mk_array(v___x_672_, v___x_671_);
return v___x_673_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1(void){
_start:
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v___x_674_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__0);
v___x_675_ = lean_unsigned_to_nat(0u);
v___x_676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
lean_ctor_set(v___x_676_, 1, v___x_674_);
return v___x_676_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default(void){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1);
return v___x_677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems(void){
_start:
{
lean_object* v___x_678_; 
v___x_678_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default;
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheorem_getProof(lean_object* v_thm_679_, lean_object* v_a_680_, lean_object* v_a_681_, lean_object* v_a_682_, lean_object* v_a_683_){
_start:
{
lean_object* v_thmName_685_; lean_object* v___x_686_; 
v_thmName_685_ = lean_ctor_get(v_thm_679_, 1);
lean_inc(v_thmName_685_);
lean_dec_ref(v_thm_679_);
v___x_686_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_thmName_685_, v_a_680_, v_a_681_, v_a_682_, v_a_683_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_LambdaTheorem_getProof___boxed(lean_object* v_thm_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheorem_getProof(v_thm_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_);
lean_dec(v_a_691_);
lean_dec_ref(v_a_690_);
lean_dec(v_a_689_);
lean_dec_ref(v_a_688_);
return v_res_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object* v_x_694_, lean_object* v_a_695_){
_start:
{
lean_object* v___x_696_; lean_object* v___x_697_; 
v___x_696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_696_, 0, v_a_695_);
lean_inc_ref_n(v___x_696_, 2);
v___x_697_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_697_, 0, v___x_696_);
lean_ctor_set(v___x_697_, 1, v___x_696_);
lean_ctor_set(v___x_697_, 2, v___x_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object* v_x_698_, lean_object* v_a_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(v_x_698_, v_a_699_);
lean_dec_ref(v_x_698_);
return v_res_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4___redArg(lean_object* v_a_701_, lean_object* v_b_702_, lean_object* v_x_703_){
_start:
{
if (lean_obj_tag(v_x_703_) == 0)
{
lean_dec(v_b_702_);
lean_dec_ref(v_a_701_);
return v_x_703_;
}
else
{
lean_object* v_key_704_; lean_object* v_value_705_; lean_object* v_tail_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_727_; 
v_key_704_ = lean_ctor_get(v_x_703_, 0);
v_value_705_ = lean_ctor_get(v_x_703_, 1);
v_tail_706_ = lean_ctor_get(v_x_703_, 2);
v_isSharedCheck_727_ = !lean_is_exclusive(v_x_703_);
if (v_isSharedCheck_727_ == 0)
{
v___x_708_ = v_x_703_;
v_isShared_709_ = v_isSharedCheck_727_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_tail_706_);
lean_inc(v_value_705_);
lean_inc(v_key_704_);
lean_dec(v_x_703_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_727_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
uint8_t v___y_711_; lean_object* v_fst_719_; lean_object* v_snd_720_; lean_object* v_fst_721_; lean_object* v_snd_722_; uint8_t v___x_723_; 
v_fst_719_ = lean_ctor_get(v_key_704_, 0);
v_snd_720_ = lean_ctor_get(v_key_704_, 1);
v_fst_721_ = lean_ctor_get(v_a_701_, 0);
v_snd_722_ = lean_ctor_get(v_a_701_, 1);
v___x_723_ = lean_name_eq(v_fst_719_, v_fst_721_);
if (v___x_723_ == 0)
{
v___y_711_ = v___x_723_;
goto v___jp_710_;
}
else
{
uint8_t v___x_724_; uint8_t v___x_725_; uint8_t v___x_726_; 
v___x_724_ = lean_unbox(v_snd_720_);
v___x_725_ = lean_unbox(v_snd_722_);
v___x_726_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(v___x_724_, v___x_725_);
v___y_711_ = v___x_726_;
goto v___jp_710_;
}
v___jp_710_:
{
if (v___y_711_ == 0)
{
lean_object* v___x_712_; lean_object* v___x_714_; 
v___x_712_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4___redArg(v_a_701_, v_b_702_, v_tail_706_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 2, v___x_712_);
v___x_714_ = v___x_708_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_key_704_);
lean_ctor_set(v_reuseFailAlloc_715_, 1, v_value_705_);
lean_ctor_set(v_reuseFailAlloc_715_, 2, v___x_712_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
else
{
lean_object* v___x_717_; 
lean_dec(v_value_705_);
lean_dec(v_key_704_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 1, v_b_702_);
lean_ctor_set(v___x_708_, 0, v_a_701_);
v___x_717_ = v___x_708_;
goto v_reusejp_716_;
}
else
{
lean_object* v_reuseFailAlloc_718_; 
v_reuseFailAlloc_718_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_718_, 0, v_a_701_);
lean_ctor_set(v_reuseFailAlloc_718_, 1, v_b_702_);
lean_ctor_set(v_reuseFailAlloc_718_, 2, v_tail_706_);
v___x_717_ = v_reuseFailAlloc_718_;
goto v_reusejp_716_;
}
v_reusejp_716_:
{
return v___x_717_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_x_728_, lean_object* v_x_729_){
_start:
{
if (lean_obj_tag(v_x_729_) == 0)
{
return v_x_728_;
}
else
{
lean_object* v_key_730_; lean_object* v_value_731_; lean_object* v_tail_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_763_; 
v_key_730_ = lean_ctor_get(v_x_729_, 0);
v_value_731_ = lean_ctor_get(v_x_729_, 1);
v_tail_732_ = lean_ctor_get(v_x_729_, 2);
v_isSharedCheck_763_ = !lean_is_exclusive(v_x_729_);
if (v_isSharedCheck_763_ == 0)
{
v___x_734_ = v_x_729_;
v_isShared_735_ = v_isSharedCheck_763_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_tail_732_);
lean_inc(v_value_731_);
lean_inc(v_key_730_);
lean_dec(v_x_729_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_763_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v_fst_736_; lean_object* v_snd_737_; lean_object* v___x_738_; uint64_t v___y_740_; 
v_fst_736_ = lean_ctor_get(v_key_730_, 0);
v_snd_737_ = lean_ctor_get(v_key_730_, 1);
v___x_738_ = lean_array_get_size(v_x_728_);
if (lean_obj_tag(v_fst_736_) == 0)
{
uint64_t v___x_761_; 
v___x_761_ = 1723ULL;
v___y_740_ = v___x_761_;
goto v___jp_739_;
}
else
{
uint64_t v_hash_762_; 
v_hash_762_ = lean_ctor_get_uint64(v_fst_736_, sizeof(void*)*2);
v___y_740_ = v_hash_762_;
goto v___jp_739_;
}
v___jp_739_:
{
uint8_t v___x_741_; uint64_t v___x_742_; uint64_t v___x_743_; uint64_t v___x_744_; uint64_t v___x_745_; uint64_t v_fold_746_; uint64_t v___x_747_; uint64_t v___x_748_; uint64_t v___x_749_; size_t v___x_750_; size_t v___x_751_; size_t v___x_752_; size_t v___x_753_; size_t v___x_754_; lean_object* v___x_755_; lean_object* v___x_757_; 
v___x_741_ = lean_unbox(v_snd_737_);
v___x_742_ = lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(v___x_741_);
v___x_743_ = lean_uint64_mix_hash(v___y_740_, v___x_742_);
v___x_744_ = 32ULL;
v___x_745_ = lean_uint64_shift_right(v___x_743_, v___x_744_);
v_fold_746_ = lean_uint64_xor(v___x_743_, v___x_745_);
v___x_747_ = 16ULL;
v___x_748_ = lean_uint64_shift_right(v_fold_746_, v___x_747_);
v___x_749_ = lean_uint64_xor(v_fold_746_, v___x_748_);
v___x_750_ = lean_uint64_to_usize(v___x_749_);
v___x_751_ = lean_usize_of_nat(v___x_738_);
v___x_752_ = ((size_t)1ULL);
v___x_753_ = lean_usize_sub(v___x_751_, v___x_752_);
v___x_754_ = lean_usize_land(v___x_750_, v___x_753_);
v___x_755_ = lean_array_uget_borrowed(v_x_728_, v___x_754_);
lean_inc(v___x_755_);
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 2, v___x_755_);
v___x_757_ = v___x_734_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_key_730_);
lean_ctor_set(v_reuseFailAlloc_760_, 1, v_value_731_);
lean_ctor_set(v_reuseFailAlloc_760_, 2, v___x_755_);
v___x_757_ = v_reuseFailAlloc_760_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
lean_object* v___x_758_; 
v___x_758_ = lean_array_uset(v_x_728_, v___x_754_, v___x_757_);
v_x_728_ = v___x_758_;
v_x_729_ = v_tail_732_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4___redArg(lean_object* v_i_764_, lean_object* v_source_765_, lean_object* v_target_766_){
_start:
{
lean_object* v___x_767_; uint8_t v___x_768_; 
v___x_767_ = lean_array_get_size(v_source_765_);
v___x_768_ = lean_nat_dec_lt(v_i_764_, v___x_767_);
if (v___x_768_ == 0)
{
lean_dec_ref(v_source_765_);
lean_dec(v_i_764_);
return v_target_766_;
}
else
{
lean_object* v_es_769_; lean_object* v___x_770_; lean_object* v_source_771_; lean_object* v_target_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v_es_769_ = lean_array_fget(v_source_765_, v_i_764_);
v___x_770_ = lean_box(0);
v_source_771_ = lean_array_fset(v_source_765_, v_i_764_, v___x_770_);
v_target_772_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5___redArg(v_target_766_, v_es_769_);
v___x_773_ = lean_unsigned_to_nat(1u);
v___x_774_ = lean_nat_add(v_i_764_, v___x_773_);
lean_dec(v_i_764_);
v_i_764_ = v___x_774_;
v_source_765_ = v_source_771_;
v_target_766_ = v_target_772_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3___redArg(lean_object* v_data_776_){
_start:
{
lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v_nbuckets_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v___x_777_ = lean_array_get_size(v_data_776_);
v___x_778_ = lean_unsigned_to_nat(2u);
v_nbuckets_779_ = lean_nat_mul(v___x_777_, v___x_778_);
v___x_780_ = lean_unsigned_to_nat(0u);
v___x_781_ = lean_box(0);
v___x_782_ = lean_mk_array(v_nbuckets_779_, v___x_781_);
v___x_783_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4___redArg(v___x_780_, v_data_776_, v___x_782_);
return v___x_783_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg(lean_object* v_a_784_, lean_object* v_x_785_){
_start:
{
if (lean_obj_tag(v_x_785_) == 0)
{
uint8_t v___x_786_; 
v___x_786_ = 0;
return v___x_786_;
}
else
{
lean_object* v_key_787_; lean_object* v_tail_788_; uint8_t v___y_790_; lean_object* v_fst_792_; lean_object* v_snd_793_; lean_object* v_fst_794_; lean_object* v_snd_795_; uint8_t v___x_796_; 
v_key_787_ = lean_ctor_get(v_x_785_, 0);
v_tail_788_ = lean_ctor_get(v_x_785_, 2);
v_fst_792_ = lean_ctor_get(v_key_787_, 0);
v_snd_793_ = lean_ctor_get(v_key_787_, 1);
v_fst_794_ = lean_ctor_get(v_a_784_, 0);
v_snd_795_ = lean_ctor_get(v_a_784_, 1);
v___x_796_ = lean_name_eq(v_fst_792_, v_fst_794_);
if (v___x_796_ == 0)
{
v___y_790_ = v___x_796_;
goto v___jp_789_;
}
else
{
uint8_t v___x_797_; uint8_t v___x_798_; uint8_t v___x_799_; 
v___x_797_ = lean_unbox(v_snd_793_);
v___x_798_ = lean_unbox(v_snd_795_);
v___x_799_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(v___x_797_, v___x_798_);
v___y_790_ = v___x_799_;
goto v___jp_789_;
}
v___jp_789_:
{
if (v___y_790_ == 0)
{
v_x_785_ = v_tail_788_;
goto _start;
}
else
{
return v___y_790_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg___boxed(lean_object* v_a_800_, lean_object* v_x_801_){
_start:
{
uint8_t v_res_802_; lean_object* v_r_803_; 
v_res_802_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg(v_a_800_, v_x_801_);
lean_dec(v_x_801_);
lean_dec_ref(v_a_800_);
v_r_803_ = lean_box(v_res_802_);
return v_r_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1___redArg(lean_object* v_m_804_, lean_object* v_a_805_, lean_object* v_b_806_){
_start:
{
lean_object* v_size_807_; lean_object* v_buckets_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_859_; 
v_size_807_ = lean_ctor_get(v_m_804_, 0);
v_buckets_808_ = lean_ctor_get(v_m_804_, 1);
v_isSharedCheck_859_ = !lean_is_exclusive(v_m_804_);
if (v_isSharedCheck_859_ == 0)
{
v___x_810_ = v_m_804_;
v_isShared_811_ = v_isSharedCheck_859_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_buckets_808_);
lean_inc(v_size_807_);
lean_dec(v_m_804_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_859_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v_fst_812_; lean_object* v_snd_813_; lean_object* v___x_814_; uint64_t v___y_816_; 
v_fst_812_ = lean_ctor_get(v_a_805_, 0);
v_snd_813_ = lean_ctor_get(v_a_805_, 1);
v___x_814_ = lean_array_get_size(v_buckets_808_);
if (lean_obj_tag(v_fst_812_) == 0)
{
uint64_t v___x_857_; 
v___x_857_ = 1723ULL;
v___y_816_ = v___x_857_;
goto v___jp_815_;
}
else
{
uint64_t v_hash_858_; 
v_hash_858_ = lean_ctor_get_uint64(v_fst_812_, sizeof(void*)*2);
v___y_816_ = v_hash_858_;
goto v___jp_815_;
}
v___jp_815_:
{
uint8_t v___x_817_; uint64_t v___x_818_; uint64_t v___x_819_; uint64_t v___x_820_; uint64_t v___x_821_; uint64_t v_fold_822_; uint64_t v___x_823_; uint64_t v___x_824_; uint64_t v___x_825_; size_t v___x_826_; size_t v___x_827_; size_t v___x_828_; size_t v___x_829_; size_t v___x_830_; lean_object* v_bkt_831_; uint8_t v___x_832_; 
v___x_817_ = lean_unbox(v_snd_813_);
v___x_818_ = lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(v___x_817_);
v___x_819_ = lean_uint64_mix_hash(v___y_816_, v___x_818_);
v___x_820_ = 32ULL;
v___x_821_ = lean_uint64_shift_right(v___x_819_, v___x_820_);
v_fold_822_ = lean_uint64_xor(v___x_819_, v___x_821_);
v___x_823_ = 16ULL;
v___x_824_ = lean_uint64_shift_right(v_fold_822_, v___x_823_);
v___x_825_ = lean_uint64_xor(v_fold_822_, v___x_824_);
v___x_826_ = lean_uint64_to_usize(v___x_825_);
v___x_827_ = lean_usize_of_nat(v___x_814_);
v___x_828_ = ((size_t)1ULL);
v___x_829_ = lean_usize_sub(v___x_827_, v___x_828_);
v___x_830_ = lean_usize_land(v___x_826_, v___x_829_);
v_bkt_831_ = lean_array_uget_borrowed(v_buckets_808_, v___x_830_);
v___x_832_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg(v_a_805_, v_bkt_831_);
if (v___x_832_ == 0)
{
lean_object* v___x_833_; lean_object* v_size_x27_834_; lean_object* v___x_835_; lean_object* v_buckets_x27_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; uint8_t v___x_842_; 
v___x_833_ = lean_unsigned_to_nat(1u);
v_size_x27_834_ = lean_nat_add(v_size_807_, v___x_833_);
lean_dec(v_size_807_);
lean_inc(v_bkt_831_);
v___x_835_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_835_, 0, v_a_805_);
lean_ctor_set(v___x_835_, 1, v_b_806_);
lean_ctor_set(v___x_835_, 2, v_bkt_831_);
v_buckets_x27_836_ = lean_array_uset(v_buckets_808_, v___x_830_, v___x_835_);
v___x_837_ = lean_unsigned_to_nat(4u);
v___x_838_ = lean_nat_mul(v_size_x27_834_, v___x_837_);
v___x_839_ = lean_unsigned_to_nat(3u);
v___x_840_ = lean_nat_div(v___x_838_, v___x_839_);
lean_dec(v___x_838_);
v___x_841_ = lean_array_get_size(v_buckets_x27_836_);
v___x_842_ = lean_nat_dec_le(v___x_840_, v___x_841_);
lean_dec(v___x_840_);
if (v___x_842_ == 0)
{
lean_object* v_val_843_; lean_object* v___x_845_; 
v_val_843_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3___redArg(v_buckets_x27_836_);
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 1, v_val_843_);
lean_ctor_set(v___x_810_, 0, v_size_x27_834_);
v___x_845_ = v___x_810_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_846_; 
v_reuseFailAlloc_846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_846_, 0, v_size_x27_834_);
lean_ctor_set(v_reuseFailAlloc_846_, 1, v_val_843_);
v___x_845_ = v_reuseFailAlloc_846_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
return v___x_845_;
}
}
else
{
lean_object* v___x_848_; 
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 1, v_buckets_x27_836_);
lean_ctor_set(v___x_810_, 0, v_size_x27_834_);
v___x_848_ = v___x_810_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_size_x27_834_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v_buckets_x27_836_);
v___x_848_ = v_reuseFailAlloc_849_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
return v___x_848_;
}
}
}
else
{
lean_object* v___x_850_; lean_object* v_buckets_x27_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_855_; 
lean_inc(v_bkt_831_);
v___x_850_ = lean_box(0);
v_buckets_x27_851_ = lean_array_uset(v_buckets_808_, v___x_830_, v___x_850_);
v___x_852_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4___redArg(v_a_805_, v_b_806_, v_bkt_831_);
v___x_853_ = lean_array_uset(v_buckets_x27_851_, v___x_830_, v___x_852_);
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 1, v___x_853_);
v___x_855_ = v___x_810_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_size_807_);
lean_ctor_set(v_reuseFailAlloc_856_, 1, v___x_853_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_a_860_, lean_object* v_fallback_861_, lean_object* v_x_862_){
_start:
{
if (lean_obj_tag(v_x_862_) == 0)
{
lean_inc(v_fallback_861_);
return v_fallback_861_;
}
else
{
lean_object* v_key_863_; lean_object* v_value_864_; lean_object* v_tail_865_; uint8_t v___y_867_; lean_object* v_fst_869_; lean_object* v_snd_870_; lean_object* v_fst_871_; lean_object* v_snd_872_; uint8_t v___x_873_; 
v_key_863_ = lean_ctor_get(v_x_862_, 0);
v_value_864_ = lean_ctor_get(v_x_862_, 1);
v_tail_865_ = lean_ctor_get(v_x_862_, 2);
v_fst_869_ = lean_ctor_get(v_key_863_, 0);
v_snd_870_ = lean_ctor_get(v_key_863_, 1);
v_fst_871_ = lean_ctor_get(v_a_860_, 0);
v_snd_872_ = lean_ctor_get(v_a_860_, 1);
v___x_873_ = lean_name_eq(v_fst_869_, v_fst_871_);
if (v___x_873_ == 0)
{
v___y_867_ = v___x_873_;
goto v___jp_866_;
}
else
{
uint8_t v___x_874_; uint8_t v___x_875_; uint8_t v___x_876_; 
v___x_874_ = lean_unbox(v_snd_870_);
v___x_875_ = lean_unbox(v_snd_872_);
v___x_876_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqLambdaTheoremType_beq(v___x_874_, v___x_875_);
v___y_867_ = v___x_876_;
goto v___jp_866_;
}
v___jp_866_:
{
if (v___y_867_ == 0)
{
v_x_862_ = v_tail_865_;
goto _start;
}
else
{
lean_inc(v_value_864_);
return v_value_864_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_a_877_, lean_object* v_fallback_878_, lean_object* v_x_879_){
_start:
{
lean_object* v_res_880_; 
v_res_880_ = lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_877_, v_fallback_878_, v_x_879_);
lean_dec(v_x_879_);
lean_dec(v_fallback_878_);
lean_dec_ref(v_a_877_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(lean_object* v_m_881_, lean_object* v_a_882_, lean_object* v_fallback_883_){
_start:
{
lean_object* v_buckets_884_; lean_object* v_fst_885_; lean_object* v_snd_886_; lean_object* v___x_887_; uint64_t v___y_889_; 
v_buckets_884_ = lean_ctor_get(v_m_881_, 1);
v_fst_885_ = lean_ctor_get(v_a_882_, 0);
v_snd_886_ = lean_ctor_get(v_a_882_, 1);
v___x_887_ = lean_array_get_size(v_buckets_884_);
if (lean_obj_tag(v_fst_885_) == 0)
{
uint64_t v___x_906_; 
v___x_906_ = 1723ULL;
v___y_889_ = v___x_906_;
goto v___jp_888_;
}
else
{
uint64_t v_hash_907_; 
v_hash_907_ = lean_ctor_get_uint64(v_fst_885_, sizeof(void*)*2);
v___y_889_ = v_hash_907_;
goto v___jp_888_;
}
v___jp_888_:
{
uint8_t v___x_890_; uint64_t v___x_891_; uint64_t v___x_892_; uint64_t v___x_893_; uint64_t v___x_894_; uint64_t v_fold_895_; uint64_t v___x_896_; uint64_t v___x_897_; uint64_t v___x_898_; size_t v___x_899_; size_t v___x_900_; size_t v___x_901_; size_t v___x_902_; size_t v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; 
v___x_890_ = lean_unbox(v_snd_886_);
v___x_891_ = lp_mathlib_Mathlib_Meta_FunProp_instHashableLambdaTheoremType_hash(v___x_890_);
v___x_892_ = lean_uint64_mix_hash(v___y_889_, v___x_891_);
v___x_893_ = 32ULL;
v___x_894_ = lean_uint64_shift_right(v___x_892_, v___x_893_);
v_fold_895_ = lean_uint64_xor(v___x_892_, v___x_894_);
v___x_896_ = 16ULL;
v___x_897_ = lean_uint64_shift_right(v_fold_895_, v___x_896_);
v___x_898_ = lean_uint64_xor(v_fold_895_, v___x_897_);
v___x_899_ = lean_uint64_to_usize(v___x_898_);
v___x_900_ = lean_usize_of_nat(v___x_887_);
v___x_901_ = ((size_t)1ULL);
v___x_902_ = lean_usize_sub(v___x_900_, v___x_901_);
v___x_903_ = lean_usize_land(v___x_899_, v___x_902_);
v___x_904_ = lean_array_uget_borrowed(v_buckets_884_, v___x_903_);
v___x_905_ = lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_882_, v_fallback_883_, v___x_904_);
return v___x_905_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_m_908_, lean_object* v_a_909_, lean_object* v_fallback_910_){
_start:
{
lean_object* v_res_911_; 
v_res_911_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(v_m_908_, v_a_909_, v_fallback_910_);
lean_dec(v_fallback_910_);
lean_dec_ref(v_a_909_);
lean_dec_ref(v_m_908_);
return v_res_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object* v_d_914_, lean_object* v_e_915_){
_start:
{
lean_object* v_funPropName_916_; lean_object* v_thmArgs_917_; uint8_t v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v_es_922_; lean_object* v___x_923_; lean_object* v___x_924_; 
v_funPropName_916_ = lean_ctor_get(v_e_915_, 0);
v_thmArgs_917_ = lean_ctor_get(v_e_915_, 2);
v___x_918_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type(v_thmArgs_917_);
v___x_919_ = lean_box(v___x_918_);
lean_inc(v_funPropName_916_);
v___x_920_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_920_, 0, v_funPropName_916_);
lean_ctor_set(v___x_920_, 1, v___x_919_);
v___x_921_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v_es_922_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(v_d_914_, v___x_920_, v___x_921_);
v___x_923_ = lean_array_push(v_es_922_, v_e_915_);
v___x_924_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1___redArg(v_d_914_, v___x_920_, v___x_923_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(lean_object* v___y_925_){
_start:
{
lean_inc_ref(v___y_925_);
return v___y_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object* v___y_926_){
_start:
{
lean_object* v_res_927_; 
v_res_927_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(v___y_926_);
lean_dec_ref(v___y_926_);
return v_res_927_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_940_; lean_object* v___f_941_; lean_object* v___x_942_; lean_object* v___f_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___f_940_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v___f_941_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v___x_942_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default___closed__1);
v___f_943_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v___x_944_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v___x_945_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
lean_ctor_set(v___x_945_, 1, v___f_943_);
lean_ctor_set(v___x_945_, 2, v___x_942_);
lean_ctor_set(v___x_945_, 3, v___f_941_);
lean_ctor_set(v___x_945_, 4, v___f_940_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_947_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_);
v___x_948_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_947_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2____boxed(lean_object* v_a_949_){
_start:
{
lean_object* v_res_950_; 
v_res_950_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_();
return v_res_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b2_951_, lean_object* v_m_952_, lean_object* v_a_953_, lean_object* v_fallback_954_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(v_m_952_, v_a_953_, v_fallback_954_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b2_956_, lean_object* v_m_957_, lean_object* v_a_958_, lean_object* v_fallback_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0(v_00_u03b2_956_, v_m_957_, v_a_958_, v_fallback_959_);
lean_dec(v_fallback_959_);
lean_dec_ref(v_a_958_);
lean_dec_ref(v_m_957_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b2_961_, lean_object* v_m_962_, lean_object* v_a_963_, lean_object* v_b_964_){
_start:
{
lean_object* v___x_965_; 
v___x_965_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1___redArg(v_m_962_, v_a_963_, v_b_964_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b2_966_, lean_object* v_a_967_, lean_object* v_fallback_968_, lean_object* v_x_969_){
_start:
{
lean_object* v___x_970_; 
v___x_970_ = lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___redArg(v_a_967_, v_fallback_968_, v_x_969_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b2_971_, lean_object* v_a_972_, lean_object* v_fallback_973_, lean_object* v_x_974_){
_start:
{
lean_object* v_res_975_; 
v_res_975_ = lp_mathlib_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b2_971_, v_a_972_, v_fallback_973_, v_x_974_);
lean_dec(v_x_974_);
lean_dec(v_fallback_973_);
lean_dec_ref(v_a_972_);
return v_res_975_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_00_u03b2_976_, lean_object* v_a_977_, lean_object* v_x_978_){
_start:
{
uint8_t v___x_979_; 
v___x_979_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___redArg(v_a_977_, v_x_978_);
return v___x_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_00_u03b2_980_, lean_object* v_a_981_, lean_object* v_x_982_){
_start:
{
uint8_t v_res_983_; lean_object* v_r_984_; 
v_res_983_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__2(v_00_u03b2_980_, v_a_981_, v_x_982_);
lean_dec(v_x_982_);
lean_dec_ref(v_a_981_);
v_r_984_ = lean_box(v_res_983_);
return v_r_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3(lean_object* v_00_u03b2_985_, lean_object* v_data_986_){
_start:
{
lean_object* v___x_987_; 
v___x_987_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3___redArg(v_data_986_);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4(lean_object* v_00_u03b2_988_, lean_object* v_a_989_, lean_object* v_b_990_, lean_object* v_x_991_){
_start:
{
lean_object* v___x_992_; 
v___x_992_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__4___redArg(v_a_989_, v_b_990_, v_x_991_);
return v___x_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4(lean_object* v_00_u03b2_993_, lean_object* v_i_994_, lean_object* v_source_995_, lean_object* v_target_996_){
_start:
{
lean_object* v___x_997_; 
v___x_997_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4___redArg(v_i_994_, v_source_995_, v_target_996_);
return v___x_997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_998_, lean_object* v_x_999_, lean_object* v_x_1000_){
_start:
{
lean_object* v___x_1001_; 
v___x_1001_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__1_spec__3_spec__4_spec__5___redArg(v_x_999_, v_x_1000_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg(lean_object* v_funPropName_1002_, uint8_t v_type_1003_, lean_object* v_a_1004_){
_start:
{
lean_object* v___x_1006_; lean_object* v_env_1007_; lean_object* v___x_1008_; lean_object* v_ext_1009_; lean_object* v_toEnvExtension_1010_; lean_object* v_asyncMode_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1006_ = lean_st_ref_get(v_a_1004_);
v_env_1007_ = lean_ctor_get(v___x_1006_, 0);
lean_inc_ref(v_env_1007_);
lean_dec(v___x_1006_);
v___x_1008_ = lp_mathlib_Mathlib_Meta_FunProp_lambdaTheoremsExt;
v_ext_1009_ = lean_ctor_get(v___x_1008_, 1);
v_toEnvExtension_1010_ = lean_ctor_get(v_ext_1009_, 0);
v_asyncMode_1011_ = lean_ctor_get(v_toEnvExtension_1010_, 2);
v___x_1012_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default;
v___x_1013_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1012_, v___x_1008_, v_env_1007_, v_asyncMode_1011_);
v___x_1014_ = lean_box(v_type_1003_);
v___x_1015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1015_, 0, v_funPropName_1002_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
v___x_1016_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_));
v___x_1017_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2__spec__0___redArg(v___x_1013_, v___x_1015_, v___x_1016_);
lean_dec_ref_known(v___x_1015_, 2);
lean_dec(v___x_1013_);
v___x_1018_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1018_, 0, v___x_1017_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg___boxed(lean_object* v_funPropName_1019_, lean_object* v_type_1020_, lean_object* v_a_1021_, lean_object* v_a_1022_){
_start:
{
uint8_t v_type_boxed_1023_; lean_object* v_res_1024_; 
v_type_boxed_1023_ = lean_unbox(v_type_1020_);
v_res_1024_ = lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg(v_funPropName_1019_, v_type_boxed_1023_, v_a_1021_);
lean_dec(v_a_1021_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems(lean_object* v_funPropName_1025_, uint8_t v_type_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_){
_start:
{
lean_object* v___x_1030_; 
v___x_1030_ = lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___redArg(v_funPropName_1025_, v_type_1026_, v_a_1028_);
return v___x_1030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems___boxed(lean_object* v_funPropName_1031_, lean_object* v_type_1032_, lean_object* v_a_1033_, lean_object* v_a_1034_, lean_object* v_a_1035_){
_start:
{
uint8_t v_type_boxed_1036_; lean_object* v_res_1037_; 
v_type_boxed_1036_ = lean_unbox(v_type_1032_);
v_res_1037_ = lp_mathlib_Mathlib_Meta_FunProp_getLambdaTheorems(v_funPropName_1031_, v_type_boxed_1036_, v_a_1033_, v_a_1034_);
lean_dec(v_a_1034_);
lean_dec_ref(v_a_1033_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx(uint8_t v_x_1038_){
_start:
{
if (v_x_1038_ == 0)
{
lean_object* v___x_1039_; 
v___x_1039_ = lean_unsigned_to_nat(0u);
return v___x_1039_;
}
else
{
lean_object* v___x_1040_; 
v___x_1040_ = lean_unsigned_to_nat(1u);
return v___x_1040_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx___boxed(lean_object* v_x_1041_){
_start:
{
uint8_t v_x_boxed_1042_; lean_object* v_res_1043_; 
v_x_boxed_1042_ = lean_unbox(v_x_1041_);
v_res_1043_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx(v_x_boxed_1042_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___redArg(lean_object* v_k_1044_){
_start:
{
lean_inc(v_k_1044_);
return v_k_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___redArg___boxed(lean_object* v_k_1045_){
_start:
{
lean_object* v_res_1046_; 
v_res_1046_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___redArg(v_k_1045_);
lean_dec(v_k_1045_);
return v_res_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim(lean_object* v_motive_1047_, lean_object* v_ctorIdx_1048_, uint8_t v_t_1049_, lean_object* v_h_1050_, lean_object* v_k_1051_){
_start:
{
lean_inc(v_k_1051_);
return v_k_1051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim___boxed(lean_object* v_motive_1052_, lean_object* v_ctorIdx_1053_, lean_object* v_t_1054_, lean_object* v_h_1055_, lean_object* v_k_1056_){
_start:
{
uint8_t v_t_boxed_1057_; lean_object* v_res_1058_; 
v_t_boxed_1057_ = lean_unbox(v_t_1054_);
v_res_1058_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorElim(v_motive_1052_, v_ctorIdx_1053_, v_t_boxed_1057_, v_h_1055_, v_k_1056_);
lean_dec(v_k_1056_);
lean_dec(v_ctorIdx_1053_);
return v_res_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___redArg(lean_object* v_uncurried_1059_){
_start:
{
lean_inc(v_uncurried_1059_);
return v_uncurried_1059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___redArg___boxed(lean_object* v_uncurried_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___redArg(v_uncurried_1060_);
lean_dec(v_uncurried_1060_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim(lean_object* v_motive_1062_, uint8_t v_t_1063_, lean_object* v_h_1064_, lean_object* v_uncurried_1065_){
_start:
{
lean_inc(v_uncurried_1065_);
return v_uncurried_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim___boxed(lean_object* v_motive_1066_, lean_object* v_t_1067_, lean_object* v_h_1068_, lean_object* v_uncurried_1069_){
_start:
{
uint8_t v_t_boxed_1070_; lean_object* v_res_1071_; 
v_t_boxed_1070_ = lean_unbox(v_t_1067_);
v_res_1071_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_uncurried_elim(v_motive_1066_, v_t_boxed_1070_, v_h_1068_, v_uncurried_1069_);
lean_dec(v_uncurried_1069_);
return v_res_1071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___redArg(lean_object* v_comp_1072_){
_start:
{
lean_inc(v_comp_1072_);
return v_comp_1072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___redArg___boxed(lean_object* v_comp_1073_){
_start:
{
lean_object* v_res_1074_; 
v_res_1074_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___redArg(v_comp_1073_);
lean_dec(v_comp_1073_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim(lean_object* v_motive_1075_, uint8_t v_t_1076_, lean_object* v_h_1077_, lean_object* v_comp_1078_){
_start:
{
lean_inc(v_comp_1078_);
return v_comp_1078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim___boxed(lean_object* v_motive_1079_, lean_object* v_t_1080_, lean_object* v_h_1081_, lean_object* v_comp_1082_){
_start:
{
uint8_t v_t_boxed_1083_; lean_object* v_res_1084_; 
v_t_boxed_1083_ = lean_unbox(v_t_1080_);
v_res_1084_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_comp_elim(v_motive_1079_, v_t_boxed_1083_, v_h_1081_, v_comp_1082_);
lean_dec(v_comp_1082_);
return v_res_1084_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm_default(void){
_start:
{
uint8_t v___x_1085_; 
v___x_1085_ = 0;
return v___x_1085_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm(void){
_start:
{
uint8_t v___x_1086_; 
v___x_1086_ = 0;
return v___x_1086_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq(uint8_t v_x_1087_, uint8_t v_y_1088_){
_start:
{
lean_object* v___x_1089_; lean_object* v___x_1090_; uint8_t v___x_1091_; 
v___x_1089_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx(v_x_1087_);
v___x_1090_ = lp_mathlib_Mathlib_Meta_FunProp_TheoremForm_ctorIdx(v_y_1088_);
v___x_1091_ = lean_nat_dec_eq(v___x_1089_, v___x_1090_);
lean_dec(v___x_1090_);
lean_dec(v___x_1089_);
return v___x_1091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq___boxed(lean_object* v_x_1092_, lean_object* v_y_1093_){
_start:
{
uint8_t v_x_17__boxed_1094_; uint8_t v_y_18__boxed_1095_; uint8_t v_res_1096_; lean_object* v_r_1097_; 
v_x_17__boxed_1094_ = lean_unbox(v_x_1092_);
v_y_18__boxed_1095_ = lean_unbox(v_y_1093_);
v_res_1096_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq(v_x_17__boxed_1094_, v_y_18__boxed_1095_);
v_r_1097_ = lean_box(v_res_1096_);
return v_r_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr(uint8_t v_x_1106_, lean_object* v_prec_1107_){
_start:
{
lean_object* v___y_1109_; lean_object* v___y_1116_; 
if (v_x_1106_ == 0)
{
lean_object* v___x_1122_; uint8_t v___x_1123_; 
v___x_1122_ = lean_unsigned_to_nat(1024u);
v___x_1123_ = lean_nat_dec_le(v___x_1122_, v_prec_1107_);
if (v___x_1123_ == 0)
{
lean_object* v___x_1124_; 
v___x_1124_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_1109_ = v___x_1124_;
goto v___jp_1108_;
}
else
{
lean_object* v___x_1125_; 
v___x_1125_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_1109_ = v___x_1125_;
goto v___jp_1108_;
}
}
else
{
lean_object* v___x_1126_; uint8_t v___x_1127_; 
v___x_1126_ = lean_unsigned_to_nat(1024u);
v___x_1127_ = lean_nat_dec_le(v___x_1126_, v_prec_1107_);
if (v___x_1127_ == 0)
{
lean_object* v___x_1128_; 
v___x_1128_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__8);
v___y_1116_ = v___x_1128_;
goto v___jp_1115_;
}
else
{
lean_object* v___x_1129_; 
v___x_1129_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremArgs_repr___closed__9);
v___y_1116_ = v___x_1129_;
goto v___jp_1115_;
}
}
v___jp_1108_:
{
lean_object* v___x_1110_; lean_object* v___x_1111_; uint8_t v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; 
v___x_1110_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__1));
lean_inc(v___y_1109_);
v___x_1111_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1111_, 0, v___y_1109_);
lean_ctor_set(v___x_1111_, 1, v___x_1110_);
v___x_1112_ = 0;
v___x_1113_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1113_, 0, v___x_1111_);
lean_ctor_set_uint8(v___x_1113_, sizeof(void*)*1, v___x_1112_);
v___x_1114_ = l_Repr_addAppParen(v___x_1113_, v_prec_1107_);
return v___x_1114_;
}
v___jp_1115_:
{
lean_object* v___x_1117_; lean_object* v___x_1118_; uint8_t v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; 
v___x_1117_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___closed__3));
lean_inc(v___y_1116_);
v___x_1118_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1118_, 0, v___y_1116_);
lean_ctor_set(v___x_1118_, 1, v___x_1117_);
v___x_1119_ = 0;
v___x_1120_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1120_, 0, v___x_1118_);
lean_ctor_set_uint8(v___x_1120_, sizeof(void*)*1, v___x_1119_);
v___x_1121_ = l_Repr_addAppParen(v___x_1120_, v_prec_1107_);
return v___x_1121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr___boxed(lean_object* v_x_1130_, lean_object* v_prec_1131_){
_start:
{
uint8_t v_x_117__boxed_1132_; lean_object* v_res_1133_; 
v_x_117__boxed_1132_ = lean_unbox(v_x_1130_);
v_res_1133_ = lp_mathlib_Mathlib_Meta_FunProp_instReprTheoremForm_repr(v_x_117__boxed_1132_, v_prec_1131_);
lean_dec(v_prec_1131_);
return v_res_1133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0(uint8_t v_x_1138_){
_start:
{
if (v_x_1138_ == 0)
{
lean_object* v___x_1139_; 
v___x_1139_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__0));
return v___x_1139_;
}
else
{
lean_object* v___x_1140_; 
v___x_1140_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__1));
return v___x_1140_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___boxed(lean_object* v_x_1141_){
_start:
{
uint8_t v_x_boxed_1142_; lean_object* v_res_1143_; 
v_x_boxed_1142_ = lean_unbox(v_x_1141_);
v_res_1143_ = lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0(v_x_boxed_1142_);
return v_res_1143_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm(lean_object* v_x_1146_){
_start:
{
if (lean_obj_tag(v_x_1146_) == 1)
{
uint8_t v___x_1147_; 
v___x_1147_ = 0;
return v___x_1147_;
}
else
{
uint8_t v___x_1148_; 
v___x_1148_ = 1;
return v___x_1148_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm___boxed(lean_object* v_x_1149_){
_start:
{
uint8_t v_res_1150_; lean_object* v_r_1151_; 
v_res_1150_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm(v_x_1149_);
lean_dec(v_x_1149_);
v_r_1151_ = lean_box(v_res_1150_);
return v_r_1151_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1(void){
_start:
{
uint8_t v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_1154_ = 0;
v___x_1155_ = lean_unsigned_to_nat(1000u);
v___x_1156_ = lean_unsigned_to_nat(0u);
v___x_1157_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__0));
v___x_1158_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default;
v___x_1159_ = lean_box(0);
v___x_1160_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1160_, 0, v___x_1159_);
lean_ctor_set(v___x_1160_, 1, v___x_1158_);
lean_ctor_set(v___x_1160_, 2, v___x_1158_);
lean_ctor_set(v___x_1160_, 3, v___x_1157_);
lean_ctor_set(v___x_1160_, 4, v___x_1156_);
lean_ctor_set(v___x_1160_, 5, v___x_1155_);
lean_ctor_set_uint8(v___x_1160_, sizeof(void*)*6, v___x_1154_);
return v___x_1160_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default(void){
_start:
{
lean_object* v___x_1161_; 
v___x_1161_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default___closed__1);
return v___x_1161_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem(void){
_start:
{
lean_object* v___x_1162_; 
v___x_1162_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default;
return v___x_1162_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg(lean_object* v_xs_1163_, lean_object* v_ys_1164_, lean_object* v_x_1165_){
_start:
{
lean_object* v_zero_1166_; uint8_t v_isZero_1167_; 
v_zero_1166_ = lean_unsigned_to_nat(0u);
v_isZero_1167_ = lean_nat_dec_eq(v_x_1165_, v_zero_1166_);
if (v_isZero_1167_ == 1)
{
lean_dec(v_x_1165_);
return v_isZero_1167_;
}
else
{
lean_object* v_one_1168_; lean_object* v_n_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; uint8_t v___x_1172_; 
v_one_1168_ = lean_unsigned_to_nat(1u);
v_n_1169_ = lean_nat_sub(v_x_1165_, v_one_1168_);
lean_dec(v_x_1165_);
v___x_1170_ = lean_array_fget_borrowed(v_xs_1163_, v_n_1169_);
v___x_1171_ = lean_array_fget_borrowed(v_ys_1164_, v_n_1169_);
v___x_1172_ = lean_nat_dec_eq(v___x_1170_, v___x_1171_);
if (v___x_1172_ == 0)
{
lean_dec(v_n_1169_);
return v___x_1172_;
}
else
{
v_x_1165_ = v_n_1169_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg___boxed(lean_object* v_xs_1174_, lean_object* v_ys_1175_, lean_object* v_x_1176_){
_start:
{
uint8_t v_res_1177_; lean_object* v_r_1178_; 
v_res_1177_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg(v_xs_1174_, v_ys_1175_, v_x_1176_);
lean_dec_ref(v_ys_1175_);
lean_dec_ref(v_xs_1174_);
v_r_1178_ = lean_box(v_res_1177_);
return v_r_1178_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq(lean_object* v_x_1179_, lean_object* v_x_1180_){
_start:
{
lean_object* v_funPropName_1181_; lean_object* v_thmOrigin_1182_; lean_object* v_funOrigin_1183_; lean_object* v_mainArgs_1184_; lean_object* v_appliedArgs_1185_; lean_object* v_priority_1186_; uint8_t v_form_1187_; lean_object* v_funPropName_1188_; lean_object* v_thmOrigin_1189_; lean_object* v_funOrigin_1190_; lean_object* v_mainArgs_1191_; lean_object* v_appliedArgs_1192_; lean_object* v_priority_1193_; uint8_t v_form_1194_; uint8_t v___x_1195_; 
v_funPropName_1181_ = lean_ctor_get(v_x_1179_, 0);
v_thmOrigin_1182_ = lean_ctor_get(v_x_1179_, 1);
v_funOrigin_1183_ = lean_ctor_get(v_x_1179_, 2);
v_mainArgs_1184_ = lean_ctor_get(v_x_1179_, 3);
v_appliedArgs_1185_ = lean_ctor_get(v_x_1179_, 4);
v_priority_1186_ = lean_ctor_get(v_x_1179_, 5);
v_form_1187_ = lean_ctor_get_uint8(v_x_1179_, sizeof(void*)*6);
v_funPropName_1188_ = lean_ctor_get(v_x_1180_, 0);
v_thmOrigin_1189_ = lean_ctor_get(v_x_1180_, 1);
v_funOrigin_1190_ = lean_ctor_get(v_x_1180_, 2);
v_mainArgs_1191_ = lean_ctor_get(v_x_1180_, 3);
v_appliedArgs_1192_ = lean_ctor_get(v_x_1180_, 4);
v_priority_1193_ = lean_ctor_get(v_x_1180_, 5);
v_form_1194_ = lean_ctor_get_uint8(v_x_1180_, sizeof(void*)*6);
v___x_1195_ = lean_name_eq(v_funPropName_1181_, v_funPropName_1188_);
if (v___x_1195_ == 0)
{
return v___x_1195_;
}
else
{
uint8_t v___x_1196_; 
v___x_1196_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(v_thmOrigin_1182_, v_thmOrigin_1189_);
if (v___x_1196_ == 0)
{
return v___x_1196_;
}
else
{
uint8_t v___x_1197_; 
v___x_1197_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(v_funOrigin_1183_, v_funOrigin_1190_);
if (v___x_1197_ == 0)
{
return v___x_1197_;
}
else
{
lean_object* v___x_1198_; lean_object* v___x_1199_; uint8_t v___x_1200_; 
v___x_1198_ = lean_array_get_size(v_mainArgs_1184_);
v___x_1199_ = lean_array_get_size(v_mainArgs_1191_);
v___x_1200_ = lean_nat_dec_eq(v___x_1198_, v___x_1199_);
if (v___x_1200_ == 0)
{
return v___x_1200_;
}
else
{
uint8_t v___x_1201_; 
v___x_1201_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg(v_mainArgs_1184_, v_mainArgs_1191_, v___x_1198_);
if (v___x_1201_ == 0)
{
return v___x_1201_;
}
else
{
uint8_t v___x_1202_; 
v___x_1202_ = lean_nat_dec_eq(v_appliedArgs_1185_, v_appliedArgs_1192_);
if (v___x_1202_ == 0)
{
return v___x_1202_;
}
else
{
uint8_t v___x_1203_; 
v___x_1203_ = lean_nat_dec_eq(v_priority_1186_, v_priority_1193_);
if (v___x_1203_ == 0)
{
return v___x_1203_;
}
else
{
uint8_t v___x_1204_; 
v___x_1204_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqTheoremForm_beq(v_form_1187_, v_form_1194_);
return v___x_1204_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq___boxed(lean_object* v_x_1205_, lean_object* v_x_1206_){
_start:
{
uint8_t v_res_1207_; lean_object* v_r_1208_; 
v_res_1207_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq(v_x_1205_, v_x_1206_);
lean_dec_ref(v_x_1206_);
lean_dec_ref(v_x_1205_);
v_r_1208_ = lean_box(v_res_1207_);
return v_r_1208_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0(lean_object* v_xs_1209_, lean_object* v_ys_1210_, lean_object* v_hsz_1211_, lean_object* v_x_1212_, lean_object* v_x_1213_){
_start:
{
uint8_t v___x_1214_; 
v___x_1214_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___redArg(v_xs_1209_, v_ys_1210_, v_x_1212_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0___boxed(lean_object* v_xs_1215_, lean_object* v_ys_1216_, lean_object* v_hsz_1217_, lean_object* v_x_1218_, lean_object* v_x_1219_){
_start:
{
uint8_t v_res_1220_; lean_object* v_r_1221_; 
v_res_1220_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Meta_FunProp_instBEqFunctionTheorem_beq_spec__0(v_xs_1215_, v_ys_1216_, v_hsz_1217_, v_x_1218_, v_x_1219_);
lean_dec_ref(v_ys_1216_);
lean_dec_ref(v_xs_1215_);
v_r_1221_ = lean_box(v_res_1220_);
return v_r_1221_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems_default(void){
_start:
{
lean_object* v___x_1224_; 
v___x_1224_ = lean_box(1);
return v___x_1224_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems(void){
_start:
{
lean_object* v___x_1225_; 
v___x_1225_ = lean_box(1);
return v___x_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionTheorem_getProof(lean_object* v_thm_1226_, lean_object* v_a_1227_, lean_object* v_a_1228_, lean_object* v_a_1229_, lean_object* v_a_1230_){
_start:
{
lean_object* v_thmOrigin_1232_; 
v_thmOrigin_1232_ = lean_ctor_get(v_thm_1226_, 1);
lean_inc_ref(v_thmOrigin_1232_);
lean_dec_ref(v_thm_1226_);
if (lean_obj_tag(v_thmOrigin_1232_) == 0)
{
lean_object* v_name_1233_; lean_object* v___x_1234_; 
v_name_1233_ = lean_ctor_get(v_thmOrigin_1232_, 0);
lean_inc(v_name_1233_);
lean_dec_ref_known(v_thmOrigin_1232_, 1);
v___x_1234_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_name_1233_, v_a_1227_, v_a_1228_, v_a_1229_, v_a_1230_);
return v___x_1234_;
}
else
{
lean_object* v_fvarId_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1243_; 
v_fvarId_1235_ = lean_ctor_get(v_thmOrigin_1232_, 0);
v_isSharedCheck_1243_ = !lean_is_exclusive(v_thmOrigin_1232_);
if (v_isSharedCheck_1243_ == 0)
{
v___x_1237_ = v_thmOrigin_1232_;
v_isShared_1238_ = v_isSharedCheck_1243_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_fvarId_1235_);
lean_dec(v_thmOrigin_1232_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1243_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v___x_1239_; lean_object* v___x_1241_; 
v___x_1239_ = l_Lean_Expr_fvar___override(v_fvarId_1235_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set_tag(v___x_1237_, 0);
lean_ctor_set(v___x_1237_, 0, v___x_1239_);
v___x_1241_ = v___x_1237_;
goto v_reusejp_1240_;
}
else
{
lean_object* v_reuseFailAlloc_1242_; 
v_reuseFailAlloc_1242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1242_, 0, v___x_1239_);
v___x_1241_ = v_reuseFailAlloc_1242_;
goto v_reusejp_1240_;
}
v_reusejp_1240_:
{
return v___x_1241_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionTheorem_getProof___boxed(lean_object* v_thm_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_, lean_object* v_a_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_){
_start:
{
lean_object* v_res_1250_; 
v_res_1250_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionTheorem_getProof(v_thm_1244_, v_a_1245_, v_a_1246_, v_a_1247_, v_a_1248_);
lean_dec(v_a_1248_);
lean_dec_ref(v_a_1247_);
lean_dec(v_a_1246_);
lean_dec_ref(v_a_1245_);
return v_res_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0(lean_object* v_e_1253_, lean_object* v_thms_1254_){
_start:
{
lean_object* v___y_1256_; 
if (lean_obj_tag(v_thms_1254_) == 0)
{
lean_object* v___x_1259_; 
v___x_1259_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0___closed__0));
v___y_1256_ = v___x_1259_;
goto v___jp_1255_;
}
else
{
lean_object* v_val_1260_; 
v_val_1260_ = lean_ctor_get(v_thms_1254_, 0);
lean_inc(v_val_1260_);
lean_dec_ref_known(v_thms_1254_, 1);
v___y_1256_ = v_val_1260_;
goto v___jp_1255_;
}
v___jp_1255_:
{
lean_object* v___x_1257_; lean_object* v___x_1258_; 
v___x_1257_ = lean_array_push(v___y_1256_, v_e_1253_);
v___x_1258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1258_, 0, v___x_1257_);
return v___x_1258_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(lean_object* v_e_1261_, lean_object* v_k_1262_, lean_object* v_t_1263_){
_start:
{
if (lean_obj_tag(v_t_1263_) == 0)
{
lean_object* v_size_1264_; lean_object* v_k_1265_; lean_object* v_v_1266_; lean_object* v_l_1267_; lean_object* v_r_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1283_; 
v_size_1264_ = lean_ctor_get(v_t_1263_, 0);
v_k_1265_ = lean_ctor_get(v_t_1263_, 1);
v_v_1266_ = lean_ctor_get(v_t_1263_, 2);
v_l_1267_ = lean_ctor_get(v_t_1263_, 3);
v_r_1268_ = lean_ctor_get(v_t_1263_, 4);
v_isSharedCheck_1283_ = !lean_is_exclusive(v_t_1263_);
if (v_isSharedCheck_1283_ == 0)
{
v___x_1270_ = v_t_1263_;
v_isShared_1271_ = v_isSharedCheck_1283_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_r_1268_);
lean_inc(v_l_1267_);
lean_inc(v_v_1266_);
lean_inc(v_k_1265_);
lean_inc(v_size_1264_);
lean_dec(v_t_1263_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1283_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
uint8_t v___x_1272_; 
v___x_1272_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_1262_, v_k_1265_);
switch(v___x_1272_)
{
case 0:
{
lean_object* v_impl_1273_; lean_object* v___x_1274_; 
lean_del_object(v___x_1270_);
lean_dec(v_size_1264_);
v_impl_1273_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(v_e_1261_, v_k_1262_, v_l_1267_);
v___x_1274_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_1265_, v_v_1266_, v_impl_1273_, v_r_1268_);
return v___x_1274_;
}
case 1:
{
lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v_val_1277_; lean_object* v___x_1279_; 
lean_dec(v_k_1265_);
v___x_1275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1275_, 0, v_v_1266_);
v___x_1276_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0(v_e_1261_, v___x_1275_);
v_val_1277_ = lean_ctor_get(v___x_1276_, 0);
lean_inc(v_val_1277_);
lean_dec(v___x_1276_);
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 2, v_val_1277_);
lean_ctor_set(v___x_1270_, 1, v_k_1262_);
v___x_1279_ = v___x_1270_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v_size_1264_);
lean_ctor_set(v_reuseFailAlloc_1280_, 1, v_k_1262_);
lean_ctor_set(v_reuseFailAlloc_1280_, 2, v_val_1277_);
lean_ctor_set(v_reuseFailAlloc_1280_, 3, v_l_1267_);
lean_ctor_set(v_reuseFailAlloc_1280_, 4, v_r_1268_);
v___x_1279_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
return v___x_1279_;
}
}
default: 
{
lean_object* v_impl_1281_; lean_object* v___x_1282_; 
lean_del_object(v___x_1270_);
lean_dec(v_size_1264_);
v_impl_1281_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(v_e_1261_, v_k_1262_, v_r_1268_);
v___x_1282_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_1265_, v_v_1266_, v_l_1267_, v_impl_1281_);
return v___x_1282_;
}
}
}
}
else
{
lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v_val_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1284_ = lean_box(0);
v___x_1285_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0(v_e_1261_, v___x_1284_);
v_val_1286_ = lean_ctor_get(v___x_1285_, 0);
lean_inc(v_val_1286_);
lean_dec(v___x_1285_);
v___x_1287_ = lean_unsigned_to_nat(1u);
v___x_1288_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
lean_ctor_set(v___x_1288_, 1, v_k_1262_);
lean_ctor_set(v___x_1288_, 2, v_val_1286_);
lean_ctor_set(v___x_1288_, 3, v_t_1263_);
lean_ctor_set(v___x_1288_, 4, v_t_1263_);
return v___x_1288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg___lam__0(lean_object* v_e_1289_, lean_object* v_funProperties_1290_){
_start:
{
lean_object* v___y_1292_; 
if (lean_obj_tag(v_funProperties_1290_) == 0)
{
lean_object* v___x_1296_; 
v___x_1296_ = lean_box(1);
v___y_1292_ = v___x_1296_;
goto v___jp_1291_;
}
else
{
lean_object* v_val_1297_; 
v_val_1297_ = lean_ctor_get(v_funProperties_1290_, 0);
lean_inc(v_val_1297_);
lean_dec_ref_known(v_funProperties_1290_, 1);
v___y_1292_ = v_val_1297_;
goto v___jp_1291_;
}
v___jp_1291_:
{
lean_object* v_funPropName_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v_funPropName_1293_ = lean_ctor_get(v_e_1289_, 0);
lean_inc(v_funPropName_1293_);
v___x_1294_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(v_e_1289_, v_funPropName_1293_, v___y_1292_);
v___x_1295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1295_, 0, v___x_1294_);
return v___x_1295_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(lean_object* v_e_1298_, lean_object* v_k_1299_, lean_object* v_t_1300_){
_start:
{
if (lean_obj_tag(v_t_1300_) == 0)
{
lean_object* v_size_1301_; lean_object* v_k_1302_; lean_object* v_v_1303_; lean_object* v_l_1304_; lean_object* v_r_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1320_; 
v_size_1301_ = lean_ctor_get(v_t_1300_, 0);
v_k_1302_ = lean_ctor_get(v_t_1300_, 1);
v_v_1303_ = lean_ctor_get(v_t_1300_, 2);
v_l_1304_ = lean_ctor_get(v_t_1300_, 3);
v_r_1305_ = lean_ctor_get(v_t_1300_, 4);
v_isSharedCheck_1320_ = !lean_is_exclusive(v_t_1300_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1307_ = v_t_1300_;
v_isShared_1308_ = v_isSharedCheck_1320_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_r_1305_);
lean_inc(v_l_1304_);
lean_inc(v_v_1303_);
lean_inc(v_k_1302_);
lean_inc(v_size_1301_);
lean_dec(v_t_1300_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1320_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
uint8_t v___x_1309_; 
v___x_1309_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_1299_, v_k_1302_);
switch(v___x_1309_)
{
case 0:
{
lean_object* v_impl_1310_; lean_object* v___x_1311_; 
lean_del_object(v___x_1307_);
lean_dec(v_size_1301_);
v_impl_1310_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(v_e_1298_, v_k_1299_, v_l_1304_);
v___x_1311_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_1302_, v_v_1303_, v_impl_1310_, v_r_1305_);
return v___x_1311_;
}
case 1:
{
lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v_val_1314_; lean_object* v___x_1316_; 
lean_dec(v_k_1302_);
v___x_1312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1312_, 0, v_v_1303_);
v___x_1313_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg___lam__0(v_e_1298_, v___x_1312_);
v_val_1314_ = lean_ctor_get(v___x_1313_, 0);
lean_inc(v_val_1314_);
lean_dec(v___x_1313_);
if (v_isShared_1308_ == 0)
{
lean_ctor_set(v___x_1307_, 2, v_val_1314_);
lean_ctor_set(v___x_1307_, 1, v_k_1299_);
v___x_1316_ = v___x_1307_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_size_1301_);
lean_ctor_set(v_reuseFailAlloc_1317_, 1, v_k_1299_);
lean_ctor_set(v_reuseFailAlloc_1317_, 2, v_val_1314_);
lean_ctor_set(v_reuseFailAlloc_1317_, 3, v_l_1304_);
lean_ctor_set(v_reuseFailAlloc_1317_, 4, v_r_1305_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
default: 
{
lean_object* v_impl_1318_; lean_object* v___x_1319_; 
lean_del_object(v___x_1307_);
lean_dec(v_size_1301_);
v_impl_1318_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(v_e_1298_, v_k_1299_, v_r_1305_);
v___x_1319_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_1302_, v_v_1303_, v_l_1304_, v_impl_1318_);
return v___x_1319_;
}
}
}
}
else
{
lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v_val_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; 
v___x_1321_ = lean_box(0);
v___x_1322_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg___lam__0(v_e_1298_, v___x_1321_);
v_val_1323_ = lean_ctor_get(v___x_1322_, 0);
lean_inc(v_val_1323_);
lean_dec(v___x_1322_);
v___x_1324_ = lean_unsigned_to_nat(1u);
v___x_1325_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1325_, 0, v___x_1324_);
lean_ctor_set(v___x_1325_, 1, v_k_1299_);
lean_ctor_set(v___x_1325_, 2, v_val_1323_);
lean_ctor_set(v___x_1325_, 3, v_t_1300_);
lean_ctor_set(v___x_1325_, 4, v_t_1300_);
return v___x_1325_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object* v_d_1326_, lean_object* v_e_1327_){
_start:
{
lean_object* v_funOrigin_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; 
v_funOrigin_1328_ = lean_ctor_get(v_e_1327_, 2);
v___x_1329_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_name(v_funOrigin_1328_);
v___x_1330_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(v_e_1327_, v___x_1329_, v_d_1326_);
return v___x_1330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object* v_x_1331_, lean_object* v_a_1332_){
_start:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1333_, 0, v_a_1332_);
lean_inc_ref_n(v___x_1333_, 2);
v___x_1334_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1333_);
lean_ctor_set(v___x_1334_, 1, v___x_1333_);
lean_ctor_set(v___x_1334_, 2, v___x_1333_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object* v_x_1335_, lean_object* v_a_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(v_x_1335_, v_a_1336_);
lean_dec_ref(v_x_1335_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(lean_object* v___y_1338_){
_start:
{
lean_inc(v___y_1338_);
return v___y_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object* v___y_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(v___y_1339_);
lean_dec(v___y_1339_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1357_; lean_object* v___x_1358_; 
v___x_1357_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_));
v___x_1358_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_1357_);
return v___x_1358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2____boxed(lean_object* v_a_1359_){
_start:
{
lean_object* v_res_1360_; 
v_res_1360_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_();
return v_res_1360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0(lean_object* v_e_1361_, lean_object* v_k_1362_, lean_object* v_t_1363_, lean_object* v_hl_1364_){
_start:
{
lean_object* v___x_1365_; 
v___x_1365_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg(v_e_1361_, v_k_1362_, v_t_1363_);
return v___x_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1(lean_object* v_e_1366_, lean_object* v_k_1367_, lean_object* v_t_1368_, lean_object* v_hl_1369_){
_start:
{
lean_object* v___x_1370_; 
v___x_1370_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__1___redArg(v_e_1366_, v_k_1367_, v_t_1368_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(lean_object* v_t_1371_, lean_object* v_k_1372_, lean_object* v_fallback_1373_){
_start:
{
if (lean_obj_tag(v_t_1371_) == 0)
{
lean_object* v_k_1374_; lean_object* v_v_1375_; lean_object* v_l_1376_; lean_object* v_r_1377_; uint8_t v___x_1378_; 
v_k_1374_ = lean_ctor_get(v_t_1371_, 1);
v_v_1375_ = lean_ctor_get(v_t_1371_, 2);
v_l_1376_ = lean_ctor_get(v_t_1371_, 3);
v_r_1377_ = lean_ctor_get(v_t_1371_, 4);
v___x_1378_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_1372_, v_k_1374_);
switch(v___x_1378_)
{
case 0:
{
v_t_1371_ = v_l_1376_;
goto _start;
}
case 1:
{
lean_inc(v_v_1375_);
return v_v_1375_;
}
default: 
{
v_t_1371_ = v_r_1377_;
goto _start;
}
}
}
else
{
lean_inc(v_fallback_1373_);
return v_fallback_1373_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg___boxed(lean_object* v_t_1381_, lean_object* v_k_1382_, lean_object* v_fallback_1383_){
_start:
{
lean_object* v_res_1384_; 
v_res_1384_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(v_t_1381_, v_k_1382_, v_fallback_1383_);
lean_dec(v_fallback_1383_);
lean_dec(v_k_1382_);
lean_dec(v_t_1381_);
return v_res_1384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg(lean_object* v_funName_1385_, lean_object* v_funPropName_1386_, lean_object* v_a_1387_){
_start:
{
lean_object* v___x_1389_; lean_object* v_env_1390_; lean_object* v___x_1391_; lean_object* v_ext_1392_; lean_object* v_toEnvExtension_1393_; lean_object* v_asyncMode_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; 
v___x_1389_ = lean_st_ref_get(v_a_1387_);
v_env_1390_ = lean_ctor_get(v___x_1389_, 0);
lean_inc_ref(v_env_1390_);
lean_dec(v___x_1389_);
v___x_1391_ = lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt;
v_ext_1392_ = lean_ctor_get(v___x_1391_, 1);
v_toEnvExtension_1393_ = lean_ctor_get(v_ext_1392_, 0);
v_asyncMode_1394_ = lean_ctor_get(v_toEnvExtension_1393_, 2);
v___x_1395_ = lean_box(1);
v___x_1396_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1395_, v___x_1391_, v_env_1390_, v_asyncMode_1394_);
v___x_1397_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(v___x_1396_, v_funName_1385_, v___x_1395_);
lean_dec(v___x_1396_);
v___x_1398_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2__spec__0___redArg___lam__0___closed__0));
v___x_1399_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(v___x_1397_, v_funPropName_1386_, v___x_1398_);
lean_dec(v___x_1397_);
v___x_1400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1400_, 0, v___x_1399_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg___boxed(lean_object* v_funName_1401_, lean_object* v_funPropName_1402_, lean_object* v_a_1403_, lean_object* v_a_1404_){
_start:
{
lean_object* v_res_1405_; 
v_res_1405_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg(v_funName_1401_, v_funPropName_1402_, v_a_1403_);
lean_dec(v_a_1403_);
lean_dec(v_funPropName_1402_);
lean_dec(v_funName_1401_);
return v_res_1405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction(lean_object* v_funName_1406_, lean_object* v_funPropName_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_){
_start:
{
lean_object* v___x_1411_; 
v___x_1411_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___redArg(v_funName_1406_, v_funPropName_1407_, v_a_1409_);
return v___x_1411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction___boxed(lean_object* v_funName_1412_, lean_object* v_funPropName_1413_, lean_object* v_a_1414_, lean_object* v_a_1415_, lean_object* v_a_1416_){
_start:
{
lean_object* v_res_1417_; 
v_res_1417_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremsForFunction(v_funName_1412_, v_funPropName_1413_, v_a_1414_, v_a_1415_);
lean_dec(v_a_1415_);
lean_dec_ref(v_a_1414_);
lean_dec(v_funPropName_1413_);
lean_dec(v_funName_1412_);
return v_res_1417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0(lean_object* v_00_u03b4_1418_, lean_object* v_t_1419_, lean_object* v_k_1420_, lean_object* v_fallback_1421_){
_start:
{
lean_object* v___x_1422_; 
v___x_1422_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___redArg(v_t_1419_, v_k_1420_, v_fallback_1421_);
return v___x_1422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0___boxed(lean_object* v_00_u03b4_1423_, lean_object* v_t_1424_, lean_object* v_k_1425_, lean_object* v_fallback_1426_){
_start:
{
lean_object* v_res_1427_; 
v_res_1427_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp_getTheoremsForFunction_spec__0(v_00_u03b4_1423_, v_t_1424_, v_k_1425_, v_fallback_1426_);
lean_dec(v_fallback_1426_);
lean_dec(v_k_1425_);
lean_dec(v_t_1424_);
return v_res_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_GeneralTheorem_getProof(lean_object* v_thm_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_){
_start:
{
lean_object* v_thmName_1434_; lean_object* v___x_1435_; 
v_thmName_1434_ = lean_ctor_get(v_thm_1428_, 1);
lean_inc(v_thmName_1434_);
lean_dec_ref(v_thm_1428_);
v___x_1435_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_thmName_1434_, v_a_1429_, v_a_1430_, v_a_1431_, v_a_1432_);
return v___x_1435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_GeneralTheorem_getProof___boxed(lean_object* v_thm_1436_, lean_object* v_a_1437_, lean_object* v_a_1438_, lean_object* v_a_1439_, lean_object* v_a_1440_, lean_object* v_a_1441_){
_start:
{
lean_object* v_res_1442_; 
v_res_1442_ = lp_mathlib_Mathlib_Meta_FunProp_GeneralTheorem_getProof(v_thm_1436_, v_a_1437_, v_a_1438_, v_a_1439_, v_a_1440_);
lean_dec(v_a_1440_);
lean_dec_ref(v_a_1439_);
lean_dec(v_a_1438_);
lean_dec_ref(v_a_1437_);
return v_res_1442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__spec__0(lean_object* v_e_1443_, lean_object* v_x_1444_, lean_object* v_x_1445_){
_start:
{
if (lean_obj_tag(v_x_1445_) == 0)
{
lean_dec_ref(v_e_1443_);
return v_x_1444_;
}
else
{
lean_object* v_head_1446_; lean_object* v_tail_1447_; lean_object* v_fst_1448_; lean_object* v_snd_1449_; lean_object* v___x_1451_; uint8_t v_isShared_1452_; uint8_t v_isSharedCheck_1458_; 
v_head_1446_ = lean_ctor_get(v_x_1445_, 0);
lean_inc(v_head_1446_);
v_tail_1447_ = lean_ctor_get(v_x_1445_, 1);
lean_inc(v_tail_1447_);
lean_dec_ref_known(v_x_1445_, 2);
v_fst_1448_ = lean_ctor_get(v_head_1446_, 0);
v_snd_1449_ = lean_ctor_get(v_head_1446_, 1);
v_isSharedCheck_1458_ = !lean_is_exclusive(v_head_1446_);
if (v_isSharedCheck_1458_ == 0)
{
v___x_1451_ = v_head_1446_;
v_isShared_1452_ = v_isSharedCheck_1458_;
goto v_resetjp_1450_;
}
else
{
lean_inc(v_snd_1449_);
lean_inc(v_fst_1448_);
lean_dec(v_head_1446_);
v___x_1451_ = lean_box(0);
v_isShared_1452_ = v_isSharedCheck_1458_;
goto v_resetjp_1450_;
}
v_resetjp_1450_:
{
lean_object* v___x_1454_; 
lean_inc_ref(v_e_1443_);
if (v_isShared_1452_ == 0)
{
lean_ctor_set(v___x_1451_, 1, v_e_1443_);
lean_ctor_set(v___x_1451_, 0, v_snd_1449_);
v___x_1454_ = v___x_1451_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1457_; 
v_reuseFailAlloc_1457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1457_, 0, v_snd_1449_);
lean_ctor_set(v_reuseFailAlloc_1457_, 1, v_e_1443_);
v___x_1454_ = v_reuseFailAlloc_1457_;
goto v_reusejp_1453_;
}
v_reusejp_1453_:
{
lean_object* v___x_1455_; 
v___x_1455_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_insert___redArg(v_x_1444_, v_fst_1448_, v___x_1454_);
v_x_1444_ = v___x_1455_;
v_x_1445_ = v_tail_1447_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object* v_d_1459_, lean_object* v_e_1460_){
_start:
{
lean_object* v_keys_1461_; lean_object* v___x_1462_; 
v_keys_1461_ = lean_ctor_get(v_e_1460_, 2);
lean_inc(v_keys_1461_);
v___x_1462_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__spec__0(v_e_1460_, v_d_1459_, v_keys_1461_);
return v___x_1462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object* v_x_1463_, lean_object* v_a_1464_){
_start:
{
lean_object* v___x_1465_; lean_object* v___x_1466_; 
v___x_1465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1465_, 0, v_a_1464_);
lean_inc_ref_n(v___x_1465_, 2);
v___x_1466_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1466_, 0, v___x_1465_);
lean_ctor_set(v___x_1466_, 1, v___x_1465_);
lean_ctor_set(v___x_1466_, 2, v___x_1465_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object* v_x_1467_, lean_object* v_a_1468_){
_start:
{
lean_object* v_res_1469_; 
v_res_1469_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(v_x_1467_, v_a_1468_);
lean_dec_ref(v_x_1467_);
return v_res_1469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(lean_object* v___y_1470_){
_start:
{
lean_inc_ref(v___y_1470_);
return v___y_1470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object* v___y_1471_){
_start:
{
lean_object* v_res_1472_; 
v_res_1472_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___lam__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(v___y_1471_);
lean_dec_ref(v___y_1471_);
return v_res_1472_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; 
v___x_1482_ = lean_box(0);
v___x_1483_ = lean_unsigned_to_nat(16u);
v___x_1484_ = lean_mk_array(v___x_1483_, v___x_1482_);
return v___x_1484_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; 
v___x_1485_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_);
v___x_1486_ = lean_unsigned_to_nat(0u);
v___x_1487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1486_);
lean_ctor_set(v___x_1487_, 1, v___x_1485_);
return v___x_1487_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; 
v___x_1490_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1491_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_);
v___x_1492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1492_, 0, v___x_1491_);
lean_ctor_set(v___x_1492_, 1, v___x_1490_);
return v___x_1492_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1493_; lean_object* v___f_1494_; lean_object* v___x_1495_; lean_object* v___f_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___f_1493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___f_1494_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1495_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_);
v___f_1496_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1497_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1498_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1497_);
lean_ctor_set(v___x_1498_, 1, v___f_1496_);
lean_ctor_set(v___x_1498_, 2, v___x_1495_);
lean_ctor_set(v___x_1498_, 3, v___f_1494_);
lean_ctor_set(v___x_1498_, 4, v___f_1493_);
return v___x_1498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1500_; lean_object* v___x_1501_; 
v___x_1500_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_);
v___x_1501_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_1500_);
return v___x_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2____boxed(lean_object* v_a_1502_){
_start:
{
lean_object* v_res_1503_; 
v_res_1503_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_();
return v_res_1503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg(lean_object* v_e_1504_, lean_object* v_a_1505_, lean_object* v_a_1506_, lean_object* v_a_1507_, lean_object* v_a_1508_, lean_object* v_a_1509_){
_start:
{
lean_object* v_cache_1511_; lean_object* v_failureCache_1512_; lean_object* v_numSteps_1513_; lean_object* v_msgLog_1514_; lean_object* v_morTheorems_1515_; lean_object* v_transitionTheorems_1516_; lean_object* v___x_1518_; uint8_t v_isShared_1519_; uint8_t v_isSharedCheck_1591_; 
v_cache_1511_ = lean_ctor_get(v_a_1505_, 0);
v_failureCache_1512_ = lean_ctor_get(v_a_1505_, 1);
v_numSteps_1513_ = lean_ctor_get(v_a_1505_, 2);
v_msgLog_1514_ = lean_ctor_get(v_a_1505_, 3);
v_morTheorems_1515_ = lean_ctor_get(v_a_1505_, 4);
v_transitionTheorems_1516_ = lean_ctor_get(v_a_1505_, 5);
v_isSharedCheck_1591_ = !lean_is_exclusive(v_a_1505_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1518_ = v_a_1505_;
v_isShared_1519_ = v_isSharedCheck_1591_;
goto v_resetjp_1517_;
}
else
{
lean_inc(v_transitionTheorems_1516_);
lean_inc(v_morTheorems_1515_);
lean_inc(v_msgLog_1514_);
lean_inc(v_numSteps_1513_);
lean_inc(v_failureCache_1512_);
lean_inc(v_cache_1511_);
lean_dec(v_a_1505_);
v___x_1518_ = lean_box(0);
v_isShared_1519_ = v_isSharedCheck_1591_;
goto v_resetjp_1517_;
}
v_resetjp_1517_:
{
lean_object* v___x_1520_; uint8_t v_foApprox_1521_; uint8_t v_ctxApprox_1522_; uint8_t v_quasiPatternApprox_1523_; uint8_t v_constApprox_1524_; uint8_t v_isDefEqStuckEx_1525_; uint8_t v_unificationHints_1526_; uint8_t v_proofIrrelevance_1527_; uint8_t v_assignSyntheticOpaque_1528_; uint8_t v_offsetCnstrs_1529_; uint8_t v_transparency_1530_; uint8_t v_etaStruct_1531_; uint8_t v_univApprox_1532_; uint8_t v_beta_1533_; uint8_t v_proj_1534_; uint8_t v_zetaDelta_1535_; uint8_t v_zetaUnused_1536_; uint8_t v_zetaHave_1537_; uint8_t v_canUnfoldPredicateConfig_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1590_; 
v___x_1520_ = l_Lean_Meta_Context_config(v_a_1506_);
v_foApprox_1521_ = lean_ctor_get_uint8(v___x_1520_, 0);
v_ctxApprox_1522_ = lean_ctor_get_uint8(v___x_1520_, 1);
v_quasiPatternApprox_1523_ = lean_ctor_get_uint8(v___x_1520_, 2);
v_constApprox_1524_ = lean_ctor_get_uint8(v___x_1520_, 3);
v_isDefEqStuckEx_1525_ = lean_ctor_get_uint8(v___x_1520_, 4);
v_unificationHints_1526_ = lean_ctor_get_uint8(v___x_1520_, 5);
v_proofIrrelevance_1527_ = lean_ctor_get_uint8(v___x_1520_, 6);
v_assignSyntheticOpaque_1528_ = lean_ctor_get_uint8(v___x_1520_, 7);
v_offsetCnstrs_1529_ = lean_ctor_get_uint8(v___x_1520_, 8);
v_transparency_1530_ = lean_ctor_get_uint8(v___x_1520_, 9);
v_etaStruct_1531_ = lean_ctor_get_uint8(v___x_1520_, 10);
v_univApprox_1532_ = lean_ctor_get_uint8(v___x_1520_, 11);
v_beta_1533_ = lean_ctor_get_uint8(v___x_1520_, 13);
v_proj_1534_ = lean_ctor_get_uint8(v___x_1520_, 14);
v_zetaDelta_1535_ = lean_ctor_get_uint8(v___x_1520_, 16);
v_zetaUnused_1536_ = lean_ctor_get_uint8(v___x_1520_, 17);
v_zetaHave_1537_ = lean_ctor_get_uint8(v___x_1520_, 18);
v_canUnfoldPredicateConfig_1538_ = lean_ctor_get_uint8(v___x_1520_, 19);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1540_ = v___x_1520_;
v_isShared_1541_ = v_isSharedCheck_1590_;
goto v_resetjp_1539_;
}
else
{
lean_dec(v___x_1520_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1590_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
uint8_t v___x_1542_; lean_object* v___x_1544_; 
v___x_1542_ = 0;
if (v_isShared_1541_ == 0)
{
v___x_1544_ = v___x_1540_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 0, v_foApprox_1521_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 1, v_ctxApprox_1522_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 2, v_quasiPatternApprox_1523_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 3, v_constApprox_1524_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 4, v_isDefEqStuckEx_1525_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 5, v_unificationHints_1526_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 6, v_proofIrrelevance_1527_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 7, v_assignSyntheticOpaque_1528_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 8, v_offsetCnstrs_1529_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 9, v_transparency_1530_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 10, v_etaStruct_1531_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 11, v_univApprox_1532_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 13, v_beta_1533_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 14, v_proj_1534_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 16, v_zetaDelta_1535_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 17, v_zetaUnused_1536_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 18, v_zetaHave_1537_);
lean_ctor_set_uint8(v_reuseFailAlloc_1589_, 19, v_canUnfoldPredicateConfig_1538_);
v___x_1544_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
uint8_t v_trackZetaDelta_1545_; lean_object* v_zetaDeltaSet_1546_; lean_object* v_lctx_1547_; lean_object* v_localInstances_1548_; lean_object* v_defEqCtx_x3f_1549_; lean_object* v_synthPendingDepth_1550_; lean_object* v_customCanUnfoldPredicate_x3f_1551_; uint8_t v_univApprox_1552_; uint8_t v_inTypeClassResolution_1553_; uint8_t v_cacheInferType_1554_; uint64_t v___x_1555_; uint8_t v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; 
lean_ctor_set_uint8(v___x_1544_, 12, v___x_1542_);
lean_ctor_set_uint8(v___x_1544_, 15, v___x_1542_);
v_trackZetaDelta_1545_ = lean_ctor_get_uint8(v_a_1506_, sizeof(void*)*7);
v_zetaDeltaSet_1546_ = lean_ctor_get(v_a_1506_, 1);
v_lctx_1547_ = lean_ctor_get(v_a_1506_, 2);
v_localInstances_1548_ = lean_ctor_get(v_a_1506_, 3);
v_defEqCtx_x3f_1549_ = lean_ctor_get(v_a_1506_, 4);
v_synthPendingDepth_1550_ = lean_ctor_get(v_a_1506_, 5);
v_customCanUnfoldPredicate_x3f_1551_ = lean_ctor_get(v_a_1506_, 6);
v_univApprox_1552_ = lean_ctor_get_uint8(v_a_1506_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1553_ = lean_ctor_get_uint8(v_a_1506_, sizeof(void*)*7 + 2);
v_cacheInferType_1554_ = lean_ctor_get_uint8(v_a_1506_, sizeof(void*)*7 + 3);
v___x_1555_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1544_);
v___x_1556_ = 1;
v___x_1557_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1557_, 0, v___x_1544_);
lean_ctor_set_uint64(v___x_1557_, sizeof(void*)*1, v___x_1555_);
lean_inc(v_customCanUnfoldPredicate_x3f_1551_);
lean_inc(v_synthPendingDepth_1550_);
lean_inc(v_defEqCtx_x3f_1549_);
lean_inc_ref(v_localInstances_1548_);
lean_inc_ref(v_lctx_1547_);
lean_inc(v_zetaDeltaSet_1546_);
v___x_1558_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1558_, 0, v___x_1557_);
lean_ctor_set(v___x_1558_, 1, v_zetaDeltaSet_1546_);
lean_ctor_set(v___x_1558_, 2, v_lctx_1547_);
lean_ctor_set(v___x_1558_, 3, v_localInstances_1548_);
lean_ctor_set(v___x_1558_, 4, v_defEqCtx_x3f_1549_);
lean_ctor_set(v___x_1558_, 5, v_synthPendingDepth_1550_);
lean_ctor_set(v___x_1558_, 6, v_customCanUnfoldPredicate_x3f_1551_);
lean_ctor_set_uint8(v___x_1558_, sizeof(void*)*7, v_trackZetaDelta_1545_);
lean_ctor_set_uint8(v___x_1558_, sizeof(void*)*7 + 1, v_univApprox_1552_);
lean_ctor_set_uint8(v___x_1558_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1553_);
lean_ctor_set_uint8(v___x_1558_, sizeof(void*)*7 + 3, v_cacheInferType_1554_);
v___x_1559_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(v_transitionTheorems_1516_, v_e_1504_, v___x_1542_, v___x_1556_, v___x_1558_, v_a_1507_, v_a_1508_, v_a_1509_);
lean_dec_ref_known(v___x_1558_, 7);
if (lean_obj_tag(v___x_1559_) == 0)
{
lean_object* v_a_1560_; lean_object* v___x_1562_; uint8_t v_isShared_1563_; uint8_t v_isSharedCheck_1580_; 
v_a_1560_ = lean_ctor_get(v___x_1559_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1559_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1562_ = v___x_1559_;
v_isShared_1563_ = v_isSharedCheck_1580_;
goto v_resetjp_1561_;
}
else
{
lean_inc(v_a_1560_);
lean_dec(v___x_1559_);
v___x_1562_ = lean_box(0);
v_isShared_1563_ = v_isSharedCheck_1580_;
goto v_resetjp_1561_;
}
v_resetjp_1561_:
{
lean_object* v_fst_1564_; lean_object* v_snd_1565_; lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1579_; 
v_fst_1564_ = lean_ctor_get(v_a_1560_, 0);
v_snd_1565_ = lean_ctor_get(v_a_1560_, 1);
v_isSharedCheck_1579_ = !lean_is_exclusive(v_a_1560_);
if (v_isSharedCheck_1579_ == 0)
{
v___x_1567_ = v_a_1560_;
v_isShared_1568_ = v_isSharedCheck_1579_;
goto v_resetjp_1566_;
}
else
{
lean_inc(v_snd_1565_);
lean_inc(v_fst_1564_);
lean_dec(v_a_1560_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1579_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
lean_object* v___x_1570_; 
if (v_isShared_1519_ == 0)
{
lean_ctor_set(v___x_1518_, 5, v_snd_1565_);
v___x_1570_ = v___x_1518_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1578_; 
v_reuseFailAlloc_1578_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1578_, 0, v_cache_1511_);
lean_ctor_set(v_reuseFailAlloc_1578_, 1, v_failureCache_1512_);
lean_ctor_set(v_reuseFailAlloc_1578_, 2, v_numSteps_1513_);
lean_ctor_set(v_reuseFailAlloc_1578_, 3, v_msgLog_1514_);
lean_ctor_set(v_reuseFailAlloc_1578_, 4, v_morTheorems_1515_);
lean_ctor_set(v_reuseFailAlloc_1578_, 5, v_snd_1565_);
v___x_1570_ = v_reuseFailAlloc_1578_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
lean_object* v___x_1571_; lean_object* v___x_1573_; 
v___x_1571_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_MatchResult_toArray___redArg(v_fst_1564_);
lean_dec(v_fst_1564_);
if (v_isShared_1568_ == 0)
{
lean_ctor_set(v___x_1567_, 1, v___x_1570_);
lean_ctor_set(v___x_1567_, 0, v___x_1571_);
v___x_1573_ = v___x_1567_;
goto v_reusejp_1572_;
}
else
{
lean_object* v_reuseFailAlloc_1577_; 
v_reuseFailAlloc_1577_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1577_, 0, v___x_1571_);
lean_ctor_set(v_reuseFailAlloc_1577_, 1, v___x_1570_);
v___x_1573_ = v_reuseFailAlloc_1577_;
goto v_reusejp_1572_;
}
v_reusejp_1572_:
{
lean_object* v___x_1575_; 
if (v_isShared_1563_ == 0)
{
lean_ctor_set(v___x_1562_, 0, v___x_1573_);
v___x_1575_ = v___x_1562_;
goto v_reusejp_1574_;
}
else
{
lean_object* v_reuseFailAlloc_1576_; 
v_reuseFailAlloc_1576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1576_, 0, v___x_1573_);
v___x_1575_ = v_reuseFailAlloc_1576_;
goto v_reusejp_1574_;
}
v_reusejp_1574_:
{
return v___x_1575_;
}
}
}
}
}
}
else
{
lean_object* v_a_1581_; lean_object* v___x_1583_; uint8_t v_isShared_1584_; uint8_t v_isSharedCheck_1588_; 
lean_del_object(v___x_1518_);
lean_dec_ref(v_morTheorems_1515_);
lean_dec(v_msgLog_1514_);
lean_dec(v_numSteps_1513_);
lean_dec_ref(v_failureCache_1512_);
lean_dec_ref(v_cache_1511_);
v_a_1581_ = lean_ctor_get(v___x_1559_, 0);
v_isSharedCheck_1588_ = !lean_is_exclusive(v___x_1559_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1583_ = v___x_1559_;
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
else
{
lean_inc(v_a_1581_);
lean_dec(v___x_1559_);
v___x_1583_ = lean_box(0);
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
v_resetjp_1582_:
{
lean_object* v___x_1586_; 
if (v_isShared_1584_ == 0)
{
v___x_1586_ = v___x_1583_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_a_1581_);
v___x_1586_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
return v___x_1586_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg___boxed(lean_object* v_e_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_, lean_object* v_a_1595_, lean_object* v_a_1596_, lean_object* v_a_1597_, lean_object* v_a_1598_){
_start:
{
lean_object* v_res_1599_; 
v_res_1599_ = lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg(v_e_1592_, v_a_1593_, v_a_1594_, v_a_1595_, v_a_1596_, v_a_1597_);
lean_dec(v_a_1597_);
lean_dec_ref(v_a_1596_);
lean_dec(v_a_1595_);
lean_dec_ref(v_a_1594_);
return v_res_1599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems(lean_object* v_e_1600_, lean_object* v_a_1601_, lean_object* v_a_1602_, lean_object* v_a_1603_, lean_object* v_a_1604_, lean_object* v_a_1605_, lean_object* v_a_1606_){
_start:
{
lean_object* v___x_1608_; 
v___x_1608_ = lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___redArg(v_e_1600_, v_a_1602_, v_a_1603_, v_a_1604_, v_a_1605_, v_a_1606_);
return v___x_1608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems___boxed(lean_object* v_e_1609_, lean_object* v_a_1610_, lean_object* v_a_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_, lean_object* v_a_1615_, lean_object* v_a_1616_){
_start:
{
lean_object* v_res_1617_; 
v_res_1617_ = lp_mathlib_Mathlib_Meta_FunProp_getTransitionTheorems(v_e_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, v_a_1615_);
lean_dec(v_a_1615_);
lean_dec_ref(v_a_1614_);
lean_dec(v_a_1613_);
lean_dec_ref(v_a_1612_);
lean_dec_ref(v_a_1610_);
return v_res_1617_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1624_; lean_object* v___f_1625_; lean_object* v___x_1626_; lean_object* v___f_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; 
v___f_1624_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___f_1625_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1626_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_);
v___f_1627_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_));
v___x_1628_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_));
v___x_1629_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1629_, 0, v___x_1628_);
lean_ctor_set(v___x_1629_, 1, v___f_1627_);
lean_ctor_set(v___x_1629_, 2, v___x_1626_);
lean_ctor_set(v___x_1629_, 3, v___f_1625_);
lean_ctor_set(v___x_1629_, 4, v___f_1624_);
return v___x_1629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1631_; lean_object* v___x_1632_; 
v___x_1631_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_);
v___x_1632_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_1631_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2____boxed(lean_object* v_a_1633_){
_start:
{
lean_object* v_res_1634_; 
v_res_1634_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_();
return v_res_1634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg(lean_object* v_e_1635_, lean_object* v_a_1636_, lean_object* v_a_1637_, lean_object* v_a_1638_, lean_object* v_a_1639_, lean_object* v_a_1640_){
_start:
{
lean_object* v_cache_1642_; lean_object* v_failureCache_1643_; lean_object* v_numSteps_1644_; lean_object* v_msgLog_1645_; lean_object* v_morTheorems_1646_; lean_object* v_transitionTheorems_1647_; lean_object* v___x_1649_; uint8_t v_isShared_1650_; uint8_t v_isSharedCheck_1722_; 
v_cache_1642_ = lean_ctor_get(v_a_1636_, 0);
v_failureCache_1643_ = lean_ctor_get(v_a_1636_, 1);
v_numSteps_1644_ = lean_ctor_get(v_a_1636_, 2);
v_msgLog_1645_ = lean_ctor_get(v_a_1636_, 3);
v_morTheorems_1646_ = lean_ctor_get(v_a_1636_, 4);
v_transitionTheorems_1647_ = lean_ctor_get(v_a_1636_, 5);
v_isSharedCheck_1722_ = !lean_is_exclusive(v_a_1636_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1649_ = v_a_1636_;
v_isShared_1650_ = v_isSharedCheck_1722_;
goto v_resetjp_1648_;
}
else
{
lean_inc(v_transitionTheorems_1647_);
lean_inc(v_morTheorems_1646_);
lean_inc(v_msgLog_1645_);
lean_inc(v_numSteps_1644_);
lean_inc(v_failureCache_1643_);
lean_inc(v_cache_1642_);
lean_dec(v_a_1636_);
v___x_1649_ = lean_box(0);
v_isShared_1650_ = v_isSharedCheck_1722_;
goto v_resetjp_1648_;
}
v_resetjp_1648_:
{
lean_object* v___x_1651_; uint8_t v_foApprox_1652_; uint8_t v_ctxApprox_1653_; uint8_t v_quasiPatternApprox_1654_; uint8_t v_constApprox_1655_; uint8_t v_isDefEqStuckEx_1656_; uint8_t v_unificationHints_1657_; uint8_t v_proofIrrelevance_1658_; uint8_t v_assignSyntheticOpaque_1659_; uint8_t v_offsetCnstrs_1660_; uint8_t v_transparency_1661_; uint8_t v_etaStruct_1662_; uint8_t v_univApprox_1663_; uint8_t v_beta_1664_; uint8_t v_proj_1665_; uint8_t v_zetaDelta_1666_; uint8_t v_zetaUnused_1667_; uint8_t v_zetaHave_1668_; uint8_t v_canUnfoldPredicateConfig_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1721_; 
v___x_1651_ = l_Lean_Meta_Context_config(v_a_1637_);
v_foApprox_1652_ = lean_ctor_get_uint8(v___x_1651_, 0);
v_ctxApprox_1653_ = lean_ctor_get_uint8(v___x_1651_, 1);
v_quasiPatternApprox_1654_ = lean_ctor_get_uint8(v___x_1651_, 2);
v_constApprox_1655_ = lean_ctor_get_uint8(v___x_1651_, 3);
v_isDefEqStuckEx_1656_ = lean_ctor_get_uint8(v___x_1651_, 4);
v_unificationHints_1657_ = lean_ctor_get_uint8(v___x_1651_, 5);
v_proofIrrelevance_1658_ = lean_ctor_get_uint8(v___x_1651_, 6);
v_assignSyntheticOpaque_1659_ = lean_ctor_get_uint8(v___x_1651_, 7);
v_offsetCnstrs_1660_ = lean_ctor_get_uint8(v___x_1651_, 8);
v_transparency_1661_ = lean_ctor_get_uint8(v___x_1651_, 9);
v_etaStruct_1662_ = lean_ctor_get_uint8(v___x_1651_, 10);
v_univApprox_1663_ = lean_ctor_get_uint8(v___x_1651_, 11);
v_beta_1664_ = lean_ctor_get_uint8(v___x_1651_, 13);
v_proj_1665_ = lean_ctor_get_uint8(v___x_1651_, 14);
v_zetaDelta_1666_ = lean_ctor_get_uint8(v___x_1651_, 16);
v_zetaUnused_1667_ = lean_ctor_get_uint8(v___x_1651_, 17);
v_zetaHave_1668_ = lean_ctor_get_uint8(v___x_1651_, 18);
v_canUnfoldPredicateConfig_1669_ = lean_ctor_get_uint8(v___x_1651_, 19);
v_isSharedCheck_1721_ = !lean_is_exclusive(v___x_1651_);
if (v_isSharedCheck_1721_ == 0)
{
v___x_1671_ = v___x_1651_;
v_isShared_1672_ = v_isSharedCheck_1721_;
goto v_resetjp_1670_;
}
else
{
lean_dec(v___x_1651_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1721_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
uint8_t v___x_1673_; lean_object* v___x_1675_; 
v___x_1673_ = 0;
if (v_isShared_1672_ == 0)
{
v___x_1675_ = v___x_1671_;
goto v_reusejp_1674_;
}
else
{
lean_object* v_reuseFailAlloc_1720_; 
v_reuseFailAlloc_1720_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 0, v_foApprox_1652_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 1, v_ctxApprox_1653_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 2, v_quasiPatternApprox_1654_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 3, v_constApprox_1655_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 4, v_isDefEqStuckEx_1656_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 5, v_unificationHints_1657_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 6, v_proofIrrelevance_1658_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 7, v_assignSyntheticOpaque_1659_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 8, v_offsetCnstrs_1660_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 9, v_transparency_1661_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 10, v_etaStruct_1662_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 11, v_univApprox_1663_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 13, v_beta_1664_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 14, v_proj_1665_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 16, v_zetaDelta_1666_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 17, v_zetaUnused_1667_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 18, v_zetaHave_1668_);
lean_ctor_set_uint8(v_reuseFailAlloc_1720_, 19, v_canUnfoldPredicateConfig_1669_);
v___x_1675_ = v_reuseFailAlloc_1720_;
goto v_reusejp_1674_;
}
v_reusejp_1674_:
{
uint8_t v_trackZetaDelta_1676_; lean_object* v_zetaDeltaSet_1677_; lean_object* v_lctx_1678_; lean_object* v_localInstances_1679_; lean_object* v_defEqCtx_x3f_1680_; lean_object* v_synthPendingDepth_1681_; lean_object* v_customCanUnfoldPredicate_x3f_1682_; uint8_t v_univApprox_1683_; uint8_t v_inTypeClassResolution_1684_; uint8_t v_cacheInferType_1685_; uint64_t v___x_1686_; uint8_t v___x_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___x_1690_; 
lean_ctor_set_uint8(v___x_1675_, 12, v___x_1673_);
lean_ctor_set_uint8(v___x_1675_, 15, v___x_1673_);
v_trackZetaDelta_1676_ = lean_ctor_get_uint8(v_a_1637_, sizeof(void*)*7);
v_zetaDeltaSet_1677_ = lean_ctor_get(v_a_1637_, 1);
v_lctx_1678_ = lean_ctor_get(v_a_1637_, 2);
v_localInstances_1679_ = lean_ctor_get(v_a_1637_, 3);
v_defEqCtx_x3f_1680_ = lean_ctor_get(v_a_1637_, 4);
v_synthPendingDepth_1681_ = lean_ctor_get(v_a_1637_, 5);
v_customCanUnfoldPredicate_x3f_1682_ = lean_ctor_get(v_a_1637_, 6);
v_univApprox_1683_ = lean_ctor_get_uint8(v_a_1637_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1684_ = lean_ctor_get_uint8(v_a_1637_, sizeof(void*)*7 + 2);
v_cacheInferType_1685_ = lean_ctor_get_uint8(v_a_1637_, sizeof(void*)*7 + 3);
v___x_1686_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1675_);
v___x_1687_ = 1;
v___x_1688_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1688_, 0, v___x_1675_);
lean_ctor_set_uint64(v___x_1688_, sizeof(void*)*1, v___x_1686_);
lean_inc(v_customCanUnfoldPredicate_x3f_1682_);
lean_inc(v_synthPendingDepth_1681_);
lean_inc(v_defEqCtx_x3f_1680_);
lean_inc_ref(v_localInstances_1679_);
lean_inc_ref(v_lctx_1678_);
lean_inc(v_zetaDeltaSet_1677_);
v___x_1689_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1689_, 0, v___x_1688_);
lean_ctor_set(v___x_1689_, 1, v_zetaDeltaSet_1677_);
lean_ctor_set(v___x_1689_, 2, v_lctx_1678_);
lean_ctor_set(v___x_1689_, 3, v_localInstances_1679_);
lean_ctor_set(v___x_1689_, 4, v_defEqCtx_x3f_1680_);
lean_ctor_set(v___x_1689_, 5, v_synthPendingDepth_1681_);
lean_ctor_set(v___x_1689_, 6, v_customCanUnfoldPredicate_x3f_1682_);
lean_ctor_set_uint8(v___x_1689_, sizeof(void*)*7, v_trackZetaDelta_1676_);
lean_ctor_set_uint8(v___x_1689_, sizeof(void*)*7 + 1, v_univApprox_1683_);
lean_ctor_set_uint8(v___x_1689_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1684_);
lean_ctor_set_uint8(v___x_1689_, sizeof(void*)*7 + 3, v_cacheInferType_1685_);
v___x_1690_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(v_morTheorems_1646_, v_e_1635_, v___x_1673_, v___x_1687_, v___x_1689_, v_a_1638_, v_a_1639_, v_a_1640_);
lean_dec_ref_known(v___x_1689_, 7);
if (lean_obj_tag(v___x_1690_) == 0)
{
lean_object* v_a_1691_; lean_object* v___x_1693_; uint8_t v_isShared_1694_; uint8_t v_isSharedCheck_1711_; 
v_a_1691_ = lean_ctor_get(v___x_1690_, 0);
v_isSharedCheck_1711_ = !lean_is_exclusive(v___x_1690_);
if (v_isSharedCheck_1711_ == 0)
{
v___x_1693_ = v___x_1690_;
v_isShared_1694_ = v_isSharedCheck_1711_;
goto v_resetjp_1692_;
}
else
{
lean_inc(v_a_1691_);
lean_dec(v___x_1690_);
v___x_1693_ = lean_box(0);
v_isShared_1694_ = v_isSharedCheck_1711_;
goto v_resetjp_1692_;
}
v_resetjp_1692_:
{
lean_object* v_fst_1695_; lean_object* v_snd_1696_; lean_object* v___x_1698_; uint8_t v_isShared_1699_; uint8_t v_isSharedCheck_1710_; 
v_fst_1695_ = lean_ctor_get(v_a_1691_, 0);
v_snd_1696_ = lean_ctor_get(v_a_1691_, 1);
v_isSharedCheck_1710_ = !lean_is_exclusive(v_a_1691_);
if (v_isSharedCheck_1710_ == 0)
{
v___x_1698_ = v_a_1691_;
v_isShared_1699_ = v_isSharedCheck_1710_;
goto v_resetjp_1697_;
}
else
{
lean_inc(v_snd_1696_);
lean_inc(v_fst_1695_);
lean_dec(v_a_1691_);
v___x_1698_ = lean_box(0);
v_isShared_1699_ = v_isSharedCheck_1710_;
goto v_resetjp_1697_;
}
v_resetjp_1697_:
{
lean_object* v___x_1701_; 
if (v_isShared_1650_ == 0)
{
lean_ctor_set(v___x_1649_, 4, v_snd_1696_);
v___x_1701_ = v___x_1649_;
goto v_reusejp_1700_;
}
else
{
lean_object* v_reuseFailAlloc_1709_; 
v_reuseFailAlloc_1709_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1709_, 0, v_cache_1642_);
lean_ctor_set(v_reuseFailAlloc_1709_, 1, v_failureCache_1643_);
lean_ctor_set(v_reuseFailAlloc_1709_, 2, v_numSteps_1644_);
lean_ctor_set(v_reuseFailAlloc_1709_, 3, v_msgLog_1645_);
lean_ctor_set(v_reuseFailAlloc_1709_, 4, v_snd_1696_);
lean_ctor_set(v_reuseFailAlloc_1709_, 5, v_transitionTheorems_1647_);
v___x_1701_ = v_reuseFailAlloc_1709_;
goto v_reusejp_1700_;
}
v_reusejp_1700_:
{
lean_object* v___x_1702_; lean_object* v___x_1704_; 
v___x_1702_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_MatchResult_toArray___redArg(v_fst_1695_);
lean_dec(v_fst_1695_);
if (v_isShared_1699_ == 0)
{
lean_ctor_set(v___x_1698_, 1, v___x_1701_);
lean_ctor_set(v___x_1698_, 0, v___x_1702_);
v___x_1704_ = v___x_1698_;
goto v_reusejp_1703_;
}
else
{
lean_object* v_reuseFailAlloc_1708_; 
v_reuseFailAlloc_1708_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1708_, 0, v___x_1702_);
lean_ctor_set(v_reuseFailAlloc_1708_, 1, v___x_1701_);
v___x_1704_ = v_reuseFailAlloc_1708_;
goto v_reusejp_1703_;
}
v_reusejp_1703_:
{
lean_object* v___x_1706_; 
if (v_isShared_1694_ == 0)
{
lean_ctor_set(v___x_1693_, 0, v___x_1704_);
v___x_1706_ = v___x_1693_;
goto v_reusejp_1705_;
}
else
{
lean_object* v_reuseFailAlloc_1707_; 
v_reuseFailAlloc_1707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1707_, 0, v___x_1704_);
v___x_1706_ = v_reuseFailAlloc_1707_;
goto v_reusejp_1705_;
}
v_reusejp_1705_:
{
return v___x_1706_;
}
}
}
}
}
}
else
{
lean_object* v_a_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1719_; 
lean_del_object(v___x_1649_);
lean_dec_ref(v_transitionTheorems_1647_);
lean_dec(v_msgLog_1645_);
lean_dec(v_numSteps_1644_);
lean_dec_ref(v_failureCache_1643_);
lean_dec_ref(v_cache_1642_);
v_a_1712_ = lean_ctor_get(v___x_1690_, 0);
v_isSharedCheck_1719_ = !lean_is_exclusive(v___x_1690_);
if (v_isSharedCheck_1719_ == 0)
{
v___x_1714_ = v___x_1690_;
v_isShared_1715_ = v_isSharedCheck_1719_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_a_1712_);
lean_dec(v___x_1690_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1719_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v___x_1717_; 
if (v_isShared_1715_ == 0)
{
v___x_1717_ = v___x_1714_;
goto v_reusejp_1716_;
}
else
{
lean_object* v_reuseFailAlloc_1718_; 
v_reuseFailAlloc_1718_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1718_, 0, v_a_1712_);
v___x_1717_ = v_reuseFailAlloc_1718_;
goto v_reusejp_1716_;
}
v_reusejp_1716_:
{
return v___x_1717_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg___boxed(lean_object* v_e_1723_, lean_object* v_a_1724_, lean_object* v_a_1725_, lean_object* v_a_1726_, lean_object* v_a_1727_, lean_object* v_a_1728_, lean_object* v_a_1729_){
_start:
{
lean_object* v_res_1730_; 
v_res_1730_ = lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg(v_e_1723_, v_a_1724_, v_a_1725_, v_a_1726_, v_a_1727_, v_a_1728_);
lean_dec(v_a_1728_);
lean_dec_ref(v_a_1727_);
lean_dec(v_a_1726_);
lean_dec_ref(v_a_1725_);
return v_res_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems(lean_object* v_e_1731_, lean_object* v_a_1732_, lean_object* v_a_1733_, lean_object* v_a_1734_, lean_object* v_a_1735_, lean_object* v_a_1736_, lean_object* v_a_1737_){
_start:
{
lean_object* v___x_1739_; 
v___x_1739_ = lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___redArg(v_e_1731_, v_a_1733_, v_a_1734_, v_a_1735_, v_a_1736_, v_a_1737_);
return v___x_1739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems___boxed(lean_object* v_e_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_, lean_object* v_a_1747_){
_start:
{
lean_object* v_res_1748_; 
v_res_1748_ = lp_mathlib_Mathlib_Meta_FunProp_getMorphismTheorems(v_e_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
lean_dec(v_a_1746_);
lean_dec_ref(v_a_1745_);
lean_dec(v_a_1744_);
lean_dec_ref(v_a_1743_);
lean_dec_ref(v_a_1741_);
return v_res_1748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorIdx(lean_object* v_x_1749_){
_start:
{
switch(lean_obj_tag(v_x_1749_))
{
case 0:
{
lean_object* v___x_1750_; 
v___x_1750_ = lean_unsigned_to_nat(0u);
return v___x_1750_;
}
case 1:
{
lean_object* v___x_1751_; 
v___x_1751_ = lean_unsigned_to_nat(1u);
return v___x_1751_;
}
case 2:
{
lean_object* v___x_1752_; 
v___x_1752_ = lean_unsigned_to_nat(2u);
return v___x_1752_;
}
default: 
{
lean_object* v___x_1753_; 
v___x_1753_ = lean_unsigned_to_nat(3u);
return v___x_1753_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorIdx___boxed(lean_object* v_x_1754_){
_start:
{
lean_object* v_res_1755_; 
v_res_1755_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorIdx(v_x_1754_);
lean_dec_ref(v_x_1754_);
return v_res_1755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(lean_object* v_t_1756_, lean_object* v_k_1757_){
_start:
{
lean_object* v_thm_1758_; lean_object* v___x_1759_; 
v_thm_1758_ = lean_ctor_get(v_t_1756_, 0);
lean_inc_ref(v_thm_1758_);
lean_dec_ref(v_t_1756_);
v___x_1759_ = lean_apply_1(v_k_1757_, v_thm_1758_);
return v___x_1759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim(lean_object* v_motive_1760_, lean_object* v_ctorIdx_1761_, lean_object* v_t_1762_, lean_object* v_h_1763_, lean_object* v_k_1764_){
_start:
{
lean_object* v___x_1765_; 
v___x_1765_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1762_, v_k_1764_);
return v___x_1765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___boxed(lean_object* v_motive_1766_, lean_object* v_ctorIdx_1767_, lean_object* v_t_1768_, lean_object* v_h_1769_, lean_object* v_k_1770_){
_start:
{
lean_object* v_res_1771_; 
v_res_1771_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim(v_motive_1766_, v_ctorIdx_1767_, v_t_1768_, v_h_1769_, v_k_1770_);
lean_dec(v_ctorIdx_1767_);
return v_res_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_lam_elim___redArg(lean_object* v_t_1772_, lean_object* v_lam_1773_){
_start:
{
lean_object* v___x_1774_; 
v___x_1774_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1772_, v_lam_1773_);
return v___x_1774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_lam_elim(lean_object* v_motive_1775_, lean_object* v_t_1776_, lean_object* v_h_1777_, lean_object* v_lam_1778_){
_start:
{
lean_object* v___x_1779_; 
v___x_1779_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1776_, v_lam_1778_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_function_elim___redArg(lean_object* v_t_1780_, lean_object* v_function_1781_){
_start:
{
lean_object* v___x_1782_; 
v___x_1782_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1780_, v_function_1781_);
return v___x_1782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_function_elim(lean_object* v_motive_1783_, lean_object* v_t_1784_, lean_object* v_h_1785_, lean_object* v_function_1786_){
_start:
{
lean_object* v___x_1787_; 
v___x_1787_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1784_, v_function_1786_);
return v___x_1787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_mor_elim___redArg(lean_object* v_t_1788_, lean_object* v_mor_1789_){
_start:
{
lean_object* v___x_1790_; 
v___x_1790_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1788_, v_mor_1789_);
return v___x_1790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_mor_elim(lean_object* v_motive_1791_, lean_object* v_t_1792_, lean_object* v_h_1793_, lean_object* v_mor_1794_){
_start:
{
lean_object* v___x_1795_; 
v___x_1795_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1792_, v_mor_1794_);
return v___x_1795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_transition_elim___redArg(lean_object* v_t_1796_, lean_object* v_transition_1797_){
_start:
{
lean_object* v___x_1798_; 
v___x_1798_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1796_, v_transition_1797_);
return v___x_1798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Theorem_transition_elim(lean_object* v_motive_1799_, lean_object* v_t_1800_, lean_object* v_h_1801_, lean_object* v_transition_1802_){
_start:
{
lean_object* v___x_1803_; 
v___x_1803_ = lp_mathlib_Mathlib_Meta_FunProp_Theorem_ctorElim___redArg(v_t_1800_, v_transition_1802_);
return v___x_1803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2(lean_object* v_msgData_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_){
_start:
{
lean_object* v___x_1810_; lean_object* v_env_1811_; lean_object* v___x_1812_; lean_object* v_mctx_1813_; lean_object* v_lctx_1814_; lean_object* v_options_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; 
v___x_1810_ = lean_st_ref_get(v___y_1808_);
v_env_1811_ = lean_ctor_get(v___x_1810_, 0);
lean_inc_ref(v_env_1811_);
lean_dec(v___x_1810_);
v___x_1812_ = lean_st_ref_get(v___y_1806_);
v_mctx_1813_ = lean_ctor_get(v___x_1812_, 0);
lean_inc_ref(v_mctx_1813_);
lean_dec(v___x_1812_);
v_lctx_1814_ = lean_ctor_get(v___y_1805_, 2);
v_options_1815_ = lean_ctor_get(v___y_1807_, 2);
lean_inc_ref(v_options_1815_);
lean_inc_ref(v_lctx_1814_);
v___x_1816_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1816_, 0, v_env_1811_);
lean_ctor_set(v___x_1816_, 1, v_mctx_1813_);
lean_ctor_set(v___x_1816_, 2, v_lctx_1814_);
lean_ctor_set(v___x_1816_, 3, v_options_1815_);
v___x_1817_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1817_, 0, v___x_1816_);
lean_ctor_set(v___x_1817_, 1, v_msgData_1804_);
v___x_1818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1818_, 0, v___x_1817_);
return v___x_1818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2___boxed(lean_object* v_msgData_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_){
_start:
{
lean_object* v_res_1825_; 
v_res_1825_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2(v_msgData_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_);
lean_dec(v___y_1823_);
lean_dec_ref(v___y_1822_);
lean_dec(v___y_1821_);
lean_dec_ref(v___y_1820_);
return v_res_1825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(lean_object* v_msg_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_){
_start:
{
lean_object* v_ref_1832_; lean_object* v___x_1833_; lean_object* v_a_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_1842_; 
v_ref_1832_ = lean_ctor_get(v___y_1829_, 5);
v___x_1833_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2(v_msg_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_);
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
v_isSharedCheck_1842_ = !lean_is_exclusive(v___x_1833_);
if (v_isSharedCheck_1842_ == 0)
{
v___x_1836_ = v___x_1833_;
v_isShared_1837_ = v_isSharedCheck_1842_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_a_1834_);
lean_dec(v___x_1833_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_1842_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___x_1838_; lean_object* v___x_1840_; 
lean_inc(v_ref_1832_);
v___x_1838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1838_, 0, v_ref_1832_);
lean_ctor_set(v___x_1838_, 1, v_a_1834_);
if (v_isShared_1837_ == 0)
{
lean_ctor_set_tag(v___x_1836_, 1);
lean_ctor_set(v___x_1836_, 0, v___x_1838_);
v___x_1840_ = v___x_1836_;
goto v_reusejp_1839_;
}
else
{
lean_object* v_reuseFailAlloc_1841_; 
v_reuseFailAlloc_1841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1841_, 0, v___x_1838_);
v___x_1840_ = v_reuseFailAlloc_1841_;
goto v_reusejp_1839_;
}
v_reusejp_1839_:
{
return v___x_1840_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg___boxed(lean_object* v_msg_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_){
_start:
{
lean_object* v_res_1849_; 
v_res_1849_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v_msg_1843_, v___y_1844_, v___y_1845_, v___y_1846_, v___y_1847_);
lean_dec(v___y_1847_);
lean_dec_ref(v___y_1846_);
lean_dec(v___y_1845_);
lean_dec_ref(v___y_1844_);
return v_res_1849_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1852_; lean_object* v___x_1853_; 
v___x_1852_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__1));
v___x_1853_ = l_Lean_stringToMessageData(v___x_1852_);
return v___x_1853_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4(void){
_start:
{
lean_object* v___x_1855_; lean_object* v___x_1856_; 
v___x_1855_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__3));
v___x_1856_ = l_Lean_stringToMessageData(v___x_1855_);
return v___x_1856_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6(void){
_start:
{
lean_object* v___x_1858_; lean_object* v___x_1859_; 
v___x_1858_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__5));
v___x_1859_ = l_Lean_stringToMessageData(v___x_1858_);
return v___x_1859_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8(void){
_start:
{
lean_object* v___x_1861_; lean_object* v___x_1862_; 
v___x_1861_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__7));
v___x_1862_ = l_Lean_stringToMessageData(v___x_1861_);
return v___x_1862_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11(void){
_start:
{
lean_object* v___x_1865_; lean_object* v___x_1866_; 
v___x_1865_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__10));
v___x_1866_ = l_Lean_stringToMessageData(v___x_1865_);
return v___x_1866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0(lean_object* v_declName_1867_, lean_object* v_prio_1868_, lean_object* v___x_1869_, lean_object* v___x_1870_, lean_object* v_xs_1871_, lean_object* v_b_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_){
_start:
{
lean_object* v___x_1878_; 
lean_inc_ref(v_b_1872_);
v___x_1878_ = lp_mathlib_Mathlib_Meta_FunProp_getFunProp_x3f(v_b_1872_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1878_) == 0)
{
lean_object* v_a_1879_; 
v_a_1879_ = lean_ctor_get(v___x_1878_, 0);
lean_inc(v_a_1879_);
lean_dec_ref_known(v___x_1878_, 1);
if (lean_obj_tag(v_a_1879_) == 1)
{
lean_object* v_val_1880_; lean_object* v___x_1882_; uint8_t v_isShared_1883_; uint8_t v_isSharedCheck_2136_; 
v_val_1880_ = lean_ctor_get(v_a_1879_, 0);
v_isSharedCheck_2136_ = !lean_is_exclusive(v_a_1879_);
if (v_isSharedCheck_2136_ == 0)
{
v___x_1882_ = v_a_1879_;
v_isShared_1883_ = v_isSharedCheck_2136_;
goto v_resetjp_1881_;
}
else
{
lean_inc(v_val_1880_);
lean_dec(v_a_1879_);
v___x_1882_ = lean_box(0);
v_isShared_1883_ = v_isSharedCheck_2136_;
goto v_resetjp_1881_;
}
v_resetjp_1881_:
{
lean_object* v_fst_1884_; lean_object* v_snd_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_2135_; 
v_fst_1884_ = lean_ctor_get(v_val_1880_, 0);
v_snd_1885_ = lean_ctor_get(v_val_1880_, 1);
v_isSharedCheck_2135_ = !lean_is_exclusive(v_val_1880_);
if (v_isSharedCheck_2135_ == 0)
{
v___x_1887_ = v_val_1880_;
v_isShared_1888_ = v_isSharedCheck_2135_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_snd_1885_);
lean_inc(v_fst_1884_);
lean_dec(v_val_1880_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_2135_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v___x_1889_; uint8_t v_foApprox_1890_; uint8_t v_ctxApprox_1891_; uint8_t v_quasiPatternApprox_1892_; uint8_t v_constApprox_1893_; uint8_t v_isDefEqStuckEx_1894_; uint8_t v_unificationHints_1895_; uint8_t v_proofIrrelevance_1896_; uint8_t v_assignSyntheticOpaque_1897_; uint8_t v_offsetCnstrs_1898_; uint8_t v_transparency_1899_; uint8_t v_etaStruct_1900_; uint8_t v_univApprox_1901_; uint8_t v_iota_1902_; uint8_t v_beta_1903_; uint8_t v_proj_1904_; uint8_t v_zetaDelta_1905_; uint8_t v_zetaUnused_1906_; uint8_t v_zetaHave_1907_; uint8_t v_canUnfoldPredicateConfig_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_2134_; 
v___x_1889_ = l_Lean_Meta_Context_config(v___y_1873_);
v_foApprox_1890_ = lean_ctor_get_uint8(v___x_1889_, 0);
v_ctxApprox_1891_ = lean_ctor_get_uint8(v___x_1889_, 1);
v_quasiPatternApprox_1892_ = lean_ctor_get_uint8(v___x_1889_, 2);
v_constApprox_1893_ = lean_ctor_get_uint8(v___x_1889_, 3);
v_isDefEqStuckEx_1894_ = lean_ctor_get_uint8(v___x_1889_, 4);
v_unificationHints_1895_ = lean_ctor_get_uint8(v___x_1889_, 5);
v_proofIrrelevance_1896_ = lean_ctor_get_uint8(v___x_1889_, 6);
v_assignSyntheticOpaque_1897_ = lean_ctor_get_uint8(v___x_1889_, 7);
v_offsetCnstrs_1898_ = lean_ctor_get_uint8(v___x_1889_, 8);
v_transparency_1899_ = lean_ctor_get_uint8(v___x_1889_, 9);
v_etaStruct_1900_ = lean_ctor_get_uint8(v___x_1889_, 10);
v_univApprox_1901_ = lean_ctor_get_uint8(v___x_1889_, 11);
v_iota_1902_ = lean_ctor_get_uint8(v___x_1889_, 12);
v_beta_1903_ = lean_ctor_get_uint8(v___x_1889_, 13);
v_proj_1904_ = lean_ctor_get_uint8(v___x_1889_, 14);
v_zetaDelta_1905_ = lean_ctor_get_uint8(v___x_1889_, 16);
v_zetaUnused_1906_ = lean_ctor_get_uint8(v___x_1889_, 17);
v_zetaHave_1907_ = lean_ctor_get_uint8(v___x_1889_, 18);
v_canUnfoldPredicateConfig_1908_ = lean_ctor_get_uint8(v___x_1889_, 19);
v_isSharedCheck_2134_ = !lean_is_exclusive(v___x_1889_);
if (v_isSharedCheck_2134_ == 0)
{
v___x_1910_ = v___x_1889_;
v_isShared_1911_ = v_isSharedCheck_2134_;
goto v_resetjp_1909_;
}
else
{
lean_dec(v___x_1889_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_2134_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
uint8_t v_trackZetaDelta_1912_; lean_object* v_zetaDeltaSet_1913_; lean_object* v_lctx_1914_; lean_object* v_localInstances_1915_; lean_object* v_defEqCtx_x3f_1916_; lean_object* v_synthPendingDepth_1917_; lean_object* v_customCanUnfoldPredicate_x3f_1918_; uint8_t v_univApprox_1919_; uint8_t v_inTypeClassResolution_1920_; uint8_t v_cacheInferType_1921_; uint8_t v___x_1922_; lean_object* v___x_1924_; 
v_trackZetaDelta_1912_ = lean_ctor_get_uint8(v___y_1873_, sizeof(void*)*7);
v_zetaDeltaSet_1913_ = lean_ctor_get(v___y_1873_, 1);
v_lctx_1914_ = lean_ctor_get(v___y_1873_, 2);
v_localInstances_1915_ = lean_ctor_get(v___y_1873_, 3);
v_defEqCtx_x3f_1916_ = lean_ctor_get(v___y_1873_, 4);
v_synthPendingDepth_1917_ = lean_ctor_get(v___y_1873_, 5);
v_customCanUnfoldPredicate_x3f_1918_ = lean_ctor_get(v___y_1873_, 6);
v_univApprox_1919_ = lean_ctor_get_uint8(v___y_1873_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1920_ = lean_ctor_get_uint8(v___y_1873_, sizeof(void*)*7 + 2);
v_cacheInferType_1921_ = lean_ctor_get_uint8(v___y_1873_, sizeof(void*)*7 + 3);
v___x_1922_ = 0;
if (v_isShared_1911_ == 0)
{
v___x_1924_ = v___x_1910_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_2133_; 
v_reuseFailAlloc_2133_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 0, v_foApprox_1890_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 1, v_ctxApprox_1891_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 2, v_quasiPatternApprox_1892_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 3, v_constApprox_1893_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 4, v_isDefEqStuckEx_1894_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 5, v_unificationHints_1895_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 6, v_proofIrrelevance_1896_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 7, v_assignSyntheticOpaque_1897_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 8, v_offsetCnstrs_1898_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 9, v_transparency_1899_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 10, v_etaStruct_1900_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 11, v_univApprox_1901_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 12, v_iota_1902_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 13, v_beta_1903_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 14, v_proj_1904_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 16, v_zetaDelta_1905_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 17, v_zetaUnused_1906_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 18, v_zetaHave_1907_);
lean_ctor_set_uint8(v_reuseFailAlloc_2133_, 19, v_canUnfoldPredicateConfig_1908_);
v___x_1924_ = v_reuseFailAlloc_2133_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
uint64_t v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; 
lean_ctor_set_uint8(v___x_1924_, 15, v___x_1922_);
v___x_1925_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1924_);
v___x_1926_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__0));
v___x_1927_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1927_, 0, v___x_1924_);
lean_ctor_set_uint64(v___x_1927_, sizeof(void*)*1, v___x_1925_);
lean_inc(v_customCanUnfoldPredicate_x3f_1918_);
lean_inc(v_synthPendingDepth_1917_);
lean_inc(v_defEqCtx_x3f_1916_);
lean_inc_ref(v_localInstances_1915_);
lean_inc_ref(v_lctx_1914_);
lean_inc(v_zetaDeltaSet_1913_);
v___x_1928_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1928_, 0, v___x_1927_);
lean_ctor_set(v___x_1928_, 1, v_zetaDeltaSet_1913_);
lean_ctor_set(v___x_1928_, 2, v_lctx_1914_);
lean_ctor_set(v___x_1928_, 3, v_localInstances_1915_);
lean_ctor_set(v___x_1928_, 4, v_defEqCtx_x3f_1916_);
lean_ctor_set(v___x_1928_, 5, v_synthPendingDepth_1917_);
lean_ctor_set(v___x_1928_, 6, v_customCanUnfoldPredicate_x3f_1918_);
lean_ctor_set_uint8(v___x_1928_, sizeof(void*)*7, v_trackZetaDelta_1912_);
lean_ctor_set_uint8(v___x_1928_, sizeof(void*)*7 + 1, v_univApprox_1919_);
lean_ctor_set_uint8(v___x_1928_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1920_);
lean_ctor_set_uint8(v___x_1928_, sizeof(void*)*7 + 3, v_cacheInferType_1921_);
lean_inc(v_snd_1885_);
v___x_1929_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f(v_snd_1885_, v___x_1926_, v___x_1928_, v___y_1874_, v___y_1875_, v___y_1876_);
lean_dec_ref_known(v___x_1928_, 7);
if (lean_obj_tag(v___x_1929_) == 0)
{
lean_object* v_a_1930_; lean_object* v___x_1931_; 
v_a_1930_ = lean_ctor_get(v___x_1929_, 0);
lean_inc_n(v_a_1930_, 2);
lean_dec_ref_known(v___x_1929_, 1);
v___x_1931_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get(v_a_1930_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1931_) == 0)
{
lean_object* v_a_1932_; lean_object* v___x_1933_; 
v_a_1932_ = lean_ctor_get(v___x_1931_, 0);
lean_inc(v_a_1932_);
lean_dec_ref_known(v___x_1931_, 1);
v___x_1933_ = lp_mathlib_Mathlib_Meta_FunProp_detectLambdaTheoremArgs(v_a_1932_, v_xs_1871_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1933_) == 0)
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_2108_; 
v_a_1934_ = lean_ctor_get(v___x_1933_, 0);
v_isSharedCheck_2108_ = !lean_is_exclusive(v___x_1933_);
if (v_isSharedCheck_2108_ == 0)
{
v___x_1936_ = v___x_1933_;
v_isShared_1937_ = v_isSharedCheck_2108_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1933_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_2108_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
if (lean_obj_tag(v_a_1934_) == 1)
{
lean_object* v_funPropName_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1956_; 
lean_dec(v_a_1930_);
lean_del_object(v___x_1887_);
lean_dec(v_snd_1885_);
lean_del_object(v___x_1882_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
v_funPropName_1938_ = lean_ctor_get(v_fst_1884_, 0);
v_isSharedCheck_1956_ = !lean_is_exclusive(v_fst_1884_);
if (v_isSharedCheck_1956_ == 0)
{
lean_object* v_unused_1957_; lean_object* v_unused_1958_; 
v_unused_1957_ = lean_ctor_get(v_fst_1884_, 2);
lean_dec(v_unused_1957_);
v_unused_1958_ = lean_ctor_get(v_fst_1884_, 1);
lean_dec(v_unused_1958_);
v___x_1940_ = v_fst_1884_;
v_isShared_1941_ = v_isSharedCheck_1956_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_funPropName_1938_);
lean_dec(v_fst_1884_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1956_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v_val_1942_; lean_object* v___x_1944_; uint8_t v_isShared_1945_; uint8_t v_isSharedCheck_1955_; 
v_val_1942_ = lean_ctor_get(v_a_1934_, 0);
v_isSharedCheck_1955_ = !lean_is_exclusive(v_a_1934_);
if (v_isSharedCheck_1955_ == 0)
{
v___x_1944_ = v_a_1934_;
v_isShared_1945_ = v_isSharedCheck_1955_;
goto v_resetjp_1943_;
}
else
{
lean_inc(v_val_1942_);
lean_dec(v_a_1934_);
v___x_1944_ = lean_box(0);
v_isShared_1945_ = v_isSharedCheck_1955_;
goto v_resetjp_1943_;
}
v_resetjp_1943_:
{
lean_object* v___x_1947_; 
if (v_isShared_1941_ == 0)
{
lean_ctor_set(v___x_1940_, 2, v_val_1942_);
lean_ctor_set(v___x_1940_, 1, v_declName_1867_);
v___x_1947_ = v___x_1940_;
goto v_reusejp_1946_;
}
else
{
lean_object* v_reuseFailAlloc_1954_; 
v_reuseFailAlloc_1954_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1954_, 0, v_funPropName_1938_);
lean_ctor_set(v_reuseFailAlloc_1954_, 1, v_declName_1867_);
lean_ctor_set(v_reuseFailAlloc_1954_, 2, v_val_1942_);
v___x_1947_ = v_reuseFailAlloc_1954_;
goto v_reusejp_1946_;
}
v_reusejp_1946_:
{
lean_object* v___x_1949_; 
if (v_isShared_1945_ == 0)
{
lean_ctor_set_tag(v___x_1944_, 0);
lean_ctor_set(v___x_1944_, 0, v___x_1947_);
v___x_1949_ = v___x_1944_;
goto v_reusejp_1948_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v___x_1947_);
v___x_1949_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1948_;
}
v_reusejp_1948_:
{
lean_object* v___x_1951_; 
if (v_isShared_1937_ == 0)
{
lean_ctor_set(v___x_1936_, 0, v___x_1949_);
v___x_1951_ = v___x_1936_;
goto v_reusejp_1950_;
}
else
{
lean_object* v_reuseFailAlloc_1952_; 
v_reuseFailAlloc_1952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1952_, 0, v___x_1949_);
v___x_1951_ = v_reuseFailAlloc_1952_;
goto v_reusejp_1950_;
}
v_reusejp_1950_:
{
return v___x_1951_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_1936_);
lean_dec(v_a_1934_);
if (lean_obj_tag(v_a_1930_) == 2)
{
lean_object* v_fData_1959_; lean_object* v___x_1961_; uint8_t v_isShared_1962_; uint8_t v_isSharedCheck_2087_; 
lean_dec(v_snd_1885_);
v_fData_1959_ = lean_ctor_get(v_a_1930_, 0);
v_isSharedCheck_2087_ = !lean_is_exclusive(v_a_1930_);
if (v_isSharedCheck_2087_ == 0)
{
v___x_1961_ = v_a_1930_;
v_isShared_1962_ = v_isSharedCheck_2087_;
goto v_resetjp_1960_;
}
else
{
lean_inc(v_fData_1959_);
lean_dec(v_a_1930_);
v___x_1961_ = lean_box(0);
v_isShared_1962_ = v_isSharedCheck_2087_;
goto v_resetjp_1960_;
}
v_resetjp_1960_:
{
lean_object* v_fn_1963_; 
v_fn_1963_ = lean_ctor_get(v_fData_1959_, 2);
switch(lean_obj_tag(v_fn_1963_))
{
case 4:
{
lean_object* v_funPropName_1964_; lean_object* v_args_1965_; lean_object* v_mainArgs_1966_; lean_object* v_declName_1967_; lean_object* v___x_1968_; 
lean_del_object(v___x_1887_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
v_funPropName_1964_ = lean_ctor_get(v_fst_1884_, 0);
lean_inc(v_funPropName_1964_);
lean_dec(v_fst_1884_);
v_args_1965_ = lean_ctor_get(v_fData_1959_, 3);
lean_inc_ref(v_args_1965_);
v_mainArgs_1966_ = lean_ctor_get(v_fData_1959_, 5);
lean_inc_ref(v_mainArgs_1966_);
v_declName_1967_ = lean_ctor_get(v_fn_1963_, 0);
lean_inc(v_declName_1967_);
v___x_1968_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition(v_fData_1959_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1968_) == 0)
{
lean_object* v_a_1969_; lean_object* v___x_1971_; uint8_t v_isShared_1972_; uint8_t v_isSharedCheck_1986_; 
v_a_1969_ = lean_ctor_get(v___x_1968_, 0);
v_isSharedCheck_1986_ = !lean_is_exclusive(v___x_1968_);
if (v_isSharedCheck_1986_ == 0)
{
v___x_1971_ = v___x_1968_;
v_isShared_1972_ = v_isSharedCheck_1986_;
goto v_resetjp_1970_;
}
else
{
lean_inc(v_a_1969_);
lean_dec(v___x_1968_);
v___x_1971_ = lean_box(0);
v_isShared_1972_ = v_isSharedCheck_1986_;
goto v_resetjp_1970_;
}
v_resetjp_1970_:
{
lean_object* v___x_1974_; 
if (v_isShared_1962_ == 0)
{
lean_ctor_set_tag(v___x_1961_, 0);
lean_ctor_set(v___x_1961_, 0, v_declName_1867_);
v___x_1974_ = v___x_1961_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v_declName_1867_);
v___x_1974_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
lean_object* v___x_1976_; 
if (v_isShared_1883_ == 0)
{
lean_ctor_set_tag(v___x_1882_, 0);
lean_ctor_set(v___x_1882_, 0, v_declName_1967_);
v___x_1976_ = v___x_1882_;
goto v_reusejp_1975_;
}
else
{
lean_object* v_reuseFailAlloc_1984_; 
v_reuseFailAlloc_1984_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1984_, 0, v_declName_1967_);
v___x_1976_ = v_reuseFailAlloc_1984_;
goto v_reusejp_1975_;
}
v_reusejp_1975_:
{
lean_object* v___x_1977_; uint8_t v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1982_; 
v___x_1977_ = lean_array_get_size(v_args_1965_);
lean_dec_ref(v_args_1965_);
v___x_1978_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_toTheoremForm(v_a_1969_);
lean_dec(v_a_1969_);
v___x_1979_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1979_, 0, v_funPropName_1964_);
lean_ctor_set(v___x_1979_, 1, v___x_1974_);
lean_ctor_set(v___x_1979_, 2, v___x_1976_);
lean_ctor_set(v___x_1979_, 3, v_mainArgs_1966_);
lean_ctor_set(v___x_1979_, 4, v___x_1977_);
lean_ctor_set(v___x_1979_, 5, v_prio_1868_);
lean_ctor_set_uint8(v___x_1979_, sizeof(void*)*6, v___x_1978_);
v___x_1980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1980_, 0, v___x_1979_);
if (v_isShared_1972_ == 0)
{
lean_ctor_set(v___x_1971_, 0, v___x_1980_);
v___x_1982_ = v___x_1971_;
goto v_reusejp_1981_;
}
else
{
lean_object* v_reuseFailAlloc_1983_; 
v_reuseFailAlloc_1983_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1983_, 0, v___x_1980_);
v___x_1982_ = v_reuseFailAlloc_1983_;
goto v_reusejp_1981_;
}
v_reusejp_1981_:
{
return v___x_1982_;
}
}
}
}
}
else
{
lean_object* v_a_1987_; lean_object* v___x_1989_; uint8_t v_isShared_1990_; uint8_t v_isSharedCheck_1994_; 
lean_dec(v_declName_1967_);
lean_dec_ref(v_mainArgs_1966_);
lean_dec_ref(v_args_1965_);
lean_dec(v_funPropName_1964_);
lean_del_object(v___x_1961_);
lean_del_object(v___x_1882_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_1987_ = lean_ctor_get(v___x_1968_, 0);
v_isSharedCheck_1994_ = !lean_is_exclusive(v___x_1968_);
if (v_isSharedCheck_1994_ == 0)
{
v___x_1989_ = v___x_1968_;
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
else
{
lean_inc(v_a_1987_);
lean_dec(v___x_1968_);
v___x_1989_ = lean_box(0);
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
v_resetjp_1988_:
{
lean_object* v___x_1992_; 
if (v_isShared_1990_ == 0)
{
v___x_1992_ = v___x_1989_;
goto v_reusejp_1991_;
}
else
{
lean_object* v_reuseFailAlloc_1993_; 
v_reuseFailAlloc_1993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1993_, 0, v_a_1987_);
v___x_1992_ = v_reuseFailAlloc_1993_;
goto v_reusejp_1991_;
}
v_reusejp_1991_:
{
return v___x_1992_;
}
}
}
}
case 1:
{
lean_object* v_funPropName_1995_; lean_object* v_args_1996_; lean_object* v_mainVar_1997_; uint8_t v___x_1998_; lean_object* v___x_1999_; 
lean_inc_ref(v_fn_1963_);
lean_del_object(v___x_1887_);
lean_dec_ref(v_b_1872_);
v_funPropName_1995_ = lean_ctor_get(v_fst_1884_, 0);
lean_inc(v_funPropName_1995_);
lean_dec(v_fst_1884_);
v_args_1996_ = lean_ctor_get(v_fData_1959_, 3);
lean_inc_ref(v_args_1996_);
v_mainVar_1997_ = lean_ctor_get(v_fData_1959_, 4);
lean_inc_ref(v_mainVar_1997_);
v___x_1998_ = 0;
v___x_1999_ = l_Lean_Meta_forallMetaTelescope(v___x_1869_, v___x_1998_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1999_) == 0)
{
lean_object* v_a_2000_; lean_object* v_snd_2001_; lean_object* v_snd_2002_; uint8_t v___x_2003_; lean_object* v___x_2004_; 
v_a_2000_ = lean_ctor_get(v___x_1999_, 0);
lean_inc(v_a_2000_);
lean_dec_ref_known(v___x_1999_, 1);
v_snd_2001_ = lean_ctor_get(v_a_2000_, 1);
lean_inc(v_snd_2001_);
lean_dec(v_a_2000_);
v_snd_2002_ = lean_ctor_get(v_snd_2001_, 1);
lean_inc(v_snd_2002_);
lean_dec(v_snd_2001_);
v___x_2003_ = 1;
v___x_2004_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(v_snd_2002_, v___x_2003_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_2004_) == 0)
{
lean_object* v_a_2005_; lean_object* v___x_2007_; uint8_t v_isShared_2008_; uint8_t v_isSharedCheck_2052_; 
v_a_2005_ = lean_ctor_get(v___x_2004_, 0);
v_isSharedCheck_2052_ = !lean_is_exclusive(v___x_2004_);
if (v_isSharedCheck_2052_ == 0)
{
v___x_2007_ = v___x_2004_;
v_isShared_2008_ = v_isSharedCheck_2052_;
goto v_resetjp_2006_;
}
else
{
lean_inc(v_a_2005_);
lean_dec(v___x_2004_);
v___x_2007_ = lean_box(0);
v_isShared_2008_ = v_isSharedCheck_2052_;
goto v_resetjp_2006_;
}
v_resetjp_2006_:
{
lean_object* v___x_2009_; 
v___x_2009_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication(v_fData_1959_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
lean_dec_ref(v_fData_1959_);
if (lean_obj_tag(v___x_2009_) == 0)
{
lean_object* v_a_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2043_; 
v_a_2010_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2043_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2043_ == 0)
{
v___x_2012_ = v___x_2009_;
v_isShared_2013_ = v_isSharedCheck_2043_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_a_2010_);
lean_dec(v___x_2009_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2043_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v___x_2017_; uint8_t v___y_2019_; uint8_t v___x_2030_; 
v___x_2017_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2017_, 0, v_funPropName_1995_);
lean_ctor_set(v___x_2017_, 1, v_declName_1867_);
lean_ctor_set(v___x_2017_, 2, v_a_2005_);
lean_ctor_set(v___x_2017_, 3, v_prio_1868_);
v___x_2030_ = lean_unbox(v_a_2010_);
lean_dec(v_a_2010_);
switch(v___x_2030_)
{
case 0:
{
lean_object* v___x_2031_; lean_object* v___x_2032_; 
lean_dec_ref_known(v___x_2017_, 4);
lean_del_object(v___x_2012_);
lean_del_object(v___x_2007_);
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_dec_ref_known(v_fn_1963_, 1);
lean_del_object(v___x_1961_);
lean_del_object(v___x_1882_);
v___x_2031_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__4);
v___x_2032_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v___x_2031_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
return v___x_2032_;
}
case 3:
{
uint8_t v___x_2033_; 
lean_del_object(v___x_2007_);
lean_del_object(v___x_1882_);
v___x_2033_ = l_Lean_Expr_isFVar(v_fn_1963_);
lean_dec_ref_known(v_fn_1963_, 1);
if (v___x_2033_ == 0)
{
v___y_2019_ = v___x_2033_;
goto v___jp_2018_;
}
else
{
lean_object* v___x_2034_; lean_object* v___x_2035_; uint8_t v___x_2036_; 
v___x_2034_ = lean_array_get_size(v_args_1996_);
v___x_2035_ = lean_unsigned_to_nat(1u);
v___x_2036_ = lean_nat_dec_eq(v___x_2034_, v___x_2035_);
v___y_2019_ = v___x_2036_;
goto v___jp_2018_;
}
}
default: 
{
lean_object* v___x_2038_; 
lean_del_object(v___x_2012_);
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_dec_ref_known(v_fn_1963_, 1);
lean_del_object(v___x_1961_);
if (v_isShared_1883_ == 0)
{
lean_ctor_set_tag(v___x_1882_, 2);
lean_ctor_set(v___x_1882_, 0, v___x_2017_);
v___x_2038_ = v___x_1882_;
goto v_reusejp_2037_;
}
else
{
lean_object* v_reuseFailAlloc_2042_; 
v_reuseFailAlloc_2042_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2042_, 0, v___x_2017_);
v___x_2038_ = v_reuseFailAlloc_2042_;
goto v_reusejp_2037_;
}
v_reusejp_2037_:
{
lean_object* v___x_2040_; 
if (v_isShared_2008_ == 0)
{
lean_ctor_set(v___x_2007_, 0, v___x_2038_);
v___x_2040_ = v___x_2007_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_2038_);
v___x_2040_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2039_;
}
v_reusejp_2039_:
{
return v___x_2040_;
}
}
}
}
v___jp_2014_:
{
lean_object* v___x_2015_; lean_object* v___x_2016_; 
v___x_2015_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__2);
v___x_2016_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v___x_2015_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
return v___x_2016_;
}
v___jp_2018_:
{
if (v___y_2019_ == 0)
{
lean_dec_ref_known(v___x_2017_, 4);
lean_del_object(v___x_2012_);
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_del_object(v___x_1961_);
goto v___jp_2014_;
}
else
{
lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v_expr_2022_; uint8_t v___x_2023_; 
v___x_2020_ = lean_unsigned_to_nat(0u);
v___x_2021_ = lean_array_get(v___x_1870_, v_args_1996_, v___x_2020_);
lean_dec_ref(v_args_1996_);
v_expr_2022_ = lean_ctor_get(v___x_2021_, 0);
lean_inc_ref(v_expr_2022_);
lean_dec(v___x_2021_);
v___x_2023_ = lean_expr_eqv(v_expr_2022_, v_mainVar_1997_);
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_expr_2022_);
if (v___x_2023_ == 0)
{
lean_dec_ref_known(v___x_2017_, 4);
lean_del_object(v___x_2012_);
lean_del_object(v___x_1961_);
goto v___jp_2014_;
}
else
{
lean_object* v___x_2025_; 
if (v_isShared_1962_ == 0)
{
lean_ctor_set_tag(v___x_1961_, 3);
lean_ctor_set(v___x_1961_, 0, v___x_2017_);
v___x_2025_ = v___x_1961_;
goto v_reusejp_2024_;
}
else
{
lean_object* v_reuseFailAlloc_2029_; 
v_reuseFailAlloc_2029_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2029_, 0, v___x_2017_);
v___x_2025_ = v_reuseFailAlloc_2029_;
goto v_reusejp_2024_;
}
v_reusejp_2024_:
{
lean_object* v___x_2027_; 
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 0, v___x_2025_);
v___x_2027_ = v___x_2012_;
goto v_reusejp_2026_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v___x_2025_);
v___x_2027_ = v_reuseFailAlloc_2028_;
goto v_reusejp_2026_;
}
v_reusejp_2026_:
{
return v___x_2027_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2044_; lean_object* v___x_2046_; uint8_t v_isShared_2047_; uint8_t v_isSharedCheck_2051_; 
lean_del_object(v___x_2007_);
lean_dec(v_a_2005_);
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_dec(v_funPropName_1995_);
lean_dec_ref_known(v_fn_1963_, 1);
lean_del_object(v___x_1961_);
lean_del_object(v___x_1882_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2044_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2051_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2051_ == 0)
{
v___x_2046_ = v___x_2009_;
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
else
{
lean_inc(v_a_2044_);
lean_dec(v___x_2009_);
v___x_2046_ = lean_box(0);
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
v_resetjp_2045_:
{
lean_object* v___x_2049_; 
if (v_isShared_2047_ == 0)
{
v___x_2049_ = v___x_2046_;
goto v_reusejp_2048_;
}
else
{
lean_object* v_reuseFailAlloc_2050_; 
v_reuseFailAlloc_2050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2050_, 0, v_a_2044_);
v___x_2049_ = v_reuseFailAlloc_2050_;
goto v_reusejp_2048_;
}
v_reusejp_2048_:
{
return v___x_2049_;
}
}
}
}
}
else
{
lean_object* v_a_2053_; lean_object* v___x_2055_; uint8_t v_isShared_2056_; uint8_t v_isSharedCheck_2060_; 
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_dec(v_funPropName_1995_);
lean_dec_ref_known(v_fn_1963_, 1);
lean_del_object(v___x_1961_);
lean_dec_ref(v_fData_1959_);
lean_del_object(v___x_1882_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2053_ = lean_ctor_get(v___x_2004_, 0);
v_isSharedCheck_2060_ = !lean_is_exclusive(v___x_2004_);
if (v_isSharedCheck_2060_ == 0)
{
v___x_2055_ = v___x_2004_;
v_isShared_2056_ = v_isSharedCheck_2060_;
goto v_resetjp_2054_;
}
else
{
lean_inc(v_a_2053_);
lean_dec(v___x_2004_);
v___x_2055_ = lean_box(0);
v_isShared_2056_ = v_isSharedCheck_2060_;
goto v_resetjp_2054_;
}
v_resetjp_2054_:
{
lean_object* v___x_2058_; 
if (v_isShared_2056_ == 0)
{
v___x_2058_ = v___x_2055_;
goto v_reusejp_2057_;
}
else
{
lean_object* v_reuseFailAlloc_2059_; 
v_reuseFailAlloc_2059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2059_, 0, v_a_2053_);
v___x_2058_ = v_reuseFailAlloc_2059_;
goto v_reusejp_2057_;
}
v_reusejp_2057_:
{
return v___x_2058_;
}
}
}
}
else
{
lean_object* v_a_2061_; lean_object* v___x_2063_; uint8_t v_isShared_2064_; uint8_t v_isSharedCheck_2068_; 
lean_dec_ref(v_mainVar_1997_);
lean_dec_ref(v_args_1996_);
lean_dec(v_funPropName_1995_);
lean_dec_ref_known(v_fn_1963_, 1);
lean_del_object(v___x_1961_);
lean_dec_ref(v_fData_1959_);
lean_del_object(v___x_1882_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2061_ = lean_ctor_get(v___x_1999_, 0);
v_isSharedCheck_2068_ = !lean_is_exclusive(v___x_1999_);
if (v_isSharedCheck_2068_ == 0)
{
v___x_2063_ = v___x_1999_;
v_isShared_2064_ = v_isSharedCheck_2068_;
goto v_resetjp_2062_;
}
else
{
lean_inc(v_a_2061_);
lean_dec(v___x_1999_);
v___x_2063_ = lean_box(0);
v_isShared_2064_ = v_isSharedCheck_2068_;
goto v_resetjp_2062_;
}
v_resetjp_2062_:
{
lean_object* v___x_2066_; 
if (v_isShared_2064_ == 0)
{
v___x_2066_ = v___x_2063_;
goto v_reusejp_2065_;
}
else
{
lean_object* v_reuseFailAlloc_2067_; 
v_reuseFailAlloc_2067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2067_, 0, v_a_2061_);
v___x_2066_ = v_reuseFailAlloc_2067_;
goto v_reusejp_2065_;
}
v_reusejp_2065_:
{
return v___x_2066_;
}
}
}
}
default: 
{
lean_object* v___x_2069_; 
lean_del_object(v___x_1961_);
lean_dec_ref(v_fData_1959_);
lean_dec(v_fst_1884_);
lean_del_object(v___x_1882_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v___x_2069_ = l_Lean_Meta_ppExpr(v_b_1872_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_2069_) == 0)
{
lean_object* v_a_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2074_; 
v_a_2070_ = lean_ctor_get(v___x_2069_, 0);
lean_inc(v_a_2070_);
lean_dec_ref_known(v___x_2069_, 1);
v___x_2071_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__6);
v___x_2072_ = l_Lean_MessageData_ofFormat(v_a_2070_);
if (v_isShared_1888_ == 0)
{
lean_ctor_set_tag(v___x_1887_, 7);
lean_ctor_set(v___x_1887_, 1, v___x_2072_);
lean_ctor_set(v___x_1887_, 0, v___x_2071_);
v___x_2074_ = v___x_1887_;
goto v_reusejp_2073_;
}
else
{
lean_object* v_reuseFailAlloc_2078_; 
v_reuseFailAlloc_2078_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2078_, 0, v___x_2071_);
lean_ctor_set(v_reuseFailAlloc_2078_, 1, v___x_2072_);
v___x_2074_ = v_reuseFailAlloc_2078_;
goto v_reusejp_2073_;
}
v_reusejp_2073_:
{
lean_object* v___x_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; 
v___x_2075_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8);
v___x_2076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2076_, 0, v___x_2074_);
lean_ctor_set(v___x_2076_, 1, v___x_2075_);
v___x_2077_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v___x_2076_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
return v___x_2077_;
}
}
else
{
lean_object* v_a_2079_; lean_object* v___x_2081_; uint8_t v_isShared_2082_; uint8_t v_isSharedCheck_2086_; 
lean_del_object(v___x_1887_);
v_a_2079_ = lean_ctor_get(v___x_2069_, 0);
v_isSharedCheck_2086_ = !lean_is_exclusive(v___x_2069_);
if (v_isSharedCheck_2086_ == 0)
{
v___x_2081_ = v___x_2069_;
v_isShared_2082_ = v_isSharedCheck_2086_;
goto v_resetjp_2080_;
}
else
{
lean_inc(v_a_2079_);
lean_dec(v___x_2069_);
v___x_2081_ = lean_box(0);
v_isShared_2082_ = v_isSharedCheck_2086_;
goto v_resetjp_2080_;
}
v_resetjp_2080_:
{
lean_object* v___x_2084_; 
if (v_isShared_2082_ == 0)
{
v___x_2084_ = v___x_2081_;
goto v_reusejp_2083_;
}
else
{
lean_object* v_reuseFailAlloc_2085_; 
v_reuseFailAlloc_2085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2085_, 0, v_a_2079_);
v___x_2084_ = v_reuseFailAlloc_2085_;
goto v_reusejp_2083_;
}
v_reusejp_2083_:
{
return v___x_2084_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2088_; 
lean_dec(v_a_1930_);
lean_del_object(v___x_1887_);
lean_dec(v_fst_1884_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v___x_2088_ = l_Lean_Meta_ppExpr(v_snd_1885_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_2088_) == 0)
{
lean_object* v_a_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2096_; 
v_a_2089_ = lean_ctor_get(v___x_2088_, 0);
lean_inc(v_a_2089_);
lean_dec_ref_known(v___x_2088_, 1);
v___x_2090_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__9));
v___x_2091_ = l_Std_Format_defWidth;
v___x_2092_ = lean_unsigned_to_nat(0u);
v___x_2093_ = l_Std_Format_pretty(v_a_2089_, v___x_2091_, v___x_2092_, v___x_2092_);
v___x_2094_ = lean_string_append(v___x_2090_, v___x_2093_);
lean_dec_ref(v___x_2093_);
if (v_isShared_1883_ == 0)
{
lean_ctor_set_tag(v___x_1882_, 3);
lean_ctor_set(v___x_1882_, 0, v___x_2094_);
v___x_2096_ = v___x_1882_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2099_; 
v_reuseFailAlloc_2099_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2099_, 0, v___x_2094_);
v___x_2096_ = v_reuseFailAlloc_2099_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
lean_object* v___x_2097_; lean_object* v___x_2098_; 
v___x_2097_ = l_Lean_MessageData_ofFormat(v___x_2096_);
v___x_2098_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v___x_2097_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
return v___x_2098_;
}
}
else
{
lean_object* v_a_2100_; lean_object* v___x_2102_; uint8_t v_isShared_2103_; uint8_t v_isSharedCheck_2107_; 
lean_del_object(v___x_1882_);
v_a_2100_ = lean_ctor_get(v___x_2088_, 0);
v_isSharedCheck_2107_ = !lean_is_exclusive(v___x_2088_);
if (v_isSharedCheck_2107_ == 0)
{
v___x_2102_ = v___x_2088_;
v_isShared_2103_ = v_isSharedCheck_2107_;
goto v_resetjp_2101_;
}
else
{
lean_inc(v_a_2100_);
lean_dec(v___x_2088_);
v___x_2102_ = lean_box(0);
v_isShared_2103_ = v_isSharedCheck_2107_;
goto v_resetjp_2101_;
}
v_resetjp_2101_:
{
lean_object* v___x_2105_; 
if (v_isShared_2103_ == 0)
{
v___x_2105_ = v___x_2102_;
goto v_reusejp_2104_;
}
else
{
lean_object* v_reuseFailAlloc_2106_; 
v_reuseFailAlloc_2106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2106_, 0, v_a_2100_);
v___x_2105_ = v_reuseFailAlloc_2106_;
goto v_reusejp_2104_;
}
v_reusejp_2104_:
{
return v___x_2105_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2109_; lean_object* v___x_2111_; uint8_t v_isShared_2112_; uint8_t v_isSharedCheck_2116_; 
lean_dec(v_a_1930_);
lean_del_object(v___x_1887_);
lean_dec(v_snd_1885_);
lean_dec(v_fst_1884_);
lean_del_object(v___x_1882_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2109_ = lean_ctor_get(v___x_1933_, 0);
v_isSharedCheck_2116_ = !lean_is_exclusive(v___x_1933_);
if (v_isSharedCheck_2116_ == 0)
{
v___x_2111_ = v___x_1933_;
v_isShared_2112_ = v_isSharedCheck_2116_;
goto v_resetjp_2110_;
}
else
{
lean_inc(v_a_2109_);
lean_dec(v___x_1933_);
v___x_2111_ = lean_box(0);
v_isShared_2112_ = v_isSharedCheck_2116_;
goto v_resetjp_2110_;
}
v_resetjp_2110_:
{
lean_object* v___x_2114_; 
if (v_isShared_2112_ == 0)
{
v___x_2114_ = v___x_2111_;
goto v_reusejp_2113_;
}
else
{
lean_object* v_reuseFailAlloc_2115_; 
v_reuseFailAlloc_2115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2115_, 0, v_a_2109_);
v___x_2114_ = v_reuseFailAlloc_2115_;
goto v_reusejp_2113_;
}
v_reusejp_2113_:
{
return v___x_2114_;
}
}
}
}
else
{
lean_object* v_a_2117_; lean_object* v___x_2119_; uint8_t v_isShared_2120_; uint8_t v_isSharedCheck_2124_; 
lean_dec(v_a_1930_);
lean_del_object(v___x_1887_);
lean_dec(v_snd_1885_);
lean_dec(v_fst_1884_);
lean_del_object(v___x_1882_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2117_ = lean_ctor_get(v___x_1931_, 0);
v_isSharedCheck_2124_ = !lean_is_exclusive(v___x_1931_);
if (v_isSharedCheck_2124_ == 0)
{
v___x_2119_ = v___x_1931_;
v_isShared_2120_ = v_isSharedCheck_2124_;
goto v_resetjp_2118_;
}
else
{
lean_inc(v_a_2117_);
lean_dec(v___x_1931_);
v___x_2119_ = lean_box(0);
v_isShared_2120_ = v_isSharedCheck_2124_;
goto v_resetjp_2118_;
}
v_resetjp_2118_:
{
lean_object* v___x_2122_; 
if (v_isShared_2120_ == 0)
{
v___x_2122_ = v___x_2119_;
goto v_reusejp_2121_;
}
else
{
lean_object* v_reuseFailAlloc_2123_; 
v_reuseFailAlloc_2123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2123_, 0, v_a_2117_);
v___x_2122_ = v_reuseFailAlloc_2123_;
goto v_reusejp_2121_;
}
v_reusejp_2121_:
{
return v___x_2122_;
}
}
}
}
else
{
lean_object* v_a_2125_; lean_object* v___x_2127_; uint8_t v_isShared_2128_; uint8_t v_isSharedCheck_2132_; 
lean_del_object(v___x_1887_);
lean_dec(v_snd_1885_);
lean_dec(v_fst_1884_);
lean_del_object(v___x_1882_);
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2125_ = lean_ctor_get(v___x_1929_, 0);
v_isSharedCheck_2132_ = !lean_is_exclusive(v___x_1929_);
if (v_isSharedCheck_2132_ == 0)
{
v___x_2127_ = v___x_1929_;
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
else
{
lean_inc(v_a_2125_);
lean_dec(v___x_1929_);
v___x_2127_ = lean_box(0);
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
v_resetjp_2126_:
{
lean_object* v___x_2130_; 
if (v_isShared_2128_ == 0)
{
v___x_2130_ = v___x_2127_;
goto v_reusejp_2129_;
}
else
{
lean_object* v_reuseFailAlloc_2131_; 
v_reuseFailAlloc_2131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2131_, 0, v_a_2125_);
v___x_2130_ = v_reuseFailAlloc_2131_;
goto v_reusejp_2129_;
}
v_reusejp_2129_:
{
return v___x_2130_;
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
lean_object* v___x_2137_; 
lean_dec(v_a_1879_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v___x_2137_ = l_Lean_Meta_ppExpr(v_b_1872_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_2137_) == 0)
{
lean_object* v_a_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; 
v_a_2138_ = lean_ctor_get(v___x_2137_, 0);
lean_inc(v_a_2138_);
lean_dec_ref_known(v___x_2137_, 1);
v___x_2139_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__11);
v___x_2140_ = l_Lean_MessageData_ofFormat(v_a_2138_);
v___x_2141_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2141_, 0, v___x_2139_);
lean_ctor_set(v___x_2141_, 1, v___x_2140_);
v___x_2142_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8);
v___x_2143_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2143_, 0, v___x_2141_);
lean_ctor_set(v___x_2143_, 1, v___x_2142_);
v___x_2144_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v___x_2143_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_);
return v___x_2144_;
}
else
{
lean_object* v_a_2145_; lean_object* v___x_2147_; uint8_t v_isShared_2148_; uint8_t v_isSharedCheck_2152_; 
v_a_2145_ = lean_ctor_get(v___x_2137_, 0);
v_isSharedCheck_2152_ = !lean_is_exclusive(v___x_2137_);
if (v_isSharedCheck_2152_ == 0)
{
v___x_2147_ = v___x_2137_;
v_isShared_2148_ = v_isSharedCheck_2152_;
goto v_resetjp_2146_;
}
else
{
lean_inc(v_a_2145_);
lean_dec(v___x_2137_);
v___x_2147_ = lean_box(0);
v_isShared_2148_ = v_isSharedCheck_2152_;
goto v_resetjp_2146_;
}
v_resetjp_2146_:
{
lean_object* v___x_2150_; 
if (v_isShared_2148_ == 0)
{
v___x_2150_ = v___x_2147_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2151_; 
v_reuseFailAlloc_2151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2151_, 0, v_a_2145_);
v___x_2150_ = v_reuseFailAlloc_2151_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
return v___x_2150_;
}
}
}
}
}
else
{
lean_object* v_a_2153_; lean_object* v___x_2155_; uint8_t v_isShared_2156_; uint8_t v_isSharedCheck_2160_; 
lean_dec_ref(v_b_1872_);
lean_dec_ref(v___x_1869_);
lean_dec(v_prio_1868_);
lean_dec(v_declName_1867_);
v_a_2153_ = lean_ctor_get(v___x_1878_, 0);
v_isSharedCheck_2160_ = !lean_is_exclusive(v___x_1878_);
if (v_isSharedCheck_2160_ == 0)
{
v___x_2155_ = v___x_1878_;
v_isShared_2156_ = v_isSharedCheck_2160_;
goto v_resetjp_2154_;
}
else
{
lean_inc(v_a_2153_);
lean_dec(v___x_1878_);
v___x_2155_ = lean_box(0);
v_isShared_2156_ = v_isSharedCheck_2160_;
goto v_resetjp_2154_;
}
v_resetjp_2154_:
{
lean_object* v___x_2158_; 
if (v_isShared_2156_ == 0)
{
v___x_2158_ = v___x_2155_;
goto v_reusejp_2157_;
}
else
{
lean_object* v_reuseFailAlloc_2159_; 
v_reuseFailAlloc_2159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2159_, 0, v_a_2153_);
v___x_2158_ = v_reuseFailAlloc_2159_;
goto v_reusejp_2157_;
}
v_reusejp_2157_:
{
return v___x_2158_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___boxed(lean_object* v_declName_2161_, lean_object* v_prio_2162_, lean_object* v___x_2163_, lean_object* v___x_2164_, lean_object* v_xs_2165_, lean_object* v_b_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_){
_start:
{
lean_object* v_res_2172_; 
v_res_2172_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0(v_declName_2161_, v_prio_2162_, v___x_2163_, v___x_2164_, v_xs_2165_, v_b_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v_xs_2165_);
lean_dec_ref(v___x_2164_);
return v_res_2172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(lean_object* v_ref_2173_, lean_object* v_msg_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_, lean_object* v___y_2178_){
_start:
{
lean_object* v_fileName_2180_; lean_object* v_fileMap_2181_; lean_object* v_options_2182_; lean_object* v_currRecDepth_2183_; lean_object* v_maxRecDepth_2184_; lean_object* v_ref_2185_; lean_object* v_currNamespace_2186_; lean_object* v_openDecls_2187_; lean_object* v_initHeartbeats_2188_; lean_object* v_maxHeartbeats_2189_; lean_object* v_quotContext_2190_; lean_object* v_currMacroScope_2191_; uint8_t v_diag_2192_; lean_object* v_cancelTk_x3f_2193_; uint8_t v_suppressElabErrors_2194_; lean_object* v_inheritedTraceOptions_2195_; lean_object* v_ref_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; 
v_fileName_2180_ = lean_ctor_get(v___y_2177_, 0);
v_fileMap_2181_ = lean_ctor_get(v___y_2177_, 1);
v_options_2182_ = lean_ctor_get(v___y_2177_, 2);
v_currRecDepth_2183_ = lean_ctor_get(v___y_2177_, 3);
v_maxRecDepth_2184_ = lean_ctor_get(v___y_2177_, 4);
v_ref_2185_ = lean_ctor_get(v___y_2177_, 5);
v_currNamespace_2186_ = lean_ctor_get(v___y_2177_, 6);
v_openDecls_2187_ = lean_ctor_get(v___y_2177_, 7);
v_initHeartbeats_2188_ = lean_ctor_get(v___y_2177_, 8);
v_maxHeartbeats_2189_ = lean_ctor_get(v___y_2177_, 9);
v_quotContext_2190_ = lean_ctor_get(v___y_2177_, 10);
v_currMacroScope_2191_ = lean_ctor_get(v___y_2177_, 11);
v_diag_2192_ = lean_ctor_get_uint8(v___y_2177_, sizeof(void*)*14);
v_cancelTk_x3f_2193_ = lean_ctor_get(v___y_2177_, 12);
v_suppressElabErrors_2194_ = lean_ctor_get_uint8(v___y_2177_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2195_ = lean_ctor_get(v___y_2177_, 13);
v_ref_2196_ = l_Lean_replaceRef(v_ref_2173_, v_ref_2185_);
lean_inc_ref(v_inheritedTraceOptions_2195_);
lean_inc(v_cancelTk_x3f_2193_);
lean_inc(v_currMacroScope_2191_);
lean_inc(v_quotContext_2190_);
lean_inc(v_maxHeartbeats_2189_);
lean_inc(v_initHeartbeats_2188_);
lean_inc(v_openDecls_2187_);
lean_inc(v_currNamespace_2186_);
lean_inc(v_maxRecDepth_2184_);
lean_inc(v_currRecDepth_2183_);
lean_inc_ref(v_options_2182_);
lean_inc_ref(v_fileMap_2181_);
lean_inc_ref(v_fileName_2180_);
v___x_2197_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2197_, 0, v_fileName_2180_);
lean_ctor_set(v___x_2197_, 1, v_fileMap_2181_);
lean_ctor_set(v___x_2197_, 2, v_options_2182_);
lean_ctor_set(v___x_2197_, 3, v_currRecDepth_2183_);
lean_ctor_set(v___x_2197_, 4, v_maxRecDepth_2184_);
lean_ctor_set(v___x_2197_, 5, v_ref_2196_);
lean_ctor_set(v___x_2197_, 6, v_currNamespace_2186_);
lean_ctor_set(v___x_2197_, 7, v_openDecls_2187_);
lean_ctor_set(v___x_2197_, 8, v_initHeartbeats_2188_);
lean_ctor_set(v___x_2197_, 9, v_maxHeartbeats_2189_);
lean_ctor_set(v___x_2197_, 10, v_quotContext_2190_);
lean_ctor_set(v___x_2197_, 11, v_currMacroScope_2191_);
lean_ctor_set(v___x_2197_, 12, v_cancelTk_x3f_2193_);
lean_ctor_set(v___x_2197_, 13, v_inheritedTraceOptions_2195_);
lean_ctor_set_uint8(v___x_2197_, sizeof(void*)*14, v_diag_2192_);
lean_ctor_set_uint8(v___x_2197_, sizeof(void*)*14 + 1, v_suppressElabErrors_2194_);
v___x_2198_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v_msg_2174_, v___y_2175_, v___y_2176_, v___x_2197_, v___y_2178_);
lean_dec_ref_known(v___x_2197_, 14);
return v___x_2198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg___boxed(lean_object* v_ref_2199_, lean_object* v_msg_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_){
_start:
{
lean_object* v_res_2206_; 
v_res_2206_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_ref_2199_, v_msg_2200_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_);
lean_dec(v___y_2204_);
lean_dec_ref(v___y_2203_);
lean_dec(v___y_2202_);
lean_dec_ref(v___y_2201_);
lean_dec(v_ref_2199_);
return v_res_2206_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_2207_; 
v___x_2207_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2207_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_2208_; lean_object* v___x_2209_; 
v___x_2208_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__0);
v___x_2209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2209_, 0, v___x_2208_);
return v___x_2209_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; 
v___x_2210_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_2211_ = lean_unsigned_to_nat(0u);
v___x_2212_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2212_, 0, v___x_2211_);
lean_ctor_set(v___x_2212_, 1, v___x_2211_);
lean_ctor_set(v___x_2212_, 2, v___x_2211_);
lean_ctor_set(v___x_2212_, 3, v___x_2211_);
lean_ctor_set(v___x_2212_, 4, v___x_2210_);
lean_ctor_set(v___x_2212_, 5, v___x_2210_);
lean_ctor_set(v___x_2212_, 6, v___x_2210_);
lean_ctor_set(v___x_2212_, 7, v___x_2210_);
lean_ctor_set(v___x_2212_, 8, v___x_2210_);
lean_ctor_set(v___x_2212_, 9, v___x_2210_);
return v___x_2212_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; 
v___x_2213_ = lean_unsigned_to_nat(32u);
v___x_2214_ = lean_mk_empty_array_with_capacity(v___x_2213_);
v___x_2215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2215_, 0, v___x_2214_);
return v___x_2215_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; 
v___x_2216_ = ((size_t)5ULL);
v___x_2217_ = lean_unsigned_to_nat(0u);
v___x_2218_ = lean_unsigned_to_nat(32u);
v___x_2219_ = lean_mk_empty_array_with_capacity(v___x_2218_);
v___x_2220_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__3);
v___x_2221_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2221_, 0, v___x_2220_);
lean_ctor_set(v___x_2221_, 1, v___x_2219_);
lean_ctor_set(v___x_2221_, 2, v___x_2217_);
lean_ctor_set(v___x_2221_, 3, v___x_2217_);
lean_ctor_set_usize(v___x_2221_, 4, v___x_2216_);
return v___x_2221_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; 
v___x_2222_ = lean_box(1);
v___x_2223_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__4);
v___x_2224_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_2225_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2225_, 0, v___x_2224_);
lean_ctor_set(v___x_2225_, 1, v___x_2223_);
lean_ctor_set(v___x_2225_, 2, v___x_2222_);
return v___x_2225_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7(void){
_start:
{
lean_object* v___x_2227_; lean_object* v___x_2228_; 
v___x_2227_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__6));
v___x_2228_ = l_Lean_stringToMessageData(v___x_2227_);
return v___x_2228_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9(void){
_start:
{
lean_object* v___x_2230_; lean_object* v___x_2231_; 
v___x_2230_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__8));
v___x_2231_ = l_Lean_stringToMessageData(v___x_2230_);
return v___x_2231_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11(void){
_start:
{
lean_object* v___x_2233_; lean_object* v___x_2234_; 
v___x_2233_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__10));
v___x_2234_ = l_Lean_stringToMessageData(v___x_2233_);
return v___x_2234_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13(void){
_start:
{
lean_object* v___x_2236_; lean_object* v___x_2237_; 
v___x_2236_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__12));
v___x_2237_ = l_Lean_stringToMessageData(v___x_2236_);
return v___x_2237_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15(void){
_start:
{
lean_object* v___x_2239_; lean_object* v___x_2240_; 
v___x_2239_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__14));
v___x_2240_ = l_Lean_stringToMessageData(v___x_2239_);
return v___x_2240_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17(void){
_start:
{
lean_object* v___x_2242_; lean_object* v___x_2243_; 
v___x_2242_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__16));
v___x_2243_ = l_Lean_stringToMessageData(v___x_2242_);
return v___x_2243_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19(void){
_start:
{
lean_object* v___x_2245_; lean_object* v___x_2246_; 
v___x_2245_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__18));
v___x_2246_ = l_Lean_stringToMessageData(v___x_2245_);
return v___x_2246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg(lean_object* v_msg_2247_, lean_object* v_declHint_2248_, lean_object* v___y_2249_){
_start:
{
lean_object* v___x_2251_; lean_object* v_env_2252_; uint8_t v___x_2253_; 
v___x_2251_ = lean_st_ref_get(v___y_2249_);
v_env_2252_ = lean_ctor_get(v___x_2251_, 0);
lean_inc_ref(v_env_2252_);
lean_dec(v___x_2251_);
v___x_2253_ = l_Lean_Name_isAnonymous(v_declHint_2248_);
if (v___x_2253_ == 0)
{
uint8_t v_isExporting_2254_; 
v_isExporting_2254_ = lean_ctor_get_uint8(v_env_2252_, sizeof(void*)*8);
if (v_isExporting_2254_ == 0)
{
lean_object* v___x_2255_; 
lean_dec_ref(v_env_2252_);
lean_dec(v_declHint_2248_);
v___x_2255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2255_, 0, v_msg_2247_);
return v___x_2255_;
}
else
{
lean_object* v___x_2256_; uint8_t v___x_2257_; 
lean_inc_ref(v_env_2252_);
v___x_2256_ = l_Lean_Environment_setExporting(v_env_2252_, v___x_2253_);
lean_inc(v_declHint_2248_);
lean_inc_ref(v___x_2256_);
v___x_2257_ = l_Lean_Environment_contains(v___x_2256_, v_declHint_2248_, v_isExporting_2254_);
if (v___x_2257_ == 0)
{
lean_object* v___x_2258_; 
lean_dec_ref(v___x_2256_);
lean_dec_ref(v_env_2252_);
lean_dec(v_declHint_2248_);
v___x_2258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2258_, 0, v_msg_2247_);
return v___x_2258_;
}
else
{
lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v_c_2264_; lean_object* v___x_2265_; 
v___x_2259_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__2);
v___x_2260_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__5);
v___x_2261_ = l_Lean_Options_empty;
v___x_2262_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2262_, 0, v___x_2256_);
lean_ctor_set(v___x_2262_, 1, v___x_2259_);
lean_ctor_set(v___x_2262_, 2, v___x_2260_);
lean_ctor_set(v___x_2262_, 3, v___x_2261_);
lean_inc(v_declHint_2248_);
v___x_2263_ = l_Lean_MessageData_ofConstName(v_declHint_2248_, v___x_2253_);
v_c_2264_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_2264_, 0, v___x_2262_);
lean_ctor_set(v_c_2264_, 1, v___x_2263_);
v___x_2265_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_2252_, v_declHint_2248_);
if (lean_obj_tag(v___x_2265_) == 0)
{
lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; 
lean_dec_ref(v_env_2252_);
lean_dec(v_declHint_2248_);
v___x_2266_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7);
v___x_2267_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2267_, 0, v___x_2266_);
lean_ctor_set(v___x_2267_, 1, v_c_2264_);
v___x_2268_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__9);
v___x_2269_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2267_);
lean_ctor_set(v___x_2269_, 1, v___x_2268_);
v___x_2270_ = l_Lean_MessageData_note(v___x_2269_);
v___x_2271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2271_, 0, v_msg_2247_);
lean_ctor_set(v___x_2271_, 1, v___x_2270_);
v___x_2272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2272_, 0, v___x_2271_);
return v___x_2272_;
}
else
{
lean_object* v_val_2273_; lean_object* v___x_2275_; uint8_t v_isShared_2276_; uint8_t v_isSharedCheck_2308_; 
v_val_2273_ = lean_ctor_get(v___x_2265_, 0);
v_isSharedCheck_2308_ = !lean_is_exclusive(v___x_2265_);
if (v_isSharedCheck_2308_ == 0)
{
v___x_2275_ = v___x_2265_;
v_isShared_2276_ = v_isSharedCheck_2308_;
goto v_resetjp_2274_;
}
else
{
lean_inc(v_val_2273_);
lean_dec(v___x_2265_);
v___x_2275_ = lean_box(0);
v_isShared_2276_ = v_isSharedCheck_2308_;
goto v_resetjp_2274_;
}
v_resetjp_2274_:
{
lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v_mod_2280_; uint8_t v___x_2281_; 
v___x_2277_ = lean_box(0);
v___x_2278_ = l_Lean_Environment_header(v_env_2252_);
lean_dec_ref(v_env_2252_);
v___x_2279_ = l_Lean_EnvironmentHeader_moduleNames(v___x_2278_);
v_mod_2280_ = lean_array_get(v___x_2277_, v___x_2279_, v_val_2273_);
lean_dec(v_val_2273_);
lean_dec_ref(v___x_2279_);
v___x_2281_ = l_Lean_isPrivateName(v_declHint_2248_);
lean_dec(v_declHint_2248_);
if (v___x_2281_ == 0)
{
lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2293_; 
v___x_2282_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__11);
v___x_2283_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2283_, 0, v___x_2282_);
lean_ctor_set(v___x_2283_, 1, v_c_2264_);
v___x_2284_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__13);
v___x_2285_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2285_, 0, v___x_2283_);
lean_ctor_set(v___x_2285_, 1, v___x_2284_);
v___x_2286_ = l_Lean_MessageData_ofName(v_mod_2280_);
v___x_2287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2287_, 0, v___x_2285_);
lean_ctor_set(v___x_2287_, 1, v___x_2286_);
v___x_2288_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__15);
v___x_2289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2287_);
lean_ctor_set(v___x_2289_, 1, v___x_2288_);
v___x_2290_ = l_Lean_MessageData_note(v___x_2289_);
v___x_2291_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2291_, 0, v_msg_2247_);
lean_ctor_set(v___x_2291_, 1, v___x_2290_);
if (v_isShared_2276_ == 0)
{
lean_ctor_set_tag(v___x_2275_, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2291_);
v___x_2293_ = v___x_2275_;
goto v_reusejp_2292_;
}
else
{
lean_object* v_reuseFailAlloc_2294_; 
v_reuseFailAlloc_2294_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2294_, 0, v___x_2291_);
v___x_2293_ = v_reuseFailAlloc_2294_;
goto v_reusejp_2292_;
}
v_reusejp_2292_:
{
return v___x_2293_;
}
}
else
{
lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2306_; 
v___x_2295_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__7);
v___x_2296_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2296_, 0, v___x_2295_);
lean_ctor_set(v___x_2296_, 1, v_c_2264_);
v___x_2297_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__17);
v___x_2298_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2298_, 0, v___x_2296_);
lean_ctor_set(v___x_2298_, 1, v___x_2297_);
v___x_2299_ = l_Lean_MessageData_ofName(v_mod_2280_);
v___x_2300_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2300_, 0, v___x_2298_);
lean_ctor_set(v___x_2300_, 1, v___x_2299_);
v___x_2301_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___closed__19);
v___x_2302_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2302_, 0, v___x_2300_);
lean_ctor_set(v___x_2302_, 1, v___x_2301_);
v___x_2303_ = l_Lean_MessageData_note(v___x_2302_);
v___x_2304_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2304_, 0, v_msg_2247_);
lean_ctor_set(v___x_2304_, 1, v___x_2303_);
if (v_isShared_2276_ == 0)
{
lean_ctor_set_tag(v___x_2275_, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2304_);
v___x_2306_ = v___x_2275_;
goto v_reusejp_2305_;
}
else
{
lean_object* v_reuseFailAlloc_2307_; 
v_reuseFailAlloc_2307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2307_, 0, v___x_2304_);
v___x_2306_ = v_reuseFailAlloc_2307_;
goto v_reusejp_2305_;
}
v_reusejp_2305_:
{
return v___x_2306_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2309_; 
lean_dec_ref(v_env_2252_);
lean_dec(v_declHint_2248_);
v___x_2309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2309_, 0, v_msg_2247_);
return v___x_2309_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg___boxed(lean_object* v_msg_2310_, lean_object* v_declHint_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_){
_start:
{
lean_object* v_res_2314_; 
v_res_2314_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg(v_msg_2310_, v_declHint_2311_, v___y_2312_);
lean_dec(v___y_2312_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object* v_msg_2315_, lean_object* v_declHint_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_){
_start:
{
lean_object* v___x_2322_; lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2332_; 
v___x_2322_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg(v_msg_2315_, v_declHint_2316_, v___y_2320_);
v_a_2323_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2332_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2332_ == 0)
{
v___x_2325_ = v___x_2322_;
v_isShared_2326_ = v_isSharedCheck_2332_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2322_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2332_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2330_; 
v___x_2327_ = l_Lean_unknownIdentifierMessageTag;
v___x_2328_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2328_, 0, v___x_2327_);
lean_ctor_set(v___x_2328_, 1, v_a_2323_);
if (v_isShared_2326_ == 0)
{
lean_ctor_set(v___x_2325_, 0, v___x_2328_);
v___x_2330_ = v___x_2325_;
goto v_reusejp_2329_;
}
else
{
lean_object* v_reuseFailAlloc_2331_; 
v_reuseFailAlloc_2331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2331_, 0, v___x_2328_);
v___x_2330_ = v_reuseFailAlloc_2331_;
goto v_reusejp_2329_;
}
v_reusejp_2329_:
{
return v___x_2330_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5___boxed(lean_object* v_msg_2333_, lean_object* v_declHint_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_){
_start:
{
lean_object* v_res_2340_; 
v_res_2340_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5(v_msg_2333_, v_declHint_2334_, v___y_2335_, v___y_2336_, v___y_2337_, v___y_2338_);
lean_dec(v___y_2338_);
lean_dec_ref(v___y_2337_);
lean_dec(v___y_2336_);
lean_dec_ref(v___y_2335_);
return v_res_2340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_ref_2341_, lean_object* v_msg_2342_, lean_object* v_declHint_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_){
_start:
{
lean_object* v___x_2349_; lean_object* v_a_2350_; lean_object* v___x_2351_; 
v___x_2349_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5(v_msg_2342_, v_declHint_2343_, v___y_2344_, v___y_2345_, v___y_2346_, v___y_2347_);
v_a_2350_ = lean_ctor_get(v___x_2349_, 0);
lean_inc(v_a_2350_);
lean_dec_ref(v___x_2349_);
v___x_2351_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_ref_2341_, v_a_2350_, v___y_2344_, v___y_2345_, v___y_2346_, v___y_2347_);
return v___x_2351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_ref_2352_, lean_object* v_msg_2353_, lean_object* v_declHint_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v_res_2360_; 
v_res_2360_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg(v_ref_2352_, v_msg_2353_, v_declHint_2354_, v___y_2355_, v___y_2356_, v___y_2357_, v___y_2358_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
lean_dec(v___y_2356_);
lean_dec_ref(v___y_2355_);
lean_dec(v_ref_2352_);
return v_res_2360_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_2362_; lean_object* v___x_2363_; 
v___x_2362_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__0));
v___x_2363_ = l_Lean_stringToMessageData(v___x_2362_);
return v___x_2363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_2364_, lean_object* v_constName_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_){
_start:
{
lean_object* v___x_2371_; uint8_t v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; 
v___x_2371_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___closed__1);
v___x_2372_ = 0;
lean_inc(v_constName_2365_);
v___x_2373_ = l_Lean_MessageData_ofConstName(v_constName_2365_, v___x_2372_);
v___x_2374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___x_2371_);
lean_ctor_set(v___x_2374_, 1, v___x_2373_);
v___x_2375_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___closed__8);
v___x_2376_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2376_, 0, v___x_2374_);
lean_ctor_set(v___x_2376_, 1, v___x_2375_);
v___x_2377_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg(v_ref_2364_, v___x_2376_, v_constName_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
return v___x_2377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_2378_, lean_object* v_constName_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_){
_start:
{
lean_object* v_res_2385_; 
v_res_2385_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg(v_ref_2378_, v_constName_2379_, v___y_2380_, v___y_2381_, v___y_2382_, v___y_2383_);
lean_dec(v___y_2383_);
lean_dec_ref(v___y_2382_);
lean_dec(v___y_2381_);
lean_dec_ref(v___y_2380_);
lean_dec(v_ref_2378_);
return v_res_2385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg(lean_object* v_constName_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_){
_start:
{
lean_object* v_ref_2392_; lean_object* v___x_2393_; 
v_ref_2392_ = lean_ctor_get(v___y_2389_, 5);
v___x_2393_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg(v_ref_2392_, v_constName_2386_, v___y_2387_, v___y_2388_, v___y_2389_, v___y_2390_);
return v___x_2393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg___boxed(lean_object* v_constName_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_){
_start:
{
lean_object* v_res_2400_; 
v_res_2400_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg(v_constName_2394_, v___y_2395_, v___y_2396_, v___y_2397_, v___y_2398_);
lean_dec(v___y_2398_);
lean_dec_ref(v___y_2397_);
lean_dec(v___y_2396_);
lean_dec_ref(v___y_2395_);
return v_res_2400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0(lean_object* v_constName_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_){
_start:
{
lean_object* v___x_2407_; lean_object* v_env_2408_; uint8_t v___x_2409_; lean_object* v___x_2410_; 
v___x_2407_ = lean_st_ref_get(v___y_2405_);
v_env_2408_ = lean_ctor_get(v___x_2407_, 0);
lean_inc_ref(v_env_2408_);
lean_dec(v___x_2407_);
v___x_2409_ = 0;
lean_inc(v_constName_2401_);
v___x_2410_ = l_Lean_Environment_find_x3f(v_env_2408_, v_constName_2401_, v___x_2409_);
if (lean_obj_tag(v___x_2410_) == 0)
{
lean_object* v___x_2411_; 
v___x_2411_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg(v_constName_2401_, v___y_2402_, v___y_2403_, v___y_2404_, v___y_2405_);
return v___x_2411_;
}
else
{
lean_object* v_val_2412_; lean_object* v___x_2414_; uint8_t v_isShared_2415_; uint8_t v_isSharedCheck_2419_; 
lean_dec(v_constName_2401_);
v_val_2412_ = lean_ctor_get(v___x_2410_, 0);
v_isSharedCheck_2419_ = !lean_is_exclusive(v___x_2410_);
if (v_isSharedCheck_2419_ == 0)
{
v___x_2414_ = v___x_2410_;
v_isShared_2415_ = v_isSharedCheck_2419_;
goto v_resetjp_2413_;
}
else
{
lean_inc(v_val_2412_);
lean_dec(v___x_2410_);
v___x_2414_ = lean_box(0);
v_isShared_2415_ = v_isSharedCheck_2419_;
goto v_resetjp_2413_;
}
v_resetjp_2413_:
{
lean_object* v___x_2417_; 
if (v_isShared_2415_ == 0)
{
lean_ctor_set_tag(v___x_2414_, 0);
v___x_2417_ = v___x_2414_;
goto v_reusejp_2416_;
}
else
{
lean_object* v_reuseFailAlloc_2418_; 
v_reuseFailAlloc_2418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2418_, 0, v_val_2412_);
v___x_2417_ = v_reuseFailAlloc_2418_;
goto v_reusejp_2416_;
}
v_reusejp_2416_:
{
return v___x_2417_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0___boxed(lean_object* v_constName_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_res_2426_; 
v_res_2426_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0(v_constName_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
lean_dec(v___y_2424_);
lean_dec_ref(v___y_2423_);
lean_dec(v___y_2422_);
lean_dec_ref(v___y_2421_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst(lean_object* v_declName_2427_, lean_object* v_prio_2428_, lean_object* v_a_2429_, lean_object* v_a_2430_, lean_object* v_a_2431_, lean_object* v_a_2432_){
_start:
{
lean_object* v___x_2434_; 
lean_inc(v_declName_2427_);
v___x_2434_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0(v_declName_2427_, v_a_2429_, v_a_2430_, v_a_2431_, v_a_2432_);
if (lean_obj_tag(v___x_2434_) == 0)
{
lean_object* v_a_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___f_2438_; uint8_t v___x_2439_; lean_object* v___x_2440_; 
v_a_2435_ = lean_ctor_get(v___x_2434_, 0);
lean_inc(v_a_2435_);
lean_dec_ref_known(v___x_2434_, 1);
v___x_2436_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
v___x_2437_ = l_Lean_ConstantInfo_type(v_a_2435_);
lean_dec(v_a_2435_);
lean_inc_ref(v___x_2437_);
v___f_2438_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___lam__0___boxed), 11, 4);
lean_closure_set(v___f_2438_, 0, v_declName_2427_);
lean_closure_set(v___f_2438_, 1, v_prio_2428_);
lean_closure_set(v___f_2438_, 2, v___x_2437_);
lean_closure_set(v___f_2438_, 3, v___x_2436_);
v___x_2439_ = 0;
v___x_2440_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Mathlib_Meta_FunProp_detectLambdaTheoremArgs_spec__0___redArg(v___x_2437_, v___f_2438_, v___x_2439_, v_a_2429_, v_a_2430_, v_a_2431_, v_a_2432_);
return v___x_2440_;
}
else
{
lean_object* v_a_2441_; lean_object* v___x_2443_; uint8_t v_isShared_2444_; uint8_t v_isSharedCheck_2448_; 
lean_dec(v_prio_2428_);
lean_dec(v_declName_2427_);
v_a_2441_ = lean_ctor_get(v___x_2434_, 0);
v_isSharedCheck_2448_ = !lean_is_exclusive(v___x_2434_);
if (v_isSharedCheck_2448_ == 0)
{
v___x_2443_ = v___x_2434_;
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
else
{
lean_inc(v_a_2441_);
lean_dec(v___x_2434_);
v___x_2443_ = lean_box(0);
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
v_resetjp_2442_:
{
lean_object* v___x_2446_; 
if (v_isShared_2444_ == 0)
{
v___x_2446_ = v___x_2443_;
goto v_reusejp_2445_;
}
else
{
lean_object* v_reuseFailAlloc_2447_; 
v_reuseFailAlloc_2447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2447_, 0, v_a_2441_);
v___x_2446_ = v_reuseFailAlloc_2447_;
goto v_reusejp_2445_;
}
v_reusejp_2445_:
{
return v___x_2446_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst___boxed(lean_object* v_declName_2449_, lean_object* v_prio_2450_, lean_object* v_a_2451_, lean_object* v_a_2452_, lean_object* v_a_2453_, lean_object* v_a_2454_, lean_object* v_a_2455_){
_start:
{
lean_object* v_res_2456_; 
v_res_2456_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst(v_declName_2449_, v_prio_2450_, v_a_2451_, v_a_2452_, v_a_2453_, v_a_2454_);
lean_dec(v_a_2454_);
lean_dec_ref(v_a_2453_);
lean_dec(v_a_2452_);
lean_dec_ref(v_a_2451_);
return v_res_2456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1(lean_object* v_00_u03b1_2457_, lean_object* v_msg_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_){
_start:
{
lean_object* v___x_2464_; 
v___x_2464_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___redArg(v_msg_2458_, v___y_2459_, v___y_2460_, v___y_2461_, v___y_2462_);
return v___x_2464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1___boxed(lean_object* v_00_u03b1_2465_, lean_object* v_msg_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_, lean_object* v___y_2470_, lean_object* v___y_2471_){
_start:
{
lean_object* v_res_2472_; 
v_res_2472_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1(v_00_u03b1_2465_, v_msg_2466_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_);
lean_dec(v___y_2470_);
lean_dec_ref(v___y_2469_);
lean_dec(v___y_2468_);
lean_dec_ref(v___y_2467_);
return v_res_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0(lean_object* v_00_u03b1_2473_, lean_object* v_constName_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_){
_start:
{
lean_object* v___x_2480_; 
v___x_2480_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___redArg(v_constName_2474_, v___y_2475_, v___y_2476_, v___y_2477_, v___y_2478_);
return v___x_2480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2481_, lean_object* v_constName_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_){
_start:
{
lean_object* v_res_2488_; 
v_res_2488_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0(v_00_u03b1_2481_, v_constName_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_);
lean_dec(v___y_2486_);
lean_dec_ref(v___y_2485_);
lean_dec(v___y_2484_);
lean_dec_ref(v___y_2483_);
return v_res_2488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_2489_, lean_object* v_ref_2490_, lean_object* v_constName_2491_, lean_object* v___y_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_){
_start:
{
lean_object* v___x_2497_; 
v___x_2497_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___redArg(v_ref_2490_, v_constName_2491_, v___y_2492_, v___y_2493_, v___y_2494_, v___y_2495_);
return v___x_2497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_2498_, lean_object* v_ref_2499_, lean_object* v_constName_2500_, lean_object* v___y_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_){
_start:
{
lean_object* v_res_2506_; 
v_res_2506_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1(v_00_u03b1_2498_, v_ref_2499_, v_constName_2500_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_);
lean_dec(v___y_2504_);
lean_dec_ref(v___y_2503_);
lean_dec(v___y_2502_);
lean_dec_ref(v___y_2501_);
lean_dec(v_ref_2499_);
return v_res_2506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4(lean_object* v_00_u03b1_2507_, lean_object* v_ref_2508_, lean_object* v_msg_2509_, lean_object* v_declHint_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_){
_start:
{
lean_object* v___x_2516_; 
v___x_2516_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___redArg(v_ref_2508_, v_msg_2509_, v_declHint_2510_, v___y_2511_, v___y_2512_, v___y_2513_, v___y_2514_);
return v___x_2516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_00_u03b1_2517_, lean_object* v_ref_2518_, lean_object* v_msg_2519_, lean_object* v_declHint_2520_, lean_object* v___y_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_){
_start:
{
lean_object* v_res_2526_; 
v_res_2526_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4(v_00_u03b1_2517_, v_ref_2518_, v_msg_2519_, v_declHint_2520_, v___y_2521_, v___y_2522_, v___y_2523_, v___y_2524_);
lean_dec(v___y_2524_);
lean_dec_ref(v___y_2523_);
lean_dec(v___y_2522_);
lean_dec_ref(v___y_2521_);
lean_dec(v_ref_2518_);
return v_res_2526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6(lean_object* v_msg_2527_, lean_object* v_declHint_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_){
_start:
{
lean_object* v___x_2534_; 
v___x_2534_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___redArg(v_msg_2527_, v_declHint_2528_, v___y_2532_);
return v___x_2534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6___boxed(lean_object* v_msg_2535_, lean_object* v_declHint_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_){
_start:
{
lean_object* v_res_2542_; 
v_res_2542_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__5_spec__6(v_msg_2535_, v_declHint_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_);
lean_dec(v___y_2540_);
lean_dec_ref(v___y_2539_);
lean_dec(v___y_2538_);
lean_dec_ref(v___y_2537_);
return v_res_2542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6(lean_object* v_00_u03b1_2543_, lean_object* v_ref_2544_, lean_object* v_msg_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_){
_start:
{
lean_object* v___x_2551_; 
v___x_2551_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_ref_2544_, v_msg_2545_, v___y_2546_, v___y_2547_, v___y_2548_, v___y_2549_);
return v___x_2551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6___boxed(lean_object* v_00_u03b1_2552_, lean_object* v_ref_2553_, lean_object* v_msg_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_, lean_object* v___y_2559_){
_start:
{
lean_object* v_res_2560_; 
v_res_2560_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__0_spec__0_spec__1_spec__4_spec__6(v_00_u03b1_2552_, v_ref_2553_, v_msg_2554_, v___y_2555_, v___y_2556_, v___y_2557_, v___y_2558_);
lean_dec(v___y_2558_);
lean_dec_ref(v___y_2557_);
lean_dec(v___y_2556_);
lean_dec_ref(v___y_2555_);
lean_dec(v_ref_2553_);
return v_res_2560_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2561_; 
v___x_2561_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2561_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_2562_; lean_object* v___x_2563_; 
v___x_2562_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__0);
v___x_2563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2563_, 0, v___x_2562_);
return v___x_2563_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_2564_; lean_object* v___x_2565_; 
v___x_2564_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1);
v___x_2565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2565_, 0, v___x_2564_);
lean_ctor_set(v___x_2565_, 1, v___x_2564_);
return v___x_2565_;
}
}
static lean_object* _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_2566_; lean_object* v___x_2567_; 
v___x_2566_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__1);
v___x_2567_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2567_, 0, v___x_2566_);
lean_ctor_set(v___x_2567_, 1, v___x_2566_);
lean_ctor_set(v___x_2567_, 2, v___x_2566_);
lean_ctor_set(v___x_2567_, 3, v___x_2566_);
lean_ctor_set(v___x_2567_, 4, v___x_2566_);
lean_ctor_set(v___x_2567_, 5, v___x_2566_);
return v___x_2567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(lean_object* v_ext_2568_, lean_object* v_b_2569_, uint8_t v_kind_2570_, lean_object* v___y_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_){
_start:
{
lean_object* v_currNamespace_2575_; lean_object* v___x_2576_; lean_object* v_env_2577_; lean_object* v_nextMacroScope_2578_; lean_object* v_ngen_2579_; lean_object* v_auxDeclNGen_2580_; lean_object* v_traceState_2581_; lean_object* v_messages_2582_; lean_object* v_infoState_2583_; lean_object* v_snapshotTasks_2584_; lean_object* v___x_2586_; uint8_t v_isShared_2587_; uint8_t v_isSharedCheck_2611_; 
v_currNamespace_2575_ = lean_ctor_get(v___y_2572_, 6);
v___x_2576_ = lean_st_ref_take(v___y_2573_);
v_env_2577_ = lean_ctor_get(v___x_2576_, 0);
v_nextMacroScope_2578_ = lean_ctor_get(v___x_2576_, 1);
v_ngen_2579_ = lean_ctor_get(v___x_2576_, 2);
v_auxDeclNGen_2580_ = lean_ctor_get(v___x_2576_, 3);
v_traceState_2581_ = lean_ctor_get(v___x_2576_, 4);
v_messages_2582_ = lean_ctor_get(v___x_2576_, 6);
v_infoState_2583_ = lean_ctor_get(v___x_2576_, 7);
v_snapshotTasks_2584_ = lean_ctor_get(v___x_2576_, 8);
v_isSharedCheck_2611_ = !lean_is_exclusive(v___x_2576_);
if (v_isSharedCheck_2611_ == 0)
{
lean_object* v_unused_2612_; 
v_unused_2612_ = lean_ctor_get(v___x_2576_, 5);
lean_dec(v_unused_2612_);
v___x_2586_ = v___x_2576_;
v_isShared_2587_ = v_isSharedCheck_2611_;
goto v_resetjp_2585_;
}
else
{
lean_inc(v_snapshotTasks_2584_);
lean_inc(v_infoState_2583_);
lean_inc(v_messages_2582_);
lean_inc(v_traceState_2581_);
lean_inc(v_auxDeclNGen_2580_);
lean_inc(v_ngen_2579_);
lean_inc(v_nextMacroScope_2578_);
lean_inc(v_env_2577_);
lean_dec(v___x_2576_);
v___x_2586_ = lean_box(0);
v_isShared_2587_ = v_isSharedCheck_2611_;
goto v_resetjp_2585_;
}
v_resetjp_2585_:
{
lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2591_; 
lean_inc(v_currNamespace_2575_);
v___x_2588_ = l_Lean_ScopedEnvExtension_addCore___redArg(v_env_2577_, v_ext_2568_, v_b_2569_, v_kind_2570_, v_currNamespace_2575_);
v___x_2589_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__2);
if (v_isShared_2587_ == 0)
{
lean_ctor_set(v___x_2586_, 5, v___x_2589_);
lean_ctor_set(v___x_2586_, 0, v___x_2588_);
v___x_2591_ = v___x_2586_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v___x_2588_);
lean_ctor_set(v_reuseFailAlloc_2610_, 1, v_nextMacroScope_2578_);
lean_ctor_set(v_reuseFailAlloc_2610_, 2, v_ngen_2579_);
lean_ctor_set(v_reuseFailAlloc_2610_, 3, v_auxDeclNGen_2580_);
lean_ctor_set(v_reuseFailAlloc_2610_, 4, v_traceState_2581_);
lean_ctor_set(v_reuseFailAlloc_2610_, 5, v___x_2589_);
lean_ctor_set(v_reuseFailAlloc_2610_, 6, v_messages_2582_);
lean_ctor_set(v_reuseFailAlloc_2610_, 7, v_infoState_2583_);
lean_ctor_set(v_reuseFailAlloc_2610_, 8, v_snapshotTasks_2584_);
v___x_2591_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v_mctx_2594_; lean_object* v_zetaDeltaFVarIds_2595_; lean_object* v_postponed_2596_; lean_object* v_diag_2597_; lean_object* v___x_2599_; uint8_t v_isShared_2600_; uint8_t v_isSharedCheck_2608_; 
v___x_2592_ = lean_st_ref_set(v___y_2573_, v___x_2591_);
v___x_2593_ = lean_st_ref_take(v___y_2571_);
v_mctx_2594_ = lean_ctor_get(v___x_2593_, 0);
v_zetaDeltaFVarIds_2595_ = lean_ctor_get(v___x_2593_, 2);
v_postponed_2596_ = lean_ctor_get(v___x_2593_, 3);
v_diag_2597_ = lean_ctor_get(v___x_2593_, 4);
v_isSharedCheck_2608_ = !lean_is_exclusive(v___x_2593_);
if (v_isSharedCheck_2608_ == 0)
{
lean_object* v_unused_2609_; 
v_unused_2609_ = lean_ctor_get(v___x_2593_, 1);
lean_dec(v_unused_2609_);
v___x_2599_ = v___x_2593_;
v_isShared_2600_ = v_isSharedCheck_2608_;
goto v_resetjp_2598_;
}
else
{
lean_inc(v_diag_2597_);
lean_inc(v_postponed_2596_);
lean_inc(v_zetaDeltaFVarIds_2595_);
lean_inc(v_mctx_2594_);
lean_dec(v___x_2593_);
v___x_2599_ = lean_box(0);
v_isShared_2600_ = v_isSharedCheck_2608_;
goto v_resetjp_2598_;
}
v_resetjp_2598_:
{
lean_object* v___x_2601_; lean_object* v___x_2603_; 
v___x_2601_ = lean_obj_once(&lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3, &lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___closed__3);
if (v_isShared_2600_ == 0)
{
lean_ctor_set(v___x_2599_, 1, v___x_2601_);
v___x_2603_ = v___x_2599_;
goto v_reusejp_2602_;
}
else
{
lean_object* v_reuseFailAlloc_2607_; 
v_reuseFailAlloc_2607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2607_, 0, v_mctx_2594_);
lean_ctor_set(v_reuseFailAlloc_2607_, 1, v___x_2601_);
lean_ctor_set(v_reuseFailAlloc_2607_, 2, v_zetaDeltaFVarIds_2595_);
lean_ctor_set(v_reuseFailAlloc_2607_, 3, v_postponed_2596_);
lean_ctor_set(v_reuseFailAlloc_2607_, 4, v_diag_2597_);
v___x_2603_ = v_reuseFailAlloc_2607_;
goto v_reusejp_2602_;
}
v_reusejp_2602_:
{
lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; 
v___x_2604_ = lean_st_ref_set(v___y_2571_, v___x_2603_);
v___x_2605_ = lean_box(0);
v___x_2606_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2606_, 0, v___x_2605_);
return v___x_2606_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg___boxed(lean_object* v_ext_2613_, lean_object* v_b_2614_, lean_object* v_kind_2615_, lean_object* v___y_2616_, lean_object* v___y_2617_, lean_object* v___y_2618_, lean_object* v___y_2619_){
_start:
{
uint8_t v_kind_boxed_2620_; lean_object* v_res_2621_; 
v_kind_boxed_2620_ = lean_unbox(v_kind_2615_);
v_res_2621_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v_ext_2613_, v_b_2614_, v_kind_boxed_2620_, v___y_2616_, v___y_2617_, v___y_2618_);
lean_dec(v___y_2618_);
lean_dec_ref(v___y_2617_);
lean_dec(v___y_2616_);
return v_res_2621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0(lean_object* v_00_u03b1_2622_, lean_object* v_00_u03b2_2623_, lean_object* v_00_u03c3_2624_, lean_object* v_ext_2625_, lean_object* v_b_2626_, uint8_t v_kind_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
lean_object* v___x_2633_; 
v___x_2633_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v_ext_2625_, v_b_2626_, v_kind_2627_, v___y_2629_, v___y_2630_, v___y_2631_);
return v___x_2633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___boxed(lean_object* v_00_u03b1_2634_, lean_object* v_00_u03b2_2635_, lean_object* v_00_u03c3_2636_, lean_object* v_ext_2637_, lean_object* v_b_2638_, lean_object* v_kind_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_){
_start:
{
uint8_t v_kind_boxed_2645_; lean_object* v_res_2646_; 
v_kind_boxed_2645_ = lean_unbox(v_kind_2639_);
v_res_2646_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0(v_00_u03b1_2634_, v_00_u03b2_2635_, v_00_u03c3_2636_, v_ext_2637_, v_b_2638_, v_kind_boxed_2645_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_);
lean_dec(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v___y_2641_);
lean_dec_ref(v___y_2640_);
return v_res_2646_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0(void){
_start:
{
lean_object* v___x_2647_; double v___x_2648_; 
v___x_2647_ = lean_unsigned_to_nat(0u);
v___x_2648_ = lean_float_of_nat(v___x_2647_);
return v___x_2648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(lean_object* v_cls_2652_, lean_object* v_msg_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_){
_start:
{
lean_object* v_ref_2659_; lean_object* v___x_2660_; lean_object* v_a_2661_; lean_object* v___x_2663_; uint8_t v_isShared_2664_; uint8_t v_isSharedCheck_2705_; 
v_ref_2659_ = lean_ctor_get(v___y_2656_, 5);
v___x_2660_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getTheoremFromConst_spec__1_spec__2(v_msg_2653_, v___y_2654_, v___y_2655_, v___y_2656_, v___y_2657_);
v_a_2661_ = lean_ctor_get(v___x_2660_, 0);
v_isSharedCheck_2705_ = !lean_is_exclusive(v___x_2660_);
if (v_isSharedCheck_2705_ == 0)
{
v___x_2663_ = v___x_2660_;
v_isShared_2664_ = v_isSharedCheck_2705_;
goto v_resetjp_2662_;
}
else
{
lean_inc(v_a_2661_);
lean_dec(v___x_2660_);
v___x_2663_ = lean_box(0);
v_isShared_2664_ = v_isSharedCheck_2705_;
goto v_resetjp_2662_;
}
v_resetjp_2662_:
{
lean_object* v___x_2665_; lean_object* v_traceState_2666_; lean_object* v_env_2667_; lean_object* v_nextMacroScope_2668_; lean_object* v_ngen_2669_; lean_object* v_auxDeclNGen_2670_; lean_object* v_cache_2671_; lean_object* v_messages_2672_; lean_object* v_infoState_2673_; lean_object* v_snapshotTasks_2674_; lean_object* v___x_2676_; uint8_t v_isShared_2677_; uint8_t v_isSharedCheck_2704_; 
v___x_2665_ = lean_st_ref_take(v___y_2657_);
v_traceState_2666_ = lean_ctor_get(v___x_2665_, 4);
v_env_2667_ = lean_ctor_get(v___x_2665_, 0);
v_nextMacroScope_2668_ = lean_ctor_get(v___x_2665_, 1);
v_ngen_2669_ = lean_ctor_get(v___x_2665_, 2);
v_auxDeclNGen_2670_ = lean_ctor_get(v___x_2665_, 3);
v_cache_2671_ = lean_ctor_get(v___x_2665_, 5);
v_messages_2672_ = lean_ctor_get(v___x_2665_, 6);
v_infoState_2673_ = lean_ctor_get(v___x_2665_, 7);
v_snapshotTasks_2674_ = lean_ctor_get(v___x_2665_, 8);
v_isSharedCheck_2704_ = !lean_is_exclusive(v___x_2665_);
if (v_isSharedCheck_2704_ == 0)
{
v___x_2676_ = v___x_2665_;
v_isShared_2677_ = v_isSharedCheck_2704_;
goto v_resetjp_2675_;
}
else
{
lean_inc(v_snapshotTasks_2674_);
lean_inc(v_infoState_2673_);
lean_inc(v_messages_2672_);
lean_inc(v_cache_2671_);
lean_inc(v_traceState_2666_);
lean_inc(v_auxDeclNGen_2670_);
lean_inc(v_ngen_2669_);
lean_inc(v_nextMacroScope_2668_);
lean_inc(v_env_2667_);
lean_dec(v___x_2665_);
v___x_2676_ = lean_box(0);
v_isShared_2677_ = v_isSharedCheck_2704_;
goto v_resetjp_2675_;
}
v_resetjp_2675_:
{
uint64_t v_tid_2678_; lean_object* v_traces_2679_; lean_object* v___x_2681_; uint8_t v_isShared_2682_; uint8_t v_isSharedCheck_2703_; 
v_tid_2678_ = lean_ctor_get_uint64(v_traceState_2666_, sizeof(void*)*1);
v_traces_2679_ = lean_ctor_get(v_traceState_2666_, 0);
v_isSharedCheck_2703_ = !lean_is_exclusive(v_traceState_2666_);
if (v_isSharedCheck_2703_ == 0)
{
v___x_2681_ = v_traceState_2666_;
v_isShared_2682_ = v_isSharedCheck_2703_;
goto v_resetjp_2680_;
}
else
{
lean_inc(v_traces_2679_);
lean_dec(v_traceState_2666_);
v___x_2681_ = lean_box(0);
v_isShared_2682_ = v_isSharedCheck_2703_;
goto v_resetjp_2680_;
}
v_resetjp_2680_:
{
lean_object* v___x_2683_; double v___x_2684_; uint8_t v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2693_; 
v___x_2683_ = lean_box(0);
v___x_2684_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__0);
v___x_2685_ = 0;
v___x_2686_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__1));
v___x_2687_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2687_, 0, v_cls_2652_);
lean_ctor_set(v___x_2687_, 1, v___x_2683_);
lean_ctor_set(v___x_2687_, 2, v___x_2686_);
lean_ctor_set_float(v___x_2687_, sizeof(void*)*3, v___x_2684_);
lean_ctor_set_float(v___x_2687_, sizeof(void*)*3 + 8, v___x_2684_);
lean_ctor_set_uint8(v___x_2687_, sizeof(void*)*3 + 16, v___x_2685_);
v___x_2688_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___closed__2));
v___x_2689_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2689_, 0, v___x_2687_);
lean_ctor_set(v___x_2689_, 1, v_a_2661_);
lean_ctor_set(v___x_2689_, 2, v___x_2688_);
lean_inc(v_ref_2659_);
v___x_2690_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2690_, 0, v_ref_2659_);
lean_ctor_set(v___x_2690_, 1, v___x_2689_);
v___x_2691_ = l_Lean_PersistentArray_push___redArg(v_traces_2679_, v___x_2690_);
if (v_isShared_2682_ == 0)
{
lean_ctor_set(v___x_2681_, 0, v___x_2691_);
v___x_2693_ = v___x_2681_;
goto v_reusejp_2692_;
}
else
{
lean_object* v_reuseFailAlloc_2702_; 
v_reuseFailAlloc_2702_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2702_, 0, v___x_2691_);
lean_ctor_set_uint64(v_reuseFailAlloc_2702_, sizeof(void*)*1, v_tid_2678_);
v___x_2693_ = v_reuseFailAlloc_2702_;
goto v_reusejp_2692_;
}
v_reusejp_2692_:
{
lean_object* v___x_2695_; 
if (v_isShared_2677_ == 0)
{
lean_ctor_set(v___x_2676_, 4, v___x_2693_);
v___x_2695_ = v___x_2676_;
goto v_reusejp_2694_;
}
else
{
lean_object* v_reuseFailAlloc_2701_; 
v_reuseFailAlloc_2701_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2701_, 0, v_env_2667_);
lean_ctor_set(v_reuseFailAlloc_2701_, 1, v_nextMacroScope_2668_);
lean_ctor_set(v_reuseFailAlloc_2701_, 2, v_ngen_2669_);
lean_ctor_set(v_reuseFailAlloc_2701_, 3, v_auxDeclNGen_2670_);
lean_ctor_set(v_reuseFailAlloc_2701_, 4, v___x_2693_);
lean_ctor_set(v_reuseFailAlloc_2701_, 5, v_cache_2671_);
lean_ctor_set(v_reuseFailAlloc_2701_, 6, v_messages_2672_);
lean_ctor_set(v_reuseFailAlloc_2701_, 7, v_infoState_2673_);
lean_ctor_set(v_reuseFailAlloc_2701_, 8, v_snapshotTasks_2674_);
v___x_2695_ = v_reuseFailAlloc_2701_;
goto v_reusejp_2694_;
}
v_reusejp_2694_:
{
lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2699_; 
v___x_2696_ = lean_st_ref_set(v___y_2657_, v___x_2695_);
v___x_2697_ = lean_box(0);
if (v_isShared_2664_ == 0)
{
lean_ctor_set(v___x_2663_, 0, v___x_2697_);
v___x_2699_ = v___x_2663_;
goto v_reusejp_2698_;
}
else
{
lean_object* v_reuseFailAlloc_2700_; 
v_reuseFailAlloc_2700_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2700_, 0, v___x_2697_);
v___x_2699_ = v_reuseFailAlloc_2700_;
goto v_reusejp_2698_;
}
v_reusejp_2698_:
{
return v___x_2699_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1___boxed(lean_object* v_cls_2706_, lean_object* v_msg_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_){
_start:
{
lean_object* v_res_2713_; 
v_res_2713_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(v_cls_2706_, v_msg_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_);
lean_dec(v___y_2711_);
lean_dec_ref(v___y_2710_);
lean_dec(v___y_2709_);
lean_dec_ref(v___y_2708_);
return v_res_2713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp_addTheorem_spec__2(lean_object* v_a_2714_, lean_object* v_a_2715_){
_start:
{
if (lean_obj_tag(v_a_2714_) == 0)
{
lean_object* v___x_2716_; 
v___x_2716_ = l_List_reverse___redArg(v_a_2715_);
return v___x_2716_;
}
else
{
lean_object* v_head_2717_; lean_object* v_tail_2718_; lean_object* v___x_2720_; uint8_t v_isShared_2721_; uint8_t v_isSharedCheck_2729_; 
v_head_2717_ = lean_ctor_get(v_a_2714_, 0);
v_tail_2718_ = lean_ctor_get(v_a_2714_, 1);
v_isSharedCheck_2729_ = !lean_is_exclusive(v_a_2714_);
if (v_isSharedCheck_2729_ == 0)
{
v___x_2720_ = v_a_2714_;
v_isShared_2721_ = v_isSharedCheck_2729_;
goto v_resetjp_2719_;
}
else
{
lean_inc(v_tail_2718_);
lean_inc(v_head_2717_);
lean_dec(v_a_2714_);
v___x_2720_ = lean_box(0);
v_isShared_2721_ = v_isSharedCheck_2729_;
goto v_resetjp_2719_;
}
v_resetjp_2719_:
{
lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2726_; 
v___x_2722_ = l_Nat_reprFast(v_head_2717_);
v___x_2723_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2723_, 0, v___x_2722_);
v___x_2724_ = l_Lean_MessageData_ofFormat(v___x_2723_);
if (v_isShared_2721_ == 0)
{
lean_ctor_set(v___x_2720_, 1, v_a_2715_);
lean_ctor_set(v___x_2720_, 0, v___x_2724_);
v___x_2726_ = v___x_2720_;
goto v_reusejp_2725_;
}
else
{
lean_object* v_reuseFailAlloc_2728_; 
v_reuseFailAlloc_2728_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2728_, 0, v___x_2724_);
lean_ctor_set(v_reuseFailAlloc_2728_, 1, v_a_2715_);
v___x_2726_ = v_reuseFailAlloc_2728_;
goto v_reusejp_2725_;
}
v_reusejp_2725_:
{
v_a_2714_ = v_tail_2718_;
v_a_2715_ = v___x_2726_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6(void){
_start:
{
lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; 
v___x_2741_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3));
v___x_2742_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__5));
v___x_2743_ = l_Lean_Name_append(v___x_2742_, v___x_2741_);
return v___x_2743_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8(void){
_start:
{
lean_object* v___x_2745_; lean_object* v___x_2746_; 
v___x_2745_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__7));
v___x_2746_ = l_Lean_stringToMessageData(v___x_2745_);
return v___x_2746_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10(void){
_start:
{
lean_object* v___x_2748_; lean_object* v___x_2749_; 
v___x_2748_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__9));
v___x_2749_ = l_Lean_stringToMessageData(v___x_2748_);
return v___x_2749_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12(void){
_start:
{
lean_object* v___x_2751_; lean_object* v___x_2752_; 
v___x_2751_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__11));
v___x_2752_ = l_Lean_stringToMessageData(v___x_2751_);
return v___x_2752_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14(void){
_start:
{
lean_object* v___x_2754_; lean_object* v___x_2755_; 
v___x_2754_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__13));
v___x_2755_ = l_Lean_stringToMessageData(v___x_2754_);
return v___x_2755_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16(void){
_start:
{
lean_object* v___x_2757_; lean_object* v___x_2758_; 
v___x_2757_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__15));
v___x_2758_ = l_Lean_stringToMessageData(v___x_2757_);
return v___x_2758_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18(void){
_start:
{
lean_object* v___x_2760_; lean_object* v___x_2761_; 
v___x_2760_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__17));
v___x_2761_ = l_Lean_stringToMessageData(v___x_2760_);
return v___x_2761_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20(void){
_start:
{
lean_object* v___x_2763_; lean_object* v___x_2764_; 
v___x_2763_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__19));
v___x_2764_ = l_Lean_stringToMessageData(v___x_2763_);
return v___x_2764_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22(void){
_start:
{
lean_object* v___x_2766_; lean_object* v___x_2767_; 
v___x_2766_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__21));
v___x_2767_ = l_Lean_stringToMessageData(v___x_2766_);
return v___x_2767_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24(void){
_start:
{
lean_object* v___x_2769_; lean_object* v___x_2770_; 
v___x_2769_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__23));
v___x_2770_ = l_Lean_stringToMessageData(v___x_2769_);
return v___x_2770_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26(void){
_start:
{
lean_object* v___x_2772_; lean_object* v___x_2773_; 
v___x_2772_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__25));
v___x_2773_ = l_Lean_stringToMessageData(v___x_2772_);
return v___x_2773_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28(void){
_start:
{
lean_object* v___x_2775_; lean_object* v___x_2776_; 
v___x_2775_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__27));
v___x_2776_ = l_Lean_stringToMessageData(v___x_2775_);
return v___x_2776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem(lean_object* v_declName_2777_, uint8_t v_attrKind_2778_, lean_object* v_prio_2779_, lean_object* v_a_2780_, lean_object* v_a_2781_, lean_object* v_a_2782_, lean_object* v_a_2783_){
_start:
{
lean_object* v___x_2785_; 
v___x_2785_ = lp_mathlib_Mathlib_Meta_FunProp_getTheoremFromConst(v_declName_2777_, v_prio_2779_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_);
if (lean_obj_tag(v___x_2785_) == 0)
{
lean_object* v_a_2786_; 
v_a_2786_ = lean_ctor_get(v___x_2785_, 0);
lean_inc(v_a_2786_);
lean_dec_ref_known(v___x_2785_, 1);
switch(lean_obj_tag(v_a_2786_))
{
case 0:
{
lean_object* v_thm_2787_; lean_object* v___y_2789_; lean_object* v___y_2790_; lean_object* v___y_2791_; lean_object* v___y_2792_; lean_object* v_options_2795_; uint8_t v_hasTrace_2796_; 
v_thm_2787_ = lean_ctor_get(v_a_2786_, 0);
lean_inc_ref(v_thm_2787_);
lean_dec_ref_known(v_a_2786_, 1);
v_options_2795_ = lean_ctor_get(v_a_2782_, 2);
v_hasTrace_2796_ = lean_ctor_get_uint8(v_options_2795_, sizeof(void*)*1);
if (v_hasTrace_2796_ == 0)
{
v___y_2789_ = v_a_2780_;
v___y_2790_ = v_a_2781_;
v___y_2791_ = v_a_2782_;
v___y_2792_ = v_a_2783_;
goto v___jp_2788_;
}
else
{
lean_object* v_inheritedTraceOptions_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; uint8_t v___x_2800_; 
v_inheritedTraceOptions_2797_ = lean_ctor_get(v_a_2782_, 13);
v___x_2798_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3));
v___x_2799_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6);
v___x_2800_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2797_, v_options_2795_, v___x_2799_);
if (v___x_2800_ == 0)
{
v___y_2789_ = v_a_2780_;
v___y_2790_ = v_a_2781_;
v___y_2791_ = v_a_2782_;
v___y_2792_ = v_a_2783_;
goto v___jp_2788_;
}
else
{
lean_object* v_funPropName_2801_; lean_object* v_thmName_2802_; lean_object* v_thmArgs_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; uint8_t v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; 
v_funPropName_2801_ = lean_ctor_get(v_thm_2787_, 0);
v_thmName_2802_ = lean_ctor_get(v_thm_2787_, 1);
v_thmArgs_2803_ = lean_ctor_get(v_thm_2787_, 2);
v___x_2804_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__8);
lean_inc(v_thmName_2802_);
v___x_2805_ = l_Lean_MessageData_ofName(v_thmName_2802_);
v___x_2806_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2806_, 0, v___x_2804_);
lean_ctor_set(v___x_2806_, 1, v___x_2805_);
v___x_2807_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10);
v___x_2808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2808_, 0, v___x_2806_);
lean_ctor_set(v___x_2808_, 1, v___x_2807_);
lean_inc(v_funPropName_2801_);
v___x_2809_ = l_Lean_MessageData_ofName(v_funPropName_2801_);
v___x_2810_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2810_, 0, v___x_2808_);
lean_ctor_set(v___x_2810_, 1, v___x_2809_);
v___x_2811_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__12);
v___x_2812_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2812_, 0, v___x_2810_);
lean_ctor_set(v___x_2812_, 1, v___x_2811_);
v___x_2813_ = lp_mathlib_Mathlib_Meta_FunProp_LambdaTheoremArgs_type(v_thmArgs_2803_);
v___x_2814_ = lean_unsigned_to_nat(0u);
v___x_2815_ = lp_mathlib_Mathlib_Meta_FunProp_instReprLambdaTheoremType_repr(v___x_2813_, v___x_2814_);
v___x_2816_ = l_Lean_MessageData_ofFormat(v___x_2815_);
v___x_2817_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2817_, 0, v___x_2812_);
lean_ctor_set(v___x_2817_, 1, v___x_2816_);
v___x_2818_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(v___x_2798_, v___x_2817_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_);
if (lean_obj_tag(v___x_2818_) == 0)
{
lean_dec_ref_known(v___x_2818_, 1);
v___y_2789_ = v_a_2780_;
v___y_2790_ = v_a_2781_;
v___y_2791_ = v_a_2782_;
v___y_2792_ = v_a_2783_;
goto v___jp_2788_;
}
else
{
lean_dec_ref(v_thm_2787_);
return v___x_2818_;
}
}
}
v___jp_2788_:
{
lean_object* v___x_2793_; lean_object* v___x_2794_; 
v___x_2793_ = lp_mathlib_Mathlib_Meta_FunProp_lambdaTheoremsExt;
v___x_2794_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v___x_2793_, v_thm_2787_, v_attrKind_2778_, v___y_2790_, v___y_2791_, v___y_2792_);
return v___x_2794_;
}
}
case 1:
{
lean_object* v_thm_2819_; lean_object* v___x_2821_; uint8_t v_isShared_2822_; uint8_t v_isSharedCheck_2881_; 
v_thm_2819_ = lean_ctor_get(v_a_2786_, 0);
v_isSharedCheck_2881_ = !lean_is_exclusive(v_a_2786_);
if (v_isSharedCheck_2881_ == 0)
{
v___x_2821_ = v_a_2786_;
v_isShared_2822_ = v_isSharedCheck_2881_;
goto v_resetjp_2820_;
}
else
{
lean_inc(v_thm_2819_);
lean_dec(v_a_2786_);
v___x_2821_ = lean_box(0);
v_isShared_2822_ = v_isSharedCheck_2881_;
goto v_resetjp_2820_;
}
v_resetjp_2820_:
{
lean_object* v___y_2824_; lean_object* v___y_2825_; lean_object* v___y_2826_; lean_object* v___y_2827_; lean_object* v_options_2830_; uint8_t v_hasTrace_2831_; 
v_options_2830_ = lean_ctor_get(v_a_2782_, 2);
v_hasTrace_2831_ = lean_ctor_get_uint8(v_options_2830_, sizeof(void*)*1);
if (v_hasTrace_2831_ == 0)
{
lean_del_object(v___x_2821_);
v___y_2824_ = v_a_2780_;
v___y_2825_ = v_a_2781_;
v___y_2826_ = v_a_2782_;
v___y_2827_ = v_a_2783_;
goto v___jp_2823_;
}
else
{
lean_object* v_inheritedTraceOptions_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; uint8_t v___x_2835_; 
v_inheritedTraceOptions_2832_ = lean_ctor_get(v_a_2782_, 13);
v___x_2833_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3));
v___x_2834_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6);
v___x_2835_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2832_, v_options_2830_, v___x_2834_);
if (v___x_2835_ == 0)
{
lean_del_object(v___x_2821_);
v___y_2824_ = v_a_2780_;
v___y_2825_ = v_a_2781_;
v___y_2826_ = v_a_2782_;
v___y_2827_ = v_a_2783_;
goto v___jp_2823_;
}
else
{
lean_object* v_funPropName_2836_; lean_object* v_thmOrigin_2837_; lean_object* v_funOrigin_2838_; lean_object* v_mainArgs_2839_; lean_object* v_appliedArgs_2840_; uint8_t v_form_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2866_; 
v_funPropName_2836_ = lean_ctor_get(v_thm_2819_, 0);
v_thmOrigin_2837_ = lean_ctor_get(v_thm_2819_, 1);
v_funOrigin_2838_ = lean_ctor_get(v_thm_2819_, 2);
v_mainArgs_2839_ = lean_ctor_get(v_thm_2819_, 3);
v_appliedArgs_2840_ = lean_ctor_get(v_thm_2819_, 4);
v_form_2841_ = lean_ctor_get_uint8(v_thm_2819_, sizeof(void*)*6);
v___x_2842_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__14);
v___x_2843_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_name(v_thmOrigin_2837_);
v___x_2844_ = l_Lean_MessageData_ofName(v___x_2843_);
v___x_2845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2845_, 0, v___x_2842_);
lean_ctor_set(v___x_2845_, 1, v___x_2844_);
v___x_2846_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10);
v___x_2847_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2847_, 0, v___x_2845_);
lean_ctor_set(v___x_2847_, 1, v___x_2846_);
lean_inc(v_funPropName_2836_);
v___x_2848_ = l_Lean_MessageData_ofName(v_funPropName_2836_);
v___x_2849_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2849_, 0, v___x_2847_);
lean_ctor_set(v___x_2849_, 1, v___x_2848_);
v___x_2850_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__16);
v___x_2851_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2851_, 0, v___x_2849_);
lean_ctor_set(v___x_2851_, 1, v___x_2850_);
v___x_2852_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_name(v_funOrigin_2838_);
v___x_2853_ = l_Lean_MessageData_ofName(v___x_2852_);
v___x_2854_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2854_, 0, v___x_2851_);
lean_ctor_set(v___x_2854_, 1, v___x_2853_);
v___x_2855_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__18);
v___x_2856_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2856_, 0, v___x_2854_);
lean_ctor_set(v___x_2856_, 1, v___x_2855_);
lean_inc_ref(v_mainArgs_2839_);
v___x_2857_ = lean_array_to_list(v_mainArgs_2839_);
v___x_2858_ = lean_box(0);
v___x_2859_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp_addTheorem_spec__2(v___x_2857_, v___x_2858_);
v___x_2860_ = l_Lean_MessageData_ofList(v___x_2859_);
v___x_2861_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2861_, 0, v___x_2856_);
lean_ctor_set(v___x_2861_, 1, v___x_2860_);
v___x_2862_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__20);
v___x_2863_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2863_, 0, v___x_2861_);
lean_ctor_set(v___x_2863_, 1, v___x_2862_);
lean_inc(v_appliedArgs_2840_);
v___x_2864_ = l_Nat_reprFast(v_appliedArgs_2840_);
if (v_isShared_2822_ == 0)
{
lean_ctor_set_tag(v___x_2821_, 3);
lean_ctor_set(v___x_2821_, 0, v___x_2864_);
v___x_2866_ = v___x_2821_;
goto v_reusejp_2865_;
}
else
{
lean_object* v_reuseFailAlloc_2880_; 
v_reuseFailAlloc_2880_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2880_, 0, v___x_2864_);
v___x_2866_ = v_reuseFailAlloc_2880_;
goto v_reusejp_2865_;
}
v_reusejp_2865_:
{
lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; lean_object* v___y_2872_; 
v___x_2867_ = l_Lean_MessageData_ofFormat(v___x_2866_);
v___x_2868_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2868_, 0, v___x_2863_);
lean_ctor_set(v___x_2868_, 1, v___x_2867_);
v___x_2869_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__22);
v___x_2870_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2870_, 0, v___x_2868_);
lean_ctor_set(v___x_2870_, 1, v___x_2869_);
if (v_form_2841_ == 0)
{
lean_object* v___x_2878_; 
v___x_2878_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__0));
v___y_2872_ = v___x_2878_;
goto v___jp_2871_;
}
else
{
lean_object* v___x_2879_; 
v___x_2879_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instToStringTheoremForm___lam__0___closed__1));
v___y_2872_ = v___x_2879_;
goto v___jp_2871_;
}
v___jp_2871_:
{
lean_object* v___x_2873_; lean_object* v___x_2874_; lean_object* v___x_2875_; lean_object* v___x_2876_; lean_object* v___x_2877_; 
lean_inc_ref(v___y_2872_);
v___x_2873_ = l_Lean_stringToMessageData(v___y_2872_);
v___x_2874_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2874_, 0, v___x_2870_);
lean_ctor_set(v___x_2874_, 1, v___x_2873_);
v___x_2875_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__24);
v___x_2876_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2876_, 0, v___x_2874_);
lean_ctor_set(v___x_2876_, 1, v___x_2875_);
v___x_2877_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(v___x_2833_, v___x_2876_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_);
if (lean_obj_tag(v___x_2877_) == 0)
{
lean_dec_ref_known(v___x_2877_, 1);
v___y_2824_ = v_a_2780_;
v___y_2825_ = v_a_2781_;
v___y_2826_ = v_a_2782_;
v___y_2827_ = v_a_2783_;
goto v___jp_2823_;
}
else
{
lean_dec_ref(v_thm_2819_);
return v___x_2877_;
}
}
}
}
}
v___jp_2823_:
{
lean_object* v___x_2828_; lean_object* v___x_2829_; 
v___x_2828_ = lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt;
v___x_2829_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v___x_2828_, v_thm_2819_, v_attrKind_2778_, v___y_2825_, v___y_2826_, v___y_2827_);
return v___x_2829_;
}
}
}
case 2:
{
lean_object* v_thm_2882_; lean_object* v___y_2884_; lean_object* v___y_2885_; lean_object* v___y_2886_; lean_object* v___y_2887_; lean_object* v_options_2890_; uint8_t v_hasTrace_2891_; 
v_thm_2882_ = lean_ctor_get(v_a_2786_, 0);
lean_inc_ref(v_thm_2882_);
lean_dec_ref_known(v_a_2786_, 1);
v_options_2890_ = lean_ctor_get(v_a_2782_, 2);
v_hasTrace_2891_ = lean_ctor_get_uint8(v_options_2890_, sizeof(void*)*1);
if (v_hasTrace_2891_ == 0)
{
v___y_2884_ = v_a_2780_;
v___y_2885_ = v_a_2781_;
v___y_2886_ = v_a_2782_;
v___y_2887_ = v_a_2783_;
goto v___jp_2883_;
}
else
{
lean_object* v_inheritedTraceOptions_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; uint8_t v___x_2895_; 
v_inheritedTraceOptions_2892_ = lean_ctor_get(v_a_2782_, 13);
v___x_2893_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3));
v___x_2894_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6);
v___x_2895_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2892_, v_options_2890_, v___x_2894_);
if (v___x_2895_ == 0)
{
v___y_2884_ = v_a_2780_;
v___y_2885_ = v_a_2781_;
v___y_2886_ = v_a_2782_;
v___y_2887_ = v_a_2783_;
goto v___jp_2883_;
}
else
{
lean_object* v_funPropName_2896_; lean_object* v_thmName_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2902_; lean_object* v___x_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; 
v_funPropName_2896_ = lean_ctor_get(v_thm_2882_, 0);
v_thmName_2897_ = lean_ctor_get(v_thm_2882_, 1);
v___x_2898_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__26);
lean_inc(v_thmName_2897_);
v___x_2899_ = l_Lean_MessageData_ofName(v_thmName_2897_);
v___x_2900_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2900_, 0, v___x_2898_);
lean_ctor_set(v___x_2900_, 1, v___x_2899_);
v___x_2901_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10);
v___x_2902_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2902_, 0, v___x_2900_);
lean_ctor_set(v___x_2902_, 1, v___x_2901_);
lean_inc(v_funPropName_2896_);
v___x_2903_ = l_Lean_MessageData_ofName(v_funPropName_2896_);
v___x_2904_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2904_, 0, v___x_2902_);
lean_ctor_set(v___x_2904_, 1, v___x_2903_);
v___x_2905_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(v___x_2893_, v___x_2904_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_);
if (lean_obj_tag(v___x_2905_) == 0)
{
lean_dec_ref_known(v___x_2905_, 1);
v___y_2884_ = v_a_2780_;
v___y_2885_ = v_a_2781_;
v___y_2886_ = v_a_2782_;
v___y_2887_ = v_a_2783_;
goto v___jp_2883_;
}
else
{
lean_dec_ref(v_thm_2882_);
return v___x_2905_;
}
}
}
v___jp_2883_:
{
lean_object* v___x_2888_; lean_object* v___x_2889_; 
v___x_2888_ = lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt;
v___x_2889_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v___x_2888_, v_thm_2882_, v_attrKind_2778_, v___y_2885_, v___y_2886_, v___y_2887_);
return v___x_2889_;
}
}
default: 
{
lean_object* v_thm_2906_; lean_object* v___y_2908_; lean_object* v___y_2909_; lean_object* v___y_2910_; lean_object* v___y_2911_; lean_object* v_options_2914_; uint8_t v_hasTrace_2915_; 
v_thm_2906_ = lean_ctor_get(v_a_2786_, 0);
lean_inc_ref(v_thm_2906_);
lean_dec_ref_known(v_a_2786_, 1);
v_options_2914_ = lean_ctor_get(v_a_2782_, 2);
v_hasTrace_2915_ = lean_ctor_get_uint8(v_options_2914_, sizeof(void*)*1);
if (v_hasTrace_2915_ == 0)
{
v___y_2908_ = v_a_2780_;
v___y_2909_ = v_a_2781_;
v___y_2910_ = v_a_2782_;
v___y_2911_ = v_a_2783_;
goto v___jp_2907_;
}
else
{
lean_object* v_inheritedTraceOptions_2916_; lean_object* v___x_2917_; lean_object* v___x_2918_; uint8_t v___x_2919_; 
v_inheritedTraceOptions_2916_ = lean_ctor_get(v_a_2782_, 13);
v___x_2917_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__3));
v___x_2918_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__6);
v___x_2919_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2916_, v_options_2914_, v___x_2918_);
if (v___x_2919_ == 0)
{
v___y_2908_ = v_a_2780_;
v___y_2909_ = v_a_2781_;
v___y_2910_ = v_a_2782_;
v___y_2911_ = v_a_2783_;
goto v___jp_2907_;
}
else
{
lean_object* v_funPropName_2920_; lean_object* v_thmName_2921_; lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; lean_object* v___x_2925_; lean_object* v___x_2926_; lean_object* v___x_2927_; lean_object* v___x_2928_; lean_object* v___x_2929_; 
v_funPropName_2920_ = lean_ctor_get(v_thm_2906_, 0);
v_thmName_2921_ = lean_ctor_get(v_thm_2906_, 1);
v___x_2922_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__28);
lean_inc(v_thmName_2921_);
v___x_2923_ = l_Lean_MessageData_ofName(v_thmName_2921_);
v___x_2924_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2924_, 0, v___x_2922_);
lean_ctor_set(v___x_2924_, 1, v___x_2923_);
v___x_2925_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10, &lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10_once, _init_lp_mathlib_Mathlib_Meta_FunProp_addTheorem___closed__10);
v___x_2926_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2926_, 0, v___x_2924_);
lean_ctor_set(v___x_2926_, 1, v___x_2925_);
lean_inc(v_funPropName_2920_);
v___x_2927_ = l_Lean_MessageData_ofName(v_funPropName_2920_);
v___x_2928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2928_, 0, v___x_2926_);
lean_ctor_set(v___x_2928_, 1, v___x_2927_);
v___x_2929_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_addTheorem_spec__1(v___x_2917_, v___x_2928_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_);
if (lean_obj_tag(v___x_2929_) == 0)
{
lean_dec_ref_known(v___x_2929_, 1);
v___y_2908_ = v_a_2780_;
v___y_2909_ = v_a_2781_;
v___y_2910_ = v_a_2782_;
v___y_2911_ = v_a_2783_;
goto v___jp_2907_;
}
else
{
lean_dec_ref(v_thm_2906_);
return v___x_2929_;
}
}
}
v___jp_2907_:
{
lean_object* v___x_2912_; lean_object* v___x_2913_; 
v___x_2912_ = lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt;
v___x_2913_ = lp_mathlib_Lean_ScopedEnvExtension_add___at___00Mathlib_Meta_FunProp_addTheorem_spec__0___redArg(v___x_2912_, v_thm_2906_, v_attrKind_2778_, v___y_2909_, v___y_2910_, v___y_2911_);
return v___x_2913_;
}
}
}
}
else
{
lean_object* v_a_2930_; lean_object* v___x_2932_; uint8_t v_isShared_2933_; uint8_t v_isSharedCheck_2937_; 
v_a_2930_ = lean_ctor_get(v___x_2785_, 0);
v_isSharedCheck_2937_ = !lean_is_exclusive(v___x_2785_);
if (v_isSharedCheck_2937_ == 0)
{
v___x_2932_ = v___x_2785_;
v_isShared_2933_ = v_isSharedCheck_2937_;
goto v_resetjp_2931_;
}
else
{
lean_inc(v_a_2930_);
lean_dec(v___x_2785_);
v___x_2932_ = lean_box(0);
v_isShared_2933_ = v_isSharedCheck_2937_;
goto v_resetjp_2931_;
}
v_resetjp_2931_:
{
lean_object* v___x_2935_; 
if (v_isShared_2933_ == 0)
{
v___x_2935_ = v___x_2932_;
goto v_reusejp_2934_;
}
else
{
lean_object* v_reuseFailAlloc_2936_; 
v_reuseFailAlloc_2936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2936_, 0, v_a_2930_);
v___x_2935_ = v_reuseFailAlloc_2936_;
goto v_reusejp_2934_;
}
v_reusejp_2934_:
{
return v___x_2935_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_addTheorem___boxed(lean_object* v_declName_2938_, lean_object* v_attrKind_2939_, lean_object* v_prio_2940_, lean_object* v_a_2941_, lean_object* v_a_2942_, lean_object* v_a_2943_, lean_object* v_a_2944_, lean_object* v_a_2945_){
_start:
{
uint8_t v_attrKind_boxed_2946_; lean_object* v_res_2947_; 
v_attrKind_boxed_2946_ = lean_unbox(v_attrKind_2939_);
v_res_2947_ = lp_mathlib_Mathlib_Meta_FunProp_addTheorem(v_declName_2938_, v_attrKind_boxed_2946_, v_prio_2940_, v_a_2941_, v_a_2942_, v_a_2943_, v_a_2944_);
lean_dec(v_a_2944_);
lean_dec_ref(v_a_2943_);
lean_dec(v_a_2942_);
lean_dec_ref(v_a_2941_);
return v_res_2947_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Decl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Decl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Decl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Decl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs_default);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremArgs);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType_default();
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheoremType();
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems_default);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedLambdaTheorems);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_2854048689____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Meta_FunProp_lambdaTheoremsExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_lambdaTheoremsExt);
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm_default();
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedTheoremForm();
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem_default);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorem);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems_default);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedFunctionTheorems);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1545051141____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_1054951927____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Theorems_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Theorems_3643713033____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Decl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Decl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Decl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Decl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_Theorems(builtin);
}
#ifdef __cplusplus
}
#endif
