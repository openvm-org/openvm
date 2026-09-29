// Lean compiler output
// Module: Batteries.Tactic.Lint.Basic
// Imports: public import Init public meta import Init public meta import Lean.Structure public meta import Lean.Elab.InfoTree.Main public meta import Lean.Elab.Exception public meta import Lean.ExtraModUses
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
uint8_t lean_has_compile_error(lean_object*, lean_object*);
lean_object* l_Lean_Environment_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
extern lean_object* l_Lean_Elab_abortCommandExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
extern lean_object* l_Lean_LocalContext_empty;
extern lean_object* l_Lean_instInhabitedEffectiveImport_default;
lean_object* l_Lean_instHashableExtraModUse_hash___boxed(lean_object*);
lean_object* l_Lean_instBEqExtraModUse_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_extraModUses;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableExtraModUse_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqExtraModUse_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Std_HashMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
extern lean_object* l_Lean_indirectModUseExt;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_sub(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_isMarkedMeta(lean_object*, lean_object*);
lean_object* l_Lean_registerParametricAttribute___redArg(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
uint8_t l_Lean_Environment_isConstructor(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_casesOnSuffix;
lean_object* l_instDecidableEqString___boxed(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_belowSuffix;
extern lean_object* l_Lean_brecOnSuffix;
extern lean_object* l_Lean_recOnSuffix;
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_isSubobjectField_x3f(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_is_reserved_name(lean_object*, lean_object*);
lean_object* l_Lean_privateToUserName(lean_object*);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
uint8_t l_Lean_Name_isInternal(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_ensureAttrDeclIsMeta(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqAttributeKind_beq(uint8_t, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadEnvCoreM;
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
extern lean_object* l_Lean_Core_instMonadRefCoreM;
extern lean_object* l_Lean_Core_instAddMessageContextCoreM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mutual"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__0 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__0_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_functor"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__1 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__1_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "functor_unfold"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__2 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__3;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ndrec"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__4 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__4_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ndrecOn"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__5 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__5_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "noConfusionType"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__6 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__6_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "noConfusion"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__7 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__7_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__8 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__8_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toCtorIdx"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__9 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__9_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ctorIdx"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__10 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__10_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ctorElim"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__11 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__11_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ctorElimType"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__12 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__12_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__13 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__13_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__11_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__13_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__14 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__14_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__10_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__14_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__15 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__15_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__9_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__15_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__16 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__16_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__8_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__16_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__17 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__17_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__7_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__17_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__18 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__18_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__6_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__18_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__19 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__19_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__5_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__19_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__20 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__20_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__4_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__20_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__21 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__21_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__22;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__23;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__24;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__25;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "below_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__26 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__26_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__27;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "brecOn_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__28 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__28_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__29;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "injEq"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__30 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__30_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inj"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__31 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__31_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "sizeOf_spec"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__32 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__32_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__33 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__33_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__34 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__34_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__33_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__34_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__35 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__35_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__32_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__35_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__36 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__36_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__31_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__36_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__37 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__37_value;
static const lean_ctor_object lp_batteries_Lean_Environment_isAutoDecl___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__30_value),((lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__37_value)}};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__38 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__38_value;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "grind_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__39 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__39_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__40;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "unsafe_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__41 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__41_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__42;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "match_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__43 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__43_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__44;
static const lean_string_object lp_batteries_Lean_Environment_isAutoDecl___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "proof_"};
static const lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__45 = (const lean_object*)&lp_batteries_Lean_Environment_isAutoDecl___closed__45_value;
static lean_once_cell_t lp_batteries_Lean_Environment_isAutoDecl___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Environment_isAutoDecl___closed__46;
LEAN_EXPORT uint8_t lp_batteries_Lean_Environment_isAutoDecl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Environment_isAutoDecl___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Environment_isPrivateOrAutoDecl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Environment_isPrivateOrAutoDecl___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__2_value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__3 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__3_value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__4 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__4_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lint"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__8 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__8_value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(213, 8, 11, 122, 51, 224, 18, 145)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getLinter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getLinter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "batteriesLinterExt"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(57, 77, 155, 37, 92, 76, 124, 33)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "env_linter"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(8, 42, 52, 200, 21, 191, 142, 24)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " disabled"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__6_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_env__linter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__11_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_env__linter = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__11_value;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Invalid attribute scope: Attribute `["};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "]` must be global, not `"};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__4 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__6_value;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__7 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__7_value;
static const lean_string_object lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "scoped"};
static const lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0;
static lean_once_cell_t lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__4 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__8_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__10 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__10_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__12 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__12_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13;
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "` must have type `"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "`, got `"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "invalid attribute `env_linter`, linter `"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "` has already been declared"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(90, 175, 18, 163, 178, 203, 59, 243)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(47, 92, 120, 195, 85, 23, 119, 138)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(211, 182, 232, 123, 14, 144, 43, 55)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(12, 131, 210, 42, 229, 209, 125, 66)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(13, 76, 235, 76, 85, 63, 198, 207)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(248, 238, 55, 58, 212, 140, 64, 133)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(229, 54, 125, 253, 110, 141, 17, 121)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(145, 178, 5, 110, 233, 85, 140, 8)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(104, 204, 177, 63, 224, 35, 91, 138)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(25, 42, 163, 239, 246, 79, 121, 109)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(252, 113, 216, 72, 242, 222, 45, 78)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(57, 192, 169, 227, 205, 75, 166, 22)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(165, 90, 131, 142, 73, 125, 214, 134)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(242, 47, 173, 146, 125, 216, 243, 127)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 171, 221, 1, 181, 103, 114, 50)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__26_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*8, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed, .m_arity = 14, .m_num_fixed = 8, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__26_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__26_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__27_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__27_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__27_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__28_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "Use this declaration as a linting test in #lint"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__28_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__28_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nolint"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value),LEAN_SCALAR_PTR_LITERAL(66, 125, 253, 49, 249, 131, 0, 220)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__3_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__5_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_env__linter___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__2_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_nolint___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__14_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_nolint = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__14_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0;
static const lean_string_object lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__1 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__1_value;
static const lean_array_object lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__2 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqExtraModUse_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__0 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__0_value;
static const lean_closure_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableExtraModUse_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__1 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__1_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__3 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__3_value;
static const lean_ctor_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__4 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__4_value;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " extra mod use "};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__5 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__5_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " of "};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__7 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__7_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__10 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__10_value;
static const lean_ctor_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__11 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__11_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recording "};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__13 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__13_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__15 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__15_value;
static lean_once_cell_t lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "regular"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__17 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__17_value;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__18 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__18_value;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__19 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__19_value;
static const lean_string_object lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__20 = (const lean_object*)&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__20_value;
LEAN_EXPORT lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__0 = (const lean_object*)&lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__0_value;
static const lean_closure_object lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__1 = (const lean_object*)&lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__1_value;
static lean_once_cell_t lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2;
static const lean_array_object lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__3 = (const lean_object*)&lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linter '"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "' not found"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "nolintAttr"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(213, 146, 108, 162, 226, 104, 221, 133)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__5_value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__6_value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value)} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_nolint___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 116, 162, 235, 23, 16, 204, 24)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Do not report this declaration in any of the tests of `#lint`"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_nolintAttr;
static const lean_array_object lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__3(void){
_start:
{
lean_object* v___x_4_; lean_object* v___f_5_; 
v___x_4_ = lean_alloc_closure((void*)(l_instDecidableEqString___boxed), 2, 0);
v___f_5_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_5_, 0, v___x_4_);
return v___f_5_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__22(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__21));
v___x_43_ = l_Lean_belowSuffix;
v___x_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___x_42_);
return v___x_44_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__23(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__22, &lp_batteries_Lean_Environment_isAutoDecl___closed__22_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__22);
v___x_46_ = l_Lean_brecOnSuffix;
v___x_47_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
return v___x_47_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__24(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__23, &lp_batteries_Lean_Environment_isAutoDecl___closed__23_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__23);
v___x_49_ = l_Lean_recOnSuffix;
v___x_50_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v___x_48_);
return v___x_50_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__25(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__24, &lp_batteries_Lean_Environment_isAutoDecl___closed__24_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__24);
v___x_52_ = l_Lean_casesOnSuffix;
v___x_53_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_53_, 0, v___x_52_);
lean_ctor_set(v___x_53_, 1, v___x_51_);
return v___x_53_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__27(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__26));
v___x_56_ = lean_string_utf8_byte_size(v___x_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__29(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__28));
v___x_59_ = lean_string_utf8_byte_size(v___x_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__40(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__39));
v___x_81_ = lean_string_utf8_byte_size(v___x_80_);
return v___x_81_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__42(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__41));
v___x_84_ = lean_string_utf8_byte_size(v___x_83_);
return v___x_84_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__44(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__43));
v___x_87_ = lean_string_utf8_byte_size(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_batteries_Lean_Environment_isAutoDecl___closed__46(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__45));
v___x_90_ = lean_string_utf8_byte_size(v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Environment_isAutoDecl(lean_object* v_env_91_, lean_object* v_decl_92_){
_start:
{
lean_object* v___y_94_; uint8_t v___y_95_; lean_object* v___y_96_; uint8_t v___y_97_; lean_object* v___y_105_; uint8_t v___y_106_; uint8_t v___y_107_; lean_object* v___y_108_; lean_object* v___y_118_; uint8_t v___y_119_; uint8_t v___y_120_; lean_object* v___y_121_; lean_object* v___y_129_; uint8_t v___y_130_; uint8_t v___y_131_; lean_object* v___y_132_; lean_object* v___y_140_; uint8_t v___y_141_; lean_object* v___y_142_; uint8_t v___y_143_; lean_object* v___y_153_; uint8_t v___y_154_; lean_object* v___y_155_; lean_object* v___y_161_; uint8_t v___y_162_; lean_object* v___y_163_; lean_object* v___y_171_; uint8_t v___y_172_; lean_object* v___y_173_; lean_object* v___y_181_; uint8_t v___y_182_; lean_object* v___y_183_; uint8_t v___y_184_; uint8_t v___y_192_; lean_object* v_declUserName_204_; uint8_t v___x_205_; 
lean_inc(v_decl_92_);
v_declUserName_204_ = l_Lean_privateToUserName(v_decl_92_);
v___x_205_ = l_Lean_Name_hasMacroScopes(v_declUserName_204_);
if (v___x_205_ == 0)
{
uint8_t v___x_206_; 
v___x_206_ = l_Lean_Name_isInternal(v_declUserName_204_);
lean_dec(v_declUserName_204_);
v___y_192_ = v___x_206_;
goto v___jp_191_;
}
else
{
lean_dec(v_declUserName_204_);
v___y_192_ = v___x_205_;
goto v___jp_191_;
}
v___jp_93_:
{
if (v___y_97_ == 0)
{
lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_98_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__0));
v___x_99_ = lean_string_dec_eq(v___y_96_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__1));
v___x_101_ = l_Lean_Name_str___override(v___y_94_, v___x_100_);
v___x_102_ = l_Lean_Name_str___override(v___x_101_, v___y_96_);
v___x_103_ = l_Lean_Environment_isConstructor(v_env_91_, v___x_102_);
if (v___x_103_ == 0)
{
return v___x_103_;
}
else
{
return v___y_95_;
}
}
else
{
lean_dec_ref(v___y_96_);
lean_dec(v___y_94_);
lean_dec_ref(v_env_91_);
return v___y_95_;
}
}
else
{
lean_dec_ref(v___y_96_);
lean_dec(v___y_94_);
lean_dec_ref(v_env_91_);
return v___y_95_;
}
}
v___jp_104_:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_109_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__1));
lean_inc(v___y_105_);
v___x_110_ = l_Lean_Name_str___override(v___y_105_, v___x_109_);
lean_inc_ref(v_env_91_);
v___x_111_ = l_Lean_Environment_find_x3f(v_env_91_, v___x_110_, v___y_106_);
if (lean_obj_tag(v___x_111_) == 1)
{
lean_object* v_val_112_; 
v_val_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_val_112_);
lean_dec_ref_known(v___x_111_, 1);
if (lean_obj_tag(v_val_112_) == 5)
{
lean_object* v___x_113_; uint8_t v___x_114_; 
lean_dec_ref_known(v_val_112_, 1);
v___x_113_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__2));
v___x_114_ = lean_string_dec_eq(v___y_108_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_115_ = l_Lean_casesOnSuffix;
v___x_116_ = lean_string_dec_eq(v___y_108_, v___x_115_);
v___y_94_ = v___y_105_;
v___y_95_ = v___y_107_;
v___y_96_ = v___y_108_;
v___y_97_ = v___x_116_;
goto v___jp_93_;
}
else
{
v___y_94_ = v___y_105_;
v___y_95_ = v___y_107_;
v___y_96_ = v___y_108_;
v___y_97_ = v___x_114_;
goto v___jp_93_;
}
}
else
{
lean_dec(v_val_112_);
lean_dec_ref(v___y_108_);
lean_dec(v___y_105_);
lean_dec_ref(v_env_91_);
return v___y_106_;
}
}
else
{
lean_dec(v___x_111_);
lean_dec_ref(v___y_108_);
lean_dec(v___y_105_);
lean_dec_ref(v_env_91_);
return v___y_106_;
}
}
v___jp_117_:
{
lean_object* v___f_122_; lean_object* v___x_123_; uint8_t v___x_124_; 
v___f_122_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__3, &lp_batteries_Lean_Environment_isAutoDecl___closed__3_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__3);
v___x_123_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__25, &lp_batteries_Lean_Environment_isAutoDecl___closed__25_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__25);
lean_inc_ref(v___y_121_);
v___x_124_ = l_List_elem___redArg(v___f_122_, v___y_121_, v___x_123_);
if (v___x_124_ == 0)
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = lean_box(0);
lean_inc_ref(v___y_121_);
v___x_126_ = l_Lean_Name_str___override(v___x_125_, v___y_121_);
lean_inc(v___y_118_);
lean_inc_ref(v_env_91_);
v___x_127_ = l_Lean_isSubobjectField_x3f(v_env_91_, v___y_118_, v___x_126_);
if (lean_obj_tag(v___x_127_) == 1)
{
lean_dec_ref_known(v___x_127_, 1);
lean_dec_ref(v___y_121_);
lean_dec(v___y_118_);
lean_dec_ref(v_env_91_);
return v___y_120_;
}
else
{
lean_dec(v___x_127_);
v___y_105_ = v___y_118_;
v___y_106_ = v___y_119_;
v___y_107_ = v___y_120_;
v___y_108_ = v___y_121_;
goto v___jp_104_;
}
}
else
{
lean_dec_ref(v___y_121_);
lean_dec(v___y_118_);
lean_dec_ref(v_env_91_);
return v___y_120_;
}
}
v___jp_128_:
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; uint8_t v___x_136_; 
v___x_133_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__26));
v___x_134_ = lean_string_utf8_byte_size(v___y_132_);
v___x_135_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__27, &lp_batteries_Lean_Environment_isAutoDecl___closed__27_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__27);
v___x_136_ = lean_nat_dec_le(v___x_135_, v___x_134_);
if (v___x_136_ == 0)
{
v___y_118_ = v___y_129_;
v___y_119_ = v___y_130_;
v___y_120_ = v___y_131_;
v___y_121_ = v___y_132_;
goto v___jp_117_;
}
else
{
lean_object* v___x_137_; uint8_t v___x_138_; 
v___x_137_ = lean_unsigned_to_nat(0u);
v___x_138_ = lean_string_memcmp(v___y_132_, v___x_133_, v___x_137_, v___x_137_, v___x_135_);
if (v___x_138_ == 0)
{
v___y_118_ = v___y_129_;
v___y_119_ = v___y_130_;
v___y_120_ = v___y_131_;
v___y_121_ = v___y_132_;
goto v___jp_117_;
}
else
{
lean_dec_ref(v___y_132_);
lean_dec(v___y_129_);
lean_dec_ref(v_env_91_);
return v___y_131_;
}
}
}
v___jp_139_:
{
if (v___y_143_ == 0)
{
lean_object* v___x_144_; 
lean_inc(v___y_140_);
lean_inc_ref(v_env_91_);
v___x_144_ = l_Lean_Environment_find_x3f(v_env_91_, v___y_140_, v___y_143_);
if (lean_obj_tag(v___x_144_) == 1)
{
lean_object* v_val_145_; 
v_val_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_val_145_);
lean_dec_ref_known(v___x_144_, 1);
if (lean_obj_tag(v_val_145_) == 5)
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; uint8_t v___x_149_; 
lean_dec_ref_known(v_val_145_, 1);
v___x_146_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__28));
v___x_147_ = lean_string_utf8_byte_size(v___y_142_);
v___x_148_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__29, &lp_batteries_Lean_Environment_isAutoDecl___closed__29_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__29);
v___x_149_ = lean_nat_dec_le(v___x_148_, v___x_147_);
if (v___x_149_ == 0)
{
v___y_129_ = v___y_140_;
v___y_130_ = v___y_143_;
v___y_131_ = v___y_141_;
v___y_132_ = v___y_142_;
goto v___jp_128_;
}
else
{
lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_150_ = lean_unsigned_to_nat(0u);
v___x_151_ = lean_string_memcmp(v___y_142_, v___x_146_, v___x_150_, v___x_150_, v___x_148_);
if (v___x_151_ == 0)
{
v___y_129_ = v___y_140_;
v___y_130_ = v___y_143_;
v___y_131_ = v___y_141_;
v___y_132_ = v___y_142_;
goto v___jp_128_;
}
else
{
lean_dec_ref(v___y_142_);
lean_dec(v___y_140_);
lean_dec_ref(v_env_91_);
return v___y_141_;
}
}
}
else
{
lean_dec(v_val_145_);
v___y_105_ = v___y_140_;
v___y_106_ = v___y_143_;
v___y_107_ = v___y_141_;
v___y_108_ = v___y_142_;
goto v___jp_104_;
}
}
else
{
lean_dec(v___x_144_);
v___y_105_ = v___y_140_;
v___y_106_ = v___y_143_;
v___y_107_ = v___y_141_;
v___y_108_ = v___y_142_;
goto v___jp_104_;
}
}
else
{
lean_dec_ref(v___y_142_);
lean_dec(v___y_140_);
lean_dec_ref(v_env_91_);
return v___y_141_;
}
}
v___jp_152_:
{
uint8_t v___x_156_; 
lean_inc(v___y_153_);
lean_inc_ref(v_env_91_);
v___x_156_ = l_Lean_Environment_isConstructor(v_env_91_, v___y_153_);
if (v___x_156_ == 0)
{
v___y_140_ = v___y_153_;
v___y_141_ = v___y_154_;
v___y_142_ = v___y_155_;
v___y_143_ = v___x_156_;
goto v___jp_139_;
}
else
{
lean_object* v___f_157_; lean_object* v___x_158_; uint8_t v___x_159_; 
v___f_157_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__3, &lp_batteries_Lean_Environment_isAutoDecl___closed__3_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__3);
v___x_158_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__38));
lean_inc_ref(v___y_155_);
v___x_159_ = l_List_elem___redArg(v___f_157_, v___y_155_, v___x_158_);
v___y_140_ = v___y_153_;
v___y_141_ = v___y_154_;
v___y_142_ = v___y_155_;
v___y_143_ = v___x_159_;
goto v___jp_139_;
}
}
v___jp_160_:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_164_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__39));
v___x_165_ = lean_string_utf8_byte_size(v___y_163_);
v___x_166_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__40, &lp_batteries_Lean_Environment_isAutoDecl___closed__40_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__40);
v___x_167_ = lean_nat_dec_le(v___x_166_, v___x_165_);
if (v___x_167_ == 0)
{
v___y_153_ = v___y_161_;
v___y_154_ = v___y_162_;
v___y_155_ = v___y_163_;
goto v___jp_152_;
}
else
{
lean_object* v___x_168_; uint8_t v___x_169_; 
v___x_168_ = lean_unsigned_to_nat(0u);
v___x_169_ = lean_string_memcmp(v___y_163_, v___x_164_, v___x_168_, v___x_168_, v___x_166_);
if (v___x_169_ == 0)
{
v___y_153_ = v___y_161_;
v___y_154_ = v___y_162_;
v___y_155_ = v___y_163_;
goto v___jp_152_;
}
else
{
lean_dec_ref(v___y_163_);
lean_dec(v___y_161_);
lean_dec_ref(v_env_91_);
return v___y_162_;
}
}
}
v___jp_170_:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; uint8_t v___x_177_; 
v___x_174_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__41));
v___x_175_ = lean_string_utf8_byte_size(v___y_173_);
v___x_176_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__42, &lp_batteries_Lean_Environment_isAutoDecl___closed__42_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__42);
v___x_177_ = lean_nat_dec_le(v___x_176_, v___x_175_);
if (v___x_177_ == 0)
{
v___y_161_ = v___y_171_;
v___y_162_ = v___y_172_;
v___y_163_ = v___y_173_;
goto v___jp_160_;
}
else
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = lean_unsigned_to_nat(0u);
v___x_179_ = lean_string_memcmp(v___y_173_, v___x_174_, v___x_178_, v___x_178_, v___x_176_);
if (v___x_179_ == 0)
{
v___y_161_ = v___y_171_;
v___y_162_ = v___y_172_;
v___y_163_ = v___y_173_;
goto v___jp_160_;
}
else
{
lean_dec_ref(v___y_173_);
lean_dec(v___y_171_);
lean_dec_ref(v_env_91_);
return v___y_172_;
}
}
}
v___jp_180_:
{
if (v___y_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; uint8_t v___x_188_; 
v___x_185_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__43));
v___x_186_ = lean_string_utf8_byte_size(v___y_183_);
v___x_187_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__44, &lp_batteries_Lean_Environment_isAutoDecl___closed__44_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__44);
v___x_188_ = lean_nat_dec_le(v___x_187_, v___x_186_);
if (v___x_188_ == 0)
{
v___y_171_ = v___y_181_;
v___y_172_ = v___y_182_;
v___y_173_ = v___y_183_;
goto v___jp_170_;
}
else
{
lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_189_ = lean_unsigned_to_nat(0u);
v___x_190_ = lean_string_memcmp(v___y_183_, v___x_185_, v___x_189_, v___x_189_, v___x_187_);
if (v___x_190_ == 0)
{
v___y_171_ = v___y_181_;
v___y_172_ = v___y_182_;
v___y_173_ = v___y_183_;
goto v___jp_170_;
}
else
{
lean_dec_ref(v___y_183_);
lean_dec(v___y_181_);
lean_dec_ref(v_env_91_);
return v___y_182_;
}
}
}
else
{
lean_dec_ref(v___y_183_);
lean_dec(v___y_181_);
lean_dec_ref(v_env_91_);
return v___y_182_;
}
}
v___jp_191_:
{
uint8_t v___x_193_; 
v___x_193_ = 1;
if (v___y_192_ == 0)
{
uint8_t v___x_194_; 
lean_inc(v_decl_92_);
lean_inc_ref(v_env_91_);
v___x_194_ = lean_is_reserved_name(v_env_91_, v_decl_92_);
if (v___x_194_ == 0)
{
if (lean_obj_tag(v_decl_92_) == 1)
{
lean_object* v_pre_195_; lean_object* v_str_196_; uint8_t v___x_197_; 
v_pre_195_ = lean_ctor_get(v_decl_92_, 0);
lean_inc_n(v_pre_195_, 2);
v_str_196_ = lean_ctor_get(v_decl_92_, 1);
lean_inc_ref(v_str_196_);
lean_dec_ref_known(v_decl_92_, 2);
lean_inc_ref(v_env_91_);
v___x_197_ = lp_batteries_Lean_Environment_isAutoDecl(v_env_91_, v_pre_195_);
if (v___x_197_ == 0)
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; uint8_t v___x_201_; 
v___x_198_ = ((lean_object*)(lp_batteries_Lean_Environment_isAutoDecl___closed__45));
v___x_199_ = lean_string_utf8_byte_size(v_str_196_);
v___x_200_ = lean_obj_once(&lp_batteries_Lean_Environment_isAutoDecl___closed__46, &lp_batteries_Lean_Environment_isAutoDecl___closed__46_once, _init_lp_batteries_Lean_Environment_isAutoDecl___closed__46);
v___x_201_ = lean_nat_dec_le(v___x_200_, v___x_199_);
if (v___x_201_ == 0)
{
v___y_181_ = v_pre_195_;
v___y_182_ = v___x_193_;
v___y_183_ = v_str_196_;
v___y_184_ = v___x_197_;
goto v___jp_180_;
}
else
{
lean_object* v___x_202_; uint8_t v___x_203_; 
v___x_202_ = lean_unsigned_to_nat(0u);
v___x_203_ = lean_string_memcmp(v_str_196_, v___x_198_, v___x_202_, v___x_202_, v___x_200_);
v___y_181_ = v_pre_195_;
v___y_182_ = v___x_193_;
v___y_183_ = v_str_196_;
v___y_184_ = v___x_203_;
goto v___jp_180_;
}
}
else
{
lean_dec_ref(v_str_196_);
lean_dec(v_pre_195_);
lean_dec_ref(v_env_91_);
return v___x_193_;
}
}
else
{
lean_dec(v_decl_92_);
lean_dec_ref(v_env_91_);
return v___x_194_;
}
}
else
{
lean_dec(v_decl_92_);
lean_dec_ref(v_env_91_);
return v___x_193_;
}
}
else
{
lean_dec(v_decl_92_);
lean_dec_ref(v_env_91_);
return v___x_193_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Environment_isAutoDecl___boxed(lean_object* v_env_207_, lean_object* v_decl_208_){
_start:
{
uint8_t v_res_209_; lean_object* v_r_210_; 
v_res_209_ = lp_batteries_Lean_Environment_isAutoDecl(v_env_207_, v_decl_208_);
v_r_210_ = lean_box(v_res_209_);
return v_r_210_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg___lam__0(lean_object* v_decl_211_, lean_object* v_toPure_212_, lean_object* v_____do__lift_213_){
_start:
{
uint8_t v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_214_ = lp_batteries_Lean_Environment_isAutoDecl(v_____do__lift_213_, v_decl_211_);
v___x_215_ = lean_box(v___x_214_);
v___x_216_ = lean_apply_2(v_toPure_212_, lean_box(0), v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg(lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_decl_219_){
_start:
{
lean_object* v_toApplicative_220_; lean_object* v_toBind_221_; lean_object* v_getEnv_222_; lean_object* v_toPure_223_; lean_object* v___f_224_; lean_object* v___x_225_; 
v_toApplicative_220_ = lean_ctor_get(v_inst_217_, 0);
lean_inc_ref(v_toApplicative_220_);
v_toBind_221_ = lean_ctor_get(v_inst_217_, 1);
lean_inc(v_toBind_221_);
lean_dec_ref(v_inst_217_);
v_getEnv_222_ = lean_ctor_get(v_inst_218_, 0);
lean_inc(v_getEnv_222_);
lean_dec_ref(v_inst_218_);
v_toPure_223_ = lean_ctor_get(v_toApplicative_220_, 1);
lean_inc(v_toPure_223_);
lean_dec_ref(v_toApplicative_220_);
v___f_224_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_224_, 0, v_decl_219_);
lean_closure_set(v___f_224_, 1, v_toPure_223_);
v___x_225_ = lean_apply_4(v_toBind_221_, lean_box(0), lean_box(0), v_getEnv_222_, v___f_224_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isAutoDecl(lean_object* v_m_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_decl_229_){
_start:
{
lean_object* v_toApplicative_230_; lean_object* v_toBind_231_; lean_object* v_getEnv_232_; lean_object* v_toPure_233_; lean_object* v___f_234_; lean_object* v___x_235_; 
v_toApplicative_230_ = lean_ctor_get(v_inst_227_, 0);
lean_inc_ref(v_toApplicative_230_);
v_toBind_231_ = lean_ctor_get(v_inst_227_, 1);
lean_inc(v_toBind_231_);
lean_dec_ref(v_inst_227_);
v_getEnv_232_ = lean_ctor_get(v_inst_228_, 0);
lean_inc(v_getEnv_232_);
lean_dec_ref(v_inst_228_);
v_toPure_233_ = lean_ctor_get(v_toApplicative_230_, 1);
lean_inc(v_toPure_233_);
lean_dec_ref(v_toApplicative_230_);
v___f_234_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_isAutoDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_234_, 0, v_decl_229_);
lean_closure_set(v___f_234_, 1, v_toPure_233_);
v___x_235_ = lean_apply_4(v_toBind_231_, lean_box(0), lean_box(0), v_getEnv_232_, v___f_234_);
return v___x_235_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Environment_isPrivateOrAutoDecl(lean_object* v_env_236_, lean_object* v_decl_237_){
_start:
{
uint8_t v___x_238_; 
v___x_238_ = l_Lean_isPrivateName(v_decl_237_);
if (v___x_238_ == 0)
{
uint8_t v___x_239_; 
v___x_239_ = lp_batteries_Lean_Environment_isAutoDecl(v_env_236_, v_decl_237_);
return v___x_239_;
}
else
{
lean_dec(v_decl_237_);
lean_dec_ref(v_env_236_);
return v___x_238_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Environment_isPrivateOrAutoDecl___boxed(lean_object* v_env_240_, lean_object* v_decl_241_){
_start:
{
uint8_t v_res_242_; lean_object* v_r_243_; 
v_res_242_ = lp_batteries_Lean_Environment_isPrivateOrAutoDecl(v_env_240_, v_decl_241_);
v_r_243_ = lean_box(v_res_242_);
return v_r_243_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg___lam__0(lean_object* v_decl_244_, lean_object* v_toPure_245_, lean_object* v_____do__lift_246_){
_start:
{
uint8_t v___x_247_; 
v___x_247_ = l_Lean_isPrivateName(v_decl_244_);
if (v___x_247_ == 0)
{
uint8_t v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_248_ = lp_batteries_Lean_Environment_isAutoDecl(v_____do__lift_246_, v_decl_244_);
v___x_249_ = lean_box(v___x_248_);
v___x_250_ = lean_apply_2(v_toPure_245_, lean_box(0), v___x_249_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec_ref(v_____do__lift_246_);
lean_dec(v_decl_244_);
v___x_251_ = lean_box(v___x_247_);
v___x_252_ = lean_apply_2(v_toPure_245_, lean_box(0), v___x_251_);
return v___x_252_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg(lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_decl_255_){
_start:
{
lean_object* v_toApplicative_256_; lean_object* v_toBind_257_; lean_object* v_getEnv_258_; lean_object* v_toPure_259_; lean_object* v___f_260_; lean_object* v___x_261_; 
v_toApplicative_256_ = lean_ctor_get(v_inst_253_, 0);
lean_inc_ref(v_toApplicative_256_);
v_toBind_257_ = lean_ctor_get(v_inst_253_, 1);
lean_inc(v_toBind_257_);
lean_dec_ref(v_inst_253_);
v_getEnv_258_ = lean_ctor_get(v_inst_254_, 0);
lean_inc(v_getEnv_258_);
lean_dec_ref(v_inst_254_);
v_toPure_259_ = lean_ctor_get(v_toApplicative_256_, 1);
lean_inc(v_toPure_259_);
lean_dec_ref(v_toApplicative_256_);
v___f_260_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_260_, 0, v_decl_255_);
lean_closure_set(v___f_260_, 1, v_toPure_259_);
v___x_261_ = lean_apply_4(v_toBind_257_, lean_box(0), lean_box(0), v_getEnv_258_, v___f_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl(lean_object* v_m_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_decl_265_){
_start:
{
lean_object* v_toApplicative_266_; lean_object* v_toBind_267_; lean_object* v_getEnv_268_; lean_object* v_toPure_269_; lean_object* v___f_270_; lean_object* v___x_271_; 
v_toApplicative_266_ = lean_ctor_get(v_inst_263_, 0);
lean_inc_ref(v_toApplicative_266_);
v_toBind_267_ = lean_ctor_get(v_inst_263_, 1);
lean_inc(v_toBind_267_);
lean_dec_ref(v_inst_263_);
v_getEnv_268_ = lean_ctor_get(v_inst_264_, 0);
lean_inc(v_getEnv_268_);
lean_dec_ref(v_inst_264_);
v_toPure_269_ = lean_ctor_get(v_toApplicative_266_, 1);
lean_inc(v_toPure_269_);
lean_dec_ref(v_toApplicative_266_);
v___f_270_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_isPrivateOrAutoDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_270_, 0, v_decl_265_);
lean_closure_set(v___f_270_, 1, v_toPure_269_);
v___x_271_ = lean_apply_4(v_toBind_267_, lean_box(0), lean_box(0), v_getEnv_268_, v___f_270_);
return v___x_271_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0(void){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = l_instMonadEIO(lean_box(0));
return v___x_272_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1(void){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0_once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__0);
v___x_274_ = l_StateRefT_x27_instMonad___redArg(v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1(lean_object* v_name_287_, lean_object* v_declName_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v___x_292_; lean_object* v_toApplicative_293_; lean_object* v_toFunctor_294_; lean_object* v_toSeq_295_; lean_object* v_toSeqLeft_296_; lean_object* v_toSeqRight_297_; lean_object* v___f_298_; lean_object* v___f_299_; lean_object* v___f_300_; lean_object* v___f_301_; lean_object* v___x_302_; lean_object* v___f_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___f_314_; lean_object* v___x_315_; lean_object* v___x_148__overap_316_; lean_object* v___x_317_; 
v___x_292_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__1);
v_toApplicative_293_ = lean_ctor_get(v___x_292_, 0);
v_toFunctor_294_ = lean_ctor_get(v_toApplicative_293_, 0);
v_toSeq_295_ = lean_ctor_get(v_toApplicative_293_, 2);
v_toSeqLeft_296_ = lean_ctor_get(v_toApplicative_293_, 3);
v_toSeqRight_297_ = lean_ctor_get(v_toApplicative_293_, 4);
v___f_298_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__2));
v___f_299_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__3));
lean_inc_ref_n(v_toFunctor_294_, 2);
v___f_300_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_300_, 0, v_toFunctor_294_);
v___f_301_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_301_, 0, v_toFunctor_294_);
v___x_302_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_302_, 0, v___f_300_);
lean_ctor_set(v___x_302_, 1, v___f_301_);
lean_inc(v_toSeqRight_297_);
v___f_303_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_303_, 0, v_toSeqRight_297_);
lean_inc(v_toSeqLeft_296_);
v___f_304_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_304_, 0, v_toSeqLeft_296_);
lean_inc(v_toSeq_295_);
v___f_305_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_305_, 0, v_toSeq_295_);
v___x_306_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_306_, 0, v___x_302_);
lean_ctor_set(v___x_306_, 1, v___f_298_);
lean_ctor_set(v___x_306_, 2, v___f_305_);
lean_ctor_set(v___x_306_, 3, v___f_304_);
lean_ctor_set(v___x_306_, 4, v___f_303_);
v___x_307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_306_);
lean_ctor_set(v___x_307_, 1, v___f_299_);
v___x_308_ = l_Lean_Core_instMonadEnvCoreM;
v___x_309_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___x_310_ = l_Lean_Core_instMonadRefCoreM;
v___x_311_ = l_Lean_Core_instAddMessageContextCoreM;
lean_inc_ref(v___x_307_);
v___x_312_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_311_, v___x_307_);
v___x_313_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_313_, 0, v___x_309_);
lean_ctor_set(v___x_313_, 1, v___x_310_);
lean_ctor_set(v___x_313_, 2, v___x_312_);
v___f_314_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__4));
v___x_315_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9));
lean_inc(v_declName_288_);
v___x_148__overap_316_ = l_Lean_evalConstCheck___redArg(v___x_307_, v___x_308_, v___x_313_, v___f_314_, v___x_315_, v_declName_288_);
lean_inc(v_a_290_);
lean_inc_ref(v_a_289_);
v___x_317_ = lean_apply_3(v___x_148__overap_316_, v_a_289_, v_a_290_, lean_box(0));
if (lean_obj_tag(v___x_317_) == 0)
{
lean_object* v_a_318_; lean_object* v___x_320_; uint8_t v_isShared_321_; uint8_t v_isSharedCheck_326_; 
v_a_318_ = lean_ctor_get(v___x_317_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_317_);
if (v_isSharedCheck_326_ == 0)
{
v___x_320_ = v___x_317_;
v_isShared_321_ = v_isSharedCheck_326_;
goto v_resetjp_319_;
}
else
{
lean_inc(v_a_318_);
lean_dec(v___x_317_);
v___x_320_ = lean_box(0);
v_isShared_321_ = v_isSharedCheck_326_;
goto v_resetjp_319_;
}
v_resetjp_319_:
{
lean_object* v___x_322_; lean_object* v___x_324_; 
v___x_322_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_322_, 0, v_a_318_);
lean_ctor_set(v___x_322_, 1, v_name_287_);
lean_ctor_set(v___x_322_, 2, v_declName_288_);
if (v_isShared_321_ == 0)
{
lean_ctor_set(v___x_320_, 0, v___x_322_);
v___x_324_ = v___x_320_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_322_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
else
{
lean_object* v_a_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_334_; 
lean_dec(v_declName_288_);
lean_dec(v_name_287_);
v_a_327_ = lean_ctor_get(v___x_317_, 0);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_317_);
if (v_isSharedCheck_334_ == 0)
{
v___x_329_ = v___x_317_;
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_a_327_);
lean_dec(v___x_317_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_332_; 
if (v_isShared_330_ == 0)
{
v___x_332_ = v___x_329_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_a_327_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___boxed(lean_object* v_name_335_, lean_object* v_declName_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1(v_name_335_, v_declName_336_, v_a_337_, v_a_338_);
lean_dec(v_a_338_);
lean_dec_ref(v_a_337_);
return v_res_340_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_341_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__0);
v___x_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
return v___x_343_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1);
v___x_345_ = lean_unsigned_to_nat(0u);
v___x_346_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
lean_ctor_set(v___x_346_, 2, v___x_345_);
lean_ctor_set(v___x_346_, 3, v___x_345_);
lean_ctor_set(v___x_346_, 4, v___x_344_);
lean_ctor_set(v___x_346_, 5, v___x_344_);
lean_ctor_set(v___x_346_, 6, v___x_344_);
lean_ctor_set(v___x_346_, 7, v___x_344_);
lean_ctor_set(v___x_346_, 8, v___x_344_);
lean_ctor_set(v___x_346_, 9, v___x_344_);
return v___x_346_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_347_ = lean_unsigned_to_nat(32u);
v___x_348_ = lean_mk_empty_array_with_capacity(v___x_347_);
v___x_349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
return v___x_349_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4(void){
_start:
{
size_t v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_350_ = ((size_t)5ULL);
v___x_351_ = lean_unsigned_to_nat(0u);
v___x_352_ = lean_unsigned_to_nat(32u);
v___x_353_ = lean_mk_empty_array_with_capacity(v___x_352_);
v___x_354_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3);
v___x_355_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v___x_353_);
lean_ctor_set(v___x_355_, 2, v___x_351_);
lean_ctor_set(v___x_355_, 3, v___x_351_);
lean_ctor_set_usize(v___x_355_, 4, v___x_350_);
return v___x_355_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_356_ = lean_box(1);
v___x_357_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__4);
v___x_358_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__1);
v___x_359_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_359_, 0, v___x_358_);
lean_ctor_set(v___x_359_, 1, v___x_357_);
lean_ctor_set(v___x_359_, 2, v___x_356_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3(lean_object* v_msgData_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; lean_object* v_env_365_; lean_object* v_options_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_364_ = lean_st_ref_get(v___y_362_);
v_env_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc_ref(v_env_365_);
lean_dec(v___x_364_);
v_options_366_ = lean_ctor_get(v___y_361_, 2);
v___x_367_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2);
v___x_368_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5);
lean_inc_ref(v_options_366_);
v___x_369_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_369_, 0, v_env_365_);
lean_ctor_set(v___x_369_, 1, v___x_367_);
lean_ctor_set(v___x_369_, 2, v___x_368_);
lean_ctor_set(v___x_369_, 3, v_options_366_);
v___x_370_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set(v___x_370_, 1, v_msgData_360_);
v___x_371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_msgData_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3(v_msgData_372_, v___y_373_, v___y_374_);
lean_dec(v___y_374_);
lean_dec_ref(v___y_373_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(lean_object* v_msg_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_ref_381_; lean_object* v___x_382_; lean_object* v_a_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_391_; 
v_ref_381_ = lean_ctor_get(v___y_378_, 5);
v___x_382_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3(v_msg_377_, v___y_378_, v___y_379_);
v_a_383_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_391_ == 0)
{
v___x_385_ = v___x_382_;
v_isShared_386_ = v_isSharedCheck_391_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_a_383_);
lean_dec(v___x_382_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_391_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; lean_object* v___x_389_; 
lean_inc(v_ref_381_);
v___x_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_387_, 0, v_ref_381_);
lean_ctor_set(v___x_387_, 1, v_a_383_);
if (v_isShared_386_ == 0)
{
lean_ctor_set_tag(v___x_385_, 1);
lean_ctor_set(v___x_385_, 0, v___x_387_);
v___x_389_ = v___x_385_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_387_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_msg_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v_res_396_; 
v_res_396_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v_msg_392_, v___y_393_, v___y_394_);
lean_dec(v___y_394_);
lean_dec_ref(v___y_393_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(lean_object* v_x_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
if (lean_obj_tag(v_x_397_) == 0)
{
lean_object* v_a_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v_a_401_ = lean_ctor_get(v_x_397_, 0);
lean_inc(v_a_401_);
lean_dec_ref_known(v_x_397_, 1);
v___x_402_ = l_Lean_stringToMessageData(v_a_401_);
v___x_403_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_402_, v___y_398_, v___y_399_);
return v___x_403_;
}
else
{
lean_object* v_a_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_411_; 
v_a_404_ = lean_ctor_get(v_x_397_, 0);
v_isSharedCheck_411_ = !lean_is_exclusive(v_x_397_);
if (v_isSharedCheck_411_ == 0)
{
v___x_406_ = v_x_397_;
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_a_404_);
lean_dec(v_x_397_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_409_; 
if (v_isShared_407_ == 0)
{
lean_ctor_set_tag(v___x_406_, 0);
v___x_409_ = v___x_406_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v_a_404_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg___boxed(lean_object* v_x_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(v_x_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
return v_res_416_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_417_ = lean_box(0);
v___x_418_ = l_Lean_Elab_abortCommandExceptionId;
v___x_419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v___x_417_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg(){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = lean_obj_once(&lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0, &lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___closed__0);
v___x_422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg___boxed(lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg();
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg(lean_object* v_typeName_425_, lean_object* v_constName_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v___x_430_; lean_object* v_env_431_; uint8_t v___x_432_; 
v___x_430_ = lean_st_ref_get(v___y_428_);
v_env_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc_ref(v_env_431_);
lean_dec(v___x_430_);
lean_inc(v_constName_426_);
v___x_432_ = lean_has_compile_error(v_env_431_, v_constName_426_);
if (v___x_432_ == 0)
{
lean_object* v___x_433_; lean_object* v_env_434_; lean_object* v_options_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_433_ = lean_st_ref_get(v___y_428_);
v_env_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc_ref(v_env_434_);
lean_dec(v___x_433_);
v_options_435_ = lean_ctor_get(v___y_427_, 2);
v___x_436_ = l_Lean_Environment_evalConstCheck___redArg(v_env_434_, v_options_435_, v_typeName_425_, v_constName_426_);
v___x_437_ = lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(v___x_436_, v___y_427_, v___y_428_);
return v___x_437_;
}
else
{
lean_object* v___x_438_; 
v___x_438_ = lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg();
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v___x_439_; lean_object* v_env_440_; lean_object* v_options_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
lean_dec_ref_known(v___x_438_, 1);
v___x_439_ = lean_st_ref_get(v___y_428_);
v_env_440_ = lean_ctor_get(v___x_439_, 0);
lean_inc_ref(v_env_440_);
lean_dec(v___x_439_);
v_options_441_ = lean_ctor_get(v___y_427_, 2);
v___x_442_ = l_Lean_Environment_evalConstCheck___redArg(v_env_440_, v_options_441_, v_typeName_425_, v_constName_426_);
v___x_443_ = lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(v___x_442_, v___y_427_, v___y_428_);
return v___x_443_;
}
else
{
lean_object* v_a_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_451_; 
lean_dec(v_constName_426_);
lean_dec(v_typeName_425_);
v_a_444_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_451_ == 0)
{
v___x_446_ = v___x_438_;
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_a_444_);
lean_dec(v___x_438_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_449_; 
if (v_isShared_447_ == 0)
{
v___x_449_ = v___x_446_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_a_444_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg___boxed(lean_object* v_typeName_452_, lean_object* v_constName_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg(v_typeName_452_, v_constName_453_, v___y_454_, v___y_455_);
lean_dec(v___y_455_);
lean_dec_ref(v___y_454_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getLinter(lean_object* v_name_458_, lean_object* v_declName_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_463_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__9));
lean_inc(v_declName_459_);
v___x_464_ = lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg(v___x_463_, v_declName_459_, v_a_460_, v_a_461_);
if (lean_obj_tag(v___x_464_) == 0)
{
lean_object* v_a_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_473_; 
v_a_465_ = lean_ctor_get(v___x_464_, 0);
v_isSharedCheck_473_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_473_ == 0)
{
v___x_467_ = v___x_464_;
v_isShared_468_ = v_isSharedCheck_473_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_a_465_);
lean_dec(v___x_464_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_473_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___x_469_; lean_object* v___x_471_; 
v___x_469_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_469_, 0, v_a_465_);
lean_ctor_set(v___x_469_, 1, v_name_458_);
lean_ctor_set(v___x_469_, 2, v_declName_459_);
if (v_isShared_468_ == 0)
{
lean_ctor_set(v___x_467_, 0, v___x_469_);
v___x_471_ = v___x_467_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_472_; 
v_reuseFailAlloc_472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_472_, 0, v___x_469_);
v___x_471_ = v_reuseFailAlloc_472_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
return v___x_471_;
}
}
}
else
{
lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_481_; 
lean_dec(v_declName_459_);
lean_dec(v_name_458_);
v_a_474_ = lean_ctor_get(v___x_464_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_481_ == 0)
{
v___x_476_ = v___x_464_;
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v___x_464_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_477_ == 0)
{
v___x_479_ = v___x_476_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_a_474_);
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
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getLinter___boxed(lean_object* v_name_482_, lean_object* v_declName_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_batteries_Batteries_Tactic_Lint_getLinter(v_name_482_, v_declName_483_, v_a_484_, v_a_485_);
lean_dec(v_a_485_);
lean_dec_ref(v_a_484_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1(lean_object* v_00_u03b1_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___redArg();
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1___boxed(lean_object* v_00_u03b1_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_batteries_Lean_Elab_throwAbortCommand___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__1(v_00_u03b1_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0(lean_object* v_00_u03b1_498_, lean_object* v_typeName_499_, lean_object* v_constName_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
lean_object* v___x_504_; 
v___x_504_ = lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___redArg(v_typeName_499_, v_constName_500_, v___y_501_, v___y_502_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0___boxed(lean_object* v_00_u03b1_505_, lean_object* v_typeName_506_, lean_object* v_constName_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_batteries_Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0(v_00_u03b1_505_, v_typeName_506_, v_constName_507_, v___y_508_, v___y_509_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0(lean_object* v_00_u03b1_512_, lean_object* v_x_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___redArg(v_x_513_, v___y_514_, v___y_515_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0___boxed(lean_object* v_00_u03b1_518_, lean_object* v_x_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_batteries_Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0(v_00_u03b1_518_, v_x_519_, v___y_520_, v___y_521_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_524_, lean_object* v_msg_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v_msg_525_, v___y_526_, v___y_527_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_530_, lean_object* v_msg_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1(v_00_u03b1_530_, v_msg_531_, v___y_532_, v___y_533_);
lean_dec(v___y_533_);
lean_dec_ref(v___y_532_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object* v_m_536_, lean_object* v_x_537_){
_start:
{
lean_object* v_fst_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v_fst_538_ = lean_ctor_get(v_x_537_, 0);
v___x_539_ = lean_box(0);
lean_inc(v_fst_538_);
v___x_540_ = l_Lean_Name_updatePrefix(v_fst_538_, v___x_539_);
v___x_541_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_540_, v_x_537_, v_m_536_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_as_542_, size_t v_i_543_, size_t v_stop_544_, lean_object* v_b_545_){
_start:
{
uint8_t v___x_546_; 
v___x_546_ = lean_usize_dec_eq(v_i_543_, v_stop_544_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; lean_object* v_fst_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; size_t v___x_552_; size_t v___x_553_; 
v___x_547_ = lean_array_uget_borrowed(v_as_542_, v_i_543_);
v_fst_548_ = lean_ctor_get(v___x_547_, 0);
v___x_549_ = lean_box(0);
lean_inc(v_fst_548_);
v___x_550_ = l_Lean_Name_updatePrefix(v_fst_548_, v___x_549_);
lean_inc(v___x_547_);
v___x_551_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_550_, v___x_547_, v_b_545_);
v___x_552_ = ((size_t)1ULL);
v___x_553_ = lean_usize_add(v_i_543_, v___x_552_);
v_i_543_ = v___x_553_;
v_b_545_ = v___x_551_;
goto _start;
}
else
{
return v_b_545_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_as_555_, lean_object* v_i_556_, lean_object* v_stop_557_, lean_object* v_b_558_){
_start:
{
size_t v_i_boxed_559_; size_t v_stop_boxed_560_; lean_object* v_res_561_; 
v_i_boxed_559_ = lean_unbox_usize(v_i_556_);
lean_dec(v_i_556_);
v_stop_boxed_560_ = lean_unbox_usize(v_stop_557_);
lean_dec(v_stop_557_);
v_res_561_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0(v_as_555_, v_i_boxed_559_, v_stop_boxed_560_, v_b_558_);
lean_dec_ref(v_as_555_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0(lean_object* v_as_562_, size_t v_i_563_, size_t v_stop_564_, lean_object* v_b_565_){
_start:
{
uint8_t v___x_566_; 
v___x_566_ = lean_usize_dec_eq(v_i_563_, v_stop_564_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; lean_object* v_fst_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; size_t v___x_572_; size_t v___x_573_; lean_object* v___x_574_; 
v___x_567_ = lean_array_uget_borrowed(v_as_562_, v_i_563_);
v_fst_568_ = lean_ctor_get(v___x_567_, 0);
v___x_569_ = lean_box(0);
lean_inc(v_fst_568_);
v___x_570_ = l_Lean_Name_updatePrefix(v_fst_568_, v___x_569_);
lean_inc(v___x_567_);
v___x_571_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_570_, v___x_567_, v_b_565_);
v___x_572_ = ((size_t)1ULL);
v___x_573_ = lean_usize_add(v_i_563_, v___x_572_);
v___x_574_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0_spec__0(v_as_562_, v___x_573_, v_stop_564_, v___x_571_);
return v___x_574_;
}
else
{
return v_b_565_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_575_, lean_object* v_i_576_, lean_object* v_stop_577_, lean_object* v_b_578_){
_start:
{
size_t v_i_boxed_579_; size_t v_stop_boxed_580_; lean_object* v_res_581_; 
v_i_boxed_579_ = lean_unbox_usize(v_i_576_);
lean_dec(v_i_576_);
v_stop_boxed_580_ = lean_unbox_usize(v_stop_577_);
lean_dec(v_stop_577_);
v_res_581_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0(v_as_575_, v_i_boxed_579_, v_stop_boxed_580_, v_b_578_);
lean_dec_ref(v_as_575_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1(lean_object* v_as_582_, size_t v_i_583_, size_t v_stop_584_, lean_object* v_b_585_){
_start:
{
lean_object* v___y_587_; uint8_t v___x_591_; 
v___x_591_ = lean_usize_dec_eq(v_i_583_, v_stop_584_);
if (v___x_591_ == 0)
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; uint8_t v___x_595_; 
v___x_592_ = lean_array_uget_borrowed(v_as_582_, v_i_583_);
v___x_593_ = lean_unsigned_to_nat(0u);
v___x_594_ = lean_array_get_size(v___x_592_);
v___x_595_ = lean_nat_dec_lt(v___x_593_, v___x_594_);
if (v___x_595_ == 0)
{
v___y_587_ = v_b_585_;
goto v___jp_586_;
}
else
{
uint8_t v___x_596_; 
v___x_596_ = lean_nat_dec_le(v___x_594_, v___x_594_);
if (v___x_596_ == 0)
{
if (v___x_595_ == 0)
{
v___y_587_ = v_b_585_;
goto v___jp_586_;
}
else
{
size_t v___x_597_; size_t v___x_598_; lean_object* v___x_599_; 
v___x_597_ = ((size_t)0ULL);
v___x_598_ = lean_usize_of_nat(v___x_594_);
v___x_599_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0(v___x_592_, v___x_597_, v___x_598_, v_b_585_);
v___y_587_ = v___x_599_;
goto v___jp_586_;
}
}
else
{
size_t v___x_600_; size_t v___x_601_; lean_object* v___x_602_; 
v___x_600_ = ((size_t)0ULL);
v___x_601_ = lean_usize_of_nat(v___x_594_);
v___x_602_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__0(v___x_592_, v___x_600_, v___x_601_, v_b_585_);
v___y_587_ = v___x_602_;
goto v___jp_586_;
}
}
}
else
{
return v_b_585_;
}
v___jp_586_:
{
size_t v___x_588_; size_t v___x_589_; 
v___x_588_ = ((size_t)1ULL);
v___x_589_ = lean_usize_add(v_i_583_, v___x_588_);
v_i_583_ = v___x_589_;
v_b_585_ = v___y_587_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1___boxed(lean_object* v_as_603_, lean_object* v_i_604_, lean_object* v_stop_605_, lean_object* v_b_606_){
_start:
{
size_t v_i_boxed_607_; size_t v_stop_boxed_608_; lean_object* v_res_609_; 
v_i_boxed_607_ = lean_unbox_usize(v_i_604_);
lean_dec(v_i_604_);
v_stop_boxed_608_ = lean_unbox_usize(v_stop_605_);
lean_dec(v_stop_605_);
v_res_609_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1(v_as_603_, v_i_boxed_607_, v_stop_boxed_608_, v_b_606_);
lean_dec_ref(v_as_603_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object* v_nss_610_){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; uint8_t v___x_614_; 
v___x_611_ = lean_box(1);
v___x_612_ = lean_unsigned_to_nat(0u);
v___x_613_ = lean_array_get_size(v_nss_610_);
v___x_614_ = lean_nat_dec_lt(v___x_612_, v___x_613_);
if (v___x_614_ == 0)
{
return v___x_611_;
}
else
{
uint8_t v___x_615_; 
v___x_615_ = lean_nat_dec_le(v___x_613_, v___x_613_);
if (v___x_615_ == 0)
{
if (v___x_614_ == 0)
{
return v___x_611_;
}
else
{
size_t v___x_616_; size_t v___x_617_; lean_object* v___x_618_; 
v___x_616_ = ((size_t)0ULL);
v___x_617_ = lean_usize_of_nat(v___x_613_);
v___x_618_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1(v_nss_610_, v___x_616_, v___x_617_, v___x_611_);
return v___x_618_;
}
}
else
{
size_t v___x_619_; size_t v___x_620_; lean_object* v___x_621_; 
v___x_619_ = ((size_t)0ULL);
v___x_620_ = lean_usize_of_nat(v___x_613_);
v___x_621_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2__spec__1(v_nss_610_, v___x_619_, v___x_620_, v___x_611_);
return v___x_621_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2____boxed(lean_object* v_nss_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(v_nss_622_);
lean_dec_ref(v_nss_622_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(lean_object* v_es_624_){
_start:
{
lean_object* v___x_625_; 
v___x_625_ = lean_array_mk(v_es_624_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_643_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_));
v___x_644_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2____boxed(lean_object* v_a_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_();
return v_res_646_;
}
}
static lean_object* _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_679_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__0));
v___x_680_ = l_Lean_stringToMessageData(v___x_679_);
return v___x_680_;
}
}
static lean_object* _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_682_; lean_object* v___x_683_; 
v___x_682_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__2));
v___x_683_ = l_Lean_stringToMessageData(v___x_682_);
return v___x_683_;
}
}
static lean_object* _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_685_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__4));
v___x_686_ = l_Lean_stringToMessageData(v___x_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg(lean_object* v_name_690_, uint8_t v_kind_691_, lean_object* v___y_692_, lean_object* v___y_693_){
_start:
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___y_701_; 
v___x_695_ = lean_obj_once(&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1, &lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__1);
v___x_696_ = l_Lean_MessageData_ofName(v_name_690_);
v___x_697_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_697_, 0, v___x_695_);
lean_ctor_set(v___x_697_, 1, v___x_696_);
v___x_698_ = lean_obj_once(&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3, &lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3_once, _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__3);
v___x_699_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_697_);
lean_ctor_set(v___x_699_, 1, v___x_698_);
switch(v_kind_691_)
{
case 0:
{
lean_object* v___x_708_; 
v___x_708_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__6));
v___y_701_ = v___x_708_;
goto v___jp_700_;
}
case 1:
{
lean_object* v___x_709_; 
v___x_709_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__7));
v___y_701_ = v___x_709_;
goto v___jp_700_;
}
default: 
{
lean_object* v___x_710_; 
v___x_710_ = ((lean_object*)(lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__8));
v___y_701_ = v___x_710_;
goto v___jp_700_;
}
}
v___jp_700_:
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
lean_inc_ref(v___y_701_);
v___x_702_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_702_, 0, v___y_701_);
v___x_703_ = l_Lean_MessageData_ofFormat(v___x_702_);
v___x_704_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_704_, 0, v___x_699_);
lean_ctor_set(v___x_704_, 1, v___x_703_);
v___x_705_ = lean_obj_once(&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5, &lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5_once, _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5);
v___x_706_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_706_, 0, v___x_704_);
lean_ctor_set(v___x_706_, 1, v___x_705_);
v___x_707_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_706_, v___y_692_, v___y_693_);
return v___x_707_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_name_711_, lean_object* v_kind_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
uint8_t v_kind_boxed_716_; lean_object* v_res_717_; 
v_kind_boxed_716_ = lean_unbox(v_kind_712_);
v_res_717_ = lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg(v_name_711_, v_kind_boxed_716_, v___y_713_, v___y_714_);
lean_dec(v___y_714_);
lean_dec_ref(v___y_713_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg(lean_object* v_t_718_, lean_object* v___y_719_){
_start:
{
lean_object* v___x_721_; lean_object* v_infoState_722_; uint8_t v_enabled_723_; 
v___x_721_ = lean_st_ref_get(v___y_719_);
v_infoState_722_ = lean_ctor_get(v___x_721_, 7);
lean_inc_ref(v_infoState_722_);
lean_dec(v___x_721_);
v_enabled_723_ = lean_ctor_get_uint8(v_infoState_722_, sizeof(void*)*3);
lean_dec_ref(v_infoState_722_);
if (v_enabled_723_ == 0)
{
lean_object* v___x_724_; lean_object* v___x_725_; 
lean_dec_ref(v_t_718_);
v___x_724_ = lean_box(0);
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
else
{
lean_object* v___x_726_; lean_object* v_infoState_727_; lean_object* v_env_728_; lean_object* v_nextMacroScope_729_; lean_object* v_ngen_730_; lean_object* v_auxDeclNGen_731_; lean_object* v_traceState_732_; lean_object* v_cache_733_; lean_object* v_messages_734_; lean_object* v_snapshotTasks_735_; lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_757_; 
v___x_726_ = lean_st_ref_take(v___y_719_);
v_infoState_727_ = lean_ctor_get(v___x_726_, 7);
v_env_728_ = lean_ctor_get(v___x_726_, 0);
v_nextMacroScope_729_ = lean_ctor_get(v___x_726_, 1);
v_ngen_730_ = lean_ctor_get(v___x_726_, 2);
v_auxDeclNGen_731_ = lean_ctor_get(v___x_726_, 3);
v_traceState_732_ = lean_ctor_get(v___x_726_, 4);
v_cache_733_ = lean_ctor_get(v___x_726_, 5);
v_messages_734_ = lean_ctor_get(v___x_726_, 6);
v_snapshotTasks_735_ = lean_ctor_get(v___x_726_, 8);
v_isSharedCheck_757_ = !lean_is_exclusive(v___x_726_);
if (v_isSharedCheck_757_ == 0)
{
v___x_737_ = v___x_726_;
v_isShared_738_ = v_isSharedCheck_757_;
goto v_resetjp_736_;
}
else
{
lean_inc(v_snapshotTasks_735_);
lean_inc(v_infoState_727_);
lean_inc(v_messages_734_);
lean_inc(v_cache_733_);
lean_inc(v_traceState_732_);
lean_inc(v_auxDeclNGen_731_);
lean_inc(v_ngen_730_);
lean_inc(v_nextMacroScope_729_);
lean_inc(v_env_728_);
lean_dec(v___x_726_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_757_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
uint8_t v_enabled_739_; lean_object* v_assignment_740_; lean_object* v_lazyAssignment_741_; lean_object* v_trees_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_756_; 
v_enabled_739_ = lean_ctor_get_uint8(v_infoState_727_, sizeof(void*)*3);
v_assignment_740_ = lean_ctor_get(v_infoState_727_, 0);
v_lazyAssignment_741_ = lean_ctor_get(v_infoState_727_, 1);
v_trees_742_ = lean_ctor_get(v_infoState_727_, 2);
v_isSharedCheck_756_ = !lean_is_exclusive(v_infoState_727_);
if (v_isSharedCheck_756_ == 0)
{
v___x_744_ = v_infoState_727_;
v_isShared_745_ = v_isSharedCheck_756_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_trees_742_);
lean_inc(v_lazyAssignment_741_);
lean_inc(v_assignment_740_);
lean_dec(v_infoState_727_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_756_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_746_; lean_object* v___x_748_; 
v___x_746_ = l_Lean_PersistentArray_push___redArg(v_trees_742_, v_t_718_);
if (v_isShared_745_ == 0)
{
lean_ctor_set(v___x_744_, 2, v___x_746_);
v___x_748_ = v___x_744_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v_assignment_740_);
lean_ctor_set(v_reuseFailAlloc_755_, 1, v_lazyAssignment_741_);
lean_ctor_set(v_reuseFailAlloc_755_, 2, v___x_746_);
lean_ctor_set_uint8(v_reuseFailAlloc_755_, sizeof(void*)*3, v_enabled_739_);
v___x_748_ = v_reuseFailAlloc_755_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
lean_object* v___x_750_; 
if (v_isShared_738_ == 0)
{
lean_ctor_set(v___x_737_, 7, v___x_748_);
v___x_750_ = v___x_737_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v_env_728_);
lean_ctor_set(v_reuseFailAlloc_754_, 1, v_nextMacroScope_729_);
lean_ctor_set(v_reuseFailAlloc_754_, 2, v_ngen_730_);
lean_ctor_set(v_reuseFailAlloc_754_, 3, v_auxDeclNGen_731_);
lean_ctor_set(v_reuseFailAlloc_754_, 4, v_traceState_732_);
lean_ctor_set(v_reuseFailAlloc_754_, 5, v_cache_733_);
lean_ctor_set(v_reuseFailAlloc_754_, 6, v_messages_734_);
lean_ctor_set(v_reuseFailAlloc_754_, 7, v___x_748_);
lean_ctor_set(v_reuseFailAlloc_754_, 8, v_snapshotTasks_735_);
v___x_750_ = v_reuseFailAlloc_754_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_751_ = lean_st_ref_set(v___y_719_, v___x_750_);
v___x_752_ = lean_box(0);
v___x_753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_753_, 0, v___x_752_);
return v___x_753_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg___boxed(lean_object* v_t_758_, lean_object* v___y_759_, lean_object* v___y_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg(v_t_758_, v___y_759_);
lean_dec(v___y_759_);
return v_res_761_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_762_ = lean_unsigned_to_nat(32u);
v___x_763_ = lean_mk_empty_array_with_capacity(v___x_762_);
v___x_764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_764_, 0, v___x_763_);
return v___x_764_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1(void){
_start:
{
size_t v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_765_ = ((size_t)5ULL);
v___x_766_ = lean_unsigned_to_nat(0u);
v___x_767_ = lean_unsigned_to_nat(32u);
v___x_768_ = lean_mk_empty_array_with_capacity(v___x_767_);
v___x_769_ = lean_obj_once(&lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0, &lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0_once, _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__0);
v___x_770_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_770_, 0, v___x_769_);
lean_ctor_set(v___x_770_, 1, v___x_768_);
lean_ctor_set(v___x_770_, 2, v___x_766_);
lean_ctor_set(v___x_770_, 3, v___x_766_);
lean_ctor_set_usize(v___x_770_, 4, v___x_765_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3(lean_object* v_t_771_, lean_object* v___y_772_, lean_object* v___y_773_){
_start:
{
lean_object* v___x_775_; lean_object* v_infoState_776_; uint8_t v_enabled_777_; 
v___x_775_ = lean_st_ref_get(v___y_773_);
v_infoState_776_ = lean_ctor_get(v___x_775_, 7);
lean_inc_ref(v_infoState_776_);
lean_dec(v___x_775_);
v_enabled_777_ = lean_ctor_get_uint8(v_infoState_776_, sizeof(void*)*3);
lean_dec_ref(v_infoState_776_);
if (v_enabled_777_ == 0)
{
lean_object* v___x_778_; lean_object* v___x_779_; 
lean_dec_ref(v_t_771_);
v___x_778_ = lean_box(0);
v___x_779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_779_, 0, v___x_778_);
return v___x_779_;
}
else
{
lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
v___x_780_ = lean_obj_once(&lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1, &lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1_once, _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___closed__1);
v___x_781_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_781_, 0, v_t_771_);
lean_ctor_set(v___x_781_, 1, v___x_780_);
v___x_782_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg(v___x_781_, v___y_773_);
return v___x_782_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3___boxed(lean_object* v_t_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3(v_t_783_, v___y_784_, v___y_785_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
return v_res_787_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_789_; lean_object* v___x_790_; 
v___x_789_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__0));
v___x_790_ = l_Lean_stringToMessageData(v___x_789_);
return v___x_790_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_792_; lean_object* v___x_793_; 
v___x_792_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__2));
v___x_793_ = l_Lean_stringToMessageData(v___x_792_);
return v___x_793_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5(void){
_start:
{
lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_795_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__4));
v___x_796_ = l_Lean_stringToMessageData(v___x_795_);
return v___x_796_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7(void){
_start:
{
lean_object* v___x_798_; lean_object* v___x_799_; 
v___x_798_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__6));
v___x_799_ = l_Lean_stringToMessageData(v___x_798_);
return v___x_799_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9(void){
_start:
{
lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_801_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__8));
v___x_802_ = l_Lean_stringToMessageData(v___x_801_);
return v___x_802_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11(void){
_start:
{
lean_object* v___x_804_; lean_object* v___x_805_; 
v___x_804_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__10));
v___x_805_ = l_Lean_stringToMessageData(v___x_804_);
return v___x_805_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13(void){
_start:
{
lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_807_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__12));
v___x_808_ = l_Lean_stringToMessageData(v___x_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(lean_object* v_msg_809_, lean_object* v_declHint_810_, lean_object* v___y_811_){
_start:
{
lean_object* v___x_813_; lean_object* v_env_814_; uint8_t v___x_815_; 
v___x_813_ = lean_st_ref_get(v___y_811_);
v_env_814_ = lean_ctor_get(v___x_813_, 0);
lean_inc_ref(v_env_814_);
lean_dec(v___x_813_);
v___x_815_ = l_Lean_Name_isAnonymous(v_declHint_810_);
if (v___x_815_ == 0)
{
uint8_t v_isExporting_816_; 
v_isExporting_816_ = lean_ctor_get_uint8(v_env_814_, sizeof(void*)*8);
if (v_isExporting_816_ == 0)
{
lean_object* v___x_817_; 
lean_dec_ref(v_env_814_);
lean_dec(v_declHint_810_);
v___x_817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_817_, 0, v_msg_809_);
return v___x_817_;
}
else
{
lean_object* v___x_818_; uint8_t v___x_819_; 
lean_inc_ref(v_env_814_);
v___x_818_ = l_Lean_Environment_setExporting(v_env_814_, v___x_815_);
lean_inc(v_declHint_810_);
lean_inc_ref(v___x_818_);
v___x_819_ = l_Lean_Environment_contains(v___x_818_, v_declHint_810_, v_isExporting_816_);
if (v___x_819_ == 0)
{
lean_object* v___x_820_; 
lean_dec_ref(v___x_818_);
lean_dec_ref(v_env_814_);
lean_dec(v_declHint_810_);
v___x_820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_820_, 0, v_msg_809_);
return v___x_820_;
}
else
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v_c_826_; lean_object* v___x_827_; 
v___x_821_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__2);
v___x_822_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__5);
v___x_823_ = l_Lean_Options_empty;
v___x_824_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_824_, 0, v___x_818_);
lean_ctor_set(v___x_824_, 1, v___x_821_);
lean_ctor_set(v___x_824_, 2, v___x_822_);
lean_ctor_set(v___x_824_, 3, v___x_823_);
lean_inc(v_declHint_810_);
v___x_825_ = l_Lean_MessageData_ofConstName(v_declHint_810_, v___x_815_);
v_c_826_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_826_, 0, v___x_824_);
lean_ctor_set(v_c_826_, 1, v___x_825_);
v___x_827_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_814_, v_declHint_810_);
if (lean_obj_tag(v___x_827_) == 0)
{
lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
lean_dec_ref(v_env_814_);
lean_dec(v_declHint_810_);
v___x_828_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1);
v___x_829_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
lean_ctor_set(v___x_829_, 1, v_c_826_);
v___x_830_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__3);
v___x_831_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_831_, 0, v___x_829_);
lean_ctor_set(v___x_831_, 1, v___x_830_);
v___x_832_ = l_Lean_MessageData_note(v___x_831_);
v___x_833_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_833_, 0, v_msg_809_);
lean_ctor_set(v___x_833_, 1, v___x_832_);
v___x_834_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_834_, 0, v___x_833_);
return v___x_834_;
}
else
{
lean_object* v_val_835_; lean_object* v___x_837_; uint8_t v_isShared_838_; uint8_t v_isSharedCheck_870_; 
v_val_835_ = lean_ctor_get(v___x_827_, 0);
v_isSharedCheck_870_ = !lean_is_exclusive(v___x_827_);
if (v_isSharedCheck_870_ == 0)
{
v___x_837_ = v___x_827_;
v_isShared_838_ = v_isSharedCheck_870_;
goto v_resetjp_836_;
}
else
{
lean_inc(v_val_835_);
lean_dec(v___x_827_);
v___x_837_ = lean_box(0);
v_isShared_838_ = v_isSharedCheck_870_;
goto v_resetjp_836_;
}
v_resetjp_836_:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v_mod_842_; uint8_t v___x_843_; 
v___x_839_ = lean_box(0);
v___x_840_ = l_Lean_Environment_header(v_env_814_);
lean_dec_ref(v_env_814_);
v___x_841_ = l_Lean_EnvironmentHeader_moduleNames(v___x_840_);
v_mod_842_ = lean_array_get(v___x_839_, v___x_841_, v_val_835_);
lean_dec(v_val_835_);
lean_dec_ref(v___x_841_);
v___x_843_ = l_Lean_isPrivateName(v_declHint_810_);
lean_dec(v_declHint_810_);
if (v___x_843_ == 0)
{
lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_855_; 
v___x_844_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__5);
v___x_845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_845_, 0, v___x_844_);
lean_ctor_set(v___x_845_, 1, v_c_826_);
v___x_846_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__7);
v___x_847_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_845_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
v___x_848_ = l_Lean_MessageData_ofName(v_mod_842_);
v___x_849_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_849_, 0, v___x_847_);
lean_ctor_set(v___x_849_, 1, v___x_848_);
v___x_850_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__9);
v___x_851_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_851_, 0, v___x_849_);
lean_ctor_set(v___x_851_, 1, v___x_850_);
v___x_852_ = l_Lean_MessageData_note(v___x_851_);
v___x_853_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_853_, 0, v_msg_809_);
lean_ctor_set(v___x_853_, 1, v___x_852_);
if (v_isShared_838_ == 0)
{
lean_ctor_set_tag(v___x_837_, 0);
lean_ctor_set(v___x_837_, 0, v___x_853_);
v___x_855_ = v___x_837_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v___x_853_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
else
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_868_; 
v___x_857_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__1);
v___x_858_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_858_, 0, v___x_857_);
lean_ctor_set(v___x_858_, 1, v_c_826_);
v___x_859_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__11);
v___x_860_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_860_, 0, v___x_858_);
lean_ctor_set(v___x_860_, 1, v___x_859_);
v___x_861_ = l_Lean_MessageData_ofName(v_mod_842_);
v___x_862_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_862_, 0, v___x_860_);
lean_ctor_set(v___x_862_, 1, v___x_861_);
v___x_863_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___closed__13);
v___x_864_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_864_, 0, v___x_862_);
lean_ctor_set(v___x_864_, 1, v___x_863_);
v___x_865_ = l_Lean_MessageData_note(v___x_864_);
v___x_866_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_866_, 0, v_msg_809_);
lean_ctor_set(v___x_866_, 1, v___x_865_);
if (v_isShared_838_ == 0)
{
lean_ctor_set_tag(v___x_837_, 0);
lean_ctor_set(v___x_837_, 0, v___x_866_);
v___x_868_ = v___x_837_;
goto v_reusejp_867_;
}
else
{
lean_object* v_reuseFailAlloc_869_; 
v_reuseFailAlloc_869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_869_, 0, v___x_866_);
v___x_868_ = v_reuseFailAlloc_869_;
goto v_reusejp_867_;
}
v_reusejp_867_:
{
return v___x_868_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_871_; 
lean_dec_ref(v_env_814_);
lean_dec(v_declHint_810_);
v___x_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_871_, 0, v_msg_809_);
return v___x_871_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___boxed(lean_object* v_msg_872_, lean_object* v_declHint_873_, lean_object* v___y_874_, lean_object* v___y_875_){
_start:
{
lean_object* v_res_876_; 
v_res_876_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_msg_872_, v_declHint_873_, v___y_874_);
lean_dec(v___y_874_);
return v_res_876_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8(lean_object* v_msg_877_, lean_object* v_declHint_878_, lean_object* v___y_879_, lean_object* v___y_880_){
_start:
{
lean_object* v___x_882_; lean_object* v_a_883_; lean_object* v___x_885_; uint8_t v_isShared_886_; uint8_t v_isSharedCheck_892_; 
v___x_882_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_msg_877_, v_declHint_878_, v___y_880_);
v_a_883_ = lean_ctor_get(v___x_882_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_882_);
if (v_isSharedCheck_892_ == 0)
{
v___x_885_ = v___x_882_;
v_isShared_886_ = v_isSharedCheck_892_;
goto v_resetjp_884_;
}
else
{
lean_inc(v_a_883_);
lean_dec(v___x_882_);
v___x_885_ = lean_box(0);
v_isShared_886_ = v_isSharedCheck_892_;
goto v_resetjp_884_;
}
v_resetjp_884_:
{
lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_890_; 
v___x_887_ = l_Lean_unknownIdentifierMessageTag;
v___x_888_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_888_, 0, v___x_887_);
lean_ctor_set(v___x_888_, 1, v_a_883_);
if (v_isShared_886_ == 0)
{
lean_ctor_set(v___x_885_, 0, v___x_888_);
v___x_890_ = v___x_885_;
goto v_reusejp_889_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v___x_888_);
v___x_890_ = v_reuseFailAlloc_891_;
goto v_reusejp_889_;
}
v_reusejp_889_:
{
return v___x_890_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8___boxed(lean_object* v_msg_893_, lean_object* v_declHint_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
lean_object* v_res_898_; 
v_res_898_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8(v_msg_893_, v_declHint_894_, v___y_895_, v___y_896_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
return v_res_898_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg(lean_object* v_ref_899_, lean_object* v_msg_900_, lean_object* v___y_901_, lean_object* v___y_902_){
_start:
{
lean_object* v_fileName_904_; lean_object* v_fileMap_905_; lean_object* v_options_906_; lean_object* v_currRecDepth_907_; lean_object* v_maxRecDepth_908_; lean_object* v_ref_909_; lean_object* v_currNamespace_910_; lean_object* v_openDecls_911_; lean_object* v_initHeartbeats_912_; lean_object* v_maxHeartbeats_913_; lean_object* v_quotContext_914_; lean_object* v_currMacroScope_915_; uint8_t v_diag_916_; lean_object* v_cancelTk_x3f_917_; uint8_t v_suppressElabErrors_918_; lean_object* v_inheritedTraceOptions_919_; lean_object* v_ref_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v_fileName_904_ = lean_ctor_get(v___y_901_, 0);
v_fileMap_905_ = lean_ctor_get(v___y_901_, 1);
v_options_906_ = lean_ctor_get(v___y_901_, 2);
v_currRecDepth_907_ = lean_ctor_get(v___y_901_, 3);
v_maxRecDepth_908_ = lean_ctor_get(v___y_901_, 4);
v_ref_909_ = lean_ctor_get(v___y_901_, 5);
v_currNamespace_910_ = lean_ctor_get(v___y_901_, 6);
v_openDecls_911_ = lean_ctor_get(v___y_901_, 7);
v_initHeartbeats_912_ = lean_ctor_get(v___y_901_, 8);
v_maxHeartbeats_913_ = lean_ctor_get(v___y_901_, 9);
v_quotContext_914_ = lean_ctor_get(v___y_901_, 10);
v_currMacroScope_915_ = lean_ctor_get(v___y_901_, 11);
v_diag_916_ = lean_ctor_get_uint8(v___y_901_, sizeof(void*)*14);
v_cancelTk_x3f_917_ = lean_ctor_get(v___y_901_, 12);
v_suppressElabErrors_918_ = lean_ctor_get_uint8(v___y_901_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_919_ = lean_ctor_get(v___y_901_, 13);
v_ref_920_ = l_Lean_replaceRef(v_ref_899_, v_ref_909_);
lean_inc_ref(v_inheritedTraceOptions_919_);
lean_inc(v_cancelTk_x3f_917_);
lean_inc(v_currMacroScope_915_);
lean_inc(v_quotContext_914_);
lean_inc(v_maxHeartbeats_913_);
lean_inc(v_initHeartbeats_912_);
lean_inc(v_openDecls_911_);
lean_inc(v_currNamespace_910_);
lean_inc(v_maxRecDepth_908_);
lean_inc(v_currRecDepth_907_);
lean_inc_ref(v_options_906_);
lean_inc_ref(v_fileMap_905_);
lean_inc_ref(v_fileName_904_);
v___x_921_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_921_, 0, v_fileName_904_);
lean_ctor_set(v___x_921_, 1, v_fileMap_905_);
lean_ctor_set(v___x_921_, 2, v_options_906_);
lean_ctor_set(v___x_921_, 3, v_currRecDepth_907_);
lean_ctor_set(v___x_921_, 4, v_maxRecDepth_908_);
lean_ctor_set(v___x_921_, 5, v_ref_920_);
lean_ctor_set(v___x_921_, 6, v_currNamespace_910_);
lean_ctor_set(v___x_921_, 7, v_openDecls_911_);
lean_ctor_set(v___x_921_, 8, v_initHeartbeats_912_);
lean_ctor_set(v___x_921_, 9, v_maxHeartbeats_913_);
lean_ctor_set(v___x_921_, 10, v_quotContext_914_);
lean_ctor_set(v___x_921_, 11, v_currMacroScope_915_);
lean_ctor_set(v___x_921_, 12, v_cancelTk_x3f_917_);
lean_ctor_set(v___x_921_, 13, v_inheritedTraceOptions_919_);
lean_ctor_set_uint8(v___x_921_, sizeof(void*)*14, v_diag_916_);
lean_ctor_set_uint8(v___x_921_, sizeof(void*)*14 + 1, v_suppressElabErrors_918_);
v___x_922_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v_msg_900_, v___x_921_, v___y_902_);
lean_dec_ref_known(v___x_921_, 14);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg___boxed(lean_object* v_ref_923_, lean_object* v_msg_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_){
_start:
{
lean_object* v_res_928_; 
v_res_928_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg(v_ref_923_, v_msg_924_, v___y_925_, v___y_926_);
lean_dec(v___y_926_);
lean_dec_ref(v___y_925_);
lean_dec(v_ref_923_);
return v_res_928_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_ref_929_, lean_object* v_msg_930_, lean_object* v_declHint_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v___x_935_; lean_object* v_a_936_; lean_object* v___x_937_; 
v___x_935_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8(v_msg_930_, v_declHint_931_, v___y_932_, v___y_933_);
v_a_936_ = lean_ctor_get(v___x_935_, 0);
lean_inc(v_a_936_);
lean_dec_ref(v___x_935_);
v___x_937_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg(v_ref_929_, v_a_936_, v___y_932_, v___y_933_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_ref_938_, lean_object* v_msg_939_, lean_object* v_declHint_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v_res_944_; 
v_res_944_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg(v_ref_938_, v_msg_939_, v_declHint_940_, v___y_941_, v___y_942_);
lean_dec(v___y_942_);
lean_dec_ref(v___y_941_);
lean_dec(v_ref_938_);
return v_res_944_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_946_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__0));
v___x_947_ = l_Lean_stringToMessageData(v___x_946_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(lean_object* v_ref_948_, lean_object* v_constName_949_, lean_object* v___y_950_, lean_object* v___y_951_){
_start:
{
lean_object* v___x_953_; uint8_t v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; 
v___x_953_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___closed__1);
v___x_954_ = 0;
lean_inc(v_constName_949_);
v___x_955_ = l_Lean_MessageData_ofConstName(v_constName_949_, v___x_954_);
v___x_956_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_956_, 0, v___x_953_);
lean_ctor_set(v___x_956_, 1, v___x_955_);
v___x_957_ = lean_obj_once(&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5, &lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5_once, _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5);
v___x_958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_958_, 0, v___x_956_);
lean_ctor_set(v___x_958_, 1, v___x_957_);
v___x_959_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg(v_ref_948_, v___x_958_, v_constName_949_, v___y_950_, v___y_951_);
return v___x_959_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_960_, lean_object* v_constName_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_ref_960_, v_constName_961_, v___y_962_, v___y_963_);
lean_dec(v___y_963_);
lean_dec_ref(v___y_962_);
lean_dec(v_ref_960_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_constName_966_, lean_object* v___y_967_, lean_object* v___y_968_){
_start:
{
lean_object* v_ref_970_; lean_object* v___x_971_; 
v_ref_970_ = lean_ctor_get(v___y_967_, 5);
v___x_971_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_ref_970_, v_constName_966_, v___y_967_, v___y_968_);
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_constName_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_){
_start:
{
lean_object* v_res_976_; 
v_res_976_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_972_, v___y_973_, v___y_974_);
lean_dec(v___y_974_);
lean_dec_ref(v___y_973_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4(lean_object* v_constName_977_, lean_object* v___y_978_, lean_object* v___y_979_){
_start:
{
lean_object* v___x_981_; lean_object* v_env_982_; uint8_t v___x_983_; lean_object* v___x_984_; 
v___x_981_ = lean_st_ref_get(v___y_979_);
v_env_982_ = lean_ctor_get(v___x_981_, 0);
lean_inc_ref(v_env_982_);
lean_dec(v___x_981_);
v___x_983_ = 0;
lean_inc(v_constName_977_);
v___x_984_ = l_Lean_Environment_findConstVal_x3f(v_env_982_, v_constName_977_, v___x_983_);
if (lean_obj_tag(v___x_984_) == 0)
{
lean_object* v___x_985_; 
v___x_985_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_977_, v___y_978_, v___y_979_);
return v___x_985_;
}
else
{
lean_object* v_val_986_; lean_object* v___x_988_; uint8_t v_isShared_989_; uint8_t v_isSharedCheck_993_; 
lean_dec(v_constName_977_);
v_val_986_ = lean_ctor_get(v___x_984_, 0);
v_isSharedCheck_993_ = !lean_is_exclusive(v___x_984_);
if (v_isSharedCheck_993_ == 0)
{
v___x_988_ = v___x_984_;
v_isShared_989_ = v_isSharedCheck_993_;
goto v_resetjp_987_;
}
else
{
lean_inc(v_val_986_);
lean_dec(v___x_984_);
v___x_988_ = lean_box(0);
v_isShared_989_ = v_isSharedCheck_993_;
goto v_resetjp_987_;
}
v_resetjp_987_:
{
lean_object* v___x_991_; 
if (v_isShared_989_ == 0)
{
lean_ctor_set_tag(v___x_988_, 0);
v___x_991_ = v___x_988_;
goto v_reusejp_990_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v_val_986_);
v___x_991_ = v_reuseFailAlloc_992_;
goto v_reusejp_990_;
}
v_reusejp_990_:
{
return v___x_991_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4___boxed(lean_object* v_constName_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_){
_start:
{
lean_object* v_res_998_; 
v_res_998_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4(v_constName_994_, v___y_995_, v___y_996_);
lean_dec(v___y_996_);
lean_dec_ref(v___y_995_);
return v_res_998_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__5(lean_object* v_a_999_, lean_object* v_a_1000_){
_start:
{
if (lean_obj_tag(v_a_999_) == 0)
{
lean_object* v___x_1001_; 
v___x_1001_ = l_List_reverse___redArg(v_a_1000_);
return v___x_1001_;
}
else
{
lean_object* v_head_1002_; lean_object* v_tail_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1012_; 
v_head_1002_ = lean_ctor_get(v_a_999_, 0);
v_tail_1003_ = lean_ctor_get(v_a_999_, 1);
v_isSharedCheck_1012_ = !lean_is_exclusive(v_a_999_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_1005_ = v_a_999_;
v_isShared_1006_ = v_isSharedCheck_1012_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_tail_1003_);
lean_inc(v_head_1002_);
lean_dec(v_a_999_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1012_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1007_; lean_object* v___x_1009_; 
v___x_1007_ = l_Lean_mkLevelParam(v_head_1002_);
if (v_isShared_1006_ == 0)
{
lean_ctor_set(v___x_1005_, 1, v_a_1000_);
lean_ctor_set(v___x_1005_, 0, v___x_1007_);
v___x_1009_ = v___x_1005_;
goto v_reusejp_1008_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_1007_);
lean_ctor_set(v_reuseFailAlloc_1011_, 1, v_a_1000_);
v___x_1009_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1008_;
}
v_reusejp_1008_:
{
v_a_999_ = v_tail_1003_;
v_a_1000_ = v___x_1009_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_constName_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_){
_start:
{
lean_object* v___x_1017_; 
lean_inc(v_constName_1013_);
v___x_1017_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__4(v_constName_1013_, v___y_1014_, v___y_1015_);
if (lean_obj_tag(v___x_1017_) == 0)
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1029_; 
v_a_1018_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1020_ = v___x_1017_;
v_isShared_1021_ = v_isSharedCheck_1029_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_1017_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1029_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v_levelParams_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1027_; 
v_levelParams_1022_ = lean_ctor_get(v_a_1018_, 1);
lean_inc(v_levelParams_1022_);
lean_dec(v_a_1018_);
v___x_1023_ = lean_box(0);
v___x_1024_ = lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2_spec__5(v_levelParams_1022_, v___x_1023_);
v___x_1025_ = l_Lean_mkConst(v_constName_1013_, v___x_1024_);
if (v_isShared_1021_ == 0)
{
lean_ctor_set(v___x_1020_, 0, v___x_1025_);
v___x_1027_ = v___x_1020_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v___x_1025_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
else
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1037_; 
lean_dec(v_constName_1013_);
v_a_1030_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1037_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1037_ == 0)
{
v___x_1032_ = v___x_1017_;
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_1017_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1035_; 
if (v_isShared_1033_ == 0)
{
v___x_1035_ = v___x_1032_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_a_1030_);
v___x_1035_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
return v___x_1035_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_constName_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v_res_1042_; 
v_res_1042_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2(v_constName_1038_, v___y_1039_, v___y_1040_);
lean_dec(v___y_1040_);
lean_dec_ref(v___y_1039_);
return v_res_1042_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1(lean_object* v_stx_1043_, lean_object* v_n_1044_, lean_object* v_expectedType_x3f_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_){
_start:
{
lean_object* v___x_1049_; 
v___x_1049_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__2(v_n_1044_, v___y_1046_, v___y_1047_);
if (lean_obj_tag(v___x_1049_) == 0)
{
lean_object* v_a_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; uint8_t v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v_a_1050_ = lean_ctor_get(v___x_1049_, 0);
lean_inc(v_a_1050_);
lean_dec_ref_known(v___x_1049_, 1);
v___x_1051_ = lean_box(0);
v___x_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1051_);
lean_ctor_set(v___x_1052_, 1, v_stx_1043_);
v___x_1053_ = l_Lean_LocalContext_empty;
v___x_1054_ = 0;
v___x_1055_ = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(v___x_1055_, 0, v___x_1052_);
lean_ctor_set(v___x_1055_, 1, v___x_1053_);
lean_ctor_set(v___x_1055_, 2, v_expectedType_x3f_1045_);
lean_ctor_set(v___x_1055_, 3, v_a_1050_);
lean_ctor_set_uint8(v___x_1055_, sizeof(void*)*4, v___x_1054_);
lean_ctor_set_uint8(v___x_1055_, sizeof(void*)*4 + 1, v___x_1054_);
v___x_1056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1055_);
v___x_1057_ = lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3(v___x_1056_, v___y_1046_, v___y_1047_);
return v___x_1057_;
}
else
{
lean_object* v_a_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1065_; 
lean_dec(v_expectedType_x3f_1045_);
lean_dec(v_stx_1043_);
v_a_1058_ = lean_ctor_get(v___x_1049_, 0);
v_isSharedCheck_1065_ = !lean_is_exclusive(v___x_1049_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1060_ = v___x_1049_;
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_a_1058_);
lean_dec(v___x_1049_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v___x_1063_; 
if (v_isShared_1061_ == 0)
{
v___x_1063_ = v___x_1060_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1064_; 
v_reuseFailAlloc_1064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1064_, 0, v_a_1058_);
v___x_1063_ = v_reuseFailAlloc_1064_;
goto v_reusejp_1062_;
}
v_reusejp_1062_:
{
return v___x_1063_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1___boxed(lean_object* v_stx_1066_, lean_object* v_n_1067_, lean_object* v_expectedType_x3f_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_){
_start:
{
lean_object* v_res_1072_; 
v_res_1072_ = lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1(v_stx_1066_, v_n_1067_, v_expectedType_x3f_1068_, v___y_1069_, v___y_1070_);
lean_dec(v___y_1070_);
lean_dec_ref(v___y_1069_);
return v_res_1072_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0(lean_object* v_constName_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_){
_start:
{
lean_object* v___x_1077_; lean_object* v_env_1078_; uint8_t v___x_1079_; lean_object* v___x_1080_; 
v___x_1077_ = lean_st_ref_get(v___y_1075_);
v_env_1078_ = lean_ctor_get(v___x_1077_, 0);
lean_inc_ref(v_env_1078_);
lean_dec(v___x_1077_);
v___x_1079_ = 0;
lean_inc(v_constName_1073_);
v___x_1080_ = l_Lean_Environment_find_x3f(v_env_1078_, v_constName_1073_, v___x_1079_);
if (lean_obj_tag(v___x_1080_) == 0)
{
lean_object* v___x_1081_; 
v___x_1081_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_1073_, v___y_1074_, v___y_1075_);
return v___x_1081_;
}
else
{
lean_object* v_val_1082_; lean_object* v___x_1084_; uint8_t v_isShared_1085_; uint8_t v_isSharedCheck_1089_; 
lean_dec(v_constName_1073_);
v_val_1082_ = lean_ctor_get(v___x_1080_, 0);
v_isSharedCheck_1089_ = !lean_is_exclusive(v___x_1080_);
if (v_isSharedCheck_1089_ == 0)
{
v___x_1084_ = v___x_1080_;
v_isShared_1085_ = v_isSharedCheck_1089_;
goto v_resetjp_1083_;
}
else
{
lean_inc(v_val_1082_);
lean_dec(v___x_1080_);
v___x_1084_ = lean_box(0);
v_isShared_1085_ = v_isSharedCheck_1089_;
goto v_resetjp_1083_;
}
v_resetjp_1083_:
{
lean_object* v___x_1087_; 
if (v_isShared_1085_ == 0)
{
lean_ctor_set_tag(v___x_1084_, 0);
v___x_1087_ = v___x_1084_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v_val_1082_);
v___x_1087_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
return v___x_1087_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0___boxed(lean_object* v_constName_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
lean_object* v_res_1094_; 
v_res_1094_ = lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0(v_constName_1090_, v___y_1091_, v___y_1092_);
lean_dec(v___y_1092_);
lean_dec_ref(v___y_1091_);
return v_res_1094_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1095_; 
v___x_1095_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1095_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; 
v___x_1096_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1097_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1097_, 0, v___x_1096_);
return v___x_1097_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1098_; lean_object* v___x_1099_; 
v___x_1098_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1099_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1098_);
lean_ctor_set(v___x_1099_, 1, v___x_1098_);
return v___x_1099_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; 
v___x_1101_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1102_ = l_Lean_stringToMessageData(v___x_1101_);
return v___x_1102_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1105_ = l_Lean_stringToMessageData(v___x_1104_);
return v___x_1105_;
}
}
static uint64_t _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1112_; uint64_t v___x_1113_; 
v___x_1112_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1113_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1112_);
return v___x_1113_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; 
v___x_1114_ = lean_uint64_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1115_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1116_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1116_, 0, v___x_1115_);
lean_ctor_set_uint64(v___x_1116_, sizeof(void*)*1, v___x_1114_);
return v___x_1116_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1117_; 
v___x_1117_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1117_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1118_; lean_object* v___x_1119_; 
v___x_1118_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1118_);
return v___x_1119_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1120_; lean_object* v___x_1121_; 
v___x_1120_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1121_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1121_, 0, v___x_1120_);
lean_ctor_set(v___x_1121_, 1, v___x_1120_);
lean_ctor_set(v___x_1121_, 2, v___x_1120_);
lean_ctor_set(v___x_1121_, 3, v___x_1120_);
lean_ctor_set(v___x_1121_, 4, v___x_1120_);
lean_ctor_set(v___x_1121_, 5, v___x_1120_);
return v___x_1121_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1122_; lean_object* v___x_1123_; 
v___x_1122_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1123_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1123_, 0, v___x_1122_);
lean_ctor_set(v___x_1123_, 1, v___x_1122_);
lean_ctor_set(v___x_1123_, 2, v___x_1122_);
lean_ctor_set(v___x_1123_, 3, v___x_1122_);
lean_ctor_set(v___x_1123_, 4, v___x_1122_);
return v___x_1123_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; 
v___x_1125_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__14_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1126_ = l_Lean_stringToMessageData(v___x_1125_);
return v___x_1126_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1128_; lean_object* v___x_1129_; 
v___x_1128_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__16_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1129_ = l_Lean_stringToMessageData(v___x_1128_);
return v___x_1129_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(lean_object* v___x_1130_, lean_object* v___x_1131_, lean_object* v___x_1132_, lean_object* v___x_1133_, lean_object* v___x_1134_, lean_object* v___x_1135_, lean_object* v___x_1136_, lean_object* v___x_1137_, lean_object* v_decl_1138_, lean_object* v_stx_1139_, uint8_t v_kind_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v___x_1144_; lean_object* v___x_1145_; uint8_t v_dflt_1146_; lean_object* v___y_1148_; lean_object* v___y_1177_; uint8_t v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; uint8_t v_a_1182_; lean_object* v___y_1197_; lean_object* v___y_1198_; uint8_t v___x_1274_; uint8_t v___x_1275_; 
v___x_1144_ = lean_unsigned_to_nat(1u);
v___x_1145_ = l_Lean_Syntax_getArg(v_stx_1139_, v___x_1144_);
v_dflt_1146_ = l_Lean_Syntax_isNone(v___x_1145_);
lean_dec(v___x_1145_);
v___x_1274_ = 0;
v___x_1275_ = l_Lean_instBEqAttributeKind_beq(v_kind_1140_, v___x_1274_);
if (v___x_1275_ == 0)
{
lean_object* v___x_1276_; 
lean_dec(v_stx_1139_);
lean_dec(v_decl_1138_);
lean_dec(v___x_1137_);
lean_dec_ref(v___x_1136_);
lean_dec_ref(v___x_1135_);
lean_dec_ref(v___x_1134_);
lean_dec(v___x_1133_);
lean_dec(v___x_1132_);
lean_dec(v___x_1130_);
v___x_1276_ = lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg(v___x_1131_, v_kind_1140_, v___y_1141_, v___y_1142_);
return v___x_1276_;
}
else
{
goto v___jp_1248_;
}
v___jp_1147_:
{
lean_object* v___x_1149_; lean_object* v_env_1150_; lean_object* v_nextMacroScope_1151_; lean_object* v_ngen_1152_; lean_object* v_auxDeclNGen_1153_; lean_object* v_traceState_1154_; lean_object* v_messages_1155_; lean_object* v_infoState_1156_; lean_object* v_snapshotTasks_1157_; lean_object* v___x_1159_; uint8_t v_isShared_1160_; uint8_t v_isSharedCheck_1174_; 
v___x_1149_ = lean_st_ref_take(v___y_1148_);
v_env_1150_ = lean_ctor_get(v___x_1149_, 0);
v_nextMacroScope_1151_ = lean_ctor_get(v___x_1149_, 1);
v_ngen_1152_ = lean_ctor_get(v___x_1149_, 2);
v_auxDeclNGen_1153_ = lean_ctor_get(v___x_1149_, 3);
v_traceState_1154_ = lean_ctor_get(v___x_1149_, 4);
v_messages_1155_ = lean_ctor_get(v___x_1149_, 6);
v_infoState_1156_ = lean_ctor_get(v___x_1149_, 7);
v_snapshotTasks_1157_ = lean_ctor_get(v___x_1149_, 8);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___x_1149_);
if (v_isSharedCheck_1174_ == 0)
{
lean_object* v_unused_1175_; 
v_unused_1175_ = lean_ctor_get(v___x_1149_, 5);
lean_dec(v_unused_1175_);
v___x_1159_ = v___x_1149_;
v_isShared_1160_ = v_isSharedCheck_1174_;
goto v_resetjp_1158_;
}
else
{
lean_inc(v_snapshotTasks_1157_);
lean_inc(v_infoState_1156_);
lean_inc(v_messages_1155_);
lean_inc(v_traceState_1154_);
lean_inc(v_auxDeclNGen_1153_);
lean_inc(v_ngen_1152_);
lean_inc(v_nextMacroScope_1151_);
lean_inc(v_env_1150_);
lean_dec(v___x_1149_);
v___x_1159_ = lean_box(0);
v_isShared_1160_ = v_isSharedCheck_1174_;
goto v_resetjp_1158_;
}
v_resetjp_1158_:
{
lean_object* v___x_1161_; lean_object* v_toEnvExtension_1162_; lean_object* v_asyncMode_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1169_; 
v___x_1161_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_1162_ = lean_ctor_get(v___x_1161_, 0);
v_asyncMode_1163_ = lean_ctor_get(v_toEnvExtension_1162_, 2);
v___x_1164_ = lean_box(v_dflt_1146_);
v___x_1165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1165_, 0, v_decl_1138_);
lean_ctor_set(v___x_1165_, 1, v___x_1164_);
v___x_1166_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_1161_, v_env_1150_, v___x_1165_, v_asyncMode_1163_, v___x_1130_);
v___x_1167_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
if (v_isShared_1160_ == 0)
{
lean_ctor_set(v___x_1159_, 5, v___x_1167_);
lean_ctor_set(v___x_1159_, 0, v___x_1166_);
v___x_1169_ = v___x_1159_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v___x_1166_);
lean_ctor_set(v_reuseFailAlloc_1173_, 1, v_nextMacroScope_1151_);
lean_ctor_set(v_reuseFailAlloc_1173_, 2, v_ngen_1152_);
lean_ctor_set(v_reuseFailAlloc_1173_, 3, v_auxDeclNGen_1153_);
lean_ctor_set(v_reuseFailAlloc_1173_, 4, v_traceState_1154_);
lean_ctor_set(v_reuseFailAlloc_1173_, 5, v___x_1167_);
lean_ctor_set(v_reuseFailAlloc_1173_, 6, v_messages_1155_);
lean_ctor_set(v_reuseFailAlloc_1173_, 7, v_infoState_1156_);
lean_ctor_set(v_reuseFailAlloc_1173_, 8, v_snapshotTasks_1157_);
v___x_1169_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; 
v___x_1170_ = lean_st_ref_set(v___y_1148_, v___x_1169_);
v___x_1171_ = lean_box(0);
v___x_1172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1172_, 0, v___x_1171_);
return v___x_1172_;
}
}
}
v___jp_1176_:
{
if (v_a_1182_ == 0)
{
lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; 
lean_dec(v___x_1130_);
v___x_1183_ = lean_obj_once(&lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5, &lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5_once, _init_lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg___closed__5);
v___x_1184_ = l_Lean_MessageData_ofConstName(v_decl_1138_, v___y_1178_);
v___x_1185_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1185_, 0, v___x_1183_);
lean_ctor_set(v___x_1185_, 1, v___x_1184_);
v___x_1186_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1187_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1185_);
lean_ctor_set(v___x_1187_, 1, v___x_1186_);
v___x_1188_ = l_Lean_MessageData_ofConstName(v___y_1179_, v___y_1178_);
v___x_1189_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1187_);
lean_ctor_set(v___x_1189_, 1, v___x_1188_);
v___x_1190_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1189_);
lean_ctor_set(v___x_1191_, 1, v___x_1190_);
v___x_1192_ = l_Lean_MessageData_ofExpr(v___y_1181_);
v___x_1193_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1191_);
lean_ctor_set(v___x_1193_, 1, v___x_1192_);
v___x_1194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1194_, 0, v___x_1193_);
lean_ctor_set(v___x_1194_, 1, v___x_1183_);
v___x_1195_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_1194_, v___y_1180_, v___y_1177_);
return v___x_1195_;
}
else
{
lean_dec_ref(v___y_1181_);
lean_dec(v___y_1179_);
v___y_1148_ = v___y_1177_;
goto v___jp_1147_;
}
}
v___jp_1196_:
{
lean_object* v___x_1199_; 
lean_inc(v_decl_1138_);
v___x_1199_ = l_Lean_ensureAttrDeclIsMeta(v___x_1131_, v_decl_1138_, v_kind_1140_, v___y_1197_, v___y_1198_);
if (lean_obj_tag(v___x_1199_) == 0)
{
lean_object* v___x_1200_; 
lean_dec_ref_known(v___x_1199_, 1);
lean_inc(v_decl_1138_);
v___x_1200_ = lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0(v_decl_1138_, v___y_1197_, v___y_1198_);
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v_a_1201_; uint8_t v___x_1202_; uint8_t v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; size_t v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; 
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
lean_inc(v_a_1201_);
lean_dec_ref_known(v___x_1200_, 1);
v___x_1202_ = 0;
v___x_1203_ = 1;
v___x_1204_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1205_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1206_ = lean_unsigned_to_nat(32u);
v___x_1207_ = lean_mk_empty_array_with_capacity(v___x_1206_);
v___x_1208_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3___closed__3);
v___x_1209_ = ((size_t)5ULL);
lean_inc_n(v___x_1132_, 6);
v___x_1210_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1210_, 0, v___x_1208_);
lean_ctor_set(v___x_1210_, 1, v___x_1207_);
lean_ctor_set(v___x_1210_, 2, v___x_1132_);
lean_ctor_set(v___x_1210_, 3, v___x_1132_);
lean_ctor_set_usize(v___x_1210_, 4, v___x_1209_);
v___x_1211_ = lean_box(1);
lean_inc_ref(v___x_1210_);
v___x_1212_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1205_);
lean_ctor_set(v___x_1212_, 1, v___x_1210_);
lean_ctor_set(v___x_1212_, 2, v___x_1211_);
v___x_1213_ = lean_mk_empty_array_with_capacity(v___x_1132_);
v___x_1214_ = lean_box(0);
lean_inc(v___x_1133_);
v___x_1215_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1215_, 0, v___x_1204_);
lean_ctor_set(v___x_1215_, 1, v___x_1133_);
lean_ctor_set(v___x_1215_, 2, v___x_1212_);
lean_ctor_set(v___x_1215_, 3, v___x_1213_);
lean_ctor_set(v___x_1215_, 4, v___x_1214_);
lean_ctor_set(v___x_1215_, 5, v___x_1132_);
lean_ctor_set(v___x_1215_, 6, v___x_1214_);
lean_ctor_set_uint8(v___x_1215_, sizeof(void*)*7, v___x_1202_);
lean_ctor_set_uint8(v___x_1215_, sizeof(void*)*7 + 1, v___x_1202_);
lean_ctor_set_uint8(v___x_1215_, sizeof(void*)*7 + 2, v___x_1202_);
lean_ctor_set_uint8(v___x_1215_, sizeof(void*)*7 + 3, v___x_1203_);
v___x_1216_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1216_, 0, v___x_1132_);
lean_ctor_set(v___x_1216_, 1, v___x_1132_);
lean_ctor_set(v___x_1216_, 2, v___x_1132_);
lean_ctor_set(v___x_1216_, 3, v___x_1132_);
lean_ctor_set(v___x_1216_, 4, v___x_1205_);
lean_ctor_set(v___x_1216_, 5, v___x_1205_);
lean_ctor_set(v___x_1216_, 6, v___x_1205_);
lean_ctor_set(v___x_1216_, 7, v___x_1205_);
lean_ctor_set(v___x_1216_, 8, v___x_1205_);
lean_ctor_set(v___x_1216_, 9, v___x_1205_);
v___x_1217_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__12_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1218_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__13_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1219_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1216_);
lean_ctor_set(v___x_1219_, 1, v___x_1217_);
lean_ctor_set(v___x_1219_, 2, v___x_1133_);
lean_ctor_set(v___x_1219_, 3, v___x_1210_);
lean_ctor_set(v___x_1219_, 4, v___x_1218_);
v___x_1220_ = lean_st_mk_ref(v___x_1219_);
v___x_1221_ = l_Lean_ConstantInfo_type(v_a_1201_);
lean_dec(v_a_1201_);
v___x_1222_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_getLinter_unsafe__1___closed__8));
v___x_1223_ = l_Lean_Name_mkStr4(v___x_1134_, v___x_1135_, v___x_1136_, v___x_1222_);
v___x_1224_ = lean_box(0);
lean_inc(v___x_1223_);
v___x_1225_ = l_Lean_mkConst(v___x_1223_, v___x_1224_);
lean_inc_ref(v___x_1221_);
v___x_1226_ = l_Lean_Meta_isExprDefEq(v___x_1221_, v___x_1225_, v___x_1215_, v___x_1220_, v___y_1197_, v___y_1198_);
lean_dec_ref_known(v___x_1215_, 7);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v_a_1227_; lean_object* v___x_1228_; uint8_t v___x_1229_; 
v_a_1227_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_a_1227_);
lean_dec_ref_known(v___x_1226_, 1);
v___x_1228_ = lean_st_ref_get(v___x_1220_);
lean_dec(v___x_1220_);
lean_dec(v___x_1228_);
v___x_1229_ = lean_unbox(v_a_1227_);
lean_dec(v_a_1227_);
v___y_1177_ = v___y_1198_;
v___y_1178_ = v___x_1202_;
v___y_1179_ = v___x_1223_;
v___y_1180_ = v___y_1197_;
v___y_1181_ = v___x_1221_;
v_a_1182_ = v___x_1229_;
goto v___jp_1176_;
}
else
{
lean_dec(v___x_1220_);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v_a_1230_; uint8_t v___x_1231_; 
v_a_1230_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_a_1230_);
lean_dec_ref_known(v___x_1226_, 1);
v___x_1231_ = lean_unbox(v_a_1230_);
lean_dec(v_a_1230_);
v___y_1177_ = v___y_1198_;
v___y_1178_ = v___x_1202_;
v___y_1179_ = v___x_1223_;
v___y_1180_ = v___y_1197_;
v___y_1181_ = v___x_1221_;
v_a_1182_ = v___x_1231_;
goto v___jp_1176_;
}
else
{
lean_object* v_a_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
lean_dec(v___x_1223_);
lean_dec_ref(v___x_1221_);
lean_dec(v_decl_1138_);
lean_dec(v___x_1130_);
v_a_1232_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1234_ = v___x_1226_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_a_1232_);
lean_dec(v___x_1226_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v_a_1232_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
return v___x_1237_;
}
}
}
}
}
else
{
lean_object* v_a_1240_; lean_object* v___x_1242_; uint8_t v_isShared_1243_; uint8_t v_isSharedCheck_1247_; 
lean_dec(v_decl_1138_);
lean_dec_ref(v___x_1136_);
lean_dec_ref(v___x_1135_);
lean_dec_ref(v___x_1134_);
lean_dec(v___x_1133_);
lean_dec(v___x_1132_);
lean_dec(v___x_1130_);
v_a_1240_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1247_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1247_ == 0)
{
v___x_1242_ = v___x_1200_;
v_isShared_1243_ = v_isSharedCheck_1247_;
goto v_resetjp_1241_;
}
else
{
lean_inc(v_a_1240_);
lean_dec(v___x_1200_);
v___x_1242_ = lean_box(0);
v_isShared_1243_ = v_isSharedCheck_1247_;
goto v_resetjp_1241_;
}
v_resetjp_1241_:
{
lean_object* v___x_1245_; 
if (v_isShared_1243_ == 0)
{
v___x_1245_ = v___x_1242_;
goto v_reusejp_1244_;
}
else
{
lean_object* v_reuseFailAlloc_1246_; 
v_reuseFailAlloc_1246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1246_, 0, v_a_1240_);
v___x_1245_ = v_reuseFailAlloc_1246_;
goto v_reusejp_1244_;
}
v_reusejp_1244_:
{
return v___x_1245_;
}
}
}
}
else
{
lean_dec(v_decl_1138_);
lean_dec_ref(v___x_1136_);
lean_dec_ref(v___x_1135_);
lean_dec_ref(v___x_1134_);
lean_dec(v___x_1133_);
lean_dec(v___x_1132_);
lean_dec(v___x_1130_);
return v___x_1199_;
}
}
v___jp_1248_:
{
lean_object* v___x_1249_; lean_object* v_env_1250_; lean_object* v___x_1251_; lean_object* v_toEnvExtension_1252_; lean_object* v_asyncMode_1253_; lean_object* v_shortName_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; 
v___x_1249_ = lean_st_ref_get(v___y_1142_);
v_env_1250_ = lean_ctor_get(v___x_1249_, 0);
lean_inc_ref(v_env_1250_);
lean_dec(v___x_1249_);
v___x_1251_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_1252_ = lean_ctor_get(v___x_1251_, 0);
v_asyncMode_1253_ = lean_ctor_get(v_toEnvExtension_1252_, 2);
lean_inc_n(v___x_1130_, 2);
lean_inc(v_decl_1138_);
v_shortName_1254_ = l_Lean_Name_updatePrefix(v_decl_1138_, v___x_1130_);
v___x_1255_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_1137_, v___x_1251_, v_env_1250_, v_asyncMode_1253_, v___x_1130_);
v___x_1256_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_1255_, v_shortName_1254_);
lean_dec(v___x_1255_);
if (lean_obj_tag(v___x_1256_) == 1)
{
lean_object* v_val_1257_; lean_object* v_fst_1258_; lean_object* v___x_1260_; uint8_t v_isShared_1261_; uint8_t v_isSharedCheck_1272_; 
lean_dec(v_decl_1138_);
lean_dec_ref(v___x_1136_);
lean_dec_ref(v___x_1135_);
lean_dec_ref(v___x_1134_);
lean_dec(v___x_1133_);
lean_dec(v___x_1132_);
lean_dec(v___x_1131_);
lean_dec(v___x_1130_);
v_val_1257_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_val_1257_);
lean_dec_ref_known(v___x_1256_, 1);
v_fst_1258_ = lean_ctor_get(v_val_1257_, 0);
v_isSharedCheck_1272_ = !lean_is_exclusive(v_val_1257_);
if (v_isSharedCheck_1272_ == 0)
{
lean_object* v_unused_1273_; 
v_unused_1273_ = lean_ctor_get(v_val_1257_, 1);
lean_dec(v_unused_1273_);
v___x_1260_ = v_val_1257_;
v_isShared_1261_ = v_isSharedCheck_1272_;
goto v_resetjp_1259_;
}
else
{
lean_inc(v_fst_1258_);
lean_dec(v_val_1257_);
v___x_1260_ = lean_box(0);
v_isShared_1261_ = v_isSharedCheck_1272_;
goto v_resetjp_1259_;
}
v_resetjp_1259_:
{
lean_object* v___x_1262_; lean_object* v___x_1263_; 
v___x_1262_ = lean_box(0);
v___x_1263_ = lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1(v_stx_1139_, v_fst_1258_, v___x_1262_, v___y_1141_, v___y_1142_);
if (lean_obj_tag(v___x_1263_) == 0)
{
lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1267_; 
lean_dec_ref_known(v___x_1263_, 1);
v___x_1264_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__15_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1265_ = l_Lean_MessageData_ofName(v_shortName_1254_);
if (v_isShared_1261_ == 0)
{
lean_ctor_set_tag(v___x_1260_, 7);
lean_ctor_set(v___x_1260_, 1, v___x_1265_);
lean_ctor_set(v___x_1260_, 0, v___x_1264_);
v___x_1267_ = v___x_1260_;
goto v_reusejp_1266_;
}
else
{
lean_object* v_reuseFailAlloc_1271_; 
v_reuseFailAlloc_1271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1271_, 0, v___x_1264_);
lean_ctor_set(v_reuseFailAlloc_1271_, 1, v___x_1265_);
v___x_1267_ = v_reuseFailAlloc_1271_;
goto v_reusejp_1266_;
}
v_reusejp_1266_:
{
lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; 
v___x_1268_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__17_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1269_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1269_, 0, v___x_1267_);
lean_ctor_set(v___x_1269_, 1, v___x_1268_);
v___x_1270_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_1269_, v___y_1141_, v___y_1142_);
return v___x_1270_;
}
}
else
{
lean_del_object(v___x_1260_);
lean_dec(v_shortName_1254_);
return v___x_1263_;
}
}
}
else
{
lean_dec(v___x_1256_);
lean_dec(v_shortName_1254_);
lean_dec(v_stx_1139_);
v___y_1197_ = v___y_1141_;
v___y_1198_ = v___y_1142_;
goto v___jp_1196_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object* v___x_1277_, lean_object* v___x_1278_, lean_object* v___x_1279_, lean_object* v___x_1280_, lean_object* v___x_1281_, lean_object* v___x_1282_, lean_object* v___x_1283_, lean_object* v___x_1284_, lean_object* v_decl_1285_, lean_object* v_stx_1286_, lean_object* v_kind_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_){
_start:
{
uint8_t v_kind_boxed_1291_; lean_object* v_res_1292_; 
v_kind_boxed_1291_ = lean_unbox(v_kind_1287_);
v_res_1292_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(v___x_1277_, v___x_1278_, v___x_1279_, v___x_1280_, v___x_1281_, v___x_1282_, v___x_1283_, v___x_1284_, v_decl_1285_, v_stx_1286_, v_kind_boxed_1291_, v___y_1288_, v___y_1289_);
lean_dec(v___y_1289_);
lean_dec_ref(v___y_1288_);
return v_res_1292_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___x_1294_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1295_ = l_Lean_stringToMessageData(v___x_1294_);
return v___x_1295_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1297_; lean_object* v___x_1298_; 
v___x_1297_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1298_ = l_Lean_stringToMessageData(v___x_1297_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(lean_object* v___x_1299_, lean_object* v_decl_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_){
_start:
{
lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; 
v___x_1304_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1305_ = l_Lean_MessageData_ofName(v___x_1299_);
v___x_1306_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1306_, 0, v___x_1304_);
lean_ctor_set(v___x_1306_, 1, v___x_1305_);
v___x_1307_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1308_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1306_);
lean_ctor_set(v___x_1308_, 1, v___x_1307_);
v___x_1309_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_1308_, v___y_1301_, v___y_1302_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object* v___x_1310_, lean_object* v_decl_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(v___x_1310_, v_decl_1311_, v___y_1312_, v___y_1313_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
lean_dec(v_decl_1311_);
return v_res_1315_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1365_ = lean_unsigned_to_nat(3164034710u);
v___x_1366_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1367_ = l_Lean_Name_num___override(v___x_1366_, v___x_1365_);
return v___x_1367_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1369_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1370_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1371_ = l_Lean_Name_str___override(v___x_1370_, v___x_1369_);
return v___x_1371_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___x_1373_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1374_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1375_ = l_Lean_Name_str___override(v___x_1374_, v___x_1373_);
return v___x_1375_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
v___x_1376_ = lean_unsigned_to_nat(2u);
v___x_1377_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1378_ = l_Lean_Name_num___override(v___x_1377_, v___x_1376_);
return v___x_1378_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; 
v___x_1392_ = 0;
v___x_1393_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__28_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1394_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__25_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1395_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1396_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1396_, 0, v___x_1395_);
lean_ctor_set(v___x_1396_, 1, v___x_1394_);
lean_ctor_set(v___x_1396_, 2, v___x_1393_);
lean_ctor_set_uint8(v___x_1396_, sizeof(void*)*3, v___x_1392_);
return v___x_1396_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1397_; lean_object* v___f_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; 
v___f_1397_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__27_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___f_1398_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__26_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_));
v___x_1399_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__29_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1400_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1400_, 0, v___x_1399_);
lean_ctor_set(v___x_1400_, 1, v___f_1398_);
lean_ctor_set(v___x_1400_, 2, v___f_1397_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1402_; lean_object* v___x_1403_; 
v___x_1402_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__30_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
v___x_1403_ = l_Lean_registerBuiltinAttribute(v___x_1402_);
return v___x_1403_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2____boxed(lean_object* v_a_1404_){
_start:
{
lean_object* v_res_1405_; 
v_res_1405_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_();
return v_res_1405_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_1406_, lean_object* v_name_1407_, uint8_t v_kind_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_){
_start:
{
lean_object* v___x_1412_; 
v___x_1412_ = lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___redArg(v_name_1407_, v_kind_1408_, v___y_1409_, v___y_1410_);
return v___x_1412_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_1413_, lean_object* v_name_1414_, lean_object* v_kind_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_){
_start:
{
uint8_t v_kind_boxed_1419_; lean_object* v_res_1420_; 
v_kind_boxed_1419_ = lean_unbox(v_kind_1415_);
v_res_1420_ = lp_batteries_Lean_throwAttrMustBeGlobal___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__2(v_00_u03b1_1413_, v_name_1414_, v_kind_boxed_1419_, v___y_1416_, v___y_1417_);
lean_dec(v___y_1417_);
lean_dec_ref(v___y_1416_);
return v_res_1420_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b1_1421_, lean_object* v_constName_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_){
_start:
{
lean_object* v___x_1426_; 
v___x_1426_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_1422_, v___y_1423_, v___y_1424_);
return v___x_1426_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b1_1427_, lean_object* v_constName_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_){
_start:
{
lean_object* v_res_1432_; 
v_res_1432_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b1_1427_, v_constName_1428_, v___y_1429_, v___y_1430_);
lean_dec(v___y_1430_);
lean_dec_ref(v___y_1429_);
return v_res_1432_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7(lean_object* v_t_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___redArg(v_t_1433_, v___y_1435_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7___boxed(lean_object* v_t_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_){
_start:
{
lean_object* v_res_1442_; 
v_res_1442_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1_spec__3_spec__7(v_t_1438_, v___y_1439_, v___y_1440_);
lean_dec(v___y_1440_);
lean_dec_ref(v___y_1439_);
return v_res_1442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_00_u03b1_1443_, lean_object* v_ref_1444_, lean_object* v_constName_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_){
_start:
{
lean_object* v___x_1449_; 
v___x_1449_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___redArg(v_ref_1444_, v_constName_1445_, v___y_1446_, v___y_1447_);
return v___x_1449_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_1450_, lean_object* v_ref_1451_, lean_object* v_constName_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_){
_start:
{
lean_object* v_res_1456_; 
v_res_1456_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_00_u03b1_1450_, v_ref_1451_, v_constName_1452_, v___y_1453_, v___y_1454_);
lean_dec(v___y_1454_);
lean_dec_ref(v___y_1453_);
lean_dec(v_ref_1451_);
return v_res_1456_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(lean_object* v_00_u03b1_1457_, lean_object* v_ref_1458_, lean_object* v_msg_1459_, lean_object* v_declHint_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_){
_start:
{
lean_object* v___x_1464_; 
v___x_1464_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___redArg(v_ref_1458_, v_msg_1459_, v_declHint_1460_, v___y_1461_, v___y_1462_);
return v___x_1464_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1465_, lean_object* v_ref_1466_, lean_object* v_msg_1467_, lean_object* v_declHint_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_){
_start:
{
lean_object* v_res_1472_; 
v_res_1472_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(v_00_u03b1_1465_, v_ref_1466_, v_msg_1467_, v_declHint_1468_, v___y_1469_, v___y_1470_);
lean_dec(v___y_1470_);
lean_dec_ref(v___y_1469_);
lean_dec(v_ref_1466_);
return v_res_1472_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(lean_object* v_msg_1473_, lean_object* v_declHint_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_){
_start:
{
lean_object* v___x_1478_; 
v___x_1478_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_msg_1473_, v_declHint_1474_, v___y_1476_);
return v___x_1478_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___boxed(lean_object* v_msg_1479_, lean_object* v_declHint_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(v_msg_1479_, v_declHint_1480_, v___y_1481_, v___y_1482_);
lean_dec(v___y_1482_);
lean_dec_ref(v___y_1481_);
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9(lean_object* v_00_u03b1_1485_, lean_object* v_ref_1486_, lean_object* v_msg_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_){
_start:
{
lean_object* v___x_1491_; 
v___x_1491_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___redArg(v_ref_1486_, v_msg_1487_, v___y_1488_, v___y_1489_);
return v___x_1491_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9___boxed(lean_object* v_00_u03b1_1492_, lean_object* v_ref_1493_, lean_object* v_msg_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4_spec__9(v_00_u03b1_1492_, v_ref_1493_, v_msg_1494_, v___y_1495_, v___y_1496_);
lean_dec(v___y_1496_);
lean_dec_ref(v___y_1495_);
lean_dec(v_ref_1493_);
return v_res_1498_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; 
v___x_1537_ = lean_box(0);
v___x_1538_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1539_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1538_);
lean_ctor_set(v___x_1539_, 1, v___x_1537_);
return v___x_1539_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg(){
_start:
{
lean_object* v___x_1541_; lean_object* v___x_1542_; 
v___x_1541_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_1542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1542_, 0, v___x_1541_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v___y_1543_){
_start:
{
lean_object* v_res_1544_; 
v_res_1544_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg();
return v_res_1544_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; 
v___x_1549_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg();
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_){
_start:
{
lean_object* v_res_1554_; 
v_res_1554_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0(v_00_u03b1_1550_, v___y_1551_, v___y_1552_);
lean_dec(v___y_1552_);
lean_dec_ref(v___y_1551_);
return v_res_1554_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(lean_object* v_x_1555_, lean_object* v_x_1556_, lean_object* v_x_1557_, lean_object* v___y_1558_){
_start:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1560_ = lean_box(0);
v___x_1561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1561_, 0, v___x_1560_);
return v___x_1561_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object* v_x_1562_, lean_object* v_x_1563_, lean_object* v_x_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_){
_start:
{
lean_object* v_res_1567_; 
v_res_1567_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(v_x_1562_, v_x_1563_, v_x_1564_, v___y_1565_);
lean_dec(v___y_1565_);
lean_dec_ref(v_x_1564_);
lean_dec_ref(v_x_1563_);
lean_dec(v_x_1562_);
return v_res_1567_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1(size_t v_sz_1568_, size_t v_i_1569_, lean_object* v_bs_1570_){
_start:
{
uint8_t v___x_1571_; 
v___x_1571_ = lean_usize_dec_lt(v_i_1569_, v_sz_1568_);
if (v___x_1571_ == 0)
{
lean_object* v___x_1572_; 
v___x_1572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1572_, 0, v_bs_1570_);
return v___x_1572_;
}
else
{
lean_object* v_v_1573_; lean_object* v___x_1574_; lean_object* v_bs_x27_1575_; size_t v___x_1576_; size_t v___x_1577_; lean_object* v___x_1578_; 
v_v_1573_ = lean_array_uget(v_bs_1570_, v_i_1569_);
v___x_1574_ = lean_unsigned_to_nat(0u);
v_bs_x27_1575_ = lean_array_uset(v_bs_1570_, v_i_1569_, v___x_1574_);
v___x_1576_ = ((size_t)1ULL);
v___x_1577_ = lean_usize_add(v_i_1569_, v___x_1576_);
v___x_1578_ = lean_array_uset(v_bs_x27_1575_, v_i_1569_, v_v_1573_);
v_i_1569_ = v___x_1577_;
v_bs_1570_ = v___x_1578_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1___boxed(lean_object* v_sz_1580_, lean_object* v_i_1581_, lean_object* v_bs_1582_){
_start:
{
size_t v_sz_boxed_1583_; size_t v_i_boxed_1584_; lean_object* v_res_1585_; 
v_sz_boxed_1583_ = lean_unbox_usize(v_sz_1580_);
lean_dec(v_sz_1580_);
v_i_boxed_1584_ = lean_unbox_usize(v_i_1581_);
lean_dec(v_i_1581_);
v_res_1585_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1(v_sz_boxed_1583_, v_i_boxed_1584_, v_bs_1582_);
return v_res_1585_;
}
}
static double _init_lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_1586_; double v___x_1587_; 
v___x_1586_ = lean_unsigned_to_nat(0u);
v___x_1587_ = lean_float_of_nat(v___x_1586_);
return v___x_1587_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4(lean_object* v_cls_1591_, lean_object* v_msg_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_){
_start:
{
lean_object* v_ref_1596_; lean_object* v___x_1597_; lean_object* v_a_1598_; lean_object* v___x_1600_; uint8_t v_isShared_1601_; uint8_t v_isSharedCheck_1642_; 
v_ref_1596_ = lean_ctor_get(v___y_1593_, 5);
v___x_1597_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1_spec__3(v_msg_1592_, v___y_1593_, v___y_1594_);
v_a_1598_ = lean_ctor_get(v___x_1597_, 0);
v_isSharedCheck_1642_ = !lean_is_exclusive(v___x_1597_);
if (v_isSharedCheck_1642_ == 0)
{
v___x_1600_ = v___x_1597_;
v_isShared_1601_ = v_isSharedCheck_1642_;
goto v_resetjp_1599_;
}
else
{
lean_inc(v_a_1598_);
lean_dec(v___x_1597_);
v___x_1600_ = lean_box(0);
v_isShared_1601_ = v_isSharedCheck_1642_;
goto v_resetjp_1599_;
}
v_resetjp_1599_:
{
lean_object* v___x_1602_; lean_object* v_traceState_1603_; lean_object* v_env_1604_; lean_object* v_nextMacroScope_1605_; lean_object* v_ngen_1606_; lean_object* v_auxDeclNGen_1607_; lean_object* v_cache_1608_; lean_object* v_messages_1609_; lean_object* v_infoState_1610_; lean_object* v_snapshotTasks_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1641_; 
v___x_1602_ = lean_st_ref_take(v___y_1594_);
v_traceState_1603_ = lean_ctor_get(v___x_1602_, 4);
v_env_1604_ = lean_ctor_get(v___x_1602_, 0);
v_nextMacroScope_1605_ = lean_ctor_get(v___x_1602_, 1);
v_ngen_1606_ = lean_ctor_get(v___x_1602_, 2);
v_auxDeclNGen_1607_ = lean_ctor_get(v___x_1602_, 3);
v_cache_1608_ = lean_ctor_get(v___x_1602_, 5);
v_messages_1609_ = lean_ctor_get(v___x_1602_, 6);
v_infoState_1610_ = lean_ctor_get(v___x_1602_, 7);
v_snapshotTasks_1611_ = lean_ctor_get(v___x_1602_, 8);
v_isSharedCheck_1641_ = !lean_is_exclusive(v___x_1602_);
if (v_isSharedCheck_1641_ == 0)
{
v___x_1613_ = v___x_1602_;
v_isShared_1614_ = v_isSharedCheck_1641_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_snapshotTasks_1611_);
lean_inc(v_infoState_1610_);
lean_inc(v_messages_1609_);
lean_inc(v_cache_1608_);
lean_inc(v_traceState_1603_);
lean_inc(v_auxDeclNGen_1607_);
lean_inc(v_ngen_1606_);
lean_inc(v_nextMacroScope_1605_);
lean_inc(v_env_1604_);
lean_dec(v___x_1602_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1641_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
uint64_t v_tid_1615_; lean_object* v_traces_1616_; lean_object* v___x_1618_; uint8_t v_isShared_1619_; uint8_t v_isSharedCheck_1640_; 
v_tid_1615_ = lean_ctor_get_uint64(v_traceState_1603_, sizeof(void*)*1);
v_traces_1616_ = lean_ctor_get(v_traceState_1603_, 0);
v_isSharedCheck_1640_ = !lean_is_exclusive(v_traceState_1603_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1618_ = v_traceState_1603_;
v_isShared_1619_ = v_isSharedCheck_1640_;
goto v_resetjp_1617_;
}
else
{
lean_inc(v_traces_1616_);
lean_dec(v_traceState_1603_);
v___x_1618_ = lean_box(0);
v_isShared_1619_ = v_isSharedCheck_1640_;
goto v_resetjp_1617_;
}
v_resetjp_1617_:
{
lean_object* v___x_1620_; double v___x_1621_; uint8_t v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1630_; 
v___x_1620_ = lean_box(0);
v___x_1621_ = lean_float_once(&lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0, &lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0_once, _init_lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__0);
v___x_1622_ = 0;
v___x_1623_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__1));
v___x_1624_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1624_, 0, v_cls_1591_);
lean_ctor_set(v___x_1624_, 1, v___x_1620_);
lean_ctor_set(v___x_1624_, 2, v___x_1623_);
lean_ctor_set_float(v___x_1624_, sizeof(void*)*3, v___x_1621_);
lean_ctor_set_float(v___x_1624_, sizeof(void*)*3 + 8, v___x_1621_);
lean_ctor_set_uint8(v___x_1624_, sizeof(void*)*3 + 16, v___x_1622_);
v___x_1625_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__2));
v___x_1626_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1626_, 0, v___x_1624_);
lean_ctor_set(v___x_1626_, 1, v_a_1598_);
lean_ctor_set(v___x_1626_, 2, v___x_1625_);
lean_inc(v_ref_1596_);
v___x_1627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1627_, 0, v_ref_1596_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = l_Lean_PersistentArray_push___redArg(v_traces_1616_, v___x_1627_);
if (v_isShared_1619_ == 0)
{
lean_ctor_set(v___x_1618_, 0, v___x_1628_);
v___x_1630_ = v___x_1618_;
goto v_reusejp_1629_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v___x_1628_);
lean_ctor_set_uint64(v_reuseFailAlloc_1639_, sizeof(void*)*1, v_tid_1615_);
v___x_1630_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1629_;
}
v_reusejp_1629_:
{
lean_object* v___x_1632_; 
if (v_isShared_1614_ == 0)
{
lean_ctor_set(v___x_1613_, 4, v___x_1630_);
v___x_1632_ = v___x_1613_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1638_; 
v_reuseFailAlloc_1638_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1638_, 0, v_env_1604_);
lean_ctor_set(v_reuseFailAlloc_1638_, 1, v_nextMacroScope_1605_);
lean_ctor_set(v_reuseFailAlloc_1638_, 2, v_ngen_1606_);
lean_ctor_set(v_reuseFailAlloc_1638_, 3, v_auxDeclNGen_1607_);
lean_ctor_set(v_reuseFailAlloc_1638_, 4, v___x_1630_);
lean_ctor_set(v_reuseFailAlloc_1638_, 5, v_cache_1608_);
lean_ctor_set(v_reuseFailAlloc_1638_, 6, v_messages_1609_);
lean_ctor_set(v_reuseFailAlloc_1638_, 7, v_infoState_1610_);
lean_ctor_set(v_reuseFailAlloc_1638_, 8, v_snapshotTasks_1611_);
v___x_1632_ = v_reuseFailAlloc_1638_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1636_; 
v___x_1633_ = lean_st_ref_set(v___y_1594_, v___x_1632_);
v___x_1634_ = lean_box(0);
if (v_isShared_1601_ == 0)
{
lean_ctor_set(v___x_1600_, 0, v___x_1634_);
v___x_1636_ = v___x_1600_;
goto v_reusejp_1635_;
}
else
{
lean_object* v_reuseFailAlloc_1637_; 
v_reuseFailAlloc_1637_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1637_, 0, v___x_1634_);
v___x_1636_ = v_reuseFailAlloc_1637_;
goto v_reusejp_1635_;
}
v_reusejp_1635_:
{
return v___x_1636_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___boxed(lean_object* v_cls_1643_, lean_object* v_msg_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_){
_start:
{
lean_object* v_res_1648_; 
v_res_1648_ = lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4(v_cls_1643_, v_msg_1644_, v___y_1645_, v___y_1646_);
lean_dec(v___y_1646_);
lean_dec_ref(v___y_1645_);
return v_res_1648_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg(lean_object* v_keys_1649_, lean_object* v_i_1650_, lean_object* v_k_1651_){
_start:
{
lean_object* v___x_1652_; uint8_t v___x_1653_; 
v___x_1652_ = lean_array_get_size(v_keys_1649_);
v___x_1653_ = lean_nat_dec_lt(v_i_1650_, v___x_1652_);
if (v___x_1653_ == 0)
{
lean_dec(v_i_1650_);
return v___x_1653_;
}
else
{
lean_object* v_k_x27_1654_; uint8_t v___x_1655_; 
v_k_x27_1654_ = lean_array_fget_borrowed(v_keys_1649_, v_i_1650_);
v___x_1655_ = l_Lean_instBEqExtraModUse_beq(v_k_1651_, v_k_x27_1654_);
if (v___x_1655_ == 0)
{
lean_object* v___x_1656_; lean_object* v___x_1657_; 
v___x_1656_ = lean_unsigned_to_nat(1u);
v___x_1657_ = lean_nat_add(v_i_1650_, v___x_1656_);
lean_dec(v_i_1650_);
v_i_1650_ = v___x_1657_;
goto _start;
}
else
{
lean_dec(v_i_1650_);
return v___x_1655_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg___boxed(lean_object* v_keys_1659_, lean_object* v_i_1660_, lean_object* v_k_1661_){
_start:
{
uint8_t v_res_1662_; lean_object* v_r_1663_; 
v_res_1662_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg(v_keys_1659_, v_i_1660_, v_k_1661_);
lean_dec_ref(v_k_1661_);
lean_dec_ref(v_keys_1659_);
v_r_1663_ = lean_box(v_res_1662_);
return v_r_1663_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg(lean_object* v_x_1664_, size_t v_x_1665_, lean_object* v_x_1666_){
_start:
{
if (lean_obj_tag(v_x_1664_) == 0)
{
lean_object* v_es_1667_; lean_object* v___x_1668_; size_t v___x_1669_; size_t v___x_1670_; lean_object* v_j_1671_; lean_object* v___x_1672_; 
v_es_1667_ = lean_ctor_get(v_x_1664_, 0);
v___x_1668_ = lean_box(2);
v___x_1669_ = ((size_t)31ULL);
v___x_1670_ = lean_usize_land(v_x_1665_, v___x_1669_);
v_j_1671_ = lean_usize_to_nat(v___x_1670_);
v___x_1672_ = lean_array_get_borrowed(v___x_1668_, v_es_1667_, v_j_1671_);
lean_dec(v_j_1671_);
switch(lean_obj_tag(v___x_1672_))
{
case 0:
{
lean_object* v_key_1673_; uint8_t v___x_1674_; 
v_key_1673_ = lean_ctor_get(v___x_1672_, 0);
v___x_1674_ = l_Lean_instBEqExtraModUse_beq(v_x_1666_, v_key_1673_);
return v___x_1674_;
}
case 1:
{
lean_object* v_node_1675_; size_t v___x_1676_; size_t v___x_1677_; 
v_node_1675_ = lean_ctor_get(v___x_1672_, 0);
v___x_1676_ = ((size_t)5ULL);
v___x_1677_ = lean_usize_shift_right(v_x_1665_, v___x_1676_);
v_x_1664_ = v_node_1675_;
v_x_1665_ = v___x_1677_;
goto _start;
}
default: 
{
uint8_t v___x_1679_; 
v___x_1679_ = 0;
return v___x_1679_;
}
}
}
else
{
lean_object* v_ks_1680_; lean_object* v___x_1681_; uint8_t v___x_1682_; 
v_ks_1680_ = lean_ctor_get(v_x_1664_, 0);
v___x_1681_ = lean_unsigned_to_nat(0u);
v___x_1682_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg(v_ks_1680_, v___x_1681_, v_x_1666_);
return v___x_1682_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_x_1683_, lean_object* v_x_1684_, lean_object* v_x_1685_){
_start:
{
size_t v_x_4932__boxed_1686_; uint8_t v_res_1687_; lean_object* v_r_1688_; 
v_x_4932__boxed_1686_ = lean_unbox_usize(v_x_1684_);
lean_dec(v_x_1684_);
v_res_1687_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg(v_x_1683_, v_x_4932__boxed_1686_, v_x_1685_);
lean_dec_ref(v_x_1685_);
lean_dec_ref(v_x_1683_);
v_r_1688_ = lean_box(v_res_1687_);
return v_r_1688_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg(lean_object* v_x_1689_, lean_object* v_x_1690_){
_start:
{
uint64_t v___x_1691_; size_t v___x_1692_; uint8_t v___x_1693_; 
v___x_1691_ = l_Lean_instHashableExtraModUse_hash(v_x_1690_);
v___x_1692_ = lean_uint64_to_usize(v___x_1691_);
v___x_1693_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg(v_x_1689_, v___x_1692_, v_x_1690_);
return v___x_1693_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_x_1694_, lean_object* v_x_1695_){
_start:
{
uint8_t v_res_1696_; lean_object* v_r_1697_; 
v_res_1696_ = lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg(v_x_1694_, v_x_1695_);
lean_dec_ref(v_x_1695_);
lean_dec_ref(v_x_1694_);
v_r_1697_ = lean_box(v_res_1696_);
return v_r_1697_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2(void){
_start:
{
lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; 
v___x_1700_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__1));
v___x_1701_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__0));
v___x_1702_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_1701_, v___x_1700_);
return v___x_1702_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6(void){
_start:
{
lean_object* v___x_1707_; lean_object* v___x_1708_; 
v___x_1707_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__5));
v___x_1708_ = l_Lean_stringToMessageData(v___x_1707_);
return v___x_1708_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8(void){
_start:
{
lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1710_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__7));
v___x_1711_ = l_Lean_stringToMessageData(v___x_1710_);
return v___x_1711_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9(void){
_start:
{
lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1712_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4___closed__1));
v___x_1713_ = l_Lean_stringToMessageData(v___x_1712_);
return v___x_1713_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12(void){
_start:
{
lean_object* v_cls_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; 
v_cls_1717_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__4));
v___x_1718_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__11));
v___x_1719_ = l_Lean_Name_append(v___x_1718_, v_cls_1717_);
return v___x_1719_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14(void){
_start:
{
lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1721_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__13));
v___x_1722_ = l_Lean_stringToMessageData(v___x_1721_);
return v___x_1722_;
}
}
static lean_object* _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16(void){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; 
v___x_1724_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__15));
v___x_1725_ = l_Lean_stringToMessageData(v___x_1724_);
return v___x_1725_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2(lean_object* v_mod_1730_, uint8_t v_isMeta_1731_, lean_object* v_hint_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_){
_start:
{
lean_object* v___x_1736_; lean_object* v_env_1737_; uint8_t v_isExporting_1738_; lean_object* v___x_1739_; lean_object* v_env_1740_; lean_object* v___x_1741_; lean_object* v_entry_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___y_1747_; lean_object* v___x_1772_; uint8_t v___x_1773_; 
v___x_1736_ = lean_st_ref_get(v___y_1734_);
v_env_1737_ = lean_ctor_get(v___x_1736_, 0);
lean_inc_ref(v_env_1737_);
lean_dec(v___x_1736_);
v_isExporting_1738_ = lean_ctor_get_uint8(v_env_1737_, sizeof(void*)*8);
lean_dec_ref(v_env_1737_);
v___x_1739_ = lean_st_ref_get(v___y_1734_);
v_env_1740_ = lean_ctor_get(v___x_1739_, 0);
lean_inc_ref(v_env_1740_);
lean_dec(v___x_1739_);
v___x_1741_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__2);
lean_inc(v_mod_1730_);
v_entry_1742_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_entry_1742_, 0, v_mod_1730_);
lean_ctor_set_uint8(v_entry_1742_, sizeof(void*)*1, v_isExporting_1738_);
lean_ctor_set_uint8(v_entry_1742_, sizeof(void*)*1 + 1, v_isMeta_1731_);
v___x_1743_ = l___private_Lean_ExtraModUses_0__Lean_extraModUses;
v___x_1744_ = lean_box(1);
v___x_1745_ = lean_box(0);
v___x_1772_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_1741_, v___x_1743_, v_env_1740_, v___x_1744_, v___x_1745_);
v___x_1773_ = lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg(v___x_1772_, v_entry_1742_);
lean_dec(v___x_1772_);
if (v___x_1773_ == 0)
{
lean_object* v_options_1774_; uint8_t v_hasTrace_1775_; 
v_options_1774_ = lean_ctor_get(v___y_1733_, 2);
v_hasTrace_1775_ = lean_ctor_get_uint8(v_options_1774_, sizeof(void*)*1);
if (v_hasTrace_1775_ == 0)
{
lean_dec(v_hint_1732_);
lean_dec(v_mod_1730_);
v___y_1747_ = v___y_1734_;
goto v___jp_1746_;
}
else
{
lean_object* v_inheritedTraceOptions_1776_; lean_object* v_cls_1777_; lean_object* v___y_1779_; lean_object* v___y_1780_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___x_1797_; uint8_t v___x_1798_; 
v_inheritedTraceOptions_1776_ = lean_ctor_get(v___y_1733_, 13);
v_cls_1777_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__4));
v___x_1797_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__12);
v___x_1798_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1776_, v_options_1774_, v___x_1797_);
if (v___x_1798_ == 0)
{
lean_dec(v_hint_1732_);
lean_dec(v_mod_1730_);
v___y_1747_ = v___y_1734_;
goto v___jp_1746_;
}
else
{
lean_object* v___x_1799_; lean_object* v___y_1801_; 
v___x_1799_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__14);
if (v_isExporting_1738_ == 0)
{
lean_object* v___x_1808_; 
v___x_1808_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__19));
v___y_1801_ = v___x_1808_;
goto v___jp_1800_;
}
else
{
lean_object* v___x_1809_; 
v___x_1809_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__20));
v___y_1801_ = v___x_1809_;
goto v___jp_1800_;
}
v___jp_1800_:
{
lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; 
lean_inc_ref(v___y_1801_);
v___x_1802_ = l_Lean_stringToMessageData(v___y_1801_);
v___x_1803_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1803_, 0, v___x_1799_);
lean_ctor_set(v___x_1803_, 1, v___x_1802_);
v___x_1804_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__16);
v___x_1805_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1805_, 0, v___x_1803_);
lean_ctor_set(v___x_1805_, 1, v___x_1804_);
if (v_isMeta_1731_ == 0)
{
lean_object* v___x_1806_; 
v___x_1806_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__17));
v___y_1784_ = v___x_1805_;
v___y_1785_ = v___x_1806_;
goto v___jp_1783_;
}
else
{
lean_object* v___x_1807_; 
v___x_1807_ = ((lean_object*)(lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__18));
v___y_1784_ = v___x_1805_;
v___y_1785_ = v___x_1807_;
goto v___jp_1783_;
}
}
}
v___jp_1778_:
{
lean_object* v___x_1781_; lean_object* v___x_1782_; 
v___x_1781_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1781_, 0, v___y_1779_);
lean_ctor_set(v___x_1781_, 1, v___y_1780_);
v___x_1782_ = lp_batteries_Lean_addTrace___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__4(v_cls_1777_, v___x_1781_, v___y_1733_, v___y_1734_);
if (lean_obj_tag(v___x_1782_) == 0)
{
lean_dec_ref_known(v___x_1782_, 1);
v___y_1747_ = v___y_1734_;
goto v___jp_1746_;
}
else
{
lean_dec_ref_known(v_entry_1742_, 1);
return v___x_1782_;
}
}
v___jp_1783_:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; uint8_t v___x_1792_; 
lean_inc_ref(v___y_1785_);
v___x_1786_ = l_Lean_stringToMessageData(v___y_1785_);
v___x_1787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1787_, 0, v___y_1784_);
lean_ctor_set(v___x_1787_, 1, v___x_1786_);
v___x_1788_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__6);
v___x_1789_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1789_, 0, v___x_1787_);
lean_ctor_set(v___x_1789_, 1, v___x_1788_);
v___x_1790_ = l_Lean_MessageData_ofName(v_mod_1730_);
v___x_1791_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1791_, 0, v___x_1789_);
lean_ctor_set(v___x_1791_, 1, v___x_1790_);
v___x_1792_ = l_Lean_Name_isAnonymous(v_hint_1732_);
if (v___x_1792_ == 0)
{
lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; 
v___x_1793_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__8);
v___x_1794_ = l_Lean_MessageData_ofName(v_hint_1732_);
v___x_1795_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1795_, 0, v___x_1793_);
lean_ctor_set(v___x_1795_, 1, v___x_1794_);
v___y_1779_ = v___x_1791_;
v___y_1780_ = v___x_1795_;
goto v___jp_1778_;
}
else
{
lean_object* v___x_1796_; 
lean_dec(v_hint_1732_);
v___x_1796_ = lean_obj_once(&lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9, &lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9_once, _init_lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___closed__9);
v___y_1779_ = v___x_1791_;
v___y_1780_ = v___x_1796_;
goto v___jp_1778_;
}
}
}
}
else
{
lean_object* v___x_1810_; lean_object* v___x_1811_; 
lean_dec_ref_known(v_entry_1742_, 1);
lean_dec(v_hint_1732_);
lean_dec(v_mod_1730_);
v___x_1810_ = lean_box(0);
v___x_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1811_, 0, v___x_1810_);
return v___x_1811_;
}
v___jp_1746_:
{
lean_object* v___x_1748_; lean_object* v_toEnvExtension_1749_; lean_object* v_env_1750_; lean_object* v_nextMacroScope_1751_; lean_object* v_ngen_1752_; lean_object* v_auxDeclNGen_1753_; lean_object* v_traceState_1754_; lean_object* v_messages_1755_; lean_object* v_infoState_1756_; lean_object* v_snapshotTasks_1757_; lean_object* v___x_1759_; uint8_t v_isShared_1760_; uint8_t v_isSharedCheck_1770_; 
v___x_1748_ = lean_st_ref_take(v___y_1747_);
v_toEnvExtension_1749_ = lean_ctor_get(v___x_1743_, 0);
v_env_1750_ = lean_ctor_get(v___x_1748_, 0);
v_nextMacroScope_1751_ = lean_ctor_get(v___x_1748_, 1);
v_ngen_1752_ = lean_ctor_get(v___x_1748_, 2);
v_auxDeclNGen_1753_ = lean_ctor_get(v___x_1748_, 3);
v_traceState_1754_ = lean_ctor_get(v___x_1748_, 4);
v_messages_1755_ = lean_ctor_get(v___x_1748_, 6);
v_infoState_1756_ = lean_ctor_get(v___x_1748_, 7);
v_snapshotTasks_1757_ = lean_ctor_get(v___x_1748_, 8);
v_isSharedCheck_1770_ = !lean_is_exclusive(v___x_1748_);
if (v_isSharedCheck_1770_ == 0)
{
lean_object* v_unused_1771_; 
v_unused_1771_ = lean_ctor_get(v___x_1748_, 5);
lean_dec(v_unused_1771_);
v___x_1759_ = v___x_1748_;
v_isShared_1760_ = v_isSharedCheck_1770_;
goto v_resetjp_1758_;
}
else
{
lean_inc(v_snapshotTasks_1757_);
lean_inc(v_infoState_1756_);
lean_inc(v_messages_1755_);
lean_inc(v_traceState_1754_);
lean_inc(v_auxDeclNGen_1753_);
lean_inc(v_ngen_1752_);
lean_inc(v_nextMacroScope_1751_);
lean_inc(v_env_1750_);
lean_dec(v___x_1748_);
v___x_1759_ = lean_box(0);
v_isShared_1760_ = v_isSharedCheck_1770_;
goto v_resetjp_1758_;
}
v_resetjp_1758_:
{
lean_object* v_asyncMode_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1765_; 
v_asyncMode_1761_ = lean_ctor_get(v_toEnvExtension_1749_, 2);
v___x_1762_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_1743_, v_env_1750_, v_entry_1742_, v_asyncMode_1761_, v___x_1745_);
v___x_1763_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_);
if (v_isShared_1760_ == 0)
{
lean_ctor_set(v___x_1759_, 5, v___x_1763_);
lean_ctor_set(v___x_1759_, 0, v___x_1762_);
v___x_1765_ = v___x_1759_;
goto v_reusejp_1764_;
}
else
{
lean_object* v_reuseFailAlloc_1769_; 
v_reuseFailAlloc_1769_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1769_, 0, v___x_1762_);
lean_ctor_set(v_reuseFailAlloc_1769_, 1, v_nextMacroScope_1751_);
lean_ctor_set(v_reuseFailAlloc_1769_, 2, v_ngen_1752_);
lean_ctor_set(v_reuseFailAlloc_1769_, 3, v_auxDeclNGen_1753_);
lean_ctor_set(v_reuseFailAlloc_1769_, 4, v_traceState_1754_);
lean_ctor_set(v_reuseFailAlloc_1769_, 5, v___x_1763_);
lean_ctor_set(v_reuseFailAlloc_1769_, 6, v_messages_1755_);
lean_ctor_set(v_reuseFailAlloc_1769_, 7, v_infoState_1756_);
lean_ctor_set(v_reuseFailAlloc_1769_, 8, v_snapshotTasks_1757_);
v___x_1765_ = v_reuseFailAlloc_1769_;
goto v_reusejp_1764_;
}
v_reusejp_1764_:
{
lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; 
v___x_1766_ = lean_st_ref_set(v___y_1747_, v___x_1765_);
v___x_1767_ = lean_box(0);
v___x_1768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1768_, 0, v___x_1767_);
return v___x_1768_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2___boxed(lean_object* v_mod_1812_, lean_object* v_isMeta_1813_, lean_object* v_hint_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_){
_start:
{
uint8_t v_isMeta_boxed_1818_; lean_object* v_res_1819_; 
v_isMeta_boxed_1818_ = lean_unbox(v_isMeta_1813_);
v_res_1819_ = lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2(v_mod_1812_, v_isMeta_boxed_1818_, v_hint_1814_, v___y_1815_, v___y_1816_);
lean_dec(v___y_1816_);
lean_dec_ref(v___y_1815_);
return v_res_1819_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3(lean_object* v___x_1820_, lean_object* v_declName_1821_, lean_object* v_as_1822_, size_t v_sz_1823_, size_t v_i_1824_, lean_object* v_b_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_){
_start:
{
uint8_t v___x_1829_; 
v___x_1829_ = lean_usize_dec_lt(v_i_1824_, v_sz_1823_);
if (v___x_1829_ == 0)
{
lean_object* v___x_1830_; 
lean_dec(v_declName_1821_);
v___x_1830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1830_, 0, v_b_1825_);
return v___x_1830_;
}
else
{
lean_object* v___x_1831_; lean_object* v_modules_1832_; lean_object* v___x_1833_; lean_object* v_a_1834_; lean_object* v___x_1835_; lean_object* v_toImport_1836_; lean_object* v_module_1837_; uint8_t v___x_1838_; lean_object* v___x_1839_; 
v___x_1831_ = l_Lean_Environment_header(v___x_1820_);
v_modules_1832_ = lean_ctor_get(v___x_1831_, 3);
lean_inc_ref(v_modules_1832_);
lean_dec_ref(v___x_1831_);
v___x_1833_ = l_Lean_instInhabitedEffectiveImport_default;
v_a_1834_ = lean_array_uget_borrowed(v_as_1822_, v_i_1824_);
v___x_1835_ = lean_array_get(v___x_1833_, v_modules_1832_, v_a_1834_);
lean_dec_ref(v_modules_1832_);
v_toImport_1836_ = lean_ctor_get(v___x_1835_, 0);
lean_inc_ref(v_toImport_1836_);
lean_dec(v___x_1835_);
v_module_1837_ = lean_ctor_get(v_toImport_1836_, 0);
lean_inc(v_module_1837_);
lean_dec_ref(v_toImport_1836_);
v___x_1838_ = 0;
lean_inc(v_declName_1821_);
v___x_1839_ = lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2(v_module_1837_, v___x_1838_, v_declName_1821_, v___y_1826_, v___y_1827_);
if (lean_obj_tag(v___x_1839_) == 0)
{
lean_object* v___x_1840_; size_t v___x_1841_; size_t v___x_1842_; 
lean_dec_ref_known(v___x_1839_, 1);
v___x_1840_ = lean_box(0);
v___x_1841_ = ((size_t)1ULL);
v___x_1842_ = lean_usize_add(v_i_1824_, v___x_1841_);
v_i_1824_ = v___x_1842_;
v_b_1825_ = v___x_1840_;
goto _start;
}
else
{
lean_dec(v_declName_1821_);
return v___x_1839_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3___boxed(lean_object* v___x_1844_, lean_object* v_declName_1845_, lean_object* v_as_1846_, lean_object* v_sz_1847_, lean_object* v_i_1848_, lean_object* v_b_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
size_t v_sz_boxed_1853_; size_t v_i_boxed_1854_; lean_object* v_res_1855_; 
v_sz_boxed_1853_ = lean_unbox_usize(v_sz_1847_);
lean_dec(v_sz_1847_);
v_i_boxed_1854_ = lean_unbox_usize(v_i_1848_);
lean_dec(v_i_1848_);
v_res_1855_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3(v___x_1844_, v_declName_1845_, v_as_1846_, v_sz_boxed_1853_, v_i_boxed_1854_, v_b_1849_, v___y_1850_, v___y_1851_);
lean_dec(v___y_1851_);
lean_dec_ref(v___y_1850_);
lean_dec_ref(v_as_1846_);
lean_dec_ref(v___x_1844_);
return v_res_1855_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg(lean_object* v_a_1856_, lean_object* v_x_1857_){
_start:
{
if (lean_obj_tag(v_x_1857_) == 0)
{
lean_object* v___x_1858_; 
v___x_1858_ = lean_box(0);
return v___x_1858_;
}
else
{
lean_object* v_key_1859_; lean_object* v_value_1860_; lean_object* v_tail_1861_; uint8_t v___x_1862_; 
v_key_1859_ = lean_ctor_get(v_x_1857_, 0);
v_value_1860_ = lean_ctor_get(v_x_1857_, 1);
v_tail_1861_ = lean_ctor_get(v_x_1857_, 2);
v___x_1862_ = lean_name_eq(v_key_1859_, v_a_1856_);
if (v___x_1862_ == 0)
{
v_x_1857_ = v_tail_1861_;
goto _start;
}
else
{
lean_object* v___x_1864_; 
lean_inc(v_value_1860_);
v___x_1864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1864_, 0, v_value_1860_);
return v___x_1864_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg___boxed(lean_object* v_a_1865_, lean_object* v_x_1866_){
_start:
{
lean_object* v_res_1867_; 
v_res_1867_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg(v_a_1865_, v_x_1866_);
lean_dec(v_x_1866_);
lean_dec(v_a_1865_);
return v_res_1867_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg(lean_object* v_m_1868_, lean_object* v_a_1869_){
_start:
{
lean_object* v_buckets_1870_; lean_object* v___x_1871_; uint64_t v___y_1873_; 
v_buckets_1870_ = lean_ctor_get(v_m_1868_, 1);
v___x_1871_ = lean_array_get_size(v_buckets_1870_);
if (lean_obj_tag(v_a_1869_) == 0)
{
uint64_t v___x_1887_; 
v___x_1887_ = 1723ULL;
v___y_1873_ = v___x_1887_;
goto v___jp_1872_;
}
else
{
uint64_t v_hash_1888_; 
v_hash_1888_ = lean_ctor_get_uint64(v_a_1869_, sizeof(void*)*2);
v___y_1873_ = v_hash_1888_;
goto v___jp_1872_;
}
v___jp_1872_:
{
uint64_t v___x_1874_; uint64_t v___x_1875_; uint64_t v_fold_1876_; uint64_t v___x_1877_; uint64_t v___x_1878_; uint64_t v___x_1879_; size_t v___x_1880_; size_t v___x_1881_; size_t v___x_1882_; size_t v___x_1883_; size_t v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; 
v___x_1874_ = 32ULL;
v___x_1875_ = lean_uint64_shift_right(v___y_1873_, v___x_1874_);
v_fold_1876_ = lean_uint64_xor(v___y_1873_, v___x_1875_);
v___x_1877_ = 16ULL;
v___x_1878_ = lean_uint64_shift_right(v_fold_1876_, v___x_1877_);
v___x_1879_ = lean_uint64_xor(v_fold_1876_, v___x_1878_);
v___x_1880_ = lean_uint64_to_usize(v___x_1879_);
v___x_1881_ = lean_usize_of_nat(v___x_1871_);
v___x_1882_ = ((size_t)1ULL);
v___x_1883_ = lean_usize_sub(v___x_1881_, v___x_1882_);
v___x_1884_ = lean_usize_land(v___x_1880_, v___x_1883_);
v___x_1885_ = lean_array_uget_borrowed(v_buckets_1870_, v___x_1884_);
v___x_1886_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg(v_a_1869_, v___x_1885_);
return v___x_1886_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg___boxed(lean_object* v_m_1889_, lean_object* v_a_1890_){
_start:
{
lean_object* v_res_1891_; 
v_res_1891_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg(v_m_1889_, v_a_1890_);
lean_dec(v_a_1890_);
lean_dec_ref(v_m_1889_);
return v_res_1891_;
}
}
static lean_object* _init_lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2(void){
_start:
{
lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; 
v___x_1894_ = ((lean_object*)(lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__1));
v___x_1895_ = ((lean_object*)(lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__0));
v___x_1896_ = l_Std_HashMap_instInhabited(lean_box(0), lean_box(0), v___x_1895_, v___x_1894_);
return v___x_1896_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2(lean_object* v_declName_1899_, uint8_t v_isMeta_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_){
_start:
{
lean_object* v___x_1904_; lean_object* v_env_1908_; lean_object* v___y_1910_; lean_object* v___x_1923_; 
v___x_1904_ = lean_st_ref_get(v___y_1902_);
v_env_1908_ = lean_ctor_get(v___x_1904_, 0);
lean_inc_ref(v_env_1908_);
lean_dec(v___x_1904_);
v___x_1923_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1908_, v_declName_1899_);
if (lean_obj_tag(v___x_1923_) == 0)
{
lean_dec_ref(v_env_1908_);
lean_dec(v_declName_1899_);
goto v___jp_1905_;
}
else
{
lean_object* v_val_1924_; lean_object* v___x_1925_; lean_object* v_modules_1926_; lean_object* v___x_1927_; uint8_t v___x_1928_; 
v_val_1924_ = lean_ctor_get(v___x_1923_, 0);
lean_inc(v_val_1924_);
lean_dec_ref_known(v___x_1923_, 1);
v___x_1925_ = l_Lean_Environment_header(v_env_1908_);
v_modules_1926_ = lean_ctor_get(v___x_1925_, 3);
lean_inc_ref(v_modules_1926_);
lean_dec_ref(v___x_1925_);
v___x_1927_ = lean_array_get_size(v_modules_1926_);
v___x_1928_ = lean_nat_dec_lt(v_val_1924_, v___x_1927_);
if (v___x_1928_ == 0)
{
lean_dec_ref(v_modules_1926_);
lean_dec(v_val_1924_);
lean_dec_ref(v_env_1908_);
lean_dec(v_declName_1899_);
goto v___jp_1905_;
}
else
{
lean_object* v___x_1929_; lean_object* v_env_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; uint8_t v___y_1934_; 
v___x_1929_ = lean_st_ref_get(v___y_1902_);
v_env_1930_ = lean_ctor_get(v___x_1929_, 0);
lean_inc_ref(v_env_1930_);
lean_dec(v___x_1929_);
v___x_1931_ = lean_obj_once(&lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2, &lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2_once, _init_lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__2);
v___x_1932_ = lean_array_fget(v_modules_1926_, v_val_1924_);
lean_dec(v_val_1924_);
lean_dec_ref(v_modules_1926_);
if (v_isMeta_1900_ == 0)
{
lean_dec_ref(v_env_1930_);
v___y_1934_ = v_isMeta_1900_;
goto v___jp_1933_;
}
else
{
uint8_t v___x_1945_; 
lean_inc(v_declName_1899_);
v___x_1945_ = l_Lean_isMarkedMeta(v_env_1930_, v_declName_1899_);
if (v___x_1945_ == 0)
{
v___y_1934_ = v_isMeta_1900_;
goto v___jp_1933_;
}
else
{
uint8_t v___x_1946_; 
v___x_1946_ = 0;
v___y_1934_ = v___x_1946_;
goto v___jp_1933_;
}
}
v___jp_1933_:
{
lean_object* v_toImport_1935_; lean_object* v_module_1936_; lean_object* v___x_1937_; 
v_toImport_1935_ = lean_ctor_get(v___x_1932_, 0);
lean_inc_ref(v_toImport_1935_);
lean_dec(v___x_1932_);
v_module_1936_ = lean_ctor_get(v_toImport_1935_, 0);
lean_inc(v_module_1936_);
lean_dec_ref(v_toImport_1935_);
lean_inc(v_declName_1899_);
v___x_1937_ = lp_batteries___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2(v_module_1936_, v___y_1934_, v_declName_1899_, v___y_1901_, v___y_1902_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; 
lean_dec_ref_known(v___x_1937_, 1);
v___x_1938_ = l_Lean_indirectModUseExt;
v___x_1939_ = lean_box(1);
v___x_1940_ = lean_box(0);
lean_inc_ref(v_env_1908_);
v___x_1941_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_1931_, v___x_1938_, v_env_1908_, v___x_1939_, v___x_1940_);
v___x_1942_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg(v___x_1941_, v_declName_1899_);
lean_dec(v___x_1941_);
if (lean_obj_tag(v___x_1942_) == 0)
{
lean_object* v___x_1943_; 
v___x_1943_ = ((lean_object*)(lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__3));
v___y_1910_ = v___x_1943_;
goto v___jp_1909_;
}
else
{
lean_object* v_val_1944_; 
v_val_1944_ = lean_ctor_get(v___x_1942_, 0);
lean_inc(v_val_1944_);
lean_dec_ref_known(v___x_1942_, 1);
v___y_1910_ = v_val_1944_;
goto v___jp_1909_;
}
}
else
{
lean_dec_ref(v_env_1908_);
lean_dec(v_declName_1899_);
return v___x_1937_;
}
}
}
}
v___jp_1905_:
{
lean_object* v___x_1906_; lean_object* v___x_1907_; 
v___x_1906_ = lean_box(0);
v___x_1907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1907_, 0, v___x_1906_);
return v___x_1907_;
}
v___jp_1909_:
{
lean_object* v___x_1911_; size_t v_sz_1912_; size_t v___x_1913_; lean_object* v___x_1914_; 
v___x_1911_ = lean_box(0);
v_sz_1912_ = lean_array_size(v___y_1910_);
v___x_1913_ = ((size_t)0ULL);
v___x_1914_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__3(v_env_1908_, v_declName_1899_, v___y_1910_, v_sz_1912_, v___x_1913_, v___x_1911_, v___y_1901_, v___y_1902_);
lean_dec_ref(v___y_1910_);
lean_dec_ref(v_env_1908_);
if (lean_obj_tag(v___x_1914_) == 0)
{
lean_object* v___x_1916_; uint8_t v_isShared_1917_; uint8_t v_isSharedCheck_1921_; 
v_isSharedCheck_1921_ = !lean_is_exclusive(v___x_1914_);
if (v_isSharedCheck_1921_ == 0)
{
lean_object* v_unused_1922_; 
v_unused_1922_ = lean_ctor_get(v___x_1914_, 0);
lean_dec(v_unused_1922_);
v___x_1916_ = v___x_1914_;
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
else
{
lean_dec(v___x_1914_);
v___x_1916_ = lean_box(0);
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
v_resetjp_1915_:
{
lean_object* v___x_1919_; 
if (v_isShared_1917_ == 0)
{
lean_ctor_set(v___x_1916_, 0, v___x_1911_);
v___x_1919_ = v___x_1916_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1920_; 
v_reuseFailAlloc_1920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1920_, 0, v___x_1911_);
v___x_1919_ = v_reuseFailAlloc_1920_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
return v___x_1919_;
}
}
}
else
{
return v___x_1914_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___boxed(lean_object* v_declName_1947_, lean_object* v_isMeta_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_){
_start:
{
uint8_t v_isMeta_boxed_1952_; lean_object* v_res_1953_; 
v_isMeta_boxed_1952_ = lean_unbox(v_isMeta_1948_);
v_res_1953_ = lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2(v_declName_1947_, v_isMeta_boxed_1952_, v___y_1949_, v___y_1950_);
lean_dec(v___y_1950_);
lean_dec_ref(v___y_1949_);
return v_res_1953_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1(void){
_start:
{
lean_object* v___x_1955_; lean_object* v___x_1956_; 
v___x_1955_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__0));
v___x_1956_ = l_Lean_stringToMessageData(v___x_1955_);
return v___x_1956_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3(void){
_start:
{
lean_object* v___x_1958_; lean_object* v___x_1959_; 
v___x_1958_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__2));
v___x_1959_ = l_Lean_stringToMessageData(v___x_1958_);
return v___x_1959_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3(size_t v_sz_1960_, size_t v_i_1961_, lean_object* v_bs_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_){
_start:
{
uint8_t v___x_1966_; 
v___x_1966_ = lean_usize_dec_lt(v_i_1961_, v_sz_1960_);
if (v___x_1966_ == 0)
{
lean_object* v___x_1967_; 
v___x_1967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1967_, 0, v_bs_1962_);
return v___x_1967_;
}
else
{
lean_object* v___x_1968_; lean_object* v_fileName_1969_; lean_object* v_fileMap_1970_; lean_object* v_options_1971_; lean_object* v_currRecDepth_1972_; lean_object* v_maxRecDepth_1973_; lean_object* v_ref_1974_; lean_object* v_currNamespace_1975_; lean_object* v_openDecls_1976_; lean_object* v_initHeartbeats_1977_; lean_object* v_maxHeartbeats_1978_; lean_object* v_quotContext_1979_; lean_object* v_currMacroScope_1980_; uint8_t v_diag_1981_; lean_object* v_cancelTk_x3f_1982_; uint8_t v_suppressElabErrors_1983_; lean_object* v_inheritedTraceOptions_1984_; lean_object* v_env_1985_; lean_object* v___x_1986_; lean_object* v_toEnvExtension_1987_; lean_object* v_asyncMode_1988_; lean_object* v_v_1989_; lean_object* v___x_1990_; lean_object* v_bs_x27_1991_; lean_object* v_a_1993_; lean_object* v___x_1998_; lean_object* v_shortName_1999_; lean_object* v_ref_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; 
v___x_1968_ = lean_st_ref_get(v___y_1964_);
v_fileName_1969_ = lean_ctor_get(v___y_1963_, 0);
v_fileMap_1970_ = lean_ctor_get(v___y_1963_, 1);
v_options_1971_ = lean_ctor_get(v___y_1963_, 2);
v_currRecDepth_1972_ = lean_ctor_get(v___y_1963_, 3);
v_maxRecDepth_1973_ = lean_ctor_get(v___y_1963_, 4);
v_ref_1974_ = lean_ctor_get(v___y_1963_, 5);
v_currNamespace_1975_ = lean_ctor_get(v___y_1963_, 6);
v_openDecls_1976_ = lean_ctor_get(v___y_1963_, 7);
v_initHeartbeats_1977_ = lean_ctor_get(v___y_1963_, 8);
v_maxHeartbeats_1978_ = lean_ctor_get(v___y_1963_, 9);
v_quotContext_1979_ = lean_ctor_get(v___y_1963_, 10);
v_currMacroScope_1980_ = lean_ctor_get(v___y_1963_, 11);
v_diag_1981_ = lean_ctor_get_uint8(v___y_1963_, sizeof(void*)*14);
v_cancelTk_x3f_1982_ = lean_ctor_get(v___y_1963_, 12);
v_suppressElabErrors_1983_ = lean_ctor_get_uint8(v___y_1963_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1984_ = lean_ctor_get(v___y_1963_, 13);
v_env_1985_ = lean_ctor_get(v___x_1968_, 0);
lean_inc_ref(v_env_1985_);
lean_dec(v___x_1968_);
v___x_1986_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_1987_ = lean_ctor_get(v___x_1986_, 0);
v_asyncMode_1988_ = lean_ctor_get(v_toEnvExtension_1987_, 2);
v_v_1989_ = lean_array_uget(v_bs_1962_, v_i_1961_);
v___x_1990_ = lean_unsigned_to_nat(0u);
v_bs_x27_1991_ = lean_array_uset(v_bs_1962_, v_i_1961_, v___x_1990_);
v___x_1998_ = l_Lean_Syntax_getId(v_v_1989_);
v_shortName_1999_ = l_Lean_Name_eraseMacroScopes(v___x_1998_);
lean_dec(v___x_1998_);
v_ref_2000_ = l_Lean_replaceRef(v_v_1989_, v_ref_1974_);
lean_inc_ref(v_inheritedTraceOptions_1984_);
lean_inc(v_cancelTk_x3f_1982_);
lean_inc(v_currMacroScope_1980_);
lean_inc(v_quotContext_1979_);
lean_inc(v_maxHeartbeats_1978_);
lean_inc(v_initHeartbeats_1977_);
lean_inc(v_openDecls_1976_);
lean_inc(v_currNamespace_1975_);
lean_inc(v_maxRecDepth_1973_);
lean_inc(v_currRecDepth_1972_);
lean_inc_ref(v_options_1971_);
lean_inc_ref(v_fileMap_1970_);
lean_inc_ref(v_fileName_1969_);
v___x_2001_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2001_, 0, v_fileName_1969_);
lean_ctor_set(v___x_2001_, 1, v_fileMap_1970_);
lean_ctor_set(v___x_2001_, 2, v_options_1971_);
lean_ctor_set(v___x_2001_, 3, v_currRecDepth_1972_);
lean_ctor_set(v___x_2001_, 4, v_maxRecDepth_1973_);
lean_ctor_set(v___x_2001_, 5, v_ref_2000_);
lean_ctor_set(v___x_2001_, 6, v_currNamespace_1975_);
lean_ctor_set(v___x_2001_, 7, v_openDecls_1976_);
lean_ctor_set(v___x_2001_, 8, v_initHeartbeats_1977_);
lean_ctor_set(v___x_2001_, 9, v_maxHeartbeats_1978_);
lean_ctor_set(v___x_2001_, 10, v_quotContext_1979_);
lean_ctor_set(v___x_2001_, 11, v_currMacroScope_1980_);
lean_ctor_set(v___x_2001_, 12, v_cancelTk_x3f_1982_);
lean_ctor_set(v___x_2001_, 13, v_inheritedTraceOptions_1984_);
lean_ctor_set_uint8(v___x_2001_, sizeof(void*)*14, v_diag_1981_);
lean_ctor_set_uint8(v___x_2001_, sizeof(void*)*14 + 1, v_suppressElabErrors_1983_);
v___x_2002_ = lean_box(1);
v___x_2003_ = lean_box(0);
v___x_2004_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_2002_, v___x_1986_, v_env_1985_, v_asyncMode_1988_, v___x_2003_);
v___x_2005_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_2004_, v_shortName_1999_);
lean_dec(v___x_2004_);
if (lean_obj_tag(v___x_2005_) == 1)
{
lean_object* v_val_2006_; lean_object* v_fst_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; 
v_val_2006_ = lean_ctor_get(v___x_2005_, 0);
lean_inc(v_val_2006_);
lean_dec_ref_known(v___x_2005_, 1);
v_fst_2007_ = lean_ctor_get(v_val_2006_, 0);
lean_inc_n(v_fst_2007_, 2);
lean_dec(v_val_2006_);
v___x_2008_ = lean_box(0);
v___x_2009_ = lp_batteries_Lean_Elab_addConstInfo___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2__spec__1(v_v_1989_, v_fst_2007_, v___x_2008_, v___x_2001_, v___y_1964_);
if (lean_obj_tag(v___x_2009_) == 0)
{
uint8_t v___x_2010_; lean_object* v___x_2011_; 
lean_dec_ref_known(v___x_2009_, 1);
v___x_2010_ = 0;
v___x_2011_ = lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2(v_fst_2007_, v___x_2010_, v___x_2001_, v___y_1964_);
lean_dec_ref_known(v___x_2001_, 14);
if (lean_obj_tag(v___x_2011_) == 0)
{
lean_dec_ref_known(v___x_2011_, 1);
v_a_1993_ = v_shortName_1999_;
goto v___jp_1992_;
}
else
{
lean_object* v_a_2012_; lean_object* v___x_2014_; uint8_t v_isShared_2015_; uint8_t v_isSharedCheck_2019_; 
lean_dec(v_shortName_1999_);
lean_dec_ref(v_bs_x27_1991_);
v_a_2012_ = lean_ctor_get(v___x_2011_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___x_2011_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_2014_ = v___x_2011_;
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
else
{
lean_inc(v_a_2012_);
lean_dec(v___x_2011_);
v___x_2014_ = lean_box(0);
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
v_resetjp_2013_:
{
lean_object* v___x_2017_; 
if (v_isShared_2015_ == 0)
{
v___x_2017_ = v___x_2014_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2018_; 
v_reuseFailAlloc_2018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2018_, 0, v_a_2012_);
v___x_2017_ = v_reuseFailAlloc_2018_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
return v___x_2017_;
}
}
}
}
else
{
lean_object* v_a_2020_; lean_object* v___x_2022_; uint8_t v_isShared_2023_; uint8_t v_isSharedCheck_2027_; 
lean_dec(v_fst_2007_);
lean_dec_ref_known(v___x_2001_, 14);
lean_dec(v_shortName_1999_);
lean_dec_ref(v_bs_x27_1991_);
v_a_2020_ = lean_ctor_get(v___x_2009_, 0);
v_isSharedCheck_2027_ = !lean_is_exclusive(v___x_2009_);
if (v_isSharedCheck_2027_ == 0)
{
v___x_2022_ = v___x_2009_;
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
else
{
lean_inc(v_a_2020_);
lean_dec(v___x_2009_);
v___x_2022_ = lean_box(0);
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
v_resetjp_2021_:
{
lean_object* v___x_2025_; 
if (v_isShared_2023_ == 0)
{
v___x_2025_ = v___x_2022_;
goto v_reusejp_2024_;
}
else
{
lean_object* v_reuseFailAlloc_2026_; 
v_reuseFailAlloc_2026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2026_, 0, v_a_2020_);
v___x_2025_ = v_reuseFailAlloc_2026_;
goto v_reusejp_2024_;
}
v_reusejp_2024_:
{
return v___x_2025_;
}
}
}
}
else
{
lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; 
lean_dec(v___x_2005_);
lean_dec(v_v_1989_);
v___x_2028_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__1);
v___x_2029_ = l_Lean_MessageData_ofName(v_shortName_1999_);
v___x_2030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2030_, 0, v___x_2028_);
lean_ctor_set(v___x_2030_, 1, v___x_2029_);
v___x_2031_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___closed__3);
v___x_2032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2032_, 0, v___x_2030_);
lean_ctor_set(v___x_2032_, 1, v___x_2031_);
v___x_2033_ = lp_batteries_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConstCheck___at___00Batteries_Tactic_Lint_getLinter_spec__0_spec__0_spec__1___redArg(v___x_2032_, v___x_2001_, v___y_1964_);
lean_dec_ref_known(v___x_2001_, 14);
if (lean_obj_tag(v___x_2033_) == 0)
{
lean_object* v_a_2034_; 
v_a_2034_ = lean_ctor_get(v___x_2033_, 0);
lean_inc(v_a_2034_);
lean_dec_ref_known(v___x_2033_, 1);
v_a_1993_ = v_a_2034_;
goto v___jp_1992_;
}
else
{
lean_object* v_a_2035_; lean_object* v___x_2037_; uint8_t v_isShared_2038_; uint8_t v_isSharedCheck_2042_; 
lean_dec_ref(v_bs_x27_1991_);
v_a_2035_ = lean_ctor_get(v___x_2033_, 0);
v_isSharedCheck_2042_ = !lean_is_exclusive(v___x_2033_);
if (v_isSharedCheck_2042_ == 0)
{
v___x_2037_ = v___x_2033_;
v_isShared_2038_ = v_isSharedCheck_2042_;
goto v_resetjp_2036_;
}
else
{
lean_inc(v_a_2035_);
lean_dec(v___x_2033_);
v___x_2037_ = lean_box(0);
v_isShared_2038_ = v_isSharedCheck_2042_;
goto v_resetjp_2036_;
}
v_resetjp_2036_:
{
lean_object* v___x_2040_; 
if (v_isShared_2038_ == 0)
{
v___x_2040_ = v___x_2037_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v_a_2035_);
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
v___jp_1992_:
{
size_t v___x_1994_; size_t v___x_1995_; lean_object* v___x_1996_; 
v___x_1994_ = ((size_t)1ULL);
v___x_1995_ = lean_usize_add(v_i_1961_, v___x_1994_);
v___x_1996_ = lean_array_uset(v_bs_x27_1991_, v_i_1961_, v_a_1993_);
v_i_1961_ = v___x_1995_;
v_bs_1962_ = v___x_1996_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3___boxed(lean_object* v_sz_2043_, lean_object* v_i_2044_, lean_object* v_bs_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_){
_start:
{
size_t v_sz_boxed_2049_; size_t v_i_boxed_2050_; lean_object* v_res_2051_; 
v_sz_boxed_2049_ = lean_unbox_usize(v_sz_2043_);
lean_dec(v_sz_2043_);
v_i_boxed_2050_ = lean_unbox_usize(v_i_2044_);
lean_dec(v_i_2044_);
v_res_2051_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3(v_sz_boxed_2049_, v_i_boxed_2050_, v_bs_2045_, v___y_2046_, v___y_2047_);
lean_dec(v___y_2047_);
lean_dec_ref(v___y_2046_);
return v_res_2051_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(lean_object* v___x_2052_, lean_object* v___x_2053_, lean_object* v___x_2054_, lean_object* v___x_2055_, lean_object* v_x_2056_, lean_object* v_x_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_){
_start:
{
lean_object* v___x_2061_; uint8_t v___x_2062_; 
v___x_2061_ = l_Lean_Name_mkStr4(v___x_2052_, v___x_2053_, v___x_2054_, v___x_2055_);
lean_inc(v_x_2057_);
v___x_2062_ = l_Lean_Syntax_isOfKind(v_x_2057_, v___x_2061_);
lean_dec(v___x_2061_);
if (v___x_2062_ == 0)
{
lean_object* v___x_2063_; 
lean_dec(v_x_2057_);
v___x_2063_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg();
return v___x_2063_;
}
else
{
lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; size_t v_sz_2067_; size_t v___x_2068_; lean_object* v___x_2069_; 
v___x_2064_ = lean_unsigned_to_nat(1u);
v___x_2065_ = l_Lean_Syntax_getArg(v_x_2057_, v___x_2064_);
lean_dec(v_x_2057_);
v___x_2066_ = l_Lean_Syntax_getArgs(v___x_2065_);
lean_dec(v___x_2065_);
v_sz_2067_ = lean_array_size(v___x_2066_);
v___x_2068_ = ((size_t)0ULL);
v___x_2069_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__1(v_sz_2067_, v___x_2068_, v___x_2066_);
if (lean_obj_tag(v___x_2069_) == 0)
{
lean_object* v___x_2070_; 
v___x_2070_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__0___redArg();
return v___x_2070_;
}
else
{
lean_object* v_val_2071_; size_t v_sz_2072_; lean_object* v___x_2073_; 
v_val_2071_ = lean_ctor_get(v___x_2069_, 0);
lean_inc(v_val_2071_);
lean_dec_ref_known(v___x_2069_, 1);
v_sz_2072_ = lean_array_size(v_val_2071_);
v___x_2073_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__3(v_sz_2072_, v___x_2068_, v_val_2071_, v___y_2058_, v___y_2059_);
return v___x_2073_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object* v___x_2074_, lean_object* v___x_2075_, lean_object* v___x_2076_, lean_object* v___x_2077_, lean_object* v_x_2078_, lean_object* v_x_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_){
_start:
{
lean_object* v_res_2083_; 
v_res_2083_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__1_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(v___x_2074_, v___x_2075_, v___x_2076_, v___x_2077_, v_x_2078_, v_x_2079_, v___y_2080_, v___y_2081_);
lean_dec(v___y_2081_);
lean_dec_ref(v___y_2080_);
lean_dec(v_x_2078_);
return v_res_2083_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(uint8_t v___x_2084_, lean_object* v_env_2085_, lean_object* v_n_2086_, lean_object* v_x_2087_){
_start:
{
uint8_t v___x_2088_; 
v___x_2088_ = l_Lean_Environment_contains(v_env_2085_, v_n_2086_, v___x_2084_);
return v___x_2088_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object* v___x_2089_, lean_object* v_env_2090_, lean_object* v_n_2091_, lean_object* v_x_2092_){
_start:
{
uint8_t v___x_5599__boxed_2093_; uint8_t v_res_2094_; lean_object* v_r_2095_; 
v___x_5599__boxed_2093_ = lean_unbox(v___x_2089_);
v_res_2094_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___lam__2_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(v___x_5599__boxed_2093_, v_env_2090_, v_n_2091_, v_x_2092_);
lean_dec_ref(v_x_2092_);
v_r_2095_ = lean_box(v_res_2094_);
return v_r_2095_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2126_; lean_object* v___x_2127_; 
v___x_2126_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_));
v___x_2127_ = l_Lean_registerParametricAttribute___redArg(v___x_2126_);
return v___x_2127_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2____boxed(lean_object* v_a_2128_){
_start:
{
lean_object* v_res_2129_; 
v_res_2129_ = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_();
return v_res_2129_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4(lean_object* v_00_u03b2_2130_, lean_object* v_m_2131_, lean_object* v_a_2132_){
_start:
{
lean_object* v___x_2133_; 
v___x_2133_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___redArg(v_m_2131_, v_a_2132_);
return v___x_2133_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4___boxed(lean_object* v_00_u03b2_2134_, lean_object* v_m_2135_, lean_object* v_a_2136_){
_start:
{
lean_object* v_res_2137_; 
v_res_2137_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4(v_00_u03b2_2134_, v_m_2135_, v_a_2136_);
lean_dec(v_a_2136_);
lean_dec_ref(v_m_2135_);
return v_res_2137_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3(lean_object* v_00_u03b2_2138_, lean_object* v_x_2139_, lean_object* v_x_2140_){
_start:
{
uint8_t v___x_2141_; 
v___x_2141_ = lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___redArg(v_x_2139_, v_x_2140_);
return v___x_2141_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3___boxed(lean_object* v_00_u03b2_2142_, lean_object* v_x_2143_, lean_object* v_x_2144_){
_start:
{
uint8_t v_res_2145_; lean_object* v_r_2146_; 
v_res_2145_ = lp_batteries_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3(v_00_u03b2_2142_, v_x_2143_, v_x_2144_);
lean_dec_ref(v_x_2144_);
lean_dec_ref(v_x_2143_);
v_r_2146_ = lean_box(v_res_2145_);
return v_r_2146_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7(lean_object* v_00_u03b2_2147_, lean_object* v_a_2148_, lean_object* v_x_2149_){
_start:
{
lean_object* v___x_2150_; 
v___x_2150_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___redArg(v_a_2148_, v_x_2149_);
return v___x_2150_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7___boxed(lean_object* v_00_u03b2_2151_, lean_object* v_a_2152_, lean_object* v_x_2153_){
_start:
{
lean_object* v_res_2154_; 
v_res_2154_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__4_spec__7(v_00_u03b2_2151_, v_a_2152_, v_x_2153_);
lean_dec(v_x_2153_);
lean_dec(v_a_2152_);
return v_res_2154_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_2155_, lean_object* v_x_2156_, size_t v_x_2157_, lean_object* v_x_2158_){
_start:
{
uint8_t v___x_2159_; 
v___x_2159_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___redArg(v_x_2156_, v_x_2157_, v_x_2158_);
return v___x_2159_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_2160_, lean_object* v_x_2161_, lean_object* v_x_2162_, lean_object* v_x_2163_){
_start:
{
size_t v_x_5720__boxed_2164_; uint8_t v_res_2165_; lean_object* v_r_2166_; 
v_x_5720__boxed_2164_ = lean_unbox_usize(v_x_2162_);
lean_dec(v_x_2162_);
v_res_2165_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5(v_00_u03b2_2160_, v_x_2161_, v_x_5720__boxed_2164_, v_x_2163_);
lean_dec_ref(v_x_2163_);
lean_dec_ref(v_x_2161_);
v_r_2166_ = lean_box(v_res_2165_);
return v_r_2166_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8(lean_object* v_00_u03b2_2167_, lean_object* v_keys_2168_, lean_object* v_vals_2169_, lean_object* v_heq_2170_, lean_object* v_i_2171_, lean_object* v_k_2172_){
_start:
{
uint8_t v___x_2173_; 
v___x_2173_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___redArg(v_keys_2168_, v_i_2171_, v_k_2172_);
return v___x_2173_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8___boxed(lean_object* v_00_u03b2_2174_, lean_object* v_keys_2175_, lean_object* v_vals_2176_, lean_object* v_heq_2177_, lean_object* v_i_2178_, lean_object* v_k_2179_){
_start:
{
uint8_t v_res_2180_; lean_object* v_r_2181_; 
v_res_2180_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2_spec__2_spec__3_spec__5_spec__8(v_00_u03b2_2174_, v_keys_2175_, v_vals_2176_, v_heq_2177_, v_i_2178_, v_k_2179_);
lean_dec_ref(v_k_2179_);
lean_dec_ref(v_vals_2176_);
lean_dec_ref(v_keys_2175_);
v_r_2181_ = lean_box(v_res_2180_);
return v_r_2181_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0(lean_object* v___x_2184_, lean_object* v_linter_2185_, lean_object* v_toPure_2186_, lean_object* v___x_2187_, lean_object* v_decl_2188_, lean_object* v_____do__lift_2189_){
_start:
{
lean_object* v___y_2191_; lean_object* v___x_2199_; lean_object* v___x_2200_; 
v___x_2199_ = lp_batteries_Batteries_Tactic_Lint_nolintAttr;
v___x_2200_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_2187_, v___x_2199_, v_____do__lift_2189_, v_decl_2188_);
if (lean_obj_tag(v___x_2200_) == 0)
{
lean_object* v___x_2201_; 
v___x_2201_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0___closed__0));
v___y_2191_ = v___x_2201_;
goto v___jp_2190_;
}
else
{
lean_object* v_val_2202_; 
v_val_2202_ = lean_ctor_get(v___x_2200_, 0);
lean_inc(v_val_2202_);
lean_dec_ref_known(v___x_2200_, 1);
v___y_2191_ = v_val_2202_;
goto v___jp_2190_;
}
v___jp_2190_:
{
uint8_t v___x_2192_; 
v___x_2192_ = l_Array_contains___redArg(v___x_2184_, v___y_2191_, v_linter_2185_);
if (v___x_2192_ == 0)
{
uint8_t v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; 
v___x_2193_ = 1;
v___x_2194_ = lean_box(v___x_2193_);
v___x_2195_ = lean_apply_2(v_toPure_2186_, lean_box(0), v___x_2194_);
return v___x_2195_;
}
else
{
uint8_t v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; 
v___x_2196_ = 0;
v___x_2197_ = lean_box(v___x_2196_);
v___x_2198_ = lean_apply_2(v_toPure_2186_, lean_box(0), v___x_2197_);
return v___x_2198_;
}
}
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0(void){
_start:
{
lean_object* v___x_2203_; 
v___x_2203_ = l_Array_instInhabited(lean_box(0));
return v___x_2203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg(lean_object* v_inst_2204_, lean_object* v_inst_2205_, lean_object* v_linter_2206_, lean_object* v_decl_2207_){
_start:
{
lean_object* v_toApplicative_2208_; lean_object* v_toBind_2209_; lean_object* v_getEnv_2210_; lean_object* v_toPure_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___f_2214_; lean_object* v___x_2215_; 
v_toApplicative_2208_ = lean_ctor_get(v_inst_2204_, 0);
lean_inc_ref(v_toApplicative_2208_);
v_toBind_2209_ = lean_ctor_get(v_inst_2204_, 1);
lean_inc(v_toBind_2209_);
lean_dec_ref(v_inst_2204_);
v_getEnv_2210_ = lean_ctor_get(v_inst_2205_, 0);
lean_inc(v_getEnv_2210_);
lean_dec_ref(v_inst_2205_);
v_toPure_2211_ = lean_ctor_get(v_toApplicative_2208_, 1);
lean_inc(v_toPure_2211_);
lean_dec_ref(v_toApplicative_2208_);
v___x_2212_ = ((lean_object*)(lp_batteries_Lean_recordExtraModUseFromDecl___at___00__private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2__spec__2___closed__0));
v___x_2213_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0, &lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___closed__0);
v___f_2214_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg___lam__0), 6, 5);
lean_closure_set(v___f_2214_, 0, v___x_2212_);
lean_closure_set(v___f_2214_, 1, v_linter_2206_);
lean_closure_set(v___f_2214_, 2, v_toPure_2211_);
lean_closure_set(v___f_2214_, 3, v___x_2213_);
lean_closure_set(v___f_2214_, 4, v_decl_2207_);
v___x_2215_ = lean_apply_4(v_toBind_2209_, lean_box(0), lean_box(0), v_getEnv_2210_, v___f_2214_);
return v___x_2215_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted(lean_object* v_m_2216_, lean_object* v_inst_2217_, lean_object* v_inst_2218_, lean_object* v_linter_2219_, lean_object* v_decl_2220_){
_start:
{
lean_object* v___x_2221_; 
v___x_2221_ = lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___redArg(v_inst_2217_, v_inst_2218_, v_linter_2219_, v_decl_2220_);
return v___x_2221_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Structure(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* runtime_initialize_Lean_ExtraModUses(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Structure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_ExtraModUses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_880347739____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_3164034710____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Lint_Basic_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Basic_1944612687____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Tactic_Lint_nolintAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Tactic_Lint_nolintAttr);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Structure(uint8_t builtin);
lean_object* initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
lean_object* initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* initialize_Lean_ExtraModUses(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Structure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_ExtraModUses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
