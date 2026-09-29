// Lean compiler output
// Module: Mathlib.Tactic.ToFun
// Imports: public import Init public meta import Init public import Mathlib.Util.AddRelatedDecl public import Mathlib.Tactic.Push public import Mathlib.Tactic.Translate.Attributes
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
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
extern lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_name_append_before(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_rootNamespace;
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Name_getNumParts(lean_object*);
lean_object* lp_mathlib_Lean_Name_splitAt(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_removeRoot(lean_object*);
uint8_t l_Lean_instBEqAttributeKind_beq(uint8_t, uint8_t);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_registerGeneratingAttr(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "to_fun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__2_value),LEAN_SCALAR_PTR_LITERAL(178, 137, 112, 231, 41, 143, 229, 67)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_to__fun___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_to__fun___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_to__fun___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_to__fun___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_to__fun___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_to__fun___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_to__fun;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 194, 82, 68, 109, 146, 236, 67)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "`@[to_fun]` failed to eta-expand any part of `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "Eta-expanded form of `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fun_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "`to_fun` correctly autogenerated the provided name `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "`.\nYou may remove it."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "`to_fun` can only be used as a global attribute"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__2_value),LEAN_SCALAR_PTR_LITERAL(98, 88, 4, 107, 202, 36, 124, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ToFun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(49, 27, 127, 48, 51, 161, 155, 102)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(116, 125, 142, 75, 161, 71, 196, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 123, 46, 246, 166, 182, 209, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 29, 221, 11, 206, 95, 135, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(201, 82, 73, 0, 139, 37, 195, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(100, 38, 73, 86, 250, 9, 255, 231)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__0_value),LEAN_SCALAR_PTR_LITERAL(85, 47, 79, 110, 141, 77, 112, 132)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_to__fun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(108, 18, 230, 236, 60, 231, 11, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 233, 171, 9, 76, 30, 155, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2055784324) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(214, 250, 234, 138, 131, 139, 118, 40)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(113, 152, 232, 32, 230, 151, 173, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(137, 79, 163, 199, 148, 151, 0, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(116, 62, 209, 166, 53, 35, 44, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 87, .m_capacity = 87, .m_length = 86, .m_data = "generate a copy of a lemma where point-free functions are expanded to their `fun` form"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__27_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__27_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__27_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__28_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__27_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__28_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__28_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__7(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_14_ = lp_mathlib_Mathlib_Tactic_optAttrArg;
v___x_15_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__6));
v___x_16_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__5));
v___x_17_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
lean_ctor_set(v___x_17_, 1, v___x_15_);
lean_ctor_set(v___x_17_, 2, v___x_14_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__18(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_38_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__17));
v___x_39_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_to__fun___closed__7, &lp_mathlib_Mathlib_Tactic_to__fun___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__7);
v___x_40_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__5));
v___x_41_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v___x_39_);
lean_ctor_set(v___x_41_, 2, v___x_38_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__19(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_42_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_to__fun___closed__18, &lp_mathlib_Mathlib_Tactic_to__fun___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__18);
v___x_43_ = lean_unsigned_to_nat(1022u);
v___x_44_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__3));
v___x_45_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_42_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_to__fun(void){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_to__fun___closed__19, &lp_mathlib_Mathlib_Tactic_to__fun___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_to__fun___closed__19);
return v___x_46_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_47_ = lean_box(0);
v___x_48_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_49_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v___x_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg(){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___closed__0);
v___x_52_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg___boxed(lean_object* v___y_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg();
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1(lean_object* v_00_u03b1_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg();
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___boxed(lean_object* v_00_u03b1_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1(v_00_u03b1_60_, v___y_61_, v___y_62_);
lean_dec(v___y_62_);
lean_dec_ref(v___y_61_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0(lean_object* v_msgData_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v___x_71_; lean_object* v_env_72_; lean_object* v___x_73_; lean_object* v_mctx_74_; lean_object* v_lctx_75_; lean_object* v_options_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_71_ = lean_st_ref_get(v___y_69_);
v_env_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc_ref(v_env_72_);
lean_dec(v___x_71_);
v___x_73_ = lean_st_ref_get(v___y_67_);
v_mctx_74_ = lean_ctor_get(v___x_73_, 0);
lean_inc_ref(v_mctx_74_);
lean_dec(v___x_73_);
v_lctx_75_ = lean_ctor_get(v___y_66_, 2);
v_options_76_ = lean_ctor_get(v___y_68_, 2);
lean_inc_ref(v_options_76_);
lean_inc_ref(v_lctx_75_);
v___x_77_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_77_, 0, v_env_72_);
lean_ctor_set(v___x_77_, 1, v_mctx_74_);
lean_ctor_set(v___x_77_, 2, v_lctx_75_);
lean_ctor_set(v___x_77_, 3, v_options_76_);
v___x_78_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_msgData_65_);
v___x_79_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0___boxed(lean_object* v_msgData_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0(v_msgData_80_, v___y_81_, v___y_82_, v___y_83_, v___y_84_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg(lean_object* v_msg_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_ref_93_; lean_object* v___x_94_; lean_object* v_a_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_103_; 
v_ref_93_ = lean_ctor_get(v___y_90_, 5);
v___x_94_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0_spec__0(v_msg_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
v_a_95_ = lean_ctor_get(v___x_94_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_103_ == 0)
{
v___x_97_ = v___x_94_;
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_a_95_);
lean_dec(v___x_94_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_99_; lean_object* v___x_101_; 
lean_inc(v_ref_93_);
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v_ref_93_);
lean_ctor_set(v___x_99_, 1, v_a_95_);
if (v_isShared_98_ == 0)
{
lean_ctor_set_tag(v___x_97_, 1);
lean_ctor_set(v___x_97_, 0, v___x_99_);
v___x_101_ = v___x_97_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v___x_99_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg___boxed(lean_object* v_msg_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg(v_msg_104_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec(v___y_106_);
lean_dec_ref(v___y_105_);
return v_res_110_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__2));
v___x_116_ = l_Lean_stringToMessageData(v___x_115_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__4));
v___x_119_ = l_Lean_stringToMessageData(v___x_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0(lean_object* v_src_120_, lean_object* v_value_121_, lean_object* v_levels_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
lean_object* v_value_129_; lean_object* v___x_132_; 
lean_inc(v___y_126_);
lean_inc_ref(v___y_125_);
lean_inc(v___y_124_);
lean_inc_ref(v___y_123_);
lean_inc_ref(v_value_121_);
v___x_132_ = lean_infer_type(v_value_121_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
if (lean_obj_tag(v___x_132_) == 0)
{
lean_object* v_a_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v_a_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc_n(v_a_133_, 2);
lean_dec_ref_known(v___x_132_, 1);
v___x_134_ = lean_box(1);
v___x_135_ = lean_box(0);
v___x_136_ = lp_mathlib_Mathlib_Tactic_Push_pullCore(v___x_134_, v_a_133_, v___x_135_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
if (lean_obj_tag(v___x_136_) == 0)
{
lean_object* v_a_137_; lean_object* v_expr_138_; lean_object* v_proof_x3f_139_; lean_object* v___y_141_; lean_object* v___y_142_; lean_object* v___y_143_; lean_object* v___y_144_; uint8_t v___x_175_; 
v_a_137_ = lean_ctor_get(v___x_136_, 0);
lean_inc(v_a_137_);
lean_dec_ref_known(v___x_136_, 1);
v_expr_138_ = lean_ctor_get(v_a_137_, 0);
lean_inc_ref(v_expr_138_);
v_proof_x3f_139_ = lean_ctor_get(v_a_137_, 1);
lean_inc(v_proof_x3f_139_);
lean_dec(v_a_137_);
v___x_175_ = lean_expr_eqv(v_expr_138_, v_a_133_);
if (v___x_175_ == 0)
{
lean_dec(v_src_120_);
v___y_141_ = v___y_123_;
v___y_142_ = v___y_124_;
v___y_143_ = v___y_125_;
v___y_144_ = v___y_126_;
goto v___jp_140_;
}
else
{
lean_object* v___x_176_; uint8_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_190_; 
lean_dec(v_proof_x3f_139_);
lean_dec_ref(v_expr_138_);
lean_dec(v_a_133_);
lean_dec(v_levels_122_);
lean_dec_ref(v_value_121_);
v___x_176_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__3);
v___x_177_ = 0;
v___x_178_ = l_Lean_MessageData_ofConstName(v_src_120_, v___x_177_);
v___x_179_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_176_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__5);
v___x_181_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_179_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
v___x_182_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg(v___x_181_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
v_a_183_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_190_ == 0)
{
v___x_185_ = v___x_182_;
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___x_182_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_188_; 
if (v_isShared_186_ == 0)
{
v___x_188_ = v___x_185_;
goto v_reusejp_187_;
}
else
{
lean_object* v_reuseFailAlloc_189_; 
v_reuseFailAlloc_189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_189_, 0, v_a_183_);
v___x_188_ = v_reuseFailAlloc_189_;
goto v_reusejp_187_;
}
v_reusejp_187_:
{
return v___x_188_;
}
}
}
v___jp_140_:
{
if (lean_obj_tag(v_proof_x3f_139_) == 0)
{
lean_object* v___x_145_; 
lean_dec(v_a_133_);
v___x_145_ = l_Lean_Meta_mkExpectedTypeHint(v_value_121_, v_expr_138_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
if (lean_obj_tag(v___x_145_) == 0)
{
lean_object* v_a_146_; 
v_a_146_ = lean_ctor_get(v___x_145_, 0);
lean_inc(v_a_146_);
lean_dec_ref_known(v___x_145_, 1);
v_value_129_ = v_a_146_;
goto v___jp_128_;
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
lean_dec(v_levels_122_);
v_a_147_ = lean_ctor_get(v___x_145_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_145_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_145_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_145_);
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
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_155_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___closed__1));
v___x_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_156_, 0, v_a_133_);
v___x_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_157_, 0, v_expr_138_);
v___x_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_158_, 0, v_value_121_);
v___x_159_ = lean_unsigned_to_nat(4u);
v___x_160_ = lean_mk_empty_array_with_capacity(v___x_159_);
v___x_161_ = lean_array_push(v___x_160_, v___x_156_);
v___x_162_ = lean_array_push(v___x_161_, v___x_157_);
v___x_163_ = lean_array_push(v___x_162_, v_proof_x3f_139_);
v___x_164_ = lean_array_push(v___x_163_, v___x_158_);
v___x_165_ = l_Lean_Meta_mkAppOptM(v___x_155_, v___x_164_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_a_166_);
lean_dec_ref_known(v___x_165_, 1);
v_value_129_ = v_a_166_;
goto v___jp_128_;
}
else
{
lean_object* v_a_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_174_; 
lean_dec(v_levels_122_);
v_a_167_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_174_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_174_ == 0)
{
v___x_169_ = v___x_165_;
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_a_167_);
lean_dec(v___x_165_);
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
v_reuseFailAlloc_173_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
else
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_198_; 
lean_dec(v_a_133_);
lean_dec(v_levels_122_);
lean_dec_ref(v_value_121_);
lean_dec(v_src_120_);
v_a_191_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_198_ == 0)
{
v___x_193_ = v___x_136_;
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_136_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_a_191_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
else
{
lean_object* v_a_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_206_; 
lean_dec(v_levels_122_);
lean_dec_ref(v_value_121_);
lean_dec(v_src_120_);
v_a_199_ = lean_ctor_get(v___x_132_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_132_);
if (v_isSharedCheck_206_ == 0)
{
v___x_201_ = v___x_132_;
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_a_199_);
lean_dec(v___x_132_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_204_; 
if (v_isShared_202_ == 0)
{
v___x_204_ = v___x_201_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_a_199_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
v___jp_128_:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v_value_129_);
lean_ctor_set(v___x_130_, 1, v_levels_122_);
v___x_131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
return v___x_131_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___boxed(lean_object* v_src_207_, lean_object* v_value_208_, lean_object* v_levels_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0(v_src_207_, v_value_208_, v_levels_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
return v_res_215_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0(void){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_216_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__0);
v___x_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_219_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1);
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v___x_220_);
lean_ctor_set(v___x_221_, 2, v___x_220_);
lean_ctor_set(v___x_221_, 3, v___x_220_);
lean_ctor_set(v___x_221_, 4, v___x_219_);
lean_ctor_set(v___x_221_, 5, v___x_219_);
lean_ctor_set(v___x_221_, 6, v___x_219_);
lean_ctor_set(v___x_221_, 7, v___x_219_);
lean_ctor_set(v___x_221_, 8, v___x_219_);
lean_ctor_set(v___x_221_, 9, v___x_219_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_222_ = lean_unsigned_to_nat(32u);
v___x_223_ = lean_mk_empty_array_with_capacity(v___x_222_);
v___x_224_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
return v___x_224_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4(void){
_start:
{
size_t v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_225_ = ((size_t)5ULL);
v___x_226_ = lean_unsigned_to_nat(0u);
v___x_227_ = lean_unsigned_to_nat(32u);
v___x_228_ = lean_mk_empty_array_with_capacity(v___x_227_);
v___x_229_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__3);
v___x_230_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v___x_228_);
lean_ctor_set(v___x_230_, 2, v___x_226_);
lean_ctor_set(v___x_230_, 3, v___x_226_);
lean_ctor_set_usize(v___x_230_, 4, v___x_225_);
return v___x_230_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5(void){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_231_ = lean_box(1);
v___x_232_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4);
v___x_233_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__1);
v___x_234_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
lean_ctor_set(v___x_234_, 2, v___x_231_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5(lean_object* v_msgData_235_, lean_object* v___y_236_, lean_object* v___y_237_){
_start:
{
lean_object* v___x_239_; lean_object* v_env_240_; lean_object* v_options_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_239_ = lean_st_ref_get(v___y_237_);
v_env_240_ = lean_ctor_get(v___x_239_, 0);
lean_inc_ref(v_env_240_);
lean_dec(v___x_239_);
v_options_241_ = lean_ctor_get(v___y_236_, 2);
v___x_242_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__2);
v___x_243_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__5);
lean_inc_ref(v_options_241_);
v___x_244_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_244_, 0, v_env_240_);
lean_ctor_set(v___x_244_, 1, v___x_242_);
lean_ctor_set(v___x_244_, 2, v___x_243_);
lean_ctor_set(v___x_244_, 3, v_options_241_);
v___x_245_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v_msgData_235_);
v___x_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___boxed(lean_object* v_msgData_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5(v_msgData_247_, v___y_248_, v___y_249_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
return v_res_251_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4(lean_object* v_opts_252_, lean_object* v_opt_253_){
_start:
{
lean_object* v_name_254_; lean_object* v_defValue_255_; lean_object* v_map_256_; lean_object* v___x_257_; 
v_name_254_ = lean_ctor_get(v_opt_253_, 0);
v_defValue_255_ = lean_ctor_get(v_opt_253_, 1);
v_map_256_ = lean_ctor_get(v_opts_252_, 0);
v___x_257_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_256_, v_name_254_);
if (lean_obj_tag(v___x_257_) == 0)
{
uint8_t v___x_258_; 
v___x_258_ = lean_unbox(v_defValue_255_);
return v___x_258_;
}
else
{
lean_object* v_val_259_; 
v_val_259_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_val_259_);
lean_dec_ref_known(v___x_257_, 1);
if (lean_obj_tag(v_val_259_) == 1)
{
uint8_t v_v_260_; 
v_v_260_ = lean_ctor_get_uint8(v_val_259_, 0);
lean_dec_ref_known(v_val_259_, 0);
return v_v_260_;
}
else
{
uint8_t v___x_261_; 
lean_dec(v_val_259_);
v___x_261_ = lean_unbox(v_defValue_255_);
return v___x_261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4___boxed(lean_object* v_opts_262_, lean_object* v_opt_263_){
_start:
{
uint8_t v_res_264_; lean_object* v_r_265_; 
v_res_264_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4(v_opts_262_, v_opt_263_);
lean_dec_ref(v_opt_263_);
lean_dec_ref(v_opts_262_);
v_r_265_ = lean_box(v_res_264_);
return v_r_265_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0(uint8_t v___y_273_, uint8_t v_suppressElabErrors_274_, lean_object* v_x_275_){
_start:
{
if (lean_obj_tag(v_x_275_) == 1)
{
lean_object* v_pre_276_; 
v_pre_276_ = lean_ctor_get(v_x_275_, 0);
switch(lean_obj_tag(v_pre_276_))
{
case 1:
{
lean_object* v_pre_277_; 
v_pre_277_ = lean_ctor_get(v_pre_276_, 0);
switch(lean_obj_tag(v_pre_277_))
{
case 0:
{
lean_object* v_str_278_; lean_object* v_str_279_; lean_object* v___x_280_; uint8_t v___x_281_; 
v_str_278_ = lean_ctor_get(v_x_275_, 1);
v_str_279_ = lean_ctor_get(v_pre_276_, 1);
v___x_280_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__0));
v___x_281_ = lean_string_dec_eq(v_str_279_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; uint8_t v___x_283_; 
v___x_282_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__1));
v___x_283_ = lean_string_dec_eq(v_str_279_, v___x_282_);
if (v___x_283_ == 0)
{
return v___y_273_;
}
else
{
lean_object* v___x_284_; uint8_t v___x_285_; 
v___x_284_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__1));
v___x_285_ = lean_string_dec_eq(v_str_278_, v___x_284_);
if (v___x_285_ == 0)
{
return v___y_273_;
}
else
{
return v_suppressElabErrors_274_;
}
}
}
else
{
lean_object* v___x_286_; uint8_t v___x_287_; 
v___x_286_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__2));
v___x_287_ = lean_string_dec_eq(v_str_278_, v___x_286_);
if (v___x_287_ == 0)
{
return v___y_273_;
}
else
{
return v_suppressElabErrors_274_;
}
}
}
case 1:
{
lean_object* v_pre_288_; 
v_pre_288_ = lean_ctor_get(v_pre_277_, 0);
if (lean_obj_tag(v_pre_288_) == 0)
{
lean_object* v_str_289_; lean_object* v_str_290_; lean_object* v_str_291_; lean_object* v___x_292_; uint8_t v___x_293_; 
v_str_289_ = lean_ctor_get(v_x_275_, 1);
v_str_290_ = lean_ctor_get(v_pre_276_, 1);
v_str_291_ = lean_ctor_get(v_pre_277_, 1);
v___x_292_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__3));
v___x_293_ = lean_string_dec_eq(v_str_291_, v___x_292_);
if (v___x_293_ == 0)
{
return v___y_273_;
}
else
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__4));
v___x_295_ = lean_string_dec_eq(v_str_290_, v___x_294_);
if (v___x_295_ == 0)
{
return v___y_273_;
}
else
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__5));
v___x_297_ = lean_string_dec_eq(v_str_289_, v___x_296_);
if (v___x_297_ == 0)
{
return v___y_273_;
}
else
{
return v_suppressElabErrors_274_;
}
}
}
}
else
{
return v___y_273_;
}
}
default: 
{
return v___y_273_;
}
}
}
case 0:
{
lean_object* v_str_298_; lean_object* v___x_299_; uint8_t v___x_300_; 
v_str_298_ = lean_ctor_get(v_x_275_, 1);
v___x_299_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___closed__6));
v___x_300_ = lean_string_dec_eq(v_str_298_, v___x_299_);
if (v___x_300_ == 0)
{
return v___y_273_;
}
else
{
return v_suppressElabErrors_274_;
}
}
default: 
{
return v___y_273_;
}
}
}
else
{
return v___y_273_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___boxed(lean_object* v___y_301_, lean_object* v_suppressElabErrors_302_, lean_object* v_x_303_){
_start:
{
uint8_t v___y_7450__boxed_304_; uint8_t v_suppressElabErrors_boxed_305_; uint8_t v_res_306_; lean_object* v_r_307_; 
v___y_7450__boxed_304_ = lean_unbox(v___y_301_);
v_suppressElabErrors_boxed_305_ = lean_unbox(v_suppressElabErrors_302_);
v_res_306_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0(v___y_7450__boxed_304_, v_suppressElabErrors_boxed_305_, v_x_303_);
lean_dec(v_x_303_);
v_r_307_ = lean_box(v_res_306_);
return v_r_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3(lean_object* v_ref_309_, lean_object* v_msgData_310_, uint8_t v_severity_311_, uint8_t v_isSilent_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___y_317_; lean_object* v___y_318_; lean_object* v___y_319_; uint8_t v___y_320_; lean_object* v___y_321_; uint8_t v___y_322_; lean_object* v___y_323_; lean_object* v___y_324_; lean_object* v___y_325_; lean_object* v___y_353_; lean_object* v___y_354_; uint8_t v___y_355_; lean_object* v___y_356_; uint8_t v___y_357_; lean_object* v___y_358_; uint8_t v___y_359_; lean_object* v___y_360_; lean_object* v___y_378_; lean_object* v___y_379_; uint8_t v___y_380_; lean_object* v___y_381_; lean_object* v___y_382_; uint8_t v___y_383_; uint8_t v___y_384_; lean_object* v___y_385_; lean_object* v___y_389_; uint8_t v___y_390_; lean_object* v___y_391_; lean_object* v___y_392_; uint8_t v___y_393_; lean_object* v___y_394_; uint8_t v___y_395_; uint8_t v___x_400_; lean_object* v___y_402_; uint8_t v___y_403_; lean_object* v___y_404_; lean_object* v___y_405_; lean_object* v___y_406_; uint8_t v___y_407_; uint8_t v___y_408_; uint8_t v___y_410_; uint8_t v___x_425_; 
v___x_400_ = 2;
v___x_425_ = l_Lean_instBEqMessageSeverity_beq(v_severity_311_, v___x_400_);
if (v___x_425_ == 0)
{
v___y_410_ = v___x_425_;
goto v___jp_409_;
}
else
{
uint8_t v___x_426_; 
lean_inc_ref(v_msgData_310_);
v___x_426_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_310_);
v___y_410_ = v___x_426_;
goto v___jp_409_;
}
v___jp_316_:
{
lean_object* v___x_326_; lean_object* v_currNamespace_327_; lean_object* v_openDecls_328_; lean_object* v_env_329_; lean_object* v_nextMacroScope_330_; lean_object* v_ngen_331_; lean_object* v_auxDeclNGen_332_; lean_object* v_traceState_333_; lean_object* v_cache_334_; lean_object* v_messages_335_; lean_object* v_infoState_336_; lean_object* v_snapshotTasks_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_351_; 
v___x_326_ = lean_st_ref_take(v___y_325_);
v_currNamespace_327_ = lean_ctor_get(v___y_324_, 6);
v_openDecls_328_ = lean_ctor_get(v___y_324_, 7);
v_env_329_ = lean_ctor_get(v___x_326_, 0);
v_nextMacroScope_330_ = lean_ctor_get(v___x_326_, 1);
v_ngen_331_ = lean_ctor_get(v___x_326_, 2);
v_auxDeclNGen_332_ = lean_ctor_get(v___x_326_, 3);
v_traceState_333_ = lean_ctor_get(v___x_326_, 4);
v_cache_334_ = lean_ctor_get(v___x_326_, 5);
v_messages_335_ = lean_ctor_get(v___x_326_, 6);
v_infoState_336_ = lean_ctor_get(v___x_326_, 7);
v_snapshotTasks_337_ = lean_ctor_get(v___x_326_, 8);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_351_ == 0)
{
v___x_339_ = v___x_326_;
v_isShared_340_ = v_isSharedCheck_351_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_snapshotTasks_337_);
lean_inc(v_infoState_336_);
lean_inc(v_messages_335_);
lean_inc(v_cache_334_);
lean_inc(v_traceState_333_);
lean_inc(v_auxDeclNGen_332_);
lean_inc(v_ngen_331_);
lean_inc(v_nextMacroScope_330_);
lean_inc(v_env_329_);
lean_dec(v___x_326_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_351_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_346_; 
lean_inc(v_openDecls_328_);
lean_inc(v_currNamespace_327_);
v___x_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_341_, 0, v_currNamespace_327_);
lean_ctor_set(v___x_341_, 1, v_openDecls_328_);
v___x_342_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v___y_323_);
lean_inc_ref(v___y_317_);
lean_inc_ref(v___y_319_);
v___x_343_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_343_, 0, v___y_319_);
lean_ctor_set(v___x_343_, 1, v___y_321_);
lean_ctor_set(v___x_343_, 2, v___y_318_);
lean_ctor_set(v___x_343_, 3, v___y_317_);
lean_ctor_set(v___x_343_, 4, v___x_342_);
lean_ctor_set_uint8(v___x_343_, sizeof(void*)*5, v___y_320_);
lean_ctor_set_uint8(v___x_343_, sizeof(void*)*5 + 1, v___y_322_);
lean_ctor_set_uint8(v___x_343_, sizeof(void*)*5 + 2, v_isSilent_312_);
v___x_344_ = l_Lean_MessageLog_add(v___x_343_, v_messages_335_);
if (v_isShared_340_ == 0)
{
lean_ctor_set(v___x_339_, 6, v___x_344_);
v___x_346_ = v___x_339_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v_env_329_);
lean_ctor_set(v_reuseFailAlloc_350_, 1, v_nextMacroScope_330_);
lean_ctor_set(v_reuseFailAlloc_350_, 2, v_ngen_331_);
lean_ctor_set(v_reuseFailAlloc_350_, 3, v_auxDeclNGen_332_);
lean_ctor_set(v_reuseFailAlloc_350_, 4, v_traceState_333_);
lean_ctor_set(v_reuseFailAlloc_350_, 5, v_cache_334_);
lean_ctor_set(v_reuseFailAlloc_350_, 6, v___x_344_);
lean_ctor_set(v_reuseFailAlloc_350_, 7, v_infoState_336_);
lean_ctor_set(v_reuseFailAlloc_350_, 8, v_snapshotTasks_337_);
v___x_346_ = v_reuseFailAlloc_350_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_347_ = lean_st_ref_set(v___y_325_, v___x_346_);
v___x_348_ = lean_box(0);
v___x_349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
return v___x_349_;
}
}
}
v___jp_352_:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v_a_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_376_; 
v___x_361_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_310_);
v___x_362_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5(v___x_361_, v___y_313_, v___y_314_);
v_a_363_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_376_ == 0)
{
v___x_365_ = v___x_362_;
v_isShared_366_ = v_isSharedCheck_376_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_a_363_);
lean_dec(v___x_362_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_376_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
lean_inc_ref_n(v___y_358_, 2);
v___x_367_ = l_Lean_FileMap_toPosition(v___y_358_, v___y_354_);
lean_dec(v___y_354_);
v___x_368_ = l_Lean_FileMap_toPosition(v___y_358_, v___y_360_);
lean_dec(v___y_360_);
v___x_369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
v___x_370_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___closed__0));
if (v___y_355_ == 0)
{
lean_del_object(v___x_365_);
lean_dec_ref(v___y_353_);
v___y_317_ = v___x_370_;
v___y_318_ = v___x_369_;
v___y_319_ = v___y_356_;
v___y_320_ = v___y_357_;
v___y_321_ = v___x_367_;
v___y_322_ = v___y_359_;
v___y_323_ = v_a_363_;
v___y_324_ = v___y_313_;
v___y_325_ = v___y_314_;
goto v___jp_316_;
}
else
{
uint8_t v___x_371_; 
lean_inc(v_a_363_);
v___x_371_ = l_Lean_MessageData_hasTag(v___y_353_, v_a_363_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_374_; 
lean_dec_ref_known(v___x_369_, 1);
lean_dec_ref(v___x_367_);
lean_dec(v_a_363_);
v___x_372_ = lean_box(0);
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 0, v___x_372_);
v___x_374_ = v___x_365_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_372_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
else
{
lean_del_object(v___x_365_);
v___y_317_ = v___x_370_;
v___y_318_ = v___x_369_;
v___y_319_ = v___y_356_;
v___y_320_ = v___y_357_;
v___y_321_ = v___x_367_;
v___y_322_ = v___y_359_;
v___y_323_ = v_a_363_;
v___y_324_ = v___y_313_;
v___y_325_ = v___y_314_;
goto v___jp_316_;
}
}
}
}
v___jp_377_:
{
lean_object* v___x_386_; 
v___x_386_ = l_Lean_Syntax_getTailPos_x3f(v___y_379_, v___y_383_);
lean_dec(v___y_379_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_inc(v___y_385_);
v___y_353_ = v___y_378_;
v___y_354_ = v___y_385_;
v___y_355_ = v___y_380_;
v___y_356_ = v___y_381_;
v___y_357_ = v___y_383_;
v___y_358_ = v___y_382_;
v___y_359_ = v___y_384_;
v___y_360_ = v___y_385_;
goto v___jp_352_;
}
else
{
lean_object* v_val_387_; 
v_val_387_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_val_387_);
lean_dec_ref_known(v___x_386_, 1);
v___y_353_ = v___y_378_;
v___y_354_ = v___y_385_;
v___y_355_ = v___y_380_;
v___y_356_ = v___y_381_;
v___y_357_ = v___y_383_;
v___y_358_ = v___y_382_;
v___y_359_ = v___y_384_;
v___y_360_ = v_val_387_;
goto v___jp_352_;
}
}
v___jp_388_:
{
lean_object* v_ref_396_; lean_object* v___x_397_; 
v_ref_396_ = l_Lean_replaceRef(v_ref_309_, v___y_392_);
v___x_397_ = l_Lean_Syntax_getPos_x3f(v_ref_396_, v___y_393_);
if (lean_obj_tag(v___x_397_) == 0)
{
lean_object* v___x_398_; 
v___x_398_ = lean_unsigned_to_nat(0u);
v___y_378_ = v___y_389_;
v___y_379_ = v_ref_396_;
v___y_380_ = v___y_390_;
v___y_381_ = v___y_391_;
v___y_382_ = v___y_394_;
v___y_383_ = v___y_393_;
v___y_384_ = v___y_395_;
v___y_385_ = v___x_398_;
goto v___jp_377_;
}
else
{
lean_object* v_val_399_; 
v_val_399_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_val_399_);
lean_dec_ref_known(v___x_397_, 1);
v___y_378_ = v___y_389_;
v___y_379_ = v_ref_396_;
v___y_380_ = v___y_390_;
v___y_381_ = v___y_391_;
v___y_382_ = v___y_394_;
v___y_383_ = v___y_393_;
v___y_384_ = v___y_395_;
v___y_385_ = v_val_399_;
goto v___jp_377_;
}
}
v___jp_401_:
{
if (v___y_408_ == 0)
{
v___y_389_ = v___y_402_;
v___y_390_ = v___y_403_;
v___y_391_ = v___y_405_;
v___y_392_ = v___y_404_;
v___y_393_ = v___y_407_;
v___y_394_ = v___y_406_;
v___y_395_ = v_severity_311_;
goto v___jp_388_;
}
else
{
v___y_389_ = v___y_402_;
v___y_390_ = v___y_403_;
v___y_391_ = v___y_405_;
v___y_392_ = v___y_404_;
v___y_393_ = v___y_407_;
v___y_394_ = v___y_406_;
v___y_395_ = v___x_400_;
goto v___jp_388_;
}
}
v___jp_409_:
{
if (v___y_410_ == 0)
{
lean_object* v_fileName_411_; lean_object* v_fileMap_412_; lean_object* v_options_413_; lean_object* v_ref_414_; uint8_t v_suppressElabErrors_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___f_418_; uint8_t v___x_419_; uint8_t v___x_420_; 
v_fileName_411_ = lean_ctor_get(v___y_313_, 0);
v_fileMap_412_ = lean_ctor_get(v___y_313_, 1);
v_options_413_ = lean_ctor_get(v___y_313_, 2);
v_ref_414_ = lean_ctor_get(v___y_313_, 5);
v_suppressElabErrors_415_ = lean_ctor_get_uint8(v___y_313_, sizeof(void*)*14 + 1);
v___x_416_ = lean_box(v___y_410_);
v___x_417_ = lean_box(v_suppressElabErrors_415_);
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_418_, 0, v___x_416_);
lean_closure_set(v___f_418_, 1, v___x_417_);
v___x_419_ = 1;
v___x_420_ = l_Lean_instBEqMessageSeverity_beq(v_severity_311_, v___x_419_);
if (v___x_420_ == 0)
{
v___y_402_ = v___f_418_;
v___y_403_ = v_suppressElabErrors_415_;
v___y_404_ = v_ref_414_;
v___y_405_ = v_fileName_411_;
v___y_406_ = v_fileMap_412_;
v___y_407_ = v___y_410_;
v___y_408_ = v___x_420_;
goto v___jp_401_;
}
else
{
lean_object* v___x_421_; uint8_t v___x_422_; 
v___x_421_ = l_Lean_warningAsError;
v___x_422_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3_spec__4(v_options_413_, v___x_421_);
v___y_402_ = v___f_418_;
v___y_403_ = v_suppressElabErrors_415_;
v___y_404_ = v_ref_414_;
v___y_405_ = v_fileName_411_;
v___y_406_ = v_fileMap_412_;
v___y_407_ = v___y_410_;
v___y_408_ = v___x_422_;
goto v___jp_401_;
}
}
else
{
lean_object* v___x_423_; lean_object* v___x_424_; 
lean_dec_ref(v_msgData_310_);
v___x_423_ = lean_box(0);
v___x_424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
return v___x_424_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3___boxed(lean_object* v_ref_427_, lean_object* v_msgData_428_, lean_object* v_severity_429_, lean_object* v_isSilent_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
uint8_t v_severity_boxed_434_; uint8_t v_isSilent_boxed_435_; lean_object* v_res_436_; 
v_severity_boxed_434_ = lean_unbox(v_severity_429_);
v_isSilent_boxed_435_ = lean_unbox(v_isSilent_430_);
v_res_436_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3(v_ref_427_, v_msgData_428_, v_severity_boxed_434_, v_isSilent_boxed_435_, v___y_431_, v___y_432_);
lean_dec(v___y_432_);
lean_dec_ref(v___y_431_);
lean_dec(v_ref_427_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2(lean_object* v_ref_437_, lean_object* v_msgData_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
uint8_t v___x_442_; uint8_t v___x_443_; lean_object* v___x_444_; 
v___x_442_ = 1;
v___x_443_ = 0;
v___x_444_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2_spec__3(v_ref_437_, v_msgData_438_, v___x_442_, v___x_443_, v___y_439_, v___y_440_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2___boxed(lean_object* v_ref_445_, lean_object* v_msgData_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2(v_ref_445_, v_msgData_446_, v___y_447_, v___y_448_);
lean_dec(v___y_448_);
lean_dec_ref(v___y_447_);
lean_dec(v_ref_445_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(lean_object* v_msg_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v_ref_455_; lean_object* v___x_456_; lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_465_; 
v_ref_455_ = lean_ctor_get(v___y_452_, 5);
v___x_456_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5(v_msg_451_, v___y_452_, v___y_453_);
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_465_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_465_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_465_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_465_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_461_; lean_object* v___x_463_; 
lean_inc(v_ref_455_);
v___x_461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_461_, 0, v_ref_455_);
lean_ctor_set(v___x_461_, 1, v_a_457_);
if (v_isShared_460_ == 0)
{
lean_ctor_set_tag(v___x_459_, 1);
lean_ctor_set(v___x_459_, 0, v___x_461_);
v___x_463_ = v___x_459_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(1, 1, 0);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg___boxed(lean_object* v_msg_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(v_msg_466_, v___y_467_, v___y_468_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
return v_res_470_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0(void){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1(void){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_472_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__0);
v___x_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
return v___x_473_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2(void){
_start:
{
lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_474_ = lean_box(1);
v___x_475_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4);
v___x_476_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1);
v___x_477_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
lean_ctor_set(v___x_477_, 1, v___x_475_);
lean_ctor_set(v___x_477_, 2, v___x_474_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4(void){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_480_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1);
v___x_481_ = lean_unsigned_to_nat(0u);
v___x_482_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
lean_ctor_set(v___x_482_, 1, v___x_481_);
lean_ctor_set(v___x_482_, 2, v___x_481_);
lean_ctor_set(v___x_482_, 3, v___x_481_);
lean_ctor_set(v___x_482_, 4, v___x_480_);
lean_ctor_set(v___x_482_, 5, v___x_480_);
lean_ctor_set(v___x_482_, 6, v___x_480_);
lean_ctor_set(v___x_482_, 7, v___x_480_);
lean_ctor_set(v___x_482_, 8, v___x_480_);
lean_ctor_set(v___x_482_, 9, v___x_480_);
return v___x_482_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1);
v___x_484_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_484_, 0, v___x_483_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
lean_ctor_set(v___x_484_, 2, v___x_483_);
lean_ctor_set(v___x_484_, 3, v___x_483_);
lean_ctor_set(v___x_484_, 4, v___x_483_);
lean_ctor_set(v___x_484_, 5, v___x_483_);
return v___x_484_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6(void){
_start:
{
lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_485_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__1);
v___x_486_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_486_, 0, v___x_485_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
lean_ctor_set(v___x_486_, 2, v___x_485_);
lean_ctor_set(v___x_486_, 3, v___x_485_);
lean_ctor_set(v___x_486_, 4, v___x_485_);
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_487_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__6);
v___x_488_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3_spec__5___closed__4);
v___x_489_ = lean_box(1);
v___x_490_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__5);
v___x_491_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__4);
v___x_492_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
lean_ctor_set(v___x_492_, 1, v___x_490_);
lean_ctor_set(v___x_492_, 2, v___x_489_);
lean_ctor_set(v___x_492_, 3, v___x_488_);
lean_ctor_set(v___x_492_, 4, v___x_487_);
return v___x_492_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12(void){
_start:
{
lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_497_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__11));
v___x_498_ = l_Lean_stringToMessageData(v___x_497_);
return v___x_498_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14(void){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__13));
v___x_501_ = l_Lean_stringToMessageData(v___x_500_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16(void){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_503_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__15));
v___x_504_ = l_Lean_stringToMessageData(v___x_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl(lean_object* v_src_505_, lean_object* v_stx_506_, uint8_t v_kind_507_, lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
lean_object* v___x_511_; uint8_t v___x_512_; 
v___x_511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_to__fun___closed__3));
lean_inc(v_stx_506_);
v___x_512_ = l_Lean_Syntax_isOfKind(v_stx_506_, v___x_511_);
if (v___x_512_ == 0)
{
lean_object* v___x_513_; 
lean_dec(v_stx_506_);
lean_dec(v_src_505_);
v___x_513_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg();
return v___x_513_;
}
else
{
lean_object* v___f_514_; lean_object* v___x_515_; lean_object* v_tk_516_; lean_object* v___x_517_; lean_object* v_optAttr_518_; lean_object* v___y_520_; lean_object* v___y_521_; lean_object* v___y_522_; lean_object* v___y_523_; lean_object* v_val_571_; lean_object* v___y_572_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v___y_596_; lean_object* v_id_609_; lean_object* v___y_610_; lean_object* v___y_611_; lean_object* v___x_624_; lean_object* v___x_625_; uint8_t v___x_626_; 
lean_inc(v_src_505_);
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___lam__0___boxed), 8, 1);
lean_closure_set(v___f_514_, 0, v_src_505_);
v___x_515_ = lean_unsigned_to_nat(0u);
v_tk_516_ = l_Lean_Syntax_getArg(v_stx_506_, v___x_515_);
v___x_517_ = lean_unsigned_to_nat(1u);
v_optAttr_518_ = l_Lean_Syntax_getArg(v_stx_506_, v___x_517_);
v___x_624_ = lean_unsigned_to_nat(2u);
v___x_625_ = l_Lean_Syntax_getArg(v_stx_506_, v___x_624_);
lean_dec(v_stx_506_);
v___x_626_ = l_Lean_Syntax_isNone(v___x_625_);
if (v___x_626_ == 0)
{
uint8_t v___x_627_; 
lean_inc(v___x_625_);
v___x_627_ = l_Lean_Syntax_matchesNull(v___x_625_, v___x_517_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; 
lean_dec(v___x_625_);
lean_dec(v_optAttr_518_);
lean_dec(v_tk_516_);
lean_dec_ref(v___f_514_);
lean_dec(v_src_505_);
v___x_628_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__1___redArg();
return v___x_628_;
}
else
{
lean_object* v_id_629_; lean_object* v___x_630_; 
v_id_629_ = l_Lean_Syntax_getArg(v___x_625_, v___x_515_);
lean_dec(v___x_625_);
v___x_630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_630_, 0, v_id_629_);
v_id_609_ = v___x_630_;
v___y_610_ = v_a_508_;
v___y_611_ = v_a_509_;
goto v___jp_608_;
}
}
else
{
lean_object* v___x_631_; 
lean_dec(v___x_625_);
v___x_631_ = lean_box(0);
v_id_609_ = v___x_631_;
v___y_610_ = v_a_508_;
v___y_611_ = v_a_509_;
goto v___jp_608_;
}
v___jp_519_:
{
lean_object* v___x_524_; uint8_t v___x_525_; uint8_t v___x_526_; uint8_t v___x_527_; uint8_t v___x_528_; lean_object* v___x_529_; uint64_t v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
lean_inc(v_src_505_);
v___x_524_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_src_505_, v___x_512_);
v___x_525_ = 0;
v___x_526_ = 1;
v___x_527_ = 0;
v___x_528_ = 2;
v___x_529_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v___x_529_, 0, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 1, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 2, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 3, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 4, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 5, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 6, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 7, v___x_525_);
lean_ctor_set_uint8(v___x_529_, 8, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 9, v___x_526_);
lean_ctor_set_uint8(v___x_529_, 10, v___x_527_);
lean_ctor_set_uint8(v___x_529_, 11, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 12, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 13, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 14, v___x_528_);
lean_ctor_set_uint8(v___x_529_, 15, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 16, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 17, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 18, v___x_512_);
lean_ctor_set_uint8(v___x_529_, 19, v___x_525_);
v___x_530_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_529_);
v___x_531_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_531_, 0, v___x_529_);
lean_ctor_set_uint64(v___x_531_, sizeof(void*)*1, v___x_530_);
v___x_532_ = lean_box(1);
v___x_533_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__2);
v___x_534_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__3));
v___x_535_ = lean_box(0);
v___x_536_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_536_, 0, v___x_531_);
lean_ctor_set(v___x_536_, 1, v___x_532_);
lean_ctor_set(v___x_536_, 2, v___x_533_);
lean_ctor_set(v___x_536_, 3, v___x_534_);
lean_ctor_set(v___x_536_, 4, v___x_535_);
lean_ctor_set(v___x_536_, 5, v___x_515_);
lean_ctor_set(v___x_536_, 6, v___x_535_);
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*7, v___x_525_);
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*7 + 1, v___x_525_);
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*7 + 2, v___x_525_);
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*7 + 3, v___x_512_);
v___x_537_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__7);
v___x_538_ = lean_st_mk_ref(v___x_537_);
v___x_539_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__8));
v___x_540_ = lean_string_append(v___x_539_, v___x_524_);
lean_dec_ref(v___x_524_);
v___x_541_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__9));
v___x_542_ = lean_string_append(v___x_540_, v___x_541_);
v___x_543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_543_, 0, v___x_542_);
lean_inc(v___y_520_);
v___x_544_ = lp_mathlib_Mathlib_Tactic_addRelatedDecl(v_src_505_, v___y_520_, v___y_523_, v_optAttr_518_, v___f_514_, v___x_543_, v___x_512_, v___x_536_, v___x_538_, v___y_522_, v___y_521_);
lean_dec_ref_known(v___x_536_, 7);
if (lean_obj_tag(v___x_544_) == 0)
{
lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_552_; 
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_552_ == 0)
{
lean_object* v_unused_553_; 
v_unused_553_ = lean_ctor_get(v___x_544_, 0);
lean_dec(v_unused_553_);
v___x_546_ = v___x_544_;
v_isShared_547_ = v_isSharedCheck_552_;
goto v_resetjp_545_;
}
else
{
lean_dec(v___x_544_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_552_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v___x_548_; lean_object* v___x_550_; 
v___x_548_ = lean_st_ref_get(v___x_538_);
lean_dec(v___x_538_);
lean_dec(v___x_548_);
if (v_isShared_547_ == 0)
{
lean_ctor_set(v___x_546_, 0, v___y_520_);
v___x_550_ = v___x_546_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v___y_520_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
else
{
lean_dec(v___x_538_);
if (lean_obj_tag(v___x_544_) == 0)
{
lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_560_; 
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_560_ == 0)
{
lean_object* v_unused_561_; 
v_unused_561_ = lean_ctor_get(v___x_544_, 0);
lean_dec(v_unused_561_);
v___x_555_ = v___x_544_;
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
else
{
lean_dec(v___x_544_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_558_; 
if (v_isShared_556_ == 0)
{
lean_ctor_set_tag(v___x_555_, 0);
lean_ctor_set(v___x_555_, 0, v___y_520_);
v___x_558_ = v___x_555_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v___y_520_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
else
{
lean_object* v_a_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_569_; 
lean_dec(v___y_520_);
v_a_562_ = lean_ctor_get(v___x_544_, 0);
v_isSharedCheck_569_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_569_ == 0)
{
v___x_564_ = v___x_544_;
v_isShared_565_ = v_isSharedCheck_569_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_a_562_);
lean_dec(v___x_544_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_569_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_567_; 
if (v_isShared_565_ == 0)
{
v___x_567_ = v___x_564_;
goto v_reusejp_566_;
}
else
{
lean_object* v_reuseFailAlloc_568_; 
v_reuseFailAlloc_568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_568_, 0, v_a_562_);
v___x_567_ = v_reuseFailAlloc_568_;
goto v_reusejp_566_;
}
v_reusejp_566_:
{
return v___x_567_;
}
}
}
}
}
v___jp_570_:
{
lean_object* v___x_575_; lean_object* v___x_576_; uint8_t v___x_577_; 
v___x_575_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__10));
lean_inc(v_src_505_);
v___x_576_ = lean_name_append_before(v_src_505_, v___x_575_);
v___x_577_ = lean_name_eq(v___y_574_, v___x_576_);
lean_dec(v___x_576_);
if (v___x_577_ == 0)
{
v___y_520_ = v___y_574_;
v___y_521_ = v___y_572_;
v___y_522_ = v___y_573_;
v___y_523_ = v_val_571_;
goto v___jp_519_;
}
else
{
lean_object* v___x_578_; uint8_t v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_578_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__12);
v___x_579_ = 0;
lean_inc(v___y_574_);
v___x_580_ = l_Lean_MessageData_ofConstName(v___y_574_, v___x_579_);
v___x_581_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_581_, 0, v___x_578_);
lean_ctor_set(v___x_581_, 1, v___x_580_);
v___x_582_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__14);
v___x_583_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_583_, 0, v___x_581_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
v___x_584_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__2(v_val_571_, v___x_583_, v___y_573_, v___y_572_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_dec_ref_known(v___x_584_, 1);
v___y_520_ = v___y_574_;
v___y_521_ = v___y_572_;
v___y_522_ = v___y_573_;
v___y_523_ = v_val_571_;
goto v___jp_519_;
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
lean_dec(v___y_574_);
lean_dec(v_val_571_);
lean_dec(v_optAttr_518_);
lean_dec_ref(v___f_514_);
lean_dec(v_src_505_);
v_a_585_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_584_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_584_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
}
v___jp_593_:
{
if (lean_obj_tag(v___y_594_) == 0)
{
lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_597_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__10));
lean_inc(v_src_505_);
v___x_598_ = lean_name_append_before(v_src_505_, v___x_597_);
v___y_520_ = v___x_598_;
v___y_521_ = v___y_596_;
v___y_522_ = v___y_595_;
v___y_523_ = v_tk_516_;
goto v___jp_519_;
}
else
{
lean_object* v_val_599_; lean_object* v___x_600_; lean_object* v___x_601_; uint8_t v___x_602_; 
lean_dec(v_tk_516_);
v_val_599_ = lean_ctor_get(v___y_594_, 0);
lean_inc(v_val_599_);
lean_dec_ref_known(v___y_594_, 1);
v___x_600_ = l_Lean_rootNamespace;
v___x_601_ = l_Lean_TSyntax_getId(v_val_599_);
v___x_602_ = l_Lean_Name_isPrefixOf(v___x_600_, v___x_601_);
if (v___x_602_ == 0)
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v_fst_605_; lean_object* v___x_606_; 
v___x_603_ = l_Lean_Name_getNumParts(v___x_601_);
lean_inc(v_src_505_);
v___x_604_ = lp_mathlib_Lean_Name_splitAt(v_src_505_, v___x_603_);
v_fst_605_ = lean_ctor_get(v___x_604_, 0);
lean_inc(v_fst_605_);
lean_dec_ref(v___x_604_);
v___x_606_ = l_Lean_Name_append(v_fst_605_, v___x_601_);
v_val_571_ = v_val_599_;
v___y_572_ = v___y_596_;
v___y_573_ = v___y_595_;
v___y_574_ = v___x_606_;
goto v___jp_570_;
}
else
{
lean_object* v___x_607_; 
v___x_607_ = l_Lean_removeRoot(v___x_601_);
v_val_571_ = v_val_599_;
v___y_572_ = v___y_596_;
v___y_573_ = v___y_595_;
v___y_574_ = v___x_607_;
goto v___jp_570_;
}
}
}
v___jp_608_:
{
uint8_t v___x_612_; uint8_t v___x_613_; 
v___x_612_ = 0;
v___x_613_ = l_Lean_instBEqAttributeKind_beq(v_kind_507_, v___x_612_);
if (v___x_613_ == 0)
{
if (v___x_512_ == 0)
{
v___y_594_ = v_id_609_;
v___y_595_ = v___y_610_;
v___y_596_ = v___y_611_;
goto v___jp_593_;
}
else
{
lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v_a_616_; lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_623_; 
lean_dec(v_id_609_);
lean_dec(v_optAttr_518_);
lean_dec(v_tk_516_);
lean_dec_ref(v___f_514_);
lean_dec(v_src_505_);
v___x_614_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___closed__16);
v___x_615_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(v___x_614_, v___y_610_, v___y_611_);
v_a_616_ = lean_ctor_get(v___x_615_, 0);
v_isSharedCheck_623_ = !lean_is_exclusive(v___x_615_);
if (v_isSharedCheck_623_ == 0)
{
v___x_618_ = v___x_615_;
v_isShared_619_ = v_isSharedCheck_623_;
goto v_resetjp_617_;
}
else
{
lean_inc(v_a_616_);
lean_dec(v___x_615_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_623_;
goto v_resetjp_617_;
}
v_resetjp_617_:
{
lean_object* v___x_621_; 
if (v_isShared_619_ == 0)
{
v___x_621_ = v___x_618_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v_a_616_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
}
else
{
v___y_594_ = v_id_609_;
v___y_595_ = v___y_610_;
v___y_596_ = v___y_611_;
goto v___jp_593_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl___boxed(lean_object* v_src_632_, lean_object* v_stx_633_, lean_object* v_kind_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_){
_start:
{
uint8_t v_kind_boxed_638_; lean_object* v_res_639_; 
v_kind_boxed_638_ = lean_unbox(v_kind_634_);
v_res_639_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl(v_src_632_, v_stx_633_, v_kind_boxed_638_, v_a_635_, v_a_636_);
lean_dec(v_a_636_);
lean_dec_ref(v_a_635_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0(lean_object* v_00_u03b1_640_, lean_object* v_msg_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___redArg(v_msg_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0___boxed(lean_object* v_00_u03b1_648_, lean_object* v_msg_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_){
_start:
{
lean_object* v_res_655_; 
v_res_655_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__0(v_00_u03b1_648_, v_msg_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3(lean_object* v_00_u03b1_656_, lean_object* v_msg_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
lean_object* v___x_661_; 
v___x_661_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(v_msg_657_, v___y_658_, v___y_659_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___boxed(lean_object* v_00_u03b1_662_, lean_object* v_msg_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_){
_start:
{
lean_object* v_res_667_; 
v_res_667_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3(v_00_u03b1_662_, v_msg_663_, v___y_664_, v___y_665_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object* v_x1_668_, lean_object* v_x2_669_, uint8_t v_x3_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl(v_x1_668_, v_x2_669_, v_x3_670_, v___y_671_, v___y_672_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v_a_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_685_; 
v_a_675_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_685_ == 0)
{
v___x_677_ = v___x_674_;
v_isShared_678_ = v_isSharedCheck_685_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_a_675_);
lean_dec(v___x_674_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_685_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_683_; 
v___x_679_ = lean_unsigned_to_nat(1u);
v___x_680_ = lean_mk_empty_array_with_capacity(v___x_679_);
v___x_681_ = lean_array_push(v___x_680_, v_a_675_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 0, v___x_681_);
v___x_683_ = v___x_677_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v___x_681_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
else
{
lean_object* v_a_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_693_; 
v_a_686_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_693_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_693_ == 0)
{
v___x_688_ = v___x_674_;
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_a_686_);
lean_dec(v___x_674_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_691_; 
if (v_isShared_689_ == 0)
{
v___x_691_ = v___x_688_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v_a_686_);
v___x_691_ = v_reuseFailAlloc_692_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
return v___x_691_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object* v_x1_694_, lean_object* v_x2_695_, lean_object* v_x3_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
uint8_t v_x3_462__boxed_700_; lean_object* v_res_701_; 
v_x3_462__boxed_700_ = lean_unbox(v_x3_696_);
v_res_701_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(v_x1_694_, v_x2_695_, v_x3_462__boxed_700_, v___y_697_, v___y_698_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object* v_x1_702_, lean_object* v_x2_703_, uint8_t v_x3_704_, lean_object* v___y_705_, lean_object* v___y_706_){
_start:
{
lean_object* v___x_708_; 
v___x_708_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl(v_x1_702_, v_x2_703_, v_x3_704_, v___y_705_, v___y_706_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_716_; 
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_716_ == 0)
{
lean_object* v_unused_717_; 
v_unused_717_ = lean_ctor_get(v___x_708_, 0);
lean_dec(v_unused_717_);
v___x_710_ = v___x_708_;
v_isShared_711_ = v_isSharedCheck_716_;
goto v_resetjp_709_;
}
else
{
lean_dec(v___x_708_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_716_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_712_; lean_object* v___x_714_; 
v___x_712_ = lean_box(0);
if (v_isShared_711_ == 0)
{
lean_ctor_set(v___x_710_, 0, v___x_712_);
v___x_714_ = v___x_710_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v___x_712_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_725_; 
v_a_718_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_725_ == 0)
{
v___x_720_ = v___x_708_;
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_708_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_718_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object* v_x1_726_, lean_object* v_x2_727_, lean_object* v_x3_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
uint8_t v_x3_520__boxed_732_; lean_object* v_res_733_; 
v_x3_520__boxed_732_ = lean_unbox(v_x3_728_);
v_res_733_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(v_x1_726_, v_x2_727_, v_x3_520__boxed_732_, v___y_729_, v___y_730_);
lean_dec(v___y_730_);
lean_dec_ref(v___y_729_);
return v_res_733_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_735_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_));
v___x_736_ = l_Lean_stringToMessageData(v___x_735_);
return v___x_736_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_738_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_));
v___x_739_ = l_Lean_stringToMessageData(v___x_738_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(lean_object* v___x_740_, lean_object* v_decl_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
v___x_745_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_);
v___x_746_ = l_Lean_MessageData_ofName(v___x_740_);
v___x_747_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_747_, 0, v___x_745_);
lean_ctor_set(v___x_747_, 1, v___x_746_);
v___x_748_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2___closed__3_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_);
v___x_749_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_749_, 0, v___x_747_);
lean_ctor_set(v___x_749_, 1, v___x_748_);
v___x_750_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_toFunImpl_spec__3___redArg(v___x_749_, v___y_742_, v___y_743_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object* v___x_751_, lean_object* v_decl_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_){
_start:
{
lean_object* v_res_756_; 
v_res_756_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___lam__2_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(v___x_751_, v_decl_752_, v___y_753_, v___y_754_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v_decl_752_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_828_; lean_object* v___x_829_; lean_object* v___x_830_; 
v___f_828_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_));
v___x_829_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_));
v___x_830_ = lp_mathlib_Mathlib_Tactic_registerGeneratingAttr(v___x_829_, v___f_828_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_object* v___x_831_; lean_object* v___x_832_; 
lean_dec_ref_known(v___x_830_, 1);
v___x_831_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn___closed__28_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_));
v___x_832_ = l_Lean_registerBuiltinAttribute(v___x_831_);
return v___x_832_;
}
else
{
return v___x_830_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2____boxed(lean_object* v_a_833_){
_start:
{
lean_object* v_res_834_; 
v_res_834_ = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_();
return v_res_834_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AddRelatedDecl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AddRelatedDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_to__fun = _init_lp_mathlib_Mathlib_Tactic_to__fun();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_to__fun);
res = lp_mathlib___private_Mathlib_Tactic_ToFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ToFun_2055784324____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AddRelatedDecl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Translate_Attributes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AddRelatedDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Translate_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
}
#ifdef __cplusplus
}
#endif
