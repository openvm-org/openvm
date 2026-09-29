// Lean compiler output
// Module: Mathlib.Tactic.Linter.UpstreamableDecl
// Imports: public import Init public meta import Init public import Mathlib.Init public import ImportGraph.Tools.FindHome
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
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minKey_x3f___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_NameSet_transitivelyUsedConstants___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_ConstantInfo_name(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
extern lean_object* lp_importGraph_GoToModuleLink;
lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Widget_WidgetInstance_ofHash___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getId(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Name_isLocal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isLocal___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "upstreamableDecl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(131, 46, 243, 194, 40, 141, 48, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "enable the upstreamableDecl linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(136, 178, 191, 13, 247, 255, 191, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_upstreamableDecl;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "defs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(131, 46, 243, 194, 40, 141, 48, 206)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(194, 217, 210, 170, 12, 88, 3, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "upstreamableDecl warns on definitions"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(136, 178, 191, 13, 247, 255, 191, 193)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(133, 39, 200, 58, 139, 165, 205, 212)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_defs;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(131, 46, 243, 194, 40, 141, 48, 206)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(164, 107, 175, 48, 94, 81, 84, 205)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "upstreamableDecl warns on private declarations"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(136, 178, 191, 13, 247, 255, 191, 193)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(11, 169, 174, 44, 67, 108, 12, 162)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_private;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "Consider moving this declaration to the module "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "set_option"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(216, 223, 149, 245, 150, 86, 134, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__5_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "UpstreamableDecl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__8_value),LEAN_SCALAR_PTR_LITERAL(152, 155, 184, 226, 13, 246, 123, 154)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(145, 54, 202, 158, 58, 203, 8, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(68, 56, 87, 71, 235, 93, 201, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(242, 142, 155, 48, 124, 17, 179, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "DoubleImports"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__13_value),LEAN_SCALAR_PTR_LITERAL(109, 31, 97, 64, 165, 130, 89, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "upstreamableDeclLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__15_value),LEAN_SCALAR_PTR_LITERAL(93, 200, 93, 36, 66, 99, 154, 56)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Name_isLocal(lean_object* v_env_1_, lean_object* v_decl_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1_, v_decl_2_);
if (lean_obj_tag(v___x_3_) == 0)
{
uint8_t v___x_4_; 
v___x_4_ = 1;
return v___x_4_;
}
else
{
uint8_t v___x_5_; 
lean_dec_ref_known(v___x_3_, 1);
v___x_5_ = 0;
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isLocal___boxed(lean_object* v_env_6_, lean_object* v_decl_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_mathlib_Lean_Name_isLocal(v_env_6_, v_decl_7_);
lean_dec(v_decl_7_);
lean_dec_ref(v_env_6_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__3(lean_object* v_a_10_, lean_object* v_a_11_){
_start:
{
if (lean_obj_tag(v_a_10_) == 0)
{
lean_object* v___x_12_; 
v___x_12_ = l_List_reverse___redArg(v_a_11_);
return v___x_12_;
}
else
{
lean_object* v_head_13_; 
v_head_13_ = lean_ctor_get(v_a_10_, 0);
switch(lean_obj_tag(v_head_13_))
{
case 2:
{
lean_object* v_tail_14_; 
v_tail_14_ = lean_ctor_get(v_a_10_, 1);
lean_inc(v_tail_14_);
lean_dec_ref_known(v_a_10_, 2);
v_a_10_ = v_tail_14_;
goto _start;
}
case 6:
{
lean_object* v_tail_16_; 
v_tail_16_ = lean_ctor_get(v_a_10_, 1);
lean_inc(v_tail_16_);
lean_dec_ref_known(v_a_10_, 2);
v_a_10_ = v_tail_16_;
goto _start;
}
default: 
{
lean_object* v_tail_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_26_; 
lean_inc(v_head_13_);
v_tail_18_ = lean_ctor_get(v_a_10_, 1);
v_isSharedCheck_26_ = !lean_is_exclusive(v_a_10_);
if (v_isSharedCheck_26_ == 0)
{
lean_object* v_unused_27_; 
v_unused_27_ = lean_ctor_get(v_a_10_, 0);
lean_dec(v_unused_27_);
v___x_20_ = v_a_10_;
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_tail_18_);
lean_dec(v_a_10_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 1, v_a_11_);
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v_head_13_);
lean_ctor_set(v_reuseFailAlloc_25_, 1, v_a_11_);
v___x_23_ = v_reuseFailAlloc_25_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
v_a_10_ = v_tail_18_;
v_a_11_ = v___x_23_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__2(lean_object* v_env_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
if (lean_obj_tag(v_a_29_) == 0)
{
lean_object* v___x_31_; 
lean_dec_ref(v_env_28_);
v___x_31_ = lean_array_to_list(v_a_30_);
return v___x_31_;
}
else
{
lean_object* v_head_32_; lean_object* v_tail_33_; uint8_t v___x_34_; lean_object* v___x_35_; 
v_head_32_ = lean_ctor_get(v_a_29_, 0);
lean_inc(v_head_32_);
v_tail_33_ = lean_ctor_get(v_a_29_, 1);
lean_inc(v_tail_33_);
lean_dec_ref_known(v_a_29_, 2);
v___x_34_ = 0;
lean_inc_ref(v_env_28_);
v___x_35_ = l_Lean_Environment_find_x3f(v_env_28_, v_head_32_, v___x_34_);
if (lean_obj_tag(v___x_35_) == 0)
{
v_a_29_ = v_tail_33_;
goto _start;
}
else
{
lean_object* v_val_37_; lean_object* v___x_38_; 
v_val_37_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_val_37_);
lean_dec_ref_known(v___x_35_, 1);
v___x_38_ = lean_array_push(v_a_30_, v_val_37_);
v_a_29_ = v_tail_33_;
v_a_30_ = v___x_38_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0_spec__0(lean_object* v_env_40_, lean_object* v_init_41_, lean_object* v_x_42_){
_start:
{
if (lean_obj_tag(v_x_42_) == 0)
{
lean_object* v_k_43_; lean_object* v_l_44_; lean_object* v_r_45_; lean_object* v___x_46_; uint8_t v___x_47_; lean_object* v___x_48_; 
v_k_43_ = lean_ctor_get(v_x_42_, 1);
lean_inc_n(v_k_43_, 2);
v_l_44_ = lean_ctor_get(v_x_42_, 3);
lean_inc(v_l_44_);
v_r_45_ = lean_ctor_get(v_x_42_, 4);
lean_inc(v_r_45_);
lean_dec_ref_known(v_x_42_, 5);
lean_inc_ref_n(v_env_40_, 2);
v___x_46_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0_spec__0(v_env_40_, v_init_41_, v_l_44_);
v___x_47_ = 0;
v___x_48_ = l_Lean_Environment_find_x3f(v_env_40_, v_k_43_, v___x_47_);
if (lean_obj_tag(v___x_48_) == 0)
{
lean_dec(v_k_43_);
v_init_41_ = v___x_46_;
v_x_42_ = v_r_45_;
goto _start;
}
else
{
lean_object* v___x_50_; 
lean_dec_ref_known(v___x_48_, 1);
v___x_50_ = l_Lean_NameSet_insert(v___x_46_, v_k_43_);
v_init_41_ = v___x_50_;
v_x_42_ = v_r_45_;
goto _start;
}
}
else
{
lean_dec_ref(v_env_40_);
return v_init_41_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4(lean_object* v_a_52_, lean_object* v_env_53_, lean_object* v_x_54_){
_start:
{
if (lean_obj_tag(v_x_54_) == 0)
{
uint8_t v___x_55_; 
v___x_55_ = 0;
return v___x_55_;
}
else
{
lean_object* v_head_56_; lean_object* v_tail_57_; lean_object* v___x_58_; uint8_t v___x_59_; 
v_head_56_ = lean_ctor_get(v_x_54_, 0);
v_tail_57_ = lean_ctor_get(v_x_54_, 1);
v___x_58_ = l_Lean_ConstantInfo_name(v_head_56_);
v___x_59_ = lean_name_eq(v_a_52_, v___x_58_);
if (v___x_59_ == 0)
{
uint8_t v___x_60_; 
v___x_60_ = lp_mathlib_Lean_Name_isLocal(v_env_53_, v___x_58_);
lean_dec(v___x_58_);
if (v___x_60_ == 0)
{
v_x_54_ = v_tail_57_;
goto _start;
}
else
{
return v___x_60_;
}
}
else
{
lean_dec(v___x_58_);
v_x_54_ = v_tail_57_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4___boxed(lean_object* v_a_63_, lean_object* v_env_64_, lean_object* v_x_65_){
_start:
{
uint8_t v_res_66_; lean_object* v_r_67_; 
v_res_66_ = lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4(v_a_63_, v_env_64_, v_x_65_);
lean_dec(v_x_65_);
lean_dec_ref(v_env_64_);
lean_dec(v_a_63_);
v_r_67_ = lean_box(v_res_66_);
return v_r_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1(lean_object* v_init_68_, lean_object* v_x_69_){
_start:
{
if (lean_obj_tag(v_x_69_) == 0)
{
lean_object* v_k_70_; lean_object* v_l_71_; lean_object* v_r_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v_k_70_ = lean_ctor_get(v_x_69_, 1);
v_l_71_ = lean_ctor_get(v_x_69_, 3);
v_r_72_ = lean_ctor_get(v_x_69_, 4);
v___x_73_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1(v_init_68_, v_r_72_);
lean_inc(v_k_70_);
v___x_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_74_, 0, v_k_70_);
lean_ctor_set(v___x_74_, 1, v___x_73_);
v_init_68_ = v___x_74_;
v_x_69_ = v_l_71_;
goto _start;
}
else
{
return v_init_68_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1___boxed(lean_object* v_init_76_, lean_object* v_x_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1(v_init_76_, v_x_77_);
lean_dec(v_x_77_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies(lean_object* v_env_81_, lean_object* v_stx_82_, lean_object* v_id_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___x_87_; 
lean_inc(v_stx_82_);
v___x_87_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName(v_stx_82_, v_a_84_, v_a_85_);
if (lean_obj_tag(v___x_87_) == 0)
{
lean_object* v_a_88_; lean_object* v___x_89_; 
v_a_88_ = lean_ctor_get(v___x_87_, 0);
lean_inc(v_a_88_);
lean_dec_ref_known(v___x_87_, 1);
v___x_89_ = lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(v_stx_82_, v_id_83_, v_a_84_, v_a_85_);
if (lean_obj_tag(v___x_89_) == 0)
{
lean_object* v_a_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v_a_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc(v_a_90_);
lean_dec_ref_known(v___x_89_, 1);
v___x_91_ = l_Lean_NameSet_empty;
lean_inc_ref(v_env_81_);
v___x_92_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0_spec__0(v_env_81_, v___x_91_, v_a_90_);
v___x_93_ = lean_alloc_closure((void*)(lp_importGraph_Lean_NameSet_transitivelyUsedConstants___boxed), 4, 1);
lean_closure_set(v___x_93_, 0, v___x_92_);
v___x_94_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_93_, v_a_84_, v_a_85_);
if (lean_obj_tag(v___x_94_) == 0)
{
lean_object* v_a_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_109_; 
v_a_95_ = lean_ctor_get(v___x_94_, 0);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_109_ == 0)
{
v___x_97_ = v___x_94_;
v_isShared_98_ = v_isSharedCheck_109_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_a_95_);
lean_dec(v___x_94_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_109_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; uint8_t v___x_104_; lean_object* v___x_105_; lean_object* v___x_107_; 
v___x_99_ = lean_box(0);
v___x_100_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__1(v___x_99_, v_a_95_);
lean_dec(v_a_95_);
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___closed__0));
lean_inc_ref(v_env_81_);
v___x_102_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__2(v_env_81_, v___x_100_, v___x_101_);
v___x_103_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__3(v___x_102_, v___x_99_);
v___x_104_ = lp_mathlib_List_any___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__4(v_a_88_, v_env_81_, v___x_103_);
lean_dec(v___x_103_);
lean_dec_ref(v_env_81_);
lean_dec(v_a_88_);
v___x_105_ = lean_box(v___x_104_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 0, v___x_105_);
v___x_107_ = v___x_97_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v___x_105_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
else
{
lean_object* v_a_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_117_; 
lean_dec(v_a_88_);
lean_dec_ref(v_env_81_);
v_a_110_ = lean_ctor_get(v___x_94_, 0);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_117_ == 0)
{
v___x_112_ = v___x_94_;
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_a_110_);
lean_dec(v___x_94_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_115_; 
if (v_isShared_113_ == 0)
{
v___x_115_ = v___x_112_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_a_110_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
}
else
{
lean_object* v_a_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_125_; 
lean_dec(v_a_88_);
lean_dec_ref(v_env_81_);
v_a_118_ = lean_ctor_get(v___x_89_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_89_);
if (v_isSharedCheck_125_ == 0)
{
v___x_120_ = v___x_89_;
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_a_118_);
lean_dec(v___x_89_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_a_118_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
else
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_133_; 
lean_dec(v_stx_82_);
lean_dec_ref(v_env_81_);
v_a_126_ = lean_ctor_get(v___x_87_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_87_);
if (v_isSharedCheck_133_ == 0)
{
v___x_128_ = v___x_87_;
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_87_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_129_ == 0)
{
v___x_131_ = v___x_128_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_a_126_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies___boxed(lean_object* v_env_134_, lean_object* v_stx_135_, lean_object* v_id_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies(v_env_134_, v_stx_135_, v_id_136_, v_a_137_, v_a_138_);
lean_dec(v_a_138_);
lean_dec_ref(v_a_137_);
lean_dec(v_id_136_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0(lean_object* v_env_141_, lean_object* v_init_142_, lean_object* v_t_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies_spec__0_spec__0(v_env_141_, v_init_142_, v_t_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(lean_object* v_name_145_, lean_object* v_decl_146_, lean_object* v_ref_147_){
_start:
{
lean_object* v_defValue_149_; lean_object* v_descr_150_; lean_object* v_deprecation_x3f_151_; lean_object* v___x_152_; uint8_t v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v_defValue_149_ = lean_ctor_get(v_decl_146_, 0);
v_descr_150_ = lean_ctor_get(v_decl_146_, 1);
v_deprecation_x3f_151_ = lean_ctor_get(v_decl_146_, 2);
v___x_152_ = lean_alloc_ctor(1, 0, 1);
v___x_153_ = lean_unbox(v_defValue_149_);
lean_ctor_set_uint8(v___x_152_, 0, v___x_153_);
lean_inc(v_deprecation_x3f_151_);
lean_inc_ref(v_descr_150_);
lean_inc_n(v_name_145_, 2);
v___x_154_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_154_, 0, v_name_145_);
lean_ctor_set(v___x_154_, 1, v_ref_147_);
lean_ctor_set(v___x_154_, 2, v___x_152_);
lean_ctor_set(v___x_154_, 3, v_descr_150_);
lean_ctor_set(v___x_154_, 4, v_deprecation_x3f_151_);
v___x_155_ = lean_register_option(v_name_145_, v___x_154_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_163_; 
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; 
v_unused_164_ = lean_ctor_get(v___x_155_, 0);
lean_dec(v_unused_164_);
v___x_157_ = v___x_155_;
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
else
{
lean_dec(v___x_155_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_159_; lean_object* v___x_161_; 
lean_inc(v_defValue_149_);
v___x_159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_159_, 0, v_name_145_);
lean_ctor_set(v___x_159_, 1, v_defValue_149_);
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v___x_159_);
v___x_161_ = v___x_157_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_159_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
else
{
lean_object* v_a_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_172_; 
lean_dec(v_name_145_);
v_a_165_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_172_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_172_ == 0)
{
v___x_167_ = v___x_155_;
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_a_165_);
lean_dec(v___x_155_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v___x_170_; 
if (v_isShared_168_ == 0)
{
v___x_170_ = v___x_167_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v_a_165_);
v___x_170_ = v_reuseFailAlloc_171_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
return v___x_170_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_173_, lean_object* v_decl_174_, lean_object* v_ref_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(v_name_173_, v_decl_174_, v_ref_175_);
lean_dec_ref(v_decl_174_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_));
v___x_198_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_));
v___x_199_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_));
v___x_200_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(v___x_197_, v___x_198_, v___x_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4____boxed(lean_object* v_a_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_();
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_));
v___x_222_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_));
v___x_223_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_));
v___x_224_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(v___x_221_, v___x_222_, v___x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4____boxed(lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_();
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_245_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_));
v___x_246_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_));
v___x_247_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_));
v___x_248_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4__spec__0(v___x_245_, v___x_246_, v___x_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4____boxed(lean_object* v_a_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_();
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg(lean_object* v___y_251_){
_start:
{
lean_object* v___x_253_; lean_object* v_env_254_; lean_object* v___x_255_; lean_object* v_mainModule_256_; lean_object* v___x_257_; 
v___x_253_ = lean_st_ref_get(v___y_251_);
v_env_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc_ref(v_env_254_);
lean_dec(v___x_253_);
v___x_255_ = l_Lean_Environment_header(v_env_254_);
lean_dec_ref(v_env_254_);
v_mainModule_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_mainModule_256_);
lean_dec_ref(v___x_255_);
v___x_257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_257_, 0, v_mainModule_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg___boxed(lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg(v___y_258_);
lean_dec(v___y_258_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2(lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg(v___y_262_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___boxed(lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2(v___y_265_, v___y_266_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
return v_res_268_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0(uint8_t v___y_270_, uint8_t v_suppressElabErrors_271_, lean_object* v_x_272_){
_start:
{
if (lean_obj_tag(v_x_272_) == 1)
{
lean_object* v_pre_273_; 
v_pre_273_ = lean_ctor_get(v_x_272_, 0);
if (lean_obj_tag(v_pre_273_) == 0)
{
lean_object* v_str_274_; lean_object* v___x_275_; uint8_t v___x_276_; 
v_str_274_ = lean_ctor_get(v_x_272_, 1);
v___x_275_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___closed__0));
v___x_276_ = lean_string_dec_eq(v_str_274_, v___x_275_);
if (v___x_276_ == 0)
{
return v___y_270_;
}
else
{
return v_suppressElabErrors_271_;
}
}
else
{
return v___y_270_;
}
}
else
{
return v___y_270_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___boxed(lean_object* v___y_277_, lean_object* v_suppressElabErrors_278_, lean_object* v_x_279_){
_start:
{
uint8_t v___y_10828__boxed_280_; uint8_t v_suppressElabErrors_boxed_281_; uint8_t v_res_282_; lean_object* v_r_283_; 
v___y_10828__boxed_280_ = lean_unbox(v___y_277_);
v_suppressElabErrors_boxed_281_ = lean_unbox(v_suppressElabErrors_278_);
v_res_282_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0(v___y_10828__boxed_280_, v_suppressElabErrors_boxed_281_, v_x_279_);
lean_dec(v_x_279_);
v_r_283_ = lean_box(v_res_282_);
return v_r_283_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__0);
v___x_286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
return v___x_286_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_287_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_288_ = lean_unsigned_to_nat(0u);
v___x_289_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v___x_288_);
lean_ctor_set(v___x_289_, 2, v___x_288_);
lean_ctor_set(v___x_289_, 3, v___x_288_);
lean_ctor_set(v___x_289_, 4, v___x_287_);
lean_ctor_set(v___x_289_, 5, v___x_287_);
lean_ctor_set(v___x_289_, 6, v___x_287_);
lean_ctor_set(v___x_289_, 7, v___x_287_);
lean_ctor_set(v___x_289_, 8, v___x_287_);
lean_ctor_set(v___x_289_, 9, v___x_287_);
return v___x_289_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_290_ = lean_unsigned_to_nat(32u);
v___x_291_ = lean_mk_empty_array_with_capacity(v___x_290_);
v___x_292_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_292_, 0, v___x_291_);
return v___x_292_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4(void){
_start:
{
size_t v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_293_ = ((size_t)5ULL);
v___x_294_ = lean_unsigned_to_nat(0u);
v___x_295_ = lean_unsigned_to_nat(32u);
v___x_296_ = lean_mk_empty_array_with_capacity(v___x_295_);
v___x_297_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__3);
v___x_298_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_298_, 0, v___x_297_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
lean_ctor_set(v___x_298_, 2, v___x_294_);
lean_ctor_set(v___x_298_, 3, v___x_294_);
lean_ctor_set_usize(v___x_298_, 4, v___x_293_);
return v___x_298_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_299_ = lean_box(1);
v___x_300_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__4);
v___x_301_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_302_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v___x_300_);
lean_ctor_set(v___x_302_, 2, v___x_299_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg(lean_object* v_msgData_303_, lean_object* v___y_304_){
_start:
{
lean_object* v___x_306_; lean_object* v_env_307_; lean_object* v___x_308_; lean_object* v_scopes_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v_opts_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_306_ = lean_st_ref_get(v___y_304_);
v_env_307_ = lean_ctor_get(v___x_306_, 0);
lean_inc_ref(v_env_307_);
lean_dec(v___x_306_);
v___x_308_ = lean_st_ref_get(v___y_304_);
v_scopes_309_ = lean_ctor_get(v___x_308_, 2);
lean_inc(v_scopes_309_);
lean_dec(v___x_308_);
v___x_310_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_311_ = l_List_head_x21___redArg(v___x_310_, v_scopes_309_);
lean_dec(v_scopes_309_);
v_opts_312_ = lean_ctor_get(v___x_311_, 1);
lean_inc_ref(v_opts_312_);
lean_dec(v___x_311_);
v___x_313_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__2);
v___x_314_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___closed__5);
v___x_315_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_315_, 0, v_env_307_);
lean_ctor_set(v___x_315_, 1, v___x_313_);
lean_ctor_set(v___x_315_, 2, v___x_314_);
lean_ctor_set(v___x_315_, 3, v_opts_312_);
v___x_316_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_msgData_303_);
v___x_317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg___boxed(lean_object* v_msgData_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg(v_msgData_318_, v___y_319_);
lean_dec(v___y_319_);
return v_res_321_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6(lean_object* v_opts_322_, lean_object* v_opt_323_){
_start:
{
lean_object* v_name_324_; lean_object* v_defValue_325_; lean_object* v_map_326_; lean_object* v___x_327_; 
v_name_324_ = lean_ctor_get(v_opt_323_, 0);
v_defValue_325_ = lean_ctor_get(v_opt_323_, 1);
v_map_326_ = lean_ctor_get(v_opts_322_, 0);
v___x_327_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_326_, v_name_324_);
if (lean_obj_tag(v___x_327_) == 0)
{
uint8_t v___x_328_; 
v___x_328_ = lean_unbox(v_defValue_325_);
return v___x_328_;
}
else
{
lean_object* v_val_329_; 
v_val_329_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_val_329_);
lean_dec_ref_known(v___x_327_, 1);
if (lean_obj_tag(v_val_329_) == 1)
{
uint8_t v_v_330_; 
v_v_330_ = lean_ctor_get_uint8(v_val_329_, 0);
lean_dec_ref_known(v_val_329_, 0);
return v_v_330_;
}
else
{
uint8_t v___x_331_; 
lean_dec(v_val_329_);
v___x_331_ = lean_unbox(v_defValue_325_);
return v___x_331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6___boxed(lean_object* v_opts_332_, lean_object* v_opt_333_){
_start:
{
uint8_t v_res_334_; lean_object* v_r_335_; 
v_res_334_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6(v_opts_332_, v_opt_333_);
lean_dec_ref(v_opt_333_);
lean_dec_ref(v_opts_332_);
v_r_335_ = lean_box(v_res_334_);
return v_r_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4(lean_object* v_ref_337_, lean_object* v_msgData_338_, uint8_t v_severity_339_, uint8_t v_isSilent_340_, lean_object* v___y_341_, lean_object* v___y_342_){
_start:
{
uint8_t v___y_345_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; uint8_t v___y_350_; lean_object* v___y_351_; lean_object* v___y_352_; uint8_t v___y_409_; uint8_t v___y_410_; uint8_t v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; uint8_t v___y_437_; uint8_t v___y_438_; lean_object* v___y_439_; uint8_t v___y_440_; lean_object* v___y_441_; uint8_t v___y_445_; uint8_t v___y_446_; uint8_t v___y_447_; uint8_t v___x_462_; uint8_t v___y_464_; uint8_t v___y_465_; uint8_t v___y_466_; uint8_t v___y_468_; uint8_t v___x_480_; 
v___x_462_ = 2;
v___x_480_ = l_Lean_instBEqMessageSeverity_beq(v_severity_339_, v___x_462_);
if (v___x_480_ == 0)
{
v___y_468_ = v___x_480_;
goto v___jp_467_;
}
else
{
uint8_t v___x_481_; 
lean_inc_ref(v_msgData_338_);
v___x_481_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_338_);
v___y_468_ = v___x_481_;
goto v___jp_467_;
}
v___jp_344_:
{
lean_object* v___x_353_; 
v___x_353_ = l_Lean_Elab_Command_getScope___redArg(v___y_352_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v_a_354_; lean_object* v___x_355_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
lean_inc(v_a_354_);
lean_dec_ref_known(v___x_353_, 1);
v___x_355_ = l_Lean_Elab_Command_getScope___redArg(v___y_352_);
if (lean_obj_tag(v___x_355_) == 0)
{
lean_object* v_a_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_391_; 
v_a_356_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_391_ == 0)
{
v___x_358_ = v___x_355_;
v_isShared_359_ = v_isSharedCheck_391_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_a_356_);
lean_dec(v___x_355_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_391_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_360_; lean_object* v_currNamespace_361_; lean_object* v_openDecls_362_; lean_object* v_env_363_; lean_object* v_messages_364_; lean_object* v_scopes_365_; lean_object* v_usedQuotCtxts_366_; lean_object* v_nextMacroScope_367_; lean_object* v_maxRecDepth_368_; lean_object* v_ngen_369_; lean_object* v_auxDeclNGen_370_; lean_object* v_infoState_371_; lean_object* v_traceState_372_; lean_object* v_snapshotTasks_373_; lean_object* v_prevLinterStates_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_390_; 
v___x_360_ = lean_st_ref_take(v___y_352_);
v_currNamespace_361_ = lean_ctor_get(v_a_354_, 2);
lean_inc(v_currNamespace_361_);
lean_dec(v_a_354_);
v_openDecls_362_ = lean_ctor_get(v_a_356_, 3);
lean_inc(v_openDecls_362_);
lean_dec(v_a_356_);
v_env_363_ = lean_ctor_get(v___x_360_, 0);
v_messages_364_ = lean_ctor_get(v___x_360_, 1);
v_scopes_365_ = lean_ctor_get(v___x_360_, 2);
v_usedQuotCtxts_366_ = lean_ctor_get(v___x_360_, 3);
v_nextMacroScope_367_ = lean_ctor_get(v___x_360_, 4);
v_maxRecDepth_368_ = lean_ctor_get(v___x_360_, 5);
v_ngen_369_ = lean_ctor_get(v___x_360_, 6);
v_auxDeclNGen_370_ = lean_ctor_get(v___x_360_, 7);
v_infoState_371_ = lean_ctor_get(v___x_360_, 8);
v_traceState_372_ = lean_ctor_get(v___x_360_, 9);
v_snapshotTasks_373_ = lean_ctor_get(v___x_360_, 10);
v_prevLinterStates_374_ = lean_ctor_get(v___x_360_, 11);
v_isSharedCheck_390_ = !lean_is_exclusive(v___x_360_);
if (v_isSharedCheck_390_ == 0)
{
v___x_376_ = v___x_360_;
v_isShared_377_ = v_isSharedCheck_390_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_prevLinterStates_374_);
lean_inc(v_snapshotTasks_373_);
lean_inc(v_traceState_372_);
lean_inc(v_infoState_371_);
lean_inc(v_auxDeclNGen_370_);
lean_inc(v_ngen_369_);
lean_inc(v_maxRecDepth_368_);
lean_inc(v_nextMacroScope_367_);
lean_inc(v_usedQuotCtxts_366_);
lean_inc(v_scopes_365_);
lean_inc(v_messages_364_);
lean_inc(v_env_363_);
lean_dec(v___x_360_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_390_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_383_; 
v___x_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_378_, 0, v_currNamespace_361_);
lean_ctor_set(v___x_378_, 1, v_openDecls_362_);
v___x_379_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
lean_ctor_set(v___x_379_, 1, v___y_348_);
lean_inc_ref(v___y_351_);
lean_inc_ref(v___y_346_);
v___x_380_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_380_, 0, v___y_346_);
lean_ctor_set(v___x_380_, 1, v___y_347_);
lean_ctor_set(v___x_380_, 2, v___y_349_);
lean_ctor_set(v___x_380_, 3, v___y_351_);
lean_ctor_set(v___x_380_, 4, v___x_379_);
lean_ctor_set_uint8(v___x_380_, sizeof(void*)*5, v___y_345_);
lean_ctor_set_uint8(v___x_380_, sizeof(void*)*5 + 1, v___y_350_);
lean_ctor_set_uint8(v___x_380_, sizeof(void*)*5 + 2, v_isSilent_340_);
v___x_381_ = l_Lean_MessageLog_add(v___x_380_, v_messages_364_);
if (v_isShared_377_ == 0)
{
lean_ctor_set(v___x_376_, 1, v___x_381_);
v___x_383_ = v___x_376_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_389_; 
v_reuseFailAlloc_389_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_389_, 0, v_env_363_);
lean_ctor_set(v_reuseFailAlloc_389_, 1, v___x_381_);
lean_ctor_set(v_reuseFailAlloc_389_, 2, v_scopes_365_);
lean_ctor_set(v_reuseFailAlloc_389_, 3, v_usedQuotCtxts_366_);
lean_ctor_set(v_reuseFailAlloc_389_, 4, v_nextMacroScope_367_);
lean_ctor_set(v_reuseFailAlloc_389_, 5, v_maxRecDepth_368_);
lean_ctor_set(v_reuseFailAlloc_389_, 6, v_ngen_369_);
lean_ctor_set(v_reuseFailAlloc_389_, 7, v_auxDeclNGen_370_);
lean_ctor_set(v_reuseFailAlloc_389_, 8, v_infoState_371_);
lean_ctor_set(v_reuseFailAlloc_389_, 9, v_traceState_372_);
lean_ctor_set(v_reuseFailAlloc_389_, 10, v_snapshotTasks_373_);
lean_ctor_set(v_reuseFailAlloc_389_, 11, v_prevLinterStates_374_);
v___x_383_ = v_reuseFailAlloc_389_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_387_; 
v___x_384_ = lean_st_ref_set(v___y_352_, v___x_383_);
v___x_385_ = lean_box(0);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 0, v___x_385_);
v___x_387_ = v___x_358_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v___x_385_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
}
}
else
{
lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_399_; 
lean_dec(v_a_354_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec_ref(v___y_347_);
v_a_392_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_399_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_399_ == 0)
{
v___x_394_ = v___x_355_;
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_355_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_397_; 
if (v_isShared_395_ == 0)
{
v___x_397_ = v___x_394_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_a_392_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec_ref(v___y_347_);
v_a_400_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_353_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_353_);
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
v___jp_408_:
{
lean_object* v_fileName_414_; lean_object* v_fileMap_415_; uint8_t v_suppressElabErrors_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_435_; 
v_fileName_414_ = lean_ctor_get(v___y_341_, 0);
v_fileMap_415_ = lean_ctor_get(v___y_341_, 1);
v_suppressElabErrors_416_ = lean_ctor_get_uint8(v___y_341_, sizeof(void*)*10);
v___x_417_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_338_);
v___x_418_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg(v___x_417_, v___y_342_);
v_a_419_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_435_ == 0)
{
v___x_421_ = v___x_418_;
v_isShared_422_ = v_isSharedCheck_435_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_418_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_435_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
lean_inc_ref_n(v_fileMap_415_, 2);
v___x_423_ = l_Lean_FileMap_toPosition(v_fileMap_415_, v___y_412_);
lean_dec(v___y_412_);
v___x_424_ = l_Lean_FileMap_toPosition(v_fileMap_415_, v___y_413_);
lean_dec(v___y_413_);
v___x_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_425_, 0, v___x_424_);
v___x_426_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___closed__0));
if (v_suppressElabErrors_416_ == 0)
{
lean_del_object(v___x_421_);
v___y_345_ = v___y_410_;
v___y_346_ = v_fileName_414_;
v___y_347_ = v___x_423_;
v___y_348_ = v_a_419_;
v___y_349_ = v___x_425_;
v___y_350_ = v___y_411_;
v___y_351_ = v___x_426_;
v___y_352_ = v___y_342_;
goto v___jp_344_;
}
else
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___f_429_; uint8_t v___x_430_; 
v___x_427_ = lean_box(v___y_409_);
v___x_428_ = lean_box(v_suppressElabErrors_416_);
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_429_, 0, v___x_427_);
lean_closure_set(v___f_429_, 1, v___x_428_);
lean_inc(v_a_419_);
v___x_430_ = l_Lean_MessageData_hasTag(v___f_429_, v_a_419_);
if (v___x_430_ == 0)
{
lean_object* v___x_431_; lean_object* v___x_433_; 
lean_dec_ref_known(v___x_425_, 1);
lean_dec_ref(v___x_423_);
lean_dec(v_a_419_);
v___x_431_ = lean_box(0);
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 0, v___x_431_);
v___x_433_ = v___x_421_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v___x_431_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
else
{
lean_del_object(v___x_421_);
v___y_345_ = v___y_410_;
v___y_346_ = v_fileName_414_;
v___y_347_ = v___x_423_;
v___y_348_ = v_a_419_;
v___y_349_ = v___x_425_;
v___y_350_ = v___y_411_;
v___y_351_ = v___x_426_;
v___y_352_ = v___y_342_;
goto v___jp_344_;
}
}
}
}
v___jp_436_:
{
lean_object* v___x_442_; 
v___x_442_ = l_Lean_Syntax_getTailPos_x3f(v___y_439_, v___y_438_);
lean_dec(v___y_439_);
if (lean_obj_tag(v___x_442_) == 0)
{
lean_inc(v___y_441_);
v___y_409_ = v___y_437_;
v___y_410_ = v___y_438_;
v___y_411_ = v___y_440_;
v___y_412_ = v___y_441_;
v___y_413_ = v___y_441_;
goto v___jp_408_;
}
else
{
lean_object* v_val_443_; 
v_val_443_ = lean_ctor_get(v___x_442_, 0);
lean_inc(v_val_443_);
lean_dec_ref_known(v___x_442_, 1);
v___y_409_ = v___y_437_;
v___y_410_ = v___y_438_;
v___y_411_ = v___y_440_;
v___y_412_ = v___y_441_;
v___y_413_ = v_val_443_;
goto v___jp_408_;
}
}
v___jp_444_:
{
lean_object* v___x_448_; 
v___x_448_ = l_Lean_Elab_Command_getRef___redArg(v___y_341_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v_a_449_; lean_object* v_ref_450_; lean_object* v___x_451_; 
v_a_449_ = lean_ctor_get(v___x_448_, 0);
lean_inc(v_a_449_);
lean_dec_ref_known(v___x_448_, 1);
v_ref_450_ = l_Lean_replaceRef(v_ref_337_, v_a_449_);
lean_dec(v_a_449_);
v___x_451_ = l_Lean_Syntax_getPos_x3f(v_ref_450_, v___y_446_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v___x_452_; 
v___x_452_ = lean_unsigned_to_nat(0u);
v___y_437_ = v___y_445_;
v___y_438_ = v___y_446_;
v___y_439_ = v_ref_450_;
v___y_440_ = v___y_447_;
v___y_441_ = v___x_452_;
goto v___jp_436_;
}
else
{
lean_object* v_val_453_; 
v_val_453_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_val_453_);
lean_dec_ref_known(v___x_451_, 1);
v___y_437_ = v___y_445_;
v___y_438_ = v___y_446_;
v___y_439_ = v_ref_450_;
v___y_440_ = v___y_447_;
v___y_441_ = v_val_453_;
goto v___jp_436_;
}
}
else
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_461_; 
lean_dec_ref(v_msgData_338_);
v_a_454_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_461_ == 0)
{
v___x_456_ = v___x_448_;
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_448_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_459_; 
if (v_isShared_457_ == 0)
{
v___x_459_ = v___x_456_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_a_454_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
v___jp_463_:
{
if (v___y_466_ == 0)
{
v___y_445_ = v___y_464_;
v___y_446_ = v___y_465_;
v___y_447_ = v_severity_339_;
goto v___jp_444_;
}
else
{
v___y_445_ = v___y_464_;
v___y_446_ = v___y_465_;
v___y_447_ = v___x_462_;
goto v___jp_444_;
}
}
v___jp_467_:
{
if (v___y_468_ == 0)
{
lean_object* v___x_469_; lean_object* v_scopes_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v_opts_473_; uint8_t v___x_474_; uint8_t v___x_475_; 
v___x_469_ = lean_st_ref_get(v___y_342_);
v_scopes_470_ = lean_ctor_get(v___x_469_, 2);
lean_inc(v_scopes_470_);
lean_dec(v___x_469_);
v___x_471_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_472_ = l_List_head_x21___redArg(v___x_471_, v_scopes_470_);
lean_dec(v_scopes_470_);
v_opts_473_ = lean_ctor_get(v___x_472_, 1);
lean_inc_ref(v_opts_473_);
lean_dec(v___x_472_);
v___x_474_ = 1;
v___x_475_ = l_Lean_instBEqMessageSeverity_beq(v_severity_339_, v___x_474_);
if (v___x_475_ == 0)
{
lean_dec_ref(v_opts_473_);
v___y_464_ = v___y_468_;
v___y_465_ = v___y_468_;
v___y_466_ = v___x_475_;
goto v___jp_463_;
}
else
{
lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_476_ = l_Lean_warningAsError;
v___x_477_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__6(v_opts_473_, v___x_476_);
lean_dec_ref(v_opts_473_);
v___y_464_ = v___y_468_;
v___y_465_ = v___y_468_;
v___y_466_ = v___x_477_;
goto v___jp_463_;
}
}
else
{
lean_object* v___x_478_; lean_object* v___x_479_; 
lean_dec_ref(v_msgData_338_);
v___x_478_ = lean_box(0);
v___x_479_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_479_, 0, v___x_478_);
return v___x_479_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4___boxed(lean_object* v_ref_482_, lean_object* v_msgData_483_, lean_object* v_severity_484_, lean_object* v_isSilent_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
uint8_t v_severity_boxed_489_; uint8_t v_isSilent_boxed_490_; lean_object* v_res_491_; 
v_severity_boxed_489_ = lean_unbox(v_severity_484_);
v_isSilent_boxed_490_ = lean_unbox(v_isSilent_485_);
v_res_491_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4(v_ref_482_, v_msgData_483_, v_severity_boxed_489_, v_isSilent_boxed_490_, v___y_486_, v___y_487_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec(v_ref_482_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2(lean_object* v_ref_492_, lean_object* v_msgData_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
uint8_t v___x_497_; uint8_t v___x_498_; lean_object* v___x_499_; 
v___x_497_ = 1;
v___x_498_ = 0;
v___x_499_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4(v_ref_492_, v_msgData_493_, v___x_497_, v___x_498_, v___y_494_, v___y_495_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2___boxed(lean_object* v_ref_500_, lean_object* v_msgData_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2(v_ref_500_, v_msgData_501_, v___y_502_, v___y_503_);
lean_dec(v___y_503_);
lean_dec_ref(v___y_502_);
lean_dec(v_ref_500_);
return v_res_505_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_507_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__0));
v___x_508_ = l_Lean_stringToMessageData(v___x_507_);
return v___x_508_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_510_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__2));
v___x_511_ = l_Lean_stringToMessageData(v___x_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1(lean_object* v_linterOption_512_, lean_object* v_stx_513_, lean_object* v_msg_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v_name_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_536_; 
v_name_518_ = lean_ctor_get(v_linterOption_512_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v_linterOption_512_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; 
v_unused_537_ = lean_ctor_get(v_linterOption_512_, 1);
lean_dec(v_unused_537_);
v___x_520_ = v_linterOption_512_;
v_isShared_521_ = v_isSharedCheck_536_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_name_518_);
lean_dec(v_linterOption_512_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_536_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_525_; 
v___x_522_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__1);
lean_inc(v_name_518_);
v___x_523_ = l_Lean_MessageData_ofName(v_name_518_);
if (v_isShared_521_ == 0)
{
lean_ctor_set_tag(v___x_520_, 7);
lean_ctor_set(v___x_520_, 1, v___x_523_);
lean_ctor_set(v___x_520_, 0, v___x_522_);
v___x_525_ = v___x_520_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v___x_522_);
lean_ctor_set(v_reuseFailAlloc_535_, 1, v___x_523_);
v___x_525_ = v_reuseFailAlloc_535_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v_disable_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_526_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___closed__3);
v___x_527_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_525_);
lean_ctor_set(v___x_527_, 1, v___x_526_);
v_disable_528_ = l_Lean_MessageData_note(v___x_527_);
v___x_529_ = l_Lean_Linter_linterMessageTag;
v___x_530_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_530_, 0, v_msg_514_);
lean_ctor_set(v___x_530_, 1, v_disable_528_);
v___x_531_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_529_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_532_, 0, v_name_518_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
lean_inc(v_stx_513_);
v___x_533_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_533_, 0, v_stx_513_);
lean_ctor_set(v___x_533_, 1, v___x_532_);
v___x_534_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2(v_stx_513_, v___x_533_, v___y_515_, v___y_516_);
lean_dec(v_stx_513_);
return v___x_534_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1___boxed(lean_object* v_linterOption_538_, lean_object* v_stx_539_, lean_object* v_msg_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_){
_start:
{
lean_object* v_res_544_; 
v_res_544_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1(v_linterOption_538_, v_stx_539_, v_msg_540_, v___y_541_, v___y_542_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg(lean_object* v_o_545_, lean_object* v___y_546_){
_start:
{
lean_object* v___x_548_; lean_object* v_env_549_; lean_object* v___x_550_; lean_object* v_toEnvExtension_551_; lean_object* v_asyncMode_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v_merged_556_; lean_object* v___x_558_; uint8_t v_isShared_559_; uint8_t v_isSharedCheck_564_; 
v___x_548_ = lean_st_ref_get(v___y_546_);
v_env_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc_ref(v_env_549_);
lean_dec(v___x_548_);
v___x_550_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_551_ = lean_ctor_get(v___x_550_, 0);
v_asyncMode_552_ = lean_ctor_get(v_toEnvExtension_551_, 2);
v___x_553_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_554_ = lean_box(0);
v___x_555_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_553_, v___x_550_, v_env_549_, v_asyncMode_552_, v___x_554_);
v_merged_556_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_564_ == 0)
{
lean_object* v_unused_565_; 
v_unused_565_ = lean_ctor_get(v___x_555_, 1);
lean_dec(v_unused_565_);
v___x_558_ = v___x_555_;
v_isShared_559_ = v_isSharedCheck_564_;
goto v_resetjp_557_;
}
else
{
lean_inc(v_merged_556_);
lean_dec(v___x_555_);
v___x_558_ = lean_box(0);
v_isShared_559_ = v_isSharedCheck_564_;
goto v_resetjp_557_;
}
v_resetjp_557_:
{
lean_object* v___x_561_; 
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 1, v_merged_556_);
lean_ctor_set(v___x_558_, 0, v_o_545_);
v___x_561_ = v___x_558_;
goto v_reusejp_560_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_o_545_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v_merged_556_);
v___x_561_ = v_reuseFailAlloc_563_;
goto v_reusejp_560_;
}
v_reusejp_560_:
{
lean_object* v___x_562_; 
v___x_562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_562_, 0, v___x_561_);
return v___x_562_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_566_, lean_object* v___y_567_, lean_object* v___y_568_){
_start:
{
lean_object* v_res_569_; 
v_res_569_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg(v_o_566_, v___y_567_);
lean_dec(v___y_567_);
return v_res_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(lean_object* v___y_570_, lean_object* v___y_571_){
_start:
{
lean_object* v___x_573_; lean_object* v_scopes_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v_opts_577_; lean_object* v___x_578_; 
v___x_573_ = lean_st_ref_get(v___y_571_);
v_scopes_574_ = lean_ctor_get(v___x_573_, 2);
lean_inc(v_scopes_574_);
lean_dec(v___x_573_);
v___x_575_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_576_ = l_List_head_x21___redArg(v___x_575_, v_scopes_574_);
lean_dec(v_scopes_574_);
v_opts_577_ = lean_ctor_get(v___x_576_, 1);
lean_inc_ref(v_opts_577_);
lean_dec(v___x_576_);
v___x_578_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg(v_opts_577_, v___y_571_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0___boxed(lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(v___y_579_, v___y_580_);
lean_dec(v___y_580_);
lean_dec_ref(v___y_579_);
return v_res_582_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1(void){
_start:
{
lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_584_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__0));
v___x_585_ = l_Lean_stringToMessageData(v___x_584_);
return v___x_585_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3(void){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_587_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__2));
v___x_588_ = l_Lean_stringToMessageData(v___x_587_);
return v___x_588_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9(void){
_start:
{
lean_object* v___x_598_; lean_object* v___x_599_; 
v___x_598_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_));
v___x_599_ = l_Lean_mkIdent(v___x_598_);
return v___x_599_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12(void){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = l_Array_mkArray0(lean_box(0));
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0(lean_object* v_stx_605_, lean_object* v___y_606_, lean_object* v___y_607_){
_start:
{
lean_object* v___x_615_; lean_object* v_a_616_; lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_810_; 
v___x_615_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(v___y_606_, v___y_607_);
v_a_616_ = lean_ctor_get(v___x_615_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_615_);
if (v_isSharedCheck_810_ == 0)
{
v___x_618_ = v___x_615_;
v_isShared_619_ = v_isSharedCheck_810_;
goto v_resetjp_617_;
}
else
{
lean_inc(v_a_616_);
lean_dec(v___x_615_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_810_;
goto v_resetjp_617_;
}
v___jp_609_:
{
lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_610_ = lean_box(0);
v___x_611_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_611_, 0, v___x_610_);
return v___x_611_;
}
v___jp_612_:
{
lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_613_ = lean_box(0);
v___x_614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_614_, 0, v___x_613_);
return v___x_614_;
}
v_resetjp_617_:
{
lean_object* v___x_620_; uint8_t v___y_622_; lean_object* v___y_623_; lean_object* v___y_624_; lean_object* v___y_625_; lean_object* v___y_626_; uint8_t v___y_681_; lean_object* v___y_682_; lean_object* v___y_683_; uint8_t v___y_684_; uint8_t v___y_699_; lean_object* v___y_700_; uint8_t v___y_701_; lean_object* v___y_702_; lean_object* v___y_703_; uint8_t v___y_706_; lean_object* v___y_707_; uint8_t v___y_708_; lean_object* v___y_709_; lean_object* v___y_710_; uint8_t v___y_711_; uint8_t v___x_712_; uint8_t v___y_714_; uint8_t v___y_715_; uint8_t v___y_716_; lean_object* v___y_717_; lean_object* v___y_718_; uint8_t v___y_719_; lean_object* v___y_738_; uint8_t v___y_739_; uint8_t v___y_740_; 
v___x_620_ = lp_mathlib_Mathlib_Linter_linter_upstreamableDecl;
v___x_712_ = l_Lean_Linter_getLinterValue(v___x_620_, v_a_616_);
lean_dec(v_a_616_);
if (v___x_712_ == 0)
{
lean_object* v___x_768_; lean_object* v___x_769_; 
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v___x_768_ = lean_box(0);
v___x_769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
return v___x_769_;
}
else
{
lean_object* v___x_770_; lean_object* v_messages_771_; uint8_t v___x_772_; uint8_t v___y_774_; uint8_t v___y_775_; 
v___x_770_ = lean_st_ref_get(v___y_607_);
v_messages_771_ = lean_ctor_get(v___x_770_, 1);
lean_inc_ref(v_messages_771_);
lean_dec(v___x_770_);
v___x_772_ = l_Lean_MessageLog_hasErrors(v_messages_771_);
lean_dec_ref(v_messages_771_);
if (v___x_772_ == 0)
{
lean_object* v___x_798_; lean_object* v_a_799_; uint8_t v___y_801_; lean_object* v___x_806_; uint8_t v___x_807_; 
v___x_798_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(v___y_606_, v___y_607_);
v_a_799_ = lean_ctor_get(v___x_798_, 0);
lean_inc(v_a_799_);
lean_dec_ref(v___x_798_);
v___x_806_ = lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_defs;
v___x_807_ = l_Lean_Linter_getLinterValue(v___x_806_, v_a_799_);
lean_dec(v_a_799_);
if (v___x_807_ == 0)
{
v___y_801_ = v___x_712_;
goto v___jp_800_;
}
else
{
v___y_801_ = v___x_772_;
goto v___jp_800_;
}
v___jp_800_:
{
lean_object* v___x_802_; lean_object* v_a_803_; lean_object* v___x_804_; uint8_t v___x_805_; 
v___x_802_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0(v___y_606_, v___y_607_);
v_a_803_ = lean_ctor_get(v___x_802_, 0);
lean_inc(v_a_803_);
lean_dec_ref(v___x_802_);
v___x_804_ = lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_private;
v___x_805_ = l_Lean_Linter_getLinterValue(v___x_804_, v_a_803_);
lean_dec(v_a_803_);
if (v___x_805_ == 0)
{
v___y_774_ = v___y_801_;
v___y_775_ = v___x_712_;
goto v___jp_773_;
}
else
{
v___y_774_ = v___y_801_;
v___y_775_ = v___x_772_;
goto v___jp_773_;
}
}
}
else
{
lean_object* v___x_808_; lean_object* v___x_809_; 
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v___x_808_ = lean_box(0);
v___x_809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_809_, 0, v___x_808_);
return v___x_809_;
}
v___jp_773_:
{
lean_object* v___x_776_; 
v___x_776_ = l_Lean_Elab_Command_getRef___redArg(v___y_606_);
if (lean_obj_tag(v___x_776_) == 0)
{
lean_object* v_a_777_; lean_object* v___x_778_; 
v_a_777_ = lean_ctor_get(v___x_776_, 0);
lean_inc(v_a_777_);
lean_dec_ref_known(v___x_776_, 1);
v___x_778_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_606_);
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v_quotContext_x3f_779_; lean_object* v___x_780_; 
lean_dec_ref_known(v___x_778_, 1);
v_quotContext_x3f_779_ = lean_ctor_get(v___y_606_, 5);
v___x_780_ = l_Lean_SourceInfo_fromRef(v_a_777_, v___x_772_);
lean_dec(v_a_777_);
if (lean_obj_tag(v_quotContext_x3f_779_) == 0)
{
lean_object* v___x_781_; 
v___x_781_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__2___redArg(v___y_607_);
lean_dec_ref(v___x_781_);
v___y_738_ = v___x_780_;
v___y_739_ = v___y_775_;
v___y_740_ = v___y_774_;
goto v___jp_737_;
}
else
{
v___y_738_ = v___x_780_;
v___y_739_ = v___y_775_;
v___y_740_ = v___y_774_;
goto v___jp_737_;
}
}
else
{
lean_object* v_a_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_789_; 
lean_dec(v_a_777_);
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v_a_782_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_789_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_789_ == 0)
{
v___x_784_ = v___x_778_;
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_a_782_);
lean_dec(v___x_778_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_787_; 
if (v_isShared_785_ == 0)
{
v___x_787_ = v___x_784_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v_a_782_);
v___x_787_ = v_reuseFailAlloc_788_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
return v___x_787_;
}
}
}
}
else
{
lean_object* v_a_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_797_; 
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v_a_790_ = lean_ctor_get(v___x_776_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_776_);
if (v_isSharedCheck_797_ == 0)
{
v___x_792_ = v___x_776_;
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_a_790_);
lean_dec(v___x_776_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_795_; 
if (v_isShared_793_ == 0)
{
v___x_795_ = v___x_792_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_a_790_);
v___x_795_ = v_reuseFailAlloc_796_;
goto v_reusejp_794_;
}
v_reusejp_794_:
{
return v___x_795_;
}
}
}
}
}
v___jp_621_:
{
lean_object* v___x_627_; uint8_t v___x_628_; 
v___x_627_ = lean_unsigned_to_nat(1u);
v___x_628_ = lean_nat_dec_eq(v___y_626_, v___x_627_);
lean_dec(v___y_626_);
if (v___x_628_ == 0)
{
lean_dec_ref(v___y_625_);
lean_dec(v___y_624_);
lean_dec(v___y_623_);
lean_dec(v_stx_605_);
goto v___jp_609_;
}
else
{
lean_object* v___x_629_; 
v___x_629_ = l_Std_DTreeMap_Internal_Impl_minKey_x3f___redArg(v___y_623_);
lean_dec(v___y_623_);
if (lean_obj_tag(v___x_629_) == 1)
{
lean_object* v_val_630_; lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_679_; 
v_val_630_ = lean_ctor_get(v___x_629_, 0);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_679_ == 0)
{
v___x_632_ = v___x_629_;
v_isShared_633_ = v_isSharedCheck_679_;
goto v_resetjp_631_;
}
else
{
lean_inc(v_val_630_);
lean_dec(v___x_629_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_679_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
lean_object* v___x_634_; 
v___x_634_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Lean_Environment_localDefinitionDependencies(v___y_625_, v_stx_605_, v___y_624_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_634_) == 0)
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_670_; 
v_a_635_ = lean_ctor_get(v___x_634_, 0);
v_isSharedCheck_670_ = !lean_is_exclusive(v___x_634_);
if (v_isSharedCheck_670_ == 0)
{
v___x_637_ = v___x_634_;
v_isShared_638_ = v_isSharedCheck_670_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_634_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_670_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
uint8_t v___x_639_; 
v___x_639_ = lean_unbox(v_a_635_);
lean_dec(v_a_635_);
if (v___x_639_ == 0)
{
lean_object* v___x_640_; uint64_t v_javascriptHash_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
lean_del_object(v___x_637_);
v___x_640_ = lp_importGraph_GoToModuleLink;
v_javascriptHash_641_ = lean_ctor_get_uint64(v___x_640_, sizeof(void*)*1);
lean_inc(v_val_630_);
v___x_642_ = lean_alloc_closure((void*)(lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_), 2, 1);
lean_closure_set(v___x_642_, 0, v_val_630_);
v___x_643_ = lean_box_uint64(v_javascriptHash_641_);
v___x_644_ = lean_alloc_closure((void*)(l_Lean_Widget_WidgetInstance_ofHash___boxed), 5, 2);
lean_closure_set(v___x_644_, 0, v___x_643_);
lean_closure_set(v___x_644_, 1, v___x_642_);
v___x_645_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_644_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_645_) == 0)
{
lean_object* v_a_646_; lean_object* v___x_647_; lean_object* v___x_649_; 
v_a_646_ = lean_ctor_get(v___x_645_, 0);
lean_inc(v_a_646_);
lean_dec_ref_known(v___x_645_, 1);
v___x_647_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_630_, v___y_622_);
if (v_isShared_633_ == 0)
{
lean_ctor_set_tag(v___x_632_, 3);
lean_ctor_set(v___x_632_, 0, v___x_647_);
v___x_649_ = v___x_632_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_657_; 
v_reuseFailAlloc_657_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_657_, 0, v___x_647_);
v___x_649_ = v_reuseFailAlloc_657_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_650_ = l_Lean_MessageData_ofFormat(v___x_649_);
v___x_651_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_651_, 0, v_a_646_);
lean_ctor_set(v___x_651_, 1, v___x_650_);
v___x_652_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__1);
v___x_653_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_653_, 0, v___x_652_);
lean_ctor_set(v___x_653_, 1, v___x_651_);
v___x_654_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__3);
v___x_655_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_655_, 0, v___x_653_);
lean_ctor_set(v___x_655_, 1, v___x_654_);
v___x_656_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1(v___x_620_, v___y_624_, v___x_655_, v___y_606_, v___y_607_);
return v___x_656_;
}
}
else
{
lean_object* v_a_658_; lean_object* v___x_660_; uint8_t v_isShared_661_; uint8_t v_isSharedCheck_665_; 
lean_del_object(v___x_632_);
lean_dec(v_val_630_);
lean_dec(v___y_624_);
v_a_658_ = lean_ctor_get(v___x_645_, 0);
v_isSharedCheck_665_ = !lean_is_exclusive(v___x_645_);
if (v_isSharedCheck_665_ == 0)
{
v___x_660_ = v___x_645_;
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
else
{
lean_inc(v_a_658_);
lean_dec(v___x_645_);
v___x_660_ = lean_box(0);
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
v_resetjp_659_:
{
lean_object* v___x_663_; 
if (v_isShared_661_ == 0)
{
v___x_663_ = v___x_660_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_664_; 
v_reuseFailAlloc_664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_664_, 0, v_a_658_);
v___x_663_ = v_reuseFailAlloc_664_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
return v___x_663_;
}
}
}
}
else
{
lean_object* v___x_666_; lean_object* v___x_668_; 
lean_del_object(v___x_632_);
lean_dec(v_val_630_);
lean_dec(v___y_624_);
v___x_666_ = lean_box(0);
if (v_isShared_638_ == 0)
{
lean_ctor_set(v___x_637_, 0, v___x_666_);
v___x_668_ = v___x_637_;
goto v_reusejp_667_;
}
else
{
lean_object* v_reuseFailAlloc_669_; 
v_reuseFailAlloc_669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_669_, 0, v___x_666_);
v___x_668_ = v_reuseFailAlloc_669_;
goto v_reusejp_667_;
}
v_reusejp_667_:
{
return v___x_668_;
}
}
}
}
else
{
lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
lean_del_object(v___x_632_);
lean_dec(v_val_630_);
lean_dec(v___y_624_);
v_a_671_ = lean_ctor_get(v___x_634_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_634_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_634_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_dec(v___x_634_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
else
{
lean_dec(v___x_629_);
lean_dec_ref(v___y_625_);
lean_dec(v___y_624_);
lean_dec(v_stx_605_);
goto v___jp_609_;
}
}
}
v___jp_680_:
{
lean_object* v___x_685_; 
lean_inc(v_stx_605_);
v___x_685_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports(v_stx_605_, v___y_683_, v___y_684_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_685_) == 0)
{
lean_object* v_a_686_; lean_object* v___x_687_; 
v_a_686_ = lean_ctor_get(v___x_685_, 0);
lean_inc(v_a_686_);
lean_dec_ref_known(v___x_685_, 1);
v___x_687_ = lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(v___y_682_, v_a_686_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v_size_688_; 
v_size_688_ = lean_ctor_get(v___x_687_, 0);
lean_inc(v_size_688_);
v___y_622_ = v___y_681_;
v___y_623_ = v___x_687_;
v___y_624_ = v___y_683_;
v___y_625_ = v___y_682_;
v___y_626_ = v_size_688_;
goto v___jp_621_;
}
else
{
lean_object* v___x_689_; 
v___x_689_ = lean_unsigned_to_nat(0u);
v___y_622_ = v___y_681_;
v___y_623_ = v___x_687_;
v___y_624_ = v___y_683_;
v___y_625_ = v___y_682_;
v___y_626_ = v___x_689_;
goto v___jp_621_;
}
}
else
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec(v___y_683_);
lean_dec_ref(v___y_682_);
lean_dec(v_stx_605_);
v_a_690_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_685_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_685_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_695_; 
if (v_isShared_693_ == 0)
{
v___x_695_ = v___x_692_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_a_690_);
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
v___jp_698_:
{
if (v___y_701_ == 0)
{
lean_dec(v___y_700_);
v___y_681_ = v___y_699_;
v___y_682_ = v___y_703_;
v___y_683_ = v___y_702_;
v___y_684_ = v___y_701_;
goto v___jp_680_;
}
else
{
uint8_t v___x_704_; 
v___x_704_ = l_Lean_isPrivateName(v___y_700_);
lean_dec(v___y_700_);
if (v___x_704_ == 0)
{
v___y_681_ = v___y_699_;
v___y_682_ = v___y_703_;
v___y_683_ = v___y_702_;
v___y_684_ = v___x_704_;
goto v___jp_680_;
}
else
{
lean_dec_ref(v___y_703_);
lean_dec(v___y_702_);
lean_dec(v_stx_605_);
goto v___jp_612_;
}
}
}
v___jp_705_:
{
if (v___y_711_ == 0)
{
v___y_699_ = v___y_706_;
v___y_700_ = v___y_707_;
v___y_701_ = v___y_708_;
v___y_702_ = v___y_710_;
v___y_703_ = v___y_709_;
goto v___jp_698_;
}
else
{
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_707_);
lean_dec(v_stx_605_);
goto v___jp_612_;
}
}
v___jp_713_:
{
if (v___y_719_ == 0)
{
lean_object* v___x_720_; lean_object* v___x_722_; 
lean_dec_ref(v___y_718_);
lean_dec(v___y_717_);
lean_dec(v_stx_605_);
v___x_720_ = lean_box(0);
if (v_isShared_619_ == 0)
{
lean_ctor_set(v___x_618_, 0, v___x_720_);
v___x_722_ = v___x_618_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v___x_720_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
else
{
lean_object* v___x_724_; 
lean_del_object(v___x_618_);
lean_inc(v_stx_605_);
v___x_724_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName(v_stx_605_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_724_) == 0)
{
if (v___y_716_ == 0)
{
lean_object* v_a_725_; 
v_a_725_ = lean_ctor_get(v___x_724_, 0);
lean_inc(v_a_725_);
lean_dec_ref_known(v___x_724_, 1);
v___y_699_ = v___y_719_;
v___y_700_ = v_a_725_;
v___y_701_ = v___y_715_;
v___y_702_ = v___y_717_;
v___y_703_ = v___y_718_;
goto v___jp_698_;
}
else
{
lean_object* v_a_726_; lean_object* v___x_727_; 
v_a_726_ = lean_ctor_get(v___x_724_, 0);
lean_inc_n(v_a_726_, 2);
lean_dec_ref_known(v___x_724_, 1);
lean_inc_ref(v___y_718_);
v___x_727_ = l_Lean_Environment_find_x3f(v___y_718_, v_a_726_, v___y_714_);
if (lean_obj_tag(v___x_727_) == 1)
{
lean_object* v_val_728_; 
v_val_728_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_val_728_);
lean_dec_ref_known(v___x_727_, 1);
switch(lean_obj_tag(v_val_728_))
{
case 2:
{
lean_dec_ref_known(v_val_728_, 1);
v___y_706_ = v___y_719_;
v___y_707_ = v_a_726_;
v___y_708_ = v___y_715_;
v___y_709_ = v___y_718_;
v___y_710_ = v___y_717_;
v___y_711_ = v___y_714_;
goto v___jp_705_;
}
case 6:
{
lean_dec_ref_known(v_val_728_, 1);
v___y_706_ = v___y_719_;
v___y_707_ = v_a_726_;
v___y_708_ = v___y_715_;
v___y_709_ = v___y_718_;
v___y_710_ = v___y_717_;
v___y_711_ = v___y_714_;
goto v___jp_705_;
}
default: 
{
lean_dec(v_val_728_);
lean_dec(v_a_726_);
lean_dec_ref(v___y_718_);
lean_dec(v___y_717_);
lean_dec(v_stx_605_);
goto v___jp_612_;
}
}
}
else
{
lean_dec(v___x_727_);
v___y_706_ = v___y_719_;
v___y_707_ = v_a_726_;
v___y_708_ = v___y_715_;
v___y_709_ = v___y_718_;
v___y_710_ = v___y_717_;
v___y_711_ = v___x_712_;
goto v___jp_705_;
}
}
}
else
{
lean_object* v_a_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_736_; 
lean_dec_ref(v___y_718_);
lean_dec(v___y_717_);
lean_dec(v_stx_605_);
v_a_729_ = lean_ctor_get(v___x_724_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_724_);
if (v_isSharedCheck_736_ == 0)
{
v___x_731_ = v___x_724_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_a_729_);
lean_dec(v___x_724_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___x_734_; 
if (v_isShared_732_ == 0)
{
v___x_734_ = v___x_731_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v_a_729_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
}
}
v___jp_737_:
{
lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; uint8_t v___x_751_; 
v___x_741_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__7));
v___x_742_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__8));
lean_inc_n(v___y_738_, 3);
v___x_743_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_743_, 0, v___y_738_);
lean_ctor_set(v___x_743_, 1, v___x_741_);
v___x_744_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__9);
v___x_745_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__11));
v___x_746_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12, &lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__12);
v___x_747_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_747_, 0, v___y_738_);
lean_ctor_set(v___x_747_, 1, v___x_745_);
lean_ctor_set(v___x_747_, 2, v___x_746_);
v___x_748_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___closed__13));
v___x_749_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_749_, 0, v___y_738_);
lean_ctor_set(v___x_749_, 1, v___x_748_);
v___x_750_ = l_Lean_Syntax_node4(v___y_738_, v___x_742_, v___x_743_, v___x_744_, v___x_747_, v___x_749_);
v___x_751_ = l_Lean_Syntax_structEq(v_stx_605_, v___x_750_);
lean_dec(v___x_750_);
if (v___x_751_ == 0)
{
lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_752_ = lean_st_ref_get(v___y_607_);
lean_inc(v_stx_605_);
v___x_753_ = lp_mathlib_Mathlib_Command_MinImports_getId(v_stx_605_, v___y_606_, v___y_607_);
if (lean_obj_tag(v___x_753_) == 0)
{
lean_object* v_a_754_; lean_object* v_env_755_; lean_object* v___x_756_; uint8_t v___x_757_; 
v_a_754_ = lean_ctor_get(v___x_753_, 0);
lean_inc(v_a_754_);
lean_dec_ref_known(v___x_753_, 1);
v_env_755_ = lean_ctor_get(v___x_752_, 0);
lean_inc_ref(v_env_755_);
lean_dec(v___x_752_);
v___x_756_ = lean_box(0);
v___x_757_ = l_Lean_Syntax_structEq(v_a_754_, v___x_756_);
if (v___x_757_ == 0)
{
v___y_714_ = v___x_751_;
v___y_715_ = v___y_739_;
v___y_716_ = v___y_740_;
v___y_717_ = v_a_754_;
v___y_718_ = v_env_755_;
v___y_719_ = v___x_712_;
goto v___jp_713_;
}
else
{
v___y_714_ = v___x_751_;
v___y_715_ = v___y_739_;
v___y_716_ = v___y_740_;
v___y_717_ = v_a_754_;
v___y_718_ = v_env_755_;
v___y_719_ = v___x_751_;
goto v___jp_713_;
}
}
else
{
lean_object* v_a_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_765_; 
lean_dec(v___x_752_);
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v_a_758_ = lean_ctor_get(v___x_753_, 0);
v_isSharedCheck_765_ = !lean_is_exclusive(v___x_753_);
if (v_isSharedCheck_765_ == 0)
{
v___x_760_ = v___x_753_;
v_isShared_761_ = v_isSharedCheck_765_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_a_758_);
lean_dec(v___x_753_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_765_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v___x_763_; 
if (v_isShared_761_ == 0)
{
v___x_763_ = v___x_760_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_764_; 
v_reuseFailAlloc_764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_764_, 0, v_a_758_);
v___x_763_ = v_reuseFailAlloc_764_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
return v___x_763_;
}
}
}
}
else
{
lean_object* v___x_766_; lean_object* v___x_767_; 
lean_del_object(v___x_618_);
lean_dec(v_stx_605_);
v___x_766_ = lean_box(0);
v___x_767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_767_, 0, v___x_766_);
return v___x_767_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0___boxed(lean_object* v_stx_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter___lam__0(v_stx_811_, v___y_812_, v___y_813_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0(lean_object* v_o_858_, lean_object* v___y_859_, lean_object* v___y_860_){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___redArg(v_o_858_, v___y_860_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0___boxed(lean_object* v_o_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v_res_867_; 
v_res_867_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__0_spec__0(v_o_863_, v___y_864_, v___y_865_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
return v_res_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5(lean_object* v_msgData_868_, lean_object* v___y_869_, lean_object* v___y_870_){
_start:
{
lean_object* v___x_872_; 
v___x_872_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___redArg(v_msgData_868_, v___y_870_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5___boxed(lean_object* v_msgData_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter_spec__1_spec__2_spec__4_spec__5(v_msgData_873_, v___y_874_, v___y_875_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_879_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_upstreamableDeclLinter));
v___x_880_ = l_Lean_Elab_Command_addLinter(v___x_879_);
return v___x_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2____boxed(lean_object* v_a_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2_();
return v_res_882_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Tools_FindHome(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(uint8_t builtin) {
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
res = runtime_initialize_importGraph_ImportGraph_Tools_FindHome(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1466986157____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_upstreamableDecl = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_upstreamableDecl);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_3034280370____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_defs = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_defs);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_1218957806____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_private = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_upstreamableDecl_private);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_UpstreamableDecl_0__Mathlib_Linter_DoubleImports_initFn_00___x40_Mathlib_Tactic_Linter_UpstreamableDecl_380890088____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Tools_FindHome(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Tools_FindHome(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_UpstreamableDecl(builtin);
}
#ifdef __cplusplus
}
#endif
