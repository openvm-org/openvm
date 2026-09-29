// Lean compiler output
// Module: Mathlib.Tactic.Linter.HaveILetI
// Imports: public import Init public meta import Init public meta import Lean.Meta.Hint public import Mathlib.Tactic.Linter.Header public import Lean.Meta.TryThis
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Hint_mkSuggestionsMessage(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
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
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "haveILetI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(109, 31, 218, 255, 150, 232, 112, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "enable the `haveILetI` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "HaveILetI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 89, 237, 100, 137, 154, 97, 152)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 149, 196, 100, 240, 152, 224, 188)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(196, 243, 144, 204, 248, 162, 254, 197)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(100, 168, 77, 89, 83, 223, 75, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_linter_style_haveILetI;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "have"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__3_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Try this: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 209, .m_capacity = 209, .m_length = 208, .m_data = "\n\nThe goal is a proposition, so `have` is preferred over `haveI`.\nThe difference between `have` and `haveI` is that `haveI` inlines the value.\nBut this is not relevant for proofs because of proof irrelevance."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticHaveI__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__2_value),LEAN_SCALAR_PTR_LITERAL(17, 169, 15, 78, 195, 212, 131, 76)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "haveI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 89, 237, 100, 137, 154, 97, 152)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__2_value),LEAN_SCALAR_PTR_LITERAL(33, 104, 160, 24, 243, 114, 216, 52)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(213, 164, 178, 9, 35, 200, 242, 88)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(237, 158, 72, 239, 156, 118, 8, 209)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI____ = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "let"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__3_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 204, .m_capacity = 204, .m_length = 203, .m_data = "\n\nThe goal is a proposition, so `let` is preferred over `letI`.\nThe difference between `let` and `letI` is that `letI` inlines the value.\nBut this is not relevant for proofs because of proof irrelevance."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticLetI__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__0_value),LEAN_SCALAR_PTR_LITERAL(190, 147, 42, 183, 35, 252, 246, 67)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "letI"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 89, 237, 100, 137, 154, 97, 152)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 199, 73, 213, 253, 17, 200, 162)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI____ = (const lean_object*)&lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticLetI______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticLetI______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_));
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_));
v___x_60_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn___closed__9_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_));
v___x_61_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4__spec__0(v___x_58_, v___x_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4____boxed(lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_();
return v_res_63_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5(lean_object* v_opts_64_, lean_object* v_opt_65_){
_start:
{
lean_object* v_name_66_; lean_object* v_defValue_67_; lean_object* v_map_68_; lean_object* v___x_69_; 
v_name_66_ = lean_ctor_get(v_opt_65_, 0);
v_defValue_67_ = lean_ctor_get(v_opt_65_, 1);
v_map_68_ = lean_ctor_get(v_opts_64_, 0);
v___x_69_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_68_, v_name_66_);
if (lean_obj_tag(v___x_69_) == 0)
{
uint8_t v___x_70_; 
v___x_70_ = lean_unbox(v_defValue_67_);
return v___x_70_;
}
else
{
lean_object* v_val_71_; 
v_val_71_ = lean_ctor_get(v___x_69_, 0);
lean_inc(v_val_71_);
lean_dec_ref_known(v___x_69_, 1);
if (lean_obj_tag(v_val_71_) == 1)
{
uint8_t v_v_72_; 
v_v_72_ = lean_ctor_get_uint8(v_val_71_, 0);
lean_dec_ref_known(v_val_71_, 0);
return v_v_72_;
}
else
{
uint8_t v___x_73_; 
lean_dec(v_val_71_);
v___x_73_ = lean_unbox(v_defValue_67_);
return v___x_73_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5___boxed(lean_object* v_opts_74_, lean_object* v_opt_75_){
_start:
{
uint8_t v_res_76_; lean_object* v_r_77_; 
v_res_76_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5(v_opts_74_, v_opt_75_);
lean_dec_ref(v_opt_75_);
lean_dec_ref(v_opts_74_);
v_r_77_ = lean_box(v_res_76_);
return v_r_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4(lean_object* v_msgData_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v___x_84_; lean_object* v_env_85_; lean_object* v___x_86_; lean_object* v_mctx_87_; lean_object* v_lctx_88_; lean_object* v_options_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_84_ = lean_st_ref_get(v___y_82_);
v_env_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc_ref(v_env_85_);
lean_dec(v___x_84_);
v___x_86_ = lean_st_ref_get(v___y_80_);
v_mctx_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc_ref(v_mctx_87_);
lean_dec(v___x_86_);
v_lctx_88_ = lean_ctor_get(v___y_79_, 2);
v_options_89_ = lean_ctor_get(v___y_81_, 2);
lean_inc_ref(v_options_89_);
lean_inc_ref(v_lctx_88_);
v___x_90_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_90_, 0, v_env_85_);
lean_ctor_set(v___x_90_, 1, v_mctx_87_);
lean_ctor_set(v___x_90_, 2, v_lctx_88_);
lean_ctor_set(v___x_90_, 3, v_options_89_);
v___x_91_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v_msgData_78_);
v___x_92_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4___boxed(lean_object* v_msgData_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4(v_msgData_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
return v_res_99_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0(uint8_t v___y_108_, uint8_t v_suppressElabErrors_109_, lean_object* v_x_110_){
_start:
{
if (lean_obj_tag(v_x_110_) == 1)
{
lean_object* v_pre_111_; 
v_pre_111_ = lean_ctor_get(v_x_110_, 0);
switch(lean_obj_tag(v_pre_111_))
{
case 1:
{
lean_object* v_pre_112_; 
v_pre_112_ = lean_ctor_get(v_pre_111_, 0);
switch(lean_obj_tag(v_pre_112_))
{
case 0:
{
lean_object* v_str_113_; lean_object* v_str_114_; lean_object* v___x_115_; uint8_t v___x_116_; 
v_str_113_ = lean_ctor_get(v_x_110_, 1);
v_str_114_ = lean_ctor_get(v_pre_111_, 1);
v___x_115_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__0));
v___x_116_ = lean_string_dec_eq(v_str_114_, v___x_115_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__1));
v___x_118_ = lean_string_dec_eq(v_str_114_, v___x_117_);
if (v___x_118_ == 0)
{
return v___y_108_;
}
else
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__2));
v___x_120_ = lean_string_dec_eq(v_str_113_, v___x_119_);
if (v___x_120_ == 0)
{
return v___y_108_;
}
else
{
return v_suppressElabErrors_109_;
}
}
}
else
{
lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_121_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__3));
v___x_122_ = lean_string_dec_eq(v_str_113_, v___x_121_);
if (v___x_122_ == 0)
{
return v___y_108_;
}
else
{
return v_suppressElabErrors_109_;
}
}
}
case 1:
{
lean_object* v_pre_123_; 
v_pre_123_ = lean_ctor_get(v_pre_112_, 0);
if (lean_obj_tag(v_pre_123_) == 0)
{
lean_object* v_str_124_; lean_object* v_str_125_; lean_object* v_str_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v_str_124_ = lean_ctor_get(v_x_110_, 1);
v_str_125_ = lean_ctor_get(v_pre_111_, 1);
v_str_126_ = lean_ctor_get(v_pre_112_, 1);
v___x_127_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__4));
v___x_128_ = lean_string_dec_eq(v_str_126_, v___x_127_);
if (v___x_128_ == 0)
{
return v___y_108_;
}
else
{
lean_object* v___x_129_; uint8_t v___x_130_; 
v___x_129_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__5));
v___x_130_ = lean_string_dec_eq(v_str_125_, v___x_129_);
if (v___x_130_ == 0)
{
return v___y_108_;
}
else
{
lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_131_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__6));
v___x_132_ = lean_string_dec_eq(v_str_124_, v___x_131_);
if (v___x_132_ == 0)
{
return v___y_108_;
}
else
{
return v_suppressElabErrors_109_;
}
}
}
}
else
{
return v___y_108_;
}
}
default: 
{
return v___y_108_;
}
}
}
case 0:
{
lean_object* v_str_133_; lean_object* v___x_134_; uint8_t v___x_135_; 
v_str_133_ = lean_ctor_get(v_x_110_, 1);
v___x_134_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___closed__7));
v___x_135_ = lean_string_dec_eq(v_str_133_, v___x_134_);
if (v___x_135_ == 0)
{
return v___y_108_;
}
else
{
return v_suppressElabErrors_109_;
}
}
default: 
{
return v___y_108_;
}
}
}
else
{
return v___y_108_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v___y_136_, lean_object* v_suppressElabErrors_137_, lean_object* v_x_138_){
_start:
{
uint8_t v___y_8595__boxed_139_; uint8_t v_suppressElabErrors_boxed_140_; uint8_t v_res_141_; lean_object* v_r_142_; 
v___y_8595__boxed_139_ = lean_unbox(v___y_136_);
v_suppressElabErrors_boxed_140_ = lean_unbox(v_suppressElabErrors_137_);
v_res_141_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0(v___y_8595__boxed_139_, v_suppressElabErrors_boxed_140_, v_x_138_);
lean_dec(v_x_138_);
v_r_142_ = lean_box(v_res_141_);
return v_r_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg(lean_object* v_ref_144_, lean_object* v_msgData_145_, uint8_t v_severity_146_, uint8_t v_isSilent_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v___y_154_; lean_object* v___y_155_; lean_object* v___y_156_; lean_object* v___y_157_; uint8_t v___y_158_; uint8_t v___y_159_; lean_object* v___y_160_; lean_object* v___y_161_; lean_object* v___y_162_; lean_object* v___y_190_; uint8_t v___y_191_; lean_object* v___y_192_; lean_object* v___y_193_; uint8_t v___y_194_; uint8_t v___y_195_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_215_; uint8_t v___y_216_; lean_object* v___y_217_; uint8_t v___y_218_; uint8_t v___y_219_; lean_object* v___y_220_; lean_object* v___y_221_; lean_object* v___y_222_; lean_object* v___y_226_; uint8_t v___y_227_; lean_object* v___y_228_; lean_object* v___y_229_; uint8_t v___y_230_; lean_object* v___y_231_; uint8_t v___y_232_; uint8_t v___x_237_; lean_object* v___y_239_; uint8_t v___y_240_; lean_object* v___y_241_; lean_object* v___y_242_; lean_object* v___y_243_; uint8_t v___y_244_; uint8_t v___y_245_; uint8_t v___y_247_; uint8_t v___x_262_; 
v___x_237_ = 2;
v___x_262_ = l_Lean_instBEqMessageSeverity_beq(v_severity_146_, v___x_237_);
if (v___x_262_ == 0)
{
v___y_247_ = v___x_262_;
goto v___jp_246_;
}
else
{
uint8_t v___x_263_; 
lean_inc_ref(v_msgData_145_);
v___x_263_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_145_);
v___y_247_ = v___x_263_;
goto v___jp_246_;
}
v___jp_153_:
{
lean_object* v___x_163_; lean_object* v_currNamespace_164_; lean_object* v_openDecls_165_; lean_object* v_env_166_; lean_object* v_nextMacroScope_167_; lean_object* v_ngen_168_; lean_object* v_auxDeclNGen_169_; lean_object* v_traceState_170_; lean_object* v_cache_171_; lean_object* v_messages_172_; lean_object* v_infoState_173_; lean_object* v_snapshotTasks_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_188_; 
v___x_163_ = lean_st_ref_take(v___y_162_);
v_currNamespace_164_ = lean_ctor_get(v___y_161_, 6);
v_openDecls_165_ = lean_ctor_get(v___y_161_, 7);
v_env_166_ = lean_ctor_get(v___x_163_, 0);
v_nextMacroScope_167_ = lean_ctor_get(v___x_163_, 1);
v_ngen_168_ = lean_ctor_get(v___x_163_, 2);
v_auxDeclNGen_169_ = lean_ctor_get(v___x_163_, 3);
v_traceState_170_ = lean_ctor_get(v___x_163_, 4);
v_cache_171_ = lean_ctor_get(v___x_163_, 5);
v_messages_172_ = lean_ctor_get(v___x_163_, 6);
v_infoState_173_ = lean_ctor_get(v___x_163_, 7);
v_snapshotTasks_174_ = lean_ctor_get(v___x_163_, 8);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_163_);
if (v_isSharedCheck_188_ == 0)
{
v___x_176_ = v___x_163_;
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_snapshotTasks_174_);
lean_inc(v_infoState_173_);
lean_inc(v_messages_172_);
lean_inc(v_cache_171_);
lean_inc(v_traceState_170_);
lean_inc(v_auxDeclNGen_169_);
lean_inc(v_ngen_168_);
lean_inc(v_nextMacroScope_167_);
lean_inc(v_env_166_);
lean_dec(v___x_163_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_183_; 
lean_inc(v_openDecls_165_);
lean_inc(v_currNamespace_164_);
v___x_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_178_, 0, v_currNamespace_164_);
lean_ctor_set(v___x_178_, 1, v_openDecls_165_);
v___x_179_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
lean_ctor_set(v___x_179_, 1, v___y_157_);
lean_inc_ref(v___y_154_);
lean_inc_ref(v___y_160_);
v___x_180_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_180_, 0, v___y_160_);
lean_ctor_set(v___x_180_, 1, v___y_156_);
lean_ctor_set(v___x_180_, 2, v___y_155_);
lean_ctor_set(v___x_180_, 3, v___y_154_);
lean_ctor_set(v___x_180_, 4, v___x_179_);
lean_ctor_set_uint8(v___x_180_, sizeof(void*)*5, v___y_159_);
lean_ctor_set_uint8(v___x_180_, sizeof(void*)*5 + 1, v___y_158_);
lean_ctor_set_uint8(v___x_180_, sizeof(void*)*5 + 2, v_isSilent_147_);
v___x_181_ = l_Lean_MessageLog_add(v___x_180_, v_messages_172_);
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 6, v___x_181_);
v___x_183_ = v___x_176_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_env_166_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_nextMacroScope_167_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_ngen_168_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_auxDeclNGen_169_);
lean_ctor_set(v_reuseFailAlloc_187_, 4, v_traceState_170_);
lean_ctor_set(v_reuseFailAlloc_187_, 5, v_cache_171_);
lean_ctor_set(v_reuseFailAlloc_187_, 6, v___x_181_);
lean_ctor_set(v_reuseFailAlloc_187_, 7, v_infoState_173_);
lean_ctor_set(v_reuseFailAlloc_187_, 8, v_snapshotTasks_174_);
v___x_183_ = v_reuseFailAlloc_187_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_184_ = lean_st_ref_set(v___y_162_, v___x_183_);
v___x_185_ = lean_box(0);
v___x_186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
return v___x_186_;
}
}
}
v___jp_189_:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_a_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_213_; 
v___x_198_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_145_);
v___x_199_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__4(v___x_198_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
v_a_200_ = lean_ctor_get(v___x_199_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_199_);
if (v_isSharedCheck_213_ == 0)
{
v___x_202_ = v___x_199_;
v_isShared_203_ = v_isSharedCheck_213_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_a_200_);
lean_dec(v___x_199_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_213_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
lean_inc_ref_n(v___y_196_, 2);
v___x_204_ = l_Lean_FileMap_toPosition(v___y_196_, v___y_192_);
lean_dec(v___y_192_);
v___x_205_ = l_Lean_FileMap_toPosition(v___y_196_, v___y_197_);
lean_dec(v___y_197_);
v___x_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
v___x_207_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___closed__0));
if (v___y_191_ == 0)
{
lean_del_object(v___x_202_);
lean_dec_ref(v___y_190_);
v___y_154_ = v___x_207_;
v___y_155_ = v___x_206_;
v___y_156_ = v___x_204_;
v___y_157_ = v_a_200_;
v___y_158_ = v___y_195_;
v___y_159_ = v___y_194_;
v___y_160_ = v___y_193_;
v___y_161_ = v___y_150_;
v___y_162_ = v___y_151_;
goto v___jp_153_;
}
else
{
uint8_t v___x_208_; 
lean_inc(v_a_200_);
v___x_208_ = l_Lean_MessageData_hasTag(v___y_190_, v_a_200_);
if (v___x_208_ == 0)
{
lean_object* v___x_209_; lean_object* v___x_211_; 
lean_dec_ref_known(v___x_206_, 1);
lean_dec_ref(v___x_204_);
lean_dec(v_a_200_);
v___x_209_ = lean_box(0);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 0, v___x_209_);
v___x_211_ = v___x_202_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_209_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
else
{
lean_del_object(v___x_202_);
v___y_154_ = v___x_207_;
v___y_155_ = v___x_206_;
v___y_156_ = v___x_204_;
v___y_157_ = v_a_200_;
v___y_158_ = v___y_195_;
v___y_159_ = v___y_194_;
v___y_160_ = v___y_193_;
v___y_161_ = v___y_150_;
v___y_162_ = v___y_151_;
goto v___jp_153_;
}
}
}
}
v___jp_214_:
{
lean_object* v___x_223_; 
v___x_223_ = l_Lean_Syntax_getTailPos_x3f(v___y_217_, v___y_219_);
lean_dec(v___y_217_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_inc(v___y_222_);
v___y_190_ = v___y_215_;
v___y_191_ = v___y_216_;
v___y_192_ = v___y_222_;
v___y_193_ = v___y_220_;
v___y_194_ = v___y_219_;
v___y_195_ = v___y_218_;
v___y_196_ = v___y_221_;
v___y_197_ = v___y_222_;
goto v___jp_189_;
}
else
{
lean_object* v_val_224_; 
v_val_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc(v_val_224_);
lean_dec_ref_known(v___x_223_, 1);
v___y_190_ = v___y_215_;
v___y_191_ = v___y_216_;
v___y_192_ = v___y_222_;
v___y_193_ = v___y_220_;
v___y_194_ = v___y_219_;
v___y_195_ = v___y_218_;
v___y_196_ = v___y_221_;
v___y_197_ = v_val_224_;
goto v___jp_189_;
}
}
v___jp_225_:
{
lean_object* v_ref_233_; lean_object* v___x_234_; 
v_ref_233_ = l_Lean_replaceRef(v_ref_144_, v___y_228_);
v___x_234_ = l_Lean_Syntax_getPos_x3f(v_ref_233_, v___y_230_);
if (lean_obj_tag(v___x_234_) == 0)
{
lean_object* v___x_235_; 
v___x_235_ = lean_unsigned_to_nat(0u);
v___y_215_ = v___y_226_;
v___y_216_ = v___y_227_;
v___y_217_ = v_ref_233_;
v___y_218_ = v___y_232_;
v___y_219_ = v___y_230_;
v___y_220_ = v___y_229_;
v___y_221_ = v___y_231_;
v___y_222_ = v___x_235_;
goto v___jp_214_;
}
else
{
lean_object* v_val_236_; 
v_val_236_ = lean_ctor_get(v___x_234_, 0);
lean_inc(v_val_236_);
lean_dec_ref_known(v___x_234_, 1);
v___y_215_ = v___y_226_;
v___y_216_ = v___y_227_;
v___y_217_ = v_ref_233_;
v___y_218_ = v___y_232_;
v___y_219_ = v___y_230_;
v___y_220_ = v___y_229_;
v___y_221_ = v___y_231_;
v___y_222_ = v_val_236_;
goto v___jp_214_;
}
}
v___jp_238_:
{
if (v___y_245_ == 0)
{
v___y_226_ = v___y_241_;
v___y_227_ = v___y_240_;
v___y_228_ = v___y_239_;
v___y_229_ = v___y_242_;
v___y_230_ = v___y_244_;
v___y_231_ = v___y_243_;
v___y_232_ = v_severity_146_;
goto v___jp_225_;
}
else
{
v___y_226_ = v___y_241_;
v___y_227_ = v___y_240_;
v___y_228_ = v___y_239_;
v___y_229_ = v___y_242_;
v___y_230_ = v___y_244_;
v___y_231_ = v___y_243_;
v___y_232_ = v___x_237_;
goto v___jp_225_;
}
}
v___jp_246_:
{
if (v___y_247_ == 0)
{
lean_object* v_fileName_248_; lean_object* v_fileMap_249_; lean_object* v_options_250_; lean_object* v_ref_251_; uint8_t v_suppressElabErrors_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___f_255_; uint8_t v___x_256_; uint8_t v___x_257_; 
v_fileName_248_ = lean_ctor_get(v___y_150_, 0);
v_fileMap_249_ = lean_ctor_get(v___y_150_, 1);
v_options_250_ = lean_ctor_get(v___y_150_, 2);
v_ref_251_ = lean_ctor_get(v___y_150_, 5);
v_suppressElabErrors_252_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*14 + 1);
v___x_253_ = lean_box(v___y_247_);
v___x_254_ = lean_box(v_suppressElabErrors_252_);
v___f_255_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_255_, 0, v___x_253_);
lean_closure_set(v___f_255_, 1, v___x_254_);
v___x_256_ = 1;
v___x_257_ = l_Lean_instBEqMessageSeverity_beq(v_severity_146_, v___x_256_);
if (v___x_257_ == 0)
{
v___y_239_ = v_ref_251_;
v___y_240_ = v_suppressElabErrors_252_;
v___y_241_ = v___f_255_;
v___y_242_ = v_fileName_248_;
v___y_243_ = v_fileMap_249_;
v___y_244_ = v___y_247_;
v___y_245_ = v___x_257_;
goto v___jp_238_;
}
else
{
lean_object* v___x_258_; uint8_t v___x_259_; 
v___x_258_ = l_Lean_warningAsError;
v___x_259_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3_spec__5(v_options_250_, v___x_258_);
v___y_239_ = v_ref_251_;
v___y_240_ = v_suppressElabErrors_252_;
v___y_241_ = v___f_255_;
v___y_242_ = v_fileName_248_;
v___y_243_ = v_fileMap_249_;
v___y_244_ = v___y_247_;
v___y_245_ = v___x_259_;
goto v___jp_238_;
}
}
else
{
lean_object* v___x_260_; lean_object* v___x_261_; 
lean_dec_ref(v_msgData_145_);
v___x_260_ = lean_box(0);
v___x_261_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
return v___x_261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg___boxed(lean_object* v_ref_264_, lean_object* v_msgData_265_, lean_object* v_severity_266_, lean_object* v_isSilent_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
uint8_t v_severity_boxed_273_; uint8_t v_isSilent_boxed_274_; lean_object* v_res_275_; 
v_severity_boxed_273_ = lean_unbox(v_severity_266_);
v_isSilent_boxed_274_ = lean_unbox(v_isSilent_267_);
v_res_275_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg(v_ref_264_, v_msgData_265_, v_severity_boxed_273_, v_isSilent_boxed_274_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v_ref_264_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2(lean_object* v_ref_276_, lean_object* v_msgData_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_){
_start:
{
uint8_t v___x_287_; uint8_t v___x_288_; lean_object* v___x_289_; 
v___x_287_ = 1;
v___x_288_ = 0;
v___x_289_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg(v_ref_276_, v_msgData_277_, v___x_287_, v___x_288_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2___boxed(lean_object* v_ref_290_, lean_object* v_msgData_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2(v_ref_290_, v_msgData_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v_ref_290_);
return v_res_301_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_303_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__0));
v___x_304_ = l_Lean_stringToMessageData(v___x_303_);
return v___x_304_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3(void){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_306_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__2));
v___x_307_ = l_Lean_stringToMessageData(v___x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1(lean_object* v_linterOption_308_, lean_object* v_stx_309_, lean_object* v_msg_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
lean_object* v_name_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_338_; 
v_name_320_ = lean_ctor_get(v_linterOption_308_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v_linterOption_308_);
if (v_isSharedCheck_338_ == 0)
{
lean_object* v_unused_339_; 
v_unused_339_ = lean_ctor_get(v_linterOption_308_, 1);
lean_dec(v_unused_339_);
v___x_322_ = v_linterOption_308_;
v_isShared_323_ = v_isSharedCheck_338_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_name_320_);
lean_dec(v_linterOption_308_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_338_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_327_; 
v___x_324_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__1);
lean_inc(v_name_320_);
v___x_325_ = l_Lean_MessageData_ofName(v_name_320_);
if (v_isShared_323_ == 0)
{
lean_ctor_set_tag(v___x_322_, 7);
lean_ctor_set(v___x_322_, 1, v___x_325_);
lean_ctor_set(v___x_322_, 0, v___x_324_);
v___x_327_ = v___x_322_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_324_);
lean_ctor_set(v_reuseFailAlloc_337_, 1, v___x_325_);
v___x_327_ = v_reuseFailAlloc_337_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v_disable_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_328_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___closed__3);
v___x_329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_327_);
lean_ctor_set(v___x_329_, 1, v___x_328_);
v_disable_330_ = l_Lean_MessageData_note(v___x_329_);
v___x_331_ = l_Lean_Linter_linterMessageTag;
v___x_332_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_332_, 0, v_msg_310_);
lean_ctor_set(v___x_332_, 1, v_disable_330_);
v___x_333_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_331_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_334_, 0, v_name_320_);
lean_ctor_set(v___x_334_, 1, v___x_333_);
lean_inc(v_stx_309_);
v___x_335_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_335_, 0, v_stx_309_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
v___x_336_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2(v_stx_309_, v___x_335_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
lean_dec(v_stx_309_);
return v___x_336_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1___boxed(lean_object* v_linterOption_340_, lean_object* v_stx_341_, lean_object* v_msg_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1(v_linterOption_340_, v_stx_341_, v_msg_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
return v_res_352_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__5));
v___x_369_ = l_Lean_stringToMessageData(v___x_368_);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_371_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__7));
v___x_372_ = l_Lean_stringToMessageData(v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0(lean_object* v_tk_373_, uint8_t v___x_374_, lean_object* v___x_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = l_Lean_Elab_Tactic_getMainTarget(v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
if (lean_obj_tag(v___x_385_) == 0)
{
lean_object* v_a_386_; lean_object* v___x_387_; 
v_a_386_ = lean_ctor_get(v___x_385_, 0);
lean_inc(v_a_386_);
lean_dec_ref_known(v___x_385_, 1);
v___x_387_ = l_Lean_Meta_isProp(v_a_386_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_415_; 
v_a_388_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_415_ == 0)
{
v___x_390_ = v___x_387_;
v_isShared_391_ = v_isSharedCheck_415_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_415_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
uint8_t v___x_392_; 
v___x_392_ = lean_unbox(v_a_388_);
lean_dec(v_a_388_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; lean_object* v___x_395_; 
lean_dec_ref(v___y_382_);
lean_dec_ref(v___x_375_);
lean_dec(v_tk_373_);
v___x_393_ = lean_box(0);
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 0, v___x_393_);
v___x_395_ = v___x_390_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v___x_393_);
v___x_395_ = v_reuseFailAlloc_396_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
return v___x_395_;
}
}
else
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
lean_del_object(v___x_390_);
v___x_397_ = lean_box(0);
v___x_398_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__4));
v___x_399_ = l_Lean_Meta_Hint_mkSuggestionsMessage(v___x_398_, v_tk_373_, v___x_397_, v___x_374_, v___y_382_, v___y_383_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v_ref_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
lean_inc(v_a_400_);
lean_dec_ref_known(v___x_399_, 1);
v_ref_401_ = lean_ctor_get(v___y_382_, 5);
lean_inc(v_ref_401_);
v___x_402_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6);
v___x_403_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
lean_ctor_set(v___x_403_, 1, v_a_400_);
v___x_404_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8, &lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__8);
v___x_405_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_403_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1(v___x_375_, v_ref_401_, v___x_405_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec_ref(v___y_382_);
return v___x_406_;
}
else
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_414_; 
lean_dec_ref(v___y_382_);
lean_dec_ref(v___x_375_);
v_a_407_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_414_ == 0)
{
v___x_409_ = v___x_399_;
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_399_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_412_; 
if (v_isShared_410_ == 0)
{
v___x_412_ = v___x_409_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v_a_407_);
v___x_412_ = v_reuseFailAlloc_413_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
return v___x_412_;
}
}
}
}
}
}
else
{
lean_object* v_a_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_423_; 
lean_dec_ref(v___y_382_);
lean_dec_ref(v___x_375_);
lean_dec(v_tk_373_);
v_a_416_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_423_ == 0)
{
v___x_418_ = v___x_387_;
v_isShared_419_ = v_isSharedCheck_423_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_a_416_);
lean_dec(v___x_387_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_423_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v___x_421_; 
if (v_isShared_419_ == 0)
{
v___x_421_ = v___x_418_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v_a_416_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
}
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
lean_dec_ref(v___y_382_);
lean_dec_ref(v___x_375_);
lean_dec(v_tk_373_);
v_a_424_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_385_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_385_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___boxed(lean_object* v_tk_432_, lean_object* v___x_433_, lean_object* v___x_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
uint8_t v___x_9038__boxed_444_; lean_object* v_res_445_; 
v___x_9038__boxed_444_ = lean_unbox(v___x_433_);
v_res_445_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0(v_tk_432_, v___x_9038__boxed_444_, v___x_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___y_442_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
lean_dec(v___y_436_);
lean_dec_ref(v___y_435_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg(lean_object* v_o_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___x_449_; lean_object* v_env_450_; lean_object* v___x_451_; lean_object* v_toEnvExtension_452_; lean_object* v_asyncMode_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v_merged_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_465_; 
v___x_449_ = lean_st_ref_get(v___y_447_);
v_env_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc_ref(v_env_450_);
lean_dec(v___x_449_);
v___x_451_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_452_ = lean_ctor_get(v___x_451_, 0);
v_asyncMode_453_ = lean_ctor_get(v_toEnvExtension_452_, 2);
v___x_454_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_455_ = lean_box(0);
v___x_456_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_454_, v___x_451_, v_env_450_, v_asyncMode_453_, v___x_455_);
v_merged_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_465_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_465_ == 0)
{
lean_object* v_unused_466_; 
v_unused_466_ = lean_ctor_get(v___x_456_, 1);
lean_dec(v_unused_466_);
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_465_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_merged_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_465_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 1, v_merged_457_);
lean_ctor_set(v___x_459_, 0, v_o_446_);
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v_o_446_);
lean_ctor_set(v_reuseFailAlloc_464_, 1, v_merged_457_);
v___x_462_ = v_reuseFailAlloc_464_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
lean_object* v___x_463_; 
v___x_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_463_, 0, v___x_462_);
return v___x_463_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg___boxed(lean_object* v_o_467_, lean_object* v___y_468_, lean_object* v___y_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg(v_o_467_, v___y_468_);
lean_dec(v___y_468_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0(lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
lean_object* v_options_480_; lean_object* v___x_481_; 
v_options_480_ = lean_ctor_get(v___y_477_, 2);
lean_inc_ref(v_options_480_);
v___x_481_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg(v_options_480_, v___y_478_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0___boxed(lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0(v___y_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_);
lean_dec(v___y_489_);
lean_dec_ref(v___y_488_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec(v___y_485_);
lean_dec_ref(v___y_484_);
lean_dec(v___y_483_);
lean_dec_ref(v___y_482_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI(lean_object* v_tk_501_, lean_object* v_c_502_, lean_object* v_d_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_, lean_object* v_a_510_, lean_object* v_a_511_){
_start:
{
lean_object* v_ref_513_; uint8_t v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
v_ref_513_ = lean_ctor_get(v_a_510_, 5);
v___x_514_ = 0;
v___x_515_ = l_Lean_SourceInfo_fromRef(v_ref_513_, v___x_514_);
v___x_516_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__3));
v___x_517_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___closed__4));
lean_inc(v___x_515_);
v___x_518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_518_, 0, v___x_515_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
v___x_519_ = l_Lean_Syntax_node3(v___x_515_, v___x_516_, v___x_518_, v_c_502_, v_d_503_);
v___x_520_ = l_Lean_Elab_Tactic_evalTactic(v___x_519_, v_a_504_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_, v_a_511_);
if (lean_obj_tag(v___x_520_) == 0)
{
lean_object* v___x_521_; lean_object* v_a_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_535_; 
lean_dec_ref_known(v___x_520_, 1);
v___x_521_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0(v_a_504_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_, v_a_511_);
v_a_522_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_535_ == 0)
{
v___x_524_ = v___x_521_;
v_isShared_525_ = v_isSharedCheck_535_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_a_522_);
lean_dec(v___x_521_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_535_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v___x_526_; uint8_t v___x_527_; 
v___x_526_ = lp_mathlib_Mathlib_Linter_HaveILetI_linter_style_haveILetI;
v___x_527_ = l_Lean_Linter_getLinterValue(v___x_526_, v_a_522_);
lean_dec(v_a_522_);
if (v___x_527_ == 0)
{
lean_object* v___x_528_; lean_object* v___x_530_; 
lean_dec(v_tk_501_);
v___x_528_ = lean_box(0);
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 0, v___x_528_);
v___x_530_ = v___x_524_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v___x_528_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
return v___x_530_;
}
}
else
{
lean_object* v___x_532_; lean_object* v___f_533_; lean_object* v___x_534_; 
lean_del_object(v___x_524_);
v___x_532_ = lean_box(v___x_514_);
v___f_533_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___boxed), 12, 3);
lean_closure_set(v___f_533_, 0, v_tk_501_);
lean_closure_set(v___f_533_, 1, v___x_532_);
lean_closure_set(v___f_533_, 2, v___x_526_);
v___x_534_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_533_, v_a_504_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_, v_a_511_);
return v___x_534_;
}
}
}
else
{
lean_dec(v_tk_501_);
return v___x_520_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___boxed(lean_object* v_tk_536_, lean_object* v_c_537_, lean_object* v_d_538_, lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v_a_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_, lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI(v_tk_536_, v_c_537_, v_d_538_, v_a_539_, v_a_540_, v_a_541_, v_a_542_, v_a_543_, v_a_544_, v_a_545_, v_a_546_);
lean_dec(v_a_546_);
lean_dec_ref(v_a_545_);
lean_dec(v_a_544_);
lean_dec_ref(v_a_543_);
lean_dec(v_a_542_);
lean_dec_ref(v_a_541_);
lean_dec(v_a_540_);
lean_dec_ref(v_a_539_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0(lean_object* v_o_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___redArg(v_o_549_, v___y_557_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0___boxed(lean_object* v_o_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0_spec__0(v_o_560_, v___y_561_, v___y_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec(v___y_564_);
lean_dec_ref(v___y_563_);
lean_dec(v___y_562_);
lean_dec_ref(v___y_561_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3(lean_object* v_ref_571_, lean_object* v_msgData_572_, uint8_t v_severity_573_, uint8_t v_isSilent_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___redArg(v_ref_571_, v_msgData_572_, v_severity_573_, v_isSilent_574_, v___y_579_, v___y_580_, v___y_581_, v___y_582_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3___boxed(lean_object* v_ref_585_, lean_object* v_msgData_586_, lean_object* v_severity_587_, lean_object* v_isSilent_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
uint8_t v_severity_boxed_598_; uint8_t v_isSilent_boxed_599_; lean_object* v_res_600_; 
v_severity_boxed_598_ = lean_unbox(v_severity_587_);
v_isSilent_boxed_599_ = lean_unbox(v_isSilent_588_);
v_res_600_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1_spec__2_spec__3(v_ref_585_, v_msgData_586_, v_severity_boxed_598_, v_isSilent_boxed_599_, v___y_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
lean_dec(v___y_594_);
lean_dec_ref(v___y_593_);
lean_dec(v___y_592_);
lean_dec_ref(v___y_591_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v_ref_585_);
return v_res_600_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = lean_box(0);
v___x_636_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_637_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_637_, 0, v___x_636_);
lean_ctor_set(v___x_637_, 1, v___x_635_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg(){
_start:
{
lean_object* v___x_639_; lean_object* v___x_640_; 
v___x_639_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___closed__0);
v___x_640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_640_, 0, v___x_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg___boxed(lean_object* v___y_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg();
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0(lean_object* v_00_u03b1_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_){
_start:
{
lean_object* v___x_653_; 
v___x_653_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg();
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___boxed(lean_object* v_00_u03b1_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0(v_00_u03b1_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
lean_dec(v___y_662_);
lean_dec_ref(v___y_661_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1(lean_object* v_x_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_){
_start:
{
lean_object* v___x_675_; uint8_t v___x_676_; 
v___x_675_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_HaveILetI_tacticHaveI_____00__closed__0));
lean_inc(v_x_665_);
v___x_676_ = l_Lean_Syntax_isOfKind(v_x_665_, v___x_675_);
if (v___x_676_ == 0)
{
lean_object* v___x_677_; 
lean_dec(v_x_665_);
v___x_677_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg();
return v___x_677_;
}
else
{
lean_object* v___x_678_; lean_object* v_tk_679_; lean_object* v___x_680_; lean_object* v_c_681_; lean_object* v___x_682_; lean_object* v_d_683_; lean_object* v___x_684_; 
v___x_678_ = lean_unsigned_to_nat(0u);
v_tk_679_ = l_Lean_Syntax_getArg(v_x_665_, v___x_678_);
v___x_680_ = lean_unsigned_to_nat(1u);
v_c_681_ = l_Lean_Syntax_getArg(v_x_665_, v___x_680_);
v___x_682_ = lean_unsigned_to_nat(2u);
v_d_683_ = l_Lean_Syntax_getArg(v_x_665_, v___x_682_);
lean_dec(v_x_665_);
v___x_684_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI(v_tk_679_, v_c_681_, v_d_683_, v_a_666_, v_a_667_, v_a_668_, v_a_669_, v_a_670_, v_a_671_, v_a_672_, v_a_673_);
return v___x_684_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1___boxed(lean_object* v_x_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1(v_x_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_);
lean_dec(v_a_693_);
lean_dec_ref(v_a_692_);
lean_dec(v_a_691_);
lean_dec_ref(v_a_690_);
lean_dec(v_a_689_);
lean_dec_ref(v_a_688_);
lean_dec(v_a_687_);
lean_dec_ref(v_a_686_);
return v_res_695_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_711_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__5));
v___x_712_ = l_Lean_stringToMessageData(v___x_711_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0(lean_object* v_tk_713_, uint8_t v___x_714_, lean_object* v___x_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_){
_start:
{
lean_object* v___x_725_; 
v___x_725_ = l_Lean_Elab_Tactic_getMainTarget(v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_725_) == 0)
{
lean_object* v_a_726_; lean_object* v___x_727_; 
v_a_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_a_726_);
lean_dec_ref_known(v___x_725_, 1);
v___x_727_ = l_Lean_Meta_isProp(v_a_726_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_object* v_a_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_755_; 
v_a_728_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_755_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_755_ == 0)
{
v___x_730_ = v___x_727_;
v_isShared_731_ = v_isSharedCheck_755_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_a_728_);
lean_dec(v___x_727_);
v___x_730_ = lean_box(0);
v_isShared_731_ = v_isSharedCheck_755_;
goto v_resetjp_729_;
}
v_resetjp_729_:
{
uint8_t v___x_732_; 
v___x_732_ = lean_unbox(v_a_728_);
lean_dec(v_a_728_);
if (v___x_732_ == 0)
{
lean_object* v___x_733_; lean_object* v___x_735_; 
lean_dec_ref(v___y_722_);
lean_dec_ref(v___x_715_);
lean_dec(v_tk_713_);
v___x_733_ = lean_box(0);
if (v_isShared_731_ == 0)
{
lean_ctor_set(v___x_730_, 0, v___x_733_);
v___x_735_ = v___x_730_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v___x_733_);
v___x_735_ = v_reuseFailAlloc_736_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
return v___x_735_;
}
}
else
{
lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
lean_del_object(v___x_730_);
v___x_737_ = lean_box(0);
v___x_738_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__4));
v___x_739_ = l_Lean_Meta_Hint_mkSuggestionsMessage(v___x_738_, v_tk_713_, v___x_737_, v___x_714_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; lean_object* v_ref_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_a_740_);
lean_dec_ref_known(v___x_739_, 1);
v_ref_741_ = lean_ctor_get(v___y_722_, 5);
lean_inc(v_ref_741_);
v___x_742_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI___lam__0___closed__6);
v___x_743_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_742_);
lean_ctor_set(v___x_743_, 1, v_a_740_);
v___x_744_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___closed__6);
v___x_745_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_743_);
lean_ctor_set(v___x_745_, 1, v___x_744_);
v___x_746_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__1(v___x_715_, v_ref_741_, v___x_745_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
lean_dec_ref(v___y_722_);
return v___x_746_;
}
else
{
lean_object* v_a_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_754_; 
lean_dec_ref(v___y_722_);
lean_dec_ref(v___x_715_);
v_a_747_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_754_ == 0)
{
v___x_749_ = v___x_739_;
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_a_747_);
lean_dec(v___x_739_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_752_; 
if (v_isShared_750_ == 0)
{
v___x_752_ = v___x_749_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v_a_747_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
}
}
}
else
{
lean_object* v_a_756_; lean_object* v___x_758_; uint8_t v_isShared_759_; uint8_t v_isSharedCheck_763_; 
lean_dec_ref(v___y_722_);
lean_dec_ref(v___x_715_);
lean_dec(v_tk_713_);
v_a_756_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_763_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_763_ == 0)
{
v___x_758_ = v___x_727_;
v_isShared_759_ = v_isSharedCheck_763_;
goto v_resetjp_757_;
}
else
{
lean_inc(v_a_756_);
lean_dec(v___x_727_);
v___x_758_ = lean_box(0);
v_isShared_759_ = v_isSharedCheck_763_;
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
lean_object* v_reuseFailAlloc_762_; 
v_reuseFailAlloc_762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_762_, 0, v_a_756_);
v___x_761_ = v_reuseFailAlloc_762_;
goto v_reusejp_760_;
}
v_reusejp_760_:
{
return v___x_761_;
}
}
}
}
else
{
lean_object* v_a_764_; lean_object* v___x_766_; uint8_t v_isShared_767_; uint8_t v_isSharedCheck_771_; 
lean_dec_ref(v___y_722_);
lean_dec_ref(v___x_715_);
lean_dec(v_tk_713_);
v_a_764_ = lean_ctor_get(v___x_725_, 0);
v_isSharedCheck_771_ = !lean_is_exclusive(v___x_725_);
if (v_isSharedCheck_771_ == 0)
{
v___x_766_ = v___x_725_;
v_isShared_767_ = v_isSharedCheck_771_;
goto v_resetjp_765_;
}
else
{
lean_inc(v_a_764_);
lean_dec(v___x_725_);
v___x_766_ = lean_box(0);
v_isShared_767_ = v_isSharedCheck_771_;
goto v_resetjp_765_;
}
v_resetjp_765_:
{
lean_object* v___x_769_; 
if (v_isShared_767_ == 0)
{
v___x_769_ = v___x_766_;
goto v_reusejp_768_;
}
else
{
lean_object* v_reuseFailAlloc_770_; 
v_reuseFailAlloc_770_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_770_, 0, v_a_764_);
v___x_769_ = v_reuseFailAlloc_770_;
goto v_reusejp_768_;
}
v_reusejp_768_:
{
return v___x_769_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___boxed(lean_object* v_tk_772_, lean_object* v___x_773_, lean_object* v___x_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_){
_start:
{
uint8_t v___x_4149__boxed_784_; lean_object* v_res_785_; 
v___x_4149__boxed_784_ = lean_unbox(v___x_773_);
v_res_785_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0(v_tk_772_, v___x_4149__boxed_784_, v___x_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
lean_dec(v___y_782_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI(lean_object* v_tk_793_, lean_object* v_c_794_, lean_object* v_d_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_){
_start:
{
lean_object* v_ref_805_; uint8_t v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; 
v_ref_805_ = lean_ctor_get(v_a_802_, 5);
v___x_806_ = 0;
v___x_807_ = l_Lean_SourceInfo_fromRef(v_ref_805_, v___x_806_);
v___x_808_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__1));
v___x_809_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___closed__2));
lean_inc(v___x_807_);
v___x_810_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_810_, 0, v___x_807_);
lean_ctor_set(v___x_810_, 1, v___x_809_);
v___x_811_ = l_Lean_Syntax_node3(v___x_807_, v___x_808_, v___x_810_, v_c_794_, v_d_795_);
v___x_812_ = l_Lean_Elab_Tactic_evalTactic(v___x_811_, v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_, v_a_803_);
if (lean_obj_tag(v___x_812_) == 0)
{
lean_object* v___x_813_; lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_827_; 
lean_dec_ref_known(v___x_812_, 1);
v___x_813_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runHaveI_spec__0(v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_, v_a_803_);
v_a_814_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_827_ == 0)
{
v___x_816_ = v___x_813_;
v_isShared_817_ = v_isSharedCheck_827_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_813_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_827_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_818_; uint8_t v___x_819_; 
v___x_818_ = lp_mathlib_Mathlib_Linter_HaveILetI_linter_style_haveILetI;
v___x_819_ = l_Lean_Linter_getLinterValue(v___x_818_, v_a_814_);
lean_dec(v_a_814_);
if (v___x_819_ == 0)
{
lean_object* v___x_820_; lean_object* v___x_822_; 
lean_dec(v_tk_793_);
v___x_820_ = lean_box(0);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v___x_820_);
v___x_822_ = v___x_816_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_820_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
else
{
lean_object* v___x_824_; lean_object* v___f_825_; lean_object* v___x_826_; 
lean_del_object(v___x_816_);
v___x_824_ = lean_box(v___x_806_);
v___f_825_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___lam__0___boxed), 12, 3);
lean_closure_set(v___f_825_, 0, v_tk_793_);
lean_closure_set(v___f_825_, 1, v___x_824_);
lean_closure_set(v___f_825_, 2, v___x_818_);
v___x_826_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_825_, v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_, v_a_803_);
return v___x_826_;
}
}
}
else
{
lean_dec(v_tk_793_);
return v___x_812_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI___boxed(lean_object* v_tk_828_, lean_object* v_c_829_, lean_object* v_d_830_, lean_object* v_a_831_, lean_object* v_a_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI(v_tk_828_, v_c_829_, v_d_830_, v_a_831_, v_a_832_, v_a_833_, v_a_834_, v_a_835_, v_a_836_, v_a_837_, v_a_838_);
lean_dec(v_a_838_);
lean_dec_ref(v_a_837_);
lean_dec(v_a_836_);
lean_dec_ref(v_a_835_);
lean_dec(v_a_834_);
lean_dec_ref(v_a_833_);
lean_dec(v_a_832_);
lean_dec_ref(v_a_831_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticLetI______1(lean_object* v_x_862_, lean_object* v_a_863_, lean_object* v_a_864_, lean_object* v_a_865_, lean_object* v_a_866_, lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_a_869_, lean_object* v_a_870_){
_start:
{
lean_object* v___x_872_; uint8_t v___x_873_; 
v___x_872_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_HaveILetI_tacticLetI_____00__closed__0));
lean_inc(v_x_862_);
v___x_873_ = l_Lean_Syntax_isOfKind(v_x_862_, v___x_872_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; 
lean_dec(v_x_862_);
v___x_874_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticHaveI______1_spec__0___redArg();
return v___x_874_;
}
else
{
lean_object* v___x_875_; lean_object* v_tk_876_; lean_object* v___x_877_; lean_object* v_c_878_; lean_object* v___x_879_; lean_object* v_d_880_; lean_object* v___x_881_; 
v___x_875_ = lean_unsigned_to_nat(0u);
v_tk_876_ = l_Lean_Syntax_getArg(v_x_862_, v___x_875_);
v___x_877_ = lean_unsigned_to_nat(1u);
v_c_878_ = l_Lean_Syntax_getArg(v_x_862_, v___x_877_);
v___x_879_ = lean_unsigned_to_nat(2u);
v_d_880_ = l_Lean_Syntax_getArg(v_x_862_, v___x_879_);
lean_dec(v_x_862_);
v___x_881_ = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_runLetI(v_tk_876_, v_c_878_, v_d_880_, v_a_863_, v_a_864_, v_a_865_, v_a_866_, v_a_867_, v_a_868_, v_a_869_, v_a_870_);
return v___x_881_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticLetI______1___boxed(lean_object* v_x_882_, lean_object* v_a_883_, lean_object* v_a_884_, lean_object* v_a_885_, lean_object* v_a_886_, lean_object* v_a_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_){
_start:
{
lean_object* v_res_892_; 
v_res_892_ = lp_mathlib_Mathlib_Linter_HaveILetI___aux__Mathlib__Tactic__Linter__HaveILetI______elabRules__Mathlib__Linter__HaveILetI__tacticLetI______1(v_x_882_, v_a_883_, v_a_884_, v_a_885_, v_a_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_);
lean_dec(v_a_890_);
lean_dec_ref(v_a_889_);
lean_dec(v_a_888_);
lean_dec_ref(v_a_887_);
lean_dec(v_a_886_);
lean_dec_ref(v_a_885_);
lean_dec(v_a_884_);
lean_dec_ref(v_a_883_);
return v_res_892_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_TryThis(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(uint8_t builtin) {
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
res = runtime_initialize_Lean_Meta_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Hint(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_HaveILetI_0__Mathlib_Linter_HaveILetI_initFn_00___x40_Mathlib_Tactic_Linter_HaveILetI_2599143525____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_HaveILetI_linter_style_haveILetI = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_HaveILetI_linter_style_haveILetI);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Hint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Meta_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(builtin);
}
#ifdef __cplusplus
}
#endif
