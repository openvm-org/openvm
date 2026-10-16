// Lean compiler output
// Module: Mathlib.Tactic.Linter.OldObtain
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Message
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "obtain"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__3_value),LEAN_SCALAR_PTR_LITERAL(11, 177, 143, 165, 56, 37, 104, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "oldObtain"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(218, 45, 52, 10, 248, 233, 120, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "enable the `oldObtain` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 70, 102, 253, 204, 74, 141)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 0, 222, 115, 76, 186, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(73, 149, 98, 192, 89, 83, 26, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_linter_oldObtain;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Please remove stream-of-consciousness `obtain` syntax"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__2_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "OldObtain"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__7_value),LEAN_SCALAR_PTR_LITERAL(84, 61, 21, 205, 1, 216, 51, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(85, 125, 196, 50, 176, 203, 2, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(176, 196, 135, 124, 179, 53, 188, 106)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(182, 82, 42, 214, 244, 52, 22, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(54, 148, 113, 19, 255, 94, 228, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "oldObtainLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__13_value),LEAN_SCALAR_PTR_LITERAL(108, 70, 81, 8, 15, 77, 58, 106)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof(lean_object* v_x_10_){
_start:
{
lean_object* v___x_11_; uint8_t v___x_12_; 
v___x_11_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___closed__4));
lean_inc(v_x_10_);
v___x_12_ = l_Lean_Syntax_isOfKind(v_x_10_, v___x_11_);
if (v___x_12_ == 0)
{
lean_dec(v_x_10_);
return v___x_12_;
}
else
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
v___x_13_ = lean_unsigned_to_nat(0u);
v___x_14_ = lean_unsigned_to_nat(1u);
v___x_15_ = l_Lean_Syntax_getArg(v_x_10_, v___x_14_);
lean_inc(v___x_15_);
v___x_16_ = l_Lean_Syntax_matchesNull(v___x_15_, v___x_13_);
if (v___x_16_ == 0)
{
uint8_t v___x_17_; 
v___x_17_ = l_Lean_Syntax_matchesNull(v___x_15_, v___x_14_);
if (v___x_17_ == 0)
{
lean_dec(v_x_10_);
return v___x_17_;
}
else
{
lean_object* v___x_18_; lean_object* v___x_19_; uint8_t v___x_20_; 
v___x_18_ = lean_unsigned_to_nat(2u);
v___x_19_ = l_Lean_Syntax_getArg(v_x_10_, v___x_18_);
v___x_20_ = l_Lean_Syntax_matchesNull(v___x_19_, v___x_18_);
if (v___x_20_ == 0)
{
lean_dec(v_x_10_);
return v___x_20_;
}
else
{
lean_object* v___x_21_; lean_object* v___x_22_; uint8_t v___x_23_; 
v___x_21_ = lean_unsigned_to_nat(3u);
v___x_22_ = l_Lean_Syntax_getArg(v_x_10_, v___x_21_);
lean_dec(v_x_10_);
v___x_23_ = l_Lean_Syntax_matchesNull(v___x_22_, v___x_13_);
if (v___x_23_ == 0)
{
return v___x_23_;
}
else
{
return v___x_12_;
}
}
}
}
else
{
lean_object* v___x_24_; lean_object* v___x_25_; uint8_t v___x_26_; 
lean_dec(v___x_15_);
v___x_24_ = lean_unsigned_to_nat(2u);
v___x_25_ = l_Lean_Syntax_getArg(v_x_10_, v___x_24_);
v___x_26_ = l_Lean_Syntax_matchesNull(v___x_25_, v___x_24_);
if (v___x_26_ == 0)
{
lean_dec(v_x_10_);
return v___x_26_;
}
else
{
lean_object* v___x_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
v___x_27_ = lean_unsigned_to_nat(3u);
v___x_28_ = l_Lean_Syntax_getArg(v_x_10_, v___x_27_);
lean_dec(v_x_10_);
v___x_29_ = l_Lean_Syntax_matchesNull(v___x_28_, v___x_13_);
if (v___x_29_ == 0)
{
return v___x_29_;
}
else
{
return v___x_12_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof___boxed(lean_object* v_x_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_isObtainWithoutProof(v_x_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0(lean_object* v_name_33_, lean_object* v_decl_34_, lean_object* v_ref_35_){
_start:
{
lean_object* v_defValue_37_; lean_object* v_descr_38_; lean_object* v_deprecation_x3f_39_; lean_object* v___x_40_; uint8_t v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v_defValue_37_ = lean_ctor_get(v_decl_34_, 0);
v_descr_38_ = lean_ctor_get(v_decl_34_, 1);
v_deprecation_x3f_39_ = lean_ctor_get(v_decl_34_, 2);
v___x_40_ = lean_alloc_ctor(1, 0, 1);
v___x_41_ = lean_unbox(v_defValue_37_);
lean_ctor_set_uint8(v___x_40_, 0, v___x_41_);
lean_inc(v_deprecation_x3f_39_);
lean_inc_ref(v_descr_38_);
lean_inc_n(v_name_33_, 2);
v___x_42_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_42_, 0, v_name_33_);
lean_ctor_set(v___x_42_, 1, v_ref_35_);
lean_ctor_set(v___x_42_, 2, v___x_40_);
lean_ctor_set(v___x_42_, 3, v_descr_38_);
lean_ctor_set(v___x_42_, 4, v_deprecation_x3f_39_);
v___x_43_ = lean_register_option(v_name_33_, v___x_42_);
if (lean_obj_tag(v___x_43_) == 0)
{
lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_51_; 
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_43_);
if (v_isSharedCheck_51_ == 0)
{
lean_object* v_unused_52_; 
v_unused_52_ = lean_ctor_get(v___x_43_, 0);
lean_dec(v_unused_52_);
v___x_45_ = v___x_43_;
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
else
{
lean_dec(v___x_43_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_47_; lean_object* v___x_49_; 
lean_inc(v_defValue_37_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v_name_33_);
lean_ctor_set(v___x_47_, 1, v_defValue_37_);
if (v_isShared_46_ == 0)
{
lean_ctor_set(v___x_45_, 0, v___x_47_);
v___x_49_ = v___x_45_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v___x_47_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
else
{
lean_object* v_a_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_60_; 
lean_dec(v_name_33_);
v_a_53_ = lean_ctor_get(v___x_43_, 0);
v_isSharedCheck_60_ = !lean_is_exclusive(v___x_43_);
if (v_isSharedCheck_60_ == 0)
{
v___x_55_ = v___x_43_;
v_isShared_56_ = v_isSharedCheck_60_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_a_53_);
lean_dec(v___x_43_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_60_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_58_; 
if (v_isShared_56_ == 0)
{
v___x_58_ = v___x_55_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v_a_53_);
v___x_58_ = v_reuseFailAlloc_59_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
return v___x_58_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_61_, lean_object* v_decl_62_, lean_object* v_ref_63_, lean_object* v_a_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0(v_name_61_, v_decl_62_, v_ref_63_);
lean_dec_ref(v_decl_62_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_));
v___x_88_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_));
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_));
v___x_90_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4__spec__0(v___x_87_, v___x_88_, v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4____boxed(lean_object* v_a_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_();
return v_res_92_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5(lean_object* v_opts_93_, lean_object* v_opt_94_){
_start:
{
lean_object* v_name_95_; lean_object* v_defValue_96_; lean_object* v_map_97_; lean_object* v___x_98_; 
v_name_95_ = lean_ctor_get(v_opt_94_, 0);
v_defValue_96_ = lean_ctor_get(v_opt_94_, 1);
v_map_97_ = lean_ctor_get(v_opts_93_, 0);
v___x_98_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_97_, v_name_95_);
if (lean_obj_tag(v___x_98_) == 0)
{
uint8_t v___x_99_; 
v___x_99_ = lean_unbox(v_defValue_96_);
return v___x_99_;
}
else
{
lean_object* v_val_100_; 
v_val_100_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_val_100_);
lean_dec_ref_known(v___x_98_, 1);
if (lean_obj_tag(v_val_100_) == 1)
{
uint8_t v_v_101_; 
v_v_101_ = lean_ctor_get_uint8(v_val_100_, 0);
lean_dec_ref_known(v_val_100_, 0);
return v_v_101_;
}
else
{
uint8_t v___x_102_; 
lean_dec(v_val_100_);
v___x_102_ = lean_unbox(v_defValue_96_);
return v___x_102_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5___boxed(lean_object* v_opts_103_, lean_object* v_opt_104_){
_start:
{
uint8_t v_res_105_; lean_object* v_r_106_; 
v_res_105_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5(v_opts_103_, v_opt_104_);
lean_dec_ref(v_opt_104_);
lean_dec_ref(v_opts_103_);
v_r_106_ = lean_box(v_res_105_);
return v_r_106_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__0);
v___x_109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1);
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
lean_ctor_set(v___x_112_, 2, v___x_111_);
lean_ctor_set(v___x_112_, 3, v___x_111_);
lean_ctor_set(v___x_112_, 4, v___x_110_);
lean_ctor_set(v___x_112_, 5, v___x_110_);
lean_ctor_set(v___x_112_, 6, v___x_110_);
lean_ctor_set(v___x_112_, 7, v___x_110_);
lean_ctor_set(v___x_112_, 8, v___x_110_);
lean_ctor_set(v___x_112_, 9, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lean_unsigned_to_nat(32u);
v___x_114_ = lean_mk_empty_array_with_capacity(v___x_113_);
v___x_115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4(void){
_start:
{
size_t v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_116_ = ((size_t)5ULL);
v___x_117_ = lean_unsigned_to_nat(0u);
v___x_118_ = lean_unsigned_to_nat(32u);
v___x_119_ = lean_mk_empty_array_with_capacity(v___x_118_);
v___x_120_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__3);
v___x_121_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v___x_119_);
lean_ctor_set(v___x_121_, 2, v___x_117_);
lean_ctor_set(v___x_121_, 3, v___x_117_);
lean_ctor_set_usize(v___x_121_, 4, v___x_116_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_122_ = lean_box(1);
v___x_123_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__4);
v___x_124_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__1);
v___x_125_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v___x_123_);
lean_ctor_set(v___x_125_, 2, v___x_122_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg(lean_object* v_msgData_126_, lean_object* v___y_127_){
_start:
{
lean_object* v___x_129_; lean_object* v_env_130_; lean_object* v___x_131_; lean_object* v_scopes_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v_opts_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_129_ = lean_st_ref_get(v___y_127_);
v_env_130_ = lean_ctor_get(v___x_129_, 0);
lean_inc_ref(v_env_130_);
lean_dec(v___x_129_);
v___x_131_ = lean_st_ref_get(v___y_127_);
v_scopes_132_ = lean_ctor_get(v___x_131_, 2);
lean_inc(v_scopes_132_);
lean_dec(v___x_131_);
v___x_133_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_134_ = l_List_head_x21___redArg(v___x_133_, v_scopes_132_);
lean_dec(v_scopes_132_);
v_opts_135_ = lean_ctor_get(v___x_134_, 1);
lean_inc_ref(v_opts_135_);
lean_dec(v___x_134_);
v___x_136_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__2);
v___x_137_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___closed__5);
v___x_138_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_138_, 0, v_env_130_);
lean_ctor_set(v___x_138_, 1, v___x_136_);
lean_ctor_set(v___x_138_, 2, v___x_137_);
lean_ctor_set(v___x_138_, 3, v_opts_135_);
v___x_139_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v_msgData_126_);
v___x_140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg(v_msgData_141_, v___y_142_);
lean_dec(v___y_142_);
return v_res_144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0(uint8_t v___y_146_, uint8_t v_suppressElabErrors_147_, lean_object* v_x_148_){
_start:
{
if (lean_obj_tag(v_x_148_) == 1)
{
lean_object* v_pre_149_; 
v_pre_149_ = lean_ctor_get(v_x_148_, 0);
if (lean_obj_tag(v_pre_149_) == 0)
{
lean_object* v_str_150_; lean_object* v___x_151_; uint8_t v___x_152_; 
v_str_150_ = lean_ctor_get(v_x_148_, 1);
v___x_151_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___closed__0));
v___x_152_ = lean_string_dec_eq(v_str_150_, v___x_151_);
if (v___x_152_ == 0)
{
return v___y_146_;
}
else
{
return v_suppressElabErrors_147_;
}
}
else
{
return v___y_146_;
}
}
else
{
return v___y_146_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object* v___y_153_, lean_object* v_suppressElabErrors_154_, lean_object* v_x_155_){
_start:
{
uint8_t v___y_3461__boxed_156_; uint8_t v_suppressElabErrors_boxed_157_; uint8_t v_res_158_; lean_object* v_r_159_; 
v___y_3461__boxed_156_ = lean_unbox(v___y_153_);
v_suppressElabErrors_boxed_157_ = lean_unbox(v_suppressElabErrors_154_);
v_res_158_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0(v___y_3461__boxed_156_, v_suppressElabErrors_boxed_157_, v_x_155_);
lean_dec(v_x_155_);
v_r_159_ = lean_box(v_res_158_);
return v_r_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3(lean_object* v_ref_161_, lean_object* v_msgData_162_, uint8_t v_severity_163_, uint8_t v_isSilent_164_, lean_object* v___y_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___y_169_; lean_object* v___y_170_; uint8_t v___y_171_; uint8_t v___y_172_; lean_object* v___y_173_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; uint8_t v___y_233_; uint8_t v___y_234_; uint8_t v___y_235_; lean_object* v___y_236_; lean_object* v___y_237_; uint8_t v___y_261_; uint8_t v___y_262_; uint8_t v___y_263_; lean_object* v___y_264_; lean_object* v___y_265_; uint8_t v___y_269_; uint8_t v___y_270_; uint8_t v___y_271_; uint8_t v___x_286_; uint8_t v___y_288_; uint8_t v___y_289_; uint8_t v___y_290_; uint8_t v___y_292_; uint8_t v___x_304_; 
v___x_286_ = 2;
v___x_304_ = l_Lean_instBEqMessageSeverity_beq(v_severity_163_, v___x_286_);
if (v___x_304_ == 0)
{
v___y_292_ = v___x_304_;
goto v___jp_291_;
}
else
{
uint8_t v___x_305_; 
lean_inc_ref(v_msgData_162_);
v___x_305_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_162_);
v___y_292_ = v___x_305_;
goto v___jp_291_;
}
v___jp_168_:
{
lean_object* v___x_177_; 
v___x_177_ = l_Lean_Elab_Command_getScope___redArg(v___y_176_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v_a_178_; lean_object* v___x_179_; 
v_a_178_ = lean_ctor_get(v___x_177_, 0);
lean_inc(v_a_178_);
lean_dec_ref_known(v___x_177_, 1);
v___x_179_ = l_Lean_Elab_Command_getScope___redArg(v___y_176_);
if (lean_obj_tag(v___x_179_) == 0)
{
lean_object* v_a_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_215_; 
v_a_180_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_215_ == 0)
{
v___x_182_ = v___x_179_;
v_isShared_183_ = v_isSharedCheck_215_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_a_180_);
lean_dec(v___x_179_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_215_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_184_; lean_object* v_currNamespace_185_; lean_object* v_openDecls_186_; lean_object* v_env_187_; lean_object* v_messages_188_; lean_object* v_scopes_189_; lean_object* v_usedQuotCtxts_190_; lean_object* v_nextMacroScope_191_; lean_object* v_maxRecDepth_192_; lean_object* v_ngen_193_; lean_object* v_auxDeclNGen_194_; lean_object* v_infoState_195_; lean_object* v_traceState_196_; lean_object* v_snapshotTasks_197_; lean_object* v_prevLinterStates_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_214_; 
v___x_184_ = lean_st_ref_take(v___y_176_);
v_currNamespace_185_ = lean_ctor_get(v_a_178_, 2);
lean_inc(v_currNamespace_185_);
lean_dec(v_a_178_);
v_openDecls_186_ = lean_ctor_get(v_a_180_, 3);
lean_inc(v_openDecls_186_);
lean_dec(v_a_180_);
v_env_187_ = lean_ctor_get(v___x_184_, 0);
v_messages_188_ = lean_ctor_get(v___x_184_, 1);
v_scopes_189_ = lean_ctor_get(v___x_184_, 2);
v_usedQuotCtxts_190_ = lean_ctor_get(v___x_184_, 3);
v_nextMacroScope_191_ = lean_ctor_get(v___x_184_, 4);
v_maxRecDepth_192_ = lean_ctor_get(v___x_184_, 5);
v_ngen_193_ = lean_ctor_get(v___x_184_, 6);
v_auxDeclNGen_194_ = lean_ctor_get(v___x_184_, 7);
v_infoState_195_ = lean_ctor_get(v___x_184_, 8);
v_traceState_196_ = lean_ctor_get(v___x_184_, 9);
v_snapshotTasks_197_ = lean_ctor_get(v___x_184_, 10);
v_prevLinterStates_198_ = lean_ctor_get(v___x_184_, 11);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_214_ == 0)
{
v___x_200_ = v___x_184_;
v_isShared_201_ = v_isSharedCheck_214_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_prevLinterStates_198_);
lean_inc(v_snapshotTasks_197_);
lean_inc(v_traceState_196_);
lean_inc(v_infoState_195_);
lean_inc(v_auxDeclNGen_194_);
lean_inc(v_ngen_193_);
lean_inc(v_maxRecDepth_192_);
lean_inc(v_nextMacroScope_191_);
lean_inc(v_usedQuotCtxts_190_);
lean_inc(v_scopes_189_);
lean_inc(v_messages_188_);
lean_inc(v_env_187_);
lean_dec(v___x_184_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_214_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_207_; 
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v_currNamespace_185_);
lean_ctor_set(v___x_202_, 1, v_openDecls_186_);
v___x_203_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
lean_ctor_set(v___x_203_, 1, v___y_170_);
lean_inc_ref(v___y_169_);
lean_inc_ref(v___y_175_);
v___x_204_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_204_, 0, v___y_175_);
lean_ctor_set(v___x_204_, 1, v___y_173_);
lean_ctor_set(v___x_204_, 2, v___y_174_);
lean_ctor_set(v___x_204_, 3, v___y_169_);
lean_ctor_set(v___x_204_, 4, v___x_203_);
lean_ctor_set_uint8(v___x_204_, sizeof(void*)*5, v___y_172_);
lean_ctor_set_uint8(v___x_204_, sizeof(void*)*5 + 1, v___y_171_);
lean_ctor_set_uint8(v___x_204_, sizeof(void*)*5 + 2, v_isSilent_164_);
v___x_205_ = l_Lean_MessageLog_add(v___x_204_, v_messages_188_);
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 1, v___x_205_);
v___x_207_ = v___x_200_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_env_187_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v___x_205_);
lean_ctor_set(v_reuseFailAlloc_213_, 2, v_scopes_189_);
lean_ctor_set(v_reuseFailAlloc_213_, 3, v_usedQuotCtxts_190_);
lean_ctor_set(v_reuseFailAlloc_213_, 4, v_nextMacroScope_191_);
lean_ctor_set(v_reuseFailAlloc_213_, 5, v_maxRecDepth_192_);
lean_ctor_set(v_reuseFailAlloc_213_, 6, v_ngen_193_);
lean_ctor_set(v_reuseFailAlloc_213_, 7, v_auxDeclNGen_194_);
lean_ctor_set(v_reuseFailAlloc_213_, 8, v_infoState_195_);
lean_ctor_set(v_reuseFailAlloc_213_, 9, v_traceState_196_);
lean_ctor_set(v_reuseFailAlloc_213_, 10, v_snapshotTasks_197_);
lean_ctor_set(v_reuseFailAlloc_213_, 11, v_prevLinterStates_198_);
v___x_207_ = v_reuseFailAlloc_213_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_208_ = lean_st_ref_set(v___y_176_, v___x_207_);
v___x_209_ = lean_box(0);
if (v_isShared_183_ == 0)
{
lean_ctor_set(v___x_182_, 0, v___x_209_);
v___x_211_ = v___x_182_;
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
}
}
}
else
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_223_; 
lean_dec(v_a_178_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec_ref(v___y_170_);
v_a_216_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_223_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_223_ == 0)
{
v___x_218_ = v___x_179_;
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_179_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_221_; 
if (v_isShared_219_ == 0)
{
v___x_221_ = v___x_218_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v_a_216_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
else
{
lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_231_; 
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec_ref(v___y_170_);
v_a_224_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_231_ == 0)
{
v___x_226_ = v___x_177_;
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_177_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_229_; 
if (v_isShared_227_ == 0)
{
v___x_229_ = v___x_226_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_a_224_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
}
v___jp_232_:
{
lean_object* v_fileName_238_; lean_object* v_fileMap_239_; uint8_t v_suppressElabErrors_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_259_; 
v_fileName_238_ = lean_ctor_get(v___y_165_, 0);
v_fileMap_239_ = lean_ctor_get(v___y_165_, 1);
v_suppressElabErrors_240_ = lean_ctor_get_uint8(v___y_165_, sizeof(void*)*10);
v___x_241_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_162_);
v___x_242_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg(v___x_241_, v___y_166_);
v_a_243_ = lean_ctor_get(v___x_242_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_259_ == 0)
{
v___x_245_ = v___x_242_;
v_isShared_246_ = v_isSharedCheck_259_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_242_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_259_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
lean_inc_ref_n(v_fileMap_239_, 2);
v___x_247_ = l_Lean_FileMap_toPosition(v_fileMap_239_, v___y_236_);
lean_dec(v___y_236_);
v___x_248_ = l_Lean_FileMap_toPosition(v_fileMap_239_, v___y_237_);
lean_dec(v___y_237_);
v___x_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
v___x_250_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___closed__0));
if (v_suppressElabErrors_240_ == 0)
{
lean_del_object(v___x_245_);
v___y_169_ = v___x_250_;
v___y_170_ = v_a_243_;
v___y_171_ = v___y_235_;
v___y_172_ = v___y_234_;
v___y_173_ = v___x_247_;
v___y_174_ = v___x_249_;
v___y_175_ = v_fileName_238_;
v___y_176_ = v___y_166_;
goto v___jp_168_;
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___f_253_; uint8_t v___x_254_; 
v___x_251_ = lean_box(v___y_233_);
v___x_252_ = lean_box(v_suppressElabErrors_240_);
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_253_, 0, v___x_251_);
lean_closure_set(v___f_253_, 1, v___x_252_);
lean_inc(v_a_243_);
v___x_254_ = l_Lean_MessageData_hasTag(v___f_253_, v_a_243_);
if (v___x_254_ == 0)
{
lean_object* v___x_255_; lean_object* v___x_257_; 
lean_dec_ref_known(v___x_249_, 1);
lean_dec_ref(v___x_247_);
lean_dec(v_a_243_);
v___x_255_ = lean_box(0);
if (v_isShared_246_ == 0)
{
lean_ctor_set(v___x_245_, 0, v___x_255_);
v___x_257_ = v___x_245_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v___x_255_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
else
{
lean_del_object(v___x_245_);
v___y_169_ = v___x_250_;
v___y_170_ = v_a_243_;
v___y_171_ = v___y_235_;
v___y_172_ = v___y_234_;
v___y_173_ = v___x_247_;
v___y_174_ = v___x_249_;
v___y_175_ = v_fileName_238_;
v___y_176_ = v___y_166_;
goto v___jp_168_;
}
}
}
}
v___jp_260_:
{
lean_object* v___x_266_; 
v___x_266_ = l_Lean_Syntax_getTailPos_x3f(v___y_264_, v___y_263_);
lean_dec(v___y_264_);
if (lean_obj_tag(v___x_266_) == 0)
{
lean_inc(v___y_265_);
v___y_233_ = v___y_261_;
v___y_234_ = v___y_263_;
v___y_235_ = v___y_262_;
v___y_236_ = v___y_265_;
v___y_237_ = v___y_265_;
goto v___jp_232_;
}
else
{
lean_object* v_val_267_; 
v_val_267_ = lean_ctor_get(v___x_266_, 0);
lean_inc(v_val_267_);
lean_dec_ref_known(v___x_266_, 1);
v___y_233_ = v___y_261_;
v___y_234_ = v___y_263_;
v___y_235_ = v___y_262_;
v___y_236_ = v___y_265_;
v___y_237_ = v_val_267_;
goto v___jp_232_;
}
}
v___jp_268_:
{
lean_object* v___x_272_; 
v___x_272_ = l_Lean_Elab_Command_getRef___redArg(v___y_165_);
if (lean_obj_tag(v___x_272_) == 0)
{
lean_object* v_a_273_; lean_object* v_ref_274_; lean_object* v___x_275_; 
v_a_273_ = lean_ctor_get(v___x_272_, 0);
lean_inc(v_a_273_);
lean_dec_ref_known(v___x_272_, 1);
v_ref_274_ = l_Lean_replaceRef(v_ref_161_, v_a_273_);
lean_dec(v_a_273_);
v___x_275_ = l_Lean_Syntax_getPos_x3f(v_ref_274_, v___y_270_);
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v___x_276_; 
v___x_276_ = lean_unsigned_to_nat(0u);
v___y_261_ = v___y_269_;
v___y_262_ = v___y_271_;
v___y_263_ = v___y_270_;
v___y_264_ = v_ref_274_;
v___y_265_ = v___x_276_;
goto v___jp_260_;
}
else
{
lean_object* v_val_277_; 
v_val_277_ = lean_ctor_get(v___x_275_, 0);
lean_inc(v_val_277_);
lean_dec_ref_known(v___x_275_, 1);
v___y_261_ = v___y_269_;
v___y_262_ = v___y_271_;
v___y_263_ = v___y_270_;
v___y_264_ = v_ref_274_;
v___y_265_ = v_val_277_;
goto v___jp_260_;
}
}
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
lean_dec_ref(v_msgData_162_);
v_a_278_ = lean_ctor_get(v___x_272_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_272_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___x_272_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v___x_272_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_283_; 
if (v_isShared_281_ == 0)
{
v___x_283_ = v___x_280_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_a_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
}
v___jp_287_:
{
if (v___y_290_ == 0)
{
v___y_269_ = v___y_288_;
v___y_270_ = v___y_289_;
v___y_271_ = v_severity_163_;
goto v___jp_268_;
}
else
{
v___y_269_ = v___y_288_;
v___y_270_ = v___y_289_;
v___y_271_ = v___x_286_;
goto v___jp_268_;
}
}
v___jp_291_:
{
if (v___y_292_ == 0)
{
lean_object* v___x_293_; lean_object* v_scopes_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v_opts_297_; uint8_t v___x_298_; uint8_t v___x_299_; 
v___x_293_ = lean_st_ref_get(v___y_166_);
v_scopes_294_ = lean_ctor_get(v___x_293_, 2);
lean_inc(v_scopes_294_);
lean_dec(v___x_293_);
v___x_295_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_296_ = l_List_head_x21___redArg(v___x_295_, v_scopes_294_);
lean_dec(v_scopes_294_);
v_opts_297_ = lean_ctor_get(v___x_296_, 1);
lean_inc_ref(v_opts_297_);
lean_dec(v___x_296_);
v___x_298_ = 1;
v___x_299_ = l_Lean_instBEqMessageSeverity_beq(v_severity_163_, v___x_298_);
if (v___x_299_ == 0)
{
lean_dec_ref(v_opts_297_);
v___y_288_ = v___y_292_;
v___y_289_ = v___y_292_;
v___y_290_ = v___x_299_;
goto v___jp_287_;
}
else
{
lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_300_ = l_Lean_warningAsError;
v___x_301_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__5(v_opts_297_, v___x_300_);
lean_dec_ref(v_opts_297_);
v___y_288_ = v___y_292_;
v___y_289_ = v___y_292_;
v___y_290_ = v___x_301_;
goto v___jp_287_;
}
}
else
{
lean_object* v___x_302_; lean_object* v___x_303_; 
lean_dec_ref(v_msgData_162_);
v___x_302_ = lean_box(0);
v___x_303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_303_, 0, v___x_302_);
return v___x_303_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3___boxed(lean_object* v_ref_306_, lean_object* v_msgData_307_, lean_object* v_severity_308_, lean_object* v_isSilent_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
uint8_t v_severity_boxed_313_; uint8_t v_isSilent_boxed_314_; lean_object* v_res_315_; 
v_severity_boxed_313_ = lean_unbox(v_severity_308_);
v_isSilent_boxed_314_ = lean_unbox(v_isSilent_309_);
v_res_315_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3(v_ref_306_, v_msgData_307_, v_severity_boxed_313_, v_isSilent_boxed_314_, v___y_310_, v___y_311_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v_ref_306_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2(lean_object* v_ref_316_, lean_object* v_msgData_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
uint8_t v___x_321_; uint8_t v___x_322_; lean_object* v___x_323_; 
v___x_321_ = 1;
v___x_322_ = 0;
v___x_323_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3(v_ref_316_, v_msgData_317_, v___x_321_, v___x_322_, v___y_318_, v___y_319_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2___boxed(lean_object* v_ref_324_, lean_object* v_msgData_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2(v_ref_324_, v_msgData_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v_ref_324_);
return v_res_329_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_331_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__0));
v___x_332_ = l_Lean_stringToMessageData(v___x_331_);
return v___x_332_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__2));
v___x_335_ = l_Lean_stringToMessageData(v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1(lean_object* v_linterOption_336_, lean_object* v_stx_337_, lean_object* v_msg_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_name_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_360_; 
v_name_342_ = lean_ctor_get(v_linterOption_336_, 0);
v_isSharedCheck_360_ = !lean_is_exclusive(v_linterOption_336_);
if (v_isSharedCheck_360_ == 0)
{
lean_object* v_unused_361_; 
v_unused_361_ = lean_ctor_get(v_linterOption_336_, 1);
lean_dec(v_unused_361_);
v___x_344_ = v_linterOption_336_;
v_isShared_345_ = v_isSharedCheck_360_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_name_342_);
lean_dec(v_linterOption_336_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_360_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_349_; 
v___x_346_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__1);
lean_inc(v_name_342_);
v___x_347_ = l_Lean_MessageData_ofName(v_name_342_);
if (v_isShared_345_ == 0)
{
lean_ctor_set_tag(v___x_344_, 7);
lean_ctor_set(v___x_344_, 1, v___x_347_);
lean_ctor_set(v___x_344_, 0, v___x_346_);
v___x_349_ = v___x_344_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_359_, 1, v___x_347_);
v___x_349_ = v_reuseFailAlloc_359_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v_disable_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_350_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___closed__3);
v___x_351_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_349_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
v_disable_352_ = l_Lean_MessageData_note(v___x_351_);
v___x_353_ = l_Lean_Linter_linterMessageTag;
v___x_354_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_354_, 0, v_msg_338_);
lean_ctor_set(v___x_354_, 1, v_disable_352_);
v___x_355_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_353_);
lean_ctor_set(v___x_355_, 1, v___x_354_);
v___x_356_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_356_, 0, v_name_342_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
lean_inc(v_stx_337_);
v___x_357_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_357_, 0, v_stx_337_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v___x_358_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2(v_stx_337_, v___x_357_, v___y_339_, v___y_340_);
lean_dec(v_stx_337_);
return v___x_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1___boxed(lean_object* v_linterOption_362_, lean_object* v_stx_363_, lean_object* v_msg_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1(v_linterOption_362_, v_stx_363_, v_msg_364_, v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg(lean_object* v_o_369_, lean_object* v___y_370_){
_start:
{
lean_object* v___x_372_; lean_object* v_env_373_; lean_object* v___x_374_; lean_object* v_toEnvExtension_375_; lean_object* v_asyncMode_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v_merged_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_388_; 
v___x_372_ = lean_st_ref_get(v___y_370_);
v_env_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc_ref(v_env_373_);
lean_dec(v___x_372_);
v___x_374_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_375_ = lean_ctor_get(v___x_374_, 0);
v_asyncMode_376_ = lean_ctor_get(v_toEnvExtension_375_, 2);
v___x_377_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_378_ = lean_box(0);
v___x_379_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_377_, v___x_374_, v_env_373_, v_asyncMode_376_, v___x_378_);
v_merged_380_ = lean_ctor_get(v___x_379_, 0);
v_isSharedCheck_388_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_388_ == 0)
{
lean_object* v_unused_389_; 
v_unused_389_ = lean_ctor_get(v___x_379_, 1);
lean_dec(v_unused_389_);
v___x_382_ = v___x_379_;
v_isShared_383_ = v_isSharedCheck_388_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_merged_380_);
lean_dec(v___x_379_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_388_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
if (v_isShared_383_ == 0)
{
lean_ctor_set(v___x_382_, 1, v_merged_380_);
lean_ctor_set(v___x_382_, 0, v_o_369_);
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_o_369_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v_merged_380_);
v___x_385_ = v_reuseFailAlloc_387_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
lean_object* v___x_386_; 
v___x_386_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
return v___x_386_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg(v_o_390_, v___y_391_);
lean_dec(v___y_391_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0(lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v___x_397_; lean_object* v_scopes_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v_opts_401_; lean_object* v___x_402_; 
v___x_397_ = lean_st_ref_get(v___y_395_);
v_scopes_398_ = lean_ctor_get(v___x_397_, 2);
lean_inc(v_scopes_398_);
lean_dec(v___x_397_);
v___x_399_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_400_ = l_List_head_x21___redArg(v___x_399_, v_scopes_398_);
lean_dec(v_scopes_398_);
v_opts_401_ = lean_ctor_get(v___x_400_, 1);
lean_inc_ref(v_opts_401_);
lean_dec(v___x_400_);
v___x_402_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg(v_opts_401_, v___y_395_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0___boxed(lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0(v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
return v_res_406_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__1));
v___x_410_ = l_Lean_stringToMessageData(v___x_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0(lean_object* v_stx_411_, lean_object* v___y_412_, lean_object* v___y_413_){
_start:
{
lean_object* v___x_415_; lean_object* v_a_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_442_; 
v___x_415_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0(v___y_412_, v___y_413_);
v_a_416_ = lean_ctor_get(v___x_415_, 0);
v_isSharedCheck_442_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_442_ == 0)
{
v___x_418_ = v___x_415_;
v_isShared_419_ = v_isSharedCheck_442_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_a_416_);
lean_dec(v___x_415_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_442_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v___x_420_; uint8_t v___x_421_; 
v___x_420_ = lp_mathlib_Mathlib_Linter_Style_linter_oldObtain;
v___x_421_ = l_Lean_Linter_getLinterValue(v___x_420_, v_a_416_);
lean_dec(v_a_416_);
if (v___x_421_ == 0)
{
lean_object* v___x_422_; lean_object* v___x_424_; 
lean_dec(v_stx_411_);
v___x_422_ = lean_box(0);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 0, v___x_422_);
v___x_424_ = v___x_418_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v___x_422_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
else
{
lean_object* v___x_426_; lean_object* v_messages_427_; uint8_t v___x_428_; 
v___x_426_ = lean_st_ref_get(v___y_413_);
v_messages_427_ = lean_ctor_get(v___x_426_, 1);
lean_inc_ref(v_messages_427_);
lean_dec(v___x_426_);
v___x_428_ = l_Lean_MessageLog_hasErrors(v_messages_427_);
lean_dec_ref(v_messages_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_429_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__0));
v___x_430_ = l_Lean_Syntax_find_x3f(v_stx_411_, v___x_429_);
if (lean_obj_tag(v___x_430_) == 1)
{
lean_object* v_val_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
lean_del_object(v___x_418_);
v_val_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_val_431_);
lean_dec_ref_known(v___x_430_, 1);
v___x_432_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___closed__2);
v___x_433_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1(v___x_420_, v_val_431_, v___x_432_, v___y_412_, v___y_413_);
return v___x_433_;
}
else
{
lean_object* v___x_434_; lean_object* v___x_436_; 
lean_dec(v___x_430_);
v___x_434_ = lean_box(0);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 0, v___x_434_);
v___x_436_ = v___x_418_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v___x_434_);
v___x_436_ = v_reuseFailAlloc_437_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
return v___x_436_;
}
}
}
else
{
lean_object* v___x_438_; lean_object* v___x_440_; 
lean_dec(v_stx_411_);
v___x_438_ = lean_box(0);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 0, v___x_438_);
v___x_440_ = v___x_418_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v___x_438_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0___boxed(lean_object* v_stx_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter___lam__0(v_stx_443_, v___y_444_, v___y_445_);
lean_dec(v___y_445_);
lean_dec_ref(v___y_444_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0(lean_object* v_o_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___redArg(v_o_488_, v___y_490_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0___boxed(lean_object* v_o_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__0_spec__0(v_o_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4(lean_object* v_msgData_498_, lean_object* v___y_499_, lean_object* v___y_500_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___redArg(v_msgData_498_, v___y_500_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4___boxed(lean_object* v_msgData_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter_spec__1_spec__2_spec__3_spec__4(v_msgData_503_, v___y_504_, v___y_505_);
lean_dec(v___y_505_);
lean_dec_ref(v___y_504_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_509_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_oldObtainLinter));
v___x_510_ = l_Lean_Elab_Command_addLinter(v___x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2____boxed(lean_object* v_a_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2_();
return v_res_512_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Message(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_741022444____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_Style_linter_oldObtain = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_Style_linter_oldObtain);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_OldObtain_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_OldObtain_3115300298____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Message(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
}
#ifdef __cplusplus
}
#endif
