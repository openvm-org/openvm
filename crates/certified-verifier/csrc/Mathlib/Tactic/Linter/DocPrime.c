// Lean compiler output
// Module: Mathlib.Tactic.Linter.DocPrime
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Command
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_System_FilePath_pathExists(lean_object*);
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
lean_object* l_IO_FS_lines(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prev_x3f(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_components(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "docPrime"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(216, 81, 61, 14, 8, 143, 225, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "enable the docPrime linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(51, 143, 226, 234, 204, 249, 102, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_docPrime;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "example"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__5_value),LEAN_SCALAR_PTR_LITERAL(117, 34, 246, 137, 114, 183, 220, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "scripts/nolints_prime_decls.txt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 317, .m_capacity = 317, .m_length = 316, .m_data = "` is missing a doc-string, please add one.\nDeclarations whose name ends with a `'` are expected to contain an explanation for the presence of a `'` in their doc-string. This may consist of discussion of the difference relative to the unprimed version, or an explanation as to why no better naming scheme is possible."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__15_value),LEAN_SCALAR_PTR_LITERAL(37, 156, 84, 218, 244, 57, 142, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__17_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___boxed, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__5_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "DocPrime"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__8_value),LEAN_SCALAR_PTR_LITERAL(51, 159, 150, 177, 240, 79, 170, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(158, 36, 27, 70, 57, 73, 209, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(87, 193, 85, 60, 216, 13, 10, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(173, 155, 69, 75, 185, 230, 96, 226)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__8_value),LEAN_SCALAR_PTR_LITERAL(91, 70, 228, 21, 111, 142, 125, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "docPrimeLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__14_value),LEAN_SCALAR_PTR_LITERAL(240, 162, 236, 61, 119, 197, 15, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_));
v___x_56_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0(lean_object* v___x_60_, lean_object* v___x_61_, lean_object* v___x_62_, lean_object* v_x_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___closed__0));
v___x_65_ = l_Lean_Name_mkStr4(v___x_60_, v___x_61_, v___x_62_, v___x_64_);
v___x_66_ = l_Lean_Syntax_isOfKind(v_x_63_, v___x_65_);
lean_dec(v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0___boxed(lean_object* v___x_67_, lean_object* v___x_68_, lean_object* v___x_69_, lean_object* v_x_70_){
_start:
{
uint8_t v_res_71_; lean_object* v_r_72_; 
v_res_71_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__0(v___x_67_, v___x_68_, v___x_69_, v_x_70_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1(lean_object* v___x_74_, lean_object* v___x_75_, lean_object* v___x_76_, lean_object* v_x_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_78_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___closed__0));
v___x_79_ = l_Lean_Name_mkStr4(v___x_74_, v___x_75_, v___x_76_, v___x_78_);
v___x_80_ = l_Lean_Syntax_isOfKind(v_x_77_, v___x_79_);
lean_dec(v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1___boxed(lean_object* v___x_81_, lean_object* v___x_82_, lean_object* v___x_83_, lean_object* v_x_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__1(v___x_81_, v___x_82_, v___x_83_, v_x_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(lean_object* v___y_87_, lean_object* v___x_88_, lean_object* v_currNamespace_89_, lean_object* v_x_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_91_ = l_Lean_Syntax_getArg(v___y_87_, v___x_88_);
v___x_92_ = l_Lean_Syntax_getId(v___x_91_);
lean_dec(v___x_91_);
v___x_93_ = l_Lean_Name_append(v_currNamespace_89_, v___x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2___boxed(lean_object* v___y_94_, lean_object* v___x_95_, lean_object* v_currNamespace_96_, lean_object* v_x_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(v___y_94_, v___x_95_, v_currNamespace_96_, v_x_97_);
lean_dec(v_x_97_);
lean_dec(v___x_95_);
lean_dec(v___y_94_);
return v_res_98_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8(lean_object* v_opts_99_, lean_object* v_opt_100_){
_start:
{
lean_object* v_name_101_; lean_object* v_defValue_102_; lean_object* v_map_103_; lean_object* v___x_104_; 
v_name_101_ = lean_ctor_get(v_opt_100_, 0);
v_defValue_102_ = lean_ctor_get(v_opt_100_, 1);
v_map_103_ = lean_ctor_get(v_opts_99_, 0);
v___x_104_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_103_, v_name_101_);
if (lean_obj_tag(v___x_104_) == 0)
{
uint8_t v___x_105_; 
v___x_105_ = lean_unbox(v_defValue_102_);
return v___x_105_;
}
else
{
lean_object* v_val_106_; 
v_val_106_ = lean_ctor_get(v___x_104_, 0);
lean_inc(v_val_106_);
lean_dec_ref_known(v___x_104_, 1);
if (lean_obj_tag(v_val_106_) == 1)
{
uint8_t v_v_107_; 
v_v_107_ = lean_ctor_get_uint8(v_val_106_, 0);
lean_dec_ref_known(v_val_106_, 0);
return v_v_107_;
}
else
{
uint8_t v___x_108_; 
lean_dec(v_val_106_);
v___x_108_ = lean_unbox(v_defValue_102_);
return v___x_108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8___boxed(lean_object* v_opts_109_, lean_object* v_opt_110_){
_start:
{
uint8_t v_res_111_; lean_object* v_r_112_; 
v_res_111_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8(v_opts_109_, v_opt_110_);
lean_dec_ref(v_opt_110_);
lean_dec_ref(v_opts_109_);
v_r_112_ = lean_box(v_res_111_);
return v_r_112_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0(uint8_t v___y_114_, uint8_t v_suppressElabErrors_115_, lean_object* v_x_116_){
_start:
{
if (lean_obj_tag(v_x_116_) == 1)
{
lean_object* v_pre_117_; 
v_pre_117_ = lean_ctor_get(v_x_116_, 0);
if (lean_obj_tag(v_pre_117_) == 0)
{
lean_object* v_str_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v_str_118_ = lean_ctor_get(v_x_116_, 1);
v___x_119_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___closed__0));
v___x_120_ = lean_string_dec_eq(v_str_118_, v___x_119_);
if (v___x_120_ == 0)
{
return v___y_114_;
}
else
{
return v_suppressElabErrors_115_;
}
}
else
{
return v___y_114_;
}
}
else
{
return v___y_114_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___boxed(lean_object* v___y_121_, lean_object* v_suppressElabErrors_122_, lean_object* v_x_123_){
_start:
{
uint8_t v___y_8795__boxed_124_; uint8_t v_suppressElabErrors_boxed_125_; uint8_t v_res_126_; lean_object* v_r_127_; 
v___y_8795__boxed_124_ = lean_unbox(v___y_121_);
v_suppressElabErrors_boxed_125_ = lean_unbox(v_suppressElabErrors_122_);
v_res_126_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0(v___y_8795__boxed_124_, v_suppressElabErrors_boxed_125_, v_x_123_);
lean_dec(v_x_123_);
v_r_127_ = lean_box(v_res_126_);
return v_r_127_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__0);
v___x_130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1);
v___x_132_ = lean_unsigned_to_nat(0u);
v___x_133_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
lean_ctor_set(v___x_133_, 2, v___x_132_);
lean_ctor_set(v___x_133_, 3, v___x_132_);
lean_ctor_set(v___x_133_, 4, v___x_131_);
lean_ctor_set(v___x_133_, 5, v___x_131_);
lean_ctor_set(v___x_133_, 6, v___x_131_);
lean_ctor_set(v___x_133_, 7, v___x_131_);
lean_ctor_set(v___x_133_, 8, v___x_131_);
lean_ctor_set(v___x_133_, 9, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = lean_unsigned_to_nat(32u);
v___x_135_ = lean_mk_empty_array_with_capacity(v___x_134_);
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4(void){
_start:
{
size_t v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_137_ = ((size_t)5ULL);
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = lean_unsigned_to_nat(32u);
v___x_140_ = lean_mk_empty_array_with_capacity(v___x_139_);
v___x_141_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__3);
v___x_142_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v___x_140_);
lean_ctor_set(v___x_142_, 2, v___x_138_);
lean_ctor_set(v___x_142_, 3, v___x_138_);
lean_ctor_set_usize(v___x_142_, 4, v___x_137_);
return v___x_142_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_143_ = lean_box(1);
v___x_144_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__4);
v___x_145_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__1);
v___x_146_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v___x_144_);
lean_ctor_set(v___x_146_, 2, v___x_143_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg(lean_object* v_msgData_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___x_150_; lean_object* v_env_151_; lean_object* v___x_152_; lean_object* v_scopes_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v_opts_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_150_ = lean_st_ref_get(v___y_148_);
v_env_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_env_151_);
lean_dec(v___x_150_);
v___x_152_ = lean_st_ref_get(v___y_148_);
v_scopes_153_ = lean_ctor_get(v___x_152_, 2);
lean_inc(v_scopes_153_);
lean_dec(v___x_152_);
v___x_154_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_155_ = l_List_head_x21___redArg(v___x_154_, v_scopes_153_);
lean_dec(v_scopes_153_);
v_opts_156_ = lean_ctor_get(v___x_155_, 1);
lean_inc_ref(v_opts_156_);
lean_dec(v___x_155_);
v___x_157_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__2);
v___x_158_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___closed__5);
v___x_159_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_159_, 0, v_env_151_);
lean_ctor_set(v___x_159_, 1, v___x_157_);
lean_ctor_set(v___x_159_, 2, v___x_158_);
lean_ctor_set(v___x_159_, 3, v_opts_156_);
v___x_160_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v_msgData_147_);
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object* v_msgData_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg(v_msgData_162_, v___y_163_);
lean_dec(v___y_163_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4(lean_object* v_ref_167_, lean_object* v_msgData_168_, uint8_t v_severity_169_, uint8_t v_isSilent_170_, lean_object* v___y_171_, lean_object* v___y_172_){
_start:
{
uint8_t v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_178_; uint8_t v___y_179_; lean_object* v___y_180_; lean_object* v___y_181_; lean_object* v___y_182_; uint8_t v___y_239_; uint8_t v___y_240_; lean_object* v___y_241_; uint8_t v___y_242_; lean_object* v___y_243_; uint8_t v___y_267_; lean_object* v___y_268_; uint8_t v___y_269_; uint8_t v___y_270_; lean_object* v___y_271_; uint8_t v___y_275_; uint8_t v___y_276_; uint8_t v___y_277_; uint8_t v___x_292_; uint8_t v___y_294_; uint8_t v___y_295_; uint8_t v___y_296_; uint8_t v___y_298_; uint8_t v___x_310_; 
v___x_292_ = 2;
v___x_310_ = l_Lean_instBEqMessageSeverity_beq(v_severity_169_, v___x_292_);
if (v___x_310_ == 0)
{
v___y_298_ = v___x_310_;
goto v___jp_297_;
}
else
{
uint8_t v___x_311_; 
lean_inc_ref(v_msgData_168_);
v___x_311_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_168_);
v___y_298_ = v___x_311_;
goto v___jp_297_;
}
v___jp_174_:
{
lean_object* v___x_183_; 
v___x_183_ = l_Lean_Elab_Command_getScope___redArg(v___y_182_);
if (lean_obj_tag(v___x_183_) == 0)
{
lean_object* v_a_184_; lean_object* v___x_185_; 
v_a_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc(v_a_184_);
lean_dec_ref_known(v___x_183_, 1);
v___x_185_ = l_Lean_Elab_Command_getScope___redArg(v___y_182_);
if (lean_obj_tag(v___x_185_) == 0)
{
lean_object* v_a_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_221_; 
v_a_186_ = lean_ctor_get(v___x_185_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_185_);
if (v_isSharedCheck_221_ == 0)
{
v___x_188_ = v___x_185_;
v_isShared_189_ = v_isSharedCheck_221_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_a_186_);
lean_dec(v___x_185_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_221_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_190_; lean_object* v_currNamespace_191_; lean_object* v_openDecls_192_; lean_object* v_env_193_; lean_object* v_messages_194_; lean_object* v_scopes_195_; lean_object* v_usedQuotCtxts_196_; lean_object* v_nextMacroScope_197_; lean_object* v_maxRecDepth_198_; lean_object* v_ngen_199_; lean_object* v_auxDeclNGen_200_; lean_object* v_infoState_201_; lean_object* v_traceState_202_; lean_object* v_snapshotTasks_203_; lean_object* v_prevLinterStates_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_220_; 
v___x_190_ = lean_st_ref_take(v___y_182_);
v_currNamespace_191_ = lean_ctor_get(v_a_184_, 2);
lean_inc(v_currNamespace_191_);
lean_dec(v_a_184_);
v_openDecls_192_ = lean_ctor_get(v_a_186_, 3);
lean_inc(v_openDecls_192_);
lean_dec(v_a_186_);
v_env_193_ = lean_ctor_get(v___x_190_, 0);
v_messages_194_ = lean_ctor_get(v___x_190_, 1);
v_scopes_195_ = lean_ctor_get(v___x_190_, 2);
v_usedQuotCtxts_196_ = lean_ctor_get(v___x_190_, 3);
v_nextMacroScope_197_ = lean_ctor_get(v___x_190_, 4);
v_maxRecDepth_198_ = lean_ctor_get(v___x_190_, 5);
v_ngen_199_ = lean_ctor_get(v___x_190_, 6);
v_auxDeclNGen_200_ = lean_ctor_get(v___x_190_, 7);
v_infoState_201_ = lean_ctor_get(v___x_190_, 8);
v_traceState_202_ = lean_ctor_get(v___x_190_, 9);
v_snapshotTasks_203_ = lean_ctor_get(v___x_190_, 10);
v_prevLinterStates_204_ = lean_ctor_get(v___x_190_, 11);
v_isSharedCheck_220_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_220_ == 0)
{
v___x_206_ = v___x_190_;
v_isShared_207_ = v_isSharedCheck_220_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_prevLinterStates_204_);
lean_inc(v_snapshotTasks_203_);
lean_inc(v_traceState_202_);
lean_inc(v_infoState_201_);
lean_inc(v_auxDeclNGen_200_);
lean_inc(v_ngen_199_);
lean_inc(v_maxRecDepth_198_);
lean_inc(v_nextMacroScope_197_);
lean_inc(v_usedQuotCtxts_196_);
lean_inc(v_scopes_195_);
lean_inc(v_messages_194_);
lean_inc(v_env_193_);
lean_dec(v___x_190_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_220_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_213_; 
v___x_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_208_, 0, v_currNamespace_191_);
lean_ctor_set(v___x_208_, 1, v_openDecls_192_);
v___x_209_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v___y_181_);
lean_inc_ref(v___y_177_);
lean_inc_ref(v___y_180_);
v___x_210_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_210_, 0, v___y_180_);
lean_ctor_set(v___x_210_, 1, v___y_176_);
lean_ctor_set(v___x_210_, 2, v___y_178_);
lean_ctor_set(v___x_210_, 3, v___y_177_);
lean_ctor_set(v___x_210_, 4, v___x_209_);
lean_ctor_set_uint8(v___x_210_, sizeof(void*)*5, v___y_179_);
lean_ctor_set_uint8(v___x_210_, sizeof(void*)*5 + 1, v___y_175_);
lean_ctor_set_uint8(v___x_210_, sizeof(void*)*5 + 2, v_isSilent_170_);
v___x_211_ = l_Lean_MessageLog_add(v___x_210_, v_messages_194_);
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 1, v___x_211_);
v___x_213_ = v___x_206_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v_env_193_);
lean_ctor_set(v_reuseFailAlloc_219_, 1, v___x_211_);
lean_ctor_set(v_reuseFailAlloc_219_, 2, v_scopes_195_);
lean_ctor_set(v_reuseFailAlloc_219_, 3, v_usedQuotCtxts_196_);
lean_ctor_set(v_reuseFailAlloc_219_, 4, v_nextMacroScope_197_);
lean_ctor_set(v_reuseFailAlloc_219_, 5, v_maxRecDepth_198_);
lean_ctor_set(v_reuseFailAlloc_219_, 6, v_ngen_199_);
lean_ctor_set(v_reuseFailAlloc_219_, 7, v_auxDeclNGen_200_);
lean_ctor_set(v_reuseFailAlloc_219_, 8, v_infoState_201_);
lean_ctor_set(v_reuseFailAlloc_219_, 9, v_traceState_202_);
lean_ctor_set(v_reuseFailAlloc_219_, 10, v_snapshotTasks_203_);
lean_ctor_set(v_reuseFailAlloc_219_, 11, v_prevLinterStates_204_);
v___x_213_ = v_reuseFailAlloc_219_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_217_; 
v___x_214_ = lean_st_ref_set(v___y_182_, v___x_213_);
v___x_215_ = lean_box(0);
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 0, v___x_215_);
v___x_217_ = v___x_188_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v___x_215_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
}
}
else
{
lean_object* v_a_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
lean_dec(v_a_184_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_176_);
v_a_222_ = lean_ctor_get(v___x_185_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_185_);
if (v_isSharedCheck_229_ == 0)
{
v___x_224_ = v___x_185_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_a_222_);
lean_dec(v___x_185_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_222_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
else
{
lean_object* v_a_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_237_; 
lean_dec_ref(v___y_181_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_176_);
v_a_230_ = lean_ctor_get(v___x_183_, 0);
v_isSharedCheck_237_ = !lean_is_exclusive(v___x_183_);
if (v_isSharedCheck_237_ == 0)
{
v___x_232_ = v___x_183_;
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_a_230_);
lean_dec(v___x_183_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_235_; 
if (v_isShared_233_ == 0)
{
v___x_235_ = v___x_232_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_a_230_);
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
v___jp_238_:
{
lean_object* v_fileName_244_; lean_object* v_fileMap_245_; uint8_t v_suppressElabErrors_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_265_; 
v_fileName_244_ = lean_ctor_get(v___y_171_, 0);
v_fileMap_245_ = lean_ctor_get(v___y_171_, 1);
v_suppressElabErrors_246_ = lean_ctor_get_uint8(v___y_171_, sizeof(void*)*10);
v___x_247_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_168_);
v___x_248_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg(v___x_247_, v___y_172_);
v_a_249_ = lean_ctor_get(v___x_248_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_248_);
if (v_isSharedCheck_265_ == 0)
{
v___x_251_ = v___x_248_;
v_isShared_252_ = v_isSharedCheck_265_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_248_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_265_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
lean_inc_ref_n(v_fileMap_245_, 2);
v___x_253_ = l_Lean_FileMap_toPosition(v_fileMap_245_, v___y_241_);
lean_dec(v___y_241_);
v___x_254_ = l_Lean_FileMap_toPosition(v_fileMap_245_, v___y_243_);
lean_dec(v___y_243_);
v___x_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
v___x_256_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___closed__0));
if (v_suppressElabErrors_246_ == 0)
{
lean_del_object(v___x_251_);
v___y_175_ = v___y_240_;
v___y_176_ = v___x_253_;
v___y_177_ = v___x_256_;
v___y_178_ = v___x_255_;
v___y_179_ = v___y_242_;
v___y_180_ = v_fileName_244_;
v___y_181_ = v_a_249_;
v___y_182_ = v___y_172_;
goto v___jp_174_;
}
else
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___f_259_; uint8_t v___x_260_; 
v___x_257_ = lean_box(v___y_239_);
v___x_258_ = lean_box(v_suppressElabErrors_246_);
v___f_259_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_259_, 0, v___x_257_);
lean_closure_set(v___f_259_, 1, v___x_258_);
lean_inc(v_a_249_);
v___x_260_ = l_Lean_MessageData_hasTag(v___f_259_, v_a_249_);
if (v___x_260_ == 0)
{
lean_object* v___x_261_; lean_object* v___x_263_; 
lean_dec_ref_known(v___x_255_, 1);
lean_dec_ref(v___x_253_);
lean_dec(v_a_249_);
v___x_261_ = lean_box(0);
if (v_isShared_252_ == 0)
{
lean_ctor_set(v___x_251_, 0, v___x_261_);
v___x_263_ = v___x_251_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_261_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
else
{
lean_del_object(v___x_251_);
v___y_175_ = v___y_240_;
v___y_176_ = v___x_253_;
v___y_177_ = v___x_256_;
v___y_178_ = v___x_255_;
v___y_179_ = v___y_242_;
v___y_180_ = v_fileName_244_;
v___y_181_ = v_a_249_;
v___y_182_ = v___y_172_;
goto v___jp_174_;
}
}
}
}
v___jp_266_:
{
lean_object* v___x_272_; 
v___x_272_ = l_Lean_Syntax_getTailPos_x3f(v___y_268_, v___y_270_);
lean_dec(v___y_268_);
if (lean_obj_tag(v___x_272_) == 0)
{
lean_inc(v___y_271_);
v___y_239_ = v___y_267_;
v___y_240_ = v___y_269_;
v___y_241_ = v___y_271_;
v___y_242_ = v___y_270_;
v___y_243_ = v___y_271_;
goto v___jp_238_;
}
else
{
lean_object* v_val_273_; 
v_val_273_ = lean_ctor_get(v___x_272_, 0);
lean_inc(v_val_273_);
lean_dec_ref_known(v___x_272_, 1);
v___y_239_ = v___y_267_;
v___y_240_ = v___y_269_;
v___y_241_ = v___y_271_;
v___y_242_ = v___y_270_;
v___y_243_ = v_val_273_;
goto v___jp_238_;
}
}
v___jp_274_:
{
lean_object* v___x_278_; 
v___x_278_ = l_Lean_Elab_Command_getRef___redArg(v___y_171_);
if (lean_obj_tag(v___x_278_) == 0)
{
lean_object* v_a_279_; lean_object* v_ref_280_; lean_object* v___x_281_; 
v_a_279_ = lean_ctor_get(v___x_278_, 0);
lean_inc(v_a_279_);
lean_dec_ref_known(v___x_278_, 1);
v_ref_280_ = l_Lean_replaceRef(v_ref_167_, v_a_279_);
lean_dec(v_a_279_);
v___x_281_ = l_Lean_Syntax_getPos_x3f(v_ref_280_, v___y_276_);
if (lean_obj_tag(v___x_281_) == 0)
{
lean_object* v___x_282_; 
v___x_282_ = lean_unsigned_to_nat(0u);
v___y_267_ = v___y_275_;
v___y_268_ = v_ref_280_;
v___y_269_ = v___y_277_;
v___y_270_ = v___y_276_;
v___y_271_ = v___x_282_;
goto v___jp_266_;
}
else
{
lean_object* v_val_283_; 
v_val_283_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_val_283_);
lean_dec_ref_known(v___x_281_, 1);
v___y_267_ = v___y_275_;
v___y_268_ = v_ref_280_;
v___y_269_ = v___y_277_;
v___y_270_ = v___y_276_;
v___y_271_ = v_val_283_;
goto v___jp_266_;
}
}
else
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec_ref(v_msgData_168_);
v_a_284_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_278_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_278_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
v___jp_293_:
{
if (v___y_296_ == 0)
{
v___y_275_ = v___y_294_;
v___y_276_ = v___y_295_;
v___y_277_ = v_severity_169_;
goto v___jp_274_;
}
else
{
v___y_275_ = v___y_294_;
v___y_276_ = v___y_295_;
v___y_277_ = v___x_292_;
goto v___jp_274_;
}
}
v___jp_297_:
{
if (v___y_298_ == 0)
{
lean_object* v___x_299_; lean_object* v_scopes_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v_opts_303_; uint8_t v___x_304_; uint8_t v___x_305_; 
v___x_299_ = lean_st_ref_get(v___y_172_);
v_scopes_300_ = lean_ctor_get(v___x_299_, 2);
lean_inc(v_scopes_300_);
lean_dec(v___x_299_);
v___x_301_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_302_ = l_List_head_x21___redArg(v___x_301_, v_scopes_300_);
lean_dec(v_scopes_300_);
v_opts_303_ = lean_ctor_get(v___x_302_, 1);
lean_inc_ref(v_opts_303_);
lean_dec(v___x_302_);
v___x_304_ = 1;
v___x_305_ = l_Lean_instBEqMessageSeverity_beq(v_severity_169_, v___x_304_);
if (v___x_305_ == 0)
{
lean_dec_ref(v_opts_303_);
v___y_294_ = v___y_298_;
v___y_295_ = v___y_298_;
v___y_296_ = v___x_305_;
goto v___jp_293_;
}
else
{
lean_object* v___x_306_; uint8_t v___x_307_; 
v___x_306_ = l_Lean_warningAsError;
v___x_307_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__8(v_opts_303_, v___x_306_);
lean_dec_ref(v_opts_303_);
v___y_294_ = v___y_298_;
v___y_295_ = v___y_298_;
v___y_296_ = v___x_307_;
goto v___jp_293_;
}
}
else
{
lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec_ref(v_msgData_168_);
v___x_308_ = lean_box(0);
v___x_309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
return v___x_309_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4___boxed(lean_object* v_ref_312_, lean_object* v_msgData_313_, lean_object* v_severity_314_, lean_object* v_isSilent_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
uint8_t v_severity_boxed_319_; uint8_t v_isSilent_boxed_320_; lean_object* v_res_321_; 
v_severity_boxed_319_ = lean_unbox(v_severity_314_);
v_isSilent_boxed_320_ = lean_unbox(v_isSilent_315_);
v_res_321_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4(v_ref_312_, v_msgData_313_, v_severity_boxed_319_, v_isSilent_boxed_320_, v___y_316_, v___y_317_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
lean_dec(v_ref_312_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3(lean_object* v_ref_322_, lean_object* v_msgData_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
uint8_t v___x_327_; uint8_t v___x_328_; lean_object* v___x_329_; 
v___x_327_ = 1;
v___x_328_ = 0;
v___x_329_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4(v_ref_322_, v_msgData_323_, v___x_327_, v___x_328_, v___y_324_, v___y_325_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3___boxed(lean_object* v_ref_330_, lean_object* v_msgData_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3(v_ref_330_, v_msgData_331_, v___y_332_, v___y_333_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
lean_dec(v_ref_330_);
return v_res_335_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__0));
v___x_338_ = l_Lean_stringToMessageData(v___x_337_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_340_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__2));
v___x_341_ = l_Lean_stringToMessageData(v___x_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2(lean_object* v_linterOption_342_, lean_object* v_stx_343_, lean_object* v_msg_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v_name_348_; lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_366_; 
v_name_348_ = lean_ctor_get(v_linterOption_342_, 0);
v_isSharedCheck_366_ = !lean_is_exclusive(v_linterOption_342_);
if (v_isSharedCheck_366_ == 0)
{
lean_object* v_unused_367_; 
v_unused_367_ = lean_ctor_get(v_linterOption_342_, 1);
lean_dec(v_unused_367_);
v___x_350_ = v_linterOption_342_;
v_isShared_351_ = v_isSharedCheck_366_;
goto v_resetjp_349_;
}
else
{
lean_inc(v_name_348_);
lean_dec(v_linterOption_342_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_366_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_355_; 
v___x_352_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__1);
lean_inc(v_name_348_);
v___x_353_ = l_Lean_MessageData_ofName(v_name_348_);
if (v_isShared_351_ == 0)
{
lean_ctor_set_tag(v___x_350_, 7);
lean_ctor_set(v___x_350_, 1, v___x_353_);
lean_ctor_set(v___x_350_, 0, v___x_352_);
v___x_355_ = v___x_350_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v___x_352_);
lean_ctor_set(v_reuseFailAlloc_365_, 1, v___x_353_);
v___x_355_ = v_reuseFailAlloc_365_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v_disable_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_356_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___closed__3);
v___x_357_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_355_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v_disable_358_ = l_Lean_MessageData_note(v___x_357_);
v___x_359_ = l_Lean_Linter_linterMessageTag;
v___x_360_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_360_, 0, v_msg_344_);
lean_ctor_set(v___x_360_, 1, v_disable_358_);
v___x_361_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_359_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
v___x_362_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_362_, 0, v_name_348_);
lean_ctor_set(v___x_362_, 1, v___x_361_);
lean_inc(v_stx_343_);
v___x_363_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_363_, 0, v_stx_343_);
lean_ctor_set(v___x_363_, 1, v___x_362_);
v___x_364_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3(v_stx_343_, v___x_363_, v___y_345_, v___y_346_);
lean_dec(v_stx_343_);
return v___x_364_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2___boxed(lean_object* v_linterOption_368_, lean_object* v_stx_369_, lean_object* v_msg_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2(v_linterOption_368_, v_stx_369_, v_msg_370_, v___y_371_, v___y_372_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg(lean_object* v_o_375_, lean_object* v___y_376_){
_start:
{
lean_object* v___x_378_; lean_object* v_env_379_; lean_object* v___x_380_; lean_object* v_toEnvExtension_381_; lean_object* v_asyncMode_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v_merged_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_394_; 
v___x_378_ = lean_st_ref_get(v___y_376_);
v_env_379_ = lean_ctor_get(v___x_378_, 0);
lean_inc_ref(v_env_379_);
lean_dec(v___x_378_);
v___x_380_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_381_ = lean_ctor_get(v___x_380_, 0);
v_asyncMode_382_ = lean_ctor_get(v_toEnvExtension_381_, 2);
v___x_383_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_384_ = lean_box(0);
v___x_385_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_383_, v___x_380_, v_env_379_, v_asyncMode_382_, v___x_384_);
v_merged_386_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_394_ == 0)
{
lean_object* v_unused_395_; 
v_unused_395_ = lean_ctor_get(v___x_385_, 1);
lean_dec(v_unused_395_);
v___x_388_ = v___x_385_;
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_merged_386_);
lean_dec(v___x_385_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_391_; 
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 1, v_merged_386_);
lean_ctor_set(v___x_388_, 0, v_o_375_);
v___x_391_ = v___x_388_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_o_375_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_merged_386_);
v___x_391_ = v_reuseFailAlloc_393_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
lean_object* v___x_392_; 
v___x_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
return v___x_392_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_396_, lean_object* v___y_397_, lean_object* v___y_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg(v_o_396_, v___y_397_);
lean_dec(v___y_397_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0(lean_object* v___y_400_, lean_object* v___y_401_){
_start:
{
lean_object* v___x_403_; lean_object* v_scopes_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v_opts_407_; lean_object* v___x_408_; 
v___x_403_ = lean_st_ref_get(v___y_401_);
v_scopes_404_ = lean_ctor_get(v___x_403_, 2);
lean_inc(v_scopes_404_);
lean_dec(v___x_403_);
v___x_405_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_406_ = l_List_head_x21___redArg(v___x_405_, v_scopes_404_);
lean_dec(v_scopes_404_);
v_opts_407_ = lean_ctor_get(v___x_406_, 1);
lean_inc_ref(v_opts_407_);
lean_dec(v___x_406_);
v___x_408_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg(v_opts_407_, v___y_401_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0___boxed(lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0(v___y_409_, v___y_410_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__4(lean_object* v_x_413_, lean_object* v_x_414_){
_start:
{
if (lean_obj_tag(v_x_414_) == 0)
{
return v_x_413_;
}
else
{
lean_object* v_head_415_; lean_object* v_tail_416_; lean_object* v___x_417_; 
v_head_415_ = lean_ctor_get(v_x_414_, 0);
lean_inc(v_head_415_);
v_tail_416_ = lean_ctor_get(v_x_414_, 1);
lean_inc(v_tail_416_);
lean_dec_ref_known(v_x_414_, 2);
v___x_417_ = l_Lean_Name_append(v_x_413_, v_head_415_);
v_x_413_ = v___x_417_;
v_x_414_ = v_tail_416_;
goto _start;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5(lean_object* v_a_419_, lean_object* v_as_420_, size_t v_i_421_, size_t v_stop_422_){
_start:
{
uint8_t v___x_423_; 
v___x_423_ = lean_usize_dec_eq(v_i_421_, v_stop_422_);
if (v___x_423_ == 0)
{
lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_424_ = lean_array_uget_borrowed(v_as_420_, v_i_421_);
v___x_425_ = lean_string_dec_eq(v_a_419_, v___x_424_);
if (v___x_425_ == 0)
{
size_t v___x_426_; size_t v___x_427_; 
v___x_426_ = ((size_t)1ULL);
v___x_427_ = lean_usize_add(v_i_421_, v___x_426_);
v_i_421_ = v___x_427_;
goto _start;
}
else
{
return v___x_425_;
}
}
else
{
uint8_t v___x_429_; 
v___x_429_ = 0;
return v___x_429_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5___boxed(lean_object* v_a_430_, lean_object* v_as_431_, lean_object* v_i_432_, lean_object* v_stop_433_){
_start:
{
size_t v_i_boxed_434_; size_t v_stop_boxed_435_; uint8_t v_res_436_; lean_object* v_r_437_; 
v_i_boxed_434_ = lean_unbox_usize(v_i_432_);
lean_dec(v_i_432_);
v_stop_boxed_435_ = lean_unbox_usize(v_stop_433_);
lean_dec(v_stop_433_);
v_res_436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5(v_a_430_, v_as_431_, v_i_boxed_434_, v_stop_boxed_435_);
lean_dec_ref(v_as_431_);
lean_dec_ref(v_a_430_);
v_r_437_ = lean_box(v_res_436_);
return v_r_437_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3(lean_object* v_as_438_, lean_object* v_a_439_){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; uint8_t v___x_442_; 
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = lean_array_get_size(v_as_438_);
v___x_442_ = lean_nat_dec_lt(v___x_440_, v___x_441_);
if (v___x_442_ == 0)
{
return v___x_442_;
}
else
{
if (v___x_442_ == 0)
{
return v___x_442_;
}
else
{
size_t v___x_443_; size_t v___x_444_; uint8_t v___x_445_; 
v___x_443_ = ((size_t)0ULL);
v___x_444_ = lean_usize_of_nat(v___x_441_);
v___x_445_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3_spec__5(v_a_439_, v_as_438_, v___x_443_, v___x_444_);
return v___x_445_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3___boxed(lean_object* v_as_446_, lean_object* v_a_447_){
_start:
{
uint8_t v_res_448_; lean_object* v_r_449_; 
v_res_448_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3(v_as_446_, v_a_447_);
lean_dec_ref(v_a_447_);
lean_dec_ref(v_as_446_);
v_r_449_ = lean_box(v_res_448_);
return v_r_449_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1(lean_object* v_a_450_, lean_object* v_x_451_){
_start:
{
if (lean_obj_tag(v_x_451_) == 0)
{
uint8_t v___x_452_; 
v___x_452_ = 0;
return v___x_452_;
}
else
{
lean_object* v_head_453_; lean_object* v_tail_454_; uint8_t v___x_455_; 
v_head_453_ = lean_ctor_get(v_x_451_, 0);
v_tail_454_ = lean_ctor_get(v_x_451_, 1);
v___x_455_ = lean_name_eq(v_a_450_, v_head_453_);
if (v___x_455_ == 0)
{
v_x_451_ = v_tail_454_;
goto _start;
}
else
{
return v___x_455_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1___boxed(lean_object* v_a_457_, lean_object* v_x_458_){
_start:
{
uint8_t v_res_459_; lean_object* v_r_460_; 
v_res_459_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1(v_a_457_, v_x_458_);
lean_dec(v_x_458_);
lean_dec(v_a_457_);
v_r_460_ = lean_box(v_res_459_);
return v_r_460_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11(void){
_start:
{
lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_481_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__10));
v___x_482_ = l_Lean_stringToMessageData(v___x_481_);
return v___x_482_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13(void){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_484_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__12));
v___x_485_ = l_Lean_stringToMessageData(v___x_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3(lean_object* v_stx_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
lean_object* v___x_505_; lean_object* v_a_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_661_; 
v___x_505_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0(v___y_502_, v___y_503_);
v_a_506_ = lean_ctor_get(v___x_505_, 0);
v_isSharedCheck_661_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_661_ == 0)
{
v___x_508_ = v___x_505_;
v_isShared_509_ = v_isSharedCheck_661_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_a_506_);
lean_dec(v___x_505_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_661_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_510_ = lp_mathlib_Mathlib_Linter_linter_docPrime;
v___x_511_ = l_Lean_Linter_getLinterValue(v___x_510_, v_a_506_);
lean_dec(v_a_506_);
if (v___x_511_ == 0)
{
lean_object* v___x_512_; lean_object* v___x_514_; 
lean_dec(v_stx_501_);
v___x_512_ = lean_box(0);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v___x_512_);
v___x_514_ = v___x_508_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_512_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
else
{
lean_object* v___x_516_; lean_object* v_messages_517_; uint8_t v___x_518_; 
v___x_516_ = lean_st_ref_get(v___y_503_);
v_messages_517_ = lean_ctor_get(v___x_516_, 1);
lean_inc_ref(v_messages_517_);
lean_dec(v___x_516_);
v___x_518_ = l_Lean_MessageLog_hasErrors(v_messages_517_);
lean_dec_ref(v_messages_517_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; lean_object* v___x_520_; uint8_t v___x_521_; lean_object* v___y_523_; lean_object* v___y_524_; lean_object* v___y_525_; uint8_t v___y_526_; lean_object* v___y_561_; lean_object* v___y_562_; lean_object* v___y_563_; uint32_t v___y_564_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v___y_596_; lean_object* v___y_597_; 
v___x_519_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__8));
lean_inc(v_stx_501_);
v___x_520_ = l_Lean_Syntax_getKind(v_stx_501_);
v___x_521_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__1(v___x_520_, v___x_519_);
lean_dec(v___x_520_);
if (v___x_521_ == 0)
{
lean_object* v___x_638_; lean_object* v___x_639_; 
lean_del_object(v___x_508_);
lean_dec(v_stx_501_);
v___x_638_ = lean_box(0);
v___x_639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_639_, 0, v___x_638_);
return v___x_639_;
}
else
{
lean_object* v___f_640_; uint8_t v___y_642_; lean_object* v___f_655_; lean_object* v___x_656_; 
v___f_640_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__17));
v___f_655_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__18));
lean_inc(v_stx_501_);
v___x_656_ = l_Lean_Syntax_find_x3f(v_stx_501_, v___f_655_);
if (lean_obj_tag(v___x_656_) == 0)
{
v___y_642_ = v___x_518_;
goto v___jp_641_;
}
else
{
lean_dec_ref_known(v___x_656_, 1);
v___y_642_ = v___x_521_;
goto v___jp_641_;
}
v___jp_641_:
{
if (v___y_642_ == 0)
{
lean_object* v___x_643_; 
lean_inc(v_stx_501_);
v___x_643_ = l_Lean_Syntax_find_x3f(v_stx_501_, v___f_640_);
if (lean_obj_tag(v___x_643_) == 0)
{
goto v___jp_626_;
}
else
{
lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_651_; 
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_643_);
if (v_isSharedCheck_651_ == 0)
{
lean_object* v_unused_652_; 
v_unused_652_ = lean_ctor_get(v___x_643_, 0);
lean_dec(v_unused_652_);
v___x_645_ = v___x_643_;
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
else
{
lean_dec(v___x_643_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
if (v___x_521_ == 0)
{
lean_del_object(v___x_645_);
goto v___jp_626_;
}
else
{
lean_object* v___x_647_; lean_object* v___x_649_; 
lean_del_object(v___x_508_);
lean_dec(v_stx_501_);
v___x_647_ = lean_box(0);
if (v_isShared_646_ == 0)
{
lean_ctor_set_tag(v___x_645_, 0);
lean_ctor_set(v___x_645_, 0, v___x_647_);
v___x_649_ = v___x_645_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_647_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
}
}
}
else
{
lean_object* v___x_653_; lean_object* v___x_654_; 
lean_del_object(v___x_508_);
lean_dec(v_stx_501_);
v___x_653_ = lean_box(0);
v___x_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
return v___x_654_;
}
}
}
v___jp_522_:
{
if (v___y_526_ == 0)
{
lean_object* v___x_527_; lean_object* v___x_529_; 
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
v___x_527_ = lean_box(0);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v___x_527_);
v___x_529_ = v___x_508_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_527_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
else
{
lean_object* v___x_531_; uint8_t v___x_532_; 
lean_del_object(v___x_508_);
v___x_531_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__9));
v___x_532_ = l_System_FilePath_pathExists(v___x_531_);
if (v___x_532_ == 0)
{
lean_object* v___x_533_; 
lean_dec(v___y_523_);
v___x_533_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2(v___x_510_, v___y_525_, v___y_524_, v___y_502_, v___y_503_);
return v___x_533_;
}
else
{
lean_object* v___x_534_; 
v___x_534_ = l_IO_FS_lines(v___x_531_);
if (lean_obj_tag(v___x_534_) == 0)
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_546_; 
v_a_535_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_546_ == 0)
{
v___x_537_ = v___x_534_;
v_isShared_538_ = v_isSharedCheck_546_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_534_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_546_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_539_; uint8_t v___x_540_; 
v___x_539_ = l_Lean_Name_toString(v___y_523_, v___x_521_);
v___x_540_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__3(v_a_535_, v___x_539_);
lean_dec_ref(v___x_539_);
lean_dec(v_a_535_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; 
lean_del_object(v___x_537_);
v___x_541_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2(v___x_510_, v___y_525_, v___y_524_, v___y_502_, v___y_503_);
return v___x_541_;
}
else
{
lean_object* v___x_542_; lean_object* v___x_544_; 
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
v___x_542_ = lean_box(0);
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 0, v___x_542_);
v___x_544_ = v___x_537_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
else
{
lean_object* v_a_547_; lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_559_; 
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
v_a_547_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_559_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_559_ == 0)
{
v___x_549_ = v___x_534_;
v_isShared_550_ = v_isSharedCheck_559_;
goto v_resetjp_548_;
}
else
{
lean_inc(v_a_547_);
lean_dec(v___x_534_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_559_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
lean_object* v_ref_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_557_; 
v_ref_551_ = lean_ctor_get(v___y_502_, 7);
v___x_552_ = lean_io_error_to_string(v_a_547_);
v___x_553_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_553_, 0, v___x_552_);
v___x_554_ = l_Lean_MessageData_ofFormat(v___x_553_);
lean_inc(v_ref_551_);
v___x_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_555_, 0, v_ref_551_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
if (v_isShared_550_ == 0)
{
lean_ctor_set(v___x_549_, 0, v___x_555_);
v___x_557_ = v___x_549_;
goto v_reusejp_556_;
}
else
{
lean_object* v_reuseFailAlloc_558_; 
v_reuseFailAlloc_558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_558_, 0, v___x_555_);
v___x_557_ = v_reuseFailAlloc_558_;
goto v_reusejp_556_;
}
v_reusejp_556_:
{
return v___x_557_;
}
}
}
}
}
}
v___jp_560_:
{
uint32_t v___x_565_; uint8_t v___x_566_; 
v___x_565_ = 39;
v___x_566_ = lean_uint32_dec_eq(v___y_564_, v___x_565_);
v___y_523_ = v___y_561_;
v___y_524_ = v___y_562_;
v___y_525_ = v___y_563_;
v___y_526_ = v___x_566_;
goto v___jp_522_;
}
v___jp_567_:
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; uint8_t v___x_582_; 
v___x_573_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__11);
lean_inc(v___y_572_);
v___x_574_ = l_Lean_MessageData_ofName(v___y_572_);
v___x_575_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_575_, 0, v___x_573_);
lean_ctor_set(v___x_575_, 1, v___x_574_);
v___x_576_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13, &lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__13);
v___x_577_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_577_, 0, v___x_575_);
lean_ctor_set(v___x_577_, 1, v___x_576_);
v___x_578_ = l_Lean_Syntax_getArg(v___y_569_, v___y_568_);
lean_dec(v___y_569_);
v___x_579_ = l_Lean_Syntax_getArg(v___x_578_, v___y_571_);
lean_dec(v___x_578_);
v___x_580_ = l_Lean_Syntax_getAtomVal(v___x_579_);
lean_dec(v___x_579_);
v___x_581_ = lean_string_utf8_byte_size(v___x_580_);
lean_dec_ref(v___x_580_);
v___x_582_ = lean_nat_dec_eq(v___x_581_, v___y_568_);
if (v___x_582_ == 0)
{
lean_dec(v___y_568_);
v___y_523_ = v___y_572_;
v___y_524_ = v___x_577_;
v___y_525_ = v___y_570_;
v___y_526_ = v___x_582_;
goto v___jp_522_;
}
else
{
lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
lean_inc(v___y_572_);
v___x_583_ = l_Lean_Name_toString(v___y_572_, v___x_521_);
v___x_584_ = lean_string_utf8_byte_size(v___x_583_);
v___x_585_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_585_, 0, v___x_583_);
lean_ctor_set(v___x_585_, 1, v___y_568_);
lean_ctor_set(v___x_585_, 2, v___x_584_);
v___x_586_ = l_String_Slice_Pos_prev_x3f(v___x_585_, v___x_584_);
if (lean_obj_tag(v___x_586_) == 0)
{
uint32_t v___x_587_; 
lean_dec_ref_known(v___x_585_, 3);
v___x_587_ = 65;
v___y_561_ = v___y_572_;
v___y_562_ = v___x_577_;
v___y_563_ = v___y_570_;
v___y_564_ = v___x_587_;
goto v___jp_560_;
}
else
{
lean_object* v_val_588_; lean_object* v___x_589_; 
v_val_588_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_val_588_);
lean_dec_ref_known(v___x_586_, 1);
v___x_589_ = l_String_Slice_Pos_get_x3f(v___x_585_, v_val_588_);
lean_dec(v_val_588_);
lean_dec_ref_known(v___x_585_, 3);
if (lean_obj_tag(v___x_589_) == 0)
{
uint32_t v___x_590_; 
v___x_590_ = 65;
v___y_561_ = v___y_572_;
v___y_562_ = v___x_577_;
v___y_563_ = v___y_570_;
v___y_564_ = v___x_590_;
goto v___jp_560_;
}
else
{
lean_object* v_val_591_; uint32_t v___x_592_; 
v_val_591_ = lean_ctor_get(v___x_589_, 0);
lean_inc(v_val_591_);
lean_dec_ref_known(v___x_589_, 1);
v___x_592_ = lean_unbox_uint32(v_val_591_);
lean_dec(v_val_591_);
v___y_561_ = v___y_572_;
v___y_562_ = v___x_577_;
v___y_563_ = v___y_570_;
v___y_564_ = v___x_592_;
goto v___jp_560_;
}
}
}
}
v___jp_593_:
{
if (lean_obj_tag(v___y_597_) == 0)
{
lean_object* v___x_598_; lean_object* v___x_599_; 
lean_dec(v___y_595_);
lean_dec(v___y_594_);
lean_del_object(v___x_508_);
v___x_598_ = lean_box(0);
v___x_599_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_599_, 0, v___x_598_);
return v___x_599_;
}
else
{
lean_object* v___x_600_; 
v___x_600_ = l_Lean_Elab_Command_getScope___redArg(v___y_503_);
if (lean_obj_tag(v___x_600_) == 0)
{
lean_object* v_a_601_; lean_object* v_currNamespace_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v_a_601_ = lean_ctor_get(v___x_600_, 0);
lean_inc(v_a_601_);
lean_dec_ref_known(v___x_600_, 1);
v_currNamespace_602_ = lean_ctor_get(v_a_601_, 2);
lean_inc(v_currNamespace_602_);
lean_dec(v_a_601_);
v___x_603_ = l_Lean_Syntax_getArg(v___y_597_, v___y_594_);
v___x_604_ = l_Lean_Syntax_getId(v___x_603_);
lean_dec(v___x_603_);
v___x_605_ = l_Lean_Name_components(v___x_604_);
if (lean_obj_tag(v___x_605_) == 1)
{
lean_object* v_head_606_; 
v_head_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_head_606_);
if (lean_obj_tag(v_head_606_) == 1)
{
lean_object* v_pre_607_; 
v_pre_607_ = lean_ctor_get(v_head_606_, 0);
if (lean_obj_tag(v_pre_607_) == 0)
{
lean_object* v_tail_608_; lean_object* v_str_609_; lean_object* v___x_610_; uint8_t v___x_611_; 
v_tail_608_ = lean_ctor_get(v___x_605_, 1);
lean_inc(v_tail_608_);
v_str_609_ = lean_ctor_get(v_head_606_, 1);
lean_inc_ref(v_str_609_);
lean_dec_ref_known(v_head_606_, 2);
v___x_610_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__14));
v___x_611_ = lean_string_dec_eq(v_str_609_, v___x_610_);
lean_dec_ref(v_str_609_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; 
lean_dec(v_tail_608_);
v___x_612_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(v___y_597_, v___y_594_, v_currNamespace_602_, v___x_605_);
lean_dec_ref_known(v___x_605_, 2);
v___y_568_ = v___y_594_;
v___y_569_ = v___y_595_;
v___y_570_ = v___y_597_;
v___y_571_ = v___y_596_;
v___y_572_ = v___x_612_;
goto v___jp_567_;
}
else
{
lean_object* v___x_613_; lean_object* v___x_614_; 
lean_dec_ref_known(v___x_605_, 2);
lean_dec(v_currNamespace_602_);
v___x_613_ = lean_box(0);
v___x_614_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__4(v___x_613_, v_tail_608_);
v___y_568_ = v___y_594_;
v___y_569_ = v___y_595_;
v___y_570_ = v___y_597_;
v___y_571_ = v___y_596_;
v___y_572_ = v___x_614_;
goto v___jp_567_;
}
}
else
{
lean_object* v___x_615_; 
lean_dec_ref_known(v_head_606_, 2);
v___x_615_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(v___y_597_, v___y_594_, v_currNamespace_602_, v___x_605_);
lean_dec_ref_known(v___x_605_, 2);
v___y_568_ = v___y_594_;
v___y_569_ = v___y_595_;
v___y_570_ = v___y_597_;
v___y_571_ = v___y_596_;
v___y_572_ = v___x_615_;
goto v___jp_567_;
}
}
else
{
lean_object* v___x_616_; 
lean_dec(v_head_606_);
v___x_616_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(v___y_597_, v___y_594_, v_currNamespace_602_, v___x_605_);
lean_dec_ref_known(v___x_605_, 2);
v___y_568_ = v___y_594_;
v___y_569_ = v___y_595_;
v___y_570_ = v___y_597_;
v___y_571_ = v___y_596_;
v___y_572_ = v___x_616_;
goto v___jp_567_;
}
}
else
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__2(v___y_597_, v___y_594_, v_currNamespace_602_, v___x_605_);
lean_dec(v___x_605_);
v___y_568_ = v___y_594_;
v___y_569_ = v___y_595_;
v___y_570_ = v___y_597_;
v___y_571_ = v___y_596_;
v___y_572_ = v___x_617_;
goto v___jp_567_;
}
}
else
{
lean_object* v_a_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
lean_dec(v___y_597_);
lean_dec(v___y_595_);
lean_dec(v___y_594_);
lean_del_object(v___x_508_);
v_a_618_ = lean_ctor_get(v___x_600_, 0);
v_isSharedCheck_625_ = !lean_is_exclusive(v___x_600_);
if (v_isSharedCheck_625_ == 0)
{
v___x_620_ = v___x_600_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_a_618_);
lean_dec(v___x_600_);
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
v_reuseFailAlloc_624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_a_618_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
}
}
v___jp_626_:
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; uint8_t v___x_633_; 
v___x_627_ = lean_unsigned_to_nat(0u);
v___x_628_ = l_Lean_Syntax_getArg(v_stx_501_, v___x_627_);
v___x_629_ = l_Lean_Syntax_getArg(v___x_628_, v___x_627_);
lean_dec(v___x_628_);
v___x_630_ = lean_unsigned_to_nat(1u);
v___x_631_ = l_Lean_Syntax_getArg(v_stx_501_, v___x_630_);
lean_dec(v_stx_501_);
v___x_632_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___closed__16));
lean_inc(v___x_631_);
v___x_633_ = l_Lean_Syntax_isOfKind(v___x_631_, v___x_632_);
if (v___x_633_ == 0)
{
lean_object* v___x_634_; 
v___x_634_ = l_Lean_Syntax_getArg(v___x_631_, v___x_630_);
lean_dec(v___x_631_);
v___y_594_ = v___x_627_;
v___y_595_ = v___x_629_;
v___y_596_ = v___x_630_;
v___y_597_ = v___x_634_;
goto v___jp_593_;
}
else
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = lean_unsigned_to_nat(3u);
v___x_636_ = l_Lean_Syntax_getArg(v___x_631_, v___x_635_);
lean_dec(v___x_631_);
v___x_637_ = l_Lean_Syntax_getArg(v___x_636_, v___x_627_);
lean_dec(v___x_636_);
v___y_594_ = v___x_627_;
v___y_595_ = v___x_629_;
v___y_596_ = v___x_630_;
v___y_597_ = v___x_637_;
goto v___jp_593_;
}
}
}
else
{
lean_object* v___x_657_; lean_object* v___x_659_; 
lean_dec(v_stx_501_);
v___x_657_ = lean_box(0);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v___x_657_);
v___x_659_ = v___x_508_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v___x_657_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3___boxed(lean_object* v_stx_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter___lam__3(v_stx_662_, v___y_663_, v___y_664_);
lean_dec(v___y_664_);
lean_dec_ref(v___y_663_);
return v_res_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0(lean_object* v_o_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___redArg(v_o_708_, v___y_710_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0___boxed(lean_object* v_o_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_){
_start:
{
lean_object* v_res_717_; 
v_res_717_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__0_spec__0(v_o_713_, v___y_714_, v___y_715_);
lean_dec(v___y_715_);
lean_dec_ref(v___y_714_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7(lean_object* v_msgData_718_, lean_object* v___y_719_, lean_object* v___y_720_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___redArg(v_msgData_718_, v___y_720_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7___boxed(lean_object* v_msgData_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_){
_start:
{
lean_object* v_res_727_; 
v_res_727_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter_spec__2_spec__3_spec__4_spec__7(v_msgData_723_, v___y_724_, v___y_725_);
lean_dec(v___y_725_);
lean_dec_ref(v___y_724_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_729_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_docPrimeLinter));
v___x_730_ = l_Lean_Elab_Command_addLinter(v___x_729_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2____boxed(lean_object* v_a_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2_();
return v_res_732_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Command(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(uint8_t builtin) {
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
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_380224307____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_docPrime = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_docPrime);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocPrime_0__Mathlib_Linter_DocPrime_initFn_00___x40_Mathlib_Tactic_Linter_DocPrime_856601495____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(uint8_t builtin) {
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
res = initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(builtin);
}
#ifdef __cplusplus
}
#endif
