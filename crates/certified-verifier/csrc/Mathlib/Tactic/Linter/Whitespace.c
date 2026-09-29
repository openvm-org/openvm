// Lean compiler output
// Module: Mathlib.Tactic.Linter.Whitespace
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Linter.Header
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
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
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
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Syntax_instHashableRange_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
extern lean_object* l_Lean_Syntax_instInhabitedRange_default;
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_SourceInfo_getRangeWithTrailing_x3f(uint8_t, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prevn(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_String_Slice_Pos_nextn(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_substring_tostring(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
lean_object* l_String_Slice_positions(lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getSubstring_x3f(lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* l_Lean_Syntax_getHead_x3f(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "whitespace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(143, 177, 71, 245, 187, 208, 143, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "enable the whitespace linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 86, 179, 36, 77, 118, 133, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_whitespace;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "verbose"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(143, 177, 71, 245, 187, 208, 143, 117)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(39, 38, 165, 117, 183, 197, 47, 103)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "report diagnostic information for the `whitespace` linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 86, 179, 36, 77, 118, 133, 202)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(236, 52, 21, 245, 234, 187, 5, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(117, 34, 246, 137, 114, 183, 220, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__6_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 246}, .m_size = 2, .m_capacity = 2, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "inductive"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(167, 178, 72, 69, 244, 64, 6, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__3(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__1_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__3, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "unreachable\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__4_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__5_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "variable"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__7_value),LEAN_SCALAR_PTR_LITERAL(250, 93, 226, 106, 76, 14, 69, 165)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "omit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__9_value),LEAN_SCALAR_PTR_LITERAL(248, 151, 249, 80, 160, 104, 42, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default = (const lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instInhabitedFormatError = (const lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "srcNat: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", srcPos: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", fmtPos: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = ", msg: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", length: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "extra space"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "remove line break"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "missing space"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Oh no! (Unreachable\?)"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "-/"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "--"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "/--"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "term{_:_//_}"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 133, 82, 74, 101, 189, 164, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term{_}"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 26, 220, 95, 138, 254, 219, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term{}"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__4_value),LEAN_SCALAR_PTR_LITERAL(44, 141, 217, 101, 193, 131, 35, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "setBuilder"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__6_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__7_value),LEAN_SCALAR_PTR_LITERAL(55, 252, 174, 2, 80, 49, 173, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__9_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_::_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__11_value),LEAN_SCALAR_PTR_LITERAL(20, 118, 107, 117, 215, 167, 224, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 6, .m_data = "term¬_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__13_value),LEAN_SCALAR_PTR_LITERAL(222, 122, 71, 36, 131, 84, 176, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__15_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "superscriptTerm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__18_value),LEAN_SCALAR_PTR_LITERAL(161, 106, 247, 168, 24, 38, 154, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "subscript"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__20_value),LEAN_SCALAR_PTR_LITERAL(45, 40, 231, 27, 100, 155, 214, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Bundle"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 7, .m_data = "termπ__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__22_value),LEAN_SCALAR_PTR_LITERAL(134, 3, 224, 131, 59, 212, 87, 196)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__23_value),LEAN_SCALAR_PTR_LITERAL(187, 19, 243, 230, 173, 110, 180, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_#_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__25_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__26_value),LEAN_SCALAR_PTR_LITERAL(232, 73, 131, 87, 161, 252, 182, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__28_value),LEAN_SCALAR_PTR_LITERAL(44, 76, 179, 33, 27, 4, 201, 125)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "doubleQuotedName"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__30_value),LEAN_SCALAR_PTR_LITERAL(194, 121, 78, 150, 98, 156, 35, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*14, .m_other = 0, .m_tag = 246}, .m_size = 14, .m_capacity = 14, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__21_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__27_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__31_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__32_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__32_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "where"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "where"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Array_map__unattach_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Array_map__unattach_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__2_value)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "runCmd"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 158, 215, 209, 131, 110, 142, 142)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "macro_rules"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(125, 80, 75, 5, 165, 87, 197, 1)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = " in the source\n\nThis part of the code\n  '"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "'\nshould be written as\n  '"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "'\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Formatted string:\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "\nOriginal string:\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Oh no"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "This should not have happened: please report this issue!"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 96, .m_capacity = 96, .m_length = 95, .m_data = "The `whitespace` linter had some parsing issues: feel free to silence it and report this error!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__4_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "' starts on column "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = ", but all commands should start at the beginning of the line."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__3_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__17_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Whitespace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 3, 190, 147, 197, 36, 103, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(118, 145, 182, 81, 193, 218, 163, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(207, 66, 243, 211, 220, 12, 106, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(53, 172, 189, 16, 232, 92, 247, 80)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__15_value),LEAN_SCALAR_PTR_LITERAL(33, 214, 80, 48, 195, 24, 217, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__10_value),LEAN_SCALAR_PTR_LITERAL(55, 223, 106, 70, 47, 90, 142, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "whitespaceLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__18_value),LEAN_SCALAR_PTR_LITERAL(224, 98, 177, 37, 38, 126, 198, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_));
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_));
v___x_59_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0(v___x_56_, v___x_57_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4____boxed(lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_();
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_82_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_));
v___x_83_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_));
v___x_84_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_));
v___x_85_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4__spec__0(v___x_82_, v___x_83_, v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4____boxed(lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_();
return v_res_87_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0(lean_object* v_a_88_, lean_object* v_as_89_, size_t v_i_90_, size_t v_stop_91_){
_start:
{
uint8_t v___x_92_; 
v___x_92_ = lean_usize_dec_eq(v_i_90_, v_stop_91_);
if (v___x_92_ == 0)
{
lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_93_ = lean_array_uget_borrowed(v_as_89_, v_i_90_);
v___x_94_ = lean_name_eq(v_a_88_, v___x_93_);
if (v___x_94_ == 0)
{
size_t v___x_95_; size_t v___x_96_; 
v___x_95_ = ((size_t)1ULL);
v___x_96_ = lean_usize_add(v_i_90_, v___x_95_);
v_i_90_ = v___x_96_;
goto _start;
}
else
{
return v___x_94_;
}
}
else
{
uint8_t v___x_98_; 
v___x_98_ = 0;
return v___x_98_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0___boxed(lean_object* v_a_99_, lean_object* v_as_100_, lean_object* v_i_101_, lean_object* v_stop_102_){
_start:
{
size_t v_i_boxed_103_; size_t v_stop_boxed_104_; uint8_t v_res_105_; lean_object* v_r_106_; 
v_i_boxed_103_ = lean_unbox_usize(v_i_101_);
lean_dec(v_i_101_);
v_stop_boxed_104_ = lean_unbox_usize(v_stop_102_);
lean_dec(v_stop_102_);
v_res_105_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0(v_a_99_, v_as_100_, v_i_boxed_103_, v_stop_boxed_104_);
lean_dec_ref(v_as_100_);
lean_dec(v_a_99_);
v_r_106_ = lean_box(v_res_105_);
return v_r_106_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0(lean_object* v_as_107_, lean_object* v_a_108_){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_109_ = lean_unsigned_to_nat(0u);
v___x_110_ = lean_array_get_size(v_as_107_);
v___x_111_ = lean_nat_dec_lt(v___x_109_, v___x_110_);
if (v___x_111_ == 0)
{
return v___x_111_;
}
else
{
if (v___x_111_ == 0)
{
return v___x_111_;
}
else
{
size_t v___x_112_; size_t v___x_113_; uint8_t v___x_114_; 
v___x_112_ = ((size_t)0ULL);
v___x_113_ = lean_usize_of_nat(v___x_110_);
v___x_114_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0_spec__0(v_a_108_, v_as_107_, v___x_112_, v___x_113_);
return v___x_114_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0___boxed(lean_object* v_as_115_, lean_object* v_a_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0(v_as_115_, v_a_116_);
lean_dec(v_a_116_);
lean_dec_ref(v_as_115_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0(lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_138_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___closed__7));
v___x_139_ = l_Lean_Syntax_getKind(v_x_137_);
v___x_140_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0(v___x_138_, v___x_139_);
lean_dec(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0___boxed(lean_object* v_x_141_){
_start:
{
uint8_t v_res_142_; lean_object* v_r_143_; 
v_res_142_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__0(v_x_141_);
v_r_143_ = lean_box(v_res_142_);
return v_r_143_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1(lean_object* v_x_150_){
_start:
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___closed__1));
v___x_152_ = l_Lean_Syntax_isOfKind(v_x_150_, v___x_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1___boxed(lean_object* v_x_153_){
_start:
{
uint8_t v_res_154_; lean_object* v_r_155_; 
v_res_154_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__1(v_x_153_);
v_r_155_ = lean_box(v_res_154_);
return v_r_155_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2(lean_object* v_x_162_){
_start:
{
lean_object* v___x_163_; uint8_t v___x_164_; 
v___x_163_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___closed__1));
v___x_164_ = l_Lean_Syntax_isOfKind(v_x_162_, v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2___boxed(lean_object* v_x_165_){
_start:
{
uint8_t v_res_166_; lean_object* v_r_167_; 
v_res_166_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__2(v_x_165_);
v_r_167_ = lean_box(v_res_166_);
return v_r_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__3(lean_object* v_x_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lean_box(0);
return v___x_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4(lean_object* v_x_177_){
_start:
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___closed__2));
v___x_179_ = l_Lean_Syntax_isOfKind(v_x_177_, v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4___boxed(lean_object* v_x_180_){
_start:
{
uint8_t v_res_181_; lean_object* v_r_182_; 
v_res_181_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__4(v_x_180_);
v_r_182_ = lean_box(v_res_181_);
return v_r_182_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5(lean_object* v_x_189_){
_start:
{
lean_object* v___x_190_; uint8_t v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___closed__1));
v___x_191_ = l_Lean_Syntax_isOfKind(v_x_189_, v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5___boxed(lean_object* v_x_192_){
_start:
{
uint8_t v_res_193_; lean_object* v_r_194_; 
v_res_193_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___lam__5(v_x_192_);
v_r_194_ = lean_box(v_res_193_);
return v_r_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos(lean_object* v_stx_214_){
_start:
{
uint8_t v___y_216_; lean_object* v___f_220_; lean_object* v___x_221_; 
v___f_220_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__0));
lean_inc(v_stx_214_);
v___x_221_ = l_Lean_Syntax_find_x3f(v_stx_214_, v___f_220_);
if (lean_obj_tag(v___x_221_) == 1)
{
lean_object* v_val_222_; lean_object* v___f_223_; lean_object* v___x_224_; 
lean_dec(v_stx_214_);
v_val_222_ = lean_ctor_get(v___x_221_, 0);
lean_inc_n(v_val_222_, 2);
lean_dec_ref_known(v___x_221_, 1);
v___f_223_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__1));
v___x_224_ = l_Lean_Syntax_find_x3f(v_val_222_, v___f_223_);
if (lean_obj_tag(v___x_224_) == 1)
{
lean_object* v_val_225_; lean_object* v___f_226_; lean_object* v___x_227_; 
lean_dec(v_val_222_);
v_val_225_ = lean_ctor_get(v___x_224_, 0);
lean_inc(v_val_225_);
lean_dec_ref_known(v___x_224_, 1);
v___f_226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__2));
v___x_227_ = l_Lean_Syntax_find_x3f(v_val_225_, v___f_226_);
if (lean_obj_tag(v___x_227_) == 0)
{
lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___f_228_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__3));
v___x_229_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__4));
v___x_230_ = lean_dbg_trace(v___x_229_, v___f_228_);
return v___x_230_;
}
else
{
lean_object* v_val_231_; uint8_t v___x_232_; lean_object* v___x_233_; 
v_val_231_ = lean_ctor_get(v___x_227_, 0);
lean_inc(v_val_231_);
lean_dec_ref_known(v___x_227_, 1);
v___x_232_ = 0;
v___x_233_ = l_Lean_Syntax_getTailPos_x3f(v_val_231_, v___x_232_);
lean_dec(v_val_231_);
return v___x_233_;
}
}
else
{
lean_object* v___f_234_; lean_object* v___x_235_; 
lean_dec(v___x_224_);
v___f_234_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__5));
lean_inc(v_val_222_);
v___x_235_ = l_Lean_Syntax_find_x3f(v_val_222_, v___f_234_);
if (lean_obj_tag(v___x_235_) == 0)
{
lean_object* v___f_236_; lean_object* v___x_237_; 
v___f_236_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__6));
v___x_237_ = l_Lean_Syntax_find_x3f(v_val_222_, v___f_236_);
if (lean_obj_tag(v___x_237_) == 0)
{
lean_object* v___x_238_; 
v___x_238_ = lean_box(0);
return v___x_238_;
}
else
{
lean_object* v_val_239_; uint8_t v___x_240_; lean_object* v___x_241_; 
v_val_239_ = lean_ctor_get(v___x_237_, 0);
lean_inc(v_val_239_);
lean_dec_ref_known(v___x_237_, 1);
v___x_240_ = 0;
v___x_241_ = l_Lean_Syntax_getPos_x3f(v_val_239_, v___x_240_);
lean_dec(v_val_239_);
return v___x_241_;
}
}
else
{
lean_object* v_val_242_; lean_object* v___x_243_; lean_object* v___x_244_; uint8_t v___x_245_; lean_object* v___x_246_; 
lean_dec(v_val_222_);
v_val_242_ = lean_ctor_get(v___x_235_, 0);
lean_inc(v_val_242_);
lean_dec_ref_known(v___x_235_, 1);
v___x_243_ = lean_unsigned_to_nat(0u);
v___x_244_ = l_Lean_Syntax_getArg(v_val_242_, v___x_243_);
lean_dec(v_val_242_);
v___x_245_ = 0;
v___x_246_ = l_Lean_Syntax_getTailPos_x3f(v___x_244_, v___x_245_);
lean_dec(v___x_244_);
return v___x_246_;
}
}
}
else
{
lean_object* v___x_247_; uint8_t v___x_248_; 
lean_dec(v___x_221_);
v___x_247_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__8));
lean_inc(v_stx_214_);
v___x_248_ = l_Lean_Syntax_isOfKind(v_stx_214_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos___closed__10));
lean_inc(v_stx_214_);
v___x_250_ = l_Lean_Syntax_isOfKind(v_stx_214_, v___x_249_);
v___y_216_ = v___x_250_;
goto v___jp_215_;
}
else
{
v___y_216_ = v___x_248_;
goto v___jp_215_;
}
}
v___jp_215_:
{
if (v___y_216_ == 0)
{
lean_object* v___x_217_; 
lean_dec(v_stx_214_);
v___x_217_ = lean_box(0);
return v___x_217_;
}
else
{
uint8_t v___x_218_; lean_object* v___x_219_; 
v___x_218_ = 0;
v___x_219_ = l_Lean_Syntax_getTailPos_x3f(v_stx_214_, v___x_218_);
lean_dec(v_stx_214_);
return v___x_219_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0(lean_object* v_f_263_){
_start:
{
lean_object* v_srcNat_264_; lean_object* v_srcEndPos_265_; lean_object* v_fmtPos_266_; lean_object* v_msg_267_; lean_object* v_length_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v_srcNat_264_ = lean_ctor_get(v_f_263_, 0);
lean_inc(v_srcNat_264_);
v_srcEndPos_265_ = lean_ctor_get(v_f_263_, 1);
lean_inc(v_srcEndPos_265_);
v_fmtPos_266_ = lean_ctor_get(v_f_263_, 2);
lean_inc(v_fmtPos_266_);
v_msg_267_ = lean_ctor_get(v_f_263_, 3);
lean_inc_ref(v_msg_267_);
v_length_268_ = lean_ctor_get(v_f_263_, 4);
lean_inc(v_length_268_);
lean_dec_ref(v_f_263_);
v___x_269_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__0));
v___x_270_ = l_Nat_reprFast(v_srcNat_264_);
v___x_271_ = lean_string_append(v___x_269_, v___x_270_);
lean_dec_ref(v___x_270_);
v___x_272_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__1));
v___x_273_ = lean_string_append(v___x_271_, v___x_272_);
v___x_274_ = l_Nat_reprFast(v_srcEndPos_265_);
v___x_275_ = lean_string_append(v___x_273_, v___x_274_);
lean_dec_ref(v___x_274_);
v___x_276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__2));
v___x_277_ = lean_string_append(v___x_275_, v___x_276_);
v___x_278_ = l_Nat_reprFast(v_fmtPos_266_);
v___x_279_ = lean_string_append(v___x_277_, v___x_278_);
lean_dec_ref(v___x_278_);
v___x_280_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__3));
v___x_281_ = lean_string_append(v___x_279_, v___x_280_);
v___x_282_ = lean_string_append(v___x_281_, v_msg_267_);
lean_dec_ref(v_msg_267_);
v___x_283_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__4));
v___x_284_ = lean_string_append(v___x_282_, v___x_283_);
v___x_285_ = l_Nat_reprFast(v_length_268_);
v___x_286_ = lean_string_append(v___x_284_, v___x_285_);
lean_dec_ref(v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_instToStringFormatError___lam__0___closed__5));
v___x_288_ = lean_string_append(v___x_286_, v___x_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(lean_object* v_ls_291_, lean_object* v_ms_292_, lean_object* v_msg_293_, lean_object* v_length_294_){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_295_ = lean_string_length(v_ls_291_);
v___x_296_ = lean_string_utf8_byte_size(v_ls_291_);
v___x_297_ = lean_string_length(v_ms_292_);
v___x_298_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_298_, 0, v___x_295_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
lean_ctor_set(v___x_298_, 2, v___x_297_);
lean_ctor_set(v___x_298_, 3, v_msg_293_);
lean_ctor_set(v___x_298_, 4, v_length_294_);
lean_ctor_set(v___x_298_, 5, v___x_296_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError___boxed(lean_object* v_ls_299_, lean_object* v_ms_300_, lean_object* v_msg_301_, lean_object* v_length_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(v_ls_299_, v_ms_300_, v_msg_301_, v_length_302_);
lean_dec_ref(v_ms_300_);
lean_dec_ref(v_ls_299_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(lean_object* v_fs_304_, lean_object* v_f_305_){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; uint8_t v___x_308_; 
v___x_306_ = lean_array_get_size(v_fs_304_);
v___x_307_ = lean_unsigned_to_nat(0u);
v___x_308_ = lean_nat_dec_eq(v___x_306_, v___x_307_);
if (v___x_308_ == 0)
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v_back_312_; lean_object* v_srcNat_313_; lean_object* v_srcEndPos_314_; lean_object* v_fmtPos_315_; lean_object* v_msg_316_; lean_object* v_length_317_; lean_object* v_srcNat_318_; lean_object* v_srcEndPos_319_; lean_object* v_msg_320_; lean_object* v_length_321_; uint8_t v___x_322_; 
v___x_309_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default));
v___x_310_ = lean_unsigned_to_nat(1u);
v___x_311_ = lean_nat_sub(v___x_306_, v___x_310_);
v_back_312_ = lean_array_get_borrowed(v___x_309_, v_fs_304_, v___x_311_);
lean_dec(v___x_311_);
v_srcNat_313_ = lean_ctor_get(v_back_312_, 0);
v_srcEndPos_314_ = lean_ctor_get(v_back_312_, 1);
v_fmtPos_315_ = lean_ctor_get(v_back_312_, 2);
v_msg_316_ = lean_ctor_get(v_back_312_, 3);
v_length_317_ = lean_ctor_get(v_back_312_, 4);
v_srcNat_318_ = lean_ctor_get(v_f_305_, 0);
v_srcEndPos_319_ = lean_ctor_get(v_f_305_, 1);
v_msg_320_ = lean_ctor_get(v_f_305_, 3);
v_length_321_ = lean_ctor_get(v_f_305_, 4);
v___x_322_ = lean_string_dec_eq(v_msg_316_, v_msg_320_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; 
v___x_323_ = lean_array_push(v_fs_304_, v_f_305_);
return v___x_323_;
}
else
{
if (v___x_308_ == 0)
{
lean_object* v___x_324_; uint8_t v___x_325_; 
v___x_324_ = lean_nat_sub(v_srcNat_313_, v_length_317_);
v___x_325_ = lean_nat_dec_eq(v___x_324_, v_srcNat_318_);
lean_dec(v___x_324_);
if (v___x_325_ == 0)
{
lean_object* v___x_326_; 
v___x_326_ = lean_array_push(v_fs_304_, v_f_305_);
return v___x_326_;
}
else
{
lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_336_; 
lean_inc(v_length_321_);
lean_inc(v_srcEndPos_319_);
lean_inc(v_length_317_);
lean_inc_ref(v_msg_316_);
lean_inc(v_fmtPos_315_);
lean_inc(v_srcEndPos_314_);
lean_inc(v_srcNat_313_);
v_isSharedCheck_336_ = !lean_is_exclusive(v_f_305_);
if (v_isSharedCheck_336_ == 0)
{
lean_object* v_unused_337_; lean_object* v_unused_338_; lean_object* v_unused_339_; lean_object* v_unused_340_; lean_object* v_unused_341_; lean_object* v_unused_342_; 
v_unused_337_ = lean_ctor_get(v_f_305_, 5);
lean_dec(v_unused_337_);
v_unused_338_ = lean_ctor_get(v_f_305_, 4);
lean_dec(v_unused_338_);
v_unused_339_ = lean_ctor_get(v_f_305_, 3);
lean_dec(v_unused_339_);
v_unused_340_ = lean_ctor_get(v_f_305_, 2);
lean_dec(v_unused_340_);
v_unused_341_ = lean_ctor_get(v_f_305_, 1);
lean_dec(v_unused_341_);
v_unused_342_ = lean_ctor_get(v_f_305_, 0);
lean_dec(v_unused_342_);
v___x_328_ = v_f_305_;
v_isShared_329_ = v_isSharedCheck_336_;
goto v_resetjp_327_;
}
else
{
lean_dec(v_f_305_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_336_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_333_; 
v___x_330_ = lean_array_pop(v_fs_304_);
v___x_331_ = lean_nat_add(v_length_317_, v_length_321_);
lean_dec(v_length_321_);
lean_dec(v_length_317_);
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 5, v_srcEndPos_319_);
lean_ctor_set(v___x_328_, 4, v___x_331_);
lean_ctor_set(v___x_328_, 3, v_msg_316_);
lean_ctor_set(v___x_328_, 2, v_fmtPos_315_);
lean_ctor_set(v___x_328_, 1, v_srcEndPos_314_);
lean_ctor_set(v___x_328_, 0, v_srcNat_313_);
v___x_333_ = v___x_328_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_srcNat_313_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v_srcEndPos_314_);
lean_ctor_set(v_reuseFailAlloc_335_, 2, v_fmtPos_315_);
lean_ctor_set(v_reuseFailAlloc_335_, 3, v_msg_316_);
lean_ctor_set(v_reuseFailAlloc_335_, 4, v___x_331_);
lean_ctor_set(v_reuseFailAlloc_335_, 5, v_srcEndPos_319_);
v___x_333_ = v_reuseFailAlloc_335_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
lean_object* v___x_334_; 
v___x_334_ = lean_array_push(v___x_330_, v___x_333_);
return v___x_334_;
}
}
}
}
else
{
lean_object* v___x_343_; 
v___x_343_ = lean_array_push(v_fs_304_, v_f_305_);
return v___x_343_;
}
}
}
else
{
lean_object* v___x_344_; 
v___x_344_ = lean_array_push(v_fs_304_, v_f_305_);
return v___x_344_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1(uint8_t v___y_345_, lean_object* v___x_346_, lean_object* v_s_347_, lean_object* v_pos_348_){
_start:
{
lean_object* v_str_349_; lean_object* v_startInclusive_350_; lean_object* v_endExclusive_351_; lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___y_355_; lean_object* v___x_361_; uint8_t v___x_362_; 
v_str_349_ = lean_ctor_get(v_s_347_, 0);
v_startInclusive_350_ = lean_ctor_get(v_s_347_, 1);
v_endExclusive_351_ = lean_ctor_get(v_s_347_, 2);
v___x_352_ = lean_unsigned_to_nat(0u);
v___x_353_ = lean_nat_add(v_startInclusive_350_, v_pos_348_);
v___x_361_ = lean_nat_sub(v_endExclusive_351_, v___x_353_);
v___x_362_ = lean_nat_dec_eq(v___x_352_, v___x_361_);
lean_dec(v___x_361_);
if (v___x_362_ == 0)
{
uint32_t v___x_363_; uint32_t v___x_364_; uint8_t v___x_365_; 
v___x_363_ = lean_string_utf8_get_fast(v_str_349_, v___x_353_);
v___x_364_ = 45;
v___x_365_ = lean_uint32_dec_eq(v___x_363_, v___x_364_);
if (v___x_365_ == 0)
{
v___y_355_ = v___y_345_;
goto v___jp_354_;
}
else
{
uint8_t v___x_366_; 
v___x_366_ = lean_nat_dec_eq(v___x_346_, v___x_352_);
v___y_355_ = v___x_366_;
goto v___jp_354_;
}
}
else
{
lean_dec(v___x_353_);
return v_pos_348_;
}
v___jp_354_:
{
if (v___y_355_ == 0)
{
lean_dec(v___x_353_);
return v_pos_348_;
}
else
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; uint8_t v___x_359_; 
v___x_356_ = lean_string_utf8_next_fast(v_str_349_, v___x_353_);
v___x_357_ = lean_nat_sub(v___x_356_, v___x_353_);
lean_dec(v___x_353_);
v___x_358_ = lean_nat_add(v_pos_348_, v___x_357_);
lean_dec(v___x_357_);
v___x_359_ = lean_nat_dec_lt(v_pos_348_, v___x_358_);
if (v___x_359_ == 0)
{
lean_dec(v___x_358_);
return v_pos_348_;
}
else
{
lean_dec(v_pos_348_);
v_pos_348_ = v___x_358_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1___boxed(lean_object* v___y_367_, lean_object* v___x_368_, lean_object* v_s_369_, lean_object* v_pos_370_){
_start:
{
uint8_t v___y_6074__boxed_371_; lean_object* v_res_372_; 
v___y_6074__boxed_371_ = lean_unbox(v___y_367_);
v_res_372_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1(v___y_6074__boxed_371_, v___x_368_, v_s_369_, v_pos_370_);
lean_dec_ref(v_s_369_);
lean_dec(v___x_368_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg(lean_object* v_L_373_, lean_object* v_a_374_, lean_object* v_b_375_){
_start:
{
lean_object* v_str_376_; lean_object* v_startInclusive_377_; lean_object* v_endExclusive_378_; lean_object* v___x_379_; uint8_t v___x_380_; 
v_str_376_ = lean_ctor_get(v_L_373_, 0);
v_startInclusive_377_ = lean_ctor_get(v_L_373_, 1);
v_endExclusive_378_ = lean_ctor_get(v_L_373_, 2);
v___x_379_ = lean_nat_sub(v_endExclusive_378_, v_startInclusive_377_);
v___x_380_ = lean_nat_dec_eq(v_a_374_, v___x_379_);
lean_dec(v___x_379_);
if (v___x_380_ == 0)
{
lean_object* v_snd_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_403_; 
v_snd_381_ = lean_ctor_get(v_b_375_, 1);
v_isSharedCheck_403_ = !lean_is_exclusive(v_b_375_);
if (v_isSharedCheck_403_ == 0)
{
lean_object* v_unused_404_; 
v_unused_404_ = lean_ctor_get(v_b_375_, 0);
lean_dec(v_unused_404_);
v___x_383_ = v_b_375_;
v_isShared_384_ = v_isSharedCheck_403_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_snd_381_);
lean_dec(v_b_375_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_403_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
lean_object* v___x_385_; uint32_t v___x_386_; uint32_t v___x_387_; uint8_t v___x_388_; 
v___x_385_ = lean_nat_add(v_startInclusive_377_, v_a_374_);
v___x_386_ = lean_string_utf8_get_fast(v_str_376_, v___x_385_);
v___x_387_ = 10;
v___x_388_ = lean_uint32_dec_eq(v___x_386_, v___x_387_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_395_; 
lean_dec(v_a_374_);
v___x_389_ = lean_box(0);
v___x_390_ = lean_string_utf8_next_fast(v_str_376_, v___x_385_);
lean_dec(v___x_385_);
v___x_391_ = lean_nat_sub(v___x_390_, v_startInclusive_377_);
v___x_392_ = lean_unsigned_to_nat(1u);
v___x_393_ = lean_nat_add(v_snd_381_, v___x_392_);
lean_dec(v_snd_381_);
if (v_isShared_384_ == 0)
{
lean_ctor_set(v___x_383_, 1, v___x_393_);
lean_ctor_set(v___x_383_, 0, v___x_389_);
v___x_395_ = v___x_383_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v___x_389_);
lean_ctor_set(v_reuseFailAlloc_397_, 1, v___x_393_);
v___x_395_ = v_reuseFailAlloc_397_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
v_a_374_ = v___x_391_;
v_b_375_ = v___x_395_;
goto _start;
}
}
else
{
lean_object* v___x_399_; 
lean_dec(v___x_385_);
lean_inc(v_snd_381_);
if (v_isShared_384_ == 0)
{
lean_ctor_set(v___x_383_, 0, v_a_374_);
v___x_399_ = v___x_383_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_a_374_);
lean_ctor_set(v_reuseFailAlloc_402_, 1, v_snd_381_);
v___x_399_ = v_reuseFailAlloc_402_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
v___x_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_400_);
lean_ctor_set(v___x_401_, 1, v_snd_381_);
return v___x_401_;
}
}
}
}
else
{
lean_dec(v_a_374_);
return v_b_375_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg___boxed(lean_object* v_L_405_, lean_object* v_a_406_, lean_object* v_b_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg(v_L_405_, v_a_406_, v_b_407_);
lean_dec_ref(v_L_405_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(lean_object* v_s_409_, lean_object* v_pos_410_){
_start:
{
lean_object* v_str_411_; lean_object* v_startInclusive_412_; lean_object* v_endExclusive_413_; lean_object* v___x_414_; uint8_t v___y_422_; lean_object* v___x_423_; lean_object* v___x_424_; uint8_t v___x_425_; 
v_str_411_ = lean_ctor_get(v_s_409_, 0);
v_startInclusive_412_ = lean_ctor_get(v_s_409_, 1);
v_endExclusive_413_ = lean_ctor_get(v_s_409_, 2);
v___x_414_ = lean_nat_add(v_startInclusive_412_, v_pos_410_);
v___x_423_ = lean_unsigned_to_nat(0u);
v___x_424_ = lean_nat_sub(v_endExclusive_413_, v___x_414_);
v___x_425_ = lean_nat_dec_eq(v___x_423_, v___x_424_);
lean_dec(v___x_424_);
if (v___x_425_ == 0)
{
uint32_t v___x_426_; uint8_t v___y_428_; uint32_t v___x_433_; uint8_t v___x_434_; 
v___x_426_ = lean_string_utf8_get_fast(v_str_411_, v___x_414_);
v___x_433_ = 32;
v___x_434_ = lean_uint32_dec_eq(v___x_426_, v___x_433_);
if (v___x_434_ == 0)
{
uint32_t v___x_435_; uint8_t v___x_436_; 
v___x_435_ = 9;
v___x_436_ = lean_uint32_dec_eq(v___x_426_, v___x_435_);
v___y_428_ = v___x_436_;
goto v___jp_427_;
}
else
{
v___y_428_ = v___x_434_;
goto v___jp_427_;
}
v___jp_427_:
{
if (v___y_428_ == 0)
{
uint32_t v___x_429_; uint8_t v___x_430_; 
v___x_429_ = 13;
v___x_430_ = lean_uint32_dec_eq(v___x_426_, v___x_429_);
if (v___x_430_ == 0)
{
uint32_t v___x_431_; uint8_t v___x_432_; 
v___x_431_ = 10;
v___x_432_ = lean_uint32_dec_eq(v___x_426_, v___x_431_);
v___y_422_ = v___x_432_;
goto v___jp_421_;
}
else
{
v___y_422_ = v___x_430_;
goto v___jp_421_;
}
}
else
{
goto v___jp_415_;
}
}
}
else
{
lean_dec(v___x_414_);
return v_pos_410_;
}
v___jp_415_:
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; uint8_t v___x_419_; 
v___x_416_ = lean_string_utf8_next_fast(v_str_411_, v___x_414_);
v___x_417_ = lean_nat_sub(v___x_416_, v___x_414_);
lean_dec(v___x_414_);
v___x_418_ = lean_nat_add(v_pos_410_, v___x_417_);
lean_dec(v___x_417_);
v___x_419_ = lean_nat_dec_lt(v_pos_410_, v___x_418_);
if (v___x_419_ == 0)
{
lean_dec(v___x_418_);
return v_pos_410_;
}
else
{
lean_dec(v_pos_410_);
v_pos_410_ = v___x_418_;
goto _start;
}
}
v___jp_421_:
{
if (v___y_422_ == 0)
{
lean_dec(v___x_414_);
return v_pos_410_;
}
else
{
goto v___jp_415_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0___boxed(lean_object* v_s_437_, lean_object* v_pos_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v_s_437_, v_pos_438_);
lean_dec_ref(v_s_437_);
return v_res_439_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5(void){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_445_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__4));
v___x_446_ = lean_string_utf8_byte_size(v___x_445_);
return v___x_446_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8(void){
_start:
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__7));
v___x_452_ = lean_string_utf8_byte_size(v___x_451_);
return v___x_452_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10(void){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_454_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__9));
v___x_455_ = lean_string_utf8_byte_size(v___x_454_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux(lean_object* v_as_456_, lean_object* v_L_457_, lean_object* v_M_458_){
_start:
{
lean_object* v___y_460_; lean_object* v___y_461_; lean_object* v___y_462_; lean_object* v___y_463_; lean_object* v___y_464_; lean_object* v___y_465_; lean_object* v___y_471_; lean_object* v___y_472_; lean_object* v___y_473_; lean_object* v___y_474_; lean_object* v___y_475_; lean_object* v___y_476_; lean_object* v___y_477_; lean_object* v___y_478_; lean_object* v___y_479_; lean_object* v___y_480_; lean_object* v___y_481_; uint8_t v___y_482_; lean_object* v___y_490_; lean_object* v___y_491_; lean_object* v___y_492_; lean_object* v___y_493_; lean_object* v___y_494_; uint32_t v___y_495_; lean_object* v___y_496_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___y_501_; uint8_t v___y_502_; lean_object* v___y_508_; lean_object* v___y_509_; lean_object* v___y_510_; lean_object* v___y_511_; lean_object* v___y_512_; lean_object* v___y_513_; lean_object* v___y_514_; lean_object* v___y_515_; lean_object* v___y_516_; lean_object* v___y_525_; lean_object* v___y_526_; lean_object* v___y_527_; lean_object* v___y_528_; lean_object* v___y_529_; lean_object* v___y_530_; lean_object* v___y_531_; lean_object* v___y_532_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_535_; lean_object* v___y_536_; uint8_t v___y_537_; lean_object* v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___y_558_; uint32_t v___y_559_; lean_object* v___y_560_; uint8_t v___y_561_; lean_object* v___y_567_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_575_; lean_object* v___y_576_; lean_object* v___y_587_; lean_object* v___y_588_; lean_object* v___y_589_; lean_object* v___y_590_; lean_object* v___y_591_; lean_object* v___y_592_; lean_object* v___y_593_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v___y_596_; lean_object* v___y_597_; uint8_t v___y_598_; lean_object* v___y_605_; lean_object* v___y_606_; lean_object* v___y_607_; lean_object* v___y_608_; lean_object* v___y_609_; uint32_t v___y_610_; lean_object* v___y_611_; lean_object* v___y_612_; lean_object* v___y_613_; lean_object* v___y_614_; lean_object* v___y_615_; lean_object* v___y_616_; uint8_t v___y_617_; lean_object* v___y_623_; lean_object* v___y_624_; lean_object* v___y_625_; lean_object* v___y_626_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v___y_629_; lean_object* v___y_630_; lean_object* v___y_631_; lean_object* v___y_632_; lean_object* v___y_633_; uint32_t v___y_634_; lean_object* v___y_635_; uint32_t v___y_636_; lean_object* v___y_653_; lean_object* v___y_654_; lean_object* v___y_655_; lean_object* v___y_656_; lean_object* v___y_657_; lean_object* v___y_658_; lean_object* v___y_659_; lean_object* v___y_660_; lean_object* v___y_661_; lean_object* v___y_662_; lean_object* v___y_663_; lean_object* v___y_664_; uint32_t v___y_665_; lean_object* v___x_670_; lean_object* v_startInclusive_671_; lean_object* v_endExclusive_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_820_; 
lean_inc_ref(v_M_458_);
v___x_670_ = l_String_Slice_trimAscii(v_M_458_);
v_startInclusive_671_ = lean_ctor_get(v___x_670_, 1);
v_endExclusive_672_ = lean_ctor_get(v___x_670_, 2);
v_isSharedCheck_820_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_820_ == 0)
{
lean_object* v_unused_821_; 
v_unused_821_ = lean_ctor_get(v___x_670_, 0);
lean_dec(v_unused_821_);
v___x_674_ = v___x_670_;
v_isShared_675_ = v_isSharedCheck_820_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_endExclusive_672_);
lean_inc(v_startInclusive_671_);
lean_dec(v___x_670_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_820_;
goto v_resetjp_673_;
}
v___jp_459_:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_466_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___y_460_, v___y_465_);
lean_dec_ref(v___y_460_);
v___x_467_ = lean_nat_add(v___y_464_, v___x_466_);
lean_dec(v___x_466_);
lean_dec(v___y_464_);
v___x_468_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_468_, 0, v___y_462_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
lean_ctor_set(v___x_468_, 2, v___y_461_);
v_L_457_ = v___y_463_;
v_M_458_ = v___x_468_;
goto _start;
}
v___jp_470_:
{
if (v___y_482_ == 0)
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
lean_dec(v___y_479_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_471_);
v___x_483_ = lean_string_utf8_extract_fast(v___y_481_, v___y_478_, v___y_480_);
lean_dec(v___y_480_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_481_);
v___x_484_ = lean_string_utf8_extract_fast(v___y_474_, v___y_472_, v___y_473_);
lean_dec(v___y_473_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_474_);
v___x_485_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__0));
v___x_486_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(v___x_483_, v___x_484_, v___x_485_, v___y_475_);
lean_dec_ref(v___x_484_);
lean_dec_ref(v___x_483_);
v___x_487_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(v_as_456_, v___x_486_);
v_as_456_ = v___x_487_;
v_L_457_ = v___y_477_;
goto _start;
}
else
{
lean_dec_ref(v___y_481_);
lean_dec(v___y_480_);
lean_dec(v___y_478_);
lean_dec(v___y_475_);
lean_dec(v___y_472_);
lean_dec_ref(v_M_458_);
v___y_460_ = v___y_471_;
v___y_461_ = v___y_473_;
v___y_462_ = v___y_474_;
v___y_463_ = v___y_477_;
v___y_464_ = v___y_476_;
v___y_465_ = v___y_479_;
goto v___jp_459_;
}
}
v___jp_489_:
{
if (v___y_502_ == 0)
{
uint32_t v___x_503_; uint8_t v___x_504_; 
v___x_503_ = 13;
v___x_504_ = lean_uint32_dec_eq(v___y_495_, v___x_503_);
if (v___x_504_ == 0)
{
uint32_t v___x_505_; uint8_t v___x_506_; 
v___x_505_ = 10;
v___x_506_ = lean_uint32_dec_eq(v___y_495_, v___x_505_);
v___y_471_ = v___y_491_;
v___y_472_ = v___y_490_;
v___y_473_ = v___y_492_;
v___y_474_ = v___y_493_;
v___y_475_ = v___y_494_;
v___y_476_ = v___y_497_;
v___y_477_ = v___y_496_;
v___y_478_ = v___y_498_;
v___y_479_ = v___y_499_;
v___y_480_ = v___y_501_;
v___y_481_ = v___y_500_;
v___y_482_ = v___x_506_;
goto v___jp_470_;
}
else
{
v___y_471_ = v___y_491_;
v___y_472_ = v___y_490_;
v___y_473_ = v___y_492_;
v___y_474_ = v___y_493_;
v___y_475_ = v___y_494_;
v___y_476_ = v___y_497_;
v___y_477_ = v___y_496_;
v___y_478_ = v___y_498_;
v___y_479_ = v___y_499_;
v___y_480_ = v___y_501_;
v___y_481_ = v___y_500_;
v___y_482_ = v___x_504_;
goto v___jp_470_;
}
}
else
{
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v___y_498_);
lean_dec(v___y_494_);
lean_dec(v___y_490_);
lean_dec_ref(v_M_458_);
v___y_460_ = v___y_491_;
v___y_461_ = v___y_492_;
v___y_462_ = v___y_493_;
v___y_463_ = v___y_496_;
v___y_464_ = v___y_497_;
v___y_465_ = v___y_499_;
goto v___jp_459_;
}
}
v___jp_507_:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
lean_inc(v___y_514_);
v___x_517_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___y_512_, v___y_514_);
lean_dec_ref(v___y_512_);
v___x_518_ = lean_nat_add(v___y_513_, v___x_517_);
lean_dec(v___x_517_);
lean_dec(v___y_513_);
v___x_519_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_519_, 0, v___y_516_);
lean_ctor_set(v___x_519_, 1, v___x_518_);
lean_ctor_set(v___x_519_, 2, v___y_515_);
v___x_520_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___y_508_, v___y_514_);
lean_dec_ref(v___y_508_);
v___x_521_ = lean_nat_add(v___y_511_, v___x_520_);
lean_dec(v___x_520_);
lean_dec(v___y_511_);
v___x_522_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_522_, 0, v___y_510_);
lean_ctor_set(v___x_522_, 1, v___x_521_);
lean_ctor_set(v___x_522_, 2, v___y_509_);
v_L_457_ = v___x_519_;
v_M_458_ = v___x_522_;
goto _start;
}
v___jp_524_:
{
if (v___y_537_ == 0)
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
lean_dec(v___y_530_);
lean_dec_ref(v___y_525_);
v___x_538_ = lean_string_utf8_extract_fast(v___y_536_, v___y_533_, v___y_535_);
lean_dec(v___y_533_);
v___x_539_ = lean_string_utf8_extract_fast(v___y_528_, v___y_526_, v___y_527_);
lean_dec(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_528_);
v___x_540_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__1));
v___x_541_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(v___x_538_, v___x_539_, v___x_540_, v___y_529_);
lean_dec_ref(v___x_539_);
lean_dec_ref(v___x_538_);
v___x_542_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(v_as_456_, v___x_541_);
v___x_543_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___y_531_, v___y_534_);
lean_dec_ref(v___y_531_);
v___x_544_ = lean_nat_add(v___y_532_, v___x_543_);
lean_dec(v___x_543_);
lean_dec(v___y_532_);
v___x_545_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_545_, 0, v___y_536_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
lean_ctor_set(v___x_545_, 2, v___y_535_);
v_as_456_ = v___x_542_;
v_L_457_ = v___x_545_;
goto _start;
}
else
{
lean_dec(v___y_533_);
lean_dec(v___y_529_);
lean_dec(v___y_526_);
lean_dec_ref(v_M_458_);
v___y_508_ = v___y_525_;
v___y_509_ = v___y_527_;
v___y_510_ = v___y_528_;
v___y_511_ = v___y_530_;
v___y_512_ = v___y_531_;
v___y_513_ = v___y_532_;
v___y_514_ = v___y_534_;
v___y_515_ = v___y_535_;
v___y_516_ = v___y_536_;
goto v___jp_507_;
}
}
v___jp_547_:
{
if (v___y_561_ == 0)
{
uint32_t v___x_562_; uint8_t v___x_563_; 
v___x_562_ = 13;
v___x_563_ = lean_uint32_dec_eq(v___y_559_, v___x_562_);
if (v___x_563_ == 0)
{
uint32_t v___x_564_; uint8_t v___x_565_; 
v___x_564_ = 10;
v___x_565_ = lean_uint32_dec_eq(v___y_559_, v___x_564_);
v___y_525_ = v___y_556_;
v___y_526_ = v___y_557_;
v___y_527_ = v___y_558_;
v___y_528_ = v___y_548_;
v___y_529_ = v___y_549_;
v___y_530_ = v___y_560_;
v___y_531_ = v___y_550_;
v___y_532_ = v___y_551_;
v___y_533_ = v___y_552_;
v___y_534_ = v___y_553_;
v___y_535_ = v___y_554_;
v___y_536_ = v___y_555_;
v___y_537_ = v___x_565_;
goto v___jp_524_;
}
else
{
v___y_525_ = v___y_556_;
v___y_526_ = v___y_557_;
v___y_527_ = v___y_558_;
v___y_528_ = v___y_548_;
v___y_529_ = v___y_549_;
v___y_530_ = v___y_560_;
v___y_531_ = v___y_550_;
v___y_532_ = v___y_551_;
v___y_533_ = v___y_552_;
v___y_534_ = v___y_553_;
v___y_535_ = v___y_554_;
v___y_536_ = v___y_555_;
v___y_537_ = v___x_563_;
goto v___jp_524_;
}
}
else
{
lean_dec(v___y_557_);
lean_dec(v___y_552_);
lean_dec(v___y_549_);
lean_dec_ref(v_M_458_);
v___y_508_ = v___y_556_;
v___y_509_ = v___y_558_;
v___y_510_ = v___y_548_;
v___y_511_ = v___y_560_;
v___y_512_ = v___y_550_;
v___y_513_ = v___y_551_;
v___y_514_ = v___y_553_;
v___y_515_ = v___y_554_;
v___y_516_ = v___y_555_;
goto v___jp_507_;
}
}
v___jp_566_:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_577_ = lean_string_utf8_extract_fast(v___y_576_, v___y_573_, v___y_575_);
lean_dec(v___y_575_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_576_);
v___x_578_ = lean_string_utf8_extract_fast(v___y_570_, v___y_568_, v___y_569_);
lean_dec(v___y_568_);
v___x_579_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__2));
v___x_580_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(v___x_577_, v___x_578_, v___x_579_, v___y_571_);
lean_dec_ref(v___x_578_);
lean_dec_ref(v___x_577_);
v___x_581_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(v_as_456_, v___x_580_);
v___x_582_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___y_567_, v___y_574_);
lean_dec_ref(v___y_567_);
v___x_583_ = lean_nat_add(v___y_572_, v___x_582_);
lean_dec(v___x_582_);
lean_dec(v___y_572_);
v___x_584_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_584_, 0, v___y_570_);
lean_ctor_set(v___x_584_, 1, v___x_583_);
lean_ctor_set(v___x_584_, 2, v___y_569_);
v_as_456_ = v___x_581_;
v_M_458_ = v___x_584_;
goto _start;
}
v___jp_586_:
{
if (v___y_598_ == 0)
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
lean_dec(v___y_595_);
lean_dec(v___y_594_);
lean_dec_ref(v___y_588_);
lean_dec(v___y_587_);
lean_dec_ref(v_L_457_);
v___x_599_ = lean_string_utf8_extract_fast(v___y_597_, v___y_593_, v___y_596_);
lean_dec(v___y_596_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_597_);
v___x_600_ = lean_string_utf8_extract_fast(v___y_590_, v___y_592_, v___y_589_);
lean_dec(v___y_589_);
lean_dec(v___y_592_);
lean_dec_ref(v___y_590_);
v___x_601_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__3));
v___x_602_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_mkFormatError(v___x_599_, v___x_600_, v___x_601_, v___y_591_);
lean_dec_ref(v___x_600_);
lean_dec_ref(v___x_599_);
v___x_603_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_pushFormatError(v_as_456_, v___x_602_);
return v___x_603_;
}
else
{
lean_dec(v___y_593_);
v___y_567_ = v___y_588_;
v___y_568_ = v___y_587_;
v___y_569_ = v___y_589_;
v___y_570_ = v___y_590_;
v___y_571_ = v___y_591_;
v___y_572_ = v___y_592_;
v___y_573_ = v___y_594_;
v___y_574_ = v___y_595_;
v___y_575_ = v___y_596_;
v___y_576_ = v___y_597_;
goto v___jp_566_;
}
}
v___jp_604_:
{
if (v___y_617_ == 0)
{
uint32_t v___x_618_; uint8_t v___x_619_; 
v___x_618_ = 13;
v___x_619_ = lean_uint32_dec_eq(v___y_610_, v___x_618_);
if (v___x_619_ == 0)
{
uint32_t v___x_620_; uint8_t v___x_621_; 
v___x_620_ = 10;
v___x_621_ = lean_uint32_dec_eq(v___y_610_, v___x_620_);
v___y_587_ = v___y_606_;
v___y_588_ = v___y_605_;
v___y_589_ = v___y_607_;
v___y_590_ = v___y_608_;
v___y_591_ = v___y_609_;
v___y_592_ = v___y_611_;
v___y_593_ = v___y_612_;
v___y_594_ = v___y_613_;
v___y_595_ = v___y_614_;
v___y_596_ = v___y_616_;
v___y_597_ = v___y_615_;
v___y_598_ = v___x_621_;
goto v___jp_586_;
}
else
{
v___y_587_ = v___y_606_;
v___y_588_ = v___y_605_;
v___y_589_ = v___y_607_;
v___y_590_ = v___y_608_;
v___y_591_ = v___y_609_;
v___y_592_ = v___y_611_;
v___y_593_ = v___y_612_;
v___y_594_ = v___y_613_;
v___y_595_ = v___y_614_;
v___y_596_ = v___y_616_;
v___y_597_ = v___y_615_;
v___y_598_ = v___x_619_;
goto v___jp_586_;
}
}
else
{
lean_dec(v___y_612_);
v___y_567_ = v___y_605_;
v___y_568_ = v___y_606_;
v___y_569_ = v___y_607_;
v___y_570_ = v___y_608_;
v___y_571_ = v___y_609_;
v___y_572_ = v___y_611_;
v___y_573_ = v___y_613_;
v___y_574_ = v___y_614_;
v___y_575_ = v___y_616_;
v___y_576_ = v___y_615_;
goto v___jp_566_;
}
}
v___jp_622_:
{
uint32_t v___x_637_; uint8_t v___x_638_; 
v___x_637_ = 32;
v___x_638_ = lean_uint32_dec_eq(v___y_636_, v___x_637_);
if (v___x_638_ == 0)
{
uint32_t v___x_639_; uint8_t v___x_640_; 
v___x_639_ = 10;
v___x_640_ = lean_uint32_dec_eq(v___y_636_, v___x_639_);
if (v___x_640_ == 0)
{
uint8_t v___x_641_; 
lean_dec_ref(v_M_458_);
v___x_641_ = lean_uint32_dec_eq(v___y_636_, v___y_634_);
if (v___x_641_ == 0)
{
uint8_t v___x_642_; 
lean_dec_ref(v___y_625_);
v___x_642_ = lean_uint32_dec_eq(v___y_634_, v___x_637_);
if (v___x_642_ == 0)
{
uint32_t v___x_643_; uint8_t v___x_644_; 
v___x_643_ = 9;
v___x_644_ = lean_uint32_dec_eq(v___y_634_, v___x_643_);
v___y_605_ = v___y_631_;
v___y_606_ = v___y_632_;
v___y_607_ = v___y_633_;
v___y_608_ = v___y_623_;
v___y_609_ = v___y_624_;
v___y_610_ = v___y_634_;
v___y_611_ = v___y_635_;
v___y_612_ = v___y_626_;
v___y_613_ = v___y_627_;
v___y_614_ = v___y_628_;
v___y_615_ = v___y_629_;
v___y_616_ = v___y_630_;
v___y_617_ = v___x_644_;
goto v___jp_604_;
}
else
{
v___y_605_ = v___y_631_;
v___y_606_ = v___y_632_;
v___y_607_ = v___y_633_;
v___y_608_ = v___y_623_;
v___y_609_ = v___y_624_;
v___y_610_ = v___y_634_;
v___y_611_ = v___y_635_;
v___y_612_ = v___y_626_;
v___y_613_ = v___y_627_;
v___y_614_ = v___y_628_;
v___y_615_ = v___y_629_;
v___y_616_ = v___y_630_;
v___y_617_ = v___x_642_;
goto v___jp_604_;
}
}
else
{
lean_dec(v___y_635_);
lean_dec(v___y_633_);
lean_dec(v___y_632_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec(v___y_627_);
lean_dec(v___y_626_);
lean_dec(v___y_624_);
lean_dec_ref(v___y_623_);
lean_dec_ref(v_L_457_);
v_L_457_ = v___y_625_;
v_M_458_ = v___y_631_;
goto _start;
}
}
else
{
uint8_t v___x_646_; 
lean_dec_ref(v_L_457_);
v___x_646_ = lean_uint32_dec_eq(v___y_634_, v___x_637_);
if (v___x_646_ == 0)
{
uint32_t v___x_647_; uint8_t v___x_648_; 
v___x_647_ = 9;
v___x_648_ = lean_uint32_dec_eq(v___y_634_, v___x_647_);
v___y_548_ = v___y_623_;
v___y_549_ = v___y_624_;
v___y_550_ = v___y_625_;
v___y_551_ = v___y_626_;
v___y_552_ = v___y_627_;
v___y_553_ = v___y_628_;
v___y_554_ = v___y_630_;
v___y_555_ = v___y_629_;
v___y_556_ = v___y_631_;
v___y_557_ = v___y_632_;
v___y_558_ = v___y_633_;
v___y_559_ = v___y_634_;
v___y_560_ = v___y_635_;
v___y_561_ = v___x_648_;
goto v___jp_547_;
}
else
{
v___y_548_ = v___y_623_;
v___y_549_ = v___y_624_;
v___y_550_ = v___y_625_;
v___y_551_ = v___y_626_;
v___y_552_ = v___y_627_;
v___y_553_ = v___y_628_;
v___y_554_ = v___y_630_;
v___y_555_ = v___y_629_;
v___y_556_ = v___y_631_;
v___y_557_ = v___y_632_;
v___y_558_ = v___y_633_;
v___y_559_ = v___y_634_;
v___y_560_ = v___y_635_;
v___y_561_ = v___x_646_;
goto v___jp_547_;
}
}
}
else
{
uint8_t v___x_649_; 
lean_dec(v___y_626_);
lean_dec_ref(v_L_457_);
v___x_649_ = lean_uint32_dec_eq(v___y_634_, v___x_637_);
if (v___x_649_ == 0)
{
uint32_t v___x_650_; uint8_t v___x_651_; 
v___x_650_ = 9;
v___x_651_ = lean_uint32_dec_eq(v___y_634_, v___x_650_);
v___y_490_ = v___y_632_;
v___y_491_ = v___y_631_;
v___y_492_ = v___y_633_;
v___y_493_ = v___y_623_;
v___y_494_ = v___y_624_;
v___y_495_ = v___y_634_;
v___y_496_ = v___y_625_;
v___y_497_ = v___y_635_;
v___y_498_ = v___y_627_;
v___y_499_ = v___y_628_;
v___y_500_ = v___y_629_;
v___y_501_ = v___y_630_;
v___y_502_ = v___x_651_;
goto v___jp_489_;
}
else
{
v___y_490_ = v___y_632_;
v___y_491_ = v___y_631_;
v___y_492_ = v___y_633_;
v___y_493_ = v___y_623_;
v___y_494_ = v___y_624_;
v___y_495_ = v___y_634_;
v___y_496_ = v___y_625_;
v___y_497_ = v___y_635_;
v___y_498_ = v___y_627_;
v___y_499_ = v___y_628_;
v___y_500_ = v___y_629_;
v___y_501_ = v___y_630_;
v___y_502_ = v___x_649_;
goto v___jp_489_;
}
}
}
v___jp_652_:
{
lean_object* v___x_666_; 
v___x_666_ = l_String_Slice_Pos_get_x3f(v_L_457_, v___y_662_);
if (lean_obj_tag(v___x_666_) == 0)
{
uint32_t v___x_667_; 
v___x_667_ = 65;
v___y_623_ = v___y_656_;
v___y_624_ = v___y_657_;
v___y_625_ = v___y_659_;
v___y_626_ = v___y_660_;
v___y_627_ = v___y_661_;
v___y_628_ = v___y_662_;
v___y_629_ = v___y_664_;
v___y_630_ = v___y_663_;
v___y_631_ = v___y_653_;
v___y_632_ = v___y_654_;
v___y_633_ = v___y_655_;
v___y_634_ = v___y_665_;
v___y_635_ = v___y_658_;
v___y_636_ = v___x_667_;
goto v___jp_622_;
}
else
{
lean_object* v_val_668_; uint32_t v___x_669_; 
v_val_668_ = lean_ctor_get(v___x_666_, 0);
lean_inc(v_val_668_);
lean_dec_ref_known(v___x_666_, 1);
v___x_669_ = lean_unbox_uint32(v_val_668_);
lean_dec(v_val_668_);
v___y_623_ = v___y_656_;
v___y_624_ = v___y_657_;
v___y_625_ = v___y_659_;
v___y_626_ = v___y_660_;
v___y_627_ = v___y_661_;
v___y_628_ = v___y_662_;
v___y_629_ = v___y_664_;
v___y_630_ = v___y_663_;
v___y_631_ = v___y_653_;
v___y_632_ = v___y_654_;
v___y_633_ = v___y_655_;
v___y_634_ = v___y_665_;
v___y_635_ = v___y_658_;
v___y_636_ = v___x_669_;
goto v___jp_622_;
}
}
v_resetjp_673_:
{
lean_object* v___x_676_; lean_object* v___x_677_; uint8_t v___x_678_; 
v___x_676_ = lean_nat_sub(v_endExclusive_672_, v_startInclusive_671_);
lean_dec(v_startInclusive_671_);
lean_dec(v_endExclusive_672_);
v___x_677_ = lean_unsigned_to_nat(0u);
v___x_678_ = lean_nat_dec_eq(v___x_676_, v___x_677_);
if (v___x_678_ == 0)
{
lean_object* v_str_679_; lean_object* v_startInclusive_680_; lean_object* v_endExclusive_681_; uint8_t v___y_683_; lean_object* v_fst_684_; lean_object* v_snd_685_; uint8_t v___y_731_; lean_object* v___y_788_; lean_object* v___y_789_; lean_object* v___y_794_; lean_object* v___x_796_; lean_object* v___y_798_; lean_object* v___x_812_; lean_object* v___x_813_; uint8_t v___x_814_; 
v_str_679_ = lean_ctor_get(v_L_457_, 0);
v_startInclusive_680_ = lean_ctor_get(v_L_457_, 1);
v_endExclusive_681_ = lean_ctor_get(v_L_457_, 2);
v___x_796_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__9));
v___x_812_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10);
v___x_813_ = lean_nat_sub(v_endExclusive_681_, v_startInclusive_680_);
v___x_814_ = lean_nat_dec_le(v___x_812_, v___x_813_);
lean_dec(v___x_813_);
if (v___x_814_ == 0)
{
goto v___jp_810_;
}
else
{
uint8_t v___x_815_; 
v___x_815_ = lean_string_memcmp(v_str_679_, v___x_796_, v_startInclusive_680_, v___x_677_, v___x_812_);
if (v___x_815_ == 0)
{
goto v___jp_810_;
}
else
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_816_ = l_String_Slice_pos_x21(v_L_457_, v___x_812_);
v___x_817_ = lean_nat_add(v_startInclusive_680_, v___x_816_);
lean_dec(v___x_816_);
lean_inc(v_endExclusive_681_);
lean_inc_ref(v_str_679_);
v___x_818_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_818_, 0, v_str_679_);
lean_ctor_set(v___x_818_, 1, v___x_817_);
lean_ctor_set(v___x_818_, 2, v_endExclusive_681_);
v___x_819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
v___y_798_ = v___x_819_;
goto v___jp_797_;
}
}
v___jp_682_:
{
lean_object* v_str_686_; lean_object* v_startInclusive_687_; lean_object* v_endExclusive_688_; lean_object* v___x_689_; lean_object* v_newL_691_; 
v_str_686_ = lean_ctor_get(v_M_458_, 0);
lean_inc_ref(v_str_686_);
v_startInclusive_687_ = lean_ctor_get(v_M_458_, 1);
lean_inc(v_startInclusive_687_);
v_endExclusive_688_ = lean_ctor_get(v_M_458_, 2);
lean_inc(v_endExclusive_688_);
v___x_689_ = lean_nat_add(v_startInclusive_680_, v_fst_684_);
lean_dec(v_fst_684_);
lean_dec(v_startInclusive_680_);
lean_inc(v_endExclusive_681_);
lean_inc(v___x_689_);
lean_inc_ref(v_str_679_);
if (v_isShared_675_ == 0)
{
lean_ctor_set(v___x_674_, 2, v_endExclusive_681_);
lean_ctor_set(v___x_674_, 1, v___x_689_);
lean_ctor_set(v___x_674_, 0, v_str_679_);
v_newL_691_ = v___x_674_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_str_679_);
lean_ctor_set(v_reuseFailAlloc_714_, 1, v___x_689_);
lean_ctor_set(v_reuseFailAlloc_714_, 2, v_endExclusive_681_);
v_newL_691_ = v_reuseFailAlloc_714_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
lean_object* v___x_692_; lean_object* v___x_694_; uint8_t v_isShared_695_; uint8_t v_isSharedCheck_710_; 
v___x_692_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__1(v___y_683_, v___x_676_, v_M_458_, v___x_677_);
lean_dec(v___x_676_);
v_isSharedCheck_710_ = !lean_is_exclusive(v_M_458_);
if (v_isSharedCheck_710_ == 0)
{
lean_object* v_unused_711_; lean_object* v_unused_712_; lean_object* v_unused_713_; 
v_unused_711_ = lean_ctor_get(v_M_458_, 2);
lean_dec(v_unused_711_);
v_unused_712_ = lean_ctor_get(v_M_458_, 1);
lean_dec(v_unused_712_);
v_unused_713_ = lean_ctor_get(v_M_458_, 0);
lean_dec(v_unused_713_);
v___x_694_ = v_M_458_;
v_isShared_695_ = v_isSharedCheck_710_;
goto v_resetjp_693_;
}
else
{
lean_dec(v_M_458_);
v___x_694_ = lean_box(0);
v_isShared_695_ = v_isSharedCheck_710_;
goto v_resetjp_693_;
}
v_resetjp_693_:
{
lean_object* v___x_696_; lean_object* v___x_698_; 
v___x_696_ = lean_nat_add(v_startInclusive_687_, v___x_692_);
lean_dec(v___x_692_);
lean_dec(v_startInclusive_687_);
lean_inc(v_endExclusive_688_);
lean_inc(v___x_696_);
lean_inc_ref(v_str_686_);
if (v_isShared_695_ == 0)
{
lean_ctor_set(v___x_694_, 1, v___x_696_);
v___x_698_ = v___x_694_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_str_686_);
lean_ctor_set(v_reuseFailAlloc_709_, 1, v___x_696_);
lean_ctor_set(v_reuseFailAlloc_709_, 2, v_endExclusive_688_);
v___x_698_ = v_reuseFailAlloc_709_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v_newM_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_699_ = l_String_Slice_Pos_nextn(v___x_698_, v___x_677_, v_snd_685_);
lean_dec_ref(v___x_698_);
v___x_700_ = lean_nat_add(v___x_696_, v___x_699_);
lean_dec(v___x_699_);
lean_dec(v___x_696_);
lean_inc(v_endExclusive_688_);
lean_inc(v___x_700_);
lean_inc_ref(v_str_686_);
v_newM_701_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_newM_701_, 0, v_str_686_);
lean_ctor_set(v_newM_701_, 1, v___x_700_);
lean_ctor_set(v_newM_701_, 2, v_endExclusive_688_);
v___x_702_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v_newL_691_, v___x_677_);
lean_dec_ref(v_newL_691_);
v___x_703_ = lean_nat_add(v___x_689_, v___x_702_);
lean_dec(v___x_702_);
lean_dec(v___x_689_);
v___x_704_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_704_, 0, v_str_679_);
lean_ctor_set(v___x_704_, 1, v___x_703_);
lean_ctor_set(v___x_704_, 2, v_endExclusive_681_);
v___x_705_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v_newM_701_, v___x_677_);
lean_dec_ref_known(v_newM_701_, 3);
v___x_706_ = lean_nat_add(v___x_700_, v___x_705_);
lean_dec(v___x_705_);
lean_dec(v___x_700_);
v___x_707_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_707_, 0, v_str_686_);
lean_ctor_set(v___x_707_, 1, v___x_706_);
lean_ctor_set(v___x_707_, 2, v_endExclusive_688_);
v_L_457_ = v___x_704_;
v_M_458_ = v___x_707_;
goto _start;
}
}
}
}
v___jp_715_:
{
lean_object* v_str_716_; lean_object* v_startInclusive_717_; lean_object* v_endExclusive_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v_ls_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v_ms_725_; lean_object* v___x_726_; 
v_str_716_ = lean_ctor_get(v_M_458_, 0);
v_startInclusive_717_ = lean_ctor_get(v_M_458_, 1);
v_endExclusive_718_ = lean_ctor_get(v_M_458_, 2);
v___x_719_ = lean_unsigned_to_nat(1u);
v___x_720_ = l_String_Slice_Pos_nextn(v_L_457_, v___x_677_, v___x_719_);
v___x_721_ = lean_nat_add(v_startInclusive_680_, v___x_720_);
lean_dec(v___x_720_);
lean_inc(v_endExclusive_681_);
lean_inc(v___x_721_);
lean_inc_ref(v_str_679_);
v_ls_722_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_ls_722_, 0, v_str_679_);
lean_ctor_set(v_ls_722_, 1, v___x_721_);
lean_ctor_set(v_ls_722_, 2, v_endExclusive_681_);
v___x_723_ = l_String_Slice_Pos_nextn(v_M_458_, v___x_677_, v___x_719_);
v___x_724_ = lean_nat_add(v_startInclusive_717_, v___x_723_);
lean_dec(v___x_723_);
lean_inc(v_endExclusive_718_);
lean_inc(v___x_724_);
lean_inc_ref(v_str_716_);
v_ms_725_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_ms_725_, 0, v_str_716_);
lean_ctor_set(v_ms_725_, 1, v___x_724_);
lean_ctor_set(v_ms_725_, 2, v_endExclusive_718_);
v___x_726_ = l_String_Slice_Pos_get_x3f(v_M_458_, v___x_677_);
if (lean_obj_tag(v___x_726_) == 0)
{
uint32_t v___x_727_; 
v___x_727_ = 65;
lean_inc_ref(v_str_679_);
lean_inc(v_endExclusive_681_);
lean_inc(v_startInclusive_680_);
lean_inc_ref(v_str_716_);
lean_inc(v_endExclusive_718_);
lean_inc(v_startInclusive_717_);
v___y_653_ = v_ms_725_;
v___y_654_ = v_startInclusive_717_;
v___y_655_ = v_endExclusive_718_;
v___y_656_ = v_str_716_;
v___y_657_ = v___x_719_;
v___y_658_ = v___x_724_;
v___y_659_ = v_ls_722_;
v___y_660_ = v___x_721_;
v___y_661_ = v_startInclusive_680_;
v___y_662_ = v___x_677_;
v___y_663_ = v_endExclusive_681_;
v___y_664_ = v_str_679_;
v___y_665_ = v___x_727_;
goto v___jp_652_;
}
else
{
lean_object* v_val_728_; uint32_t v___x_729_; 
v_val_728_ = lean_ctor_get(v___x_726_, 0);
lean_inc(v_val_728_);
lean_dec_ref_known(v___x_726_, 1);
v___x_729_ = lean_unbox_uint32(v_val_728_);
lean_dec(v_val_728_);
lean_inc_ref(v_str_679_);
lean_inc(v_endExclusive_681_);
lean_inc(v_startInclusive_680_);
lean_inc_ref(v_str_716_);
lean_inc(v_endExclusive_718_);
lean_inc(v_startInclusive_717_);
v___y_653_ = v_ms_725_;
v___y_654_ = v_startInclusive_717_;
v___y_655_ = v_endExclusive_718_;
v___y_656_ = v_str_716_;
v___y_657_ = v___x_719_;
v___y_658_ = v___x_724_;
v___y_659_ = v_ls_722_;
v___y_660_ = v___x_721_;
v___y_661_ = v_startInclusive_680_;
v___y_662_ = v___x_677_;
v___y_663_ = v_endExclusive_681_;
v___y_664_ = v_str_679_;
v___y_665_ = v___x_729_;
goto v___jp_652_;
}
}
v___jp_730_:
{
if (v___y_731_ == 0)
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; uint8_t v___x_735_; 
lean_dec(v___x_676_);
lean_del_object(v___x_674_);
v___x_732_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__4));
v___x_733_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__5);
v___x_734_ = lean_nat_sub(v_endExclusive_681_, v_startInclusive_680_);
v___x_735_ = lean_nat_dec_le(v___x_733_, v___x_734_);
lean_dec(v___x_734_);
if (v___x_735_ == 0)
{
goto v___jp_715_;
}
else
{
uint8_t v___x_736_; 
v___x_736_ = lean_string_memcmp(v_str_679_, v___x_732_, v_startInclusive_680_, v___x_677_, v___x_733_);
if (v___x_736_ == 0)
{
goto v___jp_715_;
}
else
{
lean_object* v_str_737_; lean_object* v_startInclusive_738_; lean_object* v_endExclusive_739_; lean_object* v___x_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_768_; 
lean_inc(v_endExclusive_681_);
lean_inc(v_startInclusive_680_);
lean_inc_ref(v_str_679_);
v_str_737_ = lean_ctor_get(v_M_458_, 0);
lean_inc_ref(v_str_737_);
v_startInclusive_738_ = lean_ctor_get(v_M_458_, 1);
lean_inc(v_startInclusive_738_);
v_endExclusive_739_ = lean_ctor_get(v_M_458_, 2);
lean_inc(v_endExclusive_739_);
v___x_740_ = l_String_Slice_pos_x21(v_L_457_, v___x_733_);
v_isSharedCheck_768_ = !lean_is_exclusive(v_L_457_);
if (v_isSharedCheck_768_ == 0)
{
lean_object* v_unused_769_; lean_object* v_unused_770_; lean_object* v_unused_771_; 
v_unused_769_ = lean_ctor_get(v_L_457_, 2);
lean_dec(v_unused_769_);
v_unused_770_ = lean_ctor_get(v_L_457_, 1);
lean_dec(v_unused_770_);
v_unused_771_ = lean_ctor_get(v_L_457_, 0);
lean_dec(v_unused_771_);
v___x_742_ = v_L_457_;
v_isShared_743_ = v_isSharedCheck_768_;
goto v_resetjp_741_;
}
else
{
lean_dec(v_L_457_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_768_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; lean_object* v___x_746_; 
v___x_744_ = lean_nat_add(v_startInclusive_680_, v___x_740_);
lean_dec(v___x_740_);
lean_dec(v_startInclusive_680_);
lean_inc(v_endExclusive_681_);
lean_inc(v___x_744_);
lean_inc_ref(v_str_679_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 1, v___x_744_);
v___x_746_ = v___x_742_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_str_679_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v___x_744_);
lean_ctor_set(v_reuseFailAlloc_767_, 2, v_endExclusive_681_);
v___x_746_ = v_reuseFailAlloc_767_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v_newL_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_763_; 
v___x_747_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___x_746_, v___x_677_);
lean_dec_ref(v___x_746_);
v___x_748_ = lean_nat_add(v___x_744_, v___x_747_);
lean_dec(v___x_747_);
lean_dec(v___x_744_);
v_newL_749_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_newL_749_, 0, v_str_679_);
lean_ctor_set(v_newL_749_, 1, v___x_748_);
lean_ctor_set(v_newL_749_, 2, v_endExclusive_681_);
v___x_750_ = lean_unsigned_to_nat(2u);
v___x_751_ = l_String_Slice_Pos_nextn(v_M_458_, v___x_677_, v___x_750_);
v_isSharedCheck_763_ = !lean_is_exclusive(v_M_458_);
if (v_isSharedCheck_763_ == 0)
{
lean_object* v_unused_764_; lean_object* v_unused_765_; lean_object* v_unused_766_; 
v_unused_764_ = lean_ctor_get(v_M_458_, 2);
lean_dec(v_unused_764_);
v_unused_765_ = lean_ctor_get(v_M_458_, 1);
lean_dec(v_unused_765_);
v_unused_766_ = lean_ctor_get(v_M_458_, 0);
lean_dec(v_unused_766_);
v___x_753_ = v_M_458_;
v_isShared_754_ = v_isSharedCheck_763_;
goto v_resetjp_752_;
}
else
{
lean_dec(v_M_458_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_763_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v___x_755_; lean_object* v___x_757_; 
v___x_755_ = lean_nat_add(v_startInclusive_738_, v___x_751_);
lean_dec(v___x_751_);
lean_dec(v_startInclusive_738_);
lean_inc(v_endExclusive_739_);
lean_inc(v___x_755_);
lean_inc_ref(v_str_737_);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 1, v___x_755_);
v___x_757_ = v___x_753_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_762_; 
v_reuseFailAlloc_762_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_762_, 0, v_str_737_);
lean_ctor_set(v_reuseFailAlloc_762_, 1, v___x_755_);
lean_ctor_set(v_reuseFailAlloc_762_, 2, v_endExclusive_739_);
v___x_757_ = v_reuseFailAlloc_762_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v_newM_760_; 
v___x_758_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__0(v___x_757_, v___x_677_);
lean_dec_ref(v___x_757_);
v___x_759_ = lean_nat_add(v___x_755_, v___x_758_);
lean_dec(v___x_758_);
lean_dec(v___x_755_);
v_newM_760_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_newM_760_, 0, v_str_737_);
lean_ctor_set(v_newM_760_, 1, v___x_759_);
lean_ctor_set(v_newM_760_, 2, v_endExclusive_739_);
v_L_457_ = v_newL_749_;
v_M_458_ = v_newM_760_;
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
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v_fst_775_; 
lean_inc(v_endExclusive_681_);
lean_inc(v_startInclusive_680_);
lean_inc_ref(v_str_679_);
v___x_772_ = l_String_Slice_positions(v_L_457_);
v___x_773_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__6));
v___x_774_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg(v_L_457_, v___x_772_, v___x_773_);
lean_dec_ref(v_L_457_);
v_fst_775_ = lean_ctor_get(v___x_774_, 0);
lean_inc(v_fst_775_);
if (lean_obj_tag(v_fst_775_) == 0)
{
lean_object* v_snd_776_; lean_object* v___x_777_; 
v_snd_776_ = lean_ctor_get(v___x_774_, 1);
lean_inc(v_snd_776_);
lean_dec_ref(v___x_774_);
v___x_777_ = lean_nat_sub(v_endExclusive_681_, v_startInclusive_680_);
v___y_683_ = v___y_731_;
v_fst_684_ = v___x_777_;
v_snd_685_ = v_snd_776_;
goto v___jp_682_;
}
else
{
lean_object* v_val_778_; lean_object* v_fst_779_; lean_object* v_snd_780_; 
lean_dec_ref(v___x_774_);
v_val_778_ = lean_ctor_get(v_fst_775_, 0);
lean_inc(v_val_778_);
lean_dec_ref_known(v_fst_775_, 1);
v_fst_779_ = lean_ctor_get(v_val_778_, 0);
lean_inc(v_fst_779_);
v_snd_780_ = lean_ctor_get(v_val_778_, 1);
lean_inc(v_snd_780_);
lean_dec(v_val_778_);
v___y_683_ = v___y_731_;
v_fst_684_ = v_fst_779_;
v_snd_685_ = v_snd_780_;
goto v___jp_682_;
}
}
}
v___jp_781_:
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v___x_782_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__7));
v___x_783_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__8);
v___x_784_ = lean_nat_sub(v_endExclusive_681_, v_startInclusive_680_);
v___x_785_ = lean_nat_dec_le(v___x_783_, v___x_784_);
lean_dec(v___x_784_);
if (v___x_785_ == 0)
{
v___y_731_ = v___x_678_;
goto v___jp_730_;
}
else
{
uint8_t v___x_786_; 
v___x_786_ = lean_string_memcmp(v_str_679_, v___x_782_, v_startInclusive_680_, v___x_677_, v___x_783_);
v___y_731_ = v___x_786_;
goto v___jp_730_;
}
}
v___jp_787_:
{
if (lean_obj_tag(v___y_788_) == 1)
{
if (lean_obj_tag(v___y_789_) == 1)
{
lean_object* v_val_790_; lean_object* v_val_791_; 
lean_dec(v___x_676_);
lean_del_object(v___x_674_);
lean_dec_ref(v_M_458_);
lean_dec_ref(v_L_457_);
v_val_790_ = lean_ctor_get(v___y_788_, 0);
lean_inc(v_val_790_);
lean_dec_ref_known(v___y_788_, 1);
v_val_791_ = lean_ctor_get(v___y_789_, 0);
lean_inc(v_val_791_);
lean_dec_ref_known(v___y_789_, 1);
v_L_457_ = v_val_790_;
v_M_458_ = v_val_791_;
goto _start;
}
else
{
lean_dec_ref_known(v___y_788_, 1);
lean_dec(v___y_789_);
goto v___jp_781_;
}
}
else
{
lean_dec(v___y_789_);
lean_dec(v___y_788_);
goto v___jp_781_;
}
}
v___jp_793_:
{
lean_object* v___x_795_; 
v___x_795_ = lean_box(0);
v___y_788_ = v___y_794_;
v___y_789_ = v___x_795_;
goto v___jp_787_;
}
v___jp_797_:
{
lean_object* v_str_799_; lean_object* v_startInclusive_800_; lean_object* v_endExclusive_801_; lean_object* v___x_802_; lean_object* v___x_803_; uint8_t v___x_804_; 
v_str_799_ = lean_ctor_get(v_M_458_, 0);
v_startInclusive_800_ = lean_ctor_get(v_M_458_, 1);
v_endExclusive_801_ = lean_ctor_get(v_M_458_, 2);
v___x_802_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux___closed__10);
v___x_803_ = lean_nat_sub(v_endExclusive_801_, v_startInclusive_800_);
v___x_804_ = lean_nat_dec_le(v___x_802_, v___x_803_);
lean_dec(v___x_803_);
if (v___x_804_ == 0)
{
v___y_794_ = v___y_798_;
goto v___jp_793_;
}
else
{
uint8_t v___x_805_; 
v___x_805_ = lean_string_memcmp(v_str_799_, v___x_796_, v_startInclusive_800_, v___x_677_, v___x_802_);
if (v___x_805_ == 0)
{
v___y_794_ = v___y_798_;
goto v___jp_793_;
}
else
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_806_ = l_String_Slice_pos_x21(v_M_458_, v___x_802_);
v___x_807_ = lean_nat_add(v_startInclusive_800_, v___x_806_);
lean_dec(v___x_806_);
lean_inc(v_endExclusive_801_);
lean_inc_ref(v_str_799_);
v___x_808_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_808_, 0, v_str_799_);
lean_ctor_set(v___x_808_, 1, v___x_807_);
lean_ctor_set(v___x_808_, 2, v_endExclusive_801_);
v___x_809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_809_, 0, v___x_808_);
v___y_788_ = v___y_798_;
v___y_789_ = v___x_809_;
goto v___jp_787_;
}
}
}
v___jp_810_:
{
lean_object* v___x_811_; 
v___x_811_ = lean_box(0);
v___y_798_ = v___x_811_;
goto v___jp_797_;
}
}
else
{
lean_dec(v___x_676_);
lean_del_object(v___x_674_);
lean_dec_ref(v_M_458_);
lean_dec_ref(v_L_457_);
return v_as_456_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2(lean_object* v_L_822_, lean_object* v_inst_823_, lean_object* v_R_824_, lean_object* v_a_825_, lean_object* v_b_826_, lean_object* v_c_827_){
_start:
{
lean_object* v___x_828_; 
v___x_828_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___redArg(v_L_822_, v_a_825_, v_b_826_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2___boxed(lean_object* v_L_829_, lean_object* v_inst_830_, lean_object* v_R_831_, lean_object* v_a_832_, lean_object* v_b_833_, lean_object* v_c_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux_spec__2(v_L_829_, v_inst_830_, v_R_831_, v_a_832_, v_b_833_, v_c_834_);
lean_dec_ref(v_L_829_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan(lean_object* v_src_838_, lean_object* v_fmt_839_){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_840_ = lean_unsigned_to_nat(0u);
v___x_841_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan___closed__0));
v___x_842_ = lean_string_utf8_byte_size(v_src_838_);
v___x_843_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_843_, 0, v_src_838_);
lean_ctor_set(v___x_843_, 1, v___x_840_);
lean_ctor_set(v___x_843_, 2, v___x_842_);
v___x_844_ = lean_string_utf8_byte_size(v_fmt_839_);
v___x_845_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_845_, 0, v_fmt_839_);
lean_ctor_set(v___x_845_, 1, v___x_840_);
lean_ctor_set(v___x_845_, 2, v___x_844_);
v___x_846_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScanAux(v___x_841_, v___x_843_, v___x_845_);
return v___x_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7___redArg(lean_object* v_x_941_, lean_object* v_x_942_){
_start:
{
if (lean_obj_tag(v_x_942_) == 0)
{
return v_x_941_;
}
else
{
lean_object* v_key_943_; lean_object* v_value_944_; lean_object* v_tail_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_968_; 
v_key_943_ = lean_ctor_get(v_x_942_, 0);
v_value_944_ = lean_ctor_get(v_x_942_, 1);
v_tail_945_ = lean_ctor_get(v_x_942_, 2);
v_isSharedCheck_968_ = !lean_is_exclusive(v_x_942_);
if (v_isSharedCheck_968_ == 0)
{
v___x_947_ = v_x_942_;
v_isShared_948_ = v_isSharedCheck_968_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_tail_945_);
lean_inc(v_value_944_);
lean_inc(v_key_943_);
lean_dec(v_x_942_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_968_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_949_; uint64_t v___x_950_; uint64_t v___x_951_; uint64_t v___x_952_; uint64_t v_fold_953_; uint64_t v___x_954_; uint64_t v___x_955_; uint64_t v___x_956_; size_t v___x_957_; size_t v___x_958_; size_t v___x_959_; size_t v___x_960_; size_t v___x_961_; lean_object* v___x_962_; lean_object* v___x_964_; 
v___x_949_ = lean_array_get_size(v_x_941_);
v___x_950_ = l_Lean_Syntax_instHashableRange_hash(v_key_943_);
v___x_951_ = 32ULL;
v___x_952_ = lean_uint64_shift_right(v___x_950_, v___x_951_);
v_fold_953_ = lean_uint64_xor(v___x_950_, v___x_952_);
v___x_954_ = 16ULL;
v___x_955_ = lean_uint64_shift_right(v_fold_953_, v___x_954_);
v___x_956_ = lean_uint64_xor(v_fold_953_, v___x_955_);
v___x_957_ = lean_uint64_to_usize(v___x_956_);
v___x_958_ = lean_usize_of_nat(v___x_949_);
v___x_959_ = ((size_t)1ULL);
v___x_960_ = lean_usize_sub(v___x_958_, v___x_959_);
v___x_961_ = lean_usize_land(v___x_957_, v___x_960_);
v___x_962_ = lean_array_uget_borrowed(v_x_941_, v___x_961_);
lean_inc(v___x_962_);
if (v_isShared_948_ == 0)
{
lean_ctor_set(v___x_947_, 2, v___x_962_);
v___x_964_ = v___x_947_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_967_; 
v_reuseFailAlloc_967_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_967_, 0, v_key_943_);
lean_ctor_set(v_reuseFailAlloc_967_, 1, v_value_944_);
lean_ctor_set(v_reuseFailAlloc_967_, 2, v___x_962_);
v___x_964_ = v_reuseFailAlloc_967_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
lean_object* v___x_965_; 
v___x_965_ = lean_array_uset(v_x_941_, v___x_961_, v___x_964_);
v_x_941_ = v___x_965_;
v_x_942_ = v_tail_945_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2___redArg(lean_object* v_i_969_, lean_object* v_source_970_, lean_object* v_target_971_){
_start:
{
lean_object* v___x_972_; uint8_t v___x_973_; 
v___x_972_ = lean_array_get_size(v_source_970_);
v___x_973_ = lean_nat_dec_lt(v_i_969_, v___x_972_);
if (v___x_973_ == 0)
{
lean_dec_ref(v_source_970_);
lean_dec(v_i_969_);
return v_target_971_;
}
else
{
lean_object* v_es_974_; lean_object* v___x_975_; lean_object* v_source_976_; lean_object* v_target_977_; lean_object* v___x_978_; lean_object* v___x_979_; 
v_es_974_ = lean_array_fget(v_source_970_, v_i_969_);
v___x_975_ = lean_box(0);
v_source_976_ = lean_array_fset(v_source_970_, v_i_969_, v___x_975_);
v_target_977_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7___redArg(v_target_971_, v_es_974_);
v___x_978_ = lean_unsigned_to_nat(1u);
v___x_979_ = lean_nat_add(v_i_969_, v___x_978_);
lean_dec(v_i_969_);
v_i_969_ = v___x_979_;
v_source_970_ = v_source_976_;
v_target_971_ = v_target_977_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1___redArg(lean_object* v_data_981_){
_start:
{
lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v_nbuckets_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_982_ = lean_array_get_size(v_data_981_);
v___x_983_ = lean_unsigned_to_nat(2u);
v_nbuckets_984_ = lean_nat_mul(v___x_982_, v___x_983_);
v___x_985_ = lean_unsigned_to_nat(0u);
v___x_986_ = lean_box(0);
v___x_987_ = lean_mk_array(v_nbuckets_984_, v___x_986_);
v___x_988_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2___redArg(v___x_985_, v_data_981_, v___x_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5___redArg(lean_object* v_a_989_, lean_object* v_b_990_, lean_object* v_x_991_){
_start:
{
if (lean_obj_tag(v_x_991_) == 0)
{
lean_dec(v_b_990_);
lean_dec_ref(v_a_989_);
return v_x_991_;
}
else
{
lean_object* v_key_992_; lean_object* v_value_993_; lean_object* v_tail_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1006_; 
v_key_992_ = lean_ctor_get(v_x_991_, 0);
v_value_993_ = lean_ctor_get(v_x_991_, 1);
v_tail_994_ = lean_ctor_get(v_x_991_, 2);
v_isSharedCheck_1006_ = !lean_is_exclusive(v_x_991_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_996_ = v_x_991_;
v_isShared_997_ = v_isSharedCheck_1006_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_tail_994_);
lean_inc(v_value_993_);
lean_inc(v_key_992_);
lean_dec(v_x_991_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1006_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
uint8_t v___x_998_; 
v___x_998_ = l_Lean_Syntax_instBEqRange_beq(v_key_992_, v_a_989_);
if (v___x_998_ == 0)
{
lean_object* v___x_999_; lean_object* v___x_1001_; 
v___x_999_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5___redArg(v_a_989_, v_b_990_, v_tail_994_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 2, v___x_999_);
v___x_1001_ = v___x_996_;
goto v_reusejp_1000_;
}
else
{
lean_object* v_reuseFailAlloc_1002_; 
v_reuseFailAlloc_1002_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1002_, 0, v_key_992_);
lean_ctor_set(v_reuseFailAlloc_1002_, 1, v_value_993_);
lean_ctor_set(v_reuseFailAlloc_1002_, 2, v___x_999_);
v___x_1001_ = v_reuseFailAlloc_1002_;
goto v_reusejp_1000_;
}
v_reusejp_1000_:
{
return v___x_1001_;
}
}
else
{
lean_object* v___x_1004_; 
lean_dec(v_value_993_);
lean_dec(v_key_992_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 1, v_b_990_);
lean_ctor_set(v___x_996_, 0, v_a_989_);
v___x_1004_ = v___x_996_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v_a_989_);
lean_ctor_set(v_reuseFailAlloc_1005_, 1, v_b_990_);
lean_ctor_set(v_reuseFailAlloc_1005_, 2, v_tail_994_);
v___x_1004_ = v_reuseFailAlloc_1005_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
return v___x_1004_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(lean_object* v_a_1007_, lean_object* v_x_1008_){
_start:
{
if (lean_obj_tag(v_x_1008_) == 0)
{
uint8_t v___x_1009_; 
v___x_1009_ = 0;
return v___x_1009_;
}
else
{
lean_object* v_key_1010_; lean_object* v_tail_1011_; uint8_t v___x_1012_; 
v_key_1010_ = lean_ctor_get(v_x_1008_, 0);
v_tail_1011_ = lean_ctor_get(v_x_1008_, 2);
v___x_1012_ = l_Lean_Syntax_instBEqRange_beq(v_key_1010_, v_a_1007_);
if (v___x_1012_ == 0)
{
v_x_1008_ = v_tail_1011_;
goto _start;
}
else
{
return v___x_1012_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg___boxed(lean_object* v_a_1014_, lean_object* v_x_1015_){
_start:
{
uint8_t v_res_1016_; lean_object* v_r_1017_; 
v_res_1016_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(v_a_1014_, v_x_1015_);
lean_dec(v_x_1015_);
lean_dec_ref(v_a_1014_);
v_r_1017_ = lean_box(v_res_1016_);
return v_r_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3___redArg(lean_object* v_m_1018_, lean_object* v_a_1019_, lean_object* v_b_1020_){
_start:
{
lean_object* v_size_1021_; lean_object* v_buckets_1022_; lean_object* v___x_1024_; uint8_t v_isShared_1025_; uint8_t v_isSharedCheck_1065_; 
v_size_1021_ = lean_ctor_get(v_m_1018_, 0);
v_buckets_1022_ = lean_ctor_get(v_m_1018_, 1);
v_isSharedCheck_1065_ = !lean_is_exclusive(v_m_1018_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1024_ = v_m_1018_;
v_isShared_1025_ = v_isSharedCheck_1065_;
goto v_resetjp_1023_;
}
else
{
lean_inc(v_buckets_1022_);
lean_inc(v_size_1021_);
lean_dec(v_m_1018_);
v___x_1024_ = lean_box(0);
v_isShared_1025_ = v_isSharedCheck_1065_;
goto v_resetjp_1023_;
}
v_resetjp_1023_:
{
lean_object* v___x_1026_; uint64_t v___x_1027_; uint64_t v___x_1028_; uint64_t v___x_1029_; uint64_t v_fold_1030_; uint64_t v___x_1031_; uint64_t v___x_1032_; uint64_t v___x_1033_; size_t v___x_1034_; size_t v___x_1035_; size_t v___x_1036_; size_t v___x_1037_; size_t v___x_1038_; lean_object* v_bkt_1039_; uint8_t v___x_1040_; 
v___x_1026_ = lean_array_get_size(v_buckets_1022_);
v___x_1027_ = l_Lean_Syntax_instHashableRange_hash(v_a_1019_);
v___x_1028_ = 32ULL;
v___x_1029_ = lean_uint64_shift_right(v___x_1027_, v___x_1028_);
v_fold_1030_ = lean_uint64_xor(v___x_1027_, v___x_1029_);
v___x_1031_ = 16ULL;
v___x_1032_ = lean_uint64_shift_right(v_fold_1030_, v___x_1031_);
v___x_1033_ = lean_uint64_xor(v_fold_1030_, v___x_1032_);
v___x_1034_ = lean_uint64_to_usize(v___x_1033_);
v___x_1035_ = lean_usize_of_nat(v___x_1026_);
v___x_1036_ = ((size_t)1ULL);
v___x_1037_ = lean_usize_sub(v___x_1035_, v___x_1036_);
v___x_1038_ = lean_usize_land(v___x_1034_, v___x_1037_);
v_bkt_1039_ = lean_array_uget_borrowed(v_buckets_1022_, v___x_1038_);
v___x_1040_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(v_a_1019_, v_bkt_1039_);
if (v___x_1040_ == 0)
{
lean_object* v___x_1041_; lean_object* v_size_x27_1042_; lean_object* v___x_1043_; lean_object* v_buckets_x27_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; uint8_t v___x_1050_; 
v___x_1041_ = lean_unsigned_to_nat(1u);
v_size_x27_1042_ = lean_nat_add(v_size_1021_, v___x_1041_);
lean_dec(v_size_1021_);
lean_inc(v_bkt_1039_);
v___x_1043_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1043_, 0, v_a_1019_);
lean_ctor_set(v___x_1043_, 1, v_b_1020_);
lean_ctor_set(v___x_1043_, 2, v_bkt_1039_);
v_buckets_x27_1044_ = lean_array_uset(v_buckets_1022_, v___x_1038_, v___x_1043_);
v___x_1045_ = lean_unsigned_to_nat(4u);
v___x_1046_ = lean_nat_mul(v_size_x27_1042_, v___x_1045_);
v___x_1047_ = lean_unsigned_to_nat(3u);
v___x_1048_ = lean_nat_div(v___x_1046_, v___x_1047_);
lean_dec(v___x_1046_);
v___x_1049_ = lean_array_get_size(v_buckets_x27_1044_);
v___x_1050_ = lean_nat_dec_le(v___x_1048_, v___x_1049_);
lean_dec(v___x_1048_);
if (v___x_1050_ == 0)
{
lean_object* v_val_1051_; lean_object* v___x_1053_; 
v_val_1051_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1___redArg(v_buckets_x27_1044_);
if (v_isShared_1025_ == 0)
{
lean_ctor_set(v___x_1024_, 1, v_val_1051_);
lean_ctor_set(v___x_1024_, 0, v_size_x27_1042_);
v___x_1053_ = v___x_1024_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v_size_x27_1042_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v_val_1051_);
v___x_1053_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
return v___x_1053_;
}
}
else
{
lean_object* v___x_1056_; 
if (v_isShared_1025_ == 0)
{
lean_ctor_set(v___x_1024_, 1, v_buckets_x27_1044_);
lean_ctor_set(v___x_1024_, 0, v_size_x27_1042_);
v___x_1056_ = v___x_1024_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_size_x27_1042_);
lean_ctor_set(v_reuseFailAlloc_1057_, 1, v_buckets_x27_1044_);
v___x_1056_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
return v___x_1056_;
}
}
}
else
{
lean_object* v___x_1058_; lean_object* v_buckets_x27_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1063_; 
lean_inc(v_bkt_1039_);
v___x_1058_ = lean_box(0);
v_buckets_x27_1059_ = lean_array_uset(v_buckets_1022_, v___x_1038_, v___x_1058_);
v___x_1060_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5___redArg(v_a_1019_, v_b_1020_, v_bkt_1039_);
v___x_1061_ = lean_array_uset(v_buckets_x27_1059_, v___x_1038_, v___x_1060_);
if (v_isShared_1025_ == 0)
{
lean_ctor_set(v___x_1024_, 1, v___x_1061_);
v___x_1063_ = v___x_1024_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1064_; 
v_reuseFailAlloc_1064_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1064_, 0, v_size_1021_);
lean_ctor_set(v_reuseFailAlloc_1064_, 1, v___x_1061_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__4(lean_object* v_a_1066_, lean_object* v_a_1067_){
_start:
{
if (lean_obj_tag(v_a_1066_) == 0)
{
lean_object* v___x_1068_; 
v___x_1068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1068_, 0, v_a_1067_);
return v___x_1068_;
}
else
{
lean_object* v_key_1069_; lean_object* v_value_1070_; lean_object* v_tail_1071_; lean_object* v_r_1072_; 
v_key_1069_ = lean_ctor_get(v_a_1066_, 0);
lean_inc(v_key_1069_);
v_value_1070_ = lean_ctor_get(v_a_1066_, 1);
lean_inc(v_value_1070_);
v_tail_1071_ = lean_ctor_get(v_a_1066_, 2);
lean_inc(v_tail_1071_);
lean_dec_ref_known(v_a_1066_, 3);
v_r_1072_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3___redArg(v_a_1067_, v_key_1069_, v_value_1070_);
v_a_1066_ = v_tail_1071_;
v_a_1067_ = v_r_1072_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5(lean_object* v_as_1074_, size_t v_sz_1075_, size_t v_i_1076_, lean_object* v_b_1077_){
_start:
{
uint8_t v___x_1078_; 
v___x_1078_ = lean_usize_dec_lt(v_i_1076_, v_sz_1075_);
if (v___x_1078_ == 0)
{
return v_b_1077_;
}
else
{
lean_object* v_a_1079_; lean_object* v___x_1080_; 
v_a_1079_ = lean_array_uget_borrowed(v_as_1074_, v_i_1076_);
lean_inc(v_a_1079_);
v___x_1080_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__4(v_a_1079_, v_b_1077_);
if (lean_obj_tag(v___x_1080_) == 0)
{
lean_object* v_a_1081_; 
v_a_1081_ = lean_ctor_get(v___x_1080_, 0);
lean_inc(v_a_1081_);
lean_dec_ref_known(v___x_1080_, 1);
return v_a_1081_;
}
else
{
lean_object* v_a_1082_; size_t v___x_1083_; size_t v___x_1084_; 
v_a_1082_ = lean_ctor_get(v___x_1080_, 0);
lean_inc(v_a_1082_);
lean_dec_ref_known(v___x_1080_, 1);
v___x_1083_ = ((size_t)1ULL);
v___x_1084_ = lean_usize_add(v_i_1076_, v___x_1083_);
v_i_1076_ = v___x_1084_;
v_b_1077_ = v_a_1082_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5___boxed(lean_object* v_as_1086_, lean_object* v_sz_1087_, lean_object* v_i_1088_, lean_object* v_b_1089_){
_start:
{
size_t v_sz_boxed_1090_; size_t v_i_boxed_1091_; lean_object* v_res_1092_; 
v_sz_boxed_1090_ = lean_unbox_usize(v_sz_1087_);
lean_dec(v_sz_1087_);
v_i_boxed_1091_ = lean_unbox_usize(v_i_1088_);
lean_dec(v_i_1088_);
v_res_1092_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5(v_as_1086_, v_sz_boxed_1090_, v_i_boxed_1091_, v_b_1089_);
lean_dec_ref(v_as_1086_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1(lean_object* v_m_1093_, lean_object* v_l_1094_){
_start:
{
lean_object* v_buckets_1095_; size_t v_sz_1096_; size_t v___x_1097_; lean_object* v___x_1098_; 
v_buckets_1095_ = lean_ctor_get(v_l_1094_, 1);
v_sz_1096_ = lean_array_size(v_buckets_1095_);
v___x_1097_ = ((size_t)0ULL);
v___x_1098_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__5(v_buckets_1095_, v_sz_1096_, v___x_1097_, v_m_1093_);
return v___x_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1___boxed(lean_object* v_m_1099_, lean_object* v_l_1100_){
_start:
{
lean_object* v_res_1101_; 
v_res_1101_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1(v_m_1099_, v_l_1100_);
lean_dec_ref(v_l_1100_);
return v_res_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(lean_object* v_m_1102_, lean_object* v_a_1103_, lean_object* v_b_1104_){
_start:
{
lean_object* v_size_1105_; lean_object* v_buckets_1106_; lean_object* v___x_1107_; uint64_t v___x_1108_; uint64_t v___x_1109_; uint64_t v___x_1110_; uint64_t v_fold_1111_; uint64_t v___x_1112_; uint64_t v___x_1113_; uint64_t v___x_1114_; size_t v___x_1115_; size_t v___x_1116_; size_t v___x_1117_; size_t v___x_1118_; size_t v___x_1119_; lean_object* v_bkt_1120_; uint8_t v___x_1121_; 
v_size_1105_ = lean_ctor_get(v_m_1102_, 0);
v_buckets_1106_ = lean_ctor_get(v_m_1102_, 1);
v___x_1107_ = lean_array_get_size(v_buckets_1106_);
v___x_1108_ = l_Lean_Syntax_instHashableRange_hash(v_a_1103_);
v___x_1109_ = 32ULL;
v___x_1110_ = lean_uint64_shift_right(v___x_1108_, v___x_1109_);
v_fold_1111_ = lean_uint64_xor(v___x_1108_, v___x_1110_);
v___x_1112_ = 16ULL;
v___x_1113_ = lean_uint64_shift_right(v_fold_1111_, v___x_1112_);
v___x_1114_ = lean_uint64_xor(v_fold_1111_, v___x_1113_);
v___x_1115_ = lean_uint64_to_usize(v___x_1114_);
v___x_1116_ = lean_usize_of_nat(v___x_1107_);
v___x_1117_ = ((size_t)1ULL);
v___x_1118_ = lean_usize_sub(v___x_1116_, v___x_1117_);
v___x_1119_ = lean_usize_land(v___x_1115_, v___x_1118_);
v_bkt_1120_ = lean_array_uget_borrowed(v_buckets_1106_, v___x_1119_);
v___x_1121_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(v_a_1103_, v_bkt_1120_);
if (v___x_1121_ == 0)
{
lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1142_; 
lean_inc_ref(v_buckets_1106_);
lean_inc(v_size_1105_);
v_isSharedCheck_1142_ = !lean_is_exclusive(v_m_1102_);
if (v_isSharedCheck_1142_ == 0)
{
lean_object* v_unused_1143_; lean_object* v_unused_1144_; 
v_unused_1143_ = lean_ctor_get(v_m_1102_, 1);
lean_dec(v_unused_1143_);
v_unused_1144_ = lean_ctor_get(v_m_1102_, 0);
lean_dec(v_unused_1144_);
v___x_1123_ = v_m_1102_;
v_isShared_1124_ = v_isSharedCheck_1142_;
goto v_resetjp_1122_;
}
else
{
lean_dec(v_m_1102_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1142_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1125_; lean_object* v_size_x27_1126_; lean_object* v___x_1127_; lean_object* v_buckets_x27_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; uint8_t v___x_1134_; 
v___x_1125_ = lean_unsigned_to_nat(1u);
v_size_x27_1126_ = lean_nat_add(v_size_1105_, v___x_1125_);
lean_dec(v_size_1105_);
lean_inc(v_bkt_1120_);
v___x_1127_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1127_, 0, v_a_1103_);
lean_ctor_set(v___x_1127_, 1, v_b_1104_);
lean_ctor_set(v___x_1127_, 2, v_bkt_1120_);
v_buckets_x27_1128_ = lean_array_uset(v_buckets_1106_, v___x_1119_, v___x_1127_);
v___x_1129_ = lean_unsigned_to_nat(4u);
v___x_1130_ = lean_nat_mul(v_size_x27_1126_, v___x_1129_);
v___x_1131_ = lean_unsigned_to_nat(3u);
v___x_1132_ = lean_nat_div(v___x_1130_, v___x_1131_);
lean_dec(v___x_1130_);
v___x_1133_ = lean_array_get_size(v_buckets_x27_1128_);
v___x_1134_ = lean_nat_dec_le(v___x_1132_, v___x_1133_);
lean_dec(v___x_1132_);
if (v___x_1134_ == 0)
{
lean_object* v_val_1135_; lean_object* v___x_1137_; 
v_val_1135_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1___redArg(v_buckets_x27_1128_);
if (v_isShared_1124_ == 0)
{
lean_ctor_set(v___x_1123_, 1, v_val_1135_);
lean_ctor_set(v___x_1123_, 0, v_size_x27_1126_);
v___x_1137_ = v___x_1123_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_size_x27_1126_);
lean_ctor_set(v_reuseFailAlloc_1138_, 1, v_val_1135_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
else
{
lean_object* v___x_1140_; 
if (v_isShared_1124_ == 0)
{
lean_ctor_set(v___x_1123_, 1, v_buckets_x27_1128_);
lean_ctor_set(v___x_1123_, 0, v_size_x27_1126_);
v___x_1140_ = v___x_1123_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_size_x27_1126_);
lean_ctor_set(v_reuseFailAlloc_1141_, 1, v_buckets_x27_1128_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
else
{
lean_dec(v_b_1104_);
lean_dec_ref(v_a_1103_);
return v_m_1102_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__2(lean_object* v_a_1145_, lean_object* v_a_1146_){
_start:
{
if (lean_obj_tag(v_a_1145_) == 0)
{
lean_object* v___x_1147_; 
v___x_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1147_, 0, v_a_1146_);
return v___x_1147_;
}
else
{
lean_object* v_key_1148_; lean_object* v_value_1149_; lean_object* v_tail_1150_; lean_object* v_r_1151_; 
v_key_1148_ = lean_ctor_get(v_a_1145_, 0);
lean_inc(v_key_1148_);
v_value_1149_ = lean_ctor_get(v_a_1145_, 1);
lean_inc(v_value_1149_);
v_tail_1150_ = lean_ctor_get(v_a_1145_, 2);
lean_inc(v_tail_1150_);
lean_dec_ref_known(v_a_1145_, 3);
v_r_1151_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(v_a_1146_, v_key_1148_, v_value_1149_);
v_a_1145_ = v_tail_1150_;
v_a_1146_ = v_r_1151_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3(lean_object* v_as_1153_, size_t v_sz_1154_, size_t v_i_1155_, lean_object* v_b_1156_){
_start:
{
uint8_t v___x_1157_; 
v___x_1157_ = lean_usize_dec_lt(v_i_1155_, v_sz_1154_);
if (v___x_1157_ == 0)
{
return v_b_1156_;
}
else
{
lean_object* v_a_1158_; lean_object* v___x_1159_; 
v_a_1158_ = lean_array_uget_borrowed(v_as_1153_, v_i_1155_);
lean_inc(v_a_1158_);
v___x_1159_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__2(v_a_1158_, v_b_1156_);
if (lean_obj_tag(v___x_1159_) == 0)
{
lean_object* v_a_1160_; 
v_a_1160_ = lean_ctor_get(v___x_1159_, 0);
lean_inc(v_a_1160_);
lean_dec_ref_known(v___x_1159_, 1);
return v_a_1160_;
}
else
{
lean_object* v_a_1161_; size_t v___x_1162_; size_t v___x_1163_; 
v_a_1161_ = lean_ctor_get(v___x_1159_, 0);
lean_inc(v_a_1161_);
lean_dec_ref_known(v___x_1159_, 1);
v___x_1162_ = ((size_t)1ULL);
v___x_1163_ = lean_usize_add(v_i_1155_, v___x_1162_);
v_i_1155_ = v___x_1163_;
v_b_1156_ = v_a_1161_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3___boxed(lean_object* v_as_1165_, lean_object* v_sz_1166_, lean_object* v_i_1167_, lean_object* v_b_1168_){
_start:
{
size_t v_sz_boxed_1169_; size_t v_i_boxed_1170_; lean_object* v_res_1171_; 
v_sz_boxed_1169_ = lean_unbox_usize(v_sz_1166_);
lean_dec(v_sz_1166_);
v_i_boxed_1170_ = lean_unbox_usize(v_i_1167_);
lean_dec(v_i_1167_);
v_res_1171_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3(v_as_1165_, v_sz_boxed_1169_, v_i_boxed_1170_, v_b_1168_);
lean_dec_ref(v_as_1165_);
return v_res_1171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges(lean_object* v_a_1173_, lean_object* v_x_1174_, lean_object* v_x_1175_){
_start:
{
lean_object* v___y_1177_; lean_object* v___y_1178_; 
switch(lean_obj_tag(v_x_1175_))
{
case 1:
{
lean_object* v_kind_1181_; lean_object* v_args_1182_; lean_object* v___y_1184_; lean_object* v___x_1190_; lean_object* v___x_1191_; uint8_t v___x_1192_; 
v_kind_1181_ = lean_ctor_get(v_x_1175_, 1);
v_args_1182_ = lean_ctor_get(v_x_1175_, 2);
v___x_1190_ = lean_unsigned_to_nat(0u);
v___x_1191_ = lean_array_get_size(v_args_1182_);
v___x_1192_ = lean_nat_dec_lt(v___x_1190_, v___x_1191_);
if (v___x_1192_ == 0)
{
v___y_1184_ = v_x_1174_;
goto v___jp_1183_;
}
else
{
uint8_t v___x_1193_; 
v___x_1193_ = lean_nat_dec_le(v___x_1191_, v___x_1191_);
if (v___x_1193_ == 0)
{
if (v___x_1192_ == 0)
{
v___y_1184_ = v_x_1174_;
goto v___jp_1183_;
}
else
{
size_t v___x_1194_; size_t v___x_1195_; lean_object* v___x_1196_; 
v___x_1194_ = ((size_t)0ULL);
v___x_1195_ = lean_usize_of_nat(v___x_1191_);
lean_inc_ref(v_x_1174_);
v___x_1196_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4(v_a_1173_, v_x_1174_, v_args_1182_, v___x_1194_, v___x_1195_, v_x_1174_);
v___y_1184_ = v___x_1196_;
goto v___jp_1183_;
}
}
else
{
size_t v___x_1197_; size_t v___x_1198_; lean_object* v___x_1199_; 
v___x_1197_ = ((size_t)0ULL);
v___x_1198_ = lean_usize_of_nat(v___x_1191_);
lean_inc_ref(v_x_1174_);
v___x_1199_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4(v_a_1173_, v_x_1174_, v_args_1182_, v___x_1197_, v___x_1198_, v_x_1174_);
v___y_1184_ = v___x_1199_;
goto v___jp_1183_;
}
}
v___jp_1183_:
{
uint8_t v___x_1185_; 
v___x_1185_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos_spec__0(v_a_1173_, v_kind_1181_);
if (v___x_1185_ == 0)
{
return v___y_1184_;
}
else
{
uint8_t v___x_1186_; lean_object* v___x_1187_; 
v___x_1186_ = 0;
v___x_1187_ = l_Lean_Syntax_getRange_x3f(v_x_1175_, v___x_1186_);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v___x_1188_; 
v___x_1188_ = l_Lean_Syntax_instInhabitedRange_default;
v___y_1177_ = v___y_1184_;
v___y_1178_ = v___x_1188_;
goto v___jp_1176_;
}
else
{
lean_object* v_val_1189_; 
v_val_1189_ = lean_ctor_get(v___x_1187_, 0);
lean_inc(v_val_1189_);
lean_dec_ref_known(v___x_1187_, 1);
v___y_1177_ = v___y_1184_;
v___y_1178_ = v_val_1189_;
goto v___jp_1176_;
}
}
}
}
case 2:
{
lean_object* v_info_1200_; lean_object* v_val_1201_; lean_object* v___x_1202_; uint8_t v___x_1203_; 
v_info_1200_ = lean_ctor_get(v_x_1175_, 0);
v_val_1201_ = lean_ctor_get(v_x_1175_, 1);
v___x_1202_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___closed__0));
v___x_1203_ = lean_string_dec_eq(v_val_1201_, v___x_1202_);
if (v___x_1203_ == 0)
{
return v_x_1174_;
}
else
{
uint8_t v___x_1204_; lean_object* v___x_1205_; 
v___x_1204_ = 0;
v___x_1205_ = l_Lean_SourceInfo_getRangeWithTrailing_x3f(v___x_1204_, v_info_1200_);
if (lean_obj_tag(v___x_1205_) == 1)
{
lean_object* v_val_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; 
v_val_1206_ = lean_ctor_get(v___x_1205_, 0);
lean_inc(v_val_1206_);
lean_dec_ref_known(v___x_1205_, 1);
v___x_1207_ = lean_box(0);
v___x_1208_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(v_x_1174_, v_val_1206_, v___x_1207_);
return v___x_1208_;
}
else
{
lean_dec(v___x_1205_);
return v_x_1174_;
}
}
}
default: 
{
return v_x_1174_;
}
}
v___jp_1176_:
{
lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1179_ = lean_box(0);
v___x_1180_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(v___y_1177_, v___y_1178_, v___x_1179_);
return v___x_1180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4(lean_object* v_a_1209_, lean_object* v_x_1210_, lean_object* v_as_1211_, size_t v_i_1212_, size_t v_stop_1213_, lean_object* v_b_1214_){
_start:
{
lean_object* v___y_1216_; uint8_t v___x_1220_; 
v___x_1220_ = lean_usize_dec_eq(v_i_1212_, v_stop_1213_);
if (v___x_1220_ == 0)
{
lean_object* v_size_1221_; lean_object* v_buckets_1222_; lean_object* v___x_1223_; lean_object* v_r_1224_; lean_object* v_size_1225_; uint8_t v___x_1226_; 
v_size_1221_ = lean_ctor_get(v_b_1214_, 0);
v_buckets_1222_ = lean_ctor_get(v_b_1214_, 1);
v___x_1223_ = lean_array_uget_borrowed(v_as_1211_, v_i_1212_);
lean_inc_ref(v_x_1210_);
v_r_1224_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges(v_a_1209_, v_x_1210_, v___x_1223_);
v_size_1225_ = lean_ctor_get(v_r_1224_, 0);
lean_inc(v_size_1225_);
v___x_1226_ = lean_nat_dec_le(v_size_1221_, v_size_1225_);
lean_dec(v_size_1225_);
if (v___x_1226_ == 0)
{
lean_object* v___x_1227_; 
v___x_1227_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1(v_b_1214_, v_r_1224_);
lean_dec_ref(v_r_1224_);
v___y_1216_ = v___x_1227_;
goto v___jp_1215_;
}
else
{
size_t v_sz_1228_; size_t v___x_1229_; lean_object* v___x_1230_; 
lean_inc_ref(v_buckets_1222_);
lean_dec_ref(v_b_1214_);
v_sz_1228_ = lean_array_size(v_buckets_1222_);
v___x_1229_ = ((size_t)0ULL);
v___x_1230_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__3(v_buckets_1222_, v_sz_1228_, v___x_1229_, v_r_1224_);
lean_dec_ref(v_buckets_1222_);
v___y_1216_ = v___x_1230_;
goto v___jp_1215_;
}
}
else
{
lean_dec_ref(v_x_1210_);
return v_b_1214_;
}
v___jp_1215_:
{
size_t v___x_1217_; size_t v___x_1218_; 
v___x_1217_ = ((size_t)1ULL);
v___x_1218_ = lean_usize_add(v_i_1212_, v___x_1217_);
v_i_1212_ = v___x_1218_;
v_b_1214_ = v___y_1216_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4___boxed(lean_object* v_a_1231_, lean_object* v_x_1232_, lean_object* v_as_1233_, lean_object* v_i_1234_, lean_object* v_stop_1235_, lean_object* v_b_1236_){
_start:
{
size_t v_i_boxed_1237_; size_t v_stop_boxed_1238_; lean_object* v_res_1239_; 
v_i_boxed_1237_ = lean_unbox_usize(v_i_1234_);
lean_dec(v_i_1234_);
v_stop_boxed_1238_ = lean_unbox_usize(v_stop_1235_);
lean_dec(v_stop_1235_);
v_res_1239_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__4(v_a_1231_, v_x_1232_, v_as_1233_, v_i_boxed_1237_, v_stop_boxed_1238_, v_b_1236_);
lean_dec_ref(v_as_1233_);
lean_dec_ref(v_a_1231_);
return v_res_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges___boxed(lean_object* v_a_1240_, lean_object* v_x_1241_, lean_object* v_x_1242_){
_start:
{
lean_object* v_res_1243_; 
v_res_1243_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges(v_a_1240_, v_x_1241_, v_x_1242_);
lean_dec(v_x_1242_);
lean_dec_ref(v_a_1240_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0(lean_object* v_00_u03b2_1244_, lean_object* v_m_1245_, lean_object* v_a_1246_, lean_object* v_b_1247_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0___redArg(v_m_1245_, v_a_1246_, v_b_1247_);
return v___x_1248_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0(lean_object* v_00_u03b2_1249_, lean_object* v_a_1250_, lean_object* v_x_1251_){
_start:
{
uint8_t v___x_1252_; 
v___x_1252_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___redArg(v_a_1250_, v_x_1251_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1253_, lean_object* v_a_1254_, lean_object* v_x_1255_){
_start:
{
uint8_t v_res_1256_; lean_object* v_r_1257_; 
v_res_1256_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__0(v_00_u03b2_1253_, v_a_1254_, v_x_1255_);
lean_dec(v_x_1255_);
lean_dec_ref(v_a_1254_);
v_r_1257_ = lean_box(v_res_1256_);
return v_r_1257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1(lean_object* v_00_u03b2_1258_, lean_object* v_data_1259_){
_start:
{
lean_object* v___x_1260_; 
v___x_1260_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1___redArg(v_data_1259_);
return v___x_1260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3(lean_object* v_00_u03b2_1261_, lean_object* v_m_1262_, lean_object* v_a_1263_, lean_object* v_b_1264_){
_start:
{
lean_object* v___x_1265_; 
v___x_1265_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3___redArg(v_m_1262_, v_a_1263_, v_b_1264_);
return v___x_1265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1266_, lean_object* v_i_1267_, lean_object* v_source_1268_, lean_object* v_target_1269_){
_start:
{
lean_object* v___x_1270_; 
v___x_1270_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2___redArg(v_i_1267_, v_source_1268_, v_target_1269_);
return v___x_1270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_1271_, lean_object* v_a_1272_, lean_object* v_b_1273_, lean_object* v_x_1274_){
_start:
{
lean_object* v___x_1275_; 
v___x_1275_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__1_spec__3_spec__5___redArg(v_a_1272_, v_b_1273_, v_x_1274_);
return v___x_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7(lean_object* v_00_u03b2_1276_, lean_object* v_x_1277_, lean_object* v_x_1278_){
_start:
{
lean_object* v___x_1279_; 
v___x_1279_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_spec__0_spec__1_spec__2_spec__7___redArg(v_x_1277_, v_x_1278_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg(lean_object* v_x_1281_, lean_object* v_x_1282_, lean_object* v_h__1_1283_, lean_object* v_h__2_1284_, lean_object* v_h__3_1285_){
_start:
{
switch(lean_obj_tag(v_x_1282_))
{
case 1:
{
lean_object* v_info_1286_; lean_object* v_kind_1287_; lean_object* v_args_1288_; lean_object* v___x_1289_; 
lean_dec(v_h__3_1285_);
lean_dec(v_h__2_1284_);
v_info_1286_ = lean_ctor_get(v_x_1282_, 0);
lean_inc(v_info_1286_);
v_kind_1287_ = lean_ctor_get(v_x_1282_, 1);
lean_inc(v_kind_1287_);
v_args_1288_ = lean_ctor_get(v_x_1282_, 2);
lean_inc_ref(v_args_1288_);
lean_dec_ref_known(v_x_1282_, 3);
v___x_1289_ = lean_apply_4(v_h__1_1283_, v_x_1281_, v_info_1286_, v_kind_1287_, v_args_1288_);
return v___x_1289_;
}
case 2:
{
lean_object* v_info_1290_; lean_object* v_val_1291_; lean_object* v___x_1292_; uint8_t v___x_1293_; 
lean_dec(v_h__1_1283_);
v_info_1290_ = lean_ctor_get(v_x_1282_, 0);
v_val_1291_ = lean_ctor_get(v_x_1282_, 1);
v___x_1292_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg___closed__0));
v___x_1293_ = lean_string_dec_eq(v_val_1291_, v___x_1292_);
if (v___x_1293_ == 0)
{
lean_object* v___x_1294_; 
lean_dec(v_h__2_1284_);
v___x_1294_ = lean_apply_4(v_h__3_1285_, v_x_1281_, v_x_1282_, lean_box(0), lean_box(0));
return v___x_1294_;
}
else
{
lean_object* v___x_1295_; 
lean_inc(v_info_1290_);
lean_dec_ref_known(v_x_1282_, 2);
lean_dec(v_h__3_1285_);
v___x_1295_ = lean_apply_2(v_h__2_1284_, v_x_1281_, v_info_1290_);
return v___x_1295_;
}
}
default: 
{
lean_object* v___x_1296_; 
lean_dec(v_h__2_1284_);
lean_dec(v_h__1_1283_);
v___x_1296_ = lean_apply_4(v_h__3_1285_, v_x_1281_, v_x_1282_, lean_box(0), lean_box(0));
return v___x_1296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter(lean_object* v_motive_1297_, lean_object* v_x_1298_, lean_object* v_x_1299_, lean_object* v_h__1_1300_, lean_object* v_h__2_1301_, lean_object* v_h__3_1302_){
_start:
{
switch(lean_obj_tag(v_x_1299_))
{
case 1:
{
lean_object* v_info_1303_; lean_object* v_kind_1304_; lean_object* v_args_1305_; lean_object* v___x_1306_; 
lean_dec(v_h__3_1302_);
lean_dec(v_h__2_1301_);
v_info_1303_ = lean_ctor_get(v_x_1299_, 0);
lean_inc(v_info_1303_);
v_kind_1304_ = lean_ctor_get(v_x_1299_, 1);
lean_inc(v_kind_1304_);
v_args_1305_ = lean_ctor_get(v_x_1299_, 2);
lean_inc_ref(v_args_1305_);
lean_dec_ref_known(v_x_1299_, 3);
v___x_1306_ = lean_apply_4(v_h__1_1300_, v_x_1298_, v_info_1303_, v_kind_1304_, v_args_1305_);
return v___x_1306_;
}
case 2:
{
lean_object* v_info_1307_; lean_object* v_val_1308_; lean_object* v___x_1309_; uint8_t v___x_1310_; 
lean_dec(v_h__1_1300_);
v_info_1307_ = lean_ctor_get(v_x_1299_, 0);
v_val_1308_ = lean_ctor_get(v_x_1299_, 1);
v___x_1309_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__3_splitter___redArg___closed__0));
v___x_1310_ = lean_string_dec_eq(v_val_1308_, v___x_1309_);
if (v___x_1310_ == 0)
{
lean_object* v___x_1311_; 
lean_dec(v_h__2_1301_);
v___x_1311_ = lean_apply_4(v_h__3_1302_, v_x_1298_, v_x_1299_, lean_box(0), lean_box(0));
return v___x_1311_;
}
else
{
lean_object* v___x_1312_; 
lean_inc(v_info_1307_);
lean_dec_ref_known(v_x_1299_, 2);
lean_dec(v_h__3_1302_);
v___x_1312_ = lean_apply_2(v_h__2_1301_, v_x_1298_, v_info_1307_);
return v___x_1312_;
}
}
default: 
{
lean_object* v___x_1313_; 
lean_dec(v_h__2_1301_);
lean_dec(v_h__1_1300_);
v___x_1313_ = lean_apply_4(v_h__3_1302_, v_x_1298_, v_x_1299_, lean_box(0), lean_box(0));
return v___x_1313_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Array_map__unattach_match__1_splitter___redArg(lean_object* v_x_1314_, lean_object* v_h__1_1315_){
_start:
{
lean_object* v___x_1316_; 
v___x_1316_ = lean_apply_2(v_h__1_1315_, v_x_1314_, lean_box(0));
return v___x_1316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Array_map__unattach_match__1_splitter(lean_object* v_00_u03b1_1317_, lean_object* v_P_1318_, lean_object* v_motive_1319_, lean_object* v_x_1320_, lean_object* v_h__1_1321_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lean_apply_2(v_h__1_1321_, v_x_1320_, lean_box(0));
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__1_splitter___redArg(lean_object* v_x_1323_, lean_object* v_h__1_1324_, lean_object* v_h__2_1325_){
_start:
{
if (lean_obj_tag(v_x_1323_) == 1)
{
lean_object* v_val_1326_; lean_object* v___x_1327_; 
lean_dec(v_h__2_1325_);
v_val_1326_ = lean_ctor_get(v_x_1323_, 0);
lean_inc(v_val_1326_);
lean_dec_ref_known(v_x_1323_, 1);
v___x_1327_ = lean_apply_1(v_h__1_1324_, v_val_1326_);
return v___x_1327_;
}
else
{
lean_object* v___x_1328_; 
lean_dec(v_h__1_1324_);
v___x_1328_ = lean_apply_2(v_h__2_1325_, v_x_1323_, lean_box(0));
return v___x_1328_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges_match__1_splitter(lean_object* v_motive_1329_, lean_object* v_x_1330_, lean_object* v_h__1_1331_, lean_object* v_h__2_1332_){
_start:
{
if (lean_obj_tag(v_x_1330_) == 1)
{
lean_object* v_val_1333_; lean_object* v___x_1334_; 
lean_dec(v_h__2_1332_);
v_val_1333_ = lean_ctor_get(v_x_1330_, 0);
lean_inc(v_val_1333_);
lean_dec_ref_known(v_x_1330_, 1);
v___x_1334_ = lean_apply_1(v_h__1_1331_, v_val_1333_);
return v___x_1334_;
}
else
{
lean_object* v___x_1335_; 
lean_dec(v_h__1_1331_);
v___x_1335_ = lean_apply_2(v_h__2_1332_, v_x_1330_, lean_box(0));
return v___x_1335_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0(lean_object* v_rg_1347_, lean_object* v_a_1348_, lean_object* v_a_1349_){
_start:
{
if (lean_obj_tag(v_a_1348_) == 0)
{
lean_object* v___x_1350_; 
v___x_1350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1350_, 0, v_a_1349_);
return v___x_1350_;
}
else
{
lean_object* v_key_1351_; lean_object* v_tail_1352_; lean_object* v_start_1353_; lean_object* v_stop_1354_; lean_object* v_start_1355_; lean_object* v_stop_1356_; lean_object* v___x_1357_; uint8_t v___y_1359_; uint8_t v___x_1362_; 
lean_dec_ref(v_a_1349_);
v_key_1351_ = lean_ctor_get(v_a_1348_, 0);
v_tail_1352_ = lean_ctor_get(v_a_1348_, 2);
v_start_1353_ = lean_ctor_get(v_key_1351_, 0);
v_stop_1354_ = lean_ctor_get(v_key_1351_, 1);
v_start_1355_ = lean_ctor_get(v_rg_1347_, 0);
v_stop_1356_ = lean_ctor_get(v_rg_1347_, 1);
v___x_1357_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__0));
v___x_1362_ = lean_nat_dec_le(v_start_1353_, v_start_1355_);
if (v___x_1362_ == 0)
{
v___y_1359_ = v___x_1362_;
goto v___jp_1358_;
}
else
{
uint8_t v___x_1363_; 
v___x_1363_ = lean_nat_dec_le(v_stop_1356_, v_stop_1354_);
v___y_1359_ = v___x_1363_;
goto v___jp_1358_;
}
v___jp_1358_:
{
if (v___y_1359_ == 0)
{
v_a_1348_ = v_tail_1352_;
v_a_1349_ = v___x_1357_;
goto _start;
}
else
{
lean_object* v___x_1361_; 
v___x_1361_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__3));
return v___x_1361_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___boxed(lean_object* v_rg_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_){
_start:
{
lean_object* v_res_1367_; 
v_res_1367_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0(v_rg_1364_, v_a_1365_, v_a_1366_);
lean_dec(v_a_1365_);
lean_dec_ref(v_rg_1364_);
return v_res_1367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1(lean_object* v_rg_1368_, lean_object* v_as_1369_, size_t v_sz_1370_, size_t v_i_1371_, lean_object* v_b_1372_){
_start:
{
uint8_t v___x_1373_; 
v___x_1373_ = lean_usize_dec_lt(v_i_1371_, v_sz_1370_);
if (v___x_1373_ == 0)
{
return v_b_1372_;
}
else
{
lean_object* v_a_1374_; lean_object* v___x_1375_; 
v_a_1374_ = lean_array_uget_borrowed(v_as_1369_, v_i_1371_);
v___x_1375_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0(v_rg_1368_, v_a_1374_, v_b_1372_);
if (lean_obj_tag(v___x_1375_) == 0)
{
lean_object* v_a_1376_; 
v_a_1376_ = lean_ctor_get(v___x_1375_, 0);
lean_inc(v_a_1376_);
lean_dec_ref_known(v___x_1375_, 1);
return v_a_1376_;
}
else
{
lean_object* v_a_1377_; size_t v___x_1378_; size_t v___x_1379_; 
v_a_1377_ = lean_ctor_get(v___x_1375_, 0);
lean_inc(v_a_1377_);
lean_dec_ref_known(v___x_1375_, 1);
v___x_1378_ = ((size_t)1ULL);
v___x_1379_ = lean_usize_add(v_i_1371_, v___x_1378_);
v_i_1371_ = v___x_1379_;
v_b_1372_ = v_a_1377_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1___boxed(lean_object* v_rg_1381_, lean_object* v_as_1382_, lean_object* v_sz_1383_, lean_object* v_i_1384_, lean_object* v_b_1385_){
_start:
{
size_t v_sz_boxed_1386_; size_t v_i_boxed_1387_; lean_object* v_res_1388_; 
v_sz_boxed_1386_ = lean_unbox_usize(v_sz_1383_);
lean_dec(v_sz_1383_);
v_i_boxed_1387_ = lean_unbox_usize(v_i_1384_);
lean_dec(v_i_1384_);
v_res_1388_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1(v_rg_1381_, v_as_1382_, v_sz_boxed_1386_, v_i_boxed_1387_, v_b_1385_);
lean_dec_ref(v_as_1382_);
lean_dec_ref(v_rg_1381_);
return v_res_1388_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside(lean_object* v_rgs_1389_, lean_object* v_rg_1390_){
_start:
{
lean_object* v_buckets_1391_; lean_object* v___x_1392_; size_t v_sz_1393_; size_t v___x_1394_; lean_object* v___x_1395_; lean_object* v_fst_1396_; 
v_buckets_1391_ = lean_ctor_get(v_rgs_1389_, 1);
v___x_1392_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__0___closed__0));
v_sz_1393_ = lean_array_size(v_buckets_1391_);
v___x_1394_ = ((size_t)0ULL);
v___x_1395_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside_spec__1(v_rg_1390_, v_buckets_1391_, v_sz_1393_, v___x_1394_, v___x_1392_);
v_fst_1396_ = lean_ctor_get(v___x_1395_, 0);
lean_inc(v_fst_1396_);
lean_dec_ref(v___x_1395_);
if (lean_obj_tag(v_fst_1396_) == 0)
{
uint8_t v___x_1397_; 
v___x_1397_ = 1;
return v___x_1397_;
}
else
{
lean_object* v_val_1398_; uint8_t v___x_1399_; 
v_val_1398_ = lean_ctor_get(v_fst_1396_, 0);
lean_inc(v_val_1398_);
lean_dec_ref_known(v_fst_1396_, 1);
v___x_1399_ = lean_unbox(v_val_1398_);
lean_dec(v_val_1398_);
return v___x_1399_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside___boxed(lean_object* v_rgs_1400_, lean_object* v_rg_1401_){
_start:
{
uint8_t v_res_1402_; lean_object* v_r_1403_; 
v_res_1402_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside(v_rgs_1400_, v_rg_1401_);
lean_dec_ref(v_rg_1401_);
lean_dec_ref(v_rgs_1400_);
v_r_1403_ = lean_box(v_res_1402_);
return v_r_1403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1(lean_object* v_s_1404_, lean_object* v_pos_1405_){
_start:
{
lean_object* v_str_1406_; lean_object* v_startInclusive_1407_; lean_object* v_endExclusive_1408_; lean_object* v___x_1409_; uint8_t v___y_1411_; lean_object* v___x_1417_; lean_object* v___x_1418_; uint8_t v___x_1419_; 
v_str_1406_ = lean_ctor_get(v_s_1404_, 0);
v_startInclusive_1407_ = lean_ctor_get(v_s_1404_, 1);
v_endExclusive_1408_ = lean_ctor_get(v_s_1404_, 2);
v___x_1409_ = lean_nat_add(v_startInclusive_1407_, v_pos_1405_);
v___x_1417_ = lean_unsigned_to_nat(0u);
v___x_1418_ = lean_nat_sub(v_endExclusive_1408_, v___x_1409_);
v___x_1419_ = lean_nat_dec_eq(v___x_1417_, v___x_1418_);
lean_dec(v___x_1418_);
if (v___x_1419_ == 0)
{
uint32_t v___x_1420_; uint8_t v___y_1422_; uint32_t v___x_1427_; uint8_t v___x_1428_; 
v___x_1420_ = lean_string_utf8_get_fast(v_str_1406_, v___x_1409_);
v___x_1427_ = 32;
v___x_1428_ = lean_uint32_dec_eq(v___x_1420_, v___x_1427_);
if (v___x_1428_ == 0)
{
uint32_t v___x_1429_; uint8_t v___x_1430_; 
v___x_1429_ = 9;
v___x_1430_ = lean_uint32_dec_eq(v___x_1420_, v___x_1429_);
v___y_1422_ = v___x_1430_;
goto v___jp_1421_;
}
else
{
v___y_1422_ = v___x_1428_;
goto v___jp_1421_;
}
v___jp_1421_:
{
if (v___y_1422_ == 0)
{
uint32_t v___x_1423_; uint8_t v___x_1424_; 
v___x_1423_ = 13;
v___x_1424_ = lean_uint32_dec_eq(v___x_1420_, v___x_1423_);
if (v___x_1424_ == 0)
{
uint32_t v___x_1425_; uint8_t v___x_1426_; 
v___x_1425_ = 10;
v___x_1426_ = lean_uint32_dec_eq(v___x_1420_, v___x_1425_);
v___y_1411_ = v___x_1426_;
goto v___jp_1410_;
}
else
{
v___y_1411_ = v___x_1424_;
goto v___jp_1410_;
}
}
else
{
lean_dec(v___x_1409_);
return v_pos_1405_;
}
}
}
else
{
lean_dec(v___x_1409_);
return v_pos_1405_;
}
v___jp_1410_:
{
if (v___y_1411_ == 0)
{
lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; uint8_t v___x_1415_; 
v___x_1412_ = lean_string_utf8_next_fast(v_str_1406_, v___x_1409_);
v___x_1413_ = lean_nat_sub(v___x_1412_, v___x_1409_);
lean_dec(v___x_1409_);
v___x_1414_ = lean_nat_add(v_pos_1405_, v___x_1413_);
lean_dec(v___x_1413_);
v___x_1415_ = lean_nat_dec_lt(v_pos_1405_, v___x_1414_);
if (v___x_1415_ == 0)
{
lean_dec(v___x_1414_);
return v_pos_1405_;
}
else
{
lean_dec(v_pos_1405_);
v_pos_1405_ = v___x_1414_;
goto _start;
}
}
else
{
lean_dec(v___x_1409_);
return v_pos_1405_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1___boxed(lean_object* v_s_1431_, lean_object* v_pos_1432_){
_start:
{
lean_object* v_res_1433_; 
v_res_1433_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1(v_s_1431_, v_pos_1432_);
lean_dec_ref(v_s_1431_);
return v_res_1433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0(lean_object* v_s_1434_, lean_object* v_pos_1435_){
_start:
{
lean_object* v_str_1436_; lean_object* v_startInclusive_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; uint8_t v___x_1441_; 
v_str_1436_ = lean_ctor_get(v_s_1434_, 0);
v_startInclusive_1437_ = lean_ctor_get(v_s_1434_, 1);
v___x_1438_ = lean_nat_add(v_startInclusive_1437_, v_pos_1435_);
v___x_1439_ = lean_nat_sub(v___x_1438_, v_startInclusive_1437_);
v___x_1440_ = lean_unsigned_to_nat(0u);
v___x_1441_ = lean_nat_dec_eq(v___x_1439_, v___x_1440_);
if (v___x_1441_ == 0)
{
lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; uint8_t v___y_1447_; lean_object* v___x_1450_; uint32_t v___x_1451_; uint8_t v___y_1453_; uint32_t v___x_1458_; uint8_t v___x_1459_; 
lean_inc(v_startInclusive_1437_);
lean_inc_ref(v_str_1436_);
v___x_1442_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1442_, 0, v_str_1436_);
lean_ctor_set(v___x_1442_, 1, v_startInclusive_1437_);
lean_ctor_set(v___x_1442_, 2, v___x_1438_);
v___x_1443_ = lean_unsigned_to_nat(1u);
v___x_1444_ = lean_nat_sub(v___x_1439_, v___x_1443_);
lean_dec(v___x_1439_);
v___x_1445_ = l_String_Slice_posLE(v___x_1442_, v___x_1444_);
lean_dec_ref_known(v___x_1442_, 3);
v___x_1450_ = lean_nat_add(v_startInclusive_1437_, v___x_1445_);
v___x_1451_ = lean_string_utf8_get_fast(v_str_1436_, v___x_1450_);
lean_dec(v___x_1450_);
v___x_1458_ = 32;
v___x_1459_ = lean_uint32_dec_eq(v___x_1451_, v___x_1458_);
if (v___x_1459_ == 0)
{
uint32_t v___x_1460_; uint8_t v___x_1461_; 
v___x_1460_ = 9;
v___x_1461_ = lean_uint32_dec_eq(v___x_1451_, v___x_1460_);
v___y_1453_ = v___x_1461_;
goto v___jp_1452_;
}
else
{
v___y_1453_ = v___x_1459_;
goto v___jp_1452_;
}
v___jp_1446_:
{
if (v___y_1447_ == 0)
{
uint8_t v___x_1448_; 
v___x_1448_ = lean_nat_dec_lt(v___x_1445_, v_pos_1435_);
if (v___x_1448_ == 0)
{
lean_dec(v___x_1445_);
return v_pos_1435_;
}
else
{
lean_dec(v_pos_1435_);
v_pos_1435_ = v___x_1445_;
goto _start;
}
}
else
{
lean_dec(v___x_1445_);
return v_pos_1435_;
}
}
v___jp_1452_:
{
if (v___y_1453_ == 0)
{
uint32_t v___x_1454_; uint8_t v___x_1455_; 
v___x_1454_ = 13;
v___x_1455_ = lean_uint32_dec_eq(v___x_1451_, v___x_1454_);
if (v___x_1455_ == 0)
{
uint32_t v___x_1456_; uint8_t v___x_1457_; 
v___x_1456_ = 10;
v___x_1457_ = lean_uint32_dec_eq(v___x_1451_, v___x_1456_);
v___y_1447_ = v___x_1457_;
goto v___jp_1446_;
}
else
{
v___y_1447_ = v___x_1455_;
goto v___jp_1446_;
}
}
else
{
lean_dec(v___x_1445_);
return v_pos_1435_;
}
}
}
else
{
lean_dec(v___x_1439_);
lean_dec(v___x_1438_);
return v_pos_1435_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0___boxed(lean_object* v_s_1462_, lean_object* v_pos_1463_){
_start:
{
lean_object* v_res_1464_; 
v_res_1464_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0(v_s_1462_, v_pos_1463_);
lean_dec_ref(v_s_1462_);
return v_res_1464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow(lean_object* v_orig_1465_, lean_object* v_start_1466_, lean_object* v_ctx_1467_){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v_head_1474_; lean_object* v_middle_1475_; lean_object* v___x_1476_; lean_object* v_headCtx_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v_tail_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; 
v___x_1468_ = lean_unsigned_to_nat(1u);
v___x_1469_ = lean_nat_add(v_start_1466_, v___x_1468_);
v___x_1470_ = lean_unsigned_to_nat(0u);
v___x_1471_ = lean_string_utf8_byte_size(v_orig_1465_);
lean_inc_ref_n(v_orig_1465_, 6);
v___x_1472_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1472_, 0, v_orig_1465_);
lean_ctor_set(v___x_1472_, 1, v___x_1470_);
lean_ctor_set(v___x_1472_, 2, v___x_1471_);
v___x_1473_ = l_String_Slice_Pos_prevn(v___x_1472_, v___x_1471_, v___x_1469_);
lean_dec_ref_known(v___x_1472_, 3);
lean_inc_n(v___x_1473_, 4);
v_head_1474_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_head_1474_, 0, v_orig_1465_);
lean_ctor_set(v_head_1474_, 1, v___x_1470_);
lean_ctor_set(v_head_1474_, 2, v___x_1473_);
v_middle_1475_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_middle_1475_, 0, v_orig_1465_);
lean_ctor_set(v_middle_1475_, 1, v___x_1473_);
lean_ctor_set(v_middle_1475_, 2, v___x_1471_);
v___x_1476_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__0(v_head_1474_, v___x_1473_);
lean_dec_ref_known(v_head_1474_, 3);
v_headCtx_1477_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_headCtx_1477_, 0, v_orig_1465_);
lean_ctor_set(v_headCtx_1477_, 1, v___x_1476_);
lean_ctor_set(v_headCtx_1477_, 2, v___x_1473_);
v___x_1478_ = l_String_Slice_Pos_nextn(v_middle_1475_, v___x_1470_, v_ctx_1467_);
lean_dec_ref_known(v_middle_1475_, 3);
v___x_1479_ = lean_nat_add(v___x_1473_, v___x_1478_);
lean_dec(v___x_1478_);
lean_inc_n(v___x_1479_, 2);
v___x_1480_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1480_, 0, v_orig_1465_);
lean_ctor_set(v___x_1480_, 1, v___x_1479_);
lean_ctor_set(v___x_1480_, 2, v___x_1471_);
v___x_1481_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_Linter_Style_Whitespace_mkWindow_spec__1(v___x_1480_, v___x_1470_);
lean_dec_ref_known(v___x_1480_, 3);
v___x_1482_ = lean_nat_add(v___x_1479_, v___x_1481_);
lean_dec(v___x_1481_);
v_tail_1483_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_tail_1483_, 0, v_orig_1465_);
lean_ctor_set(v_tail_1483_, 1, v___x_1479_);
lean_ctor_set(v_tail_1483_, 2, v___x_1482_);
v___x_1484_ = l_String_Slice_toString(v_headCtx_1477_);
lean_dec_ref_known(v_headCtx_1477_, 3);
v___x_1485_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1485_, 0, v_orig_1465_);
lean_ctor_set(v___x_1485_, 1, v___x_1473_);
lean_ctor_set(v___x_1485_, 2, v___x_1479_);
v___x_1486_ = l_String_Slice_toString(v___x_1485_);
lean_dec_ref_known(v___x_1485_, 3);
v___x_1487_ = lean_string_append(v___x_1484_, v___x_1486_);
lean_dec_ref(v___x_1486_);
v___x_1488_ = l_String_Slice_toString(v_tail_1483_);
lean_dec_ref_known(v_tail_1483_, 3);
v___x_1489_ = lean_string_append(v___x_1487_, v___x_1488_);
lean_dec_ref(v___x_1488_);
return v___x_1489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow___boxed(lean_object* v_orig_1490_, lean_object* v_start_1491_, lean_object* v_ctx_1492_){
_start:
{
lean_object* v_res_1493_; 
v_res_1493_ = lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow(v_orig_1490_, v_start_1491_, v_ctx_1492_);
lean_dec(v_start_1491_);
return v_res_1493_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0(lean_object* v_x_1498_){
_start:
{
lean_object* v___x_1499_; uint8_t v___x_1500_; 
v___x_1499_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___closed__1));
v___x_1500_ = l_Lean_Syntax_isOfKind(v_x_1498_, v___x_1499_);
return v___x_1500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0___boxed(lean_object* v_x_1501_){
_start:
{
uint8_t v_res_1502_; lean_object* v_r_1503_; 
v_res_1502_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__0(v_x_1501_);
v_r_1503_ = lean_box(v_res_1502_);
return v_r_1503_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1(lean_object* v_x_1510_){
_start:
{
lean_object* v___x_1511_; uint8_t v___x_1512_; 
v___x_1511_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___closed__1));
v___x_1512_ = l_Lean_Syntax_isOfKind(v_x_1510_, v___x_1511_);
return v___x_1512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1___boxed(lean_object* v_x_1513_){
_start:
{
uint8_t v_res_1514_; lean_object* v_r_1515_; 
v_res_1514_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__1(v_x_1513_);
v_r_1515_ = lean_box(v_res_1514_);
return v_r_1515_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2(lean_object* v_x_1516_){
_start:
{
lean_object* v___x_1517_; uint8_t v___x_1518_; 
v___x_1517_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__29));
v___x_1518_ = l_Lean_Syntax_isOfKind(v_x_1516_, v___x_1517_);
return v___x_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2___boxed(lean_object* v_x_1519_){
_start:
{
uint8_t v_res_1520_; lean_object* v_r_1521_; 
v_res_1520_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__2(v_x_1519_);
v_r_1521_ = lean_box(v_res_1520_);
return v_r_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3(lean_object* v___x_1522_, lean_object* v_stx_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_){
_start:
{
lean_object* v___x_1527_; 
v___x_1527_ = l_Lean_PrettyPrinter_ppCategory(v___x_1522_, v_stx_1523_, v___y_1524_, v___y_1525_);
if (lean_obj_tag(v___x_1527_) == 0)
{
lean_object* v_a_1528_; lean_object* v___x_1530_; uint8_t v_isShared_1531_; uint8_t v_isSharedCheck_1536_; 
v_a_1528_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1536_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1536_ == 0)
{
v___x_1530_ = v___x_1527_;
v_isShared_1531_ = v_isSharedCheck_1536_;
goto v_resetjp_1529_;
}
else
{
lean_inc(v_a_1528_);
lean_dec(v___x_1527_);
v___x_1530_ = lean_box(0);
v_isShared_1531_ = v_isSharedCheck_1536_;
goto v_resetjp_1529_;
}
v_resetjp_1529_:
{
lean_object* v___x_1532_; lean_object* v___x_1534_; 
v___x_1532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1532_, 0, v_a_1528_);
if (v_isShared_1531_ == 0)
{
lean_ctor_set(v___x_1530_, 0, v___x_1532_);
v___x_1534_ = v___x_1530_;
goto v_reusejp_1533_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v___x_1532_);
v___x_1534_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1533_;
}
v_reusejp_1533_:
{
return v___x_1534_;
}
}
}
else
{
lean_object* v_a_1537_; lean_object* v___x_1539_; uint8_t v_isShared_1540_; uint8_t v_isSharedCheck_1544_; 
v_a_1537_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1544_ == 0)
{
v___x_1539_ = v___x_1527_;
v_isShared_1540_ = v_isSharedCheck_1544_;
goto v_resetjp_1538_;
}
else
{
lean_inc(v_a_1537_);
lean_dec(v___x_1527_);
v___x_1539_ = lean_box(0);
v_isShared_1540_ = v_isSharedCheck_1544_;
goto v_resetjp_1538_;
}
v_resetjp_1538_:
{
lean_object* v___x_1542_; 
if (v_isShared_1540_ == 0)
{
v___x_1542_ = v___x_1539_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v_a_1537_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3___boxed(lean_object* v___x_1545_, lean_object* v_stx_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3(v___x_1545_, v_stx_1546_, v___y_1547_, v___y_1548_);
lean_dec(v___y_1548_);
lean_dec_ref(v___y_1547_);
return v_res_1550_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7(lean_object* v_opts_1551_, lean_object* v_opt_1552_){
_start:
{
lean_object* v_name_1553_; lean_object* v_defValue_1554_; lean_object* v_map_1555_; lean_object* v___x_1556_; 
v_name_1553_ = lean_ctor_get(v_opt_1552_, 0);
v_defValue_1554_ = lean_ctor_get(v_opt_1552_, 1);
v_map_1555_ = lean_ctor_get(v_opts_1551_, 0);
v___x_1556_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1555_, v_name_1553_);
if (lean_obj_tag(v___x_1556_) == 0)
{
uint8_t v___x_1557_; 
v___x_1557_ = lean_unbox(v_defValue_1554_);
return v___x_1557_;
}
else
{
lean_object* v_val_1558_; 
v_val_1558_ = lean_ctor_get(v___x_1556_, 0);
lean_inc(v_val_1558_);
lean_dec_ref_known(v___x_1556_, 1);
if (lean_obj_tag(v_val_1558_) == 1)
{
uint8_t v_v_1559_; 
v_v_1559_ = lean_ctor_get_uint8(v_val_1558_, 0);
lean_dec_ref_known(v_val_1558_, 0);
return v_v_1559_;
}
else
{
uint8_t v___x_1560_; 
lean_dec(v_val_1558_);
v___x_1560_ = lean_unbox(v_defValue_1554_);
return v___x_1560_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7___boxed(lean_object* v_opts_1561_, lean_object* v_opt_1562_){
_start:
{
uint8_t v_res_1563_; lean_object* v_r_1564_; 
v_res_1563_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7(v_opts_1561_, v_opt_1562_);
lean_dec_ref(v_opt_1562_);
lean_dec_ref(v_opts_1561_);
v_r_1564_ = lean_box(v_res_1563_);
return v_r_1564_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0(uint8_t v___y_1566_, uint8_t v_suppressElabErrors_1567_, lean_object* v_x_1568_){
_start:
{
if (lean_obj_tag(v_x_1568_) == 1)
{
lean_object* v_pre_1569_; 
v_pre_1569_ = lean_ctor_get(v_x_1568_, 0);
if (lean_obj_tag(v_pre_1569_) == 0)
{
lean_object* v_str_1570_; lean_object* v___x_1571_; uint8_t v___x_1572_; 
v_str_1570_ = lean_ctor_get(v_x_1568_, 1);
v___x_1571_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___closed__0));
v___x_1572_ = lean_string_dec_eq(v_str_1570_, v___x_1571_);
if (v___x_1572_ == 0)
{
return v___y_1566_;
}
else
{
return v_suppressElabErrors_1567_;
}
}
else
{
return v___y_1566_;
}
}
else
{
return v___y_1566_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___boxed(lean_object* v___y_1573_, lean_object* v_suppressElabErrors_1574_, lean_object* v_x_1575_){
_start:
{
uint8_t v___y_14660__boxed_1576_; uint8_t v_suppressElabErrors_boxed_1577_; uint8_t v_res_1578_; lean_object* v_r_1579_; 
v___y_14660__boxed_1576_ = lean_unbox(v___y_1573_);
v_suppressElabErrors_boxed_1577_ = lean_unbox(v_suppressElabErrors_1574_);
v_res_1578_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0(v___y_14660__boxed_1576_, v_suppressElabErrors_boxed_1577_, v_x_1575_);
lean_dec(v_x_1575_);
v_r_1579_ = lean_box(v_res_1578_);
return v_r_1579_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_1580_; 
v___x_1580_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1580_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_1581_; lean_object* v___x_1582_; 
v___x_1581_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__0);
v___x_1582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1582_, 0, v___x_1581_);
return v___x_1582_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; 
v___x_1583_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_1584_ = lean_unsigned_to_nat(0u);
v___x_1585_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1585_, 0, v___x_1584_);
lean_ctor_set(v___x_1585_, 1, v___x_1584_);
lean_ctor_set(v___x_1585_, 2, v___x_1584_);
lean_ctor_set(v___x_1585_, 3, v___x_1584_);
lean_ctor_set(v___x_1585_, 4, v___x_1583_);
lean_ctor_set(v___x_1585_, 5, v___x_1583_);
lean_ctor_set(v___x_1585_, 6, v___x_1583_);
lean_ctor_set(v___x_1585_, 7, v___x_1583_);
lean_ctor_set(v___x_1585_, 8, v___x_1583_);
lean_ctor_set(v___x_1585_, 9, v___x_1583_);
return v___x_1585_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; 
v___x_1586_ = lean_unsigned_to_nat(32u);
v___x_1587_ = lean_mk_empty_array_with_capacity(v___x_1586_);
v___x_1588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1588_, 0, v___x_1587_);
return v___x_1588_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; 
v___x_1589_ = ((size_t)5ULL);
v___x_1590_ = lean_unsigned_to_nat(0u);
v___x_1591_ = lean_unsigned_to_nat(32u);
v___x_1592_ = lean_mk_empty_array_with_capacity(v___x_1591_);
v___x_1593_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__3);
v___x_1594_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1594_, 0, v___x_1593_);
lean_ctor_set(v___x_1594_, 1, v___x_1592_);
lean_ctor_set(v___x_1594_, 2, v___x_1590_);
lean_ctor_set(v___x_1594_, 3, v___x_1590_);
lean_ctor_set_usize(v___x_1594_, 4, v___x_1589_);
return v___x_1594_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; 
v___x_1595_ = lean_box(1);
v___x_1596_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__4);
v___x_1597_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__1);
v___x_1598_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1598_, 0, v___x_1597_);
lean_ctor_set(v___x_1598_, 1, v___x_1596_);
lean_ctor_set(v___x_1598_, 2, v___x_1595_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg(lean_object* v_msgData_1599_, lean_object* v___y_1600_){
_start:
{
lean_object* v___x_1602_; lean_object* v_env_1603_; lean_object* v___x_1604_; lean_object* v_scopes_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v_opts_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
v___x_1602_ = lean_st_ref_get(v___y_1600_);
v_env_1603_ = lean_ctor_get(v___x_1602_, 0);
lean_inc_ref(v_env_1603_);
lean_dec(v___x_1602_);
v___x_1604_ = lean_st_ref_get(v___y_1600_);
v_scopes_1605_ = lean_ctor_get(v___x_1604_, 2);
lean_inc(v_scopes_1605_);
lean_dec(v___x_1604_);
v___x_1606_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1607_ = l_List_head_x21___redArg(v___x_1606_, v_scopes_1605_);
lean_dec(v_scopes_1605_);
v_opts_1608_ = lean_ctor_get(v___x_1607_, 1);
lean_inc_ref(v_opts_1608_);
lean_dec(v___x_1607_);
v___x_1609_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__2);
v___x_1610_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___closed__5);
v___x_1611_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1611_, 0, v_env_1603_);
lean_ctor_set(v___x_1611_, 1, v___x_1609_);
lean_ctor_set(v___x_1611_, 2, v___x_1610_);
lean_ctor_set(v___x_1611_, 3, v_opts_1608_);
v___x_1612_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1611_);
lean_ctor_set(v___x_1612_, 1, v_msgData_1599_);
v___x_1613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1613_, 0, v___x_1612_);
return v___x_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_msgData_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_){
_start:
{
lean_object* v_res_1617_; 
v_res_1617_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg(v_msgData_1614_, v___y_1615_);
lean_dec(v___y_1615_);
return v_res_1617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3(lean_object* v_ref_1618_, lean_object* v_msgData_1619_, uint8_t v_severity_1620_, uint8_t v_isSilent_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___y_1626_; uint8_t v___y_1627_; lean_object* v___y_1628_; lean_object* v___y_1629_; uint8_t v___y_1630_; lean_object* v___y_1631_; lean_object* v___y_1632_; lean_object* v___y_1633_; uint8_t v___y_1690_; uint8_t v___y_1691_; uint8_t v___y_1692_; lean_object* v___y_1693_; lean_object* v___y_1694_; uint8_t v___y_1718_; uint8_t v___y_1719_; lean_object* v___y_1720_; uint8_t v___y_1721_; lean_object* v___y_1722_; uint8_t v___y_1726_; uint8_t v___y_1727_; uint8_t v___y_1728_; uint8_t v___x_1743_; uint8_t v___y_1745_; uint8_t v___y_1746_; uint8_t v___y_1747_; uint8_t v___y_1749_; uint8_t v___x_1761_; 
v___x_1743_ = 2;
v___x_1761_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1620_, v___x_1743_);
if (v___x_1761_ == 0)
{
v___y_1749_ = v___x_1761_;
goto v___jp_1748_;
}
else
{
uint8_t v___x_1762_; 
lean_inc_ref(v_msgData_1619_);
v___x_1762_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1619_);
v___y_1749_ = v___x_1762_;
goto v___jp_1748_;
}
v___jp_1625_:
{
lean_object* v___x_1634_; 
v___x_1634_ = l_Lean_Elab_Command_getScope___redArg(v___y_1633_);
if (lean_obj_tag(v___x_1634_) == 0)
{
lean_object* v_a_1635_; lean_object* v___x_1636_; 
v_a_1635_ = lean_ctor_get(v___x_1634_, 0);
lean_inc(v_a_1635_);
lean_dec_ref_known(v___x_1634_, 1);
v___x_1636_ = l_Lean_Elab_Command_getScope___redArg(v___y_1633_);
if (lean_obj_tag(v___x_1636_) == 0)
{
lean_object* v_a_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1672_; 
v_a_1637_ = lean_ctor_get(v___x_1636_, 0);
v_isSharedCheck_1672_ = !lean_is_exclusive(v___x_1636_);
if (v_isSharedCheck_1672_ == 0)
{
v___x_1639_ = v___x_1636_;
v_isShared_1640_ = v_isSharedCheck_1672_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_a_1637_);
lean_dec(v___x_1636_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1672_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1641_; lean_object* v_currNamespace_1642_; lean_object* v_openDecls_1643_; lean_object* v_env_1644_; lean_object* v_messages_1645_; lean_object* v_scopes_1646_; lean_object* v_usedQuotCtxts_1647_; lean_object* v_nextMacroScope_1648_; lean_object* v_maxRecDepth_1649_; lean_object* v_ngen_1650_; lean_object* v_auxDeclNGen_1651_; lean_object* v_infoState_1652_; lean_object* v_traceState_1653_; lean_object* v_snapshotTasks_1654_; lean_object* v_prevLinterStates_1655_; lean_object* v___x_1657_; uint8_t v_isShared_1658_; uint8_t v_isSharedCheck_1671_; 
v___x_1641_ = lean_st_ref_take(v___y_1633_);
v_currNamespace_1642_ = lean_ctor_get(v_a_1635_, 2);
lean_inc(v_currNamespace_1642_);
lean_dec(v_a_1635_);
v_openDecls_1643_ = lean_ctor_get(v_a_1637_, 3);
lean_inc(v_openDecls_1643_);
lean_dec(v_a_1637_);
v_env_1644_ = lean_ctor_get(v___x_1641_, 0);
v_messages_1645_ = lean_ctor_get(v___x_1641_, 1);
v_scopes_1646_ = lean_ctor_get(v___x_1641_, 2);
v_usedQuotCtxts_1647_ = lean_ctor_get(v___x_1641_, 3);
v_nextMacroScope_1648_ = lean_ctor_get(v___x_1641_, 4);
v_maxRecDepth_1649_ = lean_ctor_get(v___x_1641_, 5);
v_ngen_1650_ = lean_ctor_get(v___x_1641_, 6);
v_auxDeclNGen_1651_ = lean_ctor_get(v___x_1641_, 7);
v_infoState_1652_ = lean_ctor_get(v___x_1641_, 8);
v_traceState_1653_ = lean_ctor_get(v___x_1641_, 9);
v_snapshotTasks_1654_ = lean_ctor_get(v___x_1641_, 10);
v_prevLinterStates_1655_ = lean_ctor_get(v___x_1641_, 11);
v_isSharedCheck_1671_ = !lean_is_exclusive(v___x_1641_);
if (v_isSharedCheck_1671_ == 0)
{
v___x_1657_ = v___x_1641_;
v_isShared_1658_ = v_isSharedCheck_1671_;
goto v_resetjp_1656_;
}
else
{
lean_inc(v_prevLinterStates_1655_);
lean_inc(v_snapshotTasks_1654_);
lean_inc(v_traceState_1653_);
lean_inc(v_infoState_1652_);
lean_inc(v_auxDeclNGen_1651_);
lean_inc(v_ngen_1650_);
lean_inc(v_maxRecDepth_1649_);
lean_inc(v_nextMacroScope_1648_);
lean_inc(v_usedQuotCtxts_1647_);
lean_inc(v_scopes_1646_);
lean_inc(v_messages_1645_);
lean_inc(v_env_1644_);
lean_dec(v___x_1641_);
v___x_1657_ = lean_box(0);
v_isShared_1658_ = v_isSharedCheck_1671_;
goto v_resetjp_1656_;
}
v_resetjp_1656_:
{
lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1664_; 
v___x_1659_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1659_, 0, v_currNamespace_1642_);
lean_ctor_set(v___x_1659_, 1, v_openDecls_1643_);
v___x_1660_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1660_, 0, v___x_1659_);
lean_ctor_set(v___x_1660_, 1, v___y_1632_);
lean_inc_ref(v___y_1626_);
lean_inc_ref(v___y_1628_);
v___x_1661_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1661_, 0, v___y_1628_);
lean_ctor_set(v___x_1661_, 1, v___y_1629_);
lean_ctor_set(v___x_1661_, 2, v___y_1631_);
lean_ctor_set(v___x_1661_, 3, v___y_1626_);
lean_ctor_set(v___x_1661_, 4, v___x_1660_);
lean_ctor_set_uint8(v___x_1661_, sizeof(void*)*5, v___y_1627_);
lean_ctor_set_uint8(v___x_1661_, sizeof(void*)*5 + 1, v___y_1630_);
lean_ctor_set_uint8(v___x_1661_, sizeof(void*)*5 + 2, v_isSilent_1621_);
v___x_1662_ = l_Lean_MessageLog_add(v___x_1661_, v_messages_1645_);
if (v_isShared_1658_ == 0)
{
lean_ctor_set(v___x_1657_, 1, v___x_1662_);
v___x_1664_ = v___x_1657_;
goto v_reusejp_1663_;
}
else
{
lean_object* v_reuseFailAlloc_1670_; 
v_reuseFailAlloc_1670_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1670_, 0, v_env_1644_);
lean_ctor_set(v_reuseFailAlloc_1670_, 1, v___x_1662_);
lean_ctor_set(v_reuseFailAlloc_1670_, 2, v_scopes_1646_);
lean_ctor_set(v_reuseFailAlloc_1670_, 3, v_usedQuotCtxts_1647_);
lean_ctor_set(v_reuseFailAlloc_1670_, 4, v_nextMacroScope_1648_);
lean_ctor_set(v_reuseFailAlloc_1670_, 5, v_maxRecDepth_1649_);
lean_ctor_set(v_reuseFailAlloc_1670_, 6, v_ngen_1650_);
lean_ctor_set(v_reuseFailAlloc_1670_, 7, v_auxDeclNGen_1651_);
lean_ctor_set(v_reuseFailAlloc_1670_, 8, v_infoState_1652_);
lean_ctor_set(v_reuseFailAlloc_1670_, 9, v_traceState_1653_);
lean_ctor_set(v_reuseFailAlloc_1670_, 10, v_snapshotTasks_1654_);
lean_ctor_set(v_reuseFailAlloc_1670_, 11, v_prevLinterStates_1655_);
v___x_1664_ = v_reuseFailAlloc_1670_;
goto v_reusejp_1663_;
}
v_reusejp_1663_:
{
lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1668_; 
v___x_1665_ = lean_st_ref_set(v___y_1633_, v___x_1664_);
v___x_1666_ = lean_box(0);
if (v_isShared_1640_ == 0)
{
lean_ctor_set(v___x_1639_, 0, v___x_1666_);
v___x_1668_ = v___x_1639_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v___x_1666_);
v___x_1668_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
return v___x_1668_;
}
}
}
}
}
else
{
lean_object* v_a_1673_; lean_object* v___x_1675_; uint8_t v_isShared_1676_; uint8_t v_isSharedCheck_1680_; 
lean_dec(v_a_1635_);
lean_dec_ref(v___y_1632_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1629_);
v_a_1673_ = lean_ctor_get(v___x_1636_, 0);
v_isSharedCheck_1680_ = !lean_is_exclusive(v___x_1636_);
if (v_isSharedCheck_1680_ == 0)
{
v___x_1675_ = v___x_1636_;
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
else
{
lean_inc(v_a_1673_);
lean_dec(v___x_1636_);
v___x_1675_ = lean_box(0);
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
v_resetjp_1674_:
{
lean_object* v___x_1678_; 
if (v_isShared_1676_ == 0)
{
v___x_1678_ = v___x_1675_;
goto v_reusejp_1677_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v_a_1673_);
v___x_1678_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1677_;
}
v_reusejp_1677_:
{
return v___x_1678_;
}
}
}
}
else
{
lean_object* v_a_1681_; lean_object* v___x_1683_; uint8_t v_isShared_1684_; uint8_t v_isSharedCheck_1688_; 
lean_dec_ref(v___y_1632_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1629_);
v_a_1681_ = lean_ctor_get(v___x_1634_, 0);
v_isSharedCheck_1688_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1688_ == 0)
{
v___x_1683_ = v___x_1634_;
v_isShared_1684_ = v_isSharedCheck_1688_;
goto v_resetjp_1682_;
}
else
{
lean_inc(v_a_1681_);
lean_dec(v___x_1634_);
v___x_1683_ = lean_box(0);
v_isShared_1684_ = v_isSharedCheck_1688_;
goto v_resetjp_1682_;
}
v_resetjp_1682_:
{
lean_object* v___x_1686_; 
if (v_isShared_1684_ == 0)
{
v___x_1686_ = v___x_1683_;
goto v_reusejp_1685_;
}
else
{
lean_object* v_reuseFailAlloc_1687_; 
v_reuseFailAlloc_1687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1687_, 0, v_a_1681_);
v___x_1686_ = v_reuseFailAlloc_1687_;
goto v_reusejp_1685_;
}
v_reusejp_1685_:
{
return v___x_1686_;
}
}
}
}
v___jp_1689_:
{
lean_object* v_fileName_1695_; lean_object* v_fileMap_1696_; uint8_t v_suppressElabErrors_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v_a_1700_; lean_object* v___x_1702_; uint8_t v_isShared_1703_; uint8_t v_isSharedCheck_1716_; 
v_fileName_1695_ = lean_ctor_get(v___y_1622_, 0);
v_fileMap_1696_ = lean_ctor_get(v___y_1622_, 1);
v_suppressElabErrors_1697_ = lean_ctor_get_uint8(v___y_1622_, sizeof(void*)*10);
v___x_1698_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1619_);
v___x_1699_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg(v___x_1698_, v___y_1623_);
v_a_1700_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1716_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1716_ == 0)
{
v___x_1702_ = v___x_1699_;
v_isShared_1703_ = v_isSharedCheck_1716_;
goto v_resetjp_1701_;
}
else
{
lean_inc(v_a_1700_);
lean_dec(v___x_1699_);
v___x_1702_ = lean_box(0);
v_isShared_1703_ = v_isSharedCheck_1716_;
goto v_resetjp_1701_;
}
v_resetjp_1701_:
{
lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; 
lean_inc_ref_n(v_fileMap_1696_, 2);
v___x_1704_ = l_Lean_FileMap_toPosition(v_fileMap_1696_, v___y_1693_);
lean_dec(v___y_1693_);
v___x_1705_ = l_Lean_FileMap_toPosition(v_fileMap_1696_, v___y_1694_);
lean_dec(v___y_1694_);
v___x_1706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1706_, 0, v___x_1705_);
v___x_1707_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0));
if (v_suppressElabErrors_1697_ == 0)
{
lean_del_object(v___x_1702_);
v___y_1626_ = v___x_1707_;
v___y_1627_ = v___y_1691_;
v___y_1628_ = v_fileName_1695_;
v___y_1629_ = v___x_1704_;
v___y_1630_ = v___y_1692_;
v___y_1631_ = v___x_1706_;
v___y_1632_ = v_a_1700_;
v___y_1633_ = v___y_1623_;
goto v___jp_1625_;
}
else
{
lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___f_1710_; uint8_t v___x_1711_; 
v___x_1708_ = lean_box(v___y_1690_);
v___x_1709_ = lean_box(v_suppressElabErrors_1697_);
v___f_1710_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1710_, 0, v___x_1708_);
lean_closure_set(v___f_1710_, 1, v___x_1709_);
lean_inc(v_a_1700_);
v___x_1711_ = l_Lean_MessageData_hasTag(v___f_1710_, v_a_1700_);
if (v___x_1711_ == 0)
{
lean_object* v___x_1712_; lean_object* v___x_1714_; 
lean_dec_ref_known(v___x_1706_, 1);
lean_dec_ref(v___x_1704_);
lean_dec(v_a_1700_);
v___x_1712_ = lean_box(0);
if (v_isShared_1703_ == 0)
{
lean_ctor_set(v___x_1702_, 0, v___x_1712_);
v___x_1714_ = v___x_1702_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v___x_1712_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
else
{
lean_del_object(v___x_1702_);
v___y_1626_ = v___x_1707_;
v___y_1627_ = v___y_1691_;
v___y_1628_ = v_fileName_1695_;
v___y_1629_ = v___x_1704_;
v___y_1630_ = v___y_1692_;
v___y_1631_ = v___x_1706_;
v___y_1632_ = v_a_1700_;
v___y_1633_ = v___y_1623_;
goto v___jp_1625_;
}
}
}
}
v___jp_1717_:
{
lean_object* v___x_1723_; 
v___x_1723_ = l_Lean_Syntax_getTailPos_x3f(v___y_1720_, v___y_1719_);
lean_dec(v___y_1720_);
if (lean_obj_tag(v___x_1723_) == 0)
{
lean_inc(v___y_1722_);
v___y_1690_ = v___y_1718_;
v___y_1691_ = v___y_1719_;
v___y_1692_ = v___y_1721_;
v___y_1693_ = v___y_1722_;
v___y_1694_ = v___y_1722_;
goto v___jp_1689_;
}
else
{
lean_object* v_val_1724_; 
v_val_1724_ = lean_ctor_get(v___x_1723_, 0);
lean_inc(v_val_1724_);
lean_dec_ref_known(v___x_1723_, 1);
v___y_1690_ = v___y_1718_;
v___y_1691_ = v___y_1719_;
v___y_1692_ = v___y_1721_;
v___y_1693_ = v___y_1722_;
v___y_1694_ = v_val_1724_;
goto v___jp_1689_;
}
}
v___jp_1725_:
{
lean_object* v___x_1729_; 
v___x_1729_ = l_Lean_Elab_Command_getRef___redArg(v___y_1622_);
if (lean_obj_tag(v___x_1729_) == 0)
{
lean_object* v_a_1730_; lean_object* v_ref_1731_; lean_object* v___x_1732_; 
v_a_1730_ = lean_ctor_get(v___x_1729_, 0);
lean_inc(v_a_1730_);
lean_dec_ref_known(v___x_1729_, 1);
v_ref_1731_ = l_Lean_replaceRef(v_ref_1618_, v_a_1730_);
lean_dec(v_a_1730_);
v___x_1732_ = l_Lean_Syntax_getPos_x3f(v_ref_1731_, v___y_1727_);
if (lean_obj_tag(v___x_1732_) == 0)
{
lean_object* v___x_1733_; 
v___x_1733_ = lean_unsigned_to_nat(0u);
v___y_1718_ = v___y_1726_;
v___y_1719_ = v___y_1727_;
v___y_1720_ = v_ref_1731_;
v___y_1721_ = v___y_1728_;
v___y_1722_ = v___x_1733_;
goto v___jp_1717_;
}
else
{
lean_object* v_val_1734_; 
v_val_1734_ = lean_ctor_get(v___x_1732_, 0);
lean_inc(v_val_1734_);
lean_dec_ref_known(v___x_1732_, 1);
v___y_1718_ = v___y_1726_;
v___y_1719_ = v___y_1727_;
v___y_1720_ = v_ref_1731_;
v___y_1721_ = v___y_1728_;
v___y_1722_ = v_val_1734_;
goto v___jp_1717_;
}
}
else
{
lean_object* v_a_1735_; lean_object* v___x_1737_; uint8_t v_isShared_1738_; uint8_t v_isSharedCheck_1742_; 
lean_dec_ref(v_msgData_1619_);
v_a_1735_ = lean_ctor_get(v___x_1729_, 0);
v_isSharedCheck_1742_ = !lean_is_exclusive(v___x_1729_);
if (v_isSharedCheck_1742_ == 0)
{
v___x_1737_ = v___x_1729_;
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
else
{
lean_inc(v_a_1735_);
lean_dec(v___x_1729_);
v___x_1737_ = lean_box(0);
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
v_resetjp_1736_:
{
lean_object* v___x_1740_; 
if (v_isShared_1738_ == 0)
{
v___x_1740_ = v___x_1737_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v_a_1735_);
v___x_1740_ = v_reuseFailAlloc_1741_;
goto v_reusejp_1739_;
}
v_reusejp_1739_:
{
return v___x_1740_;
}
}
}
}
v___jp_1744_:
{
if (v___y_1747_ == 0)
{
v___y_1726_ = v___y_1745_;
v___y_1727_ = v___y_1746_;
v___y_1728_ = v_severity_1620_;
goto v___jp_1725_;
}
else
{
v___y_1726_ = v___y_1745_;
v___y_1727_ = v___y_1746_;
v___y_1728_ = v___x_1743_;
goto v___jp_1725_;
}
}
v___jp_1748_:
{
if (v___y_1749_ == 0)
{
lean_object* v___x_1750_; lean_object* v_scopes_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v_opts_1754_; uint8_t v___x_1755_; uint8_t v___x_1756_; 
v___x_1750_ = lean_st_ref_get(v___y_1623_);
v_scopes_1751_ = lean_ctor_get(v___x_1750_, 2);
lean_inc(v_scopes_1751_);
lean_dec(v___x_1750_);
v___x_1752_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1753_ = l_List_head_x21___redArg(v___x_1752_, v_scopes_1751_);
lean_dec(v_scopes_1751_);
v_opts_1754_ = lean_ctor_get(v___x_1753_, 1);
lean_inc_ref(v_opts_1754_);
lean_dec(v___x_1753_);
v___x_1755_ = 1;
v___x_1756_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1620_, v___x_1755_);
if (v___x_1756_ == 0)
{
lean_dec_ref(v_opts_1754_);
v___y_1745_ = v___y_1749_;
v___y_1746_ = v___y_1749_;
v___y_1747_ = v___x_1756_;
goto v___jp_1744_;
}
else
{
lean_object* v___x_1757_; uint8_t v___x_1758_; 
v___x_1757_ = l_Lean_warningAsError;
v___x_1758_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__7(v_opts_1754_, v___x_1757_);
lean_dec_ref(v_opts_1754_);
v___y_1745_ = v___y_1749_;
v___y_1746_ = v___y_1749_;
v___y_1747_ = v___x_1758_;
goto v___jp_1744_;
}
}
else
{
lean_object* v___x_1759_; lean_object* v___x_1760_; 
lean_dec_ref(v_msgData_1619_);
v___x_1759_ = lean_box(0);
v___x_1760_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1760_, 0, v___x_1759_);
return v___x_1760_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3___boxed(lean_object* v_ref_1763_, lean_object* v_msgData_1764_, lean_object* v_severity_1765_, lean_object* v_isSilent_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_){
_start:
{
uint8_t v_severity_boxed_1770_; uint8_t v_isSilent_boxed_1771_; lean_object* v_res_1772_; 
v_severity_boxed_1770_ = lean_unbox(v_severity_1765_);
v_isSilent_boxed_1771_ = lean_unbox(v_isSilent_1766_);
v_res_1772_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3(v_ref_1763_, v_msgData_1764_, v_severity_boxed_1770_, v_isSilent_boxed_1771_, v___y_1767_, v___y_1768_);
lean_dec(v___y_1768_);
lean_dec_ref(v___y_1767_);
lean_dec(v_ref_1763_);
return v_res_1772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2(lean_object* v_ref_1773_, lean_object* v_msgData_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_){
_start:
{
uint8_t v___x_1778_; uint8_t v___x_1779_; lean_object* v___x_1780_; 
v___x_1778_ = 1;
v___x_1779_ = 0;
v___x_1780_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3(v_ref_1773_, v_msgData_1774_, v___x_1778_, v___x_1779_, v___y_1775_, v___y_1776_);
return v___x_1780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2___boxed(lean_object* v_ref_1781_, lean_object* v_msgData_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_){
_start:
{
lean_object* v_res_1786_; 
v_res_1786_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2(v_ref_1781_, v_msgData_1782_, v___y_1783_, v___y_1784_);
lean_dec(v___y_1784_);
lean_dec_ref(v___y_1783_);
lean_dec(v_ref_1781_);
return v_res_1786_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1788_; lean_object* v___x_1789_; 
v___x_1788_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__0));
v___x_1789_ = l_Lean_stringToMessageData(v___x_1788_);
return v___x_1789_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1791_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__2));
v___x_1792_ = l_Lean_stringToMessageData(v___x_1791_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(lean_object* v_linterOption_1793_, lean_object* v_stx_1794_, lean_object* v_msg_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v_name_1799_; lean_object* v___x_1801_; uint8_t v_isShared_1802_; uint8_t v_isSharedCheck_1817_; 
v_name_1799_ = lean_ctor_get(v_linterOption_1793_, 0);
v_isSharedCheck_1817_ = !lean_is_exclusive(v_linterOption_1793_);
if (v_isSharedCheck_1817_ == 0)
{
lean_object* v_unused_1818_; 
v_unused_1818_ = lean_ctor_get(v_linterOption_1793_, 1);
lean_dec(v_unused_1818_);
v___x_1801_ = v_linterOption_1793_;
v_isShared_1802_ = v_isSharedCheck_1817_;
goto v_resetjp_1800_;
}
else
{
lean_inc(v_name_1799_);
lean_dec(v_linterOption_1793_);
v___x_1801_ = lean_box(0);
v_isShared_1802_ = v_isSharedCheck_1817_;
goto v_resetjp_1800_;
}
v_resetjp_1800_:
{
lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1806_; 
v___x_1803_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__1);
lean_inc(v_name_1799_);
v___x_1804_ = l_Lean_MessageData_ofName(v_name_1799_);
if (v_isShared_1802_ == 0)
{
lean_ctor_set_tag(v___x_1801_, 7);
lean_ctor_set(v___x_1801_, 1, v___x_1804_);
lean_ctor_set(v___x_1801_, 0, v___x_1803_);
v___x_1806_ = v___x_1801_;
goto v_reusejp_1805_;
}
else
{
lean_object* v_reuseFailAlloc_1816_; 
v_reuseFailAlloc_1816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1816_, 0, v___x_1803_);
lean_ctor_set(v_reuseFailAlloc_1816_, 1, v___x_1804_);
v___x_1806_ = v_reuseFailAlloc_1816_;
goto v_reusejp_1805_;
}
v_reusejp_1805_:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v_disable_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; 
v___x_1807_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___closed__3);
v___x_1808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1806_);
lean_ctor_set(v___x_1808_, 1, v___x_1807_);
v_disable_1809_ = l_Lean_MessageData_note(v___x_1808_);
v___x_1810_ = l_Lean_Linter_linterMessageTag;
v___x_1811_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1811_, 0, v_msg_1795_);
lean_ctor_set(v___x_1811_, 1, v_disable_1809_);
v___x_1812_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1812_, 0, v___x_1810_);
lean_ctor_set(v___x_1812_, 1, v___x_1811_);
v___x_1813_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1813_, 0, v_name_1799_);
lean_ctor_set(v___x_1813_, 1, v___x_1812_);
lean_inc(v_stx_1794_);
v___x_1814_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1814_, 0, v_stx_1794_);
lean_ctor_set(v___x_1814_, 1, v___x_1813_);
v___x_1815_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2(v_stx_1794_, v___x_1814_, v___y_1796_, v___y_1797_);
lean_dec(v_stx_1794_);
return v___x_1815_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1___boxed(lean_object* v_linterOption_1819_, lean_object* v_stx_1820_, lean_object* v_msg_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_){
_start:
{
lean_object* v_res_1825_; 
v_res_1825_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(v_linterOption_1819_, v_stx_1820_, v_msg_1821_, v___y_1822_, v___y_1823_);
lean_dec(v___y_1823_);
lean_dec_ref(v___y_1822_);
return v_res_1825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg(lean_object* v_o_1826_, lean_object* v___y_1827_){
_start:
{
lean_object* v___x_1829_; lean_object* v_env_1830_; lean_object* v___x_1831_; lean_object* v_toEnvExtension_1832_; lean_object* v_asyncMode_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v_merged_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1845_; 
v___x_1829_ = lean_st_ref_get(v___y_1827_);
v_env_1830_ = lean_ctor_get(v___x_1829_, 0);
lean_inc_ref(v_env_1830_);
lean_dec(v___x_1829_);
v___x_1831_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1832_ = lean_ctor_get(v___x_1831_, 0);
v_asyncMode_1833_ = lean_ctor_get(v_toEnvExtension_1832_, 2);
v___x_1834_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1835_ = lean_box(0);
v___x_1836_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1834_, v___x_1831_, v_env_1830_, v_asyncMode_1833_, v___x_1835_);
v_merged_1837_ = lean_ctor_get(v___x_1836_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1836_);
if (v_isSharedCheck_1845_ == 0)
{
lean_object* v_unused_1846_; 
v_unused_1846_ = lean_ctor_get(v___x_1836_, 1);
lean_dec(v_unused_1846_);
v___x_1839_ = v___x_1836_;
v_isShared_1840_ = v_isSharedCheck_1845_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_merged_1837_);
lean_dec(v___x_1836_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1845_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v___x_1842_; 
if (v_isShared_1840_ == 0)
{
lean_ctor_set(v___x_1839_, 1, v_merged_1837_);
lean_ctor_set(v___x_1839_, 0, v_o_1826_);
v___x_1842_ = v___x_1839_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v_o_1826_);
lean_ctor_set(v_reuseFailAlloc_1844_, 1, v_merged_1837_);
v___x_1842_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
lean_object* v___x_1843_; 
v___x_1843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1843_, 0, v___x_1842_);
return v___x_1843_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_){
_start:
{
lean_object* v_res_1850_; 
v_res_1850_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg(v_o_1847_, v___y_1848_);
lean_dec(v___y_1848_);
return v_res_1850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0(lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
lean_object* v___x_1854_; lean_object* v_scopes_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v_opts_1858_; lean_object* v___x_1859_; 
v___x_1854_ = lean_st_ref_get(v___y_1852_);
v_scopes_1855_ = lean_ctor_get(v___x_1854_, 2);
lean_inc(v_scopes_1855_);
lean_dec(v___x_1854_);
v___x_1856_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1857_ = l_List_head_x21___redArg(v___x_1856_, v_scopes_1855_);
lean_dec(v_scopes_1855_);
v_opts_1858_ = lean_ctor_get(v___x_1857_, 1);
lean_inc_ref(v_opts_1858_);
lean_dec(v___x_1857_);
v___x_1859_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg(v_opts_1858_, v___y_1852_);
return v___x_1859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0___boxed(lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
lean_object* v_res_1863_; 
v_res_1863_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0(v___y_1860_, v___y_1861_);
lean_dec(v___y_1861_);
lean_dec_ref(v___y_1860_);
return v_res_1863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(lean_object* v_linterOption_1864_, lean_object* v_stx_1865_, lean_object* v_msg_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
lean_object* v___x_1870_; lean_object* v_a_1871_; lean_object* v___x_1873_; uint8_t v_isShared_1874_; uint8_t v_isSharedCheck_1881_; 
v___x_1870_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0(v___y_1867_, v___y_1868_);
v_a_1871_ = lean_ctor_get(v___x_1870_, 0);
v_isSharedCheck_1881_ = !lean_is_exclusive(v___x_1870_);
if (v_isSharedCheck_1881_ == 0)
{
v___x_1873_ = v___x_1870_;
v_isShared_1874_ = v_isSharedCheck_1881_;
goto v_resetjp_1872_;
}
else
{
lean_inc(v_a_1871_);
lean_dec(v___x_1870_);
v___x_1873_ = lean_box(0);
v_isShared_1874_ = v_isSharedCheck_1881_;
goto v_resetjp_1872_;
}
v_resetjp_1872_:
{
uint8_t v___x_1875_; 
v___x_1875_ = l_Lean_Linter_getLinterValue(v_linterOption_1864_, v_a_1871_);
lean_dec(v_a_1871_);
if (v___x_1875_ == 0)
{
lean_object* v___x_1876_; lean_object* v___x_1878_; 
lean_dec_ref(v_msg_1866_);
lean_dec(v_stx_1865_);
lean_dec_ref(v_linterOption_1864_);
v___x_1876_ = lean_box(0);
if (v_isShared_1874_ == 0)
{
lean_ctor_set(v___x_1873_, 0, v___x_1876_);
v___x_1878_ = v___x_1873_;
goto v_reusejp_1877_;
}
else
{
lean_object* v_reuseFailAlloc_1879_; 
v_reuseFailAlloc_1879_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1879_, 0, v___x_1876_);
v___x_1878_ = v_reuseFailAlloc_1879_;
goto v_reusejp_1877_;
}
v_reusejp_1877_:
{
return v___x_1878_;
}
}
else
{
lean_object* v___x_1880_; 
lean_del_object(v___x_1873_);
v___x_1880_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(v_linterOption_1864_, v_stx_1865_, v_msg_1866_, v___y_1867_, v___y_1868_);
return v___x_1880_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2___boxed(lean_object* v_linterOption_1882_, lean_object* v_stx_1883_, lean_object* v_msg_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_){
_start:
{
lean_object* v_res_1888_; 
v_res_1888_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(v_linterOption_1882_, v_stx_1883_, v_msg_1884_, v___y_1885_, v___y_1886_);
lean_dec(v___y_1886_);
lean_dec_ref(v___y_1885_);
return v_res_1888_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4(void){
_start:
{
lean_object* v___x_1898_; lean_object* v___x_1899_; 
v___x_1898_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__3));
v___x_1899_ = l_Lean_stringToMessageData(v___x_1898_);
return v___x_1899_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6(void){
_start:
{
lean_object* v___x_1901_; lean_object* v___x_1902_; 
v___x_1901_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__5));
v___x_1902_ = l_Lean_stringToMessageData(v___x_1901_);
return v___x_1902_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8(void){
_start:
{
lean_object* v___x_1904_; lean_object* v___x_1905_; 
v___x_1904_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__7));
v___x_1905_ = l_Lean_stringToMessageData(v___x_1904_);
return v___x_1905_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10(void){
_start:
{
lean_object* v___x_1907_; lean_object* v___x_1908_; 
v___x_1907_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__9));
v___x_1908_ = l_Lean_stringToMessageData(v___x_1907_);
return v___x_1908_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12(void){
_start:
{
lean_object* v___x_1910_; lean_object* v___x_1911_; 
v___x_1910_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__11));
v___x_1911_ = l_Lean_stringToMessageData(v___x_1910_);
return v___x_1911_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14(void){
_start:
{
lean_object* v___x_1913_; lean_object* v___x_1914_; 
v___x_1913_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__13));
v___x_1914_ = lean_string_utf8_byte_size(v___x_1913_);
return v___x_1914_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16(void){
_start:
{
lean_object* v___x_1916_; lean_object* v___x_1917_; 
v___x_1916_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__15));
v___x_1917_ = l_Lean_stringToMessageData(v___x_1916_);
return v___x_1917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3(lean_object* v_stopPos_1918_, lean_object* v___x_1919_, lean_object* v_val_1920_, lean_object* v___y_1921_, lean_object* v___x_1922_, lean_object* v___x_1923_, uint8_t v___x_1924_, lean_object* v_val_1925_, lean_object* v___y_1926_, lean_object* v_as_1927_, size_t v_sz_1928_, size_t v_i_1929_, lean_object* v_b_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_){
_start:
{
lean_object* v_a_1935_; uint8_t v___x_1939_; 
v___x_1939_ = lean_usize_dec_lt(v_i_1929_, v_sz_1928_);
if (v___x_1939_ == 0)
{
lean_object* v___x_1940_; 
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v___x_1940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1940_, 0, v_b_1930_);
return v___x_1940_;
}
else
{
lean_object* v_a_1941_; lean_object* v_srcNat_1942_; lean_object* v_srcEndPos_1943_; lean_object* v_fmtPos_1944_; lean_object* v_msg_1945_; lean_object* v_length_1946_; lean_object* v_srcStartPos_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; uint8_t v___x_2012_; 
lean_dec_ref(v_b_1930_);
v_a_1941_ = lean_array_uget_borrowed(v_as_1927_, v_i_1929_);
v_srcNat_1942_ = lean_ctor_get(v_a_1941_, 0);
v_srcEndPos_1943_ = lean_ctor_get(v_a_1941_, 1);
v_fmtPos_1944_ = lean_ctor_get(v_a_1941_, 2);
v_msg_1945_ = lean_ctor_get(v_a_1941_, 3);
v_length_1946_ = lean_ctor_get(v_a_1941_, 4);
v_srcStartPos_1947_ = lean_ctor_get(v_a_1941_, 5);
v___x_1948_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__0));
v___x_1949_ = lp_mathlib_Mathlib_Linter_linter_style_whitespace;
v___x_1950_ = lean_nat_sub(v_stopPos_1918_, v_srcEndPos_1943_);
v___x_1951_ = lean_nat_add(v_srcEndPos_1943_, v___x_1950_);
v___x_1952_ = lean_nat_sub(v___x_1951_, v_srcStartPos_1947_);
lean_dec(v___x_1951_);
v___x_1953_ = lean_unsigned_to_nat(1u);
v___x_1954_ = lean_nat_add(v___x_1952_, v___x_1953_);
lean_dec(v___x_1952_);
lean_inc(v___x_1954_);
lean_inc(v___x_1950_);
v___x_1955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1955_, 0, v___x_1950_);
lean_ctor_set(v___x_1955_, 1, v___x_1954_);
v___x_2009_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__13));
v___x_2010_ = lean_string_utf8_byte_size(v_msg_1945_);
v___x_2011_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__14);
v___x_2012_ = lean_nat_dec_le(v___x_2011_, v___x_2010_);
if (v___x_2012_ == 0)
{
goto v___jp_1956_;
}
else
{
lean_object* v___x_2013_; uint8_t v___x_2014_; 
v___x_2013_ = lean_unsigned_to_nat(0u);
v___x_2014_ = lean_string_memcmp(v_msg_1945_, v___x_2009_, v___x_2013_, v___x_2013_, v___x_2011_);
if (v___x_2014_ == 0)
{
goto v___jp_1956_;
}
else
{
lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; 
lean_dec(v___x_1954_);
lean_dec(v___x_1950_);
v___x_2015_ = lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose;
v___x_2016_ = l_Lean_Syntax_ofRange(v___x_1955_, v___x_1924_);
v___x_2017_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__16);
lean_inc(v___x_2016_);
v___x_2018_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(v___x_2015_, v___x_2016_, v___x_2017_, v___y_1931_, v___y_1932_);
if (lean_obj_tag(v___x_2018_) == 0)
{
lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; 
lean_dec_ref_known(v___x_2018_, 1);
v___x_2019_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10);
lean_inc(v_val_1925_);
v___x_2020_ = l_Lean_MessageData_ofFormat(v_val_1925_);
v___x_2021_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2021_, 0, v___x_2019_);
lean_ctor_set(v___x_2021_, 1, v___x_2020_);
v___x_2022_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12);
v___x_2023_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2023_, 0, v___x_2021_);
lean_ctor_set(v___x_2023_, 1, v___x_2022_);
lean_inc_ref(v___y_1926_);
v___x_2024_ = lean_substring_tostring(v___y_1926_);
v___x_2025_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2025_, 0, v___x_2024_);
v___x_2026_ = l_Lean_MessageData_ofFormat(v___x_2025_);
v___x_2027_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2027_, 0, v___x_2023_);
lean_ctor_set(v___x_2027_, 1, v___x_2026_);
v___x_2028_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(v___x_2015_, v___x_2016_, v___x_2027_, v___y_1931_, v___y_1932_);
if (lean_obj_tag(v___x_2028_) == 0)
{
lean_dec_ref_known(v___x_2028_, 1);
v_a_1935_ = v___x_1948_;
goto v___jp_1934_;
}
else
{
lean_object* v_a_2029_; lean_object* v___x_2031_; uint8_t v_isShared_2032_; uint8_t v_isSharedCheck_2036_; 
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v_a_2029_ = lean_ctor_get(v___x_2028_, 0);
v_isSharedCheck_2036_ = !lean_is_exclusive(v___x_2028_);
if (v_isSharedCheck_2036_ == 0)
{
v___x_2031_ = v___x_2028_;
v_isShared_2032_ = v_isSharedCheck_2036_;
goto v_resetjp_2030_;
}
else
{
lean_inc(v_a_2029_);
lean_dec(v___x_2028_);
v___x_2031_ = lean_box(0);
v_isShared_2032_ = v_isSharedCheck_2036_;
goto v_resetjp_2030_;
}
v_resetjp_2030_:
{
lean_object* v___x_2034_; 
if (v_isShared_2032_ == 0)
{
v___x_2034_ = v___x_2031_;
goto v_reusejp_2033_;
}
else
{
lean_object* v_reuseFailAlloc_2035_; 
v_reuseFailAlloc_2035_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2035_, 0, v_a_2029_);
v___x_2034_ = v_reuseFailAlloc_2035_;
goto v_reusejp_2033_;
}
v_reusejp_2033_:
{
return v___x_2034_;
}
}
}
}
else
{
lean_object* v_a_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2044_; 
lean_dec(v___x_2016_);
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v_a_2037_ = lean_ctor_get(v___x_2018_, 0);
v_isSharedCheck_2044_ = !lean_is_exclusive(v___x_2018_);
if (v_isSharedCheck_2044_ == 0)
{
v___x_2039_ = v___x_2018_;
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_a_2037_);
lean_dec(v___x_2018_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2042_; 
if (v_isShared_2040_ == 0)
{
v___x_2042_ = v___x_2039_;
goto v_reusejp_2041_;
}
else
{
lean_object* v_reuseFailAlloc_2043_; 
v_reuseFailAlloc_2043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2043_, 0, v_a_2037_);
v___x_2042_ = v_reuseFailAlloc_2043_;
goto v_reusejp_2041_;
}
v_reusejp_2041_:
{
return v___x_2042_;
}
}
}
}
}
v___jp_1956_:
{
uint8_t v___x_1957_; 
v___x_1957_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_isOutside(v___x_1919_, v___x_1955_);
if (v___x_1957_ == 0)
{
lean_dec_ref_known(v___x_1955_, 2);
lean_dec(v___x_1954_);
lean_dec(v___x_1950_);
v_a_1935_ = v___x_1948_;
goto v___jp_1934_;
}
else
{
uint8_t v___x_1958_; 
v___x_1958_ = lean_nat_dec_le(v___x_1954_, v_val_1920_);
lean_dec(v___x_1954_);
if (v___x_1958_ == 0)
{
lean_object* v___x_1959_; lean_object* v___x_1960_; 
lean_dec_ref_known(v___x_1955_, 2);
lean_dec(v___x_1950_);
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v___x_1959_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__2));
v___x_1960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1960_, 0, v___x_1959_);
return v___x_1960_;
}
else
{
uint8_t v___x_1961_; 
v___x_1961_ = lean_nat_dec_le(v___y_1921_, v___x_1950_);
lean_dec(v___x_1950_);
if (v___x_1961_ == 0)
{
lean_object* v___x_1962_; lean_object* v___x_1963_; 
lean_dec_ref_known(v___x_1955_, 2);
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v___x_1962_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__2));
v___x_1963_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1963_, 0, v___x_1962_);
return v___x_1963_;
}
else
{
lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; 
v___x_1964_ = lean_unsigned_to_nat(4u);
v___x_1965_ = lean_nat_add(v___x_1964_, v_length_1946_);
lean_inc_ref(v___x_1922_);
v___x_1966_ = lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow(v___x_1922_, v_srcNat_1942_, v___x_1965_);
v___x_1967_ = lean_unsigned_to_nat(5u);
lean_inc_ref(v___x_1923_);
v___x_1968_ = lp_mathlib_Mathlib_Linter_Style_Whitespace_mkWindow(v___x_1923_, v_fmtPos_1944_, v___x_1967_);
v___x_1969_ = l_Lean_Syntax_ofRange(v___x_1955_, v___x_1924_);
lean_inc_ref(v_msg_1945_);
v___x_1970_ = l_Lean_stringToMessageData(v_msg_1945_);
v___x_1971_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__4);
v___x_1972_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1972_, 0, v___x_1970_);
lean_ctor_set(v___x_1972_, 1, v___x_1971_);
v___x_1973_ = l_Lean_stringToMessageData(v___x_1966_);
v___x_1974_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1974_, 0, v___x_1972_);
lean_ctor_set(v___x_1974_, 1, v___x_1973_);
v___x_1975_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__6);
v___x_1976_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1976_, 0, v___x_1974_);
lean_ctor_set(v___x_1976_, 1, v___x_1975_);
v___x_1977_ = l_Lean_stringToMessageData(v___x_1968_);
v___x_1978_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1978_, 0, v___x_1976_);
lean_ctor_set(v___x_1978_, 1, v___x_1977_);
v___x_1979_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__8);
v___x_1980_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1980_, 0, v___x_1978_);
lean_ctor_set(v___x_1980_, 1, v___x_1979_);
lean_inc(v___x_1969_);
v___x_1981_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(v___x_1949_, v___x_1969_, v___x_1980_, v___y_1931_, v___y_1932_);
if (lean_obj_tag(v___x_1981_) == 0)
{
lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; 
lean_dec_ref_known(v___x_1981_, 1);
v___x_1982_ = lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose;
v___x_1983_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__10);
lean_inc(v_val_1925_);
v___x_1984_ = l_Lean_MessageData_ofFormat(v_val_1925_);
v___x_1985_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1985_, 0, v___x_1983_);
lean_ctor_set(v___x_1985_, 1, v___x_1984_);
v___x_1986_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__12);
v___x_1987_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1987_, 0, v___x_1985_);
lean_ctor_set(v___x_1987_, 1, v___x_1986_);
lean_inc_ref(v___y_1926_);
v___x_1988_ = lean_substring_tostring(v___y_1926_);
v___x_1989_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1989_, 0, v___x_1988_);
v___x_1990_ = l_Lean_MessageData_ofFormat(v___x_1989_);
v___x_1991_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1991_, 0, v___x_1987_);
lean_ctor_set(v___x_1991_, 1, v___x_1990_);
v___x_1992_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(v___x_1982_, v___x_1969_, v___x_1991_, v___y_1931_, v___y_1932_);
if (lean_obj_tag(v___x_1992_) == 0)
{
lean_dec_ref_known(v___x_1992_, 1);
v_a_1935_ = v___x_1948_;
goto v___jp_1934_;
}
else
{
lean_object* v_a_1993_; lean_object* v___x_1995_; uint8_t v_isShared_1996_; uint8_t v_isSharedCheck_2000_; 
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v_a_1993_ = lean_ctor_get(v___x_1992_, 0);
v_isSharedCheck_2000_ = !lean_is_exclusive(v___x_1992_);
if (v_isSharedCheck_2000_ == 0)
{
v___x_1995_ = v___x_1992_;
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
else
{
lean_inc(v_a_1993_);
lean_dec(v___x_1992_);
v___x_1995_ = lean_box(0);
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
v_resetjp_1994_:
{
lean_object* v___x_1998_; 
if (v_isShared_1996_ == 0)
{
v___x_1998_ = v___x_1995_;
goto v_reusejp_1997_;
}
else
{
lean_object* v_reuseFailAlloc_1999_; 
v_reuseFailAlloc_1999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1999_, 0, v_a_1993_);
v___x_1998_ = v_reuseFailAlloc_1999_;
goto v_reusejp_1997_;
}
v_reusejp_1997_:
{
return v___x_1998_;
}
}
}
}
else
{
lean_object* v_a_2001_; lean_object* v___x_2003_; uint8_t v_isShared_2004_; uint8_t v_isSharedCheck_2008_; 
lean_dec(v___x_1969_);
lean_dec_ref(v___y_1926_);
lean_dec(v_val_1925_);
lean_dec_ref(v___x_1923_);
lean_dec_ref(v___x_1922_);
v_a_2001_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_2008_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_2008_ == 0)
{
v___x_2003_ = v___x_1981_;
v_isShared_2004_ = v_isSharedCheck_2008_;
goto v_resetjp_2002_;
}
else
{
lean_inc(v_a_2001_);
lean_dec(v___x_1981_);
v___x_2003_ = lean_box(0);
v_isShared_2004_ = v_isSharedCheck_2008_;
goto v_resetjp_2002_;
}
v_resetjp_2002_:
{
lean_object* v___x_2006_; 
if (v_isShared_2004_ == 0)
{
v___x_2006_ = v___x_2003_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v_a_2001_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
return v___x_2006_;
}
}
}
}
}
}
}
}
v___jp_1934_:
{
size_t v___x_1936_; size_t v___x_1937_; 
v___x_1936_ = ((size_t)1ULL);
v___x_1937_ = lean_usize_add(v_i_1929_, v___x_1936_);
lean_inc_ref(v_a_1935_);
v_i_1929_ = v___x_1937_;
v_b_1930_ = v_a_1935_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___boxed(lean_object* v_stopPos_2045_, lean_object* v___x_2046_, lean_object* v_val_2047_, lean_object* v___y_2048_, lean_object* v___x_2049_, lean_object* v___x_2050_, lean_object* v___x_2051_, lean_object* v_val_2052_, lean_object* v___y_2053_, lean_object* v_as_2054_, lean_object* v_sz_2055_, lean_object* v_i_2056_, lean_object* v_b_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_){
_start:
{
uint8_t v___x_15263__boxed_2061_; size_t v_sz_boxed_2062_; size_t v_i_boxed_2063_; lean_object* v_res_2064_; 
v___x_15263__boxed_2061_ = lean_unbox(v___x_2051_);
v_sz_boxed_2062_ = lean_unbox_usize(v_sz_2055_);
lean_dec(v_sz_2055_);
v_i_boxed_2063_ = lean_unbox_usize(v_i_2056_);
lean_dec(v_i_2056_);
v_res_2064_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3(v_stopPos_2045_, v___x_2046_, v_val_2047_, v___y_2048_, v___x_2049_, v___x_2050_, v___x_15263__boxed_2061_, v_val_2052_, v___y_2053_, v_as_2054_, v_sz_boxed_2062_, v_i_boxed_2063_, v_b_2057_, v___y_2058_, v___y_2059_);
lean_dec(v___y_2059_);
lean_dec_ref(v___y_2058_);
lean_dec_ref(v_as_2054_);
lean_dec(v___y_2048_);
lean_dec(v_val_2047_);
lean_dec_ref(v___x_2046_);
lean_dec(v_stopPos_2045_);
return v_res_2064_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1(void){
_start:
{
lean_object* v___x_2066_; lean_object* v___x_2067_; 
v___x_2066_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__0));
v___x_2067_ = l_Lean_stringToMessageData(v___x_2066_);
return v___x_2067_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2(void){
_start:
{
lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; 
v___x_2068_ = lean_box(0);
v___x_2069_ = lean_unsigned_to_nat(16u);
v___x_2070_ = lean_mk_array(v___x_2069_, v___x_2068_);
return v___x_2070_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7(void){
_start:
{
lean_object* v___x_2078_; lean_object* v___x_2079_; 
v___x_2078_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__6));
v___x_2079_ = l_Lean_stringToMessageData(v___x_2078_);
return v___x_2079_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9(void){
_start:
{
lean_object* v___x_2081_; lean_object* v___x_2082_; 
v___x_2081_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__8));
v___x_2082_ = l_Lean_stringToMessageData(v___x_2081_);
return v___x_2082_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11(void){
_start:
{
lean_object* v___x_2084_; lean_object* v___x_2085_; 
v___x_2084_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__10));
v___x_2085_ = l_Lean_stringToMessageData(v___x_2084_);
return v___x_2085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4(lean_object* v___f_2086_, lean_object* v___f_2087_, lean_object* v___f_2088_, lean_object* v_stx_2089_, lean_object* v___y_2090_, lean_object* v___y_2091_){
_start:
{
lean_object* v___y_2097_; lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___x_2103_; lean_object* v_a_2104_; lean_object* v___x_2106_; uint8_t v_isShared_2107_; uint8_t v_isSharedCheck_2286_; 
v___x_2103_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0(v___y_2090_, v___y_2091_);
v_a_2104_ = lean_ctor_get(v___x_2103_, 0);
v_isSharedCheck_2286_ = !lean_is_exclusive(v___x_2103_);
if (v_isSharedCheck_2286_ == 0)
{
v___x_2106_ = v___x_2103_;
v_isShared_2107_ = v_isSharedCheck_2286_;
goto v_resetjp_2105_;
}
else
{
lean_inc(v_a_2104_);
lean_dec(v___x_2103_);
v___x_2106_ = lean_box(0);
v_isShared_2107_ = v_isSharedCheck_2286_;
goto v_resetjp_2105_;
}
v___jp_2093_:
{
lean_object* v___x_2094_; lean_object* v___x_2095_; 
v___x_2094_ = lean_box(0);
v___x_2095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2095_, 0, v___x_2094_);
return v___x_2095_;
}
v___jp_2096_:
{
lean_object* v___x_2101_; lean_object* v___x_2102_; 
v___x_2101_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__1);
lean_inc_ref(v___y_2097_);
v___x_2102_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__2(v___y_2097_, v___y_2100_, v___x_2101_, v___y_2098_, v___y_2099_);
if (lean_obj_tag(v___x_2102_) == 0)
{
lean_dec_ref_known(v___x_2102_, 1);
goto v___jp_2093_;
}
else
{
return v___x_2102_;
}
}
v_resetjp_2105_:
{
lean_object* v___x_2108_; uint8_t v___x_2109_; lean_object* v___y_2111_; lean_object* v___y_2112_; lean_object* v___y_2113_; lean_object* v___y_2114_; lean_object* v___y_2115_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2153_; lean_object* v___y_2154_; lean_object* v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2158_; lean_object* v___y_2159_; lean_object* v___y_2160_; lean_object* v___y_2161_; uint8_t v___y_2162_; lean_object* v___y_2163_; lean_object* v___y_2164_; lean_object* v___y_2168_; lean_object* v___y_2169_; lean_object* v___y_2170_; lean_object* v___y_2171_; uint8_t v___y_2172_; lean_object* v___y_2173_; lean_object* v___y_2174_; lean_object* v___y_2175_; lean_object* v_str_2176_; lean_object* v_startPos_2177_; lean_object* v_stopPos_2178_; lean_object* v___y_2185_; lean_object* v___y_2186_; uint8_t v___y_2187_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___y_2211_; lean_object* v___y_2212_; uint8_t v___y_2213_; uint8_t v___y_2229_; lean_object* v___y_2230_; lean_object* v___y_2231_; 
v___x_2108_ = lp_mathlib_Mathlib_Linter_linter_style_whitespace;
v___x_2109_ = l_Lean_Linter_getLinterValue(v___x_2108_, v_a_2104_);
lean_dec(v_a_2104_);
if (v___x_2109_ == 0)
{
lean_object* v___x_2242_; lean_object* v___x_2243_; 
lean_del_object(v___x_2106_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2088_);
lean_dec_ref(v___f_2087_);
lean_dec_ref(v___f_2086_);
v___x_2242_ = lean_box(0);
v___x_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2242_);
return v___x_2243_;
}
else
{
lean_object* v___x_2244_; uint8_t v___y_2246_; lean_object* v_messages_2281_; uint8_t v___x_2282_; 
v___x_2244_ = lean_st_ref_get(v___y_2091_);
v_messages_2281_ = lean_ctor_get(v___x_2244_, 1);
lean_inc_ref(v_messages_2281_);
lean_dec(v___x_2244_);
v___x_2282_ = l_Lean_MessageLog_hasErrors(v_messages_2281_);
lean_dec_ref(v_messages_2281_);
if (v___x_2282_ == 0)
{
lean_object* v___x_2283_; 
lean_inc(v_stx_2089_);
v___x_2283_ = l_Lean_Syntax_find_x3f(v_stx_2089_, v___f_2088_);
if (lean_obj_tag(v___x_2283_) == 0)
{
v___y_2246_ = v___x_2282_;
goto v___jp_2245_;
}
else
{
lean_dec_ref_known(v___x_2283_, 1);
v___y_2246_ = v___x_2109_;
goto v___jp_2245_;
}
}
else
{
lean_object* v___x_2284_; lean_object* v___x_2285_; 
lean_del_object(v___x_2106_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2088_);
lean_dec_ref(v___f_2087_);
lean_dec_ref(v___f_2086_);
v___x_2284_ = lean_box(0);
v___x_2285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2285_, 0, v___x_2284_);
return v___x_2285_;
}
v___jp_2245_:
{
if (v___y_2246_ == 0)
{
lean_object* v___x_2247_; 
v___x_2247_ = l_Lean_Syntax_getPos_x3f(v_stx_2089_, v___y_2246_);
if (lean_obj_tag(v___x_2247_) == 1)
{
lean_object* v_val_2248_; lean_object* v___x_2250_; uint8_t v_isShared_2251_; uint8_t v_isSharedCheck_2278_; 
v_val_2248_ = lean_ctor_get(v___x_2247_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v___x_2247_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2250_ = v___x_2247_;
v_isShared_2251_ = v_isSharedCheck_2278_;
goto v_resetjp_2249_;
}
else
{
lean_inc(v_val_2248_);
lean_dec(v___x_2247_);
v___x_2250_ = lean_box(0);
v_isShared_2251_ = v_isSharedCheck_2278_;
goto v_resetjp_2249_;
}
v_resetjp_2249_:
{
lean_object* v_fileMap_2252_; lean_object* v___x_2253_; lean_object* v_column_2254_; lean_object* v___x_2256_; uint8_t v_isShared_2257_; uint8_t v_isSharedCheck_2276_; 
v_fileMap_2252_ = lean_ctor_get(v___y_2090_, 1);
lean_inc_ref(v_fileMap_2252_);
v___x_2253_ = l_Lean_FileMap_toPosition(v_fileMap_2252_, v_val_2248_);
lean_dec(v_val_2248_);
v_column_2254_ = lean_ctor_get(v___x_2253_, 1);
v_isSharedCheck_2276_ = !lean_is_exclusive(v___x_2253_);
if (v_isSharedCheck_2276_ == 0)
{
lean_object* v_unused_2277_; 
v_unused_2277_ = lean_ctor_get(v___x_2253_, 0);
lean_dec(v_unused_2277_);
v___x_2256_ = v___x_2253_;
v_isShared_2257_ = v_isSharedCheck_2276_;
goto v_resetjp_2255_;
}
else
{
lean_inc(v_column_2254_);
lean_dec(v___x_2253_);
v___x_2256_ = lean_box(0);
v_isShared_2257_ = v_isSharedCheck_2276_;
goto v_resetjp_2255_;
}
v_resetjp_2255_:
{
lean_object* v___x_2258_; uint8_t v___x_2259_; 
v___x_2258_ = lean_unsigned_to_nat(0u);
v___x_2259_ = lean_nat_dec_eq(v_column_2254_, v___x_2258_);
if (v___x_2259_ == 0)
{
if (v___x_2109_ == 0)
{
lean_del_object(v___x_2256_);
lean_dec(v_column_2254_);
lean_del_object(v___x_2250_);
v___y_2229_ = v___y_2246_;
v___y_2230_ = v___y_2090_;
v___y_2231_ = v___y_2091_;
goto v___jp_2228_;
}
else
{
lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2263_; 
v___x_2260_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__7);
lean_inc(v_stx_2089_);
v___x_2261_ = l_Lean_MessageData_ofSyntax(v_stx_2089_);
if (v_isShared_2257_ == 0)
{
lean_ctor_set_tag(v___x_2256_, 7);
lean_ctor_set(v___x_2256_, 1, v___x_2261_);
lean_ctor_set(v___x_2256_, 0, v___x_2260_);
v___x_2263_ = v___x_2256_;
goto v_reusejp_2262_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v___x_2260_);
lean_ctor_set(v_reuseFailAlloc_2275_, 1, v___x_2261_);
v___x_2263_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2262_;
}
v_reusejp_2262_:
{
lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2268_; 
v___x_2264_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__9);
v___x_2265_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2265_, 0, v___x_2263_);
lean_ctor_set(v___x_2265_, 1, v___x_2264_);
v___x_2266_ = l_Nat_reprFast(v_column_2254_);
if (v_isShared_2251_ == 0)
{
lean_ctor_set_tag(v___x_2250_, 3);
lean_ctor_set(v___x_2250_, 0, v___x_2266_);
v___x_2268_ = v___x_2250_;
goto v_reusejp_2267_;
}
else
{
lean_object* v_reuseFailAlloc_2274_; 
v_reuseFailAlloc_2274_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2274_, 0, v___x_2266_);
v___x_2268_ = v_reuseFailAlloc_2274_;
goto v_reusejp_2267_;
}
v_reusejp_2267_:
{
lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; 
v___x_2269_ = l_Lean_MessageData_ofFormat(v___x_2268_);
v___x_2270_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2270_, 0, v___x_2265_);
lean_ctor_set(v___x_2270_, 1, v___x_2269_);
v___x_2271_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__11);
v___x_2272_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2272_, 0, v___x_2270_);
lean_ctor_set(v___x_2272_, 1, v___x_2271_);
lean_inc(v_stx_2089_);
v___x_2273_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1(v___x_2108_, v_stx_2089_, v___x_2272_, v___y_2090_, v___y_2091_);
if (lean_obj_tag(v___x_2273_) == 0)
{
lean_dec_ref_known(v___x_2273_, 1);
v___y_2229_ = v___y_2246_;
v___y_2230_ = v___y_2090_;
v___y_2231_ = v___y_2091_;
goto v___jp_2228_;
}
else
{
lean_del_object(v___x_2106_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2087_);
lean_dec_ref(v___f_2086_);
return v___x_2273_;
}
}
}
}
}
else
{
lean_del_object(v___x_2256_);
lean_dec(v_column_2254_);
lean_del_object(v___x_2250_);
v___y_2229_ = v___y_2246_;
v___y_2230_ = v___y_2090_;
v___y_2231_ = v___y_2091_;
goto v___jp_2228_;
}
}
}
}
else
{
lean_dec(v___x_2247_);
v___y_2229_ = v___y_2246_;
v___y_2230_ = v___y_2090_;
v___y_2231_ = v___y_2091_;
goto v___jp_2228_;
}
}
else
{
lean_object* v___x_2279_; lean_object* v___x_2280_; 
lean_del_object(v___x_2106_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2087_);
lean_dec_ref(v___f_2086_);
v___x_2279_ = lean_box(0);
v___x_2280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2280_, 0, v___x_2279_);
return v___x_2280_;
}
}
}
v___jp_2110_:
{
lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; size_t v_sz_2128_; size_t v___x_2129_; lean_object* v___x_2130_; 
v___x_2122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_unlintedNodes___closed__32));
v___x_2123_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__2);
v___x_2124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2124_, 0, v___y_2112_);
lean_ctor_set(v___x_2124_, 1, v___x_2123_);
v___x_2125_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_getUnlintedRanges(v___x_2122_, v___x_2124_, v_stx_2089_);
lean_dec(v_stx_2089_);
v___x_2126_ = lean_box(0);
v___x_2127_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3___closed__0));
v_sz_2128_ = lean_array_size(v___y_2111_);
v___x_2129_ = ((size_t)0ULL);
v___x_2130_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__3(v___y_2118_, v___x_2125_, v___y_2117_, v___y_2121_, v___y_2113_, v___y_2119_, v___x_2109_, v___y_2115_, v___y_2114_, v___y_2111_, v_sz_2128_, v___x_2129_, v___x_2127_, v___y_2116_, v___y_2120_);
lean_dec_ref(v___y_2111_);
lean_dec(v___y_2121_);
lean_dec(v___y_2117_);
lean_dec_ref(v___x_2125_);
lean_dec(v___y_2118_);
if (lean_obj_tag(v___x_2130_) == 0)
{
lean_object* v_a_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2143_; 
v_a_2131_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2143_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2143_ == 0)
{
v___x_2133_ = v___x_2130_;
v_isShared_2134_ = v_isSharedCheck_2143_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_a_2131_);
lean_dec(v___x_2130_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2143_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v_fst_2135_; 
v_fst_2135_ = lean_ctor_get(v_a_2131_, 0);
lean_inc(v_fst_2135_);
lean_dec(v_a_2131_);
if (lean_obj_tag(v_fst_2135_) == 0)
{
lean_object* v___x_2137_; 
if (v_isShared_2134_ == 0)
{
lean_ctor_set(v___x_2133_, 0, v___x_2126_);
v___x_2137_ = v___x_2133_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v___x_2126_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
return v___x_2137_;
}
}
else
{
lean_object* v_val_2139_; lean_object* v___x_2141_; 
v_val_2139_ = lean_ctor_get(v_fst_2135_, 0);
lean_inc(v_val_2139_);
lean_dec_ref_known(v_fst_2135_, 1);
if (v_isShared_2134_ == 0)
{
lean_ctor_set(v___x_2133_, 0, v_val_2139_);
v___x_2141_ = v___x_2133_;
goto v_reusejp_2140_;
}
else
{
lean_object* v_reuseFailAlloc_2142_; 
v_reuseFailAlloc_2142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2142_, 0, v_val_2139_);
v___x_2141_ = v_reuseFailAlloc_2142_;
goto v_reusejp_2140_;
}
v_reusejp_2140_:
{
return v___x_2141_;
}
}
}
}
else
{
lean_object* v_a_2144_; lean_object* v___x_2146_; uint8_t v_isShared_2147_; uint8_t v_isSharedCheck_2151_; 
v_a_2144_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2151_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2151_ == 0)
{
v___x_2146_ = v___x_2130_;
v_isShared_2147_ = v_isSharedCheck_2151_;
goto v_resetjp_2145_;
}
else
{
lean_inc(v_a_2144_);
lean_dec(v___x_2130_);
v___x_2146_ = lean_box(0);
v_isShared_2147_ = v_isSharedCheck_2151_;
goto v_resetjp_2145_;
}
v_resetjp_2145_:
{
lean_object* v___x_2149_; 
if (v_isShared_2147_ == 0)
{
v___x_2149_ = v___x_2146_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2150_; 
v_reuseFailAlloc_2150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2150_, 0, v_a_2144_);
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
v___jp_2152_:
{
lean_object* v___x_2165_; 
v___x_2165_ = l_Lean_Syntax_getTailPos_x3f(v___y_2164_, v___y_2162_);
lean_dec(v___y_2164_);
if (lean_obj_tag(v___x_2165_) == 0)
{
lean_inc(v___y_2154_);
v___y_2111_ = v___y_2153_;
v___y_2112_ = v___y_2154_;
v___y_2113_ = v___y_2155_;
v___y_2114_ = v___y_2156_;
v___y_2115_ = v___y_2157_;
v___y_2116_ = v___y_2160_;
v___y_2117_ = v___y_2159_;
v___y_2118_ = v___y_2158_;
v___y_2119_ = v___y_2161_;
v___y_2120_ = v___y_2163_;
v___y_2121_ = v___y_2154_;
goto v___jp_2110_;
}
else
{
lean_object* v_val_2166_; 
v_val_2166_ = lean_ctor_get(v___x_2165_, 0);
lean_inc(v_val_2166_);
lean_dec_ref_known(v___x_2165_, 1);
v___y_2111_ = v___y_2153_;
v___y_2112_ = v___y_2154_;
v___y_2113_ = v___y_2155_;
v___y_2114_ = v___y_2156_;
v___y_2115_ = v___y_2157_;
v___y_2116_ = v___y_2160_;
v___y_2117_ = v___y_2159_;
v___y_2118_ = v___y_2158_;
v___y_2119_ = v___y_2161_;
v___y_2120_ = v___y_2163_;
v___y_2121_ = v_val_2166_;
goto v___jp_2110_;
}
}
v___jp_2167_:
{
lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; 
v___x_2179_ = lean_string_utf8_extract(v_str_2176_, v_startPos_2177_, v_stopPos_2178_);
lean_dec(v_startPos_2177_);
lean_dec_ref(v_str_2176_);
lean_inc_ref(v___y_2173_);
lean_inc_ref(v___x_2179_);
v___x_2180_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_parallelScan(v___x_2179_, v___y_2173_);
lean_inc(v_stx_2089_);
v___x_2181_ = l_Lean_Syntax_find_x3f(v_stx_2089_, v___f_2086_);
if (lean_obj_tag(v___x_2181_) == 0)
{
lean_object* v___x_2182_; 
v___x_2182_ = lean_box(0);
v___y_2153_ = v___x_2180_;
v___y_2154_ = v___y_2168_;
v___y_2155_ = v___x_2179_;
v___y_2156_ = v___y_2175_;
v___y_2157_ = v___y_2169_;
v___y_2158_ = v_stopPos_2178_;
v___y_2159_ = v___y_2170_;
v___y_2160_ = v___y_2171_;
v___y_2161_ = v___y_2173_;
v___y_2162_ = v___y_2172_;
v___y_2163_ = v___y_2174_;
v___y_2164_ = v___x_2182_;
goto v___jp_2152_;
}
else
{
lean_object* v_val_2183_; 
v_val_2183_ = lean_ctor_get(v___x_2181_, 0);
lean_inc(v_val_2183_);
lean_dec_ref_known(v___x_2181_, 1);
v___y_2153_ = v___x_2180_;
v___y_2154_ = v___y_2168_;
v___y_2155_ = v___x_2179_;
v___y_2156_ = v___y_2175_;
v___y_2157_ = v___y_2169_;
v___y_2158_ = v_stopPos_2178_;
v___y_2159_ = v___y_2170_;
v___y_2160_ = v___y_2171_;
v___y_2161_ = v___y_2173_;
v___y_2162_ = v___y_2172_;
v___y_2163_ = v___y_2174_;
v___y_2164_ = v_val_2183_;
goto v___jp_2152_;
}
}
v___jp_2184_:
{
if (lean_obj_tag(v___y_2189_) == 0)
{
lean_object* v_a_2190_; 
v_a_2190_ = lean_ctor_get(v___y_2189_, 0);
lean_inc(v_a_2190_);
lean_dec_ref_known(v___y_2189_, 1);
if (lean_obj_tag(v_a_2190_) == 1)
{
lean_object* v_val_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; 
v_val_2191_ = lean_ctor_get(v_a_2190_, 0);
lean_inc_n(v_val_2191_, 2);
lean_dec_ref_known(v_a_2190_, 1);
v___x_2192_ = l_Std_Format_defWidth;
v___x_2193_ = lean_unsigned_to_nat(0u);
v___x_2194_ = l_Std_Format_pretty(v_val_2191_, v___x_2192_, v___x_2193_, v___x_2193_);
v___x_2195_ = l_Lean_Syntax_getSubstring_x3f(v_stx_2089_, v___x_2109_, v___x_2109_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_object* v___x_2196_; lean_object* v___x_2197_; 
v___x_2196_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_instInhabitedFormatError_default___closed__0));
v___x_2197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__3));
v___y_2168_ = v___x_2193_;
v___y_2169_ = v_val_2191_;
v___y_2170_ = v___y_2186_;
v___y_2171_ = v___y_2185_;
v___y_2172_ = v___y_2187_;
v___y_2173_ = v___x_2194_;
v___y_2174_ = v___y_2188_;
v___y_2175_ = v___x_2197_;
v_str_2176_ = v___x_2196_;
v_startPos_2177_ = v___x_2193_;
v_stopPos_2178_ = v___x_2193_;
goto v___jp_2167_;
}
else
{
lean_object* v_val_2198_; lean_object* v_str_2199_; lean_object* v_startPos_2200_; lean_object* v_stopPos_2201_; 
v_val_2198_ = lean_ctor_get(v___x_2195_, 0);
lean_inc(v_val_2198_);
lean_dec_ref_known(v___x_2195_, 1);
v_str_2199_ = lean_ctor_get(v_val_2198_, 0);
lean_inc_ref(v_str_2199_);
v_startPos_2200_ = lean_ctor_get(v_val_2198_, 1);
lean_inc(v_startPos_2200_);
v_stopPos_2201_ = lean_ctor_get(v_val_2198_, 2);
lean_inc(v_stopPos_2201_);
v___y_2168_ = v___x_2193_;
v___y_2169_ = v_val_2191_;
v___y_2170_ = v___y_2186_;
v___y_2171_ = v___y_2185_;
v___y_2172_ = v___y_2187_;
v___y_2173_ = v___x_2194_;
v___y_2174_ = v___y_2188_;
v___y_2175_ = v_val_2198_;
v_str_2176_ = v_str_2199_;
v_startPos_2177_ = v_startPos_2200_;
v_stopPos_2178_ = v_stopPos_2201_;
goto v___jp_2167_;
}
}
else
{
lean_dec(v_a_2190_);
lean_dec(v___y_2186_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2086_);
goto v___jp_2093_;
}
}
else
{
lean_object* v_a_2202_; lean_object* v___x_2204_; uint8_t v_isShared_2205_; uint8_t v_isSharedCheck_2209_; 
lean_dec(v___y_2186_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2086_);
v_a_2202_ = lean_ctor_get(v___y_2189_, 0);
v_isSharedCheck_2209_ = !lean_is_exclusive(v___y_2189_);
if (v_isSharedCheck_2209_ == 0)
{
v___x_2204_ = v___y_2189_;
v_isShared_2205_ = v_isSharedCheck_2209_;
goto v_resetjp_2203_;
}
else
{
lean_inc(v_a_2202_);
lean_dec(v___y_2189_);
v___x_2204_ = lean_box(0);
v_isShared_2205_ = v_isSharedCheck_2209_;
goto v_resetjp_2203_;
}
v_resetjp_2203_:
{
lean_object* v___x_2207_; 
if (v_isShared_2205_ == 0)
{
v___x_2207_ = v___x_2204_;
goto v_reusejp_2206_;
}
else
{
lean_object* v_reuseFailAlloc_2208_; 
v_reuseFailAlloc_2208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2208_, 0, v_a_2202_);
v___x_2207_ = v_reuseFailAlloc_2208_;
goto v_reusejp_2206_;
}
v_reusejp_2206_:
{
return v___x_2207_;
}
}
}
}
v___jp_2210_:
{
lean_object* v___x_2214_; 
lean_inc(v_stx_2089_);
v___x_2214_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_CommandStart_endPos(v_stx_2089_);
if (lean_obj_tag(v___x_2214_) == 1)
{
lean_object* v_val_2215_; lean_object* v___x_2216_; lean_object* v___f_2217_; lean_object* v___x_2218_; 
lean_del_object(v___x_2106_);
v_val_2215_ = lean_ctor_get(v___x_2214_, 0);
lean_inc(v_val_2215_);
lean_dec_ref_known(v___x_2214_, 1);
v___x_2216_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___closed__5));
lean_inc(v_stx_2089_);
v___f_2217_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__3___boxed), 5, 2);
lean_closure_set(v___f_2217_, 0, v___x_2216_);
lean_closure_set(v___f_2217_, 1, v_stx_2089_);
v___x_2218_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_2217_, v___y_2211_, v___y_2212_);
if (lean_obj_tag(v___x_2218_) == 0)
{
v___y_2185_ = v___y_2211_;
v___y_2186_ = v_val_2215_;
v___y_2187_ = v___y_2213_;
v___y_2188_ = v___y_2212_;
v___y_2189_ = v___x_2218_;
goto v___jp_2184_;
}
else
{
lean_object* v_a_2219_; uint8_t v___x_2220_; 
v_a_2219_ = lean_ctor_get(v___x_2218_, 0);
lean_inc(v_a_2219_);
v___x_2220_ = l_Lean_Exception_isInterrupt(v_a_2219_);
lean_dec(v_a_2219_);
if (v___x_2220_ == 0)
{
lean_object* v___x_2221_; lean_object* v___x_2222_; 
lean_dec_ref_known(v___x_2218_, 1);
lean_dec(v_val_2215_);
lean_dec_ref(v___f_2086_);
v___x_2221_ = lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose;
lean_inc(v_stx_2089_);
v___x_2222_ = l_Lean_Syntax_getHead_x3f(v_stx_2089_);
if (lean_obj_tag(v___x_2222_) == 0)
{
v___y_2097_ = v___x_2221_;
v___y_2098_ = v___y_2211_;
v___y_2099_ = v___y_2212_;
v___y_2100_ = v_stx_2089_;
goto v___jp_2096_;
}
else
{
lean_object* v_val_2223_; 
lean_dec(v_stx_2089_);
v_val_2223_ = lean_ctor_get(v___x_2222_, 0);
lean_inc(v_val_2223_);
lean_dec_ref_known(v___x_2222_, 1);
v___y_2097_ = v___x_2221_;
v___y_2098_ = v___y_2211_;
v___y_2099_ = v___y_2212_;
v___y_2100_ = v_val_2223_;
goto v___jp_2096_;
}
}
else
{
v___y_2185_ = v___y_2211_;
v___y_2186_ = v_val_2215_;
v___y_2187_ = v___y_2213_;
v___y_2188_ = v___y_2212_;
v___y_2189_ = v___x_2218_;
goto v___jp_2184_;
}
}
}
else
{
lean_object* v___x_2224_; lean_object* v___x_2226_; 
lean_dec(v___x_2214_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2086_);
v___x_2224_ = lean_box(0);
if (v_isShared_2107_ == 0)
{
lean_ctor_set(v___x_2106_, 0, v___x_2224_);
v___x_2226_ = v___x_2106_;
goto v_reusejp_2225_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v___x_2224_);
v___x_2226_ = v_reuseFailAlloc_2227_;
goto v_reusejp_2225_;
}
v_reusejp_2225_:
{
return v___x_2226_;
}
}
}
v___jp_2228_:
{
lean_object* v___x_2232_; 
lean_inc(v_stx_2089_);
v___x_2232_ = l_Lean_Syntax_find_x3f(v_stx_2089_, v___f_2087_);
if (lean_obj_tag(v___x_2232_) == 0)
{
v___y_2211_ = v___y_2230_;
v___y_2212_ = v___y_2231_;
v___y_2213_ = v___y_2229_;
goto v___jp_2210_;
}
else
{
lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2240_; 
v_isSharedCheck_2240_ = !lean_is_exclusive(v___x_2232_);
if (v_isSharedCheck_2240_ == 0)
{
lean_object* v_unused_2241_; 
v_unused_2241_ = lean_ctor_get(v___x_2232_, 0);
lean_dec(v_unused_2241_);
v___x_2234_ = v___x_2232_;
v_isShared_2235_ = v_isSharedCheck_2240_;
goto v_resetjp_2233_;
}
else
{
lean_dec(v___x_2232_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2240_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
if (v___x_2109_ == 0)
{
lean_del_object(v___x_2234_);
v___y_2211_ = v___y_2230_;
v___y_2212_ = v___y_2231_;
v___y_2213_ = v___x_2109_;
goto v___jp_2210_;
}
else
{
lean_object* v___x_2236_; lean_object* v___x_2238_; 
lean_del_object(v___x_2106_);
lean_dec(v_stx_2089_);
lean_dec_ref(v___f_2086_);
v___x_2236_ = lean_box(0);
if (v_isShared_2235_ == 0)
{
lean_ctor_set_tag(v___x_2234_, 0);
lean_ctor_set(v___x_2234_, 0, v___x_2236_);
v___x_2238_ = v___x_2234_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v___x_2236_);
v___x_2238_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
return v___x_2238_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4___boxed(lean_object* v___f_2287_, lean_object* v___f_2288_, lean_object* v___f_2289_, lean_object* v_stx_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_){
_start:
{
lean_object* v_res_2294_; 
v_res_2294_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter___lam__4(v___f_2287_, v___f_2288_, v___f_2289_, v_stx_2290_, v___y_2291_, v___y_2292_);
lean_dec(v___y_2292_);
lean_dec_ref(v___y_2291_);
return v_res_2294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0(lean_object* v_o_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_){
_start:
{
lean_object* v___x_2349_; 
v___x_2349_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___redArg(v_o_2345_, v___y_2347_);
return v___x_2349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0___boxed(lean_object* v_o_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_){
_start:
{
lean_object* v_res_2354_; 
v_res_2354_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__0_spec__0(v_o_2350_, v___y_2351_, v___y_2352_);
lean_dec(v___y_2352_);
lean_dec_ref(v___y_2351_);
return v_res_2354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6(lean_object* v_msgData_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_){
_start:
{
lean_object* v___x_2359_; 
v___x_2359_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___redArg(v_msgData_2355_, v___y_2357_);
return v___x_2359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6___boxed(lean_object* v_msgData_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_){
_start:
{
lean_object* v_res_2364_; 
v_res_2364_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter_spec__1_spec__2_spec__3_spec__6(v_msgData_2360_, v___y_2361_, v___y_2362_);
lean_dec(v___y_2362_);
lean_dec_ref(v___y_2361_);
return v_res_2364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; 
v___x_2366_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_whitespaceLinter));
v___x_2367_ = l_Lean_Elab_Command_addLinter(v___x_2366_);
return v___x_2367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2____boxed(lean_object* v_a_2368_){
_start:
{
lean_object* v_res_2369_; 
v_res_2369_ = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2_();
return v_res_2369_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_721641949____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_whitespace = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_whitespace);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_3220585800____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_whitespace_verbose);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Whitespace_0__Mathlib_Linter_Style_Whitespace_initFn_00___x40_Mathlib_Tactic_Linter_Whitespace_391367517____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(builtin);
}
#ifdef __cplusplus
}
#endif
