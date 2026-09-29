// Lean compiler output
// Module: Batteries.Util.LibraryNote
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
lean_object* l_String_Slice_slice_x21(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_push___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Name_componentsRev(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommandTopLevel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_instInhabitedLibraryNote___aux__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_instInhabitedLibraryNote;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry___aux__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry;
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__0 = (const lean_object*)&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__0_value;
static const lean_string_object lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__1 = (const lean_object*)&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2;
static lean_once_cell_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3;
static lean_once_cell_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4;
static lean_once_cell_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5;
static lean_once_cell_t lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6;
static const lean_ctor_object lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__7 = (const lean_object*)&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1___closed__0 = (const lean_object*)&lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote_encodeNameForExport(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2____boxed(lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LibraryNote"};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__5_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "libraryNoteExt"};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__5_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__5_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(224, 67, 2, 0, 253, 100, 187, 147)}};
static const lean_ctor_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(89, 66, 232, 17, 108, 130, 10, 156)}};
static const lean_ctor_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__5_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(92, 208, 188, 244, 6, 255, 163, 253)}};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__7_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_push___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__7_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__7_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__8_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__6_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__7_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__8_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__8_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "commandLibrary_note___"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(224, 67, 2, 0, 253, 100, 187, 147)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(89, 66, 232, 17, 108, 130, 10, 156)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(162, 41, 190, 137, 170, 209, 123, 70)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "library_note "};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__9_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__11_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__11_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__16_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(229, 56, 215, 222, 243, 187, 251, 54)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__19 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__16_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__20 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__20_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__21 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__21_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note______ = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__21_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(124, 247, 59, 43, 44, 177, 111, 66)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "definition"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(248, 187, 217, 228, 39, 184, 218, 135)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__14_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__15 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__17 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(241, 126, 175, 40, 55, 129, 2, 13)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18_value;
static const lean_array_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__19 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__8_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__20 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__20_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__21 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__23 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__23_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__24 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__24_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__26 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__26_value;
static lean_once_cell_t lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(134, 197, 35, 181, 239, 168, 57, 237)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__28 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__28_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__2_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__3_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(224, 67, 2, 0, 253, 100, 187, 147)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(89, 66, 232, 17, 108, 130, 10, 156)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__30 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__30_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__29_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__31 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__31_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__31_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__32 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__32_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__30_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__32_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__33 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__33_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__34 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__34_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__34_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__36 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__36_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37_value;
static lean_once_cell_t lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(29, 214, 131, 210, 10, 90, 37, 134)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__39 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__39_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Inhabited"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__40 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__40_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(164, 88, 86, 106, 191, 136, 33, 185)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(174, 152, 115, 107, 166, 56, 116, 8)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__41_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__42 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__42_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__39_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__43 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__43_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__43_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__44 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__44_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__42_value),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__44_value)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__45 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__45_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__46 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__46_value;
static const lean_string_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__47 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__47_value;
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_0),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_1),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__46_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value_aux_2),((lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__47_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48 = (const lean_object*)&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Batteries_Util_instInhabitedLibraryNote___aux__1(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_instInhabitedLibraryNote(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry___aux__1(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(lean_object* v_s_5_, lean_object* v_replacement_6_, lean_object* v_a_7_, lean_object* v_b_8_){
_start:
{
lean_object* v_it_10_; lean_object* v_startPos_11_; lean_object* v_endPos_12_; lean_object* v_it_21_; 
switch(lean_obj_tag(v_a_7_))
{
case 0:
{
lean_object* v_pos_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_39_; 
v_pos_27_ = lean_ctor_get(v_a_7_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v_a_7_);
if (v_isSharedCheck_39_ == 0)
{
v___x_29_ = v_a_7_;
v_isShared_30_ = v_isSharedCheck_39_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_pos_27_);
lean_dec(v_a_7_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_39_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v_startInclusive_31_; lean_object* v_endExclusive_32_; lean_object* v___x_33_; uint8_t v___x_34_; 
v_startInclusive_31_ = lean_ctor_get(v_s_5_, 1);
v_endExclusive_32_ = lean_ctor_get(v_s_5_, 2);
v___x_33_ = lean_nat_sub(v_endExclusive_32_, v_startInclusive_31_);
v___x_34_ = lean_nat_dec_eq(v_pos_27_, v___x_33_);
lean_dec(v___x_33_);
if (v___x_34_ == 0)
{
lean_object* v___x_36_; 
if (v_isShared_30_ == 0)
{
lean_ctor_set_tag(v___x_29_, 1);
v___x_36_ = v___x_29_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v_pos_27_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
v_it_21_ = v___x_36_;
goto v___jp_20_;
}
}
else
{
lean_object* v___x_38_; 
lean_del_object(v___x_29_);
lean_dec(v_pos_27_);
v___x_38_ = lean_box(3);
v_it_21_ = v___x_38_;
goto v___jp_20_;
}
}
}
case 1:
{
lean_object* v_pos_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_52_; 
v_pos_40_ = lean_ctor_get(v_a_7_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v_a_7_);
if (v_isSharedCheck_52_ == 0)
{
v___x_42_ = v_a_7_;
v_isShared_43_ = v_isSharedCheck_52_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_pos_40_);
lean_dec(v_a_7_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_52_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v_str_44_; lean_object* v_startInclusive_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_50_; 
v_str_44_ = lean_ctor_get(v_s_5_, 0);
v_startInclusive_45_ = lean_ctor_get(v_s_5_, 1);
v___x_46_ = lean_nat_add(v_startInclusive_45_, v_pos_40_);
v___x_47_ = lean_string_utf8_next_fast(v_str_44_, v___x_46_);
lean_dec(v___x_46_);
v___x_48_ = lean_nat_sub(v___x_47_, v_startInclusive_45_);
lean_inc(v___x_48_);
if (v_isShared_43_ == 0)
{
lean_ctor_set_tag(v___x_42_, 0);
lean_ctor_set(v___x_42_, 0, v___x_48_);
v___x_50_ = v___x_42_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_48_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
v_it_10_ = v___x_50_;
v_startPos_11_ = v_pos_40_;
v_endPos_12_ = v___x_48_;
goto v___jp_9_;
}
}
}
case 2:
{
lean_object* v_needle_53_; lean_object* v_table_54_; lean_object* v_stackPos_55_; lean_object* v_needlePos_56_; lean_object* v___x_58_; uint8_t v_isShared_59_; uint8_t v_isSharedCheck_115_; 
v_needle_53_ = lean_ctor_get(v_a_7_, 0);
v_table_54_ = lean_ctor_get(v_a_7_, 1);
v_stackPos_55_ = lean_ctor_get(v_a_7_, 2);
v_needlePos_56_ = lean_ctor_get(v_a_7_, 3);
v_isSharedCheck_115_ = !lean_is_exclusive(v_a_7_);
if (v_isSharedCheck_115_ == 0)
{
v___x_58_ = v_a_7_;
v_isShared_59_ = v_isSharedCheck_115_;
goto v_resetjp_57_;
}
else
{
lean_inc(v_needlePos_56_);
lean_inc(v_stackPos_55_);
lean_inc(v_table_54_);
lean_inc(v_needle_53_);
lean_dec(v_a_7_);
v___x_58_ = lean_box(0);
v_isShared_59_ = v_isSharedCheck_115_;
goto v_resetjp_57_;
}
v_resetjp_57_:
{
lean_object* v_str_60_; lean_object* v_startInclusive_61_; lean_object* v_endExclusive_62_; lean_object* v_str_63_; lean_object* v_startInclusive_64_; lean_object* v_endExclusive_65_; lean_object* v_basePos_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v_str_60_ = lean_ctor_get(v_needle_53_, 0);
v_startInclusive_61_ = lean_ctor_get(v_needle_53_, 1);
v_endExclusive_62_ = lean_ctor_get(v_needle_53_, 2);
v_str_63_ = lean_ctor_get(v_s_5_, 0);
v_startInclusive_64_ = lean_ctor_get(v_s_5_, 1);
v_endExclusive_65_ = lean_ctor_get(v_s_5_, 2);
v_basePos_66_ = lean_nat_sub(v_stackPos_55_, v_needlePos_56_);
v___x_67_ = lean_nat_sub(v_endExclusive_62_, v_startInclusive_61_);
v___x_68_ = lean_nat_add(v_basePos_66_, v___x_67_);
v___x_69_ = lean_nat_sub(v_endExclusive_65_, v_startInclusive_64_);
v___x_70_ = lean_nat_dec_le(v___x_68_, v___x_69_);
lean_dec(v___x_68_);
if (v___x_70_ == 0)
{
uint8_t v___x_71_; 
lean_dec(v___x_67_);
lean_del_object(v___x_58_);
lean_dec(v_needlePos_56_);
lean_dec(v_stackPos_55_);
lean_dec_ref(v_table_54_);
lean_dec_ref(v_needle_53_);
v___x_71_ = lean_nat_dec_lt(v_basePos_66_, v___x_69_);
if (v___x_71_ == 0)
{
lean_dec(v___x_69_);
lean_dec(v_basePos_66_);
lean_dec_ref(v_s_5_);
return v_b_8_;
}
else
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = l_String_Slice_pos_x21(v_s_5_, v_basePos_66_);
lean_dec(v_basePos_66_);
v___x_73_ = lean_box(3);
v_it_10_ = v___x_73_;
v_startPos_11_ = v___x_72_;
v_endPos_12_ = v___x_69_;
goto v___jp_9_;
}
}
else
{
lean_object* v___x_74_; uint8_t v_stackByte_75_; lean_object* v___x_76_; uint8_t v_patByte_77_; uint8_t v___x_78_; 
lean_dec(v___x_69_);
v___x_74_ = lean_nat_add(v_startInclusive_64_, v_stackPos_55_);
v_stackByte_75_ = lean_string_get_byte_fast(v_str_63_, v___x_74_);
v___x_76_ = lean_nat_add(v_startInclusive_61_, v_needlePos_56_);
v_patByte_77_ = lean_string_get_byte_fast(v_str_60_, v___x_76_);
v___x_78_ = lean_uint8_dec_eq(v_stackByte_75_, v_patByte_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; uint8_t v___x_80_; 
lean_dec(v___x_67_);
v___x_79_ = lean_unsigned_to_nat(0u);
v___x_80_ = lean_nat_dec_eq(v_needlePos_56_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v_newNeedlePos_83_; uint8_t v___x_84_; 
v___x_81_ = lean_unsigned_to_nat(1u);
v___x_82_ = lean_nat_sub(v_needlePos_56_, v___x_81_);
lean_dec(v_needlePos_56_);
v_newNeedlePos_83_ = lean_array_fget_borrowed(v_table_54_, v___x_82_);
lean_dec(v___x_82_);
v___x_84_ = lean_nat_dec_eq(v_newNeedlePos_83_, v___x_79_);
if (v___x_84_ == 0)
{
lean_object* v_oldBasePos_85_; lean_object* v___x_86_; lean_object* v_newBasePos_87_; lean_object* v___x_89_; 
lean_inc(v_newNeedlePos_83_);
v_oldBasePos_85_ = l_String_Slice_pos_x21(v_s_5_, v_basePos_66_);
lean_dec(v_basePos_66_);
v___x_86_ = lean_nat_sub(v_stackPos_55_, v_newNeedlePos_83_);
v_newBasePos_87_ = l_String_Slice_pos_x21(v_s_5_, v___x_86_);
lean_dec(v___x_86_);
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 3, v_newNeedlePos_83_);
v___x_89_ = v___x_58_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v_needle_53_);
lean_ctor_set(v_reuseFailAlloc_90_, 1, v_table_54_);
lean_ctor_set(v_reuseFailAlloc_90_, 2, v_stackPos_55_);
lean_ctor_set(v_reuseFailAlloc_90_, 3, v_newNeedlePos_83_);
v___x_89_ = v_reuseFailAlloc_90_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
v_it_10_ = v___x_89_;
v_startPos_11_ = v_oldBasePos_85_;
v_endPos_12_ = v_newBasePos_87_;
goto v___jp_9_;
}
}
else
{
lean_object* v_basePos_91_; lean_object* v_nextStackPos_92_; lean_object* v___x_94_; 
v_basePos_91_ = l_String_Slice_pos_x21(v_s_5_, v_basePos_66_);
lean_dec(v_basePos_66_);
v_nextStackPos_92_ = l_String_Slice_posGE___redArg(v_s_5_, v_stackPos_55_);
lean_inc(v_nextStackPos_92_);
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 3, v___x_79_);
lean_ctor_set(v___x_58_, 2, v_nextStackPos_92_);
v___x_94_ = v___x_58_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_needle_53_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_table_54_);
lean_ctor_set(v_reuseFailAlloc_95_, 2, v_nextStackPos_92_);
lean_ctor_set(v_reuseFailAlloc_95_, 3, v___x_79_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
v_it_10_ = v___x_94_;
v_startPos_11_ = v_basePos_91_;
v_endPos_12_ = v_nextStackPos_92_;
goto v___jp_9_;
}
}
}
else
{
lean_object* v_basePos_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v_nextStackPos_99_; lean_object* v___x_101_; 
lean_dec(v_basePos_66_);
lean_dec(v_needlePos_56_);
v_basePos_96_ = l_String_Slice_pos_x21(v_s_5_, v_stackPos_55_);
v___x_97_ = lean_unsigned_to_nat(1u);
v___x_98_ = lean_nat_add(v_stackPos_55_, v___x_97_);
lean_dec(v_stackPos_55_);
v_nextStackPos_99_ = l_String_Slice_posGE___redArg(v_s_5_, v___x_98_);
lean_inc(v_nextStackPos_99_);
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 3, v___x_79_);
lean_ctor_set(v___x_58_, 2, v_nextStackPos_99_);
v___x_101_ = v___x_58_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_needle_53_);
lean_ctor_set(v_reuseFailAlloc_102_, 1, v_table_54_);
lean_ctor_set(v_reuseFailAlloc_102_, 2, v_nextStackPos_99_);
lean_ctor_set(v_reuseFailAlloc_102_, 3, v___x_79_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
v_it_10_ = v___x_101_;
v_startPos_11_ = v_basePos_96_;
v_endPos_12_ = v_nextStackPos_99_;
goto v___jp_9_;
}
}
}
else
{
lean_object* v___x_103_; lean_object* v_nextStackPos_104_; lean_object* v_nextNeedlePos_105_; uint8_t v___x_106_; 
lean_dec(v_basePos_66_);
v___x_103_ = lean_unsigned_to_nat(1u);
v_nextStackPos_104_ = lean_nat_add(v_stackPos_55_, v___x_103_);
lean_dec(v_stackPos_55_);
v_nextNeedlePos_105_ = lean_nat_add(v_needlePos_56_, v___x_103_);
lean_dec(v_needlePos_56_);
v___x_106_ = lean_nat_dec_eq(v_nextNeedlePos_105_, v___x_67_);
lean_dec(v___x_67_);
if (v___x_106_ == 0)
{
lean_object* v___x_108_; 
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 3, v_nextNeedlePos_105_);
lean_ctor_set(v___x_58_, 2, v_nextStackPos_104_);
v___x_108_ = v___x_58_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v_needle_53_);
lean_ctor_set(v_reuseFailAlloc_110_, 1, v_table_54_);
lean_ctor_set(v_reuseFailAlloc_110_, 2, v_nextStackPos_104_);
lean_ctor_set(v_reuseFailAlloc_110_, 3, v_nextNeedlePos_105_);
v___x_108_ = v_reuseFailAlloc_110_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
v_a_7_ = v___x_108_;
goto _start;
}
}
else
{
lean_object* v___x_111_; lean_object* v___x_113_; 
lean_dec(v_nextNeedlePos_105_);
v___x_111_ = lean_unsigned_to_nat(0u);
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 3, v___x_111_);
lean_ctor_set(v___x_58_, 2, v_nextStackPos_104_);
v___x_113_ = v___x_58_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v_needle_53_);
lean_ctor_set(v_reuseFailAlloc_114_, 1, v_table_54_);
lean_ctor_set(v_reuseFailAlloc_114_, 2, v_nextStackPos_104_);
lean_ctor_set(v_reuseFailAlloc_114_, 3, v___x_111_);
v___x_113_ = v_reuseFailAlloc_114_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
v_it_21_ = v___x_113_;
goto v___jp_20_;
}
}
}
}
}
}
default: 
{
lean_dec_ref(v_s_5_);
return v_b_8_;
}
}
v___jp_9_:
{
lean_object* v___x_13_; lean_object* v_str_14_; lean_object* v_startInclusive_15_; lean_object* v_endExclusive_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
lean_inc_ref(v_s_5_);
v___x_13_ = l_String_Slice_slice_x21(v_s_5_, v_startPos_11_, v_endPos_12_);
lean_dec(v_endPos_12_);
lean_dec(v_startPos_11_);
v_str_14_ = lean_ctor_get(v___x_13_, 0);
lean_inc_ref(v_str_14_);
v_startInclusive_15_ = lean_ctor_get(v___x_13_, 1);
lean_inc(v_startInclusive_15_);
v_endExclusive_16_ = lean_ctor_get(v___x_13_, 2);
lean_inc(v_endExclusive_16_);
lean_dec_ref(v___x_13_);
v___x_17_ = lean_string_utf8_extract_fast(v_str_14_, v_startInclusive_15_, v_endExclusive_16_);
lean_dec(v_endExclusive_16_);
lean_dec(v_startInclusive_15_);
lean_dec_ref(v_str_14_);
v___x_18_ = lean_string_append(v_b_8_, v___x_17_);
lean_dec_ref(v___x_17_);
v_a_7_ = v_it_10_;
v_b_8_ = v___x_18_;
goto _start;
}
v___jp_20_:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_22_ = lean_unsigned_to_nat(0u);
v___x_23_ = lean_string_utf8_byte_size(v_replacement_6_);
v___x_24_ = lean_string_utf8_extract_fast(v_replacement_6_, v___x_22_, v___x_23_);
v___x_25_ = lean_string_append(v_b_8_, v___x_24_);
lean_dec_ref(v___x_24_);
v_a_7_ = v_it_21_;
v_b_8_ = v___x_25_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg___boxed(lean_object* v_s_116_, lean_object* v_replacement_117_, lean_object* v_a_118_, lean_object* v_b_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(v_s_116_, v_replacement_117_, v_a_118_, v_b_119_);
lean_dec_ref(v_replacement_117_);
return v_res_120_;
}
}
static lean_object* _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_123_ = ((lean_object*)(lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__0));
v___x_124_ = lean_string_utf8_byte_size(v___x_123_);
return v___x_124_;
}
}
static uint8_t _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_125_ = lean_unsigned_to_nat(0u);
v___x_126_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2);
v___x_127_ = lean_nat_dec_eq(v___x_126_, v___x_125_);
return v___x_127_;
}
}
static lean_object* _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_128_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__2);
v___x_129_ = lean_unsigned_to_nat(0u);
v___x_130_ = ((lean_object*)(lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__0));
v___x_131_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v___x_129_);
lean_ctor_set(v___x_131_, 2, v___x_128_);
return v___x_131_;
}
}
static lean_object* _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4);
v___x_133_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_132_);
return v___x_133_;
}
}
static lean_object* _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_134_ = lean_unsigned_to_nat(0u);
v___x_135_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__5);
v___x_136_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__4);
v___x_137_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v___x_135_);
lean_ctor_set(v___x_137_, 2, v___x_134_);
lean_ctor_set(v___x_137_, 3, v___x_134_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg(lean_object* v_s_140_, lean_object* v_replacement_141_){
_start:
{
lean_object* v___x_142_; uint8_t v___x_143_; 
v___x_142_ = ((lean_object*)(lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__1));
v___x_143_ = lean_uint8_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__3);
if (v___x_143_ == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = lean_obj_once(&lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6, &lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6_once, _init_lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__6);
v___x_145_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(v_s_140_, v_replacement_141_, v___x_144_, v___x_142_);
return v___x_145_;
}
else
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = ((lean_object*)(lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___closed__7));
v___x_147_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(v_s_140_, v_replacement_141_, v___x_146_, v___x_142_);
return v___x_147_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg___boxed(lean_object* v_s_148_, lean_object* v_replacement_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg(v_s_148_, v_replacement_149_);
lean_dec_ref(v_replacement_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1(lean_object* v_x_152_, lean_object* v_x_153_){
_start:
{
if (lean_obj_tag(v_x_153_) == 0)
{
return v_x_152_;
}
else
{
lean_object* v_head_154_; 
v_head_154_ = lean_ctor_get(v_x_153_, 0);
switch(lean_obj_tag(v_head_154_))
{
case 0:
{
lean_object* v_tail_155_; 
v_tail_155_ = lean_ctor_get(v_x_153_, 1);
lean_inc(v_tail_155_);
lean_dec_ref_known(v_x_153_, 2);
v_x_153_ = v_tail_155_;
goto _start;
}
case 1:
{
lean_object* v_tail_157_; lean_object* v_str_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
lean_inc_ref(v_head_154_);
v_tail_157_ = lean_ctor_get(v_x_153_, 1);
lean_inc(v_tail_157_);
lean_dec_ref_known(v_x_153_, 2);
v_str_158_ = lean_ctor_get(v_head_154_, 1);
lean_inc_ref(v_str_158_);
lean_dec_ref_known(v_head_154_, 2);
v___x_159_ = ((lean_object*)(lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1___closed__0));
v___x_160_ = lean_unsigned_to_nat(0u);
v___x_161_ = lean_string_utf8_byte_size(v_str_158_);
v___x_162_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_162_, 0, v_str_158_);
lean_ctor_set(v___x_162_, 1, v___x_160_);
lean_ctor_set(v___x_162_, 2, v___x_161_);
v___x_163_ = lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg(v___x_162_, v___x_159_);
v___x_164_ = l_Lean_Name_str___override(v_x_152_, v___x_163_);
v_x_152_ = v___x_164_;
v_x_153_ = v_tail_157_;
goto _start;
}
default: 
{
lean_object* v_tail_166_; lean_object* v_i_167_; lean_object* v___x_168_; 
lean_inc_ref(v_head_154_);
v_tail_166_ = lean_ctor_get(v_x_153_, 1);
lean_inc(v_tail_166_);
lean_dec_ref_known(v_x_153_, 2);
v_i_167_ = lean_ctor_get(v_head_154_, 1);
lean_inc(v_i_167_);
lean_dec_ref_known(v_head_154_, 2);
v___x_168_ = l_Lean_Name_num___override(v_x_152_, v_i_167_);
v_x_152_ = v___x_168_;
v_x_153_ = v_tail_166_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote_encodeNameForExport(lean_object* v_n_170_){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_box(0);
v___x_172_ = l_Lean_Name_componentsRev(v_n_170_);
v___x_173_ = lp_batteries_List_foldl___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__1(v___x_171_, v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0(lean_object* v_s_174_, lean_object* v_pattern_175_, lean_object* v_replacement_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___redArg(v_s_174_, v_replacement_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0___boxed(lean_object* v_s_178_, lean_object* v_pattern_179_, lean_object* v_replacement_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_batteries_String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0(v_s_178_, v_pattern_179_, v_replacement_180_);
lean_dec_ref(v_replacement_180_);
lean_dec_ref(v_pattern_179_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0(lean_object* v_s_182_, lean_object* v_replacement_183_, lean_object* v_inst_184_, lean_object* v_R_185_, lean_object* v_a_186_, lean_object* v_b_187_, lean_object* v_c_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___redArg(v_s_182_, v_replacement_183_, v_a_186_, v_b_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0___boxed(lean_object* v_s_190_, lean_object* v_replacement_191_, lean_object* v_inst_192_, lean_object* v_R_193_, lean_object* v_a_194_, lean_object* v_b_195_, lean_object* v_c_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Batteries_Util_LibraryNote_encodeNameForExport_spec__0_spec__0(v_s_190_, v_replacement_191_, v_inst_192_, v_R_193_, v_a_194_, v_b_195_, v_c_196_);
lean_dec_ref(v_replacement_191_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(lean_object* v_es_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lean_array_mk(v_es_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0(lean_object* v_as_200_, size_t v_i_201_, size_t v_stop_202_, lean_object* v_b_203_){
_start:
{
uint8_t v___x_204_; 
v___x_204_ = lean_usize_dec_eq(v_i_201_, v_stop_202_);
if (v___x_204_ == 0)
{
lean_object* v___x_205_; lean_object* v___x_206_; size_t v___x_207_; size_t v___x_208_; 
v___x_205_ = lean_array_uget_borrowed(v_as_200_, v_i_201_);
v___x_206_ = l_Array_append___redArg(v_b_203_, v___x_205_);
v___x_207_ = ((size_t)1ULL);
v___x_208_ = lean_usize_add(v_i_201_, v___x_207_);
v_i_201_ = v___x_208_;
v_b_203_ = v___x_206_;
goto _start;
}
else
{
return v_b_203_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_210_, lean_object* v_i_211_, lean_object* v_stop_212_, lean_object* v_b_213_){
_start:
{
size_t v_i_boxed_214_; size_t v_stop_boxed_215_; lean_object* v_res_216_; 
v_i_boxed_214_ = lean_unbox_usize(v_i_211_);
lean_dec(v_i_211_);
v_stop_boxed_215_ = lean_unbox_usize(v_stop_212_);
lean_dec(v_stop_212_);
v_res_216_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0(v_as_210_, v_i_boxed_214_, v_stop_boxed_215_, v_b_213_);
lean_dec_ref(v_as_210_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(lean_object* v___y_219_){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = ((lean_object*)(lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1___closed__0_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_));
v___x_222_ = lean_array_get_size(v___y_219_);
v___x_223_ = lean_nat_dec_lt(v___x_220_, v___x_222_);
if (v___x_223_ == 0)
{
return v___x_221_;
}
else
{
uint8_t v___x_224_; 
v___x_224_ = lean_nat_dec_le(v___x_222_, v___x_222_);
if (v___x_224_ == 0)
{
if (v___x_223_ == 0)
{
return v___x_221_;
}
else
{
size_t v___x_225_; size_t v___x_226_; lean_object* v___x_227_; 
v___x_225_ = ((size_t)0ULL);
v___x_226_ = lean_usize_of_nat(v___x_222_);
v___x_227_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0(v___y_219_, v___x_225_, v___x_226_, v___x_221_);
return v___x_227_;
}
}
else
{
size_t v___x_228_; size_t v___x_229_; lean_object* v___x_230_; 
v___x_228_ = ((size_t)0ULL);
v___x_229_ = lean_usize_of_nat(v___x_222_);
v___x_230_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2__spec__0(v___y_219_, v___x_228_, v___x_229_, v___x_221_);
return v___x_230_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2____boxed(lean_object* v___y_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___lam__1_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(v___y_231_);
lean_dec_ref(v___y_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_253_ = ((lean_object*)(lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__8_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_));
v___x_254_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2____boxed(lean_object* v_a_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_();
return v_res_256_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_307_ = lean_box(0);
v___x_308_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_309_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v___x_307_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg(){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___closed__0);
v___x_312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg___boxed(lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg();
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0(lean_object* v_00_u03b1_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg();
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___boxed(lean_object* v_00_u03b1_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0(v_00_u03b1_320_, v___y_321_, v___y_322_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg(lean_object* v___y_325_){
_start:
{
lean_object* v___x_327_; lean_object* v_env_328_; lean_object* v___x_329_; lean_object* v_mainModule_330_; lean_object* v___x_331_; 
v___x_327_ = lean_st_ref_get(v___y_325_);
v_env_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc_ref(v_env_328_);
lean_dec(v___x_327_);
v___x_329_ = l_Lean_Environment_header(v_env_328_);
lean_dec_ref(v_env_328_);
v_mainModule_330_ = lean_ctor_get(v___x_329_, 0);
lean_inc(v_mainModule_330_);
lean_dec_ref(v___x_329_);
v___x_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_331_, 0, v_mainModule_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg___boxed(lean_object* v___y_332_, lean_object* v___y_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg(v___y_332_);
lean_dec(v___y_332_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1(lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg(v___y_336_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___boxed(lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1(v___y_339_, v___y_340_);
lean_dec(v___y_340_);
lean_dec_ref(v___y_339_);
return v_res_342_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9(void){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = l_Array_mkArray0(lean_box(0));
return v___x_361_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_405_ = ((lean_object*)(lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn___closed__4_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_));
v___x_406_ = l_String_toRawSubstring_x27(v___x_405_);
return v___x_406_;
}
}
static lean_object* _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38(void){
_start:
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__37));
v___x_433_ = l_String_toRawSubstring_x27(v___x_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1(lean_object* v_x_458_, lean_object* v_a_459_, lean_object* v_a_460_){
_start:
{
lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_462_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote_commandLibrary__note_______00__closed__1));
lean_inc(v_x_458_);
v___x_463_ = l_Lean_Syntax_isOfKind(v_x_458_, v___x_462_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; 
lean_dec(v_x_458_);
v___x_464_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__0___redArg();
return v___x_464_;
}
else
{
lean_object* v___x_465_; lean_object* v_env_466_; lean_object* v_messages_467_; lean_object* v_scopes_468_; lean_object* v_usedQuotCtxts_469_; lean_object* v_nextMacroScope_470_; lean_object* v_maxRecDepth_471_; lean_object* v_ngen_472_; lean_object* v_auxDeclNGen_473_; lean_object* v_infoState_474_; lean_object* v_traceState_475_; lean_object* v_snapshotTasks_476_; lean_object* v_prevLinterStates_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_577_; 
v___x_465_ = lean_st_ref_take(v_a_460_);
v_env_466_ = lean_ctor_get(v___x_465_, 0);
v_messages_467_ = lean_ctor_get(v___x_465_, 1);
v_scopes_468_ = lean_ctor_get(v___x_465_, 2);
v_usedQuotCtxts_469_ = lean_ctor_get(v___x_465_, 3);
v_nextMacroScope_470_ = lean_ctor_get(v___x_465_, 4);
v_maxRecDepth_471_ = lean_ctor_get(v___x_465_, 5);
v_ngen_472_ = lean_ctor_get(v___x_465_, 6);
v_auxDeclNGen_473_ = lean_ctor_get(v___x_465_, 7);
v_infoState_474_ = lean_ctor_get(v___x_465_, 8);
v_traceState_475_ = lean_ctor_get(v___x_465_, 9);
v_snapshotTasks_476_ = lean_ctor_get(v___x_465_, 10);
v_prevLinterStates_477_ = lean_ctor_get(v___x_465_, 11);
v_isSharedCheck_577_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_577_ == 0)
{
v___x_479_ = v___x_465_;
v_isShared_480_ = v_isSharedCheck_577_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_prevLinterStates_477_);
lean_inc(v_snapshotTasks_476_);
lean_inc(v_traceState_475_);
lean_inc(v_infoState_474_);
lean_inc(v_auxDeclNGen_473_);
lean_inc(v_ngen_472_);
lean_inc(v_maxRecDepth_471_);
lean_inc(v_nextMacroScope_470_);
lean_inc(v_usedQuotCtxts_469_);
lean_inc(v_scopes_468_);
lean_inc(v_messages_467_);
lean_inc(v_env_466_);
lean_dec(v___x_465_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_577_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; lean_object* v_toEnvExtension_482_; lean_object* v_asyncMode_483_; lean_object* v___x_484_; lean_object* v_name_485_; lean_object* v_origName_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_490_; 
v___x_481_ = lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt;
v_toEnvExtension_482_ = lean_ctor_get(v___x_481_, 0);
v_asyncMode_483_ = lean_ctor_get(v_toEnvExtension_482_, 2);
v___x_484_ = lean_unsigned_to_nat(1u);
v_name_485_ = l_Lean_Syntax_getArg(v_x_458_, v___x_484_);
v_origName_486_ = l_Lean_TSyntax_getId(v_name_485_);
lean_dec(v_name_485_);
v___x_487_ = lean_box(0);
lean_inc(v_origName_486_);
v___x_488_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_481_, v_env_466_, v_origName_486_, v_asyncMode_483_, v___x_487_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 0, v___x_488_);
v___x_490_ = v___x_479_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v___x_488_);
lean_ctor_set(v_reuseFailAlloc_576_, 1, v_messages_467_);
lean_ctor_set(v_reuseFailAlloc_576_, 2, v_scopes_468_);
lean_ctor_set(v_reuseFailAlloc_576_, 3, v_usedQuotCtxts_469_);
lean_ctor_set(v_reuseFailAlloc_576_, 4, v_nextMacroScope_470_);
lean_ctor_set(v_reuseFailAlloc_576_, 5, v_maxRecDepth_471_);
lean_ctor_set(v_reuseFailAlloc_576_, 6, v_ngen_472_);
lean_ctor_set(v_reuseFailAlloc_576_, 7, v_auxDeclNGen_473_);
lean_ctor_set(v_reuseFailAlloc_576_, 8, v_infoState_474_);
lean_ctor_set(v_reuseFailAlloc_576_, 9, v_traceState_475_);
lean_ctor_set(v_reuseFailAlloc_576_, 10, v_snapshotTasks_476_);
lean_ctor_set(v_reuseFailAlloc_576_, 11, v_prevLinterStates_477_);
v___x_490_ = v_reuseFailAlloc_576_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_491_ = lean_st_ref_set(v_a_460_, v___x_490_);
v___x_492_ = l_Lean_Elab_Command_getRef___redArg(v_a_459_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v___x_494_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
lean_inc(v_a_493_);
lean_dec_ref_known(v___x_492_, 1);
v___x_494_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_459_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; lean_object* v_quotContext_x3f_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; uint8_t v___x_500_; lean_object* v___x_501_; lean_object* v_a_503_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc(v_a_495_);
lean_dec_ref_known(v___x_494_, 1);
v_quotContext_x3f_496_ = lean_ctor_get(v_a_459_, 5);
v___x_497_ = lean_unsigned_to_nat(3u);
v___x_498_ = l_Lean_Syntax_getArg(v_x_458_, v___x_497_);
lean_dec(v_x_458_);
v___x_499_ = lp_batteries_Batteries_Util_LibraryNote_encodeNameForExport(v_origName_486_);
v___x_500_ = 0;
v___x_501_ = l_Lean_SourceInfo_fromRef(v_a_493_, v___x_500_);
lean_dec(v_a_493_);
if (lean_obj_tag(v_quotContext_x3f_496_) == 0)
{
lean_object* v___x_557_; lean_object* v_a_558_; 
v___x_557_ = lp_batteries_Lean_getMainModule___at___00Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1_spec__1___redArg(v_a_460_);
v_a_558_ = lean_ctor_get(v___x_557_, 0);
lean_inc(v_a_558_);
lean_dec_ref(v___x_557_);
v_a_503_ = v_a_558_;
goto v___jp_502_;
}
else
{
lean_object* v_val_559_; 
v_val_559_ = lean_ctor_get(v_quotContext_x3f_496_, 0);
lean_inc(v_val_559_);
v_a_503_ = v_val_559_;
goto v___jp_502_;
}
v___jp_502_:
{
lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_504_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__4));
v___x_505_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__6));
v___x_506_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__8));
lean_inc_n(v___x_501_, 17);
v___x_507_ = l_Lean_Syntax_node1(v___x_501_, v___x_506_, v___x_498_);
v___x_508_ = lean_obj_once(&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9, &lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9_once, _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__9);
v___x_509_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_509_, 0, v___x_501_);
lean_ctor_set(v___x_509_, 1, v___x_506_);
lean_ctor_set(v___x_509_, 2, v___x_508_);
v___x_510_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__10));
v___x_511_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__11));
v___x_512_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_512_, 0, v___x_501_);
lean_ctor_set(v___x_512_, 1, v___x_510_);
v___x_513_ = l_Lean_Syntax_node1(v___x_501_, v___x_511_, v___x_512_);
v___x_514_ = l_Lean_Syntax_node1(v___x_501_, v___x_506_, v___x_513_);
lean_inc_ref_n(v___x_509_, 9);
v___x_515_ = l_Lean_Syntax_node7(v___x_501_, v___x_505_, v___x_507_, v___x_509_, v___x_509_, v___x_509_, v___x_514_, v___x_509_, v___x_509_);
v___x_516_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__13));
v___x_517_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__14));
v___x_518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_518_, 0, v___x_501_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
v___x_519_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__16));
v___x_520_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__18));
v___x_521_ = l_Lean_Name_append(v___x_520_, v___x_499_);
v___x_522_ = l_Lean_mkIdent(v___x_521_);
v___x_523_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__19));
v___x_524_ = lean_box(2);
v___x_525_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__20));
v___x_526_ = lean_unsigned_to_nat(2u);
v___x_527_ = lean_mk_empty_array_with_capacity(v___x_526_);
v___x_528_ = lean_array_push(v___x_527_, v___x_522_);
v___x_529_ = lean_array_push(v___x_528_, v___x_525_);
v___x_530_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_530_, 0, v___x_524_);
lean_ctor_set(v___x_530_, 1, v___x_519_);
lean_ctor_set(v___x_530_, 2, v___x_529_);
v___x_531_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__22));
v___x_532_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__25));
v___x_533_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__26));
v___x_534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_501_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
v___x_535_ = lean_obj_once(&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27, &lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27_once, _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__27);
v___x_536_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__28));
lean_inc(v_a_495_);
lean_inc(v_a_503_);
v___x_537_ = l_Lean_addMacroScope(v_a_503_, v___x_536_, v_a_495_);
v___x_538_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__33));
v___x_539_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_539_, 0, v___x_501_);
lean_ctor_set(v___x_539_, 1, v___x_535_);
lean_ctor_set(v___x_539_, 2, v___x_537_);
lean_ctor_set(v___x_539_, 3, v___x_538_);
v___x_540_ = l_Lean_Syntax_node2(v___x_501_, v___x_532_, v___x_534_, v___x_539_);
v___x_541_ = l_Lean_Syntax_node1(v___x_501_, v___x_506_, v___x_540_);
v___x_542_ = l_Lean_Syntax_node2(v___x_501_, v___x_531_, v___x_509_, v___x_541_);
v___x_543_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__35));
v___x_544_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__36));
v___x_545_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_501_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
v___x_546_ = lean_obj_once(&lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38, &lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38_once, _init_lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__38);
v___x_547_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__39));
v___x_548_ = l_Lean_addMacroScope(v_a_503_, v___x_547_, v_a_495_);
v___x_549_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__45));
v___x_550_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_550_, 0, v___x_501_);
lean_ctor_set(v___x_550_, 1, v___x_546_);
lean_ctor_set(v___x_550_, 2, v___x_548_);
lean_ctor_set(v___x_550_, 3, v___x_549_);
v___x_551_ = ((lean_object*)(lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___closed__48));
v___x_552_ = l_Lean_Syntax_node2(v___x_501_, v___x_551_, v___x_509_, v___x_509_);
v___x_553_ = l_Lean_Syntax_node4(v___x_501_, v___x_543_, v___x_545_, v___x_550_, v___x_552_, v___x_509_);
v___x_554_ = l_Lean_Syntax_node5(v___x_501_, v___x_516_, v___x_518_, v___x_530_, v___x_542_, v___x_553_, v___x_509_);
v___x_555_ = l_Lean_Syntax_node2(v___x_501_, v___x_504_, v___x_515_, v___x_554_);
v___x_556_ = l_Lean_Elab_Command_elabCommandTopLevel(v___x_555_, v___x_523_, v_a_459_, v_a_460_);
return v___x_556_;
}
}
else
{
lean_object* v_a_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_567_; 
lean_dec(v_a_493_);
lean_dec(v_origName_486_);
lean_dec(v_x_458_);
v_a_560_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_567_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_567_ == 0)
{
v___x_562_ = v___x_494_;
v_isShared_563_ = v_isSharedCheck_567_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_a_560_);
lean_dec(v___x_494_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_567_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v___x_565_; 
if (v_isShared_563_ == 0)
{
v___x_565_ = v___x_562_;
goto v_reusejp_564_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v_a_560_);
v___x_565_ = v_reuseFailAlloc_566_;
goto v_reusejp_564_;
}
v_reusejp_564_:
{
return v___x_565_;
}
}
}
}
else
{
lean_object* v_a_568_; lean_object* v___x_570_; uint8_t v_isShared_571_; uint8_t v_isSharedCheck_575_; 
lean_dec(v_origName_486_);
lean_dec(v_x_458_);
v_a_568_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_575_ == 0)
{
v___x_570_ = v___x_492_;
v_isShared_571_ = v_isSharedCheck_575_;
goto v_resetjp_569_;
}
else
{
lean_inc(v_a_568_);
lean_dec(v___x_492_);
v___x_570_ = lean_box(0);
v_isShared_571_ = v_isSharedCheck_575_;
goto v_resetjp_569_;
}
v_resetjp_569_:
{
lean_object* v___x_573_; 
if (v_isShared_571_ == 0)
{
v___x_573_ = v___x_570_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v_a_568_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1___boxed(lean_object* v_x_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_batteries_Batteries_Util_LibraryNote___aux__Batteries__Util__LibraryNote______elabRules__Batteries__Util__LibraryNote__commandLibrary__note________1(v_x_578_, v_a_579_, v_a_580_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
return v_res_582_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_Util_instInhabitedLibraryNote___aux__1 = _init_lp_batteries_Batteries_Util_instInhabitedLibraryNote___aux__1();
lean_mark_persistent(lp_batteries_Batteries_Util_instInhabitedLibraryNote___aux__1);
lp_batteries_Batteries_Util_instInhabitedLibraryNote = _init_lp_batteries_Batteries_Util_instInhabitedLibraryNote();
lean_mark_persistent(lp_batteries_Batteries_Util_instInhabitedLibraryNote);
lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry___aux__1 = _init_lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry___aux__1();
lean_mark_persistent(lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry___aux__1);
lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry = _init_lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry();
lean_mark_persistent(lp_batteries_Batteries_Util_LibraryNote_instInhabitedLibraryNoteEntry);
res = lp_batteries___private_Batteries_Util_LibraryNote_0__Batteries_Util_LibraryNote_initFn_00___x40_Batteries_Util_LibraryNote_1081991270____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin) {
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
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Util_LibraryNote(builtin);
}
#ifdef __cplusplus
}
#endif
