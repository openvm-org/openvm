// Lean compiler output
// Module: Batteries.CodeAction.Match
// Imports: public import Init public meta import Init public meta import Batteries.CodeAction.Misc public meta import Batteries.Data.List.Basic
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_batteries_Batteries_CodeAction_findTermInfoWithCtx_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_inferType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_TermInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Server_Snapshots_Snapshot_env(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_CodeAction_getExplicitArgs(lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_CodeAction_getAllArgs(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isAtom(lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
uint8_t l_Lean_Name_hasNum(lean_object*);
uint8_t l_Lean_Name_isInternal(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
extern lean_object* l_instInhabitedError;
lean_object* l_instInhabitedEIO___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lp_batteries_List_sectionsTR___redArg(lean_object*);
lean_object* l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(lean_object*);
lean_object* l_Lean_FileMap_utf8RangeToLspRange(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_WorkspaceEdit_ofTextEdit(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Info_stx(lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_Lsp_instOrdPosition_ord(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "match"};
static const lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 208, 235, 82, 91, 230, 203, 159)}};
static const lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4_value;
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_isMatchTerm(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___boxed(lean_object*);
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "matchDiscr"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 51, 127, 238, 206, 239, 57, 130)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "motive"};
static const lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(81, 58, 182, 248, 224, 44, 170, 90)}};
static const lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "generalizingParam"};
static const lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(147, 206, 52, 232, 193, 222, 34, 109)}};
static const lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f(lean_object*);
static lean_once_cell_t lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findAllInfos_loop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_CodeAction_findAllInfos___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_findAllInfos___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_findAllInfos___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findAllInfos(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__0 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__0_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__1 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__1_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__2 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__2_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__3 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__3_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__4 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__4_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__5 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__5_value;
static const lean_closure_object lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__6 = (const lean_object*)&lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__6_value;
LEAN_EXPORT uint8_t lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___boxed(lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Batteries.CodeAction.Match"};
static const lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Batteries.CodeAction.hasImplicitNonparArg"};
static const lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "bad inductive"};
static const lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3;
static const lean_string_object lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not an inductive"};
static const lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5;
static const lean_array_object lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__6_value;
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_hasImplicitNonparArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_patternFromConstructor_spec__0(lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__2_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__3_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__4_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " _"};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Batteries.CodeAction.patternFromConstructor"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__3_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Option"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__7_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__8_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__9 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__10 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__11 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__11_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "some"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__12 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__12_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__13 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__14 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__14_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__15 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__15_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__16 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__16_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__17 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__17_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " :: "};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__18 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__18_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__19 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__20 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__20_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__21 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__21_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "succ"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__22 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__22_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " + 1"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__23 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__23_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__24 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__24_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__24_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__25 = (const lean_object*)&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__25_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_matchExpand_spec__5(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Batteries.CodeAction.matchExpand"};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__4_value;
static const lean_string_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "| "};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__0 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__0_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " => _"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__1 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "quickfix"};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Batteries_CodeAction_matchExpand___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_matchExpand___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Generate a list of equations for this match."};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_matchExpand___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Generate a list of equations with implicit arguments for this match."};
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___closed__2_value;
static const lean_closure_object lp_batteries_Batteries_CodeAction_matchExpand___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_CodeAction_isMatchTerm___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_CodeAction_matchExpand___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_matchExpand___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_isMatchTerm(lean_object* v_x_10_){
_start:
{
if (lean_obj_tag(v_x_10_) == 1)
{
lean_object* v_i_11_; lean_object* v_toElabInfo_12_; lean_object* v_stx_13_; lean_object* v___x_14_; uint8_t v___x_15_; 
v_i_11_ = lean_ctor_get(v_x_10_, 0);
lean_inc_ref(v_i_11_);
lean_dec_ref_known(v_x_10_, 1);
v_toElabInfo_12_ = lean_ctor_get(v_i_11_, 0);
lean_inc_ref(v_toElabInfo_12_);
lean_dec_ref(v_i_11_);
v_stx_13_ = lean_ctor_get(v_toElabInfo_12_, 1);
lean_inc(v_stx_13_);
lean_dec_ref(v_toElabInfo_12_);
v___x_14_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4));
v___x_15_ = l_Lean_Syntax_isOfKind(v_stx_13_, v___x_14_);
return v___x_15_;
}
else
{
uint8_t v___x_16_; 
lean_dec_ref(v_x_10_);
v___x_16_ = 0;
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_isMatchTerm___boxed(lean_object* v_x_17_){
_start:
{
uint8_t v_res_18_; lean_object* v_r_19_; 
v_res_18_ = lp_batteries_Batteries_CodeAction_isMatchTerm(v_x_17_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2(lean_object* v_as_24_, size_t v_sz_25_, size_t v_i_26_, lean_object* v_b_27_){
_start:
{
lean_object* v_a_29_; uint8_t v___x_33_; 
v___x_33_ = lean_usize_dec_lt(v_i_26_, v_sz_25_);
if (v___x_33_ == 0)
{
lean_inc_ref(v_b_27_);
return v_b_27_;
}
else
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v_a_36_; uint8_t v___x_37_; 
v___x_34_ = lean_box(0);
v___x_35_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0));
v_a_36_ = lean_array_uget_borrowed(v_as_24_, v_i_26_);
v___x_37_ = l_Lean_Syntax_isAtom(v_a_36_);
if (v___x_37_ == 0)
{
v_a_29_ = v___x_35_;
goto v___jp_28_;
}
else
{
lean_object* v___x_38_; lean_object* v___x_39_; uint8_t v___x_40_; 
v___x_38_ = l_Lean_Syntax_getAtomVal(v_a_36_);
v___x_39_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__1));
v___x_40_ = lean_string_dec_eq(v___x_38_, v___x_39_);
lean_dec_ref(v___x_38_);
if (v___x_40_ == 0)
{
v_a_29_ = v___x_35_;
goto v___jp_28_;
}
else
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
lean_inc(v_a_36_);
v___x_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_41_, 0, v_a_36_);
v___x_42_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
v___x_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v___x_34_);
return v___x_43_;
}
}
}
v___jp_28_:
{
size_t v___x_30_; size_t v___x_31_; 
v___x_30_ = ((size_t)1ULL);
v___x_31_ = lean_usize_add(v_i_26_, v___x_30_);
v_i_26_ = v___x_31_;
v_b_27_ = v_a_29_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___boxed(lean_object* v_as_44_, lean_object* v_sz_45_, lean_object* v_i_46_, lean_object* v_b_47_){
_start:
{
size_t v_sz_boxed_48_; size_t v_i_boxed_49_; lean_object* v_res_50_; 
v_sz_boxed_48_ = lean_unbox_usize(v_sz_45_);
lean_dec(v_sz_45_);
v_i_boxed_49_ = lean_unbox_usize(v_i_46_);
lean_dec(v_i_46_);
v_res_50_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2(v_as_44_, v_sz_boxed_48_, v_i_boxed_49_, v_b_47_);
lean_dec_ref(v_b_47_);
lean_dec_ref(v_as_44_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1(lean_object* v_as_51_, size_t v_sz_52_, size_t v_i_53_, lean_object* v_b_54_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = lean_usize_dec_lt(v_i_53_, v_sz_52_);
if (v___x_55_ == 0)
{
lean_inc_ref(v_b_54_);
return v_b_54_;
}
else
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v_a_58_; uint8_t v___y_60_; uint8_t v___x_67_; 
v___x_56_ = lean_box(0);
v___x_57_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0));
v_a_58_ = lean_array_uget_borrowed(v_as_51_, v_i_53_);
v___x_67_ = l_Lean_Syntax_isAtom(v_a_58_);
if (v___x_67_ == 0)
{
v___y_60_ = v___x_67_;
goto v___jp_59_;
}
else
{
lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_68_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__3));
v___x_69_ = l_Lean_Syntax_getAtomVal(v_a_58_);
v___x_70_ = lean_string_dec_eq(v___x_69_, v___x_68_);
lean_dec_ref(v___x_69_);
v___y_60_ = v___x_70_;
goto v___jp_59_;
}
v___jp_59_:
{
if (v___y_60_ == 0)
{
size_t v___x_61_; size_t v___x_62_; 
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_53_, v___x_61_);
v_i_53_ = v___x_62_;
v_b_54_ = v___x_57_;
goto _start;
}
else
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
lean_inc(v_a_58_);
v___x_64_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_64_, 0, v_a_58_);
v___x_65_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_56_);
return v___x_66_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1___boxed(lean_object* v_as_71_, lean_object* v_sz_72_, lean_object* v_i_73_, lean_object* v_b_74_){
_start:
{
size_t v_sz_boxed_75_; size_t v_i_boxed_76_; lean_object* v_res_77_; 
v_sz_boxed_75_ = lean_unbox_usize(v_sz_72_);
lean_dec(v_sz_72_);
v_i_boxed_76_ = lean_unbox_usize(v_i_73_);
lean_dec(v_i_73_);
v_res_77_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1(v_as_71_, v_sz_boxed_75_, v_i_boxed_76_, v_b_74_);
lean_dec_ref(v_b_74_);
lean_dec_ref(v_as_71_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0(size_t v_sz_84_, size_t v_i_85_, lean_object* v_bs_86_){
_start:
{
uint8_t v___x_87_; 
v___x_87_ = lean_usize_dec_lt(v_i_85_, v_sz_84_);
if (v___x_87_ == 0)
{
lean_object* v___x_88_; 
v___x_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_88_, 0, v_bs_86_);
return v___x_88_;
}
else
{
lean_object* v_v_89_; lean_object* v___x_90_; uint8_t v___x_91_; 
v_v_89_ = lean_array_uget(v_bs_86_, v_i_85_);
v___x_90_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1));
lean_inc(v_v_89_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_v_89_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; 
lean_dec(v_v_89_);
lean_dec_ref(v_bs_86_);
v___x_92_ = lean_box(0);
return v___x_92_;
}
else
{
lean_object* v___x_93_; lean_object* v_bs_x27_94_; size_t v___x_95_; size_t v___x_96_; lean_object* v___x_97_; 
v___x_93_ = lean_unsigned_to_nat(0u);
v_bs_x27_94_ = lean_array_uset(v_bs_86_, v_i_85_, v___x_93_);
v___x_95_ = ((size_t)1ULL);
v___x_96_ = lean_usize_add(v_i_85_, v___x_95_);
v___x_97_ = lean_array_uset(v_bs_x27_94_, v_i_85_, v_v_89_);
v_i_85_ = v___x_96_;
v_bs_86_ = v___x_97_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___boxed(lean_object* v_sz_99_, lean_object* v_i_100_, lean_object* v_bs_101_){
_start:
{
size_t v_sz_boxed_102_; size_t v_i_boxed_103_; lean_object* v_res_104_; 
v_sz_boxed_102_ = lean_unbox_usize(v_sz_99_);
lean_dec(v_sz_99_);
v_i_boxed_103_ = lean_unbox_usize(v_i_100_);
lean_dec(v_i_100_);
v_res_104_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0(v_sz_boxed_102_, v_i_boxed_103_, v_bs_101_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(uint8_t v___x_105_, lean_object* v_as_106_, size_t v_i_107_, size_t v_stop_108_, lean_object* v_b_109_){
_start:
{
lean_object* v___y_111_; uint8_t v___x_115_; 
v___x_115_ = lean_usize_dec_eq(v_i_107_, v_stop_108_);
if (v___x_115_ == 0)
{
lean_object* v_fst_116_; uint8_t v___x_117_; 
v_fst_116_ = lean_ctor_get(v_b_109_, 0);
v___x_117_ = lean_unbox(v_fst_116_);
if (v___x_117_ == 0)
{
lean_object* v_snd_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_126_; 
v_snd_118_ = lean_ctor_get(v_b_109_, 1);
v_isSharedCheck_126_ = !lean_is_exclusive(v_b_109_);
if (v_isSharedCheck_126_ == 0)
{
lean_object* v_unused_127_; 
v_unused_127_ = lean_ctor_get(v_b_109_, 0);
lean_dec(v_unused_127_);
v___x_120_ = v_b_109_;
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_snd_118_);
lean_dec(v_b_109_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_122_; lean_object* v___x_124_; 
v___x_122_ = lean_box(v___x_105_);
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 0, v___x_122_);
v___x_124_ = v___x_120_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_122_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v_snd_118_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
v___y_111_ = v___x_124_;
goto v___jp_110_;
}
}
}
else
{
lean_object* v_snd_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_138_; 
v_snd_128_ = lean_ctor_get(v_b_109_, 1);
v_isSharedCheck_138_ = !lean_is_exclusive(v_b_109_);
if (v_isSharedCheck_138_ == 0)
{
lean_object* v_unused_139_; 
v_unused_139_ = lean_ctor_get(v_b_109_, 0);
lean_dec(v_unused_139_);
v___x_130_ = v_b_109_;
v_isShared_131_ = v_isSharedCheck_138_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_snd_128_);
lean_dec(v_b_109_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_138_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_136_; 
v___x_132_ = lean_array_uget_borrowed(v_as_106_, v_i_107_);
lean_inc(v___x_132_);
v___x_133_ = lean_array_push(v_snd_128_, v___x_132_);
v___x_134_ = lean_box(v___x_115_);
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 1, v___x_133_);
lean_ctor_set(v___x_130_, 0, v___x_134_);
v___x_136_ = v___x_130_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v___x_133_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
v___y_111_ = v___x_136_;
goto v___jp_110_;
}
}
}
}
else
{
return v_b_109_;
}
v___jp_110_:
{
size_t v___x_112_; size_t v___x_113_; 
v___x_112_ = ((size_t)1ULL);
v___x_113_ = lean_usize_add(v_i_107_, v___x_112_);
v_i_107_ = v___x_113_;
v_b_109_ = v___y_111_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3___boxed(lean_object* v___x_140_, lean_object* v_as_141_, lean_object* v_i_142_, lean_object* v_stop_143_, lean_object* v_b_144_){
_start:
{
uint8_t v___x_2694__boxed_145_; size_t v_i_boxed_146_; size_t v_stop_boxed_147_; lean_object* v_res_148_; 
v___x_2694__boxed_145_ = lean_unbox(v___x_140_);
v_i_boxed_146_ = lean_unbox_usize(v_i_142_);
lean_dec(v_i_142_);
v_stop_boxed_147_ = lean_unbox_usize(v_stop_143_);
lean_dec(v_stop_143_);
v_res_148_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(v___x_2694__boxed_145_, v_as_141_, v_i_boxed_146_, v_stop_boxed_147_, v_b_144_);
lean_dec_ref(v_as_141_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f(lean_object* v_matchStx_163_){
_start:
{
lean_object* v___y_165_; uint8_t v___y_166_; lean_object* v___y_167_; lean_object* v___y_186_; lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_224_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4));
lean_inc(v_matchStx_163_);
v___x_225_ = l_Lean_Syntax_isOfKind(v_matchStx_163_, v___x_224_);
if (v___x_225_ == 0)
{
lean_object* v___x_245_; 
lean_dec(v_matchStx_163_);
v___x_245_ = lean_box(0);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_258_; uint8_t v___x_259_; 
v___x_246_ = lean_unsigned_to_nat(0u);
v___x_247_ = lean_unsigned_to_nat(1u);
v___x_258_ = l_Lean_Syntax_getArg(v_matchStx_163_, v___x_247_);
v___x_259_ = l_Lean_Syntax_isNone(v___x_258_);
if (v___x_259_ == 0)
{
uint8_t v___x_260_; 
lean_inc(v___x_258_);
v___x_260_ = l_Lean_Syntax_matchesNull(v___x_258_, v___x_247_);
if (v___x_260_ == 0)
{
lean_object* v___x_261_; 
lean_dec(v___x_258_);
lean_dec(v_matchStx_163_);
v___x_261_ = lean_box(0);
return v___x_261_;
}
else
{
lean_object* v___x_262_; lean_object* v___x_263_; uint8_t v___x_264_; 
v___x_262_ = l_Lean_Syntax_getArg(v___x_258_, v___x_246_);
lean_dec(v___x_258_);
v___x_263_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4));
v___x_264_ = l_Lean_Syntax_isOfKind(v___x_262_, v___x_263_);
if (v___x_264_ == 0)
{
lean_object* v___x_265_; 
lean_dec(v_matchStx_163_);
v___x_265_ = lean_box(0);
return v___x_265_;
}
else
{
goto v___jp_248_;
}
}
}
else
{
lean_dec(v___x_258_);
goto v___jp_248_;
}
v___jp_248_:
{
lean_object* v___x_249_; lean_object* v___x_250_; uint8_t v___x_251_; 
v___x_249_ = lean_unsigned_to_nat(2u);
v___x_250_ = l_Lean_Syntax_getArg(v_matchStx_163_, v___x_249_);
v___x_251_ = l_Lean_Syntax_isNone(v___x_250_);
if (v___x_251_ == 0)
{
uint8_t v___x_252_; 
lean_inc(v___x_250_);
v___x_252_ = l_Lean_Syntax_matchesNull(v___x_250_, v___x_247_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; 
lean_dec(v___x_250_);
lean_dec(v_matchStx_163_);
v___x_253_ = lean_box(0);
return v___x_253_;
}
else
{
lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_254_ = l_Lean_Syntax_getArg(v___x_250_, v___x_246_);
lean_dec(v___x_250_);
v___x_255_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__2));
v___x_256_ = l_Lean_Syntax_isOfKind(v___x_254_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; 
lean_dec(v_matchStx_163_);
v___x_257_ = lean_box(0);
return v___x_257_;
}
else
{
goto v___jp_226_;
}
}
}
else
{
lean_dec(v___x_250_);
goto v___jp_226_;
}
}
}
v___jp_164_:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_168_ = lean_array_get_size(v___y_165_);
v___x_169_ = lean_unsigned_to_nat(1u);
v___x_170_ = lean_nat_sub(v___x_168_, v___x_169_);
v___x_171_ = lean_nat_dec_lt(v___x_170_, v___x_168_);
if (v___x_171_ == 0)
{
lean_object* v___x_172_; 
lean_dec(v___x_170_);
lean_dec(v___y_167_);
lean_dec_ref(v___y_165_);
v___x_172_ = lean_box(0);
return v___x_172_;
}
else
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = lean_array_fget(v___y_165_, v___x_170_);
lean_dec(v___x_170_);
lean_dec_ref(v___y_165_);
v___x_174_ = l_Lean_Syntax_getTailPos_x3f(v___x_173_, v___y_166_);
lean_dec(v___x_173_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v___x_175_; 
lean_dec(v___y_167_);
v___x_175_ = lean_box(0);
return v___x_175_;
}
else
{
lean_object* v_val_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_184_; 
v_val_176_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_184_ == 0)
{
v___x_178_ = v___x_174_;
v_isShared_179_ = v_isSharedCheck_184_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_val_176_);
lean_dec(v___x_174_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_184_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_180_; lean_object* v___x_182_; 
v___x_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_180_, 0, v___y_167_);
lean_ctor_set(v___x_180_, 1, v_val_176_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 0, v___x_180_);
v___x_182_ = v___x_178_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v___x_180_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
}
}
v___jp_185_:
{
size_t v_sz_187_; size_t v___x_188_; lean_object* v___x_189_; 
v_sz_187_ = lean_array_size(v___y_186_);
v___x_188_ = ((size_t)0ULL);
v___x_189_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0(v_sz_187_, v___x_188_, v___y_186_);
if (lean_obj_tag(v___x_189_) == 0)
{
lean_object* v___x_190_; 
lean_dec(v_matchStx_163_);
v___x_190_ = lean_box(0);
return v___x_190_;
}
else
{
lean_object* v_val_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; size_t v_sz_195_; lean_object* v___x_196_; lean_object* v_fst_197_; 
v_val_191_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_val_191_);
lean_dec_ref_known(v___x_189_, 1);
v___x_192_ = l_Lean_Syntax_getArgs(v_matchStx_163_);
lean_dec(v_matchStx_163_);
v___x_193_ = lean_box(0);
v___x_194_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0));
v_sz_195_ = lean_array_size(v___x_192_);
v___x_196_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__1(v___x_192_, v_sz_195_, v___x_188_, v___x_194_);
v_fst_197_ = lean_ctor_get(v___x_196_, 0);
lean_inc(v_fst_197_);
lean_dec_ref(v___x_196_);
if (lean_obj_tag(v_fst_197_) == 0)
{
lean_dec_ref(v___x_192_);
lean_dec(v_val_191_);
return v___x_193_;
}
else
{
lean_object* v_val_198_; 
v_val_198_ = lean_ctor_get(v_fst_197_, 0);
lean_inc(v_val_198_);
lean_dec_ref_known(v_fst_197_, 1);
if (lean_obj_tag(v_val_198_) == 0)
{
lean_dec_ref(v___x_192_);
lean_dec(v_val_191_);
return v___x_193_;
}
else
{
lean_object* v_val_199_; uint8_t v___x_200_; lean_object* v___x_201_; 
v_val_199_ = lean_ctor_get(v_val_198_, 0);
lean_inc(v_val_199_);
lean_dec_ref_known(v_val_198_, 1);
v___x_200_ = 0;
v___x_201_ = l_Lean_Syntax_getPos_x3f(v_val_199_, v___x_200_);
lean_dec(v_val_199_);
if (lean_obj_tag(v___x_201_) == 0)
{
lean_dec_ref(v___x_192_);
lean_dec(v_val_191_);
return v___x_193_;
}
else
{
lean_object* v_val_202_; lean_object* v___x_203_; lean_object* v_fst_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_222_; 
v_val_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_val_202_);
lean_dec_ref_known(v___x_201_, 1);
v___x_203_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2(v___x_192_, v_sz_195_, v___x_188_, v___x_194_);
lean_dec_ref(v___x_192_);
v_fst_204_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_222_ == 0)
{
lean_object* v_unused_223_; 
v_unused_223_ = lean_ctor_get(v___x_203_, 1);
lean_dec(v_unused_223_);
v___x_206_ = v___x_203_;
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_fst_204_);
lean_dec(v___x_203_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
if (lean_obj_tag(v_fst_204_) == 0)
{
lean_del_object(v___x_206_);
v___y_165_ = v_val_191_;
v___y_166_ = v___x_200_;
v___y_167_ = v_val_202_;
goto v___jp_164_;
}
else
{
lean_object* v_val_208_; 
v_val_208_ = lean_ctor_get(v_fst_204_, 0);
lean_inc(v_val_208_);
lean_dec_ref_known(v_fst_204_, 1);
if (lean_obj_tag(v_val_208_) == 1)
{
lean_object* v_val_209_; lean_object* v___x_210_; 
lean_dec(v_val_191_);
v_val_209_ = lean_ctor_get(v_val_208_, 0);
lean_inc(v_val_209_);
lean_dec_ref_known(v_val_208_, 1);
v___x_210_ = l_Lean_Syntax_getTailPos_x3f(v_val_209_, v___x_200_);
lean_dec(v_val_209_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_del_object(v___x_206_);
lean_dec(v_val_202_);
return v___x_193_;
}
else
{
lean_object* v_val_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_221_; 
v_val_211_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_221_ == 0)
{
v___x_213_ = v___x_210_;
v_isShared_214_ = v_isSharedCheck_221_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_val_211_);
lean_dec(v___x_210_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_221_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v___x_216_; 
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 1, v_val_211_);
lean_ctor_set(v___x_206_, 0, v_val_202_);
v___x_216_ = v___x_206_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_val_202_);
lean_ctor_set(v_reuseFailAlloc_220_, 1, v_val_211_);
v___x_216_ = v_reuseFailAlloc_220_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
lean_object* v___x_218_; 
if (v_isShared_214_ == 0)
{
lean_ctor_set(v___x_213_, 0, v___x_216_);
v___x_218_ = v___x_213_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v___x_216_);
v___x_218_ = v_reuseFailAlloc_219_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
return v___x_218_;
}
}
}
}
}
else
{
lean_dec(v_val_208_);
lean_del_object(v___x_206_);
v___y_165_ = v_val_191_;
v___y_166_ = v___x_200_;
v___y_167_ = v_val_202_;
goto v___jp_164_;
}
}
}
}
}
}
}
}
v___jp_226_:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_227_ = lean_unsigned_to_nat(3u);
v___x_228_ = l_Lean_Syntax_getArg(v_matchStx_163_, v___x_227_);
v___x_229_ = l_Lean_Syntax_getArgs(v___x_228_);
lean_dec(v___x_228_);
v___x_230_ = lean_unsigned_to_nat(0u);
v___x_231_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__0));
v___x_232_ = lean_array_get_size(v___x_229_);
v___x_233_ = lean_nat_dec_lt(v___x_230_, v___x_232_);
if (v___x_233_ == 0)
{
lean_dec_ref(v___x_229_);
v___y_186_ = v___x_231_;
goto v___jp_185_;
}
else
{
lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_234_ = lean_box(v___x_225_);
v___x_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
lean_ctor_set(v___x_235_, 1, v___x_231_);
v___x_236_ = lean_nat_dec_le(v___x_232_, v___x_232_);
if (v___x_236_ == 0)
{
if (v___x_233_ == 0)
{
lean_dec_ref_known(v___x_235_, 2);
lean_dec_ref(v___x_229_);
v___y_186_ = v___x_231_;
goto v___jp_185_;
}
else
{
size_t v___x_237_; size_t v___x_238_; lean_object* v___x_239_; lean_object* v_snd_240_; 
v___x_237_ = ((size_t)0ULL);
v___x_238_ = lean_usize_of_nat(v___x_232_);
v___x_239_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(v___x_225_, v___x_229_, v___x_237_, v___x_238_, v___x_235_);
lean_dec_ref(v___x_229_);
v_snd_240_ = lean_ctor_get(v___x_239_, 1);
lean_inc(v_snd_240_);
lean_dec_ref(v___x_239_);
v___y_186_ = v_snd_240_;
goto v___jp_185_;
}
}
else
{
size_t v___x_241_; size_t v___x_242_; lean_object* v___x_243_; lean_object* v_snd_244_; 
v___x_241_ = ((size_t)0ULL);
v___x_242_ = lean_usize_of_nat(v___x_232_);
v___x_243_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(v___x_225_, v___x_229_, v___x_241_, v___x_242_, v___x_235_);
lean_dec_ref(v___x_229_);
v_snd_244_ = lean_ctor_get(v___x_243_, 1);
lean_inc(v_snd_244_);
lean_dec_ref(v___x_243_);
v___y_186_ = v_snd_244_;
goto v___jp_185_;
}
}
}
}
}
static lean_object* _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2(lean_object* v_p_267_, lean_object* v_x_268_, lean_object* v_x_269_){
_start:
{
if (lean_obj_tag(v_x_268_) == 0)
{
lean_object* v_cs_270_; lean_object* v___x_271_; lean_object* v___x_272_; uint8_t v___x_273_; 
v_cs_270_ = lean_ctor_get(v_x_268_, 0);
v___x_271_ = lean_unsigned_to_nat(0u);
v___x_272_ = lean_array_get_size(v_cs_270_);
v___x_273_ = lean_nat_dec_lt(v___x_271_, v___x_272_);
if (v___x_273_ == 0)
{
lean_dec_ref(v_p_267_);
return v_x_269_;
}
else
{
uint8_t v___x_274_; 
v___x_274_ = lean_nat_dec_le(v___x_272_, v___x_272_);
if (v___x_274_ == 0)
{
if (v___x_273_ == 0)
{
lean_dec_ref(v_p_267_);
return v_x_269_;
}
else
{
size_t v___x_275_; size_t v___x_276_; lean_object* v___x_277_; 
v___x_275_ = ((size_t)0ULL);
v___x_276_ = lean_usize_of_nat(v___x_272_);
v___x_277_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(v_p_267_, v_cs_270_, v___x_275_, v___x_276_, v_x_269_);
return v___x_277_;
}
}
else
{
size_t v___x_278_; size_t v___x_279_; lean_object* v___x_280_; 
v___x_278_ = ((size_t)0ULL);
v___x_279_ = lean_usize_of_nat(v___x_272_);
v___x_280_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(v_p_267_, v_cs_270_, v___x_278_, v___x_279_, v_x_269_);
return v___x_280_;
}
}
}
else
{
lean_object* v_vs_281_; lean_object* v___x_282_; lean_object* v___x_283_; uint8_t v___x_284_; 
v_vs_281_ = lean_ctor_get(v_x_268_, 0);
v___x_282_ = lean_unsigned_to_nat(0u);
v___x_283_ = lean_array_get_size(v_vs_281_);
v___x_284_ = lean_nat_dec_lt(v___x_282_, v___x_283_);
if (v___x_284_ == 0)
{
lean_dec_ref(v_p_267_);
return v_x_269_;
}
else
{
uint8_t v___x_285_; 
v___x_285_ = lean_nat_dec_le(v___x_283_, v___x_283_);
if (v___x_285_ == 0)
{
if (v___x_284_ == 0)
{
lean_dec_ref(v_p_267_);
return v_x_269_;
}
else
{
size_t v___x_286_; size_t v___x_287_; lean_object* v___x_288_; 
v___x_286_ = ((size_t)0ULL);
v___x_287_ = lean_usize_of_nat(v___x_283_);
v___x_288_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_267_, v_vs_281_, v___x_286_, v___x_287_, v_x_269_);
return v___x_288_;
}
}
else
{
size_t v___x_289_; size_t v___x_290_; lean_object* v___x_291_; 
v___x_289_ = ((size_t)0ULL);
v___x_290_ = lean_usize_of_nat(v___x_283_);
v___x_291_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_267_, v_vs_281_, v___x_289_, v___x_290_, v_x_269_);
return v___x_291_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(lean_object* v_p_292_, lean_object* v_as_293_, size_t v_i_294_, size_t v_stop_295_, lean_object* v_b_296_){
_start:
{
uint8_t v___x_297_; 
v___x_297_ = lean_usize_dec_eq(v_i_294_, v_stop_295_);
if (v___x_297_ == 0)
{
lean_object* v___x_298_; lean_object* v___x_299_; size_t v___x_300_; size_t v___x_301_; 
v___x_298_ = lean_array_uget_borrowed(v_as_293_, v_i_294_);
lean_inc_ref(v_p_292_);
v___x_299_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2(v_p_292_, v___x_298_, v_b_296_);
v___x_300_ = ((size_t)1ULL);
v___x_301_ = lean_usize_add(v_i_294_, v___x_300_);
v_i_294_ = v___x_301_;
v_b_296_ = v___x_299_;
goto _start;
}
else
{
lean_dec_ref(v_p_292_);
return v_b_296_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0(lean_object* v_p_303_, lean_object* v_x_304_, size_t v_x_305_, size_t v_x_306_, lean_object* v_x_307_){
_start:
{
if (lean_obj_tag(v_x_304_) == 0)
{
lean_object* v_cs_308_; lean_object* v___x_309_; size_t v___x_310_; lean_object* v_j_311_; lean_object* v___x_312_; size_t v___x_313_; size_t v___x_314_; size_t v___x_315_; size_t v___x_316_; size_t v___x_317_; size_t v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; uint8_t v___x_323_; 
v_cs_308_ = lean_ctor_get(v_x_304_, 0);
v___x_309_ = lean_obj_once(&lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0, &lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0_once, _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___closed__0);
v___x_310_ = lean_usize_shift_right(v_x_305_, v_x_306_);
v_j_311_ = lean_usize_to_nat(v___x_310_);
v___x_312_ = lean_array_get_borrowed(v___x_309_, v_cs_308_, v_j_311_);
v___x_313_ = ((size_t)1ULL);
v___x_314_ = lean_usize_shift_left(v___x_313_, v_x_306_);
v___x_315_ = lean_usize_sub(v___x_314_, v___x_313_);
v___x_316_ = lean_usize_land(v_x_305_, v___x_315_);
v___x_317_ = ((size_t)5ULL);
v___x_318_ = lean_usize_sub(v_x_306_, v___x_317_);
lean_inc_ref(v_p_303_);
v___x_319_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0(v_p_303_, v___x_312_, v___x_316_, v___x_318_, v_x_307_);
v___x_320_ = lean_unsigned_to_nat(1u);
v___x_321_ = lean_nat_add(v_j_311_, v___x_320_);
lean_dec(v_j_311_);
v___x_322_ = lean_array_get_size(v_cs_308_);
v___x_323_ = lean_nat_dec_lt(v___x_321_, v___x_322_);
if (v___x_323_ == 0)
{
lean_dec(v___x_321_);
lean_dec_ref(v_p_303_);
return v___x_319_;
}
else
{
uint8_t v___x_324_; 
v___x_324_ = lean_nat_dec_le(v___x_322_, v___x_322_);
if (v___x_324_ == 0)
{
if (v___x_323_ == 0)
{
lean_dec(v___x_321_);
lean_dec_ref(v_p_303_);
return v___x_319_;
}
else
{
size_t v___x_325_; size_t v___x_326_; lean_object* v___x_327_; 
v___x_325_ = lean_usize_of_nat(v___x_321_);
lean_dec(v___x_321_);
v___x_326_ = lean_usize_of_nat(v___x_322_);
v___x_327_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(v_p_303_, v_cs_308_, v___x_325_, v___x_326_, v___x_319_);
return v___x_327_;
}
}
else
{
size_t v___x_328_; size_t v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_usize_of_nat(v___x_321_);
lean_dec(v___x_321_);
v___x_329_ = lean_usize_of_nat(v___x_322_);
v___x_330_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(v_p_303_, v_cs_308_, v___x_328_, v___x_329_, v___x_319_);
return v___x_330_;
}
}
}
else
{
lean_object* v_vs_331_; lean_object* v___x_332_; lean_object* v___x_333_; uint8_t v___x_334_; 
v_vs_331_ = lean_ctor_get(v_x_304_, 0);
v___x_332_ = lean_usize_to_nat(v_x_305_);
v___x_333_ = lean_array_get_size(v_vs_331_);
v___x_334_ = lean_nat_dec_lt(v___x_332_, v___x_333_);
if (v___x_334_ == 0)
{
lean_dec(v___x_332_);
lean_dec_ref(v_p_303_);
return v_x_307_;
}
else
{
uint8_t v___x_335_; 
v___x_335_ = lean_nat_dec_le(v___x_333_, v___x_333_);
if (v___x_335_ == 0)
{
if (v___x_334_ == 0)
{
lean_dec(v___x_332_);
lean_dec_ref(v_p_303_);
return v_x_307_;
}
else
{
size_t v___x_336_; size_t v___x_337_; lean_object* v___x_338_; 
v___x_336_ = lean_usize_of_nat(v___x_332_);
lean_dec(v___x_332_);
v___x_337_ = lean_usize_of_nat(v___x_333_);
v___x_338_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_303_, v_vs_331_, v___x_336_, v___x_337_, v_x_307_);
return v___x_338_;
}
}
else
{
size_t v___x_339_; size_t v___x_340_; lean_object* v___x_341_; 
v___x_339_ = lean_usize_of_nat(v___x_332_);
lean_dec(v___x_332_);
v___x_340_ = lean_usize_of_nat(v___x_333_);
v___x_341_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_303_, v_vs_331_, v___x_339_, v___x_340_, v_x_307_);
return v___x_341_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0(lean_object* v_p_342_, lean_object* v_t_343_, lean_object* v_init_344_, lean_object* v_start_345_){
_start:
{
lean_object* v___x_346_; uint8_t v___x_347_; 
v___x_346_ = lean_unsigned_to_nat(0u);
v___x_347_ = lean_nat_dec_eq(v_start_345_, v___x_346_);
if (v___x_347_ == 0)
{
lean_object* v_root_348_; lean_object* v_tail_349_; size_t v_shift_350_; lean_object* v_tailOff_351_; uint8_t v___x_352_; 
v_root_348_ = lean_ctor_get(v_t_343_, 0);
v_tail_349_ = lean_ctor_get(v_t_343_, 1);
v_shift_350_ = lean_ctor_get_usize(v_t_343_, 4);
v_tailOff_351_ = lean_ctor_get(v_t_343_, 3);
v___x_352_ = lean_nat_dec_le(v_tailOff_351_, v_start_345_);
if (v___x_352_ == 0)
{
size_t v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_353_ = lean_usize_of_nat(v_start_345_);
lean_inc_ref(v_p_342_);
v___x_354_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0(v_p_342_, v_root_348_, v___x_353_, v_shift_350_, v_init_344_);
v___x_355_ = lean_array_get_size(v_tail_349_);
v___x_356_ = lean_nat_dec_lt(v___x_346_, v___x_355_);
if (v___x_356_ == 0)
{
lean_dec_ref(v_p_342_);
return v___x_354_;
}
else
{
uint8_t v___x_357_; 
v___x_357_ = lean_nat_dec_le(v___x_355_, v___x_355_);
if (v___x_357_ == 0)
{
if (v___x_356_ == 0)
{
lean_dec_ref(v_p_342_);
return v___x_354_;
}
else
{
size_t v___x_358_; size_t v___x_359_; lean_object* v___x_360_; 
v___x_358_ = ((size_t)0ULL);
v___x_359_ = lean_usize_of_nat(v___x_355_);
v___x_360_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_349_, v___x_358_, v___x_359_, v___x_354_);
return v___x_360_;
}
}
else
{
size_t v___x_361_; size_t v___x_362_; lean_object* v___x_363_; 
v___x_361_ = ((size_t)0ULL);
v___x_362_ = lean_usize_of_nat(v___x_355_);
v___x_363_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_349_, v___x_361_, v___x_362_, v___x_354_);
return v___x_363_;
}
}
}
else
{
lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_364_ = lean_nat_sub(v_start_345_, v_tailOff_351_);
v___x_365_ = lean_array_get_size(v_tail_349_);
v___x_366_ = lean_nat_dec_lt(v___x_364_, v___x_365_);
if (v___x_366_ == 0)
{
lean_dec(v___x_364_);
lean_dec_ref(v_p_342_);
return v_init_344_;
}
else
{
uint8_t v___x_367_; 
v___x_367_ = lean_nat_dec_le(v___x_365_, v___x_365_);
if (v___x_367_ == 0)
{
if (v___x_366_ == 0)
{
lean_dec(v___x_364_);
lean_dec_ref(v_p_342_);
return v_init_344_;
}
else
{
size_t v___x_368_; size_t v___x_369_; lean_object* v___x_370_; 
v___x_368_ = lean_usize_of_nat(v___x_364_);
lean_dec(v___x_364_);
v___x_369_ = lean_usize_of_nat(v___x_365_);
v___x_370_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_349_, v___x_368_, v___x_369_, v_init_344_);
return v___x_370_;
}
}
else
{
size_t v___x_371_; size_t v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_usize_of_nat(v___x_364_);
lean_dec(v___x_364_);
v___x_372_ = lean_usize_of_nat(v___x_365_);
v___x_373_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_349_, v___x_371_, v___x_372_, v_init_344_);
return v___x_373_;
}
}
}
}
else
{
lean_object* v_root_374_; lean_object* v_tail_375_; lean_object* v___x_376_; lean_object* v___x_377_; uint8_t v___x_378_; 
v_root_374_ = lean_ctor_get(v_t_343_, 0);
v_tail_375_ = lean_ctor_get(v_t_343_, 1);
lean_inc_ref(v_p_342_);
v___x_376_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2(v_p_342_, v_root_374_, v_init_344_);
v___x_377_ = lean_array_get_size(v_tail_375_);
v___x_378_ = lean_nat_dec_lt(v___x_346_, v___x_377_);
if (v___x_378_ == 0)
{
lean_dec_ref(v_p_342_);
return v___x_376_;
}
else
{
uint8_t v___x_379_; 
v___x_379_ = lean_nat_dec_le(v___x_377_, v___x_377_);
if (v___x_379_ == 0)
{
if (v___x_378_ == 0)
{
lean_dec_ref(v_p_342_);
return v___x_376_;
}
else
{
size_t v___x_380_; size_t v___x_381_; lean_object* v___x_382_; 
v___x_380_ = ((size_t)0ULL);
v___x_381_ = lean_usize_of_nat(v___x_377_);
v___x_382_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_375_, v___x_380_, v___x_381_, v___x_376_);
return v___x_382_;
}
}
else
{
size_t v___x_383_; size_t v___x_384_; lean_object* v___x_385_; 
v___x_383_ = ((size_t)0ULL);
v___x_384_ = lean_usize_of_nat(v___x_377_);
v___x_385_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_342_, v_tail_375_, v___x_383_, v___x_384_, v___x_376_);
return v___x_385_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findAllInfos_loop(lean_object* v_p_386_, lean_object* v_t_387_, lean_object* v_acc_388_){
_start:
{
switch(lean_obj_tag(v_t_387_))
{
case 0:
{
lean_object* v_t_389_; 
v_t_389_ = lean_ctor_get(v_t_387_, 1);
lean_inc_ref(v_t_389_);
lean_dec_ref_known(v_t_387_, 2);
v_t_387_ = v_t_389_;
goto _start;
}
case 1:
{
lean_object* v_i_391_; lean_object* v_children_392_; lean_object* v___y_394_; lean_object* v___x_397_; uint8_t v___x_398_; 
v_i_391_ = lean_ctor_get(v_t_387_, 0);
lean_inc_ref_n(v_i_391_, 2);
v_children_392_ = lean_ctor_get(v_t_387_, 1);
lean_inc_ref(v_children_392_);
lean_dec_ref_known(v_t_387_, 2);
lean_inc_ref(v_p_386_);
v___x_397_ = lean_apply_1(v_p_386_, v_i_391_);
v___x_398_ = lean_unbox(v___x_397_);
if (v___x_398_ == 0)
{
lean_dec_ref(v_i_391_);
v___y_394_ = v_acc_388_;
goto v___jp_393_;
}
else
{
lean_object* v___x_399_; 
v___x_399_ = lean_array_push(v_acc_388_, v_i_391_);
v___y_394_ = v___x_399_;
goto v___jp_393_;
}
v___jp_393_:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = lean_unsigned_to_nat(0u);
v___x_396_ = lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0(v_p_386_, v_children_392_, v___y_394_, v___x_395_);
lean_dec_ref(v_children_392_);
return v___x_396_;
}
}
default: 
{
lean_dec_ref_known(v_t_387_, 1);
lean_dec_ref(v_p_386_);
return v_acc_388_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(lean_object* v_p_400_, lean_object* v_as_401_, size_t v_i_402_, size_t v_stop_403_, lean_object* v_b_404_){
_start:
{
uint8_t v___x_405_; 
v___x_405_ = lean_usize_dec_eq(v_i_402_, v_stop_403_);
if (v___x_405_ == 0)
{
lean_object* v___x_406_; lean_object* v___x_407_; size_t v___x_408_; size_t v___x_409_; 
v___x_406_ = lean_array_uget_borrowed(v_as_401_, v_i_402_);
lean_inc(v___x_406_);
lean_inc_ref(v_p_400_);
v___x_407_ = lp_batteries_Batteries_CodeAction_findAllInfos_loop(v_p_400_, v___x_406_, v_b_404_);
v___x_408_ = ((size_t)1ULL);
v___x_409_ = lean_usize_add(v_i_402_, v___x_408_);
v_i_402_ = v___x_409_;
v_b_404_ = v___x_407_;
goto _start;
}
else
{
lean_dec_ref(v_p_400_);
return v_b_404_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1___boxed(lean_object* v_p_411_, lean_object* v_as_412_, lean_object* v_i_413_, lean_object* v_stop_414_, lean_object* v_b_415_){
_start:
{
size_t v_i_boxed_416_; size_t v_stop_boxed_417_; lean_object* v_res_418_; 
v_i_boxed_416_ = lean_unbox_usize(v_i_413_);
lean_dec(v_i_413_);
v_stop_boxed_417_ = lean_unbox_usize(v_stop_414_);
lean_dec(v_stop_414_);
v_res_418_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__1(v_p_411_, v_as_412_, v_i_boxed_416_, v_stop_boxed_417_, v_b_415_);
lean_dec_ref(v_as_412_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1___boxed(lean_object* v_p_419_, lean_object* v_as_420_, lean_object* v_i_421_, lean_object* v_stop_422_, lean_object* v_b_423_){
_start:
{
size_t v_i_boxed_424_; size_t v_stop_boxed_425_; lean_object* v_res_426_; 
v_i_boxed_424_ = lean_unbox_usize(v_i_421_);
lean_dec(v_i_421_);
v_stop_boxed_425_ = lean_unbox_usize(v_stop_422_);
lean_dec(v_stop_422_);
v_res_426_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0_spec__1(v_p_419_, v_as_420_, v_i_boxed_424_, v_stop_boxed_425_, v_b_423_);
lean_dec_ref(v_as_420_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2___boxed(lean_object* v_p_427_, lean_object* v_x_428_, lean_object* v_x_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__2(v_p_427_, v_x_428_, v_x_429_);
lean_dec_ref(v_x_428_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0___boxed(lean_object* v_p_431_, lean_object* v_x_432_, lean_object* v_x_433_, lean_object* v_x_434_, lean_object* v_x_435_){
_start:
{
size_t v_x_1496__boxed_436_; size_t v_x_1497__boxed_437_; lean_object* v_res_438_; 
v_x_1496__boxed_436_ = lean_unbox_usize(v_x_433_);
lean_dec(v_x_433_);
v_x_1497__boxed_437_ = lean_unbox_usize(v_x_434_);
lean_dec(v_x_434_);
v_res_438_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0_spec__0(v_p_431_, v_x_432_, v_x_1496__boxed_436_, v_x_1497__boxed_437_, v_x_435_);
lean_dec_ref(v_x_432_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0___boxed(lean_object* v_p_439_, lean_object* v_t_440_, lean_object* v_init_441_, lean_object* v_start_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_batteries_Lean_PersistentArray_foldlM___at___00Batteries_CodeAction_findAllInfos_loop_spec__0(v_p_439_, v_t_440_, v_init_441_, v_start_442_);
lean_dec(v_start_442_);
lean_dec_ref(v_t_440_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findAllInfos(lean_object* v_p_446_, lean_object* v_t_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_findAllInfos___closed__0));
v___x_449_ = lp_batteries_Batteries_CodeAction_findAllInfos_loop(v_p_446_, v_t_447_, v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT uint8_t lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0(lean_object* v_msg_457_){
_start:
{
lean_object* v___f_458_; lean_object* v___f_459_; lean_object* v___f_460_; lean_object* v___f_461_; lean_object* v___f_462_; lean_object* v___f_463_; lean_object* v___f_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; uint8_t v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; uint8_t v___x_472_; 
v___f_458_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__0));
v___f_459_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__1));
v___f_460_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__2));
v___f_461_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__3));
v___f_462_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__4));
v___f_463_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__5));
v___f_464_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___closed__6));
v___x_465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_465_, 0, v___f_458_);
lean_ctor_set(v___x_465_, 1, v___f_459_);
v___x_466_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v___f_460_);
lean_ctor_set(v___x_466_, 2, v___f_461_);
lean_ctor_set(v___x_466_, 3, v___f_462_);
lean_ctor_set(v___x_466_, 4, v___f_463_);
v___x_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
lean_ctor_set(v___x_467_, 1, v___f_464_);
v___x_468_ = 0;
v___x_469_ = lean_box(v___x_468_);
v___x_470_ = l_instInhabitedOfMonad___redArg(v___x_467_, v___x_469_);
v___x_471_ = lean_panic_fn_borrowed(v___x_470_, v_msg_457_);
lean_dec(v___x_470_);
v___x_472_ = lean_unbox(v___x_471_);
lean_dec(v___x_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0___boxed(lean_object* v_msg_473_){
_start:
{
uint8_t v_res_474_; lean_object* v_r_475_; 
v_res_474_ = lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0(v_msg_473_);
v_r_475_ = lean_box(v_res_474_);
return v_r_475_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_479_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2));
v___x_480_ = lean_unsigned_to_nat(54u);
v___x_481_ = lean_unsigned_to_nat(60u);
v___x_482_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__1));
v___x_483_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0));
v___x_484_ = l_mkPanicMessageWithDecl(v___x_483_, v___x_482_, v___x_481_, v___x_480_, v___x_479_);
return v___x_484_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_486_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__4));
v___x_487_ = lean_unsigned_to_nat(66u);
v___x_488_ = lean_unsigned_to_nat(63u);
v___x_489_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__1));
v___x_490_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0));
v___x_491_ = l_mkPanicMessageWithDecl(v___x_490_, v___x_489_, v___x_488_, v___x_487_, v___x_486_);
return v___x_491_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_hasImplicitNonparArg(lean_object* v_ctor_494_, lean_object* v_env_495_){
_start:
{
uint8_t v___x_502_; lean_object* v___x_503_; 
v___x_502_ = 0;
lean_inc_ref(v_env_495_);
v___x_503_ = l_Lean_Environment_find_x3f(v_env_495_, v_ctor_494_, v___x_502_);
if (lean_obj_tag(v___x_503_) == 1)
{
lean_object* v_val_504_; 
v_val_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc(v_val_504_);
lean_dec_ref_known(v___x_503_, 1);
if (lean_obj_tag(v_val_504_) == 6)
{
lean_object* v_val_505_; lean_object* v_toConstantVal_506_; lean_object* v_induct_507_; lean_object* v_type_508_; lean_object* v___x_509_; 
v_val_505_ = lean_ctor_get(v_val_504_, 0);
lean_inc_ref(v_val_505_);
lean_dec_ref_known(v_val_504_, 1);
v_toConstantVal_506_ = lean_ctor_get(v_val_505_, 0);
lean_inc_ref(v_toConstantVal_506_);
v_induct_507_ = lean_ctor_get(v_val_505_, 1);
lean_inc(v_induct_507_);
lean_dec_ref(v_val_505_);
v_type_508_ = lean_ctor_get(v_toConstantVal_506_, 2);
lean_inc_ref(v_type_508_);
lean_dec_ref(v_toConstantVal_506_);
v___x_509_ = l_Lean_Environment_find_x3f(v_env_495_, v_induct_507_, v___x_502_);
if (lean_obj_tag(v___x_509_) == 1)
{
lean_object* v_val_510_; 
v_val_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc(v_val_510_);
lean_dec_ref_known(v___x_509_, 1);
if (lean_obj_tag(v_val_510_) == 5)
{
lean_object* v_val_511_; lean_object* v_numParams_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v_explicitArgs_515_; lean_object* v_allArgs_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; uint8_t v___x_521_; 
v_val_511_ = lean_ctor_get(v_val_510_, 0);
lean_inc_ref(v_val_511_);
lean_dec_ref_known(v_val_510_, 1);
v_numParams_512_ = lean_ctor_get(v_val_511_, 1);
lean_inc(v_numParams_512_);
lean_dec_ref(v_val_511_);
v___x_513_ = lean_unsigned_to_nat(0u);
v___x_514_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__6));
lean_inc_ref(v_type_508_);
v_explicitArgs_515_ = lp_batteries_Batteries_CodeAction_getExplicitArgs(v_type_508_, v___x_514_);
v_allArgs_516_ = lp_batteries_Batteries_CodeAction_getAllArgs(v_type_508_, v___x_514_);
v___x_517_ = lean_array_get_size(v_allArgs_516_);
lean_dec_ref(v_allArgs_516_);
v___x_518_ = lean_array_get_size(v_explicitArgs_515_);
lean_dec_ref(v_explicitArgs_515_);
v___x_519_ = lean_nat_add(v___x_518_, v_numParams_512_);
lean_dec(v_numParams_512_);
v___x_520_ = lean_nat_sub(v___x_517_, v___x_519_);
lean_dec(v___x_519_);
v___x_521_ = lean_nat_dec_lt(v___x_513_, v___x_520_);
lean_dec(v___x_520_);
return v___x_521_;
}
else
{
lean_dec(v_val_510_);
lean_dec_ref(v_type_508_);
goto v___jp_499_;
}
}
else
{
lean_dec(v___x_509_);
lean_dec_ref(v_type_508_);
goto v___jp_499_;
}
}
else
{
lean_dec(v_val_504_);
lean_dec_ref(v_env_495_);
goto v___jp_496_;
}
}
else
{
lean_dec(v___x_503_);
lean_dec_ref(v_env_495_);
goto v___jp_496_;
}
v___jp_496_:
{
lean_object* v___x_497_; uint8_t v___x_498_; 
v___x_497_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3, &lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3_once, _init_lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__3);
v___x_498_ = lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0(v___x_497_);
return v___x_498_;
}
v___jp_499_:
{
lean_object* v___x_500_; uint8_t v___x_501_; 
v___x_500_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5, &lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5_once, _init_lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__5);
v___x_501_ = lp_batteries_panic___at___00Batteries_CodeAction_hasImplicitNonparArg_spec__0(v___x_500_);
return v___x_501_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___boxed(lean_object* v_ctor_522_, lean_object* v_env_523_){
_start:
{
uint8_t v_res_524_; lean_object* v_r_525_; 
v_res_524_ = lp_batteries_Batteries_CodeAction_hasImplicitNonparArg(v_ctor_522_, v_env_523_);
v_r_525_ = lean_box(v_res_524_);
return v_r_525_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_patternFromConstructor_spec__0(lean_object* v_msg_526_){
_start:
{
lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_527_ = lean_box(0);
v___x_528_ = lean_panic_fn_borrowed(v___x_527_, v_msg_526_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg(lean_object* v_allCtorArgs_535_, lean_object* v_explicitCtorArgs_536_, uint8_t v_ctor__hasImplicitNonparArg_537_, lean_object* v_suffix_538_, lean_object* v_range_539_, lean_object* v_b_540_, lean_object* v_i_541_){
_start:
{
lean_object* v_stop_542_; lean_object* v_step_543_; lean_object* v___y_545_; uint8_t v___x_549_; 
v_stop_542_ = lean_ctor_get(v_range_539_, 1);
v_step_543_ = lean_ctor_get(v_range_539_, 2);
v___x_549_ = lean_nat_dec_lt(v_i_541_, v_stop_542_);
if (v___x_549_ == 0)
{
lean_object* v___x_550_; 
lean_dec(v_i_541_);
lean_dec_ref(v_explicitCtorArgs_536_);
v___x_550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_550_, 0, v_b_540_);
return v___x_550_;
}
else
{
lean_object* v___x_551_; lean_object* v___x_552_; uint8_t v___y_554_; uint8_t v___x_571_; 
v___x_551_ = lean_box(0);
v___x_552_ = lean_array_get_borrowed(v___x_551_, v_allCtorArgs_535_, v_i_541_);
v___x_571_ = l_Lean_Name_hasNum(v___x_552_);
if (v___x_571_ == 0)
{
uint8_t v___x_572_; 
v___x_572_ = l_Lean_Name_isInternal(v___x_552_);
v___y_554_ = v___x_572_;
goto v___jp_553_;
}
else
{
v___y_554_ = v___x_571_;
goto v___jp_553_;
}
v___jp_553_:
{
if (v___y_554_ == 0)
{
lean_object* v___x_555_; uint8_t v___x_556_; 
v___x_555_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__0));
lean_inc(v___x_552_);
lean_inc_ref(v_explicitCtorArgs_536_);
v___x_556_ = l_Array_contains___redArg(v___x_555_, v_explicitCtorArgs_536_, v___x_552_);
if (v___x_556_ == 0)
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_557_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__1));
lean_inc(v___x_552_);
v___x_558_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_552_, v_ctor__hasImplicitNonparArg_537_);
v___x_559_ = lean_string_append(v___x_557_, v___x_558_);
v___x_560_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__2));
v___x_561_ = lean_string_append(v___x_559_, v___x_560_);
v___x_562_ = lean_string_append(v___x_561_, v___x_558_);
lean_dec_ref(v___x_558_);
v___x_563_ = lean_string_append(v___x_562_, v_suffix_538_);
v___x_564_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__3));
v___x_565_ = lean_string_append(v___x_563_, v___x_564_);
v___y_545_ = v___x_565_;
goto v___jp_544_;
}
else
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_566_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__4));
lean_inc(v___x_552_);
v___x_567_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_552_, v_ctor__hasImplicitNonparArg_537_);
v___x_568_ = lean_string_append(v___x_566_, v___x_567_);
lean_dec_ref(v___x_567_);
v___x_569_ = lean_string_append(v___x_568_, v_suffix_538_);
v___y_545_ = v___x_569_;
goto v___jp_544_;
}
}
else
{
lean_object* v___x_570_; 
v___x_570_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__5));
v___y_545_ = v___x_570_;
goto v___jp_544_;
}
}
}
v___jp_544_:
{
lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_546_ = lean_string_append(v_b_540_, v___y_545_);
lean_dec_ref(v___y_545_);
v___x_547_ = lean_nat_add(v_i_541_, v_step_543_);
lean_dec(v_i_541_);
v_b_540_ = v___x_546_;
v_i_541_ = v___x_547_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___boxed(lean_object* v_allCtorArgs_573_, lean_object* v_explicitCtorArgs_574_, lean_object* v_ctor__hasImplicitNonparArg_575_, lean_object* v_suffix_576_, lean_object* v_range_577_, lean_object* v_b_578_, lean_object* v_i_579_){
_start:
{
uint8_t v_ctor__hasImplicitNonparArg_boxed_580_; lean_object* v_res_581_; 
v_ctor__hasImplicitNonparArg_boxed_580_ = lean_unbox(v_ctor__hasImplicitNonparArg_575_);
v_res_581_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg(v_allCtorArgs_573_, v_explicitCtorArgs_574_, v_ctor__hasImplicitNonparArg_boxed_580_, v_suffix_576_, v_range_577_, v_b_578_, v_i_579_);
lean_dec_ref(v_range_577_);
lean_dec_ref(v_suffix_576_);
lean_dec_ref(v_allCtorArgs_573_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1(uint8_t v___y_582_, lean_object* v_suffix_583_, lean_object* v_as_584_, size_t v_sz_585_, size_t v_i_586_, lean_object* v_b_587_){
_start:
{
lean_object* v___y_589_; uint8_t v___x_594_; 
v___x_594_ = lean_usize_dec_lt(v_i_586_, v_sz_585_);
if (v___x_594_ == 0)
{
lean_object* v___x_595_; 
v___x_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_595_, 0, v_b_587_);
return v___x_595_;
}
else
{
lean_object* v_a_596_; uint8_t v___y_598_; uint8_t v___x_604_; 
v_a_596_ = lean_array_uget_borrowed(v_as_584_, v_i_586_);
v___x_604_ = l_Lean_Name_hasNum(v_a_596_);
if (v___x_604_ == 0)
{
uint8_t v___x_605_; 
v___x_605_ = l_Lean_Name_isInternal(v_a_596_);
v___y_598_ = v___x_605_;
goto v___jp_597_;
}
else
{
v___y_598_ = v___x_604_;
goto v___jp_597_;
}
v___jp_597_:
{
if (v___y_598_ == 0)
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v___x_599_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__4));
lean_inc(v_a_596_);
v___x_600_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_a_596_, v___y_582_);
v___x_601_ = lean_string_append(v___x_599_, v___x_600_);
lean_dec_ref(v___x_600_);
v___x_602_ = lean_string_append(v___x_601_, v_suffix_583_);
v___y_589_ = v___x_602_;
goto v___jp_588_;
}
else
{
lean_object* v___x_603_; 
v___x_603_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg___closed__5));
v___y_589_ = v___x_603_;
goto v___jp_588_;
}
}
}
v___jp_588_:
{
lean_object* v___x_590_; size_t v___x_591_; size_t v___x_592_; 
v___x_590_ = lean_string_append(v_b_587_, v___y_589_);
lean_dec_ref(v___y_589_);
v___x_591_ = ((size_t)1ULL);
v___x_592_ = lean_usize_add(v_i_586_, v___x_591_);
v_i_586_ = v___x_592_;
v_b_587_ = v___x_590_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1___boxed(lean_object* v___y_606_, lean_object* v_suffix_607_, lean_object* v_as_608_, lean_object* v_sz_609_, lean_object* v_i_610_, lean_object* v_b_611_){
_start:
{
uint8_t v___y_2572__boxed_612_; size_t v_sz_boxed_613_; size_t v_i_boxed_614_; lean_object* v_res_615_; 
v___y_2572__boxed_612_ = lean_unbox(v___y_606_);
v_sz_boxed_613_ = lean_unbox_usize(v_sz_609_);
lean_dec(v_sz_609_);
v_i_boxed_614_ = lean_unbox_usize(v_i_610_);
lean_dec(v_i_610_);
v_res_615_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1(v___y_2572__boxed_612_, v_suffix_607_, v_as_608_, v_sz_boxed_613_, v_i_boxed_614_, v_b_611_);
lean_dec_ref(v_as_608_);
lean_dec_ref(v_suffix_607_);
return v_res_615_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1(void){
_start:
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_617_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2));
v___x_618_ = lean_unsigned_to_nat(52u);
v___x_619_ = lean_unsigned_to_nat(72u);
v___x_620_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__0));
v___x_621_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0));
v___x_622_ = l_mkPanicMessageWithDecl(v___x_621_, v___x_620_, v___x_619_, v___x_618_, v___x_617_);
return v___x_622_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2(void){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_623_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__4));
v___x_624_ = lean_unsigned_to_nat(64u);
v___x_625_ = lean_unsigned_to_nat(73u);
v___x_626_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__0));
v___x_627_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0));
v___x_628_ = l_mkPanicMessageWithDecl(v___x_627_, v___x_626_, v___x_625_, v___x_624_, v___x_623_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor(lean_object* v_ctor_657_, lean_object* v_env_658_, lean_object* v_suffix_659_, uint8_t v_explicitArgsOnly_660_, uint8_t v_ctor__hasImplicitNonparArg_661_){
_start:
{
uint8_t v___x_668_; lean_object* v___x_669_; 
v___x_668_ = 0;
lean_inc(v_ctor_657_);
lean_inc_ref(v_env_658_);
v___x_669_ = l_Lean_Environment_find_x3f(v_env_658_, v_ctor_657_, v___x_668_);
if (lean_obj_tag(v___x_669_) == 1)
{
lean_object* v_val_670_; 
v_val_670_ = lean_ctor_get(v___x_669_, 0);
lean_inc(v_val_670_);
lean_dec_ref_known(v___x_669_, 1);
if (lean_obj_tag(v_val_670_) == 6)
{
lean_object* v_val_671_; lean_object* v_toConstantVal_672_; lean_object* v_induct_673_; lean_object* v___x_674_; 
v_val_671_ = lean_ctor_get(v_val_670_, 0);
lean_inc_ref(v_val_671_);
lean_dec_ref_known(v_val_670_, 1);
v_toConstantVal_672_ = lean_ctor_get(v_val_671_, 0);
lean_inc_ref(v_toConstantVal_672_);
v_induct_673_ = lean_ctor_get(v_val_671_, 1);
lean_inc(v_induct_673_);
lean_dec_ref(v_val_671_);
v___x_674_ = l_Lean_Environment_find_x3f(v_env_658_, v_induct_673_, v___x_668_);
if (lean_obj_tag(v___x_674_) == 1)
{
lean_object* v_val_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_774_; 
v_val_675_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_774_ == 0)
{
v___x_677_ = v___x_674_;
v_isShared_678_ = v_isSharedCheck_774_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_val_675_);
lean_dec(v___x_674_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_774_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
if (lean_obj_tag(v_val_675_) == 5)
{
lean_object* v_val_679_; lean_object* v_numParams_680_; lean_object* v_type_681_; lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_771_; 
v_val_679_ = lean_ctor_get(v_val_675_, 0);
lean_inc_ref(v_val_679_);
lean_dec_ref_known(v_val_675_, 1);
v_numParams_680_ = lean_ctor_get(v_val_679_, 1);
lean_inc(v_numParams_680_);
lean_dec_ref(v_val_679_);
v_type_681_ = lean_ctor_get(v_toConstantVal_672_, 2);
v_isSharedCheck_771_ = !lean_is_exclusive(v_toConstantVal_672_);
if (v_isSharedCheck_771_ == 0)
{
lean_object* v_unused_772_; lean_object* v_unused_773_; 
v_unused_772_ = lean_ctor_get(v_toConstantVal_672_, 1);
lean_dec(v_unused_772_);
v_unused_773_ = lean_ctor_get(v_toConstantVal_672_, 0);
lean_dec(v_unused_773_);
v___x_683_ = v_toConstantVal_672_;
v_isShared_684_ = v_isSharedCheck_771_;
goto v_resetjp_682_;
}
else
{
lean_inc(v_type_681_);
lean_dec(v_toConstantVal_672_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_771_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
lean_object* v___x_685_; lean_object* v___x_686_; uint8_t v___x_687_; lean_object* v_ctor__short_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v_explicitCtorArgs_691_; uint8_t v___y_693_; lean_object* v_allCtorArgs_699_; 
v___x_685_ = lean_box(0);
lean_inc(v_ctor_657_);
v___x_686_ = l_Lean_Name_updatePrefix(v_ctor_657_, v___x_685_);
v___x_687_ = 1;
v_ctor__short_688_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_686_, v___x_687_);
v___x_689_ = lean_unsigned_to_nat(0u);
v___x_690_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__6));
lean_inc_ref(v_type_681_);
v_explicitCtorArgs_691_ = lp_batteries_Batteries_CodeAction_getExplicitArgs(v_type_681_, v___x_690_);
v_allCtorArgs_699_ = lp_batteries_Batteries_CodeAction_getAllArgs(v_type_681_, v___x_690_);
if (lean_obj_tag(v_ctor_657_) == 1)
{
lean_object* v_pre_709_; 
v_pre_709_ = lean_ctor_get(v_ctor_657_, 0);
lean_inc(v_pre_709_);
if (lean_obj_tag(v_pre_709_) == 1)
{
lean_object* v_pre_710_; 
v_pre_710_ = lean_ctor_get(v_pre_709_, 0);
if (lean_obj_tag(v_pre_710_) == 0)
{
lean_object* v_str_711_; lean_object* v_str_712_; lean_object* v___x_713_; uint8_t v___x_714_; 
v_str_711_ = lean_ctor_get(v_ctor_657_, 1);
lean_inc_ref(v_str_711_);
lean_dec_ref_known(v_ctor_657_, 2);
v_str_712_ = lean_ctor_get(v_pre_709_, 1);
lean_inc_ref(v_str_712_);
lean_dec_ref_known(v_pre_709_, 2);
v___x_713_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__4));
v___x_714_ = lean_string_dec_eq(v_str_712_, v___x_713_);
if (v___x_714_ == 0)
{
lean_object* v___x_715_; uint8_t v___x_716_; 
v___x_715_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__5));
v___x_716_ = lean_string_dec_eq(v_str_712_, v___x_715_);
if (v___x_716_ == 0)
{
lean_object* v___x_717_; uint8_t v___x_718_; 
v___x_717_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__6));
v___x_718_ = lean_string_dec_eq(v_str_712_, v___x_717_);
if (v___x_718_ == 0)
{
lean_object* v___x_719_; uint8_t v___x_720_; 
lean_del_object(v___x_677_);
v___x_719_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__7));
v___x_720_ = lean_string_dec_eq(v_str_712_, v___x_719_);
lean_dec_ref(v_str_712_);
if (v___x_720_ == 0)
{
lean_dec_ref(v_str_711_);
goto v___jp_700_;
}
else
{
lean_object* v___x_721_; uint8_t v___x_722_; 
v___x_721_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__8));
v___x_722_ = lean_string_dec_eq(v_str_711_, v___x_721_);
if (v___x_722_ == 0)
{
lean_object* v___x_723_; uint8_t v___x_724_; 
v___x_723_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__9));
v___x_724_ = lean_string_dec_eq(v_str_711_, v___x_723_);
lean_dec_ref(v_str_711_);
if (v___x_724_ == 0)
{
goto v___jp_700_;
}
else
{
lean_object* v___x_725_; 
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_explicitCtorArgs_691_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_725_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__10));
return v___x_725_;
}
}
else
{
lean_object* v___x_726_; 
lean_dec_ref(v_str_711_);
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_explicitCtorArgs_691_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_726_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__11));
return v___x_726_;
}
}
}
else
{
lean_object* v___x_727_; uint8_t v___x_728_; 
lean_dec_ref(v_str_712_);
v___x_727_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__12));
v___x_728_ = lean_string_dec_eq(v_str_711_, v___x_727_);
if (v___x_728_ == 0)
{
lean_object* v___x_729_; uint8_t v___x_730_; 
lean_del_object(v___x_677_);
v___x_729_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__13));
v___x_730_ = lean_string_dec_eq(v_str_711_, v___x_729_);
lean_dec_ref(v_str_711_);
if (v___x_730_ == 0)
{
goto v___jp_700_;
}
else
{
lean_object* v___x_731_; 
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_explicitCtorArgs_691_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_731_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__14));
return v___x_731_;
}
}
else
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_738_; 
lean_dec_ref(v_str_711_);
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_732_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__15));
v___x_733_ = lean_array_get(v___x_685_, v_explicitCtorArgs_691_, v___x_689_);
lean_dec_ref(v_explicitCtorArgs_691_);
v___x_734_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_733_, v___x_687_);
v___x_735_ = lean_string_append(v___x_732_, v___x_734_);
lean_dec_ref(v___x_734_);
v___x_736_ = lean_string_append(v___x_735_, v_suffix_659_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 0, v___x_736_);
v___x_738_ = v___x_677_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_736_);
v___x_738_ = v_reuseFailAlloc_739_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
return v___x_738_;
}
}
}
}
else
{
lean_object* v___x_740_; uint8_t v___x_741_; 
lean_dec_ref(v_str_712_);
v___x_740_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__16));
v___x_741_ = lean_string_dec_eq(v_str_711_, v___x_740_);
if (v___x_741_ == 0)
{
lean_object* v___x_742_; uint8_t v___x_743_; 
v___x_742_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__17));
v___x_743_ = lean_string_dec_eq(v_str_711_, v___x_742_);
lean_dec_ref(v_str_711_);
if (v___x_743_ == 0)
{
lean_del_object(v___x_677_);
goto v___jp_700_;
}
else
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_755_; 
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_744_ = lean_array_get(v___x_685_, v_explicitCtorArgs_691_, v___x_689_);
v___x_745_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_744_, v___x_687_);
v___x_746_ = lean_string_append(v___x_745_, v_suffix_659_);
v___x_747_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__18));
v___x_748_ = lean_string_append(v___x_746_, v___x_747_);
v___x_749_ = lean_unsigned_to_nat(1u);
v___x_750_ = lean_array_get(v___x_685_, v_explicitCtorArgs_691_, v___x_749_);
lean_dec_ref(v_explicitCtorArgs_691_);
v___x_751_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_750_, v___x_687_);
v___x_752_ = lean_string_append(v___x_748_, v___x_751_);
lean_dec_ref(v___x_751_);
v___x_753_ = lean_string_append(v___x_752_, v_suffix_659_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 0, v___x_753_);
v___x_755_ = v___x_677_;
goto v_reusejp_754_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v___x_753_);
v___x_755_ = v_reuseFailAlloc_756_;
goto v_reusejp_754_;
}
v_reusejp_754_:
{
return v___x_755_;
}
}
}
else
{
lean_object* v___x_757_; 
lean_dec_ref(v_str_711_);
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_explicitCtorArgs_691_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
lean_del_object(v___x_677_);
v___x_757_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__20));
return v___x_757_;
}
}
}
else
{
lean_object* v___x_758_; uint8_t v___x_759_; 
lean_dec_ref(v_str_712_);
v___x_758_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__21));
v___x_759_ = lean_string_dec_eq(v_str_711_, v___x_758_);
if (v___x_759_ == 0)
{
lean_object* v___x_760_; uint8_t v___x_761_; 
v___x_760_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__22));
v___x_761_ = lean_string_dec_eq(v_str_711_, v___x_760_);
lean_dec_ref(v_str_711_);
if (v___x_761_ == 0)
{
lean_del_object(v___x_677_);
goto v___jp_700_;
}
else
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_768_; 
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___x_762_ = lean_array_get(v___x_685_, v_explicitCtorArgs_691_, v___x_689_);
lean_dec_ref(v_explicitCtorArgs_691_);
v___x_763_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_762_, v___x_687_);
v___x_764_ = lean_string_append(v___x_763_, v_suffix_659_);
v___x_765_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__23));
v___x_766_ = lean_string_append(v___x_764_, v___x_765_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 0, v___x_766_);
v___x_768_ = v___x_677_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v___x_766_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
}
else
{
lean_object* v___x_770_; 
lean_dec_ref(v_str_711_);
lean_dec_ref(v_allCtorArgs_699_);
lean_dec_ref(v_explicitCtorArgs_691_);
lean_dec_ref(v_ctor__short_688_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
lean_del_object(v___x_677_);
v___x_770_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__25));
return v___x_770_;
}
}
}
else
{
lean_dec_ref_known(v_pre_709_, 2);
lean_dec_ref_known(v_ctor_657_, 2);
lean_del_object(v___x_677_);
goto v___jp_700_;
}
}
else
{
lean_dec_ref_known(v_ctor_657_, 2);
lean_dec(v_pre_709_);
lean_del_object(v___x_677_);
goto v___jp_700_;
}
}
else
{
lean_del_object(v___x_677_);
lean_dec(v_ctor_657_);
goto v___jp_700_;
}
v___jp_692_:
{
lean_object* v___x_694_; lean_object* v_str_695_; size_t v_sz_696_; size_t v___x_697_; lean_object* v___x_698_; 
v___x_694_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__3));
v_str_695_ = lean_string_append(v___x_694_, v_ctor__short_688_);
lean_dec_ref(v_ctor__short_688_);
v_sz_696_ = lean_array_size(v_explicitCtorArgs_691_);
v___x_697_ = ((size_t)0ULL);
v___x_698_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__1(v___y_693_, v_suffix_659_, v_explicitCtorArgs_691_, v_sz_696_, v___x_697_, v_str_695_);
lean_dec_ref(v_explicitCtorArgs_691_);
return v___x_698_;
}
v___jp_700_:
{
if (v_explicitArgsOnly_660_ == 0)
{
if (v_ctor__hasImplicitNonparArg_661_ == 0)
{
lean_dec_ref(v_allCtorArgs_699_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___y_693_ = v___x_687_;
goto v___jp_692_;
}
else
{
lean_object* v___x_701_; lean_object* v_str_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_706_; 
v___x_701_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__3));
v_str_702_ = lean_string_append(v___x_701_, v_ctor__short_688_);
lean_dec_ref(v_ctor__short_688_);
v___x_703_ = lean_array_get_size(v_allCtorArgs_699_);
v___x_704_ = lean_unsigned_to_nat(1u);
lean_inc(v_numParams_680_);
if (v_isShared_684_ == 0)
{
lean_ctor_set(v___x_683_, 2, v___x_704_);
lean_ctor_set(v___x_683_, 1, v___x_703_);
lean_ctor_set(v___x_683_, 0, v_numParams_680_);
v___x_706_ = v___x_683_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v_numParams_680_);
lean_ctor_set(v_reuseFailAlloc_708_, 1, v___x_703_);
lean_ctor_set(v_reuseFailAlloc_708_, 2, v___x_704_);
v___x_706_ = v_reuseFailAlloc_708_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
lean_object* v___x_707_; 
v___x_707_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg(v_allCtorArgs_699_, v_explicitCtorArgs_691_, v_ctor__hasImplicitNonparArg_661_, v_suffix_659_, v___x_706_, v_str_702_, v_numParams_680_);
lean_dec_ref(v___x_706_);
lean_dec_ref(v_allCtorArgs_699_);
return v___x_707_;
}
}
}
else
{
lean_dec_ref(v_allCtorArgs_699_);
lean_del_object(v___x_683_);
lean_dec(v_numParams_680_);
v___y_693_ = v_explicitArgsOnly_660_;
goto v___jp_692_;
}
}
}
}
else
{
lean_del_object(v___x_677_);
lean_dec(v_val_675_);
lean_dec_ref(v_toConstantVal_672_);
lean_dec(v_ctor_657_);
goto v___jp_665_;
}
}
}
else
{
lean_dec(v___x_674_);
lean_dec_ref(v_toConstantVal_672_);
lean_dec(v_ctor_657_);
goto v___jp_665_;
}
}
else
{
lean_dec(v_val_670_);
lean_dec_ref(v_env_658_);
lean_dec(v_ctor_657_);
goto v___jp_662_;
}
}
else
{
lean_dec(v___x_669_);
lean_dec_ref(v_env_658_);
lean_dec(v_ctor_657_);
goto v___jp_662_;
}
v___jp_662_:
{
lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_663_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1, &lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1_once, _init_lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__1);
v___x_664_ = lp_batteries_panic___at___00Batteries_CodeAction_patternFromConstructor_spec__0(v___x_663_);
return v___x_664_;
}
v___jp_665_:
{
lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_666_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2, &lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2_once, _init_lp_batteries_Batteries_CodeAction_patternFromConstructor___closed__2);
v___x_667_ = lp_batteries_panic___at___00Batteries_CodeAction_patternFromConstructor_spec__0(v___x_666_);
return v___x_667_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_patternFromConstructor___boxed(lean_object* v_ctor_775_, lean_object* v_env_776_, lean_object* v_suffix_777_, lean_object* v_explicitArgsOnly_778_, lean_object* v_ctor__hasImplicitNonparArg_779_){
_start:
{
uint8_t v_explicitArgsOnly_boxed_780_; uint8_t v_ctor__hasImplicitNonparArg_boxed_781_; lean_object* v_res_782_; 
v_explicitArgsOnly_boxed_780_ = lean_unbox(v_explicitArgsOnly_778_);
v_ctor__hasImplicitNonparArg_boxed_781_ = lean_unbox(v_ctor__hasImplicitNonparArg_779_);
v_res_782_ = lp_batteries_Batteries_CodeAction_patternFromConstructor(v_ctor_775_, v_env_776_, v_suffix_777_, v_explicitArgsOnly_boxed_780_, v_ctor__hasImplicitNonparArg_boxed_781_);
lean_dec_ref(v_suffix_777_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2(lean_object* v_allCtorArgs_783_, lean_object* v_explicitCtorArgs_784_, uint8_t v_ctor__hasImplicitNonparArg_785_, lean_object* v_suffix_786_, lean_object* v_range_787_, lean_object* v_b_788_, lean_object* v_i_789_, lean_object* v_hs_790_, lean_object* v_hl_791_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___redArg(v_allCtorArgs_783_, v_explicitCtorArgs_784_, v_ctor__hasImplicitNonparArg_785_, v_suffix_786_, v_range_787_, v_b_788_, v_i_789_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2___boxed(lean_object* v_allCtorArgs_793_, lean_object* v_explicitCtorArgs_794_, lean_object* v_ctor__hasImplicitNonparArg_795_, lean_object* v_suffix_796_, lean_object* v_range_797_, lean_object* v_b_798_, lean_object* v_i_799_, lean_object* v_hs_800_, lean_object* v_hl_801_){
_start:
{
uint8_t v_ctor__hasImplicitNonparArg_boxed_802_; lean_object* v_res_803_; 
v_ctor__hasImplicitNonparArg_boxed_802_ = lean_unbox(v_ctor__hasImplicitNonparArg_795_);
v_res_803_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_patternFromConstructor_spec__2(v_allCtorArgs_793_, v_explicitCtorArgs_794_, v_ctor__hasImplicitNonparArg_boxed_802_, v_suffix_796_, v_range_797_, v_b_798_, v_i_799_, v_hs_800_, v_hl_801_);
lean_dec_ref(v_range_797_);
lean_dec_ref(v_suffix_796_);
lean_dec_ref(v_allCtorArgs_793_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1(lean_object* v___y_804_){
_start:
{
lean_object* v_doc_806_; lean_object* v___x_807_; 
v_doc_806_ = lean_ctor_get(v___y_804_, 1);
lean_inc_ref(v_doc_806_);
v___x_807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_807_, 0, v_doc_806_);
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1___boxed(lean_object* v___y_808_, lean_object* v___y_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1(v___y_808_);
lean_dec_ref(v___y_808_);
return v_res_810_;
}
}
static lean_object* _init_lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0(void){
_start:
{
lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_811_ = l_instInhabitedError;
v___x_812_ = lean_alloc_closure((void*)(l_instInhabitedEIO___aux__1___boxed), 4, 3);
lean_closure_set(v___x_812_, 0, lean_box(0));
lean_closure_set(v___x_812_, 1, lean_box(0));
lean_closure_set(v___x_812_, 2, v___x_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7(lean_object* v_msg_813_){
_start:
{
lean_object* v___x_815_; lean_object* v___x_22734__overap_816_; lean_object* v___x_817_; 
v___x_815_ = lean_obj_once(&lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0, &lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0_once, _init_lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___closed__0);
v___x_22734__overap_816_ = lean_panic_fn_borrowed(v___x_815_, v_msg_813_);
v___x_817_ = lean_apply_1(v___x_22734__overap_816_, lean_box(0));
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7___boxed(lean_object* v_msg_818_, lean_object* v___y_819_){
_start:
{
lean_object* v_res_820_; 
v_res_820_ = lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7(v_msg_818_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__6(lean_object* v_a_821_, lean_object* v_a_822_){
_start:
{
if (lean_obj_tag(v_a_821_) == 0)
{
lean_object* v___x_823_; 
v___x_823_ = l_List_reverse___redArg(v_a_822_);
return v___x_823_;
}
else
{
lean_object* v_head_824_; lean_object* v_tail_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_834_; 
v_head_824_ = lean_ctor_get(v_a_821_, 0);
v_tail_825_ = lean_ctor_get(v_a_821_, 1);
v_isSharedCheck_834_ = !lean_is_exclusive(v_a_821_);
if (v_isSharedCheck_834_ == 0)
{
v___x_827_ = v_a_821_;
v_isShared_828_ = v_isSharedCheck_834_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_tail_825_);
lean_inc(v_head_824_);
lean_dec(v_a_821_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_834_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_829_; lean_object* v___x_831_; 
v___x_829_ = l_List_reverse___redArg(v_head_824_);
if (v_isShared_828_ == 0)
{
lean_ctor_set(v___x_827_, 1, v_a_822_);
lean_ctor_set(v___x_827_, 0, v___x_829_);
v___x_831_ = v___x_827_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v___x_829_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v_a_822_);
v___x_831_ = v_reuseFailAlloc_833_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
v_a_821_ = v_tail_825_;
v_a_822_ = v___x_831_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_matchExpand_spec__5(lean_object* v_x_835_, lean_object* v_x_836_){
_start:
{
lean_object* v_zero_837_; uint8_t v_isZero_838_; 
v_zero_837_ = lean_unsigned_to_nat(0u);
v_isZero_838_ = lean_nat_dec_eq(v_x_835_, v_zero_837_);
if (v_isZero_838_ == 1)
{
lean_dec(v_x_835_);
return v_x_836_;
}
else
{
uint32_t v___x_839_; lean_object* v_one_840_; lean_object* v_n_841_; lean_object* v___x_842_; 
v___x_839_ = 32;
v_one_840_ = lean_unsigned_to_nat(1u);
v_n_841_ = lean_nat_sub(v_x_835_, v_one_840_);
lean_dec(v_x_835_);
v___x_842_ = lean_string_push(v_x_836_, v___x_839_);
v_x_835_ = v_n_841_;
v_x_836_ = v___x_842_;
goto _start;
}
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3(void){
_start:
{
lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_850_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__2));
v___x_851_ = lean_unsigned_to_nat(14u);
v___x_852_ = lean_unsigned_to_nat(264u);
v___x_853_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__2));
v___x_854_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_hasImplicitNonparArg___closed__0));
v___x_855_ = l_mkPanicMessageWithDecl(v___x_854_, v___x_853_, v___x_852_, v___x_851_, v___x_850_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg(lean_object* v_a_858_, lean_object* v_snap_859_, uint8_t v_explicitArgsOnly_860_, lean_object* v___x_861_, lean_object* v___x_862_, lean_object* v_range_863_, lean_object* v_b_864_, lean_object* v_i_865_){
_start:
{
lean_object* v_stop_867_; lean_object* v_step_868_; lean_object* v_a_870_; uint8_t v___x_873_; 
v_stop_867_ = lean_ctor_get(v_range_863_, 1);
v_step_868_ = lean_ctor_get(v_range_863_, 2);
v___x_873_ = lean_nat_dec_lt(v_i_865_, v_stop_867_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; 
lean_dec(v_i_865_);
v___x_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_874_, 0, v_b_864_);
return v___x_874_;
}
else
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v_fst_877_; lean_object* v_snd_878_; lean_object* v___x_879_; lean_object* v___y_881_; lean_object* v___x_901_; lean_object* v___x_902_; uint8_t v___x_903_; 
v___x_875_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__0));
lean_inc(v_i_865_);
v___x_876_ = l_List_get_x21Internal___redArg(v___x_875_, v_a_858_, v_i_865_);
v_fst_877_ = lean_ctor_get(v___x_876_, 0);
lean_inc(v_fst_877_);
v_snd_878_ = lean_ctor_get(v___x_876_, 1);
lean_inc(v_snd_878_);
lean_dec(v___x_876_);
v___x_879_ = lean_unsigned_to_nat(1u);
v___x_901_ = lean_unsigned_to_nat(2u);
v___x_902_ = l_List_lengthTR___redArg(v___x_862_);
v___x_903_ = lean_nat_dec_le(v___x_901_, v___x_902_);
lean_dec(v___x_902_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; 
v___x_904_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__4));
v___y_881_ = v___x_904_;
goto v___jp_880_;
}
else
{
lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_905_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__5));
v___x_906_ = lean_nat_add(v_i_865_, v___x_879_);
v___x_907_ = l_Nat_reprFast(v___x_906_);
v___x_908_ = lean_string_append(v___x_905_, v___x_907_);
lean_dec_ref(v___x_907_);
v___y_881_ = v___x_908_;
goto v___jp_880_;
}
v___jp_880_:
{
lean_object* v___x_882_; uint8_t v___x_883_; lean_object* v___x_884_; 
v___x_882_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_859_);
v___x_883_ = lean_unbox(v_snd_878_);
lean_dec(v_snd_878_);
v___x_884_ = lp_batteries_Batteries_CodeAction_patternFromConstructor(v_fst_877_, v___x_882_, v___y_881_, v_explicitArgsOnly_860_, v___x_883_);
lean_dec_ref(v___y_881_);
if (lean_obj_tag(v___x_884_) == 1)
{
lean_object* v_val_885_; lean_object* v___x_886_; lean_object* v___x_887_; uint8_t v___x_888_; 
v_val_885_ = lean_ctor_get(v___x_884_, 0);
lean_inc(v_val_885_);
lean_dec_ref_known(v___x_884_, 1);
v___x_886_ = lean_string_append(v_b_864_, v_val_885_);
lean_dec(v_val_885_);
v___x_887_ = lean_nat_sub(v___x_861_, v___x_879_);
v___x_888_ = lean_nat_dec_lt(v_i_865_, v___x_887_);
lean_dec(v___x_887_);
if (v___x_888_ == 0)
{
v_a_870_ = v___x_886_;
goto v___jp_869_;
}
else
{
lean_object* v___x_889_; lean_object* v___x_890_; 
v___x_889_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__1));
v___x_890_ = lean_string_append(v___x_886_, v___x_889_);
v_a_870_ = v___x_890_;
goto v___jp_869_;
}
}
else
{
lean_object* v___x_891_; lean_object* v___x_892_; 
lean_dec(v___x_884_);
v___x_891_ = lean_obj_once(&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3, &lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3_once, _init_lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__3);
v___x_892_ = lp_batteries_panic___at___00Batteries_CodeAction_matchExpand_spec__7(v___x_891_);
if (lean_obj_tag(v___x_892_) == 0)
{
lean_dec_ref_known(v___x_892_, 1);
v_a_870_ = v_b_864_;
goto v___jp_869_;
}
else
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_900_; 
lean_dec(v_i_865_);
lean_dec_ref(v_b_864_);
v_a_893_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_900_ == 0)
{
v___x_895_ = v___x_892_;
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_892_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_898_; 
if (v_isShared_896_ == 0)
{
v___x_898_ = v___x_895_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_893_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
}
}
}
v___jp_869_:
{
lean_object* v___x_871_; 
v___x_871_ = lean_nat_add(v_i_865_, v_step_868_);
lean_dec(v_i_865_);
v_b_864_ = v_a_870_;
v_i_865_ = v___x_871_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___boxed(lean_object* v_a_909_, lean_object* v_snap_910_, lean_object* v_explicitArgsOnly_911_, lean_object* v___x_912_, lean_object* v___x_913_, lean_object* v_range_914_, lean_object* v_b_915_, lean_object* v_i_916_, lean_object* v___y_917_){
_start:
{
uint8_t v_explicitArgsOnly_boxed_918_; lean_object* v_res_919_; 
v_explicitArgsOnly_boxed_918_ = lean_unbox(v_explicitArgsOnly_911_);
v_res_919_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg(v_a_909_, v_snap_910_, v_explicitArgsOnly_boxed_918_, v___x_912_, v___x_913_, v_range_914_, v_b_915_, v_i_916_);
lean_dec_ref(v_range_914_);
lean_dec(v___x_913_);
lean_dec(v___x_912_);
lean_dec_ref(v_snap_910_);
lean_dec(v_a_909_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg(lean_object* v___x_922_, lean_object* v_snap_923_, uint8_t v_explicitArgsOnly_924_, lean_object* v___x_925_, lean_object* v_as_x27_926_, lean_object* v_b_927_){
_start:
{
if (lean_obj_tag(v_as_x27_926_) == 0)
{
lean_object* v___x_929_; 
v___x_929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_929_, 0, v_b_927_);
return v___x_929_;
}
else
{
lean_object* v_head_930_; lean_object* v_tail_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; 
v_head_930_ = lean_ctor_get(v_as_x27_926_, 0);
v_tail_931_ = lean_ctor_get(v_as_x27_926_, 1);
v___x_932_ = lean_unsigned_to_nat(1u);
v___x_933_ = lean_unsigned_to_nat(0u);
v___x_934_ = lean_string_append(v_b_927_, v___x_922_);
v___x_935_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__0));
v___x_936_ = lean_string_append(v___x_934_, v___x_935_);
v___x_937_ = l_List_lengthTR___redArg(v_head_930_);
lean_inc(v___x_937_);
v___x_938_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_938_, 0, v___x_933_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
lean_ctor_set(v___x_938_, 2, v___x_932_);
v___x_939_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg(v_head_930_, v_snap_923_, v_explicitArgsOnly_924_, v___x_937_, v___x_925_, v___x_938_, v___x_936_, v___x_933_);
lean_dec_ref_known(v___x_938_, 3);
lean_dec(v___x_937_);
if (lean_obj_tag(v___x_939_) == 0)
{
lean_object* v_a_940_; lean_object* v___x_941_; lean_object* v___x_942_; 
v_a_940_ = lean_ctor_get(v___x_939_, 0);
lean_inc(v_a_940_);
lean_dec_ref_known(v___x_939_, 1);
v___x_941_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___closed__1));
v___x_942_ = lean_string_append(v_a_940_, v___x_941_);
v_as_x27_926_ = v_tail_931_;
v_b_927_ = v___x_942_;
goto _start;
}
else
{
return v___x_939_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg___boxed(lean_object* v___x_944_, lean_object* v_snap_945_, lean_object* v_explicitArgsOnly_946_, lean_object* v___x_947_, lean_object* v_as_x27_948_, lean_object* v_b_949_, lean_object* v___y_950_){
_start:
{
uint8_t v_explicitArgsOnly_boxed_951_; lean_object* v_res_952_; 
v_explicitArgsOnly_boxed_951_ = lean_unbox(v_explicitArgsOnly_946_);
v_res_952_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg(v___x_944_, v_snap_945_, v_explicitArgsOnly_boxed_951_, v___x_947_, v_as_x27_948_, v_b_949_);
lean_dec(v_as_x27_948_);
lean_dec(v___x_947_);
lean_dec_ref(v_snap_945_);
lean_dec_ref(v___x_944_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0(lean_object* v___x_955_, lean_object* v_snd_956_, lean_object* v___x_957_, lean_object* v_snap_958_, uint8_t v_explicitArgsOnly_959_, lean_object* v_a_960_, lean_object* v_stop_961_, lean_object* v_text_962_, lean_object* v___x_963_, lean_object* v_title_964_, lean_object* v___x_965_, lean_object* v___x_966_, lean_object* v___x_967_, lean_object* v___x_968_, lean_object* v___x_969_, lean_object* v___x_970_, uint8_t v___y_971_){
_start:
{
lean_object* v_fst_973_; lean_object* v___x_975_; uint8_t v_isShared_976_; uint8_t v_isSharedCheck_1012_; 
v_fst_973_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_1012_ == 0)
{
lean_object* v_unused_1013_; 
v_unused_1013_ = lean_ctor_get(v___x_955_, 1);
lean_dec(v_unused_1013_);
v___x_975_ = v___x_955_;
v_isShared_976_ = v_isSharedCheck_1012_;
goto v_resetjp_974_;
}
else
{
lean_inc(v_fst_973_);
lean_dec(v___x_955_);
v___x_975_ = lean_box(0);
v_isShared_976_ = v_isSharedCheck_1012_;
goto v_resetjp_974_;
}
v_resetjp_974_:
{
lean_object* v___y_978_; 
if (v___y_971_ == 0)
{
lean_object* v___x_1010_; 
v___x_1010_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__1));
v___y_978_ = v___x_1010_;
goto v___jp_977_;
}
else
{
lean_object* v___x_1011_; 
v___x_1011_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg___closed__4));
v___y_978_ = v___x_1011_;
goto v___jp_977_;
}
v___jp_977_:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_979_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___lam__0___closed__0));
v___x_980_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_matchExpand_spec__5(v_fst_973_, v___x_979_);
lean_inc(v_snd_956_);
v___x_981_ = lp_batteries_List_sectionsTR___redArg(v_snd_956_);
v___x_982_ = lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__6(v___x_981_, v___x_957_);
lean_inc_ref(v___y_978_);
v___x_983_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg(v___x_980_, v_snap_958_, v_explicitArgsOnly_959_, v_snd_956_, v___x_982_, v___y_978_);
lean_dec(v___x_982_);
lean_dec(v_snd_956_);
lean_dec_ref(v___x_980_);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_1001_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1001_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1001_ == 0)
{
v___x_986_ = v___x_983_;
v_isShared_987_ = v_isSharedCheck_1001_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_a_984_);
lean_dec(v___x_983_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_1001_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v___x_988_; lean_object* v___x_990_; 
v___x_988_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_960_);
lean_inc(v_stop_961_);
if (v_isShared_976_ == 0)
{
lean_ctor_set(v___x_975_, 1, v_stop_961_);
lean_ctor_set(v___x_975_, 0, v_stop_961_);
v___x_990_ = v___x_975_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v_stop_961_);
lean_ctor_set(v_reuseFailAlloc_1000_, 1, v_stop_961_);
v___x_990_ = v_reuseFailAlloc_1000_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_998_; 
v___x_991_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_962_, v___x_990_);
v___x_992_ = lean_box(0);
lean_inc_n(v___x_963_, 2);
v___x_993_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_993_, 0, v___x_991_);
lean_ctor_set(v___x_993_, 1, v_a_984_);
lean_ctor_set(v___x_993_, 2, v___x_992_);
lean_ctor_set(v___x_993_, 3, v___x_963_);
v___x_994_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_988_, v___x_993_);
v___x_995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_995_, 0, v___x_994_);
v___x_996_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_996_, 0, v___x_963_);
lean_ctor_set(v___x_996_, 1, v___x_963_);
lean_ctor_set(v___x_996_, 2, v_title_964_);
lean_ctor_set(v___x_996_, 3, v___x_965_);
lean_ctor_set(v___x_996_, 4, v___x_966_);
lean_ctor_set(v___x_996_, 5, v___x_967_);
lean_ctor_set(v___x_996_, 6, v___x_968_);
lean_ctor_set(v___x_996_, 7, v___x_995_);
lean_ctor_set(v___x_996_, 8, v___x_969_);
lean_ctor_set(v___x_996_, 9, v___x_970_);
if (v_isShared_987_ == 0)
{
lean_ctor_set(v___x_986_, 0, v___x_996_);
v___x_998_ = v___x_986_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v___x_996_);
v___x_998_ = v_reuseFailAlloc_999_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
return v___x_998_;
}
}
}
}
else
{
lean_object* v_a_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1009_; 
lean_del_object(v___x_975_);
lean_dec(v___x_970_);
lean_dec(v___x_969_);
lean_dec(v___x_968_);
lean_dec(v___x_967_);
lean_dec(v___x_966_);
lean_dec(v___x_965_);
lean_dec_ref(v_title_964_);
lean_dec(v___x_963_);
lean_dec_ref(v_text_962_);
lean_dec(v_stop_961_);
lean_dec_ref(v_a_960_);
v_a_1002_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1009_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_1004_ = v___x_983_;
v_isShared_1005_ = v_isSharedCheck_1009_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_a_1002_);
lean_dec(v___x_983_);
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
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__0___boxed(lean_object** _args){
lean_object* v___x_1014_ = _args[0];
lean_object* v_snd_1015_ = _args[1];
lean_object* v___x_1016_ = _args[2];
lean_object* v_snap_1017_ = _args[3];
lean_object* v_explicitArgsOnly_1018_ = _args[4];
lean_object* v_a_1019_ = _args[5];
lean_object* v_stop_1020_ = _args[6];
lean_object* v_text_1021_ = _args[7];
lean_object* v___x_1022_ = _args[8];
lean_object* v_title_1023_ = _args[9];
lean_object* v___x_1024_ = _args[10];
lean_object* v___x_1025_ = _args[11];
lean_object* v___x_1026_ = _args[12];
lean_object* v___x_1027_ = _args[13];
lean_object* v___x_1028_ = _args[14];
lean_object* v___x_1029_ = _args[15];
lean_object* v___y_1030_ = _args[16];
lean_object* v___y_1031_ = _args[17];
_start:
{
uint8_t v_explicitArgsOnly_boxed_1032_; uint8_t v___y_24151__boxed_1033_; lean_object* v_res_1034_; 
v_explicitArgsOnly_boxed_1032_ = lean_unbox(v_explicitArgsOnly_1018_);
v___y_24151__boxed_1033_ = lean_unbox(v___y_1030_);
v_res_1034_ = lp_batteries_Batteries_CodeAction_matchExpand___lam__0(v___x_1014_, v_snd_1015_, v___x_1016_, v_snap_1017_, v_explicitArgsOnly_boxed_1032_, v_a_1019_, v_stop_1020_, v_text_1021_, v___x_1022_, v_title_1023_, v___x_1024_, v___x_1025_, v___x_1026_, v___x_1027_, v___x_1028_, v___x_1029_, v___y_24151__boxed_1033_);
lean_dec_ref(v_snap_1017_);
return v_res_1034_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1(lean_object* v_val_1038_, lean_object* v_a_1039_, lean_object* v_snd_1040_, lean_object* v___x_1041_, lean_object* v_snap_1042_, uint8_t v___y_1043_, lean_object* v_title_1044_, uint8_t v_explicitArgsOnly_1045_){
_start:
{
lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v_toEditableDocumentCore_1049_; lean_object* v_meta_1050_; lean_object* v_text_1051_; lean_object* v_start_1052_; lean_object* v_stop_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1066_; 
v___x_1046_ = lean_box(0);
v___x_1047_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___lam__1___closed__1));
lean_inc_ref(v_title_1044_);
v___x_1048_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1046_);
lean_ctor_set(v___x_1048_, 1, v___x_1046_);
lean_ctor_set(v___x_1048_, 2, v_title_1044_);
lean_ctor_set(v___x_1048_, 3, v___x_1047_);
lean_ctor_set(v___x_1048_, 4, v___x_1046_);
lean_ctor_set(v___x_1048_, 5, v___x_1046_);
lean_ctor_set(v___x_1048_, 6, v___x_1046_);
lean_ctor_set(v___x_1048_, 7, v___x_1046_);
lean_ctor_set(v___x_1048_, 8, v___x_1046_);
lean_ctor_set(v___x_1048_, 9, v___x_1046_);
v_toEditableDocumentCore_1049_ = lean_ctor_get(v_a_1039_, 0);
v_meta_1050_ = lean_ctor_get(v_toEditableDocumentCore_1049_, 0);
v_text_1051_ = lean_ctor_get(v_meta_1050_, 3);
lean_inc_ref(v_text_1051_);
v_start_1052_ = lean_ctor_get(v_val_1038_, 0);
v_stop_1053_ = lean_ctor_get(v_val_1038_, 1);
v_isSharedCheck_1066_ = !lean_is_exclusive(v_val_1038_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1055_ = v_val_1038_;
v_isShared_1056_ = v_isSharedCheck_1066_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_stop_1053_);
lean_inc(v_start_1052_);
lean_dec(v_val_1038_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1066_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v_source_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___y_1061_; lean_object* v___x_1062_; lean_object* v___x_1064_; 
v_source_1057_ = lean_ctor_get(v_text_1051_, 0);
lean_inc_ref(v_source_1057_);
v___x_1058_ = lp_batteries_Lean_findIndentAndIsStart(v_source_1057_, v_start_1052_);
lean_dec(v_start_1052_);
v___x_1059_ = lean_box(v_explicitArgsOnly_1045_);
v___x_1060_ = lean_box(v___y_1043_);
v___y_1061_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_matchExpand___lam__0___boxed), 18, 17);
lean_closure_set(v___y_1061_, 0, v___x_1058_);
lean_closure_set(v___y_1061_, 1, v_snd_1040_);
lean_closure_set(v___y_1061_, 2, v___x_1041_);
lean_closure_set(v___y_1061_, 3, v_snap_1042_);
lean_closure_set(v___y_1061_, 4, v___x_1059_);
lean_closure_set(v___y_1061_, 5, v_a_1039_);
lean_closure_set(v___y_1061_, 6, v_stop_1053_);
lean_closure_set(v___y_1061_, 7, v_text_1051_);
lean_closure_set(v___y_1061_, 8, v___x_1046_);
lean_closure_set(v___y_1061_, 9, v_title_1044_);
lean_closure_set(v___y_1061_, 10, v___x_1047_);
lean_closure_set(v___y_1061_, 11, v___x_1046_);
lean_closure_set(v___y_1061_, 12, v___x_1046_);
lean_closure_set(v___y_1061_, 13, v___x_1046_);
lean_closure_set(v___y_1061_, 14, v___x_1046_);
lean_closure_set(v___y_1061_, 15, v___x_1046_);
lean_closure_set(v___y_1061_, 16, v___x_1060_);
v___x_1062_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___y_1061_);
if (v_isShared_1056_ == 0)
{
lean_ctor_set(v___x_1055_, 1, v___x_1062_);
lean_ctor_set(v___x_1055_, 0, v___x_1048_);
v___x_1064_ = v___x_1055_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v___x_1048_);
lean_ctor_set(v_reuseFailAlloc_1065_, 1, v___x_1062_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___lam__1___boxed(lean_object* v_val_1067_, lean_object* v_a_1068_, lean_object* v_snd_1069_, lean_object* v___x_1070_, lean_object* v_snap_1071_, lean_object* v___y_1072_, lean_object* v_title_1073_, lean_object* v_explicitArgsOnly_1074_){
_start:
{
uint8_t v___y_24271__boxed_1075_; uint8_t v_explicitArgsOnly_boxed_1076_; lean_object* v_res_1077_; 
v___y_24271__boxed_1075_ = lean_unbox(v___y_1072_);
v_explicitArgsOnly_boxed_1076_ = lean_unbox(v_explicitArgsOnly_1074_);
v_res_1077_ = lp_batteries_Batteries_CodeAction_matchExpand___lam__1(v_val_1067_, v_a_1068_, v_snd_1069_, v___x_1070_, v_snap_1071_, v___y_24271__boxed_1075_, v_title_1073_, v_explicitArgsOnly_boxed_1076_);
return v_res_1077_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3(lean_object* v_snap_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_){
_start:
{
if (lean_obj_tag(v_a_1079_) == 0)
{
lean_object* v___x_1081_; 
v___x_1081_ = l_List_reverse___redArg(v_a_1080_);
return v___x_1081_;
}
else
{
lean_object* v_head_1082_; lean_object* v_tail_1083_; lean_object* v___x_1085_; uint8_t v_isShared_1086_; uint8_t v_isSharedCheck_1095_; 
v_head_1082_ = lean_ctor_get(v_a_1079_, 0);
v_tail_1083_ = lean_ctor_get(v_a_1079_, 1);
v_isSharedCheck_1095_ = !lean_is_exclusive(v_a_1079_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1085_ = v_a_1079_;
v_isShared_1086_ = v_isSharedCheck_1095_;
goto v_resetjp_1084_;
}
else
{
lean_inc(v_tail_1083_);
lean_inc(v_head_1082_);
lean_dec(v_a_1079_);
v___x_1085_ = lean_box(0);
v_isShared_1086_ = v_isSharedCheck_1095_;
goto v_resetjp_1084_;
}
v_resetjp_1084_:
{
lean_object* v___x_1087_; uint8_t v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1092_; 
v___x_1087_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_1078_);
lean_inc(v_head_1082_);
v___x_1088_ = lp_batteries_Batteries_CodeAction_hasImplicitNonparArg(v_head_1082_, v___x_1087_);
v___x_1089_ = lean_box(v___x_1088_);
v___x_1090_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1090_, 0, v_head_1082_);
lean_ctor_set(v___x_1090_, 1, v___x_1089_);
if (v_isShared_1086_ == 0)
{
lean_ctor_set(v___x_1085_, 1, v_a_1080_);
lean_ctor_set(v___x_1085_, 0, v___x_1090_);
v___x_1092_ = v___x_1085_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v___x_1090_);
lean_ctor_set(v_reuseFailAlloc_1094_, 1, v_a_1080_);
v___x_1092_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
v_a_1079_ = v_tail_1083_;
v_a_1080_ = v___x_1092_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3___boxed(lean_object* v_snap_1096_, lean_object* v_a_1097_, lean_object* v_a_1098_){
_start:
{
lean_object* v_res_1099_; 
v_res_1099_ = lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3(v_snap_1096_, v_a_1097_, v_a_1098_);
lean_dec_ref(v_snap_1096_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg(lean_object* v_node_1104_, lean_object* v_ctx_1105_, lean_object* v_snap_1106_, lean_object* v_as_1107_, size_t v_sz_1108_, size_t v_i_1109_, lean_object* v_b_1110_){
_start:
{
uint8_t v___x_1112_; 
v___x_1112_ = lean_usize_dec_lt(v_i_1109_, v_sz_1108_);
if (v___x_1112_ == 0)
{
lean_object* v___x_1113_; 
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
v___x_1113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1113_, 0, v_b_1110_);
return v___x_1113_;
}
else
{
lean_object* v_snd_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1190_; 
v_snd_1114_ = lean_ctor_get(v_b_1110_, 1);
v_isSharedCheck_1190_ = !lean_is_exclusive(v_b_1110_);
if (v_isSharedCheck_1190_ == 0)
{
lean_object* v_unused_1191_; 
v_unused_1191_ = lean_ctor_get(v_b_1110_, 0);
lean_dec(v_unused_1191_);
v___x_1116_ = v_b_1110_;
v_isShared_1117_ = v_isSharedCheck_1190_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_snd_1114_);
lean_dec(v_b_1110_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1190_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v_a_1124_; lean_object* v___x_1125_; 
v_a_1124_ = lean_array_uget_borrowed(v_as_1107_, v_i_1109_);
lean_inc_ref(v_ctx_1105_);
lean_inc(v_a_1124_);
lean_inc_ref(v_node_1104_);
v___x_1125_ = lp_batteries_Batteries_CodeAction_findTermInfoWithCtx_x3f(v_node_1104_, v_a_1124_, v_ctx_1105_);
if (lean_obj_tag(v___x_1125_) == 1)
{
lean_object* v_val_1126_; lean_object* v_fst_1127_; lean_object* v_snd_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1186_; 
v_val_1126_ = lean_ctor_get(v___x_1125_, 0);
lean_inc(v_val_1126_);
lean_dec_ref_known(v___x_1125_, 1);
v_fst_1127_ = lean_ctor_get(v_val_1126_, 0);
v_snd_1128_ = lean_ctor_get(v_val_1126_, 1);
v_isSharedCheck_1186_ = !lean_is_exclusive(v_val_1126_);
if (v_isSharedCheck_1186_ == 0)
{
v___x_1130_ = v_val_1126_;
v_isShared_1131_ = v_isSharedCheck_1186_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_snd_1128_);
lean_inc(v_fst_1127_);
lean_dec(v_val_1126_);
v___x_1130_ = lean_box(0);
v_isShared_1131_ = v_isSharedCheck_1186_;
goto v_resetjp_1129_;
}
v_resetjp_1129_:
{
lean_object* v_expr_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; 
v_expr_1132_ = lean_ctor_get(v_fst_1127_, 3);
lean_inc_ref(v_expr_1132_);
v___x_1133_ = lean_alloc_closure((void*)(l_Lean_Meta_inferType___boxed), 6, 1);
lean_closure_set(v___x_1133_, 0, v_expr_1132_);
lean_inc(v_snd_1128_);
lean_inc(v_fst_1127_);
v___x_1134_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v_fst_1127_, v_snd_1128_, v___x_1133_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_object* v_a_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; 
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
lean_inc(v_a_1135_);
lean_dec_ref_known(v___x_1134_, 1);
v___x_1136_ = lean_alloc_closure((void*)(l_Lean_Meta_whnf___boxed), 6, 1);
lean_closure_set(v___x_1136_, 0, v_a_1135_);
v___x_1137_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v_fst_1127_, v_snd_1128_, v___x_1136_);
if (lean_obj_tag(v___x_1137_) == 0)
{
lean_object* v_a_1138_; lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1167_; 
v_a_1138_ = lean_ctor_get(v___x_1137_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1137_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1140_ = v___x_1137_;
v_isShared_1141_ = v_isSharedCheck_1167_;
goto v_resetjp_1139_;
}
else
{
lean_inc(v_a_1138_);
lean_dec(v___x_1137_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1167_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___x_1142_; 
v___x_1142_ = l_Lean_Expr_getAppFn(v_a_1138_);
lean_dec(v_a_1138_);
if (lean_obj_tag(v___x_1142_) == 4)
{
lean_object* v_declName_1143_; lean_object* v___x_1144_; uint8_t v___x_1145_; lean_object* v___x_1146_; 
lean_del_object(v___x_1140_);
v_declName_1143_ = lean_ctor_get(v___x_1142_, 0);
lean_inc(v_declName_1143_);
lean_dec_ref_known(v___x_1142_, 2);
v___x_1144_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_1106_);
v___x_1145_ = 0;
v___x_1146_ = l_Lean_Environment_find_x3f(v___x_1144_, v_declName_1143_, v___x_1145_);
if (lean_obj_tag(v___x_1146_) == 1)
{
lean_object* v_val_1147_; 
v_val_1147_ = lean_ctor_get(v___x_1146_, 0);
lean_inc(v_val_1147_);
lean_dec_ref_known(v___x_1146_, 1);
if (lean_obj_tag(v_val_1147_) == 5)
{
lean_object* v_val_1148_; lean_object* v_ctors_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1155_; 
lean_del_object(v___x_1116_);
v_val_1148_ = lean_ctor_get(v_val_1147_, 0);
lean_inc_ref(v_val_1148_);
lean_dec_ref_known(v_val_1147_, 1);
v_ctors_1149_ = lean_ctor_get(v_val_1148_, 4);
lean_inc(v_ctors_1149_);
lean_dec_ref(v_val_1148_);
v___x_1150_ = lean_box(0);
v___x_1151_ = lean_box(0);
v___x_1152_ = lp_batteries_List_mapTR_loop___at___00Batteries_CodeAction_matchExpand_spec__3(v_snap_1106_, v_ctors_1149_, v___x_1151_);
v___x_1153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1153_, 0, v___x_1152_);
lean_ctor_set(v___x_1153_, 1, v_snd_1114_);
if (v_isShared_1131_ == 0)
{
lean_ctor_set(v___x_1130_, 1, v___x_1153_);
lean_ctor_set(v___x_1130_, 0, v___x_1150_);
v___x_1155_ = v___x_1130_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1159_; 
v_reuseFailAlloc_1159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1159_, 0, v___x_1150_);
lean_ctor_set(v_reuseFailAlloc_1159_, 1, v___x_1153_);
v___x_1155_ = v_reuseFailAlloc_1159_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
size_t v___x_1156_; size_t v___x_1157_; 
v___x_1156_ = ((size_t)1ULL);
v___x_1157_ = lean_usize_add(v_i_1109_, v___x_1156_);
v_i_1109_ = v___x_1157_;
v_b_1110_ = v___x_1155_;
goto _start;
}
}
else
{
lean_dec(v_val_1147_);
lean_del_object(v___x_1130_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
goto v___jp_1118_;
}
}
else
{
lean_dec(v___x_1146_);
lean_del_object(v___x_1130_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
goto v___jp_1118_;
}
}
else
{
lean_object* v___x_1160_; lean_object* v___x_1162_; 
lean_dec_ref(v___x_1142_);
lean_del_object(v___x_1116_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
v___x_1160_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1));
if (v_isShared_1131_ == 0)
{
lean_ctor_set(v___x_1130_, 1, v_snd_1114_);
lean_ctor_set(v___x_1130_, 0, v___x_1160_);
v___x_1162_ = v___x_1130_;
goto v_reusejp_1161_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1160_);
lean_ctor_set(v_reuseFailAlloc_1166_, 1, v_snd_1114_);
v___x_1162_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1161_;
}
v_reusejp_1161_:
{
lean_object* v___x_1164_; 
if (v_isShared_1141_ == 0)
{
lean_ctor_set(v___x_1140_, 0, v___x_1162_);
v___x_1164_ = v___x_1140_;
goto v_reusejp_1163_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v___x_1162_);
v___x_1164_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1163_;
}
v_reusejp_1163_:
{
return v___x_1164_;
}
}
}
}
}
else
{
lean_object* v_a_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1176_; 
lean_del_object(v___x_1130_);
lean_del_object(v___x_1116_);
lean_dec(v_snd_1114_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
v_a_1168_ = lean_ctor_get(v___x_1137_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1137_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1170_ = v___x_1137_;
v_isShared_1171_ = v_isSharedCheck_1176_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_a_1168_);
lean_dec(v___x_1137_);
v___x_1170_ = lean_box(0);
v_isShared_1171_ = v_isSharedCheck_1176_;
goto v_resetjp_1169_;
}
v_resetjp_1169_:
{
lean_object* v___x_1172_; lean_object* v___x_1174_; 
v___x_1172_ = l_Lean_Server_RequestError_ofIoError(v_a_1168_);
if (v_isShared_1171_ == 0)
{
lean_ctor_set(v___x_1170_, 0, v___x_1172_);
v___x_1174_ = v___x_1170_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v___x_1172_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
}
else
{
lean_object* v_a_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1185_; 
lean_del_object(v___x_1130_);
lean_dec(v_snd_1128_);
lean_dec(v_fst_1127_);
lean_del_object(v___x_1116_);
lean_dec(v_snd_1114_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
v_a_1177_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1185_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1179_ = v___x_1134_;
v_isShared_1180_ = v_isSharedCheck_1185_;
goto v_resetjp_1178_;
}
else
{
lean_inc(v_a_1177_);
lean_dec(v___x_1134_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1185_;
goto v_resetjp_1178_;
}
v_resetjp_1178_:
{
lean_object* v___x_1181_; lean_object* v___x_1183_; 
v___x_1181_ = l_Lean_Server_RequestError_ofIoError(v_a_1177_);
if (v_isShared_1180_ == 0)
{
lean_ctor_set(v___x_1179_, 0, v___x_1181_);
v___x_1183_ = v___x_1179_;
goto v_reusejp_1182_;
}
else
{
lean_object* v_reuseFailAlloc_1184_; 
v_reuseFailAlloc_1184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1184_, 0, v___x_1181_);
v___x_1183_ = v_reuseFailAlloc_1184_;
goto v_reusejp_1182_;
}
v_reusejp_1182_:
{
return v___x_1183_;
}
}
}
}
}
else
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; 
lean_dec(v___x_1125_);
lean_del_object(v___x_1116_);
lean_dec_ref(v_ctx_1105_);
lean_dec_ref(v_node_1104_);
v___x_1187_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1));
v___x_1188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1188_, 0, v___x_1187_);
lean_ctor_set(v___x_1188_, 1, v_snd_1114_);
v___x_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1188_);
return v___x_1189_;
}
v___jp_1118_:
{
lean_object* v___x_1119_; lean_object* v___x_1121_; 
v___x_1119_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__1));
if (v_isShared_1117_ == 0)
{
lean_ctor_set(v___x_1116_, 0, v___x_1119_);
v___x_1121_ = v___x_1116_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v___x_1119_);
lean_ctor_set(v_reuseFailAlloc_1123_, 1, v_snd_1114_);
v___x_1121_ = v_reuseFailAlloc_1123_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
lean_object* v___x_1122_; 
v___x_1122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1121_);
return v___x_1122_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___boxed(lean_object* v_node_1192_, lean_object* v_ctx_1193_, lean_object* v_snap_1194_, lean_object* v_as_1195_, lean_object* v_sz_1196_, lean_object* v_i_1197_, lean_object* v_b_1198_, lean_object* v___y_1199_){
_start:
{
size_t v_sz_boxed_1200_; size_t v_i_boxed_1201_; lean_object* v_res_1202_; 
v_sz_boxed_1200_ = lean_unbox_usize(v_sz_1196_);
lean_dec(v_sz_1196_);
v_i_boxed_1201_ = lean_unbox_usize(v_i_1197_);
lean_dec(v_i_1197_);
v_res_1202_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg(v_node_1192_, v_ctx_1193_, v_snap_1194_, v_as_1195_, v_sz_boxed_1200_, v_i_boxed_1201_, v_b_1198_);
lean_dec_ref(v_as_1195_);
lean_dec_ref(v_snap_1194_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2(size_t v_sz_1206_, size_t v_i_1207_, lean_object* v_bs_1208_){
_start:
{
uint8_t v___x_1209_; 
v___x_1209_ = lean_usize_dec_lt(v_i_1207_, v_sz_1206_);
if (v___x_1209_ == 0)
{
lean_object* v___x_1210_; 
v___x_1210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1210_, 0, v_bs_1208_);
return v___x_1210_;
}
else
{
lean_object* v_v_1211_; lean_object* v___x_1212_; uint8_t v___x_1213_; 
v_v_1211_ = lean_array_uget(v_bs_1208_, v_i_1207_);
v___x_1212_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0___closed__1));
lean_inc(v_v_1211_);
v___x_1213_ = l_Lean_Syntax_isOfKind(v_v_1211_, v___x_1212_);
if (v___x_1213_ == 0)
{
lean_object* v___x_1214_; 
lean_dec(v_v_1211_);
lean_dec_ref(v_bs_1208_);
v___x_1214_ = lean_box(0);
return v___x_1214_;
}
else
{
lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v_bs_x27_1217_; lean_object* v_val_1219_; lean_object* v___x_1224_; uint8_t v___x_1225_; 
v___x_1215_ = lean_unsigned_to_nat(0u);
v___x_1216_ = lean_unsigned_to_nat(1u);
v_bs_x27_1217_ = lean_array_uset(v_bs_1208_, v_i_1207_, v___x_1215_);
v___x_1224_ = l_Lean_Syntax_getArg(v_v_1211_, v___x_1215_);
lean_inc(v___x_1224_);
v___x_1225_ = l_Lean_Syntax_matchesNull(v___x_1224_, v___x_1215_);
if (v___x_1225_ == 0)
{
lean_object* v___x_1226_; uint8_t v___x_1227_; 
v___x_1226_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1224_);
v___x_1227_ = l_Lean_Syntax_matchesNull(v___x_1224_, v___x_1226_);
if (v___x_1227_ == 0)
{
lean_object* v___x_1228_; 
lean_dec(v___x_1224_);
lean_dec_ref(v_bs_x27_1217_);
lean_dec(v_v_1211_);
v___x_1228_ = lean_box(0);
return v___x_1228_;
}
else
{
lean_object* v___x_1229_; lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1229_ = l_Lean_Syntax_getArg(v___x_1224_, v___x_1215_);
lean_dec(v___x_1224_);
v___x_1230_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___closed__1));
v___x_1231_ = l_Lean_Syntax_isOfKind(v___x_1229_, v___x_1230_);
if (v___x_1231_ == 0)
{
lean_object* v___x_1232_; 
lean_dec_ref(v_bs_x27_1217_);
lean_dec(v_v_1211_);
v___x_1232_ = lean_box(0);
return v___x_1232_;
}
else
{
lean_object* v___x_1233_; 
v___x_1233_ = l_Lean_Syntax_getArg(v_v_1211_, v___x_1216_);
lean_dec(v_v_1211_);
v_val_1219_ = v___x_1233_;
goto v___jp_1218_;
}
}
}
else
{
lean_object* v___x_1234_; 
lean_dec(v___x_1224_);
v___x_1234_ = l_Lean_Syntax_getArg(v_v_1211_, v___x_1216_);
lean_dec(v_v_1211_);
v_val_1219_ = v___x_1234_;
goto v___jp_1218_;
}
v___jp_1218_:
{
size_t v___x_1220_; size_t v___x_1221_; lean_object* v___x_1222_; 
v___x_1220_ = ((size_t)1ULL);
v___x_1221_ = lean_usize_add(v_i_1207_, v___x_1220_);
v___x_1222_ = lean_array_uset(v_bs_x27_1217_, v_i_1207_, v_val_1219_);
v_i_1207_ = v___x_1221_;
v_bs_1208_ = v___x_1222_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2___boxed(lean_object* v_sz_1235_, lean_object* v_i_1236_, lean_object* v_bs_1237_){
_start:
{
size_t v_sz_boxed_1238_; size_t v_i_boxed_1239_; lean_object* v_res_1240_; 
v_sz_boxed_1238_ = lean_unbox_usize(v_sz_1235_);
lean_dec(v_sz_1235_);
v_i_boxed_1239_ = lean_unbox_usize(v_i_1236_);
lean_dec(v_i_1236_);
v_res_1240_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2(v_sz_boxed_1238_, v_i_boxed_1239_, v_bs_1237_);
return v_res_1240_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(lean_object* v_a_1241_, lean_object* v_CodeActionParams_1242_, lean_object* v_as_1243_, size_t v_i_1244_, size_t v_stop_1245_, lean_object* v_b_1246_){
_start:
{
lean_object* v_a_1249_; uint8_t v___x_1253_; 
v___x_1253_ = lean_usize_dec_eq(v_i_1244_, v_stop_1245_);
if (v___x_1253_ == 0)
{
lean_object* v___x_1254_; uint8_t v___y_1256_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v___x_1254_ = lean_array_uget_borrowed(v_as_1243_, v_i_1244_);
v___x_1258_ = l_Lean_Elab_Info_stx(v___x_1254_);
v___x_1259_ = lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f(v___x_1258_);
if (lean_obj_tag(v___x_1259_) == 1)
{
lean_object* v_toEditableDocumentCore_1260_; lean_object* v_meta_1261_; lean_object* v_val_1262_; lean_object* v_text_1263_; lean_object* v_range_1264_; lean_object* v___x_1265_; lean_object* v_start_1266_; lean_object* v_end_1267_; lean_object* v_start_1268_; lean_object* v_end_1269_; uint8_t v___y_1271_; uint8_t v___x_1273_; 
v_toEditableDocumentCore_1260_ = lean_ctor_get(v_a_1241_, 0);
v_meta_1261_ = lean_ctor_get(v_toEditableDocumentCore_1260_, 0);
v_val_1262_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_val_1262_);
lean_dec_ref_known(v___x_1259_, 1);
v_text_1263_ = lean_ctor_get(v_meta_1261_, 3);
v_range_1264_ = lean_ctor_get(v_CodeActionParams_1242_, 3);
lean_inc_ref(v_text_1263_);
v___x_1265_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_1263_, v_val_1262_);
v_start_1266_ = lean_ctor_get(v___x_1265_, 0);
lean_inc_ref(v_start_1266_);
v_end_1267_ = lean_ctor_get(v___x_1265_, 1);
lean_inc_ref(v_end_1267_);
lean_dec_ref(v___x_1265_);
v_start_1268_ = lean_ctor_get(v_range_1264_, 0);
v_end_1269_ = lean_ctor_get(v_range_1264_, 1);
v___x_1273_ = l_Lean_Lsp_instOrdPosition_ord(v_start_1266_, v_start_1268_);
lean_dec_ref(v_start_1266_);
if (v___x_1273_ == 2)
{
if (v___x_1253_ == 0)
{
lean_dec_ref(v_end_1267_);
v_a_1249_ = v_b_1246_;
goto v___jp_1248_;
}
else
{
v___y_1271_ = v___x_1253_;
goto v___jp_1270_;
}
}
else
{
uint8_t v___x_1274_; 
v___x_1274_ = 1;
v___y_1271_ = v___x_1274_;
goto v___jp_1270_;
}
v___jp_1270_:
{
uint8_t v___x_1272_; 
v___x_1272_ = l_Lean_Lsp_instOrdPosition_ord(v_end_1269_, v_end_1267_);
lean_dec_ref(v_end_1267_);
if (v___x_1272_ == 2)
{
v___y_1256_ = v___x_1253_;
goto v___jp_1255_;
}
else
{
v___y_1256_ = v___y_1271_;
goto v___jp_1255_;
}
}
}
else
{
lean_dec(v___x_1259_);
v_a_1249_ = v_b_1246_;
goto v___jp_1248_;
}
v___jp_1255_:
{
if (v___y_1256_ == 0)
{
v_a_1249_ = v_b_1246_;
goto v___jp_1248_;
}
else
{
lean_object* v___x_1257_; 
lean_inc(v___x_1254_);
v___x_1257_ = lean_array_push(v_b_1246_, v___x_1254_);
v_a_1249_ = v___x_1257_;
goto v___jp_1248_;
}
}
}
else
{
lean_object* v___x_1275_; 
lean_dec_ref(v_a_1241_);
v___x_1275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1275_, 0, v_b_1246_);
return v___x_1275_;
}
v___jp_1248_:
{
size_t v___x_1250_; size_t v___x_1251_; 
v___x_1250_ = ((size_t)1ULL);
v___x_1251_ = lean_usize_add(v_i_1244_, v___x_1250_);
v_i_1244_ = v___x_1251_;
v_b_1246_ = v_a_1249_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg___boxed(lean_object* v_a_1276_, lean_object* v_CodeActionParams_1277_, lean_object* v_as_1278_, lean_object* v_i_1279_, lean_object* v_stop_1280_, lean_object* v_b_1281_, lean_object* v___y_1282_){
_start:
{
size_t v_i_boxed_1283_; size_t v_stop_boxed_1284_; lean_object* v_res_1285_; 
v_i_boxed_1283_ = lean_unbox_usize(v_i_1279_);
lean_dec(v_i_1279_);
v_stop_boxed_1284_ = lean_unbox_usize(v_stop_1280_);
lean_dec(v_stop_1280_);
v_res_1285_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(v_a_1276_, v_CodeActionParams_1277_, v_as_1278_, v_i_boxed_1283_, v_stop_boxed_1284_, v_b_1281_);
lean_dec_ref(v_as_1278_);
lean_dec_ref(v_CodeActionParams_1277_);
return v_res_1285_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0(lean_object* v_x_1286_){
_start:
{
if (lean_obj_tag(v_x_1286_) == 0)
{
uint8_t v___x_1287_; 
v___x_1287_ = 0;
return v___x_1287_;
}
else
{
lean_object* v_head_1288_; lean_object* v_snd_1289_; uint8_t v___x_1290_; 
v_head_1288_ = lean_ctor_get(v_x_1286_, 0);
v_snd_1289_ = lean_ctor_get(v_head_1288_, 1);
v___x_1290_ = lean_unbox(v_snd_1289_);
if (v___x_1290_ == 0)
{
lean_object* v_tail_1291_; 
v_tail_1291_ = lean_ctor_get(v_x_1286_, 1);
v_x_1286_ = v_tail_1291_;
goto _start;
}
else
{
uint8_t v___x_1293_; 
v___x_1293_ = lean_unbox(v_snd_1289_);
return v___x_1293_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0___boxed(lean_object* v_x_1294_){
_start:
{
uint8_t v_res_1295_; lean_object* v_r_1296_; 
v_res_1295_ = lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0(v_x_1294_);
lean_dec(v_x_1294_);
v_r_1296_ = lean_box(v_res_1295_);
return v_r_1296_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10(lean_object* v_x_1297_){
_start:
{
if (lean_obj_tag(v_x_1297_) == 0)
{
uint8_t v___x_1298_; 
v___x_1298_ = 0;
return v___x_1298_;
}
else
{
lean_object* v_head_1299_; lean_object* v_tail_1300_; uint8_t v___x_1301_; 
v_head_1299_ = lean_ctor_get(v_x_1297_, 0);
v_tail_1300_ = lean_ctor_get(v_x_1297_, 1);
v___x_1301_ = lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__0(v_head_1299_);
if (v___x_1301_ == 0)
{
v_x_1297_ = v_tail_1300_;
goto _start;
}
else
{
return v___x_1301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10___boxed(lean_object* v_x_1303_){
_start:
{
uint8_t v_res_1304_; lean_object* v_r_1305_; 
v_res_1304_ = lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10(v_x_1303_);
lean_dec(v_x_1303_);
v_r_1305_ = lean_box(v_res_1304_);
return v_r_1305_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11(lean_object* v_as_1306_, size_t v_sz_1307_, size_t v_i_1308_, lean_object* v_b_1309_){
_start:
{
uint8_t v___x_1310_; 
v___x_1310_ = lean_usize_dec_lt(v_i_1308_, v_sz_1307_);
if (v___x_1310_ == 0)
{
lean_inc_ref(v_b_1309_);
return v_b_1309_;
}
else
{
lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v_a_1313_; uint8_t v___y_1315_; uint8_t v___x_1322_; 
v___x_1311_ = lean_box(0);
v___x_1312_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0));
v_a_1313_ = lean_array_uget_borrowed(v_as_1306_, v_i_1308_);
v___x_1322_ = l_Lean_Syntax_isAtom(v_a_1313_);
if (v___x_1322_ == 0)
{
v___y_1315_ = v___x_1322_;
goto v___jp_1314_;
}
else
{
lean_object* v___x_1323_; lean_object* v___x_1324_; uint8_t v___x_1325_; 
v___x_1323_ = l_Lean_Syntax_getAtomVal(v_a_1313_);
v___x_1324_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__1));
v___x_1325_ = lean_string_dec_eq(v___x_1323_, v___x_1324_);
lean_dec_ref(v___x_1323_);
v___y_1315_ = v___x_1325_;
goto v___jp_1314_;
}
v___jp_1314_:
{
if (v___y_1315_ == 0)
{
size_t v___x_1316_; size_t v___x_1317_; 
v___x_1316_ = ((size_t)1ULL);
v___x_1317_ = lean_usize_add(v_i_1308_, v___x_1316_);
v_i_1308_ = v___x_1317_;
v_b_1309_ = v___x_1312_;
goto _start;
}
else
{
lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; 
lean_inc(v_a_1313_);
v___x_1319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1319_, 0, v_a_1313_);
v___x_1320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1320_, 0, v___x_1319_);
v___x_1321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
lean_ctor_set(v___x_1321_, 1, v___x_1311_);
return v___x_1321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11___boxed(lean_object* v_as_1326_, lean_object* v_sz_1327_, lean_object* v_i_1328_, lean_object* v_b_1329_){
_start:
{
size_t v_sz_boxed_1330_; size_t v_i_boxed_1331_; lean_object* v_res_1332_; 
v_sz_boxed_1330_ = lean_unbox_usize(v_sz_1327_);
lean_dec(v_sz_1327_);
v_i_boxed_1331_ = lean_unbox_usize(v_i_1328_);
lean_dec(v_i_1328_);
v_res_1332_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11(v_as_1326_, v_sz_boxed_1330_, v_i_boxed_1331_, v_b_1329_);
lean_dec_ref(v_b_1329_);
lean_dec_ref(v_as_1326_);
return v_res_1332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand(lean_object* v_CodeActionParams_1339_, lean_object* v_snap_1340_, lean_object* v_ctx_1341_, lean_object* v_node_1342_, lean_object* v_a_1343_){
_start:
{
lean_object* v___x_1345_; lean_object* v_a_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1548_; 
v___x_1345_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_matchExpand_spec__1(v_a_1343_);
v_a_1346_ = lean_ctor_get(v___x_1345_, 0);
v_isSharedCheck_1548_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1548_ == 0)
{
v___x_1348_ = v___x_1345_;
v_isShared_1349_ = v_isSharedCheck_1548_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_a_1346_);
lean_dec(v___x_1345_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1548_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v___y_1351_; lean_object* v___y_1352_; uint8_t v___y_1353_; lean_object* v___y_1354_; size_t v___y_1355_; lean_object* v___y_1356_; lean_object* v___y_1357_; uint8_t v___y_1358_; lean_object* v___y_1402_; uint8_t v___y_1403_; lean_object* v___y_1404_; size_t v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1407_; lean_object* v___y_1408_; lean_object* v___x_1410_; lean_object* v_allMatchInfos_1411_; lean_object* v___x_1412_; lean_object* v___y_1414_; uint8_t v___y_1415_; lean_object* v___y_1416_; lean_object* v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___y_1443_; lean_object* v___y_1444_; uint8_t v___y_1445_; lean_object* v___y_1446_; lean_object* v___y_1447_; lean_object* v___y_1448_; lean_object* v___y_1467_; lean_object* v___y_1468_; uint8_t v___y_1469_; lean_object* v___y_1470_; lean_object* v___y_1471_; lean_object* v___y_1472_; lean_object* v___y_1473_; lean_object* v___y_1474_; lean_object* v_a_1488_; lean_object* v___y_1528_; lean_object* v___x_1538_; lean_object* v___x_1539_; uint8_t v___x_1540_; 
v___x_1410_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___closed__3));
lean_inc_ref(v_node_1342_);
v_allMatchInfos_1411_ = lp_batteries_Batteries_CodeAction_findAllInfos(v___x_1410_, v_node_1342_);
v___x_1412_ = lean_unsigned_to_nat(0u);
v___x_1538_ = lean_array_get_size(v_allMatchInfos_1411_);
v___x_1539_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_findAllInfos___closed__0));
v___x_1540_ = lean_nat_dec_lt(v___x_1412_, v___x_1538_);
if (v___x_1540_ == 0)
{
lean_dec_ref(v_allMatchInfos_1411_);
v_a_1488_ = v___x_1539_;
goto v___jp_1487_;
}
else
{
uint8_t v___x_1541_; 
v___x_1541_ = lean_nat_dec_le(v___x_1538_, v___x_1538_);
if (v___x_1541_ == 0)
{
if (v___x_1540_ == 0)
{
lean_dec_ref(v_allMatchInfos_1411_);
v_a_1488_ = v___x_1539_;
goto v___jp_1487_;
}
else
{
size_t v___x_1542_; size_t v___x_1543_; lean_object* v___x_1544_; 
v___x_1542_ = ((size_t)0ULL);
v___x_1543_ = lean_usize_of_nat(v___x_1538_);
lean_inc(v_a_1346_);
v___x_1544_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(v_a_1346_, v_CodeActionParams_1339_, v_allMatchInfos_1411_, v___x_1542_, v___x_1543_, v___x_1539_);
lean_dec_ref(v_allMatchInfos_1411_);
v___y_1528_ = v___x_1544_;
goto v___jp_1527_;
}
}
else
{
size_t v___x_1545_; size_t v___x_1546_; lean_object* v___x_1547_; 
v___x_1545_ = ((size_t)0ULL);
v___x_1546_ = lean_usize_of_nat(v___x_1538_);
lean_inc(v_a_1346_);
v___x_1547_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(v_a_1346_, v_CodeActionParams_1339_, v_allMatchInfos_1411_, v___x_1545_, v___x_1546_, v___x_1539_);
lean_dec_ref(v_allMatchInfos_1411_);
v___y_1528_ = v___x_1547_;
goto v___jp_1527_;
}
}
v___jp_1350_:
{
lean_object* v___x_1359_; lean_object* v___x_1360_; size_t v_sz_1361_; lean_object* v___x_1362_; 
v___x_1359_ = lean_box(0);
v___x_1360_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___closed__0));
v_sz_1361_ = lean_array_size(v___y_1357_);
v___x_1362_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg(v_node_1342_, v_ctx_1341_, v_snap_1340_, v___y_1357_, v_sz_1361_, v___y_1355_, v___x_1360_);
lean_dec_ref(v___y_1357_);
if (lean_obj_tag(v___x_1362_) == 0)
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1392_; 
v_a_1363_ = lean_ctor_get(v___x_1362_, 0);
v_isSharedCheck_1392_ = !lean_is_exclusive(v___x_1362_);
if (v_isSharedCheck_1392_ == 0)
{
v___x_1365_ = v___x_1362_;
v_isShared_1366_ = v_isSharedCheck_1392_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1362_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1392_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v_fst_1367_; 
v_fst_1367_ = lean_ctor_get(v_a_1363_, 0);
if (lean_obj_tag(v_fst_1367_) == 0)
{
lean_object* v_snd_1368_; uint8_t v___x_1369_; 
v_snd_1368_ = lean_ctor_get(v_a_1363_, 1);
lean_inc(v_snd_1368_);
lean_dec(v_a_1363_);
v___x_1369_ = lp_batteries_List_any___at___00Batteries_CodeAction_matchExpand_spec__10(v_snd_1368_);
if (v___x_1369_ == 0)
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1375_; 
v___x_1370_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___closed__1));
v___x_1371_ = lp_batteries_Batteries_CodeAction_matchExpand___lam__1(v___y_1351_, v_a_1346_, v_snd_1368_, v___x_1359_, v_snap_1340_, v___y_1358_, v___x_1370_, v___y_1353_);
v___x_1372_ = lean_mk_empty_array_with_capacity(v___y_1352_);
v___x_1373_ = lean_array_push(v___x_1372_, v___x_1371_);
if (v_isShared_1366_ == 0)
{
lean_ctor_set(v___x_1365_, 0, v___x_1373_);
v___x_1375_ = v___x_1365_;
goto v_reusejp_1374_;
}
else
{
lean_object* v_reuseFailAlloc_1376_; 
v_reuseFailAlloc_1376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1376_, 0, v___x_1373_);
v___x_1375_ = v_reuseFailAlloc_1376_;
goto v_reusejp_1374_;
}
v_reusejp_1374_:
{
return v___x_1375_;
}
}
else
{
lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; uint8_t v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1386_; 
v___x_1377_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___closed__1));
lean_inc_ref(v_snap_1340_);
lean_inc(v_snd_1368_);
lean_inc(v_a_1346_);
lean_inc_ref(v___y_1351_);
v___x_1378_ = lp_batteries_Batteries_CodeAction_matchExpand___lam__1(v___y_1351_, v_a_1346_, v_snd_1368_, v___x_1359_, v_snap_1340_, v___y_1358_, v___x_1377_, v___x_1369_);
v___x_1379_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_matchExpand___closed__2));
v___x_1380_ = 0;
v___x_1381_ = lp_batteries_Batteries_CodeAction_matchExpand___lam__1(v___y_1351_, v_a_1346_, v_snd_1368_, v___x_1359_, v_snap_1340_, v___y_1358_, v___x_1379_, v___x_1380_);
v___x_1382_ = lean_mk_empty_array_with_capacity(v___y_1354_);
v___x_1383_ = lean_array_push(v___x_1382_, v___x_1378_);
v___x_1384_ = lean_array_push(v___x_1383_, v___x_1381_);
if (v_isShared_1366_ == 0)
{
lean_ctor_set(v___x_1365_, 0, v___x_1384_);
v___x_1386_ = v___x_1365_;
goto v_reusejp_1385_;
}
else
{
lean_object* v_reuseFailAlloc_1387_; 
v_reuseFailAlloc_1387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1387_, 0, v___x_1384_);
v___x_1386_ = v_reuseFailAlloc_1387_;
goto v_reusejp_1385_;
}
v_reusejp_1385_:
{
return v___x_1386_;
}
}
}
else
{
lean_object* v_val_1388_; lean_object* v___x_1390_; 
lean_inc_ref(v_fst_1367_);
lean_dec(v_a_1363_);
lean_dec_ref(v___y_1351_);
lean_dec(v_a_1346_);
lean_dec_ref(v_snap_1340_);
v_val_1388_ = lean_ctor_get(v_fst_1367_, 0);
lean_inc(v_val_1388_);
lean_dec_ref_known(v_fst_1367_, 1);
if (v_isShared_1366_ == 0)
{
lean_ctor_set(v___x_1365_, 0, v_val_1388_);
v___x_1390_ = v___x_1365_;
goto v_reusejp_1389_;
}
else
{
lean_object* v_reuseFailAlloc_1391_; 
v_reuseFailAlloc_1391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1391_, 0, v_val_1388_);
v___x_1390_ = v_reuseFailAlloc_1391_;
goto v_reusejp_1389_;
}
v_reusejp_1389_:
{
return v___x_1390_;
}
}
}
}
else
{
lean_object* v_a_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1400_; 
lean_dec_ref(v___y_1351_);
lean_dec(v_a_1346_);
lean_dec_ref(v_snap_1340_);
v_a_1393_ = lean_ctor_get(v___x_1362_, 0);
v_isSharedCheck_1400_ = !lean_is_exclusive(v___x_1362_);
if (v_isSharedCheck_1400_ == 0)
{
v___x_1395_ = v___x_1362_;
v_isShared_1396_ = v_isSharedCheck_1400_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_a_1393_);
lean_dec(v___x_1362_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1400_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
lean_object* v___x_1398_; 
if (v_isShared_1396_ == 0)
{
v___x_1398_ = v___x_1395_;
goto v_reusejp_1397_;
}
else
{
lean_object* v_reuseFailAlloc_1399_; 
v_reuseFailAlloc_1399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1399_, 0, v_a_1393_);
v___x_1398_ = v_reuseFailAlloc_1399_;
goto v_reusejp_1397_;
}
v_reusejp_1397_:
{
return v___x_1398_;
}
}
}
}
v___jp_1401_:
{
uint8_t v___x_1409_; 
v___x_1409_ = 0;
v___y_1351_ = v___y_1402_;
v___y_1352_ = v___y_1404_;
v___y_1353_ = v___y_1403_;
v___y_1354_ = v___y_1406_;
v___y_1355_ = v___y_1405_;
v___y_1356_ = v___y_1407_;
v___y_1357_ = v___y_1408_;
v___y_1358_ = v___x_1409_;
goto v___jp_1350_;
}
v___jp_1413_:
{
size_t v_sz_1421_; size_t v___x_1422_; lean_object* v___x_1423_; 
v_sz_1421_ = lean_array_size(v___y_1420_);
v___x_1422_ = ((size_t)0ULL);
v___x_1423_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__0(v_sz_1421_, v___x_1422_, v___y_1420_);
if (lean_obj_tag(v___x_1423_) == 0)
{
lean_object* v___x_1424_; lean_object* v___x_1426_; 
lean_dec(v___y_1418_);
lean_dec_ref(v___y_1414_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1424_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
if (v_isShared_1349_ == 0)
{
lean_ctor_set(v___x_1348_, 0, v___x_1424_);
v___x_1426_ = v___x_1348_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v___x_1424_);
v___x_1426_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
return v___x_1426_;
}
}
else
{
lean_object* v_val_1428_; size_t v_sz_1429_; lean_object* v___x_1430_; 
v_val_1428_ = lean_ctor_get(v___x_1423_, 0);
lean_inc(v_val_1428_);
lean_dec_ref_known(v___x_1423_, 1);
v_sz_1429_ = lean_array_size(v_val_1428_);
v___x_1430_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_matchExpand_spec__2(v_sz_1429_, v___x_1422_, v_val_1428_);
if (lean_obj_tag(v___x_1430_) == 1)
{
lean_object* v_val_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; size_t v_sz_1434_; lean_object* v___x_1435_; lean_object* v_fst_1436_; 
lean_del_object(v___x_1348_);
v_val_1431_ = lean_ctor_get(v___x_1430_, 0);
lean_inc(v_val_1431_);
lean_dec_ref_known(v___x_1430_, 1);
v___x_1432_ = l_Lean_Syntax_getArgs(v___y_1418_);
lean_dec(v___y_1418_);
v___x_1433_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__2___closed__0));
v_sz_1434_ = lean_array_size(v___x_1432_);
v___x_1435_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__11(v___x_1432_, v_sz_1434_, v___x_1422_, v___x_1433_);
lean_dec_ref(v___x_1432_);
v_fst_1436_ = lean_ctor_get(v___x_1435_, 0);
lean_inc(v_fst_1436_);
lean_dec_ref(v___x_1435_);
if (lean_obj_tag(v_fst_1436_) == 0)
{
v___y_1402_ = v___y_1414_;
v___y_1403_ = v___y_1415_;
v___y_1404_ = v___y_1416_;
v___y_1405_ = v___x_1422_;
v___y_1406_ = v___y_1417_;
v___y_1407_ = v___y_1419_;
v___y_1408_ = v_val_1431_;
goto v___jp_1401_;
}
else
{
lean_object* v_val_1437_; 
v_val_1437_ = lean_ctor_get(v_fst_1436_, 0);
lean_inc(v_val_1437_);
lean_dec_ref_known(v_fst_1436_, 1);
if (lean_obj_tag(v_val_1437_) == 0)
{
v___y_1402_ = v___y_1414_;
v___y_1403_ = v___y_1415_;
v___y_1404_ = v___y_1416_;
v___y_1405_ = v___x_1422_;
v___y_1406_ = v___y_1417_;
v___y_1407_ = v___y_1419_;
v___y_1408_ = v_val_1431_;
goto v___jp_1401_;
}
else
{
lean_dec_ref_known(v_val_1437_, 1);
v___y_1351_ = v___y_1414_;
v___y_1352_ = v___y_1416_;
v___y_1353_ = v___y_1415_;
v___y_1354_ = v___y_1417_;
v___y_1355_ = v___x_1422_;
v___y_1356_ = v___y_1419_;
v___y_1357_ = v_val_1431_;
v___y_1358_ = v___y_1415_;
goto v___jp_1350_;
}
}
}
else
{
lean_object* v___x_1438_; lean_object* v___x_1440_; 
lean_dec(v___x_1430_);
lean_dec(v___y_1418_);
lean_dec_ref(v___y_1414_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1438_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
if (v_isShared_1349_ == 0)
{
lean_ctor_set(v___x_1348_, 0, v___x_1438_);
v___x_1440_ = v___x_1348_;
goto v_reusejp_1439_;
}
else
{
lean_object* v_reuseFailAlloc_1441_; 
v_reuseFailAlloc_1441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1441_, 0, v___x_1438_);
v___x_1440_ = v_reuseFailAlloc_1441_;
goto v_reusejp_1439_;
}
v_reusejp_1439_:
{
return v___x_1440_;
}
}
}
}
v___jp_1442_:
{
lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; uint8_t v___x_1454_; 
v___x_1449_ = lean_unsigned_to_nat(3u);
v___x_1450_ = l_Lean_Syntax_getArg(v___y_1447_, v___x_1449_);
v___x_1451_ = l_Lean_Syntax_getArgs(v___x_1450_);
lean_dec(v___x_1450_);
v___x_1452_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__0));
v___x_1453_ = lean_array_get_size(v___x_1451_);
v___x_1454_ = lean_nat_dec_lt(v___x_1412_, v___x_1453_);
if (v___x_1454_ == 0)
{
lean_dec_ref(v___x_1451_);
v___y_1414_ = v___y_1443_;
v___y_1415_ = v___y_1445_;
v___y_1416_ = v___y_1444_;
v___y_1417_ = v___y_1446_;
v___y_1418_ = v___y_1447_;
v___y_1419_ = v___y_1448_;
v___y_1420_ = v___x_1452_;
goto v___jp_1413_;
}
else
{
lean_object* v___x_1455_; lean_object* v___x_1456_; uint8_t v___x_1457_; 
v___x_1455_ = lean_box(v___y_1445_);
v___x_1456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1456_, 0, v___x_1455_);
lean_ctor_set(v___x_1456_, 1, v___x_1452_);
v___x_1457_ = lean_nat_dec_le(v___x_1453_, v___x_1453_);
if (v___x_1457_ == 0)
{
if (v___x_1454_ == 0)
{
lean_dec_ref_known(v___x_1456_, 2);
lean_dec_ref(v___x_1451_);
v___y_1414_ = v___y_1443_;
v___y_1415_ = v___y_1445_;
v___y_1416_ = v___y_1444_;
v___y_1417_ = v___y_1446_;
v___y_1418_ = v___y_1447_;
v___y_1419_ = v___y_1448_;
v___y_1420_ = v___x_1452_;
goto v___jp_1413_;
}
else
{
size_t v___x_1458_; size_t v___x_1459_; lean_object* v___x_1460_; lean_object* v_snd_1461_; 
v___x_1458_ = ((size_t)0ULL);
v___x_1459_ = lean_usize_of_nat(v___x_1453_);
v___x_1460_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(v___y_1445_, v___x_1451_, v___x_1458_, v___x_1459_, v___x_1456_);
lean_dec_ref(v___x_1451_);
v_snd_1461_ = lean_ctor_get(v___x_1460_, 1);
lean_inc(v_snd_1461_);
lean_dec_ref(v___x_1460_);
v___y_1414_ = v___y_1443_;
v___y_1415_ = v___y_1445_;
v___y_1416_ = v___y_1444_;
v___y_1417_ = v___y_1446_;
v___y_1418_ = v___y_1447_;
v___y_1419_ = v___y_1448_;
v___y_1420_ = v_snd_1461_;
goto v___jp_1413_;
}
}
else
{
size_t v___x_1462_; size_t v___x_1463_; lean_object* v___x_1464_; lean_object* v_snd_1465_; 
v___x_1462_ = ((size_t)0ULL);
v___x_1463_ = lean_usize_of_nat(v___x_1453_);
v___x_1464_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_getMatchHeaderRange_x3f_spec__3(v___y_1445_, v___x_1451_, v___x_1462_, v___x_1463_, v___x_1456_);
lean_dec_ref(v___x_1451_);
v_snd_1465_ = lean_ctor_get(v___x_1464_, 1);
lean_inc(v_snd_1465_);
lean_dec_ref(v___x_1464_);
v___y_1414_ = v___y_1443_;
v___y_1415_ = v___y_1445_;
v___y_1416_ = v___y_1444_;
v___y_1417_ = v___y_1446_;
v___y_1418_ = v___y_1447_;
v___y_1419_ = v___y_1448_;
v___y_1420_ = v_snd_1465_;
goto v___jp_1413_;
}
}
}
v___jp_1466_:
{
lean_object* v___x_1475_; lean_object* v___x_1476_; uint8_t v___x_1477_; 
v___x_1475_ = lean_unsigned_to_nat(2u);
v___x_1476_ = l_Lean_Syntax_getArg(v___y_1473_, v___x_1475_);
v___x_1477_ = l_Lean_Syntax_isNone(v___x_1476_);
if (v___x_1477_ == 0)
{
uint8_t v___x_1478_; 
lean_inc(v___x_1476_);
v___x_1478_ = l_Lean_Syntax_matchesNull(v___x_1476_, v___y_1470_);
if (v___x_1478_ == 0)
{
lean_object* v___x_1479_; lean_object* v___x_1480_; 
lean_dec(v___x_1476_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1467_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1479_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
v___x_1480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1480_, 0, v___x_1479_);
return v___x_1480_;
}
else
{
lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; uint8_t v___x_1484_; 
v___x_1481_ = l_Lean_Syntax_getArg(v___x_1476_, v___x_1412_);
lean_dec(v___x_1476_);
v___x_1482_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__1));
lean_inc_ref(v___y_1472_);
lean_inc_ref(v___y_1468_);
lean_inc_ref(v___y_1471_);
v___x_1483_ = l_Lean_Name_mkStr4(v___y_1471_, v___y_1468_, v___y_1472_, v___x_1482_);
v___x_1484_ = l_Lean_Syntax_isOfKind(v___x_1481_, v___x_1483_);
lean_dec(v___x_1483_);
if (v___x_1484_ == 0)
{
lean_object* v___x_1485_; lean_object* v___x_1486_; 
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1467_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1485_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
v___x_1486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1486_, 0, v___x_1485_);
return v___x_1486_;
}
else
{
v___y_1443_ = v___y_1467_;
v___y_1444_ = v___y_1470_;
v___y_1445_ = v___y_1469_;
v___y_1446_ = v___x_1475_;
v___y_1447_ = v___y_1473_;
v___y_1448_ = v___y_1474_;
goto v___jp_1442_;
}
}
}
else
{
lean_dec(v___x_1476_);
v___y_1443_ = v___y_1467_;
v___y_1444_ = v___y_1470_;
v___y_1445_ = v___y_1469_;
v___y_1446_ = v___x_1475_;
v___y_1447_ = v___y_1473_;
v___y_1448_ = v___y_1474_;
goto v___jp_1442_;
}
}
v___jp_1487_:
{
lean_object* v___x_1489_; uint8_t v___x_1490_; 
v___x_1489_ = lean_array_get_size(v_a_1488_);
v___x_1490_ = lean_nat_dec_lt(v___x_1412_, v___x_1489_);
if (v___x_1490_ == 0)
{
lean_object* v___x_1491_; lean_object* v___x_1492_; 
lean_dec_ref(v_a_1488_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1491_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
v___x_1492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1492_, 0, v___x_1491_);
return v___x_1492_;
}
else
{
lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1493_ = lean_array_fget(v_a_1488_, v___x_1412_);
lean_dec_ref(v_a_1488_);
v___x_1494_ = l_Lean_Elab_Info_stx(v___x_1493_);
lean_dec(v___x_1493_);
lean_inc(v___x_1494_);
v___x_1495_ = lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f(v___x_1494_);
if (lean_obj_tag(v___x_1495_) == 1)
{
lean_object* v_val_1496_; lean_object* v___x_1498_; uint8_t v_isShared_1499_; uint8_t v_isSharedCheck_1524_; 
v_val_1496_ = lean_ctor_get(v___x_1495_, 0);
v_isSharedCheck_1524_ = !lean_is_exclusive(v___x_1495_);
if (v_isSharedCheck_1524_ == 0)
{
v___x_1498_ = v___x_1495_;
v_isShared_1499_ = v_isSharedCheck_1524_;
goto v_resetjp_1497_;
}
else
{
lean_inc(v_val_1496_);
lean_dec(v___x_1495_);
v___x_1498_ = lean_box(0);
v_isShared_1499_ = v_isSharedCheck_1524_;
goto v_resetjp_1497_;
}
v_resetjp_1497_:
{
lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; uint8_t v___x_1504_; 
v___x_1500_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__0));
v___x_1501_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__1));
v___x_1502_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__2));
v___x_1503_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_isMatchTerm___closed__4));
lean_inc(v___x_1494_);
v___x_1504_ = l_Lean_Syntax_isOfKind(v___x_1494_, v___x_1503_);
if (v___x_1504_ == 0)
{
lean_object* v___x_1505_; lean_object* v___x_1507_; 
lean_dec(v_val_1496_);
lean_dec(v___x_1494_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1505_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
if (v_isShared_1499_ == 0)
{
lean_ctor_set_tag(v___x_1498_, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1505_);
v___x_1507_ = v___x_1498_;
goto v_reusejp_1506_;
}
else
{
lean_object* v_reuseFailAlloc_1508_; 
v_reuseFailAlloc_1508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1508_, 0, v___x_1505_);
v___x_1507_ = v_reuseFailAlloc_1508_;
goto v_reusejp_1506_;
}
v_reusejp_1506_:
{
return v___x_1507_;
}
}
else
{
lean_object* v___x_1509_; lean_object* v___x_1510_; uint8_t v___x_1511_; 
v___x_1509_ = lean_unsigned_to_nat(1u);
v___x_1510_ = l_Lean_Syntax_getArg(v___x_1494_, v___x_1509_);
v___x_1511_ = l_Lean_Syntax_isNone(v___x_1510_);
if (v___x_1511_ == 0)
{
uint8_t v___x_1512_; 
lean_inc(v___x_1510_);
v___x_1512_ = l_Lean_Syntax_matchesNull(v___x_1510_, v___x_1509_);
if (v___x_1512_ == 0)
{
lean_object* v___x_1513_; lean_object* v___x_1515_; 
lean_dec(v___x_1510_);
lean_dec(v_val_1496_);
lean_dec(v___x_1494_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1513_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
if (v_isShared_1499_ == 0)
{
lean_ctor_set_tag(v___x_1498_, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1513_);
v___x_1515_ = v___x_1498_;
goto v_reusejp_1514_;
}
else
{
lean_object* v_reuseFailAlloc_1516_; 
v_reuseFailAlloc_1516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1516_, 0, v___x_1513_);
v___x_1515_ = v_reuseFailAlloc_1516_;
goto v_reusejp_1514_;
}
v_reusejp_1514_:
{
return v___x_1515_;
}
}
else
{
lean_object* v___x_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; 
v___x_1517_ = l_Lean_Syntax_getArg(v___x_1510_, v___x_1412_);
lean_dec(v___x_1510_);
v___x_1518_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getMatchHeaderRange_x3f___closed__4));
v___x_1519_ = l_Lean_Syntax_isOfKind(v___x_1517_, v___x_1518_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1520_; lean_object* v___x_1522_; 
lean_dec(v_val_1496_);
lean_dec(v___x_1494_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1520_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
if (v_isShared_1499_ == 0)
{
lean_ctor_set_tag(v___x_1498_, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1520_);
v___x_1522_ = v___x_1498_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1523_; 
v_reuseFailAlloc_1523_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1523_, 0, v___x_1520_);
v___x_1522_ = v_reuseFailAlloc_1523_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
return v___x_1522_;
}
}
else
{
lean_del_object(v___x_1498_);
v___y_1467_ = v_val_1496_;
v___y_1468_ = v___x_1501_;
v___y_1469_ = v___x_1504_;
v___y_1470_ = v___x_1509_;
v___y_1471_ = v___x_1500_;
v___y_1472_ = v___x_1502_;
v___y_1473_ = v___x_1494_;
v___y_1474_ = v_a_1343_;
goto v___jp_1466_;
}
}
}
else
{
lean_dec(v___x_1510_);
lean_del_object(v___x_1498_);
v___y_1467_ = v_val_1496_;
v___y_1468_ = v___x_1501_;
v___y_1469_ = v___x_1504_;
v___y_1470_ = v___x_1509_;
v___y_1471_ = v___x_1500_;
v___y_1472_ = v___x_1502_;
v___y_1473_ = v___x_1494_;
v___y_1474_ = v_a_1343_;
goto v___jp_1466_;
}
}
}
}
else
{
lean_object* v___x_1525_; lean_object* v___x_1526_; 
lean_dec(v___x_1495_);
lean_dec(v___x_1494_);
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v___x_1525_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg___closed__0));
v___x_1526_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1526_, 0, v___x_1525_);
return v___x_1526_;
}
}
}
v___jp_1527_:
{
if (lean_obj_tag(v___y_1528_) == 0)
{
lean_object* v_a_1529_; 
v_a_1529_ = lean_ctor_get(v___y_1528_, 0);
lean_inc(v_a_1529_);
lean_dec_ref_known(v___y_1528_, 1);
v_a_1488_ = v_a_1529_;
goto v___jp_1487_;
}
else
{
lean_object* v_a_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1537_; 
lean_del_object(v___x_1348_);
lean_dec(v_a_1346_);
lean_dec_ref(v_node_1342_);
lean_dec_ref(v_ctx_1341_);
lean_dec_ref(v_snap_1340_);
v_a_1530_ = lean_ctor_get(v___y_1528_, 0);
v_isSharedCheck_1537_ = !lean_is_exclusive(v___y_1528_);
if (v_isSharedCheck_1537_ == 0)
{
v___x_1532_ = v___y_1528_;
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_a_1530_);
lean_dec(v___y_1528_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___x_1535_; 
if (v_isShared_1533_ == 0)
{
v___x_1535_ = v___x_1532_;
goto v_reusejp_1534_;
}
else
{
lean_object* v_reuseFailAlloc_1536_; 
v_reuseFailAlloc_1536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1536_, 0, v_a_1530_);
v___x_1535_ = v_reuseFailAlloc_1536_;
goto v_reusejp_1534_;
}
v_reusejp_1534_:
{
return v___x_1535_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_matchExpand___boxed(lean_object* v_CodeActionParams_1549_, lean_object* v_snap_1550_, lean_object* v_ctx_1551_, lean_object* v_node_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_){
_start:
{
lean_object* v_res_1555_; 
v_res_1555_ = lp_batteries_Batteries_CodeAction_matchExpand(v_CodeActionParams_1549_, v_snap_1550_, v_ctx_1551_, v_node_1552_, v_a_1553_);
lean_dec_ref(v_a_1553_);
lean_dec_ref(v_CodeActionParams_1549_);
return v_res_1555_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4(lean_object* v_node_1556_, lean_object* v_ctx_1557_, lean_object* v_snap_1558_, lean_object* v_as_1559_, size_t v_sz_1560_, size_t v_i_1561_, lean_object* v_b_1562_, lean_object* v___y_1563_){
_start:
{
lean_object* v___x_1565_; 
v___x_1565_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___redArg(v_node_1556_, v_ctx_1557_, v_snap_1558_, v_as_1559_, v_sz_1560_, v_i_1561_, v_b_1562_);
return v___x_1565_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4___boxed(lean_object* v_node_1566_, lean_object* v_ctx_1567_, lean_object* v_snap_1568_, lean_object* v_as_1569_, lean_object* v_sz_1570_, lean_object* v_i_1571_, lean_object* v_b_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_){
_start:
{
size_t v_sz_boxed_1575_; size_t v_i_boxed_1576_; lean_object* v_res_1577_; 
v_sz_boxed_1575_ = lean_unbox_usize(v_sz_1570_);
lean_dec(v_sz_1570_);
v_i_boxed_1576_ = lean_unbox_usize(v_i_1571_);
lean_dec(v_i_1571_);
v_res_1577_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_matchExpand_spec__4(v_node_1566_, v_ctx_1567_, v_snap_1568_, v_as_1569_, v_sz_boxed_1575_, v_i_boxed_1576_, v_b_1572_, v___y_1573_);
lean_dec_ref(v___y_1573_);
lean_dec_ref(v_as_1569_);
lean_dec_ref(v_snap_1568_);
return v_res_1577_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8(lean_object* v_a_1578_, lean_object* v_snap_1579_, uint8_t v_explicitArgsOnly_1580_, lean_object* v___x_1581_, lean_object* v___x_1582_, lean_object* v_range_1583_, lean_object* v_b_1584_, lean_object* v_i_1585_, lean_object* v_hs_1586_, lean_object* v_hl_1587_){
_start:
{
lean_object* v___x_1589_; 
v___x_1589_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___redArg(v_a_1578_, v_snap_1579_, v_explicitArgsOnly_1580_, v___x_1581_, v___x_1582_, v_range_1583_, v_b_1584_, v_i_1585_);
return v___x_1589_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8___boxed(lean_object* v_a_1590_, lean_object* v_snap_1591_, lean_object* v_explicitArgsOnly_1592_, lean_object* v___x_1593_, lean_object* v___x_1594_, lean_object* v_range_1595_, lean_object* v_b_1596_, lean_object* v_i_1597_, lean_object* v_hs_1598_, lean_object* v_hl_1599_, lean_object* v___y_1600_){
_start:
{
uint8_t v_explicitArgsOnly_boxed_1601_; lean_object* v_res_1602_; 
v_explicitArgsOnly_boxed_1601_ = lean_unbox(v_explicitArgsOnly_1592_);
v_res_1602_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__8(v_a_1590_, v_snap_1591_, v_explicitArgsOnly_boxed_1601_, v___x_1593_, v___x_1594_, v_range_1595_, v_b_1596_, v_i_1597_, v_hs_1598_, v_hl_1599_);
lean_dec_ref(v_range_1595_);
lean_dec(v___x_1594_);
lean_dec(v___x_1593_);
lean_dec_ref(v_snap_1591_);
lean_dec(v_a_1590_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9(lean_object* v___x_1603_, lean_object* v_snap_1604_, uint8_t v_explicitArgsOnly_1605_, lean_object* v___x_1606_, lean_object* v_as_1607_, lean_object* v_as_x27_1608_, lean_object* v_b_1609_, lean_object* v_a_1610_){
_start:
{
lean_object* v___x_1612_; 
v___x_1612_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___redArg(v___x_1603_, v_snap_1604_, v_explicitArgsOnly_1605_, v___x_1606_, v_as_x27_1608_, v_b_1609_);
return v___x_1612_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9___boxed(lean_object* v___x_1613_, lean_object* v_snap_1614_, lean_object* v_explicitArgsOnly_1615_, lean_object* v___x_1616_, lean_object* v_as_1617_, lean_object* v_as_x27_1618_, lean_object* v_b_1619_, lean_object* v_a_1620_, lean_object* v___y_1621_){
_start:
{
uint8_t v_explicitArgsOnly_boxed_1622_; lean_object* v_res_1623_; 
v_explicitArgsOnly_boxed_1622_ = lean_unbox(v_explicitArgsOnly_1615_);
v_res_1623_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_matchExpand_spec__9(v___x_1613_, v_snap_1614_, v_explicitArgsOnly_boxed_1622_, v___x_1616_, v_as_1617_, v_as_x27_1618_, v_b_1619_, v_a_1620_);
lean_dec(v_as_x27_1618_);
lean_dec(v_as_1617_);
lean_dec(v___x_1616_);
lean_dec_ref(v_snap_1614_);
lean_dec_ref(v___x_1613_);
return v_res_1623_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12(lean_object* v_a_1624_, lean_object* v_CodeActionParams_1625_, lean_object* v_as_1626_, size_t v_i_1627_, size_t v_stop_1628_, lean_object* v_b_1629_, lean_object* v___y_1630_){
_start:
{
lean_object* v___x_1632_; 
v___x_1632_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___redArg(v_a_1624_, v_CodeActionParams_1625_, v_as_1626_, v_i_1627_, v_stop_1628_, v_b_1629_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12___boxed(lean_object* v_a_1633_, lean_object* v_CodeActionParams_1634_, lean_object* v_as_1635_, lean_object* v_i_1636_, lean_object* v_stop_1637_, lean_object* v_b_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_){
_start:
{
size_t v_i_boxed_1641_; size_t v_stop_boxed_1642_; lean_object* v_res_1643_; 
v_i_boxed_1641_ = lean_unbox_usize(v_i_1636_);
lean_dec(v_i_1636_);
v_stop_boxed_1642_ = lean_unbox_usize(v_stop_1637_);
lean_dec(v_stop_1637_);
v_res_1643_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_matchExpand_spec__12(v_a_1633_, v_CodeActionParams_1634_, v_as_1635_, v_i_boxed_1641_, v_stop_boxed_1642_, v_b_1638_, v___y_1639_);
lean_dec_ref(v___y_1639_);
lean_dec_ref(v_as_1635_);
lean_dec_ref(v_CodeActionParams_1634_);
return v_res_1643_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_CodeAction_Match(uint8_t builtin) {
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
lean_object* runtime_initialize_batteries_Batteries_CodeAction_Misc(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_CodeAction_Match(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction_Misc(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_CodeAction_Match(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_CodeAction_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_CodeAction_Match(builtin);
}
#ifdef __cplusplus
}
#endif
