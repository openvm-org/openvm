// Lean compiler output
// Module: ProofWidgets.Component.MakeEditLink
// Imports: public import Init public meta import Init public import ProofWidgets.Component.Basic
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
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instFromJsonRange_fromJson(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_String_Slice_positions(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_String_instInhabitedSlice;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t lean_string_is_valid_pos(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Json_getStr_x3f(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instToJsonTextDocumentEdit_toJson(lean_object*);
lean_object* l_Lean_Lsp_instToJsonRange_toJson(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Lsp_instFromJsonTextDocumentEdit_fromJson(lean_object*);
uint64_t lean_string_hash(lean_object*);
LEAN_EXPORT uint32_t lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size(uint32_t);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_panic___at___00Lean_Lsp_Position_advance_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_Lean_Lsp_Position_advance___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Lsp_Position_advance___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_Lsp_Position_advance___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Lsp_Position_advance___closed__1_value;
static const lean_string_object lp_proofwidgets_Lean_Lsp_Position_advance___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___closed__2 = (const lean_object*)&lp_proofwidgets_Lean_Lsp_Position_advance___closed__2_value;
static const lean_string_object lp_proofwidgets_Lean_Lsp_Position_advance___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___closed__3 = (const lean_object*)&lp_proofwidgets_Lean_Lsp_Position_advance___closed__3_value;
static lean_once_cell_t lp_proofwidgets_Lean_Lsp_Position_advance___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Lsp_Position_advance(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "edit"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "MakeEditLinkProps"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__1_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__2_value),LEAN_SCALAR_PTR_LITERAL(76, 180, 236, 135, 199, 150, 215, 203)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__5_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0_value),LEAN_SCALAR_PTR_LITERAL(195, 25, 214, 57, 9, 124, 16, 157)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__7_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "newSelection"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__12_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "newSelection\?"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__13 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__13_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__13_value),LEAN_SCALAR_PTR_LITERAL(216, 135, 251, 178, 96, 7, 117, 43)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__14 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__14_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "title"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__18 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__18_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "title\?"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__19 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__19_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__19_value),LEAN_SCALAR_PTR_LITERAL(90, 250, 148, 85, 102, 238, 50, 79)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__20 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__20_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__2(lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_MakeEditLink___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 428, .m_capacity = 428, .m_length = 427, .m_data = "window;import{jsx as e}from\"react/jsx-runtime\";import*as t from\"react\";import{EditorContext as i}from\"@leanprover/infoview\";function n(n){const o=t.useContext(i);return e(\"a\",{className:\"link pointer dim \",title:n.title\?\?\"\",onClick:async()=>{await o.api.applyEdit({documentChanges:[n.edit]}),n.newSelection&&await o.revealLocation({uri:n.edit.textDocument.uri,range:n.newSelection})},children:n.children})}export{n as default};"};
static const lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_MakeEditLink___closed__0_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2;
static const lean_string_object lp_proofwidgets_ProofWidgets_MakeEditLink___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_MakeEditLink___closed__3_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink;
LEAN_EXPORT uint32_t lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size(uint32_t v_c_1_){
_start:
{
uint32_t v___x_2_; uint8_t v___x_3_; 
v___x_2_ = 65535;
v___x_3_ = lean_uint32_dec_le(v_c_1_, v___x_2_);
if (v___x_3_ == 0)
{
uint32_t v___x_4_; 
v___x_4_ = 2;
return v___x_4_;
}
else
{
uint32_t v___x_5_; 
v___x_5_ = 1;
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size___boxed(lean_object* v_c_6_){
_start:
{
uint32_t v_c_boxed_7_; uint32_t v_res_8_; lean_object* v_r_9_; 
v_c_boxed_7_ = lean_unbox_uint32(v_c_6_);
lean_dec(v_c_6_);
v_res_8_ = lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size(v_c_boxed_7_);
v_r_9_ = lean_box_uint32(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_panic___at___00Lean_Lsp_Position_advance_spec__1(lean_object* v_msg_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = l_String_instInhabitedSlice;
v___x_12_ = lean_panic_fn_borrowed(v___x_11_, v_msg_10_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg(lean_object* v___y_13_, lean_object* v_a_14_, lean_object* v_b_15_){
_start:
{
lean_object* v_str_16_; lean_object* v_startInclusive_17_; lean_object* v_endExclusive_18_; lean_object* v___x_19_; uint8_t v___x_20_; 
v_str_16_ = lean_ctor_get(v___y_13_, 0);
v_startInclusive_17_ = lean_ctor_get(v___y_13_, 1);
v_endExclusive_18_ = lean_ctor_get(v___y_13_, 2);
v___x_19_ = lean_nat_sub(v_endExclusive_18_, v_startInclusive_17_);
v___x_20_ = lean_nat_dec_eq(v_a_14_, v___x_19_);
lean_dec(v___x_19_);
if (v___x_20_ == 0)
{
lean_object* v_fst_21_; lean_object* v_snd_22_; lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_46_; 
v_fst_21_ = lean_ctor_get(v_b_15_, 0);
v_snd_22_ = lean_ctor_get(v_b_15_, 1);
v_isSharedCheck_46_ = !lean_is_exclusive(v_b_15_);
if (v_isSharedCheck_46_ == 0)
{
v___x_24_ = v_b_15_;
v_isShared_25_ = v_isSharedCheck_46_;
goto v_resetjp_23_;
}
else
{
lean_inc(v_snd_22_);
lean_inc(v_fst_21_);
lean_dec(v_b_15_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_46_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_26_; uint32_t v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; uint32_t v___x_30_; uint8_t v___x_31_; 
v___x_26_ = lean_nat_add(v_startInclusive_17_, v_a_14_);
lean_dec(v_a_14_);
v___x_27_ = lean_string_utf8_get_fast(v_str_16_, v___x_26_);
v___x_28_ = lean_string_utf8_next_fast(v_str_16_, v___x_26_);
lean_dec(v___x_26_);
v___x_29_ = lean_nat_sub(v___x_28_, v_startInclusive_17_);
v___x_30_ = 10;
v___x_31_ = lean_uint32_dec_eq(v___x_27_, v___x_30_);
if (v___x_31_ == 0)
{
uint32_t v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_36_; 
v___x_32_ = lp_proofwidgets___private_ProofWidgets_Component_MakeEditLink_0__ProofWidgets_Internal_utf16Size(v___x_27_);
v___x_33_ = lean_uint32_to_nat(v___x_32_);
v___x_34_ = lean_nat_add(v_snd_22_, v___x_33_);
lean_dec(v___x_33_);
lean_dec(v_snd_22_);
if (v_isShared_25_ == 0)
{
lean_ctor_set(v___x_24_, 1, v___x_34_);
v___x_36_ = v___x_24_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v_fst_21_);
lean_ctor_set(v_reuseFailAlloc_38_, 1, v___x_34_);
v___x_36_ = v_reuseFailAlloc_38_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
v_a_14_ = v___x_29_;
v_b_15_ = v___x_36_;
goto _start;
}
}
else
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_43_; 
lean_dec(v_snd_22_);
v___x_39_ = lean_unsigned_to_nat(1u);
v___x_40_ = lean_nat_add(v_fst_21_, v___x_39_);
lean_dec(v_fst_21_);
v___x_41_ = lean_unsigned_to_nat(0u);
if (v_isShared_25_ == 0)
{
lean_ctor_set(v___x_24_, 1, v___x_41_);
lean_ctor_set(v___x_24_, 0, v___x_40_);
v___x_43_ = v___x_24_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v___x_40_);
lean_ctor_set(v_reuseFailAlloc_45_, 1, v___x_41_);
v___x_43_ = v_reuseFailAlloc_45_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
v_a_14_ = v___x_29_;
v_b_15_ = v___x_43_;
goto _start;
}
}
}
}
else
{
lean_dec(v_a_14_);
return v_b_15_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg___boxed(lean_object* v___y_47_, lean_object* v_a_48_, lean_object* v_b_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg(v___y_47_, v_a_48_, v_b_49_);
lean_dec_ref(v___y_47_);
return v_res_50_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Lsp_Position_advance___closed__4(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_56_ = ((lean_object*)(lp_proofwidgets_Lean_Lsp_Position_advance___closed__3));
v___x_57_ = lean_unsigned_to_nat(14u);
v___x_58_ = lean_unsigned_to_nat(22u);
v___x_59_ = ((lean_object*)(lp_proofwidgets_Lean_Lsp_Position_advance___closed__2));
v___x_60_ = ((lean_object*)(lp_proofwidgets_Lean_Lsp_Position_advance___closed__1));
v___x_61_ = l_mkPanicMessageWithDecl(v___x_60_, v___x_59_, v___x_58_, v___x_57_, v___x_56_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Lsp_Position_advance(lean_object* v_p_62_, lean_object* v_s_63_){
_start:
{
lean_object* v___y_65_; lean_object* v___y_66_; lean_object* v___y_67_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___y_73_; lean_object* v_str_85_; lean_object* v_startPos_86_; lean_object* v_stopPos_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_97_; 
v___x_70_ = lean_unsigned_to_nat(0u);
v___x_71_ = ((lean_object*)(lp_proofwidgets_Lean_Lsp_Position_advance___closed__0));
v_str_85_ = lean_ctor_get(v_s_63_, 0);
v_startPos_86_ = lean_ctor_get(v_s_63_, 1);
v_stopPos_87_ = lean_ctor_get(v_s_63_, 2);
v_isSharedCheck_97_ = !lean_is_exclusive(v_s_63_);
if (v_isSharedCheck_97_ == 0)
{
v___x_89_ = v_s_63_;
v_isShared_90_ = v_isSharedCheck_97_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_stopPos_87_);
lean_inc(v_startPos_86_);
lean_inc(v_str_85_);
lean_dec(v_s_63_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_97_;
goto v_resetjp_88_;
}
v___jp_64_:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_nat_add(v___y_67_, v___y_65_);
lean_dec(v___y_65_);
v___x_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_69_, 0, v___y_66_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
return v___x_69_;
}
v___jp_72_:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v_fst_76_; lean_object* v_snd_77_; lean_object* v_line_78_; lean_object* v_character_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_74_ = l_String_Slice_positions(v___y_73_);
v___x_75_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg(v___y_73_, v___x_74_, v___x_71_);
lean_dec_ref(v___y_73_);
v_fst_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_fst_76_);
v_snd_77_ = lean_ctor_get(v___x_75_, 1);
lean_inc(v_snd_77_);
lean_dec_ref(v___x_75_);
v_line_78_ = lean_ctor_get(v_p_62_, 0);
v_character_79_ = lean_ctor_get(v_p_62_, 1);
v___x_80_ = lean_nat_add(v_line_78_, v_fst_76_);
v___x_81_ = lean_nat_dec_eq(v_fst_76_, v___x_70_);
lean_dec(v_fst_76_);
if (v___x_81_ == 0)
{
v___y_65_ = v_snd_77_;
v___y_66_ = v___x_80_;
v___y_67_ = v___x_70_;
goto v___jp_64_;
}
else
{
v___y_65_ = v_snd_77_;
v___y_66_ = v___x_80_;
v___y_67_ = v_character_79_;
goto v___jp_64_;
}
}
v___jp_82_:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lean_obj_once(&lp_proofwidgets_Lean_Lsp_Position_advance___closed__4, &lp_proofwidgets_Lean_Lsp_Position_advance___closed__4_once, _init_lp_proofwidgets_Lean_Lsp_Position_advance___closed__4);
v___x_84_ = lp_proofwidgets_panic___at___00Lean_Lsp_Position_advance_spec__1(v___x_83_);
v___y_73_ = v___x_84_;
goto v___jp_72_;
}
v_resetjp_88_:
{
uint8_t v___x_91_; 
v___x_91_ = lean_string_is_valid_pos(v_str_85_, v_startPos_86_);
if (v___x_91_ == 0)
{
lean_del_object(v___x_89_);
lean_dec(v_stopPos_87_);
lean_dec(v_startPos_86_);
lean_dec_ref(v_str_85_);
goto v___jp_82_;
}
else
{
uint8_t v___x_92_; 
v___x_92_ = lean_string_is_valid_pos(v_str_85_, v_stopPos_87_);
if (v___x_92_ == 0)
{
lean_del_object(v___x_89_);
lean_dec(v_stopPos_87_);
lean_dec(v_startPos_86_);
lean_dec_ref(v_str_85_);
goto v___jp_82_;
}
else
{
uint8_t v___x_93_; 
v___x_93_ = lean_nat_dec_le(v_startPos_86_, v_stopPos_87_);
if (v___x_93_ == 0)
{
lean_del_object(v___x_89_);
lean_dec(v_stopPos_87_);
lean_dec(v_startPos_86_);
lean_dec_ref(v_str_85_);
goto v___jp_82_;
}
else
{
lean_object* v___x_95_; 
if (v_isShared_90_ == 0)
{
v___x_95_ = v___x_89_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_str_85_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v_startPos_86_);
lean_ctor_set(v_reuseFailAlloc_96_, 2, v_stopPos_87_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
v___y_73_ = v___x_95_;
goto v___jp_72_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Lsp_Position_advance___boxed(lean_object* v_p_98_, lean_object* v_s_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_proofwidgets_Lean_Lsp_Position_advance(v_p_98_, v_s_99_);
lean_dec_ref(v_p_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0(lean_object* v___y_101_, lean_object* v_inst_102_, lean_object* v_R_103_, lean_object* v_a_104_, lean_object* v_b_105_, lean_object* v_c_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___redArg(v___y_101_, v_a_104_, v_b_105_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0___boxed(lean_object* v___y_108_, lean_object* v_inst_109_, lean_object* v_R_110_, lean_object* v_a_111_, lean_object* v_b_112_, lean_object* v_c_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00Lean_Lsp_Position_advance_spec__0(v___y_108_, v_inst_109_, v_R_110_, v_a_111_, v_b_112_, v_c_113_);
lean_dec_ref(v___y_108_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0(lean_object* v_j_115_, lean_object* v_k_116_){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_117_ = l_Lean_Json_getObjValD(v_j_115_, v_k_116_);
v___x_118_ = l_Lean_Lsp_instFromJsonTextDocumentEdit_fromJson(v___x_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0___boxed(lean_object* v_j_119_, lean_object* v_k_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0(v_j_119_, v_k_120_);
lean_dec_ref(v_k_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3(lean_object* v_x_124_){
_start:
{
if (lean_obj_tag(v_x_124_) == 0)
{
lean_object* v___x_125_; 
v___x_125_ = ((lean_object*)(lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3___closed__0));
return v___x_125_;
}
else
{
lean_object* v___x_126_; 
v___x_126_ = l_Lean_Json_getStr_x3f(v_x_124_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_134_; 
v_a_127_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_134_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_134_ == 0)
{
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_134_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_134_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_132_; 
if (v_isShared_130_ == 0)
{
v___x_132_ = v___x_129_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v_a_127_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
else
{
lean_object* v_a_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_143_; 
v_a_135_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_143_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_143_ == 0)
{
v___x_137_ = v___x_126_;
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_a_135_);
lean_dec(v___x_126_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_139_; lean_object* v___x_141_; 
v___x_139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_139_, 0, v_a_135_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 0, v___x_139_);
v___x_141_ = v___x_137_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_139_);
v___x_141_ = v_reuseFailAlloc_142_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
return v___x_141_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2(lean_object* v_j_144_, lean_object* v_k_145_){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = l_Lean_Json_getObjValD(v_j_144_, v_k_145_);
v___x_147_ = lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2_spec__3(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2___boxed(lean_object* v_j_148_, lean_object* v_k_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2(v_j_148_, v_k_149_);
lean_dec_ref(v_k_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1(lean_object* v_x_153_){
_start:
{
if (lean_obj_tag(v_x_153_) == 0)
{
lean_object* v___x_154_; 
v___x_154_ = ((lean_object*)(lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1___closed__0));
return v___x_154_;
}
else
{
lean_object* v___x_155_; 
v___x_155_ = l_Lean_Lsp_instFromJsonRange_fromJson(v_x_153_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
v_a_156_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_155_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_155_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
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
lean_object* v_a_164_; lean_object* v___x_166_; uint8_t v_isShared_167_; uint8_t v_isSharedCheck_172_; 
v_a_164_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_172_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_172_ == 0)
{
v___x_166_ = v___x_155_;
v_isShared_167_ = v_isSharedCheck_172_;
goto v_resetjp_165_;
}
else
{
lean_inc(v_a_164_);
lean_dec(v___x_155_);
v___x_166_ = lean_box(0);
v_isShared_167_ = v_isSharedCheck_172_;
goto v_resetjp_165_;
}
v_resetjp_165_:
{
lean_object* v___x_168_; lean_object* v___x_170_; 
v___x_168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_168_, 0, v_a_164_);
if (v_isShared_167_ == 0)
{
lean_ctor_set(v___x_166_, 0, v___x_168_);
v___x_170_ = v___x_166_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v___x_168_);
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
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1(lean_object* v_j_173_, lean_object* v_k_174_){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_175_ = l_Lean_Json_getObjValD(v_j_173_, v_k_174_);
v___x_176_ = lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1_spec__1(v___x_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1___boxed(lean_object* v_j_177_, lean_object* v_k_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1(v_j_177_, v_k_178_);
lean_dec_ref(v_k_178_);
return v_res_179_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4(void){
_start:
{
uint8_t v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_186_ = 1;
v___x_187_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__3));
v___x_188_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_187_, v___x_186_);
return v___x_188_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_190_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__5));
v___x_191_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__4);
v___x_192_ = lean_string_append(v___x_191_, v___x_190_);
return v___x_192_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8(void){
_start:
{
uint8_t v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_195_ = 1;
v___x_196_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__7));
v___x_197_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_196_, v___x_195_);
return v___x_197_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_198_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__8);
v___x_199_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6);
v___x_200_ = lean_string_append(v___x_199_, v___x_198_);
return v___x_200_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_202_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10));
v___x_203_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__9);
v___x_204_ = lean_string_append(v___x_203_, v___x_202_);
return v___x_204_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15(void){
_start:
{
uint8_t v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = 1;
v___x_210_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__14));
v___x_211_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_210_, v___x_209_);
return v___x_211_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__15);
v___x_213_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6);
v___x_214_ = lean_string_append(v___x_213_, v___x_212_);
return v___x_214_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_215_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10));
v___x_216_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__16);
v___x_217_ = lean_string_append(v___x_216_, v___x_215_);
return v___x_217_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21(void){
_start:
{
uint8_t v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_222_ = 1;
v___x_223_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__20));
v___x_224_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_223_, v___x_222_);
return v___x_224_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_225_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__21);
v___x_226_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__6);
v___x_227_ = lean_string_append(v___x_226_, v___x_225_);
return v___x_227_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23(void){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_228_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__10));
v___x_229_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__22);
v___x_230_ = lean_string_append(v___x_229_, v___x_228_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson(lean_object* v_json_231_){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0));
lean_inc(v_json_231_);
v___x_233_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__0(v_json_231_, v___x_232_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_243_; 
lean_dec(v_json_231_);
v_a_234_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_243_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_243_ == 0)
{
v___x_236_ = v___x_233_;
v_isShared_237_ = v_isSharedCheck_243_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_a_234_);
lean_dec(v___x_233_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_243_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_241_; 
v___x_238_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__11);
v___x_239_ = lean_string_append(v___x_238_, v_a_234_);
lean_dec(v_a_234_);
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 0, v___x_239_);
v___x_241_ = v___x_236_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
else
{
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_251_; 
lean_dec(v_json_231_);
v_a_244_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_251_ == 0)
{
v___x_246_ = v___x_233_;
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v___x_233_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_249_; 
if (v_isShared_247_ == 0)
{
lean_ctor_set_tag(v___x_246_, 0);
v___x_249_ = v___x_246_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_a_244_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v_a_252_ = lean_ctor_get(v___x_233_, 0);
lean_inc(v_a_252_);
lean_dec_ref_known(v___x_233_, 1);
v___x_253_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__12));
lean_inc(v_json_231_);
v___x_254_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__1(v_json_231_, v___x_253_);
if (lean_obj_tag(v___x_254_) == 0)
{
lean_object* v_a_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_264_; 
lean_dec(v_a_252_);
lean_dec(v_json_231_);
v_a_255_ = lean_ctor_get(v___x_254_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_254_);
if (v_isSharedCheck_264_ == 0)
{
v___x_257_ = v___x_254_;
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_a_255_);
lean_dec(v___x_254_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_262_; 
v___x_259_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__17);
v___x_260_ = lean_string_append(v___x_259_, v_a_255_);
lean_dec(v_a_255_);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v___x_260_);
v___x_262_ = v___x_257_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v___x_260_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
else
{
if (lean_obj_tag(v___x_254_) == 0)
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
lean_dec(v_a_252_);
lean_dec(v_json_231_);
v_a_265_ = lean_ctor_get(v___x_254_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_254_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_254_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_254_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set_tag(v___x_267_, 0);
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_a_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
else
{
lean_object* v_a_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v_a_273_ = lean_ctor_get(v___x_254_, 0);
lean_inc(v_a_273_);
lean_dec_ref_known(v___x_254_, 1);
v___x_274_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__18));
v___x_275_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonMakeEditLinkProps_fromJson_spec__2(v_json_231_, v___x_274_);
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v_a_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_285_; 
lean_dec(v_a_273_);
lean_dec(v_a_252_);
v_a_276_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_285_ == 0)
{
v___x_278_ = v___x_275_;
v_isShared_279_ = v_isSharedCheck_285_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_a_276_);
lean_dec(v___x_275_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_285_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_283_; 
v___x_280_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23, &lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23_once, _init_lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__23);
v___x_281_ = lean_string_append(v___x_280_, v_a_276_);
lean_dec(v_a_276_);
if (v_isShared_279_ == 0)
{
lean_ctor_set(v___x_278_, 0, v___x_281_);
v___x_283_ = v___x_278_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v___x_281_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
else
{
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_dec(v_a_273_);
lean_dec(v_a_252_);
v_a_286_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_275_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_275_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 0);
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
else
{
lean_object* v_a_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_302_; 
v_a_294_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_302_ == 0)
{
v___x_296_ = v___x_275_;
v_isShared_297_ = v_isSharedCheck_302_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_a_294_);
lean_dec(v___x_275_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_302_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_298_; lean_object* v___x_300_; 
v___x_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_298_, 0, v_a_252_);
lean_ctor_set(v___x_298_, 1, v_a_273_);
lean_ctor_set(v___x_298_, 2, v_a_294_);
if (v_isShared_297_ == 0)
{
lean_ctor_set(v___x_296_, 0, v___x_298_);
v___x_300_ = v___x_296_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v___x_298_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__0(lean_object* v_k_305_, lean_object* v_x_306_){
_start:
{
if (lean_obj_tag(v_x_306_) == 0)
{
lean_object* v___x_307_; 
lean_dec_ref(v_k_305_);
v___x_307_ = lean_box(0);
return v___x_307_;
}
else
{
lean_object* v_val_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v_val_308_ = lean_ctor_get(v_x_306_, 0);
lean_inc(v_val_308_);
lean_dec_ref_known(v_x_306_, 1);
v___x_309_ = l_Lean_Lsp_instToJsonRange_toJson(v_val_308_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v_k_305_);
lean_ctor_set(v___x_310_, 1, v___x_309_);
v___x_311_ = lean_box(0);
v___x_312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_310_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
return v___x_312_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__1(lean_object* v_k_313_, lean_object* v_x_314_){
_start:
{
if (lean_obj_tag(v_x_314_) == 0)
{
lean_object* v___x_315_; 
lean_dec_ref(v_k_313_);
v___x_315_ = lean_box(0);
return v___x_315_;
}
else
{
lean_object* v_val_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_326_; 
v_val_316_ = lean_ctor_get(v_x_314_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v_x_314_);
if (v_isSharedCheck_326_ == 0)
{
v___x_318_ = v_x_314_;
v_isShared_319_ = v_isSharedCheck_326_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_val_316_);
lean_dec(v_x_314_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_326_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_321_; 
if (v_isShared_319_ == 0)
{
lean_ctor_set_tag(v___x_318_, 3);
v___x_321_ = v___x_318_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_val_316_);
v___x_321_ = v_reuseFailAlloc_325_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
v___x_322_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_322_, 0, v_k_313_);
lean_ctor_set(v___x_322_, 1, v___x_321_);
v___x_323_ = lean_box(0);
v___x_324_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_322_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
return v___x_324_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__2(lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
if (lean_obj_tag(v_a_327_) == 0)
{
lean_object* v___x_329_; 
v___x_329_ = lean_array_to_list(v_a_328_);
return v___x_329_;
}
else
{
lean_object* v_head_330_; lean_object* v_tail_331_; lean_object* v___x_332_; 
v_head_330_ = lean_ctor_get(v_a_327_, 0);
lean_inc(v_head_330_);
v_tail_331_ = lean_ctor_get(v_a_327_, 1);
lean_inc(v_tail_331_);
lean_dec_ref_known(v_a_327_, 2);
v___x_332_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_328_, v_head_330_);
v_a_327_ = v_tail_331_;
v_a_328_ = v___x_332_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object* v_x_336_){
_start:
{
lean_object* v_edit_337_; lean_object* v_newSelection_x3f_338_; lean_object* v_title_x3f_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v_edit_337_ = lean_ctor_get(v_x_336_, 0);
lean_inc_ref(v_edit_337_);
v_newSelection_x3f_338_ = lean_ctor_get(v_x_336_, 1);
lean_inc(v_newSelection_x3f_338_);
v_title_x3f_339_ = lean_ctor_get(v_x_336_, 2);
lean_inc(v_title_x3f_339_);
lean_dec_ref(v_x_336_);
v___x_340_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__0));
v___x_341_ = l_Lean_Lsp_instToJsonTextDocumentEdit_toJson(v_edit_337_);
v___x_342_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_340_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = lean_box(0);
v___x_344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_342_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__12));
v___x_346_ = lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__0(v___x_345_, v_newSelection_x3f_338_);
v___x_347_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson___closed__18));
v___x_348_ = lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__1(v___x_347_, v_title_x3f_339_);
v___x_349_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
lean_ctor_set(v___x_349_, 1, v___x_343_);
v___x_350_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_346_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_344_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
v___x_352_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson___closed__0));
v___x_353_ = lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonMakeEditLinkProps_toJson_spec__2(v___x_351_, v___x_352_);
v___x_354_ = l_Lean_Json_mkObj(v___x_353_);
lean_dec(v___x_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27(lean_object* v_doc_357_, lean_object* v_range_358_, lean_object* v_newText_359_, lean_object* v_newSelection_x3f_360_){
_start:
{
lean_object* v_uri_361_; lean_object* v_version_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v_edit_370_; 
v_uri_361_ = lean_ctor_get(v_doc_357_, 0);
v_version_362_ = lean_ctor_get(v_doc_357_, 2);
lean_inc(v_version_362_);
v___x_363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_363_, 0, v_version_362_);
lean_inc_ref(v_uri_361_);
v___x_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_364_, 0, v_uri_361_);
lean_ctor_set(v___x_364_, 1, v___x_363_);
v___x_365_ = lean_box(0);
lean_inc_ref(v_newText_359_);
lean_inc_ref(v_range_358_);
v___x_366_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_366_, 0, v_range_358_);
lean_ctor_set(v___x_366_, 1, v_newText_359_);
lean_ctor_set(v___x_366_, 2, v___x_365_);
lean_ctor_set(v___x_366_, 3, v___x_365_);
v___x_367_ = lean_unsigned_to_nat(1u);
v___x_368_ = lean_mk_empty_array_with_capacity(v___x_367_);
v___x_369_ = lean_array_push(v___x_368_, v___x_366_);
v_edit_370_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_edit_370_, 0, v___x_364_);
lean_ctor_set(v_edit_370_, 1, v___x_369_);
if (lean_obj_tag(v_newSelection_x3f_360_) == 0)
{
lean_object* v_start_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_384_; 
v_start_371_ = lean_ctor_get(v_range_358_, 0);
v_isSharedCheck_384_ = !lean_is_exclusive(v_range_358_);
if (v_isSharedCheck_384_ == 0)
{
lean_object* v_unused_385_; 
v_unused_385_ = lean_ctor_get(v_range_358_, 1);
lean_dec(v_unused_385_);
v___x_373_ = v_range_358_;
v_isShared_374_ = v_isSharedCheck_384_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_start_371_);
lean_dec(v_range_358_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_384_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v_endPos_378_; lean_object* v___x_380_; 
v___x_375_ = lean_unsigned_to_nat(0u);
v___x_376_ = lean_string_utf8_byte_size(v_newText_359_);
v___x_377_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_377_, 0, v_newText_359_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_376_);
v_endPos_378_ = lp_proofwidgets_Lean_Lsp_Position_advance(v_start_371_, v___x_377_);
lean_dec_ref(v_start_371_);
lean_inc_ref(v_endPos_378_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 1, v_endPos_378_);
lean_ctor_set(v___x_373_, 0, v_endPos_378_);
v___x_380_ = v___x_373_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_383_; 
v_reuseFailAlloc_383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_383_, 0, v_endPos_378_);
lean_ctor_set(v_reuseFailAlloc_383_, 1, v_endPos_378_);
v___x_380_ = v_reuseFailAlloc_383_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
v___x_382_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_382_, 0, v_edit_370_);
lean_ctor_set(v___x_382_, 1, v___x_381_);
lean_ctor_set(v___x_382_, 2, v___x_365_);
return v___x_382_;
}
}
}
else
{
lean_object* v___x_386_; 
lean_dec_ref(v_newText_359_);
lean_dec_ref(v_range_358_);
v___x_386_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_386_, 0, v_edit_370_);
lean_ctor_set(v___x_386_, 1, v_newSelection_x3f_360_);
lean_ctor_set(v___x_386_, 2, v___x_365_);
return v___x_386_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27___boxed(lean_object* v_doc_387_, lean_object* v_range_388_, lean_object* v_newText_389_, lean_object* v_newSelection_x3f_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27(v_doc_387_, v_range_388_, v_newText_389_, v_newSelection_x3f_390_);
lean_dec_ref(v_doc_387_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object* v_doc_396_, lean_object* v_range_397_, lean_object* v_newText_398_, lean_object* v_newSelection_x3f_399_){
_start:
{
lean_object* v___y_401_; lean_object* v___y_402_; 
if (lean_obj_tag(v_newSelection_x3f_399_) == 0)
{
lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_407_ = lean_box(0);
v___x_408_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27(v_doc_396_, v_range_397_, v_newText_398_, v___x_407_);
return v___x_408_;
}
else
{
lean_object* v_val_409_; lean_object* v_fst_410_; lean_object* v_snd_411_; lean_object* v_start_412_; lean_object* v___x_413_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_421_; lean_object* v___y_427_; lean_object* v___x_431_; uint8_t v___x_432_; 
v_val_409_ = lean_ctor_get(v_newSelection_x3f_399_, 0);
lean_inc(v_val_409_);
lean_dec_ref_known(v_newSelection_x3f_399_, 1);
v_fst_410_ = lean_ctor_get(v_val_409_, 0);
lean_inc(v_fst_410_);
v_snd_411_ = lean_ctor_get(v_val_409_, 1);
lean_inc(v_snd_411_);
lean_dec(v_val_409_);
v_start_412_ = lean_ctor_get(v_range_397_, 0);
v___x_413_ = lean_string_utf8_byte_size(v_newText_398_);
v___x_431_ = lean_unsigned_to_nat(0u);
v___x_432_ = lean_nat_dec_le(v_fst_410_, v___x_431_);
if (v___x_432_ == 0)
{
uint8_t v___x_433_; 
v___x_433_ = lean_nat_dec_le(v___x_413_, v___x_431_);
if (v___x_433_ == 0)
{
v___y_427_ = v___x_431_;
goto v___jp_426_;
}
else
{
v___y_427_ = v___x_413_;
goto v___jp_426_;
}
}
else
{
lean_object* v___x_434_; 
v___x_434_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__1));
v___y_421_ = v___x_434_;
goto v___jp_420_;
}
v___jp_414_:
{
uint8_t v___x_417_; 
v___x_417_ = lean_nat_dec_le(v___x_413_, v_snd_411_);
if (v___x_417_ == 0)
{
lean_object* v___x_418_; 
lean_inc_ref(v_newText_398_);
v___x_418_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_418_, 0, v_newText_398_);
lean_ctor_set(v___x_418_, 1, v___y_416_);
lean_ctor_set(v___x_418_, 2, v_snd_411_);
v___y_401_ = v___y_415_;
v___y_402_ = v___x_418_;
goto v___jp_400_;
}
else
{
lean_object* v___x_419_; 
lean_dec(v_snd_411_);
lean_inc_ref(v_newText_398_);
v___x_419_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_419_, 0, v_newText_398_);
lean_ctor_set(v___x_419_, 1, v___y_416_);
lean_ctor_set(v___x_419_, 2, v___x_413_);
v___y_401_ = v___y_415_;
v___y_402_ = v___x_419_;
goto v___jp_400_;
}
}
v___jp_420_:
{
lean_object* v_ps_422_; uint8_t v___x_423_; 
v_ps_422_ = lp_proofwidgets_Lean_Lsp_Position_advance(v_start_412_, v___y_421_);
v___x_423_ = lean_nat_dec_le(v_snd_411_, v_fst_410_);
if (v___x_423_ == 0)
{
uint8_t v___x_424_; 
v___x_424_ = lean_nat_dec_le(v___x_413_, v_fst_410_);
if (v___x_424_ == 0)
{
v___y_415_ = v_ps_422_;
v___y_416_ = v_fst_410_;
goto v___jp_414_;
}
else
{
lean_dec(v_fst_410_);
v___y_415_ = v_ps_422_;
v___y_416_ = v___x_413_;
goto v___jp_414_;
}
}
else
{
lean_object* v___x_425_; 
lean_dec(v_snd_411_);
lean_dec(v_fst_410_);
v___x_425_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___closed__1));
v___y_401_ = v_ps_422_;
v___y_402_ = v___x_425_;
goto v___jp_400_;
}
}
v___jp_426_:
{
uint8_t v___x_428_; 
v___x_428_ = lean_nat_dec_le(v___x_413_, v_fst_410_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; 
lean_inc(v_fst_410_);
lean_inc_ref(v_newText_398_);
v___x_429_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_429_, 0, v_newText_398_);
lean_ctor_set(v___x_429_, 1, v___y_427_);
lean_ctor_set(v___x_429_, 2, v_fst_410_);
v___y_421_ = v___x_429_;
goto v___jp_420_;
}
else
{
lean_object* v___x_430_; 
lean_inc_ref(v_newText_398_);
v___x_430_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_430_, 0, v_newText_398_);
lean_ctor_set(v___x_430_, 1, v___y_427_);
lean_ctor_set(v___x_430_, 2, v___x_413_);
v___y_421_ = v___x_430_;
goto v___jp_420_;
}
}
}
v___jp_400_:
{
lean_object* v_pe_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v_pe_403_ = lp_proofwidgets_Lean_Lsp_Position_advance(v___y_401_, v___y_402_);
v___x_404_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_404_, 0, v___y_401_);
lean_ctor_set(v___x_404_, 1, v_pe_403_);
v___x_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
v___x_406_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange_x27(v_doc_396_, v_range_397_, v_newText_398_, v___x_405_);
return v___x_406_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange___boxed(lean_object* v_doc_435_, lean_object* v_range_436_, lean_object* v_newText_437_, lean_object* v_newSelection_x3f_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(v_doc_435_, v_range_436_, v_newText_437_, v_newSelection_x3f_438_);
lean_dec_ref(v_doc_435_);
return v_res_439_;
}
}
static uint64_t _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1(void){
_start:
{
lean_object* v___x_441_; uint64_t v___x_442_; 
v___x_441_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_MakeEditLink___closed__0));
v___x_442_ = lean_string_hash(v___x_441_);
return v___x_442_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2(void){
_start:
{
uint64_t v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_443_ = lean_uint64_once(&lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1, &lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1_once, _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__1);
v___x_444_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_MakeEditLink___closed__0));
v___x_445_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_445_, 0, v___x_444_);
lean_ctor_set_uint64(v___x_445_, sizeof(void*)*1, v___x_443_);
return v___x_445_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_MakeEditLink___closed__3));
v___x_448_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2, &lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2_once, _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__2);
v___x_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_449_, 0, v___x_448_);
lean_ctor_set(v___x_449_, 1, v___x_447_);
return v___x_449_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_MakeEditLink(void){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4, &lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4_once, _init_lp_proofwidgets_ProofWidgets_MakeEditLink___closed__4);
return v___x_450_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_ProofWidgets_MakeEditLink = _init_lp_proofwidgets_ProofWidgets_MakeEditLink();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_MakeEditLink);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
}
#ifdef __cplusplus
}
#endif
