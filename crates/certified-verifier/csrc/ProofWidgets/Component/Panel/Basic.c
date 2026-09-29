// Lean compiler output
// Module: ProofWidgets.Component.Panel.Basic
// Imports: public import Init public meta import Init public import ProofWidgets.Component.Basic public meta import Lean.Elab.Tactic.BuiltinTactic public import Lean.Widget.Commands
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instFromJsonPosition_fromJson(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveTermGoal_dec_00___x40_Lean_Widget_InteractiveGoal_2553565095____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Widget_widgetInstanceSpec;
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Widget_elabWidgetInstanceSpec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Widget_UserWidget_0__Lean_Widget_evalWidgetInstanceUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Widget_savePanelWidgetInfo(uint64_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instToJsonPosition_toJson(lean_object*);
lean_object* l_Lean_SubExpr_instToJsonGoalsLocation_toJson(lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveTermGoal_enc_00___x40_Lean_Widget_InteractiveGoal_2553565095____hygCtx___hyg_1_(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pos"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goals"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "termGoal"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "selectedLocations"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__1(lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36____boxed(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "withPanelWidgetsTacticStx"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(49, 154, 62, 185, 185, 31, 223, 42)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "with_panel_widgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__6_value),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__9_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__10_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__12_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__15 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__15_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__15_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__16 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__16_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17;
static const lean_string_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__18 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__18_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__18_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__19 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__19_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__19_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__20 = (const lean_object*)&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__20_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx;
static lean_once_cell_t lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg();
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgets(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgets___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(lean_object* v_j_1_, lean_object* v_k_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = l_Lean_Json_getObjValD(v_j_1_, v_k_2_);
v___x_4_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4_, 0, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0___boxed(lean_object* v_j_5_, lean_object* v_k_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(v_j_5_, v_k_6_);
lean_dec_ref(v_k_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1(lean_object* v_x_10_){
_start:
{
if (lean_obj_tag(v_x_10_) == 0)
{
lean_object* v___x_11_; 
v___x_11_ = ((lean_object*)(lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1___closed__0));
return v___x_11_;
}
else
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_12_, 0, v_x_10_);
v___x_13_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1(lean_object* v_j_14_, lean_object* v_k_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = l_Lean_Json_getObjValD(v_j_14_, v_k_15_);
v___x_17_ = lp_proofwidgets_Lean_Option_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1_spec__1(v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1___boxed(lean_object* v_j_18_, lean_object* v_k_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1(v_j_18_, v_k_19_);
lean_dec_ref(v_k_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_(lean_object* v_json_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v_a_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v_a_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v_a_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_45_; 
v___x_26_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
lean_inc_n(v_json_25_, 3);
v___x_27_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(v_json_25_, v___x_26_);
v_a_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc(v_a_28_);
lean_dec_ref(v___x_27_);
v___x_29_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
v___x_30_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(v_json_25_, v___x_29_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
lean_inc(v_a_31_);
lean_dec_ref(v___x_30_);
v___x_32_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
v___x_33_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__1(v_json_25_, v___x_32_);
v_a_34_ = lean_ctor_get(v___x_33_, 0);
lean_inc(v_a_34_);
lean_dec_ref(v___x_33_);
v___x_35_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
v___x_36_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17__spec__0(v_json_25_, v___x_35_);
v_a_37_ = lean_ctor_get(v___x_36_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_45_ == 0)
{
v___x_39_ = v___x_36_;
v_isShared_40_ = v_isSharedCheck_45_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_a_37_);
lean_dec(v___x_36_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_45_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v___x_41_; lean_object* v___x_43_; 
v___x_41_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_41_, 0, v_a_28_);
lean_ctor_set(v___x_41_, 1, v_a_31_);
lean_ctor_set(v___x_41_, 2, v_a_34_);
lean_ctor_set(v___x_41_, 3, v_a_37_);
if (v_isShared_40_ == 0)
{
lean_ctor_set(v___x_39_, 0, v___x_41_);
v___x_43_ = v___x_39_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v___x_41_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0(lean_object* v_k_48_, lean_object* v_x_49_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_object* v___x_50_; 
lean_dec_ref(v_k_48_);
v___x_50_ = lean_box(0);
return v___x_50_;
}
else
{
lean_object* v_val_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v_val_51_ = lean_ctor_get(v_x_49_, 0);
lean_inc(v_val_51_);
v___x_52_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_52_, 0, v_k_48_);
lean_ctor_set(v___x_52_, 1, v_val_51_);
v___x_53_ = lean_box(0);
v___x_54_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_52_);
lean_ctor_set(v___x_54_, 1, v___x_53_);
return v___x_54_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0___boxed(lean_object* v_k_55_, lean_object* v_x_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0(v_k_55_, v_x_56_);
lean_dec(v_x_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__1(lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
if (lean_obj_tag(v_a_58_) == 0)
{
lean_object* v___x_60_; 
v___x_60_ = lean_array_to_list(v_a_59_);
return v___x_60_;
}
else
{
lean_object* v_head_61_; lean_object* v_tail_62_; lean_object* v___x_63_; 
v_head_61_ = lean_ctor_get(v_a_58_, 0);
lean_inc(v_head_61_);
v_tail_62_ = lean_ctor_get(v_a_58_, 1);
lean_inc(v_tail_62_);
lean_dec_ref_known(v_a_58_, 2);
v___x_63_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_59_, v_head_61_);
v_a_58_ = v_tail_62_;
v_a_59_ = v___x_63_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_(lean_object* v_x_67_){
_start:
{
lean_object* v_pos_68_; lean_object* v_goals_69_; lean_object* v_termGoal_x3f_70_; lean_object* v_selectedLocations_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v_pos_68_ = lean_ctor_get(v_x_67_, 0);
v_goals_69_ = lean_ctor_get(v_x_67_, 1);
v_termGoal_x3f_70_ = lean_ctor_get(v_x_67_, 2);
v_selectedLocations_71_ = lean_ctor_get(v_x_67_, 3);
v___x_72_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
lean_inc(v_pos_68_);
v___x_73_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v_pos_68_);
v___x_74_ = lean_box(0);
v___x_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_73_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
v___x_76_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
lean_inc(v_goals_69_);
v___x_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v_goals_69_);
v___x_78_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v___x_74_);
v___x_79_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
v___x_80_ = lp_proofwidgets_Lean_Json_opt___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__0(v___x_79_, v_termGoal_x3f_70_);
v___x_81_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_));
lean_inc(v_selectedLocations_71_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_selectedLocations_71_);
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___x_74_);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v___x_74_);
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_80_);
lean_ctor_set(v___x_85_, 1, v___x_84_);
v___x_86_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_78_);
lean_ctor_set(v___x_86_, 1, v___x_85_);
v___x_87_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_75_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
v___x_88_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_));
v___x_89_ = lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36__spec__1(v___x_87_, v___x_88_);
v___x_90_ = l_Lean_Json_mkObj(v___x_89_);
lean_dec(v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36____boxed(lean_object* v_x_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_(v_x_91_);
lean_dec_ref(v_x_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(size_t v_sz_95_, size_t v_i_96_, lean_object* v_bs_97_){
_start:
{
uint8_t v___x_98_; 
v___x_98_ = lean_usize_dec_lt(v_i_96_, v_sz_95_);
if (v___x_98_ == 0)
{
return v_bs_97_;
}
else
{
lean_object* v_v_99_; lean_object* v___x_100_; lean_object* v_bs_x27_101_; size_t v___x_102_; size_t v___x_103_; lean_object* v___x_104_; 
v_v_99_ = lean_array_uget(v_bs_97_, v_i_96_);
v___x_100_ = lean_unsigned_to_nat(0u);
v_bs_x27_101_ = lean_array_uset(v_bs_97_, v_i_96_, v___x_100_);
v___x_102_ = ((size_t)1ULL);
v___x_103_ = lean_usize_add(v_i_96_, v___x_102_);
v___x_104_ = lean_array_uset(v_bs_x27_101_, v_i_96_, v_v_99_);
v_i_96_ = v___x_103_;
v_bs_97_ = v___x_104_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object* v_sz_106_, lean_object* v_i_107_, lean_object* v_bs_108_){
_start:
{
size_t v_sz_boxed_109_; size_t v_i_boxed_110_; lean_object* v_res_111_; 
v_sz_boxed_109_ = lean_unbox_usize(v_sz_106_);
lean_dec(v_sz_106_);
v_i_boxed_110_ = lean_unbox_usize(v_i_107_);
lean_dec(v_i_107_);
v_res_111_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(v_sz_boxed_109_, v_i_boxed_110_, v_bs_108_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(lean_object* v_a_112_){
_start:
{
size_t v_sz_113_; size_t v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v_sz_113_ = lean_array_size(v_a_112_);
v___x_114_ = ((size_t)0ULL);
v___x_115_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(v_sz_113_, v___x_114_, v_a_112_);
v___x_116_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(size_t v_sz_117_, size_t v_i_118_, lean_object* v_bs_119_, lean_object* v___y_120_){
_start:
{
uint8_t v___x_121_; 
v___x_121_ = lean_usize_dec_lt(v_i_118_, v_sz_117_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v_bs_119_);
lean_ctor_set(v___x_122_, 1, v___y_120_);
return v___x_122_;
}
else
{
lean_object* v_v_123_; lean_object* v___x_124_; lean_object* v_fst_125_; lean_object* v_snd_126_; lean_object* v___x_127_; lean_object* v_bs_x27_128_; size_t v___x_129_; size_t v___x_130_; lean_object* v___x_131_; 
v_v_123_ = lean_array_uget_borrowed(v_bs_119_, v_i_118_);
lean_inc(v_v_123_);
v___x_124_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_123_, v___y_120_);
v_fst_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_fst_125_);
v_snd_126_ = lean_ctor_get(v___x_124_, 1);
lean_inc(v_snd_126_);
lean_dec_ref(v___x_124_);
v___x_127_ = lean_unsigned_to_nat(0u);
v_bs_x27_128_ = lean_array_uset(v_bs_119_, v_i_118_, v___x_127_);
v___x_129_ = ((size_t)1ULL);
v___x_130_ = lean_usize_add(v_i_118_, v___x_129_);
v___x_131_ = lean_array_uset(v_bs_x27_128_, v_i_118_, v_fst_125_);
v_i_118_ = v___x_130_;
v_bs_119_ = v___x_131_;
v___y_120_ = v_snd_126_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___boxed(lean_object* v_sz_133_, lean_object* v_i_134_, lean_object* v_bs_135_, lean_object* v___y_136_){
_start:
{
size_t v_sz_boxed_137_; size_t v_i_boxed_138_; lean_object* v_res_139_; 
v_sz_boxed_137_ = lean_unbox_usize(v_sz_133_);
lean_dec(v_sz_133_);
v_i_boxed_138_ = lean_unbox_usize(v_i_134_);
lean_dec(v_i_134_);
v_res_139_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(v_sz_boxed_137_, v_i_boxed_138_, v_bs_135_, v___y_136_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(size_t v_sz_140_, size_t v_i_141_, lean_object* v_bs_142_, lean_object* v___y_143_){
_start:
{
uint8_t v___x_144_; 
v___x_144_ = lean_usize_dec_lt(v_i_141_, v_sz_140_);
if (v___x_144_ == 0)
{
lean_object* v___x_145_; 
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v_bs_142_);
lean_ctor_set(v___x_145_, 1, v___y_143_);
return v___x_145_;
}
else
{
lean_object* v_v_146_; lean_object* v___x_147_; lean_object* v_bs_x27_148_; lean_object* v___x_149_; size_t v___x_150_; size_t v___x_151_; lean_object* v___x_152_; 
v_v_146_ = lean_array_uget(v_bs_142_, v_i_141_);
v___x_147_ = lean_unsigned_to_nat(0u);
v_bs_x27_148_ = lean_array_uset(v_bs_142_, v_i_141_, v___x_147_);
v___x_149_ = l_Lean_SubExpr_instToJsonGoalsLocation_toJson(v_v_146_);
v___x_150_ = ((size_t)1ULL);
v___x_151_ = lean_usize_add(v_i_141_, v___x_150_);
v___x_152_ = lean_array_uset(v_bs_x27_148_, v_i_141_, v___x_149_);
v_i_141_ = v___x_151_;
v_bs_142_ = v___x_152_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2___boxed(lean_object* v_sz_154_, lean_object* v_i_155_, lean_object* v_bs_156_, lean_object* v___y_157_){
_start:
{
size_t v_sz_boxed_158_; size_t v_i_boxed_159_; lean_object* v_res_160_; 
v_sz_boxed_158_ = lean_unbox_usize(v_sz_154_);
lean_dec(v_sz_154_);
v_i_boxed_159_ = lean_unbox_usize(v_i_155_);
lean_dec(v_i_155_);
v_res_160_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(v_sz_boxed_158_, v_i_boxed_159_, v_bs_156_, v___y_157_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v_pos_163_; lean_object* v_goals_164_; lean_object* v_termGoal_x3f_165_; lean_object* v_selectedLocations_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_208_; 
v_pos_163_ = lean_ctor_get(v_a_161_, 0);
v_goals_164_ = lean_ctor_get(v_a_161_, 1);
v_termGoal_x3f_165_ = lean_ctor_get(v_a_161_, 2);
v_selectedLocations_166_ = lean_ctor_get(v_a_161_, 3);
v_isSharedCheck_208_ = !lean_is_exclusive(v_a_161_);
if (v_isSharedCheck_208_ == 0)
{
v___x_168_ = v_a_161_;
v_isShared_169_ = v_isSharedCheck_208_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_selectedLocations_166_);
lean_inc(v_termGoal_x3f_165_);
lean_inc(v_goals_164_);
lean_inc(v_pos_163_);
lean_dec(v_a_161_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_208_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
size_t v_sz_170_; size_t v___x_171_; lean_object* v___x_172_; lean_object* v_fst_173_; lean_object* v_snd_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v_fst_178_; lean_object* v_snd_179_; 
v_sz_170_ = lean_array_size(v_goals_164_);
v___x_171_ = ((size_t)0ULL);
v___x_172_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(v_sz_170_, v___x_171_, v_goals_164_, v_a_162_);
v_fst_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_fst_173_);
v_snd_174_ = lean_ctor_get(v___x_172_, 1);
lean_inc(v_snd_174_);
lean_dec_ref(v___x_172_);
v___x_175_ = l_Lean_Lsp_instToJsonPosition_toJson(v_pos_163_);
v___x_176_ = lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(v_fst_173_);
if (lean_obj_tag(v_termGoal_x3f_165_) == 0)
{
lean_object* v___x_196_; 
v___x_196_ = lean_box(0);
v_fst_178_ = v___x_196_;
v_snd_179_ = v_snd_174_;
goto v___jp_177_;
}
else
{
lean_object* v_val_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_207_; 
v_val_197_ = lean_ctor_get(v_termGoal_x3f_165_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v_termGoal_x3f_165_);
if (v_isSharedCheck_207_ == 0)
{
v___x_199_ = v_termGoal_x3f_165_;
v_isShared_200_ = v_isSharedCheck_207_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_val_197_);
lean_dec(v_termGoal_x3f_165_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_207_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_201_; lean_object* v_fst_202_; lean_object* v_snd_203_; lean_object* v___x_205_; 
v___x_201_ = l_Lean_Widget_instRpcEncodableInteractiveTermGoal_enc_00___x40_Lean_Widget_InteractiveGoal_2553565095____hygCtx___hyg_1_(v_val_197_, v_snd_174_);
v_fst_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_fst_202_);
v_snd_203_ = lean_ctor_get(v___x_201_, 1);
lean_inc(v_snd_203_);
lean_dec_ref(v___x_201_);
if (v_isShared_200_ == 0)
{
lean_ctor_set(v___x_199_, 0, v_fst_202_);
v___x_205_ = v___x_199_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_fst_202_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
v_fst_178_ = v___x_205_;
v_snd_179_ = v_snd_203_;
goto v___jp_177_;
}
}
}
v___jp_177_:
{
size_t v_sz_180_; lean_object* v___x_181_; lean_object* v_fst_182_; lean_object* v_snd_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_195_; 
v_sz_180_ = lean_array_size(v_selectedLocations_166_);
v___x_181_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(v_sz_180_, v___x_171_, v_selectedLocations_166_, v_snd_179_);
v_fst_182_ = lean_ctor_get(v___x_181_, 0);
v_snd_183_ = lean_ctor_get(v___x_181_, 1);
v_isSharedCheck_195_ = !lean_is_exclusive(v___x_181_);
if (v_isSharedCheck_195_ == 0)
{
v___x_185_ = v___x_181_;
v_isShared_186_ = v_isSharedCheck_195_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_snd_183_);
lean_inc(v_fst_182_);
lean_dec(v___x_181_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_195_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_187_; lean_object* v___x_189_; 
v___x_187_ = lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_enc_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(v_fst_182_);
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 3, v___x_187_);
lean_ctor_set(v___x_168_, 2, v_fst_178_);
lean_ctor_set(v___x_168_, 1, v___x_176_);
lean_ctor_set(v___x_168_, 0, v___x_175_);
v___x_189_ = v___x_168_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v___x_175_);
lean_ctor_set(v_reuseFailAlloc_194_, 1, v___x_176_);
lean_ctor_set(v_reuseFailAlloc_194_, 2, v_fst_178_);
lean_ctor_set(v_reuseFailAlloc_194_, 3, v___x_187_);
v___x_189_ = v_reuseFailAlloc_194_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
lean_object* v___x_190_; lean_object* v___x_192_; 
v___x_190_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_36_(v___x_189_);
lean_dec_ref(v___x_189_);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 0, v___x_190_);
v___x_192_ = v___x_185_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_193_, 1, v_snd_183_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___redArg(lean_object* v_x_209_){
_start:
{
lean_inc_ref(v_x_209_);
return v_x_209_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object* v_x_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___redArg(v_x_210_);
lean_dec_ref(v_x_210_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(lean_object* v_00_u03b1_212_, lean_object* v_x_213_, lean_object* v___y_214_){
_start:
{
lean_inc_ref(v_x_213_);
return v_x_213_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0___boxed(lean_object* v_00_u03b1_215_, lean_object* v_x_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__0(v_00_u03b1_215_, v_x_216_, v___y_217_);
lean_dec_ref(v___y_217_);
lean_dec_ref(v_x_216_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(size_t v_sz_219_, size_t v_i_220_, lean_object* v_bs_221_, lean_object* v___y_222_){
_start:
{
uint8_t v___x_223_; 
v___x_223_ = lean_usize_dec_lt(v_i_220_, v_sz_219_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; 
v___x_224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_224_, 0, v_bs_221_);
return v___x_224_;
}
else
{
lean_object* v_v_225_; lean_object* v___x_226_; 
v_v_225_ = lean_array_uget_borrowed(v_bs_221_, v_i_220_);
lean_inc(v_v_225_);
v___x_226_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_225_, v___y_222_);
if (lean_obj_tag(v___x_226_) == 0)
{
lean_object* v_a_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_234_; 
lean_dec_ref(v_bs_221_);
v_a_227_ = lean_ctor_get(v___x_226_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_234_ == 0)
{
v___x_229_ = v___x_226_;
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_a_227_);
lean_dec(v___x_226_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
if (v_isShared_230_ == 0)
{
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_a_227_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
else
{
lean_object* v_a_235_; lean_object* v___x_236_; lean_object* v_bs_x27_237_; size_t v___x_238_; size_t v___x_239_; lean_object* v___x_240_; 
v_a_235_ = lean_ctor_get(v___x_226_, 0);
lean_inc(v_a_235_);
lean_dec_ref_known(v___x_226_, 1);
v___x_236_ = lean_unsigned_to_nat(0u);
v_bs_x27_237_ = lean_array_uset(v_bs_221_, v_i_220_, v___x_236_);
v___x_238_ = ((size_t)1ULL);
v___x_239_ = lean_usize_add(v_i_220_, v___x_238_);
v___x_240_ = lean_array_uset(v_bs_x27_237_, v_i_220_, v_a_235_);
v_i_220_ = v___x_239_;
v_bs_221_ = v___x_240_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2___boxed(lean_object* v_sz_242_, lean_object* v_i_243_, lean_object* v_bs_244_, lean_object* v___y_245_){
_start:
{
size_t v_sz_boxed_246_; size_t v_i_boxed_247_; lean_object* v_res_248_; 
v_sz_boxed_246_ = lean_unbox_usize(v_sz_242_);
lean_dec(v_sz_242_);
v_i_boxed_247_ = lean_unbox_usize(v_i_243_);
lean_dec(v_i_243_);
v_res_248_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(v_sz_boxed_246_, v_i_boxed_247_, v_bs_244_, v___y_245_);
lean_dec_ref(v___y_245_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(size_t v_sz_249_, size_t v_i_250_, lean_object* v_bs_251_){
_start:
{
uint8_t v___x_252_; 
v___x_252_ = lean_usize_dec_lt(v_i_250_, v_sz_249_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; 
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v_bs_251_);
return v___x_253_;
}
else
{
lean_object* v_v_254_; lean_object* v___x_255_; lean_object* v_bs_x27_256_; size_t v___x_257_; size_t v___x_258_; lean_object* v___x_259_; 
v_v_254_ = lean_array_uget(v_bs_251_, v_i_250_);
v___x_255_ = lean_unsigned_to_nat(0u);
v_bs_x27_256_ = lean_array_uset(v_bs_251_, v_i_250_, v___x_255_);
v___x_257_ = ((size_t)1ULL);
v___x_258_ = lean_usize_add(v_i_250_, v___x_257_);
v___x_259_ = lean_array_uset(v_bs_x27_256_, v_i_250_, v_v_254_);
v_i_250_ = v___x_258_;
v_bs_251_ = v___x_259_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object* v_sz_261_, lean_object* v_i_262_, lean_object* v_bs_263_){
_start:
{
size_t v_sz_boxed_264_; size_t v_i_boxed_265_; lean_object* v_res_266_; 
v_sz_boxed_264_ = lean_unbox_usize(v_sz_261_);
lean_dec(v_sz_261_);
v_i_boxed_265_ = lean_unbox_usize(v_i_262_);
lean_dec(v_i_262_);
v_res_266_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(v_sz_boxed_264_, v_i_boxed_265_, v_bs_263_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(lean_object* v_x_269_){
_start:
{
if (lean_obj_tag(v_x_269_) == 4)
{
lean_object* v_elems_270_; size_t v_sz_271_; size_t v___x_272_; lean_object* v___x_273_; 
v_elems_270_ = lean_ctor_get(v_x_269_, 0);
lean_inc_ref(v_elems_270_);
lean_dec_ref_known(v_x_269_, 1);
v_sz_271_ = lean_array_size(v_elems_270_);
v___x_272_ = ((size_t)0ULL);
v___x_273_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1_spec__1(v_sz_271_, v___x_272_, v_elems_270_);
return v___x_273_;
}
else
{
lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_274_ = ((lean_object*)(lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__0));
v___x_275_ = lean_unsigned_to_nat(80u);
v___x_276_ = l_Lean_Json_pretty(v_x_269_, v___x_275_);
v___x_277_ = lean_string_append(v___x_274_, v___x_276_);
lean_dec_ref(v___x_276_);
v___x_278_ = ((lean_object*)(lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1___closed__1));
v___x_279_ = lean_string_append(v___x_277_, v___x_278_);
v___x_280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
return v___x_280_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg(size_t v_sz_281_, size_t v_i_282_, lean_object* v_bs_283_){
_start:
{
uint8_t v___x_284_; 
v___x_284_ = lean_usize_dec_lt(v_i_282_, v_sz_281_);
if (v___x_284_ == 0)
{
lean_object* v___x_285_; 
v___x_285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_285_, 0, v_bs_283_);
return v___x_285_;
}
else
{
lean_object* v_v_286_; lean_object* v___x_287_; 
v_v_286_ = lean_array_uget_borrowed(v_bs_283_, v_i_282_);
lean_inc(v_v_286_);
v___x_287_ = l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(v_v_286_);
if (lean_obj_tag(v___x_287_) == 0)
{
lean_object* v_a_288_; lean_object* v___x_290_; uint8_t v_isShared_291_; uint8_t v_isSharedCheck_295_; 
lean_dec_ref(v_bs_283_);
v_a_288_ = lean_ctor_get(v___x_287_, 0);
v_isSharedCheck_295_ = !lean_is_exclusive(v___x_287_);
if (v_isSharedCheck_295_ == 0)
{
v___x_290_ = v___x_287_;
v_isShared_291_ = v_isSharedCheck_295_;
goto v_resetjp_289_;
}
else
{
lean_inc(v_a_288_);
lean_dec(v___x_287_);
v___x_290_ = lean_box(0);
v_isShared_291_ = v_isSharedCheck_295_;
goto v_resetjp_289_;
}
v_resetjp_289_:
{
lean_object* v___x_293_; 
if (v_isShared_291_ == 0)
{
v___x_293_ = v___x_290_;
goto v_reusejp_292_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v_a_288_);
v___x_293_ = v_reuseFailAlloc_294_;
goto v_reusejp_292_;
}
v_reusejp_292_:
{
return v___x_293_;
}
}
}
else
{
lean_object* v_a_296_; lean_object* v___x_297_; lean_object* v_bs_x27_298_; size_t v___x_299_; size_t v___x_300_; lean_object* v___x_301_; 
v_a_296_ = lean_ctor_get(v___x_287_, 0);
lean_inc(v_a_296_);
lean_dec_ref_known(v___x_287_, 1);
v___x_297_ = lean_unsigned_to_nat(0u);
v_bs_x27_298_ = lean_array_uset(v_bs_283_, v_i_282_, v___x_297_);
v___x_299_ = ((size_t)1ULL);
v___x_300_ = lean_usize_add(v_i_282_, v___x_299_);
v___x_301_ = lean_array_uset(v_bs_x27_298_, v_i_282_, v_a_296_);
v_i_282_ = v___x_300_;
v_bs_283_ = v___x_301_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object* v_sz_303_, lean_object* v_i_304_, lean_object* v_bs_305_){
_start:
{
size_t v_sz_boxed_306_; size_t v_i_boxed_307_; lean_object* v_res_308_; 
v_sz_boxed_306_ = lean_unbox_usize(v_sz_303_);
lean_dec(v_sz_303_);
v_i_boxed_307_ = lean_unbox_usize(v_i_304_);
lean_dec(v_i_304_);
v_res_308_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg(v_sz_boxed_306_, v_i_boxed_307_, v_bs_305_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(lean_object* v_j_309_, lean_object* v_a_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_Panel_Basic_4212595528____hygCtx___hyg_17_(v_j_309_);
if (lean_obj_tag(v___x_311_) == 0)
{
lean_object* v_a_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_319_; 
v_a_312_ = lean_ctor_get(v___x_311_, 0);
v_isSharedCheck_319_ = !lean_is_exclusive(v___x_311_);
if (v_isSharedCheck_319_ == 0)
{
v___x_314_ = v___x_311_;
v_isShared_315_ = v_isSharedCheck_319_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_a_312_);
lean_dec(v___x_311_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_319_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___x_317_; 
if (v_isShared_315_ == 0)
{
v___x_317_ = v___x_314_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v_a_312_);
v___x_317_ = v_reuseFailAlloc_318_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
return v___x_317_;
}
}
}
else
{
lean_object* v_a_320_; lean_object* v_pos_321_; lean_object* v_goals_322_; lean_object* v_termGoal_x3f_323_; lean_object* v_selectedLocations_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_413_; 
v_a_320_ = lean_ctor_get(v___x_311_, 0);
lean_inc(v_a_320_);
lean_dec_ref_known(v___x_311_, 1);
v_pos_321_ = lean_ctor_get(v_a_320_, 0);
v_goals_322_ = lean_ctor_get(v_a_320_, 1);
v_termGoal_x3f_323_ = lean_ctor_get(v_a_320_, 2);
v_selectedLocations_324_ = lean_ctor_get(v_a_320_, 3);
v_isSharedCheck_413_ = !lean_is_exclusive(v_a_320_);
if (v_isSharedCheck_413_ == 0)
{
v___x_326_ = v_a_320_;
v_isShared_327_ = v_isSharedCheck_413_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_selectedLocations_324_);
lean_inc(v_termGoal_x3f_323_);
lean_inc(v_goals_322_);
lean_inc(v_pos_321_);
lean_dec(v_a_320_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_413_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_328_; 
v___x_328_ = l_Lean_Lsp_instFromJsonPosition_fromJson(v_pos_321_);
if (lean_obj_tag(v___x_328_) == 0)
{
lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_336_; 
lean_del_object(v___x_326_);
lean_dec(v_selectedLocations_324_);
lean_dec(v_termGoal_x3f_323_);
lean_dec(v_goals_322_);
v_a_329_ = lean_ctor_get(v___x_328_, 0);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_328_);
if (v_isSharedCheck_336_ == 0)
{
v___x_331_ = v___x_328_;
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_dec(v___x_328_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_a_329_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
else
{
lean_object* v_a_337_; lean_object* v___x_338_; 
v_a_337_ = lean_ctor_get(v___x_328_, 0);
lean_inc(v_a_337_);
lean_dec_ref_known(v___x_328_, 1);
v___x_338_ = lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(v_goals_322_);
if (lean_obj_tag(v___x_338_) == 0)
{
lean_object* v_a_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_346_; 
lean_dec(v_a_337_);
lean_del_object(v___x_326_);
lean_dec(v_selectedLocations_324_);
lean_dec(v_termGoal_x3f_323_);
v_a_339_ = lean_ctor_get(v___x_338_, 0);
v_isSharedCheck_346_ = !lean_is_exclusive(v___x_338_);
if (v_isSharedCheck_346_ == 0)
{
v___x_341_ = v___x_338_;
v_isShared_342_ = v_isSharedCheck_346_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_a_339_);
lean_dec(v___x_338_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_346_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_344_; 
if (v_isShared_342_ == 0)
{
v___x_344_ = v___x_341_;
goto v_reusejp_343_;
}
else
{
lean_object* v_reuseFailAlloc_345_; 
v_reuseFailAlloc_345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_345_, 0, v_a_339_);
v___x_344_ = v_reuseFailAlloc_345_;
goto v_reusejp_343_;
}
v_reusejp_343_:
{
return v___x_344_;
}
}
}
else
{
lean_object* v_a_347_; size_t v_sz_348_; size_t v___x_349_; lean_object* v___x_350_; 
v_a_347_ = lean_ctor_get(v___x_338_, 0);
lean_inc(v_a_347_);
lean_dec_ref_known(v___x_338_, 1);
v_sz_348_ = lean_array_size(v_a_347_);
v___x_349_ = ((size_t)0ULL);
v___x_350_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__2(v_sz_348_, v___x_349_, v_a_347_, v_a_310_);
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_358_; 
lean_dec(v_a_337_);
lean_del_object(v___x_326_);
lean_dec(v_selectedLocations_324_);
lean_dec(v_termGoal_x3f_323_);
v_a_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_358_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_356_; 
if (v_isShared_354_ == 0)
{
v___x_356_ = v___x_353_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_a_351_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
else
{
lean_object* v_a_359_; lean_object* v_____do__lift_361_; lean_object* v___y_362_; 
v_a_359_ = lean_ctor_get(v___x_350_, 0);
lean_inc(v_a_359_);
lean_dec_ref_known(v___x_350_, 1);
if (lean_obj_tag(v_termGoal_x3f_323_) == 0)
{
lean_object* v___x_394_; 
v___x_394_ = lean_box(0);
v_____do__lift_361_ = v___x_394_;
v___y_362_ = v_a_310_;
goto v___jp_360_;
}
else
{
lean_object* v_val_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_412_; 
v_val_395_ = lean_ctor_get(v_termGoal_x3f_323_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v_termGoal_x3f_323_);
if (v_isSharedCheck_412_ == 0)
{
v___x_397_ = v_termGoal_x3f_323_;
v_isShared_398_ = v_isSharedCheck_412_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_val_395_);
lean_dec(v_termGoal_x3f_323_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_412_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_399_; 
v___x_399_ = l_Lean_Widget_instRpcEncodableInteractiveTermGoal_dec_00___x40_Lean_Widget_InteractiveGoal_2553565095____hygCtx___hyg_1_(v_val_395_, v_a_310_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_del_object(v___x_397_);
lean_dec(v_a_359_);
lean_dec(v_a_337_);
lean_del_object(v___x_326_);
lean_dec(v_selectedLocations_324_);
v_a_400_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_399_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_399_);
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
v_reuseFailAlloc_406_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_408_; lean_object* v___x_410_; 
v_a_408_ = lean_ctor_get(v___x_399_, 0);
lean_inc(v_a_408_);
lean_dec_ref_known(v___x_399_, 1);
if (v_isShared_398_ == 0)
{
lean_ctor_set(v___x_397_, 0, v_a_408_);
v___x_410_ = v___x_397_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_408_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
v_____do__lift_361_ = v___x_410_;
v___y_362_ = v_a_310_;
goto v___jp_360_;
}
}
}
}
v___jp_360_:
{
lean_object* v___x_363_; 
v___x_363_ = lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__1(v_selectedLocations_324_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_371_; 
lean_dec(v_____do__lift_361_);
lean_dec(v_a_359_);
lean_dec(v_a_337_);
lean_del_object(v___x_326_);
v_a_364_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_371_ == 0)
{
v___x_366_ = v___x_363_;
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_363_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v___x_369_; 
if (v_isShared_367_ == 0)
{
v___x_369_ = v___x_366_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_a_364_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
else
{
lean_object* v_a_372_; size_t v_sz_373_; lean_object* v___x_374_; 
v_a_372_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_a_372_);
lean_dec_ref_known(v___x_363_, 1);
v_sz_373_ = lean_array_size(v_a_372_);
v___x_374_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg(v_sz_373_, v___x_349_, v_a_372_);
if (lean_obj_tag(v___x_374_) == 0)
{
lean_object* v_a_375_; lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_382_; 
lean_dec(v_____do__lift_361_);
lean_dec(v_a_359_);
lean_dec(v_a_337_);
lean_del_object(v___x_326_);
v_a_375_ = lean_ctor_get(v___x_374_, 0);
v_isSharedCheck_382_ = !lean_is_exclusive(v___x_374_);
if (v_isSharedCheck_382_ == 0)
{
v___x_377_ = v___x_374_;
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
else
{
lean_inc(v_a_375_);
lean_dec(v___x_374_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
lean_object* v___x_380_; 
if (v_isShared_378_ == 0)
{
v___x_380_ = v___x_377_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_a_375_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
}
else
{
lean_object* v_a_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_393_; 
v_a_383_ = lean_ctor_get(v___x_374_, 0);
v_isSharedCheck_393_ = !lean_is_exclusive(v___x_374_);
if (v_isSharedCheck_393_ == 0)
{
v___x_385_ = v___x_374_;
v_isShared_386_ = v_isSharedCheck_393_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_a_383_);
lean_dec(v___x_374_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_393_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_388_; 
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 3, v_a_383_);
lean_ctor_set(v___x_326_, 2, v_____do__lift_361_);
lean_ctor_set(v___x_326_, 1, v_a_359_);
lean_ctor_set(v___x_326_, 0, v_a_337_);
v___x_388_ = v___x_326_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v_a_337_);
lean_ctor_set(v_reuseFailAlloc_392_, 1, v_a_359_);
lean_ctor_set(v_reuseFailAlloc_392_, 2, v_____do__lift_361_);
lean_ctor_set(v_reuseFailAlloc_392_, 3, v_a_383_);
v___x_388_ = v_reuseFailAlloc_392_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
lean_object* v___x_390_; 
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 0, v___x_388_);
v___x_390_ = v___x_385_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_388_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
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
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1____boxed(lean_object* v_j_414_, lean_object* v_a_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(v_j_414_, v_a_415_);
lean_dec_ref(v_a_415_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3(size_t v_sz_417_, size_t v_i_418_, lean_object* v_bs_419_, lean_object* v___y_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___redArg(v_sz_417_, v_i_418_, v_bs_419_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3___boxed(lean_object* v_sz_422_, lean_object* v_i_423_, lean_object* v_bs_424_, lean_object* v___y_425_){
_start:
{
size_t v_sz_boxed_426_; size_t v_i_boxed_427_; lean_object* v_res_428_; 
v_sz_boxed_426_ = lean_unbox_usize(v_sz_422_);
lean_dec(v_sz_422_);
v_i_boxed_427_ = lean_unbox_usize(v_i_423_);
lean_dec(v_i_423_);
v_res_428_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1__spec__3(v_sz_boxed_426_, v_i_boxed_427_, v_bs_424_, v___y_425_);
lean_dec_ref(v___y_425_);
return v_res_428_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13(void){
_start:
{
uint8_t v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v___x_458_ = 0;
v___x_459_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__12));
v___x_460_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__10));
v___x_461_ = l_Lean_Widget_widgetInstanceSpec;
v___x_462_ = lean_alloc_ctor(11, 3, 1);
lean_ctor_set(v___x_462_, 0, v___x_461_);
lean_ctor_set(v___x_462_, 1, v___x_460_);
lean_ctor_set(v___x_462_, 2, v___x_459_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*3, v___x_458_);
return v___x_462_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14(void){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_463_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13, &lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13_once, _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__13);
v___x_464_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__9));
v___x_465_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4));
v___x_466_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v___x_464_);
lean_ctor_set(v___x_466_, 2, v___x_463_);
return v___x_466_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_470_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__16));
v___x_471_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14, &lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14_once, _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__14);
v___x_472_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4));
v___x_473_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
lean_ctor_set(v___x_473_, 1, v___x_471_);
lean_ctor_set(v___x_473_, 2, v___x_470_);
return v___x_473_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_479_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__20));
v___x_480_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17, &lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17_once, _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__17);
v___x_481_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__4));
v___x_482_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
lean_ctor_set(v___x_482_, 1, v___x_480_);
lean_ctor_set(v___x_482_, 2, v___x_479_);
return v___x_482_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_483_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21, &lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21_once, _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__21);
v___x_484_ = lean_unsigned_to_nat(1022u);
v___x_485_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2));
v___x_486_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_486_, 0, v___x_485_);
lean_ctor_set(v___x_486_, 1, v___x_484_);
lean_ctor_set(v___x_486_, 2, v___x_483_);
return v___x_486_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx(void){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22, &lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22_once, _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__22);
return v___x_487_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_488_ = lean_box(0);
v___x_489_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_490_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
lean_ctor_set(v___x_490_, 1, v___x_488_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg(){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_492_ = lean_obj_once(&lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0, &lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0_once, _init_lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___closed__0);
v___x_493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg___boxed(lean_object* v___y_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg();
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0(lean_object* v_00_u03b1_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg();
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___boxed(lean_object* v_00_u03b1_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0(v_00_u03b1_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
lean_dec(v___y_515_);
lean_dec_ref(v___y_514_);
lean_dec(v___y_513_);
lean_dec_ref(v___y_512_);
lean_dec(v___y_511_);
lean_dec_ref(v___y_510_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(lean_object* v_x_518_, lean_object* v_as_519_, size_t v_i_520_, size_t v_stop_521_, lean_object* v_b_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
uint8_t v___x_530_; 
v___x_530_ = lean_usize_dec_eq(v_i_520_, v_stop_521_);
if (v___x_530_ == 0)
{
lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_531_ = lean_array_uget_borrowed(v_as_519_, v_i_520_);
lean_inc(v___x_531_);
v___x_532_ = l_Lean_Widget_elabWidgetInstanceSpec(v___x_531_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; lean_object* v___x_534_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
lean_inc(v_a_533_);
lean_dec_ref_known(v___x_532_, 1);
v___x_534_ = l___private_Lean_Widget_UserWidget_0__Lean_Widget_evalWidgetInstanceUnsafe(v_a_533_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_534_) == 0)
{
lean_object* v_a_535_; uint64_t v_javascriptHash_536_; lean_object* v_props_537_; lean_object* v___x_538_; 
v_a_535_ = lean_ctor_get(v___x_534_, 0);
lean_inc(v_a_535_);
lean_dec_ref_known(v___x_534_, 1);
v_javascriptHash_536_ = lean_ctor_get_uint64(v_a_535_, sizeof(void*)*2);
v_props_537_ = lean_ctor_get(v_a_535_, 1);
lean_inc_ref(v_props_537_);
lean_dec(v_a_535_);
lean_inc(v_x_518_);
v___x_538_ = l_Lean_Widget_savePanelWidgetInfo(v_javascriptHash_536_, v_props_537_, v_x_518_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_538_) == 0)
{
lean_object* v_a_539_; size_t v___x_540_; size_t v___x_541_; 
v_a_539_ = lean_ctor_get(v___x_538_, 0);
lean_inc(v_a_539_);
lean_dec_ref_known(v___x_538_, 1);
v___x_540_ = ((size_t)1ULL);
v___x_541_ = lean_usize_add(v_i_520_, v___x_540_);
v_i_520_ = v___x_541_;
v_b_522_ = v_a_539_;
goto _start;
}
else
{
lean_dec(v_x_518_);
return v___x_538_;
}
}
else
{
lean_object* v_a_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_550_; 
lean_dec(v_x_518_);
v_a_543_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_550_ == 0)
{
v___x_545_ = v___x_534_;
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_a_543_);
lean_dec(v___x_534_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
lean_object* v___x_548_; 
if (v_isShared_546_ == 0)
{
v___x_548_ = v___x_545_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v_a_543_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
return v___x_548_;
}
}
}
}
else
{
lean_object* v_a_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_558_; 
lean_dec(v_x_518_);
v_a_551_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_558_ == 0)
{
v___x_553_ = v___x_532_;
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_a_551_);
lean_dec(v___x_532_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___x_556_; 
if (v_isShared_554_ == 0)
{
v___x_556_ = v___x_553_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v_a_551_);
v___x_556_ = v_reuseFailAlloc_557_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
return v___x_556_;
}
}
}
}
else
{
lean_object* v___x_559_; 
lean_dec(v_x_518_);
v___x_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_559_, 0, v_b_522_);
return v___x_559_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg___boxed(lean_object* v_x_560_, lean_object* v_as_561_, lean_object* v_i_562_, lean_object* v_stop_563_, lean_object* v_b_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_){
_start:
{
size_t v_i_boxed_572_; size_t v_stop_boxed_573_; lean_object* v_res_574_; 
v_i_boxed_572_ = lean_unbox_usize(v_i_562_);
lean_dec(v_i_562_);
v_stop_boxed_573_ = lean_unbox_usize(v_stop_563_);
lean_dec(v_stop_563_);
v_res_574_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(v_x_560_, v_as_561_, v_i_boxed_572_, v_stop_boxed_573_, v_b_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_, v___y_570_);
lean_dec(v___y_570_);
lean_dec_ref(v___y_569_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec_ref(v_as_561_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgets(lean_object* v_x_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v___x_585_; uint8_t v___x_586_; 
v___x_585_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx___closed__2));
lean_inc(v_x_575_);
v___x_586_ = l_Lean_Syntax_isOfKind(v_x_575_, v___x_585_);
if (v___x_586_ == 0)
{
lean_object* v___x_587_; 
lean_dec(v_x_575_);
v___x_587_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets_withPanelWidgets_spec__0___redArg();
return v___x_587_;
}
else
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___y_594_; lean_object* v_specs_596_; lean_object* v___x_597_; lean_object* v___x_598_; uint8_t v___x_599_; 
v___x_588_ = lean_unsigned_to_nat(0u);
v___x_589_ = lean_unsigned_to_nat(2u);
v___x_590_ = l_Lean_Syntax_getArg(v_x_575_, v___x_589_);
v___x_591_ = lean_unsigned_to_nat(4u);
v___x_592_ = l_Lean_Syntax_getArg(v_x_575_, v___x_591_);
v_specs_596_ = l_Lean_Syntax_getArgs(v___x_590_);
lean_dec(v___x_590_);
v___x_597_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_specs_596_);
lean_dec_ref(v_specs_596_);
v___x_598_ = lean_array_get_size(v___x_597_);
v___x_599_ = lean_nat_dec_lt(v___x_588_, v___x_598_);
if (v___x_599_ == 0)
{
lean_object* v___x_600_; 
lean_dec_ref(v___x_597_);
lean_dec(v_x_575_);
v___x_600_ = l_Lean_Elab_Tactic_evalTacticSeq(v___x_592_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
return v___x_600_;
}
else
{
lean_object* v___x_601_; uint8_t v___x_602_; 
v___x_601_ = lean_box(0);
v___x_602_ = lean_nat_dec_le(v___x_598_, v___x_598_);
if (v___x_602_ == 0)
{
if (v___x_599_ == 0)
{
lean_object* v___x_603_; 
lean_dec_ref(v___x_597_);
lean_dec(v_x_575_);
v___x_603_ = l_Lean_Elab_Tactic_evalTacticSeq(v___x_592_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
return v___x_603_;
}
else
{
size_t v___x_604_; size_t v___x_605_; lean_object* v___x_606_; 
v___x_604_ = ((size_t)0ULL);
v___x_605_ = lean_usize_of_nat(v___x_598_);
v___x_606_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(v_x_575_, v___x_597_, v___x_604_, v___x_605_, v___x_601_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
lean_dec_ref(v___x_597_);
v___y_594_ = v___x_606_;
goto v___jp_593_;
}
}
else
{
size_t v___x_607_; size_t v___x_608_; lean_object* v___x_609_; 
v___x_607_ = ((size_t)0ULL);
v___x_608_ = lean_usize_of_nat(v___x_598_);
v___x_609_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(v_x_575_, v___x_597_, v___x_607_, v___x_608_, v___x_601_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
lean_dec_ref(v___x_597_);
v___y_594_ = v___x_609_;
goto v___jp_593_;
}
}
v___jp_593_:
{
if (lean_obj_tag(v___y_594_) == 0)
{
lean_object* v___x_595_; 
lean_dec_ref_known(v___y_594_, 1);
v___x_595_ = l_Lean_Elab_Tactic_evalTacticSeq(v___x_592_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
return v___x_595_;
}
else
{
lean_dec(v___x_592_);
return v___y_594_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_withPanelWidgets___boxed(lean_object* v_x_610_, lean_object* v_a_611_, lean_object* v_a_612_, lean_object* v_a_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_proofwidgets_ProofWidgets_withPanelWidgets(v_x_610_, v_a_611_, v_a_612_, v_a_613_, v_a_614_, v_a_615_, v_a_616_, v_a_617_, v_a_618_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
lean_dec(v_a_616_);
lean_dec_ref(v_a_615_);
lean_dec(v_a_614_);
lean_dec_ref(v_a_613_);
lean_dec(v_a_612_);
lean_dec_ref(v_a_611_);
return v_res_620_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1(lean_object* v_x_621_, lean_object* v_as_622_, size_t v_i_623_, size_t v_stop_624_, lean_object* v_b_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
lean_object* v___x_635_; 
v___x_635_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___redArg(v_x_621_, v_as_622_, v_i_623_, v_stop_624_, v_b_625_, v___y_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_, v___y_633_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1___boxed(lean_object* v_x_636_, lean_object* v_as_637_, lean_object* v_i_638_, lean_object* v_stop_639_, lean_object* v_b_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_){
_start:
{
size_t v_i_boxed_650_; size_t v_stop_boxed_651_; lean_object* v_res_652_; 
v_i_boxed_650_ = lean_unbox_usize(v_i_638_);
lean_dec(v_i_638_);
v_stop_boxed_651_ = lean_unbox_usize(v_stop_639_);
lean_dec(v_stop_639_);
v_res_652_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_withPanelWidgets_spec__1(v_x_636_, v_as_637_, v_i_boxed_650_, v_stop_boxed_651_, v_b_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec_ref(v_as_637_);
return v_res_652_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_Commands(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_Commands(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx = _init_lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_withPanelWidgetsTacticStx);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
lean_object* initialize_Lean_Widget_Commands(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Widget_Commands(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
