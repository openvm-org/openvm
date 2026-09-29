// Lean compiler output
// Module: ProofWidgets.Data.Html
// Imports: public import Init public meta import Init public import ProofWidgets.Component.Basic public import ProofWidgets.Util
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_mkStrLit(lean_object*, lean_object*);
lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l_Lean_Syntax_isIdent(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
uint64_t lean_string_hash(lean_object*);
lean_object* l_Lean_Macro_throwErrorAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailInfo(lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* lean_string_utf8_get_opt(lean_object*, lean_object*);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_Lean_bignumToJson(lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* l_Lean_Json_getStr_x3f(lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_visitAtom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_getTag_x3f(lean_object*);
lean_object* l_Lean_Json_parseCtorFields(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_UInt64_fromJson_x3f(lean_object*);
lean_object* l_Lean_Parser_takeWhile1Fn(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkNodeToken(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAntiquot(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(lean_object*);
lean_object* l_Lean_Parser_withAntiquot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_element_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_element_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_text_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_text_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_component_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_component_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0_value;
static const lean_array_object lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__1_value),((lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default = (const lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml = (const lean_object*)&lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_element_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_element_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_text_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_text_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_component_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_component_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "no inductive tag found"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "text"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "element"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "component"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "no inductive constructor matched"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59__value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Prod_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "expected pair, got '"};
static const lean_object* lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec___lam__0_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableHtml___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value_aux_2),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "proofWidgetsJsxElement"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(241, 61, 86, 176, 243, 14, 250, 246)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(131, 37, 199, 10, 17, 203, 120, 112)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "`(proofWidgetsJsxElement| "};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__9_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__9_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(241, 61, 86, 176, 243, 14, 250, 246)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__13 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__13_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__13_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__15 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__15_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__10_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__15_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__16 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__16_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__6_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__16_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__17 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__17_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__17_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__18 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__18_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__18_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxElement;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "proofWidgetsJsxChild"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 58, 2, 237, 118, 40, 157, 253)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(86, 142, 4, 235, 92, 21, 119, 30)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "`(proofWidgetsJsxChild| "};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 58, 2, 237, 118, 40, 157, 253)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__5_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__9_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__9_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxChild;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "proofWidgetsJsxAttr"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 10, 189, 54, 39, 75, 225, 123)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(219, 83, 31, 237, 171, 248, 70, 54)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "`(proofWidgetsJsxAttr| "};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 10, 189, 54, 39, 75, 225, 123)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__5_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__9_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__9_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttr;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "proofWidgetsJsxAttrVal"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 105, 240, 167, 209, 179, 109, 139)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(134, 112, 217, 235, 28, 220, 186, 19)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "`(proofWidgetsJsxAttrVal| "};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 105, 240, 167, 209, 179, 109, 139)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__5_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__14_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__9_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__9_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttrVal;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Jsx"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "proofWidgetsJsxAttrVal_"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 251, 35, 201, 235, 15, 235, 246)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal__ = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__7_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "proofWidgetsJsxAttrVal{_}"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 11, 96, 156, 74, 252, 130, 115)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__5_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__5_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__9_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__9_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__12_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__12_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__13 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__13_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__13_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__14 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__14_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__14_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "proofWidgetsJsxAttr_=_"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 76, 137, 173, 176, 9, 246, 211)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__7_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_quot___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__9_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d__ = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__9_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "proofWidgetsJsxAttr{..._}"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(4, 178, 80, 170, 220, 188, 165, 180)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " {..."};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__7_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__7_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "{<>}$"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg(lean_object*, uint32_t, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0(uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1;
LEAN_EXPORT uint8_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2(uint32_t);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___boxed(lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "expected JSX text"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__1_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "jsxText"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__3_value),LEAN_SCALAR_PTR_LITERAL(51, 124, 170, 56, 233, 65, 122, 138)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__2_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__5_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__1_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__7_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__8_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText;
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0(lean_object*, uint32_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_getJsxText(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_getJsxText___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "proofWidgetsJsxElement<__/>"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 241, 72, 213, 72, 227, 176, 145)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__6_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__8_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "/>"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__10_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__12_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__12_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "proofWidgetsJsxElement<__>_</_>"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 162, 30, 24, 114, 202, 19, 65)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ">"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__3_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__6_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "</"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__6_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__9_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__9_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__4_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__10_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__3_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__12_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__12_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "proofWidgetsJsxChild_"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(218, 45, 106, 199, 177, 195, 73, 168)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__3_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild__ = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__3_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "proofWidgetsJsxChild{..._}"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(68, 162, 178, 77, 37, 236, 102, 109)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "{..."};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__8_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__8_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__11_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__6_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "proofWidgetsJsxChild{_}"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 159, 185, 9, 203, 91, 251, 92)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__12_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "proofWidgetsJsxChild__1"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(63, 133, 157, 178, 92, 40, 127, 204)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "term_"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 147, 218, 120, 233, 42, 243, 221)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__12_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_Jsx_term__ = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__2_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tuple"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__0_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 24, 88, 245, 200, 250, 27, 217)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__2 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__2_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__4 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__4_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__5 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__5_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__6 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__6_value;
static lean_once_cell_t lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 62, 46, 68, 60, 141, 109, 149)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__8_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__9 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__9_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "PrettyPrinter"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__10 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__10_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(120, 167, 117, 148, 131, 202, 42, 4)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__11_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__12 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__12_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__13_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__14 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__14_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__15 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__15_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__15_value),LEAN_SCALAR_PTR_LITERAL(143, 110, 173, 49, 223, 200, 200, 169)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__16_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__17 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__17_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Server"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__18 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__18_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(251, 1, 140, 35, 91, 244, 83, 213)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__19_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__20 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__20_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__21 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__21_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__21_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__22 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__22_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__23 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__23_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__20_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__23_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__24 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__24_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__17_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__24_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__25 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__25_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__14_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__25_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__26 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__26_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__12_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__26_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__27 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__27_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__9_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__27_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__28 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__28_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__29 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__29_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__29_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__32 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__32_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__32_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__34 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__34_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Json"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35_value;
static lean_once_cell_t lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35_value),LEAN_SCALAR_PTR_LITERAL(190, 18, 71, 130, 82, 255, 111, 18)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__37 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__37_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35_value),LEAN_SCALAR_PTR_LITERAL(215, 126, 99, 176, 35, 107, 201, 11)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__39 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__39_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__38_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__40 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__40_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__40_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__41 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__41_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__39_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__41_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__42 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__42_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term#[_,]"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 119, 178, 128, 145, 112, 206, 247)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "#["};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0(size_t, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0___boxed(lean_object*);
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "unknown syntax"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0_value;
static const lean_array_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__2 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__2_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__2_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Html.text"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__4 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__4_value;
static lean_once_cell_t lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Html"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(86, 101, 33, 160, 88, 74, 41, 14)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value),LEAN_SCALAR_PTR_LITERAL(203, 46, 129, 190, 34, 222, 18, 25)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(48, 5, 23, 178, 23, 214, 71, 37)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value),LEAN_SCALAR_PTR_LITERAL(173, 226, 159, 114, 162, 117, 134, 149)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__9 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__9_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__10 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__10_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__11 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__11_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__9_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__11_value)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__12 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__12_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_++_"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 69, 86, 178, 149, 48, 216, 23)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__1 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "++"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__2 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__0_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 77, 20, 88, 28, 210, 230, 84)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "structInstLVal"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__2 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__2_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__2_value),LEAN_SCALAR_PTR_LITERAL(185, 133, 6, 147, 6, 183, 100, 198)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structInstFieldDef"};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__4 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__4_value;
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value_aux_2),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__4_value),LEAN_SCALAR_PTR_LITERAL(81, 102, 39, 227, 176, 252, 65, 103)}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5_value;
static const lean_string_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__6 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__6_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Html.element"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__0_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(86, 101, 33, 160, 88, 74, 41, 14)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value),LEAN_SCALAR_PTR_LITERAL(162, 92, 79, 19, 43, 173, 84, 253)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(48, 5, 23, 178, 23, 214, 71, 37)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32__value),LEAN_SCALAR_PTR_LITERAL(172, 103, 107, 158, 53, 198, 134, 58)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__4_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__6_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__6_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__7_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Html.ofComponent"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__8_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ofComponent"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(86, 101, 33, 160, 88, 74, 41, 14)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__10_value),LEAN_SCALAR_PTR_LITERAL(23, 105, 5, 241, 201, 13, 143, 173)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value_aux_0),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(48, 5, 23, 178, 23, 214, 71, 37)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__10_value),LEAN_SCALAR_PTR_LITERAL(97, 4, 241, 99, 151, 178, 146, 79)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__13 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__13_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__14 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__14_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__15 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__15_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value_aux_2),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__15_value),LEAN_SCALAR_PTR_LITERAL(50, 43, 73, 62, 118, 124, 31, 28)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__17 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__17_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__18 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__18_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value_aux_2),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__18_value),LEAN_SCALAR_PTR_LITERAL(0, 82, 141, 43, 62, 171, 163, 69)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__20 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__20_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value_aux_2),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__20_value),LEAN_SCALAR_PTR_LITERAL(13, 1, 242, 203, 207, 188, 181, 160)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1_value),((lean_object*)&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__22 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__22_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "expected </"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__23 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__23_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(117, 121, 37, 123, 104, 28, 189, 89)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_quot___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__1_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value)} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_quot___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__2_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__2_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__1_value)} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__3_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__3_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__4_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_proofwidgets_ProofWidgets_Html_ctorIdx(v_x_5_);
lean_dec_ref(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(lean_object* v_t_7_, lean_object* v_k_8_){
_start:
{
switch(lean_obj_tag(v_t_7_))
{
case 0:
{
lean_object* v_a_9_; lean_object* v_a_10_; lean_object* v_a_11_; lean_object* v___x_12_; 
v_a_9_ = lean_ctor_get(v_t_7_, 0);
lean_inc_ref(v_a_9_);
v_a_10_ = lean_ctor_get(v_t_7_, 1);
lean_inc_ref(v_a_10_);
v_a_11_ = lean_ctor_get(v_t_7_, 2);
lean_inc_ref(v_a_11_);
lean_dec_ref_known(v_t_7_, 3);
v___x_12_ = lean_apply_3(v_k_8_, v_a_9_, v_a_10_, v_a_11_);
return v___x_12_;
}
case 1:
{
lean_object* v_a_13_; lean_object* v___x_14_; 
v_a_13_ = lean_ctor_get(v_t_7_, 0);
lean_inc_ref(v_a_13_);
lean_dec_ref_known(v_t_7_, 1);
v___x_14_ = lean_apply_1(v_k_8_, v_a_13_);
return v___x_14_;
}
default: 
{
uint64_t v_a_15_; lean_object* v_a_16_; lean_object* v_a_17_; lean_object* v_a_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v_a_15_ = lean_ctor_get_uint64(v_t_7_, sizeof(void*)*3);
v_a_16_ = lean_ctor_get(v_t_7_, 0);
lean_inc_ref(v_a_16_);
v_a_17_ = lean_ctor_get(v_t_7_, 1);
lean_inc_ref(v_a_17_);
v_a_18_ = lean_ctor_get(v_t_7_, 2);
lean_inc_ref(v_a_18_);
lean_dec_ref_known(v_t_7_, 3);
v___x_19_ = lean_box_uint64(v_a_15_);
v___x_20_ = lean_apply_4(v_k_8_, v___x_19_, v_a_16_, v_a_17_, v_a_18_);
return v___x_20_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim(lean_object* v_motive__1_21_, lean_object* v_ctorIdx_22_, lean_object* v_t_23_, lean_object* v_h_24_, lean_object* v_k_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_23_, v_k_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ctorElim___boxed(lean_object* v_motive__1_27_, lean_object* v_ctorIdx_28_, lean_object* v_t_29_, lean_object* v_h_30_, lean_object* v_k_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_proofwidgets_ProofWidgets_Html_ctorElim(v_motive__1_27_, v_ctorIdx_28_, v_t_29_, v_h_30_, v_k_31_);
lean_dec(v_ctorIdx_28_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_element_elim___redArg(lean_object* v_t_33_, lean_object* v_element_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_33_, v_element_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_element_elim(lean_object* v_motive__1_36_, lean_object* v_t_37_, lean_object* v_h_38_, lean_object* v_element_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_37_, v_element_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_text_elim___redArg(lean_object* v_t_41_, lean_object* v_text_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_41_, v_text_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_text_elim(lean_object* v_motive__1_44_, lean_object* v_t_45_, lean_object* v_h_46_, lean_object* v_text_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_45_, v_text_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_component_elim___redArg(lean_object* v_t_49_, lean_object* v_component_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_49_, v_component_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_component_elim(lean_object* v_motive__1_52_, lean_object* v_t_53_, lean_object* v_h_54_, lean_object* v_component_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_proofwidgets_ProofWidgets_Html_ctorElim___redArg(v_t_53_, v_component_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorIdx(lean_object* v_x_65_){
_start:
{
switch(lean_obj_tag(v_x_65_))
{
case 0:
{
lean_object* v___x_66_; 
v___x_66_ = lean_unsigned_to_nat(0u);
return v___x_66_;
}
case 1:
{
lean_object* v___x_67_; 
v___x_67_ = lean_unsigned_to_nat(1u);
return v___x_67_;
}
default: 
{
lean_object* v___x_68_; 
v___x_68_ = lean_unsigned_to_nat(2u);
return v___x_68_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorIdx___boxed(lean_object* v_x_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorIdx(v_x_69_);
lean_dec_ref(v_x_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(lean_object* v_t_71_, lean_object* v_k_72_){
_start:
{
switch(lean_obj_tag(v_t_71_))
{
case 0:
{
lean_object* v_a_73_; lean_object* v_a_74_; lean_object* v_a_75_; lean_object* v___x_76_; 
v_a_73_ = lean_ctor_get(v_t_71_, 0);
lean_inc(v_a_73_);
v_a_74_ = lean_ctor_get(v_t_71_, 1);
lean_inc(v_a_74_);
v_a_75_ = lean_ctor_get(v_t_71_, 2);
lean_inc(v_a_75_);
lean_dec_ref_known(v_t_71_, 3);
v___x_76_ = lean_apply_3(v_k_72_, v_a_73_, v_a_74_, v_a_75_);
return v___x_76_;
}
case 1:
{
lean_object* v_a_77_; lean_object* v___x_78_; 
v_a_77_ = lean_ctor_get(v_t_71_, 0);
lean_inc(v_a_77_);
lean_dec_ref_known(v_t_71_, 1);
v___x_78_ = lean_apply_1(v_k_72_, v_a_77_);
return v___x_78_;
}
default: 
{
lean_object* v_a_79_; lean_object* v_a_80_; lean_object* v_a_81_; lean_object* v_a_82_; lean_object* v___x_83_; 
v_a_79_ = lean_ctor_get(v_t_71_, 0);
lean_inc(v_a_79_);
v_a_80_ = lean_ctor_get(v_t_71_, 1);
lean_inc(v_a_80_);
v_a_81_ = lean_ctor_get(v_t_71_, 2);
lean_inc(v_a_81_);
v_a_82_ = lean_ctor_get(v_t_71_, 3);
lean_inc(v_a_82_);
lean_dec_ref_known(v_t_71_, 4);
v___x_83_ = lean_apply_4(v_k_72_, v_a_79_, v_a_80_, v_a_81_, v_a_82_);
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim(lean_object* v_motive_84_, lean_object* v_ctorIdx_85_, lean_object* v_t_86_, lean_object* v_h_87_, lean_object* v_k_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_86_, v_k_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___boxed(lean_object* v_motive_90_, lean_object* v_ctorIdx_91_, lean_object* v_t_92_, lean_object* v_h_93_, lean_object* v_k_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim(v_motive_90_, v_ctorIdx_91_, v_t_92_, v_h_93_, v_k_94_);
lean_dec(v_ctorIdx_91_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_element_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object* v_t_96_, lean_object* v_ProofWidgets_RpcEncodablePacket_element_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_96_, v_ProofWidgets_RpcEncodablePacket_element_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_element_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object* v_motive_99_, lean_object* v_t_100_, lean_object* v_h_101_, lean_object* v_ProofWidgets_RpcEncodablePacket_element_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_100_, v_ProofWidgets_RpcEncodablePacket_element_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_text_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object* v_t_104_, lean_object* v_ProofWidgets_RpcEncodablePacket_text_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_104_, v_ProofWidgets_RpcEncodablePacket_text_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_text_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object* v_motive_107_, lean_object* v_t_108_, lean_object* v_h_109_, lean_object* v_ProofWidgets_RpcEncodablePacket_text_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_108_, v_ProofWidgets_RpcEncodablePacket_text_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_component_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim___redArg(lean_object* v_t_112_, lean_object* v_ProofWidgets_RpcEncodablePacket_component_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_112_, v_ProofWidgets_RpcEncodablePacket_component_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_component_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__elim(lean_object* v_motive_115_, lean_object* v_t_116_, lean_object* v_h_117_, lean_object* v_ProofWidgets_RpcEncodablePacket_component_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__ctorElim___redArg(v_t_116_, v_ProofWidgets_RpcEncodablePacket_component_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_(lean_object* v_json_129_){
_start:
{
lean_object* v___x_130_; 
lean_inc(v_json_129_);
v___x_130_ = l_Lean_Json_getTag_x3f(v_json_129_);
if (lean_obj_tag(v___x_130_) == 0)
{
lean_object* v___x_131_; 
lean_dec(v_json_129_);
v___x_131_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
return v___x_131_;
}
else
{
lean_object* v_val_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_222_; 
v_val_132_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_222_ == 0)
{
v___x_134_ = v___x_130_;
v_isShared_135_ = v_isSharedCheck_222_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_val_132_);
lean_dec(v___x_130_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_222_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_136_; lean_object* v___x_137_; uint8_t v___x_138_; 
v___x_136_ = lean_box(0);
v___x_137_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_138_ = lean_string_dec_eq(v_val_132_, v___x_137_);
if (v___x_138_ == 0)
{
lean_object* v___x_139_; uint8_t v___x_140_; 
lean_del_object(v___x_134_);
v___x_139_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_140_ = lean_string_dec_eq(v_val_132_, v___x_139_);
if (v___x_140_ == 0)
{
lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_141_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_142_ = lean_string_dec_eq(v_val_132_, v___x_141_);
lean_dec(v_val_132_);
if (v___x_142_ == 0)
{
lean_object* v___x_143_; 
lean_dec(v_json_129_);
v___x_143_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
return v___x_143_;
}
else
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_144_ = lean_unsigned_to_nat(4u);
v___x_145_ = lean_box(0);
v___x_146_ = l_Lean_Json_parseCtorFields(v_json_129_, v___x_141_, v___x_144_, v___x_145_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
v_a_147_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_146_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_146_);
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
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_171_; 
v_a_155_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_171_ == 0)
{
v___x_157_ = v___x_146_;
v_isShared_158_ = v_isSharedCheck_171_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_a_155_);
lean_dec(v___x_146_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_171_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_169_; 
v___x_159_ = lean_unsigned_to_nat(0u);
v___x_160_ = lean_array_get(v___x_136_, v_a_155_, v___x_159_);
v___x_161_ = lean_unsigned_to_nat(1u);
v___x_162_ = lean_array_get(v___x_136_, v_a_155_, v___x_161_);
v___x_163_ = lean_unsigned_to_nat(2u);
v___x_164_ = lean_array_get(v___x_136_, v_a_155_, v___x_163_);
v___x_165_ = lean_unsigned_to_nat(3u);
v___x_166_ = lean_array_get(v___x_136_, v_a_155_, v___x_165_);
lean_dec(v_a_155_);
v___x_167_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_167_, 0, v___x_160_);
lean_ctor_set(v___x_167_, 1, v___x_162_);
lean_ctor_set(v___x_167_, 2, v___x_164_);
lean_ctor_set(v___x_167_, 3, v___x_166_);
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v___x_167_);
v___x_169_ = v___x_157_;
goto v_reusejp_168_;
}
else
{
lean_object* v_reuseFailAlloc_170_; 
v_reuseFailAlloc_170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_170_, 0, v___x_167_);
v___x_169_ = v_reuseFailAlloc_170_;
goto v_reusejp_168_;
}
v_reusejp_168_:
{
return v___x_169_;
}
}
}
}
}
else
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec(v_val_132_);
v___x_172_ = lean_unsigned_to_nat(3u);
v___x_173_ = lean_box(0);
v___x_174_ = l_Lean_Json_parseCtorFields(v_json_129_, v___x_139_, v___x_172_, v___x_173_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v_a_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_182_; 
v_a_175_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_182_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_182_ == 0)
{
v___x_177_ = v___x_174_;
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_a_175_);
lean_dec(v___x_174_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_180_; 
if (v_isShared_178_ == 0)
{
v___x_180_ = v___x_177_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v_a_175_);
v___x_180_ = v_reuseFailAlloc_181_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
return v___x_180_;
}
}
}
else
{
lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_197_; 
v_a_183_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_197_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_197_ == 0)
{
v___x_185_ = v___x_174_;
v_isShared_186_ = v_isSharedCheck_197_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___x_174_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_197_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_195_; 
v___x_187_ = lean_unsigned_to_nat(0u);
v___x_188_ = lean_array_get(v___x_136_, v_a_183_, v___x_187_);
v___x_189_ = lean_unsigned_to_nat(1u);
v___x_190_ = lean_array_get(v___x_136_, v_a_183_, v___x_189_);
v___x_191_ = lean_unsigned_to_nat(2u);
v___x_192_ = lean_array_get(v___x_136_, v_a_183_, v___x_191_);
lean_dec(v_a_183_);
v___x_193_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_193_, 0, v___x_188_);
lean_ctor_set(v___x_193_, 1, v___x_190_);
lean_ctor_set(v___x_193_, 2, v___x_192_);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 0, v___x_193_);
v___x_195_ = v___x_185_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_193_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
lean_dec(v_val_132_);
v___x_198_ = lean_unsigned_to_nat(1u);
v___x_199_ = lean_box(0);
v___x_200_ = l_Lean_Json_parseCtorFields(v_json_129_, v___x_137_, v___x_198_, v___x_199_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_del_object(v___x_134_);
v_a_201_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_200_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_200_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_221_; 
v_a_209_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_221_ == 0)
{
v___x_211_ = v___x_200_;
v_isShared_212_ = v_isSharedCheck_221_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_200_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_221_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_216_; 
v___x_213_ = lean_unsigned_to_nat(0u);
v___x_214_ = lean_array_get(v___x_136_, v_a_209_, v___x_213_);
lean_dec(v_a_209_);
if (v_isShared_135_ == 0)
{
lean_ctor_set(v___x_134_, 0, v___x_214_);
v___x_216_ = v___x_134_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v___x_214_);
v___x_216_ = v_reuseFailAlloc_220_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
lean_object* v___x_218_; 
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 0, v___x_216_);
v___x_218_ = v___x_211_;
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
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_(lean_object* v_x_225_){
_start:
{
switch(lean_obj_tag(v_x_225_))
{
case 0:
{
lean_object* v_a_226_; lean_object* v_a_227_; lean_object* v_a_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v_a_226_ = lean_ctor_get(v_x_225_, 0);
lean_inc(v_a_226_);
v_a_227_ = lean_ctor_get(v_x_225_, 1);
lean_inc(v_a_227_);
v_a_228_ = lean_ctor_get(v_x_225_, 2);
lean_inc(v_a_228_);
lean_dec_ref_known(v_x_225_, 3);
v___x_229_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_230_ = lean_unsigned_to_nat(3u);
v___x_231_ = lean_mk_empty_array_with_capacity(v___x_230_);
v___x_232_ = lean_array_push(v___x_231_, v_a_226_);
v___x_233_ = lean_array_push(v___x_232_, v_a_227_);
v___x_234_ = lean_array_push(v___x_233_, v_a_228_);
v___x_235_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
v___x_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_229_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
v___x_237_ = lean_box(0);
v___x_238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_236_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = l_Lean_Json_mkObj(v___x_238_);
lean_dec_ref_known(v___x_238_, 2);
return v___x_239_;
}
case 1:
{
lean_object* v_a_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v_a_240_ = lean_ctor_get(v_x_225_, 0);
lean_inc(v_a_240_);
lean_dec_ref_known(v_x_225_, 1);
v___x_241_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v_a_240_);
v___x_243_ = lean_box(0);
v___x_244_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_242_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = l_Lean_Json_mkObj(v___x_244_);
lean_dec_ref_known(v___x_244_, 2);
return v___x_245_;
}
default: 
{
lean_object* v_a_246_; lean_object* v_a_247_; lean_object* v_a_248_; lean_object* v_a_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v_a_246_ = lean_ctor_get(v_x_225_, 0);
lean_inc(v_a_246_);
v_a_247_ = lean_ctor_get(v_x_225_, 1);
lean_inc(v_a_247_);
v_a_248_ = lean_ctor_get(v_x_225_, 2);
lean_inc(v_a_248_);
v_a_249_ = lean_ctor_get(v_x_225_, 3);
lean_inc(v_a_249_);
lean_dec_ref_known(v_x_225_, 4);
v___x_250_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_));
v___x_251_ = lean_unsigned_to_nat(4u);
v___x_252_ = lean_mk_empty_array_with_capacity(v___x_251_);
v___x_253_ = lean_array_push(v___x_252_, v_a_246_);
v___x_254_ = lean_array_push(v___x_253_, v_a_247_);
v___x_255_ = lean_array_push(v___x_254_, v_a_248_);
v___x_256_ = lean_array_push(v___x_255_, v_a_249_);
v___x_257_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_250_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
v___x_259_ = lean_box(0);
v___x_260_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_258_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = l_Lean_Json_mkObj(v___x_260_);
lean_dec_ref_known(v___x_260_, 2);
return v___x_261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Prod_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__0(lean_object* v_x_264_){
_start:
{
lean_object* v_fst_265_; lean_object* v_snd_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v_fst_265_ = lean_ctor_get(v_x_264_, 0);
lean_inc(v_fst_265_);
v_snd_266_ = lean_ctor_get(v_x_264_, 1);
lean_inc(v_snd_266_);
lean_dec_ref(v_x_264_);
v___x_267_ = lean_unsigned_to_nat(2u);
v___x_268_ = lean_mk_empty_array_with_capacity(v___x_267_);
v___x_269_ = lean_array_push(v___x_268_, v_fst_265_);
v___x_270_ = lean_array_push(v___x_269_, v_snd_266_);
v___x_271_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1(size_t v_sz_272_, size_t v_i_273_, lean_object* v_bs_274_, lean_object* v___y_275_){
_start:
{
uint8_t v___x_276_; 
v___x_276_ = lean_usize_dec_lt(v_i_273_, v_sz_272_);
if (v___x_276_ == 0)
{
lean_object* v___x_277_; 
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v_bs_274_);
lean_ctor_set(v___x_277_, 1, v___y_275_);
return v___x_277_;
}
else
{
lean_object* v_v_278_; lean_object* v_fst_279_; lean_object* v_snd_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_295_; 
v_v_278_ = lean_array_uget(v_bs_274_, v_i_273_);
v_fst_279_ = lean_ctor_get(v_v_278_, 0);
v_snd_280_ = lean_ctor_get(v_v_278_, 1);
v_isSharedCheck_295_ = !lean_is_exclusive(v_v_278_);
if (v_isSharedCheck_295_ == 0)
{
v___x_282_ = v_v_278_;
v_isShared_283_ = v_isSharedCheck_295_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_snd_280_);
lean_inc(v_fst_279_);
lean_dec(v_v_278_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_295_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_284_; lean_object* v_bs_x27_285_; lean_object* v___x_286_; lean_object* v___x_288_; 
v___x_284_ = lean_unsigned_to_nat(0u);
v_bs_x27_285_ = lean_array_uset(v_bs_274_, v_i_273_, v___x_284_);
v___x_286_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_286_, 0, v_fst_279_);
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 0, v___x_286_);
v___x_288_ = v___x_282_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v___x_286_);
lean_ctor_set(v_reuseFailAlloc_294_, 1, v_snd_280_);
v___x_288_ = v_reuseFailAlloc_294_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
lean_object* v___x_289_; size_t v___x_290_; size_t v___x_291_; lean_object* v___x_292_; 
v___x_289_ = lp_proofwidgets_Lean_Prod_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__0(v___x_288_);
v___x_290_ = ((size_t)1ULL);
v___x_291_ = lean_usize_add(v_i_273_, v___x_290_);
v___x_292_ = lean_array_uset(v_bs_x27_285_, v_i_273_, v___x_289_);
v_i_273_ = v___x_291_;
v_bs_274_ = v___x_292_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1___boxed(lean_object* v_sz_296_, lean_object* v_i_297_, lean_object* v_bs_298_, lean_object* v___y_299_){
_start:
{
size_t v_sz_boxed_300_; size_t v_i_boxed_301_; lean_object* v_res_302_; 
v_sz_boxed_300_ = lean_unbox_usize(v_sz_296_);
lean_dec(v_sz_296_);
v_i_boxed_301_ = lean_unbox_usize(v_i_297_);
lean_dec(v_i_297_);
v_res_302_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1(v_sz_boxed_300_, v_i_boxed_301_, v_bs_298_, v___y_299_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3(size_t v_sz_303_, size_t v_i_304_, lean_object* v_bs_305_){
_start:
{
uint8_t v___x_306_; 
v___x_306_ = lean_usize_dec_lt(v_i_304_, v_sz_303_);
if (v___x_306_ == 0)
{
return v_bs_305_;
}
else
{
lean_object* v_v_307_; lean_object* v___x_308_; lean_object* v_bs_x27_309_; size_t v___x_310_; size_t v___x_311_; lean_object* v___x_312_; 
v_v_307_ = lean_array_uget(v_bs_305_, v_i_304_);
v___x_308_ = lean_unsigned_to_nat(0u);
v_bs_x27_309_ = lean_array_uset(v_bs_305_, v_i_304_, v___x_308_);
v___x_310_ = ((size_t)1ULL);
v___x_311_ = lean_usize_add(v_i_304_, v___x_310_);
v___x_312_ = lean_array_uset(v_bs_x27_309_, v_i_304_, v_v_307_);
v_i_304_ = v___x_311_;
v_bs_305_ = v___x_312_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3___boxed(lean_object* v_sz_314_, lean_object* v_i_315_, lean_object* v_bs_316_){
_start:
{
size_t v_sz_boxed_317_; size_t v_i_boxed_318_; lean_object* v_res_319_; 
v_sz_boxed_317_ = lean_unbox_usize(v_sz_314_);
lean_dec(v_sz_314_);
v_i_boxed_318_ = lean_unbox_usize(v_i_315_);
lean_dec(v_i_315_);
v_res_319_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3(v_sz_boxed_317_, v_i_boxed_318_, v_bs_316_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3(lean_object* v_a_320_){
_start:
{
size_t v_sz_321_; size_t v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
v_sz_321_ = lean_array_size(v_a_320_);
v___x_322_ = ((size_t)0ULL);
v___x_323_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3_spec__3(v_sz_321_, v___x_322_, v_a_320_);
v___x_324_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_324_, 0, v___x_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object* v_x_325_, lean_object* v_a_326_){
_start:
{
switch(lean_obj_tag(v_x_325_))
{
case 0:
{
lean_object* v_a_327_; lean_object* v_a_328_; lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_356_; 
v_a_327_ = lean_ctor_get(v_x_325_, 0);
v_a_328_ = lean_ctor_get(v_x_325_, 1);
v_a_329_ = lean_ctor_get(v_x_325_, 2);
v_isSharedCheck_356_ = !lean_is_exclusive(v_x_325_);
if (v_isSharedCheck_356_ == 0)
{
v___x_331_ = v_x_325_;
v_isShared_332_ = v_isSharedCheck_356_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_inc(v_a_328_);
lean_inc(v_a_327_);
lean_dec(v_x_325_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_356_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
size_t v_sz_333_; size_t v___x_334_; lean_object* v___x_335_; lean_object* v_fst_336_; lean_object* v_snd_337_; size_t v_sz_338_; lean_object* v___x_339_; lean_object* v_fst_340_; lean_object* v_snd_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_355_; 
v_sz_333_ = lean_array_size(v_a_328_);
v___x_334_ = ((size_t)0ULL);
v___x_335_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__1(v_sz_333_, v___x_334_, v_a_328_, v_a_326_);
v_fst_336_ = lean_ctor_get(v___x_335_, 0);
lean_inc(v_fst_336_);
v_snd_337_ = lean_ctor_get(v___x_335_, 1);
lean_inc(v_snd_337_);
lean_dec_ref(v___x_335_);
v_sz_338_ = lean_array_size(v_a_329_);
v___x_339_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2(v_sz_338_, v___x_334_, v_a_329_, v_snd_337_);
v_fst_340_ = lean_ctor_get(v___x_339_, 0);
v_snd_341_ = lean_ctor_get(v___x_339_, 1);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_355_ == 0)
{
v___x_343_ = v___x_339_;
v_isShared_344_ = v_isSharedCheck_355_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_snd_341_);
lean_inc(v_fst_340_);
lean_dec(v___x_339_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_355_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_349_; 
v___x_345_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_345_, 0, v_a_327_);
v___x_346_ = lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3(v_fst_336_);
v___x_347_ = lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3(v_fst_340_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 2, v___x_347_);
lean_ctor_set(v___x_331_, 1, v___x_346_);
lean_ctor_set(v___x_331_, 0, v___x_345_);
v___x_349_ = v___x_331_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_345_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_354_, 2, v___x_347_);
v___x_349_ = v_reuseFailAlloc_354_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
lean_object* v___x_350_; lean_object* v___x_352_; 
v___x_350_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_(v___x_349_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_350_);
v___x_352_ = v___x_343_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_353_, 1, v_snd_341_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
}
case 1:
{
lean_object* v_a_357_; lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_367_; 
v_a_357_ = lean_ctor_get(v_x_325_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v_x_325_);
if (v_isSharedCheck_367_ == 0)
{
v___x_359_ = v_x_325_;
v_isShared_360_ = v_isSharedCheck_367_;
goto v_resetjp_358_;
}
else
{
lean_inc(v_a_357_);
lean_dec(v_x_325_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_367_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
lean_object* v___x_362_; 
if (v_isShared_360_ == 0)
{
lean_ctor_set_tag(v___x_359_, 3);
v___x_362_ = v___x_359_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_357_);
v___x_362_ = v_reuseFailAlloc_366_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
v___x_364_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_(v___x_363_);
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
lean_ctor_set(v___x_365_, 1, v_a_326_);
return v___x_365_;
}
}
}
default: 
{
uint64_t v_a_368_; lean_object* v_a_369_; lean_object* v_a_370_; lean_object* v_a_371_; lean_object* v___x_372_; lean_object* v_fst_373_; lean_object* v_snd_374_; size_t v_sz_375_; size_t v___x_376_; lean_object* v___x_377_; lean_object* v_fst_378_; lean_object* v_snd_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_392_; 
v_a_368_ = lean_ctor_get_uint64(v_x_325_, sizeof(void*)*3);
v_a_369_ = lean_ctor_get(v_x_325_, 0);
lean_inc_ref(v_a_369_);
v_a_370_ = lean_ctor_get(v_x_325_, 1);
lean_inc_ref(v_a_370_);
v_a_371_ = lean_ctor_get(v_x_325_, 2);
lean_inc_ref(v_a_371_);
lean_dec_ref_known(v_x_325_, 3);
v___x_372_ = lean_apply_1(v_a_370_, v_a_326_);
v_fst_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc(v_fst_373_);
v_snd_374_ = lean_ctor_get(v___x_372_, 1);
lean_inc(v_snd_374_);
lean_dec_ref(v___x_372_);
v_sz_375_ = lean_array_size(v_a_371_);
v___x_376_ = ((size_t)0ULL);
v___x_377_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2(v_sz_375_, v___x_376_, v_a_371_, v_snd_374_);
v_fst_378_ = lean_ctor_get(v___x_377_, 0);
v_snd_379_ = lean_ctor_get(v___x_377_, 1);
v_isSharedCheck_392_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_392_ == 0)
{
v___x_381_ = v___x_377_;
v_isShared_382_ = v_isSharedCheck_392_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_snd_379_);
lean_inc(v_fst_378_);
lean_dec(v___x_377_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_392_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
v___x_383_ = lean_uint64_to_nat(v_a_368_);
v___x_384_ = l_Lean_bignumToJson(v___x_383_);
v___x_385_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_385_, 0, v_a_369_);
v___x_386_ = lp_proofwidgets_Lean_Array_toJson___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__3(v_fst_378_);
v___x_387_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_387_, 0, v___x_384_);
lean_ctor_set(v___x_387_, 1, v___x_385_);
lean_ctor_set(v___x_387_, 2, v_fst_373_);
lean_ctor_set(v___x_387_, 3, v___x_386_);
v___x_388_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_59_(v___x_387_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 0, v___x_388_);
v___x_390_ = v___x_381_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_388_);
lean_ctor_set(v_reuseFailAlloc_391_, 1, v_snd_379_);
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
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2(size_t v_sz_393_, size_t v_i_394_, lean_object* v_bs_395_, lean_object* v___y_396_){
_start:
{
uint8_t v___x_397_; 
v___x_397_ = lean_usize_dec_lt(v_i_394_, v_sz_393_);
if (v___x_397_ == 0)
{
lean_object* v___x_398_; 
v___x_398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_398_, 0, v_bs_395_);
lean_ctor_set(v___x_398_, 1, v___y_396_);
return v___x_398_;
}
else
{
lean_object* v_v_399_; lean_object* v___x_400_; lean_object* v_fst_401_; lean_object* v_snd_402_; lean_object* v___x_403_; lean_object* v_bs_x27_404_; size_t v___x_405_; size_t v___x_406_; lean_object* v___x_407_; 
v_v_399_ = lean_array_uget_borrowed(v_bs_395_, v_i_394_);
lean_inc(v_v_399_);
v___x_400_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_v_399_, v___y_396_);
v_fst_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_fst_401_);
v_snd_402_ = lean_ctor_get(v___x_400_, 1);
lean_inc(v_snd_402_);
lean_dec_ref(v___x_400_);
v___x_403_ = lean_unsigned_to_nat(0u);
v_bs_x27_404_ = lean_array_uset(v_bs_395_, v_i_394_, v___x_403_);
v___x_405_ = ((size_t)1ULL);
v___x_406_ = lean_usize_add(v_i_394_, v___x_405_);
v___x_407_ = lean_array_uset(v_bs_x27_404_, v_i_394_, v_fst_401_);
v_i_394_ = v___x_406_;
v_bs_395_ = v___x_407_;
v___y_396_ = v_snd_402_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2___boxed(lean_object* v_sz_409_, lean_object* v_i_410_, lean_object* v_bs_411_, lean_object* v___y_412_){
_start:
{
size_t v_sz_boxed_413_; size_t v_i_boxed_414_; lean_object* v_res_415_; 
v_sz_boxed_413_ = lean_unbox_usize(v_sz_409_);
lean_dec(v_sz_409_);
v_i_boxed_414_ = lean_unbox_usize(v_i_410_);
lean_dec(v_i_410_);
v_res_415_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__2(v_sz_boxed_413_, v_i_boxed_414_, v_bs_411_, v___y_412_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___redArg(lean_object* v_x_416_){
_start:
{
lean_inc_ref(v_x_416_);
return v_x_416_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___redArg___boxed(lean_object* v_x_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___redArg(v_x_417_);
lean_dec_ref(v_x_417_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5(lean_object* v_00_u03b1_419_, lean_object* v_x_420_, lean_object* v___y_421_){
_start:
{
lean_inc_ref(v_x_420_);
return v_x_420_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5___boxed(lean_object* v_00_u03b1_422_, lean_object* v_x_423_, lean_object* v___y_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__5(v_00_u03b1_422_, v_x_423_, v___y_424_);
lean_dec_ref(v___y_424_);
lean_dec_ref(v_x_423_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7(lean_object* v_x_428_){
_start:
{
lean_object* v_j_430_; 
if (lean_obj_tag(v_x_428_) == 4)
{
lean_object* v_elems_438_; lean_object* v___x_439_; lean_object* v___x_440_; uint8_t v___x_441_; 
v_elems_438_ = lean_ctor_get(v_x_428_, 0);
v___x_439_ = lean_array_get_size(v_elems_438_);
v___x_440_ = lean_unsigned_to_nat(2u);
v___x_441_ = lean_nat_dec_eq(v___x_439_, v___x_440_);
if (v___x_441_ == 0)
{
v_j_430_ = v_x_428_;
goto v___jp_429_;
}
else
{
lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_453_; 
lean_inc_ref(v_elems_438_);
v_isSharedCheck_453_ = !lean_is_exclusive(v_x_428_);
if (v_isSharedCheck_453_ == 0)
{
lean_object* v_unused_454_; 
v_unused_454_ = lean_ctor_get(v_x_428_, 0);
lean_dec(v_unused_454_);
v___x_443_ = v_x_428_;
v_isShared_444_ = v_isSharedCheck_453_;
goto v_resetjp_442_;
}
else
{
lean_dec(v_x_428_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_453_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_451_; 
v___x_445_ = lean_unsigned_to_nat(0u);
v___x_446_ = lean_array_fget(v_elems_438_, v___x_445_);
v___x_447_ = lean_unsigned_to_nat(1u);
v___x_448_ = lean_array_fget(v_elems_438_, v___x_447_);
lean_dec_ref(v_elems_438_);
v___x_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_449_, 0, v___x_446_);
lean_ctor_set(v___x_449_, 1, v___x_448_);
if (v_isShared_444_ == 0)
{
lean_ctor_set_tag(v___x_443_, 1);
lean_ctor_set(v___x_443_, 0, v___x_449_);
v___x_451_ = v___x_443_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v___x_449_);
v___x_451_ = v_reuseFailAlloc_452_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
return v___x_451_;
}
}
}
}
else
{
v_j_430_ = v_x_428_;
goto v___jp_429_;
}
v___jp_429_:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_431_ = ((lean_object*)(lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__0));
v___x_432_ = lean_unsigned_to_nat(80u);
v___x_433_ = l_Lean_Json_pretty(v_j_430_, v___x_432_);
v___x_434_ = lean_string_append(v___x_431_, v___x_433_);
lean_dec_ref(v___x_433_);
v___x_435_ = ((lean_object*)(lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__1));
v___x_436_ = lean_string_append(v___x_434_, v___x_435_);
v___x_437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_437_, 0, v___x_436_);
return v___x_437_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec___lam__0_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object* v_a_455_, lean_object* v___y_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_457_, 0, v_a_455_);
lean_ctor_set(v___x_457_, 1, v___y_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7(size_t v_sz_458_, size_t v_i_459_, lean_object* v_bs_460_){
_start:
{
uint8_t v___x_461_; 
v___x_461_ = lean_usize_dec_lt(v_i_459_, v_sz_458_);
if (v___x_461_ == 0)
{
lean_object* v___x_462_; 
v___x_462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_462_, 0, v_bs_460_);
return v___x_462_;
}
else
{
lean_object* v_v_463_; lean_object* v___x_464_; lean_object* v_bs_x27_465_; size_t v___x_466_; size_t v___x_467_; lean_object* v___x_468_; 
v_v_463_ = lean_array_uget(v_bs_460_, v_i_459_);
v___x_464_ = lean_unsigned_to_nat(0u);
v_bs_x27_465_ = lean_array_uset(v_bs_460_, v_i_459_, v___x_464_);
v___x_466_ = ((size_t)1ULL);
v___x_467_ = lean_usize_add(v_i_459_, v___x_466_);
v___x_468_ = lean_array_uset(v_bs_x27_465_, v_i_459_, v_v_463_);
v_i_459_ = v___x_467_;
v_bs_460_ = v___x_468_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7___boxed(lean_object* v_sz_470_, lean_object* v_i_471_, lean_object* v_bs_472_){
_start:
{
size_t v_sz_boxed_473_; size_t v_i_boxed_474_; lean_object* v_res_475_; 
v_sz_boxed_473_ = lean_unbox_usize(v_sz_470_);
lean_dec(v_sz_470_);
v_i_boxed_474_ = lean_unbox_usize(v_i_471_);
lean_dec(v_i_471_);
v_res_475_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7(v_sz_boxed_473_, v_i_boxed_474_, v_bs_472_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6(lean_object* v_x_477_){
_start:
{
if (lean_obj_tag(v_x_477_) == 4)
{
lean_object* v_elems_478_; size_t v_sz_479_; size_t v___x_480_; lean_object* v___x_481_; 
v_elems_478_ = lean_ctor_get(v_x_477_, 0);
lean_inc_ref(v_elems_478_);
lean_dec_ref_known(v_x_477_, 1);
v_sz_479_ = lean_array_size(v_elems_478_);
v___x_480_ = ((size_t)0ULL);
v___x_481_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6_spec__7(v_sz_479_, v___x_480_, v_elems_478_);
return v___x_481_;
}
else
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_482_ = ((lean_object*)(lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6___closed__0));
v___x_483_ = lean_unsigned_to_nat(80u);
v___x_484_ = l_Lean_Json_pretty(v_x_477_, v___x_483_);
v___x_485_ = lean_string_append(v___x_482_, v___x_484_);
lean_dec_ref(v___x_484_);
v___x_486_ = ((lean_object*)(lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7___closed__1));
v___x_487_ = lean_string_append(v___x_485_, v___x_486_);
v___x_488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_488_, 0, v___x_487_);
return v___x_488_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg(size_t v_sz_489_, size_t v_i_490_, lean_object* v_bs_491_){
_start:
{
uint8_t v___x_492_; 
v___x_492_ = lean_usize_dec_lt(v_i_490_, v_sz_489_);
if (v___x_492_ == 0)
{
lean_object* v___x_493_; 
v___x_493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_493_, 0, v_bs_491_);
return v___x_493_;
}
else
{
lean_object* v_v_494_; lean_object* v___x_495_; 
v_v_494_ = lean_array_uget_borrowed(v_bs_491_, v_i_490_);
lean_inc(v_v_494_);
v___x_495_ = lp_proofwidgets_Lean_Prod_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__7(v_v_494_);
if (lean_obj_tag(v___x_495_) == 0)
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_503_; 
lean_dec_ref(v_bs_491_);
v_a_496_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_503_ == 0)
{
v___x_498_ = v___x_495_;
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_495_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_501_; 
if (v_isShared_499_ == 0)
{
v___x_501_ = v___x_498_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v_a_496_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
else
{
lean_object* v_a_504_; lean_object* v_fst_505_; lean_object* v_snd_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_529_; 
v_a_504_ = lean_ctor_get(v___x_495_, 0);
lean_inc(v_a_504_);
lean_dec_ref_known(v___x_495_, 1);
v_fst_505_ = lean_ctor_get(v_a_504_, 0);
v_snd_506_ = lean_ctor_get(v_a_504_, 1);
v_isSharedCheck_529_ = !lean_is_exclusive(v_a_504_);
if (v_isSharedCheck_529_ == 0)
{
v___x_508_ = v_a_504_;
v_isShared_509_ = v_isSharedCheck_529_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_snd_506_);
lean_inc(v_fst_505_);
lean_dec(v_a_504_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_529_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; 
v___x_510_ = l_Lean_Json_getStr_x3f(v_fst_505_);
if (lean_obj_tag(v___x_510_) == 0)
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_del_object(v___x_508_);
lean_dec(v_snd_506_);
lean_dec_ref(v_bs_491_);
v_a_511_ = lean_ctor_get(v___x_510_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_510_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_510_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_510_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
else
{
lean_object* v_a_519_; lean_object* v___x_520_; lean_object* v_bs_x27_521_; lean_object* v___x_523_; 
v_a_519_ = lean_ctor_get(v___x_510_, 0);
lean_inc(v_a_519_);
lean_dec_ref_known(v___x_510_, 1);
v___x_520_ = lean_unsigned_to_nat(0u);
v_bs_x27_521_ = lean_array_uset(v_bs_491_, v_i_490_, v___x_520_);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v_a_519_);
v___x_523_ = v___x_508_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v_a_519_);
lean_ctor_set(v_reuseFailAlloc_528_, 1, v_snd_506_);
v___x_523_ = v_reuseFailAlloc_528_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
size_t v___x_524_; size_t v___x_525_; lean_object* v___x_526_; 
v___x_524_ = ((size_t)1ULL);
v___x_525_ = lean_usize_add(v_i_490_, v___x_524_);
v___x_526_ = lean_array_uset(v_bs_x27_521_, v_i_490_, v___x_523_);
v_i_490_ = v___x_525_;
v_bs_491_ = v___x_526_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg___boxed(lean_object* v_sz_530_, lean_object* v_i_531_, lean_object* v_bs_532_){
_start:
{
size_t v_sz_boxed_533_; size_t v_i_boxed_534_; lean_object* v_res_535_; 
v_sz_boxed_533_ = lean_unbox_usize(v_sz_530_);
lean_dec(v_sz_530_);
v_i_boxed_534_ = lean_unbox_usize(v_i_531_);
lean_dec(v_i_531_);
v_res_535_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg(v_sz_boxed_533_, v_i_boxed_534_, v_bs_532_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object* v_j_536_, lean_object* v_a_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Data_Html_2463204861____hygCtx___hyg_32_(v_j_536_);
if (lean_obj_tag(v___x_538_) == 0)
{
lean_object* v_a_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_546_; 
v_a_539_ = lean_ctor_get(v___x_538_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_538_);
if (v_isSharedCheck_546_ == 0)
{
v___x_541_ = v___x_538_;
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_a_539_);
lean_dec(v___x_538_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_544_; 
if (v_isShared_542_ == 0)
{
v___x_544_ = v___x_541_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_a_539_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
else
{
lean_object* v_a_547_; 
v_a_547_ = lean_ctor_get(v___x_538_, 0);
lean_inc(v_a_547_);
lean_dec_ref_known(v___x_538_, 1);
switch(lean_obj_tag(v_a_547_))
{
case 0:
{
lean_object* v_a_548_; lean_object* v_a_549_; lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_617_; 
v_a_548_ = lean_ctor_get(v_a_547_, 0);
v_a_549_ = lean_ctor_get(v_a_547_, 1);
v_a_550_ = lean_ctor_get(v_a_547_, 2);
v_isSharedCheck_617_ = !lean_is_exclusive(v_a_547_);
if (v_isSharedCheck_617_ == 0)
{
v___x_552_ = v_a_547_;
v_isShared_553_ = v_isSharedCheck_617_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_inc(v_a_549_);
lean_inc(v_a_548_);
lean_dec(v_a_547_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_617_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
lean_object* v___x_554_; 
v___x_554_ = l_Lean_Json_getStr_x3f(v_a_548_);
if (lean_obj_tag(v___x_554_) == 0)
{
lean_object* v_a_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_562_; 
lean_del_object(v___x_552_);
lean_dec(v_a_550_);
lean_dec(v_a_549_);
v_a_555_ = lean_ctor_get(v___x_554_, 0);
v_isSharedCheck_562_ = !lean_is_exclusive(v___x_554_);
if (v_isSharedCheck_562_ == 0)
{
v___x_557_ = v___x_554_;
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_a_555_);
lean_dec(v___x_554_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_560_; 
if (v_isShared_558_ == 0)
{
v___x_560_ = v___x_557_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_a_555_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
else
{
lean_object* v_a_563_; lean_object* v___x_564_; 
v_a_563_ = lean_ctor_get(v___x_554_, 0);
lean_inc(v_a_563_);
lean_dec_ref_known(v___x_554_, 1);
v___x_564_ = lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6(v_a_549_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v_a_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_572_; 
lean_dec(v_a_563_);
lean_del_object(v___x_552_);
lean_dec(v_a_550_);
v_a_565_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_572_ == 0)
{
v___x_567_ = v___x_564_;
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_a_565_);
lean_dec(v___x_564_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v___x_570_; 
if (v_isShared_568_ == 0)
{
v___x_570_ = v___x_567_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_565_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
return v___x_570_;
}
}
}
else
{
lean_object* v_a_573_; size_t v_sz_574_; size_t v___x_575_; lean_object* v___x_576_; 
v_a_573_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_573_);
lean_dec_ref_known(v___x_564_, 1);
v_sz_574_ = lean_array_size(v_a_573_);
v___x_575_ = ((size_t)0ULL);
v___x_576_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg(v_sz_574_, v___x_575_, v_a_573_);
if (lean_obj_tag(v___x_576_) == 0)
{
lean_object* v_a_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_584_; 
lean_dec(v_a_563_);
lean_del_object(v___x_552_);
lean_dec(v_a_550_);
v_a_577_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_584_ == 0)
{
v___x_579_ = v___x_576_;
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_a_577_);
lean_dec(v___x_576_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v___x_582_; 
if (v_isShared_580_ == 0)
{
v___x_582_ = v___x_579_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v_a_577_);
v___x_582_ = v_reuseFailAlloc_583_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
return v___x_582_;
}
}
}
else
{
lean_object* v_a_585_; lean_object* v___x_586_; 
v_a_585_ = lean_ctor_get(v___x_576_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_576_, 1);
v___x_586_ = lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6(v_a_550_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v_a_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_594_; 
lean_dec(v_a_585_);
lean_dec(v_a_563_);
lean_del_object(v___x_552_);
v_a_587_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_594_ == 0)
{
v___x_589_ = v___x_586_;
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_a_587_);
lean_dec(v___x_586_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_592_; 
if (v_isShared_590_ == 0)
{
v___x_592_ = v___x_589_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v_a_587_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
else
{
lean_object* v_a_595_; size_t v_sz_596_; lean_object* v___x_597_; 
v_a_595_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_a_595_);
lean_dec_ref_known(v___x_586_, 1);
v_sz_596_ = lean_array_size(v_a_595_);
v___x_597_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9(v_sz_596_, v___x_575_, v_a_595_, v_a_537_);
if (lean_obj_tag(v___x_597_) == 0)
{
lean_object* v_a_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_605_; 
lean_dec(v_a_585_);
lean_dec(v_a_563_);
lean_del_object(v___x_552_);
v_a_598_ = lean_ctor_get(v___x_597_, 0);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_597_);
if (v_isSharedCheck_605_ == 0)
{
v___x_600_ = v___x_597_;
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_a_598_);
lean_dec(v___x_597_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_a_598_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
else
{
lean_object* v_a_606_; lean_object* v___x_608_; uint8_t v_isShared_609_; uint8_t v_isSharedCheck_616_; 
v_a_606_ = lean_ctor_get(v___x_597_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_597_);
if (v_isSharedCheck_616_ == 0)
{
v___x_608_ = v___x_597_;
v_isShared_609_ = v_isSharedCheck_616_;
goto v_resetjp_607_;
}
else
{
lean_inc(v_a_606_);
lean_dec(v___x_597_);
v___x_608_ = lean_box(0);
v_isShared_609_ = v_isSharedCheck_616_;
goto v_resetjp_607_;
}
v_resetjp_607_:
{
lean_object* v___x_611_; 
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 2, v_a_606_);
lean_ctor_set(v___x_552_, 1, v_a_585_);
lean_ctor_set(v___x_552_, 0, v_a_563_);
v___x_611_ = v___x_552_;
goto v_reusejp_610_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v_a_563_);
lean_ctor_set(v_reuseFailAlloc_615_, 1, v_a_585_);
lean_ctor_set(v_reuseFailAlloc_615_, 2, v_a_606_);
v___x_611_ = v_reuseFailAlloc_615_;
goto v_reusejp_610_;
}
v_reusejp_610_:
{
lean_object* v___x_613_; 
if (v_isShared_609_ == 0)
{
lean_ctor_set(v___x_608_, 0, v___x_611_);
v___x_613_ = v___x_608_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v___x_611_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
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
case 1:
{
lean_object* v_a_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_642_; 
v_a_618_ = lean_ctor_get(v_a_547_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v_a_547_);
if (v_isSharedCheck_642_ == 0)
{
v___x_620_ = v_a_547_;
v_isShared_621_ = v_isSharedCheck_642_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_a_618_);
lean_dec(v_a_547_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_642_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_622_; 
v___x_622_ = l_Lean_Json_getStr_x3f(v_a_618_);
if (lean_obj_tag(v___x_622_) == 0)
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_del_object(v___x_620_);
v_a_623_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
else
{
lean_object* v_a_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_641_; 
v_a_631_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_641_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_641_ == 0)
{
v___x_633_ = v___x_622_;
v_isShared_634_ = v_isSharedCheck_641_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_a_631_);
lean_dec(v___x_622_);
v___x_633_ = lean_box(0);
v_isShared_634_ = v_isSharedCheck_641_;
goto v_resetjp_632_;
}
v_resetjp_632_:
{
lean_object* v___x_636_; 
if (v_isShared_621_ == 0)
{
lean_ctor_set(v___x_620_, 0, v_a_631_);
v___x_636_ = v___x_620_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_640_; 
v_reuseFailAlloc_640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_640_, 0, v_a_631_);
v___x_636_ = v_reuseFailAlloc_640_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
lean_object* v___x_638_; 
if (v_isShared_634_ == 0)
{
lean_ctor_set(v___x_633_, 0, v___x_636_);
v___x_638_ = v___x_633_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v___x_636_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
}
}
default: 
{
lean_object* v_a_643_; lean_object* v_a_644_; lean_object* v_a_645_; lean_object* v_a_646_; lean_object* v___x_647_; 
v_a_643_ = lean_ctor_get(v_a_547_, 0);
lean_inc(v_a_643_);
v_a_644_ = lean_ctor_get(v_a_547_, 1);
lean_inc(v_a_644_);
v_a_645_ = lean_ctor_get(v_a_547_, 2);
lean_inc(v_a_645_);
v_a_646_ = lean_ctor_get(v_a_547_, 3);
lean_inc(v_a_646_);
lean_dec_ref_known(v_a_547_, 4);
v___x_647_ = l_Lean_UInt64_fromJson_x3f(v_a_643_);
if (lean_obj_tag(v___x_647_) == 0)
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
lean_dec(v_a_646_);
lean_dec(v_a_645_);
lean_dec(v_a_644_);
v_a_648_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_647_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_647_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_657_; 
v_a_656_ = lean_ctor_get(v___x_647_, 0);
lean_inc(v_a_656_);
lean_dec_ref_known(v___x_647_, 1);
v___x_657_ = l_Lean_Json_getStr_x3f(v_a_644_);
if (lean_obj_tag(v___x_657_) == 0)
{
lean_object* v_a_658_; lean_object* v___x_660_; uint8_t v_isShared_661_; uint8_t v_isSharedCheck_665_; 
lean_dec(v_a_656_);
lean_dec(v_a_646_);
lean_dec(v_a_645_);
v_a_658_ = lean_ctor_get(v___x_657_, 0);
v_isSharedCheck_665_ = !lean_is_exclusive(v___x_657_);
if (v_isSharedCheck_665_ == 0)
{
v___x_660_ = v___x_657_;
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
else
{
lean_inc(v_a_658_);
lean_dec(v___x_657_);
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
v_reuseFailAlloc_664_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_666_; lean_object* v___x_667_; 
v_a_666_ = lean_ctor_get(v___x_657_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___x_657_, 1);
v___x_667_ = lp_proofwidgets_Lean_Array_fromJson_x3f___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__6(v_a_646_);
if (lean_obj_tag(v___x_667_) == 0)
{
lean_object* v_a_668_; lean_object* v___x_670_; uint8_t v_isShared_671_; uint8_t v_isSharedCheck_675_; 
lean_dec(v_a_666_);
lean_dec(v_a_656_);
lean_dec(v_a_645_);
v_a_668_ = lean_ctor_get(v___x_667_, 0);
v_isSharedCheck_675_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_675_ == 0)
{
v___x_670_ = v___x_667_;
v_isShared_671_ = v_isSharedCheck_675_;
goto v_resetjp_669_;
}
else
{
lean_inc(v_a_668_);
lean_dec(v___x_667_);
v___x_670_ = lean_box(0);
v_isShared_671_ = v_isSharedCheck_675_;
goto v_resetjp_669_;
}
v_resetjp_669_:
{
lean_object* v___x_673_; 
if (v_isShared_671_ == 0)
{
v___x_673_ = v___x_670_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v_a_668_);
v___x_673_ = v_reuseFailAlloc_674_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
return v___x_673_;
}
}
}
else
{
lean_object* v_a_676_; size_t v_sz_677_; size_t v___x_678_; lean_object* v___x_679_; 
v_a_676_ = lean_ctor_get(v___x_667_, 0);
lean_inc(v_a_676_);
lean_dec_ref_known(v___x_667_, 1);
v_sz_677_ = lean_array_size(v_a_676_);
v___x_678_ = ((size_t)0ULL);
v___x_679_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9(v_sz_677_, v___x_678_, v_a_676_, v_a_537_);
if (lean_obj_tag(v___x_679_) == 0)
{
lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_687_; 
lean_dec(v_a_666_);
lean_dec(v_a_656_);
lean_dec(v_a_645_);
v_a_680_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_687_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_687_ == 0)
{
v___x_682_ = v___x_679_;
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_679_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_685_; 
if (v_isShared_683_ == 0)
{
v___x_685_ = v___x_682_;
goto v_reusejp_684_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v_a_680_);
v___x_685_ = v_reuseFailAlloc_686_;
goto v_reusejp_684_;
}
v_reusejp_684_:
{
return v___x_685_;
}
}
}
else
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_698_; 
v_a_688_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_698_ == 0)
{
v___x_690_ = v___x_679_;
v_isShared_691_ = v_isSharedCheck_698_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_679_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_698_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___f_692_; lean_object* v___x_693_; uint64_t v___x_694_; lean_object* v___x_696_; 
v___f_692_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec___lam__0_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_), 2, 1);
lean_closure_set(v___f_692_, 0, v_a_645_);
v___x_693_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_693_, 0, v_a_666_);
lean_ctor_set(v___x_693_, 1, v___f_692_);
lean_ctor_set(v___x_693_, 2, v_a_688_);
v___x_694_ = lean_unbox_uint64(v_a_656_);
lean_dec(v_a_656_);
lean_ctor_set_uint64(v___x_693_, sizeof(void*)*3, v___x_694_);
if (v_isShared_691_ == 0)
{
lean_ctor_set(v___x_690_, 0, v___x_693_);
v___x_696_ = v___x_690_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_693_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
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
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9(size_t v_sz_699_, size_t v_i_700_, lean_object* v_bs_701_, lean_object* v___y_702_){
_start:
{
uint8_t v___x_703_; 
v___x_703_ = lean_usize_dec_lt(v_i_700_, v_sz_699_);
if (v___x_703_ == 0)
{
lean_object* v___x_704_; 
v___x_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_704_, 0, v_bs_701_);
return v___x_704_;
}
else
{
lean_object* v_v_705_; lean_object* v___x_706_; 
v_v_705_ = lean_array_uget_borrowed(v_bs_701_, v_i_700_);
lean_inc(v_v_705_);
v___x_706_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_v_705_, v___y_702_);
if (lean_obj_tag(v___x_706_) == 0)
{
lean_object* v_a_707_; lean_object* v___x_709_; uint8_t v_isShared_710_; uint8_t v_isSharedCheck_714_; 
lean_dec_ref(v_bs_701_);
v_a_707_ = lean_ctor_get(v___x_706_, 0);
v_isSharedCheck_714_ = !lean_is_exclusive(v___x_706_);
if (v_isSharedCheck_714_ == 0)
{
v___x_709_ = v___x_706_;
v_isShared_710_ = v_isSharedCheck_714_;
goto v_resetjp_708_;
}
else
{
lean_inc(v_a_707_);
lean_dec(v___x_706_);
v___x_709_ = lean_box(0);
v_isShared_710_ = v_isSharedCheck_714_;
goto v_resetjp_708_;
}
v_resetjp_708_:
{
lean_object* v___x_712_; 
if (v_isShared_710_ == 0)
{
v___x_712_ = v___x_709_;
goto v_reusejp_711_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v_a_707_);
v___x_712_ = v_reuseFailAlloc_713_;
goto v_reusejp_711_;
}
v_reusejp_711_:
{
return v___x_712_;
}
}
}
else
{
lean_object* v_a_715_; lean_object* v___x_716_; lean_object* v_bs_x27_717_; size_t v___x_718_; size_t v___x_719_; lean_object* v___x_720_; 
v_a_715_ = lean_ctor_get(v___x_706_, 0);
lean_inc(v_a_715_);
lean_dec_ref_known(v___x_706_, 1);
v___x_716_ = lean_unsigned_to_nat(0u);
v_bs_x27_717_ = lean_array_uset(v_bs_701_, v_i_700_, v___x_716_);
v___x_718_ = ((size_t)1ULL);
v___x_719_ = lean_usize_add(v_i_700_, v___x_718_);
v___x_720_ = lean_array_uset(v_bs_x27_717_, v_i_700_, v_a_715_);
v_i_700_ = v___x_719_;
v_bs_701_ = v___x_720_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9___boxed(lean_object* v_sz_722_, lean_object* v_i_723_, lean_object* v_bs_724_, lean_object* v___y_725_){
_start:
{
size_t v_sz_boxed_726_; size_t v_i_boxed_727_; lean_object* v_res_728_; 
v_sz_boxed_726_ = lean_unbox_usize(v_sz_722_);
lean_dec(v_sz_722_);
v_i_boxed_727_ = lean_unbox_usize(v_i_723_);
lean_dec(v_i_723_);
v_res_728_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__9(v_sz_boxed_726_, v_i_boxed_727_, v_bs_724_, v___y_725_);
lean_dec_ref(v___y_725_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1____boxed(lean_object* v_j_729_, lean_object* v_a_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_j_729_, v_a_730_);
lean_dec_ref(v_a_730_);
return v_res_731_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8(size_t v_sz_732_, size_t v_i_733_, lean_object* v_bs_734_, lean_object* v___y_735_){
_start:
{
lean_object* v___x_736_; 
v___x_736_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___redArg(v_sz_732_, v_i_733_, v_bs_734_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8___boxed(lean_object* v_sz_737_, lean_object* v_i_738_, lean_object* v_bs_739_, lean_object* v___y_740_){
_start:
{
size_t v_sz_boxed_741_; size_t v_i_boxed_742_; lean_object* v_res_743_; 
v_sz_boxed_741_ = lean_unbox_usize(v_sz_737_);
lean_dec(v_sz_737_);
v_i_boxed_742_ = lean_unbox_usize(v_i_738_);
lean_dec(v_i_738_);
v_res_743_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1__spec__8(v_sz_boxed_741_, v_i_boxed_742_, v_bs_739_, v___y_740_);
lean_dec_ref(v___y_740_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(lean_object* v_inst_750_, lean_object* v_c_751_, lean_object* v_props_752_, lean_object* v_children_753_){
_start:
{
lean_object* v_toModule_754_; lean_object* v_export_755_; lean_object* v_javascript_756_; lean_object* v_rpcEncode_757_; uint64_t v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v_toModule_754_ = lean_ctor_get(v_c_751_, 0);
v_export_755_ = lean_ctor_get(v_c_751_, 1);
v_javascript_756_ = lean_ctor_get(v_toModule_754_, 0);
v_rpcEncode_757_ = lean_ctor_get(v_inst_750_, 0);
lean_inc_ref(v_rpcEncode_757_);
lean_dec_ref(v_inst_750_);
v___x_758_ = lean_string_hash(v_javascript_756_);
v___x_759_ = lean_apply_1(v_rpcEncode_757_, v_props_752_);
lean_inc_ref(v_export_755_);
v___x_760_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_760_, 0, v_export_755_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
lean_ctor_set(v___x_760_, 2, v_children_753_);
lean_ctor_set_uint64(v___x_760_, sizeof(void*)*3, v___x_758_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg___boxed(lean_object* v_inst_761_, lean_object* v_c_762_, lean_object* v_props_763_, lean_object* v_children_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(v_inst_761_, v_c_762_, v_props_763_, v_children_764_);
lean_dec_ref(v_c_762_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent(lean_object* v_Props_766_, lean_object* v_inst_767_, lean_object* v_c_768_, lean_object* v_props_769_, lean_object* v_children_770_){
_start:
{
lean_object* v___x_771_; 
v___x_771_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(v_inst_767_, v_c_768_, v_props_769_, v_children_770_);
return v___x_771_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___boxed(lean_object* v_Props_772_, lean_object* v_inst_773_, lean_object* v_c_774_, lean_object* v_props_775_, lean_object* v_children_776_){
_start:
{
lean_object* v_res_777_; 
v_res_777_ = lp_proofwidgets_ProofWidgets_Html_ofComponent(v_Props_772_, v_inst_773_, v_c_774_, v_props_775_, v_children_776_);
lean_dec_ref(v_c_774_);
return v_res_777_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorIdx(uint8_t v_x_778_){
_start:
{
if (v_x_778_ == 0)
{
lean_object* v___x_779_; 
v___x_779_ = lean_unsigned_to_nat(0u);
return v___x_779_;
}
else
{
lean_object* v___x_780_; 
v___x_780_ = lean_unsigned_to_nat(1u);
return v___x_780_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorIdx___boxed(lean_object* v_x_781_){
_start:
{
uint8_t v_x_boxed_782_; lean_object* v_res_783_; 
v_x_boxed_782_ = lean_unbox(v_x_781_);
v_res_783_ = lp_proofwidgets_ProofWidgets_LayoutKind_ctorIdx(v_x_boxed_782_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___redArg(lean_object* v_k_784_){
_start:
{
lean_inc(v_k_784_);
return v_k_784_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___redArg___boxed(lean_object* v_k_785_){
_start:
{
lean_object* v_res_786_; 
v_res_786_ = lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___redArg(v_k_785_);
lean_dec(v_k_785_);
return v_res_786_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim(lean_object* v_motive_787_, lean_object* v_ctorIdx_788_, uint8_t v_t_789_, lean_object* v_h_790_, lean_object* v_k_791_){
_start:
{
lean_inc(v_k_791_);
return v_k_791_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim___boxed(lean_object* v_motive_792_, lean_object* v_ctorIdx_793_, lean_object* v_t_794_, lean_object* v_h_795_, lean_object* v_k_796_){
_start:
{
uint8_t v_t_boxed_797_; lean_object* v_res_798_; 
v_t_boxed_797_ = lean_unbox(v_t_794_);
v_res_798_ = lp_proofwidgets_ProofWidgets_LayoutKind_ctorElim(v_motive_792_, v_ctorIdx_793_, v_t_boxed_797_, v_h_795_, v_k_796_);
lean_dec(v_k_796_);
lean_dec(v_ctorIdx_793_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___redArg(lean_object* v_block_799_){
_start:
{
lean_inc(v_block_799_);
return v_block_799_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___redArg___boxed(lean_object* v_block_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___redArg(v_block_800_);
lean_dec(v_block_800_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim(lean_object* v_motive_802_, uint8_t v_t_803_, lean_object* v_h_804_, lean_object* v_block_805_){
_start:
{
lean_inc(v_block_805_);
return v_block_805_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_block_elim___boxed(lean_object* v_motive_806_, lean_object* v_t_807_, lean_object* v_h_808_, lean_object* v_block_809_){
_start:
{
uint8_t v_t_boxed_810_; lean_object* v_res_811_; 
v_t_boxed_810_ = lean_unbox(v_t_807_);
v_res_811_ = lp_proofwidgets_ProofWidgets_LayoutKind_block_elim(v_motive_806_, v_t_boxed_810_, v_h_808_, v_block_809_);
lean_dec(v_block_809_);
return v_res_811_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___redArg(lean_object* v_inline_812_){
_start:
{
lean_inc(v_inline_812_);
return v_inline_812_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___redArg___boxed(lean_object* v_inline_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___redArg(v_inline_813_);
lean_dec(v_inline_813_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim(lean_object* v_motive_815_, uint8_t v_t_816_, lean_object* v_h_817_, lean_object* v_inline_818_){
_start:
{
lean_inc(v_inline_818_);
return v_inline_818_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim___boxed(lean_object* v_motive_819_, lean_object* v_t_820_, lean_object* v_h_821_, lean_object* v_inline_822_){
_start:
{
uint8_t v_t_boxed_823_; lean_object* v_res_824_; 
v_t_boxed_823_ = lean_unbox(v_t_820_);
v_res_824_ = lp_proofwidgets_ProofWidgets_LayoutKind_inline_elim(v_motive_819_, v_t_boxed_823_, v_h_821_, v_inline_822_);
lean_dec(v_inline_822_);
return v_res_824_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxElement(void){
_start:
{
lean_object* v___x_869_; 
v___x_869_ = lean_box(0);
return v___x_869_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxChild(void){
_start:
{
lean_object* v___x_899_; 
v___x_899_ = lean_box(0);
return v___x_899_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttr(void){
_start:
{
lean_object* v___x_929_; 
v___x_929_ = lean_box(0);
return v___x_929_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttrVal(void){
_start:
{
lean_object* v___x_959_; 
v___x_959_ = lean_box(0);
return v___x_959_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0(lean_object* v___y_1065_){
_start:
{
lean_inc_ref(v___y_1065_);
return v___y_1065_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0___boxed(lean_object* v___y_1066_){
_start:
{
lean_object* v_res_1067_; 
v_res_1067_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__0(v___y_1066_);
lean_dec_ref(v___y_1066_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1(lean_object* v___y_1068_){
_start:
{
lean_inc(v___y_1068_);
return v___y_1068_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1___boxed(lean_object* v___y_1069_){
_start:
{
lean_object* v_res_1070_; 
v_res_1070_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__1(v___y_1069_);
lean_dec(v___y_1069_);
return v_res_1070_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg(lean_object* v_s_1071_, uint32_t v_c_1072_, lean_object* v_a_1073_, uint8_t v_b_1074_){
_start:
{
lean_object* v_str_1075_; lean_object* v_startInclusive_1076_; lean_object* v_endExclusive_1077_; lean_object* v___x_1078_; uint8_t v___x_1079_; 
v_str_1075_ = lean_ctor_get(v_s_1071_, 0);
v_startInclusive_1076_ = lean_ctor_get(v_s_1071_, 1);
v_endExclusive_1077_ = lean_ctor_get(v_s_1071_, 2);
v___x_1078_ = lean_nat_sub(v_endExclusive_1077_, v_startInclusive_1076_);
v___x_1079_ = lean_nat_dec_eq(v_a_1073_, v___x_1078_);
lean_dec(v___x_1078_);
if (v___x_1079_ == 0)
{
lean_object* v___x_1080_; uint32_t v___x_1081_; uint8_t v___x_1082_; 
v___x_1080_ = lean_nat_add(v_startInclusive_1076_, v_a_1073_);
lean_dec(v_a_1073_);
v___x_1081_ = lean_string_utf8_get_fast(v_str_1075_, v___x_1080_);
v___x_1082_ = lean_uint32_dec_eq(v___x_1081_, v_c_1072_);
if (v___x_1082_ == 0)
{
lean_object* v___x_1083_; lean_object* v___x_1084_; 
v___x_1083_ = lean_string_utf8_next_fast(v_str_1075_, v___x_1080_);
lean_dec(v___x_1080_);
v___x_1084_ = lean_nat_sub(v___x_1083_, v_startInclusive_1076_);
v_a_1073_ = v___x_1084_;
v_b_1074_ = v___x_1082_;
goto _start;
}
else
{
lean_dec(v___x_1080_);
return v___x_1082_;
}
}
else
{
lean_dec(v_a_1073_);
return v_b_1074_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg___boxed(lean_object* v_s_1086_, lean_object* v_c_1087_, lean_object* v_a_1088_, lean_object* v_b_1089_){
_start:
{
uint32_t v_c_boxed_1090_; uint8_t v_b_boxed_1091_; uint8_t v_res_1092_; lean_object* v_r_1093_; 
v_c_boxed_1090_ = lean_unbox_uint32(v_c_1087_);
lean_dec(v_c_1087_);
v_b_boxed_1091_ = lean_unbox(v_b_1089_);
v_res_1092_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg(v_s_1086_, v_c_boxed_1090_, v_a_1088_, v_b_boxed_1091_);
lean_dec_ref(v_s_1086_);
v_r_1093_ = lean_box(v_res_1092_);
return v_r_1093_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0(uint32_t v_c_1094_, lean_object* v_s_1095_){
_start:
{
lean_object* v_searcher_1096_; uint8_t v___x_1097_; uint8_t v___x_1098_; 
v_searcher_1096_ = lean_unsigned_to_nat(0u);
v___x_1097_ = 0;
v___x_1098_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg(v_s_1095_, v_c_1094_, v_searcher_1096_, v___x_1097_);
return v___x_1098_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0___boxed(lean_object* v_c_1099_, lean_object* v_s_1100_){
_start:
{
uint32_t v_c_boxed_1101_; uint8_t v_res_1102_; lean_object* v_r_1103_; 
v_c_boxed_1101_ = lean_unbox_uint32(v_c_1099_);
lean_dec(v_c_1099_);
v_res_1102_ = lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0(v_c_boxed_1101_, v_s_1100_);
lean_dec_ref(v_s_1100_);
v_r_1103_ = lean_box(v_res_1102_);
return v_r_1103_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0(void){
_start:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0));
v___x_1105_ = lean_string_utf8_byte_size(v___x_1104_);
return v___x_1105_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
v___x_1106_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0, &lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0_once, _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__0);
v___x_1107_ = lean_unsigned_to_nat(0u);
v___x_1108_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxTextForbidden___closed__0));
v___x_1109_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1109_, 0, v___x_1108_);
lean_ctor_set(v___x_1109_, 1, v___x_1107_);
lean_ctor_set(v___x_1109_, 2, v___x_1106_);
return v___x_1109_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2(uint32_t v_c_1110_){
_start:
{
lean_object* v___x_1111_; uint8_t v___x_1112_; 
v___x_1111_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1, &lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1_once, _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1);
v___x_1112_ = lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0(v_c_1110_, v___x_1111_);
if (v___x_1112_ == 0)
{
uint8_t v___x_1113_; 
v___x_1113_ = 1;
return v___x_1113_;
}
else
{
uint8_t v___x_1114_; 
v___x_1114_ = 0;
return v___x_1114_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___boxed(lean_object* v_c_1115_){
_start:
{
uint32_t v_c_boxed_1116_; uint8_t v_res_1117_; lean_object* v_r_1118_; 
v_c_boxed_1116_ = lean_unbox_uint32(v_c_1115_);
lean_dec(v_c_1115_);
v_res_1117_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2(v_c_boxed_1116_);
v_r_1118_ = lean_box(v_res_1117_);
return v_r_1118_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3(lean_object* v___f_1120_, lean_object* v___x_1121_, uint8_t v___x_1122_, lean_object* v_c_1123_, lean_object* v_s_1124_){
_start:
{
lean_object* v_pos_1125_; lean_object* v___x_1126_; lean_object* v_s_1127_; lean_object* v___x_1128_; 
v_pos_1125_ = lean_ctor_get(v_s_1124_, 2);
lean_inc(v_pos_1125_);
v___x_1126_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___closed__0));
lean_inc_ref(v_c_1123_);
v_s_1127_ = l_Lean_Parser_takeWhile1Fn(v___f_1120_, v___x_1126_, v_c_1123_, v_s_1124_);
v___x_1128_ = l_Lean_Parser_mkNodeToken(v___x_1121_, v_pos_1125_, v___x_1122_, v_c_1123_, v_s_1127_);
return v___x_1128_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3___boxed(lean_object* v___f_1129_, lean_object* v___x_1130_, lean_object* v___x_1131_, lean_object* v_c_1132_, lean_object* v_s_1133_){
_start:
{
uint8_t v___x_651__boxed_1134_; lean_object* v_res_1135_; 
v___x_651__boxed_1134_ = lean_unbox(v___x_1131_);
v_res_1135_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__3(v___f_1129_, v___x_1130_, v___x_651__boxed_1134_, v_c_1132_, v_s_1133_);
return v_res_1135_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6(void){
_start:
{
uint8_t v___x_1149_; uint8_t v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___x_1149_ = 0;
v___x_1150_ = 1;
v___x_1151_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4));
v___x_1152_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__3));
v___x_1153_ = l_Lean_Parser_mkAntiquot(v___x_1152_, v___x_1151_, v___x_1150_, v___x_1149_);
return v___x_1153_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9(void){
_start:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1161_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__8));
v___x_1162_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6, &lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6_once, _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__6);
v___x_1163_ = l_Lean_Parser_withAntiquot(v___x_1162_, v___x_1161_);
return v___x_1163_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText(void){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9, &lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9_once, _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__9);
return v___x_1164_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0(lean_object* v_s_1165_, uint32_t v_c_1166_, lean_object* v_inst_1167_, lean_object* v_R_1168_, lean_object* v_a_1169_, uint8_t v_b_1170_, lean_object* v_c_1171_){
_start:
{
uint8_t v___x_1172_; 
v___x_1172_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___redArg(v_s_1165_, v_c_1166_, v_a_1169_, v_b_1170_);
return v___x_1172_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0___boxed(lean_object* v_s_1173_, lean_object* v_c_1174_, lean_object* v_inst_1175_, lean_object* v_R_1176_, lean_object* v_a_1177_, lean_object* v_b_1178_, lean_object* v_c_1179_){
_start:
{
uint32_t v_c_boxed_1180_; uint8_t v_b_boxed_1181_; uint8_t v_res_1182_; lean_object* v_r_1183_; 
v_c_boxed_1180_ = lean_unbox_uint32(v_c_1174_);
lean_dec(v_c_1174_);
v_b_boxed_1181_ = lean_unbox(v_b_1178_);
v_res_1182_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0_spec__0(v_s_1173_, v_c_boxed_1180_, v_inst_1175_, v_R_1176_, v_a_1177_, v_b_boxed_1181_, v_c_1179_);
lean_dec_ref(v_s_1173_);
v_r_1183_ = lean_box(v_res_1182_);
return v_r_1183_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_getJsxText(lean_object* v_x_1184_){
_start:
{
lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; 
v___x_1185_ = lean_unsigned_to_nat(0u);
v___x_1186_ = l_Lean_Syntax_getArg(v_x_1184_, v___x_1185_);
v___x_1187_ = l_Lean_Syntax_getAtomVal(v___x_1186_);
lean_dec(v___x_1186_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_getJsxText___boxed(lean_object* v_x_1188_){
_start:
{
lean_object* v_res_1189_; 
v_res_1189_ = lp_proofwidgets_ProofWidgets_Jsx_getJsxText(v_x_1188_);
lean_dec(v_x_1188_);
return v_res_1189_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_formatter(lean_object* v_a_1190_, lean_object* v_a_1191_, lean_object* v_a_1192_, lean_object* v_a_1193_){
_start:
{
lean_object* v___x_1195_; lean_object* v___x_1196_; 
v___x_1195_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4));
v___x_1196_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1195_, v_a_1190_, v_a_1191_, v_a_1192_, v_a_1193_);
return v___x_1196_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_formatter___boxed(lean_object* v_a_1197_, lean_object* v_a_1198_, lean_object* v_a_1199_, lean_object* v_a_1200_, lean_object* v_a_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText_formatter(v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_);
lean_dec(v_a_1200_);
lean_dec_ref(v_a_1199_);
lean_dec(v_a_1198_);
lean_dec_ref(v_a_1197_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___redArg(lean_object* v_a_1203_){
_start:
{
lean_object* v___x_1205_; 
v___x_1205_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1203_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___redArg___boxed(lean_object* v_a_1206_, lean_object* v_a_1207_){
_start:
{
lean_object* v_res_1208_; 
v_res_1208_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___redArg(v_a_1206_);
lean_dec(v_a_1206_);
return v_res_1208_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer(lean_object* v_a_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_){
_start:
{
lean_object* v___x_1214_; 
v___x_1214_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1210_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer___boxed(lean_object* v_a_1215_, lean_object* v_a_1216_, lean_object* v_a_1217_, lean_object* v_a_1218_, lean_object* v_a_1219_){
_start:
{
lean_object* v_res_1220_; 
v_res_1220_ = lp_proofwidgets_ProofWidgets_Jsx_jsxText_parenthesizer(v_a_1215_, v_a_1216_, v_a_1217_, v_a_1218_);
lean_dec(v_a_1218_);
lean_dec_ref(v_a_1217_);
lean_dec(v_a_1216_);
lean_dec_ref(v_a_1215_);
return v_res_1220_;
}
}
static lean_object* _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7(void){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1373_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0));
v___x_1374_ = l_String_toRawSubstring_x27(v___x_1373_);
return v___x_1374_;
}
}
static lean_object* _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36(void){
_start:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; 
v___x_1437_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35));
v___x_1438_ = l_String_toRawSubstring_x27(v___x_1437_);
return v___x_1438_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2(size_t v_sz_1455_, size_t v_i_1456_, lean_object* v_bs_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_){
_start:
{
uint8_t v___x_1460_; 
v___x_1460_ = lean_usize_dec_lt(v_i_1456_, v_sz_1455_);
if (v___x_1460_ == 0)
{
lean_object* v___x_1461_; 
v___x_1461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1461_, 0, v_bs_1457_);
lean_ctor_set(v___x_1461_, 1, v___y_1459_);
return v___x_1461_;
}
else
{
lean_object* v_v_1462_; lean_object* v_fst_1463_; lean_object* v_snd_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1515_; 
v_v_1462_ = lean_array_uget(v_bs_1457_, v_i_1456_);
v_fst_1463_ = lean_ctor_get(v_v_1462_, 0);
v_snd_1464_ = lean_ctor_get(v_v_1462_, 1);
v_isSharedCheck_1515_ = !lean_is_exclusive(v_v_1462_);
if (v_isSharedCheck_1515_ == 0)
{
v___x_1466_ = v_v_1462_;
v_isShared_1467_ = v_isSharedCheck_1515_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_snd_1464_);
lean_inc(v_fst_1463_);
lean_dec(v_v_1462_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1515_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v_quotContext_1468_; lean_object* v_currMacroScope_1469_; lean_object* v_ref_1470_; uint8_t v___x_1471_; lean_object* v___x_1472_; lean_object* v_bs_x27_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1479_; 
v_quotContext_1468_ = lean_ctor_get(v___y_1458_, 1);
v_currMacroScope_1469_ = lean_ctor_get(v___y_1458_, 2);
v_ref_1470_ = lean_ctor_get(v___y_1458_, 5);
v___x_1471_ = 0;
v___x_1472_ = lean_unsigned_to_nat(0u);
v_bs_x27_1473_ = lean_array_uset(v_bs_1457_, v_i_1456_, v___x_1472_);
v___x_1474_ = l_Lean_SourceInfo_fromRef(v_ref_1470_, v___x_1471_);
v___x_1475_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__1));
v___x_1476_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__3));
v___x_1477_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__4));
lean_inc(v___x_1474_);
if (v_isShared_1467_ == 0)
{
lean_ctor_set_tag(v___x_1466_, 2);
lean_ctor_set(v___x_1466_, 1, v___x_1477_);
lean_ctor_set(v___x_1466_, 0, v___x_1474_);
v___x_1479_ = v___x_1466_;
goto v_reusejp_1478_;
}
else
{
lean_object* v_reuseFailAlloc_1514_; 
v_reuseFailAlloc_1514_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1514_, 0, v___x_1474_);
lean_ctor_set(v_reuseFailAlloc_1514_, 1, v___x_1477_);
v___x_1479_ = v_reuseFailAlloc_1514_;
goto v_reusejp_1478_;
}
v_reusejp_1478_:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; size_t v___x_1510_; size_t v___x_1511_; lean_object* v___x_1512_; 
v___x_1480_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__6));
v___x_1481_ = lean_obj_once(&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7, &lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7_once, _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__7);
v___x_1482_ = lean_box(0);
lean_inc_n(v_currMacroScope_1469_, 2);
lean_inc_n(v_quotContext_1468_, 2);
v___x_1483_ = l_Lean_addMacroScope(v_quotContext_1468_, v___x_1482_, v_currMacroScope_1469_);
v___x_1484_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__28));
lean_inc_n(v___x_1474_, 11);
v___x_1485_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1474_);
lean_ctor_set(v___x_1485_, 1, v___x_1481_);
lean_ctor_set(v___x_1485_, 2, v___x_1483_);
lean_ctor_set(v___x_1485_, 3, v___x_1484_);
v___x_1486_ = l_Lean_Syntax_node1(v___x_1474_, v___x_1480_, v___x_1485_);
v___x_1487_ = l_Lean_Syntax_node2(v___x_1474_, v___x_1476_, v___x_1479_, v___x_1486_);
v___x_1488_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1489_ = l_Lean_TSyntax_getId(v_fst_1463_);
lean_dec(v_fst_1463_);
v___x_1490_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_1489_, v___x_1460_);
v___x_1491_ = lean_box(2);
v___x_1492_ = l_Lean_Syntax_mkStrLit(v___x_1490_, v___x_1491_);
v___x_1493_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31));
v___x_1494_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1474_);
lean_ctor_set(v___x_1494_, 1, v___x_1493_);
v___x_1495_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__33));
v___x_1496_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__34));
v___x_1497_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1497_, 0, v___x_1474_);
lean_ctor_set(v___x_1497_, 1, v___x_1496_);
v___x_1498_ = lean_obj_once(&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36, &lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36_once, _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__36);
v___x_1499_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__37));
v___x_1500_ = l_Lean_addMacroScope(v_quotContext_1468_, v___x_1499_, v_currMacroScope_1469_);
v___x_1501_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__42));
v___x_1502_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1502_, 0, v___x_1474_);
lean_ctor_set(v___x_1502_, 1, v___x_1498_);
lean_ctor_set(v___x_1502_, 2, v___x_1500_);
lean_ctor_set(v___x_1502_, 3, v___x_1501_);
v___x_1503_ = l_Lean_Syntax_node1(v___x_1474_, v___x_1488_, v___x_1502_);
v___x_1504_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__13));
v___x_1505_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1505_, 0, v___x_1474_);
lean_ctor_set(v___x_1505_, 1, v___x_1504_);
lean_inc_ref(v___x_1505_);
lean_inc(v___x_1487_);
v___x_1506_ = l_Lean_Syntax_node5(v___x_1474_, v___x_1495_, v___x_1487_, v_snd_1464_, v___x_1497_, v___x_1503_, v___x_1505_);
v___x_1507_ = l_Lean_Syntax_node1(v___x_1474_, v___x_1488_, v___x_1506_);
v___x_1508_ = l_Lean_Syntax_node3(v___x_1474_, v___x_1488_, v___x_1492_, v___x_1494_, v___x_1507_);
v___x_1509_ = l_Lean_Syntax_node3(v___x_1474_, v___x_1475_, v___x_1487_, v___x_1508_, v___x_1505_);
v___x_1510_ = ((size_t)1ULL);
v___x_1511_ = lean_usize_add(v_i_1456_, v___x_1510_);
v___x_1512_ = lean_array_uset(v_bs_x27_1473_, v_i_1456_, v___x_1509_);
v_i_1456_ = v___x_1511_;
v_bs_1457_ = v___x_1512_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___boxed(lean_object* v_sz_1516_, lean_object* v_i_1517_, lean_object* v_bs_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_){
_start:
{
size_t v_sz_boxed_1521_; size_t v_i_boxed_1522_; lean_object* v_res_1523_; 
v_sz_boxed_1521_ = lean_unbox_usize(v_sz_1516_);
lean_dec(v_sz_1516_);
v_i_boxed_1522_ = lean_unbox_usize(v_i_1517_);
lean_dec(v_i_1517_);
v_res_1523_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2(v_sz_boxed_1521_, v_i_boxed_1522_, v_bs_1518_, v___y_1519_, v___y_1520_);
lean_dec_ref(v___y_1519_);
return v_res_1523_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1528_; 
v___x_1528_ = l_Array_mkArray0(lean_box(0));
return v___x_1528_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0(size_t v___x_1530_, uint8_t v___x_1531_, lean_object* v_vs_x27_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
size_t v_sz_1535_; lean_object* v___x_1536_; 
v_sz_1535_ = lean_array_size(v_vs_x27_1532_);
v___x_1536_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2(v_sz_1535_, v___x_1530_, v_vs_x27_1532_, v___y_1533_, v___y_1534_);
if (lean_obj_tag(v___x_1536_) == 0)
{
lean_object* v_a_1537_; lean_object* v_a_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1559_; 
v_a_1537_ = lean_ctor_get(v___x_1536_, 0);
v_a_1538_ = lean_ctor_get(v___x_1536_, 1);
v_isSharedCheck_1559_ = !lean_is_exclusive(v___x_1536_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1540_ = v___x_1536_;
v_isShared_1541_ = v_isSharedCheck_1559_;
goto v_resetjp_1539_;
}
else
{
lean_inc(v_a_1538_);
lean_inc(v_a_1537_);
lean_dec(v___x_1536_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1559_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
lean_object* v_ref_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1557_; 
v_ref_1542_ = lean_ctor_get(v___y_1533_, 5);
v___x_1543_ = l_Lean_SourceInfo_fromRef(v_ref_1542_, v___x_1531_);
v___x_1544_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1));
v___x_1545_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2));
lean_inc_n(v___x_1543_, 3);
v___x_1546_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1546_, 0, v___x_1543_);
lean_ctor_set(v___x_1546_, 1, v___x_1545_);
v___x_1547_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1548_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v___x_1549_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31));
v___x_1550_ = l_Lean_Syntax_SepArray_ofElems(v___x_1549_, v_a_1537_);
lean_dec(v_a_1537_);
v___x_1551_ = l_Array_append___redArg(v___x_1548_, v___x_1550_);
lean_dec_ref(v___x_1550_);
v___x_1552_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1543_);
lean_ctor_set(v___x_1552_, 1, v___x_1547_);
lean_ctor_set(v___x_1552_, 2, v___x_1551_);
v___x_1553_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4));
v___x_1554_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1554_, 0, v___x_1543_);
lean_ctor_set(v___x_1554_, 1, v___x_1553_);
v___x_1555_ = l_Lean_Syntax_node3(v___x_1543_, v___x_1544_, v___x_1546_, v___x_1552_, v___x_1554_);
if (v_isShared_1541_ == 0)
{
lean_ctor_set(v___x_1540_, 0, v___x_1555_);
v___x_1557_ = v___x_1540_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1555_);
lean_ctor_set(v_reuseFailAlloc_1558_, 1, v_a_1538_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
}
else
{
lean_object* v_a_1560_; lean_object* v_a_1561_; lean_object* v___x_1563_; uint8_t v_isShared_1564_; uint8_t v_isSharedCheck_1568_; 
v_a_1560_ = lean_ctor_get(v___x_1536_, 0);
v_a_1561_ = lean_ctor_get(v___x_1536_, 1);
v_isSharedCheck_1568_ = !lean_is_exclusive(v___x_1536_);
if (v_isSharedCheck_1568_ == 0)
{
v___x_1563_ = v___x_1536_;
v_isShared_1564_ = v_isSharedCheck_1568_;
goto v_resetjp_1562_;
}
else
{
lean_inc(v_a_1561_);
lean_inc(v_a_1560_);
lean_dec(v___x_1536_);
v___x_1563_ = lean_box(0);
v_isShared_1564_ = v_isSharedCheck_1568_;
goto v_resetjp_1562_;
}
v_resetjp_1562_:
{
lean_object* v___x_1566_; 
if (v_isShared_1564_ == 0)
{
v___x_1566_ = v___x_1563_;
goto v_reusejp_1565_;
}
else
{
lean_object* v_reuseFailAlloc_1567_; 
v_reuseFailAlloc_1567_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1567_, 0, v_a_1560_);
lean_ctor_set(v_reuseFailAlloc_1567_, 1, v_a_1561_);
v___x_1566_ = v_reuseFailAlloc_1567_;
goto v_reusejp_1565_;
}
v_reusejp_1565_:
{
return v___x_1566_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___boxed(lean_object* v___x_1569_, lean_object* v___x_1570_, lean_object* v_vs_x27_1571_, lean_object* v___y_1572_, lean_object* v___y_1573_){
_start:
{
size_t v___x_41928__boxed_1574_; uint8_t v___x_41929__boxed_1575_; lean_object* v_res_1576_; 
v___x_41928__boxed_1574_ = lean_unbox_usize(v___x_1569_);
lean_dec(v___x_1569_);
v___x_41929__boxed_1575_ = lean_unbox(v___x_1570_);
v_res_1576_ = lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0(v___x_41928__boxed_1574_, v___x_41929__boxed_1575_, v_vs_x27_1571_, v___y_1572_, v___y_1573_);
lean_dec_ref(v___y_1572_);
return v_res_1576_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(lean_object* v_stx_1577_){
_start:
{
lean_object* v___x_1578_; 
v___x_1578_ = l_Lean_Syntax_getTailInfo(v_stx_1577_);
if (lean_obj_tag(v___x_1578_) == 0)
{
lean_object* v_trailing_1579_; lean_object* v_str_1580_; lean_object* v_startPos_1581_; lean_object* v_stopPos_1582_; lean_object* v___x_1583_; 
v_trailing_1579_ = lean_ctor_get(v___x_1578_, 2);
lean_inc_ref(v_trailing_1579_);
lean_dec_ref_known(v___x_1578_, 4);
v_str_1580_ = lean_ctor_get(v_trailing_1579_, 0);
lean_inc_ref(v_str_1580_);
v_startPos_1581_ = lean_ctor_get(v_trailing_1579_, 1);
lean_inc(v_startPos_1581_);
v_stopPos_1582_ = lean_ctor_get(v_trailing_1579_, 2);
lean_inc(v_stopPos_1582_);
lean_dec_ref(v_trailing_1579_);
v___x_1583_ = lean_string_utf8_extract(v_str_1580_, v_startPos_1581_, v_stopPos_1582_);
lean_dec(v_stopPos_1582_);
lean_dec(v_startPos_1581_);
lean_dec_ref(v_str_1580_);
return v___x_1583_;
}
else
{
lean_object* v___x_1584_; 
lean_dec(v___x_1578_);
v___x_1584_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0));
return v___x_1584_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0___boxed(lean_object* v_stx_1585_){
_start:
{
lean_object* v_res_1586_; 
v_res_1586_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(v_stx_1585_);
lean_dec(v_stx_1585_);
return v_res_1586_;
}
}
static lean_object* _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5(void){
_start:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; 
v___x_1597_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__4));
v___x_1598_ = l_String_toRawSubstring_x27(v___x_1597_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6(lean_object* v_as_1618_, size_t v_sz_1619_, size_t v_i_1620_, lean_object* v_b_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v_a_1625_; lean_object* v_a_1626_; uint8_t v___x_1630_; 
v___x_1630_ = lean_usize_dec_lt(v_i_1620_, v_sz_1619_);
if (v___x_1630_ == 0)
{
lean_object* v___x_1631_; 
v___x_1631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1631_, 0, v_b_1621_);
lean_ctor_set(v___x_1631_, 1, v___y_1623_);
return v___x_1631_;
}
else
{
lean_object* v_snd_1632_; lean_object* v_fst_1633_; lean_object* v___x_1635_; uint8_t v_isShared_1636_; uint8_t v_isSharedCheck_1778_; 
v_snd_1632_ = lean_ctor_get(v_b_1621_, 1);
v_fst_1633_ = lean_ctor_get(v_b_1621_, 0);
v_isSharedCheck_1778_ = !lean_is_exclusive(v_b_1621_);
if (v_isSharedCheck_1778_ == 0)
{
v___x_1635_ = v_b_1621_;
v_isShared_1636_ = v_isSharedCheck_1778_;
goto v_resetjp_1634_;
}
else
{
lean_inc(v_snd_1632_);
lean_inc(v_fst_1633_);
lean_dec(v_b_1621_);
v___x_1635_ = lean_box(0);
v_isShared_1636_ = v_isSharedCheck_1778_;
goto v_resetjp_1634_;
}
v_resetjp_1634_:
{
lean_object* v_fst_1637_; lean_object* v_snd_1638_; lean_object* v___x_1640_; uint8_t v_isShared_1641_; uint8_t v_isSharedCheck_1777_; 
v_fst_1637_ = lean_ctor_get(v_snd_1632_, 0);
v_snd_1638_ = lean_ctor_get(v_snd_1632_, 1);
v_isSharedCheck_1777_ = !lean_is_exclusive(v_snd_1632_);
if (v_isSharedCheck_1777_ == 0)
{
v___x_1640_ = v_snd_1632_;
v_isShared_1641_ = v_isSharedCheck_1777_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_snd_1638_);
lean_inc(v_fst_1637_);
lean_dec(v_snd_1632_);
v___x_1640_ = lean_box(0);
v_isShared_1641_ = v_isSharedCheck_1777_;
goto v_resetjp_1639_;
}
v_resetjp_1639_:
{
lean_object* v___x_1642_; lean_object* v_a_1643_; lean_object* v___x_1644_; uint8_t v___x_1645_; 
v___x_1642_ = lean_unsigned_to_nat(0u);
v_a_1643_ = lean_array_uget_borrowed(v_as_1618_, v_i_1620_);
v___x_1644_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1));
lean_inc(v_a_1643_);
v___x_1645_ = l_Lean_Syntax_isOfKind(v_a_1643_, v___x_1644_);
if (v___x_1645_ == 0)
{
lean_object* v___x_1646_; uint8_t v___x_1647_; 
v___x_1646_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1));
lean_inc(v_a_1643_);
v___x_1647_ = l_Lean_Syntax_isOfKind(v_a_1643_, v___x_1646_);
if (v___x_1647_ == 0)
{
lean_object* v___x_1648_; uint8_t v___x_1649_; 
v___x_1648_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1));
lean_inc(v_a_1643_);
v___x_1649_ = l_Lean_Syntax_isOfKind(v_a_1643_, v___x_1648_);
if (v___x_1649_ == 0)
{
lean_object* v___x_1650_; uint8_t v___x_1651_; 
v___x_1650_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1));
lean_inc(v_a_1643_);
v___x_1651_ = l_Lean_Syntax_isOfKind(v_a_1643_, v___x_1650_);
if (v___x_1651_ == 0)
{
lean_object* v___x_1652_; lean_object* v___x_1653_; 
v___x_1652_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_1653_ = l_Lean_Macro_throwErrorAt___redArg(v_a_1643_, v___x_1652_, v___y_1622_, v___y_1623_);
if (lean_obj_tag(v___x_1653_) == 0)
{
lean_object* v_a_1654_; lean_object* v___x_1656_; 
v_a_1654_ = lean_ctor_get(v___x_1653_, 1);
lean_inc(v_a_1654_);
lean_dec_ref_known(v___x_1653_, 2);
if (v_isShared_1641_ == 0)
{
v___x_1656_ = v___x_1640_;
goto v_reusejp_1655_;
}
else
{
lean_object* v_reuseFailAlloc_1660_; 
v_reuseFailAlloc_1660_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1660_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1660_, 1, v_snd_1638_);
v___x_1656_ = v_reuseFailAlloc_1660_;
goto v_reusejp_1655_;
}
v_reusejp_1655_:
{
lean_object* v___x_1658_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1656_);
v___x_1658_ = v___x_1635_;
goto v_reusejp_1657_;
}
else
{
lean_object* v_reuseFailAlloc_1659_; 
v_reuseFailAlloc_1659_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1659_, 0, v_fst_1633_);
lean_ctor_set(v_reuseFailAlloc_1659_, 1, v___x_1656_);
v___x_1658_ = v_reuseFailAlloc_1659_;
goto v_reusejp_1657_;
}
v_reusejp_1657_:
{
v_a_1625_ = v___x_1658_;
v_a_1626_ = v_a_1654_;
goto v___jp_1624_;
}
}
}
else
{
lean_object* v_a_1661_; lean_object* v_a_1662_; lean_object* v___x_1664_; uint8_t v_isShared_1665_; uint8_t v_isSharedCheck_1669_; 
lean_del_object(v___x_1640_);
lean_dec(v_snd_1638_);
lean_dec(v_fst_1637_);
lean_del_object(v___x_1635_);
lean_dec(v_fst_1633_);
v_a_1661_ = lean_ctor_get(v___x_1653_, 0);
v_a_1662_ = lean_ctor_get(v___x_1653_, 1);
v_isSharedCheck_1669_ = !lean_is_exclusive(v___x_1653_);
if (v_isSharedCheck_1669_ == 0)
{
v___x_1664_ = v___x_1653_;
v_isShared_1665_ = v_isSharedCheck_1669_;
goto v_resetjp_1663_;
}
else
{
lean_inc(v_a_1662_);
lean_inc(v_a_1661_);
lean_dec(v___x_1653_);
v___x_1664_ = lean_box(0);
v_isShared_1665_ = v_isSharedCheck_1669_;
goto v_resetjp_1663_;
}
v_resetjp_1663_:
{
lean_object* v___x_1667_; 
if (v_isShared_1665_ == 0)
{
v___x_1667_ = v___x_1664_;
goto v_reusejp_1666_;
}
else
{
lean_object* v_reuseFailAlloc_1668_; 
v_reuseFailAlloc_1668_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1668_, 0, v_a_1661_);
lean_ctor_set(v_reuseFailAlloc_1668_, 1, v_a_1662_);
v___x_1667_ = v_reuseFailAlloc_1668_;
goto v_reusejp_1666_;
}
v_reusejp_1666_:
{
return v___x_1667_;
}
}
}
}
else
{
lean_object* v_csArrs_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v_csArrs_1676_; lean_object* v___y_1677_; uint8_t v___y_1687_; lean_object* v___x_1703_; uint8_t v___x_1704_; 
lean_dec(v_fst_1633_);
v_csArrs_1670_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1));
v___x_1671_ = lean_unsigned_to_nat(1u);
v___x_1672_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1671_);
v___x_1673_ = lean_unsigned_to_nat(2u);
v___x_1674_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1673_);
v___x_1703_ = lean_array_get_size(v_snd_1638_);
v___x_1704_ = lean_nat_dec_eq(v___x_1703_, v___x_1642_);
if (v___x_1704_ == 0)
{
v___y_1687_ = v___x_1651_;
goto v___jp_1686_;
}
else
{
v___y_1687_ = v___x_1649_;
goto v___jp_1686_;
}
v___jp_1675_:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1681_; 
v___x_1678_ = lean_array_push(v_csArrs_1676_, v___x_1672_);
v___x_1679_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(v___x_1674_);
lean_dec(v___x_1674_);
if (v_isShared_1641_ == 0)
{
lean_ctor_set(v___x_1640_, 1, v_csArrs_1670_);
lean_ctor_set(v___x_1640_, 0, v___x_1678_);
v___x_1681_ = v___x_1640_;
goto v_reusejp_1680_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v___x_1678_);
lean_ctor_set(v_reuseFailAlloc_1685_, 1, v_csArrs_1670_);
v___x_1681_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1680_;
}
v_reusejp_1680_:
{
lean_object* v___x_1683_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1681_);
lean_ctor_set(v___x_1635_, 0, v___x_1679_);
v___x_1683_ = v___x_1635_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1684_; 
v_reuseFailAlloc_1684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1684_, 0, v___x_1679_);
lean_ctor_set(v_reuseFailAlloc_1684_, 1, v___x_1681_);
v___x_1683_ = v_reuseFailAlloc_1684_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
v_a_1625_ = v___x_1683_;
v_a_1626_ = v___y_1677_;
goto v___jp_1624_;
}
}
}
v___jp_1686_:
{
if (v___y_1687_ == 0)
{
lean_dec(v_snd_1638_);
v_csArrs_1676_ = v_fst_1637_;
v___y_1677_ = v___y_1623_;
goto v___jp_1675_;
}
else
{
lean_object* v_ref_1688_; lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; 
v_ref_1688_ = lean_ctor_get(v___y_1622_, 5);
v___x_1689_ = l_Lean_SourceInfo_fromRef(v_ref_1688_, v___x_1649_);
v___x_1690_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1));
v___x_1691_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2));
lean_inc_n(v___x_1689_, 3);
v___x_1692_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1692_, 0, v___x_1689_);
lean_ctor_set(v___x_1692_, 1, v___x_1691_);
v___x_1693_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1694_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v___x_1695_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31));
v___x_1696_ = l_Lean_Syntax_SepArray_ofElems(v___x_1695_, v_snd_1638_);
lean_dec(v_snd_1638_);
v___x_1697_ = l_Array_append___redArg(v___x_1694_, v___x_1696_);
lean_dec_ref(v___x_1696_);
v___x_1698_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1698_, 0, v___x_1689_);
lean_ctor_set(v___x_1698_, 1, v___x_1693_);
lean_ctor_set(v___x_1698_, 2, v___x_1697_);
v___x_1699_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4));
v___x_1700_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1700_, 0, v___x_1689_);
lean_ctor_set(v___x_1700_, 1, v___x_1699_);
v___x_1701_ = l_Lean_Syntax_node3(v___x_1689_, v___x_1690_, v___x_1692_, v___x_1698_, v___x_1700_);
v___x_1702_ = lean_array_push(v_fst_1637_, v___x_1701_);
v_csArrs_1676_ = v___x_1702_;
v___y_1677_ = v___y_1623_;
goto v___jp_1675_;
}
}
}
}
else
{
lean_object* v_ref_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1713_; 
lean_dec(v_fst_1633_);
v_ref_1705_ = lean_ctor_get(v___y_1622_, 5);
v___x_1706_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1642_);
v___x_1707_ = l_Lean_SourceInfo_fromRef(v_ref_1705_, v___x_1647_);
v___x_1708_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1));
lean_inc(v___x_1706_);
v___x_1709_ = l_Lean_Syntax_node1(v___x_1707_, v___x_1708_, v___x_1706_);
v___x_1710_ = lean_array_push(v_snd_1638_, v___x_1709_);
v___x_1711_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(v___x_1706_);
lean_dec(v___x_1706_);
if (v_isShared_1641_ == 0)
{
lean_ctor_set(v___x_1640_, 1, v___x_1710_);
v___x_1713_ = v___x_1640_;
goto v_reusejp_1712_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1717_, 1, v___x_1710_);
v___x_1713_ = v_reuseFailAlloc_1717_;
goto v_reusejp_1712_;
}
v_reusejp_1712_:
{
lean_object* v___x_1715_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1713_);
lean_ctor_set(v___x_1635_, 0, v___x_1711_);
v___x_1715_ = v___x_1635_;
goto v_reusejp_1714_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v___x_1711_);
lean_ctor_set(v_reuseFailAlloc_1716_, 1, v___x_1713_);
v___x_1715_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1714_;
}
v_reusejp_1714_:
{
v_a_1625_ = v___x_1715_;
v_a_1626_ = v___y_1623_;
goto v___jp_1624_;
}
}
}
}
else
{
lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1725_; 
lean_dec(v_fst_1633_);
v___x_1718_ = lean_unsigned_to_nat(1u);
v___x_1719_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1718_);
v___x_1720_ = lean_unsigned_to_nat(2u);
v___x_1721_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1720_);
v___x_1722_ = lean_array_push(v_snd_1638_, v___x_1719_);
v___x_1723_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___lam__0(v___x_1721_);
lean_dec(v___x_1721_);
if (v_isShared_1641_ == 0)
{
lean_ctor_set(v___x_1640_, 1, v___x_1722_);
v___x_1725_ = v___x_1640_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1729_, 1, v___x_1722_);
v___x_1725_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
lean_object* v___x_1727_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1725_);
lean_ctor_set(v___x_1635_, 0, v___x_1723_);
v___x_1727_ = v___x_1635_;
goto v_reusejp_1726_;
}
else
{
lean_object* v_reuseFailAlloc_1728_; 
v_reuseFailAlloc_1728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1728_, 0, v___x_1723_);
lean_ctor_set(v_reuseFailAlloc_1728_, 1, v___x_1725_);
v___x_1727_ = v_reuseFailAlloc_1728_;
goto v_reusejp_1726_;
}
v_reusejp_1726_:
{
v_a_1625_ = v___x_1727_;
v_a_1626_ = v___y_1623_;
goto v___jp_1624_;
}
}
}
}
else
{
lean_object* v___x_1730_; lean_object* v___x_1731_; uint8_t v___x_1732_; 
v___x_1730_ = l_Lean_Syntax_getArg(v_a_1643_, v___x_1642_);
v___x_1731_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4));
lean_inc(v___x_1730_);
v___x_1732_ = l_Lean_Syntax_isOfKind(v___x_1730_, v___x_1731_);
if (v___x_1732_ == 0)
{
lean_object* v___x_1733_; lean_object* v___x_1734_; 
lean_dec(v___x_1730_);
v___x_1733_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_1734_ = l_Lean_Macro_throwErrorAt___redArg(v_a_1643_, v___x_1733_, v___y_1622_, v___y_1623_);
if (lean_obj_tag(v___x_1734_) == 0)
{
lean_object* v_a_1735_; lean_object* v___x_1737_; 
v_a_1735_ = lean_ctor_get(v___x_1734_, 1);
lean_inc(v_a_1735_);
lean_dec_ref_known(v___x_1734_, 2);
if (v_isShared_1641_ == 0)
{
v___x_1737_ = v___x_1640_;
goto v_reusejp_1736_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1741_, 1, v_snd_1638_);
v___x_1737_ = v_reuseFailAlloc_1741_;
goto v_reusejp_1736_;
}
v_reusejp_1736_:
{
lean_object* v___x_1739_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1737_);
v___x_1739_ = v___x_1635_;
goto v_reusejp_1738_;
}
else
{
lean_object* v_reuseFailAlloc_1740_; 
v_reuseFailAlloc_1740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1740_, 0, v_fst_1633_);
lean_ctor_set(v_reuseFailAlloc_1740_, 1, v___x_1737_);
v___x_1739_ = v_reuseFailAlloc_1740_;
goto v_reusejp_1738_;
}
v_reusejp_1738_:
{
v_a_1625_ = v___x_1739_;
v_a_1626_ = v_a_1735_;
goto v___jp_1624_;
}
}
}
else
{
lean_object* v_a_1742_; lean_object* v_a_1743_; lean_object* v___x_1745_; uint8_t v_isShared_1746_; uint8_t v_isSharedCheck_1750_; 
lean_del_object(v___x_1640_);
lean_dec(v_snd_1638_);
lean_dec(v_fst_1637_);
lean_del_object(v___x_1635_);
lean_dec(v_fst_1633_);
v_a_1742_ = lean_ctor_get(v___x_1734_, 0);
v_a_1743_ = lean_ctor_get(v___x_1734_, 1);
v_isSharedCheck_1750_ = !lean_is_exclusive(v___x_1734_);
if (v_isSharedCheck_1750_ == 0)
{
v___x_1745_ = v___x_1734_;
v_isShared_1746_ = v_isSharedCheck_1750_;
goto v_resetjp_1744_;
}
else
{
lean_inc(v_a_1743_);
lean_inc(v_a_1742_);
lean_dec(v___x_1734_);
v___x_1745_ = lean_box(0);
v_isShared_1746_ = v_isSharedCheck_1750_;
goto v_resetjp_1744_;
}
v_resetjp_1744_:
{
lean_object* v___x_1748_; 
if (v_isShared_1746_ == 0)
{
v___x_1748_ = v___x_1745_;
goto v_reusejp_1747_;
}
else
{
lean_object* v_reuseFailAlloc_1749_; 
v_reuseFailAlloc_1749_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1749_, 0, v_a_1742_);
lean_ctor_set(v_reuseFailAlloc_1749_, 1, v_a_1743_);
v___x_1748_ = v_reuseFailAlloc_1749_;
goto v_reusejp_1747_;
}
v_reusejp_1747_:
{
return v___x_1748_;
}
}
}
}
else
{
lean_object* v_quotContext_1751_; lean_object* v_currMacroScope_1752_; lean_object* v_ref_1753_; uint8_t v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1772_; 
v_quotContext_1751_ = lean_ctor_get(v___y_1622_, 1);
v_currMacroScope_1752_ = lean_ctor_get(v___y_1622_, 2);
v_ref_1753_ = lean_ctor_get(v___y_1622_, 5);
v___x_1754_ = 0;
v___x_1755_ = l_Lean_SourceInfo_fromRef(v_ref_1753_, v___x_1754_);
v___x_1756_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3));
v___x_1757_ = lean_obj_once(&lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5, &lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5_once, _init_lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__5);
v___x_1758_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__7));
lean_inc(v_currMacroScope_1752_);
lean_inc(v_quotContext_1751_);
v___x_1759_ = l_Lean_addMacroScope(v_quotContext_1751_, v___x_1758_, v_currMacroScope_1752_);
v___x_1760_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__12));
lean_inc_n(v___x_1755_, 2);
v___x_1761_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1761_, 0, v___x_1755_);
lean_ctor_set(v___x_1761_, 1, v___x_1757_);
lean_ctor_set(v___x_1761_, 2, v___x_1759_);
lean_ctor_set(v___x_1761_, 3, v___x_1760_);
v___x_1762_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1763_ = lp_proofwidgets_ProofWidgets_Jsx_getJsxText(v___x_1730_);
lean_dec(v___x_1730_);
v___x_1764_ = lean_string_append(v_fst_1633_, v___x_1763_);
lean_dec_ref(v___x_1763_);
v___x_1765_ = lean_box(2);
v___x_1766_ = l_Lean_Syntax_mkStrLit(v___x_1764_, v___x_1765_);
v___x_1767_ = l_Lean_Syntax_node1(v___x_1755_, v___x_1762_, v___x_1766_);
v___x_1768_ = l_Lean_Syntax_node2(v___x_1755_, v___x_1756_, v___x_1761_, v___x_1767_);
v___x_1769_ = lean_array_push(v_snd_1638_, v___x_1768_);
v___x_1770_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0));
if (v_isShared_1641_ == 0)
{
lean_ctor_set(v___x_1640_, 1, v___x_1769_);
v___x_1772_ = v___x_1640_;
goto v_reusejp_1771_;
}
else
{
lean_object* v_reuseFailAlloc_1776_; 
v_reuseFailAlloc_1776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1776_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1776_, 1, v___x_1769_);
v___x_1772_ = v_reuseFailAlloc_1776_;
goto v_reusejp_1771_;
}
v_reusejp_1771_:
{
lean_object* v___x_1774_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___x_1772_);
lean_ctor_set(v___x_1635_, 0, v___x_1770_);
v___x_1774_ = v___x_1635_;
goto v_reusejp_1773_;
}
else
{
lean_object* v_reuseFailAlloc_1775_; 
v_reuseFailAlloc_1775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1775_, 0, v___x_1770_);
lean_ctor_set(v_reuseFailAlloc_1775_, 1, v___x_1772_);
v___x_1774_ = v_reuseFailAlloc_1775_;
goto v_reusejp_1773_;
}
v_reusejp_1773_:
{
v_a_1625_ = v___x_1774_;
v_a_1626_ = v___y_1623_;
goto v___jp_1624_;
}
}
}
}
}
}
}
v___jp_1624_:
{
size_t v___x_1627_; size_t v___x_1628_; 
v___x_1627_ = ((size_t)1ULL);
v___x_1628_ = lean_usize_add(v_i_1620_, v___x_1627_);
v_i_1620_ = v___x_1628_;
v_b_1621_ = v_a_1625_;
v___y_1623_ = v_a_1626_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___boxed(lean_object* v_as_1779_, lean_object* v_sz_1780_, lean_object* v_i_1781_, lean_object* v_b_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_){
_start:
{
size_t v_sz_boxed_1785_; size_t v_i_boxed_1786_; lean_object* v_res_1787_; 
v_sz_boxed_1785_ = lean_unbox_usize(v_sz_1780_);
lean_dec(v_sz_1780_);
v_i_boxed_1786_ = lean_unbox_usize(v_i_1781_);
lean_dec(v_i_1781_);
v_res_1787_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6(v_as_1779_, v_sz_boxed_1785_, v_i_boxed_1786_, v_b_1782_, v___y_1783_, v___y_1784_);
lean_dec_ref(v___y_1783_);
lean_dec_ref(v_as_1779_);
return v_res_1787_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0(lean_object* v_as_1792_, size_t v_i_1793_, size_t v_stop_1794_, lean_object* v_b_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
uint8_t v___x_1798_; 
v___x_1798_ = lean_usize_dec_eq(v_i_1793_, v_stop_1794_);
if (v___x_1798_ == 0)
{
lean_object* v_ref_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; size_t v___x_1806_; size_t v___x_1807_; 
v_ref_1799_ = lean_ctor_get(v___y_1796_, 5);
v___x_1800_ = lean_array_uget_borrowed(v_as_1792_, v_i_1793_);
v___x_1801_ = l_Lean_SourceInfo_fromRef(v_ref_1799_, v___x_1798_);
v___x_1802_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__1));
v___x_1803_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___closed__2));
lean_inc(v___x_1801_);
v___x_1804_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1804_, 0, v___x_1801_);
lean_ctor_set(v___x_1804_, 1, v___x_1803_);
lean_inc(v___x_1800_);
v___x_1805_ = l_Lean_Syntax_node3(v___x_1801_, v___x_1802_, v_b_1795_, v___x_1804_, v___x_1800_);
v___x_1806_ = ((size_t)1ULL);
v___x_1807_ = lean_usize_add(v_i_1793_, v___x_1806_);
v_i_1793_ = v___x_1807_;
v_b_1795_ = v___x_1805_;
goto _start;
}
else
{
lean_object* v___x_1809_; 
v___x_1809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1809_, 0, v_b_1795_);
lean_ctor_set(v___x_1809_, 1, v___y_1797_);
return v___x_1809_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0___boxed(lean_object* v_as_1810_, lean_object* v_i_1811_, lean_object* v_stop_1812_, lean_object* v_b_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_){
_start:
{
size_t v_i_boxed_1816_; size_t v_stop_boxed_1817_; lean_object* v_res_1818_; 
v_i_boxed_1816_ = lean_unbox_usize(v_i_1811_);
lean_dec(v_i_1811_);
v_stop_boxed_1817_ = lean_unbox_usize(v_stop_1812_);
lean_dec(v_stop_1812_);
v_res_1818_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0(v_as_1810_, v_i_boxed_1816_, v_stop_boxed_1817_, v_b_1813_, v___y_1814_, v___y_1815_);
lean_dec_ref(v___y_1814_);
lean_dec_ref(v_as_1810_);
return v_res_1818_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0(lean_object* v_arr_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v___x_1822_; lean_object* v___x_1823_; uint8_t v___x_1824_; 
v___x_1822_ = lean_unsigned_to_nat(0u);
v___x_1823_ = lean_array_get_size(v_arr_1819_);
v___x_1824_ = lean_nat_dec_lt(v___x_1822_, v___x_1823_);
if (v___x_1824_ == 0)
{
lean_object* v_ref_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; 
v_ref_1825_ = lean_ctor_get(v___y_1820_, 5);
v___x_1826_ = l_Lean_SourceInfo_fromRef(v_ref_1825_, v___x_1824_);
v___x_1827_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1));
v___x_1828_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2));
lean_inc_n(v___x_1826_, 3);
v___x_1829_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1829_, 0, v___x_1826_);
lean_ctor_set(v___x_1829_, 1, v___x_1828_);
v___x_1830_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1831_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v___x_1832_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1832_, 0, v___x_1826_);
lean_ctor_set(v___x_1832_, 1, v___x_1830_);
lean_ctor_set(v___x_1832_, 2, v___x_1831_);
v___x_1833_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4));
v___x_1834_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1826_);
lean_ctor_set(v___x_1834_, 1, v___x_1833_);
v___x_1835_ = l_Lean_Syntax_node3(v___x_1826_, v___x_1827_, v___x_1829_, v___x_1832_, v___x_1834_);
v___x_1836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1835_);
lean_ctor_set(v___x_1836_, 1, v___y_1821_);
return v___x_1836_;
}
else
{
lean_object* v___x_1837_; lean_object* v___x_1838_; uint8_t v___x_1839_; 
v___x_1837_ = lean_array_fget_borrowed(v_arr_1819_, v___x_1822_);
v___x_1838_ = lean_unsigned_to_nat(1u);
v___x_1839_ = lean_nat_dec_lt(v___x_1838_, v___x_1823_);
if (v___x_1839_ == 0)
{
lean_object* v___x_1840_; 
lean_inc(v___x_1837_);
v___x_1840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1840_, 0, v___x_1837_);
lean_ctor_set(v___x_1840_, 1, v___y_1821_);
return v___x_1840_;
}
else
{
uint8_t v___x_1841_; 
v___x_1841_ = lean_nat_dec_le(v___x_1823_, v___x_1823_);
if (v___x_1841_ == 0)
{
if (v___x_1839_ == 0)
{
lean_object* v___x_1842_; 
lean_inc(v___x_1837_);
v___x_1842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1842_, 0, v___x_1837_);
lean_ctor_set(v___x_1842_, 1, v___y_1821_);
return v___x_1842_;
}
else
{
size_t v___x_1843_; size_t v___x_1844_; lean_object* v___x_1845_; 
v___x_1843_ = ((size_t)1ULL);
v___x_1844_ = lean_usize_of_nat(v___x_1823_);
lean_inc(v___x_1837_);
v___x_1845_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0(v_arr_1819_, v___x_1843_, v___x_1844_, v___x_1837_, v___y_1820_, v___y_1821_);
return v___x_1845_;
}
}
else
{
size_t v___x_1846_; size_t v___x_1847_; lean_object* v___x_1848_; 
v___x_1846_ = ((size_t)1ULL);
v___x_1847_ = lean_usize_of_nat(v___x_1823_);
lean_inc(v___x_1837_);
v___x_1848_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0_spec__0(v_arr_1819_, v___x_1846_, v___x_1847_, v___x_1837_, v___y_1820_, v___y_1821_);
return v___x_1848_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0___boxed(lean_object* v_arr_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_){
_start:
{
lean_object* v_res_1852_; 
v_res_1852_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0(v_arr_1849_, v___y_1850_, v___y_1851_);
lean_dec_ref(v___y_1850_);
lean_dec_ref(v_arr_1849_);
return v_res_1852_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8(lean_object* v_as_1872_, size_t v_i_1873_, size_t v_stop_1874_, lean_object* v_b_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_){
_start:
{
lean_object* v_a_1879_; lean_object* v_a_1880_; uint8_t v___x_1884_; 
v___x_1884_ = lean_usize_dec_eq(v_i_1873_, v_stop_1874_);
if (v___x_1884_ == 0)
{
lean_object* v___x_1885_; 
v___x_1885_ = lean_array_uget_borrowed(v_as_1872_, v_i_1873_);
if (lean_obj_tag(v___x_1885_) == 0)
{
lean_object* v_val_1886_; lean_object* v_fst_1887_; lean_object* v_snd_1888_; lean_object* v___x_1890_; uint8_t v_isShared_1891_; uint8_t v_isSharedCheck_1909_; 
v_val_1886_ = lean_ctor_get(v___x_1885_, 0);
lean_inc(v_val_1886_);
v_fst_1887_ = lean_ctor_get(v_val_1886_, 0);
v_snd_1888_ = lean_ctor_get(v_val_1886_, 1);
v_isSharedCheck_1909_ = !lean_is_exclusive(v_val_1886_);
if (v_isSharedCheck_1909_ == 0)
{
v___x_1890_ = v_val_1886_;
v_isShared_1891_ = v_isSharedCheck_1909_;
goto v_resetjp_1889_;
}
else
{
lean_inc(v_snd_1888_);
lean_inc(v_fst_1887_);
lean_dec(v_val_1886_);
v___x_1890_ = lean_box(0);
v_isShared_1891_ = v_isSharedCheck_1909_;
goto v_resetjp_1889_;
}
v_resetjp_1889_:
{
lean_object* v_ref_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1903_; 
v_ref_1892_ = lean_ctor_get(v___y_1876_, 5);
v___x_1893_ = l_Lean_SourceInfo_fromRef(v_ref_1892_, v___x_1884_);
v___x_1894_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1));
v___x_1895_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3));
v___x_1896_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_1897_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
lean_inc_n(v___x_1893_, 3);
v___x_1898_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1898_, 0, v___x_1893_);
lean_ctor_set(v___x_1898_, 1, v___x_1896_);
lean_ctor_set(v___x_1898_, 2, v___x_1897_);
lean_inc_ref(v___x_1898_);
v___x_1899_ = l_Lean_Syntax_node2(v___x_1893_, v___x_1895_, v_fst_1887_, v___x_1898_);
v___x_1900_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5));
v___x_1901_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__6));
if (v_isShared_1891_ == 0)
{
lean_ctor_set_tag(v___x_1890_, 2);
lean_ctor_set(v___x_1890_, 1, v___x_1901_);
lean_ctor_set(v___x_1890_, 0, v___x_1893_);
v___x_1903_ = v___x_1890_;
goto v_reusejp_1902_;
}
else
{
lean_object* v_reuseFailAlloc_1908_; 
v_reuseFailAlloc_1908_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1908_, 0, v___x_1893_);
lean_ctor_set(v_reuseFailAlloc_1908_, 1, v___x_1901_);
v___x_1903_ = v_reuseFailAlloc_1908_;
goto v_reusejp_1902_;
}
v_reusejp_1902_:
{
lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; 
lean_inc_ref_n(v___x_1898_, 2);
lean_inc_n(v___x_1893_, 2);
v___x_1904_ = l_Lean_Syntax_node3(v___x_1893_, v___x_1900_, v___x_1903_, v___x_1898_, v_snd_1888_);
v___x_1905_ = l_Lean_Syntax_node3(v___x_1893_, v___x_1896_, v___x_1898_, v___x_1898_, v___x_1904_);
v___x_1906_ = l_Lean_Syntax_node2(v___x_1893_, v___x_1894_, v___x_1899_, v___x_1905_);
v___x_1907_ = lean_array_push(v_b_1875_, v___x_1906_);
v_a_1879_ = v___x_1907_;
v_a_1880_ = v___y_1877_;
goto v___jp_1878_;
}
}
}
else
{
v_a_1879_ = v_b_1875_;
v_a_1880_ = v___y_1877_;
goto v___jp_1878_;
}
}
else
{
lean_object* v___x_1910_; 
v___x_1910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1910_, 0, v_b_1875_);
lean_ctor_set(v___x_1910_, 1, v___y_1877_);
return v___x_1910_;
}
v___jp_1878_:
{
size_t v___x_1881_; size_t v___x_1882_; 
v___x_1881_ = ((size_t)1ULL);
v___x_1882_ = lean_usize_add(v_i_1873_, v___x_1881_);
v_i_1873_ = v___x_1882_;
v_b_1875_ = v_a_1879_;
v___y_1877_ = v_a_1880_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___boxed(lean_object* v_as_1911_, lean_object* v_i_1912_, lean_object* v_stop_1913_, lean_object* v_b_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
size_t v_i_boxed_1917_; size_t v_stop_boxed_1918_; lean_object* v_res_1919_; 
v_i_boxed_1917_ = lean_unbox_usize(v_i_1912_);
lean_dec(v_i_1912_);
v_stop_boxed_1918_ = lean_unbox_usize(v_stop_1913_);
lean_dec(v_stop_1913_);
v_res_1919_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8(v_as_1911_, v_i_boxed_1917_, v_stop_boxed_1918_, v_b_1914_, v___y_1915_, v___y_1916_);
lean_dec_ref(v___y_1915_);
lean_dec_ref(v_as_1911_);
return v_res_1919_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5(lean_object* v_as_1920_, lean_object* v_start_1921_, lean_object* v_stop_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_){
_start:
{
lean_object* v___x_1925_; uint8_t v___x_1926_; 
v___x_1925_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1));
v___x_1926_ = lean_nat_dec_lt(v_start_1921_, v_stop_1922_);
if (v___x_1926_ == 0)
{
lean_object* v___x_1927_; 
v___x_1927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1927_, 0, v___x_1925_);
lean_ctor_set(v___x_1927_, 1, v___y_1924_);
return v___x_1927_;
}
else
{
lean_object* v___x_1928_; uint8_t v___x_1929_; 
v___x_1928_ = lean_array_get_size(v_as_1920_);
v___x_1929_ = lean_nat_dec_le(v_stop_1922_, v___x_1928_);
if (v___x_1929_ == 0)
{
uint8_t v___x_1930_; 
v___x_1930_ = lean_nat_dec_lt(v_start_1921_, v___x_1928_);
if (v___x_1930_ == 0)
{
lean_object* v___x_1931_; 
v___x_1931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1931_, 0, v___x_1925_);
lean_ctor_set(v___x_1931_, 1, v___y_1924_);
return v___x_1931_;
}
else
{
size_t v___x_1932_; size_t v___x_1933_; lean_object* v___x_1934_; 
v___x_1932_ = lean_usize_of_nat(v_start_1921_);
v___x_1933_ = lean_usize_of_nat(v___x_1928_);
v___x_1934_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8(v_as_1920_, v___x_1932_, v___x_1933_, v___x_1925_, v___y_1923_, v___y_1924_);
return v___x_1934_;
}
}
else
{
size_t v___x_1935_; size_t v___x_1936_; lean_object* v___x_1937_; 
v___x_1935_ = lean_usize_of_nat(v_start_1921_);
v___x_1936_ = lean_usize_of_nat(v_stop_1922_);
v___x_1937_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8(v_as_1920_, v___x_1935_, v___x_1936_, v___x_1925_, v___y_1923_, v___y_1924_);
return v___x_1937_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5___boxed(lean_object* v_as_1938_, lean_object* v_start_1939_, lean_object* v_stop_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_){
_start:
{
lean_object* v_res_1943_; 
v_res_1943_ = lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5(v_as_1938_, v_start_1939_, v_stop_1940_, v___y_1941_, v___y_1942_);
lean_dec_ref(v___y_1941_);
lean_dec(v_stop_1940_);
lean_dec(v_start_1939_);
lean_dec_ref(v_as_1938_);
return v_res_1943_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1(size_t v_sz_1944_, size_t v_i_1945_, lean_object* v_bs_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
uint8_t v___x_1949_; 
v___x_1949_ = lean_usize_dec_lt(v_i_1945_, v_sz_1944_);
if (v___x_1949_ == 0)
{
lean_object* v___x_1950_; 
v___x_1950_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1950_, 0, v_bs_1946_);
lean_ctor_set(v___x_1950_, 1, v___y_1948_);
return v___x_1950_;
}
else
{
lean_object* v_v_1951_; lean_object* v___x_1952_; lean_object* v_bs_x27_1953_; lean_object* v_a_1955_; lean_object* v_a_1956_; lean_object* v___y_1962_; lean_object* v___x_1974_; uint8_t v___x_1975_; 
v_v_1951_ = lean_array_uget(v_bs_1946_, v_i_1945_);
v___x_1952_ = lean_unsigned_to_nat(0u);
v_bs_x27_1953_ = lean_array_uset(v_bs_1946_, v_i_1945_, v___x_1952_);
v___x_1974_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1));
lean_inc(v_v_1951_);
v___x_1975_ = l_Lean_Syntax_isOfKind(v_v_1951_, v___x_1974_);
if (v___x_1975_ == 0)
{
lean_object* v___x_1976_; uint8_t v___x_1977_; 
v___x_1976_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
lean_inc(v_v_1951_);
v___x_1977_ = l_Lean_Syntax_isOfKind(v_v_1951_, v___x_1976_);
if (v___x_1977_ == 0)
{
lean_object* v___x_1978_; lean_object* v___x_1979_; 
v___x_1978_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_1979_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_1978_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_1979_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_1980_; lean_object* v___x_1981_; uint8_t v___x_1982_; 
v___x_1980_ = l_Lean_Syntax_getArg(v_v_1951_, v___x_1952_);
v___x_1981_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
lean_inc(v___x_1980_);
v___x_1982_ = l_Lean_Syntax_isOfKind(v___x_1980_, v___x_1981_);
if (v___x_1982_ == 0)
{
lean_object* v___x_1983_; lean_object* v___x_1984_; 
lean_dec(v___x_1980_);
v___x_1983_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_1984_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_1983_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_1984_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; 
lean_dec(v_v_1951_);
v___x_1985_ = lean_unsigned_to_nat(1u);
v___x_1986_ = l_Lean_Syntax_getArg(v___x_1980_, v___x_1985_);
lean_dec(v___x_1980_);
v___x_1987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1987_, 0, v___x_1986_);
v_a_1955_ = v___x_1987_;
v_a_1956_ = v___y_1948_;
goto v___jp_1954_;
}
}
}
else
{
lean_object* v___x_1988_; lean_object* v___x_1989_; uint8_t v___x_1990_; 
v___x_1988_ = l_Lean_Syntax_getArg(v_v_1951_, v___x_1952_);
v___x_1989_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3));
lean_inc(v___x_1988_);
v___x_1990_ = l_Lean_Syntax_isOfKind(v___x_1988_, v___x_1989_);
if (v___x_1990_ == 0)
{
lean_object* v___x_1991_; lean_object* v___x_1992_; 
lean_dec(v___x_1988_);
v___x_1991_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_1992_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_1991_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_1992_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; uint8_t v___x_1996_; 
v___x_1993_ = lean_unsigned_to_nat(2u);
v___x_1994_ = l_Lean_Syntax_getArg(v_v_1951_, v___x_1993_);
v___x_1995_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__3));
lean_inc(v___x_1994_);
v___x_1996_ = l_Lean_Syntax_isOfKind(v___x_1994_, v___x_1995_);
if (v___x_1996_ == 0)
{
lean_object* v___x_1997_; uint8_t v___x_1998_; 
v___x_1997_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1));
lean_inc(v___x_1994_);
v___x_1998_ = l_Lean_Syntax_isOfKind(v___x_1994_, v___x_1997_);
if (v___x_1998_ == 0)
{
lean_object* v___x_1999_; lean_object* v___x_2000_; 
lean_dec(v___x_1994_);
lean_dec(v___x_1988_);
v___x_1999_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_2000_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_1999_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_2000_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_2001_; lean_object* v___x_2002_; uint8_t v___x_2003_; 
v___x_2001_ = l_Lean_Syntax_getArg(v___x_1994_, v___x_1952_);
lean_dec(v___x_1994_);
v___x_2002_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
lean_inc(v___x_2001_);
v___x_2003_ = l_Lean_Syntax_isOfKind(v___x_2001_, v___x_2002_);
if (v___x_2003_ == 0)
{
lean_object* v___x_2004_; lean_object* v___x_2005_; 
lean_dec(v___x_2001_);
lean_dec(v___x_1988_);
v___x_2004_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_2005_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_2004_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_2005_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; 
lean_dec(v_v_1951_);
v___x_2006_ = lean_unsigned_to_nat(1u);
v___x_2007_ = l_Lean_Syntax_getArg(v___x_2001_, v___x_2006_);
lean_dec(v___x_2001_);
v___x_2008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2008_, 0, v___x_1988_);
lean_ctor_set(v___x_2008_, 1, v___x_2007_);
v___x_2009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2009_, 0, v___x_2008_);
v_a_1955_ = v___x_2009_;
v_a_1956_ = v___y_1948_;
goto v___jp_1954_;
}
}
}
else
{
lean_object* v___x_2010_; lean_object* v___x_2011_; uint8_t v___x_2012_; 
v___x_2010_ = l_Lean_Syntax_getArg(v___x_1994_, v___x_1952_);
lean_dec(v___x_1994_);
v___x_2011_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__5));
lean_inc(v___x_2010_);
v___x_2012_ = l_Lean_Syntax_isOfKind(v___x_2010_, v___x_2011_);
if (v___x_2012_ == 0)
{
lean_object* v___x_2013_; lean_object* v___x_2014_; 
lean_dec(v___x_2010_);
lean_dec(v___x_1988_);
v___x_2013_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__0));
v___x_2014_ = l_Lean_Macro_throwErrorAt___redArg(v_v_1951_, v___x_2013_, v___y_1947_, v___y_1948_);
lean_dec(v_v_1951_);
v___y_1962_ = v___x_2014_;
goto v___jp_1961_;
}
else
{
lean_object* v___x_2015_; lean_object* v___x_2016_; 
lean_dec(v_v_1951_);
v___x_2015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2015_, 0, v___x_1988_);
lean_ctor_set(v___x_2015_, 1, v___x_2010_);
v___x_2016_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2016_, 0, v___x_2015_);
v_a_1955_ = v___x_2016_;
v_a_1956_ = v___y_1948_;
goto v___jp_1954_;
}
}
}
}
v___jp_1954_:
{
size_t v___x_1957_; size_t v___x_1958_; lean_object* v___x_1959_; 
v___x_1957_ = ((size_t)1ULL);
v___x_1958_ = lean_usize_add(v_i_1945_, v___x_1957_);
v___x_1959_ = lean_array_uset(v_bs_x27_1953_, v_i_1945_, v_a_1955_);
v_i_1945_ = v___x_1958_;
v_bs_1946_ = v___x_1959_;
v___y_1948_ = v_a_1956_;
goto _start;
}
v___jp_1961_:
{
if (lean_obj_tag(v___y_1962_) == 0)
{
lean_object* v_a_1963_; lean_object* v_a_1964_; 
v_a_1963_ = lean_ctor_get(v___y_1962_, 0);
lean_inc(v_a_1963_);
v_a_1964_ = lean_ctor_get(v___y_1962_, 1);
lean_inc(v_a_1964_);
lean_dec_ref_known(v___y_1962_, 2);
v_a_1955_ = v_a_1963_;
v_a_1956_ = v_a_1964_;
goto v___jp_1954_;
}
else
{
lean_object* v_a_1965_; lean_object* v_a_1966_; lean_object* v___x_1968_; uint8_t v_isShared_1969_; uint8_t v_isSharedCheck_1973_; 
lean_dec_ref(v_bs_x27_1953_);
v_a_1965_ = lean_ctor_get(v___y_1962_, 0);
v_a_1966_ = lean_ctor_get(v___y_1962_, 1);
v_isSharedCheck_1973_ = !lean_is_exclusive(v___y_1962_);
if (v_isSharedCheck_1973_ == 0)
{
v___x_1968_ = v___y_1962_;
v_isShared_1969_ = v_isSharedCheck_1973_;
goto v_resetjp_1967_;
}
else
{
lean_inc(v_a_1966_);
lean_inc(v_a_1965_);
lean_dec(v___y_1962_);
v___x_1968_ = lean_box(0);
v_isShared_1969_ = v_isSharedCheck_1973_;
goto v_resetjp_1967_;
}
v_resetjp_1967_:
{
lean_object* v___x_1971_; 
if (v_isShared_1969_ == 0)
{
v___x_1971_ = v___x_1968_;
goto v_reusejp_1970_;
}
else
{
lean_object* v_reuseFailAlloc_1972_; 
v_reuseFailAlloc_1972_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1972_, 0, v_a_1965_);
lean_ctor_set(v_reuseFailAlloc_1972_, 1, v_a_1966_);
v___x_1971_ = v_reuseFailAlloc_1972_;
goto v_reusejp_1970_;
}
v_reusejp_1970_:
{
return v___x_1971_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1___boxed(lean_object* v_sz_2017_, lean_object* v_i_2018_, lean_object* v_bs_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_){
_start:
{
size_t v_sz_boxed_2022_; size_t v_i_boxed_2023_; lean_object* v_res_2024_; 
v_sz_boxed_2022_ = lean_unbox_usize(v_sz_2017_);
lean_dec(v_sz_2017_);
v_i_boxed_2023_ = lean_unbox_usize(v_i_2018_);
lean_dec(v_i_2018_);
v_res_2024_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1(v_sz_boxed_2022_, v_i_boxed_2023_, v_bs_2019_, v___y_2020_, v___y_2021_);
lean_dec_ref(v___y_2020_);
return v_res_2024_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(lean_object* v_as_2025_, size_t v_i_2026_, size_t v_stop_2027_, lean_object* v_b_2028_, lean_object* v___y_2029_){
_start:
{
lean_object* v_a_2031_; lean_object* v_a_2032_; uint8_t v___x_2036_; 
v___x_2036_ = lean_usize_dec_eq(v_i_2026_, v_stop_2027_);
if (v___x_2036_ == 0)
{
lean_object* v___x_2037_; 
v___x_2037_ = lean_array_uget_borrowed(v_as_2025_, v_i_2026_);
if (lean_obj_tag(v___x_2037_) == 0)
{
v_a_2031_ = v_b_2028_;
v_a_2032_ = v___y_2029_;
goto v___jp_2030_;
}
else
{
lean_object* v_val_2038_; lean_object* v___x_2039_; 
v_val_2038_ = lean_ctor_get(v___x_2037_, 0);
lean_inc(v_val_2038_);
v___x_2039_ = lean_array_push(v_b_2028_, v_val_2038_);
v_a_2031_ = v___x_2039_;
v_a_2032_ = v___y_2029_;
goto v___jp_2030_;
}
}
else
{
lean_object* v___x_2040_; 
v___x_2040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2040_, 0, v_b_2028_);
lean_ctor_set(v___x_2040_, 1, v___y_2029_);
return v___x_2040_;
}
v___jp_2030_:
{
size_t v___x_2033_; size_t v___x_2034_; 
v___x_2033_ = ((size_t)1ULL);
v___x_2034_ = lean_usize_add(v_i_2026_, v___x_2033_);
v_i_2026_ = v___x_2034_;
v_b_2028_ = v_a_2031_;
v___y_2029_ = v_a_2032_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg___boxed(lean_object* v_as_2041_, lean_object* v_i_2042_, lean_object* v_stop_2043_, lean_object* v_b_2044_, lean_object* v___y_2045_){
_start:
{
size_t v_i_boxed_2046_; size_t v_stop_boxed_2047_; lean_object* v_res_2048_; 
v_i_boxed_2046_ = lean_unbox_usize(v_i_2042_);
lean_dec(v_i_2042_);
v_stop_boxed_2047_ = lean_unbox_usize(v_stop_2043_);
lean_dec(v_stop_2043_);
v_res_2048_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(v_as_2041_, v_i_boxed_2046_, v_stop_boxed_2047_, v_b_2044_, v___y_2045_);
lean_dec_ref(v_as_2041_);
return v_res_2048_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4(lean_object* v_as_2049_, lean_object* v_start_2050_, lean_object* v_stop_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_){
_start:
{
lean_object* v___x_2054_; uint8_t v___x_2055_; 
v___x_2054_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1));
v___x_2055_ = lean_nat_dec_lt(v_start_2050_, v_stop_2051_);
if (v___x_2055_ == 0)
{
lean_object* v___x_2056_; 
v___x_2056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2056_, 0, v___x_2054_);
lean_ctor_set(v___x_2056_, 1, v___y_2053_);
return v___x_2056_;
}
else
{
lean_object* v___x_2057_; uint8_t v___x_2058_; 
v___x_2057_ = lean_array_get_size(v_as_2049_);
v___x_2058_ = lean_nat_dec_le(v_stop_2051_, v___x_2057_);
if (v___x_2058_ == 0)
{
uint8_t v___x_2059_; 
v___x_2059_ = lean_nat_dec_lt(v_start_2050_, v___x_2057_);
if (v___x_2059_ == 0)
{
lean_object* v___x_2060_; 
v___x_2060_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2060_, 0, v___x_2054_);
lean_ctor_set(v___x_2060_, 1, v___y_2053_);
return v___x_2060_;
}
else
{
size_t v___x_2061_; size_t v___x_2062_; lean_object* v___x_2063_; 
v___x_2061_ = lean_usize_of_nat(v_start_2050_);
v___x_2062_ = lean_usize_of_nat(v___x_2057_);
v___x_2063_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(v_as_2049_, v___x_2061_, v___x_2062_, v___x_2054_, v___y_2053_);
return v___x_2063_;
}
}
else
{
size_t v___x_2064_; size_t v___x_2065_; lean_object* v___x_2066_; 
v___x_2064_ = lean_usize_of_nat(v_start_2050_);
v___x_2065_ = lean_usize_of_nat(v_stop_2051_);
v___x_2066_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(v_as_2049_, v___x_2064_, v___x_2065_, v___x_2054_, v___y_2053_);
return v___x_2066_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4___boxed(lean_object* v_as_2067_, lean_object* v_start_2068_, lean_object* v_stop_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_){
_start:
{
lean_object* v_res_2072_; 
v_res_2072_ = lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4(v_as_2067_, v_start_2068_, v_stop_2069_, v___y_2070_, v___y_2071_);
lean_dec_ref(v___y_2070_);
lean_dec(v_stop_2069_);
lean_dec(v_start_2068_);
lean_dec_ref(v_as_2067_);
return v_res_2072_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0(lean_object* v_val_2073_, lean_object* v_pending__inls_2074_, lean_object* v_____r_2075_, lean_object* v_ret_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_){
_start:
{
lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; 
v___x_2079_ = lean_array_push(v_ret_2076_, v_val_2073_);
v___x_2080_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2080_, 0, v___x_2079_);
lean_ctor_set(v___x_2080_, 1, v_pending__inls_2074_);
v___x_2081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2081_, 0, v___x_2080_);
v___x_2082_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2082_, 0, v___x_2081_);
lean_ctor_set(v___x_2082_, 1, v___y_2078_);
return v___x_2082_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v_val_2083_, lean_object* v_pending__inls_2084_, lean_object* v_____r_2085_, lean_object* v_ret_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_){
_start:
{
lean_object* v_res_2089_; 
v_res_2089_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0(v_val_2083_, v_pending__inls_2084_, v_____r_2085_, v_ret_2086_, v___y_2087_, v___y_2088_);
lean_dec_ref(v___y_2087_);
return v_res_2089_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg(lean_object* v_f_2092_, lean_object* v_as_2093_, size_t v_sz_2094_, size_t v_i_2095_, lean_object* v_b_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_){
_start:
{
lean_object* v_a_2100_; lean_object* v_a_2101_; lean_object* v___y_2106_; uint8_t v___x_2129_; 
v___x_2129_ = lean_usize_dec_lt(v_i_2095_, v_sz_2094_);
if (v___x_2129_ == 0)
{
lean_object* v___x_2130_; 
lean_dec_ref(v_f_2092_);
v___x_2130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2130_, 0, v_b_2096_);
lean_ctor_set(v___x_2130_, 1, v___y_2098_);
return v___x_2130_;
}
else
{
lean_object* v_fst_2131_; lean_object* v_snd_2132_; lean_object* v___x_2134_; uint8_t v_isShared_2135_; uint8_t v_isSharedCheck_2165_; 
v_fst_2131_ = lean_ctor_get(v_b_2096_, 0);
v_snd_2132_ = lean_ctor_get(v_b_2096_, 1);
v_isSharedCheck_2165_ = !lean_is_exclusive(v_b_2096_);
if (v_isSharedCheck_2165_ == 0)
{
v___x_2134_ = v_b_2096_;
v_isShared_2135_ = v_isSharedCheck_2165_;
goto v_resetjp_2133_;
}
else
{
lean_inc(v_snd_2132_);
lean_inc(v_fst_2131_);
lean_dec(v_b_2096_);
v___x_2134_ = lean_box(0);
v_isShared_2135_ = v_isSharedCheck_2165_;
goto v_resetjp_2133_;
}
v_resetjp_2133_:
{
lean_object* v_a_2136_; 
v_a_2136_ = lean_array_uget_borrowed(v_as_2093_, v_i_2095_);
if (lean_obj_tag(v_a_2136_) == 0)
{
lean_object* v_val_2137_; lean_object* v___x_2138_; lean_object* v___x_2140_; 
v_val_2137_ = lean_ctor_get(v_a_2136_, 0);
lean_inc(v_val_2137_);
v___x_2138_ = lean_array_push(v_snd_2132_, v_val_2137_);
if (v_isShared_2135_ == 0)
{
lean_ctor_set(v___x_2134_, 1, v___x_2138_);
v___x_2140_ = v___x_2134_;
goto v_reusejp_2139_;
}
else
{
lean_object* v_reuseFailAlloc_2141_; 
v_reuseFailAlloc_2141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2141_, 0, v_fst_2131_);
lean_ctor_set(v_reuseFailAlloc_2141_, 1, v___x_2138_);
v___x_2140_ = v_reuseFailAlloc_2141_;
goto v_reusejp_2139_;
}
v_reusejp_2139_:
{
v_a_2100_ = v___x_2140_;
v_a_2101_ = v___y_2098_;
goto v___jp_2099_;
}
}
else
{
lean_object* v_val_2142_; lean_object* v___x_2143_; lean_object* v_pending__inls_2144_; lean_object* v___x_2148_; uint8_t v___x_2149_; 
lean_del_object(v___x_2134_);
v_val_2142_ = lean_ctor_get(v_a_2136_, 0);
v___x_2143_ = lean_unsigned_to_nat(0u);
v_pending__inls_2144_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___closed__0));
v___x_2148_ = lean_array_get_size(v_snd_2132_);
v___x_2149_ = lean_nat_dec_eq(v___x_2148_, v___x_2143_);
if (v___x_2149_ == 0)
{
if (v___x_2129_ == 0)
{
lean_dec(v_snd_2132_);
goto v___jp_2145_;
}
else
{
lean_object* v___x_2150_; 
lean_inc_ref(v_f_2092_);
lean_inc_ref(v___y_2097_);
v___x_2150_ = lean_apply_3(v_f_2092_, v_snd_2132_, v___y_2097_, v___y_2098_);
if (lean_obj_tag(v___x_2150_) == 0)
{
lean_object* v_a_2151_; lean_object* v_a_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; 
v_a_2151_ = lean_ctor_get(v___x_2150_, 0);
lean_inc(v_a_2151_);
v_a_2152_ = lean_ctor_get(v___x_2150_, 1);
lean_inc(v_a_2152_);
lean_dec_ref_known(v___x_2150_, 2);
v___x_2153_ = lean_array_push(v_fst_2131_, v_a_2151_);
v___x_2154_ = lean_box(0);
lean_inc(v_val_2142_);
v___x_2155_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0(v_val_2142_, v_pending__inls_2144_, v___x_2154_, v___x_2153_, v___y_2097_, v_a_2152_);
v___y_2106_ = v___x_2155_;
goto v___jp_2105_;
}
else
{
lean_object* v_a_2156_; lean_object* v_a_2157_; lean_object* v___x_2159_; uint8_t v_isShared_2160_; uint8_t v_isSharedCheck_2164_; 
lean_dec(v_fst_2131_);
lean_dec_ref(v_f_2092_);
v_a_2156_ = lean_ctor_get(v___x_2150_, 0);
v_a_2157_ = lean_ctor_get(v___x_2150_, 1);
v_isSharedCheck_2164_ = !lean_is_exclusive(v___x_2150_);
if (v_isSharedCheck_2164_ == 0)
{
v___x_2159_ = v___x_2150_;
v_isShared_2160_ = v_isSharedCheck_2164_;
goto v_resetjp_2158_;
}
else
{
lean_inc(v_a_2157_);
lean_inc(v_a_2156_);
lean_dec(v___x_2150_);
v___x_2159_ = lean_box(0);
v_isShared_2160_ = v_isSharedCheck_2164_;
goto v_resetjp_2158_;
}
v_resetjp_2158_:
{
lean_object* v___x_2162_; 
if (v_isShared_2160_ == 0)
{
v___x_2162_ = v___x_2159_;
goto v_reusejp_2161_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v_a_2156_);
lean_ctor_set(v_reuseFailAlloc_2163_, 1, v_a_2157_);
v___x_2162_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2161_;
}
v_reusejp_2161_:
{
return v___x_2162_;
}
}
}
}
}
else
{
lean_dec(v_snd_2132_);
goto v___jp_2145_;
}
v___jp_2145_:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; 
v___x_2146_ = lean_box(0);
lean_inc(v_val_2142_);
v___x_2147_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___lam__0(v_val_2142_, v_pending__inls_2144_, v___x_2146_, v_fst_2131_, v___y_2097_, v___y_2098_);
v___y_2106_ = v___x_2147_;
goto v___jp_2105_;
}
}
}
}
v___jp_2099_:
{
size_t v___x_2102_; size_t v___x_2103_; 
v___x_2102_ = ((size_t)1ULL);
v___x_2103_ = lean_usize_add(v_i_2095_, v___x_2102_);
v_i_2095_ = v___x_2103_;
v_b_2096_ = v_a_2100_;
v___y_2098_ = v_a_2101_;
goto _start;
}
v___jp_2105_:
{
if (lean_obj_tag(v___y_2106_) == 0)
{
lean_object* v_a_2107_; 
v_a_2107_ = lean_ctor_get(v___y_2106_, 0);
lean_inc(v_a_2107_);
if (lean_obj_tag(v_a_2107_) == 0)
{
lean_object* v_a_2108_; lean_object* v___x_2110_; uint8_t v_isShared_2111_; uint8_t v_isSharedCheck_2116_; 
lean_dec_ref(v_f_2092_);
v_a_2108_ = lean_ctor_get(v___y_2106_, 1);
v_isSharedCheck_2116_ = !lean_is_exclusive(v___y_2106_);
if (v_isSharedCheck_2116_ == 0)
{
lean_object* v_unused_2117_; 
v_unused_2117_ = lean_ctor_get(v___y_2106_, 0);
lean_dec(v_unused_2117_);
v___x_2110_ = v___y_2106_;
v_isShared_2111_ = v_isSharedCheck_2116_;
goto v_resetjp_2109_;
}
else
{
lean_inc(v_a_2108_);
lean_dec(v___y_2106_);
v___x_2110_ = lean_box(0);
v_isShared_2111_ = v_isSharedCheck_2116_;
goto v_resetjp_2109_;
}
v_resetjp_2109_:
{
lean_object* v_a_2112_; lean_object* v___x_2114_; 
v_a_2112_ = lean_ctor_get(v_a_2107_, 0);
lean_inc(v_a_2112_);
lean_dec_ref_known(v_a_2107_, 1);
if (v_isShared_2111_ == 0)
{
lean_ctor_set(v___x_2110_, 0, v_a_2112_);
v___x_2114_ = v___x_2110_;
goto v_reusejp_2113_;
}
else
{
lean_object* v_reuseFailAlloc_2115_; 
v_reuseFailAlloc_2115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2115_, 0, v_a_2112_);
lean_ctor_set(v_reuseFailAlloc_2115_, 1, v_a_2108_);
v___x_2114_ = v_reuseFailAlloc_2115_;
goto v_reusejp_2113_;
}
v_reusejp_2113_:
{
return v___x_2114_;
}
}
}
else
{
lean_object* v_a_2118_; lean_object* v_a_2119_; 
v_a_2118_ = lean_ctor_get(v___y_2106_, 1);
lean_inc(v_a_2118_);
lean_dec_ref_known(v___y_2106_, 2);
v_a_2119_ = lean_ctor_get(v_a_2107_, 0);
lean_inc(v_a_2119_);
lean_dec_ref_known(v_a_2107_, 1);
v_a_2100_ = v_a_2119_;
v_a_2101_ = v_a_2118_;
goto v___jp_2099_;
}
}
else
{
lean_object* v_a_2120_; lean_object* v_a_2121_; lean_object* v___x_2123_; uint8_t v_isShared_2124_; uint8_t v_isSharedCheck_2128_; 
lean_dec_ref(v_f_2092_);
v_a_2120_ = lean_ctor_get(v___y_2106_, 0);
v_a_2121_ = lean_ctor_get(v___y_2106_, 1);
v_isSharedCheck_2128_ = !lean_is_exclusive(v___y_2106_);
if (v_isSharedCheck_2128_ == 0)
{
v___x_2123_ = v___y_2106_;
v_isShared_2124_ = v_isSharedCheck_2128_;
goto v_resetjp_2122_;
}
else
{
lean_inc(v_a_2121_);
lean_inc(v_a_2120_);
lean_dec(v___y_2106_);
v___x_2123_ = lean_box(0);
v_isShared_2124_ = v_isSharedCheck_2128_;
goto v_resetjp_2122_;
}
v_resetjp_2122_:
{
lean_object* v___x_2126_; 
if (v_isShared_2124_ == 0)
{
v___x_2126_ = v___x_2123_;
goto v_reusejp_2125_;
}
else
{
lean_object* v_reuseFailAlloc_2127_; 
v_reuseFailAlloc_2127_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2127_, 0, v_a_2120_);
lean_ctor_set(v_reuseFailAlloc_2127_, 1, v_a_2121_);
v___x_2126_ = v_reuseFailAlloc_2127_;
goto v_reusejp_2125_;
}
v_reusejp_2125_:
{
return v___x_2126_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg___boxed(lean_object* v_f_2166_, lean_object* v_as_2167_, lean_object* v_sz_2168_, lean_object* v_i_2169_, lean_object* v_b_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_){
_start:
{
size_t v_sz_boxed_2173_; size_t v_i_boxed_2174_; lean_object* v_res_2175_; 
v_sz_boxed_2173_ = lean_unbox_usize(v_sz_2168_);
lean_dec(v_sz_2168_);
v_i_boxed_2174_ = lean_unbox_usize(v_i_2169_);
lean_dec(v_i_2169_);
v_res_2175_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg(v_f_2166_, v_as_2167_, v_sz_boxed_2173_, v_i_boxed_2174_, v_b_2170_, v___y_2171_, v___y_2172_);
lean_dec_ref(v___y_2171_);
lean_dec_ref(v_as_2167_);
return v_res_2175_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg(lean_object* v_arr_2178_, lean_object* v_f_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_){
_start:
{
lean_object* v___x_2182_; lean_object* v___x_2183_; size_t v_sz_2184_; size_t v___x_2185_; lean_object* v___x_2186_; 
v___x_2182_ = lean_unsigned_to_nat(0u);
v___x_2183_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___closed__0));
v_sz_2184_ = lean_array_size(v_arr_2178_);
v___x_2185_ = ((size_t)0ULL);
lean_inc_ref(v_f_2179_);
v___x_2186_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg(v_f_2179_, v_arr_2178_, v_sz_2184_, v___x_2185_, v___x_2183_, v___y_2180_, v___y_2181_);
if (lean_obj_tag(v___x_2186_) == 0)
{
lean_object* v_a_2187_; lean_object* v_a_2188_; lean_object* v___x_2190_; uint8_t v_isShared_2191_; uint8_t v_isSharedCheck_2219_; 
v_a_2187_ = lean_ctor_get(v___x_2186_, 0);
v_a_2188_ = lean_ctor_get(v___x_2186_, 1);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2186_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2190_ = v___x_2186_;
v_isShared_2191_ = v_isSharedCheck_2219_;
goto v_resetjp_2189_;
}
else
{
lean_inc(v_a_2188_);
lean_inc(v_a_2187_);
lean_dec(v___x_2186_);
v___x_2190_ = lean_box(0);
v_isShared_2191_ = v_isSharedCheck_2219_;
goto v_resetjp_2189_;
}
v_resetjp_2189_:
{
lean_object* v_fst_2192_; lean_object* v_snd_2193_; lean_object* v___x_2194_; uint8_t v___x_2195_; 
v_fst_2192_ = lean_ctor_get(v_a_2187_, 0);
lean_inc(v_fst_2192_);
v_snd_2193_ = lean_ctor_get(v_a_2187_, 1);
lean_inc(v_snd_2193_);
lean_dec(v_a_2187_);
v___x_2194_ = lean_array_get_size(v_snd_2193_);
v___x_2195_ = lean_nat_dec_eq(v___x_2194_, v___x_2182_);
if (v___x_2195_ == 0)
{
lean_object* v___x_2196_; 
lean_del_object(v___x_2190_);
lean_inc_ref(v___y_2180_);
v___x_2196_ = lean_apply_3(v_f_2179_, v_snd_2193_, v___y_2180_, v_a_2188_);
if (lean_obj_tag(v___x_2196_) == 0)
{
lean_object* v_a_2197_; lean_object* v_a_2198_; lean_object* v___x_2200_; uint8_t v_isShared_2201_; uint8_t v_isSharedCheck_2206_; 
v_a_2197_ = lean_ctor_get(v___x_2196_, 0);
v_a_2198_ = lean_ctor_get(v___x_2196_, 1);
v_isSharedCheck_2206_ = !lean_is_exclusive(v___x_2196_);
if (v_isSharedCheck_2206_ == 0)
{
v___x_2200_ = v___x_2196_;
v_isShared_2201_ = v_isSharedCheck_2206_;
goto v_resetjp_2199_;
}
else
{
lean_inc(v_a_2198_);
lean_inc(v_a_2197_);
lean_dec(v___x_2196_);
v___x_2200_ = lean_box(0);
v_isShared_2201_ = v_isSharedCheck_2206_;
goto v_resetjp_2199_;
}
v_resetjp_2199_:
{
lean_object* v_ret_2202_; lean_object* v___x_2204_; 
v_ret_2202_ = lean_array_push(v_fst_2192_, v_a_2197_);
if (v_isShared_2201_ == 0)
{
lean_ctor_set(v___x_2200_, 0, v_ret_2202_);
v___x_2204_ = v___x_2200_;
goto v_reusejp_2203_;
}
else
{
lean_object* v_reuseFailAlloc_2205_; 
v_reuseFailAlloc_2205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2205_, 0, v_ret_2202_);
lean_ctor_set(v_reuseFailAlloc_2205_, 1, v_a_2198_);
v___x_2204_ = v_reuseFailAlloc_2205_;
goto v_reusejp_2203_;
}
v_reusejp_2203_:
{
return v___x_2204_;
}
}
}
else
{
lean_object* v_a_2207_; lean_object* v_a_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2215_; 
lean_dec(v_fst_2192_);
v_a_2207_ = lean_ctor_get(v___x_2196_, 0);
v_a_2208_ = lean_ctor_get(v___x_2196_, 1);
v_isSharedCheck_2215_ = !lean_is_exclusive(v___x_2196_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2210_ = v___x_2196_;
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_a_2208_);
lean_inc(v_a_2207_);
lean_dec(v___x_2196_);
v___x_2210_ = lean_box(0);
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
v_resetjp_2209_:
{
lean_object* v___x_2213_; 
if (v_isShared_2211_ == 0)
{
v___x_2213_ = v___x_2210_;
goto v_reusejp_2212_;
}
else
{
lean_object* v_reuseFailAlloc_2214_; 
v_reuseFailAlloc_2214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2214_, 0, v_a_2207_);
lean_ctor_set(v_reuseFailAlloc_2214_, 1, v_a_2208_);
v___x_2213_ = v_reuseFailAlloc_2214_;
goto v_reusejp_2212_;
}
v_reusejp_2212_:
{
return v___x_2213_;
}
}
}
}
else
{
lean_object* v___x_2217_; 
lean_dec(v_snd_2193_);
lean_dec_ref(v_f_2179_);
if (v_isShared_2191_ == 0)
{
lean_ctor_set(v___x_2190_, 0, v_fst_2192_);
v___x_2217_ = v___x_2190_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_fst_2192_);
lean_ctor_set(v_reuseFailAlloc_2218_, 1, v_a_2188_);
v___x_2217_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
return v___x_2217_;
}
}
}
}
else
{
lean_object* v_a_2220_; lean_object* v_a_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2228_; 
lean_dec_ref(v_f_2179_);
v_a_2220_ = lean_ctor_get(v___x_2186_, 0);
v_a_2221_ = lean_ctor_get(v___x_2186_, 1);
v_isSharedCheck_2228_ = !lean_is_exclusive(v___x_2186_);
if (v_isSharedCheck_2228_ == 0)
{
v___x_2223_ = v___x_2186_;
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_a_2221_);
lean_inc(v_a_2220_);
lean_dec(v___x_2186_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2226_; 
if (v_isShared_2224_ == 0)
{
v___x_2226_ = v___x_2223_;
goto v_reusejp_2225_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v_a_2220_);
lean_ctor_set(v_reuseFailAlloc_2227_, 1, v_a_2221_);
v___x_2226_ = v_reuseFailAlloc_2227_;
goto v_reusejp_2225_;
}
v_reusejp_2225_:
{
return v___x_2226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg___boxed(lean_object* v_arr_2229_, lean_object* v_f_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_){
_start:
{
lean_object* v_res_2233_; 
v_res_2233_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg(v_arr_2229_, v_f_2230_, v___y_2231_, v___y_2232_);
lean_dec_ref(v___y_2231_);
lean_dec_ref(v_arr_2229_);
return v_res_2233_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1(void){
_start:
{
lean_object* v___x_2235_; lean_object* v___x_2236_; 
v___x_2235_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__0));
v___x_2236_ = l_String_toRawSubstring_x27(v___x_2235_);
return v___x_2236_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9(void){
_start:
{
lean_object* v___x_2256_; lean_object* v___x_2257_; 
v___x_2256_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__8));
v___x_2257_ = l_String_toRawSubstring_x27(v___x_2256_);
return v___x_2257_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag(lean_object* v_tk_2294_, lean_object* v_n_2295_, lean_object* v_m_2296_, lean_object* v_vs_2297_, lean_object* v_cs_2298_, lean_object* v_a_2299_, lean_object* v_a_2300_){
_start:
{
size_t v___y_2302_; lean_object* v___y_2303_; lean_object* v___y_2304_; lean_object* v___y_2305_; lean_object* v___y_2306_; lean_object* v___y_2307_; lean_object* v___y_2350_; lean_object* v_props_2351_; lean_object* v_quotContext_2352_; lean_object* v_currMacroScope_2353_; lean_object* v_ref_2354_; lean_object* v___y_2355_; lean_object* v___y_2369_; lean_object* v___y_2370_; lean_object* v___y_2371_; lean_object* v___y_2372_; lean_object* v___y_2373_; size_t v___y_2403_; lean_object* v___y_2404_; lean_object* v___y_2405_; lean_object* v___y_2406_; lean_object* v___y_2407_; lean_object* v___y_2408_; lean_object* v___y_2409_; lean_object* v___x_2445_; lean_object* v_nId_2446_; lean_object* v_csArrs_2448_; lean_object* v___y_2449_; lean_object* v___y_2450_; lean_object* v___y_2480_; lean_object* v___y_2481_; lean_object* v___y_2482_; lean_object* v___y_2534_; lean_object* v___y_2535_; lean_object* v___x_2543_; lean_object* v_mId_2544_; uint8_t v___x_2545_; 
v___x_2445_ = l_Lean_TSyntax_getId(v_n_2295_);
v_nId_2446_ = l_Lean_Name_eraseMacroScopes(v___x_2445_);
lean_dec(v___x_2445_);
v___x_2543_ = l_Lean_TSyntax_getId(v_m_2296_);
v_mId_2544_ = l_Lean_Name_eraseMacroScopes(v___x_2543_);
lean_dec(v___x_2543_);
v___x_2545_ = lean_name_eq(v_nId_2446_, v_mId_2544_);
lean_dec(v_mId_2544_);
if (v___x_2545_ == 0)
{
uint8_t v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2546_ = 1;
v___x_2547_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__23));
lean_inc(v_nId_2446_);
v___x_2548_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_nId_2446_, v___x_2546_);
v___x_2549_ = lean_string_append(v___x_2547_, v___x_2548_);
lean_dec_ref(v___x_2548_);
v___x_2550_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2));
v___x_2551_ = lean_string_append(v___x_2549_, v___x_2550_);
v___x_2552_ = l_Lean_Macro_throwErrorAt___redArg(v_m_2296_, v___x_2551_, v_a_2299_, v_a_2300_);
if (lean_obj_tag(v___x_2552_) == 0)
{
lean_object* v_a_2553_; 
v_a_2553_ = lean_ctor_get(v___x_2552_, 1);
lean_inc(v_a_2553_);
lean_dec_ref_known(v___x_2552_, 2);
v___y_2534_ = v_a_2299_;
v___y_2535_ = v_a_2553_;
goto v___jp_2533_;
}
else
{
lean_object* v_a_2554_; lean_object* v_a_2555_; lean_object* v___x_2557_; uint8_t v_isShared_2558_; uint8_t v_isSharedCheck_2562_; 
lean_dec(v_nId_2446_);
lean_dec_ref(v_vs_2297_);
lean_dec(v_n_2295_);
v_a_2554_ = lean_ctor_get(v___x_2552_, 0);
v_a_2555_ = lean_ctor_get(v___x_2552_, 1);
v_isSharedCheck_2562_ = !lean_is_exclusive(v___x_2552_);
if (v_isSharedCheck_2562_ == 0)
{
v___x_2557_ = v___x_2552_;
v_isShared_2558_ = v_isSharedCheck_2562_;
goto v_resetjp_2556_;
}
else
{
lean_inc(v_a_2555_);
lean_inc(v_a_2554_);
lean_dec(v___x_2552_);
v___x_2557_ = lean_box(0);
v_isShared_2558_ = v_isSharedCheck_2562_;
goto v_resetjp_2556_;
}
v_resetjp_2556_:
{
lean_object* v___x_2560_; 
if (v_isShared_2558_ == 0)
{
v___x_2560_ = v___x_2557_;
goto v_reusejp_2559_;
}
else
{
lean_object* v_reuseFailAlloc_2561_; 
v_reuseFailAlloc_2561_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2561_, 0, v_a_2554_);
lean_ctor_set(v_reuseFailAlloc_2561_, 1, v_a_2555_);
v___x_2560_ = v_reuseFailAlloc_2561_;
goto v_reusejp_2559_;
}
v_reusejp_2559_:
{
return v___x_2560_;
}
}
}
}
else
{
v___y_2534_ = v_a_2299_;
v___y_2535_ = v_a_2300_;
goto v___jp_2533_;
}
v___jp_2301_:
{
uint8_t v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___f_2311_; lean_object* v___x_2312_; 
v___x_2308_ = 0;
v___x_2309_ = lean_box_usize(v___y_2302_);
v___x_2310_ = lean_box(v___x_2308_);
v___f_2311_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___boxed), 5, 2);
lean_closure_set(v___f_2311_, 0, v___x_2309_);
lean_closure_set(v___f_2311_, 1, v___x_2310_);
v___x_2312_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg(v___y_2306_, v___f_2311_, v___y_2303_, v___y_2305_);
lean_dec_ref(v___y_2306_);
if (lean_obj_tag(v___x_2312_) == 0)
{
lean_object* v_a_2313_; lean_object* v_a_2314_; lean_object* v___x_2315_; 
v_a_2313_ = lean_ctor_get(v___x_2312_, 0);
lean_inc(v_a_2313_);
v_a_2314_ = lean_ctor_get(v___x_2312_, 1);
lean_inc(v_a_2314_);
lean_dec_ref_known(v___x_2312_, 2);
v___x_2315_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0(v_a_2313_, v___y_2303_, v_a_2314_);
lean_dec(v_a_2313_);
if (lean_obj_tag(v___x_2315_) == 0)
{
lean_object* v_a_2316_; lean_object* v_a_2317_; lean_object* v___x_2319_; uint8_t v_isShared_2320_; uint8_t v_isSharedCheck_2339_; 
v_a_2316_ = lean_ctor_get(v___x_2315_, 0);
v_a_2317_ = lean_ctor_get(v___x_2315_, 1);
v_isSharedCheck_2339_ = !lean_is_exclusive(v___x_2315_);
if (v_isSharedCheck_2339_ == 0)
{
v___x_2319_ = v___x_2315_;
v_isShared_2320_ = v_isSharedCheck_2339_;
goto v_resetjp_2318_;
}
else
{
lean_inc(v_a_2317_);
lean_inc(v_a_2316_);
lean_dec(v___x_2315_);
v___x_2319_ = lean_box(0);
v_isShared_2320_ = v_isSharedCheck_2339_;
goto v_resetjp_2318_;
}
v_resetjp_2318_:
{
lean_object* v_quotContext_2321_; lean_object* v_currMacroScope_2322_; lean_object* v_ref_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2337_; 
v_quotContext_2321_ = lean_ctor_get(v___y_2303_, 1);
v_currMacroScope_2322_ = lean_ctor_get(v___y_2303_, 2);
v_ref_2323_ = lean_ctor_get(v___y_2303_, 5);
v___x_2324_ = l_Lean_SourceInfo_fromRef(v_ref_2323_, v___x_2308_);
v___x_2325_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3));
v___x_2326_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__1);
v___x_2327_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__2));
lean_inc(v_currMacroScope_2322_);
lean_inc(v_quotContext_2321_);
v___x_2328_ = l_Lean_addMacroScope(v_quotContext_2321_, v___x_2327_, v_currMacroScope_2322_);
v___x_2329_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__7));
lean_inc_n(v___x_2324_, 2);
v___x_2330_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2330_, 0, v___x_2324_);
lean_ctor_set(v___x_2330_, 1, v___x_2326_);
lean_ctor_set(v___x_2330_, 2, v___x_2328_);
lean_ctor_set(v___x_2330_, 3, v___x_2329_);
v___x_2331_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_2332_ = lean_box(2);
v___x_2333_ = l_Lean_Syntax_mkStrLit(v___y_2304_, v___x_2332_);
v___x_2334_ = l_Lean_Syntax_node3(v___x_2324_, v___x_2331_, v___x_2333_, v_a_2316_, v___y_2307_);
v___x_2335_ = l_Lean_Syntax_node2(v___x_2324_, v___x_2325_, v___x_2330_, v___x_2334_);
if (v_isShared_2320_ == 0)
{
lean_ctor_set(v___x_2319_, 0, v___x_2335_);
v___x_2337_ = v___x_2319_;
goto v_reusejp_2336_;
}
else
{
lean_object* v_reuseFailAlloc_2338_; 
v_reuseFailAlloc_2338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2338_, 0, v___x_2335_);
lean_ctor_set(v_reuseFailAlloc_2338_, 1, v_a_2317_);
v___x_2337_ = v_reuseFailAlloc_2338_;
goto v_reusejp_2336_;
}
v_reusejp_2336_:
{
return v___x_2337_;
}
}
}
else
{
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2304_);
return v___x_2315_;
}
}
else
{
lean_object* v_a_2340_; lean_object* v_a_2341_; lean_object* v___x_2343_; uint8_t v_isShared_2344_; uint8_t v_isSharedCheck_2348_; 
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2304_);
v_a_2340_ = lean_ctor_get(v___x_2312_, 0);
v_a_2341_ = lean_ctor_get(v___x_2312_, 1);
v_isSharedCheck_2348_ = !lean_is_exclusive(v___x_2312_);
if (v_isSharedCheck_2348_ == 0)
{
v___x_2343_ = v___x_2312_;
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
else
{
lean_inc(v_a_2341_);
lean_inc(v_a_2340_);
lean_dec(v___x_2312_);
v___x_2343_ = lean_box(0);
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
v_resetjp_2342_:
{
lean_object* v___x_2346_; 
if (v_isShared_2344_ == 0)
{
v___x_2346_ = v___x_2343_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2347_; 
v_reuseFailAlloc_2347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2347_, 0, v_a_2340_);
lean_ctor_set(v_reuseFailAlloc_2347_, 1, v_a_2341_);
v___x_2346_ = v_reuseFailAlloc_2347_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
return v___x_2346_;
}
}
}
}
v___jp_2349_:
{
uint8_t v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; 
v___x_2356_ = 0;
v___x_2357_ = l_Lean_SourceInfo_fromRef(v_ref_2354_, v___x_2356_);
v___x_2358_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__3));
v___x_2359_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__9);
v___x_2360_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__11));
lean_inc(v_currMacroScope_2353_);
lean_inc(v_quotContext_2352_);
v___x_2361_ = l_Lean_addMacroScope(v_quotContext_2352_, v___x_2360_, v_currMacroScope_2353_);
v___x_2362_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__14));
lean_inc_n(v___x_2357_, 2);
v___x_2363_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2363_, 0, v___x_2357_);
lean_ctor_set(v___x_2363_, 1, v___x_2359_);
lean_ctor_set(v___x_2363_, 2, v___x_2361_);
lean_ctor_set(v___x_2363_, 3, v___x_2362_);
v___x_2364_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_2365_ = l_Lean_Syntax_node3(v___x_2357_, v___x_2364_, v_n_2295_, v_props_2351_, v___y_2350_);
v___x_2366_ = l_Lean_Syntax_node2(v___x_2357_, v___x_2358_, v___x_2363_, v___x_2365_);
v___x_2367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2367_, 0, v___x_2366_);
lean_ctor_set(v___x_2367_, 1, v___y_2355_);
return v___x_2367_;
}
v___jp_2368_:
{
lean_object* v_quotContext_2374_; lean_object* v_currMacroScope_2375_; lean_object* v_ref_2376_; uint8_t v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; 
v_quotContext_2374_ = lean_ctor_get(v___y_2372_, 1);
v_currMacroScope_2375_ = lean_ctor_get(v___y_2372_, 2);
v_ref_2376_ = lean_ctor_get(v___y_2372_, 5);
v___x_2377_ = 0;
v___x_2378_ = l_Lean_SourceInfo_fromRef(v_ref_2376_, v___x_2377_);
v___x_2379_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16));
v___x_2380_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4));
lean_inc_n(v___x_2378_, 9);
v___x_2381_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2378_);
lean_ctor_set(v___x_2381_, 1, v___x_2380_);
v___x_2382_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_2383_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v___x_2384_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31));
v___x_2385_ = l_Lean_Syntax_SepArray_ofElems(v___x_2384_, v___y_2369_);
lean_dec_ref(v___y_2369_);
v___x_2386_ = l_Array_append___redArg(v___x_2383_, v___x_2385_);
lean_dec_ref(v___x_2385_);
v___x_2387_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2387_, 0, v___x_2378_);
lean_ctor_set(v___x_2387_, 1, v___x_2382_);
lean_ctor_set(v___x_2387_, 2, v___x_2386_);
v___x_2388_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__17));
v___x_2389_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2389_, 0, v___x_2378_);
lean_ctor_set(v___x_2389_, 1, v___x_2388_);
v___x_2390_ = l_Lean_Syntax_node2(v___x_2378_, v___x_2382_, v___x_2387_, v___x_2389_);
v___x_2391_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19));
v___x_2392_ = l_Lean_Syntax_SepArray_ofElems(v___x_2384_, v___y_2370_);
lean_dec_ref(v___y_2370_);
v___x_2393_ = l_Array_append___redArg(v___x_2383_, v___x_2392_);
lean_dec_ref(v___x_2392_);
v___x_2394_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2394_, 0, v___x_2378_);
lean_ctor_set(v___x_2394_, 1, v___x_2382_);
lean_ctor_set(v___x_2394_, 2, v___x_2393_);
v___x_2395_ = l_Lean_Syntax_node1(v___x_2378_, v___x_2391_, v___x_2394_);
v___x_2396_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__21));
v___x_2397_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2397_, 0, v___x_2378_);
lean_ctor_set(v___x_2397_, 1, v___x_2382_);
lean_ctor_set(v___x_2397_, 2, v___x_2383_);
lean_inc_ref(v___x_2397_);
v___x_2398_ = l_Lean_Syntax_node1(v___x_2378_, v___x_2396_, v___x_2397_);
v___x_2399_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_2400_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2400_, 0, v___x_2378_);
lean_ctor_set(v___x_2400_, 1, v___x_2399_);
v___x_2401_ = l_Lean_Syntax_node6(v___x_2378_, v___x_2379_, v___x_2381_, v___x_2390_, v___x_2395_, v___x_2398_, v___x_2397_, v___x_2400_);
v___y_2350_ = v___y_2371_;
v_props_2351_ = v___x_2401_;
v_quotContext_2352_ = v_quotContext_2374_;
v_currMacroScope_2353_ = v_currMacroScope_2375_;
v_ref_2354_ = v_ref_2376_;
v___y_2355_ = v___y_2373_;
goto v___jp_2349_;
}
v___jp_2402_:
{
if (lean_obj_tag(v___y_2409_) == 0)
{
lean_dec(v_n_2295_);
v___y_2302_ = v___y_2403_;
v___y_2303_ = v___y_2406_;
v___y_2304_ = v___y_2407_;
v___y_2305_ = v___y_2404_;
v___y_2306_ = v___y_2405_;
v___y_2307_ = v___y_2408_;
goto v___jp_2301_;
}
else
{
lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v___x_2412_; 
lean_dec_ref_known(v___y_2409_, 1);
lean_dec_ref(v___y_2407_);
v___x_2410_ = lean_unsigned_to_nat(0u);
v___x_2411_ = lean_array_get_size(v___y_2405_);
v___x_2412_ = lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4(v___y_2405_, v___x_2410_, v___x_2411_, v___y_2406_, v___y_2404_);
if (lean_obj_tag(v___x_2412_) == 0)
{
lean_object* v_a_2413_; lean_object* v_a_2414_; lean_object* v___x_2415_; 
v_a_2413_ = lean_ctor_get(v___x_2412_, 0);
lean_inc(v_a_2413_);
v_a_2414_ = lean_ctor_get(v___x_2412_, 1);
lean_inc(v_a_2414_);
lean_dec_ref_known(v___x_2412_, 2);
v___x_2415_ = lp_proofwidgets_Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5(v___y_2405_, v___x_2410_, v___x_2411_, v___y_2406_, v_a_2414_);
lean_dec_ref(v___y_2405_);
if (lean_obj_tag(v___x_2415_) == 0)
{
lean_object* v_a_2416_; lean_object* v_a_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; uint8_t v___x_2420_; 
v_a_2416_ = lean_ctor_get(v___x_2415_, 0);
lean_inc(v_a_2416_);
v_a_2417_ = lean_ctor_get(v___x_2415_, 1);
lean_inc(v_a_2417_);
lean_dec_ref_known(v___x_2415_, 2);
v___x_2418_ = lean_array_get_size(v_a_2413_);
v___x_2419_ = lean_unsigned_to_nat(1u);
v___x_2420_ = lean_nat_dec_eq(v___x_2418_, v___x_2419_);
if (v___x_2420_ == 0)
{
v___y_2369_ = v_a_2413_;
v___y_2370_ = v_a_2416_;
v___y_2371_ = v___y_2408_;
v___y_2372_ = v___y_2406_;
v___y_2373_ = v_a_2417_;
goto v___jp_2368_;
}
else
{
lean_object* v___x_2421_; uint8_t v___x_2422_; 
v___x_2421_ = lean_array_get_size(v_a_2416_);
v___x_2422_ = lean_nat_dec_eq(v___x_2421_, v___x_2410_);
if (v___x_2422_ == 0)
{
v___y_2369_ = v_a_2413_;
v___y_2370_ = v_a_2416_;
v___y_2371_ = v___y_2408_;
v___y_2372_ = v___y_2406_;
v___y_2373_ = v_a_2417_;
goto v___jp_2368_;
}
else
{
lean_object* v_quotContext_2423_; lean_object* v_currMacroScope_2424_; lean_object* v_ref_2425_; lean_object* v___x_2426_; 
lean_dec(v_a_2416_);
v_quotContext_2423_ = lean_ctor_get(v___y_2406_, 1);
v_currMacroScope_2424_ = lean_ctor_get(v___y_2406_, 2);
v_ref_2425_ = lean_ctor_get(v___y_2406_, 5);
v___x_2426_ = lean_array_fget(v_a_2413_, v___x_2410_);
lean_dec(v_a_2413_);
v___y_2350_ = v___y_2408_;
v_props_2351_ = v___x_2426_;
v_quotContext_2352_ = v_quotContext_2423_;
v_currMacroScope_2353_ = v_currMacroScope_2424_;
v_ref_2354_ = v_ref_2425_;
v___y_2355_ = v_a_2417_;
goto v___jp_2349_;
}
}
}
else
{
lean_object* v_a_2427_; lean_object* v_a_2428_; lean_object* v___x_2430_; uint8_t v_isShared_2431_; uint8_t v_isSharedCheck_2435_; 
lean_dec(v_a_2413_);
lean_dec(v___y_2408_);
lean_dec(v_n_2295_);
v_a_2427_ = lean_ctor_get(v___x_2415_, 0);
v_a_2428_ = lean_ctor_get(v___x_2415_, 1);
v_isSharedCheck_2435_ = !lean_is_exclusive(v___x_2415_);
if (v_isSharedCheck_2435_ == 0)
{
v___x_2430_ = v___x_2415_;
v_isShared_2431_ = v_isSharedCheck_2435_;
goto v_resetjp_2429_;
}
else
{
lean_inc(v_a_2428_);
lean_inc(v_a_2427_);
lean_dec(v___x_2415_);
v___x_2430_ = lean_box(0);
v_isShared_2431_ = v_isSharedCheck_2435_;
goto v_resetjp_2429_;
}
v_resetjp_2429_:
{
lean_object* v___x_2433_; 
if (v_isShared_2431_ == 0)
{
v___x_2433_ = v___x_2430_;
goto v_reusejp_2432_;
}
else
{
lean_object* v_reuseFailAlloc_2434_; 
v_reuseFailAlloc_2434_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2434_, 0, v_a_2427_);
lean_ctor_set(v_reuseFailAlloc_2434_, 1, v_a_2428_);
v___x_2433_ = v_reuseFailAlloc_2434_;
goto v_reusejp_2432_;
}
v_reusejp_2432_:
{
return v___x_2433_;
}
}
}
}
else
{
lean_object* v_a_2436_; lean_object* v_a_2437_; lean_object* v___x_2439_; uint8_t v_isShared_2440_; uint8_t v_isSharedCheck_2444_; 
lean_dec(v___y_2408_);
lean_dec_ref(v___y_2405_);
lean_dec(v_n_2295_);
v_a_2436_ = lean_ctor_get(v___x_2412_, 0);
v_a_2437_ = lean_ctor_get(v___x_2412_, 1);
v_isSharedCheck_2444_ = !lean_is_exclusive(v___x_2412_);
if (v_isSharedCheck_2444_ == 0)
{
v___x_2439_ = v___x_2412_;
v_isShared_2440_ = v_isSharedCheck_2444_;
goto v_resetjp_2438_;
}
else
{
lean_inc(v_a_2437_);
lean_inc(v_a_2436_);
lean_dec(v___x_2412_);
v___x_2439_ = lean_box(0);
v_isShared_2440_ = v_isSharedCheck_2444_;
goto v_resetjp_2438_;
}
v_resetjp_2438_:
{
lean_object* v___x_2442_; 
if (v_isShared_2440_ == 0)
{
v___x_2442_ = v___x_2439_;
goto v_reusejp_2441_;
}
else
{
lean_object* v_reuseFailAlloc_2443_; 
v_reuseFailAlloc_2443_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2443_, 0, v_a_2436_);
lean_ctor_set(v_reuseFailAlloc_2443_, 1, v_a_2437_);
v___x_2442_ = v_reuseFailAlloc_2443_;
goto v_reusejp_2441_;
}
v_reusejp_2441_:
{
return v___x_2442_;
}
}
}
}
}
v___jp_2447_:
{
lean_object* v___x_2451_; 
v___x_2451_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___at___00ProofWidgets_Jsx_transformTag_spec__0(v_csArrs_2448_, v___y_2449_, v___y_2450_);
lean_dec_ref(v_csArrs_2448_);
if (lean_obj_tag(v___x_2451_) == 0)
{
lean_object* v_a_2452_; lean_object* v_a_2453_; size_t v_sz_2454_; size_t v___x_2455_; lean_object* v___x_2456_; 
v_a_2452_ = lean_ctor_get(v___x_2451_, 0);
lean_inc(v_a_2452_);
v_a_2453_ = lean_ctor_get(v___x_2451_, 1);
lean_inc(v_a_2453_);
lean_dec_ref_known(v___x_2451_, 2);
v_sz_2454_ = lean_array_size(v_vs_2297_);
v___x_2455_ = ((size_t)0ULL);
v___x_2456_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__1(v_sz_2454_, v___x_2455_, v_vs_2297_, v___y_2449_, v_a_2453_);
if (lean_obj_tag(v___x_2456_) == 0)
{
lean_object* v_a_2457_; lean_object* v_a_2458_; uint8_t v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; 
v_a_2457_ = lean_ctor_get(v___x_2456_, 0);
lean_inc(v_a_2457_);
v_a_2458_ = lean_ctor_get(v___x_2456_, 1);
lean_inc(v_a_2458_);
lean_dec_ref_known(v___x_2456_, 2);
v___x_2459_ = 1;
v___x_2460_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_nId_2446_, v___x_2459_);
v___x_2461_ = lean_unsigned_to_nat(0u);
v___x_2462_ = lean_string_utf8_get_opt(v___x_2460_, v___x_2461_);
if (lean_obj_tag(v___x_2462_) == 0)
{
v___y_2403_ = v___x_2455_;
v___y_2404_ = v_a_2458_;
v___y_2405_ = v_a_2457_;
v___y_2406_ = v___y_2449_;
v___y_2407_ = v___x_2460_;
v___y_2408_ = v_a_2452_;
v___y_2409_ = v___x_2462_;
goto v___jp_2402_;
}
else
{
lean_object* v_val_2463_; uint32_t v___x_2464_; uint32_t v___x_2465_; uint8_t v___x_2466_; 
v_val_2463_ = lean_ctor_get(v___x_2462_, 0);
lean_inc(v_val_2463_);
v___x_2464_ = 65;
v___x_2465_ = lean_unbox_uint32(v_val_2463_);
v___x_2466_ = lean_uint32_dec_le(v___x_2464_, v___x_2465_);
if (v___x_2466_ == 0)
{
lean_dec_ref_known(v___x_2462_, 1);
lean_dec(v_val_2463_);
lean_dec(v_n_2295_);
v___y_2302_ = v___x_2455_;
v___y_2303_ = v___y_2449_;
v___y_2304_ = v___x_2460_;
v___y_2305_ = v_a_2458_;
v___y_2306_ = v_a_2457_;
v___y_2307_ = v_a_2452_;
goto v___jp_2301_;
}
else
{
uint32_t v___x_2467_; uint32_t v___x_2468_; uint8_t v___x_2469_; 
v___x_2467_ = 90;
v___x_2468_ = lean_unbox_uint32(v_val_2463_);
lean_dec(v_val_2463_);
v___x_2469_ = lean_uint32_dec_le(v___x_2468_, v___x_2467_);
if (v___x_2469_ == 0)
{
lean_dec_ref_known(v___x_2462_, 1);
lean_dec(v_n_2295_);
v___y_2302_ = v___x_2455_;
v___y_2303_ = v___y_2449_;
v___y_2304_ = v___x_2460_;
v___y_2305_ = v_a_2458_;
v___y_2306_ = v_a_2457_;
v___y_2307_ = v_a_2452_;
goto v___jp_2301_;
}
else
{
v___y_2403_ = v___x_2455_;
v___y_2404_ = v_a_2458_;
v___y_2405_ = v_a_2457_;
v___y_2406_ = v___y_2449_;
v___y_2407_ = v___x_2460_;
v___y_2408_ = v_a_2452_;
v___y_2409_ = v___x_2462_;
goto v___jp_2402_;
}
}
}
}
else
{
lean_object* v_a_2470_; lean_object* v_a_2471_; lean_object* v___x_2473_; uint8_t v_isShared_2474_; uint8_t v_isSharedCheck_2478_; 
lean_dec(v_a_2452_);
lean_dec(v_nId_2446_);
lean_dec(v_n_2295_);
v_a_2470_ = lean_ctor_get(v___x_2456_, 0);
v_a_2471_ = lean_ctor_get(v___x_2456_, 1);
v_isSharedCheck_2478_ = !lean_is_exclusive(v___x_2456_);
if (v_isSharedCheck_2478_ == 0)
{
v___x_2473_ = v___x_2456_;
v_isShared_2474_ = v_isSharedCheck_2478_;
goto v_resetjp_2472_;
}
else
{
lean_inc(v_a_2471_);
lean_inc(v_a_2470_);
lean_dec(v___x_2456_);
v___x_2473_ = lean_box(0);
v_isShared_2474_ = v_isSharedCheck_2478_;
goto v_resetjp_2472_;
}
v_resetjp_2472_:
{
lean_object* v___x_2476_; 
if (v_isShared_2474_ == 0)
{
v___x_2476_ = v___x_2473_;
goto v_reusejp_2475_;
}
else
{
lean_object* v_reuseFailAlloc_2477_; 
v_reuseFailAlloc_2477_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2477_, 0, v_a_2470_);
lean_ctor_set(v_reuseFailAlloc_2477_, 1, v_a_2471_);
v___x_2476_ = v_reuseFailAlloc_2477_;
goto v_reusejp_2475_;
}
v_reusejp_2475_:
{
return v___x_2476_;
}
}
}
}
else
{
lean_dec(v_nId_2446_);
lean_dec_ref(v_vs_2297_);
lean_dec(v_n_2295_);
return v___x_2451_;
}
}
v___jp_2479_:
{
lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; size_t v_sz_2486_; size_t v___x_2487_; lean_object* v___x_2488_; 
v___x_2483_ = lean_unsigned_to_nat(0u);
v___x_2484_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__22));
v___x_2485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2485_, 0, v___y_2482_);
lean_ctor_set(v___x_2485_, 1, v___x_2484_);
v_sz_2486_ = lean_array_size(v_cs_2298_);
v___x_2487_ = ((size_t)0ULL);
v___x_2488_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6(v_cs_2298_, v_sz_2486_, v___x_2487_, v___x_2485_, v___y_2481_, v___y_2480_);
if (lean_obj_tag(v___x_2488_) == 0)
{
lean_object* v_a_2489_; lean_object* v_snd_2490_; lean_object* v___x_2492_; uint8_t v_isShared_2493_; uint8_t v_isSharedCheck_2522_; 
v_a_2489_ = lean_ctor_get(v___x_2488_, 0);
lean_inc(v_a_2489_);
v_snd_2490_ = lean_ctor_get(v_a_2489_, 1);
v_isSharedCheck_2522_ = !lean_is_exclusive(v_a_2489_);
if (v_isSharedCheck_2522_ == 0)
{
lean_object* v_unused_2523_; 
v_unused_2523_ = lean_ctor_get(v_a_2489_, 0);
lean_dec(v_unused_2523_);
v___x_2492_ = v_a_2489_;
v_isShared_2493_ = v_isSharedCheck_2522_;
goto v_resetjp_2491_;
}
else
{
lean_inc(v_snd_2490_);
lean_dec(v_a_2489_);
v___x_2492_ = lean_box(0);
v_isShared_2493_ = v_isSharedCheck_2522_;
goto v_resetjp_2491_;
}
v_resetjp_2491_:
{
lean_object* v_a_2494_; lean_object* v_fst_2495_; lean_object* v_snd_2496_; lean_object* v___x_2498_; uint8_t v_isShared_2499_; uint8_t v_isSharedCheck_2521_; 
v_a_2494_ = lean_ctor_get(v___x_2488_, 1);
lean_inc(v_a_2494_);
lean_dec_ref_known(v___x_2488_, 2);
v_fst_2495_ = lean_ctor_get(v_snd_2490_, 0);
v_snd_2496_ = lean_ctor_get(v_snd_2490_, 1);
v_isSharedCheck_2521_ = !lean_is_exclusive(v_snd_2490_);
if (v_isSharedCheck_2521_ == 0)
{
v___x_2498_ = v_snd_2490_;
v_isShared_2499_ = v_isSharedCheck_2521_;
goto v_resetjp_2497_;
}
else
{
lean_inc(v_snd_2496_);
lean_inc(v_fst_2495_);
lean_dec(v_snd_2490_);
v___x_2498_ = lean_box(0);
v_isShared_2499_ = v_isSharedCheck_2521_;
goto v_resetjp_2497_;
}
v_resetjp_2497_:
{
lean_object* v___x_2500_; uint8_t v___x_2501_; 
v___x_2500_ = lean_array_get_size(v_snd_2496_);
v___x_2501_ = lean_nat_dec_eq(v___x_2500_, v___x_2483_);
if (v___x_2501_ == 0)
{
lean_object* v_ref_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2507_; 
v_ref_2502_ = lean_ctor_get(v___y_2481_, 5);
v___x_2503_ = l_Lean_SourceInfo_fromRef(v_ref_2502_, v___x_2501_);
v___x_2504_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__1));
v___x_2505_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__2));
lean_inc(v___x_2503_);
if (v_isShared_2499_ == 0)
{
lean_ctor_set_tag(v___x_2498_, 2);
lean_ctor_set(v___x_2498_, 1, v___x_2505_);
lean_ctor_set(v___x_2498_, 0, v___x_2503_);
v___x_2507_ = v___x_2498_;
goto v_reusejp_2506_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v___x_2503_);
lean_ctor_set(v_reuseFailAlloc_2520_, 1, v___x_2505_);
v___x_2507_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2506_;
}
v_reusejp_2506_:
{
lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2516_; 
v___x_2508_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_2509_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v___x_2510_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__31));
v___x_2511_ = l_Lean_Syntax_SepArray_ofElems(v___x_2510_, v_snd_2496_);
lean_dec(v_snd_2496_);
v___x_2512_ = l_Array_append___redArg(v___x_2509_, v___x_2511_);
lean_dec_ref(v___x_2511_);
lean_inc_n(v___x_2503_, 2);
v___x_2513_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2513_, 0, v___x_2503_);
lean_ctor_set(v___x_2513_, 1, v___x_2508_);
lean_ctor_set(v___x_2513_, 2, v___x_2512_);
v___x_2514_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__4));
if (v_isShared_2493_ == 0)
{
lean_ctor_set_tag(v___x_2492_, 2);
lean_ctor_set(v___x_2492_, 1, v___x_2514_);
lean_ctor_set(v___x_2492_, 0, v___x_2503_);
v___x_2516_ = v___x_2492_;
goto v_reusejp_2515_;
}
else
{
lean_object* v_reuseFailAlloc_2519_; 
v_reuseFailAlloc_2519_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2519_, 0, v___x_2503_);
lean_ctor_set(v_reuseFailAlloc_2519_, 1, v___x_2514_);
v___x_2516_ = v_reuseFailAlloc_2519_;
goto v_reusejp_2515_;
}
v_reusejp_2515_:
{
lean_object* v___x_2517_; lean_object* v___x_2518_; 
v___x_2517_ = l_Lean_Syntax_node3(v___x_2503_, v___x_2504_, v___x_2507_, v___x_2513_, v___x_2516_);
v___x_2518_ = lean_array_push(v_fst_2495_, v___x_2517_);
v_csArrs_2448_ = v___x_2518_;
v___y_2449_ = v___y_2481_;
v___y_2450_ = v_a_2494_;
goto v___jp_2447_;
}
}
}
else
{
lean_del_object(v___x_2498_);
lean_dec(v_snd_2496_);
lean_del_object(v___x_2492_);
v_csArrs_2448_ = v_fst_2495_;
v___y_2449_ = v___y_2481_;
v___y_2450_ = v_a_2494_;
goto v___jp_2447_;
}
}
}
}
else
{
lean_object* v_a_2524_; lean_object* v_a_2525_; lean_object* v___x_2527_; uint8_t v_isShared_2528_; uint8_t v_isSharedCheck_2532_; 
lean_dec(v_nId_2446_);
lean_dec_ref(v_vs_2297_);
lean_dec(v_n_2295_);
v_a_2524_ = lean_ctor_get(v___x_2488_, 0);
v_a_2525_ = lean_ctor_get(v___x_2488_, 1);
v_isSharedCheck_2532_ = !lean_is_exclusive(v___x_2488_);
if (v_isSharedCheck_2532_ == 0)
{
v___x_2527_ = v___x_2488_;
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
else
{
lean_inc(v_a_2525_);
lean_inc(v_a_2524_);
lean_dec(v___x_2488_);
v___x_2527_ = lean_box(0);
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
v_resetjp_2526_:
{
lean_object* v___x_2530_; 
if (v_isShared_2528_ == 0)
{
v___x_2530_ = v___x_2527_;
goto v_reusejp_2529_;
}
else
{
lean_object* v_reuseFailAlloc_2531_; 
v_reuseFailAlloc_2531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2531_, 0, v_a_2524_);
lean_ctor_set(v_reuseFailAlloc_2531_, 1, v_a_2525_);
v___x_2530_ = v_reuseFailAlloc_2531_;
goto v_reusejp_2529_;
}
v_reusejp_2529_:
{
return v___x_2530_;
}
}
}
}
v___jp_2533_:
{
lean_object* v___x_2536_; 
v___x_2536_ = l_Lean_Syntax_getTailInfo(v_tk_2294_);
if (lean_obj_tag(v___x_2536_) == 0)
{
lean_object* v_trailing_2537_; lean_object* v_str_2538_; lean_object* v_startPos_2539_; lean_object* v_stopPos_2540_; lean_object* v___x_2541_; 
v_trailing_2537_ = lean_ctor_get(v___x_2536_, 2);
lean_inc_ref(v_trailing_2537_);
lean_dec_ref_known(v___x_2536_, 4);
v_str_2538_ = lean_ctor_get(v_trailing_2537_, 0);
lean_inc_ref(v_str_2538_);
v_startPos_2539_ = lean_ctor_get(v_trailing_2537_, 1);
lean_inc(v_startPos_2539_);
v_stopPos_2540_ = lean_ctor_get(v_trailing_2537_, 2);
lean_inc(v_stopPos_2540_);
lean_dec_ref(v_trailing_2537_);
v___x_2541_ = lean_string_utf8_extract(v_str_2538_, v_startPos_2539_, v_stopPos_2540_);
lean_dec(v_stopPos_2540_);
lean_dec(v_startPos_2539_);
lean_dec_ref(v_str_2538_);
v___y_2480_ = v___y_2535_;
v___y_2481_ = v___y_2534_;
v___y_2482_ = v___x_2541_;
goto v___jp_2479_;
}
else
{
lean_object* v___x_2542_; 
lean_dec(v___x_2536_);
v___x_2542_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instInhabitedHtml_default___closed__0));
v___y_2480_ = v___y_2535_;
v___y_2481_ = v___y_2534_;
v___y_2482_ = v___x_2542_;
goto v___jp_2479_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_transformTag___boxed(lean_object* v_tk_2563_, lean_object* v_n_2564_, lean_object* v_m_2565_, lean_object* v_vs_2566_, lean_object* v_cs_2567_, lean_object* v_a_2568_, lean_object* v_a_2569_){
_start:
{
lean_object* v_res_2570_; 
v_res_2570_ = lp_proofwidgets_ProofWidgets_Jsx_transformTag(v_tk_2563_, v_n_2564_, v_m_2565_, v_vs_2566_, v_cs_2567_, v_a_2568_, v_a_2569_);
lean_dec_ref(v_a_2568_);
lean_dec_ref(v_cs_2567_);
lean_dec(v_m_2565_);
lean_dec(v_tk_2563_);
return v_res_2570_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3(lean_object* v_00_u03b1_2571_, lean_object* v_00_u03b2_2572_, lean_object* v_arr_2573_, lean_object* v_f_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_){
_start:
{
lean_object* v___x_2577_; 
v___x_2577_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___redArg(v_arr_2573_, v_f_2574_, v___y_2575_, v___y_2576_);
return v___x_2577_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3___boxed(lean_object* v_00_u03b1_2578_, lean_object* v_00_u03b2_2579_, lean_object* v_arr_2580_, lean_object* v_f_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_){
_start:
{
lean_object* v_res_2584_; 
v_res_2584_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3(v_00_u03b1_2578_, v_00_u03b2_2579_, v_arr_2580_, v_f_2581_, v___y_2582_, v___y_2583_);
lean_dec_ref(v___y_2582_);
lean_dec_ref(v_arr_2580_);
return v_res_2584_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4(lean_object* v_00_u03b1_2585_, lean_object* v_00_u03b2_2586_, lean_object* v_f_2587_, lean_object* v_as_2588_, size_t v_sz_2589_, size_t v_i_2590_, lean_object* v_b_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_){
_start:
{
lean_object* v___x_2594_; 
v___x_2594_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___redArg(v_f_2587_, v_as_2588_, v_sz_2589_, v_i_2590_, v_b_2591_, v___y_2592_, v___y_2593_);
return v___x_2594_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4___boxed(lean_object* v_00_u03b1_2595_, lean_object* v_00_u03b2_2596_, lean_object* v_f_2597_, lean_object* v_as_2598_, lean_object* v_sz_2599_, lean_object* v_i_2600_, lean_object* v_b_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_){
_start:
{
size_t v_sz_boxed_2604_; size_t v_i_boxed_2605_; lean_object* v_res_2606_; 
v_sz_boxed_2604_ = lean_unbox_usize(v_sz_2599_);
lean_dec(v_sz_2599_);
v_i_boxed_2605_ = lean_unbox_usize(v_i_2600_);
lean_dec(v_i_2600_);
v_res_2606_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Util_foldInlsM___at___00ProofWidgets_Jsx_transformTag_spec__3_spec__4(v_00_u03b1_2595_, v_00_u03b2_2596_, v_f_2597_, v_as_2598_, v_sz_boxed_2604_, v_i_boxed_2605_, v_b_2601_, v___y_2602_, v___y_2603_);
lean_dec_ref(v___y_2602_);
lean_dec_ref(v_as_2598_);
return v_res_2606_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6(lean_object* v_as_2607_, size_t v_i_2608_, size_t v_stop_2609_, lean_object* v_b_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_){
_start:
{
lean_object* v___x_2613_; 
v___x_2613_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___redArg(v_as_2607_, v_i_2608_, v_stop_2609_, v_b_2610_, v___y_2612_);
return v___x_2613_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6___boxed(lean_object* v_as_2614_, lean_object* v_i_2615_, lean_object* v_stop_2616_, lean_object* v_b_2617_, lean_object* v___y_2618_, lean_object* v___y_2619_){
_start:
{
size_t v_i_boxed_2620_; size_t v_stop_boxed_2621_; lean_object* v_res_2622_; 
v_i_boxed_2620_ = lean_unbox_usize(v_i_2615_);
lean_dec(v_i_2615_);
v_stop_boxed_2621_ = lean_unbox_usize(v_stop_2616_);
lean_dec(v_stop_2616_);
v_res_2622_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__4_spec__6(v_as_2614_, v_i_boxed_2620_, v_stop_boxed_2621_, v_b_2617_, v___y_2618_, v___y_2619_);
lean_dec_ref(v___y_2618_);
lean_dec_ref(v_as_2614_);
return v_res_2622_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0(size_t v_sz_2623_, size_t v_i_2624_, lean_object* v_bs_2625_){
_start:
{
uint8_t v___x_2626_; 
v___x_2626_ = lean_usize_dec_lt(v_i_2624_, v_sz_2623_);
if (v___x_2626_ == 0)
{
lean_object* v___x_2627_; 
v___x_2627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2627_, 0, v_bs_2625_);
return v___x_2627_;
}
else
{
lean_object* v_v_2628_; lean_object* v___x_2629_; lean_object* v_bs_x27_2630_; size_t v___x_2631_; size_t v___x_2632_; lean_object* v___x_2633_; 
v_v_2628_ = lean_array_uget(v_bs_2625_, v_i_2624_);
v___x_2629_ = lean_unsigned_to_nat(0u);
v_bs_x27_2630_ = lean_array_uset(v_bs_2625_, v_i_2624_, v___x_2629_);
v___x_2631_ = ((size_t)1ULL);
v___x_2632_ = lean_usize_add(v_i_2624_, v___x_2631_);
v___x_2633_ = lean_array_uset(v_bs_x27_2630_, v_i_2624_, v_v_2628_);
v_i_2624_ = v___x_2632_;
v_bs_2625_ = v___x_2633_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0___boxed(lean_object* v_sz_2635_, lean_object* v_i_2636_, lean_object* v_bs_2637_){
_start:
{
size_t v_sz_boxed_2638_; size_t v_i_boxed_2639_; lean_object* v_res_2640_; 
v_sz_boxed_2638_ = lean_unbox_usize(v_sz_2635_);
lean_dec(v_sz_2635_);
v_i_boxed_2639_ = lean_unbox_usize(v_i_2636_);
lean_dec(v_i_2636_);
v_res_2640_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0(v_sz_boxed_2638_, v_i_boxed_2639_, v_bs_2637_);
return v_res_2640_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1(lean_object* v_x_2641_, lean_object* v_a_2642_, lean_object* v_a_2643_){
_start:
{
lean_object* v___x_2644_; uint8_t v___x_2645_; 
v___x_2644_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1));
lean_inc(v_x_2641_);
v___x_2645_ = l_Lean_Syntax_isOfKind(v_x_2641_, v___x_2644_);
if (v___x_2645_ == 0)
{
lean_object* v___x_2646_; lean_object* v___x_2647_; 
lean_dec(v_x_2641_);
v___x_2646_ = lean_box(1);
v___x_2647_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2647_, 0, v___x_2646_);
lean_ctor_set(v___x_2647_, 1, v_a_2643_);
return v___x_2647_;
}
else
{
lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; uint8_t v___x_2651_; 
v___x_2648_ = lean_unsigned_to_nat(0u);
v___x_2649_ = l_Lean_Syntax_getArg(v_x_2641_, v___x_2648_);
lean_dec(v_x_2641_);
v___x_2650_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1));
lean_inc(v___x_2649_);
v___x_2651_ = l_Lean_Syntax_isOfKind(v___x_2649_, v___x_2650_);
if (v___x_2651_ == 0)
{
lean_object* v___x_2652_; uint8_t v___x_2653_; 
v___x_2652_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1));
lean_inc(v___x_2649_);
v___x_2653_ = l_Lean_Syntax_isOfKind(v___x_2649_, v___x_2652_);
if (v___x_2653_ == 0)
{
lean_object* v___x_2654_; lean_object* v___x_2655_; 
lean_dec(v___x_2649_);
v___x_2654_ = lean_box(1);
v___x_2655_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2655_, 0, v___x_2654_);
lean_ctor_set(v___x_2655_, 1, v_a_2643_);
return v___x_2655_;
}
else
{
lean_object* v___x_2656_; lean_object* v_n_2657_; lean_object* v___x_2658_; uint8_t v___x_2659_; 
v___x_2656_ = lean_unsigned_to_nat(1u);
v_n_2657_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2656_);
v___x_2658_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3));
lean_inc(v_n_2657_);
v___x_2659_ = l_Lean_Syntax_isOfKind(v_n_2657_, v___x_2658_);
if (v___x_2659_ == 0)
{
lean_object* v___x_2660_; lean_object* v___x_2661_; 
lean_dec(v_n_2657_);
lean_dec(v___x_2649_);
v___x_2660_ = lean_box(1);
v___x_2661_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2661_, 0, v___x_2660_);
lean_ctor_set(v___x_2661_, 1, v_a_2643_);
return v___x_2661_;
}
else
{
lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; size_t v_sz_2665_; size_t v___x_2666_; lean_object* v___x_2667_; 
v___x_2662_ = lean_unsigned_to_nat(2u);
v___x_2663_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2662_);
v___x_2664_ = l_Lean_Syntax_getArgs(v___x_2663_);
lean_dec(v___x_2663_);
v_sz_2665_ = lean_array_size(v___x_2664_);
v___x_2666_ = ((size_t)0ULL);
v___x_2667_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0(v_sz_2665_, v___x_2666_, v___x_2664_);
if (lean_obj_tag(v___x_2667_) == 0)
{
lean_object* v___x_2668_; lean_object* v___x_2669_; 
lean_dec(v_n_2657_);
lean_dec(v___x_2649_);
v___x_2668_ = lean_box(1);
v___x_2669_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2669_, 0, v___x_2668_);
lean_ctor_set(v___x_2669_, 1, v_a_2643_);
return v___x_2669_;
}
else
{
lean_object* v_val_2670_; lean_object* v___x_2671_; lean_object* v_tk_2672_; lean_object* v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v_m_2676_; lean_object* v_cs_2677_; lean_object* v___x_2678_; 
v_val_2670_ = lean_ctor_get(v___x_2667_, 0);
lean_inc(v_val_2670_);
lean_dec_ref_known(v___x_2667_, 1);
v___x_2671_ = lean_unsigned_to_nat(3u);
v_tk_2672_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2671_);
v___x_2673_ = lean_unsigned_to_nat(4u);
v___x_2674_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2673_);
v___x_2675_ = lean_unsigned_to_nat(6u);
v_m_2676_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2675_);
lean_dec(v___x_2649_);
v_cs_2677_ = l_Lean_Syntax_getArgs(v___x_2674_);
lean_dec(v___x_2674_);
v___x_2678_ = lp_proofwidgets_ProofWidgets_Jsx_transformTag(v_tk_2672_, v_n_2657_, v_m_2676_, v_val_2670_, v_cs_2677_, v_a_2642_, v_a_2643_);
lean_dec_ref(v_cs_2677_);
lean_dec(v_m_2676_);
lean_dec(v_tk_2672_);
if (lean_obj_tag(v___x_2678_) == 0)
{
lean_object* v_a_2679_; lean_object* v_a_2680_; lean_object* v___x_2682_; uint8_t v_isShared_2683_; uint8_t v_isSharedCheck_2687_; 
v_a_2679_ = lean_ctor_get(v___x_2678_, 0);
v_a_2680_ = lean_ctor_get(v___x_2678_, 1);
v_isSharedCheck_2687_ = !lean_is_exclusive(v___x_2678_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2682_ = v___x_2678_;
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
else
{
lean_inc(v_a_2680_);
lean_inc(v_a_2679_);
lean_dec(v___x_2678_);
v___x_2682_ = lean_box(0);
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
v_resetjp_2681_:
{
lean_object* v___x_2685_; 
if (v_isShared_2683_ == 0)
{
v___x_2685_ = v___x_2682_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v_a_2679_);
lean_ctor_set(v_reuseFailAlloc_2686_, 1, v_a_2680_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
else
{
lean_object* v_a_2688_; lean_object* v_a_2689_; lean_object* v___x_2691_; uint8_t v_isShared_2692_; uint8_t v_isSharedCheck_2696_; 
v_a_2688_ = lean_ctor_get(v___x_2678_, 0);
v_a_2689_ = lean_ctor_get(v___x_2678_, 1);
v_isSharedCheck_2696_ = !lean_is_exclusive(v___x_2678_);
if (v_isSharedCheck_2696_ == 0)
{
v___x_2691_ = v___x_2678_;
v_isShared_2692_ = v_isSharedCheck_2696_;
goto v_resetjp_2690_;
}
else
{
lean_inc(v_a_2689_);
lean_inc(v_a_2688_);
lean_dec(v___x_2678_);
v___x_2691_ = lean_box(0);
v_isShared_2692_ = v_isSharedCheck_2696_;
goto v_resetjp_2690_;
}
v_resetjp_2690_:
{
lean_object* v___x_2694_; 
if (v_isShared_2692_ == 0)
{
v___x_2694_ = v___x_2691_;
goto v_reusejp_2693_;
}
else
{
lean_object* v_reuseFailAlloc_2695_; 
v_reuseFailAlloc_2695_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2695_, 0, v_a_2688_);
lean_ctor_set(v_reuseFailAlloc_2695_, 1, v_a_2689_);
v___x_2694_ = v_reuseFailAlloc_2695_;
goto v_reusejp_2693_;
}
v_reusejp_2693_:
{
return v___x_2694_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2697_; lean_object* v_n_2698_; lean_object* v___x_2699_; uint8_t v___x_2700_; 
v___x_2697_ = lean_unsigned_to_nat(1u);
v_n_2698_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2697_);
v___x_2699_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3));
lean_inc(v_n_2698_);
v___x_2700_ = l_Lean_Syntax_isOfKind(v_n_2698_, v___x_2699_);
if (v___x_2700_ == 0)
{
lean_object* v___x_2701_; lean_object* v___x_2702_; 
lean_dec(v_n_2698_);
lean_dec(v___x_2649_);
v___x_2701_ = lean_box(1);
v___x_2702_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2702_, 0, v___x_2701_);
lean_ctor_set(v___x_2702_, 1, v_a_2643_);
return v___x_2702_;
}
else
{
lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; size_t v_sz_2706_; size_t v___x_2707_; lean_object* v___x_2708_; 
v___x_2703_ = lean_unsigned_to_nat(2u);
v___x_2704_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2703_);
v___x_2705_ = l_Lean_Syntax_getArgs(v___x_2704_);
lean_dec(v___x_2704_);
v_sz_2706_ = lean_array_size(v___x_2705_);
v___x_2707_ = ((size_t)0ULL);
v___x_2708_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1_spec__0(v_sz_2706_, v___x_2707_, v___x_2705_);
if (lean_obj_tag(v___x_2708_) == 0)
{
lean_object* v___x_2709_; lean_object* v___x_2710_; 
lean_dec(v_n_2698_);
lean_dec(v___x_2649_);
v___x_2709_ = lean_box(1);
v___x_2710_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2710_, 0, v___x_2709_);
lean_ctor_set(v___x_2710_, 1, v_a_2643_);
return v___x_2710_;
}
else
{
lean_object* v_val_2711_; lean_object* v___x_2712_; lean_object* v_tk_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; 
v_val_2711_ = lean_ctor_get(v___x_2708_, 0);
lean_inc(v_val_2711_);
lean_dec_ref_known(v___x_2708_, 1);
v___x_2712_ = lean_unsigned_to_nat(3u);
v_tk_2713_ = l_Lean_Syntax_getArg(v___x_2649_, v___x_2712_);
lean_dec(v___x_2649_);
v___x_2714_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__1));
lean_inc(v_n_2698_);
v___x_2715_ = lp_proofwidgets_ProofWidgets_Jsx_transformTag(v_tk_2713_, v_n_2698_, v_n_2698_, v_val_2711_, v___x_2714_, v_a_2642_, v_a_2643_);
lean_dec(v_n_2698_);
lean_dec(v_tk_2713_);
if (lean_obj_tag(v___x_2715_) == 0)
{
lean_object* v_a_2716_; lean_object* v_a_2717_; lean_object* v___x_2719_; uint8_t v_isShared_2720_; uint8_t v_isSharedCheck_2724_; 
v_a_2716_ = lean_ctor_get(v___x_2715_, 0);
v_a_2717_ = lean_ctor_get(v___x_2715_, 1);
v_isSharedCheck_2724_ = !lean_is_exclusive(v___x_2715_);
if (v_isSharedCheck_2724_ == 0)
{
v___x_2719_ = v___x_2715_;
v_isShared_2720_ = v_isSharedCheck_2724_;
goto v_resetjp_2718_;
}
else
{
lean_inc(v_a_2717_);
lean_inc(v_a_2716_);
lean_dec(v___x_2715_);
v___x_2719_ = lean_box(0);
v_isShared_2720_ = v_isSharedCheck_2724_;
goto v_resetjp_2718_;
}
v_resetjp_2718_:
{
lean_object* v___x_2722_; 
if (v_isShared_2720_ == 0)
{
v___x_2722_ = v___x_2719_;
goto v_reusejp_2721_;
}
else
{
lean_object* v_reuseFailAlloc_2723_; 
v_reuseFailAlloc_2723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2723_, 0, v_a_2716_);
lean_ctor_set(v_reuseFailAlloc_2723_, 1, v_a_2717_);
v___x_2722_ = v_reuseFailAlloc_2723_;
goto v_reusejp_2721_;
}
v_reusejp_2721_:
{
return v___x_2722_;
}
}
}
else
{
lean_object* v_a_2725_; lean_object* v_a_2726_; lean_object* v___x_2728_; uint8_t v_isShared_2729_; uint8_t v_isSharedCheck_2733_; 
v_a_2725_ = lean_ctor_get(v___x_2715_, 0);
v_a_2726_ = lean_ctor_get(v___x_2715_, 1);
v_isSharedCheck_2733_ = !lean_is_exclusive(v___x_2715_);
if (v_isSharedCheck_2733_ == 0)
{
v___x_2728_ = v___x_2715_;
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
else
{
lean_inc(v_a_2726_);
lean_inc(v_a_2725_);
lean_dec(v___x_2715_);
v___x_2728_ = lean_box(0);
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
v_resetjp_2727_:
{
lean_object* v___x_2731_; 
if (v_isShared_2729_ == 0)
{
v___x_2731_ = v___x_2728_;
goto v_reusejp_2730_;
}
else
{
lean_object* v_reuseFailAlloc_2732_; 
v_reuseFailAlloc_2732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2732_, 0, v_a_2725_);
lean_ctor_set(v_reuseFailAlloc_2732_, 1, v_a_2726_);
v___x_2731_ = v_reuseFailAlloc_2732_;
goto v_reusejp_2730_;
}
v_reusejp_2730_:
{
return v___x_2731_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1___boxed(lean_object* v_x_2734_, lean_object* v_a_2735_, lean_object* v_a_2736_){
_start:
{
lean_object* v_res_2737_; 
v_res_2737_ = lp_proofwidgets_ProofWidgets_Jsx___aux__ProofWidgets__Data__Html______macroRules__ProofWidgets__Jsx__term____1(v_x_2734_, v_a_2735_, v_a_2736_);
lean_dec_ref(v_a_2735_);
return v_res_2737_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(lean_object* v___y_2738_){
_start:
{
lean_object* v_subExpr_2740_; lean_object* v_expr_2741_; lean_object* v___x_2742_; 
v_subExpr_2740_ = lean_ctor_get(v___y_2738_, 3);
v_expr_2741_ = lean_ctor_get(v_subExpr_2740_, 0);
lean_inc_ref(v_expr_2741_);
v___x_2742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2742_, 0, v_expr_2741_);
return v___x_2742_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg___boxed(lean_object* v___y_2743_, lean_object* v___y_2744_){
_start:
{
lean_object* v_res_2745_; 
v_res_2745_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_2743_);
lean_dec_ref(v___y_2743_);
return v_res_2745_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0(lean_object* v___y_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_){
_start:
{
lean_object* v___x_2753_; 
v___x_2753_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_2746_);
return v___x_2753_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___boxed(lean_object* v___y_2754_, lean_object* v___y_2755_, lean_object* v___y_2756_, lean_object* v___y_2757_, lean_object* v___y_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_){
_start:
{
lean_object* v_res_2761_; 
v_res_2761_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0(v___y_2754_, v___y_2755_, v___y_2756_, v___y_2757_, v___y_2758_, v___y_2759_);
lean_dec(v___y_2759_);
lean_dec_ref(v___y_2758_);
lean_dec(v___y_2757_);
lean_dec_ref(v___y_2756_);
lean_dec(v___y_2755_);
lean_dec_ref(v___y_2754_);
return v_res_2761_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg(lean_object* v_s_2762_, lean_object* v_a_2763_, uint8_t v_b_2764_){
_start:
{
lean_object* v_str_2765_; lean_object* v_startInclusive_2766_; lean_object* v_endExclusive_2767_; lean_object* v___x_2768_; uint8_t v___x_2769_; 
v_str_2765_ = lean_ctor_get(v_s_2762_, 0);
v_startInclusive_2766_ = lean_ctor_get(v_s_2762_, 1);
v_endExclusive_2767_ = lean_ctor_get(v_s_2762_, 2);
v___x_2768_ = lean_nat_sub(v_endExclusive_2767_, v_startInclusive_2766_);
v___x_2769_ = lean_nat_dec_eq(v_a_2763_, v___x_2768_);
lean_dec(v___x_2768_);
if (v___x_2769_ == 0)
{
lean_object* v___x_2770_; uint32_t v___x_2771_; lean_object* v___x_2772_; uint8_t v___x_2773_; 
v___x_2770_ = lean_nat_add(v_startInclusive_2766_, v_a_2763_);
lean_dec(v_a_2763_);
v___x_2771_ = lean_string_utf8_get_fast(v_str_2765_, v___x_2770_);
v___x_2772_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1, &lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1_once, _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText___lam__2___closed__1);
v___x_2773_ = lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_jsxText_spec__0(v___x_2771_, v___x_2772_);
if (v___x_2773_ == 0)
{
lean_object* v___x_2774_; lean_object* v___x_2775_; 
v___x_2774_ = lean_string_utf8_next_fast(v_str_2765_, v___x_2770_);
lean_dec(v___x_2770_);
v___x_2775_ = lean_nat_sub(v___x_2774_, v_startInclusive_2766_);
v_a_2763_ = v___x_2775_;
v_b_2764_ = v___x_2773_;
goto _start;
}
else
{
lean_dec(v___x_2770_);
return v___x_2773_;
}
}
else
{
lean_dec(v_a_2763_);
return v_b_2764_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg___boxed(lean_object* v_s_2777_, lean_object* v_a_2778_, lean_object* v_b_2779_){
_start:
{
uint8_t v_b_boxed_2780_; uint8_t v_res_2781_; lean_object* v_r_2782_; 
v_b_boxed_2780_ = lean_unbox(v_b_2779_);
v_res_2781_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg(v_s_2777_, v_a_2778_, v_b_boxed_2780_);
lean_dec_ref(v_s_2777_);
v_r_2782_ = lean_box(v_res_2781_);
return v_r_2782_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1(lean_object* v_s_2783_){
_start:
{
lean_object* v_searcher_2784_; uint8_t v___x_2785_; uint8_t v___x_2786_; 
v_searcher_2784_ = lean_unsigned_to_nat(0u);
v___x_2785_ = 0;
v___x_2786_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg(v_s_2783_, v_searcher_2784_, v___x_2785_);
return v___x_2786_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1___boxed(lean_object* v_s_2787_){
_start:
{
uint8_t v_res_2788_; lean_object* v_r_2789_; 
v_res_2788_ = lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1(v_s_2787_);
lean_dec_ref(v_s_2787_);
v_r_2789_ = lean_box(v_res_2788_);
return v_r_2789_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText(lean_object* v_a_2790_, lean_object* v_a_2791_, lean_object* v_a_2792_, lean_object* v_a_2793_, lean_object* v_a_2794_, lean_object* v_a_2795_){
_start:
{
lean_object* v___x_2797_; lean_object* v_a_2798_; lean_object* v___x_2799_; uint8_t v___x_2800_; 
v___x_2797_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v_a_2790_);
v_a_2798_ = lean_ctor_get(v___x_2797_, 0);
lean_inc(v_a_2798_);
lean_dec_ref(v___x_2797_);
v___x_2799_ = l_Lean_Expr_cleanupAnnotations(v_a_2798_);
v___x_2800_ = l_Lean_Expr_isApp(v___x_2799_);
if (v___x_2800_ == 0)
{
lean_object* v___x_2801_; 
lean_dec_ref(v___x_2799_);
v___x_2801_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2801_;
}
else
{
lean_object* v_arg_2802_; lean_object* v___x_2803_; lean_object* v___x_2804_; uint8_t v___x_2805_; 
v_arg_2802_ = lean_ctor_get(v___x_2799_, 1);
lean_inc_ref(v_arg_2802_);
v___x_2803_ = l_Lean_Expr_appFnCleanup___redArg(v___x_2799_);
v___x_2804_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8));
v___x_2805_ = l_Lean_Expr_isConstOf(v___x_2803_, v___x_2804_);
lean_dec_ref(v___x_2803_);
if (v___x_2805_ == 0)
{
lean_object* v___x_2806_; 
lean_dec_ref(v_arg_2802_);
v___x_2806_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2806_;
}
else
{
if (lean_obj_tag(v_arg_2802_) == 9)
{
lean_object* v_a_2807_; 
v_a_2807_ = lean_ctor_get(v_arg_2802_, 0);
lean_inc_ref(v_a_2807_);
lean_dec_ref_known(v_arg_2802_, 1);
if (lean_obj_tag(v_a_2807_) == 1)
{
lean_object* v_val_2808_; lean_object* v___y_2810_; lean_object* v___y_2811_; lean_object* v___y_2812_; lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; uint8_t v___x_2824_; 
v_val_2808_ = lean_ctor_get(v_a_2807_, 0);
lean_inc_ref_n(v_val_2808_, 2);
lean_dec_ref_known(v_a_2807_, 1);
v___x_2821_ = lean_unsigned_to_nat(0u);
v___x_2822_ = lean_string_utf8_byte_size(v_val_2808_);
v___x_2823_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2823_, 0, v_val_2808_);
lean_ctor_set(v___x_2823_, 1, v___x_2821_);
lean_ctor_set(v___x_2823_, 2, v___x_2822_);
v___x_2824_ = lp_proofwidgets_String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1(v___x_2823_);
lean_dec_ref_known(v___x_2823_, 3);
if (v___x_2824_ == 0)
{
v___y_2810_ = v_a_2790_;
v___y_2811_ = v_a_2791_;
v___y_2812_ = v_a_2792_;
goto v___jp_2809_;
}
else
{
lean_object* v___x_2825_; 
v___x_2825_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_2825_) == 0)
{
lean_dec_ref_known(v___x_2825_, 1);
v___y_2810_ = v_a_2790_;
v___y_2811_ = v_a_2791_;
v___y_2812_ = v_a_2792_;
goto v___jp_2809_;
}
else
{
lean_object* v_a_2826_; lean_object* v___x_2828_; uint8_t v_isShared_2829_; uint8_t v_isSharedCheck_2833_; 
lean_dec_ref(v_val_2808_);
v_a_2826_ = lean_ctor_get(v___x_2825_, 0);
v_isSharedCheck_2833_ = !lean_is_exclusive(v___x_2825_);
if (v_isSharedCheck_2833_ == 0)
{
v___x_2828_ = v___x_2825_;
v_isShared_2829_ = v_isSharedCheck_2833_;
goto v_resetjp_2827_;
}
else
{
lean_inc(v_a_2826_);
lean_dec(v___x_2825_);
v___x_2828_ = lean_box(0);
v_isShared_2829_ = v_isSharedCheck_2833_;
goto v_resetjp_2827_;
}
v_resetjp_2827_:
{
lean_object* v___x_2831_; 
if (v_isShared_2829_ == 0)
{
v___x_2831_ = v___x_2828_;
goto v_reusejp_2830_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v_a_2826_);
v___x_2831_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2830_;
}
v_reusejp_2830_:
{
return v___x_2831_;
}
}
}
}
v___jp_2809_:
{
lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; lean_object* v___x_2819_; lean_object* v___x_2820_; 
v___x_2813_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_jsxText___closed__4));
v___x_2814_ = l_Lean_mkAtom(v_val_2808_);
v___x_2815_ = lean_unsigned_to_nat(1u);
v___x_2816_ = lean_mk_empty_array_with_capacity(v___x_2815_);
v___x_2817_ = lean_array_push(v___x_2816_, v___x_2814_);
v___x_2818_ = lean_box(2);
v___x_2819_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2819_, 0, v___x_2818_);
lean_ctor_set(v___x_2819_, 1, v___x_2813_);
lean_ctor_set(v___x_2819_, 2, v___x_2817_);
v___x_2820_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(v___x_2819_, v___y_2810_, v___y_2811_, v___y_2812_);
return v___x_2820_;
}
}
else
{
lean_object* v___x_2834_; 
lean_dec_ref(v_a_2807_);
v___x_2834_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2834_;
}
}
else
{
lean_object* v___x_2835_; 
lean_dec_ref(v_arg_2802_);
v___x_2835_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2835_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText___boxed(lean_object* v_a_2836_, lean_object* v_a_2837_, lean_object* v_a_2838_, lean_object* v_a_2839_, lean_object* v_a_2840_, lean_object* v_a_2841_, lean_object* v_a_2842_){
_start:
{
lean_object* v_res_2843_; 
v_res_2843_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText(v_a_2836_, v_a_2837_, v_a_2838_, v_a_2839_, v_a_2840_, v_a_2841_);
lean_dec(v_a_2841_);
lean_dec_ref(v_a_2840_);
lean_dec(v_a_2839_);
lean_dec_ref(v_a_2838_);
lean_dec(v_a_2837_);
lean_dec_ref(v_a_2836_);
return v_res_2843_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1(lean_object* v_s_2844_, lean_object* v_inst_2845_, lean_object* v_R_2846_, lean_object* v_a_2847_, uint8_t v_b_2848_, lean_object* v_c_2849_){
_start:
{
uint8_t v___x_2850_; 
v___x_2850_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___redArg(v_s_2844_, v_a_2847_, v_b_2848_);
return v___x_2850_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1___boxed(lean_object* v_s_2851_, lean_object* v_inst_2852_, lean_object* v_R_2853_, lean_object* v_a_2854_, lean_object* v_b_2855_, lean_object* v_c_2856_){
_start:
{
uint8_t v_b_boxed_2857_; uint8_t v_res_2858_; lean_object* v_r_2859_; 
v_b_boxed_2857_ = lean_unbox(v_b_2855_);
v_res_2858_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00ProofWidgets_Jsx_delabHtmlText_spec__1_spec__1(v_s_2851_, v_inst_2852_, v_R_2853_, v_a_2854_, v_b_boxed_2857_, v_c_2856_);
lean_dec_ref(v_s_2851_);
v_r_2859_ = lean_box(v_res_2858_);
return v_r_2859_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0(lean_object* v___x_2860_, lean_object* v_a_2861_, lean_object* v___y_2862_, lean_object* v___y_2863_, lean_object* v___y_2864_, lean_object* v___y_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_){
_start:
{
lean_object* v___x_2869_; 
v___x_2869_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_2862_);
if (lean_obj_tag(v___x_2869_) == 0)
{
lean_object* v_a_2870_; lean_object* v___y_2872_; lean_object* v___y_2873_; lean_object* v___y_2874_; lean_object* v___y_2875_; lean_object* v___y_2876_; lean_object* v___y_2877_; 
v_a_2870_ = lean_ctor_get(v___x_2869_, 0);
lean_inc(v_a_2870_);
lean_dec_ref_known(v___x_2869_, 1);
if (lean_obj_tag(v_a_2870_) == 5)
{
lean_object* v_fn_2905_; 
v_fn_2905_ = lean_ctor_get(v_a_2870_, 0);
if (lean_obj_tag(v_fn_2905_) == 4)
{
lean_object* v_declName_2906_; 
v_declName_2906_ = lean_ctor_get(v_fn_2905_, 0);
lean_inc(v_declName_2906_);
if (lean_obj_tag(v_declName_2906_) == 1)
{
lean_object* v_pre_2907_; 
v_pre_2907_ = lean_ctor_get(v_declName_2906_, 0);
lean_inc(v_pre_2907_);
if (lean_obj_tag(v_pre_2907_) == 1)
{
lean_object* v_pre_2908_; 
v_pre_2908_ = lean_ctor_get(v_pre_2907_, 0);
lean_inc(v_pre_2908_);
if (lean_obj_tag(v_pre_2908_) == 1)
{
lean_object* v_pre_2909_; 
v_pre_2909_ = lean_ctor_get(v_pre_2908_, 0);
if (lean_obj_tag(v_pre_2909_) == 0)
{
lean_object* v_arg_2910_; lean_object* v_str_2911_; lean_object* v_str_2912_; lean_object* v_str_2913_; lean_object* v___x_2914_; uint8_t v___x_2915_; 
v_arg_2910_ = lean_ctor_get(v_a_2870_, 1);
lean_inc_ref(v_arg_2910_);
lean_dec_ref_known(v_a_2870_, 2);
v_str_2911_ = lean_ctor_get(v_declName_2906_, 1);
lean_inc_ref(v_str_2911_);
lean_dec_ref_known(v_declName_2906_, 2);
v_str_2912_ = lean_ctor_get(v_pre_2907_, 1);
lean_inc_ref(v_str_2912_);
lean_dec_ref_known(v_pre_2907_, 2);
v_str_2913_ = lean_ctor_get(v_pre_2908_, 1);
lean_inc_ref(v_str_2913_);
lean_dec_ref_known(v_pre_2908_, 2);
v___x_2914_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0));
v___x_2915_ = lean_string_dec_eq(v_str_2913_, v___x_2914_);
lean_dec_ref(v_str_2913_);
if (v___x_2915_ == 0)
{
lean_dec_ref(v_str_2912_);
lean_dec_ref(v_str_2911_);
lean_dec_ref(v_arg_2910_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
else
{
lean_object* v___x_2916_; uint8_t v___x_2917_; 
v___x_2916_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__35));
v___x_2917_ = lean_string_dec_eq(v_str_2912_, v___x_2916_);
lean_dec_ref(v_str_2912_);
if (v___x_2917_ == 0)
{
lean_dec_ref(v_str_2911_);
lean_dec_ref(v_arg_2910_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
else
{
lean_object* v___x_2918_; uint8_t v___x_2919_; 
v___x_2918_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__4));
v___x_2919_ = lean_string_dec_eq(v_str_2911_, v___x_2918_);
lean_dec_ref(v_str_2911_);
if (v___x_2919_ == 0)
{
lean_dec_ref(v_arg_2910_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
else
{
if (lean_obj_tag(v_arg_2910_) == 9)
{
lean_object* v_a_2920_; 
v_a_2920_ = lean_ctor_get(v_arg_2910_, 0);
lean_inc_ref(v_a_2920_);
lean_dec_ref_known(v_arg_2910_, 1);
if (lean_obj_tag(v_a_2920_) == 1)
{
lean_object* v_val_2921_; lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; 
v_val_2921_ = lean_ctor_get(v_a_2920_, 0);
lean_inc_ref(v_val_2921_);
lean_dec_ref_known(v_a_2920_, 1);
v___x_2922_ = lean_box(2);
v___x_2923_ = l_Lean_Syntax_mkStrLit(v_val_2921_, v___x_2922_);
v___x_2924_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(v___x_2923_, v___y_2862_, v___y_2863_, v___y_2864_);
if (lean_obj_tag(v___x_2924_) == 0)
{
lean_object* v_a_2925_; lean_object* v___x_2927_; uint8_t v_isShared_2928_; uint8_t v_isSharedCheck_2944_; 
v_a_2925_ = lean_ctor_get(v___x_2924_, 0);
v_isSharedCheck_2944_ = !lean_is_exclusive(v___x_2924_);
if (v_isSharedCheck_2944_ == 0)
{
v___x_2927_ = v___x_2924_;
v_isShared_2928_ = v_isSharedCheck_2944_;
goto v_resetjp_2926_;
}
else
{
lean_inc(v_a_2925_);
lean_dec(v___x_2924_);
v___x_2927_ = lean_box(0);
v_isShared_2928_ = v_isSharedCheck_2944_;
goto v_resetjp_2926_;
}
v_resetjp_2926_:
{
lean_object* v_ref_2929_; uint8_t v___x_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; lean_object* v___x_2935_; lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2942_; 
v_ref_2929_ = lean_ctor_get(v___y_2866_, 5);
v___x_2930_ = 0;
v___x_2931_ = l_Lean_SourceInfo_fromRef(v_ref_2929_, v___x_2930_);
v___x_2932_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1));
v___x_2933_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0));
lean_inc_ref(v___x_2860_);
v___x_2934_ = l_Lean_Name_mkStr3(v___x_2860_, v___x_2932_, v___x_2933_);
v___x_2935_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5));
lean_inc_n(v___x_2931_, 2);
v___x_2936_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2936_, 0, v___x_2931_);
lean_ctor_set(v___x_2936_, 1, v___x_2935_);
v___x_2937_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__2));
v___x_2938_ = l_Lean_Name_mkStr3(v___x_2860_, v___x_2932_, v___x_2937_);
v___x_2939_ = l_Lean_Syntax_node1(v___x_2931_, v___x_2938_, v_a_2925_);
v___x_2940_ = l_Lean_Syntax_node3(v___x_2931_, v___x_2934_, v_a_2861_, v___x_2936_, v___x_2939_);
if (v_isShared_2928_ == 0)
{
lean_ctor_set(v___x_2927_, 0, v___x_2940_);
v___x_2942_ = v___x_2927_;
goto v_reusejp_2941_;
}
else
{
lean_object* v_reuseFailAlloc_2943_; 
v_reuseFailAlloc_2943_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2943_, 0, v___x_2940_);
v___x_2942_ = v_reuseFailAlloc_2943_;
goto v_reusejp_2941_;
}
v_reusejp_2941_:
{
return v___x_2942_;
}
}
}
else
{
lean_dec(v_a_2861_);
lean_dec_ref(v___x_2860_);
return v___x_2924_;
}
}
else
{
lean_dec_ref(v_a_2920_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec_ref(v_arg_2910_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_2908_, 2);
lean_dec_ref_known(v_pre_2907_, 2);
lean_dec_ref_known(v_declName_2906_, 2);
lean_dec_ref_known(v_a_2870_, 2);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec(v_pre_2908_);
lean_dec_ref_known(v_pre_2907_, 2);
lean_dec_ref_known(v_declName_2906_, 2);
lean_dec_ref_known(v_a_2870_, 2);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec_ref_known(v_declName_2906_, 2);
lean_dec(v_pre_2907_);
lean_dec_ref_known(v_a_2870_, 2);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec(v_declName_2906_);
lean_dec_ref_known(v_a_2870_, 2);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec_ref_known(v_a_2870_, 2);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
}
else
{
lean_dec(v_a_2870_);
v___y_2872_ = v___y_2862_;
v___y_2873_ = v___y_2863_;
v___y_2874_ = v___y_2864_;
v___y_2875_ = v___y_2865_;
v___y_2876_ = v___y_2866_;
v___y_2877_ = v___y_2867_;
goto v___jp_2871_;
}
v___jp_2871_:
{
lean_object* v___x_2878_; 
v___x_2878_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_2872_, v___y_2873_, v___y_2874_, v___y_2875_, v___y_2876_, v___y_2877_);
if (lean_obj_tag(v___x_2878_) == 0)
{
lean_object* v_a_2879_; lean_object* v___x_2881_; uint8_t v_isShared_2882_; uint8_t v_isSharedCheck_2904_; 
v_a_2879_ = lean_ctor_get(v___x_2878_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2878_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2881_ = v___x_2878_;
v_isShared_2882_ = v_isSharedCheck_2904_;
goto v_resetjp_2880_;
}
else
{
lean_inc(v_a_2879_);
lean_dec(v___x_2878_);
v___x_2881_ = lean_box(0);
v_isShared_2882_ = v_isSharedCheck_2904_;
goto v_resetjp_2880_;
}
v_resetjp_2880_:
{
lean_object* v_ref_2883_; uint8_t v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2902_; 
v_ref_2883_ = lean_ctor_get(v___y_2876_, 5);
v___x_2884_ = 0;
v___x_2885_ = l_Lean_SourceInfo_fromRef(v_ref_2883_, v___x_2884_);
v___x_2886_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1));
v___x_2887_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__0));
lean_inc_ref(v___x_2860_);
v___x_2888_ = l_Lean_Name_mkStr3(v___x_2860_, v___x_2886_, v___x_2887_);
v___x_2889_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5));
lean_inc_n(v___x_2885_, 5);
v___x_2890_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2890_, 0, v___x_2885_);
lean_ctor_set(v___x_2890_, 1, v___x_2889_);
v___x_2891_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__0));
v___x_2892_ = l_Lean_Name_mkStr3(v___x_2860_, v___x_2886_, v___x_2891_);
v___x_2893_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_2894_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4));
v___x_2895_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2895_, 0, v___x_2885_);
lean_ctor_set(v___x_2895_, 1, v___x_2894_);
v___x_2896_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_2897_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2897_, 0, v___x_2885_);
lean_ctor_set(v___x_2897_, 1, v___x_2896_);
v___x_2898_ = l_Lean_Syntax_node3(v___x_2885_, v___x_2893_, v___x_2895_, v_a_2879_, v___x_2897_);
v___x_2899_ = l_Lean_Syntax_node1(v___x_2885_, v___x_2892_, v___x_2898_);
v___x_2900_ = l_Lean_Syntax_node3(v___x_2885_, v___x_2888_, v_a_2861_, v___x_2890_, v___x_2899_);
if (v_isShared_2882_ == 0)
{
lean_ctor_set(v___x_2881_, 0, v___x_2900_);
v___x_2902_ = v___x_2881_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v___x_2900_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
else
{
lean_dec(v_a_2861_);
lean_dec_ref(v___x_2860_);
return v___x_2878_;
}
}
}
else
{
lean_object* v_a_2945_; lean_object* v___x_2947_; uint8_t v_isShared_2948_; uint8_t v_isSharedCheck_2952_; 
lean_dec(v_a_2861_);
lean_dec_ref(v___x_2860_);
v_a_2945_ = lean_ctor_get(v___x_2869_, 0);
v_isSharedCheck_2952_ = !lean_is_exclusive(v___x_2869_);
if (v_isSharedCheck_2952_ == 0)
{
v___x_2947_ = v___x_2869_;
v_isShared_2948_ = v_isSharedCheck_2952_;
goto v_resetjp_2946_;
}
else
{
lean_inc(v_a_2945_);
lean_dec(v___x_2869_);
v___x_2947_ = lean_box(0);
v_isShared_2948_ = v_isSharedCheck_2952_;
goto v_resetjp_2946_;
}
v_resetjp_2946_:
{
lean_object* v___x_2950_; 
if (v_isShared_2948_ == 0)
{
v___x_2950_ = v___x_2947_;
goto v_reusejp_2949_;
}
else
{
lean_object* v_reuseFailAlloc_2951_; 
v_reuseFailAlloc_2951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2951_, 0, v_a_2945_);
v___x_2950_ = v_reuseFailAlloc_2951_;
goto v_reusejp_2949_;
}
v_reusejp_2949_:
{
return v___x_2950_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0___boxed(lean_object* v___x_2953_, lean_object* v_a_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_, lean_object* v___y_2958_, lean_object* v___y_2959_, lean_object* v___y_2960_, lean_object* v___y_2961_){
_start:
{
lean_object* v_res_2962_; 
v_res_2962_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0(v___x_2953_, v_a_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_);
lean_dec(v___y_2960_);
lean_dec_ref(v___y_2959_);
lean_dec(v___y_2958_);
lean_dec_ref(v___y_2957_);
lean_dec(v___y_2956_);
lean_dec_ref(v___y_2955_);
return v_res_2962_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg(lean_object* v___y_2963_){
_start:
{
lean_object* v_subExpr_2965_; lean_object* v_pos_2966_; lean_object* v___x_2967_; 
v_subExpr_2965_ = lean_ctor_get(v___y_2963_, 3);
v_pos_2966_ = lean_ctor_get(v_subExpr_2965_, 1);
lean_inc(v_pos_2966_);
v___x_2967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2967_, 0, v_pos_2966_);
return v___x_2967_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg___boxed(lean_object* v___y_2968_, lean_object* v___y_2969_){
_start:
{
lean_object* v_res_2970_; 
v_res_2970_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg(v___y_2968_);
lean_dec_ref(v___y_2968_);
return v_res_2970_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2971_; lean_object* v_dummy_2972_; 
v___x_2971_ = lean_box(0);
v_dummy_2972_ = l_Lean_Expr_sort___override(v___x_2971_);
return v_dummy_2972_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(lean_object* v_argIdx_2973_, lean_object* v_x_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_){
_start:
{
lean_object* v___x_2982_; lean_object* v_a_2983_; lean_object* v___x_2984_; lean_object* v_a_2985_; lean_object* v_optionsPerPos_2986_; lean_object* v_currNamespace_2987_; lean_object* v_openDecls_2988_; uint8_t v_inPattern_2989_; lean_object* v_depth_2990_; lean_object* v_lctxInitIndices_2991_; lean_object* v_nargs_2992_; lean_object* v___x_2993_; lean_object* v_dummy_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v_args_2998_; lean_object* v___x_2999_; lean_object* v_newPos_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; 
v___x_2982_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_2975_);
v_a_2983_ = lean_ctor_get(v___x_2982_, 0);
lean_inc(v_a_2983_);
lean_dec_ref(v___x_2982_);
v___x_2984_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg(v___y_2975_);
v_a_2985_ = lean_ctor_get(v___x_2984_, 0);
lean_inc(v_a_2985_);
lean_dec_ref(v___x_2984_);
v_optionsPerPos_2986_ = lean_ctor_get(v___y_2975_, 0);
v_currNamespace_2987_ = lean_ctor_get(v___y_2975_, 1);
v_openDecls_2988_ = lean_ctor_get(v___y_2975_, 2);
v_inPattern_2989_ = lean_ctor_get_uint8(v___y_2975_, sizeof(void*)*6);
v_depth_2990_ = lean_ctor_get(v___y_2975_, 4);
v_lctxInitIndices_2991_ = lean_ctor_get(v___y_2975_, 5);
v_nargs_2992_ = l_Lean_Expr_getAppNumArgs(v_a_2983_);
v___x_2993_ = l_Lean_instInhabitedExpr;
v_dummy_2994_ = lean_obj_once(&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0, &lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0_once, _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___closed__0);
lean_inc(v_nargs_2992_);
v___x_2995_ = lean_mk_array(v_nargs_2992_, v_dummy_2994_);
v___x_2996_ = lean_unsigned_to_nat(1u);
v___x_2997_ = lean_nat_sub(v_nargs_2992_, v___x_2996_);
lean_dec(v_nargs_2992_);
v_args_2998_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2983_, v___x_2995_, v___x_2997_);
v___x_2999_ = lean_array_get_size(v_args_2998_);
v_newPos_3000_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_2999_, v_argIdx_2973_, v_a_2985_);
lean_dec(v_a_2985_);
v___x_3001_ = lean_array_get(v___x_2993_, v_args_2998_, v_argIdx_2973_);
lean_dec_ref(v_args_2998_);
v___x_3002_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3002_, 0, v___x_3001_);
lean_ctor_set(v___x_3002_, 1, v_newPos_3000_);
lean_inc(v_lctxInitIndices_2991_);
lean_inc(v_depth_2990_);
lean_inc(v_openDecls_2988_);
lean_inc(v_currNamespace_2987_);
lean_inc(v_optionsPerPos_2986_);
v___x_3003_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_3003_, 0, v_optionsPerPos_2986_);
lean_ctor_set(v___x_3003_, 1, v_currNamespace_2987_);
lean_ctor_set(v___x_3003_, 2, v_openDecls_2988_);
lean_ctor_set(v___x_3003_, 3, v___x_3002_);
lean_ctor_set(v___x_3003_, 4, v_depth_2990_);
lean_ctor_set(v___x_3003_, 5, v_lctxInitIndices_2991_);
lean_ctor_set_uint8(v___x_3003_, sizeof(void*)*6, v_inPattern_2989_);
lean_inc(v___y_2980_);
lean_inc_ref(v___y_2979_);
lean_inc(v___y_2978_);
lean_inc_ref(v___y_2977_);
lean_inc(v___y_2976_);
v___x_3004_ = lean_apply_7(v_x_2974_, v___x_3003_, v___y_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_, lean_box(0));
return v___x_3004_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg___boxed(lean_object* v_argIdx_3005_, lean_object* v_x_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_){
_start:
{
lean_object* v_res_3014_; 
v_res_3014_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v_argIdx_3005_, v_x_3006_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec(v___y_3010_);
lean_dec_ref(v___y_3009_);
lean_dec(v___y_3008_);
lean_dec_ref(v___y_3007_);
lean_dec(v_argIdx_3005_);
return v_res_3014_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1(lean_object* v___x_3020_, lean_object* v___x_3021_, lean_object* v___x_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_){
_start:
{
lean_object* v___x_3030_; 
v___x_3030_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_3023_);
if (lean_obj_tag(v___x_3030_) == 0)
{
lean_object* v_a_3031_; lean_object* v___x_3032_; uint8_t v___x_3033_; 
v_a_3031_ = lean_ctor_get(v___x_3030_, 0);
lean_inc(v_a_3031_);
lean_dec_ref_known(v___x_3030_, 1);
v___x_3032_ = l_Lean_Expr_cleanupAnnotations(v_a_3031_);
v___x_3033_ = l_Lean_Expr_isApp(v___x_3032_);
if (v___x_3033_ == 0)
{
lean_object* v___x_3034_; 
lean_dec_ref(v___x_3032_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3034_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3034_;
}
else
{
lean_object* v___x_3035_; uint8_t v___x_3036_; 
v___x_3035_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3032_);
v___x_3036_ = l_Lean_Expr_isApp(v___x_3035_);
if (v___x_3036_ == 0)
{
lean_object* v___x_3037_; 
lean_dec_ref(v___x_3035_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3037_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3037_;
}
else
{
lean_object* v_arg_3038_; lean_object* v___x_3039_; uint8_t v___x_3040_; 
v_arg_3038_ = lean_ctor_get(v___x_3035_, 1);
lean_inc_ref(v_arg_3038_);
v___x_3039_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3035_);
v___x_3040_ = l_Lean_Expr_isApp(v___x_3039_);
if (v___x_3040_ == 0)
{
lean_object* v___x_3041_; 
lean_dec_ref(v___x_3039_);
lean_dec_ref(v_arg_3038_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3041_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3041_;
}
else
{
lean_object* v___x_3042_; uint8_t v___x_3043_; 
v___x_3042_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3039_);
v___x_3043_ = l_Lean_Expr_isApp(v___x_3042_);
if (v___x_3043_ == 0)
{
lean_object* v___x_3044_; 
lean_dec_ref(v___x_3042_);
lean_dec_ref(v_arg_3038_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3044_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3044_;
}
else
{
lean_object* v___x_3045_; lean_object* v___x_3046_; uint8_t v___x_3047_; 
v___x_3045_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3042_);
v___x_3046_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___closed__2));
v___x_3047_ = l_Lean_Expr_isConstOf(v___x_3045_, v___x_3046_);
lean_dec_ref(v___x_3045_);
if (v___x_3047_ == 0)
{
lean_object* v___x_3048_; 
lean_dec_ref(v_arg_3038_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3048_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3048_;
}
else
{
if (lean_obj_tag(v_arg_3038_) == 9)
{
lean_object* v_a_3049_; 
v_a_3049_ = lean_ctor_get(v_arg_3038_, 0);
lean_inc_ref(v_a_3049_);
lean_dec_ref_known(v_arg_3038_, 1);
if (lean_obj_tag(v_a_3049_) == 1)
{
lean_object* v_val_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3055_; 
v_val_3050_ = lean_ctor_get(v_a_3049_, 0);
lean_inc_ref(v_val_3050_);
lean_dec_ref_known(v_a_3049_, 1);
v___x_3051_ = lean_unsigned_to_nat(2u);
v___x_3052_ = l_Lean_Name_str___override(v___x_3020_, v_val_3050_);
v___x_3053_ = l_Lean_mkIdent(v___x_3052_);
v___x_3054_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___boxed), 9, 2);
lean_closure_set(v___x_3054_, 0, v___x_3021_);
lean_closure_set(v___x_3054_, 1, v___x_3053_);
v___x_3055_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3051_, v___x_3054_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_, v___y_3027_, v___y_3028_);
if (lean_obj_tag(v___x_3055_) == 0)
{
lean_object* v_a_3056_; lean_object* v___f_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; 
v_a_3056_ = lean_ctor_get(v___x_3055_, 0);
lean_inc(v_a_3056_);
lean_dec_ref_known(v___x_3055_, 1);
v___f_3057_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__0___boxed), 9, 2);
lean_closure_set(v___f_3057_, 0, v___x_3022_);
lean_closure_set(v___f_3057_, 1, v_a_3056_);
v___x_3058_ = lean_unsigned_to_nat(3u);
v___x_3059_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3058_, v___f_3057_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_, v___y_3027_, v___y_3028_);
return v___x_3059_;
}
else
{
lean_dec_ref(v___x_3022_);
return v___x_3055_;
}
}
else
{
lean_object* v___x_3060_; 
lean_dec_ref(v_a_3049_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3060_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3060_;
}
}
else
{
lean_object* v___x_3061_; 
lean_dec_ref(v_arg_3038_);
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v___x_3061_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3061_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3062_; lean_object* v___x_3064_; uint8_t v_isShared_3065_; uint8_t v_isSharedCheck_3069_; 
lean_dec_ref(v___x_3022_);
lean_dec(v___x_3021_);
lean_dec(v___x_3020_);
v_a_3062_ = lean_ctor_get(v___x_3030_, 0);
v_isSharedCheck_3069_ = !lean_is_exclusive(v___x_3030_);
if (v_isSharedCheck_3069_ == 0)
{
v___x_3064_ = v___x_3030_;
v_isShared_3065_ = v_isSharedCheck_3069_;
goto v_resetjp_3063_;
}
else
{
lean_inc(v_a_3062_);
lean_dec(v___x_3030_);
v___x_3064_ = lean_box(0);
v_isShared_3065_ = v_isSharedCheck_3069_;
goto v_resetjp_3063_;
}
v_resetjp_3063_:
{
lean_object* v___x_3067_; 
if (v_isShared_3065_ == 0)
{
v___x_3067_ = v___x_3064_;
goto v_reusejp_3066_;
}
else
{
lean_object* v_reuseFailAlloc_3068_; 
v_reuseFailAlloc_3068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3068_, 0, v_a_3062_);
v___x_3067_ = v_reuseFailAlloc_3068_;
goto v_reusejp_3066_;
}
v_reusejp_3066_:
{
return v___x_3067_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1___boxed(lean_object* v___x_3070_, lean_object* v___x_3071_, lean_object* v___x_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_){
_start:
{
lean_object* v_res_3080_; 
v_res_3080_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__1(v___x_3070_, v___x_3071_, v___x_3072_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_);
lean_dec(v___y_3078_);
lean_dec_ref(v___y_3077_);
lean_dec(v___y_3076_);
lean_dec_ref(v___y_3075_);
lean_dec(v___y_3074_);
lean_dec_ref(v___y_3073_);
return v_res_3080_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2(lean_object* v___x_3081_, lean_object* v___x_3082_, lean_object* v___x_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_){
_start:
{
lean_object* v___x_3091_; 
v___x_3091_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(v___x_3081_, v___y_3084_, v___y_3085_, v___y_3086_, v___y_3087_, v___y_3088_, v___y_3089_);
if (lean_obj_tag(v___x_3091_) == 0)
{
lean_dec_ref(v___x_3082_);
return v___x_3091_;
}
else
{
lean_object* v_a_3092_; uint8_t v___y_3094_; uint8_t v___x_3126_; 
v_a_3092_ = lean_ctor_get(v___x_3091_, 0);
lean_inc(v_a_3092_);
v___x_3126_ = l_Lean_Exception_isInterrupt(v_a_3092_);
if (v___x_3126_ == 0)
{
uint8_t v___x_3127_; 
v___x_3127_ = l_Lean_Exception_isRuntime(v_a_3092_);
v___y_3094_ = v___x_3127_;
goto v___jp_3093_;
}
else
{
lean_dec(v_a_3092_);
v___y_3094_ = v___x_3126_;
goto v___jp_3093_;
}
v___jp_3093_:
{
if (v___y_3094_ == 0)
{
lean_object* v___x_3095_; 
lean_dec_ref_known(v___x_3091_, 1);
v___x_3095_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_3084_, v___y_3085_, v___y_3086_, v___y_3087_, v___y_3088_, v___y_3089_);
if (lean_obj_tag(v___x_3095_) == 0)
{
lean_object* v_a_3096_; lean_object* v___x_3098_; uint8_t v_isShared_3099_; uint8_t v_isSharedCheck_3117_; 
v_a_3096_ = lean_ctor_get(v___x_3095_, 0);
v_isSharedCheck_3117_ = !lean_is_exclusive(v___x_3095_);
if (v_isSharedCheck_3117_ == 0)
{
v___x_3098_ = v___x_3095_;
v_isShared_3099_ = v_isSharedCheck_3117_;
goto v_resetjp_3097_;
}
else
{
lean_inc(v_a_3096_);
lean_dec(v___x_3095_);
v___x_3098_ = lean_box(0);
v_isShared_3099_ = v_isSharedCheck_3117_;
goto v_resetjp_3097_;
}
v_resetjp_3097_:
{
lean_object* v_ref_3100_; lean_object* v___x_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; lean_object* v___x_3113_; lean_object* v___x_3115_; 
v_ref_3100_ = lean_ctor_get(v___y_3088_, 5);
v___x_3101_ = l_Lean_SourceInfo_fromRef(v_ref_3100_, v___y_3094_);
v___x_3102_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal___00__closed__1));
v___x_3103_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__0));
v___x_3104_ = l_Lean_Name_mkStr3(v___x_3082_, v___x_3102_, v___x_3103_);
v___x_3105_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3106_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3101_, 3);
v___x_3107_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3107_, 0, v___x_3101_);
lean_ctor_set(v___x_3107_, 1, v___x_3106_);
v___x_3108_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3109_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3109_, 0, v___x_3101_);
lean_ctor_set(v___x_3109_, 1, v___x_3108_);
v___x_3110_ = l_Lean_Syntax_node3(v___x_3101_, v___x_3105_, v___x_3107_, v_a_3096_, v___x_3109_);
v___x_3111_ = l_Lean_Syntax_node1(v___x_3101_, v___x_3104_, v___x_3110_);
v___x_3112_ = lean_mk_empty_array_with_capacity(v___x_3083_);
v___x_3113_ = lean_array_push(v___x_3112_, v___x_3111_);
if (v_isShared_3099_ == 0)
{
lean_ctor_set(v___x_3098_, 0, v___x_3113_);
v___x_3115_ = v___x_3098_;
goto v_reusejp_3114_;
}
else
{
lean_object* v_reuseFailAlloc_3116_; 
v_reuseFailAlloc_3116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3116_, 0, v___x_3113_);
v___x_3115_ = v_reuseFailAlloc_3116_;
goto v_reusejp_3114_;
}
v_reusejp_3114_:
{
return v___x_3115_;
}
}
}
else
{
lean_object* v_a_3118_; lean_object* v___x_3120_; uint8_t v_isShared_3121_; uint8_t v_isSharedCheck_3125_; 
lean_dec_ref(v___x_3082_);
v_a_3118_ = lean_ctor_get(v___x_3095_, 0);
v_isSharedCheck_3125_ = !lean_is_exclusive(v___x_3095_);
if (v_isSharedCheck_3125_ == 0)
{
v___x_3120_ = v___x_3095_;
v_isShared_3121_ = v_isSharedCheck_3125_;
goto v_resetjp_3119_;
}
else
{
lean_inc(v_a_3118_);
lean_dec(v___x_3095_);
v___x_3120_ = lean_box(0);
v_isShared_3121_ = v_isSharedCheck_3125_;
goto v_resetjp_3119_;
}
v_resetjp_3119_:
{
lean_object* v___x_3123_; 
if (v_isShared_3121_ == 0)
{
v___x_3123_ = v___x_3120_;
goto v_reusejp_3122_;
}
else
{
lean_object* v_reuseFailAlloc_3124_; 
v_reuseFailAlloc_3124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3124_, 0, v_a_3118_);
v___x_3123_ = v_reuseFailAlloc_3124_;
goto v_reusejp_3122_;
}
v_reusejp_3122_:
{
return v___x_3123_;
}
}
}
}
else
{
lean_dec_ref(v___x_3082_);
return v___x_3091_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2___boxed(lean_object* v___x_3128_, lean_object* v___x_3129_, lean_object* v___x_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
lean_object* v_res_3138_; 
v_res_3138_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___lam__2(v___x_3128_, v___x_3129_, v___x_3130_, v___y_3131_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_);
lean_dec(v___y_3136_);
lean_dec_ref(v___y_3135_);
lean_dec(v___y_3134_);
lean_dec_ref(v___y_3133_);
lean_dec(v___y_3132_);
lean_dec_ref(v___y_3131_);
lean_dec(v___x_3130_);
return v_res_3138_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(size_t v_sz_3139_, size_t v_i_3140_, lean_object* v_bs_3141_){
_start:
{
uint8_t v___x_3142_; 
v___x_3142_ = lean_usize_dec_lt(v_i_3140_, v_sz_3139_);
if (v___x_3142_ == 0)
{
return v_bs_3141_;
}
else
{
lean_object* v_v_3143_; lean_object* v___x_3144_; lean_object* v_bs_x27_3145_; size_t v___x_3146_; size_t v___x_3147_; lean_object* v___x_3148_; 
v_v_3143_ = lean_array_uget(v_bs_3141_, v_i_3140_);
v___x_3144_ = lean_unsigned_to_nat(0u);
v_bs_x27_3145_ = lean_array_uset(v_bs_3141_, v_i_3140_, v___x_3144_);
v___x_3146_ = ((size_t)1ULL);
v___x_3147_ = lean_usize_add(v_i_3140_, v___x_3146_);
v___x_3148_ = lean_array_uset(v_bs_x27_3145_, v_i_3140_, v_v_3143_);
v_i_3140_ = v___x_3147_;
v_bs_3141_ = v___x_3148_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2___boxed(lean_object* v_sz_3150_, lean_object* v_i_3151_, lean_object* v_bs_3152_){
_start:
{
size_t v_sz_boxed_3153_; size_t v_i_boxed_3154_; lean_object* v_res_3155_; 
v_sz_boxed_3153_ = lean_unbox_usize(v_sz_3150_);
lean_dec(v_sz_3150_);
v_i_boxed_3154_ = lean_unbox_usize(v_i_3151_);
lean_dec(v_i_3151_);
v_res_3155_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_boxed_3153_, v_i_boxed_3154_, v_bs_3152_);
return v_res_3155_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg(lean_object* v_child_3156_, lean_object* v_childIdx_3157_, lean_object* v_x_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_, lean_object* v___y_3164_){
_start:
{
lean_object* v_subExpr_3166_; lean_object* v_optionsPerPos_3167_; lean_object* v_currNamespace_3168_; lean_object* v_openDecls_3169_; uint8_t v_inPattern_3170_; lean_object* v_depth_3171_; lean_object* v_lctxInitIndices_3172_; lean_object* v_pos_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v___x_3177_; 
v_subExpr_3166_ = lean_ctor_get(v___y_3159_, 3);
v_optionsPerPos_3167_ = lean_ctor_get(v___y_3159_, 0);
v_currNamespace_3168_ = lean_ctor_get(v___y_3159_, 1);
v_openDecls_3169_ = lean_ctor_get(v___y_3159_, 2);
v_inPattern_3170_ = lean_ctor_get_uint8(v___y_3159_, sizeof(void*)*6);
v_depth_3171_ = lean_ctor_get(v___y_3159_, 4);
v_lctxInitIndices_3172_ = lean_ctor_get(v___y_3159_, 5);
v_pos_3173_ = lean_ctor_get(v_subExpr_3166_, 1);
v___x_3174_ = l_Lean_SubExpr_Pos_push(v_pos_3173_, v_childIdx_3157_);
v___x_3175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3175_, 0, v_child_3156_);
lean_ctor_set(v___x_3175_, 1, v___x_3174_);
lean_inc(v_lctxInitIndices_3172_);
lean_inc(v_depth_3171_);
lean_inc(v_openDecls_3169_);
lean_inc(v_currNamespace_3168_);
lean_inc(v_optionsPerPos_3167_);
v___x_3176_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_3176_, 0, v_optionsPerPos_3167_);
lean_ctor_set(v___x_3176_, 1, v_currNamespace_3168_);
lean_ctor_set(v___x_3176_, 2, v_openDecls_3169_);
lean_ctor_set(v___x_3176_, 3, v___x_3175_);
lean_ctor_set(v___x_3176_, 4, v_depth_3171_);
lean_ctor_set(v___x_3176_, 5, v_lctxInitIndices_3172_);
lean_ctor_set_uint8(v___x_3176_, sizeof(void*)*6, v_inPattern_3170_);
lean_inc(v___y_3164_);
lean_inc_ref(v___y_3163_);
lean_inc(v___y_3162_);
lean_inc_ref(v___y_3161_);
lean_inc(v___y_3160_);
v___x_3177_ = lean_apply_7(v_x_3158_, v___x_3176_, v___y_3160_, v___y_3161_, v___y_3162_, v___y_3163_, v___y_3164_, lean_box(0));
return v___x_3177_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg___boxed(lean_object* v_child_3178_, lean_object* v_childIdx_3179_, lean_object* v_x_3180_, lean_object* v___y_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_){
_start:
{
lean_object* v_res_3188_; 
v_res_3188_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg(v_child_3178_, v_childIdx_3179_, v_x_3180_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec(v___y_3184_);
lean_dec_ref(v___y_3183_);
lean_dec(v___y_3182_);
lean_dec_ref(v___y_3181_);
return v_res_3188_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg(lean_object* v_x_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_, lean_object* v___y_3194_, lean_object* v___y_3195_){
_start:
{
lean_object* v___x_3197_; lean_object* v_a_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; 
v___x_3197_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_3190_);
v_a_3198_ = lean_ctor_get(v___x_3197_, 0);
lean_inc(v_a_3198_);
lean_dec_ref(v___x_3197_);
v___x_3199_ = l_Lean_Expr_appArg_x21(v_a_3198_);
lean_dec(v_a_3198_);
v___x_3200_ = lean_unsigned_to_nat(1u);
v___x_3201_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg(v___x_3199_, v___x_3200_, v_x_3189_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_, v___y_3195_);
return v___x_3201_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg___boxed(lean_object* v_x_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_){
_start:
{
lean_object* v_res_3210_; 
v_res_3210_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg(v_x_3202_, v___y_3203_, v___y_3204_, v___y_3205_, v___y_3206_, v___y_3207_, v___y_3208_);
lean_dec(v___y_3208_);
lean_dec_ref(v___y_3207_);
lean_dec(v___y_3206_);
lean_dec_ref(v___y_3205_);
lean_dec(v___y_3204_);
lean_dec_ref(v___y_3203_);
return v_res_3210_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6(size_t v_sz_3211_, size_t v_i_3212_, lean_object* v_bs_3213_){
_start:
{
uint8_t v___x_3214_; 
v___x_3214_ = lean_usize_dec_lt(v_i_3212_, v_sz_3211_);
if (v___x_3214_ == 0)
{
return v_bs_3213_;
}
else
{
lean_object* v_v_3215_; lean_object* v_snd_3216_; lean_object* v___x_3217_; lean_object* v_bs_x27_3218_; size_t v___x_3219_; size_t v___x_3220_; lean_object* v___x_3221_; 
v_v_3215_ = lean_array_uget_borrowed(v_bs_3213_, v_i_3212_);
v_snd_3216_ = lean_ctor_get(v_v_3215_, 1);
lean_inc(v_snd_3216_);
v___x_3217_ = lean_unsigned_to_nat(0u);
v_bs_x27_3218_ = lean_array_uset(v_bs_3213_, v_i_3212_, v___x_3217_);
v___x_3219_ = ((size_t)1ULL);
v___x_3220_ = lean_usize_add(v_i_3212_, v___x_3219_);
v___x_3221_ = lean_array_uset(v_bs_x27_3218_, v_i_3212_, v_snd_3216_);
v_i_3212_ = v___x_3220_;
v_bs_3213_ = v___x_3221_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6___boxed(lean_object* v_sz_3223_, lean_object* v_i_3224_, lean_object* v_bs_3225_){
_start:
{
size_t v_sz_boxed_3226_; size_t v_i_boxed_3227_; lean_object* v_res_3228_; 
v_sz_boxed_3226_ = lean_unbox_usize(v_sz_3223_);
lean_dec(v_sz_3223_);
v_i_boxed_3227_ = lean_unbox_usize(v_i_3224_);
lean_dec(v_i_3224_);
v_res_3228_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6(v_sz_boxed_3226_, v_i_boxed_3227_, v_bs_3225_);
return v_res_3228_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5(size_t v_sz_3229_, size_t v_i_3230_, lean_object* v_bs_3231_){
_start:
{
uint8_t v___x_3232_; 
v___x_3232_ = lean_usize_dec_lt(v_i_3230_, v_sz_3229_);
if (v___x_3232_ == 0)
{
lean_object* v___x_3233_; 
v___x_3233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3233_, 0, v_bs_3231_);
return v___x_3233_;
}
else
{
lean_object* v_v_3234_; lean_object* v___x_3235_; uint8_t v___x_3236_; 
v_v_3234_ = lean_array_uget_borrowed(v_bs_3231_, v_i_3230_);
v___x_3235_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__1));
lean_inc(v_v_3234_);
v___x_3236_ = l_Lean_Syntax_isOfKind(v_v_3234_, v___x_3235_);
if (v___x_3236_ == 0)
{
lean_object* v___x_3237_; 
lean_dec_ref(v_bs_3231_);
v___x_3237_ = lean_box(0);
return v___x_3237_;
}
else
{
lean_object* v___x_3238_; lean_object* v___x_3239_; lean_object* v___x_3240_; uint8_t v___x_3241_; 
v___x_3238_ = lean_unsigned_to_nat(0u);
v___x_3239_ = l_Lean_Syntax_getArg(v_v_3234_, v___x_3238_);
v___x_3240_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__3));
lean_inc(v___x_3239_);
v___x_3241_ = l_Lean_Syntax_isOfKind(v___x_3239_, v___x_3240_);
if (v___x_3241_ == 0)
{
lean_object* v___x_3242_; 
lean_dec(v___x_3239_);
lean_dec_ref(v_bs_3231_);
v___x_3242_ = lean_box(0);
return v___x_3242_;
}
else
{
lean_object* v___x_3243_; lean_object* v___x_3244_; uint8_t v___x_3245_; 
v___x_3243_ = l_Lean_Syntax_getArg(v___x_3239_, v___x_3238_);
v___x_3244_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__3));
lean_inc(v___x_3243_);
v___x_3245_ = l_Lean_Syntax_isOfKind(v___x_3243_, v___x_3244_);
if (v___x_3245_ == 0)
{
lean_object* v___x_3246_; 
lean_dec(v___x_3243_);
lean_dec(v___x_3239_);
lean_dec_ref(v_bs_3231_);
v___x_3246_ = lean_box(0);
return v___x_3246_;
}
else
{
lean_object* v___x_3247_; lean_object* v___x_3248_; uint8_t v___x_3249_; 
v___x_3247_ = lean_unsigned_to_nat(1u);
v___x_3248_ = l_Lean_Syntax_getArg(v___x_3239_, v___x_3247_);
lean_dec(v___x_3239_);
v___x_3249_ = l_Lean_Syntax_matchesNull(v___x_3248_, v___x_3238_);
if (v___x_3249_ == 0)
{
lean_object* v___x_3250_; 
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3250_ = lean_box(0);
return v___x_3250_;
}
else
{
lean_object* v___x_3251_; lean_object* v___x_3252_; uint8_t v___x_3253_; 
v___x_3251_ = lean_unsigned_to_nat(3u);
v___x_3252_ = l_Lean_Syntax_getArg(v_v_3234_, v___x_3247_);
lean_inc(v___x_3252_);
v___x_3253_ = l_Lean_Syntax_matchesNull(v___x_3252_, v___x_3251_);
if (v___x_3253_ == 0)
{
lean_object* v___x_3254_; 
lean_dec(v___x_3252_);
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3254_ = lean_box(0);
return v___x_3254_;
}
else
{
lean_object* v___x_3255_; uint8_t v___x_3256_; 
v___x_3255_ = l_Lean_Syntax_getArg(v___x_3252_, v___x_3238_);
v___x_3256_ = l_Lean_Syntax_matchesNull(v___x_3255_, v___x_3238_);
if (v___x_3256_ == 0)
{
lean_object* v___x_3257_; 
lean_dec(v___x_3252_);
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3257_ = lean_box(0);
return v___x_3257_;
}
else
{
lean_object* v___x_3258_; uint8_t v___x_3259_; 
v___x_3258_ = l_Lean_Syntax_getArg(v___x_3252_, v___x_3247_);
v___x_3259_ = l_Lean_Syntax_matchesNull(v___x_3258_, v___x_3238_);
if (v___x_3259_ == 0)
{
lean_object* v___x_3260_; 
lean_dec(v___x_3252_);
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3260_ = lean_box(0);
return v___x_3260_;
}
else
{
lean_object* v___x_3261_; lean_object* v___x_3262_; lean_object* v___x_3263_; uint8_t v___x_3264_; 
v___x_3261_ = lean_unsigned_to_nat(2u);
v___x_3262_ = l_Lean_Syntax_getArg(v___x_3252_, v___x_3261_);
lean_dec(v___x_3252_);
v___x_3263_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00ProofWidgets_Jsx_transformTag_spec__5_spec__8___closed__5));
lean_inc(v___x_3262_);
v___x_3264_ = l_Lean_Syntax_isOfKind(v___x_3262_, v___x_3263_);
if (v___x_3264_ == 0)
{
lean_object* v___x_3265_; 
lean_dec(v___x_3262_);
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3265_ = lean_box(0);
return v___x_3265_;
}
else
{
lean_object* v___x_3266_; uint8_t v___x_3267_; 
v___x_3266_ = l_Lean_Syntax_getArg(v___x_3262_, v___x_3247_);
v___x_3267_ = l_Lean_Syntax_matchesNull(v___x_3266_, v___x_3238_);
if (v___x_3267_ == 0)
{
lean_object* v___x_3268_; 
lean_dec(v___x_3262_);
lean_dec(v___x_3243_);
lean_dec_ref(v_bs_3231_);
v___x_3268_ = lean_box(0);
return v___x_3268_;
}
else
{
lean_object* v_bs_x27_3269_; lean_object* v___x_3270_; lean_object* v___x_3271_; size_t v___x_3272_; size_t v___x_3273_; lean_object* v___x_3274_; 
v_bs_x27_3269_ = lean_array_uset(v_bs_3231_, v_i_3230_, v___x_3238_);
v___x_3270_ = l_Lean_Syntax_getArg(v___x_3262_, v___x_3261_);
lean_dec(v___x_3262_);
v___x_3271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3271_, 0, v___x_3243_);
lean_ctor_set(v___x_3271_, 1, v___x_3270_);
v___x_3272_ = ((size_t)1ULL);
v___x_3273_ = lean_usize_add(v_i_3230_, v___x_3272_);
v___x_3274_ = lean_array_uset(v_bs_x27_3269_, v_i_3230_, v___x_3271_);
v_i_3230_ = v___x_3273_;
v_bs_3231_ = v___x_3274_;
goto _start;
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
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5___boxed(lean_object* v_sz_3276_, lean_object* v_i_3277_, lean_object* v_bs_3278_){
_start:
{
size_t v_sz_boxed_3279_; size_t v_i_boxed_3280_; lean_object* v_res_3281_; 
v_sz_boxed_3279_ = lean_unbox_usize(v_sz_3276_);
lean_dec(v_sz_3276_);
v_i_boxed_3280_ = lean_unbox_usize(v_i_3277_);
lean_dec(v_i_3277_);
v_res_3281_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5(v_sz_boxed_3279_, v_i_boxed_3280_, v_bs_3278_);
return v_res_3281_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9(uint8_t v___x_3282_, lean_object* v_as_3283_, size_t v_i_3284_, size_t v_stop_3285_, lean_object* v_b_3286_){
_start:
{
lean_object* v___y_3288_; uint8_t v___x_3292_; 
v___x_3292_ = lean_usize_dec_eq(v_i_3284_, v_stop_3285_);
if (v___x_3292_ == 0)
{
lean_object* v_fst_3293_; uint8_t v___x_3294_; 
v_fst_3293_ = lean_ctor_get(v_b_3286_, 0);
v___x_3294_ = lean_unbox(v_fst_3293_);
if (v___x_3294_ == 0)
{
lean_object* v_snd_3295_; lean_object* v___x_3297_; uint8_t v_isShared_3298_; uint8_t v_isSharedCheck_3303_; 
v_snd_3295_ = lean_ctor_get(v_b_3286_, 1);
v_isSharedCheck_3303_ = !lean_is_exclusive(v_b_3286_);
if (v_isSharedCheck_3303_ == 0)
{
lean_object* v_unused_3304_; 
v_unused_3304_ = lean_ctor_get(v_b_3286_, 0);
lean_dec(v_unused_3304_);
v___x_3297_ = v_b_3286_;
v_isShared_3298_ = v_isSharedCheck_3303_;
goto v_resetjp_3296_;
}
else
{
lean_inc(v_snd_3295_);
lean_dec(v_b_3286_);
v___x_3297_ = lean_box(0);
v_isShared_3298_ = v_isSharedCheck_3303_;
goto v_resetjp_3296_;
}
v_resetjp_3296_:
{
lean_object* v___x_3299_; lean_object* v___x_3301_; 
v___x_3299_ = lean_box(v___x_3282_);
if (v_isShared_3298_ == 0)
{
lean_ctor_set(v___x_3297_, 0, v___x_3299_);
v___x_3301_ = v___x_3297_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3302_; 
v_reuseFailAlloc_3302_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3302_, 0, v___x_3299_);
lean_ctor_set(v_reuseFailAlloc_3302_, 1, v_snd_3295_);
v___x_3301_ = v_reuseFailAlloc_3302_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
v___y_3288_ = v___x_3301_;
goto v___jp_3287_;
}
}
}
else
{
lean_object* v_snd_3305_; lean_object* v___x_3307_; uint8_t v_isShared_3308_; uint8_t v_isSharedCheck_3315_; 
v_snd_3305_ = lean_ctor_get(v_b_3286_, 1);
v_isSharedCheck_3315_ = !lean_is_exclusive(v_b_3286_);
if (v_isSharedCheck_3315_ == 0)
{
lean_object* v_unused_3316_; 
v_unused_3316_ = lean_ctor_get(v_b_3286_, 0);
lean_dec(v_unused_3316_);
v___x_3307_ = v_b_3286_;
v_isShared_3308_ = v_isSharedCheck_3315_;
goto v_resetjp_3306_;
}
else
{
lean_inc(v_snd_3305_);
lean_dec(v_b_3286_);
v___x_3307_ = lean_box(0);
v_isShared_3308_ = v_isSharedCheck_3315_;
goto v_resetjp_3306_;
}
v_resetjp_3306_:
{
lean_object* v___x_3309_; lean_object* v___x_3310_; lean_object* v___x_3311_; lean_object* v___x_3313_; 
v___x_3309_ = lean_array_uget_borrowed(v_as_3283_, v_i_3284_);
lean_inc(v___x_3309_);
v___x_3310_ = lean_array_push(v_snd_3305_, v___x_3309_);
v___x_3311_ = lean_box(v___x_3292_);
if (v_isShared_3308_ == 0)
{
lean_ctor_set(v___x_3307_, 1, v___x_3310_);
lean_ctor_set(v___x_3307_, 0, v___x_3311_);
v___x_3313_ = v___x_3307_;
goto v_reusejp_3312_;
}
else
{
lean_object* v_reuseFailAlloc_3314_; 
v_reuseFailAlloc_3314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3314_, 0, v___x_3311_);
lean_ctor_set(v_reuseFailAlloc_3314_, 1, v___x_3310_);
v___x_3313_ = v_reuseFailAlloc_3314_;
goto v_reusejp_3312_;
}
v_reusejp_3312_:
{
v___y_3288_ = v___x_3313_;
goto v___jp_3287_;
}
}
}
}
else
{
return v_b_3286_;
}
v___jp_3287_:
{
size_t v___x_3289_; size_t v___x_3290_; 
v___x_3289_ = ((size_t)1ULL);
v___x_3290_ = lean_usize_add(v_i_3284_, v___x_3289_);
v_i_3284_ = v___x_3290_;
v_b_3286_ = v___y_3288_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9___boxed(lean_object* v___x_3317_, lean_object* v_as_3318_, lean_object* v_i_3319_, lean_object* v_stop_3320_, lean_object* v_b_3321_){
_start:
{
uint8_t v___x_131021__boxed_3322_; size_t v_i_boxed_3323_; size_t v_stop_boxed_3324_; lean_object* v_res_3325_; 
v___x_131021__boxed_3322_ = lean_unbox(v___x_3317_);
v_i_boxed_3323_ = lean_unbox_usize(v_i_3319_);
lean_dec(v_i_3319_);
v_stop_boxed_3324_ = lean_unbox_usize(v_stop_3320_);
lean_dec(v_stop_3320_);
v_res_3325_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9(v___x_131021__boxed_3322_, v_as_3318_, v_i_boxed_3323_, v_stop_boxed_3324_, v_b_3321_);
lean_dec_ref(v_as_3318_);
return v_res_3325_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg(size_t v_sz_3326_, size_t v_i_3327_, lean_object* v_bs_3328_, lean_object* v___y_3329_){
_start:
{
uint8_t v___x_3331_; 
v___x_3331_ = lean_usize_dec_lt(v_i_3327_, v_sz_3326_);
if (v___x_3331_ == 0)
{
lean_object* v___x_3332_; 
v___x_3332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3332_, 0, v_bs_3328_);
return v___x_3332_;
}
else
{
lean_object* v_v_3333_; lean_object* v_fst_3334_; lean_object* v_snd_3335_; lean_object* v___x_3337_; uint8_t v_isShared_3338_; uint8_t v_isSharedCheck_3362_; 
v_v_3333_ = lean_array_uget(v_bs_3328_, v_i_3327_);
v_fst_3334_ = lean_ctor_get(v_v_3333_, 0);
v_snd_3335_ = lean_ctor_get(v_v_3333_, 1);
v_isSharedCheck_3362_ = !lean_is_exclusive(v_v_3333_);
if (v_isSharedCheck_3362_ == 0)
{
v___x_3337_ = v_v_3333_;
v_isShared_3338_ = v_isSharedCheck_3362_;
goto v_resetjp_3336_;
}
else
{
lean_inc(v_snd_3335_);
lean_inc(v_fst_3334_);
lean_dec(v_v_3333_);
v___x_3337_ = lean_box(0);
v_isShared_3338_ = v_isSharedCheck_3362_;
goto v_resetjp_3336_;
}
v_resetjp_3336_:
{
lean_object* v_ref_3339_; lean_object* v___x_3340_; lean_object* v_bs_x27_3341_; uint8_t v___x_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3347_; 
v_ref_3339_ = lean_ctor_get(v___y_3329_, 5);
v___x_3340_ = lean_unsigned_to_nat(0u);
v_bs_x27_3341_ = lean_array_uset(v_bs_3328_, v_i_3327_, v___x_3340_);
v___x_3342_ = 0;
v___x_3343_ = l_Lean_SourceInfo_fromRef(v_ref_3339_, v___x_3342_);
v___x_3344_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__1));
v___x_3345_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr___x3d___00__closed__5));
lean_inc(v___x_3343_);
if (v_isShared_3338_ == 0)
{
lean_ctor_set_tag(v___x_3337_, 2);
lean_ctor_set(v___x_3337_, 1, v___x_3345_);
lean_ctor_set(v___x_3337_, 0, v___x_3343_);
v___x_3347_ = v___x_3337_;
goto v_reusejp_3346_;
}
else
{
lean_object* v_reuseFailAlloc_3361_; 
v_reuseFailAlloc_3361_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3361_, 0, v___x_3343_);
lean_ctor_set(v_reuseFailAlloc_3361_, 1, v___x_3345_);
v___x_3347_ = v_reuseFailAlloc_3361_;
goto v_reusejp_3346_;
}
v_reusejp_3346_:
{
lean_object* v___x_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; size_t v___x_3357_; size_t v___x_3358_; lean_object* v___x_3359_; 
v___x_3348_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__1));
v___x_3349_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3350_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4));
lean_inc_n(v___x_3343_, 4);
v___x_3351_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3351_, 0, v___x_3343_);
lean_ctor_set(v___x_3351_, 1, v___x_3350_);
v___x_3352_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3353_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3353_, 0, v___x_3343_);
lean_ctor_set(v___x_3353_, 1, v___x_3352_);
v___x_3354_ = l_Lean_Syntax_node3(v___x_3343_, v___x_3349_, v___x_3351_, v_snd_3335_, v___x_3353_);
v___x_3355_ = l_Lean_Syntax_node1(v___x_3343_, v___x_3348_, v___x_3354_);
v___x_3356_ = l_Lean_Syntax_node3(v___x_3343_, v___x_3344_, v_fst_3334_, v___x_3347_, v___x_3355_);
v___x_3357_ = ((size_t)1ULL);
v___x_3358_ = lean_usize_add(v_i_3327_, v___x_3357_);
v___x_3359_ = lean_array_uset(v_bs_x27_3341_, v_i_3327_, v___x_3356_);
v_i_3327_ = v___x_3358_;
v_bs_3328_ = v___x_3359_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg___boxed(lean_object* v_sz_3363_, lean_object* v_i_3364_, lean_object* v_bs_3365_, lean_object* v___y_3366_, lean_object* v___y_3367_){
_start:
{
size_t v_sz_boxed_3368_; size_t v_i_boxed_3369_; lean_object* v_res_3370_; 
v_sz_boxed_3368_ = lean_unbox_usize(v_sz_3363_);
lean_dec(v_sz_3363_);
v_i_boxed_3369_ = lean_unbox_usize(v_i_3364_);
lean_dec(v_i_3364_);
v_res_3370_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg(v_sz_boxed_3368_, v_i_boxed_3369_, v_bs_3365_, v___y_3366_);
lean_dec_ref(v___y_3366_);
return v_res_3370_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7(size_t v_sz_3371_, size_t v_i_3372_, lean_object* v_bs_3373_){
_start:
{
uint8_t v___x_3374_; 
v___x_3374_ = lean_usize_dec_lt(v_i_3372_, v_sz_3371_);
if (v___x_3374_ == 0)
{
return v_bs_3373_;
}
else
{
lean_object* v_v_3375_; lean_object* v_fst_3376_; lean_object* v___x_3377_; lean_object* v_bs_x27_3378_; size_t v___x_3379_; size_t v___x_3380_; lean_object* v___x_3381_; 
v_v_3375_ = lean_array_uget_borrowed(v_bs_3373_, v_i_3372_);
v_fst_3376_ = lean_ctor_get(v_v_3375_, 0);
lean_inc(v_fst_3376_);
v___x_3377_ = lean_unsigned_to_nat(0u);
v_bs_x27_3378_ = lean_array_uset(v_bs_3373_, v_i_3372_, v___x_3377_);
v___x_3379_ = ((size_t)1ULL);
v___x_3380_ = lean_usize_add(v_i_3372_, v___x_3379_);
v___x_3381_ = lean_array_uset(v_bs_x27_3378_, v_i_3372_, v_fst_3376_);
v_i_3372_ = v___x_3380_;
v_bs_3373_ = v___x_3381_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7___boxed(lean_object* v_sz_3383_, lean_object* v_i_3384_, lean_object* v_bs_3385_){
_start:
{
size_t v_sz_boxed_3386_; size_t v_i_boxed_3387_; lean_object* v_res_3388_; 
v_sz_boxed_3386_ = lean_unbox_usize(v_sz_3383_);
lean_dec(v_sz_3383_);
v_i_boxed_3387_ = lean_unbox_usize(v_i_3384_);
lean_dec(v_i_3384_);
v_res_3388_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7(v_sz_boxed_3386_, v_i_boxed_3387_, v_bs_3385_);
return v_res_3388_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0(lean_object* v_x_3389_, lean_object* v___y_3390_, lean_object* v___y_3391_, lean_object* v___y_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_){
_start:
{
lean_object* v___x_3397_; 
v___x_3397_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_3397_) == 0)
{
lean_object* v_a_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3406_; 
v_a_3398_ = lean_ctor_get(v___x_3397_, 0);
v_isSharedCheck_3406_ = !lean_is_exclusive(v___x_3397_);
if (v_isSharedCheck_3406_ == 0)
{
v___x_3400_ = v___x_3397_;
v_isShared_3401_ = v_isSharedCheck_3406_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_a_3398_);
lean_dec(v___x_3397_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3406_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3402_; lean_object* v___x_3404_; 
v___x_3402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3402_, 0, v_a_3398_);
if (v_isShared_3401_ == 0)
{
lean_ctor_set(v___x_3400_, 0, v___x_3402_);
v___x_3404_ = v___x_3400_;
goto v_reusejp_3403_;
}
else
{
lean_object* v_reuseFailAlloc_3405_; 
v_reuseFailAlloc_3405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3405_, 0, v___x_3402_);
v___x_3404_ = v_reuseFailAlloc_3405_;
goto v_reusejp_3403_;
}
v_reusejp_3403_:
{
return v___x_3404_;
}
}
}
else
{
lean_object* v_a_3407_; lean_object* v___x_3409_; uint8_t v_isShared_3410_; uint8_t v_isSharedCheck_3414_; 
v_a_3407_ = lean_ctor_get(v___x_3397_, 0);
v_isSharedCheck_3414_ = !lean_is_exclusive(v___x_3397_);
if (v_isSharedCheck_3414_ == 0)
{
v___x_3409_ = v___x_3397_;
v_isShared_3410_ = v_isSharedCheck_3414_;
goto v_resetjp_3408_;
}
else
{
lean_inc(v_a_3407_);
lean_dec(v___x_3397_);
v___x_3409_ = lean_box(0);
v_isShared_3410_ = v_isSharedCheck_3414_;
goto v_resetjp_3408_;
}
v_resetjp_3408_:
{
lean_object* v___x_3412_; 
if (v_isShared_3410_ == 0)
{
v___x_3412_ = v___x_3409_;
goto v_reusejp_3411_;
}
else
{
lean_object* v_reuseFailAlloc_3413_; 
v_reuseFailAlloc_3413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3413_, 0, v_a_3407_);
v___x_3412_ = v_reuseFailAlloc_3413_;
goto v_reusejp_3411_;
}
v_reusejp_3411_:
{
return v___x_3412_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0___boxed(lean_object* v_x_3415_, lean_object* v___y_3416_, lean_object* v___y_3417_, lean_object* v___y_3418_, lean_object* v___y_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_, lean_object* v___y_3422_){
_start:
{
lean_object* v_res_3423_; 
v_res_3423_ = lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__0(v_x_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
lean_dec(v___y_3421_);
lean_dec_ref(v___y_3420_);
lean_dec(v___y_3419_);
lean_dec_ref(v___y_3418_);
lean_dec(v___y_3417_);
lean_dec_ref(v___y_3416_);
return v_res_3423_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___boxed(lean_object* v_a_3429_, lean_object* v_a_3430_, lean_object* v_a_3431_, lean_object* v_a_3432_, lean_object* v_a_3433_, lean_object* v_a_3434_, lean_object* v_a_3435_){
_start:
{
lean_object* v_res_3436_; 
v_res_3436_ = lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren(v_a_3429_, v_a_3430_, v_a_3431_, v_a_3432_, v_a_3433_, v_a_3434_);
lean_dec(v_a_3434_);
lean_dec_ref(v_a_3433_);
lean_dec(v_a_3432_);
lean_dec_ref(v_a_3431_);
lean_dec(v_a_3430_);
lean_dec_ref(v_a_3429_);
return v_res_3436_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27(lean_object* v_a_3439_, lean_object* v_a_3440_, lean_object* v_a_3441_, lean_object* v_a_3442_, lean_object* v_a_3443_, lean_object* v_a_3444_){
_start:
{
lean_object* v___x_3446_; 
v___x_3446_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v_a_3439_);
if (lean_obj_tag(v___x_3446_) == 0)
{
lean_object* v_a_3447_; lean_object* v___x_3448_; uint8_t v___x_3449_; 
v_a_3447_ = lean_ctor_get(v___x_3446_, 0);
lean_inc(v_a_3447_);
lean_dec_ref_known(v___x_3446_, 1);
v___x_3448_ = l_Lean_Expr_cleanupAnnotations(v_a_3447_);
v___x_3449_ = l_Lean_Expr_isApp(v___x_3448_);
if (v___x_3449_ == 0)
{
lean_object* v___x_3450_; 
lean_dec_ref(v___x_3448_);
v___x_3450_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3450_;
}
else
{
lean_object* v___x_3451_; uint8_t v___x_3452_; 
v___x_3451_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3448_);
v___x_3452_ = l_Lean_Expr_isApp(v___x_3451_);
if (v___x_3452_ == 0)
{
lean_object* v___x_3453_; 
lean_dec_ref(v___x_3451_);
v___x_3453_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3453_;
}
else
{
lean_object* v___x_3454_; uint8_t v___x_3455_; 
v___x_3454_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3451_);
v___x_3455_ = l_Lean_Expr_isApp(v___x_3454_);
if (v___x_3455_ == 0)
{
lean_object* v___x_3456_; 
lean_dec_ref(v___x_3454_);
v___x_3456_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3456_;
}
else
{
lean_object* v___x_3457_; uint8_t v___x_3458_; 
v___x_3457_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3454_);
v___x_3458_ = l_Lean_Expr_isApp(v___x_3457_);
if (v___x_3458_ == 0)
{
lean_object* v___x_3459_; 
lean_dec_ref(v___x_3457_);
v___x_3459_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3459_;
}
else
{
lean_object* v___x_3460_; uint8_t v___x_3461_; 
v___x_3460_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3457_);
v___x_3461_ = l_Lean_Expr_isApp(v___x_3460_);
if (v___x_3461_ == 0)
{
lean_object* v___x_3462_; 
lean_dec_ref(v___x_3460_);
v___x_3462_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3462_;
}
else
{
lean_object* v___x_3463_; lean_object* v___x_3464_; uint8_t v___x_3465_; 
v___x_3463_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3460_);
v___x_3464_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12));
v___x_3465_ = l_Lean_Expr_isConstOf(v___x_3463_, v___x_3464_);
lean_dec_ref(v___x_3463_);
if (v___x_3465_ == 0)
{
lean_object* v___x_3466_; 
v___x_3466_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3466_;
}
else
{
lean_object* v___x_3467_; lean_object* v___x_3468_; lean_object* v___x_3469_; 
v___x_3467_ = lean_unsigned_to_nat(2u);
v___x_3468_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__0));
v___x_3469_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3467_, v___x_3468_, v_a_3439_, v_a_3440_, v_a_3441_, v_a_3442_, v_a_3443_, v_a_3444_);
if (lean_obj_tag(v___x_3469_) == 0)
{
lean_object* v_a_3470_; lean_object* v_attrs_3472_; lean_object* v___y_3473_; lean_object* v___y_3474_; lean_object* v___y_3475_; lean_object* v___y_3476_; lean_object* v___y_3477_; lean_object* v___y_3478_; lean_object* v___y_3542_; lean_object* v___y_3543_; lean_object* v___y_3544_; lean_object* v___y_3545_; lean_object* v___y_3546_; lean_object* v___y_3547_; lean_object* v___y_3548_; lean_object* v___y_3549_; lean_object* v___y_3550_; lean_object* v___y_3551_; lean_object* v___y_3552_; lean_object* v___y_3553_; lean_object* v___y_3554_; lean_object* v___y_3555_; lean_object* v___y_3634_; lean_object* v___y_3635_; lean_object* v___y_3636_; lean_object* v___y_3637_; lean_object* v___y_3638_; lean_object* v___y_3639_; uint8_t v___x_3708_; 
v_a_3470_ = lean_ctor_get(v___x_3469_, 0);
lean_inc(v_a_3470_);
lean_dec_ref_known(v___x_3469_, 1);
v___x_3708_ = l_Lean_Syntax_isIdent(v_a_3470_);
if (v___x_3708_ == 0)
{
lean_object* v___x_3709_; 
v___x_3709_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_3709_) == 0)
{
lean_dec_ref_known(v___x_3709_, 1);
v___y_3634_ = v_a_3439_;
v___y_3635_ = v_a_3440_;
v___y_3636_ = v_a_3441_;
v___y_3637_ = v_a_3442_;
v___y_3638_ = v_a_3443_;
v___y_3639_ = v_a_3444_;
goto v___jp_3633_;
}
else
{
lean_object* v_a_3710_; lean_object* v___x_3712_; uint8_t v_isShared_3713_; uint8_t v_isSharedCheck_3717_; 
lean_dec(v_a_3470_);
v_a_3710_ = lean_ctor_get(v___x_3709_, 0);
v_isSharedCheck_3717_ = !lean_is_exclusive(v___x_3709_);
if (v_isSharedCheck_3717_ == 0)
{
v___x_3712_ = v___x_3709_;
v_isShared_3713_ = v_isSharedCheck_3717_;
goto v_resetjp_3711_;
}
else
{
lean_inc(v_a_3710_);
lean_dec(v___x_3709_);
v___x_3712_ = lean_box(0);
v_isShared_3713_ = v_isSharedCheck_3717_;
goto v_resetjp_3711_;
}
v_resetjp_3711_:
{
lean_object* v___x_3715_; 
if (v_isShared_3713_ == 0)
{
v___x_3715_ = v___x_3712_;
goto v_reusejp_3714_;
}
else
{
lean_object* v_reuseFailAlloc_3716_; 
v_reuseFailAlloc_3716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3716_, 0, v_a_3710_);
v___x_3715_ = v_reuseFailAlloc_3716_;
goto v_reusejp_3714_;
}
v_reusejp_3714_:
{
return v___x_3715_;
}
}
}
}
else
{
v___y_3634_ = v_a_3439_;
v___y_3635_ = v_a_3440_;
v___y_3636_ = v_a_3441_;
v___y_3637_ = v_a_3442_;
v___y_3638_ = v_a_3443_;
v___y_3639_ = v_a_3444_;
goto v___jp_3633_;
}
v___jp_3471_:
{
lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; 
v___x_3479_ = lean_unsigned_to_nat(4u);
v___x_3480_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___boxed), 7, 0);
v___x_3481_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3479_, v___x_3480_, v___y_3473_, v___y_3474_, v___y_3475_, v___y_3476_, v___y_3477_, v___y_3478_);
if (lean_obj_tag(v___x_3481_) == 0)
{
lean_object* v_a_3482_; lean_object* v___x_3484_; uint8_t v_isShared_3485_; uint8_t v_isSharedCheck_3532_; 
v_a_3482_ = lean_ctor_get(v___x_3481_, 0);
v_isSharedCheck_3532_ = !lean_is_exclusive(v___x_3481_);
if (v_isSharedCheck_3532_ == 0)
{
v___x_3484_ = v___x_3481_;
v_isShared_3485_ = v_isSharedCheck_3532_;
goto v_resetjp_3483_;
}
else
{
lean_inc(v_a_3482_);
lean_dec(v___x_3481_);
v___x_3484_ = lean_box(0);
v_isShared_3485_ = v_isSharedCheck_3532_;
goto v_resetjp_3483_;
}
v_resetjp_3483_:
{
lean_object* v___x_3486_; lean_object* v___x_3487_; uint8_t v___x_3488_; 
v___x_3486_ = lean_array_get_size(v_a_3482_);
v___x_3487_ = lean_unsigned_to_nat(0u);
v___x_3488_ = lean_nat_dec_eq(v___x_3486_, v___x_3487_);
if (v___x_3488_ == 0)
{
lean_object* v_ref_3489_; lean_object* v___x_3490_; lean_object* v___x_3491_; lean_object* v___x_3492_; lean_object* v___x_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; size_t v_sz_3496_; size_t v___x_3497_; lean_object* v___x_3498_; lean_object* v___x_3499_; lean_object* v___x_3500_; lean_object* v___x_3501_; lean_object* v___x_3502_; size_t v_sz_3503_; lean_object* v___x_3504_; lean_object* v___x_3505_; lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3508_; lean_object* v___x_3509_; lean_object* v___x_3511_; 
v_ref_3489_ = lean_ctor_get(v___y_3477_, 5);
v___x_3490_ = l_Lean_SourceInfo_fromRef(v_ref_3489_, v___x_3488_);
v___x_3491_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1));
v___x_3492_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2));
lean_inc_n(v___x_3490_, 5);
v___x_3493_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3493_, 0, v___x_3490_);
lean_ctor_set(v___x_3493_, 1, v___x_3492_);
v___x_3494_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_3495_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v_sz_3496_ = lean_array_size(v_attrs_3472_);
v___x_3497_ = ((size_t)0ULL);
v___x_3498_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3496_, v___x_3497_, v_attrs_3472_);
v___x_3499_ = l_Array_append___redArg(v___x_3495_, v___x_3498_);
lean_dec_ref(v___x_3498_);
v___x_3500_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3500_, 0, v___x_3490_);
lean_ctor_set(v___x_3500_, 1, v___x_3494_);
lean_ctor_set(v___x_3500_, 2, v___x_3499_);
v___x_3501_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2));
v___x_3502_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3502_, 0, v___x_3490_);
lean_ctor_set(v___x_3502_, 1, v___x_3501_);
v_sz_3503_ = lean_array_size(v_a_3482_);
v___x_3504_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3503_, v___x_3497_, v_a_3482_);
v___x_3505_ = l_Array_append___redArg(v___x_3495_, v___x_3504_);
lean_dec_ref(v___x_3504_);
v___x_3506_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3506_, 0, v___x_3490_);
lean_ctor_set(v___x_3506_, 1, v___x_3494_);
lean_ctor_set(v___x_3506_, 2, v___x_3505_);
v___x_3507_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7));
v___x_3508_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3508_, 0, v___x_3490_);
lean_ctor_set(v___x_3508_, 1, v___x_3507_);
lean_inc_ref(v___x_3502_);
lean_inc(v_a_3470_);
v___x_3509_ = l_Lean_Syntax_node8(v___x_3490_, v___x_3491_, v___x_3493_, v_a_3470_, v___x_3500_, v___x_3502_, v___x_3506_, v___x_3508_, v_a_3470_, v___x_3502_);
if (v_isShared_3485_ == 0)
{
lean_ctor_set(v___x_3484_, 0, v___x_3509_);
v___x_3511_ = v___x_3484_;
goto v_reusejp_3510_;
}
else
{
lean_object* v_reuseFailAlloc_3512_; 
v_reuseFailAlloc_3512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3512_, 0, v___x_3509_);
v___x_3511_ = v_reuseFailAlloc_3512_;
goto v_reusejp_3510_;
}
v_reusejp_3510_:
{
return v___x_3511_;
}
}
else
{
lean_object* v_ref_3513_; uint8_t v___x_3514_; lean_object* v___x_3515_; lean_object* v___x_3516_; lean_object* v___x_3517_; lean_object* v___x_3518_; lean_object* v___x_3519_; lean_object* v___x_3520_; size_t v_sz_3521_; size_t v___x_3522_; lean_object* v___x_3523_; lean_object* v___x_3524_; lean_object* v___x_3525_; lean_object* v___x_3526_; lean_object* v___x_3527_; lean_object* v___x_3528_; lean_object* v___x_3530_; 
lean_dec(v_a_3482_);
v_ref_3513_ = lean_ctor_get(v___y_3477_, 5);
v___x_3514_ = 0;
v___x_3515_ = l_Lean_SourceInfo_fromRef(v_ref_3513_, v___x_3514_);
v___x_3516_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1));
v___x_3517_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2));
lean_inc_n(v___x_3515_, 3);
v___x_3518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3518_, 0, v___x_3515_);
lean_ctor_set(v___x_3518_, 1, v___x_3517_);
v___x_3519_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_3520_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v_sz_3521_ = lean_array_size(v_attrs_3472_);
v___x_3522_ = ((size_t)0ULL);
v___x_3523_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3521_, v___x_3522_, v_attrs_3472_);
v___x_3524_ = l_Array_append___redArg(v___x_3520_, v___x_3523_);
lean_dec_ref(v___x_3523_);
v___x_3525_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3525_, 0, v___x_3515_);
lean_ctor_set(v___x_3525_, 1, v___x_3519_);
lean_ctor_set(v___x_3525_, 2, v___x_3524_);
v___x_3526_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9));
v___x_3527_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3527_, 0, v___x_3515_);
lean_ctor_set(v___x_3527_, 1, v___x_3526_);
v___x_3528_ = l_Lean_Syntax_node4(v___x_3515_, v___x_3516_, v___x_3518_, v_a_3470_, v___x_3525_, v___x_3527_);
if (v_isShared_3485_ == 0)
{
lean_ctor_set(v___x_3484_, 0, v___x_3528_);
v___x_3530_ = v___x_3484_;
goto v_reusejp_3529_;
}
else
{
lean_object* v_reuseFailAlloc_3531_; 
v_reuseFailAlloc_3531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3531_, 0, v___x_3528_);
v___x_3530_ = v_reuseFailAlloc_3531_;
goto v_reusejp_3529_;
}
v_reusejp_3529_:
{
return v___x_3530_;
}
}
}
}
else
{
lean_object* v_a_3533_; lean_object* v___x_3535_; uint8_t v_isShared_3536_; uint8_t v_isSharedCheck_3540_; 
lean_dec_ref(v_attrs_3472_);
lean_dec(v_a_3470_);
v_a_3533_ = lean_ctor_get(v___x_3481_, 0);
v_isSharedCheck_3540_ = !lean_is_exclusive(v___x_3481_);
if (v_isSharedCheck_3540_ == 0)
{
v___x_3535_ = v___x_3481_;
v_isShared_3536_ = v_isSharedCheck_3540_;
goto v_resetjp_3534_;
}
else
{
lean_inc(v_a_3533_);
lean_dec(v___x_3481_);
v___x_3535_ = lean_box(0);
v_isShared_3536_ = v_isSharedCheck_3540_;
goto v_resetjp_3534_;
}
v_resetjp_3534_:
{
lean_object* v___x_3538_; 
if (v_isShared_3536_ == 0)
{
v___x_3538_ = v___x_3535_;
goto v_reusejp_3537_;
}
else
{
lean_object* v_reuseFailAlloc_3539_; 
v_reuseFailAlloc_3539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3539_, 0, v_a_3533_);
v___x_3538_ = v_reuseFailAlloc_3539_;
goto v_reusejp_3537_;
}
v_reusejp_3537_:
{
return v___x_3538_;
}
}
}
}
v___jp_3541_:
{
size_t v_sz_3556_; size_t v___x_3557_; lean_object* v___x_3558_; 
v_sz_3556_ = lean_array_size(v___y_3555_);
v___x_3557_ = ((size_t)0ULL);
v___x_3558_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__5(v_sz_3556_, v___x_3557_, v___y_3555_);
if (lean_obj_tag(v___x_3558_) == 0)
{
lean_object* v_ref_3559_; uint8_t v___x_3560_; lean_object* v___x_3561_; lean_object* v___x_3562_; lean_object* v___x_3563_; lean_object* v___x_3564_; lean_object* v___x_3565_; lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3568_; lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; 
v_ref_3559_ = lean_ctor_get(v___y_3553_, 5);
v___x_3560_ = 0;
v___x_3561_ = l_Lean_SourceInfo_fromRef(v_ref_3559_, v___x_3560_);
v___x_3562_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3563_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3564_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3561_, 3);
v___x_3565_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3565_, 0, v___x_3561_);
lean_ctor_set(v___x_3565_, 1, v___x_3564_);
v___x_3566_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3567_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3567_, 0, v___x_3561_);
lean_ctor_set(v___x_3567_, 1, v___x_3566_);
v___x_3568_ = l_Lean_Syntax_node3(v___x_3561_, v___x_3563_, v___x_3565_, v___y_3549_, v___x_3567_);
v___x_3569_ = l_Lean_Syntax_node1(v___x_3561_, v___x_3562_, v___x_3568_);
v___x_3570_ = lean_mk_empty_array_with_capacity(v___y_3548_);
v___x_3571_ = lean_array_push(v___x_3570_, v___x_3569_);
v_attrs_3472_ = v___x_3571_;
v___y_3473_ = v___y_3542_;
v___y_3474_ = v___y_3550_;
v___y_3475_ = v___y_3552_;
v___y_3476_ = v___y_3545_;
v___y_3477_ = v___y_3553_;
v___y_3478_ = v___y_3546_;
goto v___jp_3471_;
}
else
{
lean_object* v_val_3572_; lean_object* v___x_3573_; lean_object* v___x_3574_; lean_object* v___x_3575_; uint8_t v___x_3576_; 
v_val_3572_ = lean_ctor_get(v___x_3558_, 0);
lean_inc(v_val_3572_);
lean_dec_ref_known(v___x_3558_, 1);
v___x_3573_ = l_Lean_Syntax_getArg(v___y_3549_, v___y_3554_);
v___x_3574_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__20));
lean_inc_ref(v___y_3543_);
lean_inc_ref(v___y_3547_);
lean_inc_ref(v___y_3551_);
v___x_3575_ = l_Lean_Name_mkStr4(v___y_3551_, v___y_3547_, v___y_3543_, v___x_3574_);
lean_inc(v___x_3573_);
v___x_3576_ = l_Lean_Syntax_isOfKind(v___x_3573_, v___x_3575_);
lean_dec(v___x_3575_);
if (v___x_3576_ == 0)
{
lean_object* v_ref_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v___x_3585_; lean_object* v___x_3586_; lean_object* v___x_3587_; lean_object* v___x_3588_; 
lean_dec(v___x_3573_);
lean_dec(v_val_3572_);
v_ref_3577_ = lean_ctor_get(v___y_3553_, 5);
v___x_3578_ = l_Lean_SourceInfo_fromRef(v_ref_3577_, v___x_3576_);
v___x_3579_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3580_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3581_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3578_, 3);
v___x_3582_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3582_, 0, v___x_3578_);
lean_ctor_set(v___x_3582_, 1, v___x_3581_);
v___x_3583_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3584_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3584_, 0, v___x_3578_);
lean_ctor_set(v___x_3584_, 1, v___x_3583_);
v___x_3585_ = l_Lean_Syntax_node3(v___x_3578_, v___x_3580_, v___x_3582_, v___y_3549_, v___x_3584_);
v___x_3586_ = l_Lean_Syntax_node1(v___x_3578_, v___x_3579_, v___x_3585_);
v___x_3587_ = lean_mk_empty_array_with_capacity(v___y_3548_);
v___x_3588_ = lean_array_push(v___x_3587_, v___x_3586_);
v_attrs_3472_ = v___x_3588_;
v___y_3473_ = v___y_3542_;
v___y_3474_ = v___y_3550_;
v___y_3475_ = v___y_3552_;
v___y_3476_ = v___y_3545_;
v___y_3477_ = v___y_3553_;
v___y_3478_ = v___y_3546_;
goto v___jp_3471_;
}
else
{
lean_object* v___x_3589_; uint8_t v___x_3590_; 
v___x_3589_ = l_Lean_Syntax_getArg(v___x_3573_, v___y_3544_);
lean_dec(v___x_3573_);
v___x_3590_ = l_Lean_Syntax_matchesNull(v___x_3589_, v___y_3544_);
if (v___x_3590_ == 0)
{
lean_object* v_ref_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; lean_object* v___x_3594_; lean_object* v___x_3595_; lean_object* v___x_3596_; lean_object* v___x_3597_; lean_object* v___x_3598_; lean_object* v___x_3599_; lean_object* v___x_3600_; lean_object* v___x_3601_; lean_object* v___x_3602_; 
lean_dec(v_val_3572_);
v_ref_3591_ = lean_ctor_get(v___y_3553_, 5);
v___x_3592_ = l_Lean_SourceInfo_fromRef(v_ref_3591_, v___x_3590_);
v___x_3593_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3594_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3595_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3592_, 3);
v___x_3596_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3596_, 0, v___x_3592_);
lean_ctor_set(v___x_3596_, 1, v___x_3595_);
v___x_3597_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3598_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3598_, 0, v___x_3592_);
lean_ctor_set(v___x_3598_, 1, v___x_3597_);
v___x_3599_ = l_Lean_Syntax_node3(v___x_3592_, v___x_3594_, v___x_3596_, v___y_3549_, v___x_3598_);
v___x_3600_ = l_Lean_Syntax_node1(v___x_3592_, v___x_3593_, v___x_3599_);
v___x_3601_ = lean_mk_empty_array_with_capacity(v___y_3548_);
v___x_3602_ = lean_array_push(v___x_3601_, v___x_3600_);
v_attrs_3472_ = v___x_3602_;
v___y_3473_ = v___y_3542_;
v___y_3474_ = v___y_3550_;
v___y_3475_ = v___y_3552_;
v___y_3476_ = v___y_3545_;
v___y_3477_ = v___y_3553_;
v___y_3478_ = v___y_3546_;
goto v___jp_3471_;
}
else
{
lean_object* v___x_3603_; lean_object* v___x_3604_; uint8_t v___x_3605_; 
v___x_3603_ = lean_unsigned_to_nat(4u);
v___x_3604_ = l_Lean_Syntax_getArg(v___y_3549_, v___x_3603_);
v___x_3605_ = l_Lean_Syntax_matchesNull(v___x_3604_, v___y_3544_);
if (v___x_3605_ == 0)
{
lean_object* v_ref_3606_; lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v___x_3611_; lean_object* v___x_3612_; lean_object* v___x_3613_; lean_object* v___x_3614_; lean_object* v___x_3615_; lean_object* v___x_3616_; lean_object* v___x_3617_; 
lean_dec(v_val_3572_);
v_ref_3606_ = lean_ctor_get(v___y_3553_, 5);
v___x_3607_ = l_Lean_SourceInfo_fromRef(v_ref_3606_, v___x_3605_);
v___x_3608_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3609_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3610_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3607_, 3);
v___x_3611_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3611_, 0, v___x_3607_);
lean_ctor_set(v___x_3611_, 1, v___x_3610_);
v___x_3612_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3613_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3613_, 0, v___x_3607_);
lean_ctor_set(v___x_3613_, 1, v___x_3612_);
v___x_3614_ = l_Lean_Syntax_node3(v___x_3607_, v___x_3609_, v___x_3611_, v___y_3549_, v___x_3613_);
v___x_3615_ = l_Lean_Syntax_node1(v___x_3607_, v___x_3608_, v___x_3614_);
v___x_3616_ = lean_mk_empty_array_with_capacity(v___y_3548_);
v___x_3617_ = lean_array_push(v___x_3616_, v___x_3615_);
v_attrs_3472_ = v___x_3617_;
v___y_3473_ = v___y_3542_;
v___y_3474_ = v___y_3550_;
v___y_3475_ = v___y_3552_;
v___y_3476_ = v___y_3545_;
v___y_3477_ = v___y_3553_;
v___y_3478_ = v___y_3546_;
goto v___jp_3471_;
}
else
{
size_t v_sz_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; lean_object* v___x_3621_; size_t v_sz_3622_; lean_object* v___x_3623_; 
lean_dec(v___y_3549_);
v_sz_3618_ = lean_array_size(v_val_3572_);
lean_inc(v_val_3572_);
v___x_3619_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__6(v_sz_3618_, v___x_3557_, v_val_3572_);
v___x_3620_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__7(v_sz_3618_, v___x_3557_, v_val_3572_);
v___x_3621_ = l_Array_zip___redArg(v___x_3620_, v___x_3619_);
lean_dec_ref(v___x_3619_);
lean_dec_ref(v___x_3620_);
v_sz_3622_ = lean_array_size(v___x_3621_);
v___x_3623_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg(v_sz_3622_, v___x_3557_, v___x_3621_, v___y_3553_);
if (lean_obj_tag(v___x_3623_) == 0)
{
lean_object* v_a_3624_; 
v_a_3624_ = lean_ctor_get(v___x_3623_, 0);
lean_inc(v_a_3624_);
lean_dec_ref_known(v___x_3623_, 1);
v_attrs_3472_ = v_a_3624_;
v___y_3473_ = v___y_3542_;
v___y_3474_ = v___y_3550_;
v___y_3475_ = v___y_3552_;
v___y_3476_ = v___y_3545_;
v___y_3477_ = v___y_3553_;
v___y_3478_ = v___y_3546_;
goto v___jp_3471_;
}
else
{
lean_object* v_a_3625_; lean_object* v___x_3627_; uint8_t v_isShared_3628_; uint8_t v_isSharedCheck_3632_; 
lean_dec(v_a_3470_);
v_a_3625_ = lean_ctor_get(v___x_3623_, 0);
v_isSharedCheck_3632_ = !lean_is_exclusive(v___x_3623_);
if (v_isSharedCheck_3632_ == 0)
{
v___x_3627_ = v___x_3623_;
v_isShared_3628_ = v_isSharedCheck_3632_;
goto v_resetjp_3626_;
}
else
{
lean_inc(v_a_3625_);
lean_dec(v___x_3623_);
v___x_3627_ = lean_box(0);
v_isShared_3628_ = v_isSharedCheck_3632_;
goto v_resetjp_3626_;
}
v_resetjp_3626_:
{
lean_object* v___x_3630_; 
if (v_isShared_3628_ == 0)
{
v___x_3630_ = v___x_3627_;
goto v_reusejp_3629_;
}
else
{
lean_object* v_reuseFailAlloc_3631_; 
v_reuseFailAlloc_3631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3631_, 0, v_a_3625_);
v___x_3630_ = v_reuseFailAlloc_3631_;
goto v_reusejp_3629_;
}
v_reusejp_3629_:
{
return v___x_3630_;
}
}
}
}
}
}
}
}
v___jp_3633_:
{
lean_object* v___x_3640_; lean_object* v___x_3641_; 
v___x_3640_ = lean_unsigned_to_nat(3u);
v___x_3641_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3640_, v___x_3468_, v___y_3634_, v___y_3635_, v___y_3636_, v___y_3637_, v___y_3638_, v___y_3639_);
if (lean_obj_tag(v___x_3641_) == 0)
{
lean_object* v_a_3642_; lean_object* v___x_3643_; lean_object* v___x_3644_; lean_object* v___x_3645_; lean_object* v___x_3646_; uint8_t v___x_3647_; 
v_a_3642_ = lean_ctor_get(v___x_3641_, 0);
lean_inc_n(v_a_3642_, 2);
lean_dec_ref_known(v___x_3641_, 1);
v___x_3643_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__0));
v___x_3644_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__1));
v___x_3645_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_quot___closed__2));
v___x_3646_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__16));
v___x_3647_ = l_Lean_Syntax_isOfKind(v_a_3642_, v___x_3646_);
if (v___x_3647_ == 0)
{
lean_object* v_ref_3648_; lean_object* v___x_3649_; lean_object* v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_3654_; lean_object* v___x_3655_; lean_object* v___x_3656_; lean_object* v___x_3657_; lean_object* v___x_3658_; lean_object* v___x_3659_; lean_object* v___x_3660_; 
v_ref_3648_ = lean_ctor_get(v___y_3638_, 5);
v___x_3649_ = l_Lean_SourceInfo_fromRef(v_ref_3648_, v___x_3647_);
v___x_3650_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3651_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3652_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3649_, 3);
v___x_3653_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3653_, 0, v___x_3649_);
lean_ctor_set(v___x_3653_, 1, v___x_3652_);
v___x_3654_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3655_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3655_, 0, v___x_3649_);
lean_ctor_set(v___x_3655_, 1, v___x_3654_);
v___x_3656_ = l_Lean_Syntax_node3(v___x_3649_, v___x_3651_, v___x_3653_, v_a_3642_, v___x_3655_);
v___x_3657_ = l_Lean_Syntax_node1(v___x_3649_, v___x_3650_, v___x_3656_);
v___x_3658_ = lean_unsigned_to_nat(1u);
v___x_3659_ = lean_mk_empty_array_with_capacity(v___x_3658_);
v___x_3660_ = lean_array_push(v___x_3659_, v___x_3657_);
v_attrs_3472_ = v___x_3660_;
v___y_3473_ = v___y_3634_;
v___y_3474_ = v___y_3635_;
v___y_3475_ = v___y_3636_;
v___y_3476_ = v___y_3637_;
v___y_3477_ = v___y_3638_;
v___y_3478_ = v___y_3639_;
goto v___jp_3471_;
}
else
{
lean_object* v___x_3661_; lean_object* v___x_3662_; lean_object* v___x_3663_; uint8_t v___x_3664_; 
v___x_3661_ = lean_unsigned_to_nat(0u);
v___x_3662_ = lean_unsigned_to_nat(1u);
v___x_3663_ = l_Lean_Syntax_getArg(v_a_3642_, v___x_3662_);
v___x_3664_ = l_Lean_Syntax_matchesNull(v___x_3663_, v___x_3661_);
if (v___x_3664_ == 0)
{
lean_object* v_ref_3665_; lean_object* v___x_3666_; lean_object* v___x_3667_; lean_object* v___x_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; lean_object* v___x_3671_; lean_object* v___x_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; lean_object* v___x_3675_; lean_object* v___x_3676_; 
v_ref_3665_ = lean_ctor_get(v___y_3638_, 5);
v___x_3666_ = l_Lean_SourceInfo_fromRef(v_ref_3665_, v___x_3664_);
v___x_3667_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3668_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3669_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3666_, 3);
v___x_3670_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3670_, 0, v___x_3666_);
lean_ctor_set(v___x_3670_, 1, v___x_3669_);
v___x_3671_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3672_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3672_, 0, v___x_3666_);
lean_ctor_set(v___x_3672_, 1, v___x_3671_);
v___x_3673_ = l_Lean_Syntax_node3(v___x_3666_, v___x_3668_, v___x_3670_, v_a_3642_, v___x_3672_);
v___x_3674_ = l_Lean_Syntax_node1(v___x_3666_, v___x_3667_, v___x_3673_);
v___x_3675_ = lean_mk_empty_array_with_capacity(v___x_3662_);
v___x_3676_ = lean_array_push(v___x_3675_, v___x_3674_);
v_attrs_3472_ = v___x_3676_;
v___y_3473_ = v___y_3634_;
v___y_3474_ = v___y_3635_;
v___y_3475_ = v___y_3636_;
v___y_3476_ = v___y_3637_;
v___y_3477_ = v___y_3638_;
v___y_3478_ = v___y_3639_;
goto v___jp_3471_;
}
else
{
lean_object* v___x_3677_; lean_object* v___x_3678_; uint8_t v___x_3679_; 
v___x_3677_ = l_Lean_Syntax_getArg(v_a_3642_, v___x_3467_);
v___x_3678_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__19));
lean_inc(v___x_3677_);
v___x_3679_ = l_Lean_Syntax_isOfKind(v___x_3677_, v___x_3678_);
if (v___x_3679_ == 0)
{
lean_object* v_ref_3680_; lean_object* v___x_3681_; lean_object* v___x_3682_; lean_object* v___x_3683_; lean_object* v___x_3684_; lean_object* v___x_3685_; lean_object* v___x_3686_; lean_object* v___x_3687_; lean_object* v___x_3688_; lean_object* v___x_3689_; lean_object* v___x_3690_; lean_object* v___x_3691_; 
lean_dec(v___x_3677_);
v_ref_3680_ = lean_ctor_get(v___y_3638_, 5);
v___x_3681_ = l_Lean_SourceInfo_fromRef(v_ref_3680_, v___x_3679_);
v___x_3682_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttr_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_3683_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__3));
v___x_3684_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_3681_, 3);
v___x_3685_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3685_, 0, v___x_3681_);
lean_ctor_set(v___x_3685_, 1, v___x_3684_);
v___x_3686_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3687_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3687_, 0, v___x_3681_);
lean_ctor_set(v___x_3687_, 1, v___x_3686_);
v___x_3688_ = l_Lean_Syntax_node3(v___x_3681_, v___x_3683_, v___x_3685_, v_a_3642_, v___x_3687_);
v___x_3689_ = l_Lean_Syntax_node1(v___x_3681_, v___x_3682_, v___x_3688_);
v___x_3690_ = lean_mk_empty_array_with_capacity(v___x_3662_);
v___x_3691_ = lean_array_push(v___x_3690_, v___x_3689_);
v_attrs_3472_ = v___x_3691_;
v___y_3473_ = v___y_3634_;
v___y_3474_ = v___y_3635_;
v___y_3475_ = v___y_3636_;
v___y_3476_ = v___y_3637_;
v___y_3477_ = v___y_3638_;
v___y_3478_ = v___y_3639_;
goto v___jp_3471_;
}
else
{
lean_object* v___x_3692_; lean_object* v___x_3693_; lean_object* v___x_3694_; lean_object* v___x_3695_; uint8_t v___x_3696_; 
v___x_3692_ = l_Lean_Syntax_getArg(v___x_3677_, v___x_3661_);
lean_dec(v___x_3677_);
v___x_3693_ = l_Lean_Syntax_getArgs(v___x_3692_);
lean_dec(v___x_3692_);
v___x_3694_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___closed__1));
v___x_3695_ = lean_array_get_size(v___x_3693_);
v___x_3696_ = lean_nat_dec_lt(v___x_3661_, v___x_3695_);
if (v___x_3696_ == 0)
{
lean_dec_ref(v___x_3693_);
v___y_3542_ = v___y_3634_;
v___y_3543_ = v___x_3645_;
v___y_3544_ = v___x_3661_;
v___y_3545_ = v___y_3637_;
v___y_3546_ = v___y_3639_;
v___y_3547_ = v___x_3644_;
v___y_3548_ = v___x_3662_;
v___y_3549_ = v_a_3642_;
v___y_3550_ = v___y_3635_;
v___y_3551_ = v___x_3643_;
v___y_3552_ = v___y_3636_;
v___y_3553_ = v___y_3638_;
v___y_3554_ = v___x_3640_;
v___y_3555_ = v___x_3694_;
goto v___jp_3541_;
}
else
{
lean_object* v___x_3697_; lean_object* v___x_3698_; uint8_t v___x_3699_; 
v___x_3697_ = lean_box(v___x_3679_);
v___x_3698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3698_, 0, v___x_3697_);
lean_ctor_set(v___x_3698_, 1, v___x_3694_);
v___x_3699_ = lean_nat_dec_le(v___x_3695_, v___x_3695_);
if (v___x_3699_ == 0)
{
if (v___x_3696_ == 0)
{
lean_dec_ref_known(v___x_3698_, 2);
lean_dec_ref(v___x_3693_);
v___y_3542_ = v___y_3634_;
v___y_3543_ = v___x_3645_;
v___y_3544_ = v___x_3661_;
v___y_3545_ = v___y_3637_;
v___y_3546_ = v___y_3639_;
v___y_3547_ = v___x_3644_;
v___y_3548_ = v___x_3662_;
v___y_3549_ = v_a_3642_;
v___y_3550_ = v___y_3635_;
v___y_3551_ = v___x_3643_;
v___y_3552_ = v___y_3636_;
v___y_3553_ = v___y_3638_;
v___y_3554_ = v___x_3640_;
v___y_3555_ = v___x_3694_;
goto v___jp_3541_;
}
else
{
size_t v___x_3700_; size_t v___x_3701_; lean_object* v___x_3702_; lean_object* v_snd_3703_; 
v___x_3700_ = ((size_t)0ULL);
v___x_3701_ = lean_usize_of_nat(v___x_3695_);
v___x_3702_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9(v___x_3679_, v___x_3693_, v___x_3700_, v___x_3701_, v___x_3698_);
lean_dec_ref(v___x_3693_);
v_snd_3703_ = lean_ctor_get(v___x_3702_, 1);
lean_inc(v_snd_3703_);
lean_dec_ref(v___x_3702_);
v___y_3542_ = v___y_3634_;
v___y_3543_ = v___x_3645_;
v___y_3544_ = v___x_3661_;
v___y_3545_ = v___y_3637_;
v___y_3546_ = v___y_3639_;
v___y_3547_ = v___x_3644_;
v___y_3548_ = v___x_3662_;
v___y_3549_ = v_a_3642_;
v___y_3550_ = v___y_3635_;
v___y_3551_ = v___x_3643_;
v___y_3552_ = v___y_3636_;
v___y_3553_ = v___y_3638_;
v___y_3554_ = v___x_3640_;
v___y_3555_ = v_snd_3703_;
goto v___jp_3541_;
}
}
else
{
size_t v___x_3704_; size_t v___x_3705_; lean_object* v___x_3706_; lean_object* v_snd_3707_; 
v___x_3704_ = ((size_t)0ULL);
v___x_3705_ = lean_usize_of_nat(v___x_3695_);
v___x_3706_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__9(v___x_3679_, v___x_3693_, v___x_3704_, v___x_3705_, v___x_3698_);
lean_dec_ref(v___x_3693_);
v_snd_3707_ = lean_ctor_get(v___x_3706_, 1);
lean_inc(v_snd_3707_);
lean_dec_ref(v___x_3706_);
v___y_3542_ = v___y_3634_;
v___y_3543_ = v___x_3645_;
v___y_3544_ = v___x_3661_;
v___y_3545_ = v___y_3637_;
v___y_3546_ = v___y_3639_;
v___y_3547_ = v___x_3644_;
v___y_3548_ = v___x_3662_;
v___y_3549_ = v_a_3642_;
v___y_3550_ = v___y_3635_;
v___y_3551_ = v___x_3643_;
v___y_3552_ = v___y_3636_;
v___y_3553_ = v___y_3638_;
v___y_3554_ = v___x_3640_;
v___y_3555_ = v_snd_3707_;
goto v___jp_3541_;
}
}
}
}
}
}
else
{
lean_dec(v_a_3470_);
return v___x_3641_;
}
}
}
else
{
return v___x_3469_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_3718_; lean_object* v___x_3720_; uint8_t v_isShared_3721_; uint8_t v_isSharedCheck_3725_; 
v_a_3718_ = lean_ctor_get(v___x_3446_, 0);
v_isSharedCheck_3725_ = !lean_is_exclusive(v___x_3446_);
if (v_isSharedCheck_3725_ == 0)
{
v___x_3720_ = v___x_3446_;
v_isShared_3721_ = v_isSharedCheck_3725_;
goto v_resetjp_3719_;
}
else
{
lean_inc(v_a_3718_);
lean_dec(v___x_3446_);
v___x_3720_ = lean_box(0);
v_isShared_3721_ = v_isSharedCheck_3725_;
goto v_resetjp_3719_;
}
v_resetjp_3719_:
{
lean_object* v___x_3723_; 
if (v_isShared_3721_ == 0)
{
v___x_3723_ = v___x_3720_;
goto v_reusejp_3722_;
}
else
{
lean_object* v_reuseFailAlloc_3724_; 
v_reuseFailAlloc_3724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3724_, 0, v_a_3718_);
v___x_3723_ = v_reuseFailAlloc_3724_;
goto v_reusejp_3722_;
}
v_reusejp_3722_:
{
return v___x_3723_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27(lean_object* v_a_3743_, lean_object* v_a_3744_, lean_object* v_a_3745_, lean_object* v_a_3746_, lean_object* v_a_3747_, lean_object* v_a_3748_){
_start:
{
lean_object* v___x_3750_; 
v___x_3750_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v_a_3743_);
if (lean_obj_tag(v___x_3750_) == 0)
{
lean_object* v_a_3751_; lean_object* v___x_3752_; uint8_t v___x_3753_; 
v_a_3751_ = lean_ctor_get(v___x_3750_, 0);
lean_inc(v_a_3751_);
lean_dec_ref_known(v___x_3750_, 1);
v___x_3752_ = l_Lean_Expr_cleanupAnnotations(v_a_3751_);
v___x_3753_ = l_Lean_Expr_isApp(v___x_3752_);
if (v___x_3753_ == 0)
{
lean_object* v___x_3754_; 
lean_dec_ref(v___x_3752_);
v___x_3754_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3754_;
}
else
{
lean_object* v___x_3755_; uint8_t v___x_3756_; 
v___x_3755_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3752_);
v___x_3756_ = l_Lean_Expr_isApp(v___x_3755_);
if (v___x_3756_ == 0)
{
lean_object* v___x_3757_; 
lean_dec_ref(v___x_3755_);
v___x_3757_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3757_;
}
else
{
lean_object* v___x_3758_; uint8_t v___x_3759_; 
v___x_3758_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3755_);
v___x_3759_ = l_Lean_Expr_isApp(v___x_3758_);
if (v___x_3759_ == 0)
{
lean_object* v___x_3760_; 
lean_dec_ref(v___x_3758_);
v___x_3760_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3760_;
}
else
{
lean_object* v_arg_3761_; lean_object* v___x_3762_; lean_object* v___x_3763_; uint8_t v___x_3764_; 
v_arg_3761_ = lean_ctor_get(v___x_3758_, 1);
lean_inc_ref(v_arg_3761_);
v___x_3762_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3758_);
v___x_3763_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3));
v___x_3764_ = l_Lean_Expr_isConstOf(v___x_3762_, v___x_3763_);
lean_dec_ref(v___x_3762_);
if (v___x_3764_ == 0)
{
lean_object* v___x_3765_; 
lean_dec_ref(v_arg_3761_);
v___x_3765_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3765_;
}
else
{
if (lean_obj_tag(v_arg_3761_) == 9)
{
lean_object* v_a_3766_; 
v_a_3766_ = lean_ctor_get(v_arg_3761_, 0);
lean_inc_ref(v_a_3766_);
lean_dec_ref_known(v_arg_3761_, 1);
if (lean_obj_tag(v_a_3766_) == 1)
{
lean_object* v_val_3767_; lean_object* v___x_3768_; lean_object* v___x_3769_; lean_object* v___x_3770_; lean_object* v___x_3771_; lean_object* v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; 
v_val_3767_ = lean_ctor_get(v_a_3766_, 0);
lean_inc_ref(v_val_3767_);
lean_dec_ref_known(v_a_3766_, 1);
v___x_3768_ = lean_unsigned_to_nat(0u);
v___x_3769_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__0));
v___x_3770_ = lean_box(0);
v___x_3771_ = l_Lean_Name_str___override(v___x_3770_, v_val_3767_);
v___x_3772_ = l_Lean_mkIdent(v___x_3771_);
v___x_3773_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___boxed), 9, 2);
lean_closure_set(v___x_3773_, 0, v___x_3769_);
lean_closure_set(v___x_3773_, 1, v___x_3772_);
v___x_3774_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3768_, v___x_3773_, v_a_3743_, v_a_3744_, v_a_3745_, v_a_3746_, v_a_3747_, v_a_3748_);
if (lean_obj_tag(v___x_3774_) == 0)
{
lean_object* v_a_3775_; lean_object* v___x_3776_; lean_object* v___f_3777_; lean_object* v___x_3778_; 
v_a_3775_ = lean_ctor_get(v___x_3774_, 0);
lean_inc(v_a_3775_);
lean_dec_ref_known(v___x_3774_, 1);
v___x_3776_ = lean_unsigned_to_nat(1u);
v___f_3777_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___closed__4));
v___x_3778_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v___x_3776_, v___f_3777_, v_a_3743_, v_a_3744_, v_a_3745_, v_a_3746_, v_a_3747_, v_a_3748_);
if (lean_obj_tag(v___x_3778_) == 0)
{
lean_object* v_a_3779_; lean_object* v___x_3780_; lean_object* v___x_3781_; 
v_a_3779_ = lean_ctor_get(v___x_3778_, 0);
lean_inc(v_a_3779_);
lean_dec_ref_known(v___x_3778_, 1);
v___x_3780_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___boxed), 7, 0);
v___x_3781_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg(v___x_3780_, v_a_3743_, v_a_3744_, v_a_3745_, v_a_3746_, v_a_3747_, v_a_3748_);
if (lean_obj_tag(v___x_3781_) == 0)
{
lean_object* v_a_3782_; lean_object* v___x_3784_; uint8_t v_isShared_3785_; uint8_t v_isSharedCheck_3831_; 
v_a_3782_ = lean_ctor_get(v___x_3781_, 0);
v_isSharedCheck_3831_ = !lean_is_exclusive(v___x_3781_);
if (v_isSharedCheck_3831_ == 0)
{
v___x_3784_ = v___x_3781_;
v_isShared_3785_ = v_isSharedCheck_3831_;
goto v_resetjp_3783_;
}
else
{
lean_inc(v_a_3782_);
lean_dec(v___x_3781_);
v___x_3784_ = lean_box(0);
v_isShared_3785_ = v_isSharedCheck_3831_;
goto v_resetjp_3783_;
}
v_resetjp_3783_:
{
lean_object* v___x_3786_; uint8_t v___x_3787_; 
v___x_3786_ = lean_array_get_size(v_a_3782_);
v___x_3787_ = lean_nat_dec_eq(v___x_3786_, v___x_3768_);
if (v___x_3787_ == 0)
{
lean_object* v_ref_3788_; lean_object* v___x_3789_; lean_object* v___x_3790_; lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v___x_3793_; lean_object* v___x_3794_; size_t v_sz_3795_; size_t v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; size_t v_sz_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3810_; 
v_ref_3788_ = lean_ctor_get(v_a_3747_, 5);
v___x_3789_ = l_Lean_SourceInfo_fromRef(v_ref_3788_, v___x_3787_);
v___x_3790_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__1));
v___x_3791_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2));
lean_inc_n(v___x_3789_, 5);
v___x_3792_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3792_, 0, v___x_3789_);
lean_ctor_set(v___x_3792_, 1, v___x_3791_);
v___x_3793_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_3794_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v_sz_3795_ = lean_array_size(v_a_3779_);
v___x_3796_ = ((size_t)0ULL);
v___x_3797_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3795_, v___x_3796_, v_a_3779_);
v___x_3798_ = l_Array_append___redArg(v___x_3794_, v___x_3797_);
lean_dec_ref(v___x_3797_);
v___x_3799_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3799_, 0, v___x_3789_);
lean_ctor_set(v___x_3799_, 1, v___x_3793_);
lean_ctor_set(v___x_3799_, 2, v___x_3798_);
v___x_3800_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__2));
v___x_3801_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3801_, 0, v___x_3789_);
lean_ctor_set(v___x_3801_, 1, v___x_3800_);
v_sz_3802_ = lean_array_size(v_a_3782_);
v___x_3803_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3802_, v___x_3796_, v_a_3782_);
v___x_3804_ = l_Array_append___redArg(v___x_3794_, v___x_3803_);
lean_dec_ref(v___x_3803_);
v___x_3805_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3805_, 0, v___x_3789_);
lean_ctor_set(v___x_3805_, 1, v___x_3793_);
lean_ctor_set(v___x_3805_, 2, v___x_3804_);
v___x_3806_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x3e___x3c_x2f___x3e___closed__7));
v___x_3807_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3807_, 0, v___x_3789_);
lean_ctor_set(v___x_3807_, 1, v___x_3806_);
lean_inc_ref(v___x_3801_);
lean_inc(v_a_3775_);
v___x_3808_ = l_Lean_Syntax_node8(v___x_3789_, v___x_3790_, v___x_3792_, v_a_3775_, v___x_3799_, v___x_3801_, v___x_3805_, v___x_3807_, v_a_3775_, v___x_3801_);
if (v_isShared_3785_ == 0)
{
lean_ctor_set(v___x_3784_, 0, v___x_3808_);
v___x_3810_ = v___x_3784_;
goto v_reusejp_3809_;
}
else
{
lean_object* v_reuseFailAlloc_3811_; 
v_reuseFailAlloc_3811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3811_, 0, v___x_3808_);
v___x_3810_ = v_reuseFailAlloc_3811_;
goto v_reusejp_3809_;
}
v_reusejp_3809_:
{
return v___x_3810_;
}
}
else
{
lean_object* v_ref_3812_; uint8_t v___x_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; lean_object* v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3819_; size_t v_sz_3820_; size_t v___x_3821_; lean_object* v___x_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___x_3825_; lean_object* v___x_3826_; lean_object* v___x_3827_; lean_object* v___x_3829_; 
lean_dec(v_a_3782_);
v_ref_3812_ = lean_ctor_get(v_a_3747_, 5);
v___x_3813_ = 0;
v___x_3814_ = l_Lean_SourceInfo_fromRef(v_ref_3812_, v___x_3813_);
v___x_3815_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__1));
v___x_3816_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__2));
lean_inc_n(v___x_3814_, 3);
v___x_3817_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3817_, 0, v___x_3814_);
lean_ctor_set(v___x_3817_, 1, v___x_3816_);
v___x_3818_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_transformTag_spec__2___closed__30));
v___x_3819_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3, &lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3_once, _init_lp_proofwidgets_ProofWidgets_Jsx_transformTag___lam__0___closed__3);
v_sz_3820_ = lean_array_size(v_a_3779_);
v___x_3821_ = ((size_t)0ULL);
v___x_3822_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__2(v_sz_3820_, v___x_3821_, v_a_3779_);
v___x_3823_ = l_Array_append___redArg(v___x_3819_, v___x_3822_);
lean_dec_ref(v___x_3822_);
v___x_3824_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3824_, 0, v___x_3814_);
lean_ctor_set(v___x_3824_, 1, v___x_3818_);
lean_ctor_set(v___x_3824_, 2, v___x_3823_);
v___x_3825_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxElement_x3c_____x2f_x3e___closed__9));
v___x_3826_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3826_, 0, v___x_3814_);
lean_ctor_set(v___x_3826_, 1, v___x_3825_);
v___x_3827_ = l_Lean_Syntax_node4(v___x_3814_, v___x_3815_, v___x_3817_, v_a_3775_, v___x_3824_, v___x_3826_);
if (v_isShared_3785_ == 0)
{
lean_ctor_set(v___x_3784_, 0, v___x_3827_);
v___x_3829_ = v___x_3784_;
goto v_reusejp_3828_;
}
else
{
lean_object* v_reuseFailAlloc_3830_; 
v_reuseFailAlloc_3830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3830_, 0, v___x_3827_);
v___x_3829_ = v_reuseFailAlloc_3830_;
goto v_reusejp_3828_;
}
v_reusejp_3828_:
{
return v___x_3829_;
}
}
}
}
else
{
lean_object* v_a_3832_; lean_object* v___x_3834_; uint8_t v_isShared_3835_; uint8_t v_isSharedCheck_3839_; 
lean_dec(v_a_3779_);
lean_dec(v_a_3775_);
v_a_3832_ = lean_ctor_get(v___x_3781_, 0);
v_isSharedCheck_3839_ = !lean_is_exclusive(v___x_3781_);
if (v_isSharedCheck_3839_ == 0)
{
v___x_3834_ = v___x_3781_;
v_isShared_3835_ = v_isSharedCheck_3839_;
goto v_resetjp_3833_;
}
else
{
lean_inc(v_a_3832_);
lean_dec(v___x_3781_);
v___x_3834_ = lean_box(0);
v_isShared_3835_ = v_isSharedCheck_3839_;
goto v_resetjp_3833_;
}
v_resetjp_3833_:
{
lean_object* v___x_3837_; 
if (v_isShared_3835_ == 0)
{
v___x_3837_ = v___x_3834_;
goto v_reusejp_3836_;
}
else
{
lean_object* v_reuseFailAlloc_3838_; 
v_reuseFailAlloc_3838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3838_, 0, v_a_3832_);
v___x_3837_ = v_reuseFailAlloc_3838_;
goto v_reusejp_3836_;
}
v_reusejp_3836_:
{
return v___x_3837_;
}
}
}
}
else
{
lean_object* v_a_3840_; lean_object* v___x_3842_; uint8_t v_isShared_3843_; uint8_t v_isSharedCheck_3847_; 
lean_dec(v_a_3775_);
v_a_3840_ = lean_ctor_get(v___x_3778_, 0);
v_isSharedCheck_3847_ = !lean_is_exclusive(v___x_3778_);
if (v_isSharedCheck_3847_ == 0)
{
v___x_3842_ = v___x_3778_;
v_isShared_3843_ = v_isSharedCheck_3847_;
goto v_resetjp_3841_;
}
else
{
lean_inc(v_a_3840_);
lean_dec(v___x_3778_);
v___x_3842_ = lean_box(0);
v_isShared_3843_ = v_isSharedCheck_3847_;
goto v_resetjp_3841_;
}
v_resetjp_3841_:
{
lean_object* v___x_3845_; 
if (v_isShared_3843_ == 0)
{
v___x_3845_ = v___x_3842_;
goto v_reusejp_3844_;
}
else
{
lean_object* v_reuseFailAlloc_3846_; 
v_reuseFailAlloc_3846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3846_, 0, v_a_3840_);
v___x_3845_ = v_reuseFailAlloc_3846_;
goto v_reusejp_3844_;
}
v_reusejp_3844_:
{
return v___x_3845_;
}
}
}
}
else
{
return v___x_3774_;
}
}
else
{
lean_object* v___x_3848_; 
lean_dec_ref(v_a_3766_);
v___x_3848_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3848_;
}
}
else
{
lean_object* v___x_3849_; 
lean_dec_ref(v_arg_3761_);
v___x_3849_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_3849_;
}
}
}
}
}
}
else
{
lean_object* v_a_3850_; lean_object* v___x_3852_; uint8_t v_isShared_3853_; uint8_t v_isSharedCheck_3857_; 
v_a_3850_ = lean_ctor_get(v___x_3750_, 0);
v_isSharedCheck_3857_ = !lean_is_exclusive(v___x_3750_);
if (v_isSharedCheck_3857_ == 0)
{
v___x_3852_ = v___x_3750_;
v_isShared_3853_ = v_isSharedCheck_3857_;
goto v_resetjp_3851_;
}
else
{
lean_inc(v_a_3850_);
lean_dec(v___x_3750_);
v___x_3852_ = lean_box(0);
v_isShared_3853_ = v_isSharedCheck_3857_;
goto v_resetjp_3851_;
}
v_resetjp_3851_:
{
lean_object* v___x_3855_; 
if (v_isShared_3853_ == 0)
{
v___x_3855_ = v___x_3852_;
goto v_reusejp_3854_;
}
else
{
lean_object* v_reuseFailAlloc_3856_; 
v_reuseFailAlloc_3856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3856_, 0, v_a_3850_);
v___x_3855_ = v_reuseFailAlloc_3856_;
goto v_reusejp_3854_;
}
v_reusejp_3854_:
{
return v___x_3855_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1(lean_object* v___f_3858_, lean_object* v___y_3859_, lean_object* v___y_3860_, lean_object* v___y_3861_, lean_object* v___y_3862_, lean_object* v___y_3863_, lean_object* v___y_3864_){
_start:
{
lean_object* v_a_3867_; lean_object* v___y_3885_; uint8_t v___y_3886_; lean_object* v___y_3914_; lean_object* v_a_3915_; lean_object* v___y_3919_; lean_object* v___x_3922_; 
v___x_3922_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00ProofWidgets_Jsx_delabHtmlText_spec__0___redArg(v___y_3859_);
if (lean_obj_tag(v___x_3922_) == 0)
{
lean_object* v_a_3923_; lean_object* v___x_3924_; 
v_a_3923_ = lean_ctor_get(v___x_3922_, 0);
lean_inc(v_a_3923_);
lean_dec_ref_known(v___x_3922_, 1);
v___x_3924_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_3923_, v___y_3862_);
if (lean_obj_tag(v___x_3924_) == 0)
{
lean_object* v_a_3925_; lean_object* v___x_3926_; uint8_t v___x_3927_; 
v_a_3925_ = lean_ctor_get(v___x_3924_, 0);
lean_inc(v_a_3925_);
lean_dec_ref_known(v___x_3924_, 1);
v___x_3926_ = l_Lean_Expr_cleanupAnnotations(v_a_3925_);
v___x_3927_ = l_Lean_Expr_isApp(v___x_3926_);
if (v___x_3927_ == 0)
{
lean_object* v___x_3928_; lean_object* v___x_3929_; 
lean_dec_ref(v___x_3926_);
v___x_3928_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3929_ = lean_apply_8(v___f_3858_, v___x_3928_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3929_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3930_; lean_object* v___x_3931_; uint8_t v___x_3932_; 
v___x_3930_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3926_);
v___x_3931_ = ((lean_object*)(lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00ProofWidgets_Jsx_transformTag_spec__6___closed__8));
v___x_3932_ = l_Lean_Expr_isConstOf(v___x_3930_, v___x_3931_);
if (v___x_3932_ == 0)
{
uint8_t v___x_3933_; 
v___x_3933_ = l_Lean_Expr_isApp(v___x_3930_);
if (v___x_3933_ == 0)
{
lean_object* v___x_3934_; lean_object* v___x_3935_; 
lean_dec_ref(v___x_3930_);
v___x_3934_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3935_ = lean_apply_8(v___f_3858_, v___x_3934_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3935_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3936_; uint8_t v___x_3937_; 
v___x_3936_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3930_);
v___x_3937_ = l_Lean_Expr_isApp(v___x_3936_);
if (v___x_3937_ == 0)
{
lean_object* v___x_3938_; lean_object* v___x_3939_; 
lean_dec_ref(v___x_3936_);
v___x_3938_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3939_ = lean_apply_8(v___f_3858_, v___x_3938_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3939_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3940_; lean_object* v___x_3941_; uint8_t v___x_3942_; 
v___x_3940_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3936_);
v___x_3941_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__3));
v___x_3942_ = l_Lean_Expr_isConstOf(v___x_3940_, v___x_3941_);
if (v___x_3942_ == 0)
{
uint8_t v___x_3943_; 
v___x_3943_ = l_Lean_Expr_isApp(v___x_3940_);
if (v___x_3943_ == 0)
{
lean_object* v___x_3944_; lean_object* v___x_3945_; 
lean_dec_ref(v___x_3940_);
v___x_3944_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3945_ = lean_apply_8(v___f_3858_, v___x_3944_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3945_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3946_; uint8_t v___x_3947_; 
v___x_3946_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3940_);
v___x_3947_ = l_Lean_Expr_isApp(v___x_3946_);
if (v___x_3947_ == 0)
{
lean_object* v___x_3948_; lean_object* v___x_3949_; 
lean_dec_ref(v___x_3946_);
v___x_3948_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3949_ = lean_apply_8(v___f_3858_, v___x_3948_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3949_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3950_; lean_object* v___x_3951_; uint8_t v___x_3952_; 
v___x_3950_ = l_Lean_Expr_appFnCleanup___redArg(v___x_3946_);
v___x_3951_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_transformTag___closed__12));
v___x_3952_ = l_Lean_Expr_isConstOf(v___x_3950_, v___x_3951_);
lean_dec_ref(v___x_3950_);
if (v___x_3952_ == 0)
{
lean_object* v___x_3953_; lean_object* v___x_3954_; 
v___x_3953_ = lean_box(0);
lean_inc(v___y_3864_);
lean_inc_ref(v___y_3863_);
lean_inc(v___y_3862_);
lean_inc_ref(v___y_3861_);
lean_inc(v___y_3860_);
lean_inc_ref(v___y_3859_);
v___x_3954_ = lean_apply_8(v___f_3858_, v___x_3953_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, lean_box(0));
v___y_3919_ = v___x_3954_;
goto v___jp_3918_;
}
else
{
lean_object* v___x_3955_; 
lean_dec_ref(v___f_3858_);
v___x_3955_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27(v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3955_) == 0)
{
lean_object* v_a_3956_; lean_object* v___x_3958_; uint8_t v_isShared_3959_; uint8_t v_isSharedCheck_3967_; 
v_a_3956_ = lean_ctor_get(v___x_3955_, 0);
v_isSharedCheck_3967_ = !lean_is_exclusive(v___x_3955_);
if (v_isSharedCheck_3967_ == 0)
{
v___x_3958_ = v___x_3955_;
v_isShared_3959_ = v_isSharedCheck_3967_;
goto v_resetjp_3957_;
}
else
{
lean_inc(v_a_3956_);
lean_dec(v___x_3955_);
v___x_3958_ = lean_box(0);
v_isShared_3959_ = v_isSharedCheck_3967_;
goto v_resetjp_3957_;
}
v_resetjp_3957_:
{
lean_object* v_ref_3960_; lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; lean_object* v___x_3965_; 
v_ref_3960_ = lean_ctor_get(v___y_3863_, 5);
v___x_3961_ = l_Lean_SourceInfo_fromRef(v_ref_3960_, v___x_3942_);
v___x_3962_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1));
v___x_3963_ = l_Lean_Syntax_node1(v___x_3961_, v___x_3962_, v_a_3956_);
if (v_isShared_3959_ == 0)
{
lean_ctor_set(v___x_3958_, 0, v___x_3963_);
v___x_3965_ = v___x_3958_;
goto v_reusejp_3964_;
}
else
{
lean_object* v_reuseFailAlloc_3966_; 
v_reuseFailAlloc_3966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3966_, 0, v___x_3963_);
v___x_3965_ = v_reuseFailAlloc_3966_;
goto v_reusejp_3964_;
}
v_reusejp_3964_:
{
return v___x_3965_;
}
}
}
else
{
lean_object* v_a_3968_; lean_object* v___x_3970_; uint8_t v_isShared_3971_; uint8_t v_isSharedCheck_3975_; 
v_a_3968_ = lean_ctor_get(v___x_3955_, 0);
v_isSharedCheck_3975_ = !lean_is_exclusive(v___x_3955_);
if (v_isSharedCheck_3975_ == 0)
{
v___x_3970_ = v___x_3955_;
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
else
{
lean_inc(v_a_3968_);
lean_dec(v___x_3955_);
v___x_3970_ = lean_box(0);
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
v_resetjp_3969_:
{
lean_object* v___x_3973_; 
lean_inc(v_a_3968_);
if (v_isShared_3971_ == 0)
{
v___x_3973_ = v___x_3970_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_3974_; 
v_reuseFailAlloc_3974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3974_, 0, v_a_3968_);
v___x_3973_ = v_reuseFailAlloc_3974_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
v___y_3914_ = v___x_3973_;
v_a_3915_ = v_a_3968_;
goto v___jp_3913_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3976_; 
lean_dec_ref(v___x_3940_);
lean_dec_ref(v___f_3858_);
v___x_3976_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27(v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3976_) == 0)
{
lean_object* v_a_3977_; lean_object* v___x_3979_; uint8_t v_isShared_3980_; uint8_t v_isSharedCheck_3988_; 
v_a_3977_ = lean_ctor_get(v___x_3976_, 0);
v_isSharedCheck_3988_ = !lean_is_exclusive(v___x_3976_);
if (v_isSharedCheck_3988_ == 0)
{
v___x_3979_ = v___x_3976_;
v_isShared_3980_ = v_isSharedCheck_3988_;
goto v_resetjp_3978_;
}
else
{
lean_inc(v_a_3977_);
lean_dec(v___x_3976_);
v___x_3979_ = lean_box(0);
v_isShared_3980_ = v_isSharedCheck_3988_;
goto v_resetjp_3978_;
}
v_resetjp_3978_:
{
lean_object* v_ref_3981_; lean_object* v___x_3982_; lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v___x_3986_; 
v_ref_3981_ = lean_ctor_get(v___y_3863_, 5);
v___x_3982_ = l_Lean_SourceInfo_fromRef(v_ref_3981_, v___x_3932_);
v___x_3983_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild____1___closed__1));
v___x_3984_ = l_Lean_Syntax_node1(v___x_3982_, v___x_3983_, v_a_3977_);
if (v_isShared_3980_ == 0)
{
lean_ctor_set(v___x_3979_, 0, v___x_3984_);
v___x_3986_ = v___x_3979_;
goto v_reusejp_3985_;
}
else
{
lean_object* v_reuseFailAlloc_3987_; 
v_reuseFailAlloc_3987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3987_, 0, v___x_3984_);
v___x_3986_ = v_reuseFailAlloc_3987_;
goto v_reusejp_3985_;
}
v_reusejp_3985_:
{
return v___x_3986_;
}
}
}
else
{
lean_object* v_a_3989_; lean_object* v___x_3991_; uint8_t v_isShared_3992_; uint8_t v_isSharedCheck_3996_; 
v_a_3989_ = lean_ctor_get(v___x_3976_, 0);
v_isSharedCheck_3996_ = !lean_is_exclusive(v___x_3976_);
if (v_isSharedCheck_3996_ == 0)
{
v___x_3991_ = v___x_3976_;
v_isShared_3992_ = v_isSharedCheck_3996_;
goto v_resetjp_3990_;
}
else
{
lean_inc(v_a_3989_);
lean_dec(v___x_3976_);
v___x_3991_ = lean_box(0);
v_isShared_3992_ = v_isSharedCheck_3996_;
goto v_resetjp_3990_;
}
v_resetjp_3990_:
{
lean_object* v___x_3994_; 
lean_inc(v_a_3989_);
if (v_isShared_3992_ == 0)
{
v___x_3994_ = v___x_3991_;
goto v_reusejp_3993_;
}
else
{
lean_object* v_reuseFailAlloc_3995_; 
v_reuseFailAlloc_3995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3995_, 0, v_a_3989_);
v___x_3994_ = v_reuseFailAlloc_3995_;
goto v_reusejp_3993_;
}
v_reusejp_3993_:
{
v___y_3914_ = v___x_3994_;
v_a_3915_ = v_a_3989_;
goto v___jp_3913_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3997_; 
lean_dec_ref(v___x_3930_);
lean_dec_ref(v___f_3858_);
v___x_3997_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlText(v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3997_) == 0)
{
lean_object* v_a_3998_; lean_object* v___x_4000_; uint8_t v_isShared_4001_; uint8_t v_isSharedCheck_4010_; 
v_a_3998_ = lean_ctor_get(v___x_3997_, 0);
v_isSharedCheck_4010_ = !lean_is_exclusive(v___x_3997_);
if (v_isSharedCheck_4010_ == 0)
{
v___x_4000_ = v___x_3997_;
v_isShared_4001_ = v_isSharedCheck_4010_;
goto v_resetjp_3999_;
}
else
{
lean_inc(v_a_3998_);
lean_dec(v___x_3997_);
v___x_4000_ = lean_box(0);
v_isShared_4001_ = v_isSharedCheck_4010_;
goto v_resetjp_3999_;
}
v_resetjp_3999_:
{
lean_object* v_ref_4002_; uint8_t v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4008_; 
v_ref_4002_ = lean_ctor_get(v___y_3863_, 5);
v___x_4003_ = 0;
v___x_4004_ = l_Lean_SourceInfo_fromRef(v_ref_4002_, v___x_4003_);
v___x_4005_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild___00__closed__1));
v___x_4006_ = l_Lean_Syntax_node1(v___x_4004_, v___x_4005_, v_a_3998_);
if (v_isShared_4001_ == 0)
{
lean_ctor_set(v___x_4000_, 0, v___x_4006_);
v___x_4008_ = v___x_4000_;
goto v_reusejp_4007_;
}
else
{
lean_object* v_reuseFailAlloc_4009_; 
v_reuseFailAlloc_4009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4009_, 0, v___x_4006_);
v___x_4008_ = v_reuseFailAlloc_4009_;
goto v_reusejp_4007_;
}
v_reusejp_4007_:
{
return v___x_4008_;
}
}
}
else
{
lean_object* v_a_4011_; lean_object* v___x_4013_; uint8_t v_isShared_4014_; uint8_t v_isSharedCheck_4018_; 
v_a_4011_ = lean_ctor_get(v___x_3997_, 0);
v_isSharedCheck_4018_ = !lean_is_exclusive(v___x_3997_);
if (v_isSharedCheck_4018_ == 0)
{
v___x_4013_ = v___x_3997_;
v_isShared_4014_ = v_isSharedCheck_4018_;
goto v_resetjp_4012_;
}
else
{
lean_inc(v_a_4011_);
lean_dec(v___x_3997_);
v___x_4013_ = lean_box(0);
v_isShared_4014_ = v_isSharedCheck_4018_;
goto v_resetjp_4012_;
}
v_resetjp_4012_:
{
lean_object* v___x_4016_; 
lean_inc(v_a_4011_);
if (v_isShared_4014_ == 0)
{
v___x_4016_ = v___x_4013_;
goto v_reusejp_4015_;
}
else
{
lean_object* v_reuseFailAlloc_4017_; 
v_reuseFailAlloc_4017_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4017_, 0, v_a_4011_);
v___x_4016_ = v_reuseFailAlloc_4017_;
goto v_reusejp_4015_;
}
v_reusejp_4015_:
{
v___y_3914_ = v___x_4016_;
v_a_3915_ = v_a_4011_;
goto v___jp_3913_;
}
}
}
}
}
}
else
{
lean_object* v_a_4019_; lean_object* v___x_4021_; uint8_t v_isShared_4022_; uint8_t v_isSharedCheck_4026_; 
lean_dec_ref(v___f_3858_);
v_a_4019_ = lean_ctor_get(v___x_3924_, 0);
v_isSharedCheck_4026_ = !lean_is_exclusive(v___x_3924_);
if (v_isSharedCheck_4026_ == 0)
{
v___x_4021_ = v___x_3924_;
v_isShared_4022_ = v_isSharedCheck_4026_;
goto v_resetjp_4020_;
}
else
{
lean_inc(v_a_4019_);
lean_dec(v___x_3924_);
v___x_4021_ = lean_box(0);
v_isShared_4022_ = v_isSharedCheck_4026_;
goto v_resetjp_4020_;
}
v_resetjp_4020_:
{
lean_object* v___x_4024_; 
lean_inc(v_a_4019_);
if (v_isShared_4022_ == 0)
{
v___x_4024_ = v___x_4021_;
goto v_reusejp_4023_;
}
else
{
lean_object* v_reuseFailAlloc_4025_; 
v_reuseFailAlloc_4025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4025_, 0, v_a_4019_);
v___x_4024_ = v_reuseFailAlloc_4025_;
goto v_reusejp_4023_;
}
v_reusejp_4023_:
{
v___y_3914_ = v___x_4024_;
v_a_3915_ = v_a_4019_;
goto v___jp_3913_;
}
}
}
}
else
{
lean_object* v_a_4027_; lean_object* v___x_4029_; uint8_t v_isShared_4030_; uint8_t v_isSharedCheck_4034_; 
lean_dec_ref(v___f_3858_);
v_a_4027_ = lean_ctor_get(v___x_3922_, 0);
v_isSharedCheck_4034_ = !lean_is_exclusive(v___x_3922_);
if (v_isSharedCheck_4034_ == 0)
{
v___x_4029_ = v___x_3922_;
v_isShared_4030_ = v_isSharedCheck_4034_;
goto v_resetjp_4028_;
}
else
{
lean_inc(v_a_4027_);
lean_dec(v___x_3922_);
v___x_4029_ = lean_box(0);
v_isShared_4030_ = v_isSharedCheck_4034_;
goto v_resetjp_4028_;
}
v_resetjp_4028_:
{
lean_object* v___x_4032_; 
lean_inc(v_a_4027_);
if (v_isShared_4030_ == 0)
{
v___x_4032_ = v___x_4029_;
goto v_reusejp_4031_;
}
else
{
lean_object* v_reuseFailAlloc_4033_; 
v_reuseFailAlloc_4033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4033_, 0, v_a_4027_);
v___x_4032_ = v_reuseFailAlloc_4033_;
goto v_reusejp_4031_;
}
v_reusejp_4031_:
{
v___y_3914_ = v___x_4032_;
v_a_3915_ = v_a_4027_;
goto v___jp_3913_;
}
}
}
v___jp_3866_:
{
if (lean_obj_tag(v_a_3867_) == 0)
{
lean_object* v_a_3868_; lean_object* v___x_3870_; uint8_t v_isShared_3871_; uint8_t v_isSharedCheck_3875_; 
v_a_3868_ = lean_ctor_get(v_a_3867_, 0);
v_isSharedCheck_3875_ = !lean_is_exclusive(v_a_3867_);
if (v_isSharedCheck_3875_ == 0)
{
v___x_3870_ = v_a_3867_;
v_isShared_3871_ = v_isSharedCheck_3875_;
goto v_resetjp_3869_;
}
else
{
lean_inc(v_a_3868_);
lean_dec(v_a_3867_);
v___x_3870_ = lean_box(0);
v_isShared_3871_ = v_isSharedCheck_3875_;
goto v_resetjp_3869_;
}
v_resetjp_3869_:
{
lean_object* v___x_3873_; 
if (v_isShared_3871_ == 0)
{
v___x_3873_ = v___x_3870_;
goto v_reusejp_3872_;
}
else
{
lean_object* v_reuseFailAlloc_3874_; 
v_reuseFailAlloc_3874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3874_, 0, v_a_3868_);
v___x_3873_ = v_reuseFailAlloc_3874_;
goto v_reusejp_3872_;
}
v_reusejp_3872_:
{
return v___x_3873_;
}
}
}
else
{
lean_object* v_a_3876_; lean_object* v___x_3878_; uint8_t v_isShared_3879_; uint8_t v_isSharedCheck_3883_; 
v_a_3876_ = lean_ctor_get(v_a_3867_, 0);
v_isSharedCheck_3883_ = !lean_is_exclusive(v_a_3867_);
if (v_isSharedCheck_3883_ == 0)
{
v___x_3878_ = v_a_3867_;
v_isShared_3879_ = v_isSharedCheck_3883_;
goto v_resetjp_3877_;
}
else
{
lean_inc(v_a_3876_);
lean_dec(v_a_3867_);
v___x_3878_ = lean_box(0);
v_isShared_3879_ = v_isSharedCheck_3883_;
goto v_resetjp_3877_;
}
v_resetjp_3877_:
{
lean_object* v___x_3881_; 
if (v_isShared_3879_ == 0)
{
lean_ctor_set_tag(v___x_3878_, 0);
v___x_3881_ = v___x_3878_;
goto v_reusejp_3880_;
}
else
{
lean_object* v_reuseFailAlloc_3882_; 
v_reuseFailAlloc_3882_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3882_, 0, v_a_3876_);
v___x_3881_ = v_reuseFailAlloc_3882_;
goto v_reusejp_3880_;
}
v_reusejp_3880_:
{
return v___x_3881_;
}
}
}
}
v___jp_3884_:
{
if (v___y_3886_ == 0)
{
lean_object* v___x_3887_; 
lean_dec_ref(v___y_3885_);
v___x_3887_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3887_) == 0)
{
lean_object* v_a_3888_; lean_object* v___x_3890_; uint8_t v_isShared_3891_; uint8_t v_isSharedCheck_3903_; 
v_a_3888_ = lean_ctor_get(v___x_3887_, 0);
v_isSharedCheck_3903_ = !lean_is_exclusive(v___x_3887_);
if (v_isSharedCheck_3903_ == 0)
{
v___x_3890_ = v___x_3887_;
v_isShared_3891_ = v_isSharedCheck_3903_;
goto v_resetjp_3889_;
}
else
{
lean_inc(v_a_3888_);
lean_dec(v___x_3887_);
v___x_3890_ = lean_box(0);
v_isShared_3891_ = v_isSharedCheck_3903_;
goto v_resetjp_3889_;
}
v_resetjp_3889_:
{
lean_object* v_ref_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3901_; 
v_ref_3892_ = lean_ctor_get(v___y_3863_, 5);
v___x_3893_ = l_Lean_SourceInfo_fromRef(v_ref_3892_, v___y_3886_);
v___x_3894_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b___x7d___closed__1));
v___x_3895_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__4));
lean_inc_n(v___x_3893_, 2);
v___x_3896_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3896_, 0, v___x_3893_);
lean_ctor_set(v___x_3896_, 1, v___x_3895_);
v___x_3897_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_3898_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3898_, 0, v___x_3893_);
lean_ctor_set(v___x_3898_, 1, v___x_3897_);
v___x_3899_ = l_Lean_Syntax_node3(v___x_3893_, v___x_3894_, v___x_3896_, v_a_3888_, v___x_3898_);
if (v_isShared_3891_ == 0)
{
lean_ctor_set(v___x_3890_, 0, v___x_3899_);
v___x_3901_ = v___x_3890_;
goto v_reusejp_3900_;
}
else
{
lean_object* v_reuseFailAlloc_3902_; 
v_reuseFailAlloc_3902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3902_, 0, v___x_3899_);
v___x_3901_ = v_reuseFailAlloc_3902_;
goto v_reusejp_3900_;
}
v_reusejp_3900_:
{
return v___x_3901_;
}
}
}
else
{
return v___x_3887_;
}
}
else
{
if (lean_obj_tag(v___y_3885_) == 0)
{
lean_object* v_a_3904_; 
v_a_3904_ = lean_ctor_get(v___y_3885_, 0);
lean_inc(v_a_3904_);
lean_dec_ref_known(v___y_3885_, 1);
v_a_3867_ = v_a_3904_;
goto v___jp_3866_;
}
else
{
lean_object* v_a_3905_; lean_object* v___x_3907_; uint8_t v_isShared_3908_; uint8_t v_isSharedCheck_3912_; 
v_a_3905_ = lean_ctor_get(v___y_3885_, 0);
v_isSharedCheck_3912_ = !lean_is_exclusive(v___y_3885_);
if (v_isSharedCheck_3912_ == 0)
{
v___x_3907_ = v___y_3885_;
v_isShared_3908_ = v_isSharedCheck_3912_;
goto v_resetjp_3906_;
}
else
{
lean_inc(v_a_3905_);
lean_dec(v___y_3885_);
v___x_3907_ = lean_box(0);
v_isShared_3908_ = v_isSharedCheck_3912_;
goto v_resetjp_3906_;
}
v_resetjp_3906_:
{
lean_object* v___x_3910_; 
if (v_isShared_3908_ == 0)
{
v___x_3910_ = v___x_3907_;
goto v_reusejp_3909_;
}
else
{
lean_object* v_reuseFailAlloc_3911_; 
v_reuseFailAlloc_3911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3911_, 0, v_a_3905_);
v___x_3910_ = v_reuseFailAlloc_3911_;
goto v_reusejp_3909_;
}
v_reusejp_3909_:
{
return v___x_3910_;
}
}
}
}
}
v___jp_3913_:
{
uint8_t v___x_3916_; 
v___x_3916_ = l_Lean_Exception_isInterrupt(v_a_3915_);
if (v___x_3916_ == 0)
{
uint8_t v___x_3917_; 
v___x_3917_ = l_Lean_Exception_isRuntime(v_a_3915_);
v___y_3885_ = v___y_3914_;
v___y_3886_ = v___x_3917_;
goto v___jp_3884_;
}
else
{
lean_dec_ref(v_a_3915_);
v___y_3885_ = v___y_3914_;
v___y_3886_ = v___x_3916_;
goto v___jp_3884_;
}
}
v___jp_3918_:
{
if (lean_obj_tag(v___y_3919_) == 0)
{
lean_object* v_a_3920_; 
v_a_3920_ = lean_ctor_get(v___y_3919_, 0);
lean_inc(v_a_3920_);
lean_dec_ref_known(v___y_3919_, 1);
v_a_3867_ = v_a_3920_;
goto v___jp_3866_;
}
else
{
lean_object* v_a_3921_; 
v_a_3921_ = lean_ctor_get(v___y_3919_, 0);
lean_inc(v_a_3921_);
v___y_3914_ = v___y_3919_;
v_a_3915_ = v_a_3921_;
goto v___jp_3913_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1___boxed(lean_object* v___f_4035_, lean_object* v___y_4036_, lean_object* v___y_4037_, lean_object* v___y_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_){
_start:
{
lean_object* v_res_4043_; 
v_res_4043_ = lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1(v___f_4035_, v___y_4036_, v___y_4037_, v___y_4038_, v___y_4039_, v___y_4040_, v___y_4041_);
lean_dec(v___y_4041_);
lean_dec_ref(v___y_4040_);
lean_dec(v___y_4039_);
lean_dec_ref(v___y_4038_);
lean_dec(v___y_4037_);
lean_dec_ref(v___y_4036_);
return v_res_4043_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren(lean_object* v_a_4044_, lean_object* v_a_4045_, lean_object* v_a_4046_, lean_object* v_a_4047_, lean_object* v_a_4048_, lean_object* v_a_4049_){
_start:
{
lean_object* v___f_4051_; lean_object* v___f_4052_; lean_object* v___x_4053_; lean_object* v___x_4054_; lean_object* v___x_4055_; 
v___f_4051_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__0));
v___f_4052_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___lam__1___boxed), 8, 1);
lean_closure_set(v___f_4052_, 0, v___f_4051_);
v___x_4053_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_delabJsxChildren___closed__1));
v___x_4054_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___boxed), 9, 2);
lean_closure_set(v___x_4054_, 0, v___x_4053_);
lean_closure_set(v___x_4054_, 1, v___f_4052_);
v___x_4055_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(v___x_4054_, v_a_4044_, v_a_4045_, v_a_4046_, v_a_4047_, v_a_4048_, v_a_4049_);
if (lean_obj_tag(v___x_4055_) == 0)
{
return v___x_4055_;
}
else
{
lean_object* v_a_4056_; uint8_t v___y_4058_; uint8_t v___x_4087_; 
v_a_4056_ = lean_ctor_get(v___x_4055_, 0);
lean_inc(v_a_4056_);
v___x_4087_ = l_Lean_Exception_isInterrupt(v_a_4056_);
if (v___x_4087_ == 0)
{
uint8_t v___x_4088_; 
v___x_4088_ = l_Lean_Exception_isRuntime(v_a_4056_);
v___y_4058_ = v___x_4088_;
goto v___jp_4057_;
}
else
{
lean_dec(v_a_4056_);
v___y_4058_ = v___x_4087_;
goto v___jp_4057_;
}
v___jp_4057_:
{
if (v___y_4058_ == 0)
{
lean_object* v___x_4059_; 
lean_dec_ref_known(v___x_4055_, 1);
v___x_4059_ = l_Lean_PrettyPrinter_Delaborator_delab(v_a_4044_, v_a_4045_, v_a_4046_, v_a_4047_, v_a_4048_, v_a_4049_);
if (lean_obj_tag(v___x_4059_) == 0)
{
lean_object* v_a_4060_; lean_object* v___x_4062_; uint8_t v_isShared_4063_; uint8_t v_isSharedCheck_4078_; 
v_a_4060_ = lean_ctor_get(v___x_4059_, 0);
v_isSharedCheck_4078_ = !lean_is_exclusive(v___x_4059_);
if (v_isSharedCheck_4078_ == 0)
{
v___x_4062_ = v___x_4059_;
v_isShared_4063_ = v_isSharedCheck_4078_;
goto v_resetjp_4061_;
}
else
{
lean_inc(v_a_4060_);
lean_dec(v___x_4059_);
v___x_4062_ = lean_box(0);
v_isShared_4063_ = v_isSharedCheck_4078_;
goto v_resetjp_4061_;
}
v_resetjp_4061_:
{
lean_object* v_ref_4064_; lean_object* v___x_4065_; lean_object* v___x_4066_; lean_object* v___x_4067_; lean_object* v___x_4068_; lean_object* v___x_4069_; lean_object* v___x_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; lean_object* v___x_4073_; lean_object* v___x_4074_; lean_object* v___x_4076_; 
v_ref_4064_ = lean_ctor_get(v_a_4048_, 5);
v___x_4065_ = l_Lean_SourceInfo_fromRef(v_ref_4064_, v___y_4058_);
v___x_4066_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__1));
v___x_4067_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxChild_x7b_x2e_x2e_x2e___x7d___closed__2));
lean_inc_n(v___x_4065_, 2);
v___x_4068_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4068_, 0, v___x_4065_);
lean_ctor_set(v___x_4068_, 1, v___x_4067_);
v___x_4069_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_proofWidgetsJsxAttrVal_x7b___x7d___closed__10));
v___x_4070_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4070_, 0, v___x_4065_);
lean_ctor_set(v___x_4070_, 1, v___x_4069_);
v___x_4071_ = l_Lean_Syntax_node3(v___x_4065_, v___x_4066_, v___x_4068_, v_a_4060_, v___x_4070_);
v___x_4072_ = lean_unsigned_to_nat(1u);
v___x_4073_ = lean_mk_empty_array_with_capacity(v___x_4072_);
v___x_4074_ = lean_array_push(v___x_4073_, v___x_4071_);
if (v_isShared_4063_ == 0)
{
lean_ctor_set(v___x_4062_, 0, v___x_4074_);
v___x_4076_ = v___x_4062_;
goto v_reusejp_4075_;
}
else
{
lean_object* v_reuseFailAlloc_4077_; 
v_reuseFailAlloc_4077_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4077_, 0, v___x_4074_);
v___x_4076_ = v_reuseFailAlloc_4077_;
goto v_reusejp_4075_;
}
v_reusejp_4075_:
{
return v___x_4076_;
}
}
}
else
{
lean_object* v_a_4079_; lean_object* v___x_4081_; uint8_t v_isShared_4082_; uint8_t v_isSharedCheck_4086_; 
v_a_4079_ = lean_ctor_get(v___x_4059_, 0);
v_isSharedCheck_4086_ = !lean_is_exclusive(v___x_4059_);
if (v_isSharedCheck_4086_ == 0)
{
v___x_4081_ = v___x_4059_;
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
else
{
lean_inc(v_a_4079_);
lean_dec(v___x_4059_);
v___x_4081_ = lean_box(0);
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
v_resetjp_4080_:
{
lean_object* v___x_4084_; 
if (v_isShared_4082_ == 0)
{
v___x_4084_ = v___x_4081_;
goto v_reusejp_4083_;
}
else
{
lean_object* v_reuseFailAlloc_4085_; 
v_reuseFailAlloc_4085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4085_, 0, v_a_4079_);
v___x_4084_ = v_reuseFailAlloc_4085_;
goto v_reusejp_4083_;
}
v_reusejp_4083_:
{
return v___x_4084_;
}
}
}
}
else
{
return v___x_4055_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27___boxed(lean_object* v_a_4089_, lean_object* v_a_4090_, lean_object* v_a_4091_, lean_object* v_a_4092_, lean_object* v_a_4093_, lean_object* v_a_4094_, lean_object* v_a_4095_){
_start:
{
lean_object* v_res_4096_; 
v_res_4096_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27(v_a_4089_, v_a_4090_, v_a_4091_, v_a_4092_, v_a_4093_, v_a_4094_);
lean_dec(v_a_4094_);
lean_dec_ref(v_a_4093_);
lean_dec(v_a_4092_);
lean_dec_ref(v_a_4091_);
lean_dec(v_a_4090_);
lean_dec_ref(v_a_4089_);
return v_res_4096_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27___boxed(lean_object* v_a_4097_, lean_object* v_a_4098_, lean_object* v_a_4099_, lean_object* v_a_4100_, lean_object* v_a_4101_, lean_object* v_a_4102_, lean_object* v_a_4103_){
_start:
{
lean_object* v_res_4104_; 
v_res_4104_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27(v_a_4097_, v_a_4098_, v_a_4099_, v_a_4100_, v_a_4101_, v_a_4102_);
lean_dec(v_a_4102_);
lean_dec_ref(v_a_4101_);
lean_dec(v_a_4100_);
lean_dec_ref(v_a_4099_);
lean_dec(v_a_4098_);
lean_dec_ref(v_a_4097_);
return v_res_4104_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0(lean_object* v___y_4105_, lean_object* v___y_4106_, lean_object* v___y_4107_, lean_object* v___y_4108_, lean_object* v___y_4109_, lean_object* v___y_4110_){
_start:
{
lean_object* v___x_4112_; 
v___x_4112_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___redArg(v___y_4105_);
return v___x_4112_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0___boxed(lean_object* v___y_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_, lean_object* v___y_4119_){
_start:
{
lean_object* v_res_4120_; 
v_res_4120_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0_spec__0(v___y_4113_, v___y_4114_, v___y_4115_, v___y_4116_, v___y_4117_, v___y_4118_);
lean_dec(v___y_4118_);
lean_dec_ref(v___y_4117_);
lean_dec(v___y_4116_);
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4114_);
lean_dec_ref(v___y_4113_);
return v_res_4120_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0(lean_object* v_00_u03b1_4121_, lean_object* v_argIdx_4122_, lean_object* v_x_4123_, lean_object* v___y_4124_, lean_object* v___y_4125_, lean_object* v___y_4126_, lean_object* v___y_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_){
_start:
{
lean_object* v___x_4131_; 
v___x_4131_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___redArg(v_argIdx_4122_, v_x_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_, v___y_4128_, v___y_4129_);
return v___x_4131_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0___boxed(lean_object* v_00_u03b1_4132_, lean_object* v_argIdx_4133_, lean_object* v_x_4134_, lean_object* v___y_4135_, lean_object* v___y_4136_, lean_object* v___y_4137_, lean_object* v___y_4138_, lean_object* v___y_4139_, lean_object* v___y_4140_, lean_object* v___y_4141_){
_start:
{
lean_object* v_res_4142_; 
v_res_4142_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__0(v_00_u03b1_4132_, v_argIdx_4133_, v_x_4134_, v___y_4135_, v___y_4136_, v___y_4137_, v___y_4138_, v___y_4139_, v___y_4140_);
lean_dec(v___y_4140_);
lean_dec_ref(v___y_4139_);
lean_dec(v___y_4138_);
lean_dec_ref(v___y_4137_);
lean_dec(v___y_4136_);
lean_dec_ref(v___y_4135_);
lean_dec(v_argIdx_4133_);
return v_res_4142_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2(lean_object* v_00_u03b1_4143_, lean_object* v_child_4144_, lean_object* v_childIdx_4145_, lean_object* v_x_4146_, lean_object* v___y_4147_, lean_object* v___y_4148_, lean_object* v___y_4149_, lean_object* v___y_4150_, lean_object* v___y_4151_, lean_object* v___y_4152_){
_start:
{
lean_object* v___x_4154_; 
v___x_4154_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___redArg(v_child_4144_, v_childIdx_4145_, v_x_4146_, v___y_4147_, v___y_4148_, v___y_4149_, v___y_4150_, v___y_4151_, v___y_4152_);
return v___x_4154_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2___boxed(lean_object* v_00_u03b1_4155_, lean_object* v_child_4156_, lean_object* v_childIdx_4157_, lean_object* v_x_4158_, lean_object* v___y_4159_, lean_object* v___y_4160_, lean_object* v___y_4161_, lean_object* v___y_4162_, lean_object* v___y_4163_, lean_object* v___y_4164_, lean_object* v___y_4165_){
_start:
{
lean_object* v_res_4166_; 
v_res_4166_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1_spec__2(v_00_u03b1_4155_, v_child_4156_, v_childIdx_4157_, v_x_4158_, v___y_4159_, v___y_4160_, v___y_4161_, v___y_4162_, v___y_4163_, v___y_4164_);
lean_dec(v___y_4164_);
lean_dec_ref(v___y_4163_);
lean_dec(v___y_4162_);
lean_dec_ref(v___y_4161_);
lean_dec(v___y_4160_);
lean_dec_ref(v___y_4159_);
return v_res_4166_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1(lean_object* v_00_u03b1_4167_, lean_object* v_x_4168_, lean_object* v___y_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_, lean_object* v___y_4172_, lean_object* v___y_4173_, lean_object* v___y_4174_){
_start:
{
lean_object* v___x_4176_; 
v___x_4176_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___redArg(v_x_4168_, v___y_4169_, v___y_4170_, v___y_4171_, v___y_4172_, v___y_4173_, v___y_4174_);
return v___x_4176_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1___boxed(lean_object* v_00_u03b1_4177_, lean_object* v_x_4178_, lean_object* v___y_4179_, lean_object* v___y_4180_, lean_object* v___y_4181_, lean_object* v___y_4182_, lean_object* v___y_4183_, lean_object* v___y_4184_, lean_object* v___y_4185_){
_start:
{
lean_object* v_res_4186_; 
v_res_4186_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00ProofWidgets_Jsx_delabHtmlElement_x27_spec__1(v_00_u03b1_4177_, v_x_4178_, v___y_4179_, v___y_4180_, v___y_4181_, v___y_4182_, v___y_4183_, v___y_4184_);
lean_dec(v___y_4184_);
lean_dec_ref(v___y_4183_);
lean_dec(v___y_4182_);
lean_dec_ref(v___y_4181_);
lean_dec(v___y_4180_);
lean_dec_ref(v___y_4179_);
return v_res_4186_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8(size_t v_sz_4187_, size_t v_i_4188_, lean_object* v_bs_4189_, lean_object* v___y_4190_, lean_object* v___y_4191_, lean_object* v___y_4192_, lean_object* v___y_4193_, lean_object* v___y_4194_, lean_object* v___y_4195_){
_start:
{
lean_object* v___x_4197_; 
v___x_4197_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___redArg(v_sz_4187_, v_i_4188_, v_bs_4189_, v___y_4194_);
return v___x_4197_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8___boxed(lean_object* v_sz_4198_, lean_object* v_i_4199_, lean_object* v_bs_4200_, lean_object* v___y_4201_, lean_object* v___y_4202_, lean_object* v___y_4203_, lean_object* v___y_4204_, lean_object* v___y_4205_, lean_object* v___y_4206_, lean_object* v___y_4207_){
_start:
{
size_t v_sz_boxed_4208_; size_t v_i_boxed_4209_; lean_object* v_res_4210_; 
v_sz_boxed_4208_ = lean_unbox_usize(v_sz_4198_);
lean_dec(v_sz_4198_);
v_i_boxed_4209_ = lean_unbox_usize(v_i_4199_);
lean_dec(v_i_4199_);
v_res_4210_ = lp_proofwidgets___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00ProofWidgets_Jsx_delabHtmlOfComponent_x27_spec__8(v_sz_boxed_4208_, v_i_boxed_4209_, v_bs_4200_, v___y_4201_, v___y_4202_, v___y_4203_, v___y_4204_, v___y_4205_, v___y_4206_);
lean_dec(v___y_4206_);
lean_dec_ref(v___y_4205_);
lean_dec(v___y_4204_);
lean_dec_ref(v___y_4203_);
lean_dec(v___y_4202_);
lean_dec_ref(v___y_4201_);
return v_res_4210_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement(lean_object* v_a_4211_, lean_object* v_a_4212_, lean_object* v_a_4213_, lean_object* v_a_4214_, lean_object* v_a_4215_, lean_object* v_a_4216_){
_start:
{
lean_object* v___x_4218_; 
v___x_4218_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement_x27(v_a_4211_, v_a_4212_, v_a_4213_, v_a_4214_, v_a_4215_, v_a_4216_);
if (lean_obj_tag(v___x_4218_) == 0)
{
lean_object* v_a_4219_; lean_object* v___x_4221_; uint8_t v_isShared_4222_; uint8_t v_isSharedCheck_4231_; 
v_a_4219_ = lean_ctor_get(v___x_4218_, 0);
v_isSharedCheck_4231_ = !lean_is_exclusive(v___x_4218_);
if (v_isSharedCheck_4231_ == 0)
{
v___x_4221_ = v___x_4218_;
v_isShared_4222_ = v_isSharedCheck_4231_;
goto v_resetjp_4220_;
}
else
{
lean_inc(v_a_4219_);
lean_dec(v___x_4218_);
v___x_4221_ = lean_box(0);
v_isShared_4222_ = v_isSharedCheck_4231_;
goto v_resetjp_4220_;
}
v_resetjp_4220_:
{
lean_object* v_ref_4223_; uint8_t v___x_4224_; lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v___x_4229_; 
v_ref_4223_ = lean_ctor_get(v_a_4215_, 5);
v___x_4224_ = 0;
v___x_4225_ = l_Lean_SourceInfo_fromRef(v_ref_4223_, v___x_4224_);
v___x_4226_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1));
v___x_4227_ = l_Lean_Syntax_node1(v___x_4225_, v___x_4226_, v_a_4219_);
if (v_isShared_4222_ == 0)
{
lean_ctor_set(v___x_4221_, 0, v___x_4227_);
v___x_4229_ = v___x_4221_;
goto v_reusejp_4228_;
}
else
{
lean_object* v_reuseFailAlloc_4230_; 
v_reuseFailAlloc_4230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4230_, 0, v___x_4227_);
v___x_4229_ = v_reuseFailAlloc_4230_;
goto v_reusejp_4228_;
}
v_reusejp_4228_:
{
return v___x_4229_;
}
}
}
else
{
return v___x_4218_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement___boxed(lean_object* v_a_4232_, lean_object* v_a_4233_, lean_object* v_a_4234_, lean_object* v_a_4235_, lean_object* v_a_4236_, lean_object* v_a_4237_, lean_object* v_a_4238_){
_start:
{
lean_object* v_res_4239_; 
v_res_4239_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlElement(v_a_4232_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_);
lean_dec(v_a_4237_);
lean_dec_ref(v_a_4236_);
lean_dec(v_a_4235_);
lean_dec_ref(v_a_4234_);
lean_dec(v_a_4233_);
lean_dec_ref(v_a_4232_);
return v_res_4239_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent(lean_object* v_a_4240_, lean_object* v_a_4241_, lean_object* v_a_4242_, lean_object* v_a_4243_, lean_object* v_a_4244_, lean_object* v_a_4245_){
_start:
{
lean_object* v___x_4247_; 
v___x_4247_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent_x27(v_a_4240_, v_a_4241_, v_a_4242_, v_a_4243_, v_a_4244_, v_a_4245_);
if (lean_obj_tag(v___x_4247_) == 0)
{
lean_object* v_a_4248_; lean_object* v___x_4250_; uint8_t v_isShared_4251_; uint8_t v_isSharedCheck_4260_; 
v_a_4248_ = lean_ctor_get(v___x_4247_, 0);
v_isSharedCheck_4260_ = !lean_is_exclusive(v___x_4247_);
if (v_isSharedCheck_4260_ == 0)
{
v___x_4250_ = v___x_4247_;
v_isShared_4251_ = v_isSharedCheck_4260_;
goto v_resetjp_4249_;
}
else
{
lean_inc(v_a_4248_);
lean_dec(v___x_4247_);
v___x_4250_ = lean_box(0);
v_isShared_4251_ = v_isSharedCheck_4260_;
goto v_resetjp_4249_;
}
v_resetjp_4249_:
{
lean_object* v_ref_4252_; uint8_t v___x_4253_; lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___x_4256_; lean_object* v___x_4258_; 
v_ref_4252_ = lean_ctor_get(v_a_4244_, 5);
v___x_4253_ = 0;
v___x_4254_ = l_Lean_SourceInfo_fromRef(v_ref_4252_, v___x_4253_);
v___x_4255_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Jsx_term___00__closed__1));
v___x_4256_ = l_Lean_Syntax_node1(v___x_4254_, v___x_4255_, v_a_4248_);
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 0, v___x_4256_);
v___x_4258_ = v___x_4250_;
goto v_reusejp_4257_;
}
else
{
lean_object* v_reuseFailAlloc_4259_; 
v_reuseFailAlloc_4259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4259_, 0, v___x_4256_);
v___x_4258_ = v_reuseFailAlloc_4259_;
goto v_reusejp_4257_;
}
v_reusejp_4257_:
{
return v___x_4258_;
}
}
}
else
{
return v___x_4247_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent___boxed(lean_object* v_a_4261_, lean_object* v_a_4262_, lean_object* v_a_4263_, lean_object* v_a_4264_, lean_object* v_a_4265_, lean_object* v_a_4266_, lean_object* v_a_4267_){
_start:
{
lean_object* v_res_4268_; 
v_res_4268_ = lp_proofwidgets_ProofWidgets_Jsx_delabHtmlOfComponent(v_a_4261_, v_a_4262_, v_a_4263_, v_a_4264_, v_a_4265_, v_a_4266_);
lean_dec(v_a_4266_);
lean_dec_ref(v_a_4265_);
lean_dec(v_a_4264_);
lean_dec_ref(v_a_4263_);
lean_dec(v_a_4262_);
lean_dec_ref(v_a_4261_);
return v_res_4268_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin) {
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
res = runtime_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxElement = _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxElement();
lean_mark_persistent(lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxElement);
lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxChild = _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxChild();
lean_mark_persistent(lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxChild);
lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttr = _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttr();
lean_mark_persistent(lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttr);
lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttrVal = _init_lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttrVal();
lean_mark_persistent(lp_proofwidgets_Lean_Parser_Category_proofWidgetsJsxAttrVal);
lp_proofwidgets_ProofWidgets_Jsx_jsxText = _init_lp_proofwidgets_ProofWidgets_Jsx_jsxText();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_Jsx_jsxText);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin) {
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
res = initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
}
#ifdef __cplusplus
}
#endif
