// Lean compiler output
// Module: ProofWidgets.Component.OfRpcMethod
// Imports: public import Init public meta import Init import ProofWidgets.Data.Html public import ProofWidgets.Cancellable
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
lean_object* l_String_Slice_slice_x21(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_mkStrLit(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MapDeclarationExtension_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkArrow(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_cancellableSuffix;
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_existsBuiltinRpcProcedure(lean_object*);
extern lean_object* l_Lean_Server_userRpcProcedures;
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Elab_Term_withExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3838, .m_capacity = 3838, .m_length = 3837, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import{useRpcSession as o,EnvPosContext as a,useAsyncPersistent as i,mapRpcError as f,importWidgetModule as c}from\"@leanprover/infoview\";function u(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,\"default\")\?e.default:e}var s,l;var p=u(function(){if(l)return s;l=1;var e=\"undefined\"!=typeof Element,t=\"function\"==typeof Map,r=\"function\"==typeof Set,n=\"function\"==typeof ArrayBuffer&&!!ArrayBuffer.isView;function o(a,i){if(a===i)return!0;if(a&&i&&\"object\"==typeof a&&\"object\"==typeof i){if(a.constructor!==i.constructor)return!1;var f,c,u,s;if(Array.isArray(a)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(!o(a[c],i[c]))return!1;return!0}if(t&&a instanceof Map&&i instanceof Map){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;for(s=a.entries();!(c=s.next()).done;)if(!o(c.value[1],i.get(c.value[0])))return!1;return!0}if(r&&a instanceof Set&&i instanceof Set){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;return!0}if(n&&ArrayBuffer.isView(a)&&ArrayBuffer.isView(i)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(a[c]!==i[c])return!1;return!0}if(a.constructor===RegExp)return a.source===i.source&&a.flags===i.flags;if(a.valueOf!==Object.prototype.valueOf&&\"function\"==typeof a.valueOf&&\"function\"==typeof i.valueOf)return a.valueOf()===i.valueOf();if(a.toString!==Object.prototype.toString&&\"function\"==typeof a.toString&&\"function\"==typeof i.toString)return a.toString()===i.toString();if((f=(u=Object.keys(a)).length)!==Object.keys(i).length)return!1;for(c=f;0!==c--;)if(!Object.prototype.hasOwnProperty.call(i,u[c]))return!1;if(e&&a instanceof Element)return!1;for(c=f;0!==c--;)if((\"_owner\"!==u[c]&&\"__v\"!==u[c]&&\"__o\"!==u[c]||!a.$$typeof)&&!o(a[u[c]],i[u[c]]))return!1;return!0}return a!=a&&i!=i}return s=function(e,t){try{return o(e,t)}catch(e){if((e.message||\"\").match(/stack|recursion/i))return console.warn(\"react-fast-compare cannot handle circular refs\"),!1;throw e}}}());async function y(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,f]=i.element,c={};for(const[e,t]of r)c[e]=t;const u=await Promise.all(f.map(async e=>await y(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===u.length\?n.createElement(e,c):n.createElement(e,c,u)}if(\"component\"in i){const[e,t,r,f]=i.component,u=await Promise.all(f.map(async e=>await y(o,a,e))),s={...r,pos:a},l=await c(o,a,e);if(!(t in l))throw new Error(`Module '${e}' does not export '${t}'`);return 0===u.length\?n.createElement(l[t],s):n.createElement(l[t],s,u)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function d({html:c}){const u=o(),s=n.useContext(a),l=i(()=>y(u,s,c),[u,s,c]);return\"resolved\"===l.state\?l.value:\"rejected\"===l.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",f(l.error).message]}):t(r,{})}const m=\"$RPC_METHOD\",g=window.toString();var w=n.memo(e=>{const a=o(),c=n.useRef({fn:()=>{}}),u=i(async()=>{if(c.current.fn(),\"true\"===g){const[t,r]=function(e,t,r){const n={fn:()=>{}};return[new Promise(async(o,a)=>{const i=await e.call(t,r),f=window.setInterval(async()=>{try{const t=await e.call(\"ProofWidgets.checkRequest\",i);if(\"running\"===t)return;window.clearInterval(f),o(t.done.result)}catch(e){window.clearInterval(f),a(e)}},100);n.fn=()=>{e.call(\"ProofWidgets.cancelRequest\",i)}}),n]}(a,m,e);return c.current=r,t}{const t=new AbortController,r=a.call(m,e,{abortSignal:t.signal});return c.current={fn:()=>t.abort()},r}},[a,e]);return n.useEffect(()=>()=>{c.current.fn()},[]),\"rejected\"===u.state\?t(\"p\",{style:{color:\"red\"},children:f(u.error).message}):\"loading\"===u.state\?t(r,{children:\"Loading..\"}):t(d,{html:u.value})},p);export{w as default};"};
static const lean_object* lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate = (const lean_object*)&lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "termMk_rpc_widget%_"};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(31, 169, 70, 166, 222, 232, 188, 210)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mk_rpc_widget%"};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__5_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__5_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__8_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__9 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__9_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__4_value),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__6_value),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__9_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__10_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__10_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25__ = (const lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__11_value;
static lean_once_cell_t lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "$RPC_METHOD"};
static const lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__0_value;
static const lean_string_object lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__1 = (const lean_object*)&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6;
static const lean_ctor_object lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__7 = (const lean_object*)&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "window.toString()"};
static const lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__0_value;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4;
static lean_once_cell_t lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0;
static const lean_string_object lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__1 = (const lean_object*)&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__1_value;
static const lean_ctor_object lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__1_value)}};
static const lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__2 = (const lean_object*)&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__2_value;
static lean_once_cell_t lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3;
LEAN_EXPORT lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__0_value;
static const lean_ctor_object lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__0_value)}};
static const lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__1_value;
static lean_once_cell_t lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "'true'"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__3_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__4_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__5 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__5_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__6_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__7 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__7_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__8 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__8_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__10 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__10_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__11 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__11_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "structInstLVal"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__12 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__12_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "javascript"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__13 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__13_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(124, 118, 184, 62, 15, 192, 226, 192)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__15 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__15_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structInstFieldDef"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__16 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__16_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__17 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__17_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__18 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__18_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__19 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__19_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "'false'"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__20 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__20_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__21 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__21_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 72, .m_capacity = 72, .m_length = 71, .m_data = "' is not a known RPC method. Use `@[server_rpc_method]` to register it."};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__22 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__22_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__26 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__26_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__26_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__27 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__27_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Component"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__28 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__28_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__28_value),LEAN_SCALAR_PTR_LITERAL(222, 234, 130, 55, 111, 248, 107, 78)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Server"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__31 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__31_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "RequestTask"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__32 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__32_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__31_value),LEAN_SCALAR_PTR_LITERAL(251, 1, 140, 35, 91, 244, 83, 213)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__32_value),LEAN_SCALAR_PTR_LITERAL(74, 251, 196, 59, 228, 211, 83, 197)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Html"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__34 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__34_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(48, 5, 23, 178, 23, 214, 71, 37)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "RequestM"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__38 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__38_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__31_value),LEAN_SCALAR_PTR_LITERAL(251, 1, 140, 35, 91, 244, 83, 213)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(184, 87, 7, 59, 37, 78, 138, 49)}};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39_value;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Expected the name of a constant, got a complex term"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__40 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__40_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "expected type"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__42 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__42_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43;
static const lean_string_object lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "\nis not of the form"};
static const lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__44 = (const lean_object*)&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__44_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_box(0);
v___x_30_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_31_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_31_, 0, v___x_30_);
lean_ctor_set(v___x_31_, 1, v___x_29_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg(){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_obj_once(&lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0, &lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0_once, _init_lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___closed__0);
v___x_34_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg___boxed(lean_object* v___y_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg();
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0(lean_object* v_00_u03b1_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg();
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___boxed(lean_object* v_00_u03b1_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0(v_00_u03b1_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg(lean_object* v_e_55_, lean_object* v___y_56_){
_start:
{
uint8_t v___x_58_; 
v___x_58_ = l_Lean_Expr_hasMVar(v_e_55_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; 
v___x_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_59_, 0, v_e_55_);
return v___x_59_;
}
else
{
lean_object* v___x_60_; lean_object* v_mctx_61_; lean_object* v___x_62_; lean_object* v_fst_63_; lean_object* v_snd_64_; lean_object* v___x_65_; lean_object* v_cache_66_; lean_object* v_zetaDeltaFVarIds_67_; lean_object* v_postponed_68_; lean_object* v_diag_69_; lean_object* v___x_71_; uint8_t v_isShared_72_; uint8_t v_isSharedCheck_78_; 
v___x_60_ = lean_st_ref_get(v___y_56_);
v_mctx_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc_ref(v_mctx_61_);
lean_dec(v___x_60_);
v___x_62_ = l_Lean_instantiateMVarsCore(v_mctx_61_, v_e_55_);
v_fst_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_fst_63_);
v_snd_64_ = lean_ctor_get(v___x_62_, 1);
lean_inc(v_snd_64_);
lean_dec_ref(v___x_62_);
v___x_65_ = lean_st_ref_take(v___y_56_);
v_cache_66_ = lean_ctor_get(v___x_65_, 1);
v_zetaDeltaFVarIds_67_ = lean_ctor_get(v___x_65_, 2);
v_postponed_68_ = lean_ctor_get(v___x_65_, 3);
v_diag_69_ = lean_ctor_get(v___x_65_, 4);
v_isSharedCheck_78_ = !lean_is_exclusive(v___x_65_);
if (v_isSharedCheck_78_ == 0)
{
lean_object* v_unused_79_; 
v_unused_79_ = lean_ctor_get(v___x_65_, 0);
lean_dec(v_unused_79_);
v___x_71_ = v___x_65_;
v_isShared_72_ = v_isSharedCheck_78_;
goto v_resetjp_70_;
}
else
{
lean_inc(v_diag_69_);
lean_inc(v_postponed_68_);
lean_inc(v_zetaDeltaFVarIds_67_);
lean_inc(v_cache_66_);
lean_dec(v___x_65_);
v___x_71_ = lean_box(0);
v_isShared_72_ = v_isSharedCheck_78_;
goto v_resetjp_70_;
}
v_resetjp_70_:
{
lean_object* v___x_74_; 
if (v_isShared_72_ == 0)
{
lean_ctor_set(v___x_71_, 0, v_snd_64_);
v___x_74_ = v___x_71_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v_snd_64_);
lean_ctor_set(v_reuseFailAlloc_77_, 1, v_cache_66_);
lean_ctor_set(v_reuseFailAlloc_77_, 2, v_zetaDeltaFVarIds_67_);
lean_ctor_set(v_reuseFailAlloc_77_, 3, v_postponed_68_);
lean_ctor_set(v_reuseFailAlloc_77_, 4, v_diag_69_);
v___x_74_ = v_reuseFailAlloc_77_;
goto v_reusejp_73_;
}
v_reusejp_73_:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = lean_st_ref_set(v___y_56_, v___x_74_);
v___x_76_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_76_, 0, v_fst_63_);
return v___x_76_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg___boxed(lean_object* v_e_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg(v_e_80_, v___y_81_);
lean_dec(v___y_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1(lean_object* v_e_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg(v_e_84_, v___y_88_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___boxed(lean_object* v_e_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1(v_e_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(lean_object* v_s_102_, lean_object* v_replacement_103_, lean_object* v_a_104_, lean_object* v_b_105_){
_start:
{
lean_object* v_it_107_; lean_object* v_startPos_108_; lean_object* v_endPos_109_; lean_object* v_it_118_; 
switch(lean_obj_tag(v_a_104_))
{
case 0:
{
lean_object* v_pos_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_136_; 
v_pos_124_ = lean_ctor_get(v_a_104_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v_a_104_);
if (v_isSharedCheck_136_ == 0)
{
v___x_126_ = v_a_104_;
v_isShared_127_ = v_isSharedCheck_136_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_pos_124_);
lean_dec(v_a_104_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_136_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v_startInclusive_128_; lean_object* v_endExclusive_129_; lean_object* v___x_130_; uint8_t v___x_131_; 
v_startInclusive_128_ = lean_ctor_get(v_s_102_, 1);
v_endExclusive_129_ = lean_ctor_get(v_s_102_, 2);
v___x_130_ = lean_nat_sub(v_endExclusive_129_, v_startInclusive_128_);
v___x_131_ = lean_nat_dec_eq(v_pos_124_, v___x_130_);
lean_dec(v___x_130_);
if (v___x_131_ == 0)
{
lean_object* v___x_133_; 
if (v_isShared_127_ == 0)
{
lean_ctor_set_tag(v___x_126_, 1);
v___x_133_ = v___x_126_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v_pos_124_);
v___x_133_ = v_reuseFailAlloc_134_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
v_it_118_ = v___x_133_;
goto v___jp_117_;
}
}
else
{
lean_object* v___x_135_; 
lean_del_object(v___x_126_);
lean_dec(v_pos_124_);
v___x_135_ = lean_box(3);
v_it_118_ = v___x_135_;
goto v___jp_117_;
}
}
}
case 1:
{
lean_object* v_pos_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_149_; 
v_pos_137_ = lean_ctor_get(v_a_104_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v_a_104_);
if (v_isSharedCheck_149_ == 0)
{
v___x_139_ = v_a_104_;
v_isShared_140_ = v_isSharedCheck_149_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_pos_137_);
lean_dec(v_a_104_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_149_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v_str_141_; lean_object* v_startInclusive_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
v_str_141_ = lean_ctor_get(v_s_102_, 0);
v_startInclusive_142_ = lean_ctor_get(v_s_102_, 1);
v___x_143_ = lean_nat_add(v_startInclusive_142_, v_pos_137_);
v___x_144_ = lean_string_utf8_next_fast(v_str_141_, v___x_143_);
lean_dec(v___x_143_);
v___x_145_ = lean_nat_sub(v___x_144_, v_startInclusive_142_);
lean_inc(v___x_145_);
if (v_isShared_140_ == 0)
{
lean_ctor_set_tag(v___x_139_, 0);
lean_ctor_set(v___x_139_, 0, v___x_145_);
v___x_147_ = v___x_139_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
v_it_107_ = v___x_147_;
v_startPos_108_ = v_pos_137_;
v_endPos_109_ = v___x_145_;
goto v___jp_106_;
}
}
}
case 2:
{
lean_object* v_needle_150_; lean_object* v_table_151_; lean_object* v_stackPos_152_; lean_object* v_needlePos_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_212_; 
v_needle_150_ = lean_ctor_get(v_a_104_, 0);
v_table_151_ = lean_ctor_get(v_a_104_, 1);
v_stackPos_152_ = lean_ctor_get(v_a_104_, 2);
v_needlePos_153_ = lean_ctor_get(v_a_104_, 3);
v_isSharedCheck_212_ = !lean_is_exclusive(v_a_104_);
if (v_isSharedCheck_212_ == 0)
{
v___x_155_ = v_a_104_;
v_isShared_156_ = v_isSharedCheck_212_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_needlePos_153_);
lean_inc(v_stackPos_152_);
lean_inc(v_table_151_);
lean_inc(v_needle_150_);
lean_dec(v_a_104_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_212_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v_str_157_; lean_object* v_startInclusive_158_; lean_object* v_endExclusive_159_; lean_object* v_str_160_; lean_object* v_startInclusive_161_; lean_object* v_endExclusive_162_; lean_object* v_basePos_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
v_str_157_ = lean_ctor_get(v_needle_150_, 0);
v_startInclusive_158_ = lean_ctor_get(v_needle_150_, 1);
v_endExclusive_159_ = lean_ctor_get(v_needle_150_, 2);
v_str_160_ = lean_ctor_get(v_s_102_, 0);
v_startInclusive_161_ = lean_ctor_get(v_s_102_, 1);
v_endExclusive_162_ = lean_ctor_get(v_s_102_, 2);
v_basePos_163_ = lean_nat_sub(v_stackPos_152_, v_needlePos_153_);
v___x_164_ = lean_nat_sub(v_endExclusive_159_, v_startInclusive_158_);
v___x_165_ = lean_nat_add(v_basePos_163_, v___x_164_);
v___x_166_ = lean_nat_sub(v_endExclusive_162_, v_startInclusive_161_);
v___x_167_ = lean_nat_dec_le(v___x_165_, v___x_166_);
lean_dec(v___x_165_);
if (v___x_167_ == 0)
{
uint8_t v___x_168_; 
lean_dec(v___x_164_);
lean_del_object(v___x_155_);
lean_dec(v_needlePos_153_);
lean_dec(v_stackPos_152_);
lean_dec_ref(v_table_151_);
lean_dec_ref(v_needle_150_);
v___x_168_ = lean_nat_dec_lt(v_basePos_163_, v___x_166_);
if (v___x_168_ == 0)
{
lean_dec(v___x_166_);
lean_dec(v_basePos_163_);
lean_dec_ref(v_s_102_);
return v_b_105_;
}
else
{
lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_169_ = l_String_Slice_pos_x21(v_s_102_, v_basePos_163_);
lean_dec(v_basePos_163_);
v___x_170_ = lean_box(3);
v_it_107_ = v___x_170_;
v_startPos_108_ = v___x_169_;
v_endPos_109_ = v___x_166_;
goto v___jp_106_;
}
}
else
{
lean_object* v___x_171_; uint8_t v_stackByte_172_; lean_object* v___x_173_; uint8_t v_patByte_174_; uint8_t v___x_175_; 
lean_dec(v___x_166_);
v___x_171_ = lean_nat_add(v_startInclusive_161_, v_stackPos_152_);
v_stackByte_172_ = lean_string_get_byte_fast(v_str_160_, v___x_171_);
v___x_173_ = lean_nat_add(v_startInclusive_158_, v_needlePos_153_);
v_patByte_174_ = lean_string_get_byte_fast(v_str_157_, v___x_173_);
v___x_175_ = lean_uint8_dec_eq(v_stackByte_172_, v_patByte_174_);
if (v___x_175_ == 0)
{
lean_object* v___x_176_; uint8_t v___x_177_; 
lean_dec(v___x_164_);
v___x_176_ = lean_unsigned_to_nat(0u);
v___x_177_ = lean_nat_dec_eq(v_needlePos_153_, v___x_176_);
if (v___x_177_ == 0)
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v_newNeedlePos_180_; uint8_t v___x_181_; 
v___x_178_ = lean_unsigned_to_nat(1u);
v___x_179_ = lean_nat_sub(v_needlePos_153_, v___x_178_);
lean_dec(v_needlePos_153_);
v_newNeedlePos_180_ = lean_array_fget_borrowed(v_table_151_, v___x_179_);
lean_dec(v___x_179_);
v___x_181_ = lean_nat_dec_eq(v_newNeedlePos_180_, v___x_176_);
if (v___x_181_ == 0)
{
lean_object* v_oldBasePos_182_; lean_object* v___x_183_; lean_object* v_newBasePos_184_; lean_object* v___x_186_; 
lean_inc(v_newNeedlePos_180_);
v_oldBasePos_182_ = l_String_Slice_pos_x21(v_s_102_, v_basePos_163_);
lean_dec(v_basePos_163_);
v___x_183_ = lean_nat_sub(v_stackPos_152_, v_newNeedlePos_180_);
v_newBasePos_184_ = l_String_Slice_pos_x21(v_s_102_, v___x_183_);
lean_dec(v___x_183_);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 3, v_newNeedlePos_180_);
v___x_186_ = v___x_155_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_needle_150_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_table_151_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_stackPos_152_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_newNeedlePos_180_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
v_it_107_ = v___x_186_;
v_startPos_108_ = v_oldBasePos_182_;
v_endPos_109_ = v_newBasePos_184_;
goto v___jp_106_;
}
}
else
{
lean_object* v_basePos_188_; lean_object* v_nextStackPos_189_; lean_object* v___x_191_; 
v_basePos_188_ = l_String_Slice_pos_x21(v_s_102_, v_basePos_163_);
lean_dec(v_basePos_163_);
v_nextStackPos_189_ = l_String_Slice_posGE___redArg(v_s_102_, v_stackPos_152_);
lean_inc(v_nextStackPos_189_);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 3, v___x_176_);
lean_ctor_set(v___x_155_, 2, v_nextStackPos_189_);
v___x_191_ = v___x_155_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_needle_150_);
lean_ctor_set(v_reuseFailAlloc_192_, 1, v_table_151_);
lean_ctor_set(v_reuseFailAlloc_192_, 2, v_nextStackPos_189_);
lean_ctor_set(v_reuseFailAlloc_192_, 3, v___x_176_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
v_it_107_ = v___x_191_;
v_startPos_108_ = v_basePos_188_;
v_endPos_109_ = v_nextStackPos_189_;
goto v___jp_106_;
}
}
}
else
{
lean_object* v_basePos_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v_nextStackPos_196_; lean_object* v___x_198_; 
lean_dec(v_basePos_163_);
lean_dec(v_needlePos_153_);
v_basePos_193_ = l_String_Slice_pos_x21(v_s_102_, v_stackPos_152_);
v___x_194_ = lean_unsigned_to_nat(1u);
v___x_195_ = lean_nat_add(v_stackPos_152_, v___x_194_);
lean_dec(v_stackPos_152_);
v_nextStackPos_196_ = l_String_Slice_posGE___redArg(v_s_102_, v___x_195_);
lean_inc(v_nextStackPos_196_);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 3, v___x_176_);
lean_ctor_set(v___x_155_, 2, v_nextStackPos_196_);
v___x_198_ = v___x_155_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_needle_150_);
lean_ctor_set(v_reuseFailAlloc_199_, 1, v_table_151_);
lean_ctor_set(v_reuseFailAlloc_199_, 2, v_nextStackPos_196_);
lean_ctor_set(v_reuseFailAlloc_199_, 3, v___x_176_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
v_it_107_ = v___x_198_;
v_startPos_108_ = v_basePos_193_;
v_endPos_109_ = v_nextStackPos_196_;
goto v___jp_106_;
}
}
}
else
{
lean_object* v___x_200_; lean_object* v_nextStackPos_201_; lean_object* v_nextNeedlePos_202_; uint8_t v___x_203_; 
lean_dec(v_basePos_163_);
v___x_200_ = lean_unsigned_to_nat(1u);
v_nextStackPos_201_ = lean_nat_add(v_stackPos_152_, v___x_200_);
lean_dec(v_stackPos_152_);
v_nextNeedlePos_202_ = lean_nat_add(v_needlePos_153_, v___x_200_);
lean_dec(v_needlePos_153_);
v___x_203_ = lean_nat_dec_eq(v_nextNeedlePos_202_, v___x_164_);
lean_dec(v___x_164_);
if (v___x_203_ == 0)
{
lean_object* v___x_205_; 
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 3, v_nextNeedlePos_202_);
lean_ctor_set(v___x_155_, 2, v_nextStackPos_201_);
v___x_205_ = v___x_155_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_needle_150_);
lean_ctor_set(v_reuseFailAlloc_207_, 1, v_table_151_);
lean_ctor_set(v_reuseFailAlloc_207_, 2, v_nextStackPos_201_);
lean_ctor_set(v_reuseFailAlloc_207_, 3, v_nextNeedlePos_202_);
v___x_205_ = v_reuseFailAlloc_207_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
v_a_104_ = v___x_205_;
goto _start;
}
}
else
{
lean_object* v___x_208_; lean_object* v___x_210_; 
lean_dec(v_nextNeedlePos_202_);
v___x_208_ = lean_unsigned_to_nat(0u);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 3, v___x_208_);
lean_ctor_set(v___x_155_, 2, v_nextStackPos_201_);
v___x_210_ = v___x_155_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_needle_150_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v_table_151_);
lean_ctor_set(v_reuseFailAlloc_211_, 2, v_nextStackPos_201_);
lean_ctor_set(v_reuseFailAlloc_211_, 3, v___x_208_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
v_it_118_ = v___x_210_;
goto v___jp_117_;
}
}
}
}
}
}
default: 
{
lean_dec_ref(v_s_102_);
return v_b_105_;
}
}
v___jp_106_:
{
lean_object* v___x_110_; lean_object* v_str_111_; lean_object* v_startInclusive_112_; lean_object* v_endExclusive_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
lean_inc_ref(v_s_102_);
v___x_110_ = l_String_Slice_slice_x21(v_s_102_, v_startPos_108_, v_endPos_109_);
lean_dec(v_endPos_109_);
lean_dec(v_startPos_108_);
v_str_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc_ref(v_str_111_);
v_startInclusive_112_ = lean_ctor_get(v___x_110_, 1);
lean_inc(v_startInclusive_112_);
v_endExclusive_113_ = lean_ctor_get(v___x_110_, 2);
lean_inc(v_endExclusive_113_);
lean_dec_ref(v___x_110_);
v___x_114_ = lean_string_utf8_extract_fast(v_str_111_, v_startInclusive_112_, v_endExclusive_113_);
lean_dec(v_endExclusive_113_);
lean_dec(v_startInclusive_112_);
lean_dec_ref(v_str_111_);
v___x_115_ = lean_string_append(v_b_105_, v___x_114_);
lean_dec_ref(v___x_114_);
v_a_104_ = v_it_107_;
v_b_105_ = v___x_115_;
goto _start;
}
v___jp_117_:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_119_ = lean_unsigned_to_nat(0u);
v___x_120_ = lean_string_utf8_byte_size(v_replacement_103_);
v___x_121_ = lean_string_utf8_extract_fast(v_replacement_103_, v___x_119_, v___x_120_);
v___x_122_ = lean_string_append(v_b_105_, v___x_121_);
lean_dec_ref(v___x_121_);
v_a_104_ = v_it_118_;
v_b_105_ = v___x_122_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg___boxed(lean_object* v_s_213_, lean_object* v_replacement_214_, lean_object* v_a_215_, lean_object* v_b_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_213_, v_replacement_214_, v_a_215_, v_b_216_);
lean_dec_ref(v_replacement_214_);
return v_res_217_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__0));
v___x_221_ = lean_string_utf8_byte_size(v___x_220_);
return v___x_221_;
}
}
static uint8_t _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; uint8_t v___x_224_; 
v___x_222_ = lean_unsigned_to_nat(0u);
v___x_223_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2);
v___x_224_ = lean_nat_dec_eq(v___x_223_, v___x_222_);
return v___x_224_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_225_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__2);
v___x_226_ = lean_unsigned_to_nat(0u);
v___x_227_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__0));
v___x_228_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_228_, 0, v___x_227_);
lean_ctor_set(v___x_228_, 1, v___x_226_);
lean_ctor_set(v___x_228_, 2, v___x_225_);
return v___x_228_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4);
v___x_230_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_229_);
return v___x_230_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6(void){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_231_ = lean_unsigned_to_nat(0u);
v___x_232_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__5);
v___x_233_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__4);
v___x_234_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
lean_ctor_set(v___x_234_, 2, v___x_231_);
lean_ctor_set(v___x_234_, 3, v___x_231_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(lean_object* v_s_237_, lean_object* v_replacement_238_){
_start:
{
lean_object* v___x_239_; uint8_t v___x_240_; 
v___x_239_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__1));
v___x_240_ = lean_uint8_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__3);
if (v___x_240_ == 0)
{
lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_241_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__6);
v___x_242_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_237_, v_replacement_238_, v___x_241_, v___x_239_);
return v___x_242_;
}
else
{
lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_243_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__7));
v___x_244_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_237_, v_replacement_238_, v___x_243_, v___x_239_);
return v___x_244_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___boxed(lean_object* v_s_245_, lean_object* v_replacement_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(v_s_245_, v_replacement_246_);
lean_dec_ref(v_replacement_246_);
return v_res_247_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_249_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__0));
v___x_250_ = lean_string_utf8_byte_size(v___x_249_);
return v___x_250_;
}
}
static uint8_t _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_251_ = lean_unsigned_to_nat(0u);
v___x_252_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1);
v___x_253_ = lean_nat_dec_eq(v___x_252_, v___x_251_);
return v___x_253_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_254_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__1);
v___x_255_ = lean_unsigned_to_nat(0u);
v___x_256_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__0));
v___x_257_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v___x_255_);
lean_ctor_set(v___x_257_, 2, v___x_254_);
return v___x_257_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4(void){
_start:
{
lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_258_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3);
v___x_259_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_258_);
return v___x_259_;
}
}
static lean_object* _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_260_ = lean_unsigned_to_nat(0u);
v___x_261_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__4);
v___x_262_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__3);
v___x_263_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v___x_261_);
lean_ctor_set(v___x_263_, 2, v___x_260_);
lean_ctor_set(v___x_263_, 3, v___x_260_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(lean_object* v_s_264_, lean_object* v_replacement_265_){
_start:
{
lean_object* v___x_266_; uint8_t v___x_267_; 
v___x_266_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__1));
v___x_267_ = lean_uint8_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__2);
if (v___x_267_ == 0)
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = lean_obj_once(&lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5, &lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5_once, _init_lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___closed__5);
v___x_269_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_264_, v_replacement_265_, v___x_268_, v___x_266_);
return v___x_269_;
}
else
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = ((lean_object*)(lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg___closed__7));
v___x_271_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_264_, v_replacement_265_, v___x_270_, v___x_266_);
return v___x_271_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg___boxed(lean_object* v_s_272_, lean_object* v_replacement_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(v_s_272_, v_replacement_273_);
lean_dec_ref(v_replacement_273_);
return v_res_274_;
}
}
static lean_object* _init_lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0(void){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_275_ = lean_box(1);
v___x_276_ = l_Lean_MessageData_ofFormat(v___x_275_);
return v___x_276_;
}
}
static lean_object* _init_lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3(void){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_280_ = ((lean_object*)(lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__2));
v___x_281_ = l_Lean_MessageData_ofFormat(v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8(lean_object* v_x_282_, lean_object* v_x_283_){
_start:
{
if (lean_obj_tag(v_x_283_) == 0)
{
return v_x_282_;
}
else
{
lean_object* v_head_284_; lean_object* v_tail_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_307_; 
v_head_284_ = lean_ctor_get(v_x_283_, 0);
v_tail_285_ = lean_ctor_get(v_x_283_, 1);
v_isSharedCheck_307_ = !lean_is_exclusive(v_x_283_);
if (v_isSharedCheck_307_ == 0)
{
v___x_287_ = v_x_283_;
v_isShared_288_ = v_isSharedCheck_307_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_tail_285_);
lean_inc(v_head_284_);
lean_dec(v_x_283_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_307_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v_before_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_305_; 
v_before_289_ = lean_ctor_get(v_head_284_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v_head_284_);
if (v_isSharedCheck_305_ == 0)
{
lean_object* v_unused_306_; 
v_unused_306_ = lean_ctor_get(v_head_284_, 1);
lean_dec(v_unused_306_);
v___x_291_ = v_head_284_;
v_isShared_292_ = v_isSharedCheck_305_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_before_289_);
lean_dec(v_head_284_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_305_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v___x_293_; lean_object* v___x_295_; 
v___x_293_ = lean_obj_once(&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0, &lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0_once, _init_lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0);
if (v_isShared_292_ == 0)
{
lean_ctor_set_tag(v___x_291_, 7);
lean_ctor_set(v___x_291_, 1, v___x_293_);
lean_ctor_set(v___x_291_, 0, v_x_282_);
v___x_295_ = v___x_291_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_x_282_);
lean_ctor_set(v_reuseFailAlloc_304_, 1, v___x_293_);
v___x_295_ = v_reuseFailAlloc_304_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
lean_object* v___x_296_; lean_object* v___x_298_; 
v___x_296_ = lean_obj_once(&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3, &lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3_once, _init_lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__3);
if (v_isShared_288_ == 0)
{
lean_ctor_set_tag(v___x_287_, 7);
lean_ctor_set(v___x_287_, 1, v___x_296_);
lean_ctor_set(v___x_287_, 0, v___x_295_);
v___x_298_ = v___x_287_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_303_; 
v_reuseFailAlloc_303_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_303_, 0, v___x_295_);
lean_ctor_set(v_reuseFailAlloc_303_, 1, v___x_296_);
v___x_298_ = v_reuseFailAlloc_303_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_299_ = l_Lean_MessageData_ofSyntax(v_before_289_);
v___x_300_ = l_Lean_indentD(v___x_299_);
v___x_301_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_298_);
lean_ctor_set(v___x_301_, 1, v___x_300_);
v_x_282_ = v___x_301_;
v_x_283_ = v_tail_285_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7(lean_object* v_opts_308_, lean_object* v_opt_309_){
_start:
{
lean_object* v_name_310_; lean_object* v_defValue_311_; lean_object* v_map_312_; lean_object* v___x_313_; 
v_name_310_ = lean_ctor_get(v_opt_309_, 0);
v_defValue_311_ = lean_ctor_get(v_opt_309_, 1);
v_map_312_ = lean_ctor_get(v_opts_308_, 0);
v___x_313_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_312_, v_name_310_);
if (lean_obj_tag(v___x_313_) == 0)
{
uint8_t v___x_314_; 
v___x_314_ = lean_unbox(v_defValue_311_);
return v___x_314_;
}
else
{
lean_object* v_val_315_; 
v_val_315_ = lean_ctor_get(v___x_313_, 0);
lean_inc(v_val_315_);
lean_dec_ref_known(v___x_313_, 1);
if (lean_obj_tag(v_val_315_) == 1)
{
uint8_t v_v_316_; 
v_v_316_ = lean_ctor_get_uint8(v_val_315_, 0);
lean_dec_ref_known(v_val_315_, 0);
return v_v_316_;
}
else
{
uint8_t v___x_317_; 
lean_dec(v_val_315_);
v___x_317_ = lean_unbox(v_defValue_311_);
return v___x_317_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7___boxed(lean_object* v_opts_318_, lean_object* v_opt_319_){
_start:
{
uint8_t v_res_320_; lean_object* v_r_321_; 
v_res_320_ = lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7(v_opts_318_, v_opt_319_);
lean_dec_ref(v_opt_319_);
lean_dec_ref(v_opts_318_);
v_r_321_ = lean_box(v_res_320_);
return v_r_321_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = ((lean_object*)(lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__1));
v___x_326_ = l_Lean_MessageData_ofFormat(v___x_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg(lean_object* v_msgData_327_, lean_object* v_macroStack_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_options_331_; lean_object* v___x_332_; uint8_t v___x_333_; 
v_options_331_ = lean_ctor_get(v___y_329_, 2);
v___x_332_ = l_Lean_Elab_pp_macroStack;
v___x_333_ = lp_proofwidgets_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__7(v_options_331_, v___x_332_);
if (v___x_333_ == 0)
{
lean_object* v___x_334_; 
lean_dec(v_macroStack_328_);
v___x_334_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_334_, 0, v_msgData_327_);
return v___x_334_;
}
else
{
if (lean_obj_tag(v_macroStack_328_) == 0)
{
lean_object* v___x_335_; 
v___x_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_335_, 0, v_msgData_327_);
return v___x_335_;
}
else
{
lean_object* v_head_336_; lean_object* v_after_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_352_; 
v_head_336_ = lean_ctor_get(v_macroStack_328_, 0);
lean_inc(v_head_336_);
v_after_337_ = lean_ctor_get(v_head_336_, 1);
v_isSharedCheck_352_ = !lean_is_exclusive(v_head_336_);
if (v_isSharedCheck_352_ == 0)
{
lean_object* v_unused_353_; 
v_unused_353_ = lean_ctor_get(v_head_336_, 0);
lean_dec(v_unused_353_);
v___x_339_ = v_head_336_;
v_isShared_340_ = v_isSharedCheck_352_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_after_337_);
lean_dec(v_head_336_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_352_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_341_; lean_object* v___x_343_; 
v___x_341_ = lean_obj_once(&lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0, &lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0_once, _init_lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8___closed__0);
if (v_isShared_340_ == 0)
{
lean_ctor_set_tag(v___x_339_, 7);
lean_ctor_set(v___x_339_, 1, v___x_341_);
lean_ctor_set(v___x_339_, 0, v_msgData_327_);
v___x_343_ = v___x_339_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v_msgData_327_);
lean_ctor_set(v_reuseFailAlloc_351_, 1, v___x_341_);
v___x_343_ = v_reuseFailAlloc_351_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v_msgData_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_344_ = lean_obj_once(&lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2, &lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2_once, _init_lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___closed__2);
v___x_345_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_343_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = l_Lean_MessageData_ofSyntax(v_after_337_);
v___x_347_ = l_Lean_indentD(v___x_346_);
v_msgData_348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_348_, 0, v___x_345_);
lean_ctor_set(v_msgData_348_, 1, v___x_347_);
v___x_349_ = lp_proofwidgets_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6_spec__8(v_msgData_348_, v_macroStack_328_);
v___x_350_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
return v___x_350_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg___boxed(lean_object* v_msgData_354_, lean_object* v_macroStack_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg(v_msgData_354_, v_macroStack_355_, v___y_356_);
lean_dec_ref(v___y_356_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5(lean_object* v_msgData_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v___x_365_; lean_object* v_env_366_; lean_object* v___x_367_; lean_object* v_mctx_368_; lean_object* v_lctx_369_; lean_object* v_options_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_365_ = lean_st_ref_get(v___y_363_);
v_env_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc_ref(v_env_366_);
lean_dec(v___x_365_);
v___x_367_ = lean_st_ref_get(v___y_361_);
v_mctx_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc_ref(v_mctx_368_);
lean_dec(v___x_367_);
v_lctx_369_ = lean_ctor_get(v___y_360_, 2);
v_options_370_ = lean_ctor_get(v___y_362_, 2);
lean_inc_ref(v_options_370_);
lean_inc_ref(v_lctx_369_);
v___x_371_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_371_, 0, v_env_366_);
lean_ctor_set(v___x_371_, 1, v_mctx_368_);
lean_ctor_set(v___x_371_, 2, v_lctx_369_);
lean_ctor_set(v___x_371_, 3, v_options_370_);
v___x_372_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v_msgData_359_);
v___x_373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5___boxed(lean_object* v_msgData_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5(v_msgData_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
lean_dec_ref(v___y_375_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(lean_object* v_msg_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_){
_start:
{
lean_object* v_ref_389_; lean_object* v___x_390_; lean_object* v_a_391_; lean_object* v_macroStack_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v_a_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_403_; 
v_ref_389_ = lean_ctor_get(v___y_386_, 5);
v___x_390_ = lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__5(v_msg_381_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
v_a_391_ = lean_ctor_get(v___x_390_, 0);
lean_inc(v_a_391_);
lean_dec_ref(v___x_390_);
v_macroStack_392_ = lean_ctor_get(v___y_382_, 1);
v___x_393_ = l_Lean_Elab_getBetterRef(v_ref_389_, v_macroStack_392_);
lean_inc(v_macroStack_392_);
v___x_394_ = lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg(v_a_391_, v_macroStack_392_, v___y_386_);
v_a_395_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_403_ == 0)
{
v___x_397_ = v___x_394_;
v_isShared_398_ = v_isSharedCheck_403_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_a_395_);
lean_dec(v___x_394_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_403_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_399_; lean_object* v___x_401_; 
v___x_399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_393_);
lean_ctor_set(v___x_399_, 1, v_a_395_);
if (v_isShared_398_ == 0)
{
lean_ctor_set_tag(v___x_397_, 1);
lean_ctor_set(v___x_397_, 0, v___x_399_);
v___x_401_ = v___x_397_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v___x_399_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg___boxed(lean_object* v_msg_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(v_msg_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec(v___y_408_);
lean_dec_ref(v___y_407_);
lean_dec(v___y_406_);
lean_dec_ref(v___y_405_);
return v_res_412_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0(void){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0));
v___x_414_ = lean_string_utf8_byte_size(v___x_413_);
return v___x_414_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_415_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__0);
v___x_416_ = lean_unsigned_to_nat(0u);
v___x_417_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_ofRpcMethodTemplate___closed__0));
v___x_418_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
lean_ctor_set(v___x_418_, 1, v___x_416_);
lean_ctor_set(v___x_418_, 2, v___x_415_);
return v___x_418_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9(void){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = l_Array_mkArray0(lean_box(0));
return v___x_427_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14(void){
_start:
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__13));
v___x_433_ = l_String_toRawSubstring_x27(v___x_432_);
return v___x_433_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_443_ = lean_box(0);
v___x_444_ = l_Lean_Level_succ___override(v___x_443_);
return v___x_444_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24(void){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_445_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__23);
v___x_446_ = l_Lean_Expr_sort___override(v___x_445_);
return v___x_446_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_447_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__24);
v___x_448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_448_, 0, v___x_447_);
return v___x_448_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36(void){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_467_ = lean_box(0);
v___x_468_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__35));
v___x_469_ = l_Lean_Expr_const___override(v___x_468_, v___x_467_);
return v___x_469_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_470_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__36);
v___x_471_ = lean_unsigned_to_nat(1u);
v___x_472_ = lean_mk_empty_array_with_capacity(v___x_471_);
v___x_473_ = lean_array_push(v___x_472_, v___x_470_);
return v___x_473_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41(void){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_480_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__40));
v___x_481_ = l_Lean_stringToMessageData(v___x_480_);
return v___x_481_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__42));
v___x_484_ = l_Lean_stringToMessageData(v___x_483_);
return v___x_484_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_486_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__44));
v___x_487_ = l_Lean_stringToMessageData(v___x_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0(lean_object* v_stx_488_, lean_object* v_expectedType_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v___x_497_; uint8_t v___x_498_; lean_object* v___y_500_; lean_object* v___y_501_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; uint8_t v___y_508_; uint8_t v___y_563_; lean_object* v___y_564_; lean_object* v___y_565_; lean_object* v___y_566_; lean_object* v___y_567_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_625_; lean_object* v___y_626_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v___y_629_; lean_object* v___y_630_; uint8_t v___y_631_; lean_object* v___y_632_; lean_object* v___y_633_; lean_object* v___y_634_; lean_object* v___y_635_; lean_object* v___y_636_; uint8_t v___y_637_; 
v___x_497_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_termMk__rpc__widget_x25___00__closed__2));
lean_inc(v_stx_488_);
v___x_498_ = l_Lean_Syntax_isOfKind(v_stx_488_, v___x_497_);
if (v___x_498_ == 0)
{
lean_object* v___x_655_; 
lean_dec_ref(v_expectedType_489_);
lean_dec(v_stx_488_);
v___x_655_ = lp_proofwidgets_Lean_Elab_throwUnsupportedSyntax___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__0___redArg();
return v___x_655_;
}
else
{
lean_object* v___x_656_; uint8_t v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_656_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__25);
v___x_657_ = 0;
v___x_658_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__27));
v___x_659_ = l_Lean_Meta_mkFreshExprMVar(v___x_656_, v___x_657_, v___x_658_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
if (lean_obj_tag(v___x_659_) == 0)
{
lean_object* v_a_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; 
v_a_660_ = lean_ctor_get(v___x_659_, 0);
lean_inc_n(v_a_660_, 2);
lean_dec_ref_known(v___x_659_, 1);
v___x_661_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__29));
v___x_662_ = lean_unsigned_to_nat(1u);
v___x_663_ = lean_mk_empty_array_with_capacity(v___x_662_);
lean_inc_ref(v___x_663_);
v___x_664_ = lean_array_push(v___x_663_, v_a_660_);
v___x_665_ = l_Lean_Meta_mkAppM(v___x_661_, v___x_664_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v_a_666_; lean_object* v___x_667_; 
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc_n(v_a_666_, 2);
lean_dec_ref_known(v___x_665_, 1);
lean_inc_ref(v_expectedType_489_);
v___x_667_ = l_Lean_Meta_isExprDefEq(v_expectedType_489_, v_a_666_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
if (lean_obj_tag(v___x_667_) == 0)
{
lean_object* v_a_668_; lean_object* v___x_669_; lean_object* v___y_671_; lean_object* v___y_672_; lean_object* v___y_673_; lean_object* v___y_674_; lean_object* v___y_675_; lean_object* v___y_676_; uint8_t v___x_756_; 
v_a_668_ = lean_ctor_get(v___x_667_, 0);
lean_inc(v_a_668_);
lean_dec_ref_known(v___x_667_, 1);
v___x_669_ = l_Lean_Syntax_getArg(v_stx_488_, v___x_662_);
lean_dec(v_stx_488_);
v___x_756_ = lean_unbox(v_a_668_);
lean_dec(v_a_668_);
if (v___x_756_ == 0)
{
if (v___x_498_ == 0)
{
lean_dec(v_a_666_);
v___y_671_ = v___y_490_;
v___y_672_ = v___y_491_;
v___y_673_ = v___y_492_;
v___y_674_ = v___y_493_;
v___y_675_ = v___y_494_;
v___y_676_ = v___y_495_;
goto v___jp_670_;
}
else
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v_a_767_; lean_object* v___x_769_; uint8_t v_isShared_770_; uint8_t v_isSharedCheck_774_; 
lean_dec(v___x_669_);
lean_dec_ref(v___x_663_);
lean_dec(v_a_660_);
v___x_757_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__43);
v___x_758_ = l_Lean_MessageData_ofExpr(v_expectedType_489_);
v___x_759_ = l_Lean_indentD(v___x_758_);
v___x_760_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_757_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__45);
v___x_762_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_762_, 0, v___x_760_);
lean_ctor_set(v___x_762_, 1, v___x_761_);
v___x_763_ = l_Lean_MessageData_ofExpr(v_a_666_);
v___x_764_ = l_Lean_indentD(v___x_763_);
v___x_765_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_762_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
v___x_766_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(v___x_765_, v___y_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
v_a_767_ = lean_ctor_get(v___x_766_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_766_);
if (v_isSharedCheck_774_ == 0)
{
v___x_769_ = v___x_766_;
v_isShared_770_ = v_isSharedCheck_774_;
goto v_resetjp_768_;
}
else
{
lean_inc(v_a_767_);
lean_dec(v___x_766_);
v___x_769_ = lean_box(0);
v_isShared_770_ = v_isSharedCheck_774_;
goto v_resetjp_768_;
}
v_resetjp_768_:
{
lean_object* v___x_772_; 
if (v_isShared_770_ == 0)
{
v___x_772_ = v___x_769_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v_a_767_);
v___x_772_ = v_reuseFailAlloc_773_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
return v___x_772_;
}
}
}
}
else
{
lean_dec(v_a_666_);
v___y_671_ = v___y_490_;
v___y_672_ = v___y_491_;
v___y_673_ = v___y_492_;
v___y_674_ = v___y_493_;
v___y_675_ = v___y_494_;
v___y_676_ = v___y_495_;
goto v___jp_670_;
}
v___jp_670_:
{
lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_677_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__30));
v___x_678_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__33));
v___x_679_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__37);
v___x_680_ = l_Lean_Meta_mkAppM(v___x_678_, v___x_679_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
if (lean_obj_tag(v___x_680_) == 0)
{
lean_object* v_a_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v_a_681_ = lean_ctor_get(v___x_680_, 0);
lean_inc(v_a_681_);
lean_dec_ref_known(v___x_680_, 1);
v___x_682_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__39));
v___x_683_ = lean_array_push(v___x_663_, v_a_681_);
v___x_684_ = l_Lean_Meta_mkAppM(v___x_682_, v___x_683_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
if (lean_obj_tag(v___x_684_) == 0)
{
lean_object* v_a_685_; lean_object* v___x_686_; 
v_a_685_ = lean_ctor_get(v___x_684_, 0);
lean_inc(v_a_685_);
lean_dec_ref_known(v___x_684_, 1);
v___x_686_ = l_Lean_mkArrow(v_a_660_, v_a_685_, v___y_675_, v___y_676_);
if (lean_obj_tag(v___x_686_) == 0)
{
lean_object* v_a_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_755_; 
v_a_687_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_755_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_755_ == 0)
{
v___x_689_ = v___x_686_;
v_isShared_690_ = v_isSharedCheck_755_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_a_687_);
lean_dec(v___x_686_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_755_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v___x_692_; 
if (v_isShared_690_ == 0)
{
lean_ctor_set_tag(v___x_689_, 1);
v___x_692_ = v___x_689_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v_a_687_);
v___x_692_ = v_reuseFailAlloc_754_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_693_ = lean_box(0);
v___x_694_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_669_, v___x_692_, v___x_498_, v___x_498_, v___x_693_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
if (lean_obj_tag(v___x_694_) == 0)
{
lean_object* v_a_695_; lean_object* v___x_696_; lean_object* v_a_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_753_; 
v_a_695_ = lean_ctor_get(v___x_694_, 0);
lean_inc(v_a_695_);
lean_dec_ref_known(v___x_694_, 1);
v___x_696_ = lp_proofwidgets_Lean_instantiateMVars___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__1___redArg(v_a_695_, v___y_674_);
v_a_697_ = lean_ctor_get(v___x_696_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_696_);
if (v_isSharedCheck_753_ == 0)
{
v___x_699_ = v___x_696_;
v_isShared_700_ = v_isSharedCheck_753_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_a_697_);
lean_dec(v___x_696_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_753_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
if (lean_obj_tag(v_a_697_) == 4)
{
lean_object* v_declName_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
v_declName_701_ = lean_ctor_get(v_a_697_, 0);
lean_inc_n(v_declName_701_, 2);
lean_dec_ref_known(v_a_697_, 2);
v___x_702_ = lp_proofwidgets_ProofWidgets_cancellableSuffix;
v___x_703_ = l_Lean_Name_append(v_declName_701_, v___x_702_);
v___x_704_ = l_Lean_Server_existsBuiltinRpcProcedure(v___x_703_);
if (lean_obj_tag(v___x_704_) == 0)
{
lean_object* v_a_705_; lean_object* v___x_706_; uint8_t v___x_707_; 
v_a_705_ = lean_ctor_get(v___x_704_, 0);
lean_inc(v_a_705_);
lean_dec_ref_known(v___x_704_, 1);
v___x_706_ = lean_st_ref_get(v___y_676_);
v___x_707_ = lean_unbox(v_a_705_);
lean_dec(v_a_705_);
if (v___x_707_ == 0)
{
lean_object* v_env_708_; lean_object* v___x_709_; lean_object* v___x_710_; uint8_t v___x_711_; 
v_env_708_ = lean_ctor_get(v___x_706_, 0);
lean_inc_ref(v_env_708_);
lean_dec(v___x_706_);
v___x_709_ = lean_box(0);
v___x_710_ = l_Lean_Server_userRpcProcedures;
lean_inc(v___x_703_);
v___x_711_ = l_Lean_MapDeclarationExtension_contains___redArg(v___x_709_, v___x_710_, v_env_708_, v___x_703_);
if (v___x_711_ == 0)
{
lean_object* v___x_712_; 
lean_dec(v___x_703_);
v___x_712_ = l_Lean_Server_existsBuiltinRpcProcedure(v_declName_701_);
if (lean_obj_tag(v___x_712_) == 0)
{
lean_object* v_a_713_; lean_object* v___x_714_; uint8_t v___x_715_; 
lean_del_object(v___x_699_);
v_a_713_ = lean_ctor_get(v___x_712_, 0);
lean_inc(v_a_713_);
lean_dec_ref_known(v___x_712_, 1);
v___x_714_ = lean_st_ref_get(v___y_676_);
v___x_715_ = lean_unbox(v_a_713_);
lean_dec(v_a_713_);
if (v___x_715_ == 0)
{
lean_object* v_env_716_; 
v_env_716_ = lean_ctor_get(v___x_714_, 0);
lean_inc_ref(v_env_716_);
lean_dec(v___x_714_);
v___y_625_ = v_env_716_;
v___y_626_ = v___y_671_;
v___y_627_ = v___y_672_;
v___y_628_ = v_declName_701_;
v___y_629_ = v___x_710_;
v___y_630_ = v___y_676_;
v___y_631_ = v___x_711_;
v___y_632_ = v___y_675_;
v___y_633_ = v___y_673_;
v___y_634_ = v___y_674_;
v___y_635_ = v___x_677_;
v___y_636_ = v___x_709_;
v___y_637_ = v___x_498_;
goto v___jp_624_;
}
else
{
lean_object* v_env_717_; 
v_env_717_ = lean_ctor_get(v___x_714_, 0);
lean_inc_ref(v_env_717_);
lean_dec(v___x_714_);
v___y_625_ = v_env_717_;
v___y_626_ = v___y_671_;
v___y_627_ = v___y_672_;
v___y_628_ = v_declName_701_;
v___y_629_ = v___x_710_;
v___y_630_ = v___y_676_;
v___y_631_ = v___x_711_;
v___y_632_ = v___y_675_;
v___y_633_ = v___y_673_;
v___y_634_ = v___y_674_;
v___y_635_ = v___x_677_;
v___y_636_ = v___x_709_;
v___y_637_ = v___x_711_;
goto v___jp_624_;
}
}
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_732_; 
lean_dec(v_declName_701_);
lean_dec_ref(v_expectedType_489_);
v_a_718_ = lean_ctor_get(v___x_712_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v___x_712_);
if (v_isSharedCheck_732_ == 0)
{
v___x_720_ = v___x_712_;
v_isShared_721_ = v_isSharedCheck_732_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_712_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_732_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v_ref_722_; lean_object* v___x_723_; lean_object* v___x_725_; 
v_ref_722_ = lean_ctor_get(v___y_675_, 5);
v___x_723_ = lean_io_error_to_string(v_a_718_);
if (v_isShared_700_ == 0)
{
lean_ctor_set_tag(v___x_699_, 3);
lean_ctor_set(v___x_699_, 0, v___x_723_);
v___x_725_ = v___x_699_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v___x_723_);
v___x_725_ = v_reuseFailAlloc_731_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_729_; 
v___x_726_ = l_Lean_MessageData_ofFormat(v___x_725_);
lean_inc(v_ref_722_);
v___x_727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_727_, 0, v_ref_722_);
lean_ctor_set(v___x_727_, 1, v___x_726_);
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 0, v___x_727_);
v___x_729_ = v___x_720_;
goto v_reusejp_728_;
}
else
{
lean_object* v_reuseFailAlloc_730_; 
v_reuseFailAlloc_730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_730_, 0, v___x_727_);
v___x_729_ = v_reuseFailAlloc_730_;
goto v_reusejp_728_;
}
v_reusejp_728_:
{
return v___x_729_;
}
}
}
}
}
else
{
lean_dec(v_declName_701_);
lean_del_object(v___x_699_);
v___y_500_ = v___y_675_;
v___y_501_ = v___x_703_;
v___y_502_ = v___y_672_;
v___y_503_ = v___y_671_;
v___y_504_ = v___y_673_;
v___y_505_ = v___y_674_;
v___y_506_ = v___x_677_;
v___y_507_ = v___y_676_;
v___y_508_ = v___x_711_;
goto v___jp_499_;
}
}
else
{
lean_dec(v___x_706_);
lean_dec(v_declName_701_);
lean_del_object(v___x_699_);
v___y_500_ = v___y_675_;
v___y_501_ = v___x_703_;
v___y_502_ = v___y_672_;
v___y_503_ = v___y_671_;
v___y_504_ = v___y_673_;
v___y_505_ = v___y_674_;
v___y_506_ = v___x_677_;
v___y_507_ = v___y_676_;
v___y_508_ = v___x_498_;
goto v___jp_499_;
}
}
else
{
lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_747_; 
lean_dec(v___x_703_);
lean_dec(v_declName_701_);
lean_dec_ref(v_expectedType_489_);
v_a_733_ = lean_ctor_get(v___x_704_, 0);
v_isSharedCheck_747_ = !lean_is_exclusive(v___x_704_);
if (v_isSharedCheck_747_ == 0)
{
v___x_735_ = v___x_704_;
v_isShared_736_ = v_isSharedCheck_747_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_704_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_747_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v_ref_737_; lean_object* v___x_738_; lean_object* v___x_740_; 
v_ref_737_ = lean_ctor_get(v___y_675_, 5);
v___x_738_ = lean_io_error_to_string(v_a_733_);
if (v_isShared_700_ == 0)
{
lean_ctor_set_tag(v___x_699_, 3);
lean_ctor_set(v___x_699_, 0, v___x_738_);
v___x_740_ = v___x_699_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v___x_738_);
v___x_740_ = v_reuseFailAlloc_746_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_744_; 
v___x_741_ = l_Lean_MessageData_ofFormat(v___x_740_);
lean_inc(v_ref_737_);
v___x_742_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_742_, 0, v_ref_737_);
lean_ctor_set(v___x_742_, 1, v___x_741_);
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 0, v___x_742_);
v___x_744_ = v___x_735_;
goto v_reusejp_743_;
}
else
{
lean_object* v_reuseFailAlloc_745_; 
v_reuseFailAlloc_745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_745_, 0, v___x_742_);
v___x_744_ = v_reuseFailAlloc_745_;
goto v_reusejp_743_;
}
v_reusejp_743_:
{
return v___x_744_;
}
}
}
}
}
else
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; 
lean_del_object(v___x_699_);
lean_dec_ref(v_expectedType_489_);
v___x_748_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__41);
v___x_749_ = l_Lean_MessageData_ofExpr(v_a_697_);
v___x_750_ = l_Lean_indentD(v___x_749_);
v___x_751_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_748_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v___x_752_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(v___x_751_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
return v___x_752_;
}
}
}
else
{
lean_dec_ref(v_expectedType_489_);
return v___x_694_;
}
}
}
}
else
{
lean_dec(v___x_669_);
lean_dec_ref(v_expectedType_489_);
return v___x_686_;
}
}
else
{
lean_dec(v___x_669_);
lean_dec(v_a_660_);
lean_dec_ref(v_expectedType_489_);
return v___x_684_;
}
}
else
{
lean_dec(v___x_669_);
lean_dec_ref(v___x_663_);
lean_dec(v_a_660_);
lean_dec_ref(v_expectedType_489_);
return v___x_680_;
}
}
}
else
{
lean_object* v_a_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_782_; 
lean_dec(v_a_666_);
lean_dec_ref(v___x_663_);
lean_dec(v_a_660_);
lean_dec_ref(v_expectedType_489_);
lean_dec(v_stx_488_);
v_a_775_ = lean_ctor_get(v___x_667_, 0);
v_isSharedCheck_782_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_782_ == 0)
{
v___x_777_ = v___x_667_;
v_isShared_778_ = v_isSharedCheck_782_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_a_775_);
lean_dec(v___x_667_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_782_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v___x_780_; 
if (v_isShared_778_ == 0)
{
v___x_780_ = v___x_777_;
goto v_reusejp_779_;
}
else
{
lean_object* v_reuseFailAlloc_781_; 
v_reuseFailAlloc_781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_781_, 0, v_a_775_);
v___x_780_ = v_reuseFailAlloc_781_;
goto v_reusejp_779_;
}
v_reusejp_779_:
{
return v___x_780_;
}
}
}
}
else
{
lean_dec_ref(v___x_663_);
lean_dec(v_a_660_);
lean_dec_ref(v_expectedType_489_);
lean_dec(v_stx_488_);
return v___x_665_;
}
}
else
{
lean_dec_ref(v_expectedType_489_);
lean_dec(v_stx_488_);
return v___x_659_;
}
}
v___jp_499_:
{
lean_object* v_ref_509_; lean_object* v_quotContext_510_; lean_object* v_currMacroScope_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; uint8_t v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v_ref_509_ = lean_ctor_get(v___y_500_, 5);
v_quotContext_510_ = lean_ctor_get(v___y_500_, 10);
v_currMacroScope_511_ = lean_ctor_get(v___y_500_, 11);
v___x_512_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___y_501_, v___y_508_);
v___x_513_ = lean_unsigned_to_nat(0u);
v___x_514_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1);
v___x_515_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(v___x_514_, v___x_512_);
lean_dec_ref(v___x_512_);
v___x_516_ = lean_string_utf8_byte_size(v___x_515_);
v___x_517_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__2));
v___x_518_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_518_, 0, v___x_515_);
lean_ctor_set(v___x_518_, 1, v___x_513_);
lean_ctor_set(v___x_518_, 2, v___x_516_);
v___x_519_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(v___x_518_, v___x_517_);
v___x_520_ = lean_box(2);
v___x_521_ = l_Lean_Syntax_mkStrLit(v___x_519_, v___x_520_);
v___x_522_ = 0;
v___x_523_ = l_Lean_SourceInfo_fromRef(v_ref_509_, v___x_522_);
v___x_524_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__3));
v___x_525_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__4));
v___x_526_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__5));
lean_inc_ref_n(v___y_506_, 6);
v___x_527_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_526_);
v___x_528_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__6));
lean_inc_n(v___x_523_, 12);
v___x_529_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_529_, 0, v___x_523_);
lean_ctor_set(v___x_529_, 1, v___x_528_);
v___x_530_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__8));
v___x_531_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9);
v___x_532_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_532_, 0, v___x_523_);
lean_ctor_set(v___x_532_, 1, v___x_530_);
lean_ctor_set(v___x_532_, 2, v___x_531_);
v___x_533_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__10));
v___x_534_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_533_);
v___x_535_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__11));
v___x_536_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_535_);
v___x_537_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__12));
v___x_538_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_537_);
v___x_539_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14);
v___x_540_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__15));
lean_inc(v_currMacroScope_511_);
lean_inc(v_quotContext_510_);
v___x_541_ = l_Lean_addMacroScope(v_quotContext_510_, v___x_540_, v_currMacroScope_511_);
v___x_542_ = lean_box(0);
v___x_543_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_543_, 0, v___x_523_);
lean_ctor_set(v___x_543_, 1, v___x_539_);
lean_ctor_set(v___x_543_, 2, v___x_541_);
lean_ctor_set(v___x_543_, 3, v___x_542_);
lean_inc_ref_n(v___x_532_, 6);
v___x_544_ = l_Lean_Syntax_node2(v___x_523_, v___x_538_, v___x_543_, v___x_532_);
v___x_545_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__16));
v___x_546_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_545_);
v___x_547_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__17));
v___x_548_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_548_, 0, v___x_523_);
lean_ctor_set(v___x_548_, 1, v___x_547_);
v___x_549_ = l_Lean_Syntax_node3(v___x_523_, v___x_546_, v___x_548_, v___x_532_, v___x_521_);
v___x_550_ = l_Lean_Syntax_node3(v___x_523_, v___x_530_, v___x_532_, v___x_532_, v___x_549_);
v___x_551_ = l_Lean_Syntax_node2(v___x_523_, v___x_536_, v___x_544_, v___x_550_);
v___x_552_ = l_Lean_Syntax_node1(v___x_523_, v___x_530_, v___x_551_);
v___x_553_ = l_Lean_Syntax_node1(v___x_523_, v___x_534_, v___x_552_);
v___x_554_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__18));
v___x_555_ = l_Lean_Name_mkStr4(v___y_506_, v___x_524_, v___x_525_, v___x_554_);
v___x_556_ = l_Lean_Syntax_node1(v___x_523_, v___x_555_, v___x_532_);
v___x_557_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__19));
v___x_558_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_523_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
v___x_559_ = l_Lean_Syntax_node6(v___x_523_, v___x_527_, v___x_529_, v___x_532_, v___x_553_, v___x_556_, v___x_532_, v___x_558_);
v___x_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_560_, 0, v_expectedType_489_);
v___x_561_ = l_Lean_Elab_Term_elabTerm(v___x_559_, v___x_560_, v___x_498_, v___x_498_, v___y_503_, v___y_502_, v___y_504_, v___y_505_, v___y_500_, v___y_507_);
return v___x_561_;
}
v___jp_562_:
{
lean_object* v_ref_572_; lean_object* v_quotContext_573_; lean_object* v_currMacroScope_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v_ref_572_ = lean_ctor_get(v___y_570_, 5);
v_quotContext_573_ = lean_ctor_get(v___y_570_, 10);
v_currMacroScope_574_ = lean_ctor_get(v___y_570_, 11);
v___x_575_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___y_564_, v___x_498_);
v___x_576_ = lean_unsigned_to_nat(0u);
v___x_577_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__1);
v___x_578_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(v___x_577_, v___x_575_);
lean_dec_ref(v___x_575_);
v___x_579_ = lean_string_utf8_byte_size(v___x_578_);
v___x_580_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__20));
v___x_581_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_581_, 0, v___x_578_);
lean_ctor_set(v___x_581_, 1, v___x_576_);
lean_ctor_set(v___x_581_, 2, v___x_579_);
v___x_582_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(v___x_581_, v___x_580_);
v___x_583_ = lean_box(2);
v___x_584_ = l_Lean_Syntax_mkStrLit(v___x_582_, v___x_583_);
v___x_585_ = l_Lean_SourceInfo_fromRef(v_ref_572_, v___y_563_);
v___x_586_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__3));
v___x_587_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__4));
v___x_588_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__5));
lean_inc_ref_n(v___y_565_, 6);
v___x_589_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_588_);
v___x_590_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__6));
lean_inc_n(v___x_585_, 12);
v___x_591_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_591_, 0, v___x_585_);
lean_ctor_set(v___x_591_, 1, v___x_590_);
v___x_592_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__8));
v___x_593_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__9);
v___x_594_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_594_, 0, v___x_585_);
lean_ctor_set(v___x_594_, 1, v___x_592_);
lean_ctor_set(v___x_594_, 2, v___x_593_);
v___x_595_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__10));
v___x_596_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_595_);
v___x_597_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__11));
v___x_598_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_597_);
v___x_599_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__12));
v___x_600_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_599_);
v___x_601_ = lean_obj_once(&lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14, &lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14_once, _init_lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__14);
v___x_602_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__15));
lean_inc(v_currMacroScope_574_);
lean_inc(v_quotContext_573_);
v___x_603_ = l_Lean_addMacroScope(v_quotContext_573_, v___x_602_, v_currMacroScope_574_);
v___x_604_ = lean_box(0);
v___x_605_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_605_, 0, v___x_585_);
lean_ctor_set(v___x_605_, 1, v___x_601_);
lean_ctor_set(v___x_605_, 2, v___x_603_);
lean_ctor_set(v___x_605_, 3, v___x_604_);
lean_inc_ref_n(v___x_594_, 6);
v___x_606_ = l_Lean_Syntax_node2(v___x_585_, v___x_600_, v___x_605_, v___x_594_);
v___x_607_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__16));
v___x_608_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_607_);
v___x_609_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__17));
v___x_610_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_610_, 0, v___x_585_);
lean_ctor_set(v___x_610_, 1, v___x_609_);
v___x_611_ = l_Lean_Syntax_node3(v___x_585_, v___x_608_, v___x_610_, v___x_594_, v___x_584_);
v___x_612_ = l_Lean_Syntax_node3(v___x_585_, v___x_592_, v___x_594_, v___x_594_, v___x_611_);
v___x_613_ = l_Lean_Syntax_node2(v___x_585_, v___x_598_, v___x_606_, v___x_612_);
v___x_614_ = l_Lean_Syntax_node1(v___x_585_, v___x_592_, v___x_613_);
v___x_615_ = l_Lean_Syntax_node1(v___x_585_, v___x_596_, v___x_614_);
v___x_616_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__18));
v___x_617_ = l_Lean_Name_mkStr4(v___y_565_, v___x_586_, v___x_587_, v___x_616_);
v___x_618_ = l_Lean_Syntax_node1(v___x_585_, v___x_617_, v___x_594_);
v___x_619_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__19));
v___x_620_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_585_);
lean_ctor_set(v___x_620_, 1, v___x_619_);
v___x_621_ = l_Lean_Syntax_node6(v___x_585_, v___x_589_, v___x_591_, v___x_594_, v___x_615_, v___x_618_, v___x_594_, v___x_620_);
v___x_622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_622_, 0, v_expectedType_489_);
v___x_623_ = l_Lean_Elab_Term_elabTerm(v___x_621_, v___x_622_, v___x_498_, v___x_498_, v___y_566_, v___y_567_, v___y_568_, v___y_569_, v___y_570_, v___y_571_);
return v___x_623_;
}
v___jp_624_:
{
if (v___y_637_ == 0)
{
lean_dec(v___y_636_);
lean_dec_ref(v___y_625_);
v___y_563_ = v___y_631_;
v___y_564_ = v___y_628_;
v___y_565_ = v___y_635_;
v___y_566_ = v___y_626_;
v___y_567_ = v___y_627_;
v___y_568_ = v___y_633_;
v___y_569_ = v___y_634_;
v___y_570_ = v___y_632_;
v___y_571_ = v___y_630_;
goto v___jp_562_;
}
else
{
uint8_t v___x_638_; 
lean_inc(v___y_628_);
v___x_638_ = l_Lean_MapDeclarationExtension_contains___redArg(v___y_636_, v___y_629_, v___y_625_, v___y_628_);
if (v___x_638_ == 0)
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v_a_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_654_; 
lean_dec_ref(v_expectedType_489_);
v___x_639_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__21));
v___x_640_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___y_628_, v___y_637_);
v___x_641_ = lean_string_append(v___x_639_, v___x_640_);
lean_dec_ref(v___x_640_);
v___x_642_ = ((lean_object*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___closed__22));
v___x_643_ = lean_string_append(v___x_641_, v___x_642_);
v___x_644_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_644_, 0, v___x_643_);
v___x_645_ = l_Lean_MessageData_ofFormat(v___x_644_);
v___x_646_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(v___x_645_, v___y_626_, v___y_627_, v___y_633_, v___y_634_, v___y_632_, v___y_630_);
v_a_647_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_654_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_654_ == 0)
{
v___x_649_ = v___x_646_;
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_a_647_);
lean_dec(v___x_646_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v___x_652_; 
if (v_isShared_650_ == 0)
{
v___x_652_ = v___x_649_;
goto v_reusejp_651_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v_a_647_);
v___x_652_ = v_reuseFailAlloc_653_;
goto v_reusejp_651_;
}
v_reusejp_651_:
{
return v___x_652_;
}
}
}
else
{
v___y_563_ = v___y_631_;
v___y_564_ = v___y_628_;
v___y_565_ = v___y_635_;
v___y_566_ = v___y_626_;
v___y_567_ = v___y_627_;
v___y_568_ = v___y_633_;
v___y_569_ = v___y_634_;
v___y_570_ = v___y_632_;
v___y_571_ = v___y_630_;
goto v___jp_562_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___boxed(lean_object* v_stx_783_, lean_object* v_expectedType_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0(v_stx_783_, v_expectedType_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
lean_dec(v___y_788_);
lean_dec_ref(v___y_787_);
lean_dec(v___y_786_);
lean_dec_ref(v___y_785_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1(lean_object* v_stx_793_, lean_object* v_expectedType_x3f_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_){
_start:
{
lean_object* v___f_802_; lean_object* v___x_803_; 
v___f_802_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___lam__0___boxed), 9, 1);
lean_closure_set(v___f_802_, 0, v_stx_793_);
v___x_803_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_794_, v___f_802_, v_a_795_, v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_);
return v___x_803_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1___boxed(lean_object* v_stx_804_, lean_object* v_expectedType_x3f_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_){
_start:
{
lean_object* v_res_813_; 
v_res_813_ = lp_proofwidgets_ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1(v_stx_804_, v_expectedType_x3f_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_, v_a_810_, v_a_811_);
lean_dec(v_a_811_);
lean_dec_ref(v_a_810_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
return v_res_813_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2(lean_object* v_s_814_, lean_object* v_pattern_815_, lean_object* v_replacement_816_){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___redArg(v_s_814_, v_replacement_816_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2___boxed(lean_object* v_s_818_, lean_object* v_pattern_819_, lean_object* v_replacement_820_){
_start:
{
lean_object* v_res_821_; 
v_res_821_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2(v_s_818_, v_pattern_819_, v_replacement_820_);
lean_dec_ref(v_replacement_820_);
lean_dec_ref(v_pattern_819_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3(lean_object* v_s_822_, lean_object* v_pattern_823_, lean_object* v_replacement_824_){
_start:
{
lean_object* v___x_825_; 
v___x_825_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___redArg(v_s_822_, v_replacement_824_);
return v___x_825_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3___boxed(lean_object* v_s_826_, lean_object* v_pattern_827_, lean_object* v_replacement_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_proofwidgets_String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__3(v_s_826_, v_pattern_827_, v_replacement_828_);
lean_dec_ref(v_replacement_828_);
lean_dec_ref(v_pattern_827_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4(lean_object* v_00_u03b1_830_, lean_object* v_msg_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___redArg(v_msg_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4___boxed(lean_object* v_00_u03b1_840_, lean_object* v_msg_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_proofwidgets_Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4(v_00_u03b1_840_, v_msg_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___y_845_);
lean_dec_ref(v___y_844_);
lean_dec(v___y_843_);
lean_dec_ref(v___y_842_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2(lean_object* v_s_850_, lean_object* v_replacement_851_, lean_object* v_inst_852_, lean_object* v_R_853_, lean_object* v_a_854_, lean_object* v_b_855_, lean_object* v_c_856_){
_start:
{
lean_object* v___x_857_; 
v___x_857_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___redArg(v_s_850_, v_replacement_851_, v_a_854_, v_b_855_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2___boxed(lean_object* v_s_858_, lean_object* v_replacement_859_, lean_object* v_inst_860_, lean_object* v_R_861_, lean_object* v_a_862_, lean_object* v_b_863_, lean_object* v_c_864_){
_start:
{
lean_object* v_res_865_; 
v_res_865_ = lp_proofwidgets_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__2_spec__2(v_s_858_, v_replacement_859_, v_inst_860_, v_R_861_, v_a_862_, v_b_863_, v_c_864_);
lean_dec_ref(v_replacement_859_);
return v_res_865_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6(lean_object* v_msgData_866_, lean_object* v_macroStack_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___redArg(v_msgData_866_, v_macroStack_867_, v___y_872_);
return v___x_875_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6___boxed(lean_object* v_msgData_876_, lean_object* v_macroStack_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_proofwidgets_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00ProofWidgets___aux__ProofWidgets__Component__OfRpcMethod______elabRules__ProofWidgets__termMk__rpc__widget_x25____1_spec__4_spec__6(v_msgData_876_, v_macroStack_877_, v___y_878_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_);
lean_dec(v___y_883_);
lean_dec_ref(v___y_882_);
lean_dec(v___y_881_);
lean_dec_ref(v___y_880_);
lean_dec(v___y_879_);
lean_dec_ref(v___y_878_);
return v_res_885_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Cancellable(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Cancellable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Cancellable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Cancellable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
}
#ifdef __cplusplus
}
#endif
