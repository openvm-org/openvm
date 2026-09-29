// Lean compiler output
// Module: Plausible.DeriveShrinkable
// Imports: public import Init public meta import Init import Lean.Elab.Deriving.Basic import Lean.Elab.Deriving.Util import Plausible.Shrinkable
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
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_isInductiveCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Deriving_mkContext(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
extern lean_object* l_Lean_instInhabitedInductiveVal_default;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInductArgNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkImplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInductiveApp___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkInstImplicitBinders(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantInfo(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
extern lean_object* l_Lean_instInhabitedExpr;
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkLocalInstanceLetDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Deriving_mkLet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isInductiveCore(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Elab_registerDerivingHandler(lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "plausible"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__1_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "deriving"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__1_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__1_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__2_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "shrinkable"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__2_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__2_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__1_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(85, 3, 132, 224, 98, 236, 88, 42)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__2_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 135, 62, 108, 77, 135, 2, 101)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__4_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__4_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__4_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__5_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__4_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__5_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__5_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Plausible"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__7_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__5_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 49, 141, 230, 134, 235, 58, 254)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__7_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__7_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__8_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DeriveShrinkable"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__8_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__8_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__9_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__7_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__8_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(168, 199, 135, 9, 177, 131, 52, 215)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__9_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__9_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__10_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__9_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(129, 110, 11, 203, 188, 119, 204, 105)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__10_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__10_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__11_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__11_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__11_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__12_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__10_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__11_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(184, 209, 43, 101, 153, 94, 52, 159)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__12_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__12_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__13_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__13_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__13_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__14_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__12_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__13_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(9, 5, 87, 57, 87, 75, 254, 29)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__14_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__14_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__15_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__14_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(113, 14, 20, 173, 227, 183, 13, 131)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__15_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__15_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__16_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__15_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__8_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(62, 89, 178, 165, 21, 188, 230, 197)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__16_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__16_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__18_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__18_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__18_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__20_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__20_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__20_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3_value;
static const lean_closure_object lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4 = (const lean_object*)&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is not a constructor"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.MonadEnv"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4_value;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Lean.isCtor\?"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5_value;
static const lean_string_object lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6 = (const lean_object*)&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7;
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Shrinkable"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value),LEAN_SCALAR_PTR_LITERAL(106, 130, 147, 52, 116, 69, 71, 172)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value;
static const lean_array_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term[_]"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(86, 147, 168, 74, 195, 98, 232, 161)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__1_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__2_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__3 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__3_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__0_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__1_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__2_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__3 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__3_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__4_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__5_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__6 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__6_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__8 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__8_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__8_value)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__9 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__9_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Deriving"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__13 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__13_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__14 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__14_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__15 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__15_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__16 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__16_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "map"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__17 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__17_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(12, 105, 225, 203, 110, 20, 132, 37)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__19 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__19_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "x'"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__0 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(3, 12, 207, 74, 240, 119, 195, 119)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__1_value;
static const lean_array_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2_value;
static const lean_closure_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__3 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__3_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__9 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__9_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__13 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__13_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Shrinkable.shrink"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__15 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__15_value;
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "shrink"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 236, 200, 97, 115, 152, 173, 137)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(149, 3, 155, 5, 71, 245, 154, 225)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value),LEAN_SCALAR_PTR_LITERAL(106, 130, 147, 52, 116, 69, 71, 172)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(197, 166, 102, 178, 239, 188, 135, 176)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__20 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__20_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__21 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__21_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_++_"};
static const lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__0 = (const lean_object*)&lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 69, 86, 178, 149, 48, 216, 23)}};
static const lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__1 = (const lean_object*)&lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "++"};
static const lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__2 = (const lean_object*)&lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "matchAlt"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__0 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value_aux_2),((lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(178, 0, 203, 112, 215, 49, 100, 229)}};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1_value;
static const lean_string_object lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__2 = (const lean_object*)&lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__2_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__6_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0_value),LEAN_SCALAR_PTR_LITERAL(101, 27, 120, 52, 7, 113, 91, 192)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__4_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__4_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__5_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__6_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__3_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__7 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__7_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0_value),LEAN_SCALAR_PTR_LITERAL(76, 126, 178, 24, 39, 245, 155, 175)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__8_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__9 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__9_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__10 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__10_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__7_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__10_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__11 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__11_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__6_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__11_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__12 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__12_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "match"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__13 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__13_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__13_value),LEAN_SCALAR_PTR_LITERAL(9, 208, 235, 82, 91, 230, 203, 159)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "matchDiscr"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__15 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__15_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__15_value),LEAN_SCALAR_PTR_LITERAL(99, 51, 127, 238, 206, 239, 57, 130)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__17 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__17_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "matchAlts"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__18 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__18_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__18_value),LEAN_SCALAR_PTR_LITERAL(193, 186, 26, 109, 82, 172, 197, 183)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "arrow"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__20 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__20_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__20_value),LEAN_SCALAR_PTR_LITERAL(182, 146, 143, 73, 122, 115, 5, 207)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "→"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__22 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__22_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__23 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__23_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__23_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__25 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__25_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__25_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "definition"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__27 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__27_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__27_value),LEAN_SCALAR_PTR_LITERAL(248, 187, 217, 228, 39, 184, 218, 135)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__29 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__29_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__30 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__30_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__30_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__32 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__32_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__32_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__34 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__34_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__34_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__37 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__37_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__37_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__40 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__40_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__41 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__41_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__40_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__41_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "partial"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__43 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__43_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__43_value),LEAN_SCALAR_PTR_LITERAL(103, 175, 198, 167, 172, 79, 14, 207)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__45 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__45_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__45_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__48_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__49 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__49_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(230, 230, 99, 85, 138, 169, 166, 218)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__50_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__51 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__51_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__52_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__53 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__53_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__54_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__55 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__55_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__56_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__57 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__57_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__58_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__59 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__59_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__60_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__61 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__61_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__62 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__62_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__62_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__63 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__63_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__63_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__64 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__64_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__61_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__64_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__65 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__65_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__59_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__65_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__66 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__66_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__57_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__66_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__67 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__67_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__55_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__67_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__68 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__68_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__53_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__68_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__69 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__69_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__51_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__69_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__70 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__70_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__49_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__70_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__71 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__71_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__71_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__72 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__72_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__9_value),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__72_value)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__73 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__73_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mutual"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value_aux_2),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 205, 8, 5, 164, 77, 17, 1)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1_value;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "end"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 236, 200, 97, 115, 152, 173, 137)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__1 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__2 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__2_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__3 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__3_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__2_value),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__3_value)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__4 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__4_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instance"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__5 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__5_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(37, 156, 84, 218, 244, 57, 142, 153)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__7 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__7_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(32, 164, 20, 104, 12, 221, 204, 110)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__9 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__9_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(22, 101, 130, 251, 183, 19, 113, 82)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__11 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__11_value;
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_0),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_1),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value_aux_2),((lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__13 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__13_value;
static const lean_string_object lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__14 = (const lean_object*)&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__14_value;
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0;
static const lean_array_object lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__1_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2;
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__3_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4;
static lean_once_cell_t lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5;
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0;
static const lean_string_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__1 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__1_value;
static const lean_ctor_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__1_value)}};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__2 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__2_value;
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3;
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__0 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__0_value)}};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__1_value;
static lean_once_cell_t lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is not an inductive type"};
static const lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__0 = (const lean_object*)&lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__0_value;
static lean_once_cell_t lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "Cannot derive instance of Shrinkable typeclass for indexed inductive type '"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__0 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__0_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1;
static const lean_string_object lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__2 = (const lean_object*)&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__2_value;
static lean_once_cell_t lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "Cannot derive instance of Shrinkable typeclass for non-inductive types"};
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__0_value;
static lean_once_cell_t lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_ = (const lean_object*)&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = lean_unsigned_to_nat(2530155140u);
v___x_38_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__16_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_39_ = l_Lean_Name_num___override(v___x_38_, v___x_37_);
return v___x_39_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__18_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_42_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_, &lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__17_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_);
v___x_43_ = l_Lean_Name_str___override(v___x_42_, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__20_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_46_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_, &lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__19_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_);
v___x_47_ = l_Lean_Name_str___override(v___x_46_, v___x_45_);
return v___x_47_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_unsigned_to_nat(2u);
v___x_49_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_, &lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__21_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_);
v___x_50_ = l_Lean_Name_num___override(v___x_49_, v___x_48_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_52_; uint8_t v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_52_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_53_ = 0;
v___x_54_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_, &lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2__once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__22_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_);
v___x_55_ = l_Lean_registerTraceClass(v___x_52_, v___x_53_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2____boxed(lean_object* v_a_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_();
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(lean_object* v_k_58_, lean_object* v_b_59_, lean_object* v_c_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_){
_start:
{
lean_object* v___x_66_; 
lean_inc(v___y_64_);
lean_inc_ref(v___y_63_);
lean_inc(v___y_62_);
lean_inc_ref(v___y_61_);
v___x_66_ = lean_apply_7(v_k_58_, v_b_59_, v_c_60_, v___y_61_, v___y_62_, v___y_63_, v___y_64_, lean_box(0));
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed(lean_object* v_k_67_, lean_object* v_b_68_, lean_object* v_c_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0(v_k_67_, v_b_68_, v_c_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(lean_object* v_type_76_, lean_object* v_k_77_, uint8_t v_cleanupAnnotations_78_, uint8_t v_whnfType_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___f_85_; lean_object* v___x_86_; 
v___f_85_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_85_, 0, v_k_77_);
v___x_86_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_76_, v___f_85_, v_cleanupAnnotations_78_, v_whnfType_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
if (lean_obj_tag(v___x_86_) == 0)
{
lean_object* v_a_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_94_; 
v_a_87_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_94_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_94_ == 0)
{
v___x_89_ = v___x_86_;
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_a_87_);
lean_dec(v___x_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_92_; 
if (v_isShared_90_ == 0)
{
v___x_92_ = v___x_89_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v_a_87_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
else
{
lean_object* v_a_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_102_; 
v_a_95_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_102_ == 0)
{
v___x_97_ = v___x_86_;
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_a_95_);
lean_dec(v___x_86_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_100_; 
if (v_isShared_98_ == 0)
{
v___x_100_ = v___x_97_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_a_95_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg___boxed(lean_object* v_type_103_, lean_object* v_k_104_, lean_object* v_cleanupAnnotations_105_, lean_object* v_whnfType_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_112_; uint8_t v_whnfType_boxed_113_; lean_object* v_res_114_; 
v_cleanupAnnotations_boxed_112_ = lean_unbox(v_cleanupAnnotations_105_);
v_whnfType_boxed_113_ = lean_unbox(v_whnfType_106_);
v_res_114_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_103_, v_k_104_, v_cleanupAnnotations_boxed_112_, v_whnfType_boxed_113_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2(lean_object* v_00_u03b1_115_, lean_object* v_type_116_, lean_object* v_k_117_, uint8_t v_cleanupAnnotations_118_, uint8_t v_whnfType_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_116_, v_k_117_, v_cleanupAnnotations_118_, v_whnfType_119_, v___y_120_, v___y_121_, v___y_122_, v___y_123_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___boxed(lean_object* v_00_u03b1_126_, lean_object* v_type_127_, lean_object* v_k_128_, lean_object* v_cleanupAnnotations_129_, lean_object* v_whnfType_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_136_; uint8_t v_whnfType_boxed_137_; lean_object* v_res_138_; 
v_cleanupAnnotations_boxed_136_ = lean_unbox(v_cleanupAnnotations_129_);
v_whnfType_boxed_137_ = lean_unbox(v_whnfType_130_);
v_res_138_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2(v_00_u03b1_126_, v_type_127_, v_k_128_, v_cleanupAnnotations_boxed_136_, v_whnfType_boxed_137_, v___y_131_, v___y_132_, v___y_133_, v___y_134_);
lean_dec(v___y_134_);
lean_dec_ref(v___y_133_);
lean_dec(v___y_132_);
lean_dec_ref(v___y_131_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(lean_object* v_upperBound_142_, lean_object* v_indVal_143_, lean_object* v_args_144_, lean_object* v_a_145_, lean_object* v_b_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_){
_start:
{
lean_object* v_a_152_; uint8_t v___x_156_; 
v___x_156_ = lean_nat_dec_lt(v_a_145_, v_upperBound_142_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; 
lean_dec(v_a_145_);
v___x_157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_157_, 0, v_b_146_);
return v___x_157_;
}
else
{
lean_object* v_numParams_158_; uint8_t v___x_159_; 
v_numParams_158_ = lean_ctor_get(v_indVal_143_, 1);
v___x_159_ = lean_nat_dec_lt(v_a_145_, v_numParams_158_);
if (v___x_159_ == 0)
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_160_ = lean_array_fget_borrowed(v_args_144_, v_a_145_);
v___x_161_ = l_Lean_Expr_fvarId_x21(v___x_160_);
v___x_162_ = l_Lean_FVarId_getType___redArg(v___x_161_, v___y_147_, v___y_148_, v___y_149_);
if (lean_obj_tag(v___x_162_) == 0)
{
lean_object* v_a_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_a_163_ = lean_ctor_get(v___x_162_, 0);
lean_inc(v_a_163_);
lean_dec_ref_known(v___x_162_, 1);
v___x_164_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___closed__1));
v___x_165_ = l_Lean_Core_mkFreshUserName(v___x_164_, v___y_148_, v___y_149_);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_a_166_);
lean_dec_ref_known(v___x_165_, 1);
v___x_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_167_, 0, v_a_166_);
lean_ctor_set(v___x_167_, 1, v_a_163_);
v___x_168_ = lean_array_push(v_b_146_, v___x_167_);
v_a_152_ = v___x_168_;
goto v___jp_151_;
}
else
{
lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_176_; 
lean_dec(v_a_163_);
lean_dec_ref(v_b_146_);
lean_dec(v_a_145_);
v_a_169_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_176_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_176_ == 0)
{
v___x_171_ = v___x_165_;
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_dec(v___x_165_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_172_ == 0)
{
v___x_174_ = v___x_171_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_a_169_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
else
{
lean_object* v_a_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_184_; 
lean_dec_ref(v_b_146_);
lean_dec(v_a_145_);
v_a_177_ = lean_ctor_get(v___x_162_, 0);
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_162_);
if (v_isSharedCheck_184_ == 0)
{
v___x_179_ = v___x_162_;
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_a_177_);
lean_dec(v___x_162_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_182_; 
if (v_isShared_180_ == 0)
{
v___x_182_ = v___x_179_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v_a_177_);
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
else
{
v_a_152_ = v_b_146_;
goto v___jp_151_;
}
}
v___jp_151_:
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = lean_unsigned_to_nat(1u);
v___x_154_ = lean_nat_add(v_a_145_, v___x_153_);
lean_dec(v_a_145_);
v_a_145_ = v___x_154_;
v_b_146_ = v_a_152_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg___boxed(lean_object* v_upperBound_185_, lean_object* v_indVal_186_, lean_object* v_args_187_, lean_object* v_a_188_, lean_object* v_b_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v_upperBound_185_, v_indVal_186_, v_args_187_, v_a_188_, v_b_189_, v___y_190_, v___y_191_, v___y_192_);
lean_dec(v___y_192_);
lean_dec_ref(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec_ref(v_args_187_);
lean_dec_ref(v_indVal_186_);
lean_dec(v_upperBound_185_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0(lean_object* v_indVal_197_, lean_object* v_args_198_, lean_object* v_x_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_205_ = lean_unsigned_to_nat(0u);
v___x_206_ = lean_array_get_size(v_args_198_);
v___x_207_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___closed__0));
v___x_208_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v___x_206_, v_indVal_197_, v_args_198_, v___x_205_, v___x_207_, v___y_200_, v___y_202_, v___y_203_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___boxed(lean_object* v_indVal_209_, lean_object* v_args_210_, lean_object* v_x_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0(v_indVal_209_, v_args_210_, v_x_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec_ref(v_x_211_);
lean_dec_ref(v_args_210_);
lean_dec_ref(v_indVal_209_);
return v_res_217_;
}
}
static lean_object* _init_lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = l_instMonadEIO(lean_box(0));
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(lean_object* v_msg_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_toApplicative_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_292_; 
v___x_229_ = lean_obj_once(&lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0, &lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0_once, _init_lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__0);
v___x_230_ = l_StateRefT_x27_instMonad___redArg(v___x_229_);
v_toApplicative_231_ = lean_ctor_get(v___x_230_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_292_ == 0)
{
lean_object* v_unused_293_; 
v_unused_293_ = lean_ctor_get(v___x_230_, 1);
lean_dec(v_unused_293_);
v___x_233_ = v___x_230_;
v_isShared_234_ = v_isSharedCheck_292_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_toApplicative_231_);
lean_dec(v___x_230_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_292_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v_toFunctor_235_; lean_object* v_toSeq_236_; lean_object* v_toSeqLeft_237_; lean_object* v_toSeqRight_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_290_; 
v_toFunctor_235_ = lean_ctor_get(v_toApplicative_231_, 0);
v_toSeq_236_ = lean_ctor_get(v_toApplicative_231_, 2);
v_toSeqLeft_237_ = lean_ctor_get(v_toApplicative_231_, 3);
v_toSeqRight_238_ = lean_ctor_get(v_toApplicative_231_, 4);
v_isSharedCheck_290_ = !lean_is_exclusive(v_toApplicative_231_);
if (v_isSharedCheck_290_ == 0)
{
lean_object* v_unused_291_; 
v_unused_291_ = lean_ctor_get(v_toApplicative_231_, 1);
lean_dec(v_unused_291_);
v___x_240_ = v_toApplicative_231_;
v_isShared_241_ = v_isSharedCheck_290_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_toSeqRight_238_);
lean_inc(v_toSeqLeft_237_);
lean_inc(v_toSeq_236_);
lean_inc(v_toFunctor_235_);
lean_dec(v_toApplicative_231_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_290_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___f_242_; lean_object* v___f_243_; lean_object* v___f_244_; lean_object* v___f_245_; lean_object* v___x_246_; lean_object* v___f_247_; lean_object* v___f_248_; lean_object* v___f_249_; lean_object* v___x_251_; 
v___f_242_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__1));
v___f_243_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_235_);
v___f_244_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_244_, 0, v_toFunctor_235_);
v___f_245_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_245_, 0, v_toFunctor_235_);
v___x_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_246_, 0, v___f_244_);
lean_ctor_set(v___x_246_, 1, v___f_245_);
v___f_247_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_247_, 0, v_toSeqRight_238_);
v___f_248_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_248_, 0, v_toSeqLeft_237_);
v___f_249_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_249_, 0, v_toSeq_236_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 4, v___f_247_);
lean_ctor_set(v___x_240_, 3, v___f_248_);
lean_ctor_set(v___x_240_, 2, v___f_249_);
lean_ctor_set(v___x_240_, 1, v___f_242_);
lean_ctor_set(v___x_240_, 0, v___x_246_);
v___x_251_ = v___x_240_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v___f_242_);
lean_ctor_set(v_reuseFailAlloc_289_, 2, v___f_249_);
lean_ctor_set(v_reuseFailAlloc_289_, 3, v___f_248_);
lean_ctor_set(v_reuseFailAlloc_289_, 4, v___f_247_);
v___x_251_ = v_reuseFailAlloc_289_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
lean_object* v___x_253_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 1, v___f_243_);
lean_ctor_set(v___x_233_, 0, v___x_251_);
v___x_253_ = v___x_233_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v___x_251_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v___f_243_);
v___x_253_ = v_reuseFailAlloc_288_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
lean_object* v___x_254_; lean_object* v_toApplicative_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_286_; 
v___x_254_ = l_StateRefT_x27_instMonad___redArg(v___x_253_);
v_toApplicative_255_ = lean_ctor_get(v___x_254_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_254_);
if (v_isSharedCheck_286_ == 0)
{
lean_object* v_unused_287_; 
v_unused_287_ = lean_ctor_get(v___x_254_, 1);
lean_dec(v_unused_287_);
v___x_257_ = v___x_254_;
v_isShared_258_ = v_isSharedCheck_286_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_toApplicative_255_);
lean_dec(v___x_254_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_286_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v_toFunctor_259_; lean_object* v_toSeq_260_; lean_object* v_toSeqLeft_261_; lean_object* v_toSeqRight_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_284_; 
v_toFunctor_259_ = lean_ctor_get(v_toApplicative_255_, 0);
v_toSeq_260_ = lean_ctor_get(v_toApplicative_255_, 2);
v_toSeqLeft_261_ = lean_ctor_get(v_toApplicative_255_, 3);
v_toSeqRight_262_ = lean_ctor_get(v_toApplicative_255_, 4);
v_isSharedCheck_284_ = !lean_is_exclusive(v_toApplicative_255_);
if (v_isSharedCheck_284_ == 0)
{
lean_object* v_unused_285_; 
v_unused_285_ = lean_ctor_get(v_toApplicative_255_, 1);
lean_dec(v_unused_285_);
v___x_264_ = v_toApplicative_255_;
v_isShared_265_ = v_isSharedCheck_284_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_toSeqRight_262_);
lean_inc(v_toSeqLeft_261_);
lean_inc(v_toSeq_260_);
lean_inc(v_toFunctor_259_);
lean_dec(v_toApplicative_255_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_284_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___f_266_; lean_object* v___f_267_; lean_object* v___f_268_; lean_object* v___f_269_; lean_object* v___x_270_; lean_object* v___f_271_; lean_object* v___f_272_; lean_object* v___f_273_; lean_object* v___x_275_; 
v___f_266_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__3));
v___f_267_ = ((lean_object*)(lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_259_);
v___f_268_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_268_, 0, v_toFunctor_259_);
v___f_269_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_269_, 0, v_toFunctor_259_);
v___x_270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_270_, 0, v___f_268_);
lean_ctor_set(v___x_270_, 1, v___f_269_);
v___f_271_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_271_, 0, v_toSeqRight_262_);
v___f_272_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_272_, 0, v_toSeqLeft_261_);
v___f_273_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_273_, 0, v_toSeq_260_);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 4, v___f_271_);
lean_ctor_set(v___x_264_, 3, v___f_272_);
lean_ctor_set(v___x_264_, 2, v___f_273_);
lean_ctor_set(v___x_264_, 1, v___f_266_);
lean_ctor_set(v___x_264_, 0, v___x_270_);
v___x_275_ = v___x_264_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_270_);
lean_ctor_set(v_reuseFailAlloc_283_, 1, v___f_266_);
lean_ctor_set(v_reuseFailAlloc_283_, 2, v___f_273_);
lean_ctor_set(v_reuseFailAlloc_283_, 3, v___f_272_);
lean_ctor_set(v_reuseFailAlloc_283_, 4, v___f_271_);
v___x_275_ = v_reuseFailAlloc_283_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
lean_object* v___x_277_; 
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 1, v___f_267_);
lean_ctor_set(v___x_257_, 0, v___x_275_);
v___x_277_ = v___x_257_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_282_, 1, v___f_267_);
v___x_277_ = v_reuseFailAlloc_282_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_3015__overap_280_; lean_object* v___x_281_; 
v___x_278_ = lean_box(0);
v___x_279_ = l_instInhabitedOfMonad___redArg(v___x_277_, v___x_278_);
v___x_3015__overap_280_ = lean_panic_fn_borrowed(v___x_279_, v_msg_223_);
lean_dec(v___x_279_);
lean_inc(v___y_227_);
lean_inc_ref(v___y_226_);
lean_inc(v___y_225_);
lean_inc_ref(v___y_224_);
v___x_281_ = lean_apply_5(v___x_3015__overap_280_, v___y_224_, v___y_225_, v___y_226_, v___y_227_, lean_box(0));
return v___x_281_;
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
LEAN_EXPORT lean_object* lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2___boxed(lean_object* v_msg_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(v_msg_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(lean_object* v_msgData_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v___x_307_; lean_object* v_env_308_; lean_object* v___x_309_; lean_object* v_mctx_310_; lean_object* v_lctx_311_; lean_object* v_options_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_307_ = lean_st_ref_get(v___y_305_);
v_env_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc_ref(v_env_308_);
lean_dec(v___x_307_);
v___x_309_ = lean_st_ref_get(v___y_303_);
v_mctx_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc_ref(v_mctx_310_);
lean_dec(v___x_309_);
v_lctx_311_ = lean_ctor_get(v___y_302_, 2);
v_options_312_ = lean_ctor_get(v___y_304_, 2);
lean_inc_ref(v_options_312_);
lean_inc_ref(v_lctx_311_);
v___x_313_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_313_, 0, v_env_308_);
lean_ctor_set(v___x_313_, 1, v_mctx_310_);
lean_ctor_set(v___x_313_, 2, v_lctx_311_);
lean_ctor_set(v___x_313_, 3, v_options_312_);
v___x_314_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_313_);
lean_ctor_set(v___x_314_, 1, v_msgData_301_);
v___x_315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msgData_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(lean_object* v_msg_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_ref_329_; lean_object* v___x_330_; lean_object* v_a_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_339_; 
v_ref_329_ = lean_ctor_get(v___y_326_, 5);
v___x_330_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_);
v_a_331_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_339_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_339_ == 0)
{
v___x_333_ = v___x_330_;
v_isShared_334_ = v_isSharedCheck_339_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_a_331_);
lean_dec(v___x_330_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_339_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_335_; lean_object* v___x_337_; 
lean_inc(v_ref_329_);
v___x_335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_335_, 0, v_ref_329_);
lean_ctor_set(v___x_335_, 1, v_a_331_);
if (v_isShared_334_ == 0)
{
lean_ctor_set_tag(v___x_333_, 1);
lean_ctor_set(v___x_333_, 0, v___x_335_);
v___x_337_ = v___x_333_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_335_);
v___x_337_ = v_reuseFailAlloc_338_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
return v___x_337_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg___boxed(lean_object* v_msg_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v_msg_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
return v_res_346_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1(void){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_348_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__0));
v___x_349_ = l_Lean_stringToMessageData(v___x_348_);
return v___x_349_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_351_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__2));
v___x_352_ = l_Lean_stringToMessageData(v___x_351_);
return v___x_352_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_356_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__6));
v___x_357_ = lean_unsigned_to_nat(11u);
v___x_358_ = lean_unsigned_to_nat(122u);
v___x_359_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__5));
v___x_360_ = ((lean_object*)(lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__4));
v___x_361_ = l_mkPanicMessageWithDecl(v___x_360_, v___x_359_, v___x_358_, v___x_357_, v___x_356_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1(lean_object* v_constName_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_376_; lean_object* v_env_377_; uint8_t v___x_378_; lean_object* v___x_379_; 
v___x_376_ = lean_st_ref_get(v___y_366_);
v_env_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc_ref(v_env_377_);
lean_dec(v___x_376_);
v___x_378_ = 0;
lean_inc(v_constName_362_);
v___x_379_ = l_Lean_Environment_findAsync_x3f(v_env_377_, v_constName_362_, v___x_378_);
if (lean_obj_tag(v___x_379_) == 1)
{
lean_object* v_val_380_; uint8_t v_kind_381_; 
v_val_380_ = lean_ctor_get(v___x_379_, 0);
lean_inc(v_val_380_);
lean_dec_ref_known(v___x_379_, 1);
v_kind_381_ = lean_ctor_get_uint8(v_val_380_, sizeof(void*)*3);
if (v_kind_381_ == 6)
{
lean_object* v___x_382_; 
v___x_382_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_380_);
if (lean_obj_tag(v___x_382_) == 6)
{
lean_object* v_val_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_390_; 
lean_dec(v_constName_362_);
v_val_383_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_390_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_390_ == 0)
{
v___x_385_ = v___x_382_;
v_isShared_386_ = v_isSharedCheck_390_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_val_383_);
lean_dec(v___x_382_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_390_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_388_; 
if (v_isShared_386_ == 0)
{
lean_ctor_set_tag(v___x_385_, 0);
v___x_388_ = v___x_385_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_389_; 
v_reuseFailAlloc_389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_389_, 0, v_val_383_);
v___x_388_ = v_reuseFailAlloc_389_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
return v___x_388_;
}
}
}
else
{
lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec_ref(v___x_382_);
v___x_391_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__7);
v___x_392_ = lp_plausible_panic___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__2(v___x_391_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
if (lean_obj_tag(v___x_392_) == 0)
{
lean_object* v_a_393_; lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_401_; 
v_a_393_ = lean_ctor_get(v___x_392_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_392_);
if (v_isSharedCheck_401_ == 0)
{
v___x_395_ = v___x_392_;
v_isShared_396_ = v_isSharedCheck_401_;
goto v_resetjp_394_;
}
else
{
lean_inc(v_a_393_);
lean_dec(v___x_392_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_401_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
if (lean_obj_tag(v_a_393_) == 0)
{
lean_del_object(v___x_395_);
goto v___jp_368_;
}
else
{
lean_object* v_val_397_; lean_object* v___x_399_; 
lean_dec(v_constName_362_);
v_val_397_ = lean_ctor_get(v_a_393_, 0);
lean_inc(v_val_397_);
lean_dec_ref_known(v_a_393_, 1);
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 0, v_val_397_);
v___x_399_ = v___x_395_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v_val_397_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
}
else
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_409_; 
lean_dec(v_constName_362_);
v_a_402_ = lean_ctor_get(v___x_392_, 0);
v_isSharedCheck_409_ = !lean_is_exclusive(v___x_392_);
if (v_isSharedCheck_409_ == 0)
{
v___x_404_ = v___x_392_;
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_392_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v_a_402_);
v___x_407_ = v_reuseFailAlloc_408_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
return v___x_407_;
}
}
}
}
}
else
{
lean_dec(v_val_380_);
goto v___jp_368_;
}
}
else
{
lean_dec(v___x_379_);
goto v___jp_368_;
}
v___jp_368_:
{
lean_object* v___x_369_; uint8_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_369_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1);
v___x_370_ = 0;
v___x_371_ = l_Lean_MessageData_ofConstName(v_constName_362_, v___x_370_);
v___x_372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_369_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
v___x_373_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__3);
v___x_374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_372_);
lean_ctor_set(v___x_374_, 1, v___x_373_);
v___x_375_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v___x_374_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
return v___x_375_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___boxed(lean_object* v_constName_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1(v_constName_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes(lean_object* v_indVal_417_, lean_object* v_ctorName_418_, lean_object* v_a_419_, lean_object* v_a_420_, lean_object* v_a_421_, lean_object* v_a_422_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1(v_ctorName_418_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v_a_425_; lean_object* v_toConstantVal_426_; lean_object* v_type_427_; lean_object* v___f_428_; uint8_t v___x_429_; lean_object* v___x_430_; 
v_a_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_a_425_);
lean_dec_ref_known(v___x_424_, 1);
v_toConstantVal_426_ = lean_ctor_get(v_a_425_, 0);
lean_inc_ref(v_toConstantVal_426_);
lean_dec(v_a_425_);
v_type_427_ = lean_ctor_get(v_toConstantVal_426_, 2);
lean_inc_ref(v_type_427_);
lean_dec_ref(v_toConstantVal_426_);
v___f_428_ = lean_alloc_closure((void*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___lam__0___boxed), 8, 1);
lean_closure_set(v___f_428_, 0, v_indVal_417_);
v___x_429_ = 0;
v___x_430_ = lp_plausible_Lean_Meta_forallTelescopeReducing___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__2___redArg(v_type_427_, v___f_428_, v___x_429_, v___x_429_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_430_;
}
else
{
lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_438_; 
lean_dec_ref(v_indVal_417_);
v_a_431_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_438_ == 0)
{
v___x_433_ = v___x_424_;
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_424_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_436_; 
if (v_isShared_434_ == 0)
{
v___x_436_ = v___x_433_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v_a_431_);
v___x_436_ = v_reuseFailAlloc_437_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
return v___x_436_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes___boxed(lean_object* v_indVal_439_, lean_object* v_ctorName_440_, lean_object* v_a_441_, lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes(v_indVal_439_, v_ctorName_440_, v_a_441_, v_a_442_, v_a_443_, v_a_444_);
lean_dec(v_a_444_);
lean_dec_ref(v_a_443_);
lean_dec(v_a_442_);
lean_dec_ref(v_a_441_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0(lean_object* v_upperBound_447_, lean_object* v_indVal_448_, lean_object* v_args_449_, lean_object* v_inst_450_, lean_object* v_R_451_, lean_object* v_a_452_, lean_object* v_b_453_, lean_object* v_c_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___redArg(v_upperBound_447_, v_indVal_448_, v_args_449_, v_a_452_, v_b_453_, v___y_455_, v___y_457_, v___y_458_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0___boxed(lean_object* v_upperBound_461_, lean_object* v_indVal_462_, lean_object* v_args_463_, lean_object* v_inst_464_, lean_object* v_R_465_, lean_object* v_a_466_, lean_object* v_b_467_, lean_object* v_c_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__0(v_upperBound_461_, v_indVal_462_, v_args_463_, v_inst_464_, v_R_465_, v_a_466_, v_b_467_, v_c_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
lean_dec_ref(v_args_463_);
lean_dec_ref(v_indVal_462_);
lean_dec(v_upperBound_461_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(lean_object* v_00_u03b1_475_, lean_object* v_msg_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___redArg(v_msg_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1___boxed(lean_object* v_00_u03b1_483_, lean_object* v_msg_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1(v_00_u03b1_483_, v_msg_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
lean_dec(v___y_488_);
lean_dec_ref(v___y_487_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader(lean_object* v_indVal_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_){
_start:
{
lean_object* v___x_505_; 
lean_inc_ref(v_indVal_497_);
v___x_505_ = l_Lean_Elab_Deriving_mkInductArgNames(v_indVal_497_, v_a_498_, v_a_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_);
if (lean_obj_tag(v___x_505_) == 0)
{
lean_object* v_a_506_; lean_object* v___x_507_; 
v_a_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc_n(v_a_506_, 2);
lean_dec_ref_known(v___x_505_, 1);
v___x_507_ = l_Lean_Elab_Deriving_mkImplicitBinders(v_a_506_, v_a_498_, v_a_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_);
if (lean_obj_tag(v___x_507_) == 0)
{
lean_object* v_a_508_; lean_object* v___x_509_; 
v_a_508_ = lean_ctor_get(v___x_507_, 0);
lean_inc(v_a_508_);
lean_dec_ref_known(v___x_507_, 1);
lean_inc(v_a_506_);
lean_inc_ref(v_indVal_497_);
v___x_509_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v_indVal_497_, v_a_506_, v_a_502_);
if (lean_obj_tag(v___x_509_) == 0)
{
lean_object* v_a_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v_a_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc(v_a_510_);
lean_dec_ref_known(v___x_509_, 1);
v___x_511_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1));
lean_inc(v_a_506_);
v___x_512_ = l_Lean_Elab_Deriving_mkInstImplicitBinders(v___x_511_, v_indVal_497_, v_a_506_, v_a_498_, v_a_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_);
if (lean_obj_tag(v___x_512_) == 0)
{
lean_object* v_a_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_523_; 
v_a_513_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_523_ == 0)
{
v___x_515_ = v___x_512_;
v_isShared_516_ = v_isSharedCheck_523_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_a_513_);
lean_dec(v___x_512_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_523_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_521_; 
v___x_517_ = l_Array_append___redArg(v_a_508_, v_a_513_);
lean_dec(v_a_513_);
v___x_518_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__2));
v___x_519_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_519_, 0, v___x_517_);
lean_ctor_set(v___x_519_, 1, v_a_506_);
lean_ctor_set(v___x_519_, 2, v___x_518_);
lean_ctor_set(v___x_519_, 3, v_a_510_);
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 0, v___x_519_);
v___x_521_ = v___x_515_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_519_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
else
{
lean_object* v_a_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_531_; 
lean_dec(v_a_510_);
lean_dec(v_a_508_);
lean_dec(v_a_506_);
v_a_524_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_531_ == 0)
{
v___x_526_ = v___x_512_;
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_a_524_);
lean_dec(v___x_512_);
v___x_526_ = lean_box(0);
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
v_resetjp_525_:
{
lean_object* v___x_529_; 
if (v_isShared_527_ == 0)
{
v___x_529_ = v___x_526_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_a_524_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
else
{
lean_object* v_a_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_539_; 
lean_dec(v_a_508_);
lean_dec(v_a_506_);
lean_dec_ref(v_indVal_497_);
v_a_532_ = lean_ctor_get(v___x_509_, 0);
v_isSharedCheck_539_ = !lean_is_exclusive(v___x_509_);
if (v_isSharedCheck_539_ == 0)
{
v___x_534_ = v___x_509_;
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_a_532_);
lean_dec(v___x_509_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_537_; 
if (v_isShared_535_ == 0)
{
v___x_537_ = v___x_534_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_538_; 
v_reuseFailAlloc_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_538_, 0, v_a_532_);
v___x_537_ = v_reuseFailAlloc_538_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
return v___x_537_;
}
}
}
}
else
{
lean_object* v_a_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_547_; 
lean_dec(v_a_506_);
lean_dec_ref(v_indVal_497_);
v_a_540_ = lean_ctor_get(v___x_507_, 0);
v_isSharedCheck_547_ = !lean_is_exclusive(v___x_507_);
if (v_isSharedCheck_547_ == 0)
{
v___x_542_ = v___x_507_;
v_isShared_543_ = v_isSharedCheck_547_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_a_540_);
lean_dec(v___x_507_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_547_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_545_; 
if (v_isShared_543_ == 0)
{
v___x_545_ = v___x_542_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v_a_540_);
v___x_545_ = v_reuseFailAlloc_546_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
return v___x_545_;
}
}
}
}
else
{
lean_object* v_a_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_555_; 
lean_dec_ref(v_indVal_497_);
v_a_548_ = lean_ctor_get(v___x_505_, 0);
v_isSharedCheck_555_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_555_ == 0)
{
v___x_550_ = v___x_505_;
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_a_548_);
lean_dec(v___x_505_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_553_; 
if (v_isShared_551_ == 0)
{
v___x_553_ = v___x_550_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v_a_548_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___boxed(lean_object* v_indVal_556_, lean_object* v_a_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader(v_indVal_556_, v_a_557_, v_a_558_, v_a_559_, v_a_560_, v_a_561_, v_a_562_);
lean_dec(v_a_562_);
lean_dec_ref(v_a_561_);
lean_dec(v_a_560_);
lean_dec_ref(v_a_559_);
lean_dec(v_a_558_);
lean_dec_ref(v_a_557_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg(lean_object* v_upperBound_573_, lean_object* v_argTypes_574_, lean_object* v_targetTypeName_575_, lean_object* v_freshIdents_576_, lean_object* v_a_577_, lean_object* v_b_578_, lean_object* v___y_579_){
_start:
{
lean_object* v_a_582_; uint8_t v___x_586_; 
v___x_586_ = lean_nat_dec_lt(v_a_577_, v_upperBound_573_);
if (v___x_586_ == 0)
{
lean_object* v___x_587_; 
lean_dec(v_a_577_);
v___x_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_587_, 0, v_b_578_);
return v___x_587_;
}
else
{
lean_object* v___x_588_; lean_object* v___x_589_; uint8_t v___x_590_; 
v___x_588_ = l_Lean_instInhabitedExpr;
v___x_589_ = lean_array_get_borrowed(v___x_588_, v_argTypes_574_, v_a_577_);
v___x_590_ = l_Lean_Expr_isAppOf(v___x_589_, v_targetTypeName_575_);
if (v___x_590_ == 0)
{
v_a_582_ = v_b_578_;
goto v___jp_581_;
}
else
{
lean_object* v_ref_591_; uint8_t v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v_ref_591_ = lean_ctor_get(v___y_579_, 5);
v___x_592_ = 0;
v___x_593_ = l_Lean_SourceInfo_fromRef(v_ref_591_, v___x_592_);
v___x_594_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__1));
v___x_595_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__2));
lean_inc_n(v___x_593_, 3);
v___x_596_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_593_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
v___x_597_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_598_ = lean_array_fget_borrowed(v_freshIdents_576_, v_a_577_);
lean_inc(v___x_598_);
v___x_599_ = l_Lean_Syntax_node1(v___x_593_, v___x_597_, v___x_598_);
v___x_600_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__5));
v___x_601_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_593_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = l_Lean_Syntax_node3(v___x_593_, v___x_594_, v___x_596_, v___x_599_, v___x_601_);
v___x_603_ = lean_array_push(v_b_578_, v___x_602_);
v_a_582_ = v___x_603_;
goto v___jp_581_;
}
}
v___jp_581_:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = lean_unsigned_to_nat(1u);
v___x_584_ = lean_nat_add(v_a_577_, v___x_583_);
lean_dec(v_a_577_);
v_a_577_ = v___x_584_;
v_b_578_ = v_a_582_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___boxed(lean_object* v_upperBound_604_, lean_object* v_argTypes_605_, lean_object* v_targetTypeName_606_, lean_object* v_freshIdents_607_, lean_object* v_a_608_, lean_object* v_b_609_, lean_object* v___y_610_, lean_object* v___y_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg(v_upperBound_604_, v_argTypes_605_, v_targetTypeName_606_, v_freshIdents_607_, v_a_608_, v_b_609_, v___y_610_);
lean_dec_ref(v___y_610_);
lean_dec_ref(v_freshIdents_607_);
lean_dec(v_targetTypeName_606_);
lean_dec_ref(v_argTypes_605_);
lean_dec(v_upperBound_604_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg(lean_object* v_upperBound_613_, lean_object* v_next_614_, lean_object* v_freshIdents_615_, lean_object* v___x_616_, lean_object* v_a_617_, lean_object* v_b_618_){
_start:
{
lean_object* v_a_621_; uint8_t v___x_625_; 
v___x_625_ = lean_nat_dec_lt(v_a_617_, v_upperBound_613_);
if (v___x_625_ == 0)
{
lean_object* v___x_626_; 
lean_dec(v_a_617_);
lean_dec(v___x_616_);
v___x_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_626_, 0, v_b_618_);
return v___x_626_;
}
else
{
uint8_t v___x_627_; 
v___x_627_ = lean_nat_dec_eq(v_next_614_, v_a_617_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_628_ = lean_array_fget_borrowed(v_freshIdents_615_, v_a_617_);
lean_inc(v___x_628_);
v___x_629_ = lean_array_push(v_b_618_, v___x_628_);
v_a_621_ = v___x_629_;
goto v___jp_620_;
}
else
{
lean_object* v___x_630_; 
lean_inc(v___x_616_);
v___x_630_ = lean_array_push(v_b_618_, v___x_616_);
v_a_621_ = v___x_630_;
goto v___jp_620_;
}
}
v___jp_620_:
{
lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_622_ = lean_unsigned_to_nat(1u);
v___x_623_ = lean_nat_add(v_a_617_, v___x_622_);
lean_dec(v_a_617_);
v_a_617_ = v___x_623_;
v_b_618_ = v_a_621_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg___boxed(lean_object* v_upperBound_631_, lean_object* v_next_632_, lean_object* v_freshIdents_633_, lean_object* v___x_634_, lean_object* v_a_635_, lean_object* v_b_636_, lean_object* v___y_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg(v_upperBound_631_, v_next_632_, v_freshIdents_633_, v___x_634_, v_a_635_, v_b_636_);
lean_dec_ref(v_freshIdents_633_);
lean_dec(v_next_632_);
lean_dec(v_upperBound_631_);
return v_res_638_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7(void){
_start:
{
lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_647_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__6));
v___x_648_ = l_String_toRawSubstring_x27(v___x_647_);
return v___x_648_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__17));
v___x_663_ = l_String_toRawSubstring_x27(v___x_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1(lean_object* v___f_666_, lean_object* v___x_667_, lean_object* v___x_668_, lean_object* v___x_669_, lean_object* v___x_670_, lean_object* v___x_671_, lean_object* v___x_672_, lean_object* v_b_673_, lean_object* v_shrinkCall_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v___x_682_; 
lean_inc(v___y_680_);
lean_inc_ref(v___y_679_);
lean_inc(v___y_678_);
lean_inc_ref(v___y_677_);
lean_inc(v___y_676_);
lean_inc_ref(v___y_675_);
v___x_682_ = lean_apply_7(v___f_666_, v___y_675_, v___y_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, lean_box(0));
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v_a_683_; lean_object* v___x_685_; uint8_t v_isShared_686_; uint8_t v_isSharedCheck_754_; 
v_a_683_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_754_ == 0)
{
v___x_685_ = v___x_682_;
v_isShared_686_ = v_isSharedCheck_754_;
goto v_resetjp_684_;
}
else
{
lean_inc(v_a_683_);
lean_dec(v___x_682_);
v___x_685_ = lean_box(0);
v_isShared_686_ = v_isSharedCheck_754_;
goto v_resetjp_684_;
}
v_resetjp_684_:
{
lean_object* v_quotContext_687_; lean_object* v_currMacroScope_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_752_; 
v_quotContext_687_ = lean_ctor_get(v___y_679_, 10);
v_currMacroScope_688_ = lean_ctor_get(v___y_679_, 11);
v___x_689_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__0));
lean_inc_ref_n(v___x_669_, 4);
lean_inc_ref_n(v___x_668_, 4);
lean_inc_ref_n(v___x_667_, 10);
v___x_690_ = l_Lean_Name_mkStr4(v___x_667_, v___x_668_, v___x_669_, v___x_689_);
v___x_691_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__1));
v___x_692_ = l_Lean_Name_mkStr4(v___x_667_, v___x_668_, v___x_669_, v___x_691_);
v___x_693_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__2));
v___x_694_ = l_Lean_Name_mkStr4(v___x_667_, v___x_668_, v___x_669_, v___x_693_);
v___x_695_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__3));
lean_inc_n(v_a_683_, 10);
v___x_696_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_696_, 0, v_a_683_);
lean_ctor_set(v___x_696_, 1, v___x_695_);
v___x_697_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__5));
v___x_698_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7);
v___x_699_ = lean_box(0);
lean_inc_n(v_currMacroScope_688_, 2);
lean_inc_n(v_quotContext_687_, 2);
v___x_700_ = l_Lean_addMacroScope(v_quotContext_687_, v___x_699_, v_currMacroScope_688_);
v___x_701_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__9));
v___x_702_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__10));
v___x_703_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__11));
v___x_704_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__12));
v___x_705_ = l_Lean_Name_mkStr3(v___x_667_, v___x_703_, v___x_704_);
v___x_706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_706_, 0, v___x_705_);
v___x_707_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__13));
v___x_708_ = l_Lean_Name_mkStr3(v___x_667_, v___x_703_, v___x_707_);
v___x_709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_709_, 0, v___x_708_);
v___x_710_ = l_Lean_Name_mkStr3(v___x_667_, v___x_703_, v___x_669_);
v___x_711_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_711_, 0, v___x_710_);
v___x_712_ = l_Lean_Name_mkStr3(v___x_667_, v___x_668_, v___x_669_);
v___x_713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_713_, 0, v___x_712_);
v___x_714_ = l_Lean_Name_mkStr2(v___x_667_, v___x_668_);
v___x_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
v___x_716_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__14));
v___x_717_ = l_Lean_Name_mkStr2(v___x_667_, v___x_716_);
v___x_718_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_718_, 0, v___x_717_);
v___x_719_ = l_Lean_Name_mkStr2(v___x_667_, v___x_703_);
v___x_720_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_720_, 0, v___x_719_);
v___x_721_ = l_Lean_Name_mkStr1(v___x_667_);
v___x_722_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_722_, 0, v___x_721_);
v___x_723_ = lean_box(0);
v___x_724_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_722_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_725_, 0, v___x_720_);
lean_ctor_set(v___x_725_, 1, v___x_724_);
v___x_726_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_726_, 0, v___x_718_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
v___x_727_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_727_, 0, v___x_715_);
lean_ctor_set(v___x_727_, 1, v___x_726_);
v___x_728_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_713_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
v___x_729_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_729_, 0, v___x_711_);
lean_ctor_set(v___x_729_, 1, v___x_728_);
v___x_730_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_709_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_731_, 0, v___x_706_);
lean_ctor_set(v___x_731_, 1, v___x_730_);
v___x_732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_732_, 0, v___x_702_);
lean_ctor_set(v___x_732_, 1, v___x_731_);
v___x_733_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_733_, 0, v___x_701_);
lean_ctor_set(v___x_733_, 1, v___x_732_);
v___x_734_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_734_, 0, v_a_683_);
lean_ctor_set(v___x_734_, 1, v___x_698_);
lean_ctor_set(v___x_734_, 2, v___x_700_);
lean_ctor_set(v___x_734_, 3, v___x_733_);
v___x_735_ = l_Lean_Syntax_node1(v_a_683_, v___x_697_, v___x_734_);
v___x_736_ = l_Lean_Syntax_node2(v_a_683_, v___x_694_, v___x_696_, v___x_735_);
v___x_737_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__15));
v___x_738_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_738_, 0, v_a_683_);
lean_ctor_set(v___x_738_, 1, v___x_737_);
v___x_739_ = l_Lean_Syntax_node3(v_a_683_, v___x_692_, v___x_736_, v_shrinkCall_674_, v___x_738_);
v___x_740_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__16));
v___x_741_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_741_, 0, v_a_683_);
lean_ctor_set(v___x_741_, 1, v___x_740_);
v___x_742_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__18);
v___x_743_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__19));
v___x_744_ = l_Lean_addMacroScope(v_quotContext_687_, v___x_743_, v_currMacroScope_688_);
v___x_745_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_745_, 0, v_a_683_);
lean_ctor_set(v___x_745_, 1, v___x_742_);
lean_ctor_set(v___x_745_, 2, v___x_744_);
lean_ctor_set(v___x_745_, 3, v___x_723_);
v___x_746_ = l_Lean_Syntax_node3(v_a_683_, v___x_690_, v___x_739_, v___x_741_, v___x_745_);
v___x_747_ = l_Lean_Syntax_node1(v_a_683_, v___x_670_, v___x_671_);
v___x_748_ = l_Lean_Syntax_node2(v_a_683_, v___x_672_, v___x_746_, v___x_747_);
v___x_749_ = lean_array_push(v_b_673_, v___x_748_);
v___x_750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_750_, 0, v___x_749_);
if (v_isShared_686_ == 0)
{
lean_ctor_set(v___x_685_, 0, v___x_750_);
v___x_752_ = v___x_685_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v___x_750_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
else
{
lean_object* v_a_755_; lean_object* v___x_757_; uint8_t v_isShared_758_; uint8_t v_isSharedCheck_762_; 
lean_dec(v_shrinkCall_674_);
lean_dec_ref(v_b_673_);
lean_dec(v___x_672_);
lean_dec(v___x_671_);
lean_dec(v___x_670_);
lean_dec_ref(v___x_669_);
lean_dec_ref(v___x_668_);
lean_dec_ref(v___x_667_);
v_a_755_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_762_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_762_ == 0)
{
v___x_757_ = v___x_682_;
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
else
{
lean_inc(v_a_755_);
lean_dec(v___x_682_);
v___x_757_ = lean_box(0);
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
v_resetjp_756_:
{
lean_object* v___x_760_; 
if (v_isShared_758_ == 0)
{
v___x_760_ = v___x_757_;
goto v_reusejp_759_;
}
else
{
lean_object* v_reuseFailAlloc_761_; 
v_reuseFailAlloc_761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_761_, 0, v_a_755_);
v___x_760_ = v_reuseFailAlloc_761_;
goto v_reusejp_759_;
}
v_reusejp_759_:
{
return v___x_760_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___boxed(lean_object* v___f_763_, lean_object* v___x_764_, lean_object* v___x_765_, lean_object* v___x_766_, lean_object* v___x_767_, lean_object* v___x_768_, lean_object* v___x_769_, lean_object* v_b_770_, lean_object* v_shrinkCall_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_){
_start:
{
lean_object* v_res_779_; 
v_res_779_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1(v___f_763_, v___x_764_, v___x_765_, v___x_766_, v___x_767_, v___x_768_, v___x_769_, v_b_770_, v_shrinkCall_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_, v___y_777_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0(lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_){
_start:
{
lean_object* v_ref_787_; uint8_t v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; 
v_ref_787_ = lean_ctor_get(v___y_784_, 5);
v___x_788_ = 0;
v___x_789_ = l_Lean_SourceInfo_fromRef(v_ref_787_, v___x_788_);
v___x_790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_790_, 0, v___x_789_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0___boxed(lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_){
_start:
{
lean_object* v_res_798_; 
v_res_798_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0(v___y_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_);
lean_dec(v___y_796_);
lean_dec_ref(v___y_795_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
return v_res_798_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11(void){
_start:
{
lean_object* v___x_820_; 
v___x_820_ = l_Array_mkArray0(lean_box(0));
return v___x_820_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16(void){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_829_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__15));
v___x_830_ = l_String_toRawSubstring_x27(v___x_829_);
return v___x_830_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg(lean_object* v_upperBound_845_, lean_object* v___x_846_, lean_object* v_freshIdents_847_, lean_object* v_ctorIdent_848_, lean_object* v_argTypes_849_, lean_object* v_targetTypeName_850_, lean_object* v_auxFn_851_, lean_object* v_a_852_, lean_object* v_b_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
lean_object* v___y_862_; uint8_t v___x_884_; 
v___x_884_ = lean_nat_dec_lt(v_a_852_, v_upperBound_845_);
if (v___x_884_ == 0)
{
lean_object* v___x_885_; 
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v___x_885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_885_, 0, v_b_853_);
return v___x_885_;
}
else
{
lean_object* v___x_886_; lean_object* v___x_887_; 
v___x_886_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__1));
v___x_887_ = l_Lean_Core_mkFreshUserName(v___x_886_, v___y_858_, v___y_859_);
if (lean_obj_tag(v___x_887_) == 0)
{
lean_object* v_a_888_; lean_object* v___x_889_; lean_object* v_listTerms_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; 
v_a_888_ = lean_ctor_get(v___x_887_, 0);
lean_inc(v_a_888_);
lean_dec_ref_known(v___x_887_, 1);
v___x_889_ = lean_unsigned_to_nat(0u);
v_listTerms_890_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2));
v___x_891_ = lean_array_fget_borrowed(v_freshIdents_847_, v_a_852_);
v___x_892_ = l_Lean_mkIdent(v_a_888_);
lean_inc(v___x_892_);
v___x_893_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg(v___x_846_, v_a_852_, v_freshIdents_847_, v___x_892_, v___x_889_, v_listTerms_890_);
if (lean_obj_tag(v___x_893_) == 0)
{
lean_object* v_a_894_; lean_object* v_ref_895_; lean_object* v_quotContext_896_; lean_object* v_currMacroScope_897_; lean_object* v___f_898_; lean_object* v___x_899_; uint8_t v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; uint8_t v___x_922_; 
v_a_894_ = lean_ctor_get(v___x_893_, 0);
lean_inc(v_a_894_);
lean_dec_ref_known(v___x_893_, 1);
v_ref_895_ = lean_ctor_get(v___y_858_, 5);
v_quotContext_896_ = lean_ctor_get(v___y_858_, 10);
v_currMacroScope_897_ = lean_ctor_get(v___y_858_, 11);
v___f_898_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__3));
v___x_899_ = l_Lean_instInhabitedExpr;
v___x_900_ = 0;
v___x_901_ = l_Lean_SourceInfo_fromRef(v_ref_895_, v___x_900_);
v___x_902_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__4));
v___x_903_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__5));
v___x_904_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__6));
v___x_905_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7));
v___x_906_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8));
lean_inc_n(v___x_901_, 7);
v___x_907_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_907_, 0, v___x_901_);
lean_ctor_set(v___x_907_, 1, v___x_905_);
v___x_908_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10));
v___x_909_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_910_ = l_Lean_Syntax_node1(v___x_901_, v___x_909_, v___x_892_);
v___x_911_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
v___x_912_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_912_, 0, v___x_901_);
lean_ctor_set(v___x_912_, 1, v___x_909_);
lean_ctor_set(v___x_912_, 2, v___x_911_);
v___x_913_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12));
v___x_914_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_901_);
lean_ctor_set(v___x_914_, 1, v___x_913_);
v___x_915_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14));
v___x_916_ = l_Array_append___redArg(v___x_911_, v_a_894_);
lean_dec(v_a_894_);
v___x_917_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_917_, 0, v___x_901_);
lean_ctor_set(v___x_917_, 1, v___x_909_);
lean_ctor_set(v___x_917_, 2, v___x_916_);
lean_inc(v_ctorIdent_848_);
v___x_918_ = l_Lean_Syntax_node2(v___x_901_, v___x_915_, v_ctorIdent_848_, v___x_917_);
v___x_919_ = l_Lean_Syntax_node4(v___x_901_, v___x_908_, v___x_910_, v___x_912_, v___x_914_, v___x_918_);
v___x_920_ = l_Lean_Syntax_node2(v___x_901_, v___x_906_, v___x_907_, v___x_919_);
v___x_921_ = lean_array_get_borrowed(v___x_899_, v_argTypes_849_, v_a_852_);
v___x_922_ = l_Lean_Expr_isAppOf(v___x_921_, v_targetTypeName_850_);
if (v___x_922_ == 0)
{
lean_object* v___x_923_; 
v___x_923_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0(v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v_a_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
v_a_924_ = lean_ctor_get(v___x_923_, 0);
lean_inc_n(v_a_924_, 3);
lean_dec_ref_known(v___x_923_, 1);
v___x_925_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__16);
v___x_926_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__18));
lean_inc(v_currMacroScope_897_);
lean_inc(v_quotContext_896_);
v___x_927_ = l_Lean_addMacroScope(v_quotContext_896_, v___x_926_, v_currMacroScope_897_);
v___x_928_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__21));
v___x_929_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_929_, 0, v_a_924_);
lean_ctor_set(v___x_929_, 1, v___x_925_);
lean_ctor_set(v___x_929_, 2, v___x_927_);
lean_ctor_set(v___x_929_, 3, v___x_928_);
lean_inc(v___x_891_);
v___x_930_ = l_Lean_Syntax_node1(v_a_924_, v___x_909_, v___x_891_);
v___x_931_ = l_Lean_Syntax_node2(v_a_924_, v___x_915_, v___x_929_, v___x_930_);
v___x_932_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1(v___f_898_, v___x_902_, v___x_903_, v___x_904_, v___x_909_, v___x_920_, v___x_915_, v_b_853_, v___x_931_, v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
v___y_862_ = v___x_932_;
goto v___jp_861_;
}
else
{
lean_object* v_a_933_; lean_object* v___x_935_; uint8_t v_isShared_936_; uint8_t v_isSharedCheck_940_; 
lean_dec(v___x_920_);
lean_dec_ref(v_b_853_);
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v_a_933_ = lean_ctor_get(v___x_923_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_923_);
if (v_isSharedCheck_940_ == 0)
{
v___x_935_ = v___x_923_;
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
else
{
lean_inc(v_a_933_);
lean_dec(v___x_923_);
v___x_935_ = lean_box(0);
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
v_resetjp_934_:
{
lean_object* v___x_938_; 
if (v_isShared_936_ == 0)
{
v___x_938_ = v___x_935_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v_a_933_);
v___x_938_ = v_reuseFailAlloc_939_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
return v___x_938_;
}
}
}
}
else
{
lean_object* v___x_941_; 
v___x_941_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__0(v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
if (lean_obj_tag(v___x_941_) == 0)
{
lean_object* v_a_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v_a_942_ = lean_ctor_get(v___x_941_, 0);
lean_inc_n(v_a_942_, 2);
lean_dec_ref_known(v___x_941_, 1);
lean_inc(v___x_891_);
v___x_943_ = l_Lean_Syntax_node1(v_a_942_, v___x_909_, v___x_891_);
lean_inc(v_auxFn_851_);
v___x_944_ = l_Lean_Syntax_node2(v_a_942_, v___x_915_, v_auxFn_851_, v___x_943_);
v___x_945_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1(v___f_898_, v___x_902_, v___x_903_, v___x_904_, v___x_909_, v___x_920_, v___x_915_, v_b_853_, v___x_944_, v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
v___y_862_ = v___x_945_;
goto v___jp_861_;
}
else
{
lean_object* v_a_946_; lean_object* v___x_948_; uint8_t v_isShared_949_; uint8_t v_isSharedCheck_953_; 
lean_dec(v___x_920_);
lean_dec_ref(v_b_853_);
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v_a_946_ = lean_ctor_get(v___x_941_, 0);
v_isSharedCheck_953_ = !lean_is_exclusive(v___x_941_);
if (v_isSharedCheck_953_ == 0)
{
v___x_948_ = v___x_941_;
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
else
{
lean_inc(v_a_946_);
lean_dec(v___x_941_);
v___x_948_ = lean_box(0);
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
v_resetjp_947_:
{
lean_object* v___x_951_; 
if (v_isShared_949_ == 0)
{
v___x_951_ = v___x_948_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v_a_946_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
return v___x_951_;
}
}
}
}
}
else
{
lean_dec(v___x_892_);
lean_dec_ref(v_b_853_);
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
return v___x_893_;
}
}
else
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_961_; 
lean_dec_ref(v_b_853_);
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v_a_954_ = lean_ctor_get(v___x_887_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_887_);
if (v_isSharedCheck_961_ == 0)
{
v___x_956_ = v___x_887_;
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_887_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_959_; 
if (v_isShared_957_ == 0)
{
v___x_959_ = v___x_956_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_a_954_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
}
v___jp_861_:
{
if (lean_obj_tag(v___y_862_) == 0)
{
lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_875_; 
v_a_863_ = lean_ctor_get(v___y_862_, 0);
v_isSharedCheck_875_ = !lean_is_exclusive(v___y_862_);
if (v_isSharedCheck_875_ == 0)
{
v___x_865_ = v___y_862_;
v_isShared_866_ = v_isSharedCheck_875_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___y_862_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_875_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
if (lean_obj_tag(v_a_863_) == 0)
{
lean_object* v_a_867_; lean_object* v___x_869_; 
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v_a_867_ = lean_ctor_get(v_a_863_, 0);
lean_inc(v_a_867_);
lean_dec_ref_known(v_a_863_, 1);
if (v_isShared_866_ == 0)
{
lean_ctor_set(v___x_865_, 0, v_a_867_);
v___x_869_ = v___x_865_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_a_867_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
else
{
lean_object* v_a_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
lean_del_object(v___x_865_);
v_a_871_ = lean_ctor_get(v_a_863_, 0);
lean_inc(v_a_871_);
lean_dec_ref_known(v_a_863_, 1);
v___x_872_ = lean_unsigned_to_nat(1u);
v___x_873_ = lean_nat_add(v_a_852_, v___x_872_);
lean_dec(v_a_852_);
v_a_852_ = v___x_873_;
v_b_853_ = v_a_871_;
goto _start;
}
}
}
else
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_883_; 
lean_dec(v_a_852_);
lean_dec(v_auxFn_851_);
lean_dec(v_ctorIdent_848_);
v_a_876_ = lean_ctor_get(v___y_862_, 0);
v_isSharedCheck_883_ = !lean_is_exclusive(v___y_862_);
if (v_isSharedCheck_883_ == 0)
{
v___x_878_ = v___y_862_;
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___y_862_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_881_; 
if (v_isShared_879_ == 0)
{
v___x_881_ = v___x_878_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_a_876_);
v___x_881_ = v_reuseFailAlloc_882_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
return v___x_881_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___boxed(lean_object* v_upperBound_962_, lean_object* v___x_963_, lean_object* v_freshIdents_964_, lean_object* v_ctorIdent_965_, lean_object* v_argTypes_966_, lean_object* v_targetTypeName_967_, lean_object* v_auxFn_968_, lean_object* v_a_969_, lean_object* v_b_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg(v_upperBound_962_, v___x_963_, v_freshIdents_964_, v_ctorIdent_965_, v_argTypes_966_, v_targetTypeName_967_, v_auxFn_968_, v_a_969_, v_b_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
lean_dec(v___y_974_);
lean_dec_ref(v___y_973_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
lean_dec(v_targetTypeName_967_);
lean_dec_ref(v_argTypes_966_);
lean_dec_ref(v_freshIdents_964_);
lean_dec(v___x_963_);
lean_dec(v_upperBound_962_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg(lean_object* v_x_983_, lean_object* v_x_984_, lean_object* v___y_985_){
_start:
{
if (lean_obj_tag(v_x_984_) == 0)
{
lean_object* v___x_987_; 
v___x_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_987_, 0, v_x_983_);
return v___x_987_;
}
else
{
lean_object* v_head_988_; lean_object* v_tail_989_; lean_object* v___x_991_; uint8_t v_isShared_992_; uint8_t v_isSharedCheck_1003_; 
v_head_988_ = lean_ctor_get(v_x_984_, 0);
v_tail_989_ = lean_ctor_get(v_x_984_, 1);
v_isSharedCheck_1003_ = !lean_is_exclusive(v_x_984_);
if (v_isSharedCheck_1003_ == 0)
{
v___x_991_ = v_x_984_;
v_isShared_992_ = v_isSharedCheck_1003_;
goto v_resetjp_990_;
}
else
{
lean_inc(v_tail_989_);
lean_inc(v_head_988_);
lean_dec(v_x_984_);
v___x_991_ = lean_box(0);
v_isShared_992_ = v_isSharedCheck_1003_;
goto v_resetjp_990_;
}
v_resetjp_990_:
{
lean_object* v_ref_993_; uint8_t v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_999_; 
v_ref_993_ = lean_ctor_get(v___y_985_, 5);
v___x_994_ = 0;
v___x_995_ = l_Lean_SourceInfo_fromRef(v_ref_993_, v___x_994_);
v___x_996_ = ((lean_object*)(lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__1));
v___x_997_ = ((lean_object*)(lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___closed__2));
lean_inc(v___x_995_);
if (v_isShared_992_ == 0)
{
lean_ctor_set_tag(v___x_991_, 2);
lean_ctor_set(v___x_991_, 1, v___x_997_);
lean_ctor_set(v___x_991_, 0, v___x_995_);
v___x_999_ = v___x_991_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1002_; 
v_reuseFailAlloc_1002_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1002_, 0, v___x_995_);
lean_ctor_set(v_reuseFailAlloc_1002_, 1, v___x_997_);
v___x_999_ = v_reuseFailAlloc_1002_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
lean_object* v___x_1000_; 
v___x_1000_ = l_Lean_Syntax_node3(v___x_995_, v___x_996_, v_x_983_, v___x_999_, v_head_988_);
v_x_983_ = v___x_1000_;
v_x_984_ = v_tail_989_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg___boxed(lean_object* v_x_1004_, lean_object* v_x_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_){
_start:
{
lean_object* v_res_1008_; 
v_res_1008_ = lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg(v_x_1004_, v_x_1005_, v___y_1006_);
lean_dec_ref(v___y_1006_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr(lean_object* v_targetTypeName_1009_, lean_object* v_ctorIdent_1010_, lean_object* v_freshIdents_1011_, lean_object* v_argTypes_1012_, lean_object* v_auxFn_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_, lean_object* v_a_1017_, lean_object* v_a_1018_, lean_object* v_a_1019_){
_start:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v_listTerms_1023_; lean_object* v___x_1024_; 
v___x_1021_ = lean_unsigned_to_nat(0u);
v___x_1022_ = lean_array_get_size(v_freshIdents_1011_);
v_listTerms_1023_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2));
v___x_1024_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg(v___x_1022_, v_argTypes_1012_, v_targetTypeName_1009_, v_freshIdents_1011_, v___x_1021_, v_listTerms_1023_, v_a_1018_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v_a_1025_; lean_object* v___x_1026_; 
v_a_1025_ = lean_ctor_get(v___x_1024_, 0);
lean_inc(v_a_1025_);
lean_dec_ref_known(v___x_1024_, 1);
v___x_1026_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg(v___x_1022_, v___x_1022_, v_freshIdents_1011_, v_ctorIdent_1010_, v_argTypes_1012_, v_targetTypeName_1009_, v_auxFn_1013_, v___x_1021_, v_a_1025_, v_a_1014_, v_a_1015_, v_a_1016_, v_a_1017_, v_a_1018_, v_a_1019_);
if (lean_obj_tag(v___x_1026_) == 0)
{
lean_object* v_a_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1054_; 
v_a_1027_ = lean_ctor_get(v___x_1026_, 0);
v_isSharedCheck_1054_ = !lean_is_exclusive(v___x_1026_);
if (v_isSharedCheck_1054_ == 0)
{
v___x_1029_ = v___x_1026_;
v_isShared_1030_ = v_isSharedCheck_1054_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_a_1027_);
lean_dec(v___x_1026_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1054_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1031_; 
v___x_1031_ = lean_array_to_list(v_a_1027_);
if (lean_obj_tag(v___x_1031_) == 0)
{
lean_object* v_ref_1032_; uint8_t v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1045_; 
v_ref_1032_ = lean_ctor_get(v_a_1018_, 5);
v___x_1033_ = 0;
v___x_1034_ = l_Lean_SourceInfo_fromRef(v_ref_1032_, v___x_1033_);
v___x_1035_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__1));
v___x_1036_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__2));
lean_inc_n(v___x_1034_, 3);
v___x_1037_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1037_, 0, v___x_1034_);
lean_ctor_set(v___x_1037_, 1, v___x_1036_);
v___x_1038_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_1039_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
v___x_1040_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1034_);
lean_ctor_set(v___x_1040_, 1, v___x_1038_);
lean_ctor_set(v___x_1040_, 2, v___x_1039_);
v___x_1041_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__5));
v___x_1042_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1034_);
lean_ctor_set(v___x_1042_, 1, v___x_1041_);
v___x_1043_ = l_Lean_Syntax_node3(v___x_1034_, v___x_1035_, v___x_1037_, v___x_1040_, v___x_1042_);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v___x_1043_);
v___x_1045_ = v___x_1029_;
goto v_reusejp_1044_;
}
else
{
lean_object* v_reuseFailAlloc_1046_; 
v_reuseFailAlloc_1046_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1046_, 0, v___x_1043_);
v___x_1045_ = v_reuseFailAlloc_1046_;
goto v_reusejp_1044_;
}
v_reusejp_1044_:
{
return v___x_1045_;
}
}
else
{
lean_object* v_tail_1047_; 
v_tail_1047_ = lean_ctor_get(v___x_1031_, 1);
lean_inc(v_tail_1047_);
if (lean_obj_tag(v_tail_1047_) == 0)
{
lean_object* v_head_1048_; lean_object* v___x_1050_; 
v_head_1048_ = lean_ctor_get(v___x_1031_, 0);
lean_inc(v_head_1048_);
lean_dec_ref_known(v___x_1031_, 2);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v_head_1048_);
v___x_1050_ = v___x_1029_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v_head_1048_);
v___x_1050_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
return v___x_1050_;
}
}
else
{
lean_object* v_head_1052_; lean_object* v___x_1053_; 
lean_del_object(v___x_1029_);
v_head_1052_ = lean_ctor_get(v___x_1031_, 0);
lean_inc(v_head_1052_);
lean_dec_ref_known(v___x_1031_, 2);
v___x_1053_ = lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg(v_head_1052_, v_tail_1047_, v_a_1018_);
return v___x_1053_;
}
}
}
}
else
{
lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1062_; 
v_a_1055_ = lean_ctor_get(v___x_1026_, 0);
v_isSharedCheck_1062_ = !lean_is_exclusive(v___x_1026_);
if (v_isSharedCheck_1062_ == 0)
{
v___x_1057_ = v___x_1026_;
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_1026_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
lean_object* v___x_1060_; 
if (v_isShared_1058_ == 0)
{
v___x_1060_ = v___x_1057_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v_a_1055_);
v___x_1060_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
return v___x_1060_;
}
}
}
}
else
{
lean_object* v_a_1063_; lean_object* v___x_1065_; uint8_t v_isShared_1066_; uint8_t v_isSharedCheck_1070_; 
lean_dec(v_auxFn_1013_);
lean_dec(v_ctorIdent_1010_);
v_a_1063_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1065_ = v___x_1024_;
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
else
{
lean_inc(v_a_1063_);
lean_dec(v___x_1024_);
v___x_1065_ = lean_box(0);
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
v_resetjp_1064_:
{
lean_object* v___x_1068_; 
if (v_isShared_1066_ == 0)
{
v___x_1068_ = v___x_1065_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_a_1063_);
v___x_1068_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
return v___x_1068_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr___boxed(lean_object* v_targetTypeName_1071_, lean_object* v_ctorIdent_1072_, lean_object* v_freshIdents_1073_, lean_object* v_argTypes_1074_, lean_object* v_auxFn_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr(v_targetTypeName_1071_, v_ctorIdent_1072_, v_freshIdents_1073_, v_argTypes_1074_, v_auxFn_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_, v_a_1080_, v_a_1081_);
lean_dec(v_a_1081_);
lean_dec_ref(v_a_1080_);
lean_dec(v_a_1079_);
lean_dec_ref(v_a_1078_);
lean_dec(v_a_1077_);
lean_dec_ref(v_a_1076_);
lean_dec_ref(v_argTypes_1074_);
lean_dec_ref(v_freshIdents_1073_);
lean_dec(v_targetTypeName_1071_);
return v_res_1083_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0(lean_object* v_x_1084_, lean_object* v_x_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_){
_start:
{
lean_object* v___x_1093_; 
v___x_1093_ = lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___redArg(v_x_1084_, v_x_1085_, v___y_1090_);
return v___x_1093_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0___boxed(lean_object* v_x_1094_, lean_object* v_x_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_plausible_List_foldlM___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__0(v_x_1094_, v_x_1095_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
lean_dec(v___y_1097_);
lean_dec_ref(v___y_1096_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1(lean_object* v_upperBound_1104_, lean_object* v_next_1105_, lean_object* v_freshIdents_1106_, lean_object* v___x_1107_, lean_object* v_inst_1108_, lean_object* v_R_1109_, lean_object* v_a_1110_, lean_object* v_b_1111_, lean_object* v_c_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_){
_start:
{
lean_object* v___x_1120_; 
v___x_1120_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___redArg(v_upperBound_1104_, v_next_1105_, v_freshIdents_1106_, v___x_1107_, v_a_1110_, v_b_1111_);
return v___x_1120_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1___boxed(lean_object* v_upperBound_1121_, lean_object* v_next_1122_, lean_object* v_freshIdents_1123_, lean_object* v___x_1124_, lean_object* v_inst_1125_, lean_object* v_R_1126_, lean_object* v_a_1127_, lean_object* v_b_1128_, lean_object* v_c_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__1(v_upperBound_1121_, v_next_1122_, v_freshIdents_1123_, v___x_1124_, v_inst_1125_, v_R_1126_, v_a_1127_, v_b_1128_, v_c_1129_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_, v___y_1135_);
lean_dec(v___y_1135_);
lean_dec_ref(v___y_1134_);
lean_dec(v___y_1133_);
lean_dec_ref(v___y_1132_);
lean_dec(v___y_1131_);
lean_dec_ref(v___y_1130_);
lean_dec_ref(v_freshIdents_1123_);
lean_dec(v_next_1122_);
lean_dec(v_upperBound_1121_);
return v_res_1137_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2(lean_object* v_upperBound_1138_, lean_object* v___x_1139_, lean_object* v_freshIdents_1140_, lean_object* v_ctorIdent_1141_, lean_object* v_argTypes_1142_, lean_object* v_targetTypeName_1143_, lean_object* v_auxFn_1144_, lean_object* v_inst_1145_, lean_object* v_R_1146_, lean_object* v_a_1147_, lean_object* v_b_1148_, lean_object* v_c_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_){
_start:
{
lean_object* v___x_1157_; 
v___x_1157_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg(v_upperBound_1138_, v___x_1139_, v_freshIdents_1140_, v_ctorIdent_1141_, v_argTypes_1142_, v_targetTypeName_1143_, v_auxFn_1144_, v_a_1147_, v_b_1148_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_);
return v___x_1157_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___boxed(lean_object** _args){
lean_object* v_upperBound_1158_ = _args[0];
lean_object* v___x_1159_ = _args[1];
lean_object* v_freshIdents_1160_ = _args[2];
lean_object* v_ctorIdent_1161_ = _args[3];
lean_object* v_argTypes_1162_ = _args[4];
lean_object* v_targetTypeName_1163_ = _args[5];
lean_object* v_auxFn_1164_ = _args[6];
lean_object* v_inst_1165_ = _args[7];
lean_object* v_R_1166_ = _args[8];
lean_object* v_a_1167_ = _args[9];
lean_object* v_b_1168_ = _args[10];
lean_object* v_c_1169_ = _args[11];
lean_object* v___y_1170_ = _args[12];
lean_object* v___y_1171_ = _args[13];
lean_object* v___y_1172_ = _args[14];
lean_object* v___y_1173_ = _args[15];
lean_object* v___y_1174_ = _args[16];
lean_object* v___y_1175_ = _args[17];
lean_object* v___y_1176_ = _args[18];
_start:
{
lean_object* v_res_1177_; 
v_res_1177_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2(v_upperBound_1158_, v___x_1159_, v_freshIdents_1160_, v_ctorIdent_1161_, v_argTypes_1162_, v_targetTypeName_1163_, v_auxFn_1164_, v_inst_1165_, v_R_1166_, v_a_1167_, v_b_1168_, v_c_1169_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
lean_dec(v___y_1173_);
lean_dec_ref(v___y_1172_);
lean_dec(v___y_1171_);
lean_dec_ref(v___y_1170_);
lean_dec(v_targetTypeName_1163_);
lean_dec_ref(v_argTypes_1162_);
lean_dec_ref(v_freshIdents_1160_);
lean_dec(v___x_1159_);
lean_dec(v_upperBound_1158_);
return v_res_1177_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3(lean_object* v_upperBound_1178_, lean_object* v_argTypes_1179_, lean_object* v_targetTypeName_1180_, lean_object* v_freshIdents_1181_, lean_object* v_inst_1182_, lean_object* v_R_1183_, lean_object* v_a_1184_, lean_object* v_b_1185_, lean_object* v_c_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v___x_1194_; 
v___x_1194_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg(v_upperBound_1178_, v_argTypes_1179_, v_targetTypeName_1180_, v_freshIdents_1181_, v_a_1184_, v_b_1185_, v___y_1191_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___boxed(lean_object* v_upperBound_1195_, lean_object* v_argTypes_1196_, lean_object* v_targetTypeName_1197_, lean_object* v_freshIdents_1198_, lean_object* v_inst_1199_, lean_object* v_R_1200_, lean_object* v_a_1201_, lean_object* v_b_1202_, lean_object* v_c_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_){
_start:
{
lean_object* v_res_1211_; 
v_res_1211_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3(v_upperBound_1195_, v_argTypes_1196_, v_targetTypeName_1197_, v_freshIdents_1198_, v_inst_1199_, v_R_1200_, v_a_1201_, v_b_1202_, v_c_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_, v___y_1208_, v___y_1209_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
lean_dec(v___y_1207_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec_ref(v_freshIdents_1198_);
lean_dec(v_targetTypeName_1197_);
lean_dec_ref(v_argTypes_1196_);
lean_dec(v_upperBound_1195_);
return v_res_1211_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_){
_start:
{
lean_object* v_ref_1219_; uint8_t v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; 
v_ref_1219_ = lean_ctor_get(v___y_1216_, 5);
v___x_1220_ = 0;
v___x_1221_ = l_Lean_SourceInfo_fromRef(v_ref_1219_, v___x_1220_);
v___x_1222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1221_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0___boxed(lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1223_, v___y_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
lean_dec(v___y_1224_);
lean_dec_ref(v___y_1223_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1(size_t v_sz_1231_, size_t v_i_1232_, lean_object* v_bs_1233_){
_start:
{
uint8_t v___x_1234_; 
v___x_1234_ = lean_usize_dec_lt(v_i_1232_, v_sz_1231_);
if (v___x_1234_ == 0)
{
return v_bs_1233_;
}
else
{
lean_object* v_v_1235_; lean_object* v_fst_1236_; lean_object* v___x_1237_; lean_object* v_bs_x27_1238_; size_t v___x_1239_; size_t v___x_1240_; lean_object* v___x_1241_; 
v_v_1235_ = lean_array_uget_borrowed(v_bs_1233_, v_i_1232_);
v_fst_1236_ = lean_ctor_get(v_v_1235_, 0);
lean_inc(v_fst_1236_);
v___x_1237_ = lean_unsigned_to_nat(0u);
v_bs_x27_1238_ = lean_array_uset(v_bs_1233_, v_i_1232_, v___x_1237_);
v___x_1239_ = ((size_t)1ULL);
v___x_1240_ = lean_usize_add(v_i_1232_, v___x_1239_);
v___x_1241_ = lean_array_uset(v_bs_x27_1238_, v_i_1232_, v_fst_1236_);
v_i_1232_ = v___x_1240_;
v_bs_1233_ = v___x_1241_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1___boxed(lean_object* v_sz_1243_, lean_object* v_i_1244_, lean_object* v_bs_1245_){
_start:
{
size_t v_sz_boxed_1246_; size_t v_i_boxed_1247_; lean_object* v_res_1248_; 
v_sz_boxed_1246_ = lean_unbox_usize(v_sz_1243_);
lean_dec(v_sz_1243_);
v_i_boxed_1247_ = lean_unbox_usize(v_i_1244_);
lean_dec(v_i_1244_);
v_res_1248_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1(v_sz_boxed_1246_, v_i_boxed_1247_, v_bs_1245_);
return v_res_1248_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0(size_t v_sz_1249_, size_t v_i_1250_, lean_object* v_bs_1251_){
_start:
{
uint8_t v___x_1252_; 
v___x_1252_ = lean_usize_dec_lt(v_i_1250_, v_sz_1249_);
if (v___x_1252_ == 0)
{
return v_bs_1251_;
}
else
{
lean_object* v_v_1253_; lean_object* v_snd_1254_; lean_object* v___x_1255_; lean_object* v_bs_x27_1256_; size_t v___x_1257_; size_t v___x_1258_; lean_object* v___x_1259_; 
v_v_1253_ = lean_array_uget_borrowed(v_bs_1251_, v_i_1250_);
v_snd_1254_ = lean_ctor_get(v_v_1253_, 1);
lean_inc(v_snd_1254_);
v___x_1255_ = lean_unsigned_to_nat(0u);
v_bs_x27_1256_ = lean_array_uset(v_bs_1251_, v_i_1250_, v___x_1255_);
v___x_1257_ = ((size_t)1ULL);
v___x_1258_ = lean_usize_add(v_i_1250_, v___x_1257_);
v___x_1259_ = lean_array_uset(v_bs_x27_1256_, v_i_1250_, v_snd_1254_);
v_i_1250_ = v___x_1258_;
v_bs_1251_ = v___x_1259_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0___boxed(lean_object* v_sz_1261_, lean_object* v_i_1262_, lean_object* v_bs_1263_){
_start:
{
size_t v_sz_boxed_1264_; size_t v_i_boxed_1265_; lean_object* v_res_1266_; 
v_sz_boxed_1264_ = lean_unbox_usize(v_sz_1261_);
lean_dec(v_sz_1261_);
v_i_boxed_1265_ = lean_unbox_usize(v_i_1262_);
lean_dec(v_i_1262_);
v_res_1266_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0(v_sz_boxed_1264_, v_i_boxed_1265_, v_bs_1263_);
return v_res_1266_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3(size_t v_sz_1267_, size_t v_i_1268_, lean_object* v_bs_1269_){
_start:
{
uint8_t v___x_1270_; 
v___x_1270_ = lean_usize_dec_lt(v_i_1268_, v_sz_1267_);
if (v___x_1270_ == 0)
{
return v_bs_1269_;
}
else
{
lean_object* v_v_1271_; lean_object* v___x_1272_; lean_object* v_bs_x27_1273_; size_t v___x_1274_; size_t v___x_1275_; lean_object* v___x_1276_; 
v_v_1271_ = lean_array_uget(v_bs_1269_, v_i_1268_);
v___x_1272_ = lean_unsigned_to_nat(0u);
v_bs_x27_1273_ = lean_array_uset(v_bs_1269_, v_i_1268_, v___x_1272_);
v___x_1274_ = ((size_t)1ULL);
v___x_1275_ = lean_usize_add(v_i_1268_, v___x_1274_);
v___x_1276_ = lean_array_uset(v_bs_x27_1273_, v_i_1268_, v_v_1271_);
v_i_1268_ = v___x_1275_;
v_bs_1269_ = v___x_1276_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3___boxed(lean_object* v_sz_1278_, lean_object* v_i_1279_, lean_object* v_bs_1280_){
_start:
{
size_t v_sz_boxed_1281_; size_t v_i_boxed_1282_; lean_object* v_res_1283_; 
v_sz_boxed_1281_ = lean_unbox_usize(v_sz_1278_);
lean_dec(v_sz_1278_);
v_i_boxed_1282_ = lean_unbox_usize(v_i_1279_);
lean_dec(v_i_1279_);
v_res_1283_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3(v_sz_boxed_1281_, v_i_boxed_1282_, v_bs_1280_);
return v_res_1283_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2(size_t v_sz_1284_, size_t v_i_1285_, lean_object* v_bs_1286_){
_start:
{
uint8_t v___x_1287_; 
v___x_1287_ = lean_usize_dec_lt(v_i_1285_, v_sz_1284_);
if (v___x_1287_ == 0)
{
return v_bs_1286_;
}
else
{
lean_object* v_v_1288_; lean_object* v___x_1289_; lean_object* v_bs_x27_1290_; lean_object* v___x_1291_; size_t v___x_1292_; size_t v___x_1293_; lean_object* v___x_1294_; 
v_v_1288_ = lean_array_uget(v_bs_1286_, v_i_1285_);
v___x_1289_ = lean_unsigned_to_nat(0u);
v_bs_x27_1290_ = lean_array_uset(v_bs_1286_, v_i_1285_, v___x_1289_);
v___x_1291_ = l_Lean_mkIdent(v_v_1288_);
v___x_1292_ = ((size_t)1ULL);
v___x_1293_ = lean_usize_add(v_i_1285_, v___x_1292_);
v___x_1294_ = lean_array_uset(v_bs_x27_1290_, v_i_1285_, v___x_1291_);
v_i_1285_ = v___x_1293_;
v_bs_1286_ = v___x_1294_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2___boxed(lean_object* v_sz_1296_, lean_object* v_i_1297_, lean_object* v_bs_1298_){
_start:
{
size_t v_sz_boxed_1299_; size_t v_i_boxed_1300_; lean_object* v_res_1301_; 
v_sz_boxed_1299_ = lean_unbox_usize(v_sz_1296_);
lean_dec(v_sz_1296_);
v_i_boxed_1300_ = lean_unbox_usize(v_i_1297_);
lean_dec(v_i_1297_);
v_res_1301_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2(v_sz_boxed_1299_, v_i_boxed_1300_, v_bs_1298_);
return v_res_1301_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg(lean_object* v_indVal_1309_, lean_object* v___x_1310_, lean_object* v_as_x27_1311_, lean_object* v_b_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_){
_start:
{
if (lean_obj_tag(v_as_x27_1311_) == 0)
{
lean_object* v___x_1320_; 
lean_dec(v___x_1310_);
lean_dec_ref(v_indVal_1309_);
v___x_1320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1320_, 0, v_b_1312_);
return v___x_1320_;
}
else
{
lean_object* v_head_1321_; lean_object* v_tail_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; 
v_head_1321_ = lean_ctor_get(v_as_x27_1311_, 0);
v_tail_1322_ = lean_ctor_get(v_as_x27_1311_, 1);
lean_inc_n(v_head_1321_, 2);
v___x_1323_ = l_Lean_mkIdent(v_head_1321_);
lean_inc_ref(v_indVal_1309_);
v___x_1324_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes(v_indVal_1309_, v_head_1321_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_);
if (lean_obj_tag(v___x_1324_) == 0)
{
lean_object* v_a_1325_; size_t v_sz_1326_; size_t v___x_1327_; lean_object* v___x_1328_; lean_object* v_toConstantVal_1329_; lean_object* v_name_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1385_; 
v_a_1325_ = lean_ctor_get(v___x_1324_, 0);
lean_inc_n(v_a_1325_, 2);
lean_dec_ref_known(v___x_1324_, 1);
v_sz_1326_ = lean_array_size(v_a_1325_);
v___x_1327_ = ((size_t)0ULL);
v___x_1328_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__0(v_sz_1326_, v___x_1327_, v_a_1325_);
v_toConstantVal_1329_ = lean_ctor_get(v_indVal_1309_, 0);
lean_inc_ref(v_toConstantVal_1329_);
v_name_1330_ = lean_ctor_get(v_toConstantVal_1329_, 0);
v_isSharedCheck_1385_ = !lean_is_exclusive(v_toConstantVal_1329_);
if (v_isSharedCheck_1385_ == 0)
{
lean_object* v_unused_1386_; lean_object* v_unused_1387_; 
v_unused_1386_ = lean_ctor_get(v_toConstantVal_1329_, 2);
lean_dec(v_unused_1386_);
v_unused_1387_ = lean_ctor_get(v_toConstantVal_1329_, 1);
lean_dec(v_unused_1387_);
v___x_1332_ = v_toConstantVal_1329_;
v_isShared_1333_ = v_isSharedCheck_1385_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_name_1330_);
lean_dec(v_toConstantVal_1329_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1385_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___x_1334_; size_t v_sz_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1334_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__1(v_sz_1326_, v___x_1327_, v_a_1325_);
v_sz_1335_ = lean_array_size(v___x_1334_);
v___x_1336_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__2(v_sz_1335_, v___x_1327_, v___x_1334_);
lean_inc(v___x_1310_);
lean_inc(v___x_1323_);
v___x_1337_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr(v_name_1330_, v___x_1323_, v___x_1336_, v___x_1328_, v___x_1310_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_);
lean_dec_ref(v___x_1328_);
lean_dec(v_name_1330_);
if (lean_obj_tag(v___x_1337_) == 0)
{
lean_object* v_a_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; uint8_t v___x_1341_; 
v_a_1338_ = lean_ctor_get(v___x_1337_, 0);
lean_inc(v_a_1338_);
lean_dec_ref_known(v___x_1337_, 1);
v___x_1339_ = lean_array_get_size(v___x_1336_);
v___x_1340_ = lean_unsigned_to_nat(0u);
v___x_1341_ = lean_nat_dec_eq(v___x_1339_, v___x_1340_);
if (v___x_1341_ == 0)
{
lean_object* v___x_1342_; lean_object* v_a_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; size_t v_sz_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1354_; 
v___x_1342_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_);
v_a_1343_ = lean_ctor_get(v___x_1342_, 0);
lean_inc_n(v_a_1343_, 3);
lean_dec_ref(v___x_1342_);
v___x_1344_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1));
v___x_1345_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__2));
v___x_1346_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1346_, 0, v_a_1343_);
lean_ctor_set(v___x_1346_, 1, v___x_1345_);
v___x_1347_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_1348_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14));
v___x_1349_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
v_sz_1350_ = lean_array_size(v___x_1336_);
v___x_1351_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__3(v_sz_1350_, v___x_1327_, v___x_1336_);
v___x_1352_ = l_Array_append___redArg(v___x_1349_, v___x_1351_);
lean_dec_ref(v___x_1351_);
if (v_isShared_1333_ == 0)
{
lean_ctor_set_tag(v___x_1332_, 1);
lean_ctor_set(v___x_1332_, 2, v___x_1352_);
lean_ctor_set(v___x_1332_, 1, v___x_1347_);
lean_ctor_set(v___x_1332_, 0, v_a_1343_);
v___x_1354_ = v___x_1332_;
goto v_reusejp_1353_;
}
else
{
lean_object* v_reuseFailAlloc_1363_; 
v_reuseFailAlloc_1363_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1363_, 0, v_a_1343_);
lean_ctor_set(v_reuseFailAlloc_1363_, 1, v___x_1347_);
lean_ctor_set(v_reuseFailAlloc_1363_, 2, v___x_1352_);
v___x_1354_ = v_reuseFailAlloc_1363_;
goto v_reusejp_1353_;
}
v_reusejp_1353_:
{
lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
lean_inc_n(v_a_1343_, 4);
v___x_1355_ = l_Lean_Syntax_node2(v_a_1343_, v___x_1348_, v___x_1323_, v___x_1354_);
v___x_1356_ = l_Lean_Syntax_node1(v_a_1343_, v___x_1347_, v___x_1355_);
v___x_1357_ = l_Lean_Syntax_node1(v_a_1343_, v___x_1347_, v___x_1356_);
v___x_1358_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12));
v___x_1359_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1359_, 0, v_a_1343_);
lean_ctor_set(v___x_1359_, 1, v___x_1358_);
v___x_1360_ = l_Lean_Syntax_node4(v_a_1343_, v___x_1344_, v___x_1346_, v___x_1357_, v___x_1359_, v_a_1338_);
v___x_1361_ = lean_array_push(v_b_1312_, v___x_1360_);
v_as_x27_1311_ = v_tail_1322_;
v_b_1312_ = v___x_1361_;
goto _start;
}
}
else
{
lean_object* v___x_1364_; lean_object* v_a_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; 
lean_dec_ref(v___x_1336_);
lean_del_object(v___x_1332_);
v___x_1364_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_);
v_a_1365_ = lean_ctor_get(v___x_1364_, 0);
lean_inc_n(v_a_1365_, 5);
lean_dec_ref(v___x_1364_);
v___x_1366_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__1));
v___x_1367_ = ((lean_object*)(lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___closed__2));
v___x_1368_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1368_, 0, v_a_1365_);
lean_ctor_set(v___x_1368_, 1, v___x_1367_);
v___x_1369_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_1370_ = l_Lean_Syntax_node1(v_a_1365_, v___x_1369_, v___x_1323_);
v___x_1371_ = l_Lean_Syntax_node1(v_a_1365_, v___x_1369_, v___x_1370_);
v___x_1372_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12));
v___x_1373_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1373_, 0, v_a_1365_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
v___x_1374_ = l_Lean_Syntax_node4(v_a_1365_, v___x_1366_, v___x_1368_, v___x_1371_, v___x_1373_, v_a_1338_);
v___x_1375_ = lean_array_push(v_b_1312_, v___x_1374_);
v_as_x27_1311_ = v_tail_1322_;
v_b_1312_ = v___x_1375_;
goto _start;
}
}
else
{
lean_object* v_a_1377_; lean_object* v___x_1379_; uint8_t v_isShared_1380_; uint8_t v_isSharedCheck_1384_; 
lean_dec_ref(v___x_1336_);
lean_del_object(v___x_1332_);
lean_dec(v___x_1323_);
lean_dec_ref(v_b_1312_);
lean_dec(v___x_1310_);
lean_dec_ref(v_indVal_1309_);
v_a_1377_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1379_ = v___x_1337_;
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
else
{
lean_inc(v_a_1377_);
lean_dec(v___x_1337_);
v___x_1379_ = lean_box(0);
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
v_resetjp_1378_:
{
lean_object* v___x_1382_; 
if (v_isShared_1380_ == 0)
{
v___x_1382_ = v___x_1379_;
goto v_reusejp_1381_;
}
else
{
lean_object* v_reuseFailAlloc_1383_; 
v_reuseFailAlloc_1383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1383_, 0, v_a_1377_);
v___x_1382_ = v_reuseFailAlloc_1383_;
goto v_reusejp_1381_;
}
v_reusejp_1381_:
{
return v___x_1382_;
}
}
}
}
}
else
{
lean_object* v_a_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1395_; 
lean_dec(v___x_1323_);
lean_dec_ref(v_b_1312_);
lean_dec(v___x_1310_);
lean_dec_ref(v_indVal_1309_);
v_a_1388_ = lean_ctor_get(v___x_1324_, 0);
v_isSharedCheck_1395_ = !lean_is_exclusive(v___x_1324_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1390_ = v___x_1324_;
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_a_1388_);
lean_dec(v___x_1324_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v___x_1393_; 
if (v_isShared_1391_ == 0)
{
v___x_1393_ = v___x_1390_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v_a_1388_);
v___x_1393_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
return v___x_1393_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg___boxed(lean_object* v_indVal_1396_, lean_object* v___x_1397_, lean_object* v_as_x27_1398_, lean_object* v_b_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_){
_start:
{
lean_object* v_res_1407_; 
v_res_1407_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg(v_indVal_1396_, v___x_1397_, v_as_x27_1398_, v_b_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec(v___y_1401_);
lean_dec_ref(v___y_1400_);
lean_dec(v_as_x27_1398_);
return v_res_1407_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1(void){
_start:
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
v___x_1409_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__0));
v___x_1410_ = l_String_toRawSubstring_x27(v___x_1409_);
return v___x_1410_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction(lean_object* v_ctx_1606_, lean_object* v_i_1607_, lean_object* v_a_1608_, lean_object* v_a_1609_, lean_object* v_a_1610_, lean_object* v_a_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_){
_start:
{
lean_object* v_typeInfos_1615_; lean_object* v_auxFunNames_1616_; uint8_t v_usePartial_1617_; lean_object* v___x_1618_; lean_object* v_indVal_1619_; lean_object* v___x_1620_; 
v_typeInfos_1615_ = lean_ctor_get(v_ctx_1606_, 1);
v_auxFunNames_1616_ = lean_ctor_get(v_ctx_1606_, 2);
v_usePartial_1617_ = lean_ctor_get_uint8(v_ctx_1606_, sizeof(void*)*3);
v___x_1618_ = l_Lean_instInhabitedInductiveVal_default;
v_indVal_1619_ = lean_array_get_borrowed(v___x_1618_, v_typeInfos_1615_, v_i_1607_);
lean_inc(v_indVal_1619_);
v___x_1620_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader(v_indVal_1619_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
if (lean_obj_tag(v___x_1620_) == 0)
{
lean_object* v_a_1621_; lean_object* v_binders_1622_; lean_object* v_argNames_1623_; lean_object* v___x_1625_; uint8_t v_isShared_1626_; uint8_t v_isSharedCheck_1820_; 
v_a_1621_ = lean_ctor_get(v___x_1620_, 0);
lean_inc(v_a_1621_);
lean_dec_ref_known(v___x_1620_, 1);
v_binders_1622_ = lean_ctor_get(v_a_1621_, 0);
v_argNames_1623_ = lean_ctor_get(v_a_1621_, 1);
v_isSharedCheck_1820_ = !lean_is_exclusive(v_a_1621_);
if (v_isSharedCheck_1820_ == 0)
{
lean_object* v_unused_1821_; lean_object* v_unused_1822_; 
v_unused_1821_ = lean_ctor_get(v_a_1621_, 3);
lean_dec(v_unused_1821_);
v_unused_1822_ = lean_ctor_get(v_a_1621_, 2);
lean_dec(v_unused_1822_);
v___x_1625_ = v_a_1621_;
v_isShared_1626_ = v_isSharedCheck_1820_;
goto v_resetjp_1624_;
}
else
{
lean_inc(v_argNames_1623_);
lean_inc(v_binders_1622_);
lean_dec(v_a_1621_);
v___x_1625_ = lean_box(0);
v_isShared_1626_ = v_isSharedCheck_1820_;
goto v_resetjp_1624_;
}
v_resetjp_1624_:
{
lean_object* v___x_1627_; 
lean_inc_ref(v_argNames_1623_);
lean_inc(v_indVal_1619_);
v___x_1627_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v_indVal_1619_, v_argNames_1623_, v_a_1612_);
if (lean_obj_tag(v___x_1627_) == 0)
{
lean_object* v_a_1628_; lean_object* v_ref_1629_; lean_object* v_quotContext_1630_; lean_object* v_currMacroScope_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; 
v_a_1628_ = lean_ctor_get(v___x_1627_, 0);
lean_inc(v_a_1628_);
lean_dec_ref_known(v___x_1627_, 1);
v_ref_1629_ = lean_ctor_get(v_a_1612_, 5);
v_quotContext_1630_ = lean_ctor_get(v_a_1612_, 10);
v_currMacroScope_1631_ = lean_ctor_get(v_a_1612_, 11);
v___x_1632_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14));
v___x_1633_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1, &lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1_once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__1);
v___x_1634_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__2));
lean_inc(v_currMacroScope_1631_);
lean_inc(v_quotContext_1630_);
v___x_1635_ = l_Lean_addMacroScope(v_quotContext_1630_, v___x_1634_, v_currMacroScope_1631_);
v___x_1636_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__5));
v___x_1637_ = l_Lean_Core_mkFreshUserName(v___x_1636_, v_a_1612_, v_a_1613_);
if (lean_obj_tag(v___x_1637_) == 0)
{
lean_object* v_a_1638_; lean_object* v_ctors_1639_; lean_object* v___x_1640_; lean_object* v_auxFunName_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; 
v_a_1638_ = lean_ctor_get(v___x_1637_, 0);
lean_inc(v_a_1638_);
lean_dec_ref_known(v___x_1637_, 1);
v_ctors_1639_ = lean_ctor_get(v_indVal_1619_, 4);
v___x_1640_ = lean_box(0);
v_auxFunName_1641_ = lean_array_get_borrowed(v___x_1640_, v_auxFunNames_1616_, v_i_1607_);
lean_inc(v_auxFunName_1641_);
v___x_1642_ = l_Lean_mkIdent(v_auxFunName_1641_);
v___x_1643_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2));
lean_inc(v___x_1642_);
lean_inc(v_indVal_1619_);
v___x_1644_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg(v_indVal_1619_, v___x_1642_, v_ctors_1639_, v___x_1643_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
if (lean_obj_tag(v___x_1644_) == 0)
{
lean_object* v_a_1645_; lean_object* v___x_1646_; lean_object* v_a_1647_; uint8_t v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1652_; 
v_a_1645_ = lean_ctor_get(v___x_1644_, 0);
lean_inc(v_a_1645_);
lean_dec_ref_known(v___x_1644_, 1);
v___x_1646_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
v_a_1647_ = lean_ctor_get(v___x_1646_, 0);
lean_inc(v_a_1647_);
lean_dec_ref(v___x_1646_);
v___x_1648_ = 0;
v___x_1649_ = l_Lean_SourceInfo_fromRef(v_ref_1629_, v___x_1648_);
v___x_1650_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__12));
lean_inc(v___x_1649_);
if (v_isShared_1626_ == 0)
{
lean_ctor_set_tag(v___x_1625_, 3);
lean_ctor_set(v___x_1625_, 3, v___x_1650_);
lean_ctor_set(v___x_1625_, 2, v___x_1635_);
lean_ctor_set(v___x_1625_, 1, v___x_1633_);
lean_ctor_set(v___x_1625_, 0, v___x_1649_);
v___x_1652_ = v___x_1625_;
goto v_reusejp_1651_;
}
else
{
lean_object* v_reuseFailAlloc_1803_; 
v_reuseFailAlloc_1803_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1803_, 0, v___x_1649_);
lean_ctor_set(v_reuseFailAlloc_1803_, 1, v___x_1633_);
lean_ctor_set(v_reuseFailAlloc_1803_, 2, v___x_1635_);
lean_ctor_set(v_reuseFailAlloc_1803_, 3, v___x_1650_);
v___x_1652_ = v_reuseFailAlloc_1803_;
goto v_reusejp_1651_;
}
v_reusejp_1651_:
{
lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v_a_1672_; lean_object* v___x_1673_; lean_object* v_body_1675_; lean_object* v___y_1676_; lean_object* v___y_1677_; lean_object* v___y_1678_; lean_object* v___y_1679_; lean_object* v___y_1680_; lean_object* v___y_1681_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; 
v___x_1653_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
lean_inc_n(v_a_1628_, 2);
lean_inc(v___x_1649_);
v___x_1654_ = l_Lean_Syntax_node1(v___x_1649_, v___x_1653_, v_a_1628_);
v___x_1655_ = l_Lean_mkIdent(v_a_1638_);
v___x_1656_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__13));
v___x_1657_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__14));
lean_inc_n(v_a_1647_, 7);
v___x_1658_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1658_, 0, v_a_1647_);
lean_ctor_set(v___x_1658_, 1, v___x_1656_);
v___x_1659_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
v___x_1660_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1660_, 0, v_a_1647_);
lean_ctor_set(v___x_1660_, 1, v___x_1653_);
lean_ctor_set(v___x_1660_, 2, v___x_1659_);
v___x_1661_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__16));
lean_inc(v___x_1655_);
lean_inc_ref_n(v___x_1660_, 2);
v___x_1662_ = l_Lean_Syntax_node2(v_a_1647_, v___x_1661_, v___x_1660_, v___x_1655_);
v___x_1663_ = l_Lean_Syntax_node1(v_a_1647_, v___x_1653_, v___x_1662_);
v___x_1664_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__17));
v___x_1665_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1665_, 0, v_a_1647_);
lean_ctor_set(v___x_1665_, 1, v___x_1664_);
v___x_1666_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__19));
v___x_1667_ = l_Array_append___redArg(v___x_1659_, v_a_1645_);
lean_dec(v_a_1645_);
v___x_1668_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1668_, 0, v_a_1647_);
lean_ctor_set(v___x_1668_, 1, v___x_1653_);
lean_ctor_set(v___x_1668_, 2, v___x_1667_);
v___x_1669_ = l_Lean_Syntax_node1(v_a_1647_, v___x_1666_, v___x_1668_);
v___x_1670_ = l_Lean_Syntax_node6(v_a_1647_, v___x_1657_, v___x_1658_, v___x_1660_, v___x_1660_, v___x_1663_, v___x_1665_, v___x_1669_);
v___x_1671_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
v_a_1672_ = lean_ctor_get(v___x_1671_, 0);
lean_inc_n(v_a_1672_, 14);
lean_dec_ref(v___x_1671_);
v___x_1673_ = l_Lean_Syntax_node2(v___x_1649_, v___x_1632_, v___x_1652_, v___x_1654_);
v___x_1763_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__7));
v___x_1764_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__8));
v___x_1765_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1765_, 0, v_a_1672_);
lean_ctor_set(v___x_1765_, 1, v___x_1763_);
v___x_1766_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__10));
v___x_1767_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__46));
v___x_1768_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__47));
v___x_1769_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__3));
v___x_1770_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1770_, 0, v_a_1672_);
lean_ctor_set(v___x_1770_, 1, v___x_1769_);
v___x_1771_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__5));
v___x_1772_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__7);
lean_inc(v_currMacroScope_1631_);
lean_inc(v_quotContext_1630_);
v___x_1773_ = l_Lean_addMacroScope(v_quotContext_1630_, v___x_1640_, v_currMacroScope_1631_);
v___x_1774_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1));
v___x_1775_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__73));
v___x_1776_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1776_, 0, v_a_1672_);
lean_ctor_set(v___x_1776_, 1, v___x_1772_);
lean_ctor_set(v___x_1776_, 2, v___x_1773_);
lean_ctor_set(v___x_1776_, 3, v___x_1775_);
v___x_1777_ = l_Lean_Syntax_node1(v_a_1672_, v___x_1771_, v___x_1776_);
v___x_1778_ = l_Lean_Syntax_node2(v_a_1672_, v___x_1768_, v___x_1770_, v___x_1777_);
v___x_1779_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36));
v___x_1780_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1780_, 0, v_a_1672_);
lean_ctor_set(v___x_1780_, 1, v___x_1779_);
v___x_1781_ = l_Lean_Syntax_node1(v_a_1672_, v___x_1653_, v_a_1628_);
v___x_1782_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__15));
v___x_1783_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1783_, 0, v_a_1672_);
lean_ctor_set(v___x_1783_, 1, v___x_1782_);
v___x_1784_ = l_Lean_Syntax_node5(v_a_1672_, v___x_1767_, v___x_1778_, v___x_1655_, v___x_1780_, v___x_1781_, v___x_1783_);
v___x_1785_ = l_Lean_Syntax_node1(v_a_1672_, v___x_1653_, v___x_1784_);
v___x_1786_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1786_, 0, v_a_1672_);
lean_ctor_set(v___x_1786_, 1, v___x_1653_);
lean_ctor_set(v___x_1786_, 2, v___x_1659_);
v___x_1787_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__12));
v___x_1788_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1788_, 0, v_a_1672_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
v___x_1789_ = l_Lean_Syntax_node4(v_a_1672_, v___x_1766_, v___x_1785_, v___x_1786_, v___x_1788_, v___x_1670_);
v___x_1790_ = l_Lean_Syntax_node2(v_a_1672_, v___x_1764_, v___x_1765_, v___x_1789_);
if (v_usePartial_1617_ == 0)
{
lean_dec_ref(v_argNames_1623_);
v_body_1675_ = v___x_1790_;
v___y_1676_ = v_a_1608_;
v___y_1677_ = v_a_1609_;
v___y_1678_ = v_a_1610_;
v___y_1679_ = v_a_1611_;
v___y_1680_ = v_a_1612_;
v___y_1681_ = v_a_1613_;
goto v___jp_1674_;
}
else
{
lean_object* v___x_1791_; 
v___x_1791_ = l_Lean_Elab_Deriving_mkLocalInstanceLetDecls(v_ctx_1606_, v___x_1774_, v_argNames_1623_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
if (lean_obj_tag(v___x_1791_) == 0)
{
lean_object* v_a_1792_; lean_object* v___x_1793_; 
v_a_1792_ = lean_ctor_get(v___x_1791_, 0);
lean_inc(v_a_1792_);
lean_dec_ref_known(v___x_1791_, 1);
v___x_1793_ = l_Lean_Elab_Deriving_mkLet(v_a_1792_, v___x_1790_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
lean_dec(v_a_1792_);
if (lean_obj_tag(v___x_1793_) == 0)
{
lean_object* v_a_1794_; 
v_a_1794_ = lean_ctor_get(v___x_1793_, 0);
lean_inc(v_a_1794_);
lean_dec_ref_known(v___x_1793_, 1);
v_body_1675_ = v_a_1794_;
v___y_1676_ = v_a_1608_;
v___y_1677_ = v_a_1609_;
v___y_1678_ = v_a_1610_;
v___y_1679_ = v_a_1611_;
v___y_1680_ = v_a_1612_;
v___y_1681_ = v_a_1613_;
goto v___jp_1674_;
}
else
{
lean_dec(v___x_1673_);
lean_dec(v___x_1642_);
lean_dec(v_a_1628_);
lean_dec_ref(v_binders_1622_);
return v___x_1793_;
}
}
else
{
lean_object* v_a_1795_; lean_object* v___x_1797_; uint8_t v_isShared_1798_; uint8_t v_isSharedCheck_1802_; 
lean_dec(v___x_1790_);
lean_dec(v___x_1673_);
lean_dec(v___x_1642_);
lean_dec(v_a_1628_);
lean_dec_ref(v_binders_1622_);
v_a_1795_ = lean_ctor_get(v___x_1791_, 0);
v_isSharedCheck_1802_ = !lean_is_exclusive(v___x_1791_);
if (v_isSharedCheck_1802_ == 0)
{
v___x_1797_ = v___x_1791_;
v_isShared_1798_ = v_isSharedCheck_1802_;
goto v_resetjp_1796_;
}
else
{
lean_inc(v_a_1795_);
lean_dec(v___x_1791_);
v___x_1797_ = lean_box(0);
v_isShared_1798_ = v_isSharedCheck_1802_;
goto v_resetjp_1796_;
}
v_resetjp_1796_:
{
lean_object* v___x_1800_; 
if (v_isShared_1798_ == 0)
{
v___x_1800_ = v___x_1797_;
goto v_reusejp_1799_;
}
else
{
lean_object* v_reuseFailAlloc_1801_; 
v_reuseFailAlloc_1801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1801_, 0, v_a_1795_);
v___x_1800_ = v_reuseFailAlloc_1801_;
goto v_reusejp_1799_;
}
v_reusejp_1799_:
{
return v___x_1800_;
}
}
}
}
v___jp_1674_:
{
lean_object* v___x_1682_; lean_object* v_a_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; 
v___x_1682_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
v_a_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc_n(v_a_1683_, 2);
lean_dec_ref(v___x_1682_);
v___x_1684_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__21));
v___x_1685_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__22));
v___x_1686_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1686_, 0, v_a_1683_);
lean_ctor_set(v___x_1686_, 1, v___x_1685_);
v___x_1687_ = l_Lean_Syntax_node3(v_a_1683_, v___x_1684_, v_a_1628_, v___x_1686_, v___x_1673_);
if (v_usePartial_1617_ == 0)
{
lean_object* v___x_1688_; lean_object* v_a_1689_; lean_object* v___x_1691_; uint8_t v_isShared_1692_; uint8_t v_isSharedCheck_1722_; 
v___x_1688_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
v_a_1689_ = lean_ctor_get(v___x_1688_, 0);
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1688_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1691_ = v___x_1688_;
v_isShared_1692_ = v_isSharedCheck_1722_;
goto v_resetjp_1690_;
}
else
{
lean_inc(v_a_1689_);
lean_dec(v___x_1688_);
v___x_1691_ = lean_box(0);
v_isShared_1692_ = v_isSharedCheck_1722_;
goto v_resetjp_1690_;
}
v_resetjp_1690_:
{
lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1720_; 
v___x_1693_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24));
v___x_1694_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26));
lean_inc_n(v_a_1689_, 13);
v___x_1695_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1695_, 0, v_a_1689_);
lean_ctor_set(v___x_1695_, 1, v___x_1653_);
lean_ctor_set(v___x_1695_, 2, v___x_1659_);
lean_inc_ref_n(v___x_1695_, 11);
v___x_1696_ = l_Lean_Syntax_node7(v_a_1689_, v___x_1694_, v___x_1695_, v___x_1695_, v___x_1695_, v___x_1695_, v___x_1695_, v___x_1695_, v___x_1695_);
v___x_1697_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28));
v___x_1698_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__29));
v___x_1699_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1699_, 0, v_a_1689_);
lean_ctor_set(v___x_1699_, 1, v___x_1698_);
v___x_1700_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31));
v___x_1701_ = l_Lean_Syntax_node2(v_a_1689_, v___x_1700_, v___x_1642_, v___x_1695_);
v___x_1702_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33));
v___x_1703_ = l_Array_append___redArg(v___x_1659_, v_binders_1622_);
lean_dec_ref(v_binders_1622_);
v___x_1704_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1704_, 0, v_a_1689_);
lean_ctor_set(v___x_1704_, 1, v___x_1653_);
lean_ctor_set(v___x_1704_, 2, v___x_1703_);
v___x_1705_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35));
v___x_1706_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36));
v___x_1707_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1707_, 0, v_a_1689_);
lean_ctor_set(v___x_1707_, 1, v___x_1706_);
v___x_1708_ = l_Lean_Syntax_node2(v_a_1689_, v___x_1705_, v___x_1707_, v___x_1687_);
v___x_1709_ = l_Lean_Syntax_node1(v_a_1689_, v___x_1653_, v___x_1708_);
v___x_1710_ = l_Lean_Syntax_node2(v_a_1689_, v___x_1702_, v___x_1704_, v___x_1709_);
v___x_1711_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38));
v___x_1712_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39));
v___x_1713_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1713_, 0, v_a_1689_);
lean_ctor_set(v___x_1713_, 1, v___x_1712_);
v___x_1714_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42));
v___x_1715_ = l_Lean_Syntax_node2(v_a_1689_, v___x_1714_, v___x_1695_, v___x_1695_);
v___x_1716_ = l_Lean_Syntax_node4(v_a_1689_, v___x_1711_, v___x_1713_, v_body_1675_, v___x_1715_, v___x_1695_);
v___x_1717_ = l_Lean_Syntax_node5(v_a_1689_, v___x_1697_, v___x_1699_, v___x_1701_, v___x_1710_, v___x_1716_, v___x_1695_);
v___x_1718_ = l_Lean_Syntax_node2(v_a_1689_, v___x_1693_, v___x_1696_, v___x_1717_);
if (v_isShared_1692_ == 0)
{
lean_ctor_set(v___x_1691_, 0, v___x_1718_);
v___x_1720_ = v___x_1691_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v___x_1718_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
else
{
lean_object* v___x_1723_; lean_object* v_a_1724_; lean_object* v___x_1726_; uint8_t v_isShared_1727_; uint8_t v_isSharedCheck_1762_; 
v___x_1723_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___lam__0(v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
v_a_1724_ = lean_ctor_get(v___x_1723_, 0);
v_isSharedCheck_1762_ = !lean_is_exclusive(v___x_1723_);
if (v_isSharedCheck_1762_ == 0)
{
v___x_1726_ = v___x_1723_;
v_isShared_1727_ = v_isSharedCheck_1762_;
goto v_resetjp_1725_;
}
else
{
lean_inc(v_a_1724_);
lean_dec(v___x_1723_);
v___x_1726_ = lean_box(0);
v_isShared_1727_ = v_isSharedCheck_1762_;
goto v_resetjp_1725_;
}
v_resetjp_1725_:
{
lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1760_; 
v___x_1728_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24));
v___x_1729_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26));
lean_inc_n(v_a_1724_, 16);
v___x_1730_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1730_, 0, v_a_1724_);
lean_ctor_set(v___x_1730_, 1, v___x_1653_);
lean_ctor_set(v___x_1730_, 2, v___x_1659_);
v___x_1731_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__43));
v___x_1732_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__44));
v___x_1733_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1733_, 0, v_a_1724_);
lean_ctor_set(v___x_1733_, 1, v___x_1731_);
v___x_1734_ = l_Lean_Syntax_node1(v_a_1724_, v___x_1732_, v___x_1733_);
v___x_1735_ = l_Lean_Syntax_node1(v_a_1724_, v___x_1653_, v___x_1734_);
lean_inc_ref_n(v___x_1730_, 10);
v___x_1736_ = l_Lean_Syntax_node7(v_a_1724_, v___x_1729_, v___x_1730_, v___x_1730_, v___x_1730_, v___x_1730_, v___x_1730_, v___x_1730_, v___x_1735_);
v___x_1737_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__28));
v___x_1738_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__29));
v___x_1739_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1739_, 0, v_a_1724_);
lean_ctor_set(v___x_1739_, 1, v___x_1738_);
v___x_1740_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__31));
v___x_1741_ = l_Lean_Syntax_node2(v_a_1724_, v___x_1740_, v___x_1642_, v___x_1730_);
v___x_1742_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__33));
v___x_1743_ = l_Array_append___redArg(v___x_1659_, v_binders_1622_);
lean_dec_ref(v_binders_1622_);
v___x_1744_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1744_, 0, v_a_1724_);
lean_ctor_set(v___x_1744_, 1, v___x_1653_);
lean_ctor_set(v___x_1744_, 2, v___x_1743_);
v___x_1745_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35));
v___x_1746_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36));
v___x_1747_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1747_, 0, v_a_1724_);
lean_ctor_set(v___x_1747_, 1, v___x_1746_);
v___x_1748_ = l_Lean_Syntax_node2(v_a_1724_, v___x_1745_, v___x_1747_, v___x_1687_);
v___x_1749_ = l_Lean_Syntax_node1(v_a_1724_, v___x_1653_, v___x_1748_);
v___x_1750_ = l_Lean_Syntax_node2(v_a_1724_, v___x_1742_, v___x_1744_, v___x_1749_);
v___x_1751_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38));
v___x_1752_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39));
v___x_1753_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1753_, 0, v_a_1724_);
lean_ctor_set(v___x_1753_, 1, v___x_1752_);
v___x_1754_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42));
v___x_1755_ = l_Lean_Syntax_node2(v_a_1724_, v___x_1754_, v___x_1730_, v___x_1730_);
v___x_1756_ = l_Lean_Syntax_node4(v_a_1724_, v___x_1751_, v___x_1753_, v_body_1675_, v___x_1755_, v___x_1730_);
v___x_1757_ = l_Lean_Syntax_node5(v_a_1724_, v___x_1737_, v___x_1739_, v___x_1741_, v___x_1750_, v___x_1756_, v___x_1730_);
v___x_1758_ = l_Lean_Syntax_node2(v_a_1724_, v___x_1728_, v___x_1736_, v___x_1757_);
if (v_isShared_1727_ == 0)
{
lean_ctor_set(v___x_1726_, 0, v___x_1758_);
v___x_1760_ = v___x_1726_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1761_; 
v_reuseFailAlloc_1761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1761_, 0, v___x_1758_);
v___x_1760_ = v_reuseFailAlloc_1761_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
return v___x_1760_;
}
}
}
}
}
}
else
{
lean_object* v_a_1804_; lean_object* v___x_1806_; uint8_t v_isShared_1807_; uint8_t v_isSharedCheck_1811_; 
lean_dec(v___x_1642_);
lean_dec(v_a_1638_);
lean_dec(v___x_1635_);
lean_dec(v_a_1628_);
lean_del_object(v___x_1625_);
lean_dec_ref(v_argNames_1623_);
lean_dec_ref(v_binders_1622_);
v_a_1804_ = lean_ctor_get(v___x_1644_, 0);
v_isSharedCheck_1811_ = !lean_is_exclusive(v___x_1644_);
if (v_isSharedCheck_1811_ == 0)
{
v___x_1806_ = v___x_1644_;
v_isShared_1807_ = v_isSharedCheck_1811_;
goto v_resetjp_1805_;
}
else
{
lean_inc(v_a_1804_);
lean_dec(v___x_1644_);
v___x_1806_ = lean_box(0);
v_isShared_1807_ = v_isSharedCheck_1811_;
goto v_resetjp_1805_;
}
v_resetjp_1805_:
{
lean_object* v___x_1809_; 
if (v_isShared_1807_ == 0)
{
v___x_1809_ = v___x_1806_;
goto v_reusejp_1808_;
}
else
{
lean_object* v_reuseFailAlloc_1810_; 
v_reuseFailAlloc_1810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1810_, 0, v_a_1804_);
v___x_1809_ = v_reuseFailAlloc_1810_;
goto v_reusejp_1808_;
}
v_reusejp_1808_:
{
return v___x_1809_;
}
}
}
}
else
{
lean_object* v_a_1812_; lean_object* v___x_1814_; uint8_t v_isShared_1815_; uint8_t v_isSharedCheck_1819_; 
lean_dec(v___x_1635_);
lean_dec(v_a_1628_);
lean_del_object(v___x_1625_);
lean_dec_ref(v_argNames_1623_);
lean_dec_ref(v_binders_1622_);
v_a_1812_ = lean_ctor_get(v___x_1637_, 0);
v_isSharedCheck_1819_ = !lean_is_exclusive(v___x_1637_);
if (v_isSharedCheck_1819_ == 0)
{
v___x_1814_ = v___x_1637_;
v_isShared_1815_ = v_isSharedCheck_1819_;
goto v_resetjp_1813_;
}
else
{
lean_inc(v_a_1812_);
lean_dec(v___x_1637_);
v___x_1814_ = lean_box(0);
v_isShared_1815_ = v_isSharedCheck_1819_;
goto v_resetjp_1813_;
}
v_resetjp_1813_:
{
lean_object* v___x_1817_; 
if (v_isShared_1815_ == 0)
{
v___x_1817_ = v___x_1814_;
goto v_reusejp_1816_;
}
else
{
lean_object* v_reuseFailAlloc_1818_; 
v_reuseFailAlloc_1818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1818_, 0, v_a_1812_);
v___x_1817_ = v_reuseFailAlloc_1818_;
goto v_reusejp_1816_;
}
v_reusejp_1816_:
{
return v___x_1817_;
}
}
}
}
else
{
lean_del_object(v___x_1625_);
lean_dec_ref(v_argNames_1623_);
lean_dec_ref(v_binders_1622_);
return v___x_1627_;
}
}
}
else
{
lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_1830_; 
v_a_1823_ = lean_ctor_get(v___x_1620_, 0);
v_isSharedCheck_1830_ = !lean_is_exclusive(v___x_1620_);
if (v_isSharedCheck_1830_ == 0)
{
v___x_1825_ = v___x_1620_;
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1620_);
v___x_1825_ = lean_box(0);
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
v_resetjp_1824_:
{
lean_object* v___x_1828_; 
if (v_isShared_1826_ == 0)
{
v___x_1828_ = v___x_1825_;
goto v_reusejp_1827_;
}
else
{
lean_object* v_reuseFailAlloc_1829_; 
v_reuseFailAlloc_1829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1829_, 0, v_a_1823_);
v___x_1828_ = v_reuseFailAlloc_1829_;
goto v_reusejp_1827_;
}
v_reusejp_1827_:
{
return v___x_1828_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___boxed(lean_object* v_ctx_1831_, lean_object* v_i_1832_, lean_object* v_a_1833_, lean_object* v_a_1834_, lean_object* v_a_1835_, lean_object* v_a_1836_, lean_object* v_a_1837_, lean_object* v_a_1838_, lean_object* v_a_1839_){
_start:
{
lean_object* v_res_1840_; 
v_res_1840_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction(v_ctx_1831_, v_i_1832_, v_a_1833_, v_a_1834_, v_a_1835_, v_a_1836_, v_a_1837_, v_a_1838_);
lean_dec(v_a_1838_);
lean_dec_ref(v_a_1837_);
lean_dec(v_a_1836_);
lean_dec_ref(v_a_1835_);
lean_dec(v_a_1834_);
lean_dec_ref(v_a_1833_);
lean_dec(v_i_1832_);
lean_dec_ref(v_ctx_1831_);
return v_res_1840_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4(lean_object* v_indVal_1841_, lean_object* v___x_1842_, lean_object* v_as_1843_, lean_object* v_as_x27_1844_, lean_object* v_b_1845_, lean_object* v_a_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
lean_object* v___x_1854_; 
v___x_1854_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___redArg(v_indVal_1841_, v___x_1842_, v_as_x27_1844_, v_b_1845_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_);
return v___x_1854_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4___boxed(lean_object* v_indVal_1855_, lean_object* v___x_1856_, lean_object* v_as_1857_, lean_object* v_as_x27_1858_, lean_object* v_b_1859_, lean_object* v_a_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_){
_start:
{
lean_object* v_res_1868_; 
v_res_1868_ = lp_plausible_List_forIn_x27_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction_spec__4(v_indVal_1855_, v___x_1856_, v_as_1857_, v_as_x27_1858_, v_b_1859_, v_a_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_, v___y_1865_, v___y_1866_);
lean_dec(v___y_1866_);
lean_dec_ref(v___y_1865_);
lean_dec(v___y_1864_);
lean_dec_ref(v___y_1863_);
lean_dec(v___y_1862_);
lean_dec_ref(v___y_1861_);
lean_dec(v_as_x27_1858_);
lean_dec(v_as_1857_);
return v_res_1868_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg(lean_object* v_upperBound_1869_, lean_object* v_ctx_1870_, lean_object* v_a_1871_, lean_object* v_b_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_){
_start:
{
uint8_t v___x_1880_; 
v___x_1880_ = lean_nat_dec_lt(v_a_1871_, v_upperBound_1869_);
if (v___x_1880_ == 0)
{
lean_object* v___x_1881_; 
lean_dec(v_a_1871_);
v___x_1881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1881_, 0, v_b_1872_);
return v___x_1881_;
}
else
{
lean_object* v___x_1882_; 
v___x_1882_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction(v_ctx_1870_, v_a_1871_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_);
if (lean_obj_tag(v___x_1882_) == 0)
{
lean_object* v_a_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; 
v_a_1883_ = lean_ctor_get(v___x_1882_, 0);
lean_inc(v_a_1883_);
lean_dec_ref_known(v___x_1882_, 1);
v___x_1884_ = lean_array_push(v_b_1872_, v_a_1883_);
v___x_1885_ = lean_unsigned_to_nat(1u);
v___x_1886_ = lean_nat_add(v_a_1871_, v___x_1885_);
lean_dec(v_a_1871_);
v_a_1871_ = v___x_1886_;
v_b_1872_ = v___x_1884_;
goto _start;
}
else
{
lean_object* v_a_1888_; lean_object* v___x_1890_; uint8_t v_isShared_1891_; uint8_t v_isSharedCheck_1895_; 
lean_dec_ref(v_b_1872_);
lean_dec(v_a_1871_);
v_a_1888_ = lean_ctor_get(v___x_1882_, 0);
v_isSharedCheck_1895_ = !lean_is_exclusive(v___x_1882_);
if (v_isSharedCheck_1895_ == 0)
{
v___x_1890_ = v___x_1882_;
v_isShared_1891_ = v_isSharedCheck_1895_;
goto v_resetjp_1889_;
}
else
{
lean_inc(v_a_1888_);
lean_dec(v___x_1882_);
v___x_1890_ = lean_box(0);
v_isShared_1891_ = v_isSharedCheck_1895_;
goto v_resetjp_1889_;
}
v_resetjp_1889_:
{
lean_object* v___x_1893_; 
if (v_isShared_1891_ == 0)
{
v___x_1893_ = v___x_1890_;
goto v_reusejp_1892_;
}
else
{
lean_object* v_reuseFailAlloc_1894_; 
v_reuseFailAlloc_1894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1894_, 0, v_a_1888_);
v___x_1893_ = v_reuseFailAlloc_1894_;
goto v_reusejp_1892_;
}
v_reusejp_1892_:
{
return v___x_1893_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg___boxed(lean_object* v_upperBound_1896_, lean_object* v_ctx_1897_, lean_object* v_a_1898_, lean_object* v_b_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg(v_upperBound_1896_, v_ctx_1897_, v_a_1898_, v_b_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1901_);
lean_dec_ref(v___y_1900_);
lean_dec_ref(v_ctx_1897_);
lean_dec(v_upperBound_1896_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock(lean_object* v_ctx_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_, lean_object* v_a_1921_){
_start:
{
lean_object* v_typeInfos_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v_auxDefs_1926_; lean_object* v___x_1927_; 
v_typeInfos_1923_ = lean_ctor_get(v_ctx_1915_, 1);
v___x_1924_ = lean_unsigned_to_nat(0u);
v___x_1925_ = lean_array_get_size(v_typeInfos_1923_);
v_auxDefs_1926_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2));
v___x_1927_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg(v___x_1925_, v_ctx_1915_, v___x_1924_, v_auxDefs_1926_, v_a_1916_, v_a_1917_, v_a_1918_, v_a_1919_, v_a_1920_, v_a_1921_);
if (lean_obj_tag(v___x_1927_) == 0)
{
lean_object* v_a_1928_; lean_object* v___x_1930_; uint8_t v_isShared_1931_; uint8_t v_isSharedCheck_1948_; 
v_a_1928_ = lean_ctor_get(v___x_1927_, 0);
v_isSharedCheck_1948_ = !lean_is_exclusive(v___x_1927_);
if (v_isSharedCheck_1948_ == 0)
{
v___x_1930_ = v___x_1927_;
v_isShared_1931_ = v_isSharedCheck_1948_;
goto v_resetjp_1929_;
}
else
{
lean_inc(v_a_1928_);
lean_dec(v___x_1927_);
v___x_1930_ = lean_box(0);
v_isShared_1931_ = v_isSharedCheck_1948_;
goto v_resetjp_1929_;
}
v_resetjp_1929_:
{
lean_object* v_ref_1932_; uint8_t v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1946_; 
v_ref_1932_ = lean_ctor_get(v_a_1920_, 5);
v___x_1933_ = 0;
v___x_1934_ = l_Lean_SourceInfo_fromRef(v_ref_1932_, v___x_1933_);
v___x_1935_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__0));
v___x_1936_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__1));
lean_inc_n(v___x_1934_, 3);
v___x_1937_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1937_, 0, v___x_1934_);
lean_ctor_set(v___x_1937_, 1, v___x_1935_);
v___x_1938_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_1939_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
v___x_1940_ = l_Array_append___redArg(v___x_1939_, v_a_1928_);
lean_dec(v_a_1928_);
v___x_1941_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1941_, 0, v___x_1934_);
lean_ctor_set(v___x_1941_, 1, v___x_1938_);
lean_ctor_set(v___x_1941_, 2, v___x_1940_);
v___x_1942_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___closed__2));
v___x_1943_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1943_, 0, v___x_1934_);
lean_ctor_set(v___x_1943_, 1, v___x_1942_);
v___x_1944_ = l_Lean_Syntax_node3(v___x_1934_, v___x_1936_, v___x_1937_, v___x_1941_, v___x_1943_);
if (v_isShared_1931_ == 0)
{
lean_ctor_set(v___x_1930_, 0, v___x_1944_);
v___x_1946_ = v___x_1930_;
goto v_reusejp_1945_;
}
else
{
lean_object* v_reuseFailAlloc_1947_; 
v_reuseFailAlloc_1947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1947_, 0, v___x_1944_);
v___x_1946_ = v_reuseFailAlloc_1947_;
goto v_reusejp_1945_;
}
v_reusejp_1945_:
{
return v___x_1946_;
}
}
}
else
{
lean_object* v_a_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1956_; 
v_a_1949_ = lean_ctor_get(v___x_1927_, 0);
v_isSharedCheck_1956_ = !lean_is_exclusive(v___x_1927_);
if (v_isSharedCheck_1956_ == 0)
{
v___x_1951_ = v___x_1927_;
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_a_1949_);
lean_dec(v___x_1927_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v___x_1954_; 
if (v_isShared_1952_ == 0)
{
v___x_1954_ = v___x_1951_;
goto v_reusejp_1953_;
}
else
{
lean_object* v_reuseFailAlloc_1955_; 
v_reuseFailAlloc_1955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1955_, 0, v_a_1949_);
v___x_1954_ = v_reuseFailAlloc_1955_;
goto v_reusejp_1953_;
}
v_reusejp_1953_:
{
return v___x_1954_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock___boxed(lean_object* v_ctx_1957_, lean_object* v_a_1958_, lean_object* v_a_1959_, lean_object* v_a_1960_, lean_object* v_a_1961_, lean_object* v_a_1962_, lean_object* v_a_1963_, lean_object* v_a_1964_){
_start:
{
lean_object* v_res_1965_; 
v_res_1965_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock(v_ctx_1957_, v_a_1958_, v_a_1959_, v_a_1960_, v_a_1961_, v_a_1962_, v_a_1963_);
lean_dec(v_a_1963_);
lean_dec_ref(v_a_1962_);
lean_dec(v_a_1961_);
lean_dec_ref(v_a_1960_);
lean_dec(v_a_1959_);
lean_dec_ref(v_a_1958_);
lean_dec_ref(v_ctx_1957_);
return v_res_1965_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0(lean_object* v_upperBound_1966_, lean_object* v_ctx_1967_, lean_object* v_inst_1968_, lean_object* v_R_1969_, lean_object* v_a_1970_, lean_object* v_b_1971_, lean_object* v_c_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v___x_1980_; 
v___x_1980_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___redArg(v_upperBound_1966_, v_ctx_1967_, v_a_1970_, v_b_1971_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_);
return v___x_1980_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0___boxed(lean_object* v_upperBound_1981_, lean_object* v_ctx_1982_, lean_object* v_inst_1983_, lean_object* v_R_1984_, lean_object* v_a_1985_, lean_object* v_b_1986_, lean_object* v_c_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_){
_start:
{
lean_object* v_res_1995_; 
v_res_1995_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock_spec__0(v_upperBound_1981_, v_ctx_1982_, v_inst_1983_, v_R_1984_, v_a_1985_, v_b_1986_, v_c_1987_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_, v___y_1992_, v___y_1993_);
lean_dec(v___y_1993_);
lean_dec_ref(v___y_1992_);
lean_dec(v___y_1991_);
lean_dec_ref(v___y_1990_);
lean_dec(v___y_1989_);
lean_dec_ref(v___y_1988_);
lean_dec_ref(v_ctx_1982_);
lean_dec(v_upperBound_1981_);
return v_res_1995_;
}
}
LEAN_EXPORT uint8_t lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0(lean_object* v_a_1996_, lean_object* v_as_1997_, size_t v_i_1998_, size_t v_stop_1999_){
_start:
{
uint8_t v___x_2000_; 
v___x_2000_ = lean_usize_dec_eq(v_i_1998_, v_stop_1999_);
if (v___x_2000_ == 0)
{
lean_object* v___x_2001_; uint8_t v___x_2002_; 
v___x_2001_ = lean_array_uget_borrowed(v_as_1997_, v_i_1998_);
v___x_2002_ = lean_name_eq(v_a_1996_, v___x_2001_);
if (v___x_2002_ == 0)
{
size_t v___x_2003_; size_t v___x_2004_; 
v___x_2003_ = ((size_t)1ULL);
v___x_2004_ = lean_usize_add(v_i_1998_, v___x_2003_);
v_i_1998_ = v___x_2004_;
goto _start;
}
else
{
return v___x_2002_;
}
}
else
{
uint8_t v___x_2006_; 
v___x_2006_ = 0;
return v___x_2006_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0___boxed(lean_object* v_a_2007_, lean_object* v_as_2008_, lean_object* v_i_2009_, lean_object* v_stop_2010_){
_start:
{
size_t v_i_boxed_2011_; size_t v_stop_boxed_2012_; uint8_t v_res_2013_; lean_object* v_r_2014_; 
v_i_boxed_2011_ = lean_unbox_usize(v_i_2009_);
lean_dec(v_i_2009_);
v_stop_boxed_2012_ = lean_unbox_usize(v_stop_2010_);
lean_dec(v_stop_2010_);
v_res_2013_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0(v_a_2007_, v_as_2008_, v_i_boxed_2011_, v_stop_boxed_2012_);
lean_dec_ref(v_as_2008_);
lean_dec(v_a_2007_);
v_r_2014_ = lean_box(v_res_2013_);
return v_r_2014_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0(lean_object* v_as_2015_, lean_object* v_a_2016_){
_start:
{
lean_object* v___x_2017_; lean_object* v___x_2018_; uint8_t v___x_2019_; 
v___x_2017_ = lean_unsigned_to_nat(0u);
v___x_2018_ = lean_array_get_size(v_as_2015_);
v___x_2019_ = lean_nat_dec_lt(v___x_2017_, v___x_2018_);
if (v___x_2019_ == 0)
{
return v___x_2019_;
}
else
{
if (v___x_2019_ == 0)
{
return v___x_2019_;
}
else
{
size_t v___x_2020_; size_t v___x_2021_; uint8_t v___x_2022_; 
v___x_2020_ = ((size_t)0ULL);
v___x_2021_ = lean_usize_of_nat(v___x_2018_);
v___x_2022_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0_spec__0(v_a_2016_, v_as_2015_, v___x_2020_, v___x_2021_);
return v___x_2022_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0___boxed(lean_object* v_as_2023_, lean_object* v_a_2024_){
_start:
{
uint8_t v_res_2025_; lean_object* v_r_2026_; 
v_res_2025_ = lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0(v_as_2023_, v_a_2024_);
lean_dec(v_a_2024_);
lean_dec_ref(v_as_2023_);
v_r_2026_ = lean_box(v_res_2025_);
return v_r_2026_;
}
}
static lean_object* _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_2027_; lean_object* v___x_2028_; 
v___x_2027_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__0));
v___x_2028_ = l_String_toRawSubstring_x27(v___x_2027_);
return v___x_2028_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg(lean_object* v_upperBound_2066_, lean_object* v___x_2067_, lean_object* v_typeNames_2068_, lean_object* v_ctx_2069_, lean_object* v_a_2070_, lean_object* v_b_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_, lean_object* v___y_2077_){
_start:
{
lean_object* v_a_2080_; uint8_t v___x_2084_; 
v___x_2084_ = lean_nat_dec_lt(v_a_2070_, v_upperBound_2066_);
if (v___x_2084_ == 0)
{
lean_object* v___x_2085_; 
lean_dec(v_a_2070_);
v___x_2085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2085_, 0, v_b_2071_);
return v___x_2085_;
}
else
{
lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v_toConstantVal_2088_; lean_object* v_name_2089_; lean_object* v___x_2091_; uint8_t v_isShared_2092_; uint8_t v_isSharedCheck_2182_; 
v___x_2086_ = l_Lean_instInhabitedInductiveVal_default;
v___x_2087_ = lean_array_get_borrowed(v___x_2086_, v___x_2067_, v_a_2070_);
v_toConstantVal_2088_ = lean_ctor_get(v___x_2087_, 0);
lean_inc_ref(v_toConstantVal_2088_);
v_name_2089_ = lean_ctor_get(v_toConstantVal_2088_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v_toConstantVal_2088_);
if (v_isSharedCheck_2182_ == 0)
{
lean_object* v_unused_2183_; lean_object* v_unused_2184_; 
v_unused_2183_ = lean_ctor_get(v_toConstantVal_2088_, 2);
lean_dec(v_unused_2183_);
v_unused_2184_ = lean_ctor_get(v_toConstantVal_2088_, 1);
lean_dec(v_unused_2184_);
v___x_2091_ = v_toConstantVal_2088_;
v_isShared_2092_ = v_isSharedCheck_2182_;
goto v_resetjp_2090_;
}
else
{
lean_inc(v_name_2089_);
lean_dec(v_toConstantVal_2088_);
v___x_2091_ = lean_box(0);
v_isShared_2092_ = v_isSharedCheck_2182_;
goto v_resetjp_2090_;
}
v_resetjp_2090_:
{
uint8_t v___x_2093_; 
v___x_2093_ = lp_plausible_Array_contains___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__0(v_typeNames_2068_, v_name_2089_);
lean_dec(v_name_2089_);
if (v___x_2093_ == 0)
{
lean_del_object(v___x_2091_);
v_a_2080_ = v_b_2071_;
goto v___jp_2079_;
}
else
{
lean_object* v___x_2094_; 
lean_inc(v___x_2087_);
v___x_2094_ = l_Lean_Elab_Deriving_mkInductArgNames(v___x_2087_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_, v___y_2077_);
if (lean_obj_tag(v___x_2094_) == 0)
{
lean_object* v_a_2095_; lean_object* v___x_2096_; 
v_a_2095_ = lean_ctor_get(v___x_2094_, 0);
lean_inc_n(v_a_2095_, 2);
lean_dec_ref_known(v___x_2094_, 1);
v___x_2096_ = l_Lean_Elab_Deriving_mkImplicitBinders(v_a_2095_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_, v___y_2077_);
if (lean_obj_tag(v___x_2096_) == 0)
{
lean_object* v_a_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; 
v_a_2097_ = lean_ctor_get(v___x_2096_, 0);
lean_inc(v_a_2097_);
lean_dec_ref_known(v___x_2096_, 1);
v___x_2098_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1));
lean_inc(v_a_2095_);
lean_inc(v___x_2087_);
v___x_2099_ = l_Lean_Elab_Deriving_mkInstImplicitBinders(v___x_2098_, v___x_2087_, v_a_2095_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_, v___y_2077_);
if (lean_obj_tag(v___x_2099_) == 0)
{
lean_object* v_a_2100_; lean_object* v___x_2101_; 
v_a_2100_ = lean_ctor_get(v___x_2099_, 0);
lean_inc(v_a_2100_);
lean_dec_ref_known(v___x_2099_, 1);
lean_inc(v___x_2087_);
v___x_2101_ = l_Lean_Elab_Deriving_mkInductiveApp___redArg(v___x_2087_, v_a_2095_, v___y_2076_);
if (lean_obj_tag(v___x_2101_) == 0)
{
lean_object* v_a_2102_; lean_object* v_auxFunNames_2103_; lean_object* v_ref_2104_; lean_object* v_quotContext_2105_; lean_object* v_currMacroScope_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; uint8_t v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2125_; 
v_a_2102_ = lean_ctor_get(v___x_2101_, 0);
lean_inc(v_a_2102_);
lean_dec_ref_known(v___x_2101_, 1);
v_auxFunNames_2103_ = lean_ctor_get(v_ctx_2069_, 2);
v_ref_2104_ = lean_ctor_get(v___y_2076_, 5);
v_quotContext_2105_ = lean_ctor_get(v___y_2076_, 10);
v_currMacroScope_2106_ = lean_ctor_get(v___y_2076_, 11);
v___x_2107_ = lean_box(0);
v___x_2108_ = lean_array_get_borrowed(v___x_2107_, v_auxFunNames_2103_, v_a_2070_);
v___x_2109_ = l_Array_append___redArg(v_a_2097_, v_a_2100_);
lean_dec(v_a_2100_);
v___x_2110_ = 0;
v___x_2111_ = l_Lean_SourceInfo_fromRef(v_ref_2104_, v___x_2110_);
v___x_2112_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__14));
v___x_2113_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__0);
v___x_2114_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__1));
lean_inc(v_currMacroScope_2106_);
lean_inc(v_quotContext_2105_);
v___x_2115_ = l_Lean_addMacroScope(v_quotContext_2105_, v___x_2114_, v_currMacroScope_2106_);
v___x_2116_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__4));
lean_inc_n(v___x_2111_, 4);
v___x_2117_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2117_, 0, v___x_2111_);
lean_ctor_set(v___x_2117_, 1, v___x_2113_);
lean_ctor_set(v___x_2117_, 2, v___x_2115_);
lean_ctor_set(v___x_2117_, 3, v___x_2116_);
v___x_2118_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__3___redArg___closed__4));
v___x_2119_ = l_Lean_Syntax_node1(v___x_2111_, v___x_2118_, v_a_2102_);
v___x_2120_ = l_Lean_Syntax_node2(v___x_2111_, v___x_2112_, v___x_2117_, v___x_2119_);
v___x_2121_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__24));
v___x_2122_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__26));
v___x_2123_ = lean_obj_once(&lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11, &lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11_once, _init_lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__11);
if (v_isShared_2092_ == 0)
{
lean_ctor_set_tag(v___x_2091_, 1);
lean_ctor_set(v___x_2091_, 2, v___x_2123_);
lean_ctor_set(v___x_2091_, 1, v___x_2118_);
lean_ctor_set(v___x_2091_, 0, v___x_2111_);
v___x_2125_ = v___x_2091_;
goto v_reusejp_2124_;
}
else
{
lean_object* v_reuseFailAlloc_2157_; 
v_reuseFailAlloc_2157_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2157_, 0, v___x_2111_);
lean_ctor_set(v_reuseFailAlloc_2157_, 1, v___x_2118_);
lean_ctor_set(v_reuseFailAlloc_2157_, 2, v___x_2123_);
v___x_2125_ = v_reuseFailAlloc_2157_;
goto v_reusejp_2124_;
}
v_reusejp_2124_:
{
lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; 
lean_inc_ref_n(v___x_2125_, 12);
lean_inc_n(v___x_2111_, 15);
v___x_2126_ = l_Lean_Syntax_node7(v___x_2111_, v___x_2122_, v___x_2125_, v___x_2125_, v___x_2125_, v___x_2125_, v___x_2125_, v___x_2125_, v___x_2125_);
v___x_2127_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__5));
v___x_2128_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__6));
v___x_2129_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__8));
v___x_2130_ = l_Lean_Syntax_node1(v___x_2111_, v___x_2129_, v___x_2125_);
v___x_2131_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2131_, 0, v___x_2111_);
lean_ctor_set(v___x_2131_, 1, v___x_2127_);
v___x_2132_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__10));
v___x_2133_ = l_Array_append___redArg(v___x_2123_, v___x_2109_);
lean_dec_ref(v___x_2109_);
v___x_2134_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2134_, 0, v___x_2111_);
lean_ctor_set(v___x_2134_, 1, v___x_2118_);
lean_ctor_set(v___x_2134_, 2, v___x_2133_);
v___x_2135_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__35));
v___x_2136_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__36));
v___x_2137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2111_);
lean_ctor_set(v___x_2137_, 1, v___x_2136_);
v___x_2138_ = l_Lean_Syntax_node2(v___x_2111_, v___x_2135_, v___x_2137_, v___x_2120_);
v___x_2139_ = l_Lean_Syntax_node2(v___x_2111_, v___x_2132_, v___x_2134_, v___x_2138_);
v___x_2140_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__38));
v___x_2141_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__39));
v___x_2142_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2142_, 0, v___x_2111_);
lean_ctor_set(v___x_2142_, 1, v___x_2141_);
v___x_2143_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__12));
v___x_2144_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__13));
v___x_2145_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2145_, 0, v___x_2111_);
lean_ctor_set(v___x_2145_, 1, v___x_2144_);
lean_inc(v___x_2108_);
v___x_2146_ = l_Lean_mkIdent(v___x_2108_);
v___x_2147_ = l_Lean_Syntax_node1(v___x_2111_, v___x_2118_, v___x_2146_);
v___x_2148_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___closed__14));
v___x_2149_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2149_, 0, v___x_2111_);
lean_ctor_set(v___x_2149_, 1, v___x_2148_);
v___x_2150_ = l_Lean_Syntax_node3(v___x_2111_, v___x_2143_, v___x_2145_, v___x_2147_, v___x_2149_);
v___x_2151_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkAuxFunction___closed__42));
v___x_2152_ = l_Lean_Syntax_node2(v___x_2111_, v___x_2151_, v___x_2125_, v___x_2125_);
v___x_2153_ = l_Lean_Syntax_node4(v___x_2111_, v___x_2140_, v___x_2142_, v___x_2150_, v___x_2152_, v___x_2125_);
v___x_2154_ = l_Lean_Syntax_node6(v___x_2111_, v___x_2128_, v___x_2130_, v___x_2131_, v___x_2125_, v___x_2125_, v___x_2139_, v___x_2153_);
v___x_2155_ = l_Lean_Syntax_node2(v___x_2111_, v___x_2121_, v___x_2126_, v___x_2154_);
v___x_2156_ = lean_array_push(v_b_2071_, v___x_2155_);
v_a_2080_ = v___x_2156_;
goto v___jp_2079_;
}
}
else
{
lean_object* v_a_2158_; lean_object* v___x_2160_; uint8_t v_isShared_2161_; uint8_t v_isSharedCheck_2165_; 
lean_dec(v_a_2100_);
lean_dec(v_a_2097_);
lean_del_object(v___x_2091_);
lean_dec_ref(v_b_2071_);
lean_dec(v_a_2070_);
v_a_2158_ = lean_ctor_get(v___x_2101_, 0);
v_isSharedCheck_2165_ = !lean_is_exclusive(v___x_2101_);
if (v_isSharedCheck_2165_ == 0)
{
v___x_2160_ = v___x_2101_;
v_isShared_2161_ = v_isSharedCheck_2165_;
goto v_resetjp_2159_;
}
else
{
lean_inc(v_a_2158_);
lean_dec(v___x_2101_);
v___x_2160_ = lean_box(0);
v_isShared_2161_ = v_isSharedCheck_2165_;
goto v_resetjp_2159_;
}
v_resetjp_2159_:
{
lean_object* v___x_2163_; 
if (v_isShared_2161_ == 0)
{
v___x_2163_ = v___x_2160_;
goto v_reusejp_2162_;
}
else
{
lean_object* v_reuseFailAlloc_2164_; 
v_reuseFailAlloc_2164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2164_, 0, v_a_2158_);
v___x_2163_ = v_reuseFailAlloc_2164_;
goto v_reusejp_2162_;
}
v_reusejp_2162_:
{
return v___x_2163_;
}
}
}
}
else
{
lean_object* v_a_2166_; lean_object* v___x_2168_; uint8_t v_isShared_2169_; uint8_t v_isSharedCheck_2173_; 
lean_dec(v_a_2097_);
lean_dec(v_a_2095_);
lean_del_object(v___x_2091_);
lean_dec_ref(v_b_2071_);
lean_dec(v_a_2070_);
v_a_2166_ = lean_ctor_get(v___x_2099_, 0);
v_isSharedCheck_2173_ = !lean_is_exclusive(v___x_2099_);
if (v_isSharedCheck_2173_ == 0)
{
v___x_2168_ = v___x_2099_;
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
else
{
lean_inc(v_a_2166_);
lean_dec(v___x_2099_);
v___x_2168_ = lean_box(0);
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
v_resetjp_2167_:
{
lean_object* v___x_2171_; 
if (v_isShared_2169_ == 0)
{
v___x_2171_ = v___x_2168_;
goto v_reusejp_2170_;
}
else
{
lean_object* v_reuseFailAlloc_2172_; 
v_reuseFailAlloc_2172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2172_, 0, v_a_2166_);
v___x_2171_ = v_reuseFailAlloc_2172_;
goto v_reusejp_2170_;
}
v_reusejp_2170_:
{
return v___x_2171_;
}
}
}
}
else
{
lean_dec(v_a_2095_);
lean_del_object(v___x_2091_);
lean_dec_ref(v_b_2071_);
lean_dec(v_a_2070_);
return v___x_2096_;
}
}
else
{
lean_object* v_a_2174_; lean_object* v___x_2176_; uint8_t v_isShared_2177_; uint8_t v_isSharedCheck_2181_; 
lean_del_object(v___x_2091_);
lean_dec_ref(v_b_2071_);
lean_dec(v_a_2070_);
v_a_2174_ = lean_ctor_get(v___x_2094_, 0);
v_isSharedCheck_2181_ = !lean_is_exclusive(v___x_2094_);
if (v_isSharedCheck_2181_ == 0)
{
v___x_2176_ = v___x_2094_;
v_isShared_2177_ = v_isSharedCheck_2181_;
goto v_resetjp_2175_;
}
else
{
lean_inc(v_a_2174_);
lean_dec(v___x_2094_);
v___x_2176_ = lean_box(0);
v_isShared_2177_ = v_isSharedCheck_2181_;
goto v_resetjp_2175_;
}
v_resetjp_2175_:
{
lean_object* v___x_2179_; 
if (v_isShared_2177_ == 0)
{
v___x_2179_ = v___x_2176_;
goto v_reusejp_2178_;
}
else
{
lean_object* v_reuseFailAlloc_2180_; 
v_reuseFailAlloc_2180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2180_, 0, v_a_2174_);
v___x_2179_ = v_reuseFailAlloc_2180_;
goto v_reusejp_2178_;
}
v_reusejp_2178_:
{
return v___x_2179_;
}
}
}
}
}
}
v___jp_2079_:
{
lean_object* v___x_2081_; lean_object* v___x_2082_; 
v___x_2081_ = lean_unsigned_to_nat(1u);
v___x_2082_ = lean_nat_add(v_a_2070_, v___x_2081_);
lean_dec(v_a_2070_);
v_a_2070_ = v___x_2082_;
v_b_2071_ = v_a_2080_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg___boxed(lean_object* v_upperBound_2185_, lean_object* v___x_2186_, lean_object* v_typeNames_2187_, lean_object* v_ctx_2188_, lean_object* v_a_2189_, lean_object* v_b_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_){
_start:
{
lean_object* v_res_2198_; 
v_res_2198_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg(v_upperBound_2185_, v___x_2186_, v_typeNames_2187_, v_ctx_2188_, v_a_2189_, v_b_2190_, v___y_2191_, v___y_2192_, v___y_2193_, v___y_2194_, v___y_2195_, v___y_2196_);
lean_dec(v___y_2196_);
lean_dec_ref(v___y_2195_);
lean_dec(v___y_2194_);
lean_dec_ref(v___y_2193_);
lean_dec(v___y_2192_);
lean_dec_ref(v___y_2191_);
lean_dec_ref(v_ctx_2188_);
lean_dec_ref(v_typeNames_2187_);
lean_dec_ref(v___x_2186_);
lean_dec(v_upperBound_2185_);
return v_res_2198_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds(lean_object* v_ctx_2199_, lean_object* v_typeNames_2200_, lean_object* v_a_2201_, lean_object* v_a_2202_, lean_object* v_a_2203_, lean_object* v_a_2204_, lean_object* v_a_2205_, lean_object* v_a_2206_){
_start:
{
lean_object* v_typeInfos_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v_instances_2211_; lean_object* v___x_2212_; 
v_typeInfos_2208_ = lean_ctor_get(v_ctx_2199_, 1);
v___x_2209_ = lean_unsigned_to_nat(0u);
v___x_2210_ = lean_array_get_size(v_typeInfos_2208_);
v_instances_2211_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__2));
v___x_2212_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg(v___x_2210_, v_typeInfos_2208_, v_typeNames_2200_, v_ctx_2199_, v___x_2209_, v_instances_2211_, v_a_2201_, v_a_2202_, v_a_2203_, v_a_2204_, v_a_2205_, v_a_2206_);
return v___x_2212_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds___boxed(lean_object* v_ctx_2213_, lean_object* v_typeNames_2214_, lean_object* v_a_2215_, lean_object* v_a_2216_, lean_object* v_a_2217_, lean_object* v_a_2218_, lean_object* v_a_2219_, lean_object* v_a_2220_, lean_object* v_a_2221_){
_start:
{
lean_object* v_res_2222_; 
v_res_2222_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds(v_ctx_2213_, v_typeNames_2214_, v_a_2215_, v_a_2216_, v_a_2217_, v_a_2218_, v_a_2219_, v_a_2220_);
lean_dec(v_a_2220_);
lean_dec_ref(v_a_2219_);
lean_dec(v_a_2218_);
lean_dec_ref(v_a_2217_);
lean_dec(v_a_2216_);
lean_dec_ref(v_a_2215_);
lean_dec_ref(v_typeNames_2214_);
lean_dec_ref(v_ctx_2213_);
return v_res_2222_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1(lean_object* v_upperBound_2223_, lean_object* v___x_2224_, lean_object* v_typeNames_2225_, lean_object* v_ctx_2226_, lean_object* v_inst_2227_, lean_object* v_R_2228_, lean_object* v_a_2229_, lean_object* v_b_2230_, lean_object* v_c_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_){
_start:
{
lean_object* v___x_2239_; 
v___x_2239_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___redArg(v_upperBound_2223_, v___x_2224_, v_typeNames_2225_, v_ctx_2226_, v_a_2229_, v_b_2230_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_, v___y_2236_, v___y_2237_);
return v___x_2239_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1___boxed(lean_object* v_upperBound_2240_, lean_object* v___x_2241_, lean_object* v_typeNames_2242_, lean_object* v_ctx_2243_, lean_object* v_inst_2244_, lean_object* v_R_2245_, lean_object* v_a_2246_, lean_object* v_b_2247_, lean_object* v_c_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_){
_start:
{
lean_object* v_res_2256_; 
v_res_2256_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds_spec__1(v_upperBound_2240_, v___x_2241_, v_typeNames_2242_, v_ctx_2243_, v_inst_2244_, v_R_2245_, v_a_2246_, v_b_2247_, v_c_2248_, v___y_2249_, v___y_2250_, v___y_2251_, v___y_2252_, v___y_2253_, v___y_2254_);
lean_dec(v___y_2254_);
lean_dec_ref(v___y_2253_);
lean_dec(v___y_2252_);
lean_dec_ref(v___y_2251_);
lean_dec(v___y_2250_);
lean_dec_ref(v___y_2249_);
lean_dec_ref(v_ctx_2243_);
lean_dec_ref(v_typeNames_2242_);
lean_dec_ref(v___x_2241_);
lean_dec(v_upperBound_2240_);
return v_res_2256_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__0(lean_object* v_a_2257_, lean_object* v_a_2258_){
_start:
{
if (lean_obj_tag(v_a_2257_) == 0)
{
lean_object* v___x_2259_; 
v___x_2259_ = l_List_reverse___redArg(v_a_2258_);
return v___x_2259_;
}
else
{
lean_object* v_head_2260_; lean_object* v_tail_2261_; lean_object* v___x_2263_; uint8_t v_isShared_2264_; uint8_t v_isSharedCheck_2270_; 
v_head_2260_ = lean_ctor_get(v_a_2257_, 0);
v_tail_2261_ = lean_ctor_get(v_a_2257_, 1);
v_isSharedCheck_2270_ = !lean_is_exclusive(v_a_2257_);
if (v_isSharedCheck_2270_ == 0)
{
v___x_2263_ = v_a_2257_;
v_isShared_2264_ = v_isSharedCheck_2270_;
goto v_resetjp_2262_;
}
else
{
lean_inc(v_tail_2261_);
lean_inc(v_head_2260_);
lean_dec(v_a_2257_);
v___x_2263_ = lean_box(0);
v_isShared_2264_ = v_isSharedCheck_2270_;
goto v_resetjp_2262_;
}
v_resetjp_2262_:
{
lean_object* v___x_2265_; lean_object* v___x_2267_; 
v___x_2265_ = l_Lean_MessageData_ofSyntax(v_head_2260_);
if (v_isShared_2264_ == 0)
{
lean_ctor_set(v___x_2263_, 1, v_a_2258_);
lean_ctor_set(v___x_2263_, 0, v___x_2265_);
v___x_2267_ = v___x_2263_;
goto v_reusejp_2266_;
}
else
{
lean_object* v_reuseFailAlloc_2269_; 
v_reuseFailAlloc_2269_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2269_, 0, v___x_2265_);
lean_ctor_set(v_reuseFailAlloc_2269_, 1, v_a_2258_);
v___x_2267_ = v_reuseFailAlloc_2269_;
goto v_reusejp_2266_;
}
v_reusejp_2266_:
{
v_a_2257_ = v_tail_2261_;
v_a_2258_ = v___x_2267_;
goto _start;
}
}
}
}
}
static double _init_lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_2271_; double v___x_2272_; 
v___x_2271_ = lean_unsigned_to_nat(0u);
v___x_2272_ = lean_float_of_nat(v___x_2271_);
return v___x_2272_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg(lean_object* v_cls_2275_, lean_object* v_msg_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_){
_start:
{
lean_object* v_ref_2282_; lean_object* v___x_2283_; lean_object* v_a_2284_; lean_object* v___x_2286_; uint8_t v_isShared_2287_; uint8_t v_isSharedCheck_2328_; 
v_ref_2282_ = lean_ctor_get(v___y_2279_, 5);
v___x_2283_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_2276_, v___y_2277_, v___y_2278_, v___y_2279_, v___y_2280_);
v_a_2284_ = lean_ctor_get(v___x_2283_, 0);
v_isSharedCheck_2328_ = !lean_is_exclusive(v___x_2283_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2286_ = v___x_2283_;
v_isShared_2287_ = v_isSharedCheck_2328_;
goto v_resetjp_2285_;
}
else
{
lean_inc(v_a_2284_);
lean_dec(v___x_2283_);
v___x_2286_ = lean_box(0);
v_isShared_2287_ = v_isSharedCheck_2328_;
goto v_resetjp_2285_;
}
v_resetjp_2285_:
{
lean_object* v___x_2288_; lean_object* v_traceState_2289_; lean_object* v_env_2290_; lean_object* v_nextMacroScope_2291_; lean_object* v_ngen_2292_; lean_object* v_auxDeclNGen_2293_; lean_object* v_cache_2294_; lean_object* v_messages_2295_; lean_object* v_infoState_2296_; lean_object* v_snapshotTasks_2297_; lean_object* v___x_2299_; uint8_t v_isShared_2300_; uint8_t v_isSharedCheck_2327_; 
v___x_2288_ = lean_st_ref_take(v___y_2280_);
v_traceState_2289_ = lean_ctor_get(v___x_2288_, 4);
v_env_2290_ = lean_ctor_get(v___x_2288_, 0);
v_nextMacroScope_2291_ = lean_ctor_get(v___x_2288_, 1);
v_ngen_2292_ = lean_ctor_get(v___x_2288_, 2);
v_auxDeclNGen_2293_ = lean_ctor_get(v___x_2288_, 3);
v_cache_2294_ = lean_ctor_get(v___x_2288_, 5);
v_messages_2295_ = lean_ctor_get(v___x_2288_, 6);
v_infoState_2296_ = lean_ctor_get(v___x_2288_, 7);
v_snapshotTasks_2297_ = lean_ctor_get(v___x_2288_, 8);
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2288_);
if (v_isSharedCheck_2327_ == 0)
{
v___x_2299_ = v___x_2288_;
v_isShared_2300_ = v_isSharedCheck_2327_;
goto v_resetjp_2298_;
}
else
{
lean_inc(v_snapshotTasks_2297_);
lean_inc(v_infoState_2296_);
lean_inc(v_messages_2295_);
lean_inc(v_cache_2294_);
lean_inc(v_traceState_2289_);
lean_inc(v_auxDeclNGen_2293_);
lean_inc(v_ngen_2292_);
lean_inc(v_nextMacroScope_2291_);
lean_inc(v_env_2290_);
lean_dec(v___x_2288_);
v___x_2299_ = lean_box(0);
v_isShared_2300_ = v_isSharedCheck_2327_;
goto v_resetjp_2298_;
}
v_resetjp_2298_:
{
uint64_t v_tid_2301_; lean_object* v_traces_2302_; lean_object* v___x_2304_; uint8_t v_isShared_2305_; uint8_t v_isSharedCheck_2326_; 
v_tid_2301_ = lean_ctor_get_uint64(v_traceState_2289_, sizeof(void*)*1);
v_traces_2302_ = lean_ctor_get(v_traceState_2289_, 0);
v_isSharedCheck_2326_ = !lean_is_exclusive(v_traceState_2289_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2304_ = v_traceState_2289_;
v_isShared_2305_ = v_isSharedCheck_2326_;
goto v_resetjp_2303_;
}
else
{
lean_inc(v_traces_2302_);
lean_dec(v_traceState_2289_);
v___x_2304_ = lean_box(0);
v_isShared_2305_ = v_isSharedCheck_2326_;
goto v_resetjp_2303_;
}
v_resetjp_2303_:
{
lean_object* v___x_2306_; double v___x_2307_; uint8_t v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2316_; 
v___x_2306_ = lean_box(0);
v___x_2307_ = lean_float_once(&lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0, &lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0_once, _init_lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__0);
v___x_2308_ = 0;
v___x_2309_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___lam__1___closed__6));
v___x_2310_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2310_, 0, v_cls_2275_);
lean_ctor_set(v___x_2310_, 1, v___x_2306_);
lean_ctor_set(v___x_2310_, 2, v___x_2309_);
lean_ctor_set_float(v___x_2310_, sizeof(void*)*3, v___x_2307_);
lean_ctor_set_float(v___x_2310_, sizeof(void*)*3 + 8, v___x_2307_);
lean_ctor_set_uint8(v___x_2310_, sizeof(void*)*3 + 16, v___x_2308_);
v___x_2311_ = ((lean_object*)(lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___closed__1));
v___x_2312_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2312_, 0, v___x_2310_);
lean_ctor_set(v___x_2312_, 1, v_a_2284_);
lean_ctor_set(v___x_2312_, 2, v___x_2311_);
lean_inc(v_ref_2282_);
v___x_2313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2313_, 0, v_ref_2282_);
lean_ctor_set(v___x_2313_, 1, v___x_2312_);
v___x_2314_ = l_Lean_PersistentArray_push___redArg(v_traces_2302_, v___x_2313_);
if (v_isShared_2305_ == 0)
{
lean_ctor_set(v___x_2304_, 0, v___x_2314_);
v___x_2316_ = v___x_2304_;
goto v_reusejp_2315_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v___x_2314_);
lean_ctor_set_uint64(v_reuseFailAlloc_2325_, sizeof(void*)*1, v_tid_2301_);
v___x_2316_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2315_;
}
v_reusejp_2315_:
{
lean_object* v___x_2318_; 
if (v_isShared_2300_ == 0)
{
lean_ctor_set(v___x_2299_, 4, v___x_2316_);
v___x_2318_ = v___x_2299_;
goto v_reusejp_2317_;
}
else
{
lean_object* v_reuseFailAlloc_2324_; 
v_reuseFailAlloc_2324_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2324_, 0, v_env_2290_);
lean_ctor_set(v_reuseFailAlloc_2324_, 1, v_nextMacroScope_2291_);
lean_ctor_set(v_reuseFailAlloc_2324_, 2, v_ngen_2292_);
lean_ctor_set(v_reuseFailAlloc_2324_, 3, v_auxDeclNGen_2293_);
lean_ctor_set(v_reuseFailAlloc_2324_, 4, v___x_2316_);
lean_ctor_set(v_reuseFailAlloc_2324_, 5, v_cache_2294_);
lean_ctor_set(v_reuseFailAlloc_2324_, 6, v_messages_2295_);
lean_ctor_set(v_reuseFailAlloc_2324_, 7, v_infoState_2296_);
lean_ctor_set(v_reuseFailAlloc_2324_, 8, v_snapshotTasks_2297_);
v___x_2318_ = v_reuseFailAlloc_2324_;
goto v_reusejp_2317_;
}
v_reusejp_2317_:
{
lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2322_; 
v___x_2319_ = lean_st_ref_set(v___y_2280_, v___x_2318_);
v___x_2320_ = lean_box(0);
if (v_isShared_2287_ == 0)
{
lean_ctor_set(v___x_2286_, 0, v___x_2320_);
v___x_2322_ = v___x_2286_;
goto v_reusejp_2321_;
}
else
{
lean_object* v_reuseFailAlloc_2323_; 
v_reuseFailAlloc_2323_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2323_, 0, v___x_2320_);
v___x_2322_ = v_reuseFailAlloc_2323_;
goto v_reusejp_2321_;
}
v_reusejp_2321_:
{
return v___x_2322_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg___boxed(lean_object* v_cls_2329_, lean_object* v_msg_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_){
_start:
{
lean_object* v_res_2336_; 
v_res_2336_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg(v_cls_2329_, v_msg_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
lean_dec(v___y_2334_);
lean_dec_ref(v___y_2333_);
lean_dec(v___y_2332_);
lean_dec_ref(v___y_2331_);
return v_res_2336_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2(void){
_start:
{
lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_2341_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__1));
v___x_2342_ = l_Lean_Name_append(v___x_2341_, v___x_2340_);
return v___x_2342_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4(void){
_start:
{
lean_object* v___x_2344_; lean_object* v___x_2345_; 
v___x_2344_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__3));
v___x_2345_ = l_Lean_stringToMessageData(v___x_2344_);
return v___x_2345_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd(lean_object* v_declName_2346_, lean_object* v_a_2347_, lean_object* v_a_2348_, lean_object* v_a_2349_, lean_object* v_a_2350_, lean_object* v_a_2351_, lean_object* v_a_2352_){
_start:
{
lean_object* v___x_2354_; lean_object* v___x_2355_; uint8_t v___x_2356_; lean_object* v___x_2357_; 
v___x_2354_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1));
v___x_2355_ = ((lean_object*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkCtorShrinkExpr_spec__2___redArg___closed__17));
v___x_2356_ = 1;
lean_inc(v_declName_2346_);
v___x_2357_ = l_Lean_Elab_Deriving_mkContext(v___x_2354_, v___x_2355_, v_declName_2346_, v___x_2356_, v_a_2347_, v_a_2348_, v_a_2349_, v_a_2350_, v_a_2351_, v_a_2352_);
if (lean_obj_tag(v___x_2357_) == 0)
{
lean_object* v_a_2358_; lean_object* v___x_2359_; 
v_a_2358_ = lean_ctor_get(v___x_2357_, 0);
lean_inc(v_a_2358_);
lean_dec_ref_known(v___x_2357_, 1);
v___x_2359_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkMutualBlock(v_a_2358_, v_a_2347_, v_a_2348_, v_a_2349_, v_a_2350_, v_a_2351_, v_a_2352_);
if (lean_obj_tag(v___x_2359_) == 0)
{
lean_object* v_a_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; 
v_a_2360_ = lean_ctor_get(v___x_2359_, 0);
lean_inc(v_a_2360_);
lean_dec_ref_known(v___x_2359_, 1);
v___x_2361_ = lean_unsigned_to_nat(1u);
v___x_2362_ = lean_mk_empty_array_with_capacity(v___x_2361_);
lean_inc_ref(v___x_2362_);
v___x_2363_ = lean_array_push(v___x_2362_, v_declName_2346_);
v___x_2364_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmds(v_a_2358_, v___x_2363_, v_a_2347_, v_a_2348_, v_a_2349_, v_a_2350_, v_a_2351_, v_a_2352_);
lean_dec_ref(v___x_2363_);
lean_dec(v_a_2358_);
if (lean_obj_tag(v___x_2364_) == 0)
{
lean_object* v_options_2365_; lean_object* v_a_2366_; lean_object* v___x_2368_; uint8_t v_isShared_2369_; uint8_t v_isSharedCheck_2406_; 
v_options_2365_ = lean_ctor_get(v_a_2351_, 2);
v_a_2366_ = lean_ctor_get(v___x_2364_, 0);
v_isSharedCheck_2406_ = !lean_is_exclusive(v___x_2364_);
if (v_isSharedCheck_2406_ == 0)
{
v___x_2368_ = v___x_2364_;
v_isShared_2369_ = v_isSharedCheck_2406_;
goto v_resetjp_2367_;
}
else
{
lean_inc(v_a_2366_);
lean_dec(v___x_2364_);
v___x_2368_ = lean_box(0);
v_isShared_2369_ = v_isSharedCheck_2406_;
goto v_resetjp_2367_;
}
v_resetjp_2367_:
{
lean_object* v_inheritedTraceOptions_2370_; uint8_t v_hasTrace_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; 
v_inheritedTraceOptions_2370_ = lean_ctor_get(v_a_2351_, 13);
v_hasTrace_2371_ = lean_ctor_get_uint8(v_options_2365_, sizeof(void*)*1);
v___x_2372_ = lean_array_push(v___x_2362_, v_a_2360_);
v___x_2373_ = l_Array_append___redArg(v___x_2372_, v_a_2366_);
lean_dec(v_a_2366_);
if (v_hasTrace_2371_ == 0)
{
lean_object* v___x_2375_; 
if (v_isShared_2369_ == 0)
{
lean_ctor_set(v___x_2368_, 0, v___x_2373_);
v___x_2375_ = v___x_2368_;
goto v_reusejp_2374_;
}
else
{
lean_object* v_reuseFailAlloc_2376_; 
v_reuseFailAlloc_2376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2376_, 0, v___x_2373_);
v___x_2375_ = v_reuseFailAlloc_2376_;
goto v_reusejp_2374_;
}
v_reusejp_2374_:
{
return v___x_2375_;
}
}
else
{
lean_object* v___x_2377_; lean_object* v___x_2378_; uint8_t v___x_2379_; 
v___x_2377_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__initFn___closed__3_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_));
v___x_2378_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2, &lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2_once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__2);
v___x_2379_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2370_, v_options_2365_, v___x_2378_);
if (v___x_2379_ == 0)
{
lean_object* v___x_2381_; 
if (v_isShared_2369_ == 0)
{
lean_ctor_set(v___x_2368_, 0, v___x_2373_);
v___x_2381_ = v___x_2368_;
goto v_reusejp_2380_;
}
else
{
lean_object* v_reuseFailAlloc_2382_; 
v_reuseFailAlloc_2382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2382_, 0, v___x_2373_);
v___x_2381_ = v_reuseFailAlloc_2382_;
goto v_reusejp_2380_;
}
v_reusejp_2380_:
{
return v___x_2381_;
}
}
else
{
lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; 
lean_del_object(v___x_2368_);
v___x_2383_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4, &lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4_once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___closed__4);
lean_inc_ref(v___x_2373_);
v___x_2384_ = lean_array_to_list(v___x_2373_);
v___x_2385_ = lean_box(0);
v___x_2386_ = lp_plausible_List_mapTR_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__0(v___x_2384_, v___x_2385_);
v___x_2387_ = l_Lean_MessageData_ofList(v___x_2386_);
v___x_2388_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2388_, 0, v___x_2383_);
lean_ctor_set(v___x_2388_, 1, v___x_2387_);
v___x_2389_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg(v___x_2377_, v___x_2388_, v_a_2349_, v_a_2350_, v_a_2351_, v_a_2352_);
if (lean_obj_tag(v___x_2389_) == 0)
{
lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2396_; 
v_isSharedCheck_2396_ = !lean_is_exclusive(v___x_2389_);
if (v_isSharedCheck_2396_ == 0)
{
lean_object* v_unused_2397_; 
v_unused_2397_ = lean_ctor_get(v___x_2389_, 0);
lean_dec(v_unused_2397_);
v___x_2391_ = v___x_2389_;
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
else
{
lean_dec(v___x_2389_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
lean_object* v___x_2394_; 
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v___x_2373_);
v___x_2394_ = v___x_2391_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v___x_2373_);
v___x_2394_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
return v___x_2394_;
}
}
}
else
{
lean_object* v_a_2398_; lean_object* v___x_2400_; uint8_t v_isShared_2401_; uint8_t v_isSharedCheck_2405_; 
lean_dec_ref(v___x_2373_);
v_a_2398_ = lean_ctor_get(v___x_2389_, 0);
v_isSharedCheck_2405_ = !lean_is_exclusive(v___x_2389_);
if (v_isSharedCheck_2405_ == 0)
{
v___x_2400_ = v___x_2389_;
v_isShared_2401_ = v_isSharedCheck_2405_;
goto v_resetjp_2399_;
}
else
{
lean_inc(v_a_2398_);
lean_dec(v___x_2389_);
v___x_2400_ = lean_box(0);
v_isShared_2401_ = v_isSharedCheck_2405_;
goto v_resetjp_2399_;
}
v_resetjp_2399_:
{
lean_object* v___x_2403_; 
if (v_isShared_2401_ == 0)
{
v___x_2403_ = v___x_2400_;
goto v_reusejp_2402_;
}
else
{
lean_object* v_reuseFailAlloc_2404_; 
v_reuseFailAlloc_2404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2404_, 0, v_a_2398_);
v___x_2403_ = v_reuseFailAlloc_2404_;
goto v_reusejp_2402_;
}
v_reusejp_2402_:
{
return v___x_2403_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2407_; lean_object* v___x_2409_; uint8_t v_isShared_2410_; uint8_t v_isSharedCheck_2414_; 
lean_dec_ref(v___x_2362_);
lean_dec(v_a_2360_);
v_a_2407_ = lean_ctor_get(v___x_2364_, 0);
v_isSharedCheck_2414_ = !lean_is_exclusive(v___x_2364_);
if (v_isSharedCheck_2414_ == 0)
{
v___x_2409_ = v___x_2364_;
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
else
{
lean_inc(v_a_2407_);
lean_dec(v___x_2364_);
v___x_2409_ = lean_box(0);
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
v_resetjp_2408_:
{
lean_object* v___x_2412_; 
if (v_isShared_2410_ == 0)
{
v___x_2412_ = v___x_2409_;
goto v_reusejp_2411_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v_a_2407_);
v___x_2412_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2411_;
}
v_reusejp_2411_:
{
return v___x_2412_;
}
}
}
}
else
{
lean_object* v_a_2415_; lean_object* v___x_2417_; uint8_t v_isShared_2418_; uint8_t v_isSharedCheck_2422_; 
lean_dec(v_a_2358_);
lean_dec(v_declName_2346_);
v_a_2415_ = lean_ctor_get(v___x_2359_, 0);
v_isSharedCheck_2422_ = !lean_is_exclusive(v___x_2359_);
if (v_isSharedCheck_2422_ == 0)
{
v___x_2417_ = v___x_2359_;
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
else
{
lean_inc(v_a_2415_);
lean_dec(v___x_2359_);
v___x_2417_ = lean_box(0);
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
v_resetjp_2416_:
{
lean_object* v___x_2420_; 
if (v_isShared_2418_ == 0)
{
v___x_2420_ = v___x_2417_;
goto v_reusejp_2419_;
}
else
{
lean_object* v_reuseFailAlloc_2421_; 
v_reuseFailAlloc_2421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2421_, 0, v_a_2415_);
v___x_2420_ = v_reuseFailAlloc_2421_;
goto v_reusejp_2419_;
}
v_reusejp_2419_:
{
return v___x_2420_;
}
}
}
}
else
{
lean_object* v_a_2423_; lean_object* v___x_2425_; uint8_t v_isShared_2426_; uint8_t v_isSharedCheck_2430_; 
lean_dec(v_declName_2346_);
v_a_2423_ = lean_ctor_get(v___x_2357_, 0);
v_isSharedCheck_2430_ = !lean_is_exclusive(v___x_2357_);
if (v_isSharedCheck_2430_ == 0)
{
v___x_2425_ = v___x_2357_;
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
else
{
lean_inc(v_a_2423_);
lean_dec(v___x_2357_);
v___x_2425_ = lean_box(0);
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
v_resetjp_2424_:
{
lean_object* v___x_2428_; 
if (v_isShared_2426_ == 0)
{
v___x_2428_ = v___x_2425_;
goto v_reusejp_2427_;
}
else
{
lean_object* v_reuseFailAlloc_2429_; 
v_reuseFailAlloc_2429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2429_, 0, v_a_2423_);
v___x_2428_ = v_reuseFailAlloc_2429_;
goto v_reusejp_2427_;
}
v_reusejp_2427_:
{
return v___x_2428_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___boxed(lean_object* v_declName_2431_, lean_object* v_a_2432_, lean_object* v_a_2433_, lean_object* v_a_2434_, lean_object* v_a_2435_, lean_object* v_a_2436_, lean_object* v_a_2437_, lean_object* v_a_2438_){
_start:
{
lean_object* v_res_2439_; 
v_res_2439_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd(v_declName_2431_, v_a_2432_, v_a_2433_, v_a_2434_, v_a_2435_, v_a_2436_, v_a_2437_);
lean_dec(v_a_2437_);
lean_dec_ref(v_a_2436_);
lean_dec(v_a_2435_);
lean_dec_ref(v_a_2434_);
lean_dec(v_a_2433_);
lean_dec_ref(v_a_2432_);
return v_res_2439_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1(lean_object* v_cls_2440_, lean_object* v_msg_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_){
_start:
{
lean_object* v___x_2449_; 
v___x_2449_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___redArg(v_cls_2440_, v_msg_2441_, v___y_2444_, v___y_2445_, v___y_2446_, v___y_2447_);
return v___x_2449_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1___boxed(lean_object* v_cls_2450_, lean_object* v_msg_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_){
_start:
{
lean_object* v_res_2459_; 
v_res_2459_ = lp_plausible_Lean_addTrace___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd_spec__1(v_cls_2450_, v_msg_2451_, v___y_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
lean_dec(v___y_2457_);
lean_dec_ref(v___y_2456_);
lean_dec(v___y_2455_);
lean_dec_ref(v___y_2454_);
lean_dec(v___y_2453_);
lean_dec_ref(v___y_2452_);
return v_res_2459_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg(lean_object* v_declName_2460_, lean_object* v___y_2461_){
_start:
{
lean_object* v___x_2463_; lean_object* v_env_2464_; uint8_t v___x_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; 
v___x_2463_ = lean_st_ref_get(v___y_2461_);
v_env_2464_ = lean_ctor_get(v___x_2463_, 0);
lean_inc_ref(v_env_2464_);
lean_dec(v___x_2463_);
v___x_2465_ = l_Lean_isInductiveCore(v_env_2464_, v_declName_2460_);
v___x_2466_ = lean_box(v___x_2465_);
v___x_2467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2467_, 0, v___x_2466_);
return v___x_2467_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg___boxed(lean_object* v_declName_2468_, lean_object* v___y_2469_, lean_object* v___y_2470_){
_start:
{
lean_object* v_res_2471_; 
v_res_2471_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg(v_declName_2468_, v___y_2469_);
lean_dec(v___y_2469_);
return v_res_2471_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0(lean_object* v_declName_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_){
_start:
{
lean_object* v___x_2476_; 
v___x_2476_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg(v_declName_2472_, v___y_2474_);
return v___x_2476_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___boxed(lean_object* v_declName_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_){
_start:
{
lean_object* v_res_2481_; 
v_res_2481_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0(v_declName_2477_, v___y_2478_, v___y_2479_);
lean_dec(v___y_2479_);
lean_dec_ref(v___y_2478_);
return v_res_2481_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0(uint8_t v_____do__lift_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_){
_start:
{
if (v_____do__lift_2482_ == 0)
{
uint8_t v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; 
v___x_2486_ = 1;
v___x_2487_ = lean_box(v___x_2486_);
v___x_2488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2488_, 0, v___x_2487_);
return v___x_2488_;
}
else
{
uint8_t v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; 
v___x_2489_ = 0;
v___x_2490_ = lean_box(v___x_2489_);
v___x_2491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2490_);
return v___x_2491_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0___boxed(lean_object* v_____do__lift_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_){
_start:
{
uint8_t v_____do__lift_6289__boxed_2496_; lean_object* v_res_2497_; 
v_____do__lift_6289__boxed_2496_ = lean_unbox(v_____do__lift_2492_);
v_res_2497_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0(v_____do__lift_6289__boxed_2496_, v___y_2493_, v___y_2494_);
lean_dec(v___y_2494_);
lean_dec_ref(v___y_2493_);
return v_res_2497_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1(lean_object* v_as_2498_, size_t v_i_2499_, size_t v_stop_2500_, lean_object* v_b_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_){
_start:
{
uint8_t v___x_2505_; 
v___x_2505_ = lean_usize_dec_eq(v_i_2499_, v_stop_2500_);
if (v___x_2505_ == 0)
{
lean_object* v___x_2506_; lean_object* v___x_2507_; 
v___x_2506_ = lean_array_uget_borrowed(v_as_2498_, v_i_2499_);
lean_inc(v___x_2506_);
v___x_2507_ = l_Lean_Elab_Command_elabCommand(v___x_2506_, v___y_2502_, v___y_2503_);
if (lean_obj_tag(v___x_2507_) == 0)
{
lean_object* v_a_2508_; size_t v___x_2509_; size_t v___x_2510_; 
v_a_2508_ = lean_ctor_get(v___x_2507_, 0);
lean_inc(v_a_2508_);
lean_dec_ref_known(v___x_2507_, 1);
v___x_2509_ = ((size_t)1ULL);
v___x_2510_ = lean_usize_add(v_i_2499_, v___x_2509_);
v_i_2499_ = v___x_2510_;
v_b_2501_ = v_a_2508_;
goto _start;
}
else
{
return v___x_2507_;
}
}
else
{
lean_object* v___x_2512_; 
v___x_2512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2512_, 0, v_b_2501_);
return v___x_2512_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1___boxed(lean_object* v_as_2513_, lean_object* v_i_2514_, lean_object* v_stop_2515_, lean_object* v_b_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_, lean_object* v___y_2519_){
_start:
{
size_t v_i_boxed_2520_; size_t v_stop_boxed_2521_; lean_object* v_res_2522_; 
v_i_boxed_2520_ = lean_unbox_usize(v_i_2514_);
lean_dec(v_i_2514_);
v_stop_boxed_2521_ = lean_unbox_usize(v_stop_2515_);
lean_dec(v_stop_2515_);
v_res_2522_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1(v_as_2513_, v_i_boxed_2520_, v_stop_boxed_2521_, v_b_2516_, v___y_2517_, v___y_2518_);
lean_dec(v___y_2518_);
lean_dec_ref(v___y_2517_);
lean_dec_ref(v_as_2513_);
return v_res_2522_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5(lean_object* v_as_2523_, size_t v_sz_2524_, size_t v_i_2525_, lean_object* v_b_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_){
_start:
{
lean_object* v_a_2531_; uint8_t v___x_2535_; 
v___x_2535_ = lean_usize_dec_lt(v_i_2525_, v_sz_2524_);
if (v___x_2535_ == 0)
{
lean_object* v___x_2536_; 
v___x_2536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2536_, 0, v_b_2526_);
return v___x_2536_;
}
else
{
lean_object* v_a_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; 
v_a_2537_ = lean_array_uget_borrowed(v_as_2523_, v_i_2525_);
lean_inc(v_a_2537_);
v___x_2538_ = lean_alloc_closure((void*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceCmd___boxed), 8, 1);
lean_closure_set(v___x_2538_, 0, v_a_2537_);
v___x_2539_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_2538_, v___y_2527_, v___y_2528_);
if (lean_obj_tag(v___x_2539_) == 0)
{
lean_object* v_a_2540_; lean_object* v___x_2541_; lean_object* v___y_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; uint8_t v___x_2546_; 
v_a_2540_ = lean_ctor_get(v___x_2539_, 0);
lean_inc(v_a_2540_);
lean_dec_ref_known(v___x_2539_, 1);
v___x_2541_ = lean_box(0);
v___x_2544_ = lean_unsigned_to_nat(0u);
v___x_2545_ = lean_array_get_size(v_a_2540_);
v___x_2546_ = lean_nat_dec_lt(v___x_2544_, v___x_2545_);
if (v___x_2546_ == 0)
{
lean_dec(v_a_2540_);
v_a_2531_ = v___x_2541_;
goto v___jp_2530_;
}
else
{
uint8_t v___x_2547_; 
v___x_2547_ = lean_nat_dec_le(v___x_2545_, v___x_2545_);
if (v___x_2547_ == 0)
{
if (v___x_2546_ == 0)
{
lean_dec(v_a_2540_);
v_a_2531_ = v___x_2541_;
goto v___jp_2530_;
}
else
{
size_t v___x_2548_; size_t v___x_2549_; lean_object* v___x_2550_; 
v___x_2548_ = ((size_t)0ULL);
v___x_2549_ = lean_usize_of_nat(v___x_2545_);
v___x_2550_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1(v_a_2540_, v___x_2548_, v___x_2549_, v___x_2541_, v___y_2527_, v___y_2528_);
lean_dec(v_a_2540_);
v___y_2543_ = v___x_2550_;
goto v___jp_2542_;
}
}
else
{
size_t v___x_2551_; size_t v___x_2552_; lean_object* v___x_2553_; 
v___x_2551_ = ((size_t)0ULL);
v___x_2552_ = lean_usize_of_nat(v___x_2545_);
v___x_2553_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__1(v_a_2540_, v___x_2551_, v___x_2552_, v___x_2541_, v___y_2527_, v___y_2528_);
lean_dec(v_a_2540_);
v___y_2543_ = v___x_2553_;
goto v___jp_2542_;
}
}
v___jp_2542_:
{
if (lean_obj_tag(v___y_2543_) == 0)
{
lean_dec_ref_known(v___y_2543_, 1);
v_a_2531_ = v___x_2541_;
goto v___jp_2530_;
}
else
{
return v___y_2543_;
}
}
}
else
{
lean_object* v_a_2554_; lean_object* v___x_2556_; uint8_t v_isShared_2557_; uint8_t v_isSharedCheck_2561_; 
v_a_2554_ = lean_ctor_get(v___x_2539_, 0);
v_isSharedCheck_2561_ = !lean_is_exclusive(v___x_2539_);
if (v_isSharedCheck_2561_ == 0)
{
v___x_2556_ = v___x_2539_;
v_isShared_2557_ = v_isSharedCheck_2561_;
goto v_resetjp_2555_;
}
else
{
lean_inc(v_a_2554_);
lean_dec(v___x_2539_);
v___x_2556_ = lean_box(0);
v_isShared_2557_ = v_isSharedCheck_2561_;
goto v_resetjp_2555_;
}
v_resetjp_2555_:
{
lean_object* v___x_2559_; 
if (v_isShared_2557_ == 0)
{
v___x_2559_ = v___x_2556_;
goto v_reusejp_2558_;
}
else
{
lean_object* v_reuseFailAlloc_2560_; 
v_reuseFailAlloc_2560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2560_, 0, v_a_2554_);
v___x_2559_ = v_reuseFailAlloc_2560_;
goto v_reusejp_2558_;
}
v_reusejp_2558_:
{
return v___x_2559_;
}
}
}
}
v___jp_2530_:
{
size_t v___x_2532_; size_t v___x_2533_; 
v___x_2532_ = ((size_t)1ULL);
v___x_2533_ = lean_usize_add(v_i_2525_, v___x_2532_);
v_i_2525_ = v___x_2533_;
v_b_2526_ = v_a_2531_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5___boxed(lean_object* v_as_2562_, lean_object* v_sz_2563_, lean_object* v_i_2564_, lean_object* v_b_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_){
_start:
{
size_t v_sz_boxed_2569_; size_t v_i_boxed_2570_; lean_object* v_res_2571_; 
v_sz_boxed_2569_ = lean_unbox_usize(v_sz_2563_);
lean_dec(v_sz_2563_);
v_i_boxed_2570_ = lean_unbox_usize(v_i_2564_);
lean_dec(v_i_2564_);
v_res_2571_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5(v_as_2562_, v_sz_boxed_2569_, v_i_boxed_2570_, v_b_2565_, v___y_2566_, v___y_2567_);
lean_dec(v___y_2567_);
lean_dec_ref(v___y_2566_);
lean_dec_ref(v_as_2562_);
return v_res_2571_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6(lean_object* v_as_2572_, size_t v_i_2573_, size_t v_stop_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_){
_start:
{
uint8_t v___x_2578_; 
v___x_2578_ = lean_usize_dec_eq(v_i_2573_, v_stop_2574_);
if (v___x_2578_ == 0)
{
uint8_t v___x_2579_; uint8_t v_a_2581_; lean_object* v___x_2587_; lean_object* v___x_2588_; 
v___x_2579_ = 1;
v___x_2587_ = lean_array_uget_borrowed(v_as_2572_, v_i_2573_);
lean_inc(v___x_2587_);
v___x_2588_ = lp_plausible_Lean_isInductive___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__0___redArg(v___x_2587_, v___y_2576_);
if (lean_obj_tag(v___x_2588_) == 0)
{
lean_object* v_a_2589_; lean_object* v___x_2591_; uint8_t v_isShared_2592_; uint8_t v_isSharedCheck_2598_; 
v_a_2589_ = lean_ctor_get(v___x_2588_, 0);
v_isSharedCheck_2598_ = !lean_is_exclusive(v___x_2588_);
if (v_isSharedCheck_2598_ == 0)
{
v___x_2591_ = v___x_2588_;
v_isShared_2592_ = v_isSharedCheck_2598_;
goto v_resetjp_2590_;
}
else
{
lean_inc(v_a_2589_);
lean_dec(v___x_2588_);
v___x_2591_ = lean_box(0);
v_isShared_2592_ = v_isSharedCheck_2598_;
goto v_resetjp_2590_;
}
v_resetjp_2590_:
{
uint8_t v___x_2593_; 
v___x_2593_ = lean_unbox(v_a_2589_);
lean_dec(v_a_2589_);
if (v___x_2593_ == 0)
{
lean_object* v___x_2594_; lean_object* v___x_2596_; 
v___x_2594_ = lean_box(v___x_2579_);
if (v_isShared_2592_ == 0)
{
lean_ctor_set(v___x_2591_, 0, v___x_2594_);
v___x_2596_ = v___x_2591_;
goto v_reusejp_2595_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v___x_2594_);
v___x_2596_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2595_;
}
v_reusejp_2595_:
{
return v___x_2596_;
}
}
else
{
lean_del_object(v___x_2591_);
v_a_2581_ = v___x_2578_;
goto v___jp_2580_;
}
}
}
else
{
if (lean_obj_tag(v___x_2588_) == 0)
{
lean_object* v_a_2599_; uint8_t v___x_2600_; 
v_a_2599_ = lean_ctor_get(v___x_2588_, 0);
lean_inc(v_a_2599_);
lean_dec_ref_known(v___x_2588_, 1);
v___x_2600_ = lean_unbox(v_a_2599_);
lean_dec(v_a_2599_);
v_a_2581_ = v___x_2600_;
goto v___jp_2580_;
}
else
{
return v___x_2588_;
}
}
v___jp_2580_:
{
if (v_a_2581_ == 0)
{
size_t v___x_2582_; size_t v___x_2583_; 
v___x_2582_ = ((size_t)1ULL);
v___x_2583_ = lean_usize_add(v_i_2573_, v___x_2582_);
v_i_2573_ = v___x_2583_;
goto _start;
}
else
{
lean_object* v___x_2585_; lean_object* v___x_2586_; 
v___x_2585_ = lean_box(v___x_2579_);
v___x_2586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2586_, 0, v___x_2585_);
return v___x_2586_;
}
}
}
else
{
uint8_t v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; 
v___x_2601_ = 0;
v___x_2602_ = lean_box(v___x_2601_);
v___x_2603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2603_, 0, v___x_2602_);
return v___x_2603_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6___boxed(lean_object* v_as_2604_, lean_object* v_i_2605_, lean_object* v_stop_2606_, lean_object* v___y_2607_, lean_object* v___y_2608_, lean_object* v___y_2609_){
_start:
{
size_t v_i_boxed_2610_; size_t v_stop_boxed_2611_; lean_object* v_res_2612_; 
v_i_boxed_2610_ = lean_unbox_usize(v_i_2605_);
lean_dec(v_i_2605_);
v_stop_boxed_2611_ = lean_unbox_usize(v_stop_2606_);
lean_dec(v_stop_2606_);
v_res_2612_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6(v_as_2604_, v_i_boxed_2610_, v_stop_boxed_2611_, v___y_2607_, v___y_2608_);
lean_dec(v___y_2608_);
lean_dec_ref(v___y_2607_);
lean_dec_ref(v_as_2604_);
return v_res_2612_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_2613_; 
v___x_2613_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2613_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_2614_; lean_object* v___x_2615_; 
v___x_2614_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__0);
v___x_2615_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2615_, 0, v___x_2614_);
return v___x_2615_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; 
v___x_2616_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1);
v___x_2617_ = lean_unsigned_to_nat(0u);
v___x_2618_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2618_, 0, v___x_2617_);
lean_ctor_set(v___x_2618_, 1, v___x_2617_);
lean_ctor_set(v___x_2618_, 2, v___x_2617_);
lean_ctor_set(v___x_2618_, 3, v___x_2617_);
lean_ctor_set(v___x_2618_, 4, v___x_2616_);
lean_ctor_set(v___x_2618_, 5, v___x_2616_);
lean_ctor_set(v___x_2618_, 6, v___x_2616_);
lean_ctor_set(v___x_2618_, 7, v___x_2616_);
lean_ctor_set(v___x_2618_, 8, v___x_2616_);
lean_ctor_set(v___x_2618_, 9, v___x_2616_);
return v___x_2618_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2619_ = lean_unsigned_to_nat(32u);
v___x_2620_ = lean_mk_empty_array_with_capacity(v___x_2619_);
v___x_2621_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2621_, 0, v___x_2620_);
return v___x_2621_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4(void){
_start:
{
size_t v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; 
v___x_2622_ = ((size_t)5ULL);
v___x_2623_ = lean_unsigned_to_nat(0u);
v___x_2624_ = lean_unsigned_to_nat(32u);
v___x_2625_ = lean_mk_empty_array_with_capacity(v___x_2624_);
v___x_2626_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__3);
v___x_2627_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2627_, 0, v___x_2626_);
lean_ctor_set(v___x_2627_, 1, v___x_2625_);
lean_ctor_set(v___x_2627_, 2, v___x_2623_);
lean_ctor_set(v___x_2627_, 3, v___x_2623_);
lean_ctor_set_usize(v___x_2627_, 4, v___x_2622_);
return v___x_2627_;
}
}
static lean_object* _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5(void){
_start:
{
lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; 
v___x_2628_ = lean_box(1);
v___x_2629_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__4);
v___x_2630_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__1);
v___x_2631_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2630_);
lean_ctor_set(v___x_2631_, 1, v___x_2629_);
lean_ctor_set(v___x_2631_, 2, v___x_2628_);
return v___x_2631_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg(lean_object* v_msgData_2632_, lean_object* v___y_2633_){
_start:
{
lean_object* v___x_2635_; lean_object* v_env_2636_; lean_object* v___x_2637_; lean_object* v_scopes_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; lean_object* v_opts_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; 
v___x_2635_ = lean_st_ref_get(v___y_2633_);
v_env_2636_ = lean_ctor_get(v___x_2635_, 0);
lean_inc_ref(v_env_2636_);
lean_dec(v___x_2635_);
v___x_2637_ = lean_st_ref_get(v___y_2633_);
v_scopes_2638_ = lean_ctor_get(v___x_2637_, 2);
lean_inc(v_scopes_2638_);
lean_dec(v___x_2637_);
v___x_2639_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2640_ = l_List_head_x21___redArg(v___x_2639_, v_scopes_2638_);
lean_dec(v_scopes_2638_);
v_opts_2641_ = lean_ctor_get(v___x_2640_, 1);
lean_inc_ref(v_opts_2641_);
lean_dec(v___x_2640_);
v___x_2642_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__2);
v___x_2643_ = lean_obj_once(&lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5, &lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5_once, _init_lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___closed__5);
v___x_2644_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2644_, 0, v_env_2636_);
lean_ctor_set(v___x_2644_, 1, v___x_2642_);
lean_ctor_set(v___x_2644_, 2, v___x_2643_);
lean_ctor_set(v___x_2644_, 3, v_opts_2641_);
v___x_2645_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2645_, 0, v___x_2644_);
lean_ctor_set(v___x_2645_, 1, v_msgData_2632_);
v___x_2646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2646_, 0, v___x_2645_);
return v___x_2646_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_){
_start:
{
lean_object* v_res_2650_; 
v_res_2650_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg(v_msgData_2647_, v___y_2648_);
lean_dec(v___y_2648_);
return v_res_2650_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0(void){
_start:
{
lean_object* v___x_2651_; lean_object* v___x_2652_; 
v___x_2651_ = lean_box(1);
v___x_2652_ = l_Lean_MessageData_ofFormat(v___x_2651_);
return v___x_2652_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3(void){
_start:
{
lean_object* v___x_2656_; lean_object* v___x_2657_; 
v___x_2656_ = ((lean_object*)(lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__2));
v___x_2657_ = l_Lean_MessageData_ofFormat(v___x_2656_);
return v___x_2657_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8(lean_object* v_x_2658_, lean_object* v_x_2659_){
_start:
{
if (lean_obj_tag(v_x_2659_) == 0)
{
return v_x_2658_;
}
else
{
lean_object* v_head_2660_; lean_object* v_tail_2661_; lean_object* v___x_2663_; uint8_t v_isShared_2664_; uint8_t v_isSharedCheck_2683_; 
v_head_2660_ = lean_ctor_get(v_x_2659_, 0);
v_tail_2661_ = lean_ctor_get(v_x_2659_, 1);
v_isSharedCheck_2683_ = !lean_is_exclusive(v_x_2659_);
if (v_isSharedCheck_2683_ == 0)
{
v___x_2663_ = v_x_2659_;
v_isShared_2664_ = v_isSharedCheck_2683_;
goto v_resetjp_2662_;
}
else
{
lean_inc(v_tail_2661_);
lean_inc(v_head_2660_);
lean_dec(v_x_2659_);
v___x_2663_ = lean_box(0);
v_isShared_2664_ = v_isSharedCheck_2683_;
goto v_resetjp_2662_;
}
v_resetjp_2662_:
{
lean_object* v_before_2665_; lean_object* v___x_2667_; uint8_t v_isShared_2668_; uint8_t v_isSharedCheck_2681_; 
v_before_2665_ = lean_ctor_get(v_head_2660_, 0);
v_isSharedCheck_2681_ = !lean_is_exclusive(v_head_2660_);
if (v_isSharedCheck_2681_ == 0)
{
lean_object* v_unused_2682_; 
v_unused_2682_ = lean_ctor_get(v_head_2660_, 1);
lean_dec(v_unused_2682_);
v___x_2667_ = v_head_2660_;
v_isShared_2668_ = v_isSharedCheck_2681_;
goto v_resetjp_2666_;
}
else
{
lean_inc(v_before_2665_);
lean_dec(v_head_2660_);
v___x_2667_ = lean_box(0);
v_isShared_2668_ = v_isSharedCheck_2681_;
goto v_resetjp_2666_;
}
v_resetjp_2666_:
{
lean_object* v___x_2669_; lean_object* v___x_2671_; 
v___x_2669_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0);
if (v_isShared_2668_ == 0)
{
lean_ctor_set_tag(v___x_2667_, 7);
lean_ctor_set(v___x_2667_, 1, v___x_2669_);
lean_ctor_set(v___x_2667_, 0, v_x_2658_);
v___x_2671_ = v___x_2667_;
goto v_reusejp_2670_;
}
else
{
lean_object* v_reuseFailAlloc_2680_; 
v_reuseFailAlloc_2680_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2680_, 0, v_x_2658_);
lean_ctor_set(v_reuseFailAlloc_2680_, 1, v___x_2669_);
v___x_2671_ = v_reuseFailAlloc_2680_;
goto v_reusejp_2670_;
}
v_reusejp_2670_:
{
lean_object* v___x_2672_; lean_object* v___x_2674_; 
v___x_2672_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__3);
if (v_isShared_2664_ == 0)
{
lean_ctor_set_tag(v___x_2663_, 7);
lean_ctor_set(v___x_2663_, 1, v___x_2672_);
lean_ctor_set(v___x_2663_, 0, v___x_2671_);
v___x_2674_ = v___x_2663_;
goto v_reusejp_2673_;
}
else
{
lean_object* v_reuseFailAlloc_2679_; 
v_reuseFailAlloc_2679_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2679_, 0, v___x_2671_);
lean_ctor_set(v_reuseFailAlloc_2679_, 1, v___x_2672_);
v___x_2674_ = v_reuseFailAlloc_2679_;
goto v_reusejp_2673_;
}
v_reusejp_2673_:
{
lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; 
v___x_2675_ = l_Lean_MessageData_ofSyntax(v_before_2665_);
v___x_2676_ = l_Lean_indentD(v___x_2675_);
v___x_2677_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2677_, 0, v___x_2674_);
lean_ctor_set(v___x_2677_, 1, v___x_2676_);
v_x_2658_ = v___x_2677_;
v_x_2659_ = v_tail_2661_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7(lean_object* v_opts_2684_, lean_object* v_opt_2685_){
_start:
{
lean_object* v_name_2686_; lean_object* v_defValue_2687_; lean_object* v_map_2688_; lean_object* v___x_2689_; 
v_name_2686_ = lean_ctor_get(v_opt_2685_, 0);
v_defValue_2687_ = lean_ctor_get(v_opt_2685_, 1);
v_map_2688_ = lean_ctor_get(v_opts_2684_, 0);
v___x_2689_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2688_, v_name_2686_);
if (lean_obj_tag(v___x_2689_) == 0)
{
uint8_t v___x_2690_; 
v___x_2690_ = lean_unbox(v_defValue_2687_);
return v___x_2690_;
}
else
{
lean_object* v_val_2691_; 
v_val_2691_ = lean_ctor_get(v___x_2689_, 0);
lean_inc(v_val_2691_);
lean_dec_ref_known(v___x_2689_, 1);
if (lean_obj_tag(v_val_2691_) == 1)
{
uint8_t v_v_2692_; 
v_v_2692_ = lean_ctor_get_uint8(v_val_2691_, 0);
lean_dec_ref_known(v_val_2691_, 0);
return v_v_2692_;
}
else
{
uint8_t v___x_2693_; 
lean_dec(v_val_2691_);
v___x_2693_ = lean_unbox(v_defValue_2687_);
return v___x_2693_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7___boxed(lean_object* v_opts_2694_, lean_object* v_opt_2695_){
_start:
{
uint8_t v_res_2696_; lean_object* v_r_2697_; 
v_res_2696_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7(v_opts_2694_, v_opt_2695_);
lean_dec_ref(v_opt_2695_);
lean_dec_ref(v_opts_2694_);
v_r_2697_ = lean_box(v_res_2696_);
return v_r_2697_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_2701_; lean_object* v___x_2702_; 
v___x_2701_ = ((lean_object*)(lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__1));
v___x_2702_ = l_Lean_MessageData_ofFormat(v___x_2701_);
return v___x_2702_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg(lean_object* v_msgData_2703_, lean_object* v_macroStack_2704_, lean_object* v___y_2705_){
_start:
{
lean_object* v___x_2707_; lean_object* v_scopes_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v_opts_2711_; lean_object* v___x_2712_; uint8_t v___x_2713_; 
v___x_2707_ = lean_st_ref_get(v___y_2705_);
v_scopes_2708_ = lean_ctor_get(v___x_2707_, 2);
lean_inc(v_scopes_2708_);
lean_dec(v___x_2707_);
v___x_2709_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2710_ = l_List_head_x21___redArg(v___x_2709_, v_scopes_2708_);
lean_dec(v_scopes_2708_);
v_opts_2711_ = lean_ctor_get(v___x_2710_, 1);
lean_inc_ref(v_opts_2711_);
lean_dec(v___x_2710_);
v___x_2712_ = l_Lean_Elab_pp_macroStack;
v___x_2713_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7(v_opts_2711_, v___x_2712_);
lean_dec_ref(v_opts_2711_);
if (v___x_2713_ == 0)
{
lean_object* v___x_2714_; 
lean_dec(v_macroStack_2704_);
v___x_2714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2714_, 0, v_msgData_2703_);
return v___x_2714_;
}
else
{
if (lean_obj_tag(v_macroStack_2704_) == 0)
{
lean_object* v___x_2715_; 
v___x_2715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2715_, 0, v_msgData_2703_);
return v___x_2715_;
}
else
{
lean_object* v_head_2716_; lean_object* v_after_2717_; lean_object* v___x_2719_; uint8_t v_isShared_2720_; uint8_t v_isSharedCheck_2732_; 
v_head_2716_ = lean_ctor_get(v_macroStack_2704_, 0);
lean_inc(v_head_2716_);
v_after_2717_ = lean_ctor_get(v_head_2716_, 1);
v_isSharedCheck_2732_ = !lean_is_exclusive(v_head_2716_);
if (v_isSharedCheck_2732_ == 0)
{
lean_object* v_unused_2733_; 
v_unused_2733_ = lean_ctor_get(v_head_2716_, 0);
lean_dec(v_unused_2733_);
v___x_2719_ = v_head_2716_;
v_isShared_2720_ = v_isSharedCheck_2732_;
goto v_resetjp_2718_;
}
else
{
lean_inc(v_after_2717_);
lean_dec(v_head_2716_);
v___x_2719_ = lean_box(0);
v_isShared_2720_ = v_isSharedCheck_2732_;
goto v_resetjp_2718_;
}
v_resetjp_2718_:
{
lean_object* v___x_2721_; lean_object* v___x_2723_; 
v___x_2721_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0);
if (v_isShared_2720_ == 0)
{
lean_ctor_set_tag(v___x_2719_, 7);
lean_ctor_set(v___x_2719_, 1, v___x_2721_);
lean_ctor_set(v___x_2719_, 0, v_msgData_2703_);
v___x_2723_ = v___x_2719_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2731_; 
v_reuseFailAlloc_2731_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2731_, 0, v_msgData_2703_);
lean_ctor_set(v_reuseFailAlloc_2731_, 1, v___x_2721_);
v___x_2723_ = v_reuseFailAlloc_2731_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; lean_object* v_msgData_2728_; lean_object* v___x_2729_; lean_object* v___x_2730_; 
v___x_2724_ = lean_obj_once(&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2, &lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2_once, _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2);
v___x_2725_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2725_, 0, v___x_2723_);
lean_ctor_set(v___x_2725_, 1, v___x_2724_);
v___x_2726_ = l_Lean_MessageData_ofSyntax(v_after_2717_);
v___x_2727_ = l_Lean_indentD(v___x_2726_);
v_msgData_2728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2728_, 0, v___x_2725_);
lean_ctor_set(v_msgData_2728_, 1, v___x_2727_);
v___x_2729_ = lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8(v_msgData_2728_, v_macroStack_2704_);
v___x_2730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2730_, 0, v___x_2729_);
return v___x_2730_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___boxed(lean_object* v_msgData_2734_, lean_object* v_macroStack_2735_, lean_object* v___y_2736_, lean_object* v___y_2737_){
_start:
{
lean_object* v_res_2738_; 
v_res_2738_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg(v_msgData_2734_, v_macroStack_2735_, v___y_2736_);
lean_dec(v___y_2736_);
return v_res_2738_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(lean_object* v_msg_2739_, lean_object* v___y_2740_, lean_object* v___y_2741_){
_start:
{
lean_object* v___x_2743_; 
v___x_2743_ = l_Lean_Elab_Command_getRef___redArg(v___y_2740_);
if (lean_obj_tag(v___x_2743_) == 0)
{
lean_object* v_a_2744_; lean_object* v_macroStack_2745_; lean_object* v___x_2746_; lean_object* v_a_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v_a_2750_; lean_object* v___x_2752_; uint8_t v_isShared_2753_; uint8_t v_isSharedCheck_2758_; 
v_a_2744_ = lean_ctor_get(v___x_2743_, 0);
lean_inc(v_a_2744_);
lean_dec_ref_known(v___x_2743_, 1);
v_macroStack_2745_ = lean_ctor_get(v___y_2740_, 4);
v___x_2746_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg(v_msg_2739_, v___y_2741_);
v_a_2747_ = lean_ctor_get(v___x_2746_, 0);
lean_inc(v_a_2747_);
lean_dec_ref(v___x_2746_);
v___x_2748_ = l_Lean_Elab_getBetterRef(v_a_2744_, v_macroStack_2745_);
lean_dec(v_a_2744_);
lean_inc(v_macroStack_2745_);
v___x_2749_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg(v_a_2747_, v_macroStack_2745_, v___y_2741_);
v_a_2750_ = lean_ctor_get(v___x_2749_, 0);
v_isSharedCheck_2758_ = !lean_is_exclusive(v___x_2749_);
if (v_isSharedCheck_2758_ == 0)
{
v___x_2752_ = v___x_2749_;
v_isShared_2753_ = v_isSharedCheck_2758_;
goto v_resetjp_2751_;
}
else
{
lean_inc(v_a_2750_);
lean_dec(v___x_2749_);
v___x_2752_ = lean_box(0);
v_isShared_2753_ = v_isSharedCheck_2758_;
goto v_resetjp_2751_;
}
v_resetjp_2751_:
{
lean_object* v___x_2754_; lean_object* v___x_2756_; 
v___x_2754_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2754_, 0, v___x_2748_);
lean_ctor_set(v___x_2754_, 1, v_a_2750_);
if (v_isShared_2753_ == 0)
{
lean_ctor_set_tag(v___x_2752_, 1);
lean_ctor_set(v___x_2752_, 0, v___x_2754_);
v___x_2756_ = v___x_2752_;
goto v_reusejp_2755_;
}
else
{
lean_object* v_reuseFailAlloc_2757_; 
v_reuseFailAlloc_2757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2757_, 0, v___x_2754_);
v___x_2756_ = v_reuseFailAlloc_2757_;
goto v_reusejp_2755_;
}
v_reusejp_2755_:
{
return v___x_2756_;
}
}
}
else
{
lean_object* v_a_2759_; lean_object* v___x_2761_; uint8_t v_isShared_2762_; uint8_t v_isSharedCheck_2766_; 
lean_dec_ref(v_msg_2739_);
v_a_2759_ = lean_ctor_get(v___x_2743_, 0);
v_isSharedCheck_2766_ = !lean_is_exclusive(v___x_2743_);
if (v_isSharedCheck_2766_ == 0)
{
v___x_2761_ = v___x_2743_;
v_isShared_2762_ = v_isSharedCheck_2766_;
goto v_resetjp_2760_;
}
else
{
lean_inc(v_a_2759_);
lean_dec(v___x_2743_);
v___x_2761_ = lean_box(0);
v_isShared_2762_ = v_isSharedCheck_2766_;
goto v_resetjp_2760_;
}
v_resetjp_2760_:
{
lean_object* v___x_2764_; 
if (v_isShared_2762_ == 0)
{
v___x_2764_ = v___x_2761_;
goto v_reusejp_2763_;
}
else
{
lean_object* v_reuseFailAlloc_2765_; 
v_reuseFailAlloc_2765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2765_, 0, v_a_2759_);
v___x_2764_ = v_reuseFailAlloc_2765_;
goto v_reusejp_2763_;
}
v_reusejp_2763_:
{
return v___x_2764_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg___boxed(lean_object* v_msg_2767_, lean_object* v___y_2768_, lean_object* v___y_2769_, lean_object* v___y_2770_){
_start:
{
lean_object* v_res_2771_; 
v_res_2771_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(v_msg_2767_, v___y_2768_, v___y_2769_);
lean_dec(v___y_2769_);
lean_dec_ref(v___y_2768_);
return v_res_2771_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg(lean_object* v_msgData_2772_, lean_object* v_macroStack_2773_, lean_object* v___y_2774_){
_start:
{
lean_object* v_options_2776_; lean_object* v___x_2777_; uint8_t v___x_2778_; 
v_options_2776_ = lean_ctor_get(v___y_2774_, 2);
v___x_2777_ = l_Lean_Elab_pp_macroStack;
v___x_2778_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__7(v_options_2776_, v___x_2777_);
if (v___x_2778_ == 0)
{
lean_object* v___x_2779_; 
lean_dec(v_macroStack_2773_);
v___x_2779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2779_, 0, v_msgData_2772_);
return v___x_2779_;
}
else
{
if (lean_obj_tag(v_macroStack_2773_) == 0)
{
lean_object* v___x_2780_; 
v___x_2780_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2780_, 0, v_msgData_2772_);
return v___x_2780_;
}
else
{
lean_object* v_head_2781_; lean_object* v_after_2782_; lean_object* v___x_2784_; uint8_t v_isShared_2785_; uint8_t v_isSharedCheck_2797_; 
v_head_2781_ = lean_ctor_get(v_macroStack_2773_, 0);
lean_inc(v_head_2781_);
v_after_2782_ = lean_ctor_get(v_head_2781_, 1);
v_isSharedCheck_2797_ = !lean_is_exclusive(v_head_2781_);
if (v_isSharedCheck_2797_ == 0)
{
lean_object* v_unused_2798_; 
v_unused_2798_ = lean_ctor_get(v_head_2781_, 0);
lean_dec(v_unused_2798_);
v___x_2784_ = v_head_2781_;
v_isShared_2785_ = v_isSharedCheck_2797_;
goto v_resetjp_2783_;
}
else
{
lean_inc(v_after_2782_);
lean_dec(v_head_2781_);
v___x_2784_ = lean_box(0);
v_isShared_2785_ = v_isSharedCheck_2797_;
goto v_resetjp_2783_;
}
v_resetjp_2783_:
{
lean_object* v___x_2786_; lean_object* v___x_2788_; 
v___x_2786_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8___closed__0);
if (v_isShared_2785_ == 0)
{
lean_ctor_set_tag(v___x_2784_, 7);
lean_ctor_set(v___x_2784_, 1, v___x_2786_);
lean_ctor_set(v___x_2784_, 0, v_msgData_2772_);
v___x_2788_ = v___x_2784_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2796_; 
v_reuseFailAlloc_2796_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2796_, 0, v_msgData_2772_);
lean_ctor_set(v_reuseFailAlloc_2796_, 1, v___x_2786_);
v___x_2788_ = v_reuseFailAlloc_2796_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; lean_object* v_msgData_2793_; lean_object* v___x_2794_; lean_object* v___x_2795_; 
v___x_2789_ = lean_obj_once(&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2, &lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2_once, _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg___closed__2);
v___x_2790_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2790_, 0, v___x_2788_);
lean_ctor_set(v___x_2790_, 1, v___x_2789_);
v___x_2791_ = l_Lean_MessageData_ofSyntax(v_after_2782_);
v___x_2792_ = l_Lean_indentD(v___x_2791_);
v_msgData_2793_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2793_, 0, v___x_2790_);
lean_ctor_set(v_msgData_2793_, 1, v___x_2792_);
v___x_2794_ = lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5_spec__8(v_msgData_2793_, v_macroStack_2773_);
v___x_2795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2795_, 0, v___x_2794_);
return v___x_2795_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_2799_, lean_object* v_macroStack_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_){
_start:
{
lean_object* v_res_2803_; 
v_res_2803_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg(v_msgData_2799_, v_macroStack_2800_, v___y_2801_);
lean_dec_ref(v___y_2801_);
return v_res_2803_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg(lean_object* v_msg_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_){
_start:
{
lean_object* v_ref_2812_; lean_object* v___x_2813_; lean_object* v_a_2814_; lean_object* v_macroStack_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v_a_2818_; lean_object* v___x_2820_; uint8_t v_isShared_2821_; uint8_t v_isSharedCheck_2826_; 
v_ref_2812_ = lean_ctor_get(v___y_2809_, 5);
v___x_2813_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1_spec__1_spec__3(v_msg_2804_, v___y_2807_, v___y_2808_, v___y_2809_, v___y_2810_);
v_a_2814_ = lean_ctor_get(v___x_2813_, 0);
lean_inc(v_a_2814_);
lean_dec_ref(v___x_2813_);
v_macroStack_2815_ = lean_ctor_get(v___y_2805_, 1);
v___x_2816_ = l_Lean_Elab_getBetterRef(v_ref_2812_, v_macroStack_2815_);
lean_inc(v_macroStack_2815_);
v___x_2817_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg(v_a_2814_, v_macroStack_2815_, v___y_2809_);
v_a_2818_ = lean_ctor_get(v___x_2817_, 0);
v_isSharedCheck_2826_ = !lean_is_exclusive(v___x_2817_);
if (v_isSharedCheck_2826_ == 0)
{
v___x_2820_ = v___x_2817_;
v_isShared_2821_ = v_isSharedCheck_2826_;
goto v_resetjp_2819_;
}
else
{
lean_inc(v_a_2818_);
lean_dec(v___x_2817_);
v___x_2820_ = lean_box(0);
v_isShared_2821_ = v_isSharedCheck_2826_;
goto v_resetjp_2819_;
}
v_resetjp_2819_:
{
lean_object* v___x_2822_; lean_object* v___x_2824_; 
v___x_2822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2822_, 0, v___x_2816_);
lean_ctor_set(v___x_2822_, 1, v_a_2818_);
if (v_isShared_2821_ == 0)
{
lean_ctor_set_tag(v___x_2820_, 1);
lean_ctor_set(v___x_2820_, 0, v___x_2822_);
v___x_2824_ = v___x_2820_;
goto v_reusejp_2823_;
}
else
{
lean_object* v_reuseFailAlloc_2825_; 
v_reuseFailAlloc_2825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2825_, 0, v___x_2822_);
v___x_2824_ = v_reuseFailAlloc_2825_;
goto v_reusejp_2823_;
}
v_reusejp_2823_:
{
return v___x_2824_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg___boxed(lean_object* v_msg_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_, lean_object* v___y_2830_, lean_object* v___y_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_){
_start:
{
lean_object* v_res_2835_; 
v_res_2835_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg(v_msg_2827_, v___y_2828_, v___y_2829_, v___y_2830_, v___y_2831_, v___y_2832_, v___y_2833_);
lean_dec(v___y_2833_);
lean_dec_ref(v___y_2832_);
lean_dec(v___y_2831_);
lean_dec_ref(v___y_2830_);
lean_dec(v___y_2829_);
lean_dec_ref(v___y_2828_);
return v_res_2835_;
}
}
static lean_object* _init_lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1(void){
_start:
{
lean_object* v___x_2837_; lean_object* v___x_2838_; 
v___x_2837_ = ((lean_object*)(lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__0));
v___x_2838_ = l_Lean_stringToMessageData(v___x_2837_);
return v___x_2838_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2(lean_object* v_constName_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_, lean_object* v___y_2842_, lean_object* v___y_2843_, lean_object* v___y_2844_, lean_object* v___y_2845_){
_start:
{
lean_object* v___x_2847_; lean_object* v_env_2848_; lean_object* v___x_2849_; 
v___x_2847_ = lean_st_ref_get(v___y_2845_);
v_env_2848_ = lean_ctor_get(v___x_2847_, 0);
lean_inc_ref(v_env_2848_);
lean_dec(v___x_2847_);
lean_inc(v_constName_2839_);
v___x_2849_ = l_Lean_isInductiveCore_x3f(v_env_2848_, v_constName_2839_);
if (lean_obj_tag(v___x_2849_) == 0)
{
lean_object* v___x_2850_; uint8_t v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; 
v___x_2850_ = lean_obj_once(&lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1, &lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1_once, _init_lp_plausible_Lean_getConstInfoCtor___at___00__private_Plausible_DeriveShrinkable_0__Plausible_getCtorArgsNamesAndTypes_spec__1___closed__1);
v___x_2851_ = 0;
v___x_2852_ = l_Lean_MessageData_ofConstName(v_constName_2839_, v___x_2851_);
v___x_2853_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2853_, 0, v___x_2850_);
lean_ctor_set(v___x_2853_, 1, v___x_2852_);
v___x_2854_ = lean_obj_once(&lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1, &lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1_once, _init_lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___closed__1);
v___x_2855_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2855_, 0, v___x_2853_);
lean_ctor_set(v___x_2855_, 1, v___x_2854_);
v___x_2856_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg(v___x_2855_, v___y_2840_, v___y_2841_, v___y_2842_, v___y_2843_, v___y_2844_, v___y_2845_);
return v___x_2856_;
}
else
{
lean_object* v_val_2857_; lean_object* v___x_2859_; uint8_t v_isShared_2860_; uint8_t v_isSharedCheck_2864_; 
lean_dec(v_constName_2839_);
v_val_2857_ = lean_ctor_get(v___x_2849_, 0);
v_isSharedCheck_2864_ = !lean_is_exclusive(v___x_2849_);
if (v_isSharedCheck_2864_ == 0)
{
v___x_2859_ = v___x_2849_;
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
else
{
lean_inc(v_val_2857_);
lean_dec(v___x_2849_);
v___x_2859_ = lean_box(0);
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
v_resetjp_2858_:
{
lean_object* v___x_2862_; 
if (v_isShared_2860_ == 0)
{
lean_ctor_set_tag(v___x_2859_, 0);
v___x_2862_ = v___x_2859_;
goto v_reusejp_2861_;
}
else
{
lean_object* v_reuseFailAlloc_2863_; 
v_reuseFailAlloc_2863_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2863_, 0, v_val_2857_);
v___x_2862_ = v_reuseFailAlloc_2863_;
goto v_reusejp_2861_;
}
v_reusejp_2861_:
{
return v___x_2862_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___boxed(lean_object* v_constName_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_, lean_object* v___y_2870_, lean_object* v___y_2871_, lean_object* v___y_2872_){
_start:
{
lean_object* v_res_2873_; 
v_res_2873_ = lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2(v_constName_2865_, v___y_2866_, v___y_2867_, v___y_2868_, v___y_2869_, v___y_2870_, v___y_2871_);
lean_dec(v___y_2871_);
lean_dec_ref(v___y_2870_);
lean_dec(v___y_2869_);
lean_dec_ref(v___y_2868_);
lean_dec(v___y_2867_);
lean_dec_ref(v___y_2866_);
return v_res_2873_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1(void){
_start:
{
lean_object* v___x_2875_; lean_object* v___x_2876_; 
v___x_2875_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__0));
v___x_2876_ = l_Lean_stringToMessageData(v___x_2875_);
return v___x_2876_;
}
}
static lean_object* _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3(void){
_start:
{
lean_object* v___x_2878_; lean_object* v___x_2879_; 
v___x_2878_ = ((lean_object*)(lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__2));
v___x_2879_ = l_Lean_stringToMessageData(v___x_2878_);
return v___x_2879_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4(lean_object* v_as_2880_, size_t v_sz_2881_, size_t v_i_2882_, lean_object* v_b_2883_, lean_object* v___y_2884_, lean_object* v___y_2885_){
_start:
{
lean_object* v_a_2888_; uint8_t v___x_2892_; 
v___x_2892_ = lean_usize_dec_lt(v_i_2882_, v_sz_2881_);
if (v___x_2892_ == 0)
{
lean_object* v___x_2893_; 
v___x_2893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2893_, 0, v_b_2883_);
return v___x_2893_;
}
else
{
lean_object* v_a_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; 
v_a_2894_ = lean_array_uget_borrowed(v_as_2880_, v_i_2882_);
lean_inc(v_a_2894_);
v___x_2895_ = lean_alloc_closure((void*)(lp_plausible_Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2___boxed), 8, 1);
lean_closure_set(v___x_2895_, 0, v_a_2894_);
v___x_2896_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_2895_, v___y_2884_, v___y_2885_);
if (lean_obj_tag(v___x_2896_) == 0)
{
lean_object* v_a_2897_; lean_object* v_numIndices_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; uint8_t v___x_2901_; 
v_a_2897_ = lean_ctor_get(v___x_2896_, 0);
lean_inc(v_a_2897_);
lean_dec_ref_known(v___x_2896_, 1);
v_numIndices_2898_ = lean_ctor_get(v_a_2897_, 2);
lean_inc(v_numIndices_2898_);
lean_dec(v_a_2897_);
v___x_2899_ = lean_unsigned_to_nat(0u);
v___x_2900_ = lean_box(0);
v___x_2901_ = lean_nat_dec_lt(v___x_2899_, v_numIndices_2898_);
lean_dec(v_numIndices_2898_);
if (v___x_2901_ == 0)
{
v_a_2888_ = v___x_2900_;
goto v___jp_2887_;
}
else
{
lean_object* v___x_2902_; lean_object* v___x_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; 
v___x_2902_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__1);
lean_inc(v_a_2894_);
v___x_2903_ = l_Lean_MessageData_ofName(v_a_2894_);
v___x_2904_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2904_, 0, v___x_2902_);
lean_ctor_set(v___x_2904_, 1, v___x_2903_);
v___x_2905_ = lean_obj_once(&lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3, &lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3_once, _init_lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___closed__3);
v___x_2906_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2906_, 0, v___x_2904_);
lean_ctor_set(v___x_2906_, 1, v___x_2905_);
v___x_2907_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(v___x_2906_, v___y_2884_, v___y_2885_);
if (lean_obj_tag(v___x_2907_) == 0)
{
lean_dec_ref_known(v___x_2907_, 1);
v_a_2888_ = v___x_2900_;
goto v___jp_2887_;
}
else
{
return v___x_2907_;
}
}
}
else
{
lean_object* v_a_2908_; lean_object* v___x_2910_; uint8_t v_isShared_2911_; uint8_t v_isSharedCheck_2915_; 
v_a_2908_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_2915_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2915_ == 0)
{
v___x_2910_ = v___x_2896_;
v_isShared_2911_ = v_isSharedCheck_2915_;
goto v_resetjp_2909_;
}
else
{
lean_inc(v_a_2908_);
lean_dec(v___x_2896_);
v___x_2910_ = lean_box(0);
v_isShared_2911_ = v_isSharedCheck_2915_;
goto v_resetjp_2909_;
}
v_resetjp_2909_:
{
lean_object* v___x_2913_; 
if (v_isShared_2911_ == 0)
{
v___x_2913_ = v___x_2910_;
goto v_reusejp_2912_;
}
else
{
lean_object* v_reuseFailAlloc_2914_; 
v_reuseFailAlloc_2914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2914_, 0, v_a_2908_);
v___x_2913_ = v_reuseFailAlloc_2914_;
goto v_reusejp_2912_;
}
v_reusejp_2912_:
{
return v___x_2913_;
}
}
}
}
v___jp_2887_:
{
size_t v___x_2889_; size_t v___x_2890_; 
v___x_2889_ = ((size_t)1ULL);
v___x_2890_ = lean_usize_add(v_i_2882_, v___x_2889_);
v_i_2882_ = v___x_2890_;
v_b_2883_ = v_a_2888_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4___boxed(lean_object* v_as_2916_, lean_object* v_sz_2917_, lean_object* v_i_2918_, lean_object* v_b_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_){
_start:
{
size_t v_sz_boxed_2923_; size_t v_i_boxed_2924_; lean_object* v_res_2925_; 
v_sz_boxed_2923_ = lean_unbox_usize(v_sz_2917_);
lean_dec(v_sz_2917_);
v_i_boxed_2924_ = lean_unbox_usize(v_i_2918_);
lean_dec(v_i_2918_);
v_res_2925_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4(v_as_2916_, v_sz_boxed_2923_, v_i_boxed_2924_, v_b_2919_, v___y_2920_, v___y_2921_);
lean_dec(v___y_2921_);
lean_dec_ref(v___y_2920_);
lean_dec_ref(v_as_2916_);
return v_res_2925_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1(void){
_start:
{
lean_object* v___x_2927_; lean_object* v___x_2928_; 
v___x_2927_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__0));
v___x_2928_ = l_Lean_stringToMessageData(v___x_2927_);
return v___x_2928_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler(lean_object* v_declNames_2929_, lean_object* v_a_2930_, lean_object* v_a_2931_){
_start:
{
lean_object* v___y_2934_; lean_object* v___y_2935_; lean_object* v___y_2968_; lean_object* v___x_2981_; lean_object* v___x_2982_; uint8_t v___x_2983_; 
v___x_2981_ = lean_unsigned_to_nat(0u);
v___x_2982_ = lean_array_get_size(v_declNames_2929_);
v___x_2983_ = lean_nat_dec_lt(v___x_2981_, v___x_2982_);
if (v___x_2983_ == 0)
{
lean_object* v___x_2984_; 
v___x_2984_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0(v___x_2983_, v_a_2930_, v_a_2931_);
v___y_2968_ = v___x_2984_;
goto v___jp_2967_;
}
else
{
if (v___x_2983_ == 0)
{
v___y_2934_ = v_a_2930_;
v___y_2935_ = v_a_2931_;
goto v___jp_2933_;
}
else
{
size_t v___x_2985_; size_t v___x_2986_; lean_object* v___x_2987_; 
v___x_2985_ = ((size_t)0ULL);
v___x_2986_ = lean_usize_of_nat(v___x_2982_);
v___x_2987_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__6(v_declNames_2929_, v___x_2985_, v___x_2986_, v_a_2930_, v_a_2931_);
if (lean_obj_tag(v___x_2987_) == 0)
{
lean_object* v_a_2988_; uint8_t v___x_2989_; lean_object* v___x_2990_; 
v_a_2988_ = lean_ctor_get(v___x_2987_, 0);
lean_inc(v_a_2988_);
lean_dec_ref_known(v___x_2987_, 1);
v___x_2989_ = lean_unbox(v_a_2988_);
lean_dec(v_a_2988_);
v___x_2990_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___lam__0(v___x_2989_, v_a_2930_, v_a_2931_);
v___y_2968_ = v___x_2990_;
goto v___jp_2967_;
}
else
{
v___y_2968_ = v___x_2987_;
goto v___jp_2967_;
}
}
}
v___jp_2933_:
{
lean_object* v___x_2936_; size_t v_sz_2937_; size_t v___x_2938_; lean_object* v___x_2939_; 
v___x_2936_ = lean_box(0);
v_sz_2937_ = lean_array_size(v_declNames_2929_);
v___x_2938_ = ((size_t)0ULL);
v___x_2939_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__4(v_declNames_2929_, v_sz_2937_, v___x_2938_, v___x_2936_, v___y_2934_, v___y_2935_);
if (lean_obj_tag(v___x_2939_) == 0)
{
lean_object* v___x_2940_; 
lean_dec_ref_known(v___x_2939_, 1);
v___x_2940_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__5(v_declNames_2929_, v_sz_2937_, v___x_2938_, v___x_2936_, v___y_2934_, v___y_2935_);
if (lean_obj_tag(v___x_2940_) == 0)
{
lean_object* v___x_2942_; uint8_t v_isShared_2943_; uint8_t v_isSharedCheck_2949_; 
v_isSharedCheck_2949_ = !lean_is_exclusive(v___x_2940_);
if (v_isSharedCheck_2949_ == 0)
{
lean_object* v_unused_2950_; 
v_unused_2950_ = lean_ctor_get(v___x_2940_, 0);
lean_dec(v_unused_2950_);
v___x_2942_ = v___x_2940_;
v_isShared_2943_ = v_isSharedCheck_2949_;
goto v_resetjp_2941_;
}
else
{
lean_dec(v___x_2940_);
v___x_2942_ = lean_box(0);
v_isShared_2943_ = v_isSharedCheck_2949_;
goto v_resetjp_2941_;
}
v_resetjp_2941_:
{
uint8_t v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2947_; 
v___x_2944_ = 1;
v___x_2945_ = lean_box(v___x_2944_);
if (v_isShared_2943_ == 0)
{
lean_ctor_set(v___x_2942_, 0, v___x_2945_);
v___x_2947_ = v___x_2942_;
goto v_reusejp_2946_;
}
else
{
lean_object* v_reuseFailAlloc_2948_; 
v_reuseFailAlloc_2948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2948_, 0, v___x_2945_);
v___x_2947_ = v_reuseFailAlloc_2948_;
goto v_reusejp_2946_;
}
v_reusejp_2946_:
{
return v___x_2947_;
}
}
}
else
{
lean_object* v_a_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_2958_; 
v_a_2951_ = lean_ctor_get(v___x_2940_, 0);
v_isSharedCheck_2958_ = !lean_is_exclusive(v___x_2940_);
if (v_isSharedCheck_2958_ == 0)
{
v___x_2953_ = v___x_2940_;
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_a_2951_);
lean_dec(v___x_2940_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v___x_2956_; 
if (v_isShared_2954_ == 0)
{
v___x_2956_ = v___x_2953_;
goto v_reusejp_2955_;
}
else
{
lean_object* v_reuseFailAlloc_2957_; 
v_reuseFailAlloc_2957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2957_, 0, v_a_2951_);
v___x_2956_ = v_reuseFailAlloc_2957_;
goto v_reusejp_2955_;
}
v_reusejp_2955_:
{
return v___x_2956_;
}
}
}
}
else
{
lean_object* v_a_2959_; lean_object* v___x_2961_; uint8_t v_isShared_2962_; uint8_t v_isSharedCheck_2966_; 
v_a_2959_ = lean_ctor_get(v___x_2939_, 0);
v_isSharedCheck_2966_ = !lean_is_exclusive(v___x_2939_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2961_ = v___x_2939_;
v_isShared_2962_ = v_isSharedCheck_2966_;
goto v_resetjp_2960_;
}
else
{
lean_inc(v_a_2959_);
lean_dec(v___x_2939_);
v___x_2961_ = lean_box(0);
v_isShared_2962_ = v_isSharedCheck_2966_;
goto v_resetjp_2960_;
}
v_resetjp_2960_:
{
lean_object* v___x_2964_; 
if (v_isShared_2962_ == 0)
{
v___x_2964_ = v___x_2961_;
goto v_reusejp_2963_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v_a_2959_);
v___x_2964_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2963_;
}
v_reusejp_2963_:
{
return v___x_2964_;
}
}
}
}
v___jp_2967_:
{
if (lean_obj_tag(v___y_2968_) == 0)
{
lean_object* v_a_2969_; uint8_t v___x_2970_; 
v_a_2969_ = lean_ctor_get(v___y_2968_, 0);
lean_inc(v_a_2969_);
lean_dec_ref_known(v___y_2968_, 1);
v___x_2970_ = lean_unbox(v_a_2969_);
lean_dec(v_a_2969_);
if (v___x_2970_ == 0)
{
lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v_a_2973_; lean_object* v___x_2975_; uint8_t v_isShared_2976_; uint8_t v_isSharedCheck_2980_; 
v___x_2971_ = lean_obj_once(&lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1, &lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1_once, _init_lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___closed__1);
v___x_2972_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(v___x_2971_, v_a_2930_, v_a_2931_);
v_a_2973_ = lean_ctor_get(v___x_2972_, 0);
v_isSharedCheck_2980_ = !lean_is_exclusive(v___x_2972_);
if (v_isSharedCheck_2980_ == 0)
{
v___x_2975_ = v___x_2972_;
v_isShared_2976_ = v_isSharedCheck_2980_;
goto v_resetjp_2974_;
}
else
{
lean_inc(v_a_2973_);
lean_dec(v___x_2972_);
v___x_2975_ = lean_box(0);
v_isShared_2976_ = v_isSharedCheck_2980_;
goto v_resetjp_2974_;
}
v_resetjp_2974_:
{
lean_object* v___x_2978_; 
if (v_isShared_2976_ == 0)
{
v___x_2978_ = v___x_2975_;
goto v_reusejp_2977_;
}
else
{
lean_object* v_reuseFailAlloc_2979_; 
v_reuseFailAlloc_2979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2979_, 0, v_a_2973_);
v___x_2978_ = v_reuseFailAlloc_2979_;
goto v_reusejp_2977_;
}
v_reusejp_2977_:
{
return v___x_2978_;
}
}
}
else
{
v___y_2934_ = v_a_2930_;
v___y_2935_ = v_a_2931_;
goto v___jp_2933_;
}
}
else
{
return v___y_2968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler___boxed(lean_object* v_declNames_2991_, lean_object* v_a_2992_, lean_object* v_a_2993_, lean_object* v_a_2994_){
_start:
{
lean_object* v_res_2995_; 
v_res_2995_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler(v_declNames_2991_, v_a_2992_, v_a_2993_);
lean_dec(v_a_2993_);
lean_dec_ref(v_a_2992_);
lean_dec_ref(v_declNames_2991_);
return v_res_2995_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4(lean_object* v_msgData_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_){
_start:
{
lean_object* v___x_3000_; 
v___x_3000_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___redArg(v_msgData_2996_, v___y_2998_);
return v___x_3000_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4___boxed(lean_object* v_msgData_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_){
_start:
{
lean_object* v_res_3005_; 
v_res_3005_ = lp_plausible_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__4(v_msgData_3001_, v___y_3002_, v___y_3003_);
lean_dec(v___y_3003_);
lean_dec_ref(v___y_3002_);
return v_res_3005_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3(lean_object* v_00_u03b1_3006_, lean_object* v_msg_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_){
_start:
{
lean_object* v___x_3011_; 
v___x_3011_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___redArg(v_msg_3007_, v___y_3008_, v___y_3009_);
return v___x_3011_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3___boxed(lean_object* v_00_u03b1_3012_, lean_object* v_msg_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_){
_start:
{
lean_object* v_res_3017_; 
v_res_3017_ = lp_plausible_Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3(v_00_u03b1_3012_, v_msg_3013_, v___y_3014_, v___y_3015_);
lean_dec(v___y_3015_);
lean_dec_ref(v___y_3014_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2(lean_object* v_00_u03b1_3018_, lean_object* v_msg_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_){
_start:
{
lean_object* v___x_3027_; 
v___x_3027_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___redArg(v_msg_3019_, v___y_3020_, v___y_3021_, v___y_3022_, v___y_3023_, v___y_3024_, v___y_3025_);
return v___x_3027_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2___boxed(lean_object* v_00_u03b1_3028_, lean_object* v_msg_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_, lean_object* v___y_3036_){
_start:
{
lean_object* v_res_3037_; 
v_res_3037_ = lp_plausible_Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2(v_00_u03b1_3028_, v_msg_3029_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
lean_dec(v___y_3035_);
lean_dec_ref(v___y_3034_);
lean_dec(v___y_3033_);
lean_dec_ref(v___y_3032_);
lean_dec(v___y_3031_);
lean_dec_ref(v___y_3030_);
return v_res_3037_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5(lean_object* v_msgData_3038_, lean_object* v_macroStack_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_){
_start:
{
lean_object* v___x_3043_; 
v___x_3043_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___redArg(v_msgData_3038_, v_macroStack_3039_, v___y_3041_);
return v___x_3043_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5___boxed(lean_object* v_msgData_3044_, lean_object* v_macroStack_3045_, lean_object* v___y_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_){
_start:
{
lean_object* v_res_3049_; 
v_res_3049_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__3_spec__5(v_msgData_3044_, v_macroStack_3045_, v___y_3046_, v___y_3047_);
lean_dec(v___y_3047_);
lean_dec_ref(v___y_3046_);
return v_res_3049_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3(lean_object* v_msgData_3050_, lean_object* v_macroStack_3051_, lean_object* v___y_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_){
_start:
{
lean_object* v___x_3059_; 
v___x_3059_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___redArg(v_msgData_3050_, v_macroStack_3051_, v___y_3056_);
return v___x_3059_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3___boxed(lean_object* v_msgData_3060_, lean_object* v_macroStack_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_){
_start:
{
lean_object* v_res_3069_; 
v_res_3069_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00__private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableInstanceHandler_spec__2_spec__2_spec__3(v_msgData_3060_, v_macroStack_3061_, v___y_3062_, v___y_3063_, v___y_3064_, v___y_3065_, v___y_3066_, v___y_3067_);
lean_dec(v___y_3067_);
lean_dec_ref(v___y_3066_);
lean_dec(v___y_3065_);
lean_dec_ref(v___y_3064_);
lean_dec(v___y_3063_);
lean_dec_ref(v___y_3062_);
return v_res_3069_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_3072_; lean_object* v___x_3073_; lean_object* v___x_3074_; 
v___x_3072_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_mkShrinkableHeader___closed__1));
v___x_3073_ = ((lean_object*)(lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn___closed__0_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_));
v___x_3074_ = l_Lean_Elab_registerDerivingHandler(v___x_3072_, v___x_3073_);
return v___x_3074_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2____boxed(lean_object* v_a_3075_){
_start:
{
lean_object* v_res_3076_; 
v_res_3076_ = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_();
return v_res_3076_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Deriving_Util(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Shrinkable(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_DeriveShrinkable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Deriving_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_plausible___private_Plausible_DeriveShrinkable_0__initFn_00___x40_Plausible_DeriveShrinkable_2530155140____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_plausible___private_Plausible_DeriveShrinkable_0__Plausible_initFn_00___x40_Plausible_DeriveShrinkable_2607000457____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_DeriveShrinkable(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Deriving_Util(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Shrinkable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_DeriveShrinkable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Deriving_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_DeriveShrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_DeriveShrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_DeriveShrinkable(builtin);
}
#ifdef __cplusplus
}
#endif
