// Lean compiler output
// Module: Batteries.CodeAction.Attr
// Imports: public import Init public meta import Init public import Lean.Server.CodeActions.Basic public import Lean.Compiler.IR.CompilerM
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
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_registerPersistentEnvExtensionUnsafe___redArg(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_decl_get_sorry_dep(lean_object*, lean_object*);
uint8_t l_Lean_instBEqAttributeKind_beq(uint8_t, uint8_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "CodeAction"};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "TacticCodeAction"};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(173, 7, 254, 225, 102, 7, 197, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(85, 185, 161, 241, 255, 141, 165, 244)}};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "TacticSeqCodeAction"};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(173, 7, 254, 225, 102, 7, 197, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(91, 217, 69, 194, 18, 129, 10, 129)}};
static const lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActionEntry_default___closed__1_value;
static const lean_array_object lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions = (const lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_TacticCodeActions_insert(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_TacticCodeActions_insert___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "tacticSeqCodeActionExt"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(173, 7, 254, 225, 102, 7, 197, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 224, 120, 78, 188, 161, 238, 101)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_array_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticCodeActionExt"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(173, 7, 254, 225, 102, 7, 197, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 97, 180, 110, 39, 116, 241, 43)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_array_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default___closed__1_value)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionExt;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tactic_code_action"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(173, 7, 254, 225, 102, 7, 197, 223)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value),LEAN_SCALAR_PTR_LITERAL(92, 246, 200, 197, 194, 69, 29, 175)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__5_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__8_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7_value),LEAN_SCALAR_PTR_LITERAL(46, 123, 149, 63, 0, 221, 179, 78)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__10 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__7_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__11 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__11_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__12 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__12_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__13 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__13_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__14 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__14_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__15 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__15_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__16 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__16_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__17 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__17_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__18 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__18_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__19 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__3_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__16_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__20 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__13_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__20_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__21 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__6_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__11_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__21_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__22 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__22_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__3_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__4_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__22_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__23 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__23_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_tactic__code__action___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__23_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action___closed__24 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__24_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_CodeAction_tactic__code__action = (const lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__24_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "invalid attribute 'tactic_code_action', must be global"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 175, 18, 163, 178, 203, 59, 243)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(9, 99, 221, 72, 210, 79, 191, 110)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(41, 241, 91, 105, 28, 91, 24, 102)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__5_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(108, 129, 95, 32, 213, 133, 155, 3)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__6_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 239, 220, 115, 185, 64, 102, 203)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__7_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(250, 204, 201, 238, 175, 229, 150, 121)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__8_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__9_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(223, 191, 219, 176, 29, 131, 180, 179)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__12_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__10_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 48, 83, 27, 233, 17, 133, 201)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__12_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__12_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__13_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__12_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(107, 248, 219, 197, 159, 131, 176, 114)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__13_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__13_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__14_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__13_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(36, 103, 91, 68, 65, 61, 180, 161)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__14_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__14_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__15_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__14_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(136, 140, 102, 71, 106, 187, 59, 62)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__15_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__15_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__17_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__17_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__17_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__19_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__19_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__19_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__22_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__0_value),((lean_object*)&lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__1_value),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__22_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__22_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__23_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_tactic__code__action___closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 39, 240, 32, 220, 126, 91, 56)}};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__23_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__23_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__24_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__23_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__24_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__24_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__25_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 74, .m_data = "Declare a new tactic code action, to appear in the code actions on tactics"};
static const lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__25_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__25_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1(lean_object* v_n_8_, lean_object* v_env_9_, lean_object* v_opts_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3));
v___x_12_ = l_Lean_Environment_evalConstCheck___redArg(v_env_9_, v_opts_10_, v___x_11_, v_n_8_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___boxed(lean_object* v_n_13_, lean_object* v_env_14_, lean_object* v_opts_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1(v_n_13_, v_env_14_, v_opts_15_);
lean_dec_ref(v_opts_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(lean_object* v_e_17_){
_start:
{
if (lean_obj_tag(v_e_17_) == 0)
{
lean_object* v_a_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_27_; 
v_a_19_ = lean_ctor_get(v_e_17_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v_e_17_);
if (v_isSharedCheck_27_ == 0)
{
v___x_21_ = v_e_17_;
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_a_19_);
lean_dec(v_e_17_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
lean_object* v___x_23_; lean_object* v___x_25_; 
v___x_23_ = lean_mk_io_user_error(v_a_19_);
if (v_isShared_22_ == 0)
{
lean_ctor_set_tag(v___x_21_, 1);
lean_ctor_set(v___x_21_, 0, v___x_23_);
v___x_25_ = v___x_21_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_23_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
else
{
lean_object* v_a_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_35_; 
v_a_28_ = lean_ctor_get(v_e_17_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v_e_17_);
if (v_isSharedCheck_35_ == 0)
{
v___x_30_ = v_e_17_;
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_a_28_);
lean_dec(v_e_17_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_33_; 
if (v_isShared_31_ == 0)
{
lean_ctor_set_tag(v___x_30_, 0);
v___x_33_ = v___x_30_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v_a_28_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg___boxed(lean_object* v_e_36_, lean_object* v_a_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(v_e_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0(lean_object* v_00_u03b1_39_, lean_object* v_e_40_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(v_e_40_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___boxed(lean_object* v_00_u03b1_43_, lean_object* v_e_44_, lean_object* v_a_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0(v_00_u03b1_43_, v_e_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction(lean_object* v_n_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_env_50_; lean_object* v_opts_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v_env_50_ = lean_ctor_get(v_a_48_, 0);
v_opts_51_ = lean_ctor_get(v_a_48_, 1);
v___x_52_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_mkTacticCodeAction_unsafe__1___closed__3));
lean_inc_ref(v_env_50_);
v___x_53_ = l_Lean_Environment_evalConstCheck___redArg(v_env_50_, v_opts_51_, v___x_52_, v_n_47_);
v___x_54_ = lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticCodeAction___boxed(lean_object* v_n_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_batteries_Batteries_CodeAction_mkTacticCodeAction(v_n_55_, v_a_56_);
lean_dec_ref(v_a_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1(lean_object* v_n_64_, lean_object* v_env_65_, lean_object* v_opts_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1));
v___x_68_ = l_Lean_Environment_evalConstCheck___redArg(v_env_65_, v_opts_66_, v___x_67_, v_n_64_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___boxed(lean_object* v_n_69_, lean_object* v_env_70_, lean_object* v_opts_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1(v_n_69_, v_env_70_, v_opts_71_);
lean_dec_ref(v_opts_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction(lean_object* v_n_73_, lean_object* v_a_74_){
_start:
{
lean_object* v_env_76_; lean_object* v_opts_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_env_76_ = lean_ctor_get(v_a_74_, 0);
v_opts_77_ = lean_ctor_get(v_a_74_, 1);
v___x_78_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction_unsafe__1___closed__1));
lean_inc_ref(v_env_76_);
v___x_79_ = l_Lean_Environment_evalConstCheck___redArg(v_env_76_, v_opts_77_, v___x_78_, v_n_73_);
v___x_80_ = lp_batteries_IO_ofExcept___at___00Batteries_CodeAction_mkTacticCodeAction_spec__0___redArg(v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction___boxed(lean_object* v_n_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction(v_n_81_, v_a_82_);
lean_dec_ref(v_a_82_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg(lean_object* v_t_99_, lean_object* v_k_100_, lean_object* v_fallback_101_){
_start:
{
if (lean_obj_tag(v_t_99_) == 0)
{
lean_object* v_k_102_; lean_object* v_v_103_; lean_object* v_l_104_; lean_object* v_r_105_; uint8_t v___x_106_; 
v_k_102_ = lean_ctor_get(v_t_99_, 1);
v_v_103_ = lean_ctor_get(v_t_99_, 2);
v_l_104_ = lean_ctor_get(v_t_99_, 3);
v_r_105_ = lean_ctor_get(v_t_99_, 4);
v___x_106_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_100_, v_k_102_);
switch(v___x_106_)
{
case 0:
{
v_t_99_ = v_l_104_;
goto _start;
}
case 1:
{
lean_inc(v_v_103_);
return v_v_103_;
}
default: 
{
v_t_99_ = v_r_105_;
goto _start;
}
}
}
else
{
lean_inc(v_fallback_101_);
return v_fallback_101_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg___boxed(lean_object* v_t_109_, lean_object* v_k_110_, lean_object* v_fallback_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg(v_t_109_, v_k_110_, v_fallback_111_);
lean_dec(v_fallback_111_);
lean_dec(v_k_110_);
lean_dec(v_t_109_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1(lean_object* v_action_115_, lean_object* v_as_116_, size_t v_i_117_, size_t v_stop_118_, lean_object* v_b_119_){
_start:
{
uint8_t v___x_120_; 
v___x_120_ = lean_usize_dec_eq(v_i_117_, v_stop_118_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; size_t v___x_126_; size_t v___x_127_; 
v___x_121_ = lean_array_uget_borrowed(v_as_116_, v_i_117_);
v___x_122_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___closed__0));
v___x_123_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg(v_b_119_, v___x_121_, v___x_122_);
lean_inc_ref(v_action_115_);
v___x_124_ = lean_array_push(v___x_123_, v_action_115_);
lean_inc(v___x_121_);
v___x_125_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_121_, v___x_124_, v_b_119_);
v___x_126_ = ((size_t)1ULL);
v___x_127_ = lean_usize_add(v_i_117_, v___x_126_);
v_i_117_ = v___x_127_;
v_b_119_ = v___x_125_;
goto _start;
}
else
{
lean_dec_ref(v_action_115_);
return v_b_119_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1___boxed(lean_object* v_action_129_, lean_object* v_as_130_, lean_object* v_i_131_, lean_object* v_stop_132_, lean_object* v_b_133_){
_start:
{
size_t v_i_boxed_134_; size_t v_stop_boxed_135_; lean_object* v_res_136_; 
v_i_boxed_134_ = lean_unbox_usize(v_i_131_);
lean_dec(v_i_131_);
v_stop_boxed_135_ = lean_unbox_usize(v_stop_132_);
lean_dec(v_stop_132_);
v_res_136_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1(v_action_129_, v_as_130_, v_i_boxed_134_, v_stop_boxed_135_, v_b_133_);
lean_dec_ref(v_as_130_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_TacticCodeActions_insert(lean_object* v_self_137_, lean_object* v_tacticKinds_138_, lean_object* v_action_139_){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_140_ = lean_array_get_size(v_tacticKinds_138_);
v___x_141_ = lean_unsigned_to_nat(0u);
v___x_142_ = lean_nat_dec_eq(v___x_140_, v___x_141_);
if (v___x_142_ == 0)
{
lean_object* v_onAnyTactic_143_; lean_object* v_onTactic_144_; uint8_t v___x_145_; 
v_onAnyTactic_143_ = lean_ctor_get(v_self_137_, 0);
v_onTactic_144_ = lean_ctor_get(v_self_137_, 1);
v___x_145_ = lean_nat_dec_lt(v___x_141_, v___x_140_);
if (v___x_145_ == 0)
{
lean_dec_ref(v_action_139_);
return v_self_137_;
}
else
{
uint8_t v___x_146_; 
v___x_146_ = lean_nat_dec_le(v___x_140_, v___x_140_);
if (v___x_146_ == 0)
{
if (v___x_145_ == 0)
{
lean_dec_ref(v_action_139_);
return v_self_137_;
}
else
{
lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_156_; 
lean_inc(v_onTactic_144_);
lean_inc_ref(v_onAnyTactic_143_);
v_isSharedCheck_156_ = !lean_is_exclusive(v_self_137_);
if (v_isSharedCheck_156_ == 0)
{
lean_object* v_unused_157_; lean_object* v_unused_158_; 
v_unused_157_ = lean_ctor_get(v_self_137_, 1);
lean_dec(v_unused_157_);
v_unused_158_ = lean_ctor_get(v_self_137_, 0);
lean_dec(v_unused_158_);
v___x_148_ = v_self_137_;
v_isShared_149_ = v_isSharedCheck_156_;
goto v_resetjp_147_;
}
else
{
lean_dec(v_self_137_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_156_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
size_t v___x_150_; size_t v___x_151_; lean_object* v___x_152_; lean_object* v___x_154_; 
v___x_150_ = ((size_t)0ULL);
v___x_151_ = lean_usize_of_nat(v___x_140_);
v___x_152_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1(v_action_139_, v_tacticKinds_138_, v___x_150_, v___x_151_, v_onTactic_144_);
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 1, v___x_152_);
v___x_154_ = v___x_148_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v_onAnyTactic_143_);
lean_ctor_set(v_reuseFailAlloc_155_, 1, v___x_152_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
}
}
else
{
lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_168_; 
lean_inc(v_onTactic_144_);
lean_inc_ref(v_onAnyTactic_143_);
v_isSharedCheck_168_ = !lean_is_exclusive(v_self_137_);
if (v_isSharedCheck_168_ == 0)
{
lean_object* v_unused_169_; lean_object* v_unused_170_; 
v_unused_169_ = lean_ctor_get(v_self_137_, 1);
lean_dec(v_unused_169_);
v_unused_170_ = lean_ctor_get(v_self_137_, 0);
lean_dec(v_unused_170_);
v___x_160_ = v_self_137_;
v_isShared_161_ = v_isSharedCheck_168_;
goto v_resetjp_159_;
}
else
{
lean_dec(v_self_137_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_168_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
size_t v___x_162_; size_t v___x_163_; lean_object* v___x_164_; lean_object* v___x_166_; 
v___x_162_ = ((size_t)0ULL);
v___x_163_ = lean_usize_of_nat(v___x_140_);
v___x_164_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__1(v_action_139_, v_tacticKinds_138_, v___x_162_, v___x_163_, v_onTactic_144_);
if (v_isShared_161_ == 0)
{
lean_ctor_set(v___x_160_, 1, v___x_164_);
v___x_166_ = v___x_160_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_onAnyTactic_143_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v___x_164_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
}
else
{
lean_object* v_onAnyTactic_171_; lean_object* v_onTactic_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_180_; 
v_onAnyTactic_171_ = lean_ctor_get(v_self_137_, 0);
v_onTactic_172_ = lean_ctor_get(v_self_137_, 1);
v_isSharedCheck_180_ = !lean_is_exclusive(v_self_137_);
if (v_isSharedCheck_180_ == 0)
{
v___x_174_ = v_self_137_;
v_isShared_175_ = v_isSharedCheck_180_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_onTactic_172_);
lean_inc(v_onAnyTactic_171_);
lean_dec(v_self_137_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_180_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_176_; lean_object* v___x_178_; 
v___x_176_ = lean_array_push(v_onAnyTactic_171_, v_action_139_);
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 0, v___x_176_);
v___x_178_ = v___x_174_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v___x_176_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_onTactic_172_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_TacticCodeActions_insert___boxed(lean_object* v_self_181_, lean_object* v_tacticKinds_182_, lean_object* v_action_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_batteries_Batteries_CodeAction_TacticCodeActions_insert(v_self_181_, v_tacticKinds_182_, v_action_183_);
lean_dec_ref(v_tacticKinds_182_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0(lean_object* v_00_u03b4_185_, lean_object* v_t_186_, lean_object* v_k_187_, lean_object* v_fallback_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___redArg(v_t_186_, v_k_187_, v_fallback_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0___boxed(lean_object* v_00_u03b4_190_, lean_object* v_t_191_, lean_object* v_k_192_, lean_object* v_fallback_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_getD___at___00Batteries_CodeAction_TacticCodeActions_insert_spec__0(v_00_u03b4_190_, v_t_191_, v_k_192_, v_fallback_193_);
lean_dec(v_fallback_193_);
lean_dec(v_k_192_);
lean_dec(v_t_191_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v_x_195_, lean_object* v_x_196_){
_start:
{
lean_object* v_fst_197_; lean_object* v_snd_198_; lean_object* v_fst_199_; lean_object* v_snd_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_209_; 
v_fst_197_ = lean_ctor_get(v_x_195_, 0);
lean_inc(v_fst_197_);
v_snd_198_ = lean_ctor_get(v_x_195_, 1);
lean_inc(v_snd_198_);
lean_dec_ref(v_x_195_);
v_fst_199_ = lean_ctor_get(v_x_196_, 0);
v_snd_200_ = lean_ctor_get(v_x_196_, 1);
v_isSharedCheck_209_ = !lean_is_exclusive(v_x_196_);
if (v_isSharedCheck_209_ == 0)
{
v___x_202_ = v_x_196_;
v_isShared_203_ = v_isSharedCheck_209_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_snd_200_);
lean_inc(v_fst_199_);
lean_dec(v_x_196_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_209_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_207_; 
v___x_204_ = lean_array_push(v_fst_197_, v_fst_199_);
v___x_205_ = lean_array_push(v_snd_198_, v_snd_200_);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 1, v___x_205_);
lean_ctor_set(v___x_202_, 0, v___x_204_);
v___x_207_ = v___x_202_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_204_);
lean_ctor_set(v_reuseFailAlloc_208_, 1, v___x_205_);
v___x_207_ = v_reuseFailAlloc_208_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
return v___x_207_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v_x_210_, lean_object* v_s_211_){
_start:
{
lean_object* v_fst_212_; lean_object* v___x_213_; 
v_fst_212_ = lean_ctor_get(v_s_211_, 0);
lean_inc_n(v_fst_212_, 3);
v___x_213_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_213_, 0, v_fst_212_);
lean_ctor_set(v___x_213_, 1, v_fst_212_);
lean_ctor_set(v___x_213_, 2, v_fst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v_x_214_, lean_object* v_s_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(v_x_214_, v_s_215_);
lean_dec_ref(v_s_215_);
lean_dec_ref(v_x_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v_x_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_box(0);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v_x_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(v_x_219_);
lean_dec_ref(v_x_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v_x_221_){
_start:
{
lean_object* v_fst_222_; 
v_fst_222_ = lean_ctor_get(v_x_221_, 0);
lean_inc(v_fst_222_);
return v_fst_222_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v_x_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(v_x_223_);
lean_dec_ref(v_x_223_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0(lean_object* v_as_225_, size_t v_i_226_, size_t v_stop_227_, lean_object* v_b_228_, lean_object* v___y_229_){
_start:
{
uint8_t v___x_231_; 
v___x_231_ = lean_usize_dec_eq(v_i_226_, v_stop_227_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = lean_array_uget_borrowed(v_as_225_, v_i_226_);
lean_inc(v___x_232_);
v___x_233_ = lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction(v___x_232_, v___y_229_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_234_; lean_object* v___x_235_; size_t v___x_236_; size_t v___x_237_; 
v_a_234_ = lean_ctor_get(v___x_233_, 0);
lean_inc(v_a_234_);
lean_dec_ref_known(v___x_233_, 1);
v___x_235_ = lean_array_push(v_b_228_, v_a_234_);
v___x_236_ = ((size_t)1ULL);
v___x_237_ = lean_usize_add(v_i_226_, v___x_236_);
v_i_226_ = v___x_237_;
v_b_228_ = v___x_235_;
goto _start;
}
else
{
lean_object* v_a_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_246_; 
lean_dec_ref(v_b_228_);
v_a_239_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_246_ == 0)
{
v___x_241_ = v___x_233_;
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_a_239_);
lean_dec(v___x_233_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_244_; 
if (v_isShared_242_ == 0)
{
v___x_244_ = v___x_241_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_a_239_);
v___x_244_ = v_reuseFailAlloc_245_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
return v___x_244_;
}
}
}
}
else
{
lean_object* v___x_247_; 
v___x_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_247_, 0, v_b_228_);
return v___x_247_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_248_, lean_object* v_i_249_, lean_object* v_stop_250_, lean_object* v_b_251_, lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
size_t v_i_boxed_254_; size_t v_stop_boxed_255_; lean_object* v_res_256_; 
v_i_boxed_254_ = lean_unbox_usize(v_i_249_);
lean_dec(v_i_249_);
v_stop_boxed_255_ = lean_unbox_usize(v_stop_250_);
lean_dec(v_stop_250_);
v_res_256_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0(v_as_248_, v_i_boxed_254_, v_stop_boxed_255_, v_b_251_, v___y_252_);
lean_dec_ref(v___y_252_);
lean_dec_ref(v_as_248_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1(lean_object* v_as_257_, size_t v_i_258_, size_t v_stop_259_, lean_object* v_b_260_, lean_object* v___y_261_){
_start:
{
lean_object* v_a_264_; lean_object* v___y_269_; uint8_t v___x_271_; 
v___x_271_ = lean_usize_dec_eq(v_i_258_, v_stop_259_);
if (v___x_271_ == 0)
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; uint8_t v___x_275_; 
v___x_272_ = lean_array_uget_borrowed(v_as_257_, v_i_258_);
v___x_273_ = lean_unsigned_to_nat(0u);
v___x_274_ = lean_array_get_size(v___x_272_);
v___x_275_ = lean_nat_dec_lt(v___x_273_, v___x_274_);
if (v___x_275_ == 0)
{
v_a_264_ = v_b_260_;
goto v___jp_263_;
}
else
{
uint8_t v___x_276_; 
v___x_276_ = lean_nat_dec_le(v___x_274_, v___x_274_);
if (v___x_276_ == 0)
{
if (v___x_275_ == 0)
{
v_a_264_ = v_b_260_;
goto v___jp_263_;
}
else
{
size_t v___x_277_; size_t v___x_278_; lean_object* v___x_279_; 
v___x_277_ = ((size_t)0ULL);
v___x_278_ = lean_usize_of_nat(v___x_274_);
v___x_279_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0(v___x_272_, v___x_277_, v___x_278_, v_b_260_, v___y_261_);
v___y_269_ = v___x_279_;
goto v___jp_268_;
}
}
else
{
size_t v___x_280_; size_t v___x_281_; lean_object* v___x_282_; 
v___x_280_ = ((size_t)0ULL);
v___x_281_ = lean_usize_of_nat(v___x_274_);
v___x_282_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__0(v___x_272_, v___x_280_, v___x_281_, v_b_260_, v___y_261_);
v___y_269_ = v___x_282_;
goto v___jp_268_;
}
}
}
else
{
lean_object* v___x_283_; 
v___x_283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_283_, 0, v_b_260_);
return v___x_283_;
}
v___jp_263_:
{
size_t v___x_265_; size_t v___x_266_; 
v___x_265_ = ((size_t)1ULL);
v___x_266_ = lean_usize_add(v_i_258_, v___x_265_);
v_i_258_ = v___x_266_;
v_b_260_ = v_a_264_;
goto _start;
}
v___jp_268_:
{
if (lean_obj_tag(v___y_269_) == 0)
{
lean_object* v_a_270_; 
v_a_270_ = lean_ctor_get(v___y_269_, 0);
lean_inc(v_a_270_);
lean_dec_ref_known(v___y_269_, 1);
v_a_264_ = v_a_270_;
goto v___jp_263_;
}
else
{
return v___y_269_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1___boxed(lean_object* v_as_284_, lean_object* v_i_285_, lean_object* v_stop_286_, lean_object* v_b_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
size_t v_i_boxed_290_; size_t v_stop_boxed_291_; lean_object* v_res_292_; 
v_i_boxed_290_ = lean_unbox_usize(v_i_285_);
lean_dec(v_i_285_);
v_stop_boxed_291_ = lean_unbox_usize(v_stop_286_);
lean_dec(v_stop_286_);
v_res_292_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1(v_as_284_, v_i_boxed_290_, v_stop_boxed_291_, v_b_287_, v___y_288_);
lean_dec_ref(v___y_288_);
lean_dec_ref(v_as_284_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v___x_293_, lean_object* v___x_294_, lean_object* v___x_295_, lean_object* v_as_296_, lean_object* v___y_297_){
_start:
{
lean_object* v_a_300_; lean_object* v___y_304_; lean_object* v___x_314_; uint8_t v___x_315_; 
v___x_314_ = lean_array_get_size(v_as_296_);
v___x_315_ = lean_nat_dec_lt(v___x_294_, v___x_314_);
if (v___x_315_ == 0)
{
v_a_300_ = v___x_295_;
goto v___jp_299_;
}
else
{
uint8_t v___x_316_; 
v___x_316_ = lean_nat_dec_le(v___x_314_, v___x_314_);
if (v___x_316_ == 0)
{
if (v___x_315_ == 0)
{
v_a_300_ = v___x_295_;
goto v___jp_299_;
}
else
{
size_t v___x_317_; size_t v___x_318_; lean_object* v___x_319_; 
v___x_317_ = ((size_t)0ULL);
v___x_318_ = lean_usize_of_nat(v___x_314_);
v___x_319_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1(v_as_296_, v___x_317_, v___x_318_, v___x_295_, v___y_297_);
v___y_304_ = v___x_319_;
goto v___jp_303_;
}
}
else
{
size_t v___x_320_; size_t v___x_321_; lean_object* v___x_322_; 
v___x_320_ = ((size_t)0ULL);
v___x_321_ = lean_usize_of_nat(v___x_314_);
v___x_322_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2__spec__1(v_as_296_, v___x_320_, v___x_321_, v___x_295_, v___y_297_);
v___y_304_ = v___x_322_;
goto v___jp_303_;
}
}
v___jp_299_:
{
lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_293_);
lean_ctor_set(v___x_301_, 1, v_a_300_);
v___x_302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
return v___x_302_;
}
v___jp_303_:
{
if (lean_obj_tag(v___y_304_) == 0)
{
lean_object* v_a_305_; 
v_a_305_ = lean_ctor_get(v___y_304_, 0);
lean_inc(v_a_305_);
lean_dec_ref_known(v___y_304_, 1);
v_a_300_ = v_a_305_;
goto v___jp_299_;
}
else
{
lean_object* v_a_306_; lean_object* v___x_308_; uint8_t v_isShared_309_; uint8_t v_isSharedCheck_313_; 
lean_dec_ref(v___x_293_);
v_a_306_ = lean_ctor_get(v___y_304_, 0);
v_isSharedCheck_313_ = !lean_is_exclusive(v___y_304_);
if (v_isSharedCheck_313_ == 0)
{
v___x_308_ = v___y_304_;
v_isShared_309_ = v_isSharedCheck_313_;
goto v_resetjp_307_;
}
else
{
lean_inc(v_a_306_);
lean_dec(v___y_304_);
v___x_308_ = lean_box(0);
v_isShared_309_ = v_isSharedCheck_313_;
goto v_resetjp_307_;
}
v_resetjp_307_:
{
lean_object* v___x_311_; 
if (v_isShared_309_ == 0)
{
v___x_311_ = v___x_308_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v_a_306_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v___x_323_, lean_object* v___x_324_, lean_object* v___x_325_, lean_object* v_as_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(v___x_323_, v___x_324_, v___x_325_, v_as_326_, v___y_327_);
lean_dec_ref(v___y_327_);
lean_dec_ref(v_as_326_);
lean_dec(v___x_324_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(lean_object* v___x_330_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_332_, 0, v___x_330_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v___x_333_, lean_object* v___y_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(v___x_333_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_367_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_));
v___x_368_ = l_Lean_registerPersistentEnvExtensionUnsafe___redArg(v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2____boxed(lean_object* v_a_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_();
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v_x_371_, lean_object* v_x_372_){
_start:
{
lean_object* v_fst_373_; lean_object* v_fst_374_; lean_object* v_snd_375_; lean_object* v_snd_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_386_; 
v_fst_373_ = lean_ctor_get(v_x_372_, 0);
lean_inc(v_fst_373_);
v_fst_374_ = lean_ctor_get(v_x_371_, 0);
lean_inc(v_fst_374_);
v_snd_375_ = lean_ctor_get(v_x_371_, 1);
lean_inc(v_snd_375_);
lean_dec_ref(v_x_371_);
v_snd_376_ = lean_ctor_get(v_x_372_, 1);
v_isSharedCheck_386_ = !lean_is_exclusive(v_x_372_);
if (v_isSharedCheck_386_ == 0)
{
lean_object* v_unused_387_; 
v_unused_387_ = lean_ctor_get(v_x_372_, 0);
lean_dec(v_unused_387_);
v___x_378_ = v_x_372_;
v_isShared_379_ = v_isSharedCheck_386_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_snd_376_);
lean_dec(v_x_372_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_386_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v_tacticKinds_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_384_; 
v_tacticKinds_380_ = lean_ctor_get(v_fst_373_, 1);
lean_inc_ref(v_tacticKinds_380_);
v___x_381_ = lean_array_push(v_fst_374_, v_fst_373_);
v___x_382_ = lp_batteries_Batteries_CodeAction_TacticCodeActions_insert(v_snd_375_, v_tacticKinds_380_, v_snd_376_);
lean_dec_ref(v_tacticKinds_380_);
if (v_isShared_379_ == 0)
{
lean_ctor_set(v___x_378_, 1, v___x_382_);
lean_ctor_set(v___x_378_, 0, v___x_381_);
v___x_384_ = v___x_378_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_381_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v___x_382_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v_x_388_, lean_object* v_s_389_){
_start:
{
lean_object* v_fst_390_; lean_object* v___x_391_; 
v_fst_390_ = lean_ctor_get(v_s_389_, 0);
lean_inc_n(v_fst_390_, 3);
v___x_391_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_391_, 0, v_fst_390_);
lean_ctor_set(v___x_391_, 1, v_fst_390_);
lean_ctor_set(v___x_391_, 2, v_fst_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v_x_392_, lean_object* v_s_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(v_x_392_, v_s_393_);
lean_dec_ref(v_s_393_);
lean_dec_ref(v_x_392_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v_x_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lean_box(0);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v_x_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__2_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(v_x_397_);
lean_dec_ref(v_x_397_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v_x_399_){
_start:
{
lean_object* v_fst_400_; 
v_fst_400_ = lean_ctor_get(v_x_399_, 0);
lean_inc(v_fst_400_);
return v_fst_400_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v_x_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__3_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(v_x_401_);
lean_dec_ref(v_x_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0(lean_object* v_as_403_, size_t v_i_404_, size_t v_stop_405_, lean_object* v_b_406_, lean_object* v___y_407_){
_start:
{
uint8_t v___x_409_; 
v___x_409_ = lean_usize_dec_eq(v_i_404_, v_stop_405_);
if (v___x_409_ == 0)
{
lean_object* v___x_410_; lean_object* v_declName_411_; lean_object* v_tacticKinds_412_; lean_object* v___x_413_; 
v___x_410_ = lean_array_uget_borrowed(v_as_403_, v_i_404_);
v_declName_411_ = lean_ctor_get(v___x_410_, 0);
v_tacticKinds_412_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_declName_411_);
v___x_413_ = lp_batteries_Batteries_CodeAction_mkTacticCodeAction(v_declName_411_, v___y_407_);
if (lean_obj_tag(v___x_413_) == 0)
{
lean_object* v_a_414_; lean_object* v___x_415_; size_t v___x_416_; size_t v___x_417_; 
v_a_414_ = lean_ctor_get(v___x_413_, 0);
lean_inc(v_a_414_);
lean_dec_ref_known(v___x_413_, 1);
v___x_415_ = lp_batteries_Batteries_CodeAction_TacticCodeActions_insert(v_b_406_, v_tacticKinds_412_, v_a_414_);
v___x_416_ = ((size_t)1ULL);
v___x_417_ = lean_usize_add(v_i_404_, v___x_416_);
v_i_404_ = v___x_417_;
v_b_406_ = v___x_415_;
goto _start;
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref(v_b_406_);
v_a_419_ = lean_ctor_get(v___x_413_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_413_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_413_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_413_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
else
{
lean_object* v___x_427_; 
v___x_427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_427_, 0, v_b_406_);
return v___x_427_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_428_, lean_object* v_i_429_, lean_object* v_stop_430_, lean_object* v_b_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
size_t v_i_boxed_434_; size_t v_stop_boxed_435_; lean_object* v_res_436_; 
v_i_boxed_434_ = lean_unbox_usize(v_i_429_);
lean_dec(v_i_429_);
v_stop_boxed_435_ = lean_unbox_usize(v_stop_430_);
lean_dec(v_stop_430_);
v_res_436_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0(v_as_428_, v_i_boxed_434_, v_stop_boxed_435_, v_b_431_, v___y_432_);
lean_dec_ref(v___y_432_);
lean_dec_ref(v_as_428_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1(lean_object* v_as_437_, size_t v_i_438_, size_t v_stop_439_, lean_object* v_b_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_a_444_; lean_object* v___y_449_; uint8_t v___x_451_; 
v___x_451_ = lean_usize_dec_eq(v_i_438_, v_stop_439_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; uint8_t v___x_455_; 
v___x_452_ = lean_array_uget_borrowed(v_as_437_, v_i_438_);
v___x_453_ = lean_unsigned_to_nat(0u);
v___x_454_ = lean_array_get_size(v___x_452_);
v___x_455_ = lean_nat_dec_lt(v___x_453_, v___x_454_);
if (v___x_455_ == 0)
{
v_a_444_ = v_b_440_;
goto v___jp_443_;
}
else
{
uint8_t v___x_456_; 
v___x_456_ = lean_nat_dec_le(v___x_454_, v___x_454_);
if (v___x_456_ == 0)
{
if (v___x_455_ == 0)
{
v_a_444_ = v_b_440_;
goto v___jp_443_;
}
else
{
size_t v___x_457_; size_t v___x_458_; lean_object* v___x_459_; 
v___x_457_ = ((size_t)0ULL);
v___x_458_ = lean_usize_of_nat(v___x_454_);
v___x_459_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0(v___x_452_, v___x_457_, v___x_458_, v_b_440_, v___y_441_);
v___y_449_ = v___x_459_;
goto v___jp_448_;
}
}
else
{
size_t v___x_460_; size_t v___x_461_; lean_object* v___x_462_; 
v___x_460_ = ((size_t)0ULL);
v___x_461_ = lean_usize_of_nat(v___x_454_);
v___x_462_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__0(v___x_452_, v___x_460_, v___x_461_, v_b_440_, v___y_441_);
v___y_449_ = v___x_462_;
goto v___jp_448_;
}
}
}
else
{
lean_object* v___x_463_; 
v___x_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_463_, 0, v_b_440_);
return v___x_463_;
}
v___jp_443_:
{
size_t v___x_445_; size_t v___x_446_; 
v___x_445_ = ((size_t)1ULL);
v___x_446_ = lean_usize_add(v_i_438_, v___x_445_);
v_i_438_ = v___x_446_;
v_b_440_ = v_a_444_;
goto _start;
}
v___jp_448_:
{
if (lean_obj_tag(v___y_449_) == 0)
{
lean_object* v_a_450_; 
v_a_450_ = lean_ctor_get(v___y_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___y_449_, 1);
v_a_444_ = v_a_450_;
goto v___jp_443_;
}
else
{
return v___y_449_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1___boxed(lean_object* v_as_464_, lean_object* v_i_465_, lean_object* v_stop_466_, lean_object* v_b_467_, lean_object* v___y_468_, lean_object* v___y_469_){
_start:
{
size_t v_i_boxed_470_; size_t v_stop_boxed_471_; lean_object* v_res_472_; 
v_i_boxed_470_ = lean_unbox_usize(v_i_465_);
lean_dec(v_i_465_);
v_stop_boxed_471_ = lean_unbox_usize(v_stop_466_);
lean_dec(v_stop_466_);
v_res_472_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1(v_as_464_, v_i_boxed_470_, v_stop_boxed_471_, v_b_467_, v___y_468_);
lean_dec_ref(v___y_468_);
lean_dec_ref(v_as_464_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v___x_473_, lean_object* v___x_474_, lean_object* v___x_475_, lean_object* v_as_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_a_480_; lean_object* v___y_484_; lean_object* v___x_494_; uint8_t v___x_495_; 
v___x_494_ = lean_array_get_size(v_as_476_);
v___x_495_ = lean_nat_dec_lt(v___x_474_, v___x_494_);
if (v___x_495_ == 0)
{
v_a_480_ = v___x_475_;
goto v___jp_479_;
}
else
{
uint8_t v___x_496_; 
v___x_496_ = lean_nat_dec_le(v___x_494_, v___x_494_);
if (v___x_496_ == 0)
{
if (v___x_495_ == 0)
{
v_a_480_ = v___x_475_;
goto v___jp_479_;
}
else
{
size_t v___x_497_; size_t v___x_498_; lean_object* v___x_499_; 
v___x_497_ = ((size_t)0ULL);
v___x_498_ = lean_usize_of_nat(v___x_494_);
v___x_499_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1(v_as_476_, v___x_497_, v___x_498_, v___x_475_, v___y_477_);
v___y_484_ = v___x_499_;
goto v___jp_483_;
}
}
else
{
size_t v___x_500_; size_t v___x_501_; lean_object* v___x_502_; 
v___x_500_ = ((size_t)0ULL);
v___x_501_ = lean_usize_of_nat(v___x_494_);
v___x_502_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2__spec__1(v_as_476_, v___x_500_, v___x_501_, v___x_475_, v___y_477_);
v___y_484_ = v___x_502_;
goto v___jp_483_;
}
}
v___jp_479_:
{
lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_473_);
lean_ctor_set(v___x_481_, 1, v_a_480_);
v___x_482_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
return v___x_482_;
}
v___jp_483_:
{
if (lean_obj_tag(v___y_484_) == 0)
{
lean_object* v_a_485_; 
v_a_485_ = lean_ctor_get(v___y_484_, 0);
lean_inc(v_a_485_);
lean_dec_ref_known(v___y_484_, 1);
v_a_480_ = v_a_485_;
goto v___jp_479_;
}
else
{
lean_object* v_a_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_493_; 
lean_dec_ref(v___x_473_);
v_a_486_ = lean_ctor_get(v___y_484_, 0);
v_isSharedCheck_493_ = !lean_is_exclusive(v___y_484_);
if (v_isSharedCheck_493_ == 0)
{
v___x_488_ = v___y_484_;
v_isShared_489_ = v_isSharedCheck_493_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_a_486_);
lean_dec(v___y_484_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_493_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v___x_491_; 
if (v_isShared_489_ == 0)
{
v___x_491_ = v___x_488_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v_a_486_);
v___x_491_ = v_reuseFailAlloc_492_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
return v___x_491_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v___x_503_, lean_object* v___x_504_, lean_object* v___x_505_, lean_object* v_as_506_, lean_object* v___y_507_, lean_object* v___y_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__4_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(v___x_503_, v___x_504_, v___x_505_, v_as_506_, v___y_507_);
lean_dec_ref(v___y_507_);
lean_dec_ref(v_as_506_);
lean_dec(v___x_504_);
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(lean_object* v___x_510_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_512_, 0, v___x_510_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v___x_513_, lean_object* v___y_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__5_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(v___x_513_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; 
v___x_549_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__11_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_));
v___x_550_ = l_Lean_registerPersistentEnvExtensionUnsafe___redArg(v___x_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2____boxed(lean_object* v_a_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_();
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1(size_t v_sz_611_, size_t v_i_612_, lean_object* v_bs_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
uint8_t v___x_617_; 
v___x_617_ = lean_usize_dec_lt(v_i_612_, v_sz_611_);
if (v___x_617_ == 0)
{
lean_object* v___x_618_; 
v___x_618_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_618_, 0, v_bs_613_);
return v___x_618_;
}
else
{
lean_object* v_v_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v_v_619_ = lean_array_uget_borrowed(v_bs_613_, v_i_612_);
v___x_620_ = lean_box(0);
lean_inc(v_v_619_);
v___x_621_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v_v_619_, v___x_620_, v___y_614_, v___y_615_);
if (lean_obj_tag(v___x_621_) == 0)
{
lean_object* v_a_622_; lean_object* v___x_623_; lean_object* v_bs_x27_624_; size_t v___x_625_; size_t v___x_626_; lean_object* v___x_627_; 
v_a_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc(v_a_622_);
lean_dec_ref_known(v___x_621_, 1);
v___x_623_ = lean_unsigned_to_nat(0u);
v_bs_x27_624_ = lean_array_uset(v_bs_613_, v_i_612_, v___x_623_);
v___x_625_ = ((size_t)1ULL);
v___x_626_ = lean_usize_add(v_i_612_, v___x_625_);
v___x_627_ = lean_array_uset(v_bs_x27_624_, v_i_612_, v_a_622_);
v_i_612_ = v___x_626_;
v_bs_613_ = v___x_627_;
goto _start;
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec_ref(v_bs_613_);
v_a_629_ = lean_ctor_get(v___x_621_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_621_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_621_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_621_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1___boxed(lean_object* v_sz_637_, lean_object* v_i_638_, lean_object* v_bs_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
size_t v_sz_boxed_643_; size_t v_i_boxed_644_; lean_object* v_res_645_; 
v_sz_boxed_643_ = lean_unbox_usize(v_sz_637_);
lean_dec(v_sz_637_);
v_i_boxed_644_ = lean_unbox_usize(v_i_638_);
lean_dec(v_i_638_);
v_res_645_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1(v_sz_boxed_643_, v_i_boxed_644_, v_bs_639_, v___y_640_, v___y_641_);
lean_dec(v___y_641_);
lean_dec_ref(v___y_640_);
return v_res_645_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0(void){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_646_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1(void){
_start:
{
lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_647_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__0);
v___x_648_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_648_, 0, v___x_647_);
return v___x_648_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_649_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1);
v___x_650_ = lean_unsigned_to_nat(0u);
v___x_651_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_651_, 0, v___x_650_);
lean_ctor_set(v___x_651_, 1, v___x_650_);
lean_ctor_set(v___x_651_, 2, v___x_650_);
lean_ctor_set(v___x_651_, 3, v___x_650_);
lean_ctor_set(v___x_651_, 4, v___x_649_);
lean_ctor_set(v___x_651_, 5, v___x_649_);
lean_ctor_set(v___x_651_, 6, v___x_649_);
lean_ctor_set(v___x_651_, 7, v___x_649_);
lean_ctor_set(v___x_651_, 8, v___x_649_);
lean_ctor_set(v___x_651_, 9, v___x_649_);
return v___x_651_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_unsigned_to_nat(32u);
v___x_653_ = lean_mk_empty_array_with_capacity(v___x_652_);
v___x_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
return v___x_654_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4(void){
_start:
{
size_t v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_655_ = ((size_t)5ULL);
v___x_656_ = lean_unsigned_to_nat(0u);
v___x_657_ = lean_unsigned_to_nat(32u);
v___x_658_ = lean_mk_empty_array_with_capacity(v___x_657_);
v___x_659_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__3);
v___x_660_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_660_, 0, v___x_659_);
lean_ctor_set(v___x_660_, 1, v___x_658_);
lean_ctor_set(v___x_660_, 2, v___x_656_);
lean_ctor_set(v___x_660_, 3, v___x_656_);
lean_ctor_set_usize(v___x_660_, 4, v___x_655_);
return v___x_660_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5(void){
_start:
{
lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_661_ = lean_box(1);
v___x_662_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__4);
v___x_663_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__1);
v___x_664_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_664_, 0, v___x_663_);
lean_ctor_set(v___x_664_, 1, v___x_662_);
lean_ctor_set(v___x_664_, 2, v___x_661_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2(lean_object* v_msgData_665_, lean_object* v___y_666_, lean_object* v___y_667_){
_start:
{
lean_object* v___x_669_; lean_object* v_env_670_; lean_object* v_options_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v___x_669_ = lean_st_ref_get(v___y_667_);
v_env_670_ = lean_ctor_get(v___x_669_, 0);
lean_inc_ref(v_env_670_);
lean_dec(v___x_669_);
v_options_671_ = lean_ctor_get(v___y_666_, 2);
v___x_672_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__2);
v___x_673_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___closed__5);
lean_inc_ref(v_options_671_);
v___x_674_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_674_, 0, v_env_670_);
lean_ctor_set(v___x_674_, 1, v___x_672_);
lean_ctor_set(v___x_674_, 2, v___x_673_);
lean_ctor_set(v___x_674_, 3, v_options_671_);
v___x_675_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_675_, 0, v___x_674_);
lean_ctor_set(v___x_675_, 1, v_msgData_665_);
v___x_676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2___boxed(lean_object* v_msgData_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2(v_msgData_677_, v___y_678_, v___y_679_);
lean_dec(v___y_679_);
lean_dec_ref(v___y_678_);
return v_res_681_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(lean_object* v_msg_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
lean_object* v_ref_686_; lean_object* v___x_687_; lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_696_; 
v_ref_686_ = lean_ctor_get(v___y_683_, 5);
v___x_687_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2_spec__2(v_msg_682_, v___y_683_, v___y_684_);
v_a_688_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_696_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_696_ == 0)
{
v___x_690_ = v___x_687_;
v_isShared_691_ = v_isSharedCheck_696_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_687_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_696_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_692_; lean_object* v___x_694_; 
lean_inc(v_ref_686_);
v___x_692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_692_, 0, v_ref_686_);
lean_ctor_set(v___x_692_, 1, v_a_688_);
if (v_isShared_691_ == 0)
{
lean_ctor_set_tag(v___x_690_, 1);
lean_ctor_set(v___x_690_, 0, v___x_692_);
v___x_694_ = v___x_690_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_692_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
return v___x_694_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_msg_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_){
_start:
{
lean_object* v_res_701_; 
v_res_701_ = lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(v_msg_697_, v___y_698_, v___y_699_);
lean_dec(v___y_699_);
lean_dec_ref(v___y_698_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0(size_t v_sz_702_, size_t v_i_703_, lean_object* v_bs_704_){
_start:
{
uint8_t v___x_705_; 
v___x_705_ = lean_usize_dec_lt(v_i_703_, v_sz_702_);
if (v___x_705_ == 0)
{
lean_object* v___x_706_; 
v___x_706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_706_, 0, v_bs_704_);
return v___x_706_;
}
else
{
lean_object* v_v_707_; lean_object* v___x_708_; lean_object* v_bs_x27_709_; size_t v___x_710_; size_t v___x_711_; lean_object* v___x_712_; 
v_v_707_ = lean_array_uget(v_bs_704_, v_i_703_);
v___x_708_ = lean_unsigned_to_nat(0u);
v_bs_x27_709_ = lean_array_uset(v_bs_704_, v_i_703_, v___x_708_);
v___x_710_ = ((size_t)1ULL);
v___x_711_ = lean_usize_add(v_i_703_, v___x_710_);
v___x_712_ = lean_array_uset(v_bs_x27_709_, v_i_703_, v_v_707_);
v_i_703_ = v___x_711_;
v_bs_704_ = v___x_712_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0___boxed(lean_object* v_sz_714_, lean_object* v_i_715_, lean_object* v_bs_716_){
_start:
{
size_t v_sz_boxed_717_; size_t v_i_boxed_718_; lean_object* v_res_719_; 
v_sz_boxed_717_ = lean_unbox_usize(v_sz_714_);
lean_dec(v_sz_714_);
v_i_boxed_718_ = lean_unbox_usize(v_i_715_);
lean_dec(v_i_715_);
v_res_719_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0(v_sz_boxed_717_, v_i_boxed_718_, v_bs_716_);
return v_res_719_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_720_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_721_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_722_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_722_, 0, v___x_721_);
return v___x_722_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; 
v___x_723_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_723_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
return v___x_724_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_726_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_727_ = l_Lean_stringToMessageData(v___x_726_);
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(lean_object* v___x_728_, lean_object* v___x_729_, lean_object* v___x_730_, lean_object* v___x_731_, lean_object* v___x_732_, lean_object* v_decl_733_, lean_object* v_stx_734_, uint8_t v_kind_735_, lean_object* v___y_736_, lean_object* v___y_737_){
_start:
{
lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___y_796_; lean_object* v___y_797_; uint8_t v___y_798_; lean_object* v___y_853_; lean_object* v___y_854_; lean_object* v___y_855_; uint8_t v___y_856_; uint8_t v___x_958_; uint8_t v___x_959_; 
v___x_958_ = 0;
v___x_959_ = l_Lean_instBEqAttributeKind_beq(v_kind_735_, v___x_958_);
if (v___x_959_ == 0)
{
lean_object* v___x_960_; lean_object* v___x_961_; 
lean_dec(v_stx_734_);
lean_dec(v_decl_733_);
lean_dec_ref(v___x_732_);
lean_dec_ref(v___x_731_);
lean_dec_ref(v___x_730_);
lean_dec(v___x_729_);
v___x_960_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__4_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_961_ = lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(v___x_960_, v___y_736_, v___y_737_);
return v___x_961_;
}
else
{
goto v___jp_911_;
}
v___jp_739_:
{
lean_object* v___x_742_; lean_object* v_env_743_; lean_object* v_options_744_; lean_object* v_ref_745_; lean_object* v___x_746_; lean_object* v___x_747_; 
v___x_742_ = lean_st_ref_get(v___y_741_);
v_env_743_ = lean_ctor_get(v___x_742_, 0);
lean_inc_ref(v_env_743_);
lean_dec(v___x_742_);
v_options_744_ = lean_ctor_get(v___y_740_, 2);
v_ref_745_ = lean_ctor_get(v___y_740_, 5);
lean_inc_ref(v_options_744_);
v___x_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_746_, 0, v_env_743_);
lean_ctor_set(v___x_746_, 1, v_options_744_);
lean_inc(v_decl_733_);
v___x_747_ = lp_batteries_Batteries_CodeAction_mkTacticCodeAction(v_decl_733_, v___x_746_);
lean_dec_ref_known(v___x_746_, 2);
if (lean_obj_tag(v___x_747_) == 0)
{
lean_object* v_a_748_; lean_object* v___x_750_; uint8_t v_isShared_751_; uint8_t v_isSharedCheck_782_; 
v_a_748_ = lean_ctor_get(v___x_747_, 0);
v_isSharedCheck_782_ = !lean_is_exclusive(v___x_747_);
if (v_isSharedCheck_782_ == 0)
{
v___x_750_ = v___x_747_;
v_isShared_751_ = v_isSharedCheck_782_;
goto v_resetjp_749_;
}
else
{
lean_inc(v_a_748_);
lean_dec(v___x_747_);
v___x_750_ = lean_box(0);
v_isShared_751_ = v_isSharedCheck_782_;
goto v_resetjp_749_;
}
v_resetjp_749_:
{
lean_object* v___x_752_; lean_object* v_env_753_; lean_object* v_nextMacroScope_754_; lean_object* v_ngen_755_; lean_object* v_auxDeclNGen_756_; lean_object* v_traceState_757_; lean_object* v_messages_758_; lean_object* v_infoState_759_; lean_object* v_snapshotTasks_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_780_; 
v___x_752_ = lean_st_ref_take(v___y_741_);
v_env_753_ = lean_ctor_get(v___x_752_, 0);
v_nextMacroScope_754_ = lean_ctor_get(v___x_752_, 1);
v_ngen_755_ = lean_ctor_get(v___x_752_, 2);
v_auxDeclNGen_756_ = lean_ctor_get(v___x_752_, 3);
v_traceState_757_ = lean_ctor_get(v___x_752_, 4);
v_messages_758_ = lean_ctor_get(v___x_752_, 6);
v_infoState_759_ = lean_ctor_get(v___x_752_, 7);
v_snapshotTasks_760_ = lean_ctor_get(v___x_752_, 8);
v_isSharedCheck_780_ = !lean_is_exclusive(v___x_752_);
if (v_isSharedCheck_780_ == 0)
{
lean_object* v_unused_781_; 
v_unused_781_ = lean_ctor_get(v___x_752_, 5);
lean_dec(v_unused_781_);
v___x_762_ = v___x_752_;
v_isShared_763_ = v_isSharedCheck_780_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_snapshotTasks_760_);
lean_inc(v_infoState_759_);
lean_inc(v_messages_758_);
lean_inc(v_traceState_757_);
lean_inc(v_auxDeclNGen_756_);
lean_inc(v_ngen_755_);
lean_inc(v_nextMacroScope_754_);
lean_inc(v_env_753_);
lean_dec(v___x_752_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_780_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_764_; lean_object* v_toEnvExtension_765_; lean_object* v_asyncMode_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_773_; 
v___x_764_ = lp_batteries_Batteries_CodeAction_tacticCodeActionExt;
v_toEnvExtension_765_ = lean_ctor_get(v___x_764_, 0);
v_asyncMode_766_ = lean_ctor_get(v_toEnvExtension_765_, 2);
v___x_767_ = lean_mk_empty_array_with_capacity(v___x_728_);
v___x_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_768_, 0, v_decl_733_);
lean_ctor_set(v___x_768_, 1, v___x_767_);
v___x_769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
lean_ctor_set(v___x_769_, 1, v_a_748_);
v___x_770_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_764_, v_env_753_, v___x_769_, v_asyncMode_766_, v___x_729_);
v___x_771_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
if (v_isShared_763_ == 0)
{
lean_ctor_set(v___x_762_, 5, v___x_771_);
lean_ctor_set(v___x_762_, 0, v___x_770_);
v___x_773_ = v___x_762_;
goto v_reusejp_772_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v___x_770_);
lean_ctor_set(v_reuseFailAlloc_779_, 1, v_nextMacroScope_754_);
lean_ctor_set(v_reuseFailAlloc_779_, 2, v_ngen_755_);
lean_ctor_set(v_reuseFailAlloc_779_, 3, v_auxDeclNGen_756_);
lean_ctor_set(v_reuseFailAlloc_779_, 4, v_traceState_757_);
lean_ctor_set(v_reuseFailAlloc_779_, 5, v___x_771_);
lean_ctor_set(v_reuseFailAlloc_779_, 6, v_messages_758_);
lean_ctor_set(v_reuseFailAlloc_779_, 7, v_infoState_759_);
lean_ctor_set(v_reuseFailAlloc_779_, 8, v_snapshotTasks_760_);
v___x_773_ = v_reuseFailAlloc_779_;
goto v_reusejp_772_;
}
v_reusejp_772_:
{
lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_777_; 
v___x_774_ = lean_st_ref_set(v___y_741_, v___x_773_);
v___x_775_ = lean_box(0);
if (v_isShared_751_ == 0)
{
lean_ctor_set(v___x_750_, 0, v___x_775_);
v___x_777_ = v___x_750_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_775_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
}
}
}
else
{
lean_object* v_a_783_; lean_object* v___x_785_; uint8_t v_isShared_786_; uint8_t v_isSharedCheck_794_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v_a_783_ = lean_ctor_get(v___x_747_, 0);
v_isSharedCheck_794_ = !lean_is_exclusive(v___x_747_);
if (v_isSharedCheck_794_ == 0)
{
v___x_785_ = v___x_747_;
v_isShared_786_ = v_isSharedCheck_794_;
goto v_resetjp_784_;
}
else
{
lean_inc(v_a_783_);
lean_dec(v___x_747_);
v___x_785_ = lean_box(0);
v_isShared_786_ = v_isSharedCheck_794_;
goto v_resetjp_784_;
}
v_resetjp_784_:
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_792_; 
v___x_787_ = lean_io_error_to_string(v_a_783_);
v___x_788_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_788_, 0, v___x_787_);
v___x_789_ = l_Lean_MessageData_ofFormat(v___x_788_);
lean_inc(v_ref_745_);
v___x_790_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_790_, 0, v_ref_745_);
lean_ctor_set(v___x_790_, 1, v___x_789_);
if (v_isShared_786_ == 0)
{
lean_ctor_set(v___x_785_, 0, v___x_790_);
v___x_792_ = v___x_785_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v___x_790_);
v___x_792_ = v_reuseFailAlloc_793_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
return v___x_792_;
}
}
}
}
v___jp_795_:
{
if (v___y_798_ == 0)
{
lean_object* v___x_799_; lean_object* v_env_800_; lean_object* v_options_801_; lean_object* v_ref_802_; lean_object* v___x_803_; lean_object* v___x_804_; 
v___x_799_ = lean_st_ref_get(v___y_797_);
v_env_800_ = lean_ctor_get(v___x_799_, 0);
lean_inc_ref(v_env_800_);
lean_dec(v___x_799_);
v_options_801_ = lean_ctor_get(v___y_796_, 2);
v_ref_802_ = lean_ctor_get(v___y_796_, 5);
lean_inc_ref(v_options_801_);
v___x_803_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_803_, 0, v_env_800_);
lean_ctor_set(v___x_803_, 1, v_options_801_);
lean_inc(v_decl_733_);
v___x_804_ = lp_batteries_Batteries_CodeAction_mkTacticSeqCodeAction(v_decl_733_, v___x_803_);
lean_dec_ref_known(v___x_803_, 2);
if (lean_obj_tag(v___x_804_) == 0)
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_837_; 
v_a_805_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_837_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_837_ == 0)
{
v___x_807_ = v___x_804_;
v_isShared_808_ = v_isSharedCheck_837_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_804_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_837_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_809_; lean_object* v_env_810_; lean_object* v_nextMacroScope_811_; lean_object* v_ngen_812_; lean_object* v_auxDeclNGen_813_; lean_object* v_traceState_814_; lean_object* v_messages_815_; lean_object* v_infoState_816_; lean_object* v_snapshotTasks_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_835_; 
v___x_809_ = lean_st_ref_take(v___y_797_);
v_env_810_ = lean_ctor_get(v___x_809_, 0);
v_nextMacroScope_811_ = lean_ctor_get(v___x_809_, 1);
v_ngen_812_ = lean_ctor_get(v___x_809_, 2);
v_auxDeclNGen_813_ = lean_ctor_get(v___x_809_, 3);
v_traceState_814_ = lean_ctor_get(v___x_809_, 4);
v_messages_815_ = lean_ctor_get(v___x_809_, 6);
v_infoState_816_ = lean_ctor_get(v___x_809_, 7);
v_snapshotTasks_817_ = lean_ctor_get(v___x_809_, 8);
v_isSharedCheck_835_ = !lean_is_exclusive(v___x_809_);
if (v_isSharedCheck_835_ == 0)
{
lean_object* v_unused_836_; 
v_unused_836_ = lean_ctor_get(v___x_809_, 5);
lean_dec(v_unused_836_);
v___x_819_ = v___x_809_;
v_isShared_820_ = v_isSharedCheck_835_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_snapshotTasks_817_);
lean_inc(v_infoState_816_);
lean_inc(v_messages_815_);
lean_inc(v_traceState_814_);
lean_inc(v_auxDeclNGen_813_);
lean_inc(v_ngen_812_);
lean_inc(v_nextMacroScope_811_);
lean_inc(v_env_810_);
lean_dec(v___x_809_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_835_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v___x_821_; lean_object* v_toEnvExtension_822_; lean_object* v_asyncMode_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_828_; 
v___x_821_ = lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt;
v_toEnvExtension_822_ = lean_ctor_get(v___x_821_, 0);
v_asyncMode_823_ = lean_ctor_get(v_toEnvExtension_822_, 2);
v___x_824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_824_, 0, v_decl_733_);
lean_ctor_set(v___x_824_, 1, v_a_805_);
v___x_825_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_821_, v_env_810_, v___x_824_, v_asyncMode_823_, v___x_729_);
v___x_826_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
if (v_isShared_820_ == 0)
{
lean_ctor_set(v___x_819_, 5, v___x_826_);
lean_ctor_set(v___x_819_, 0, v___x_825_);
v___x_828_ = v___x_819_;
goto v_reusejp_827_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_825_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v_nextMacroScope_811_);
lean_ctor_set(v_reuseFailAlloc_834_, 2, v_ngen_812_);
lean_ctor_set(v_reuseFailAlloc_834_, 3, v_auxDeclNGen_813_);
lean_ctor_set(v_reuseFailAlloc_834_, 4, v_traceState_814_);
lean_ctor_set(v_reuseFailAlloc_834_, 5, v___x_826_);
lean_ctor_set(v_reuseFailAlloc_834_, 6, v_messages_815_);
lean_ctor_set(v_reuseFailAlloc_834_, 7, v_infoState_816_);
lean_ctor_set(v_reuseFailAlloc_834_, 8, v_snapshotTasks_817_);
v___x_828_ = v_reuseFailAlloc_834_;
goto v_reusejp_827_;
}
v_reusejp_827_:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_832_; 
v___x_829_ = lean_st_ref_set(v___y_797_, v___x_828_);
v___x_830_ = lean_box(0);
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 0, v___x_830_);
v___x_832_ = v___x_807_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v___x_830_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
}
}
}
else
{
lean_object* v_a_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_849_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v_a_838_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_849_ == 0)
{
v___x_840_ = v___x_804_;
v_isShared_841_ = v_isSharedCheck_849_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_a_838_);
lean_dec(v___x_804_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_849_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_842_ = lean_io_error_to_string(v_a_838_);
v___x_843_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_843_, 0, v___x_842_);
v___x_844_ = l_Lean_MessageData_ofFormat(v___x_843_);
lean_inc(v_ref_802_);
v___x_845_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_845_, 0, v_ref_802_);
lean_ctor_set(v___x_845_, 1, v___x_844_);
if (v_isShared_841_ == 0)
{
lean_ctor_set(v___x_840_, 0, v___x_845_);
v___x_847_ = v___x_840_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v___x_845_);
v___x_847_ = v_reuseFailAlloc_848_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
return v___x_847_;
}
}
}
}
else
{
lean_object* v___x_850_; lean_object* v___x_851_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v___x_850_ = lean_box(0);
v___x_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_851_, 0, v___x_850_);
return v___x_851_;
}
}
v___jp_852_:
{
if (v___y_856_ == 0)
{
lean_object* v___x_857_; lean_object* v_env_858_; lean_object* v_options_859_; lean_object* v_ref_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_857_ = lean_st_ref_get(v___y_855_);
v_env_858_ = lean_ctor_get(v___x_857_, 0);
lean_inc_ref(v_env_858_);
lean_dec(v___x_857_);
v_options_859_ = lean_ctor_get(v___y_854_, 2);
v_ref_860_ = lean_ctor_get(v___y_854_, 5);
lean_inc_ref(v_options_859_);
v___x_861_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_861_, 0, v_env_858_);
lean_ctor_set(v___x_861_, 1, v_options_859_);
lean_inc(v_decl_733_);
v___x_862_ = lp_batteries_Batteries_CodeAction_mkTacticCodeAction(v_decl_733_, v___x_861_);
lean_dec_ref_known(v___x_861_, 2);
if (lean_obj_tag(v___x_862_) == 0)
{
lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_896_; 
v_a_863_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_896_ == 0)
{
v___x_865_ = v___x_862_;
v_isShared_866_ = v_isSharedCheck_896_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___x_862_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_896_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v___x_867_; lean_object* v_env_868_; lean_object* v_nextMacroScope_869_; lean_object* v_ngen_870_; lean_object* v_auxDeclNGen_871_; lean_object* v_traceState_872_; lean_object* v_messages_873_; lean_object* v_infoState_874_; lean_object* v_snapshotTasks_875_; lean_object* v___x_877_; uint8_t v_isShared_878_; uint8_t v_isSharedCheck_894_; 
v___x_867_ = lean_st_ref_take(v___y_855_);
v_env_868_ = lean_ctor_get(v___x_867_, 0);
v_nextMacroScope_869_ = lean_ctor_get(v___x_867_, 1);
v_ngen_870_ = lean_ctor_get(v___x_867_, 2);
v_auxDeclNGen_871_ = lean_ctor_get(v___x_867_, 3);
v_traceState_872_ = lean_ctor_get(v___x_867_, 4);
v_messages_873_ = lean_ctor_get(v___x_867_, 6);
v_infoState_874_ = lean_ctor_get(v___x_867_, 7);
v_snapshotTasks_875_ = lean_ctor_get(v___x_867_, 8);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_894_ == 0)
{
lean_object* v_unused_895_; 
v_unused_895_ = lean_ctor_get(v___x_867_, 5);
lean_dec(v_unused_895_);
v___x_877_ = v___x_867_;
v_isShared_878_ = v_isSharedCheck_894_;
goto v_resetjp_876_;
}
else
{
lean_inc(v_snapshotTasks_875_);
lean_inc(v_infoState_874_);
lean_inc(v_messages_873_);
lean_inc(v_traceState_872_);
lean_inc(v_auxDeclNGen_871_);
lean_inc(v_ngen_870_);
lean_inc(v_nextMacroScope_869_);
lean_inc(v_env_868_);
lean_dec(v___x_867_);
v___x_877_ = lean_box(0);
v_isShared_878_ = v_isSharedCheck_894_;
goto v_resetjp_876_;
}
v_resetjp_876_:
{
lean_object* v___x_879_; lean_object* v_toEnvExtension_880_; lean_object* v_asyncMode_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_887_; 
v___x_879_ = lp_batteries_Batteries_CodeAction_tacticCodeActionExt;
v_toEnvExtension_880_ = lean_ctor_get(v___x_879_, 0);
v_asyncMode_881_ = lean_ctor_get(v_toEnvExtension_880_, 2);
v___x_882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_882_, 0, v_decl_733_);
lean_ctor_set(v___x_882_, 1, v___y_853_);
v___x_883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_883_, 0, v___x_882_);
lean_ctor_set(v___x_883_, 1, v_a_863_);
v___x_884_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_879_, v_env_868_, v___x_883_, v_asyncMode_881_, v___x_729_);
v___x_885_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
if (v_isShared_878_ == 0)
{
lean_ctor_set(v___x_877_, 5, v___x_885_);
lean_ctor_set(v___x_877_, 0, v___x_884_);
v___x_887_ = v___x_877_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_884_);
lean_ctor_set(v_reuseFailAlloc_893_, 1, v_nextMacroScope_869_);
lean_ctor_set(v_reuseFailAlloc_893_, 2, v_ngen_870_);
lean_ctor_set(v_reuseFailAlloc_893_, 3, v_auxDeclNGen_871_);
lean_ctor_set(v_reuseFailAlloc_893_, 4, v_traceState_872_);
lean_ctor_set(v_reuseFailAlloc_893_, 5, v___x_885_);
lean_ctor_set(v_reuseFailAlloc_893_, 6, v_messages_873_);
lean_ctor_set(v_reuseFailAlloc_893_, 7, v_infoState_874_);
lean_ctor_set(v_reuseFailAlloc_893_, 8, v_snapshotTasks_875_);
v___x_887_ = v_reuseFailAlloc_893_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_891_; 
v___x_888_ = lean_st_ref_set(v___y_855_, v___x_887_);
v___x_889_ = lean_box(0);
if (v_isShared_866_ == 0)
{
lean_ctor_set(v___x_865_, 0, v___x_889_);
v___x_891_ = v___x_865_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_889_);
v___x_891_ = v_reuseFailAlloc_892_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
return v___x_891_;
}
}
}
}
}
else
{
lean_object* v_a_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_908_; 
lean_dec_ref(v___y_853_);
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v_a_897_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_908_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_908_ == 0)
{
v___x_899_ = v___x_862_;
v_isShared_900_ = v_isSharedCheck_908_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_a_897_);
lean_dec(v___x_862_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_908_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_906_; 
v___x_901_ = lean_io_error_to_string(v_a_897_);
v___x_902_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_902_, 0, v___x_901_);
v___x_903_ = l_Lean_MessageData_ofFormat(v___x_902_);
lean_inc(v_ref_860_);
v___x_904_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_904_, 0, v_ref_860_);
lean_ctor_set(v___x_904_, 1, v___x_903_);
if (v_isShared_900_ == 0)
{
lean_ctor_set(v___x_899_, 0, v___x_904_);
v___x_906_ = v___x_899_;
goto v_reusejp_905_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v___x_904_);
v___x_906_ = v_reuseFailAlloc_907_;
goto v_reusejp_905_;
}
v_reusejp_905_:
{
return v___x_906_;
}
}
}
}
else
{
lean_object* v___x_909_; lean_object* v___x_910_; 
lean_dec_ref(v___y_853_);
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v___x_909_ = lean_box(0);
v___x_910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_910_, 0, v___x_909_);
return v___x_910_;
}
}
v___jp_911_:
{
lean_object* v___x_912_; uint8_t v___x_913_; 
v___x_912_ = l_Lean_Name_mkStr3(v___x_730_, v___x_731_, v___x_732_);
lean_inc(v_stx_734_);
v___x_913_ = l_Lean_Syntax_isOfKind(v_stx_734_, v___x_912_);
lean_dec(v___x_912_);
if (v___x_913_ == 0)
{
lean_object* v___x_914_; lean_object* v___x_915_; 
lean_dec(v_stx_734_);
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v___x_914_ = lean_box(0);
v___x_915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_915_, 0, v___x_914_);
return v___x_915_;
}
else
{
lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; uint8_t v___x_919_; 
v___x_916_ = lean_unsigned_to_nat(1u);
v___x_917_ = l_Lean_Syntax_getArg(v_stx_734_, v___x_916_);
lean_dec(v_stx_734_);
v___x_918_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tactic__code__action___closed__9));
lean_inc(v___x_917_);
v___x_919_ = l_Lean_Syntax_isOfKind(v___x_917_, v___x_918_);
if (v___x_919_ == 0)
{
lean_object* v___x_920_; size_t v_sz_921_; size_t v___x_922_; lean_object* v___x_923_; 
v___x_920_ = l_Lean_Syntax_getArgs(v___x_917_);
lean_dec(v___x_917_);
v_sz_921_ = lean_array_size(v___x_920_);
v___x_922_ = ((size_t)0ULL);
v___x_923_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__0(v_sz_921_, v___x_922_, v___x_920_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v___x_924_; lean_object* v___x_925_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v___x_924_ = lean_box(0);
v___x_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
return v___x_925_;
}
else
{
lean_object* v_val_926_; lean_object* v___x_927_; uint8_t v___x_928_; 
v_val_926_ = lean_ctor_get(v___x_923_, 0);
lean_inc(v_val_926_);
lean_dec_ref_known(v___x_923_, 1);
v___x_927_ = lean_array_get_size(v_val_926_);
v___x_928_ = lean_nat_dec_eq(v___x_927_, v___x_728_);
if (v___x_928_ == 0)
{
size_t v_sz_929_; lean_object* v___x_930_; 
v_sz_929_ = lean_array_size(v_val_926_);
v___x_930_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__1(v_sz_929_, v___x_922_, v_val_926_, v___y_736_, v___y_737_);
if (lean_obj_tag(v___x_930_) == 0)
{
lean_object* v_a_931_; lean_object* v___x_932_; lean_object* v_env_933_; lean_object* v___x_934_; 
v_a_931_ = lean_ctor_get(v___x_930_, 0);
lean_inc(v_a_931_);
lean_dec_ref_known(v___x_930_, 1);
v___x_932_ = lean_st_ref_get(v___y_737_);
v_env_933_ = lean_ctor_get(v___x_932_, 0);
lean_inc_ref(v_env_933_);
lean_dec(v___x_932_);
lean_inc(v_decl_733_);
v___x_934_ = lean_decl_get_sorry_dep(v_env_933_, v_decl_733_);
if (lean_obj_tag(v___x_934_) == 0)
{
v___y_853_ = v_a_931_;
v___y_854_ = v___y_736_;
v___y_855_ = v___y_737_;
v___y_856_ = v___x_928_;
goto v___jp_852_;
}
else
{
lean_dec_ref_known(v___x_934_, 1);
v___y_853_ = v_a_931_;
v___y_854_ = v___y_736_;
v___y_855_ = v___y_737_;
v___y_856_ = v___x_913_;
goto v___jp_852_;
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v_a_935_ = lean_ctor_get(v___x_930_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_930_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_930_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_930_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
lean_object* v___x_940_; 
if (v_isShared_938_ == 0)
{
v___x_940_ = v___x_937_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_a_935_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
}
}
else
{
lean_object* v___x_943_; lean_object* v_env_944_; lean_object* v___x_945_; 
lean_dec(v_val_926_);
v___x_943_ = lean_st_ref_get(v___y_737_);
v_env_944_ = lean_ctor_get(v___x_943_, 0);
lean_inc_ref(v_env_944_);
lean_dec(v___x_943_);
lean_inc(v_decl_733_);
v___x_945_ = lean_decl_get_sorry_dep(v_env_944_, v_decl_733_);
if (lean_obj_tag(v___x_945_) == 0)
{
v___y_796_ = v___y_736_;
v___y_797_ = v___y_737_;
v___y_798_ = v___x_919_;
goto v___jp_795_;
}
else
{
lean_dec_ref_known(v___x_945_, 1);
v___y_796_ = v___y_736_;
v___y_797_ = v___y_737_;
v___y_798_ = v___x_928_;
goto v___jp_795_;
}
}
}
}
else
{
lean_object* v___x_946_; lean_object* v_env_947_; lean_object* v___x_948_; 
lean_dec(v___x_917_);
v___x_946_ = lean_st_ref_get(v___y_737_);
v_env_947_ = lean_ctor_get(v___x_946_, 0);
lean_inc_ref(v_env_947_);
lean_dec(v___x_946_);
lean_inc(v_decl_733_);
v___x_948_ = lean_decl_get_sorry_dep(v_env_947_, v_decl_733_);
if (lean_obj_tag(v___x_948_) == 0)
{
v___y_740_ = v___y_736_;
v___y_741_ = v___y_737_;
goto v___jp_739_;
}
else
{
lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_956_; 
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_948_);
if (v_isSharedCheck_956_ == 0)
{
lean_object* v_unused_957_; 
v_unused_957_ = lean_ctor_get(v___x_948_, 0);
lean_dec(v_unused_957_);
v___x_950_ = v___x_948_;
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
else
{
lean_dec(v___x_948_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
if (v___x_919_ == 0)
{
lean_del_object(v___x_950_);
v___y_740_ = v___y_736_;
v___y_741_ = v___y_737_;
goto v___jp_739_;
}
else
{
lean_object* v___x_952_; lean_object* v___x_954_; 
lean_dec(v_decl_733_);
lean_dec(v___x_729_);
v___x_952_ = lean_box(0);
if (v_isShared_951_ == 0)
{
lean_ctor_set_tag(v___x_950_, 0);
lean_ctor_set(v___x_950_, 0, v___x_952_);
v___x_954_ = v___x_950_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v___x_952_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object* v___x_962_, lean_object* v___x_963_, lean_object* v___x_964_, lean_object* v___x_965_, lean_object* v___x_966_, lean_object* v_decl_967_, lean_object* v_stx_968_, lean_object* v_kind_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_){
_start:
{
uint8_t v_kind_boxed_973_; lean_object* v_res_974_; 
v_kind_boxed_973_ = lean_unbox(v_kind_969_);
v_res_974_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(v___x_962_, v___x_963_, v___x_964_, v___x_965_, v___x_966_, v_decl_967_, v_stx_968_, v_kind_boxed_973_, v___y_970_, v___y_971_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___x_962_);
return v_res_974_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_976_; lean_object* v___x_977_; 
v___x_976_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__0_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_977_ = l_Lean_stringToMessageData(v___x_976_);
return v___x_977_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__2_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_980_ = l_Lean_stringToMessageData(v___x_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(lean_object* v___x_981_, lean_object* v_decl_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_986_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_987_ = l_Lean_MessageData_ofName(v___x_981_);
v___x_988_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_988_, 0, v___x_986_);
lean_ctor_set(v___x_988_, 1, v___x_987_);
v___x_989_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1___closed__3_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_990_, 0, v___x_988_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
v___x_991_ = lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(v___x_990_, v___y_983_, v___y_984_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object* v___x_992_, lean_object* v_decl_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___lam__1_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(v___x_992_, v_decl_993_, v___y_994_, v___y_995_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
lean_dec(v_decl_993_);
return v_res_997_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v___x_1038_ = lean_unsigned_to_nat(2443277903u);
v___x_1039_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__15_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1040_ = l_Lean_Name_num___override(v___x_1039_, v___x_1038_);
return v___x_1040_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1042_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__17_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1043_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__16_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1044_ = l_Lean_Name_str___override(v___x_1043_, v___x_1042_);
return v___x_1044_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1046_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__19_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1047_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__18_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1048_ = l_Lean_Name_str___override(v___x_1047_, v___x_1046_);
return v___x_1048_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; 
v___x_1049_ = lean_unsigned_to_nat(2u);
v___x_1050_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__20_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1051_ = l_Lean_Name_num___override(v___x_1050_, v___x_1049_);
return v___x_1051_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; 
v___x_1063_ = 1;
v___x_1064_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__25_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1065_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__23_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1066_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__21_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1067_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1067_, 0, v___x_1066_);
lean_ctor_set(v___x_1067_, 1, v___x_1065_);
lean_ctor_set(v___x_1067_, 2, v___x_1064_);
lean_ctor_set_uint8(v___x_1067_, sizeof(void*)*3, v___x_1063_);
return v___x_1067_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1068_; lean_object* v___f_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___f_1068_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__24_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___f_1069_ = ((lean_object*)(lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__22_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_));
v___x_1070_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__26_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1071_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1071_, 0, v___x_1070_);
lean_ctor_set(v___x_1071_, 1, v___f_1069_);
lean_ctor_set(v___x_1071_, 2, v___f_1068_);
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___x_1073_ = lean_obj_once(&lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_, &lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn___closed__27_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_);
v___x_1074_ = l_Lean_registerBuiltinAttribute(v___x_1073_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2____boxed(lean_object* v_a_1075_){
_start:
{
lean_object* v_res_1076_; 
v_res_1076_ = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_();
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_1077_, lean_object* v_msg_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___redArg(v_msg_1078_, v___y_1079_, v___y_1080_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_1083_, lean_object* v_msg_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v_res_1088_; 
v_res_1088_ = lp_batteries_Lean_throwError___at___00__private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2__spec__2(v_00_u03b1_1083_, v_msg_1084_, v___y_1085_, v___y_1086_);
lean_dec(v___y_1086_);
lean_dec_ref(v___y_1085_);
return v_res_1088_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_CodeActions_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_CodeActions_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_3685358032____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2519722026____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_CodeAction_tacticCodeActionExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_CodeAction_tacticCodeActionExt);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_CodeAction_Attr_0__Batteries_CodeAction_initFn_00___x40_Batteries_CodeAction_Attr_2443277903____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin) {
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
lean_object* initialize_Lean_Server_CodeActions_Basic(uint8_t builtin);
lean_object* initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_CodeActions_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_CodeAction_Attr(builtin);
}
#ifdef __cplusplus
}
#endif
