// Lean compiler output
// Module: Batteries.Linter.UnnecessarySeqFocus
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Batteries.Lean.AttributeExtra public meta import Lean.Linter.Basic
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
lean_object* l_instOrdInt___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadST(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Syntax_instHashableRange_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_instInhabitedInfoTree_default;
lean_object* l_Lean_PersistentArray_get_x21___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_batteries_Lean_registerTagAttributeExtra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t lp_batteries_Lean_TagAttributeExtra_hasTag(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_instOrdNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_lexOrd___redArg(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Syntax_instBEqRange_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_instHashableRange_hash___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_runST___redArg(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "unnecessarySeqFocus"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(130, 69, 154, 29, 21, 22, 66, 38)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "enable the 'unnecessary <;>' linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(200, 5, 203, 188, 211, 209, 34, 248)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(53, 186, 4, 149, 36, 28, 231, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(25, 203, 170, 215, 231, 231, 37, 82)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_linter_unnecessarySeqFocus;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "multigoal"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__1_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(148, 138, 12, 111, 202, 12, 113, 4)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "this tactic acts on multiple goals"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticNext_=>_"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(90, 21, 53, 2, 17, 158, 67, 66)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__9_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "allGoals"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__9_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__9_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__9_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(105, 66, 138, 83, 251, 171, 29, 196)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__11_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "anyGoals"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__11_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__11_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__11_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(168, 19, 163, 3, 232, 106, 175, 32)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__13_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "case"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__13_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__13_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__13_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(216, 244, 120, 128, 139, 198, 139, 51)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__15_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "case'"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__15_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__15_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__15_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(134, 21, 185, 205, 238, 88, 7, 106)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__18_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "convNext__=>_"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__18_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__18_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__18_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(141, 255, 234, 0, 142, 69, 158, 51)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__9_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(80, 55, 182, 70, 128, 26, 115, 15)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__11_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 41, 143, 75, 238, 57, 26, 246)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__13_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(241, 23, 91, 126, 214, 77, 25, 163)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__15_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(151, 157, 98, 160, 189, 128, 94, 31)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__24_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rotateLeft"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__24_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__24_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__24_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(63, 201, 198, 124, 10, 198, 250, 123)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__26_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rotateRight"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__26_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__26_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__26_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(98, 177, 153, 112, 69, 167, 66, 136)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__28_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__28_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__28_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__28_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(151, 147, 62, 103, 130, 224, 84, 63)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__30_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticStop_"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__30_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__30_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__30_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 187, 217, 116, 133, 153, 2, 108)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__32_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__31_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__32_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__32_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__33_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__29_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__32_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__33_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__33_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__34_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__27_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__33_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__34_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__34_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__35_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__25_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__34_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__35_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__35_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__36_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__23_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__35_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__36_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__36_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__37_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__22_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__36_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__37_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__37_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__38_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__21_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__37_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__38_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__38_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__39_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__20_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__38_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__39_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__39_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__40_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__19_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__39_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__40_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__40_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__41_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__16_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__40_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__41_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__41_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__42_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__14_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__41_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__42_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__42_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__43_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__12_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__42_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__43_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__43_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__44_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__10_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__43_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__44_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__44_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__45_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__8_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__44_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__45_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__45_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__46_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "UnnecessarySeqFocus"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__46_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__46_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__47_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "multigoalAttr"};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__47_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__47_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(200, 5, 203, 188, 211, 209, 34, 248)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__46_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(204, 202, 81, 8, 110, 218, 45, 117)}};
static const lean_ctor_object lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__47_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(170, 133, 22, 125, 37, 223, 227, 162)}};
static const lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_multigoalAttr;
static const lean_string_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__0_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "conv_<;>_"};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__17_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value_aux_3),((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 57, 152, 10, 187, 180, 111, 39)}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3_value;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___boxed(lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instBEqRange_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__2_value;
static const lean_closure_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Syntax_instHashableRange_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__10 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__0 = (const lean_object*)&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1;
static const lean_string_object lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__2 = (const lean_object*)&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Used `tac1 <;> tac2` where `(tac1; tac2)` would suffice"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__0_value)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__1_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__0_value;
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__0_value;
static const lean_array_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__0 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__0_value;
static const lean_closure_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__0_value)} };
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__1 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "unnecessarySeqFocusLinter"};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__2 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__5_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__6_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(200, 5, 203, 188, 211, 209, 34, 248)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__46_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(204, 202, 81, 8, 110, 218, 45, 117)}};
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value_aux_2),((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(208, 205, 204, 46, 7, 41, 178, 202)}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__1_value),((lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__4 = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__4_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter = (const lean_object*)&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__4_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn___closed__7_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_));
v___x_56_ = lp_batteries_Lean_Option_register___at___00__private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus(lean_object* v_o_59_){
_start:
{
lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_60_ = lp_batteries_Batteries_Linter_linter_unnecessarySeqFocus;
v___x_61_ = l_Lean_Linter_getLinterValue(v___x_60_, v_o_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus___boxed(lean_object* v_o_62_){
_start:
{
uint8_t v_res_63_; lean_object* v_r_64_; 
v_res_63_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus(v_o_62_);
lean_dec_ref(v_o_62_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_(lean_object* v_x_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lean_box(0);
v___x_70_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2____boxed(lean_object* v_x_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___lam__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_(v_x_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v_x_71_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___f_220_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__0_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_));
v___x_221_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__2_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_));
v___x_222_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__3_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_));
v___x_223_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__45_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_));
v___x_224_ = ((lean_object*)(lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn___closed__48_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_));
v___x_225_ = lp_batteries_Lean_registerTagAttributeExtra(v___x_221_, v___x_222_, v___x_223_, v___f_220_, v___x_224_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2____boxed(lean_object* v_a_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_();
return v_res_227_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus(lean_object* v_k_241_){
_start:
{
lean_object* v___x_242_; uint8_t v___x_243_; 
v___x_242_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1));
v___x_243_ = lean_name_eq(v_k_241_, v___x_242_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_244_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3));
v___x_245_ = lean_name_eq(v_k_241_, v___x_244_);
return v___x_245_;
}
else
{
return v___x_243_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___boxed(lean_object* v_k_246_){
_start:
{
uint8_t v_res_247_; lean_object* v_r_248_; 
v_res_247_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus(v_k_246_);
lean_dec(v_k_246_);
v_r_248_ = lean_box(v_res_247_);
return v_r_248_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0(void){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = l_instMonadST(lean_box(0));
return v___x_249_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1(void){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_250_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__0);
v___x_251_ = l_StateRefT_x27_instMonad___redArg(v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0___boxed(lean_object* v_x_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0(v_x_252_, v___y_253_, v___y_254_);
lean_dec(v___y_254_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(lean_object* v_stx_259_, lean_object* v_a_260_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__1);
if (lean_obj_tag(v_stx_259_) == 1)
{
lean_object* v_kind_263_; lean_object* v_args_264_; lean_object* v___f_265_; lean_object* v___y_267_; uint8_t v___y_282_; lean_object* v___x_292_; uint8_t v___x_293_; 
v_kind_263_ = lean_ctor_get(v_stx_259_, 1);
v_args_264_ = lean_ctor_get(v_stx_259_, 2);
lean_inc_ref(v_args_264_);
v___f_265_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0___boxed), 4, 0);
v___x_292_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1));
v___x_293_ = lean_name_eq(v_kind_263_, v___x_292_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_294_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3));
v___x_295_ = lean_name_eq(v_kind_263_, v___x_294_);
v___y_282_ = v___x_295_;
goto v___jp_281_;
}
else
{
v___y_282_ = v___x_293_;
goto v___jp_281_;
}
v___jp_266_:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; uint8_t v___x_271_; 
v___x_268_ = lean_unsigned_to_nat(0u);
v___x_269_ = lean_array_get_size(v_args_264_);
v___x_270_ = lean_box(0);
v___x_271_ = lean_nat_dec_lt(v___x_268_, v___x_269_);
if (v___x_271_ == 0)
{
lean_dec_ref(v___f_265_);
lean_dec_ref(v_args_264_);
return v___x_270_;
}
else
{
uint8_t v___x_272_; 
v___x_272_ = lean_nat_dec_le(v___x_269_, v___x_269_);
if (v___x_272_ == 0)
{
if (v___x_271_ == 0)
{
lean_dec_ref(v___f_265_);
lean_dec_ref(v_args_264_);
return v___x_270_;
}
else
{
size_t v___x_273_; size_t v___x_274_; lean_object* v___x_747__overap_275_; lean_object* v___x_276_; 
v___x_273_ = ((size_t)0ULL);
v___x_274_ = lean_usize_of_nat(v___x_269_);
v___x_747__overap_275_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_262_, v___f_265_, v_args_264_, v___x_273_, v___x_274_, v___x_270_);
lean_inc(v___y_267_);
v___x_276_ = lean_apply_2(v___x_747__overap_275_, v___y_267_, lean_box(0));
return v___x_276_;
}
}
else
{
size_t v___x_277_; size_t v___x_278_; lean_object* v___x_752__overap_279_; lean_object* v___x_280_; 
v___x_277_ = ((size_t)0ULL);
v___x_278_ = lean_usize_of_nat(v___x_269_);
v___x_752__overap_279_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_262_, v___f_265_, v_args_264_, v___x_277_, v___x_278_, v___x_270_);
lean_inc(v___y_267_);
v___x_280_ = lean_apply_2(v___x_752__overap_279_, v___y_267_, lean_box(0));
return v___x_280_;
}
}
}
v___jp_281_:
{
if (v___y_282_ == 0)
{
lean_dec_ref_known(v_stx_259_, 3);
v___y_267_ = v_a_260_;
goto v___jp_266_;
}
else
{
lean_object* v_r_283_; 
v_r_283_ = l_Lean_Syntax_getRange_x3f(v_stx_259_, v___y_282_);
if (lean_obj_tag(v_r_283_) == 1)
{
lean_object* v_val_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; uint8_t v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v_val_284_ = lean_ctor_get(v_r_283_, 0);
lean_inc(v_val_284_);
lean_dec_ref_known(v_r_283_, 1);
v___x_285_ = lean_st_ref_take(v_a_260_);
v___x_286_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__2));
v___x_287_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___closed__3));
v___x_288_ = 0;
v___x_289_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_289_, 0, v_stx_259_);
lean_ctor_set_uint8(v___x_289_, sizeof(void*)*1, v___x_288_);
v___x_290_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_286_, v___x_287_, v___x_285_, v_val_284_, v___x_289_);
v___x_291_ = lean_st_ref_set(v_a_260_, v___x_290_);
v___y_267_ = v_a_260_;
goto v___jp_266_;
}
else
{
lean_dec(v_r_283_);
lean_dec_ref_known(v_stx_259_, 3);
v___y_267_ = v_a_260_;
goto v___jp_266_;
}
}
}
}
else
{
lean_object* v___x_296_; 
lean_dec(v_stx_259_);
v___x_296_ = lean_box(0);
return v___x_296_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___lam__0(lean_object* v_x_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(v___y_298_, v___y_299_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg___boxed(lean_object* v_stx_302_, lean_object* v_a_303_, lean_object* v_a_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(v_stx_302_, v_a_303_);
lean_dec(v_a_303_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics(lean_object* v_00_u03c9_306_, lean_object* v_stx_307_, lean_object* v_a_308_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(v_stx_307_, v_a_308_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___boxed(lean_object* v_00_u03c9_311_, lean_object* v_stx_312_, lean_object* v_a_313_, lean_object* v_a_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics(v_00_u03c9_311_, v_stx_312_, v_a_313_);
lean_dec(v_a_313_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath(lean_object* v_x_316_, lean_object* v_x_317_, lean_object* v_x_318_){
_start:
{
if (lean_obj_tag(v_x_318_) == 0)
{
lean_object* v___x_319_; 
lean_dec_ref(v_x_317_);
v___x_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_319_, 0, v_x_316_);
return v___x_319_;
}
else
{
lean_object* v_head_320_; lean_object* v_tail_321_; lean_object* v_fst_322_; lean_object* v_snd_323_; lean_object* v_size_324_; uint8_t v___x_325_; 
lean_dec_ref(v_x_316_);
v_head_320_ = lean_ctor_get(v_x_318_, 0);
v_tail_321_ = lean_ctor_get(v_x_318_, 1);
v_fst_322_ = lean_ctor_get(v_head_320_, 0);
v_snd_323_ = lean_ctor_get(v_head_320_, 1);
v_size_324_ = lean_ctor_get(v_x_317_, 2);
v___x_325_ = lean_nat_dec_eq(v_size_324_, v_fst_322_);
if (v___x_325_ == 0)
{
lean_object* v___x_326_; 
lean_dec_ref(v_x_317_);
v___x_326_ = lean_box(0);
return v___x_326_;
}
else
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = l_Lean_Elab_instInhabitedInfoTree_default;
v___x_328_ = l_Lean_PersistentArray_get_x21___redArg(v___x_327_, v_x_317_, v_snd_323_);
lean_dec_ref(v_x_317_);
if (lean_obj_tag(v___x_328_) == 1)
{
lean_object* v_i_329_; lean_object* v_children_330_; 
v_i_329_ = lean_ctor_get(v___x_328_, 0);
lean_inc_ref(v_i_329_);
v_children_330_ = lean_ctor_get(v___x_328_, 1);
lean_inc_ref(v_children_330_);
lean_dec_ref_known(v___x_328_, 2);
v_x_316_ = v_i_329_;
v_x_317_ = v_children_330_;
v_x_318_ = v_tail_321_;
goto _start;
}
else
{
lean_object* v___x_332_; 
lean_dec(v___x_328_);
v___x_332_ = lean_box(0);
return v___x_332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath___boxed(lean_object* v_x_333_, lean_object* v_x_334_, lean_object* v_x_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath(v_x_333_, v_x_334_, v_x_335_);
lean_dec(v_x_335_);
return v_res_336_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(lean_object* v_a_337_, lean_object* v_x_338_){
_start:
{
if (lean_obj_tag(v_x_338_) == 0)
{
uint8_t v___x_339_; 
v___x_339_ = 0;
return v___x_339_;
}
else
{
lean_object* v_key_340_; lean_object* v_tail_341_; uint8_t v___x_342_; 
v_key_340_ = lean_ctor_get(v_x_338_, 0);
v_tail_341_ = lean_ctor_get(v_x_338_, 2);
v___x_342_ = l_Lean_Syntax_instBEqRange_beq(v_key_340_, v_a_337_);
if (v___x_342_ == 0)
{
v_x_338_ = v_tail_341_;
goto _start;
}
else
{
return v___x_342_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg___boxed(lean_object* v_a_344_, lean_object* v_x_345_){
_start:
{
uint8_t v_res_346_; lean_object* v_r_347_; 
v_res_346_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(v_a_344_, v_x_345_);
lean_dec(v_x_345_);
lean_dec_ref(v_a_344_);
v_r_347_ = lean_box(v_res_346_);
return v_r_347_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14___redArg(lean_object* v_x_348_, lean_object* v_x_349_){
_start:
{
if (lean_obj_tag(v_x_349_) == 0)
{
return v_x_348_;
}
else
{
lean_object* v_key_350_; lean_object* v_value_351_; lean_object* v_tail_352_; lean_object* v___x_354_; uint8_t v_isShared_355_; uint8_t v_isSharedCheck_375_; 
v_key_350_ = lean_ctor_get(v_x_349_, 0);
v_value_351_ = lean_ctor_get(v_x_349_, 1);
v_tail_352_ = lean_ctor_get(v_x_349_, 2);
v_isSharedCheck_375_ = !lean_is_exclusive(v_x_349_);
if (v_isSharedCheck_375_ == 0)
{
v___x_354_ = v_x_349_;
v_isShared_355_ = v_isSharedCheck_375_;
goto v_resetjp_353_;
}
else
{
lean_inc(v_tail_352_);
lean_inc(v_value_351_);
lean_inc(v_key_350_);
lean_dec(v_x_349_);
v___x_354_ = lean_box(0);
v_isShared_355_ = v_isSharedCheck_375_;
goto v_resetjp_353_;
}
v_resetjp_353_:
{
lean_object* v___x_356_; uint64_t v___x_357_; uint64_t v___x_358_; uint64_t v___x_359_; uint64_t v_fold_360_; uint64_t v___x_361_; uint64_t v___x_362_; uint64_t v___x_363_; size_t v___x_364_; size_t v___x_365_; size_t v___x_366_; size_t v___x_367_; size_t v___x_368_; lean_object* v___x_369_; lean_object* v___x_371_; 
v___x_356_ = lean_array_get_size(v_x_348_);
v___x_357_ = l_Lean_Syntax_instHashableRange_hash(v_key_350_);
v___x_358_ = 32ULL;
v___x_359_ = lean_uint64_shift_right(v___x_357_, v___x_358_);
v_fold_360_ = lean_uint64_xor(v___x_357_, v___x_359_);
v___x_361_ = 16ULL;
v___x_362_ = lean_uint64_shift_right(v_fold_360_, v___x_361_);
v___x_363_ = lean_uint64_xor(v_fold_360_, v___x_362_);
v___x_364_ = lean_uint64_to_usize(v___x_363_);
v___x_365_ = lean_usize_of_nat(v___x_356_);
v___x_366_ = ((size_t)1ULL);
v___x_367_ = lean_usize_sub(v___x_365_, v___x_366_);
v___x_368_ = lean_usize_land(v___x_364_, v___x_367_);
v___x_369_ = lean_array_uget_borrowed(v_x_348_, v___x_368_);
lean_inc(v___x_369_);
if (v_isShared_355_ == 0)
{
lean_ctor_set(v___x_354_, 2, v___x_369_);
v___x_371_ = v___x_354_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_key_350_);
lean_ctor_set(v_reuseFailAlloc_374_, 1, v_value_351_);
lean_ctor_set(v_reuseFailAlloc_374_, 2, v___x_369_);
v___x_371_ = v_reuseFailAlloc_374_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___x_372_; 
v___x_372_ = lean_array_uset(v_x_348_, v___x_368_, v___x_371_);
v_x_348_ = v___x_372_;
v_x_349_ = v_tail_352_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13___redArg(lean_object* v_i_376_, lean_object* v_source_377_, lean_object* v_target_378_){
_start:
{
lean_object* v___x_379_; uint8_t v___x_380_; 
v___x_379_ = lean_array_get_size(v_source_377_);
v___x_380_ = lean_nat_dec_lt(v_i_376_, v___x_379_);
if (v___x_380_ == 0)
{
lean_dec_ref(v_source_377_);
lean_dec(v_i_376_);
return v_target_378_;
}
else
{
lean_object* v_es_381_; lean_object* v___x_382_; lean_object* v_source_383_; lean_object* v_target_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v_es_381_ = lean_array_fget(v_source_377_, v_i_376_);
v___x_382_ = lean_box(0);
v_source_383_ = lean_array_fset(v_source_377_, v_i_376_, v___x_382_);
v_target_384_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14___redArg(v_target_378_, v_es_381_);
v___x_385_ = lean_unsigned_to_nat(1u);
v___x_386_ = lean_nat_add(v_i_376_, v___x_385_);
lean_dec(v_i_376_);
v_i_376_ = v___x_386_;
v_source_377_ = v_source_383_;
v_target_378_ = v_target_384_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10___redArg(lean_object* v_data_388_){
_start:
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v_nbuckets_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_389_ = lean_array_get_size(v_data_388_);
v___x_390_ = lean_unsigned_to_nat(2u);
v_nbuckets_391_ = lean_nat_mul(v___x_389_, v___x_390_);
v___x_392_ = lean_unsigned_to_nat(0u);
v___x_393_ = lean_box(0);
v___x_394_ = lean_mk_array(v_nbuckets_391_, v___x_393_);
v___x_395_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13___redArg(v___x_392_, v_data_388_, v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11___redArg(lean_object* v_a_396_, lean_object* v_b_397_, lean_object* v_x_398_){
_start:
{
if (lean_obj_tag(v_x_398_) == 0)
{
lean_dec(v_b_397_);
lean_dec_ref(v_a_396_);
return v_x_398_;
}
else
{
lean_object* v_key_399_; lean_object* v_value_400_; lean_object* v_tail_401_; lean_object* v___x_403_; uint8_t v_isShared_404_; uint8_t v_isSharedCheck_413_; 
v_key_399_ = lean_ctor_get(v_x_398_, 0);
v_value_400_ = lean_ctor_get(v_x_398_, 1);
v_tail_401_ = lean_ctor_get(v_x_398_, 2);
v_isSharedCheck_413_ = !lean_is_exclusive(v_x_398_);
if (v_isSharedCheck_413_ == 0)
{
v___x_403_ = v_x_398_;
v_isShared_404_ = v_isSharedCheck_413_;
goto v_resetjp_402_;
}
else
{
lean_inc(v_tail_401_);
lean_inc(v_value_400_);
lean_inc(v_key_399_);
lean_dec(v_x_398_);
v___x_403_ = lean_box(0);
v_isShared_404_ = v_isSharedCheck_413_;
goto v_resetjp_402_;
}
v_resetjp_402_:
{
uint8_t v___x_405_; 
v___x_405_ = l_Lean_Syntax_instBEqRange_beq(v_key_399_, v_a_396_);
if (v___x_405_ == 0)
{
lean_object* v___x_406_; lean_object* v___x_408_; 
v___x_406_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11___redArg(v_a_396_, v_b_397_, v_tail_401_);
if (v_isShared_404_ == 0)
{
lean_ctor_set(v___x_403_, 2, v___x_406_);
v___x_408_ = v___x_403_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_key_399_);
lean_ctor_set(v_reuseFailAlloc_409_, 1, v_value_400_);
lean_ctor_set(v_reuseFailAlloc_409_, 2, v___x_406_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
else
{
lean_object* v___x_411_; 
lean_dec(v_value_400_);
lean_dec(v_key_399_);
if (v_isShared_404_ == 0)
{
lean_ctor_set(v___x_403_, 1, v_b_397_);
lean_ctor_set(v___x_403_, 0, v_a_396_);
v___x_411_ = v___x_403_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_a_396_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_b_397_);
lean_ctor_set(v_reuseFailAlloc_412_, 2, v_tail_401_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4___redArg(lean_object* v_m_414_, lean_object* v_a_415_, lean_object* v_b_416_){
_start:
{
lean_object* v_size_417_; lean_object* v_buckets_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_461_; 
v_size_417_ = lean_ctor_get(v_m_414_, 0);
v_buckets_418_ = lean_ctor_get(v_m_414_, 1);
v_isSharedCheck_461_ = !lean_is_exclusive(v_m_414_);
if (v_isSharedCheck_461_ == 0)
{
v___x_420_ = v_m_414_;
v_isShared_421_ = v_isSharedCheck_461_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_buckets_418_);
lean_inc(v_size_417_);
lean_dec(v_m_414_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_461_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_422_; uint64_t v___x_423_; uint64_t v___x_424_; uint64_t v___x_425_; uint64_t v_fold_426_; uint64_t v___x_427_; uint64_t v___x_428_; uint64_t v___x_429_; size_t v___x_430_; size_t v___x_431_; size_t v___x_432_; size_t v___x_433_; size_t v___x_434_; lean_object* v_bkt_435_; uint8_t v___x_436_; 
v___x_422_ = lean_array_get_size(v_buckets_418_);
v___x_423_ = l_Lean_Syntax_instHashableRange_hash(v_a_415_);
v___x_424_ = 32ULL;
v___x_425_ = lean_uint64_shift_right(v___x_423_, v___x_424_);
v_fold_426_ = lean_uint64_xor(v___x_423_, v___x_425_);
v___x_427_ = 16ULL;
v___x_428_ = lean_uint64_shift_right(v_fold_426_, v___x_427_);
v___x_429_ = lean_uint64_xor(v_fold_426_, v___x_428_);
v___x_430_ = lean_uint64_to_usize(v___x_429_);
v___x_431_ = lean_usize_of_nat(v___x_422_);
v___x_432_ = ((size_t)1ULL);
v___x_433_ = lean_usize_sub(v___x_431_, v___x_432_);
v___x_434_ = lean_usize_land(v___x_430_, v___x_433_);
v_bkt_435_ = lean_array_uget_borrowed(v_buckets_418_, v___x_434_);
v___x_436_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(v_a_415_, v_bkt_435_);
if (v___x_436_ == 0)
{
lean_object* v___x_437_; lean_object* v_size_x27_438_; lean_object* v___x_439_; lean_object* v_buckets_x27_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; uint8_t v___x_446_; 
v___x_437_ = lean_unsigned_to_nat(1u);
v_size_x27_438_ = lean_nat_add(v_size_417_, v___x_437_);
lean_dec(v_size_417_);
lean_inc(v_bkt_435_);
v___x_439_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_439_, 0, v_a_415_);
lean_ctor_set(v___x_439_, 1, v_b_416_);
lean_ctor_set(v___x_439_, 2, v_bkt_435_);
v_buckets_x27_440_ = lean_array_uset(v_buckets_418_, v___x_434_, v___x_439_);
v___x_441_ = lean_unsigned_to_nat(4u);
v___x_442_ = lean_nat_mul(v_size_x27_438_, v___x_441_);
v___x_443_ = lean_unsigned_to_nat(3u);
v___x_444_ = lean_nat_div(v___x_442_, v___x_443_);
lean_dec(v___x_442_);
v___x_445_ = lean_array_get_size(v_buckets_x27_440_);
v___x_446_ = lean_nat_dec_le(v___x_444_, v___x_445_);
lean_dec(v___x_444_);
if (v___x_446_ == 0)
{
lean_object* v_val_447_; lean_object* v___x_449_; 
v_val_447_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10___redArg(v_buckets_x27_440_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 1, v_val_447_);
lean_ctor_set(v___x_420_, 0, v_size_x27_438_);
v___x_449_ = v___x_420_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_size_x27_438_);
lean_ctor_set(v_reuseFailAlloc_450_, 1, v_val_447_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
else
{
lean_object* v___x_452_; 
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 1, v_buckets_x27_440_);
lean_ctor_set(v___x_420_, 0, v_size_x27_438_);
v___x_452_ = v___x_420_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_size_x27_438_);
lean_ctor_set(v_reuseFailAlloc_453_, 1, v_buckets_x27_440_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
else
{
lean_object* v___x_454_; lean_object* v_buckets_x27_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_459_; 
lean_inc(v_bkt_435_);
v___x_454_ = lean_box(0);
v_buckets_x27_455_ = lean_array_uset(v_buckets_418_, v___x_434_, v___x_454_);
v___x_456_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11___redArg(v_a_415_, v_b_416_, v_bkt_435_);
v___x_457_ = lean_array_uset(v_buckets_x27_455_, v___x_434_, v___x_456_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 1, v___x_457_);
v___x_459_ = v___x_420_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_size_417_);
lean_ctor_set(v_reuseFailAlloc_460_, 1, v___x_457_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(lean_object* v_a_462_, lean_object* v_x_463_){
_start:
{
if (lean_obj_tag(v_x_463_) == 0)
{
return v_x_463_;
}
else
{
lean_object* v_key_464_; lean_object* v_value_465_; lean_object* v_tail_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_475_; 
v_key_464_ = lean_ctor_get(v_x_463_, 0);
v_value_465_ = lean_ctor_get(v_x_463_, 1);
v_tail_466_ = lean_ctor_get(v_x_463_, 2);
v_isSharedCheck_475_ = !lean_is_exclusive(v_x_463_);
if (v_isSharedCheck_475_ == 0)
{
v___x_468_ = v_x_463_;
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_tail_466_);
lean_inc(v_value_465_);
lean_inc(v_key_464_);
lean_dec(v_x_463_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
uint8_t v___x_470_; 
v___x_470_ = l_Lean_Syntax_instBEqRange_beq(v_key_464_, v_a_462_);
if (v___x_470_ == 0)
{
lean_object* v___x_471_; lean_object* v___x_473_; 
v___x_471_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(v_a_462_, v_tail_466_);
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 2, v___x_471_);
v___x_473_ = v___x_468_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_key_464_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v_value_465_);
lean_ctor_set(v_reuseFailAlloc_474_, 2, v___x_471_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
else
{
lean_del_object(v___x_468_);
lean_dec(v_value_465_);
lean_dec(v_key_464_);
return v_tail_466_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg___boxed(lean_object* v_a_476_, lean_object* v_x_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(v_a_476_, v_x_477_);
lean_dec_ref(v_a_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(lean_object* v_m_479_, lean_object* v_a_480_){
_start:
{
lean_object* v_size_481_; lean_object* v_buckets_482_; lean_object* v___x_483_; uint64_t v___x_484_; uint64_t v___x_485_; uint64_t v___x_486_; uint64_t v_fold_487_; uint64_t v___x_488_; uint64_t v___x_489_; uint64_t v___x_490_; size_t v___x_491_; size_t v___x_492_; size_t v___x_493_; size_t v___x_494_; size_t v___x_495_; lean_object* v_bkt_496_; uint8_t v___x_497_; 
v_size_481_ = lean_ctor_get(v_m_479_, 0);
v_buckets_482_ = lean_ctor_get(v_m_479_, 1);
v___x_483_ = lean_array_get_size(v_buckets_482_);
v___x_484_ = l_Lean_Syntax_instHashableRange_hash(v_a_480_);
v___x_485_ = 32ULL;
v___x_486_ = lean_uint64_shift_right(v___x_484_, v___x_485_);
v_fold_487_ = lean_uint64_xor(v___x_484_, v___x_486_);
v___x_488_ = 16ULL;
v___x_489_ = lean_uint64_shift_right(v_fold_487_, v___x_488_);
v___x_490_ = lean_uint64_xor(v_fold_487_, v___x_489_);
v___x_491_ = lean_uint64_to_usize(v___x_490_);
v___x_492_ = lean_usize_of_nat(v___x_483_);
v___x_493_ = ((size_t)1ULL);
v___x_494_ = lean_usize_sub(v___x_492_, v___x_493_);
v___x_495_ = lean_usize_land(v___x_491_, v___x_494_);
v_bkt_496_ = lean_array_uget_borrowed(v_buckets_482_, v___x_495_);
v___x_497_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(v_a_480_, v_bkt_496_);
if (v___x_497_ == 0)
{
return v_m_479_;
}
else
{
lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_510_; 
lean_inc(v_bkt_496_);
lean_inc_ref(v_buckets_482_);
lean_inc(v_size_481_);
v_isSharedCheck_510_ = !lean_is_exclusive(v_m_479_);
if (v_isSharedCheck_510_ == 0)
{
lean_object* v_unused_511_; lean_object* v_unused_512_; 
v_unused_511_ = lean_ctor_get(v_m_479_, 1);
lean_dec(v_unused_511_);
v_unused_512_ = lean_ctor_get(v_m_479_, 0);
lean_dec(v_unused_512_);
v___x_499_ = v_m_479_;
v_isShared_500_ = v_isSharedCheck_510_;
goto v_resetjp_498_;
}
else
{
lean_dec(v_m_479_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_510_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___x_501_; lean_object* v_buckets_x27_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_508_; 
v___x_501_ = lean_box(0);
v_buckets_x27_502_ = lean_array_uset(v_buckets_482_, v___x_495_, v___x_501_);
v___x_503_ = lean_unsigned_to_nat(1u);
v___x_504_ = lean_nat_sub(v_size_481_, v___x_503_);
lean_dec(v_size_481_);
v___x_505_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(v_a_480_, v_bkt_496_);
v___x_506_ = lean_array_uset(v_buckets_x27_502_, v___x_495_, v___x_505_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 1, v___x_506_);
lean_ctor_set(v___x_499_, 0, v___x_504_);
v___x_508_ = v___x_499_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_504_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v___x_506_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg___boxed(lean_object* v_m_513_, lean_object* v_a_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(v_m_513_, v_a_514_);
lean_dec_ref(v_a_514_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg(lean_object* v_a_516_, lean_object* v_x_517_){
_start:
{
if (lean_obj_tag(v_x_517_) == 0)
{
lean_object* v___x_518_; 
v___x_518_ = lean_box(0);
return v___x_518_;
}
else
{
lean_object* v_key_519_; lean_object* v_value_520_; lean_object* v_tail_521_; uint8_t v___x_522_; 
v_key_519_ = lean_ctor_get(v_x_517_, 0);
v_value_520_ = lean_ctor_get(v_x_517_, 1);
v_tail_521_ = lean_ctor_get(v_x_517_, 2);
v___x_522_ = l_Lean_Syntax_instBEqRange_beq(v_key_519_, v_a_516_);
if (v___x_522_ == 0)
{
v_x_517_ = v_tail_521_;
goto _start;
}
else
{
lean_object* v___x_524_; 
lean_inc(v_value_520_);
v___x_524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_524_, 0, v_value_520_);
return v___x_524_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg___boxed(lean_object* v_a_525_, lean_object* v_x_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg(v_a_525_, v_x_526_);
lean_dec(v_x_526_);
lean_dec_ref(v_a_525_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg(lean_object* v_m_528_, lean_object* v_a_529_){
_start:
{
lean_object* v_buckets_530_; lean_object* v___x_531_; uint64_t v___x_532_; uint64_t v___x_533_; uint64_t v___x_534_; uint64_t v_fold_535_; uint64_t v___x_536_; uint64_t v___x_537_; uint64_t v___x_538_; size_t v___x_539_; size_t v___x_540_; size_t v___x_541_; size_t v___x_542_; size_t v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v_buckets_530_ = lean_ctor_get(v_m_528_, 1);
v___x_531_ = lean_array_get_size(v_buckets_530_);
v___x_532_ = l_Lean_Syntax_instHashableRange_hash(v_a_529_);
v___x_533_ = 32ULL;
v___x_534_ = lean_uint64_shift_right(v___x_532_, v___x_533_);
v_fold_535_ = lean_uint64_xor(v___x_532_, v___x_534_);
v___x_536_ = 16ULL;
v___x_537_ = lean_uint64_shift_right(v_fold_535_, v___x_536_);
v___x_538_ = lean_uint64_xor(v_fold_535_, v___x_537_);
v___x_539_ = lean_uint64_to_usize(v___x_538_);
v___x_540_ = lean_usize_of_nat(v___x_531_);
v___x_541_ = ((size_t)1ULL);
v___x_542_ = lean_usize_sub(v___x_540_, v___x_541_);
v___x_543_ = lean_usize_land(v___x_539_, v___x_542_);
v___x_544_ = lean_array_uget_borrowed(v_buckets_530_, v___x_543_);
v___x_545_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg(v_a_529_, v___x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg___boxed(lean_object* v_m_546_, lean_object* v_a_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg(v_m_546_, v_a_547_);
lean_dec_ref(v_a_547_);
lean_dec_ref(v_m_546_);
return v_res_548_;
}
}
static lean_object* _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_549_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4(void){
_start:
{
lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_550_ = lean_unsigned_to_nat(5u);
v___x_551_ = lean_unsigned_to_nat(0u);
v___x_552_ = lean_nat_mod(v___x_551_, v___x_550_);
return v___x_552_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5(void){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v___x_553_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__4);
v___x_554_ = lean_unsigned_to_nat(5u);
v___x_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_555_, 0, v___x_554_);
lean_ctor_set(v___x_555_, 1, v___x_553_);
return v___x_555_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6(void){
_start:
{
lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_556_ = lean_box(0);
v___x_557_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__5);
v___x_558_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
lean_ctor_set(v___x_558_, 1, v___x_556_);
return v___x_558_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0(void){
_start:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_559_ = lean_unsigned_to_nat(1u);
v___x_560_ = lean_unsigned_to_nat(0u);
v___x_561_ = lean_nat_mod(v___x_560_, v___x_559_);
return v___x_561_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1(void){
_start:
{
lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; 
v___x_562_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__0);
v___x_563_ = lean_unsigned_to_nat(1u);
v___x_564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_564_, 0, v___x_563_);
lean_ctor_set(v___x_564_, 1, v___x_562_);
return v___x_564_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7(void){
_start:
{
lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_565_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__6);
v___x_566_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_567_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_566_);
lean_ctor_set(v___x_567_, 1, v___x_565_);
return v___x_567_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2(void){
_start:
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_568_ = lean_unsigned_to_nat(2u);
v___x_569_ = lean_unsigned_to_nat(1u);
v___x_570_ = lean_nat_mod(v___x_569_, v___x_568_);
return v___x_570_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3(void){
_start:
{
lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_571_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__2);
v___x_572_ = lean_unsigned_to_nat(2u);
v___x_573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v___x_571_);
return v___x_573_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8(void){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_574_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__7);
v___x_575_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__3);
v___x_576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_576_, 0, v___x_575_);
lean_ctor_set(v___x_576_, 1, v___x_574_);
return v___x_576_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9(void){
_start:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_577_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__8);
v___x_578_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_579_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_579_, 0, v___x_578_);
lean_ctor_set(v___x_579_, 1, v___x_577_);
return v___x_579_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11(void){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_582_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9);
v___x_583_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_584_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v___x_582_);
return v___x_584_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
v___x_585_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__11);
v___x_586_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_587_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_587_, 0, v___x_586_);
lean_ctor_set(v___x_587_, 1, v___x_585_);
return v___x_587_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_588_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__12);
v___x_589_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_590_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_590_, 0, v___x_589_);
lean_ctor_set(v___x_590_, 1, v___x_588_);
return v___x_590_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14(void){
_start:
{
lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_591_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__13);
v___x_592_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__1);
v___x_593_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v___x_591_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg(lean_object* v_env_594_, lean_object* v_x_595_, lean_object* v_a_596_){
_start:
{
switch(lean_obj_tag(v_x_595_))
{
case 0:
{
lean_object* v_t_598_; 
v_t_598_ = lean_ctor_get(v_x_595_, 1);
lean_inc_ref(v_t_598_);
lean_dec_ref_known(v_x_595_, 2);
v_x_595_ = v_t_598_;
goto _start;
}
case 1:
{
lean_object* v_i_600_; lean_object* v_children_601_; lean_object* v_snd_603_; lean_object* v_snd_607_; 
v_i_600_ = lean_ctor_get(v_x_595_, 0);
lean_inc_ref(v_i_600_);
v_children_601_ = lean_ctor_get(v_x_595_, 1);
lean_inc_ref(v_children_601_);
lean_dec_ref_known(v_x_595_, 2);
if (lean_obj_tag(v_i_600_) == 0)
{
lean_object* v_i_610_; lean_object* v_toElabInfo_611_; lean_object* v_goalsBefore_612_; lean_object* v_stx_613_; uint8_t v___x_614_; lean_object* v___x_615_; 
v_i_610_ = lean_ctor_get(v_i_600_, 0);
v_toElabInfo_611_ = lean_ctor_get(v_i_610_, 0);
v_goalsBefore_612_ = lean_ctor_get(v_i_610_, 2);
v_stx_613_ = lean_ctor_get(v_toElabInfo_611_, 1);
v___x_614_ = 1;
v___x_615_ = l_Lean_Syntax_getRange_x3f(v_stx_613_, v___x_614_);
if (lean_obj_tag(v___x_615_) == 1)
{
lean_object* v_val_616_; lean_object* v___y_618_; lean_object* v___x_620_; lean_object* v___x_621_; 
v_val_616_ = lean_ctor_get(v___x_615_, 0);
lean_inc(v_val_616_);
lean_dec_ref_known(v___x_615_, 1);
v___x_620_ = lean_st_ref_get(v_a_596_);
v___x_621_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg(v___x_620_, v_val_616_);
lean_dec(v___x_620_);
if (lean_obj_tag(v___x_621_) == 1)
{
lean_object* v_val_622_; lean_object* v___y_624_; uint8_t v___y_625_; lean_object* v___x_636_; lean_object* v___x_637_; uint8_t v___x_638_; lean_object* v___y_640_; uint8_t v___y_665_; 
v_val_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc(v_val_622_);
lean_dec_ref_known(v___x_621_, 1);
lean_inc(v_stx_613_);
v___x_636_ = l_Lean_Syntax_getKind(v_stx_613_);
v___x_637_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__1));
v___x_638_ = lean_name_eq(v___x_636_, v___x_637_);
if (v___x_638_ == 0)
{
lean_object* v___x_667_; uint8_t v___x_668_; lean_object* v___y_670_; uint8_t v___y_686_; 
v___x_667_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_isSeqFocus___closed__3));
v___x_668_ = lean_name_eq(v___x_636_, v___x_667_);
lean_dec(v___x_636_);
if (v___x_668_ == 0)
{
lean_object* v___x_688_; 
lean_dec(v_val_622_);
lean_dec(v_val_616_);
lean_dec_ref_known(v_i_600_, 1);
v___x_688_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_688_;
}
else
{
lean_object* v___x_689_; lean_object* v___x_690_; uint8_t v___x_691_; 
v___x_689_ = l_List_lengthTR___redArg(v_goalsBefore_612_);
v___x_690_ = lean_unsigned_to_nat(1u);
v___x_691_ = lean_nat_dec_eq(v___x_689_, v___x_690_);
lean_dec(v___x_689_);
if (v___x_691_ == 0)
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; uint8_t v___x_696_; 
v___x_692_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_multigoalAttr;
v___x_693_ = lean_unsigned_to_nat(0u);
v___x_694_ = l_Lean_Syntax_getArg(v_stx_613_, v___x_693_);
v___x_695_ = l_Lean_Syntax_getKind(v___x_694_);
lean_inc_ref(v_env_594_);
v___x_696_ = lp_batteries_Lean_TagAttributeExtra_hasTag(v___x_692_, v_env_594_, v___x_695_);
lean_dec(v___x_695_);
if (v___x_696_ == 0)
{
v___y_686_ = v___x_668_;
goto v___jp_685_;
}
else
{
v___y_686_ = v___x_691_;
goto v___jp_685_;
}
}
else
{
goto v___jp_672_;
}
}
v___jp_669_:
{
lean_object* v___x_671_; 
v___x_671_ = lean_st_ref_take(v_a_596_);
if (lean_obj_tag(v___y_670_) == 0)
{
v___y_624_ = v___x_671_;
v___y_625_ = v___x_638_;
goto v___jp_623_;
}
else
{
v___y_624_ = v___x_671_;
v___y_625_ = v___x_668_;
goto v___jp_623_;
}
}
v___jp_672_:
{
lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_673_ = lean_unsigned_to_nat(1u);
v___x_674_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__14);
lean_inc_ref(v_children_601_);
v___x_675_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath(v_i_600_, v_children_601_, v___x_674_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v___x_676_; 
v___x_676_ = lean_box(0);
v___y_670_ = v___x_676_;
goto v___jp_669_;
}
else
{
lean_object* v_val_677_; 
v_val_677_ = lean_ctor_get(v___x_675_, 0);
lean_inc(v_val_677_);
lean_dec_ref_known(v___x_675_, 1);
if (lean_obj_tag(v_val_677_) == 0)
{
lean_object* v_i_678_; lean_object* v_goalsAfter_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v_i_678_ = lean_ctor_get(v_val_677_, 0);
lean_inc_ref(v_i_678_);
lean_dec_ref_known(v_val_677_, 1);
v_goalsAfter_679_ = lean_ctor_get(v_i_678_, 4);
lean_inc(v_goalsAfter_679_);
lean_dec_ref(v_i_678_);
v___x_680_ = l_List_lengthTR___redArg(v_goalsAfter_679_);
lean_dec(v_goalsAfter_679_);
v___x_681_ = lean_nat_dec_eq(v___x_680_, v___x_673_);
lean_dec(v___x_680_);
if (v___x_681_ == 0)
{
lean_object* v___x_682_; 
v___x_682_ = lean_box(0);
v___y_670_ = v___x_682_;
goto v___jp_669_;
}
else
{
lean_object* v___x_683_; 
v___x_683_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__10));
v___y_670_ = v___x_683_;
goto v___jp_669_;
}
}
else
{
lean_object* v___x_684_; 
lean_dec(v_val_677_);
v___x_684_ = lean_box(0);
v___y_670_ = v___x_684_;
goto v___jp_669_;
}
}
}
v___jp_685_:
{
if (v___y_686_ == 0)
{
lean_object* v___x_687_; 
lean_dec_ref_known(v_i_600_, 1);
v___x_687_ = lean_box(0);
v___y_670_ = v___x_687_;
goto v___jp_669_;
}
else
{
goto v___jp_672_;
}
}
}
else
{
lean_object* v___x_697_; lean_object* v___x_698_; uint8_t v___x_699_; 
lean_dec(v___x_636_);
v___x_697_ = l_List_lengthTR___redArg(v_goalsBefore_612_);
v___x_698_ = lean_unsigned_to_nat(1u);
v___x_699_ = lean_nat_dec_eq(v___x_697_, v___x_698_);
lean_dec(v___x_697_);
if (v___x_699_ == 0)
{
lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; uint8_t v___x_704_; 
v___x_700_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_multigoalAttr;
v___x_701_ = lean_unsigned_to_nat(0u);
v___x_702_ = l_Lean_Syntax_getArg(v_stx_613_, v___x_701_);
v___x_703_ = l_Lean_Syntax_getKind(v___x_702_);
lean_inc_ref(v_env_594_);
v___x_704_ = lp_batteries_Lean_TagAttributeExtra_hasTag(v___x_700_, v_env_594_, v___x_703_);
lean_dec(v___x_703_);
if (v___x_704_ == 0)
{
v___y_665_ = v___x_638_;
goto v___jp_664_;
}
else
{
v___y_665_ = v___x_699_;
goto v___jp_664_;
}
}
else
{
goto v___jp_651_;
}
}
v___jp_623_:
{
if (v___y_625_ == 0)
{
lean_object* v___x_626_; 
lean_dec(v_val_622_);
v___x_626_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(v___y_624_, v_val_616_);
lean_dec(v_val_616_);
v_snd_607_ = v___x_626_;
goto v___jp_606_;
}
else
{
lean_object* v_stx_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_635_; 
v_stx_627_ = lean_ctor_get(v_val_622_, 0);
v_isSharedCheck_635_ = !lean_is_exclusive(v_val_622_);
if (v_isSharedCheck_635_ == 0)
{
v___x_629_ = v_val_622_;
v_isShared_630_ = v_isSharedCheck_635_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_stx_627_);
lean_dec(v_val_622_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_635_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_632_; 
if (v_isShared_630_ == 0)
{
v___x_632_ = v___x_629_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_634_; 
v_reuseFailAlloc_634_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_634_, 0, v_stx_627_);
v___x_632_ = v_reuseFailAlloc_634_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
lean_object* v___x_633_; 
lean_ctor_set_uint8(v___x_632_, sizeof(void*)*1, v___x_614_);
v___x_633_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4___redArg(v___y_624_, v_val_616_, v___x_632_);
v_snd_607_ = v___x_633_;
goto v___jp_606_;
}
}
}
}
v___jp_639_:
{
lean_object* v___x_641_; 
v___x_641_ = lean_st_ref_take(v_a_596_);
if (lean_obj_tag(v___y_640_) == 0)
{
lean_dec(v_val_622_);
v___y_618_ = v___x_641_;
goto v___jp_617_;
}
else
{
if (v___x_638_ == 0)
{
lean_dec(v_val_622_);
v___y_618_ = v___x_641_;
goto v___jp_617_;
}
else
{
lean_object* v_stx_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_650_; 
v_stx_642_ = lean_ctor_get(v_val_622_, 0);
v_isSharedCheck_650_ = !lean_is_exclusive(v_val_622_);
if (v_isSharedCheck_650_ == 0)
{
v___x_644_ = v_val_622_;
v_isShared_645_ = v_isSharedCheck_650_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_stx_642_);
lean_dec(v_val_622_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_650_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v___x_647_; 
if (v_isShared_645_ == 0)
{
v___x_647_ = v___x_644_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v_stx_642_);
v___x_647_ = v_reuseFailAlloc_649_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
lean_object* v___x_648_; 
lean_ctor_set_uint8(v___x_647_, sizeof(void*)*1, v___x_614_);
v___x_648_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4___redArg(v___x_641_, v_val_616_, v___x_647_);
v_snd_603_ = v___x_648_;
goto v___jp_602_;
}
}
}
}
}
v___jp_651_:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_unsigned_to_nat(1u);
v___x_653_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__9);
lean_inc_ref(v_children_601_);
v___x_654_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getPath(v_i_600_, v_children_601_, v___x_653_);
if (lean_obj_tag(v___x_654_) == 0)
{
lean_object* v___x_655_; 
v___x_655_ = lean_box(0);
v___y_640_ = v___x_655_;
goto v___jp_639_;
}
else
{
lean_object* v_val_656_; 
v_val_656_ = lean_ctor_get(v___x_654_, 0);
lean_inc(v_val_656_);
lean_dec_ref_known(v___x_654_, 1);
if (lean_obj_tag(v_val_656_) == 0)
{
lean_object* v_i_657_; lean_object* v_goalsAfter_658_; lean_object* v___x_659_; uint8_t v___x_660_; 
v_i_657_ = lean_ctor_get(v_val_656_, 0);
lean_inc_ref(v_i_657_);
lean_dec_ref_known(v_val_656_, 1);
v_goalsAfter_658_ = lean_ctor_get(v_i_657_, 4);
lean_inc(v_goalsAfter_658_);
lean_dec_ref(v_i_657_);
v___x_659_ = l_List_lengthTR___redArg(v_goalsAfter_658_);
lean_dec(v_goalsAfter_658_);
v___x_660_ = lean_nat_dec_eq(v___x_659_, v___x_652_);
lean_dec(v___x_659_);
if (v___x_660_ == 0)
{
lean_object* v___x_661_; 
v___x_661_ = lean_box(0);
v___y_640_ = v___x_661_;
goto v___jp_639_;
}
else
{
lean_object* v___x_662_; 
v___x_662_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___closed__10));
v___y_640_ = v___x_662_;
goto v___jp_639_;
}
}
else
{
lean_object* v___x_663_; 
lean_dec(v_val_656_);
v___x_663_ = lean_box(0);
v___y_640_ = v___x_663_;
goto v___jp_639_;
}
}
}
v___jp_664_:
{
if (v___y_665_ == 0)
{
lean_object* v___x_666_; 
lean_dec_ref_known(v_i_600_, 1);
v___x_666_ = lean_box(0);
v___y_640_ = v___x_666_;
goto v___jp_639_;
}
else
{
goto v___jp_651_;
}
}
}
else
{
lean_object* v___x_705_; 
lean_dec(v___x_621_);
lean_dec(v_val_616_);
lean_dec_ref_known(v_i_600_, 1);
v___x_705_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_705_;
}
v___jp_617_:
{
lean_object* v___x_619_; 
v___x_619_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(v___y_618_, v_val_616_);
lean_dec(v_val_616_);
v_snd_603_ = v___x_619_;
goto v___jp_602_;
}
}
else
{
lean_object* v___x_706_; 
lean_dec(v___x_615_);
lean_dec_ref_known(v_i_600_, 1);
v___x_706_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_706_;
}
}
else
{
lean_object* v___x_707_; 
lean_dec_ref(v_i_600_);
v___x_707_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_707_;
}
v___jp_602_:
{
lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_604_ = lean_st_ref_set(v_a_596_, v_snd_603_);
v___x_605_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_605_;
}
v___jp_606_:
{
lean_object* v___x_608_; lean_object* v___x_609_; 
v___x_608_ = lean_st_ref_set(v_a_596_, v_snd_607_);
v___x_609_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_594_, v_children_601_, v_a_596_);
lean_dec_ref(v_children_601_);
return v___x_609_;
}
}
default: 
{
lean_object* v___x_708_; 
lean_dec_ref_known(v_x_595_, 1);
lean_dec_ref(v_env_594_);
v___x_708_ = lean_box(0);
return v___x_708_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(lean_object* v_env_709_, lean_object* v_as_710_, size_t v_i_711_, size_t v_stop_712_, lean_object* v_b_713_, lean_object* v___y_714_){
_start:
{
uint8_t v___x_716_; 
v___x_716_ = lean_usize_dec_eq(v_i_711_, v_stop_712_);
if (v___x_716_ == 0)
{
lean_object* v___x_717_; lean_object* v___x_718_; size_t v___x_719_; size_t v___x_720_; 
v___x_717_ = lean_array_uget_borrowed(v_as_710_, v_i_711_);
lean_inc(v___x_717_);
lean_inc_ref(v_env_709_);
v___x_718_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg(v_env_709_, v___x_717_, v___y_714_);
v___x_719_ = ((size_t)1ULL);
v___x_720_ = lean_usize_add(v_i_711_, v___x_719_);
v_i_711_ = v___x_720_;
v_b_713_ = v___x_718_;
goto _start;
}
else
{
lean_dec_ref(v_env_709_);
return v_b_713_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(lean_object* v_env_722_, lean_object* v_x_723_, lean_object* v___y_724_){
_start:
{
if (lean_obj_tag(v_x_723_) == 0)
{
lean_object* v_cs_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; uint8_t v___x_730_; 
v_cs_726_ = lean_ctor_get(v_x_723_, 0);
v___x_727_ = lean_unsigned_to_nat(0u);
v___x_728_ = lean_array_get_size(v_cs_726_);
v___x_729_ = lean_box(0);
v___x_730_ = lean_nat_dec_lt(v___x_727_, v___x_728_);
if (v___x_730_ == 0)
{
lean_dec_ref(v_env_722_);
return v___x_729_;
}
else
{
uint8_t v___x_731_; 
v___x_731_ = lean_nat_dec_le(v___x_728_, v___x_728_);
if (v___x_731_ == 0)
{
if (v___x_730_ == 0)
{
lean_dec_ref(v_env_722_);
return v___x_729_;
}
else
{
size_t v___x_732_; size_t v___x_733_; lean_object* v___x_734_; 
v___x_732_ = ((size_t)0ULL);
v___x_733_ = lean_usize_of_nat(v___x_728_);
v___x_734_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_722_, v_cs_726_, v___x_732_, v___x_733_, v___x_729_, v___y_724_);
return v___x_734_;
}
}
else
{
size_t v___x_735_; size_t v___x_736_; lean_object* v___x_737_; 
v___x_735_ = ((size_t)0ULL);
v___x_736_ = lean_usize_of_nat(v___x_728_);
v___x_737_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_722_, v_cs_726_, v___x_735_, v___x_736_, v___x_729_, v___y_724_);
return v___x_737_;
}
}
}
else
{
lean_object* v_vs_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; uint8_t v___x_742_; 
v_vs_738_ = lean_ctor_get(v_x_723_, 0);
v___x_739_ = lean_unsigned_to_nat(0u);
v___x_740_ = lean_array_get_size(v_vs_738_);
v___x_741_ = lean_box(0);
v___x_742_ = lean_nat_dec_lt(v___x_739_, v___x_740_);
if (v___x_742_ == 0)
{
lean_dec_ref(v_env_722_);
return v___x_741_;
}
else
{
uint8_t v___x_743_; 
v___x_743_ = lean_nat_dec_le(v___x_740_, v___x_740_);
if (v___x_743_ == 0)
{
if (v___x_742_ == 0)
{
lean_dec_ref(v_env_722_);
return v___x_741_;
}
else
{
size_t v___x_744_; size_t v___x_745_; lean_object* v___x_746_; 
v___x_744_ = ((size_t)0ULL);
v___x_745_ = lean_usize_of_nat(v___x_740_);
v___x_746_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_722_, v_vs_738_, v___x_744_, v___x_745_, v___x_741_, v___y_724_);
return v___x_746_;
}
}
else
{
size_t v___x_747_; size_t v___x_748_; lean_object* v___x_749_; 
v___x_747_ = ((size_t)0ULL);
v___x_748_ = lean_usize_of_nat(v___x_740_);
v___x_749_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_722_, v_vs_738_, v___x_747_, v___x_748_, v___x_741_, v___y_724_);
return v___x_749_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(lean_object* v_env_750_, lean_object* v_as_751_, size_t v_i_752_, size_t v_stop_753_, lean_object* v_b_754_, lean_object* v___y_755_){
_start:
{
uint8_t v___x_757_; 
v___x_757_ = lean_usize_dec_eq(v_i_752_, v_stop_753_);
if (v___x_757_ == 0)
{
lean_object* v___x_758_; lean_object* v___x_759_; size_t v___x_760_; size_t v___x_761_; 
v___x_758_ = lean_array_uget_borrowed(v_as_751_, v_i_752_);
lean_inc_ref(v_env_750_);
v___x_759_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(v_env_750_, v___x_758_, v___y_755_);
v___x_760_ = ((size_t)1ULL);
v___x_761_ = lean_usize_add(v_i_752_, v___x_760_);
v_i_752_ = v___x_761_;
v_b_754_ = v___x_759_;
goto _start;
}
else
{
lean_dec_ref(v_env_750_);
return v_b_754_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(lean_object* v_env_763_, lean_object* v_x_764_, size_t v_x_765_, size_t v_x_766_, lean_object* v___y_767_){
_start:
{
if (lean_obj_tag(v_x_764_) == 0)
{
lean_object* v_cs_769_; lean_object* v___x_770_; size_t v___x_771_; lean_object* v_j_772_; lean_object* v___x_773_; size_t v___x_774_; size_t v___x_775_; size_t v___x_776_; size_t v___x_777_; size_t v___x_778_; size_t v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v_cs_769_ = lean_ctor_get(v_x_764_, 0);
v___x_770_ = lean_obj_once(&lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0, &lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0_once, _init_lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___closed__0);
v___x_771_ = lean_usize_shift_right(v_x_765_, v_x_766_);
v_j_772_ = lean_usize_to_nat(v___x_771_);
v___x_773_ = lean_array_get_borrowed(v___x_770_, v_cs_769_, v_j_772_);
v___x_774_ = ((size_t)1ULL);
v___x_775_ = lean_usize_shift_left(v___x_774_, v_x_766_);
v___x_776_ = lean_usize_sub(v___x_775_, v___x_774_);
v___x_777_ = lean_usize_land(v_x_765_, v___x_776_);
v___x_778_ = ((size_t)5ULL);
v___x_779_ = lean_usize_sub(v_x_766_, v___x_778_);
lean_inc_ref(v_env_763_);
v___x_780_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(v_env_763_, v___x_773_, v___x_777_, v___x_779_, v___y_767_);
v___x_781_ = lean_unsigned_to_nat(1u);
v___x_782_ = lean_nat_add(v_j_772_, v___x_781_);
lean_dec(v_j_772_);
v___x_783_ = lean_array_get_size(v_cs_769_);
v___x_784_ = lean_box(0);
v___x_785_ = lean_nat_dec_lt(v___x_782_, v___x_783_);
if (v___x_785_ == 0)
{
lean_dec(v___x_782_);
lean_dec_ref(v_env_763_);
return v___x_784_;
}
else
{
uint8_t v___x_786_; 
v___x_786_ = lean_nat_dec_le(v___x_783_, v___x_783_);
if (v___x_786_ == 0)
{
if (v___x_785_ == 0)
{
lean_dec(v___x_782_);
lean_dec_ref(v_env_763_);
return v___x_784_;
}
else
{
size_t v___x_787_; size_t v___x_788_; lean_object* v___x_789_; 
v___x_787_ = lean_usize_of_nat(v___x_782_);
lean_dec(v___x_782_);
v___x_788_ = lean_usize_of_nat(v___x_783_);
v___x_789_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_763_, v_cs_769_, v___x_787_, v___x_788_, v___x_784_, v___y_767_);
return v___x_789_;
}
}
else
{
size_t v___x_790_; size_t v___x_791_; lean_object* v___x_792_; 
v___x_790_ = lean_usize_of_nat(v___x_782_);
lean_dec(v___x_782_);
v___x_791_ = lean_usize_of_nat(v___x_783_);
v___x_792_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_763_, v_cs_769_, v___x_790_, v___x_791_, v___x_784_, v___y_767_);
return v___x_792_;
}
}
}
else
{
lean_object* v_vs_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; uint8_t v___x_797_; 
v_vs_793_ = lean_ctor_get(v_x_764_, 0);
v___x_794_ = lean_usize_to_nat(v_x_765_);
v___x_795_ = lean_array_get_size(v_vs_793_);
v___x_796_ = lean_box(0);
v___x_797_ = lean_nat_dec_lt(v___x_794_, v___x_795_);
if (v___x_797_ == 0)
{
lean_dec(v___x_794_);
lean_dec_ref(v_env_763_);
return v___x_796_;
}
else
{
uint8_t v___x_798_; 
v___x_798_ = lean_nat_dec_le(v___x_795_, v___x_795_);
if (v___x_798_ == 0)
{
if (v___x_797_ == 0)
{
lean_dec(v___x_794_);
lean_dec_ref(v_env_763_);
return v___x_796_;
}
else
{
size_t v___x_799_; size_t v___x_800_; lean_object* v___x_801_; 
v___x_799_ = lean_usize_of_nat(v___x_794_);
lean_dec(v___x_794_);
v___x_800_ = lean_usize_of_nat(v___x_795_);
v___x_801_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_763_, v_vs_793_, v___x_799_, v___x_800_, v___x_796_, v___y_767_);
return v___x_801_;
}
}
else
{
size_t v___x_802_; size_t v___x_803_; lean_object* v___x_804_; 
v___x_802_ = lean_usize_of_nat(v___x_794_);
lean_dec(v___x_794_);
v___x_803_ = lean_usize_of_nat(v___x_795_);
v___x_804_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_763_, v_vs_793_, v___x_802_, v___x_803_, v___x_796_, v___y_767_);
return v___x_804_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg(lean_object* v_env_805_, lean_object* v_t_806_, lean_object* v___y_807_){
_start:
{
lean_object* v_root_809_; lean_object* v_tail_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; uint8_t v___x_815_; 
v_root_809_ = lean_ctor_get(v_t_806_, 0);
v_tail_810_ = lean_ctor_get(v_t_806_, 1);
lean_inc_ref(v_env_805_);
v___x_811_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(v_env_805_, v_root_809_, v___y_807_);
v___x_812_ = lean_unsigned_to_nat(0u);
v___x_813_ = lean_array_get_size(v_tail_810_);
v___x_814_ = lean_box(0);
v___x_815_ = lean_nat_dec_lt(v___x_812_, v___x_813_);
if (v___x_815_ == 0)
{
lean_dec_ref(v_env_805_);
return v___x_814_;
}
else
{
uint8_t v___x_816_; 
v___x_816_ = lean_nat_dec_le(v___x_813_, v___x_813_);
if (v___x_816_ == 0)
{
if (v___x_815_ == 0)
{
lean_dec_ref(v_env_805_);
return v___x_814_;
}
else
{
size_t v___x_817_; size_t v___x_818_; lean_object* v___x_819_; 
v___x_817_ = ((size_t)0ULL);
v___x_818_ = lean_usize_of_nat(v___x_813_);
v___x_819_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_805_, v_tail_810_, v___x_817_, v___x_818_, v___x_814_, v___y_807_);
return v___x_819_;
}
}
else
{
size_t v___x_820_; size_t v___x_821_; lean_object* v___x_822_; 
v___x_820_ = ((size_t)0ULL);
v___x_821_ = lean_usize_of_nat(v___x_813_);
v___x_822_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_805_, v_tail_810_, v___x_820_, v___x_821_, v___x_814_, v___y_807_);
return v___x_822_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg(lean_object* v_env_823_, lean_object* v_t_824_, lean_object* v_start_825_, lean_object* v___y_826_){
_start:
{
lean_object* v___x_828_; uint8_t v___x_829_; 
v___x_828_ = lean_unsigned_to_nat(0u);
v___x_829_ = lean_nat_dec_eq(v_start_825_, v___x_828_);
if (v___x_829_ == 0)
{
lean_object* v_root_830_; lean_object* v_tail_831_; size_t v_shift_832_; lean_object* v_tailOff_833_; uint8_t v___x_834_; 
v_root_830_ = lean_ctor_get(v_t_824_, 0);
v_tail_831_ = lean_ctor_get(v_t_824_, 1);
v_shift_832_ = lean_ctor_get_usize(v_t_824_, 4);
v_tailOff_833_ = lean_ctor_get(v_t_824_, 3);
v___x_834_ = lean_nat_dec_le(v_tailOff_833_, v_start_825_);
if (v___x_834_ == 0)
{
size_t v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; uint8_t v___x_839_; 
v___x_835_ = lean_usize_of_nat(v_start_825_);
lean_inc_ref(v_env_823_);
v___x_836_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(v_env_823_, v_root_830_, v___x_835_, v_shift_832_, v___y_826_);
v___x_837_ = lean_array_get_size(v_tail_831_);
v___x_838_ = lean_box(0);
v___x_839_ = lean_nat_dec_lt(v___x_828_, v___x_837_);
if (v___x_839_ == 0)
{
lean_dec_ref(v_env_823_);
return v___x_838_;
}
else
{
uint8_t v___x_840_; 
v___x_840_ = lean_nat_dec_le(v___x_837_, v___x_837_);
if (v___x_840_ == 0)
{
if (v___x_839_ == 0)
{
lean_dec_ref(v_env_823_);
return v___x_838_;
}
else
{
size_t v___x_841_; size_t v___x_842_; lean_object* v___x_843_; 
v___x_841_ = ((size_t)0ULL);
v___x_842_ = lean_usize_of_nat(v___x_837_);
v___x_843_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_823_, v_tail_831_, v___x_841_, v___x_842_, v___x_838_, v___y_826_);
return v___x_843_;
}
}
else
{
size_t v___x_844_; size_t v___x_845_; lean_object* v___x_846_; 
v___x_844_ = ((size_t)0ULL);
v___x_845_ = lean_usize_of_nat(v___x_837_);
v___x_846_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_823_, v_tail_831_, v___x_844_, v___x_845_, v___x_838_, v___y_826_);
return v___x_846_;
}
}
}
else
{
lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; uint8_t v___x_850_; 
v___x_847_ = lean_nat_sub(v_start_825_, v_tailOff_833_);
v___x_848_ = lean_array_get_size(v_tail_831_);
v___x_849_ = lean_box(0);
v___x_850_ = lean_nat_dec_lt(v___x_847_, v___x_848_);
if (v___x_850_ == 0)
{
lean_dec(v___x_847_);
lean_dec_ref(v_env_823_);
return v___x_849_;
}
else
{
uint8_t v___x_851_; 
v___x_851_ = lean_nat_dec_le(v___x_848_, v___x_848_);
if (v___x_851_ == 0)
{
if (v___x_850_ == 0)
{
lean_dec(v___x_847_);
lean_dec_ref(v_env_823_);
return v___x_849_;
}
else
{
size_t v___x_852_; size_t v___x_853_; lean_object* v___x_854_; 
v___x_852_ = lean_usize_of_nat(v___x_847_);
lean_dec(v___x_847_);
v___x_853_ = lean_usize_of_nat(v___x_848_);
v___x_854_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_823_, v_tail_831_, v___x_852_, v___x_853_, v___x_849_, v___y_826_);
return v___x_854_;
}
}
else
{
size_t v___x_855_; size_t v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_usize_of_nat(v___x_847_);
lean_dec(v___x_847_);
v___x_856_ = lean_usize_of_nat(v___x_848_);
v___x_857_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_823_, v_tail_831_, v___x_855_, v___x_856_, v___x_849_, v___y_826_);
return v___x_857_;
}
}
}
}
else
{
lean_object* v___x_858_; 
v___x_858_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg(v_env_823_, v_t_824_, v___y_826_);
return v___x_858_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(lean_object* v_env_859_, lean_object* v_trees_860_, lean_object* v_a_861_){
_start:
{
lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_863_ = lean_unsigned_to_nat(0u);
v___x_864_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg(v_env_859_, v_trees_860_, v___x_863_, v_a_861_);
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg___boxed(lean_object* v_env_865_, lean_object* v_trees_866_, lean_object* v_a_867_, lean_object* v_a_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_865_, v_trees_866_, v_a_867_);
lean_dec(v_a_867_);
lean_dec_ref(v_trees_866_);
return v_res_869_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg___boxed(lean_object* v_env_870_, lean_object* v_as_871_, lean_object* v_i_872_, lean_object* v_stop_873_, lean_object* v_b_874_, lean_object* v___y_875_, lean_object* v___y_876_){
_start:
{
size_t v_i_boxed_877_; size_t v_stop_boxed_878_; lean_object* v_res_879_; 
v_i_boxed_877_ = lean_unbox_usize(v_i_872_);
lean_dec(v_i_872_);
v_stop_boxed_878_ = lean_unbox_usize(v_stop_873_);
lean_dec(v_stop_873_);
v_res_879_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_870_, v_as_871_, v_i_boxed_877_, v_stop_boxed_878_, v_b_874_, v___y_875_);
lean_dec(v___y_875_);
lean_dec_ref(v_as_871_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_env_880_, lean_object* v_as_881_, lean_object* v_i_882_, lean_object* v_stop_883_, lean_object* v_b_884_, lean_object* v___y_885_, lean_object* v___y_886_){
_start:
{
size_t v_i_boxed_887_; size_t v_stop_boxed_888_; lean_object* v_res_889_; 
v_i_boxed_887_ = lean_unbox_usize(v_i_882_);
lean_dec(v_i_882_);
v_stop_boxed_888_ = lean_unbox_usize(v_stop_883_);
lean_dec(v_stop_883_);
v_res_889_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_880_, v_as_881_, v_i_boxed_887_, v_stop_boxed_888_, v_b_884_, v___y_885_);
lean_dec(v___y_885_);
lean_dec_ref(v_as_881_);
return v_res_889_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg___boxed(lean_object* v_env_890_, lean_object* v_t_891_, lean_object* v___y_892_, lean_object* v___y_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg(v_env_890_, v_t_891_, v___y_892_);
lean_dec(v___y_892_);
lean_dec_ref(v_t_891_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_env_895_, lean_object* v_x_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v_res_899_; 
v_res_899_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(v_env_895_, v_x_896_, v___y_897_);
lean_dec(v___y_897_);
lean_dec_ref(v_x_896_);
return v_res_899_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg___boxed(lean_object* v_env_900_, lean_object* v_t_901_, lean_object* v_start_902_, lean_object* v___y_903_, lean_object* v___y_904_){
_start:
{
lean_object* v_res_905_; 
v_res_905_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg(v_env_900_, v_t_901_, v_start_902_, v___y_903_);
lean_dec(v___y_903_);
lean_dec(v_start_902_);
lean_dec_ref(v_t_901_);
return v_res_905_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg___boxed(lean_object* v_env_906_, lean_object* v_x_907_, lean_object* v_x_908_, lean_object* v_x_909_, lean_object* v___y_910_, lean_object* v___y_911_){
_start:
{
size_t v_x_10240__boxed_912_; size_t v_x_10241__boxed_913_; lean_object* v_res_914_; 
v_x_10240__boxed_912_ = lean_unbox_usize(v_x_908_);
lean_dec(v_x_908_);
v_x_10241__boxed_913_ = lean_unbox_usize(v_x_909_);
lean_dec(v_x_909_);
v_res_914_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(v_env_906_, v_x_907_, v_x_10240__boxed_912_, v_x_10241__boxed_913_, v___y_910_);
lean_dec(v___y_910_);
lean_dec_ref(v_x_907_);
return v_res_914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg___boxed(lean_object* v_env_915_, lean_object* v_x_916_, lean_object* v_a_917_, lean_object* v_a_918_){
_start:
{
lean_object* v_res_919_; 
v_res_919_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg(v_env_915_, v_x_916_, v_a_917_);
lean_dec(v_a_917_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList(lean_object* v_env_920_, lean_object* v_00_u03c9_921_, lean_object* v_trees_922_, lean_object* v_a_923_){
_start:
{
lean_object* v___x_925_; 
v___x_925_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_920_, v_trees_922_, v_a_923_);
return v___x_925_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___boxed(lean_object* v_env_926_, lean_object* v_00_u03c9_927_, lean_object* v_trees_928_, lean_object* v_a_929_, lean_object* v_a_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList(v_env_926_, v_00_u03c9_927_, v_trees_928_, v_a_929_);
lean_dec(v_a_929_);
lean_dec_ref(v_trees_928_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics(lean_object* v_env_932_, lean_object* v_00_u03c9_933_, lean_object* v_x_934_, lean_object* v_a_935_){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___redArg(v_env_932_, v_x_934_, v_a_935_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics___boxed(lean_object* v_env_938_, lean_object* v_00_u03c9_939_, lean_object* v_x_940_, lean_object* v_a_941_, lean_object* v_a_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTactics(v_env_938_, v_00_u03c9_939_, v_x_940_, v_a_941_);
lean_dec(v_a_941_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0(lean_object* v_00_u03c9_944_, lean_object* v_env_945_, lean_object* v_t_946_, lean_object* v_start_947_, lean_object* v___y_948_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___redArg(v_env_945_, v_t_946_, v_start_947_, v___y_948_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0___boxed(lean_object* v_00_u03c9_951_, lean_object* v_env_952_, lean_object* v_t_953_, lean_object* v_start_954_, lean_object* v___y_955_, lean_object* v___y_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_batteries_Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0(v_00_u03c9_951_, v_env_952_, v_t_953_, v_start_954_, v___y_955_);
lean_dec(v___y_955_);
lean_dec(v_start_954_);
lean_dec_ref(v_t_953_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2(lean_object* v_00_u03b2_958_, lean_object* v_m_959_, lean_object* v_a_960_){
_start:
{
lean_object* v___x_961_; 
v___x_961_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___redArg(v_m_959_, v_a_960_);
return v___x_961_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2___boxed(lean_object* v_00_u03b2_962_, lean_object* v_m_963_, lean_object* v_a_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2(v_00_u03b2_962_, v_m_963_, v_a_964_);
lean_dec_ref(v_a_964_);
lean_dec_ref(v_m_963_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3(lean_object* v_00_u03b2_966_, lean_object* v_m_967_, lean_object* v_a_968_){
_start:
{
lean_object* v___x_969_; 
v___x_969_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___redArg(v_m_967_, v_a_968_);
return v___x_969_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3___boxed(lean_object* v_00_u03b2_970_, lean_object* v_m_971_, lean_object* v_a_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3(v_00_u03b2_970_, v_m_971_, v_a_972_);
lean_dec_ref(v_a_972_);
return v_res_973_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4(lean_object* v_00_u03b2_974_, lean_object* v_m_975_, lean_object* v_a_976_, lean_object* v_b_977_){
_start:
{
lean_object* v___x_978_; 
v___x_978_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4___redArg(v_m_975_, v_a_976_, v_b_977_);
return v___x_978_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0(lean_object* v_00_u03c9_979_, lean_object* v_env_980_, lean_object* v_x_981_, size_t v_x_982_, size_t v_x_983_, lean_object* v___y_984_){
_start:
{
lean_object* v___x_986_; 
v___x_986_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___redArg(v_env_980_, v_x_981_, v_x_982_, v_x_983_, v___y_984_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0___boxed(lean_object* v_00_u03c9_987_, lean_object* v_env_988_, lean_object* v_x_989_, lean_object* v_x_990_, lean_object* v_x_991_, lean_object* v___y_992_, lean_object* v___y_993_){
_start:
{
size_t v_x_10759__boxed_994_; size_t v_x_10760__boxed_995_; lean_object* v_res_996_; 
v_x_10759__boxed_994_ = lean_unbox_usize(v_x_990_);
lean_dec(v_x_990_);
v_x_10760__boxed_995_ = lean_unbox_usize(v_x_991_);
lean_dec(v_x_991_);
v_res_996_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0(v_00_u03c9_987_, v_env_988_, v_x_989_, v_x_10759__boxed_994_, v_x_10760__boxed_995_, v___y_992_);
lean_dec(v___y_992_);
lean_dec_ref(v_x_989_);
return v_res_996_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1(lean_object* v_00_u03c9_997_, lean_object* v_env_998_, lean_object* v_as_999_, size_t v_i_1000_, size_t v_stop_1001_, lean_object* v_b_1002_, lean_object* v___y_1003_){
_start:
{
lean_object* v___x_1005_; 
v___x_1005_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___redArg(v_env_998_, v_as_999_, v_i_1000_, v_stop_1001_, v_b_1002_, v___y_1003_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1___boxed(lean_object* v_00_u03c9_1006_, lean_object* v_env_1007_, lean_object* v_as_1008_, lean_object* v_i_1009_, lean_object* v_stop_1010_, lean_object* v_b_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_){
_start:
{
size_t v_i_boxed_1014_; size_t v_stop_boxed_1015_; lean_object* v_res_1016_; 
v_i_boxed_1014_ = lean_unbox_usize(v_i_1009_);
lean_dec(v_i_1009_);
v_stop_boxed_1015_ = lean_unbox_usize(v_stop_1010_);
lean_dec(v_stop_1010_);
v_res_1016_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__1(v_00_u03c9_1006_, v_env_1007_, v_as_1008_, v_i_boxed_1014_, v_stop_boxed_1015_, v_b_1011_, v___y_1012_);
lean_dec(v___y_1012_);
lean_dec_ref(v_as_1008_);
return v_res_1016_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2(lean_object* v_00_u03c9_1017_, lean_object* v_env_1018_, lean_object* v_t_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v___x_1022_; 
v___x_1022_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___redArg(v_env_1018_, v_t_1019_, v___y_1020_);
return v___x_1022_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2___boxed(lean_object* v_00_u03c9_1023_, lean_object* v_env_1024_, lean_object* v_t_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_batteries_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__2(v_00_u03c9_1023_, v_env_1024_, v_t_1025_, v___y_1026_);
lean_dec(v___y_1026_);
lean_dec_ref(v_t_1025_);
return v_res_1028_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5(lean_object* v_00_u03b2_1029_, lean_object* v_a_1030_, lean_object* v_x_1031_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___redArg(v_a_1030_, v_x_1031_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5___boxed(lean_object* v_00_u03b2_1033_, lean_object* v_a_1034_, lean_object* v_x_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__2_spec__5(v_00_u03b2_1033_, v_a_1034_, v_x_1035_);
lean_dec(v_x_1035_);
lean_dec_ref(v_a_1034_);
return v_res_1036_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7(lean_object* v_00_u03b2_1037_, lean_object* v_a_1038_, lean_object* v_x_1039_){
_start:
{
uint8_t v___x_1040_; 
v___x_1040_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___redArg(v_a_1038_, v_x_1039_);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7___boxed(lean_object* v_00_u03b2_1041_, lean_object* v_a_1042_, lean_object* v_x_1043_){
_start:
{
uint8_t v_res_1044_; lean_object* v_r_1045_; 
v_res_1044_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__7(v_00_u03b2_1041_, v_a_1042_, v_x_1043_);
lean_dec(v_x_1043_);
lean_dec_ref(v_a_1042_);
v_r_1045_ = lean_box(v_res_1044_);
return v_r_1045_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8(lean_object* v_00_u03b2_1046_, lean_object* v_a_1047_, lean_object* v_x_1048_){
_start:
{
lean_object* v___x_1049_; 
v___x_1049_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___redArg(v_a_1047_, v_x_1048_);
return v___x_1049_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8___boxed(lean_object* v_00_u03b2_1050_, lean_object* v_a_1051_, lean_object* v_x_1052_){
_start:
{
lean_object* v_res_1053_; 
v_res_1053_ = lp_batteries_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__3_spec__8(v_00_u03b2_1050_, v_a_1051_, v_x_1052_);
lean_dec_ref(v_a_1051_);
return v_res_1053_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10(lean_object* v_00_u03b2_1054_, lean_object* v_data_1055_){
_start:
{
lean_object* v___x_1056_; 
v___x_1056_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10___redArg(v_data_1055_);
return v___x_1056_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11(lean_object* v_00_u03b2_1057_, lean_object* v_a_1058_, lean_object* v_b_1059_, lean_object* v_x_1060_){
_start:
{
lean_object* v___x_1061_; 
v___x_1061_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__11___redArg(v_a_1058_, v_b_1059_, v_x_1060_);
return v___x_1061_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2(lean_object* v_00_u03c9_1062_, lean_object* v_env_1063_, lean_object* v_x_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v___x_1067_; 
v___x_1067_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___redArg(v_env_1063_, v_x_1064_, v___y_1065_);
return v___x_1067_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03c9_1068_, lean_object* v_env_1069_, lean_object* v_x_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v_res_1073_; 
v_res_1073_ = lp_batteries_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__2(v_00_u03c9_1068_, v_env_1069_, v_x_1070_, v___y_1071_);
lean_dec(v___y_1071_);
lean_dec_ref(v_x_1070_);
return v_res_1073_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3(lean_object* v_00_u03c9_1074_, lean_object* v_env_1075_, lean_object* v_as_1076_, size_t v_i_1077_, size_t v_stop_1078_, lean_object* v_b_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___redArg(v_env_1075_, v_as_1076_, v_i_1077_, v_stop_1078_, v_b_1079_, v___y_1080_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03c9_1083_, lean_object* v_env_1084_, lean_object* v_as_1085_, lean_object* v_i_1086_, lean_object* v_stop_1087_, lean_object* v_b_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
size_t v_i_boxed_1091_; size_t v_stop_boxed_1092_; lean_object* v_res_1093_; 
v_i_boxed_1091_ = lean_unbox_usize(v_i_1086_);
lean_dec(v_i_1086_);
v_stop_boxed_1092_ = lean_unbox_usize(v_stop_1087_);
lean_dec(v_stop_1087_);
v_res_1093_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList_spec__0_spec__0_spec__3(v_00_u03c9_1083_, v_env_1084_, v_as_1085_, v_i_boxed_1091_, v_stop_boxed_1092_, v_b_1088_, v___y_1089_);
lean_dec(v___y_1089_);
lean_dec_ref(v_as_1085_);
return v_res_1093_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13(lean_object* v_00_u03b2_1094_, lean_object* v_i_1095_, lean_object* v_source_1096_, lean_object* v_target_1097_){
_start:
{
lean_object* v___x_1098_; 
v___x_1098_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13___redArg(v_i_1095_, v_source_1096_, v_target_1097_);
return v___x_1098_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14(lean_object* v_00_u03b2_1099_, lean_object* v_x_1100_, lean_object* v_x_1101_){
_start:
{
lean_object* v___x_1102_; 
v___x_1102_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Linter_UnnecessarySeqFocus_markUsedTactics_spec__4_spec__10_spec__13_spec__14___redArg(v_x_1100_, v_x_1101_);
return v___x_1102_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__0(lean_object* v_a_1103_){
_start:
{
lean_object* v___x_1104_; 
v___x_1104_ = lean_nat_to_int(v_a_1103_);
return v___x_1104_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg(lean_object* v___y_1105_){
_start:
{
lean_object* v___x_1107_; lean_object* v_infoState_1108_; lean_object* v_trees_1109_; lean_object* v___x_1110_; 
v___x_1107_ = lean_st_ref_get(v___y_1105_);
v_infoState_1108_ = lean_ctor_get(v___x_1107_, 8);
lean_inc_ref(v_infoState_1108_);
lean_dec(v___x_1107_);
v_trees_1109_ = lean_ctor_get(v_infoState_1108_, 2);
lean_inc_ref(v_trees_1109_);
lean_dec_ref(v_infoState_1108_);
v___x_1110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1110_, 0, v_trees_1109_);
return v___x_1110_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg___boxed(lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
lean_object* v_res_1113_; 
v_res_1113_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg(v___y_1111_);
lean_dec(v___y_1111_);
return v_res_1113_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3(lean_object* v___y_1114_, lean_object* v___y_1115_){
_start:
{
lean_object* v___x_1117_; 
v___x_1117_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg(v___y_1115_);
return v___x_1117_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___boxed(lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
lean_object* v_res_1121_; 
v_res_1121_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3(v___y_1118_, v___y_1119_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
return v_res_1121_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; 
v___x_1122_ = lean_box(0);
v___x_1123_ = lean_unsigned_to_nat(16u);
v___x_1124_ = lean_mk_array(v___x_1123_, v___x_1122_);
return v___x_1124_;
}
}
static lean_object* _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1125_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__0);
v___x_1126_ = lean_unsigned_to_nat(0u);
v___x_1127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1126_);
lean_ctor_set(v___x_1127_, 1, v___x_1125_);
return v___x_1127_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0(lean_object* v_stx_1128_, lean_object* v_env_1129_, lean_object* v_a_1130_, lean_object* v_x_1131_){
_start:
{
lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; 
v___x_1133_ = lean_obj_once(&lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1, &lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1_once, _init_lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___closed__1);
v___x_1134_ = lean_st_mk_ref(v___x_1133_);
v___x_1135_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getTactics___redArg(v_stx_1128_, v___x_1134_);
v___x_1136_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_markUsedTacticsList___redArg(v_env_1129_, v_a_1130_, v___x_1134_);
v___x_1137_ = lean_st_ref_get(v___x_1134_);
lean_dec(v___x_1134_);
v___x_1138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1138_, 0, v___x_1136_);
lean_ctor_set(v___x_1138_, 1, v___x_1137_);
return v___x_1138_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___boxed(lean_object* v_stx_1139_, lean_object* v_env_1140_, lean_object* v_a_1141_, lean_object* v_x_1142_, lean_object* v___y_1143_){
_start:
{
lean_object* v_res_1144_; 
v_res_1144_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0(v_stx_1139_, v_env_1140_, v_a_1141_, v_x_1142_);
lean_dec_ref(v_a_1141_);
return v_res_1144_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg(lean_object* v_o_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v___x_1148_; lean_object* v_env_1149_; lean_object* v___x_1150_; lean_object* v_toEnvExtension_1151_; lean_object* v_asyncMode_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v_merged_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1164_; 
v___x_1148_ = lean_st_ref_get(v___y_1146_);
v_env_1149_ = lean_ctor_get(v___x_1148_, 0);
lean_inc_ref(v_env_1149_);
lean_dec(v___x_1148_);
v___x_1150_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1151_ = lean_ctor_get(v___x_1150_, 0);
v_asyncMode_1152_ = lean_ctor_get(v_toEnvExtension_1151_, 2);
v___x_1153_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1154_ = lean_box(0);
v___x_1155_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1153_, v___x_1150_, v_env_1149_, v_asyncMode_1152_, v___x_1154_);
v_merged_1156_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1164_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1164_ == 0)
{
lean_object* v_unused_1165_; 
v_unused_1165_ = lean_ctor_get(v___x_1155_, 1);
lean_dec(v_unused_1165_);
v___x_1158_ = v___x_1155_;
v_isShared_1159_ = v_isSharedCheck_1164_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_merged_1156_);
lean_dec(v___x_1155_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1164_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1161_; 
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 1, v_merged_1156_);
lean_ctor_set(v___x_1158_, 0, v_o_1145_);
v___x_1161_ = v___x_1158_;
goto v_reusejp_1160_;
}
else
{
lean_object* v_reuseFailAlloc_1163_; 
v_reuseFailAlloc_1163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1163_, 0, v_o_1145_);
lean_ctor_set(v_reuseFailAlloc_1163_, 1, v_merged_1156_);
v___x_1161_ = v_reuseFailAlloc_1163_;
goto v_reusejp_1160_;
}
v_reusejp_1160_:
{
lean_object* v___x_1162_; 
v___x_1162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1162_, 0, v___x_1161_);
return v___x_1162_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg___boxed(lean_object* v_o_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg(v_o_1166_, v___y_1167_);
lean_dec(v___y_1167_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2(lean_object* v___y_1170_, lean_object* v___y_1171_){
_start:
{
lean_object* v___x_1173_; lean_object* v_scopes_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v_opts_1177_; lean_object* v___x_1178_; 
v___x_1173_ = lean_st_ref_get(v___y_1171_);
v_scopes_1174_ = lean_ctor_get(v___x_1173_, 2);
lean_inc(v_scopes_1174_);
lean_dec(v___x_1173_);
v___x_1175_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1176_ = l_List_head_x21___redArg(v___x_1175_, v_scopes_1174_);
lean_dec(v_scopes_1174_);
v_opts_1177_ = lean_ctor_get(v___x_1176_, 1);
lean_inc_ref(v_opts_1177_);
lean_dec(v___x_1176_);
v___x_1178_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg(v_opts_1177_, v___y_1171_);
return v___x_1178_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2___boxed(lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2(v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
return v_res_1182_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11(lean_object* v_opts_1183_, lean_object* v_opt_1184_){
_start:
{
lean_object* v_name_1185_; lean_object* v_defValue_1186_; lean_object* v_map_1187_; lean_object* v___x_1188_; 
v_name_1185_ = lean_ctor_get(v_opt_1184_, 0);
v_defValue_1186_ = lean_ctor_get(v_opt_1184_, 1);
v_map_1187_ = lean_ctor_get(v_opts_1183_, 0);
v___x_1188_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1187_, v_name_1185_);
if (lean_obj_tag(v___x_1188_) == 0)
{
uint8_t v___x_1189_; 
v___x_1189_ = lean_unbox(v_defValue_1186_);
return v___x_1189_;
}
else
{
lean_object* v_val_1190_; 
v_val_1190_ = lean_ctor_get(v___x_1188_, 0);
lean_inc(v_val_1190_);
lean_dec_ref_known(v___x_1188_, 1);
if (lean_obj_tag(v_val_1190_) == 1)
{
uint8_t v_v_1191_; 
v_v_1191_ = lean_ctor_get_uint8(v_val_1190_, 0);
lean_dec_ref_known(v_val_1190_, 0);
return v_v_1191_;
}
else
{
uint8_t v___x_1192_; 
lean_dec(v_val_1190_);
v___x_1192_ = lean_unbox(v_defValue_1186_);
return v___x_1192_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11___boxed(lean_object* v_opts_1193_, lean_object* v_opt_1194_){
_start:
{
uint8_t v_res_1195_; lean_object* v_r_1196_; 
v_res_1195_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11(v_opts_1193_, v_opt_1194_);
lean_dec_ref(v_opt_1194_);
lean_dec_ref(v_opts_1193_);
v_r_1196_ = lean_box(v_res_1195_);
return v_r_1196_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0(void){
_start:
{
lean_object* v___x_1197_; 
v___x_1197_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1197_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1(void){
_start:
{
lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1198_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__0);
v___x_1199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1198_);
return v___x_1199_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2(void){
_start:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1200_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1);
v___x_1201_ = lean_unsigned_to_nat(0u);
v___x_1202_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
lean_ctor_set(v___x_1202_, 1, v___x_1201_);
lean_ctor_set(v___x_1202_, 2, v___x_1201_);
lean_ctor_set(v___x_1202_, 3, v___x_1201_);
lean_ctor_set(v___x_1202_, 4, v___x_1200_);
lean_ctor_set(v___x_1202_, 5, v___x_1200_);
lean_ctor_set(v___x_1202_, 6, v___x_1200_);
lean_ctor_set(v___x_1202_, 7, v___x_1200_);
lean_ctor_set(v___x_1202_, 8, v___x_1200_);
lean_ctor_set(v___x_1202_, 9, v___x_1200_);
return v___x_1202_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3(void){
_start:
{
lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1203_ = lean_unsigned_to_nat(32u);
v___x_1204_ = lean_mk_empty_array_with_capacity(v___x_1203_);
v___x_1205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
return v___x_1205_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4(void){
_start:
{
size_t v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1206_ = ((size_t)5ULL);
v___x_1207_ = lean_unsigned_to_nat(0u);
v___x_1208_ = lean_unsigned_to_nat(32u);
v___x_1209_ = lean_mk_empty_array_with_capacity(v___x_1208_);
v___x_1210_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__3);
v___x_1211_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
lean_ctor_set(v___x_1211_, 1, v___x_1209_);
lean_ctor_set(v___x_1211_, 2, v___x_1207_);
lean_ctor_set(v___x_1211_, 3, v___x_1207_);
lean_ctor_set_usize(v___x_1211_, 4, v___x_1206_);
return v___x_1211_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5(void){
_start:
{
lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; 
v___x_1212_ = lean_box(1);
v___x_1213_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__4);
v___x_1214_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__1);
v___x_1215_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1215_, 0, v___x_1214_);
lean_ctor_set(v___x_1215_, 1, v___x_1213_);
lean_ctor_set(v___x_1215_, 2, v___x_1212_);
return v___x_1215_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg(lean_object* v_msgData_1216_, lean_object* v___y_1217_){
_start:
{
lean_object* v___x_1219_; lean_object* v_env_1220_; lean_object* v___x_1221_; lean_object* v_scopes_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v_opts_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___x_1219_ = lean_st_ref_get(v___y_1217_);
v_env_1220_ = lean_ctor_get(v___x_1219_, 0);
lean_inc_ref(v_env_1220_);
lean_dec(v___x_1219_);
v___x_1221_ = lean_st_ref_get(v___y_1217_);
v_scopes_1222_ = lean_ctor_get(v___x_1221_, 2);
lean_inc(v_scopes_1222_);
lean_dec(v___x_1221_);
v___x_1223_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1224_ = l_List_head_x21___redArg(v___x_1223_, v_scopes_1222_);
lean_dec(v_scopes_1222_);
v_opts_1225_ = lean_ctor_get(v___x_1224_, 1);
lean_inc_ref(v_opts_1225_);
lean_dec(v___x_1224_);
v___x_1226_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__2);
v___x_1227_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___closed__5);
v___x_1228_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1228_, 0, v_env_1220_);
lean_ctor_set(v___x_1228_, 1, v___x_1226_);
lean_ctor_set(v___x_1228_, 2, v___x_1227_);
lean_ctor_set(v___x_1228_, 3, v_opts_1225_);
v___x_1229_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1229_, 0, v___x_1228_);
lean_ctor_set(v___x_1229_, 1, v_msgData_1216_);
v___x_1230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1230_, 0, v___x_1229_);
return v___x_1230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg___boxed(lean_object* v_msgData_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v_res_1234_; 
v_res_1234_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg(v_msgData_1231_, v___y_1232_);
lean_dec(v___y_1232_);
return v_res_1234_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0(uint8_t v___y_1236_, uint8_t v_suppressElabErrors_1237_, lean_object* v_x_1238_){
_start:
{
if (lean_obj_tag(v_x_1238_) == 1)
{
lean_object* v_pre_1239_; 
v_pre_1239_ = lean_ctor_get(v_x_1238_, 0);
if (lean_obj_tag(v_pre_1239_) == 0)
{
lean_object* v_str_1240_; lean_object* v___x_1241_; uint8_t v___x_1242_; 
v_str_1240_ = lean_ctor_get(v_x_1238_, 1);
v___x_1241_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___closed__0));
v___x_1242_ = lean_string_dec_eq(v_str_1240_, v___x_1241_);
if (v___x_1242_ == 0)
{
return v___y_1236_;
}
else
{
return v_suppressElabErrors_1237_;
}
}
else
{
return v___y_1236_;
}
}
else
{
return v___y_1236_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___boxed(lean_object* v___y_1243_, lean_object* v_suppressElabErrors_1244_, lean_object* v_x_1245_){
_start:
{
uint8_t v___y_8622__boxed_1246_; uint8_t v_suppressElabErrors_boxed_1247_; uint8_t v_res_1248_; lean_object* v_r_1249_; 
v___y_8622__boxed_1246_ = lean_unbox(v___y_1243_);
v_suppressElabErrors_boxed_1247_ = lean_unbox(v_suppressElabErrors_1244_);
v_res_1248_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0(v___y_8622__boxed_1246_, v_suppressElabErrors_boxed_1247_, v_x_1245_);
lean_dec(v_x_1245_);
v_r_1249_ = lean_box(v_res_1248_);
return v_r_1249_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3(lean_object* v_ref_1251_, lean_object* v_msgData_1252_, uint8_t v_severity_1253_, uint8_t v_isSilent_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_){
_start:
{
uint8_t v___y_1259_; lean_object* v___y_1260_; lean_object* v___y_1261_; lean_object* v___y_1262_; lean_object* v___y_1263_; lean_object* v___y_1264_; uint8_t v___y_1265_; lean_object* v___y_1266_; uint8_t v___y_1323_; uint8_t v___y_1324_; lean_object* v___y_1325_; uint8_t v___y_1326_; lean_object* v___y_1327_; uint8_t v___y_1351_; uint8_t v___y_1352_; lean_object* v___y_1353_; uint8_t v___y_1354_; lean_object* v___y_1355_; uint8_t v___y_1359_; uint8_t v___y_1360_; uint8_t v___y_1361_; uint8_t v___x_1376_; uint8_t v___y_1378_; uint8_t v___y_1379_; uint8_t v___y_1380_; uint8_t v___y_1382_; uint8_t v___x_1394_; 
v___x_1376_ = 2;
v___x_1394_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1253_, v___x_1376_);
if (v___x_1394_ == 0)
{
v___y_1382_ = v___x_1394_;
goto v___jp_1381_;
}
else
{
uint8_t v___x_1395_; 
lean_inc_ref(v_msgData_1252_);
v___x_1395_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1252_);
v___y_1382_ = v___x_1395_;
goto v___jp_1381_;
}
v___jp_1258_:
{
lean_object* v___x_1267_; 
v___x_1267_ = l_Lean_Elab_Command_getScope___redArg(v___y_1266_);
if (lean_obj_tag(v___x_1267_) == 0)
{
lean_object* v_a_1268_; lean_object* v___x_1269_; 
v_a_1268_ = lean_ctor_get(v___x_1267_, 0);
lean_inc(v_a_1268_);
lean_dec_ref_known(v___x_1267_, 1);
v___x_1269_ = l_Lean_Elab_Command_getScope___redArg(v___y_1266_);
if (lean_obj_tag(v___x_1269_) == 0)
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1305_; 
v_a_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1305_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1305_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1305_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1274_; lean_object* v_currNamespace_1275_; lean_object* v_openDecls_1276_; lean_object* v_env_1277_; lean_object* v_messages_1278_; lean_object* v_scopes_1279_; lean_object* v_usedQuotCtxts_1280_; lean_object* v_nextMacroScope_1281_; lean_object* v_maxRecDepth_1282_; lean_object* v_ngen_1283_; lean_object* v_auxDeclNGen_1284_; lean_object* v_infoState_1285_; lean_object* v_traceState_1286_; lean_object* v_snapshotTasks_1287_; lean_object* v_prevLinterStates_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1304_; 
v___x_1274_ = lean_st_ref_take(v___y_1266_);
v_currNamespace_1275_ = lean_ctor_get(v_a_1268_, 2);
lean_inc(v_currNamespace_1275_);
lean_dec(v_a_1268_);
v_openDecls_1276_ = lean_ctor_get(v_a_1270_, 3);
lean_inc(v_openDecls_1276_);
lean_dec(v_a_1270_);
v_env_1277_ = lean_ctor_get(v___x_1274_, 0);
v_messages_1278_ = lean_ctor_get(v___x_1274_, 1);
v_scopes_1279_ = lean_ctor_get(v___x_1274_, 2);
v_usedQuotCtxts_1280_ = lean_ctor_get(v___x_1274_, 3);
v_nextMacroScope_1281_ = lean_ctor_get(v___x_1274_, 4);
v_maxRecDepth_1282_ = lean_ctor_get(v___x_1274_, 5);
v_ngen_1283_ = lean_ctor_get(v___x_1274_, 6);
v_auxDeclNGen_1284_ = lean_ctor_get(v___x_1274_, 7);
v_infoState_1285_ = lean_ctor_get(v___x_1274_, 8);
v_traceState_1286_ = lean_ctor_get(v___x_1274_, 9);
v_snapshotTasks_1287_ = lean_ctor_get(v___x_1274_, 10);
v_prevLinterStates_1288_ = lean_ctor_get(v___x_1274_, 11);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1274_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1290_ = v___x_1274_;
v_isShared_1291_ = v_isSharedCheck_1304_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_prevLinterStates_1288_);
lean_inc(v_snapshotTasks_1287_);
lean_inc(v_traceState_1286_);
lean_inc(v_infoState_1285_);
lean_inc(v_auxDeclNGen_1284_);
lean_inc(v_ngen_1283_);
lean_inc(v_maxRecDepth_1282_);
lean_inc(v_nextMacroScope_1281_);
lean_inc(v_usedQuotCtxts_1280_);
lean_inc(v_scopes_1279_);
lean_inc(v_messages_1278_);
lean_inc(v_env_1277_);
lean_dec(v___x_1274_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1304_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1297_; 
v___x_1292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1292_, 0, v_currNamespace_1275_);
lean_ctor_set(v___x_1292_, 1, v_openDecls_1276_);
v___x_1293_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1293_, 0, v___x_1292_);
lean_ctor_set(v___x_1293_, 1, v___y_1263_);
lean_inc_ref(v___y_1264_);
lean_inc_ref(v___y_1262_);
v___x_1294_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1294_, 0, v___y_1262_);
lean_ctor_set(v___x_1294_, 1, v___y_1260_);
lean_ctor_set(v___x_1294_, 2, v___y_1261_);
lean_ctor_set(v___x_1294_, 3, v___y_1264_);
lean_ctor_set(v___x_1294_, 4, v___x_1293_);
lean_ctor_set_uint8(v___x_1294_, sizeof(void*)*5, v___y_1265_);
lean_ctor_set_uint8(v___x_1294_, sizeof(void*)*5 + 1, v___y_1259_);
lean_ctor_set_uint8(v___x_1294_, sizeof(void*)*5 + 2, v_isSilent_1254_);
v___x_1295_ = l_Lean_MessageLog_add(v___x_1294_, v_messages_1278_);
if (v_isShared_1291_ == 0)
{
lean_ctor_set(v___x_1290_, 1, v___x_1295_);
v___x_1297_ = v___x_1290_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_env_1277_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v___x_1295_);
lean_ctor_set(v_reuseFailAlloc_1303_, 2, v_scopes_1279_);
lean_ctor_set(v_reuseFailAlloc_1303_, 3, v_usedQuotCtxts_1280_);
lean_ctor_set(v_reuseFailAlloc_1303_, 4, v_nextMacroScope_1281_);
lean_ctor_set(v_reuseFailAlloc_1303_, 5, v_maxRecDepth_1282_);
lean_ctor_set(v_reuseFailAlloc_1303_, 6, v_ngen_1283_);
lean_ctor_set(v_reuseFailAlloc_1303_, 7, v_auxDeclNGen_1284_);
lean_ctor_set(v_reuseFailAlloc_1303_, 8, v_infoState_1285_);
lean_ctor_set(v_reuseFailAlloc_1303_, 9, v_traceState_1286_);
lean_ctor_set(v_reuseFailAlloc_1303_, 10, v_snapshotTasks_1287_);
lean_ctor_set(v_reuseFailAlloc_1303_, 11, v_prevLinterStates_1288_);
v___x_1297_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1301_; 
v___x_1298_ = lean_st_ref_set(v___y_1266_, v___x_1297_);
v___x_1299_ = lean_box(0);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 0, v___x_1299_);
v___x_1301_ = v___x_1272_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v___x_1299_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
}
else
{
lean_object* v_a_1306_; lean_object* v___x_1308_; uint8_t v_isShared_1309_; uint8_t v_isSharedCheck_1313_; 
lean_dec(v_a_1268_);
lean_dec_ref(v___y_1263_);
lean_dec(v___y_1261_);
lean_dec_ref(v___y_1260_);
v_a_1306_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1313_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1313_ == 0)
{
v___x_1308_ = v___x_1269_;
v_isShared_1309_ = v_isSharedCheck_1313_;
goto v_resetjp_1307_;
}
else
{
lean_inc(v_a_1306_);
lean_dec(v___x_1269_);
v___x_1308_ = lean_box(0);
v_isShared_1309_ = v_isSharedCheck_1313_;
goto v_resetjp_1307_;
}
v_resetjp_1307_:
{
lean_object* v___x_1311_; 
if (v_isShared_1309_ == 0)
{
v___x_1311_ = v___x_1308_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1312_; 
v_reuseFailAlloc_1312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1312_, 0, v_a_1306_);
v___x_1311_ = v_reuseFailAlloc_1312_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
return v___x_1311_;
}
}
}
}
else
{
lean_object* v_a_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1321_; 
lean_dec_ref(v___y_1263_);
lean_dec(v___y_1261_);
lean_dec_ref(v___y_1260_);
v_a_1314_ = lean_ctor_get(v___x_1267_, 0);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1267_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1316_ = v___x_1267_;
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_a_1314_);
lean_dec(v___x_1267_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___x_1319_; 
if (v_isShared_1317_ == 0)
{
v___x_1319_ = v___x_1316_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v_a_1314_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
}
}
v___jp_1322_:
{
lean_object* v_fileName_1328_; lean_object* v_fileMap_1329_; uint8_t v_suppressElabErrors_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1349_; 
v_fileName_1328_ = lean_ctor_get(v___y_1255_, 0);
v_fileMap_1329_ = lean_ctor_get(v___y_1255_, 1);
v_suppressElabErrors_1330_ = lean_ctor_get_uint8(v___y_1255_, sizeof(void*)*10);
v___x_1331_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1252_);
v___x_1332_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg(v___x_1331_, v___y_1256_);
v_a_1333_ = lean_ctor_get(v___x_1332_, 0);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1332_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1335_ = v___x_1332_;
v_isShared_1336_ = v_isSharedCheck_1349_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1332_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1349_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; 
lean_inc_ref_n(v_fileMap_1329_, 2);
v___x_1337_ = l_Lean_FileMap_toPosition(v_fileMap_1329_, v___y_1325_);
lean_dec(v___y_1325_);
v___x_1338_ = l_Lean_FileMap_toPosition(v_fileMap_1329_, v___y_1327_);
lean_dec(v___y_1327_);
v___x_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1339_, 0, v___x_1338_);
v___x_1340_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___closed__0));
if (v_suppressElabErrors_1330_ == 0)
{
lean_del_object(v___x_1335_);
v___y_1259_ = v___y_1324_;
v___y_1260_ = v___x_1337_;
v___y_1261_ = v___x_1339_;
v___y_1262_ = v_fileName_1328_;
v___y_1263_ = v_a_1333_;
v___y_1264_ = v___x_1340_;
v___y_1265_ = v___y_1326_;
v___y_1266_ = v___y_1256_;
goto v___jp_1258_;
}
else
{
lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___f_1343_; uint8_t v___x_1344_; 
v___x_1341_ = lean_box(v___y_1323_);
v___x_1342_ = lean_box(v_suppressElabErrors_1330_);
v___f_1343_ = lean_alloc_closure((void*)(lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1343_, 0, v___x_1341_);
lean_closure_set(v___f_1343_, 1, v___x_1342_);
lean_inc(v_a_1333_);
v___x_1344_ = l_Lean_MessageData_hasTag(v___f_1343_, v_a_1333_);
if (v___x_1344_ == 0)
{
lean_object* v___x_1345_; lean_object* v___x_1347_; 
lean_dec_ref_known(v___x_1339_, 1);
lean_dec_ref(v___x_1337_);
lean_dec(v_a_1333_);
v___x_1345_ = lean_box(0);
if (v_isShared_1336_ == 0)
{
lean_ctor_set(v___x_1335_, 0, v___x_1345_);
v___x_1347_ = v___x_1335_;
goto v_reusejp_1346_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v___x_1345_);
v___x_1347_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1346_;
}
v_reusejp_1346_:
{
return v___x_1347_;
}
}
else
{
lean_del_object(v___x_1335_);
v___y_1259_ = v___y_1324_;
v___y_1260_ = v___x_1337_;
v___y_1261_ = v___x_1339_;
v___y_1262_ = v_fileName_1328_;
v___y_1263_ = v_a_1333_;
v___y_1264_ = v___x_1340_;
v___y_1265_ = v___y_1326_;
v___y_1266_ = v___y_1256_;
goto v___jp_1258_;
}
}
}
}
v___jp_1350_:
{
lean_object* v___x_1356_; 
v___x_1356_ = l_Lean_Syntax_getTailPos_x3f(v___y_1353_, v___y_1354_);
lean_dec(v___y_1353_);
if (lean_obj_tag(v___x_1356_) == 0)
{
lean_inc(v___y_1355_);
v___y_1323_ = v___y_1351_;
v___y_1324_ = v___y_1352_;
v___y_1325_ = v___y_1355_;
v___y_1326_ = v___y_1354_;
v___y_1327_ = v___y_1355_;
goto v___jp_1322_;
}
else
{
lean_object* v_val_1357_; 
v_val_1357_ = lean_ctor_get(v___x_1356_, 0);
lean_inc(v_val_1357_);
lean_dec_ref_known(v___x_1356_, 1);
v___y_1323_ = v___y_1351_;
v___y_1324_ = v___y_1352_;
v___y_1325_ = v___y_1355_;
v___y_1326_ = v___y_1354_;
v___y_1327_ = v_val_1357_;
goto v___jp_1322_;
}
}
v___jp_1358_:
{
lean_object* v___x_1362_; 
v___x_1362_ = l_Lean_Elab_Command_getRef___redArg(v___y_1255_);
if (lean_obj_tag(v___x_1362_) == 0)
{
lean_object* v_a_1363_; lean_object* v_ref_1364_; lean_object* v___x_1365_; 
v_a_1363_ = lean_ctor_get(v___x_1362_, 0);
lean_inc(v_a_1363_);
lean_dec_ref_known(v___x_1362_, 1);
v_ref_1364_ = l_Lean_replaceRef(v_ref_1251_, v_a_1363_);
lean_dec(v_a_1363_);
v___x_1365_ = l_Lean_Syntax_getPos_x3f(v_ref_1364_, v___y_1360_);
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v___x_1366_; 
v___x_1366_ = lean_unsigned_to_nat(0u);
v___y_1351_ = v___y_1359_;
v___y_1352_ = v___y_1361_;
v___y_1353_ = v_ref_1364_;
v___y_1354_ = v___y_1360_;
v___y_1355_ = v___x_1366_;
goto v___jp_1350_;
}
else
{
lean_object* v_val_1367_; 
v_val_1367_ = lean_ctor_get(v___x_1365_, 0);
lean_inc(v_val_1367_);
lean_dec_ref_known(v___x_1365_, 1);
v___y_1351_ = v___y_1359_;
v___y_1352_ = v___y_1361_;
v___y_1353_ = v_ref_1364_;
v___y_1354_ = v___y_1360_;
v___y_1355_ = v_val_1367_;
goto v___jp_1350_;
}
}
else
{
lean_object* v_a_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1375_; 
lean_dec_ref(v_msgData_1252_);
v_a_1368_ = lean_ctor_get(v___x_1362_, 0);
v_isSharedCheck_1375_ = !lean_is_exclusive(v___x_1362_);
if (v_isSharedCheck_1375_ == 0)
{
v___x_1370_ = v___x_1362_;
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_a_1368_);
lean_dec(v___x_1362_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1373_; 
if (v_isShared_1371_ == 0)
{
v___x_1373_ = v___x_1370_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1374_; 
v_reuseFailAlloc_1374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1374_, 0, v_a_1368_);
v___x_1373_ = v_reuseFailAlloc_1374_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
return v___x_1373_;
}
}
}
}
v___jp_1377_:
{
if (v___y_1380_ == 0)
{
v___y_1359_ = v___y_1378_;
v___y_1360_ = v___y_1379_;
v___y_1361_ = v_severity_1253_;
goto v___jp_1358_;
}
else
{
v___y_1359_ = v___y_1378_;
v___y_1360_ = v___y_1379_;
v___y_1361_ = v___x_1376_;
goto v___jp_1358_;
}
}
v___jp_1381_:
{
if (v___y_1382_ == 0)
{
lean_object* v___x_1383_; lean_object* v_scopes_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v_opts_1387_; uint8_t v___x_1388_; uint8_t v___x_1389_; 
v___x_1383_ = lean_st_ref_get(v___y_1256_);
v_scopes_1384_ = lean_ctor_get(v___x_1383_, 2);
lean_inc(v_scopes_1384_);
lean_dec(v___x_1383_);
v___x_1385_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1386_ = l_List_head_x21___redArg(v___x_1385_, v_scopes_1384_);
lean_dec(v_scopes_1384_);
v_opts_1387_ = lean_ctor_get(v___x_1386_, 1);
lean_inc_ref(v_opts_1387_);
lean_dec(v___x_1386_);
v___x_1388_ = 1;
v___x_1389_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1253_, v___x_1388_);
if (v___x_1389_ == 0)
{
lean_dec_ref(v_opts_1387_);
v___y_1378_ = v___y_1382_;
v___y_1379_ = v___y_1382_;
v___y_1380_ = v___x_1389_;
goto v___jp_1377_;
}
else
{
lean_object* v___x_1390_; uint8_t v___x_1391_; 
v___x_1390_ = l_Lean_warningAsError;
v___x_1391_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__11(v_opts_1387_, v___x_1390_);
lean_dec_ref(v_opts_1387_);
v___y_1378_ = v___y_1382_;
v___y_1379_ = v___y_1382_;
v___y_1380_ = v___x_1391_;
goto v___jp_1377_;
}
}
else
{
lean_object* v___x_1392_; lean_object* v___x_1393_; 
lean_dec_ref(v_msgData_1252_);
v___x_1392_ = lean_box(0);
v___x_1393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1392_);
return v___x_1393_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3___boxed(lean_object* v_ref_1396_, lean_object* v_msgData_1397_, lean_object* v_severity_1398_, lean_object* v_isSilent_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
uint8_t v_severity_boxed_1403_; uint8_t v_isSilent_boxed_1404_; lean_object* v_res_1405_; 
v_severity_boxed_1403_ = lean_unbox(v_severity_1398_);
v_isSilent_boxed_1404_ = lean_unbox(v_isSilent_1399_);
v_res_1405_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3(v_ref_1396_, v_msgData_1397_, v_severity_boxed_1403_, v_isSilent_boxed_1404_, v___y_1400_, v___y_1401_);
lean_dec(v___y_1401_);
lean_dec_ref(v___y_1400_);
lean_dec(v_ref_1396_);
return v_res_1405_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1(lean_object* v_ref_1406_, lean_object* v_msgData_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
uint8_t v___x_1411_; uint8_t v___x_1412_; lean_object* v___x_1413_; 
v___x_1411_ = 1;
v___x_1412_ = 0;
v___x_1413_ = lp_batteries_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3(v_ref_1406_, v_msgData_1407_, v___x_1411_, v___x_1412_, v___y_1408_, v___y_1409_);
return v___x_1413_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1___boxed(lean_object* v_ref_1414_, lean_object* v_msgData_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_){
_start:
{
lean_object* v_res_1419_; 
v_res_1419_ = lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1(v_ref_1414_, v_msgData_1415_, v___y_1416_, v___y_1417_);
lean_dec(v___y_1417_);
lean_dec_ref(v___y_1416_);
lean_dec(v_ref_1414_);
return v_res_1419_;
}
}
static lean_object* _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1421_; lean_object* v___x_1422_; 
v___x_1421_ = ((lean_object*)(lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__0));
v___x_1422_ = l_Lean_stringToMessageData(v___x_1421_);
return v___x_1422_;
}
}
static lean_object* _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1424_; lean_object* v___x_1425_; 
v___x_1424_ = ((lean_object*)(lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__2));
v___x_1425_ = l_Lean_stringToMessageData(v___x_1424_);
return v___x_1425_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1(lean_object* v_linterOption_1426_, lean_object* v_stx_1427_, lean_object* v_msg_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_){
_start:
{
lean_object* v_name_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1450_; 
v_name_1432_ = lean_ctor_get(v_linterOption_1426_, 0);
v_isSharedCheck_1450_ = !lean_is_exclusive(v_linterOption_1426_);
if (v_isSharedCheck_1450_ == 0)
{
lean_object* v_unused_1451_; 
v_unused_1451_ = lean_ctor_get(v_linterOption_1426_, 1);
lean_dec(v_unused_1451_);
v___x_1434_ = v_linterOption_1426_;
v_isShared_1435_ = v_isSharedCheck_1450_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_name_1432_);
lean_dec(v_linterOption_1426_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1450_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1439_; 
v___x_1436_ = lean_obj_once(&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1, &lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1_once, _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__1);
lean_inc(v_name_1432_);
v___x_1437_ = l_Lean_MessageData_ofName(v_name_1432_);
if (v_isShared_1435_ == 0)
{
lean_ctor_set_tag(v___x_1434_, 7);
lean_ctor_set(v___x_1434_, 1, v___x_1437_);
lean_ctor_set(v___x_1434_, 0, v___x_1436_);
v___x_1439_ = v___x_1434_;
goto v_reusejp_1438_;
}
else
{
lean_object* v_reuseFailAlloc_1449_; 
v_reuseFailAlloc_1449_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1449_, 0, v___x_1436_);
lean_ctor_set(v_reuseFailAlloc_1449_, 1, v___x_1437_);
v___x_1439_ = v_reuseFailAlloc_1449_;
goto v_reusejp_1438_;
}
v_reusejp_1438_:
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v_disable_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; 
v___x_1440_ = lean_obj_once(&lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3, &lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3_once, _init_lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___closed__3);
v___x_1441_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1441_, 0, v___x_1439_);
lean_ctor_set(v___x_1441_, 1, v___x_1440_);
v_disable_1442_ = l_Lean_MessageData_note(v___x_1441_);
v___x_1443_ = l_Lean_Linter_linterMessageTag;
v___x_1444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1444_, 0, v_msg_1428_);
lean_ctor_set(v___x_1444_, 1, v_disable_1442_);
v___x_1445_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1445_, 0, v___x_1443_);
lean_ctor_set(v___x_1445_, 1, v___x_1444_);
v___x_1446_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1446_, 0, v_name_1432_);
lean_ctor_set(v___x_1446_, 1, v___x_1445_);
lean_inc(v_stx_1427_);
v___x_1447_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1447_, 0, v_stx_1427_);
lean_ctor_set(v___x_1447_, 1, v___x_1446_);
v___x_1448_ = lp_batteries_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1(v_stx_1427_, v___x_1447_, v___y_1429_, v___y_1430_);
lean_dec(v_stx_1427_);
return v___x_1448_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1___boxed(lean_object* v_linterOption_1452_, lean_object* v_stx_1453_, lean_object* v_msg_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1(v_linterOption_1452_, v_stx_1453_, v_msg_1454_, v___y_1455_, v___y_1456_);
lean_dec(v___y_1456_);
lean_dec_ref(v___y_1455_);
return v_res_1458_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2(void){
_start:
{
lean_object* v___x_1462_; lean_object* v___x_1463_; 
v___x_1462_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__1));
v___x_1463_ = l_Lean_MessageData_ofFormat(v___x_1462_);
return v___x_1463_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4(lean_object* v_as_1464_, size_t v_sz_1465_, size_t v_i_1466_, lean_object* v_b_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_){
_start:
{
lean_object* v_a_1472_; uint8_t v___x_1476_; 
v___x_1476_ = lean_usize_dec_lt(v_i_1466_, v_sz_1465_);
if (v___x_1476_ == 0)
{
lean_object* v___x_1477_; 
v___x_1477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1477_, 0, v_b_1467_);
return v___x_1477_;
}
else
{
lean_object* v_a_1478_; lean_object* v_fst_1479_; lean_object* v_snd_1480_; uint8_t v___y_1482_; lean_object* v_start_1494_; lean_object* v_stop_1495_; lean_object* v_start_1496_; lean_object* v_stop_1497_; uint8_t v___x_1498_; 
v_a_1478_ = lean_array_uget_borrowed(v_as_1464_, v_i_1466_);
v_fst_1479_ = lean_ctor_get(v_a_1478_, 0);
v_snd_1480_ = lean_ctor_get(v_a_1478_, 1);
v_start_1494_ = lean_ctor_get(v_b_1467_, 0);
v_stop_1495_ = lean_ctor_get(v_b_1467_, 1);
v_start_1496_ = lean_ctor_get(v_fst_1479_, 0);
v_stop_1497_ = lean_ctor_get(v_fst_1479_, 1);
v___x_1498_ = lean_nat_dec_le(v_start_1494_, v_start_1496_);
if (v___x_1498_ == 0)
{
v___y_1482_ = v___x_1498_;
goto v___jp_1481_;
}
else
{
uint8_t v___x_1499_; 
v___x_1499_ = lean_nat_dec_le(v_stop_1497_, v_stop_1495_);
v___y_1482_ = v___x_1499_;
goto v___jp_1481_;
}
v___jp_1481_:
{
if (v___y_1482_ == 0)
{
lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; 
lean_dec_ref(v_b_1467_);
v___x_1483_ = lp_batteries_Batteries_Linter_linter_unnecessarySeqFocus;
v___x_1484_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___closed__2);
lean_inc(v_snd_1480_);
v___x_1485_ = lp_batteries_Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1(v___x_1483_, v_snd_1480_, v___x_1484_, v___y_1468_, v___y_1469_);
if (lean_obj_tag(v___x_1485_) == 0)
{
lean_dec_ref_known(v___x_1485_, 1);
lean_inc(v_fst_1479_);
v_a_1472_ = v_fst_1479_;
goto v___jp_1471_;
}
else
{
lean_object* v_a_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1493_; 
v_a_1486_ = lean_ctor_get(v___x_1485_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1485_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1488_ = v___x_1485_;
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_a_1486_);
lean_dec(v___x_1485_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___x_1491_; 
if (v_isShared_1489_ == 0)
{
v___x_1491_ = v___x_1488_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v_a_1486_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
}
else
{
v_a_1472_ = v_b_1467_;
goto v___jp_1471_;
}
}
}
v___jp_1471_:
{
size_t v___x_1473_; size_t v___x_1474_; 
v___x_1473_ = ((size_t)1ULL);
v___x_1474_ = lean_usize_add(v_i_1466_, v___x_1473_);
v_i_1466_ = v___x_1474_;
v_b_1467_ = v_a_1472_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4___boxed(lean_object* v_as_1500_, lean_object* v_sz_1501_, lean_object* v_i_1502_, lean_object* v_b_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_){
_start:
{
size_t v_sz_boxed_1507_; size_t v_i_boxed_1508_; lean_object* v_res_1509_; 
v_sz_boxed_1507_ = lean_unbox_usize(v_sz_1501_);
lean_dec(v_sz_1501_);
v_i_boxed_1508_ = lean_unbox_usize(v_i_1502_);
lean_dec(v_i_1502_);
v_res_1509_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4(v_as_1500_, v_sz_boxed_1507_, v_i_boxed_1508_, v_b_1503_, v___y_1504_, v___y_1505_);
lean_dec(v___y_1505_);
lean_dec_ref(v___y_1504_);
lean_dec_ref(v_as_1500_);
return v_res_1509_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6(uint8_t v___x_1510_, lean_object* v_x_1511_, lean_object* v_x_1512_){
_start:
{
if (lean_obj_tag(v_x_1512_) == 0)
{
return v_x_1511_;
}
else
{
lean_object* v_value_1513_; uint8_t v_used_1514_; 
v_value_1513_ = lean_ctor_get(v_x_1512_, 1);
v_used_1514_ = lean_ctor_get_uint8(v_value_1513_, sizeof(void*)*1);
if (v_used_1514_ == 0)
{
lean_object* v_tail_1515_; 
v_tail_1515_ = lean_ctor_get(v_x_1512_, 2);
lean_inc(v_tail_1515_);
lean_dec_ref_known(v_x_1512_, 3);
v_x_1512_ = v_tail_1515_;
goto _start;
}
else
{
lean_object* v_key_1517_; lean_object* v_tail_1518_; lean_object* v_stx_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___y_1523_; lean_object* v___x_1527_; 
lean_inc(v_value_1513_);
v_key_1517_ = lean_ctor_get(v_x_1512_, 0);
lean_inc(v_key_1517_);
v_tail_1518_ = lean_ctor_get(v_x_1512_, 2);
lean_inc(v_tail_1518_);
lean_dec_ref_known(v_x_1512_, 3);
v_stx_1519_ = lean_ctor_get(v_value_1513_, 0);
lean_inc(v_stx_1519_);
lean_dec(v_value_1513_);
v___x_1520_ = lean_unsigned_to_nat(1u);
v___x_1521_ = l_Lean_Syntax_getArg(v_stx_1519_, v___x_1520_);
lean_dec(v_stx_1519_);
v___x_1527_ = l_Lean_Syntax_getRange_x3f(v___x_1521_, v___x_1510_);
if (lean_obj_tag(v___x_1527_) == 0)
{
v___y_1523_ = v_key_1517_;
goto v___jp_1522_;
}
else
{
lean_object* v_val_1528_; 
lean_dec(v_key_1517_);
v_val_1528_ = lean_ctor_get(v___x_1527_, 0);
lean_inc(v_val_1528_);
lean_dec_ref_known(v___x_1527_, 1);
v___y_1523_ = v_val_1528_;
goto v___jp_1522_;
}
v___jp_1522_:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; 
v___x_1524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1524_, 0, v___y_1523_);
lean_ctor_set(v___x_1524_, 1, v___x_1521_);
v___x_1525_ = lean_array_push(v_x_1511_, v___x_1524_);
v_x_1511_ = v___x_1525_;
v_x_1512_ = v_tail_1518_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6___boxed(lean_object* v___x_1529_, lean_object* v_x_1530_, lean_object* v_x_1531_){
_start:
{
uint8_t v___x_9041__boxed_1532_; lean_object* v_res_1533_; 
v___x_9041__boxed_1532_ = lean_unbox(v___x_1529_);
v_res_1533_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6(v___x_9041__boxed_1532_, v_x_1530_, v_x_1531_);
return v_res_1533_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7(uint8_t v___x_1534_, lean_object* v_as_1535_, size_t v_i_1536_, size_t v_stop_1537_, lean_object* v_b_1538_){
_start:
{
uint8_t v___x_1539_; 
v___x_1539_ = lean_usize_dec_eq(v_i_1536_, v_stop_1537_);
if (v___x_1539_ == 0)
{
lean_object* v___x_1540_; lean_object* v___x_1541_; size_t v___x_1542_; size_t v___x_1543_; 
v___x_1540_ = lean_array_uget_borrowed(v_as_1535_, v_i_1536_);
lean_inc(v___x_1540_);
v___x_1541_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__6(v___x_1534_, v_b_1538_, v___x_1540_);
v___x_1542_ = ((size_t)1ULL);
v___x_1543_ = lean_usize_add(v_i_1536_, v___x_1542_);
v_i_1536_ = v___x_1543_;
v_b_1538_ = v___x_1541_;
goto _start;
}
else
{
return v_b_1538_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7___boxed(lean_object* v___x_1545_, lean_object* v_as_1546_, lean_object* v_i_1547_, lean_object* v_stop_1548_, lean_object* v_b_1549_){
_start:
{
uint8_t v___x_9082__boxed_1550_; size_t v_i_boxed_1551_; size_t v_stop_boxed_1552_; lean_object* v_res_1553_; 
v___x_9082__boxed_1550_ = lean_unbox(v___x_1545_);
v_i_boxed_1551_ = lean_unbox_usize(v_i_1547_);
lean_dec(v_i_1547_);
v_stop_boxed_1552_ = lean_unbox_usize(v_stop_1548_);
lean_dec(v_stop_1548_);
v_res_1553_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7(v___x_9082__boxed_1550_, v_as_1546_, v_i_boxed_1551_, v_stop_boxed_1552_, v_b_1549_);
lean_dec_ref(v_as_1546_);
return v_res_1553_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(lean_object* v___f_1556_, uint8_t v___x_1557_, lean_object* v_x1_1558_, lean_object* v_x2_1559_){
_start:
{
lean_object* v_fst_1560_; lean_object* v_fst_1561_; lean_object* v___f_1562_; lean_object* v___f_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_8385__overap_1566_; lean_object* v___x_1567_; uint8_t v___x_1568_; 
v_fst_1560_ = lean_ctor_get(v_x1_1558_, 0);
lean_inc(v_fst_1560_);
lean_dec_ref(v_x1_1558_);
v_fst_1561_ = lean_ctor_get(v_x2_1559_, 0);
lean_inc(v_fst_1561_);
lean_dec_ref(v_x2_1559_);
v___f_1562_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__0));
v___f_1563_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__1));
lean_inc_ref(v___f_1556_);
v___x_1564_ = lean_apply_1(v___f_1556_, v_fst_1560_);
v___x_1565_ = lean_apply_1(v___f_1556_, v_fst_1561_);
v___x_8385__overap_1566_ = l_lexOrd___redArg(v___f_1562_, v___f_1563_);
v___x_1567_ = lean_apply_2(v___x_8385__overap_1566_, v___x_1564_, v___x_1565_);
v___x_1568_ = lean_unbox(v___x_1567_);
if (v___x_1568_ == 0)
{
return v___x_1557_;
}
else
{
uint8_t v___x_1569_; 
v___x_1569_ = 0;
return v___x_1569_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___boxed(lean_object* v___f_1570_, lean_object* v___x_1571_, lean_object* v_x1_1572_, lean_object* v_x2_1573_){
_start:
{
uint8_t v___x_9102__boxed_1574_; uint8_t v_res_1575_; lean_object* v_r_1576_; 
v___x_9102__boxed_1574_ = lean_unbox(v___x_1571_);
v_res_1575_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(v___f_1570_, v___x_9102__boxed_1574_, v_x1_1572_, v_x2_1573_);
v_r_1576_ = lean_box(v_res_1575_);
return v_r_1576_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__0(lean_object* v_r_1577_){
_start:
{
lean_object* v_start_1578_; lean_object* v_stop_1579_; lean_object* v___x_1581_; uint8_t v_isShared_1582_; uint8_t v_isSharedCheck_1588_; 
v_start_1578_ = lean_ctor_get(v_r_1577_, 0);
v_stop_1579_ = lean_ctor_get(v_r_1577_, 1);
v_isSharedCheck_1588_ = !lean_is_exclusive(v_r_1577_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1581_ = v_r_1577_;
v_isShared_1582_ = v_isSharedCheck_1588_;
goto v_resetjp_1580_;
}
else
{
lean_inc(v_stop_1579_);
lean_inc(v_start_1578_);
lean_dec(v_r_1577_);
v___x_1581_ = lean_box(0);
v_isShared_1582_ = v_isSharedCheck_1588_;
goto v_resetjp_1580_;
}
v_resetjp_1580_:
{
lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1586_; 
v___x_1583_ = lean_nat_to_int(v_stop_1579_);
v___x_1584_ = lean_int_neg(v___x_1583_);
lean_dec(v___x_1583_);
if (v_isShared_1582_ == 0)
{
lean_ctor_set(v___x_1581_, 1, v___x_1584_);
v___x_1586_ = v___x_1581_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_start_1578_);
lean_ctor_set(v_reuseFailAlloc_1587_, 1, v___x_1584_);
v___x_1586_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
return v___x_1586_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg(lean_object* v_hi_1589_, lean_object* v_pivot_1590_, lean_object* v_as_1591_, lean_object* v_i_1592_, lean_object* v_k_1593_){
_start:
{
uint8_t v___x_1598_; 
v___x_1598_ = lean_nat_dec_lt(v_k_1593_, v_hi_1589_);
if (v___x_1598_ == 0)
{
lean_object* v___x_1599_; lean_object* v___x_1600_; 
lean_dec(v_k_1593_);
lean_dec_ref(v_pivot_1590_);
v___x_1599_ = lean_array_fswap(v_as_1591_, v_i_1592_, v_hi_1589_);
v___x_1600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1600_, 0, v_i_1592_);
lean_ctor_set(v___x_1600_, 1, v___x_1599_);
return v___x_1600_;
}
else
{
lean_object* v___x_1601_; lean_object* v_fst_1602_; lean_object* v_fst_1603_; lean_object* v___f_1604_; lean_object* v___f_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_8189__overap_1608_; lean_object* v___x_1609_; uint8_t v___x_1610_; 
v___x_1601_ = lean_array_fget_borrowed(v_as_1591_, v_k_1593_);
v_fst_1602_ = lean_ctor_get(v___x_1601_, 0);
v_fst_1603_ = lean_ctor_get(v_pivot_1590_, 0);
v___f_1604_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__0));
v___f_1605_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1___closed__1));
lean_inc(v_fst_1602_);
v___x_1606_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__0(v_fst_1602_);
lean_inc(v_fst_1603_);
v___x_1607_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__0(v_fst_1603_);
v___x_8189__overap_1608_ = l_lexOrd___redArg(v___f_1604_, v___f_1605_);
v___x_1609_ = lean_apply_2(v___x_8189__overap_1608_, v___x_1606_, v___x_1607_);
v___x_1610_ = lean_unbox(v___x_1609_);
if (v___x_1610_ == 0)
{
if (v___x_1598_ == 0)
{
goto v___jp_1594_;
}
else
{
lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; 
v___x_1611_ = lean_array_fswap(v_as_1591_, v_i_1592_, v_k_1593_);
v___x_1612_ = lean_unsigned_to_nat(1u);
v___x_1613_ = lean_nat_add(v_i_1592_, v___x_1612_);
lean_dec(v_i_1592_);
v___x_1614_ = lean_nat_add(v_k_1593_, v___x_1612_);
lean_dec(v_k_1593_);
v_as_1591_ = v___x_1611_;
v_i_1592_ = v___x_1613_;
v_k_1593_ = v___x_1614_;
goto _start;
}
}
else
{
goto v___jp_1594_;
}
}
v___jp_1594_:
{
lean_object* v___x_1595_; lean_object* v___x_1596_; 
v___x_1595_ = lean_unsigned_to_nat(1u);
v___x_1596_ = lean_nat_add(v_k_1593_, v___x_1595_);
lean_dec(v_k_1593_);
v_k_1593_ = v___x_1596_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg___boxed(lean_object* v_hi_1616_, lean_object* v_pivot_1617_, lean_object* v_as_1618_, lean_object* v_i_1619_, lean_object* v_k_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg(v_hi_1616_, v_pivot_1617_, v_as_1618_, v_i_1619_, v_k_1620_);
lean_dec(v_hi_1616_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(lean_object* v_n_1623_, lean_object* v_as_1624_, lean_object* v_lo_1625_, lean_object* v_hi_1626_){
_start:
{
lean_object* v___y_1628_; uint8_t v___x_1638_; 
v___x_1638_ = lean_nat_dec_lt(v_lo_1625_, v_hi_1626_);
if (v___x_1638_ == 0)
{
lean_dec(v_lo_1625_);
return v_as_1624_;
}
else
{
lean_object* v___f_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v_mid_1642_; lean_object* v___y_1644_; lean_object* v___y_1650_; lean_object* v___x_1655_; lean_object* v___x_1656_; uint8_t v___x_1657_; 
v___f_1639_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___closed__0));
v___x_1640_ = lean_nat_add(v_lo_1625_, v_hi_1626_);
v___x_1641_ = lean_unsigned_to_nat(1u);
v_mid_1642_ = lean_nat_shiftr(v___x_1640_, v___x_1641_);
lean_dec(v___x_1640_);
v___x_1655_ = lean_array_fget_borrowed(v_as_1624_, v_mid_1642_);
v___x_1656_ = lean_array_fget_borrowed(v_as_1624_, v_lo_1625_);
lean_inc(v___x_1656_);
lean_inc(v___x_1655_);
v___x_1657_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(v___f_1639_, v___x_1638_, v___x_1655_, v___x_1656_);
if (v___x_1657_ == 0)
{
v___y_1650_ = v_as_1624_;
goto v___jp_1649_;
}
else
{
lean_object* v___x_1658_; 
v___x_1658_ = lean_array_fswap(v_as_1624_, v_lo_1625_, v_mid_1642_);
v___y_1650_ = v___x_1658_;
goto v___jp_1649_;
}
v___jp_1643_:
{
lean_object* v___x_1645_; lean_object* v___x_1646_; uint8_t v___x_1647_; 
v___x_1645_ = lean_array_fget_borrowed(v___y_1644_, v_mid_1642_);
v___x_1646_ = lean_array_fget_borrowed(v___y_1644_, v_hi_1626_);
lean_inc(v___x_1646_);
lean_inc(v___x_1645_);
v___x_1647_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(v___f_1639_, v___x_1638_, v___x_1645_, v___x_1646_);
if (v___x_1647_ == 0)
{
lean_dec(v_mid_1642_);
v___y_1628_ = v___y_1644_;
goto v___jp_1627_;
}
else
{
lean_object* v___x_1648_; 
v___x_1648_ = lean_array_fswap(v___y_1644_, v_mid_1642_, v_hi_1626_);
lean_dec(v_mid_1642_);
v___y_1628_ = v___x_1648_;
goto v___jp_1627_;
}
}
v___jp_1649_:
{
lean_object* v___x_1651_; lean_object* v___x_1652_; uint8_t v___x_1653_; 
v___x_1651_ = lean_array_fget_borrowed(v___y_1650_, v_hi_1626_);
v___x_1652_ = lean_array_fget_borrowed(v___y_1650_, v_lo_1625_);
lean_inc(v___x_1652_);
lean_inc(v___x_1651_);
v___x_1653_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___lam__1(v___f_1639_, v___x_1638_, v___x_1651_, v___x_1652_);
if (v___x_1653_ == 0)
{
v___y_1644_ = v___y_1650_;
goto v___jp_1643_;
}
else
{
lean_object* v___x_1654_; 
v___x_1654_ = lean_array_fswap(v___y_1650_, v_lo_1625_, v_hi_1626_);
v___y_1644_ = v___x_1654_;
goto v___jp_1643_;
}
}
}
v___jp_1627_:
{
lean_object* v_pivot_1629_; lean_object* v___x_1630_; lean_object* v_fst_1631_; lean_object* v_snd_1632_; uint8_t v___x_1633_; 
v_pivot_1629_ = lean_array_fget(v___y_1628_, v_hi_1626_);
lean_inc_n(v_lo_1625_, 2);
v___x_1630_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg(v_hi_1626_, v_pivot_1629_, v___y_1628_, v_lo_1625_, v_lo_1625_);
v_fst_1631_ = lean_ctor_get(v___x_1630_, 0);
lean_inc(v_fst_1631_);
v_snd_1632_ = lean_ctor_get(v___x_1630_, 1);
lean_inc(v_snd_1632_);
lean_dec_ref(v___x_1630_);
v___x_1633_ = lean_nat_dec_le(v_hi_1626_, v_fst_1631_);
if (v___x_1633_ == 0)
{
lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1634_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(v_n_1623_, v_snd_1632_, v_lo_1625_, v_fst_1631_);
v___x_1635_ = lean_unsigned_to_nat(1u);
v___x_1636_ = lean_nat_add(v_fst_1631_, v___x_1635_);
lean_dec(v_fst_1631_);
v_as_1624_ = v___x_1634_;
v_lo_1625_ = v___x_1636_;
goto _start;
}
else
{
lean_dec(v_fst_1631_);
lean_dec(v_lo_1625_);
return v_snd_1632_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg___boxed(lean_object* v_n_1659_, lean_object* v_as_1660_, lean_object* v_lo_1661_, lean_object* v_hi_1662_){
_start:
{
lean_object* v_res_1663_; 
v_res_1663_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(v_n_1659_, v_as_1660_, v_lo_1661_, v_hi_1662_);
lean_dec(v_hi_1662_);
lean_dec(v_n_1659_);
return v_res_1663_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1(lean_object* v_stx_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_){
_start:
{
lean_object* v___y_1673_; lean_object* v___y_1674_; lean_object* v___y_1696_; lean_object* v___y_1697_; lean_object* v___y_1698_; lean_object* v___y_1699_; lean_object* v___y_1700_; lean_object* v___y_1703_; lean_object* v___y_1704_; lean_object* v___y_1705_; lean_object* v___y_1706_; lean_object* v___y_1707_; lean_object* v___y_1710_; lean_object* v___x_1718_; lean_object* v_a_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1759_; 
v___x_1718_ = lp_batteries_Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2(v___y_1669_, v___y_1670_);
v_a_1719_ = lean_ctor_get(v___x_1718_, 0);
v_isSharedCheck_1759_ = !lean_is_exclusive(v___x_1718_);
if (v_isSharedCheck_1759_ == 0)
{
v___x_1721_ = v___x_1718_;
v_isShared_1722_ = v_isSharedCheck_1759_;
goto v_resetjp_1720_;
}
else
{
lean_inc(v_a_1719_);
lean_dec(v___x_1718_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1759_;
goto v_resetjp_1720_;
}
v___jp_1672_:
{
size_t v_sz_1675_; size_t v___x_1676_; lean_object* v___x_1677_; 
v_sz_1675_ = lean_array_size(v___y_1674_);
v___x_1676_ = ((size_t)0ULL);
lean_inc_ref(v___y_1673_);
v___x_1677_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__4(v___y_1674_, v_sz_1675_, v___x_1676_, v___y_1673_, v___y_1669_, v___y_1670_);
lean_dec_ref(v___y_1674_);
if (lean_obj_tag(v___x_1677_) == 0)
{
lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1685_; 
v_isSharedCheck_1685_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1685_ == 0)
{
lean_object* v_unused_1686_; 
v_unused_1686_ = lean_ctor_get(v___x_1677_, 0);
lean_dec(v_unused_1686_);
v___x_1679_ = v___x_1677_;
v_isShared_1680_ = v_isSharedCheck_1685_;
goto v_resetjp_1678_;
}
else
{
lean_dec(v___x_1677_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1685_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1681_; lean_object* v___x_1683_; 
v___x_1681_ = lean_box(0);
if (v_isShared_1680_ == 0)
{
lean_ctor_set(v___x_1679_, 0, v___x_1681_);
v___x_1683_ = v___x_1679_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1684_; 
v_reuseFailAlloc_1684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1684_, 0, v___x_1681_);
v___x_1683_ = v_reuseFailAlloc_1684_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
return v___x_1683_;
}
}
}
else
{
lean_object* v_a_1687_; lean_object* v___x_1689_; uint8_t v_isShared_1690_; uint8_t v_isSharedCheck_1694_; 
v_a_1687_ = lean_ctor_get(v___x_1677_, 0);
v_isSharedCheck_1694_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1694_ == 0)
{
v___x_1689_ = v___x_1677_;
v_isShared_1690_ = v_isSharedCheck_1694_;
goto v_resetjp_1688_;
}
else
{
lean_inc(v_a_1687_);
lean_dec(v___x_1677_);
v___x_1689_ = lean_box(0);
v_isShared_1690_ = v_isSharedCheck_1694_;
goto v_resetjp_1688_;
}
v_resetjp_1688_:
{
lean_object* v___x_1692_; 
if (v_isShared_1690_ == 0)
{
v___x_1692_ = v___x_1689_;
goto v_reusejp_1691_;
}
else
{
lean_object* v_reuseFailAlloc_1693_; 
v_reuseFailAlloc_1693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1693_, 0, v_a_1687_);
v___x_1692_ = v_reuseFailAlloc_1693_;
goto v_reusejp_1691_;
}
v_reusejp_1691_:
{
return v___x_1692_;
}
}
}
}
v___jp_1695_:
{
lean_object* v___x_1701_; 
v___x_1701_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(v___y_1698_, v___y_1699_, v___y_1697_, v___y_1700_);
lean_dec(v___y_1700_);
lean_dec(v___y_1698_);
v___y_1673_ = v___y_1696_;
v___y_1674_ = v___x_1701_;
goto v___jp_1672_;
}
v___jp_1702_:
{
uint8_t v___x_1708_; 
v___x_1708_ = lean_nat_dec_le(v___y_1707_, v___y_1706_);
if (v___x_1708_ == 0)
{
lean_dec(v___y_1706_);
lean_inc(v___y_1707_);
v___y_1696_ = v___y_1703_;
v___y_1697_ = v___y_1707_;
v___y_1698_ = v___y_1704_;
v___y_1699_ = v___y_1705_;
v___y_1700_ = v___y_1707_;
goto v___jp_1695_;
}
else
{
v___y_1696_ = v___y_1703_;
v___y_1697_ = v___y_1707_;
v___y_1698_ = v___y_1704_;
v___y_1699_ = v___y_1705_;
v___y_1700_ = v___y_1706_;
goto v___jp_1695_;
}
}
v___jp_1709_:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; uint8_t v___x_1714_; 
v___x_1711_ = lean_unsigned_to_nat(0u);
v___x_1712_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__0));
v___x_1713_ = lean_array_get_size(v___y_1710_);
v___x_1714_ = lean_nat_dec_eq(v___x_1713_, v___x_1711_);
if (v___x_1714_ == 0)
{
lean_object* v___x_1715_; lean_object* v___x_1716_; uint8_t v___x_1717_; 
v___x_1715_ = lean_unsigned_to_nat(1u);
v___x_1716_ = lean_nat_sub(v___x_1713_, v___x_1715_);
v___x_1717_ = lean_nat_dec_le(v___x_1711_, v___x_1716_);
if (v___x_1717_ == 0)
{
lean_inc(v___x_1716_);
v___y_1703_ = v___x_1712_;
v___y_1704_ = v___x_1713_;
v___y_1705_ = v___y_1710_;
v___y_1706_ = v___x_1716_;
v___y_1707_ = v___x_1716_;
goto v___jp_1702_;
}
else
{
v___y_1703_ = v___x_1712_;
v___y_1704_ = v___x_1713_;
v___y_1705_ = v___y_1710_;
v___y_1706_ = v___x_1716_;
v___y_1707_ = v___x_1711_;
goto v___jp_1702_;
}
}
else
{
v___y_1673_ = v___x_1712_;
v___y_1674_ = v___y_1710_;
goto v___jp_1672_;
}
}
v_resetjp_1720_:
{
lean_object* v___x_1723_; uint8_t v___y_1725_; uint8_t v___x_1756_; 
v___x_1723_ = lean_st_ref_get(v___y_1670_);
v___x_1756_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_getLinterUnnecessarySeqFocus(v_a_1719_);
lean_dec(v_a_1719_);
if (v___x_1756_ == 0)
{
lean_dec(v___x_1723_);
v___y_1725_ = v___x_1756_;
goto v___jp_1724_;
}
else
{
lean_object* v_infoState_1757_; uint8_t v_enabled_1758_; 
v_infoState_1757_ = lean_ctor_get(v___x_1723_, 8);
lean_inc_ref(v_infoState_1757_);
lean_dec(v___x_1723_);
v_enabled_1758_ = lean_ctor_get_uint8(v_infoState_1757_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1757_);
v___y_1725_ = v_enabled_1758_;
goto v___jp_1724_;
}
v___jp_1724_:
{
if (v___y_1725_ == 0)
{
lean_object* v___x_1726_; lean_object* v___x_1728_; 
lean_dec(v_stx_1668_);
v___x_1726_ = lean_box(0);
if (v_isShared_1722_ == 0)
{
lean_ctor_set(v___x_1721_, 0, v___x_1726_);
v___x_1728_ = v___x_1721_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v___x_1726_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
else
{
lean_object* v___x_1730_; lean_object* v_messages_1731_; uint8_t v___x_1732_; 
v___x_1730_ = lean_st_ref_get(v___y_1670_);
v_messages_1731_ = lean_ctor_get(v___x_1730_, 1);
lean_inc_ref(v_messages_1731_);
lean_dec(v___x_1730_);
v___x_1732_ = l_Lean_MessageLog_hasErrors(v_messages_1731_);
lean_dec_ref(v_messages_1731_);
if (v___x_1732_ == 0)
{
lean_object* v___x_1733_; lean_object* v_a_1734_; lean_object* v___x_1735_; lean_object* v_env_1736_; lean_object* v___f_1737_; lean_object* v___x_1738_; lean_object* v_snd_1739_; lean_object* v_buckets_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; uint8_t v___x_1744_; 
lean_del_object(v___x_1721_);
v___x_1733_ = lp_batteries_Lean_Elab_getInfoTrees___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__3___redArg(v___y_1670_);
v_a_1734_ = lean_ctor_get(v___x_1733_, 0);
lean_inc(v_a_1734_);
lean_dec_ref(v___x_1733_);
v___x_1735_ = lean_st_ref_get(v___y_1670_);
v_env_1736_ = lean_ctor_get(v___x_1735_, 0);
lean_inc_ref(v_env_1736_);
lean_dec(v___x_1735_);
v___f_1737_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__0___boxed), 5, 3);
lean_closure_set(v___f_1737_, 0, v_stx_1668_);
lean_closure_set(v___f_1737_, 1, v_env_1736_);
lean_closure_set(v___f_1737_, 2, v_a_1734_);
v___x_1738_ = l_runST___redArg(v___f_1737_);
v_snd_1739_ = lean_ctor_get(v___x_1738_, 1);
lean_inc(v_snd_1739_);
lean_dec(v___x_1738_);
v_buckets_1740_ = lean_ctor_get(v_snd_1739_, 1);
lean_inc_ref(v_buckets_1740_);
lean_dec(v_snd_1739_);
v___x_1741_ = lean_unsigned_to_nat(0u);
v___x_1742_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___closed__1));
v___x_1743_ = lean_array_get_size(v_buckets_1740_);
v___x_1744_ = lean_nat_dec_lt(v___x_1741_, v___x_1743_);
if (v___x_1744_ == 0)
{
lean_dec_ref(v_buckets_1740_);
v___y_1710_ = v___x_1742_;
goto v___jp_1709_;
}
else
{
uint8_t v___x_1745_; 
v___x_1745_ = lean_nat_dec_le(v___x_1743_, v___x_1743_);
if (v___x_1745_ == 0)
{
if (v___x_1744_ == 0)
{
lean_dec_ref(v_buckets_1740_);
v___y_1710_ = v___x_1742_;
goto v___jp_1709_;
}
else
{
size_t v___x_1746_; size_t v___x_1747_; lean_object* v___x_1748_; 
v___x_1746_ = ((size_t)0ULL);
v___x_1747_ = lean_usize_of_nat(v___x_1743_);
v___x_1748_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7(v___x_1732_, v_buckets_1740_, v___x_1746_, v___x_1747_, v___x_1742_);
lean_dec_ref(v_buckets_1740_);
v___y_1710_ = v___x_1748_;
goto v___jp_1709_;
}
}
else
{
size_t v___x_1749_; size_t v___x_1750_; lean_object* v___x_1751_; 
v___x_1749_ = ((size_t)0ULL);
v___x_1750_ = lean_usize_of_nat(v___x_1743_);
v___x_1751_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__7(v___x_1732_, v_buckets_1740_, v___x_1749_, v___x_1750_, v___x_1742_);
lean_dec_ref(v_buckets_1740_);
v___y_1710_ = v___x_1751_;
goto v___jp_1709_;
}
}
}
else
{
lean_object* v___x_1752_; lean_object* v___x_1754_; 
lean_dec(v_stx_1668_);
v___x_1752_ = lean_box(0);
if (v_isShared_1722_ == 0)
{
lean_ctor_set(v___x_1721_, 0, v___x_1752_);
v___x_1754_ = v___x_1721_;
goto v_reusejp_1753_;
}
else
{
lean_object* v_reuseFailAlloc_1755_; 
v_reuseFailAlloc_1755_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1755_, 0, v___x_1752_);
v___x_1754_ = v_reuseFailAlloc_1755_;
goto v_reusejp_1753_;
}
v_reusejp_1753_:
{
return v___x_1754_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1___boxed(lean_object* v_stx_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v_res_1764_; 
v_res_1764_ = lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter___lam__1(v_stx_1760_, v___y_1761_, v___y_1762_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3(lean_object* v_o_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_){
_start:
{
lean_object* v___x_1782_; 
v___x_1782_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___redArg(v_o_1778_, v___y_1780_);
return v___x_1782_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3___boxed(lean_object* v_o_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_){
_start:
{
lean_object* v_res_1787_; 
v_res_1787_ = lp_batteries_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__2_spec__3(v_o_1783_, v___y_1784_, v___y_1785_);
lean_dec(v___y_1785_);
lean_dec_ref(v___y_1784_);
return v_res_1787_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5(lean_object* v_n_1788_, lean_object* v_as_1789_, lean_object* v_lo_1790_, lean_object* v_hi_1791_, lean_object* v_w_1792_, lean_object* v_hlo_1793_, lean_object* v_hhi_1794_){
_start:
{
lean_object* v___x_1795_; 
v___x_1795_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___redArg(v_n_1788_, v_as_1789_, v_lo_1790_, v_hi_1791_);
return v___x_1795_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5___boxed(lean_object* v_n_1796_, lean_object* v_as_1797_, lean_object* v_lo_1798_, lean_object* v_hi_1799_, lean_object* v_w_1800_, lean_object* v_hlo_1801_, lean_object* v_hhi_1802_){
_start:
{
lean_object* v_res_1803_; 
v_res_1803_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5(v_n_1796_, v_as_1797_, v_lo_1798_, v_hi_1799_, v_w_1800_, v_hlo_1801_, v_hhi_1802_);
lean_dec(v_hi_1799_);
lean_dec(v_n_1796_);
return v_res_1803_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7(lean_object* v_n_1804_, lean_object* v_lo_1805_, lean_object* v_hi_1806_, lean_object* v_hhi_1807_, lean_object* v_pivot_1808_, lean_object* v_as_1809_, lean_object* v_i_1810_, lean_object* v_k_1811_, lean_object* v_ilo_1812_, lean_object* v_ik_1813_, lean_object* v_w_1814_){
_start:
{
lean_object* v___x_1815_; 
v___x_1815_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___redArg(v_hi_1806_, v_pivot_1808_, v_as_1809_, v_i_1810_, v_k_1811_);
return v___x_1815_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7___boxed(lean_object* v_n_1816_, lean_object* v_lo_1817_, lean_object* v_hi_1818_, lean_object* v_hhi_1819_, lean_object* v_pivot_1820_, lean_object* v_as_1821_, lean_object* v_i_1822_, lean_object* v_k_1823_, lean_object* v_ilo_1824_, lean_object* v_ik_1825_, lean_object* v_w_1826_){
_start:
{
lean_object* v_res_1827_; 
v_res_1827_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__5_spec__7(v_n_1816_, v_lo_1817_, v_hi_1818_, v_hhi_1819_, v_pivot_1820_, v_as_1821_, v_i_1822_, v_k_1823_, v_ilo_1824_, v_ik_1825_, v_w_1826_);
lean_dec(v_hi_1818_);
lean_dec(v_lo_1817_);
lean_dec(v_n_1816_);
return v_res_1827_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10(lean_object* v_msgData_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_){
_start:
{
lean_object* v___x_1832_; 
v___x_1832_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___redArg(v_msgData_1828_, v___y_1830_);
return v___x_1832_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10___boxed(lean_object* v_msgData_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_){
_start:
{
lean_object* v_res_1837_; 
v_res_1837_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter_spec__1_spec__1_spec__3_spec__10(v_msgData_1833_, v___y_1834_, v___y_1835_);
lean_dec(v___y_1835_);
lean_dec_ref(v___y_1834_);
return v_res_1837_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1839_; lean_object* v___x_1840_; 
v___x_1839_ = ((lean_object*)(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_unnecessarySeqFocusLinter));
v___x_1840_ = l_Lean_Elab_Command_addLinter(v___x_1839_);
return v___x_1840_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2____boxed(lean_object* v_a_1841_){
_start:
{
lean_object* v_res_1842_; 
v_res_1842_ = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2_();
return v_res_1842_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(uint8_t builtin) {
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
lean_object* runtime_initialize_batteries_Batteries_Lean_AttributeExtra(uint8_t builtin);
lean_object* runtime_initialize_Lean_Linter_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_AttributeExtra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_2411125583____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Linter_linter_unnecessarySeqFocus = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Linter_linter_unnecessarySeqFocus);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_971531596____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Linter_UnnecessarySeqFocus_multigoalAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Linter_UnnecessarySeqFocus_multigoalAttr);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Linter_UnnecessarySeqFocus_0__Batteries_Linter_UnnecessarySeqFocus_initFn_00___x40_Batteries_Linter_UnnecessarySeqFocus_3311112716____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_AttributeExtra(uint8_t builtin);
lean_object* initialize_Lean_Linter_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(uint8_t builtin) {
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
res = initialize_batteries_Batteries_Lean_AttributeExtra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Linter_UnnecessarySeqFocus(builtin);
}
#ifdef __cplusplus
}
#endif
