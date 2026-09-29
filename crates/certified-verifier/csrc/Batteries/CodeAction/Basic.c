// Lean compiler output
// Module: Batteries.CodeAction.Basic
// Imports: public import Init public meta import Init public meta import Lean.Elab.BuiltinTerm public meta import Lean.Elab.BuiltinNotation public meta import Lean.Server.InfoUtils public meta import Lean.Server.CodeActions.Provider public meta import Batteries.CodeAction.Attr
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
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_utf8PosToLspPos(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
extern lean_object* l_Lean_Server_instInhabitedRequestError_default;
lean_object* l_instInhabitedEIO___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
extern lean_object* lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default;
lean_object* l_Array_instInhabited(lean_object*);
extern lean_object* lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt;
lean_object* l_Lean_Server_Snapshots_Snapshot_env(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FileMap_lspPosToUtf8Pos(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Server_Snapshots_Snapshot_infoTree(lean_object*);
lean_object* l_Lean_CodeAction_findInfoTree_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
extern lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionExt;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
extern lean_object* l_Lean_Syntax_instInhabitedRange_default;
lean_object* l_Lean_CodeAction_findTactic_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__5(lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Batteries.CodeAction.Basic"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "Batteries.CodeAction.tacticCodeActionProvider"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1(uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4;
static const lean_closure_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__7_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__0(lean_object* v_msg_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_box(0);
v___x_3_ = lean_panic_fn_borrowed(v___x_2_, v_msg_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1(lean_object* v___y_4_){
_start:
{
lean_object* v_doc_6_; lean_object* v___x_7_; 
v_doc_6_ = lean_ctor_get(v___y_4_, 1);
lean_inc_ref(v_doc_6_);
v___x_7_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_7_, 0, v_doc_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1___boxed(lean_object* v___y_8_, lean_object* v___y_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1(v___y_8_);
lean_dec_ref(v___y_8_);
return v_res_10_;
}
}
static lean_object* _init_lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0(void){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = l_Lean_Server_instInhabitedRequestError_default;
v___x_12_ = lean_alloc_closure((void*)(l_instInhabitedEIO___aux__1___boxed), 4, 3);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, lean_box(0));
lean_closure_set(v___x_12_, 2, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2(lean_object* v_msg_13_, lean_object* v___y_14_){
_start:
{
lean_object* v___x_16_; lean_object* v___f_17_; lean_object* v___x_7222__overap_18_; lean_object* v___x_19_; 
v___x_16_ = lean_obj_once(&lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0, &lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0_once, _init_lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___closed__0);
v___f_17_ = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_17_, 0, v___x_16_);
v___x_7222__overap_18_ = lean_panic_fn_borrowed(v___f_17_, v_msg_13_);
lean_dec_ref(v___f_17_);
lean_inc_ref(v___y_14_);
v___x_19_ = lean_apply_2(v___x_7222__overap_18_, v___y_14_, lean_box(0));
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2___boxed(lean_object* v_msg_20_, lean_object* v___y_21_, lean_object* v___y_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2(v_msg_20_, v___y_21_);
lean_dec_ref(v___y_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__5(lean_object* v_msg_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = l_Lean_Syntax_instInhabitedRange_default;
v___x_26_ = lean_panic_fn_borrowed(v___x_25_, v_msg_24_);
return v___x_26_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_30_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__2));
v___x_31_ = lean_unsigned_to_nat(11u);
v___x_32_ = lean_unsigned_to_nat(45u);
v___x_33_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__1));
v___x_34_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__0));
v___x_35_ = l_mkPanicMessageWithDecl(v___x_34_, v___x_33_, v___x_32_, v___x_31_, v___x_30_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(lean_object* v_x_36_){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__3);
v___x_38_ = lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__0(v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___boxed(lean_object* v_x_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(v_x_39_);
lean_dec_ref(v_x_39_);
return v_res_40_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1(uint8_t v___x_41_, uint8_t v___x_42_, lean_object* v_x_43_, lean_object* v_info_44_){
_start:
{
if (lean_obj_tag(v_info_44_) == 0)
{
return v___x_41_;
}
else
{
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1___boxed(lean_object* v___x_45_, lean_object* v___x_46_, lean_object* v_x_47_, lean_object* v_info_48_){
_start:
{
uint8_t v___x_7982__boxed_49_; uint8_t v___x_7983__boxed_50_; uint8_t v_res_51_; lean_object* v_r_52_; 
v___x_7982__boxed_49_ = lean_unbox(v___x_45_);
v___x_7983__boxed_50_ = lean_unbox(v___x_46_);
v_res_51_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__1(v___x_7982__boxed_49_, v___x_7983__boxed_50_, v_x_47_, v_info_48_);
lean_dec_ref(v_info_48_);
lean_dec_ref(v_x_47_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2(lean_object* v_text_53_, lean_object* v___y_54_, lean_object* v_pos_55_){
_start:
{
lean_object* v___x_56_; lean_object* v_character_57_; uint8_t v___x_58_; 
v___x_56_ = l_Lean_FileMap_utf8PosToLspPos(v_text_53_, v_pos_55_);
v_character_57_ = lean_ctor_get(v___x_56_, 1);
lean_inc(v_character_57_);
lean_dec_ref(v___x_56_);
v___x_58_ = lean_nat_dec_le(v_character_57_, v___y_54_);
lean_dec(v_character_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2___boxed(lean_object* v_text_59_, lean_object* v___y_60_, lean_object* v_pos_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2(v_text_59_, v___y_60_, v_pos_61_);
lean_dec(v_pos_61_);
lean_dec(v___y_60_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4(lean_object* v_params_64_, lean_object* v_snap_65_, lean_object* v_fst_66_, lean_object* v_insertIdx_67_, lean_object* v_a_68_, lean_object* v_snd_69_, lean_object* v_as_70_, size_t v_sz_71_, size_t v_i_72_, lean_object* v_b_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_snd_77_; uint8_t v___x_81_; 
v___x_81_ = lean_usize_dec_lt(v_i_72_, v_sz_71_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; 
lean_dec(v_snd_69_);
lean_dec(v_a_68_);
lean_dec(v_insertIdx_67_);
lean_dec_ref(v_fst_66_);
lean_dec_ref(v_snap_65_);
lean_dec_ref(v_params_64_);
v___x_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_82_, 0, v_b_73_);
return v___x_82_;
}
else
{
lean_object* v___x_7826__overap_83_; lean_object* v___x_84_; 
v___x_7826__overap_83_ = lean_array_uget_borrowed(v_as_70_, v_i_72_);
lean_inc(v___x_7826__overap_83_);
lean_inc_ref(v___y_74_);
lean_inc(v_snd_69_);
lean_inc(v_a_68_);
lean_inc(v_insertIdx_67_);
lean_inc_ref(v_fst_66_);
lean_inc_ref(v_snap_65_);
lean_inc_ref(v_params_64_);
v___x_84_ = lean_apply_8(v___x_7826__overap_83_, v_params_64_, v_snap_65_, v_fst_66_, v_insertIdx_67_, v_a_68_, v_snd_69_, v___y_74_, lean_box(0));
if (lean_obj_tag(v___x_84_) == 0)
{
lean_object* v_a_85_; lean_object* v___x_86_; 
v_a_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_a_85_);
lean_dec_ref_known(v___x_84_, 1);
v___x_86_ = l_Array_append___redArg(v_b_73_, v_a_85_);
lean_dec(v_a_85_);
v_snd_77_ = v___x_86_;
goto v___jp_76_;
}
else
{
lean_dec_ref_known(v___x_84_, 1);
v_snd_77_ = v_b_73_;
goto v___jp_76_;
}
}
v___jp_76_:
{
size_t v___x_78_; size_t v___x_79_; 
v___x_78_ = ((size_t)1ULL);
v___x_79_ = lean_usize_add(v_i_72_, v___x_78_);
v_i_72_ = v___x_79_;
v_b_73_ = v_snd_77_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4___boxed(lean_object* v_params_87_, lean_object* v_snap_88_, lean_object* v_fst_89_, lean_object* v_insertIdx_90_, lean_object* v_a_91_, lean_object* v_snd_92_, lean_object* v_as_93_, lean_object* v_sz_94_, lean_object* v_i_95_, lean_object* v_b_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
size_t v_sz_boxed_99_; size_t v_i_boxed_100_; lean_object* v_res_101_; 
v_sz_boxed_99_ = lean_unbox_usize(v_sz_94_);
lean_dec(v_sz_94_);
v_i_boxed_100_ = lean_unbox_usize(v_i_95_);
lean_dec(v_i_95_);
v_res_101_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4(v_params_87_, v_snap_88_, v_fst_89_, v_insertIdx_90_, v_a_91_, v_snd_92_, v_as_93_, v_sz_boxed_99_, v_i_boxed_100_, v_b_96_, v___y_97_);
lean_dec_ref(v___y_97_);
lean_dec_ref(v_as_93_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3(lean_object* v_params_102_, lean_object* v_snap_103_, lean_object* v___x_104_, lean_object* v_a_105_, lean_object* v_snd_106_, lean_object* v_as_107_, size_t v_sz_108_, size_t v_i_109_, lean_object* v_b_110_, lean_object* v___y_111_){
_start:
{
lean_object* v_snd_114_; uint8_t v___x_118_; 
v___x_118_ = lean_usize_dec_lt(v_i_109_, v_sz_108_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; 
lean_dec_ref(v_snd_106_);
lean_dec(v_a_105_);
lean_dec_ref(v___x_104_);
lean_dec_ref(v_snap_103_);
lean_dec_ref(v_params_102_);
v___x_119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_119_, 0, v_b_110_);
return v___x_119_;
}
else
{
lean_object* v___x_7790__overap_120_; lean_object* v___x_121_; 
v___x_7790__overap_120_ = lean_array_uget_borrowed(v_as_107_, v_i_109_);
lean_inc(v___x_7790__overap_120_);
lean_inc_ref(v___y_111_);
lean_inc_ref(v_snd_106_);
lean_inc(v_a_105_);
lean_inc_ref(v___x_104_);
lean_inc_ref(v_snap_103_);
lean_inc_ref(v_params_102_);
v___x_121_ = lean_apply_7(v___x_7790__overap_120_, v_params_102_, v_snap_103_, v___x_104_, v_a_105_, v_snd_106_, v___y_111_, lean_box(0));
if (lean_obj_tag(v___x_121_) == 0)
{
lean_object* v_a_122_; lean_object* v___x_123_; 
v_a_122_ = lean_ctor_get(v___x_121_, 0);
lean_inc(v_a_122_);
lean_dec_ref_known(v___x_121_, 1);
v___x_123_ = l_Array_append___redArg(v_b_110_, v_a_122_);
lean_dec(v_a_122_);
v_snd_114_ = v___x_123_;
goto v___jp_113_;
}
else
{
lean_dec_ref_known(v___x_121_, 1);
v_snd_114_ = v_b_110_;
goto v___jp_113_;
}
}
v___jp_113_:
{
size_t v___x_115_; size_t v___x_116_; 
v___x_115_ = ((size_t)1ULL);
v___x_116_ = lean_usize_add(v_i_109_, v___x_115_);
v_i_109_ = v___x_116_;
v_b_110_ = v_snd_114_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3___boxed(lean_object* v_params_124_, lean_object* v_snap_125_, lean_object* v___x_126_, lean_object* v_a_127_, lean_object* v_snd_128_, lean_object* v_as_129_, lean_object* v_sz_130_, lean_object* v_i_131_, lean_object* v_b_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
size_t v_sz_boxed_135_; size_t v_i_boxed_136_; lean_object* v_res_137_; 
v_sz_boxed_135_ = lean_unbox_usize(v_sz_130_);
lean_dec(v_sz_130_);
v_i_boxed_136_ = lean_unbox_usize(v_i_131_);
lean_dec(v_i_131_);
v_res_137_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3(v_params_124_, v_snap_125_, v___x_126_, v_a_127_, v_snd_128_, v_as_129_, v_sz_boxed_135_, v_i_boxed_136_, v_b_132_, v___y_133_);
lean_dec_ref(v___y_133_);
lean_dec_ref(v_as_129_);
return v_res_137_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_140_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__2));
v___x_141_ = lean_unsigned_to_nat(9u);
v___x_142_ = lean_unsigned_to_nat(72u);
v___x_143_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__1));
v___x_144_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0___closed__0));
v___x_145_ = l_mkPanicMessageWithDecl(v___x_144_, v___x_143_, v___x_142_, v___x_141_, v___x_140_);
return v___x_145_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2(void){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = l_Array_instInhabited(lean_box(0));
return v___x_146_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3(void){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_147_ = lp_batteries_Batteries_CodeAction_instInhabitedTacticCodeActions_default;
v___x_148_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___x_147_);
return v___x_149_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__2);
v___x_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_150_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
return v___x_151_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_160_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__8));
v___x_161_ = lean_unsigned_to_nat(14u);
v___x_162_ = lean_unsigned_to_nat(22u);
v___x_163_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__7));
v___x_164_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__6));
v___x_165_ = l_mkPanicMessageWithDecl(v___x_164_, v___x_163_, v___x_162_, v___x_161_, v___x_160_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider(lean_object* v_params_166_, lean_object* v_snap_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_195_; lean_object* v___y_196_; lean_object* v_onAnyTactic_197_; lean_object* v___y_198_; lean_object* v_out_199_; lean_object* v___y_200_; lean_object* v___x_204_; lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_432_; 
v___x_204_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__1(v_a_168_);
v_a_205_ = lean_ctor_get(v___x_204_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_204_);
if (v_isSharedCheck_432_ == 0)
{
v___x_207_ = v___x_204_;
v_isShared_208_ = v_isSharedCheck_432_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_dec(v___x_204_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_432_;
goto v_resetjp_206_;
}
v___jp_170_:
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0));
v___x_172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
return v___x_172_;
}
v___jp_173_:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__1);
v___x_177_ = lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__2(v___x_176_, v___y_175_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_184_; 
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_184_ == 0)
{
lean_object* v_unused_185_; 
v_unused_185_ = lean_ctor_get(v___x_177_, 0);
lean_dec(v_unused_185_);
v___x_179_ = v___x_177_;
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
else
{
lean_dec(v___x_177_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_182_; 
if (v_isShared_180_ == 0)
{
lean_ctor_set(v___x_179_, 0, v___y_174_);
v___x_182_ = v___x_179_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v___y_174_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
else
{
lean_object* v_a_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_193_; 
lean_dec_ref(v___y_174_);
v_a_186_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_193_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_193_ == 0)
{
v___x_188_ = v___x_177_;
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_a_186_);
lean_dec(v___x_177_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_191_; 
if (v_isShared_189_ == 0)
{
v___x_191_ = v___x_188_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_a_186_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
}
v___jp_194_:
{
size_t v_sz_201_; size_t v___x_202_; lean_object* v___x_203_; 
v_sz_201_ = lean_array_size(v_onAnyTactic_197_);
v___x_202_ = ((size_t)0ULL);
v___x_203_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3(v_params_166_, v_snap_167_, v___y_196_, v___y_195_, v___y_198_, v_onAnyTactic_197_, v_sz_201_, v___x_202_, v_out_199_, v___y_200_);
lean_dec_ref(v_onAnyTactic_197_);
return v___x_203_;
}
v_resetjp_206_:
{
lean_object* v_toEditableDocumentCore_209_; lean_object* v_meta_210_; lean_object* v_range_211_; lean_object* v_start_212_; lean_object* v_end_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_431_; 
v_toEditableDocumentCore_209_ = lean_ctor_get(v_a_205_, 0);
lean_inc_ref(v_toEditableDocumentCore_209_);
lean_dec(v_a_205_);
v_meta_210_ = lean_ctor_get(v_toEditableDocumentCore_209_, 0);
lean_inc_ref(v_meta_210_);
lean_dec_ref(v_toEditableDocumentCore_209_);
v_range_211_ = lean_ctor_get(v_params_166_, 3);
lean_inc_ref(v_range_211_);
v_start_212_ = lean_ctor_get(v_range_211_, 0);
v_end_213_ = lean_ctor_get(v_range_211_, 1);
v_isSharedCheck_431_ = !lean_is_exclusive(v_range_211_);
if (v_isSharedCheck_431_ == 0)
{
v___x_215_ = v_range_211_;
v_isShared_216_ = v_isSharedCheck_431_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_end_213_);
lean_inc(v_start_212_);
lean_dec(v_range_211_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_431_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v_text_217_; lean_object* v_line_218_; lean_object* v_character_219_; lean_object* v_line_220_; lean_object* v_character_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___y_225_; lean_object* v___y_226_; lean_object* v___y_227_; lean_object* v_fst_228_; lean_object* v_snd_229_; lean_object* v___y_230_; lean_object* v___x_241_; lean_object* v___x_242_; uint8_t v___x_243_; uint8_t v___x_244_; lean_object* v___y_246_; lean_object* v___y_247_; lean_object* v___y_248_; uint8_t v___y_249_; lean_object* v___y_250_; lean_object* v___y_400_; lean_object* v___y_401_; lean_object* v___y_409_; 
v_text_217_ = lean_ctor_get(v_meta_210_, 3);
lean_inc_ref(v_text_217_);
lean_dec_ref(v_meta_210_);
v_line_218_ = lean_ctor_get(v_start_212_, 0);
lean_inc(v_line_218_);
v_character_219_ = lean_ctor_get(v_start_212_, 1);
lean_inc(v_character_219_);
v_line_220_ = lean_ctor_get(v_end_213_, 0);
lean_inc(v_line_220_);
v_character_221_ = lean_ctor_get(v_end_213_, 1);
lean_inc(v_character_221_);
v___x_222_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__3);
v___x_223_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__4);
v___x_241_ = l_Lean_FileMap_lspPosToUtf8Pos(v_text_217_, v_start_212_);
v___x_242_ = l_Lean_FileMap_lspPosToUtf8Pos(v_text_217_, v_end_213_);
v___x_243_ = lean_nat_dec_eq(v_line_218_, v_line_220_);
lean_dec(v_line_220_);
lean_dec(v_line_218_);
v___x_244_ = 1;
if (v___x_243_ == 0)
{
lean_object* v___x_429_; 
lean_dec(v_character_221_);
lean_dec(v_character_219_);
v___x_429_ = lean_unsigned_to_nat(0u);
v___y_409_ = v___x_429_;
goto v___jp_408_;
}
else
{
uint8_t v___x_430_; 
v___x_430_ = lean_nat_dec_le(v_character_219_, v_character_221_);
if (v___x_430_ == 0)
{
lean_dec(v_character_221_);
v___y_409_ = v_character_219_;
goto v___jp_408_;
}
else
{
lean_dec(v_character_219_);
v___y_409_ = v_character_221_;
goto v___jp_408_;
}
}
v___jp_224_:
{
lean_object* v___x_231_; lean_object* v_toEnvExtension_232_; lean_object* v_asyncMode_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v_snd_237_; size_t v_sz_238_; size_t v___x_239_; lean_object* v___x_240_; 
v___x_231_ = lp_batteries_Batteries_CodeAction_tacticSeqCodeActionExt;
v_toEnvExtension_232_ = lean_ctor_get(v___x_231_, 0);
v_asyncMode_233_ = lean_ctor_get(v_toEnvExtension_232_, 2);
v___x_234_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_167_);
v___x_235_ = lean_box(0);
v___x_236_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_223_, v___x_231_, v___x_234_, v_asyncMode_233_, v___x_235_);
v_snd_237_ = lean_ctor_get(v___x_236_, 1);
lean_inc(v_snd_237_);
lean_dec(v___x_236_);
v_sz_238_ = lean_array_size(v_snd_237_);
v___x_239_ = ((size_t)0ULL);
v___x_240_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__4(v_params_166_, v_snap_167_, v_fst_228_, v___y_226_, v___y_227_, v_snd_229_, v_snd_237_, v_sz_238_, v___x_239_, v___y_225_, v___y_230_);
lean_dec(v_snd_237_);
return v___x_240_;
}
v___jp_245_:
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_251_ = l_Lean_Syntax_getKind(v___y_246_);
v___x_252_ = lean_box(0);
lean_inc_ref(v_snap_167_);
v___x_253_ = l_Lean_Server_Snapshots_Snapshot_infoTree(v_snap_167_);
lean_inc_ref(v___y_248_);
v___x_254_ = l_Lean_CodeAction_findInfoTree_x3f(v___x_251_, v___y_250_, v___x_252_, v___x_253_, v___y_248_, v___x_244_);
lean_dec_ref(v___y_250_);
lean_dec(v___x_251_);
if (lean_obj_tag(v___x_254_) == 1)
{
lean_object* v_val_255_; lean_object* v_snd_256_; 
v_val_255_ = lean_ctor_get(v___x_254_, 0);
lean_inc(v_val_255_);
lean_dec_ref_known(v___x_254_, 1);
v_snd_256_ = lean_ctor_get(v_val_255_, 1);
lean_inc(v_snd_256_);
if (lean_obj_tag(v_snd_256_) == 1)
{
lean_object* v_i_257_; 
v_i_257_ = lean_ctor_get(v_snd_256_, 0);
if (lean_obj_tag(v_i_257_) == 0)
{
lean_object* v_fst_258_; lean_object* v_i_259_; lean_object* v___x_260_; 
v_fst_258_ = lean_ctor_get(v_val_255_, 0);
lean_inc(v_fst_258_);
lean_dec(v_val_255_);
v_i_259_ = lean_ctor_get(v_i_257_, 0);
v___x_260_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0));
if (lean_obj_tag(v___y_247_) == 0)
{
lean_object* v_a_261_; 
lean_del_object(v___x_207_);
v_a_261_ = lean_ctor_get(v___y_247_, 0);
lean_inc(v_a_261_);
lean_dec_ref_known(v___y_247_, 1);
if (lean_obj_tag(v_a_261_) == 1)
{
lean_object* v_head_262_; lean_object* v_toCommandContextInfo_263_; lean_object* v_fst_264_; lean_object* v_parentDecl_x3f_265_; lean_object* v_autoImplicits_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_305_; 
v_head_262_ = lean_ctor_get(v_a_261_, 0);
v_toCommandContextInfo_263_ = lean_ctor_get(v_fst_258_, 0);
lean_inc_ref(v_toCommandContextInfo_263_);
v_fst_264_ = lean_ctor_get(v_head_262_, 0);
v_parentDecl_x3f_265_ = lean_ctor_get(v_fst_258_, 1);
v_autoImplicits_266_ = lean_ctor_get(v_fst_258_, 2);
v_isSharedCheck_305_ = !lean_is_exclusive(v_fst_258_);
if (v_isSharedCheck_305_ == 0)
{
lean_object* v_unused_306_; 
v_unused_306_ = lean_ctor_get(v_fst_258_, 0);
lean_dec(v_unused_306_);
v___x_268_ = v_fst_258_;
v_isShared_269_ = v_isSharedCheck_305_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_autoImplicits_266_);
lean_inc(v_parentDecl_x3f_265_);
lean_dec(v_fst_258_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_305_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v_env_270_; lean_object* v_cmdEnv_x3f_271_; lean_object* v_fileMap_272_; lean_object* v_options_273_; lean_object* v_currNamespace_274_; lean_object* v_openDecls_275_; lean_object* v_ngen_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_303_; 
v_env_270_ = lean_ctor_get(v_toCommandContextInfo_263_, 0);
v_cmdEnv_x3f_271_ = lean_ctor_get(v_toCommandContextInfo_263_, 1);
v_fileMap_272_ = lean_ctor_get(v_toCommandContextInfo_263_, 2);
v_options_273_ = lean_ctor_get(v_toCommandContextInfo_263_, 4);
v_currNamespace_274_ = lean_ctor_get(v_toCommandContextInfo_263_, 5);
v_openDecls_275_ = lean_ctor_get(v_toCommandContextInfo_263_, 6);
v_ngen_276_ = lean_ctor_get(v_toCommandContextInfo_263_, 7);
v_isSharedCheck_303_ = !lean_is_exclusive(v_toCommandContextInfo_263_);
if (v_isSharedCheck_303_ == 0)
{
lean_object* v_unused_304_; 
v_unused_304_ = lean_ctor_get(v_toCommandContextInfo_263_, 3);
lean_dec(v_unused_304_);
v___x_278_ = v_toCommandContextInfo_263_;
v_isShared_279_ = v_isSharedCheck_303_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_ngen_276_);
lean_inc(v_openDecls_275_);
lean_inc(v_currNamespace_274_);
lean_inc(v_options_273_);
lean_inc(v_fileMap_272_);
lean_inc(v_cmdEnv_x3f_271_);
lean_inc(v_env_270_);
lean_dec(v_toCommandContextInfo_263_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_303_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v_mctxBefore_280_; lean_object* v___x_281_; lean_object* v_toEnvExtension_282_; lean_object* v_asyncMode_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v_snd_287_; lean_object* v_onAnyTactic_288_; lean_object* v_onTactic_289_; lean_object* v___x_291_; 
v_mctxBefore_280_ = lean_ctor_get(v_i_259_, 1);
v___x_281_ = lp_batteries_Batteries_CodeAction_tacticCodeActionExt;
v_toEnvExtension_282_ = lean_ctor_get(v___x_281_, 0);
v_asyncMode_283_ = lean_ctor_get(v_toEnvExtension_282_, 2);
v___x_284_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_167_);
v___x_285_ = lean_box(0);
v___x_286_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_222_, v___x_281_, v___x_284_, v_asyncMode_283_, v___x_285_);
v_snd_287_ = lean_ctor_get(v___x_286_, 1);
lean_inc(v_snd_287_);
lean_dec(v___x_286_);
v_onAnyTactic_288_ = lean_ctor_get(v_snd_287_, 0);
lean_inc_ref(v_onAnyTactic_288_);
v_onTactic_289_ = lean_ctor_get(v_snd_287_, 1);
lean_inc(v_onTactic_289_);
lean_dec(v_snd_287_);
lean_inc_ref(v_mctxBefore_280_);
if (v_isShared_279_ == 0)
{
lean_ctor_set(v___x_278_, 3, v_mctxBefore_280_);
v___x_291_ = v___x_278_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_302_; 
v_reuseFailAlloc_302_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_302_, 0, v_env_270_);
lean_ctor_set(v_reuseFailAlloc_302_, 1, v_cmdEnv_x3f_271_);
lean_ctor_set(v_reuseFailAlloc_302_, 2, v_fileMap_272_);
lean_ctor_set(v_reuseFailAlloc_302_, 3, v_mctxBefore_280_);
lean_ctor_set(v_reuseFailAlloc_302_, 4, v_options_273_);
lean_ctor_set(v_reuseFailAlloc_302_, 5, v_currNamespace_274_);
lean_ctor_set(v_reuseFailAlloc_302_, 6, v_openDecls_275_);
lean_ctor_set(v_reuseFailAlloc_302_, 7, v_ngen_276_);
v___x_291_ = v_reuseFailAlloc_302_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
lean_object* v___x_293_; 
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 0, v___x_291_);
v___x_293_ = v___x_268_;
goto v_reusejp_292_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v___x_291_);
lean_ctor_set(v_reuseFailAlloc_301_, 1, v_parentDecl_x3f_265_);
lean_ctor_set(v_reuseFailAlloc_301_, 2, v_autoImplicits_266_);
v___x_293_ = v_reuseFailAlloc_301_;
goto v_reusejp_292_;
}
v_reusejp_292_:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
lean_inc(v_fst_264_);
v___x_294_ = l_Lean_Syntax_getKind(v_fst_264_);
v___x_295_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_onTactic_289_, v___x_294_);
lean_dec(v___x_294_);
lean_dec(v_onTactic_289_);
if (lean_obj_tag(v___x_295_) == 1)
{
lean_object* v_val_296_; size_t v_sz_297_; size_t v___x_298_; lean_object* v___x_299_; 
v_val_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc(v_val_296_);
lean_dec_ref_known(v___x_295_, 1);
v_sz_297_ = lean_array_size(v_val_296_);
v___x_298_ = ((size_t)0ULL);
lean_inc_ref(v_snd_256_);
lean_inc_ref(v_a_261_);
lean_inc_ref(v___x_293_);
lean_inc_ref(v_snap_167_);
lean_inc_ref(v_params_166_);
v___x_299_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__3(v_params_166_, v_snap_167_, v___x_293_, v_a_261_, v_snd_256_, v_val_296_, v_sz_297_, v___x_298_, v___x_260_, v_a_168_);
lean_dec(v_val_296_);
if (lean_obj_tag(v___x_299_) == 0)
{
lean_object* v_a_300_; 
v_a_300_ = lean_ctor_get(v___x_299_, 0);
lean_inc(v_a_300_);
lean_dec_ref_known(v___x_299_, 1);
v___y_195_ = v_a_261_;
v___y_196_ = v___x_293_;
v_onAnyTactic_197_ = v_onAnyTactic_288_;
v___y_198_ = v_snd_256_;
v_out_199_ = v_a_300_;
v___y_200_ = v_a_168_;
goto v___jp_194_;
}
else
{
lean_dec_ref(v___x_293_);
lean_dec_ref(v_onAnyTactic_288_);
lean_dec_ref_known(v_a_261_, 2);
lean_dec_ref_known(v_snd_256_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
return v___x_299_;
}
}
else
{
lean_dec(v___x_295_);
v___y_195_ = v_a_261_;
v___y_196_ = v___x_293_;
v_onAnyTactic_197_ = v_onAnyTactic_288_;
v___y_198_ = v_snd_256_;
v_out_199_ = v___x_260_;
v___y_200_ = v_a_168_;
goto v___jp_194_;
}
}
}
}
}
}
else
{
lean_dec(v_a_261_);
lean_dec(v_fst_258_);
lean_dec_ref_known(v_snd_256_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
v___y_174_ = v___x_260_;
v___y_175_ = v_a_168_;
goto v___jp_173_;
}
}
else
{
lean_object* v_a_307_; 
v_a_307_ = lean_ctor_get(v___y_247_, 1);
lean_inc(v_a_307_);
if (lean_obj_tag(v_a_307_) == 1)
{
lean_object* v_head_308_; lean_object* v_insertIdx_309_; lean_object* v_fst_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v_head_308_ = lean_ctor_get(v_a_307_, 0);
v_insertIdx_309_ = lean_ctor_get(v___y_247_, 0);
lean_inc(v_insertIdx_309_);
lean_dec_ref_known(v___y_247_, 2);
v_fst_310_ = lean_ctor_get(v_head_308_, 0);
v___x_311_ = lean_unsigned_to_nat(2u);
v___x_312_ = lean_nat_mul(v___x_311_, v_insertIdx_309_);
v___x_313_ = l_Lean_Syntax_getNumArgs(v_fst_310_);
v___x_314_ = lean_nat_dec_lt(v___x_312_, v___x_313_);
lean_dec(v___x_313_);
if (v___x_314_ == 0)
{
lean_object* v_toCommandContextInfo_315_; lean_object* v_parentDecl_x3f_316_; lean_object* v_autoImplicits_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_341_; 
lean_inc_ref(v_i_259_);
lean_dec(v___x_312_);
lean_dec_ref_known(v_snd_256_, 2);
lean_del_object(v___x_207_);
v_toCommandContextInfo_315_ = lean_ctor_get(v_fst_258_, 0);
v_parentDecl_x3f_316_ = lean_ctor_get(v_fst_258_, 1);
v_autoImplicits_317_ = lean_ctor_get(v_fst_258_, 2);
v_isSharedCheck_341_ = !lean_is_exclusive(v_fst_258_);
if (v_isSharedCheck_341_ == 0)
{
v___x_319_ = v_fst_258_;
v_isShared_320_ = v_isSharedCheck_341_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_autoImplicits_317_);
lean_inc(v_parentDecl_x3f_316_);
lean_inc(v_toCommandContextInfo_315_);
lean_dec(v_fst_258_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_341_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v_env_321_; lean_object* v_cmdEnv_x3f_322_; lean_object* v_fileMap_323_; lean_object* v_options_324_; lean_object* v_currNamespace_325_; lean_object* v_openDecls_326_; lean_object* v_ngen_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_339_; 
v_env_321_ = lean_ctor_get(v_toCommandContextInfo_315_, 0);
v_cmdEnv_x3f_322_ = lean_ctor_get(v_toCommandContextInfo_315_, 1);
v_fileMap_323_ = lean_ctor_get(v_toCommandContextInfo_315_, 2);
v_options_324_ = lean_ctor_get(v_toCommandContextInfo_315_, 4);
v_currNamespace_325_ = lean_ctor_get(v_toCommandContextInfo_315_, 5);
v_openDecls_326_ = lean_ctor_get(v_toCommandContextInfo_315_, 6);
v_ngen_327_ = lean_ctor_get(v_toCommandContextInfo_315_, 7);
v_isSharedCheck_339_ = !lean_is_exclusive(v_toCommandContextInfo_315_);
if (v_isSharedCheck_339_ == 0)
{
lean_object* v_unused_340_; 
v_unused_340_ = lean_ctor_get(v_toCommandContextInfo_315_, 3);
lean_dec(v_unused_340_);
v___x_329_ = v_toCommandContextInfo_315_;
v_isShared_330_ = v_isSharedCheck_339_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_ngen_327_);
lean_inc(v_openDecls_326_);
lean_inc(v_currNamespace_325_);
lean_inc(v_options_324_);
lean_inc(v_fileMap_323_);
lean_inc(v_cmdEnv_x3f_322_);
lean_inc(v_env_321_);
lean_dec(v_toCommandContextInfo_315_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_339_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v_mctxAfter_331_; lean_object* v_goalsAfter_332_; lean_object* v___x_334_; 
v_mctxAfter_331_ = lean_ctor_get(v_i_259_, 3);
lean_inc_ref(v_mctxAfter_331_);
v_goalsAfter_332_ = lean_ctor_get(v_i_259_, 4);
lean_inc(v_goalsAfter_332_);
lean_dec_ref(v_i_259_);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 3, v_mctxAfter_331_);
v___x_334_ = v___x_329_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_env_321_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v_cmdEnv_x3f_322_);
lean_ctor_set(v_reuseFailAlloc_338_, 2, v_fileMap_323_);
lean_ctor_set(v_reuseFailAlloc_338_, 3, v_mctxAfter_331_);
lean_ctor_set(v_reuseFailAlloc_338_, 4, v_options_324_);
lean_ctor_set(v_reuseFailAlloc_338_, 5, v_currNamespace_325_);
lean_ctor_set(v_reuseFailAlloc_338_, 6, v_openDecls_326_);
lean_ctor_set(v_reuseFailAlloc_338_, 7, v_ngen_327_);
v___x_334_ = v_reuseFailAlloc_338_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
lean_object* v___x_336_; 
if (v_isShared_320_ == 0)
{
lean_ctor_set(v___x_319_, 0, v___x_334_);
v___x_336_ = v___x_319_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_334_);
lean_ctor_set(v_reuseFailAlloc_337_, 1, v_parentDecl_x3f_316_);
lean_ctor_set(v_reuseFailAlloc_337_, 2, v_autoImplicits_317_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
v___y_225_ = v___x_260_;
v___y_226_ = v_insertIdx_309_;
v___y_227_ = v_a_307_;
v_fst_228_ = v___x_336_;
v_snd_229_ = v_goalsAfter_332_;
v___y_230_ = v_a_168_;
goto v___jp_224_;
}
}
}
}
}
else
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = l_Lean_Syntax_getArg(v_fst_310_, v___x_312_);
lean_dec(v___x_312_);
v___x_343_ = l_Lean_Syntax_getRange_x3f(v___x_342_, v___y_249_);
if (lean_obj_tag(v___x_343_) == 1)
{
lean_object* v_val_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_395_; 
v_val_344_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_395_ == 0)
{
v___x_346_ = v___x_343_;
v_isShared_347_ = v_isSharedCheck_395_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_val_344_);
lean_dec(v___x_343_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_395_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_350_; 
v___x_348_ = l_Lean_Syntax_getKind(v___x_342_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 0, v_fst_258_);
v___x_350_ = v___x_346_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_fst_258_);
v___x_350_ = v_reuseFailAlloc_394_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
lean_object* v___x_351_; 
lean_inc_ref(v___y_248_);
v___x_351_ = l_Lean_CodeAction_findInfoTree_x3f(v___x_348_, v_val_344_, v___x_350_, v_snd_256_, v___y_248_, v___y_249_);
lean_dec(v_val_344_);
lean_dec(v___x_348_);
if (lean_obj_tag(v___x_351_) == 1)
{
lean_object* v_val_352_; lean_object* v_snd_353_; 
v_val_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_val_352_);
lean_dec_ref_known(v___x_351_, 1);
v_snd_353_ = lean_ctor_get(v_val_352_, 1);
if (lean_obj_tag(v_snd_353_) == 1)
{
lean_object* v_i_354_; 
v_i_354_ = lean_ctor_get(v_snd_353_, 0);
lean_inc_ref(v_i_354_);
if (lean_obj_tag(v_i_354_) == 0)
{
lean_object* v_fst_355_; lean_object* v_toCommandContextInfo_356_; lean_object* v_i_357_; lean_object* v_parentDecl_x3f_358_; lean_object* v_autoImplicits_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_383_; 
lean_del_object(v___x_207_);
v_fst_355_ = lean_ctor_get(v_val_352_, 0);
lean_inc(v_fst_355_);
lean_dec(v_val_352_);
v_toCommandContextInfo_356_ = lean_ctor_get(v_fst_355_, 0);
lean_inc_ref(v_toCommandContextInfo_356_);
v_i_357_ = lean_ctor_get(v_i_354_, 0);
lean_inc_ref(v_i_357_);
lean_dec_ref_known(v_i_354_, 1);
v_parentDecl_x3f_358_ = lean_ctor_get(v_fst_355_, 1);
v_autoImplicits_359_ = lean_ctor_get(v_fst_355_, 2);
v_isSharedCheck_383_ = !lean_is_exclusive(v_fst_355_);
if (v_isSharedCheck_383_ == 0)
{
lean_object* v_unused_384_; 
v_unused_384_ = lean_ctor_get(v_fst_355_, 0);
lean_dec(v_unused_384_);
v___x_361_ = v_fst_355_;
v_isShared_362_ = v_isSharedCheck_383_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_autoImplicits_359_);
lean_inc(v_parentDecl_x3f_358_);
lean_dec(v_fst_355_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_383_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v_env_363_; lean_object* v_cmdEnv_x3f_364_; lean_object* v_fileMap_365_; lean_object* v_options_366_; lean_object* v_currNamespace_367_; lean_object* v_openDecls_368_; lean_object* v_ngen_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_381_; 
v_env_363_ = lean_ctor_get(v_toCommandContextInfo_356_, 0);
v_cmdEnv_x3f_364_ = lean_ctor_get(v_toCommandContextInfo_356_, 1);
v_fileMap_365_ = lean_ctor_get(v_toCommandContextInfo_356_, 2);
v_options_366_ = lean_ctor_get(v_toCommandContextInfo_356_, 4);
v_currNamespace_367_ = lean_ctor_get(v_toCommandContextInfo_356_, 5);
v_openDecls_368_ = lean_ctor_get(v_toCommandContextInfo_356_, 6);
v_ngen_369_ = lean_ctor_get(v_toCommandContextInfo_356_, 7);
v_isSharedCheck_381_ = !lean_is_exclusive(v_toCommandContextInfo_356_);
if (v_isSharedCheck_381_ == 0)
{
lean_object* v_unused_382_; 
v_unused_382_ = lean_ctor_get(v_toCommandContextInfo_356_, 3);
lean_dec(v_unused_382_);
v___x_371_ = v_toCommandContextInfo_356_;
v_isShared_372_ = v_isSharedCheck_381_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_ngen_369_);
lean_inc(v_openDecls_368_);
lean_inc(v_currNamespace_367_);
lean_inc(v_options_366_);
lean_inc(v_fileMap_365_);
lean_inc(v_cmdEnv_x3f_364_);
lean_inc(v_env_363_);
lean_dec(v_toCommandContextInfo_356_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_381_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v_mctxBefore_373_; lean_object* v_goalsBefore_374_; lean_object* v___x_376_; 
v_mctxBefore_373_ = lean_ctor_get(v_i_357_, 1);
lean_inc_ref(v_mctxBefore_373_);
v_goalsBefore_374_ = lean_ctor_get(v_i_357_, 2);
lean_inc(v_goalsBefore_374_);
lean_dec_ref(v_i_357_);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 3, v_mctxBefore_373_);
v___x_376_ = v___x_371_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_env_363_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v_cmdEnv_x3f_364_);
lean_ctor_set(v_reuseFailAlloc_380_, 2, v_fileMap_365_);
lean_ctor_set(v_reuseFailAlloc_380_, 3, v_mctxBefore_373_);
lean_ctor_set(v_reuseFailAlloc_380_, 4, v_options_366_);
lean_ctor_set(v_reuseFailAlloc_380_, 5, v_currNamespace_367_);
lean_ctor_set(v_reuseFailAlloc_380_, 6, v_openDecls_368_);
lean_ctor_set(v_reuseFailAlloc_380_, 7, v_ngen_369_);
v___x_376_ = v_reuseFailAlloc_380_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
lean_object* v___x_378_; 
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 0, v___x_376_);
v___x_378_ = v___x_361_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v___x_376_);
lean_ctor_set(v_reuseFailAlloc_379_, 1, v_parentDecl_x3f_358_);
lean_ctor_set(v_reuseFailAlloc_379_, 2, v_autoImplicits_359_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
v___y_225_ = v___x_260_;
v___y_226_ = v_insertIdx_309_;
v___y_227_ = v_a_307_;
v_fst_228_ = v___x_378_;
v_snd_229_ = v_goalsBefore_374_;
v___y_230_ = v_a_168_;
goto v___jp_224_;
}
}
}
}
}
else
{
lean_object* v___x_386_; 
lean_dec_ref(v_i_354_);
lean_dec(v_val_352_);
lean_dec(v_insertIdx_309_);
lean_dec_ref_known(v_a_307_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 0, v___x_260_);
v___x_386_ = v___x_207_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_260_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
}
else
{
lean_object* v___x_389_; 
lean_dec(v_val_352_);
lean_dec(v_insertIdx_309_);
lean_dec_ref_known(v_a_307_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 0, v___x_260_);
v___x_389_ = v___x_207_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_260_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
else
{
lean_object* v___x_392_; 
lean_dec(v___x_351_);
lean_dec(v_insertIdx_309_);
lean_dec_ref_known(v_a_307_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 0, v___x_260_);
v___x_392_ = v___x_207_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_260_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
}
else
{
lean_object* v___x_397_; 
lean_dec(v___x_343_);
lean_dec(v___x_342_);
lean_dec(v_insertIdx_309_);
lean_dec_ref_known(v_a_307_, 2);
lean_dec(v_fst_258_);
lean_dec_ref_known(v_snd_256_, 2);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 0, v___x_260_);
v___x_397_ = v___x_207_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v___x_260_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
}
else
{
lean_dec_ref_known(v___y_247_, 2);
lean_dec(v_a_307_);
lean_dec(v_fst_258_);
lean_dec_ref_known(v_snd_256_, 2);
lean_del_object(v___x_207_);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
v___y_174_ = v___x_260_;
v___y_175_ = v_a_168_;
goto v___jp_173_;
}
}
}
else
{
lean_dec_ref_known(v_snd_256_, 2);
lean_dec(v_val_255_);
lean_dec_ref(v___y_247_);
lean_del_object(v___x_207_);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
goto v___jp_170_;
}
}
else
{
lean_dec(v_snd_256_);
lean_dec(v_val_255_);
lean_dec_ref(v___y_247_);
lean_del_object(v___x_207_);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
goto v___jp_170_;
}
}
else
{
lean_dec(v___x_254_);
lean_dec_ref(v___y_247_);
lean_del_object(v___x_207_);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
goto v___jp_170_;
}
}
v___jp_399_:
{
uint8_t v___x_402_; lean_object* v___f_403_; lean_object* v___x_404_; 
v___x_402_ = 0;
v___f_403_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__5));
v___x_404_ = l_Lean_Syntax_getRange_x3f(v___y_401_, v___x_402_);
if (lean_obj_tag(v___x_404_) == 0)
{
lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_405_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9, &lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9_once, _init_lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__9);
v___x_406_ = lp_batteries_panic___at___00Batteries_CodeAction_tacticCodeActionProvider_spec__5(v___x_405_);
v___y_246_ = v___y_401_;
v___y_247_ = v___y_400_;
v___y_248_ = v___f_403_;
v___y_249_ = v___x_402_;
v___y_250_ = v___x_406_;
goto v___jp_245_;
}
else
{
lean_object* v_val_407_; 
v_val_407_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_val_407_);
lean_dec_ref_known(v___x_404_, 1);
v___y_246_ = v___y_401_;
v___y_247_ = v___y_400_;
v___y_248_ = v___f_403_;
v___y_249_ = v___x_402_;
v___y_250_ = v_val_407_;
goto v___jp_245_;
}
}
v___jp_408_:
{
lean_object* v_stx_410_; lean_object* v___f_411_; lean_object* v___x_413_; 
v_stx_410_ = lean_ctor_get(v_snap_167_, 0);
v___f_411_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__2___boxed), 3, 2);
lean_closure_set(v___f_411_, 0, v_text_217_);
lean_closure_set(v___f_411_, 1, v___y_409_);
if (v_isShared_216_ == 0)
{
lean_ctor_set(v___x_215_, 1, v___x_242_);
lean_ctor_set(v___x_215_, 0, v___x_241_);
v___x_413_ = v___x_215_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_241_);
lean_ctor_set(v_reuseFailAlloc_428_, 1, v___x_242_);
v___x_413_ = v_reuseFailAlloc_428_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
lean_object* v___x_414_; 
lean_inc(v_stx_410_);
v___x_414_ = l_Lean_CodeAction_findTactic_x3f(v___f_411_, v___x_413_, v_stx_410_);
if (lean_obj_tag(v___x_414_) == 1)
{
lean_object* v_val_415_; 
v_val_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc(v_val_415_);
lean_dec_ref_known(v___x_414_, 1);
if (lean_obj_tag(v_val_415_) == 0)
{
lean_object* v_a_416_; 
v_a_416_ = lean_ctor_get(v_val_415_, 0);
if (lean_obj_tag(v_a_416_) == 1)
{
lean_object* v_head_417_; lean_object* v_fst_418_; 
v_head_417_ = lean_ctor_get(v_a_416_, 0);
v_fst_418_ = lean_ctor_get(v_head_417_, 0);
lean_inc(v_fst_418_);
v___y_400_ = v_val_415_;
v___y_401_ = v_fst_418_;
goto v___jp_399_;
}
else
{
lean_object* v___x_419_; 
v___x_419_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(v_val_415_);
v___y_400_ = v_val_415_;
v___y_401_ = v___x_419_;
goto v___jp_399_;
}
}
else
{
lean_object* v_a_420_; 
v_a_420_ = lean_ctor_get(v_val_415_, 1);
if (lean_obj_tag(v_a_420_) == 1)
{
lean_object* v_tail_421_; 
v_tail_421_ = lean_ctor_get(v_a_420_, 1);
if (lean_obj_tag(v_tail_421_) == 1)
{
lean_object* v_head_422_; lean_object* v_fst_423_; 
v_head_422_ = lean_ctor_get(v_tail_421_, 0);
v_fst_423_ = lean_ctor_get(v_head_422_, 0);
lean_inc(v_fst_423_);
v___y_400_ = v_val_415_;
v___y_401_ = v_fst_423_;
goto v___jp_399_;
}
else
{
lean_object* v___x_424_; 
v___x_424_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(v_val_415_);
v___y_400_ = v_val_415_;
v___y_401_ = v___x_424_;
goto v___jp_399_;
}
}
else
{
lean_object* v___x_425_; 
v___x_425_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___lam__0(v_val_415_);
v___y_400_ = v_val_415_;
v___y_401_ = v___x_425_;
goto v___jp_399_;
}
}
}
else
{
lean_object* v___x_426_; lean_object* v___x_427_; 
lean_dec(v___x_414_);
lean_del_object(v___x_207_);
lean_dec_ref(v_snap_167_);
lean_dec_ref(v_params_166_);
v___x_426_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___closed__0));
v___x_427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_427_, 0, v___x_426_);
return v___x_427_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_tacticCodeActionProvider___boxed(lean_object* v_params_433_, lean_object* v_snap_434_, lean_object* v_a_435_, lean_object* v_a_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_batteries_Batteries_CodeAction_tacticCodeActionProvider(v_params_433_, v_snap_434_, v_a_435_);
lean_dec_ref(v_a_435_);
return v_res_437_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_CodeAction_Basic(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_BuiltinTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_BuiltinNotation(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_CodeActions_Provider(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_CodeAction_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_BuiltinTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_BuiltinNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_CodeActions_Provider(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_BuiltinTerm(uint8_t builtin);
lean_object* initialize_Lean_Elab_BuiltinNotation(uint8_t builtin);
lean_object* initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* initialize_Lean_Server_CodeActions_Provider(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_CodeAction_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_BuiltinTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_BuiltinNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_CodeActions_Provider(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_CodeAction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_CodeAction_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
