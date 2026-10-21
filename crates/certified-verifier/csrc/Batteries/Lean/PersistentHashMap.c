// Lean compiler output
// Module: Batteries.Lean.PersistentHashMap
// Imports: public import Init public meta import Init public import Lean.Data.PersistentHashMap
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
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__2_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__3 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__3_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__4 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__4_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__5 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__5_value;
static const lean_closure_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__0_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__1_value)}};
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__7 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__7_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__2_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__3_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__4_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__5_value)}};
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__8_value),((lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__6_value)}};
static const lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__9 = (const lean_object*)&lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_m_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_fst_5_; lean_object* v_snd_6_; lean_object* v___x_7_; 
v_fst_5_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_fst_5_);
v_snd_6_ = lean_ctor_get(v_x_4_, 1);
lean_inc(v_snd_6_);
lean_dec_ref(v_x_4_);
v___x_7_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_1_, v_inst_2_, v_m_3_, v_fst_5_, v_snd_6_);
return v___x_7_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_8_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__0);
v___x_10_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList___redArg(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_xs_13_){
_start:
{
lean_object* v___f_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___f_14_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_ofList___redArg___lam__0), 4, 2);
lean_closure_set(v___f_14_, 0, v_inst_11_);
lean_closure_set(v___f_14_, 1, v_inst_12_);
v___x_15_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1, &lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1_once, _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1);
v___x_16_ = l_List_foldl___redArg(v___f_14_, v___x_15_, v_xs_13_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofList(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_00_u03b2_20_, lean_object* v_xs_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_batteries_Lean_PersistentHashMap_ofList___redArg(v_inst_18_, v_inst_19_, v_xs_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith___redArg___lam__0(lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_f_25_, lean_object* v_m_26_, lean_object* v_x_27_){
_start:
{
lean_object* v_fst_28_; lean_object* v_snd_29_; lean_object* v___x_30_; 
v_fst_28_ = lean_ctor_get(v_x_27_, 0);
lean_inc_n(v_fst_28_, 2);
v_snd_29_ = lean_ctor_get(v_x_27_, 1);
lean_inc(v_snd_29_);
lean_dec_ref(v_x_27_);
lean_inc_ref(v_inst_24_);
lean_inc_ref(v_inst_23_);
v___x_30_ = l_Lean_PersistentHashMap_find_x3f___redArg(v_inst_23_, v_inst_24_, v_m_26_, v_fst_28_);
if (lean_obj_tag(v___x_30_) == 0)
{
lean_object* v___x_31_; 
lean_dec(v_f_25_);
v___x_31_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_23_, v_inst_24_, v_m_26_, v_fst_28_, v_snd_29_);
return v___x_31_;
}
else
{
lean_object* v_val_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v_val_32_ = lean_ctor_get(v___x_30_, 0);
lean_inc(v_val_32_);
lean_dec_ref_known(v___x_30_, 1);
lean_inc(v_fst_28_);
v___x_33_ = lean_apply_3(v_f_25_, v_fst_28_, v_snd_29_, v_val_32_);
v___x_34_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_23_, v_inst_24_, v_m_26_, v_fst_28_, v___x_33_);
return v___x_34_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith___redArg(lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_xs_37_, lean_object* v_f_38_){
_start:
{
lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___f_39_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_ofListWith___redArg___lam__0), 5, 3);
lean_closure_set(v___f_39_, 0, v_inst_35_);
lean_closure_set(v___f_39_, 1, v_inst_36_);
lean_closure_set(v___f_39_, 2, v_f_38_);
v___x_40_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1, &lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1_once, _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1);
v___x_41_ = l_List_foldl___redArg(v___f_39_, v___x_40_, v_xs_37_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofListWith(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_00_u03b2_45_, lean_object* v_xs_46_, lean_object* v_f_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_batteries_Lean_PersistentHashMap_ofListWith___redArg(v_inst_43_, v_inst_44_, v_xs_46_, v_f_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg___lam__0(lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_x1_51_, lean_object* v_x2_52_){
_start:
{
lean_object* v_fst_53_; lean_object* v_snd_54_; lean_object* v___x_55_; 
v_fst_53_ = lean_ctor_get(v_x2_52_, 0);
lean_inc(v_fst_53_);
v_snd_54_ = lean_ctor_get(v_x2_52_, 1);
lean_inc(v_snd_54_);
lean_dec_ref(v_x2_52_);
v___x_55_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_49_, v_inst_50_, v_x1_51_, v_fst_53_, v_snd_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray___redArg(lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_xs_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_78_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1, &lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1_once, _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1);
v___x_79_ = lean_unsigned_to_nat(0u);
v___x_80_ = lean_array_get_size(v_xs_77_);
v___x_81_ = ((lean_object*)(lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__9));
v___x_82_ = lean_nat_dec_lt(v___x_79_, v___x_80_);
if (v___x_82_ == 0)
{
lean_dec_ref(v_xs_77_);
lean_dec_ref(v_inst_76_);
lean_dec_ref(v_inst_75_);
return v___x_78_;
}
else
{
lean_object* v___f_83_; uint8_t v___x_84_; 
v___f_83_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_ofArray___redArg___lam__0), 4, 2);
lean_closure_set(v___f_83_, 0, v_inst_75_);
lean_closure_set(v___f_83_, 1, v_inst_76_);
v___x_84_ = lean_nat_dec_le(v___x_80_, v___x_80_);
if (v___x_84_ == 0)
{
if (v___x_82_ == 0)
{
lean_dec_ref(v___f_83_);
lean_dec_ref(v_xs_77_);
return v___x_78_;
}
else
{
size_t v___x_85_; size_t v___x_86_; lean_object* v___x_87_; 
v___x_85_ = ((size_t)0ULL);
v___x_86_ = lean_usize_of_nat(v___x_80_);
v___x_87_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_81_, v___f_83_, v_xs_77_, v___x_85_, v___x_86_, v___x_78_);
return v___x_87_;
}
}
else
{
size_t v___x_88_; size_t v___x_89_; lean_object* v___x_90_; 
v___x_88_ = ((size_t)0ULL);
v___x_89_ = lean_usize_of_nat(v___x_80_);
v___x_90_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_81_, v___f_83_, v_xs_77_, v___x_88_, v___x_89_, v___x_78_);
return v___x_90_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArray(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_00_u03b2_94_, lean_object* v_xs_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_batteries_Lean_PersistentHashMap_ofArray___redArg(v_inst_92_, v_inst_93_, v_xs_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg___lam__0(lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_f_99_, lean_object* v_x1_100_, lean_object* v_x2_101_){
_start:
{
lean_object* v_fst_102_; lean_object* v_snd_103_; lean_object* v___x_104_; 
v_fst_102_ = lean_ctor_get(v_x2_101_, 0);
lean_inc_n(v_fst_102_, 2);
v_snd_103_ = lean_ctor_get(v_x2_101_, 1);
lean_inc(v_snd_103_);
lean_dec_ref(v_x2_101_);
lean_inc_ref(v_inst_98_);
lean_inc_ref(v_inst_97_);
v___x_104_ = l_Lean_PersistentHashMap_find_x3f___redArg(v_inst_97_, v_inst_98_, v_x1_100_, v_fst_102_);
if (lean_obj_tag(v___x_104_) == 0)
{
lean_object* v___x_105_; 
lean_dec(v_f_99_);
v___x_105_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_97_, v_inst_98_, v_x1_100_, v_fst_102_, v_snd_103_);
return v___x_105_;
}
else
{
lean_object* v_val_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v_val_106_ = lean_ctor_get(v___x_104_, 0);
lean_inc(v_val_106_);
lean_dec_ref_known(v___x_104_, 1);
lean_inc(v_fst_102_);
v___x_107_ = lean_apply_3(v_f_99_, v_fst_102_, v_snd_103_, v_val_106_);
v___x_108_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_97_, v_inst_98_, v_x1_100_, v_fst_102_, v___x_107_);
return v___x_108_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg(lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_xs_111_, lean_object* v_f_112_){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_113_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1, &lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1_once, _init_lp_batteries_Lean_PersistentHashMap_ofList___redArg___closed__1);
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = lean_array_get_size(v_xs_111_);
v___x_116_ = ((lean_object*)(lp_batteries_Lean_PersistentHashMap_ofArray___redArg___closed__9));
v___x_117_ = lean_nat_dec_lt(v___x_114_, v___x_115_);
if (v___x_117_ == 0)
{
lean_dec(v_f_112_);
lean_dec_ref(v_xs_111_);
lean_dec_ref(v_inst_110_);
lean_dec_ref(v_inst_109_);
return v___x_113_;
}
else
{
lean_object* v___f_118_; uint8_t v___x_119_; 
v___f_118_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg___lam__0), 5, 3);
lean_closure_set(v___f_118_, 0, v_inst_109_);
lean_closure_set(v___f_118_, 1, v_inst_110_);
lean_closure_set(v___f_118_, 2, v_f_112_);
v___x_119_ = lean_nat_dec_le(v___x_115_, v___x_115_);
if (v___x_119_ == 0)
{
if (v___x_117_ == 0)
{
lean_dec_ref(v___f_118_);
lean_dec_ref(v_xs_111_);
return v___x_113_;
}
else
{
size_t v___x_120_; size_t v___x_121_; lean_object* v___x_122_; 
v___x_120_ = ((size_t)0ULL);
v___x_121_ = lean_usize_of_nat(v___x_115_);
v___x_122_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_116_, v___f_118_, v_xs_111_, v___x_120_, v___x_121_, v___x_113_);
return v___x_122_;
}
}
else
{
size_t v___x_123_; size_t v___x_124_; lean_object* v___x_125_; 
v___x_123_ = ((size_t)0ULL);
v___x_124_ = lean_usize_of_nat(v___x_115_);
v___x_125_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_116_, v___f_118_, v_xs_111_, v___x_123_, v___x_124_, v___x_113_);
return v___x_125_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_ofArrayWith(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_00_u03b2_129_, lean_object* v_xs_130_, lean_object* v_f_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_batteries_Lean_PersistentHashMap_ofArrayWith___redArg(v_inst_127_, v_inst_128_, v_xs_130_, v_f_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__0(lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_map_135_, lean_object* v_k_136_, lean_object* v_toPure_137_, lean_object* v_____do__lift_138_){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_139_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_133_, v_inst_134_, v_map_135_, v_k_136_, v_____do__lift_138_);
v___x_140_ = lean_apply_2(v_toPure_137_, lean_box(0), v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__1(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_toPure_143_, lean_object* v_f_144_, lean_object* v_toBind_145_, lean_object* v_map_146_, lean_object* v_k_147_, lean_object* v_v_u2082_148_){
_start:
{
lean_object* v___x_149_; 
lean_inc(v_k_147_);
lean_inc_ref(v_inst_142_);
lean_inc_ref(v_inst_141_);
v___x_149_ = l_Lean_PersistentHashMap_find_x3f___redArg(v_inst_141_, v_inst_142_, v_map_146_, v_k_147_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec(v_toBind_145_);
lean_dec(v_f_144_);
v___x_150_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_141_, v_inst_142_, v_map_146_, v_k_147_, v_v_u2082_148_);
v___x_151_ = lean_apply_2(v_toPure_143_, lean_box(0), v___x_150_);
return v___x_151_;
}
else
{
lean_object* v_val_152_; lean_object* v___f_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v_val_152_ = lean_ctor_get(v___x_149_, 0);
lean_inc(v_val_152_);
lean_dec_ref_known(v___x_149_, 1);
lean_inc(v_k_147_);
v___f_153_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__0), 6, 5);
lean_closure_set(v___f_153_, 0, v_inst_141_);
lean_closure_set(v___f_153_, 1, v_inst_142_);
lean_closure_set(v___f_153_, 2, v_map_146_);
lean_closure_set(v___f_153_, 3, v_k_147_);
lean_closure_set(v___f_153_, 4, v_toPure_143_);
v___x_154_ = lean_apply_3(v_f_144_, v_k_147_, v_val_152_, v_v_u2082_148_);
v___x_155_ = lean_apply_4(v_toBind_145_, lean_box(0), lean_box(0), v___x_154_, v___f_153_);
return v___x_155_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg(lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_self_159_, lean_object* v_other_160_, lean_object* v_f_161_){
_start:
{
lean_object* v_toApplicative_162_; lean_object* v_toBind_163_; lean_object* v_toPure_164_; lean_object* v___f_165_; lean_object* v___x_166_; 
v_toApplicative_162_ = lean_ctor_get(v_inst_158_, 0);
v_toBind_163_ = lean_ctor_get(v_inst_158_, 1);
v_toPure_164_ = lean_ctor_get(v_toApplicative_162_, 1);
lean_inc(v_toBind_163_);
lean_inc(v_toPure_164_);
v___f_165_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg___lam__1), 8, 5);
lean_closure_set(v___f_165_, 0, v_inst_156_);
lean_closure_set(v___f_165_, 1, v_inst_157_);
lean_closure_set(v___f_165_, 2, v_toPure_164_);
lean_closure_set(v___f_165_, 3, v_f_161_);
lean_closure_set(v___f_165_, 4, v_toBind_163_);
v___x_166_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_158_, v___f_165_, v_other_160_, v_self_159_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWithM(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_m_170_, lean_object* v_00_u03b2_171_, lean_object* v_inst_172_, lean_object* v_self_173_, lean_object* v_other_174_, lean_object* v_f_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_batteries_Lean_PersistentHashMap_mergeWithM___redArg(v_inst_168_, v_inst_169_, v_inst_172_, v_self_173_, v_other_174_, v_f_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith___redArg___lam__0(lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_f_179_, lean_object* v_map_180_, lean_object* v_k_181_, lean_object* v_v_u2082_182_){
_start:
{
lean_object* v___x_183_; 
lean_inc(v_k_181_);
lean_inc_ref(v_inst_178_);
lean_inc_ref(v_inst_177_);
v___x_183_ = l_Lean_PersistentHashMap_find_x3f___redArg(v_inst_177_, v_inst_178_, v_map_180_, v_k_181_);
if (lean_obj_tag(v___x_183_) == 0)
{
lean_object* v___x_184_; 
lean_dec(v_f_179_);
v___x_184_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_177_, v_inst_178_, v_map_180_, v_k_181_, v_v_u2082_182_);
return v___x_184_;
}
else
{
lean_object* v_val_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v_val_185_ = lean_ctor_get(v___x_183_, 0);
lean_inc(v_val_185_);
lean_dec_ref_known(v___x_183_, 1);
lean_inc(v_k_181_);
v___x_186_ = lean_apply_3(v_f_179_, v_k_181_, v_val_185_, v_v_u2082_182_);
v___x_187_ = l_Lean_PersistentHashMap_insert___redArg(v_inst_177_, v_inst_178_, v_map_180_, v_k_181_, v___x_186_);
return v___x_187_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith___redArg(lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_self_190_, lean_object* v_other_191_, lean_object* v_f_192_){
_start:
{
lean_object* v___f_193_; lean_object* v___x_194_; 
v___f_193_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_mergeWith___redArg___lam__0), 6, 3);
lean_closure_set(v___f_193_, 0, v_inst_188_);
lean_closure_set(v___f_193_, 1, v_inst_189_);
lean_closure_set(v___f_193_, 2, v_f_192_);
v___x_194_ = l_Lean_PersistentHashMap_foldl___redArg(v_other_191_, v___f_193_, v_self_190_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_mergeWith(lean_object* v_00_u03b1_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_00_u03b2_198_, lean_object* v_self_199_, lean_object* v_other_200_, lean_object* v_f_201_){
_start:
{
lean_object* v___f_202_; lean_object* v___x_203_; 
v___f_202_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_mergeWith___redArg___lam__0), 6, 3);
lean_closure_set(v___f_202_, 0, v_inst_196_);
lean_closure_set(v___f_202_, 1, v_inst_197_);
lean_closure_set(v___f_202_, 2, v_f_201_);
v___x_203_ = l_Lean_PersistentHashMap_foldl___redArg(v_other_200_, v___f_202_, v_self_199_);
return v___x_203_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_PersistentHashMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin) {
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
lean_object* initialize_Lean_Data_PersistentHashMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
}
#ifdef __cplusplus
}
#endif
