// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vinner_product_4x16.h for the primary calling header

#include "Vinner_product_4x16__pch.h"
#include "Vinner_product_4x16___024root.h"

void Vinner_product_4x16___024root___eval_act(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vinner_product_4x16___024root___nba_sequent__TOP__0(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___nba_sequent__TOP__0\n"); );
    // Init
    CData/*0:0*/ __Vdlyvset__inner_product_4x16__DOT__activations_reg__v0;
    __Vdlyvset__inner_product_4x16__DOT__activations_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__activations_reg__v4;
    __Vdlyvval__inner_product_4x16__DOT__activations_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__inner_product_4x16__DOT__activations_reg__v4;
    __Vdlyvset__inner_product_4x16__DOT__activations_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__activations_reg__v5;
    __Vdlyvval__inner_product_4x16__DOT__activations_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__activations_reg__v6;
    __Vdlyvval__inner_product_4x16__DOT__activations_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__activations_reg__v7;
    __Vdlyvval__inner_product_4x16__DOT__activations_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__inner_product_4x16__DOT__weights_reg__v0;
    __Vdlyvset__inner_product_4x16__DOT__weights_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__weights_reg__v4;
    __Vdlyvval__inner_product_4x16__DOT__weights_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__inner_product_4x16__DOT__weights_reg__v4;
    __Vdlyvset__inner_product_4x16__DOT__weights_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__weights_reg__v5;
    __Vdlyvval__inner_product_4x16__DOT__weights_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__weights_reg__v6;
    __Vdlyvval__inner_product_4x16__DOT__weights_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__inner_product_4x16__DOT__weights_reg__v7;
    __Vdlyvval__inner_product_4x16__DOT__weights_reg__v7 = 0;
    // Body
    __Vdlyvset__inner_product_4x16__DOT__weights_reg__v0 = 0U;
    __Vdlyvset__inner_product_4x16__DOT__weights_reg__v4 = 0U;
    __Vdlyvset__inner_product_4x16__DOT__activations_reg__v0 = 0U;
    __Vdlyvset__inner_product_4x16__DOT__activations_reg__v4 = 0U;
    if (vlSelf->reset) {
        vlSelf->inner_product_4x16__DOT__i = 4U;
        __Vdlyvset__inner_product_4x16__DOT__weights_reg__v0 = 1U;
        __Vdlyvset__inner_product_4x16__DOT__activations_reg__v0 = 1U;
        vlSelf->result = 0U;
    } else if (vlSelf->enable) {
        vlSelf->inner_product_4x16__DOT__i = 4U;
        __Vdlyvval__inner_product_4x16__DOT__weights_reg__v4 
            = vlSelf->weights[0U];
        __Vdlyvset__inner_product_4x16__DOT__weights_reg__v4 = 1U;
        __Vdlyvval__inner_product_4x16__DOT__weights_reg__v5 
            = vlSelf->weights[1U];
        __Vdlyvval__inner_product_4x16__DOT__weights_reg__v6 
            = vlSelf->weights[2U];
        __Vdlyvval__inner_product_4x16__DOT__weights_reg__v7 
            = vlSelf->weights[3U];
        __Vdlyvval__inner_product_4x16__DOT__activations_reg__v4 
            = vlSelf->activations[0U];
        __Vdlyvset__inner_product_4x16__DOT__activations_reg__v4 = 1U;
        __Vdlyvval__inner_product_4x16__DOT__activations_reg__v5 
            = vlSelf->activations[1U];
        __Vdlyvval__inner_product_4x16__DOT__activations_reg__v6 
            = vlSelf->activations[2U];
        __Vdlyvval__inner_product_4x16__DOT__activations_reg__v7 
            = vlSelf->activations[3U];
        vlSelf->result = (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                        vlSelf->inner_product_4x16__DOT__activations_reg
                                                        [0U]), 
                                      VL_EXTENDS_II(32,16, 
                                                    vlSelf->inner_product_4x16__DOT__weights_reg
                                                    [0U])) 
                          + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                           vlSelf->inner_product_4x16__DOT__activations_reg
                                                           [1U]), 
                                         VL_EXTENDS_II(32,16, 
                                                       vlSelf->inner_product_4x16__DOT__weights_reg
                                                       [1U])) 
                             + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                              vlSelf->inner_product_4x16__DOT__activations_reg
                                                              [2U]), 
                                            VL_EXTENDS_II(32,16, 
                                                          vlSelf->inner_product_4x16__DOT__weights_reg
                                                          [2U])) 
                                + VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->inner_product_4x16__DOT__activations_reg
                                                                [3U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->inner_product_4x16__DOT__weights_reg
                                                            [3U])))));
    }
    if (__Vdlyvset__inner_product_4x16__DOT__activations_reg__v0) {
        vlSelf->inner_product_4x16__DOT__activations_reg[0U] = 0U;
        vlSelf->inner_product_4x16__DOT__activations_reg[1U] = 0U;
        vlSelf->inner_product_4x16__DOT__activations_reg[2U] = 0U;
        vlSelf->inner_product_4x16__DOT__activations_reg[3U] = 0U;
    }
    if (__Vdlyvset__inner_product_4x16__DOT__activations_reg__v4) {
        vlSelf->inner_product_4x16__DOT__activations_reg[0U] 
            = __Vdlyvval__inner_product_4x16__DOT__activations_reg__v4;
        vlSelf->inner_product_4x16__DOT__activations_reg[1U] 
            = __Vdlyvval__inner_product_4x16__DOT__activations_reg__v5;
        vlSelf->inner_product_4x16__DOT__activations_reg[2U] 
            = __Vdlyvval__inner_product_4x16__DOT__activations_reg__v6;
        vlSelf->inner_product_4x16__DOT__activations_reg[3U] 
            = __Vdlyvval__inner_product_4x16__DOT__activations_reg__v7;
    }
    if (__Vdlyvset__inner_product_4x16__DOT__weights_reg__v0) {
        vlSelf->inner_product_4x16__DOT__weights_reg[0U] = 0U;
        vlSelf->inner_product_4x16__DOT__weights_reg[1U] = 0U;
        vlSelf->inner_product_4x16__DOT__weights_reg[2U] = 0U;
        vlSelf->inner_product_4x16__DOT__weights_reg[3U] = 0U;
    }
    if (__Vdlyvset__inner_product_4x16__DOT__weights_reg__v4) {
        vlSelf->inner_product_4x16__DOT__weights_reg[0U] 
            = __Vdlyvval__inner_product_4x16__DOT__weights_reg__v4;
        vlSelf->inner_product_4x16__DOT__weights_reg[1U] 
            = __Vdlyvval__inner_product_4x16__DOT__weights_reg__v5;
        vlSelf->inner_product_4x16__DOT__weights_reg[2U] 
            = __Vdlyvval__inner_product_4x16__DOT__weights_reg__v6;
        vlSelf->inner_product_4x16__DOT__weights_reg[3U] 
            = __Vdlyvval__inner_product_4x16__DOT__weights_reg__v7;
    }
}

void Vinner_product_4x16___024root___eval_nba(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vinner_product_4x16___024root___nba_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[1U] = 1U;
    }
}

void Vinner_product_4x16___024root___eval_triggers__act(Vinner_product_4x16___024root* vlSelf);

bool Vinner_product_4x16___024root___eval_phase__act(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vinner_product_4x16___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vinner_product_4x16___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vinner_product_4x16___024root___eval_phase__nba(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vinner_product_4x16___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vinner_product_4x16___024root___dump_triggers__nba(Vinner_product_4x16___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vinner_product_4x16___024root___dump_triggers__act(Vinner_product_4x16___024root* vlSelf);
#endif  // VL_DEBUG

void Vinner_product_4x16___024root___eval(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vinner_product_4x16___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/inner_product_4x16.v", 1, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vinner_product_4x16___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/inner_product_4x16.v", 1, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vinner_product_4x16___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vinner_product_4x16___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vinner_product_4x16___024root___eval_debug_assertions(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->reset & 0xfeU))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY((vlSelf->enable & 0xfeU))) {
        Verilated::overWidthError("enable");}
}
#endif  // VL_DEBUG
