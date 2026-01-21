// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vinner_product_4x16.h for the primary calling header

#include "Vinner_product_4x16__pch.h"
#include "Vinner_product_4x16___024root.h"

VL_ATTR_COLD void Vinner_product_4x16___024root___eval_static(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vinner_product_4x16___024root___eval_initial(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_initial\n"); );
    // Body
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
}

VL_ATTR_COLD void Vinner_product_4x16___024root___eval_final(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_final\n"); );
}

VL_ATTR_COLD void Vinner_product_4x16___024root___eval_settle(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___eval_settle\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vinner_product_4x16___024root___dump_triggers__act(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vinner_product_4x16___024root___dump_triggers__nba(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vinner_product_4x16___024root___ctor_var_reset(Vinner_product_4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->reset = VL_RAND_RESET_I(1);
    vlSelf->enable = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->activations[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->inner_product_4x16__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->inner_product_4x16__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->inner_product_4x16__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
