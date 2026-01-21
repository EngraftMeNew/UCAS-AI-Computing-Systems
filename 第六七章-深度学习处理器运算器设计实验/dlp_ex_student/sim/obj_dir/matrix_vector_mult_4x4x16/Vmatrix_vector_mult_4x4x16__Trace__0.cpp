// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vmatrix_vector_mult_4x4x16__Syms.h"


void Vmatrix_vector_mult_4x4x16___024root__trace_chg_0_sub_0(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vmatrix_vector_mult_4x4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_chg_0\n"); );
    // Init
    Vmatrix_vector_mult_4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_vector_mult_4x4x16___024root*>(voidSelf);
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vmatrix_vector_mult_4x4x16___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vmatrix_vector_mult_4x4x16___024root__trace_chg_0_sub_0(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[1U])) {
        bufp->chgSData(oldp+0,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[0]),16);
        bufp->chgSData(oldp+1,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[1]),16);
        bufp->chgSData(oldp+2,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[2]),16);
        bufp->chgSData(oldp+3,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[3]),16);
        bufp->chgSData(oldp+4,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[0]),16);
        bufp->chgSData(oldp+5,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[1]),16);
        bufp->chgSData(oldp+6,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[2]),16);
        bufp->chgSData(oldp+7,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[3]),16);
        bufp->chgSData(oldp+8,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[0]),16);
        bufp->chgSData(oldp+9,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[1]),16);
        bufp->chgSData(oldp+10,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[2]),16);
        bufp->chgSData(oldp+11,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[3]),16);
        bufp->chgSData(oldp+12,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[0]),16);
        bufp->chgSData(oldp+13,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[1]),16);
        bufp->chgSData(oldp+14,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[2]),16);
        bufp->chgSData(oldp+15,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[3]),16);
    }
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[2U])) {
        bufp->chgIData(oldp+16,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result),32);
        bufp->chgSData(oldp+17,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+18,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+19,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+20,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+21,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+22,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+23,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+24,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+25,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                               [0U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                           [0U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                              [1U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                              [1U])) 
                                    + (VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                 [2U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                 [2U])) 
                                       + VL_MULS_III(32, 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                   [3U]), 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                   [3U])))))),32);
        bufp->chgIData(oldp+26,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__i),32);
        bufp->chgIData(oldp+27,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result),32);
        bufp->chgSData(oldp+28,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+29,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+30,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+31,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+32,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+33,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+34,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+35,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+36,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                               [0U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                           [0U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                              [1U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                              [1U])) 
                                    + (VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                 [2U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                 [2U])) 
                                       + VL_MULS_III(32, 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                   [3U]), 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                   [3U])))))),32);
        bufp->chgIData(oldp+37,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__i),32);
        bufp->chgIData(oldp+38,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result),32);
        bufp->chgSData(oldp+39,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+40,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+41,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+42,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+43,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+44,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+45,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+46,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+47,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                               [0U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                           [0U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                              [1U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                              [1U])) 
                                    + (VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                 [2U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                 [2U])) 
                                       + VL_MULS_III(32, 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                   [3U]), 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                   [3U])))))),32);
        bufp->chgIData(oldp+48,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__i),32);
        bufp->chgIData(oldp+49,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result),32);
        bufp->chgSData(oldp+50,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+51,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+52,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+53,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+54,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+55,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+56,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+57,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+58,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                               [0U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                           [0U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                              [1U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                              [1U])) 
                                    + (VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                 [2U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                 [2U])) 
                                       + VL_MULS_III(32, 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                   [3U]), 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                   [3U])))))),32);
        bufp->chgIData(oldp+59,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__i),32);
    }
    bufp->chgBit(oldp+60,(vlSelf->clk));
    bufp->chgBit(oldp+61,(vlSelf->reset));
    bufp->chgBit(oldp+62,(vlSelf->enable));
    bufp->chgSData(oldp+63,(vlSelf->activations[0]),16);
    bufp->chgSData(oldp+64,(vlSelf->activations[1]),16);
    bufp->chgSData(oldp+65,(vlSelf->activations[2]),16);
    bufp->chgSData(oldp+66,(vlSelf->activations[3]),16);
    bufp->chgSData(oldp+67,(vlSelf->weights[0U][0U]),16);
    bufp->chgSData(oldp+68,(vlSelf->weights[0U][1U]),16);
    bufp->chgSData(oldp+69,(vlSelf->weights[0U][2U]),16);
    bufp->chgSData(oldp+70,(vlSelf->weights[0U][3U]),16);
    bufp->chgSData(oldp+71,(vlSelf->weights[1U][0U]),16);
    bufp->chgSData(oldp+72,(vlSelf->weights[1U][1U]),16);
    bufp->chgSData(oldp+73,(vlSelf->weights[1U][2U]),16);
    bufp->chgSData(oldp+74,(vlSelf->weights[1U][3U]),16);
    bufp->chgSData(oldp+75,(vlSelf->weights[2U][0U]),16);
    bufp->chgSData(oldp+76,(vlSelf->weights[2U][1U]),16);
    bufp->chgSData(oldp+77,(vlSelf->weights[2U][2U]),16);
    bufp->chgSData(oldp+78,(vlSelf->weights[2U][3U]),16);
    bufp->chgSData(oldp+79,(vlSelf->weights[3U][0U]),16);
    bufp->chgSData(oldp+80,(vlSelf->weights[3U][1U]),16);
    bufp->chgSData(oldp+81,(vlSelf->weights[3U][2U]),16);
    bufp->chgSData(oldp+82,(vlSelf->weights[3U][3U]),16);
    bufp->chgIData(oldp+83,(vlSelf->results[0]),32);
    bufp->chgIData(oldp+84,(vlSelf->results[1]),32);
    bufp->chgIData(oldp+85,(vlSelf->results[2]),32);
    bufp->chgIData(oldp+86,(vlSelf->results[3]),32);
}

void Vmatrix_vector_mult_4x4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_cleanup\n"); );
    // Init
    Vmatrix_vector_mult_4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_vector_mult_4x4x16___024root*>(voidSelf);
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
}
