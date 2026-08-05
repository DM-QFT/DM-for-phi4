#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*SetDirectory[NotebookDirectory[]];*)
(*Get["academic.m"];*)
(*Needs["MaTeX`"];*)


(* ::Section:: *)
(*Propagator*)


(* ::Input:: *)
(*data=Import["L64_crossL_3d_k0.1923_propagator.csv","Dataset","HeaderLines"->1];*)
(**)
(*curves=GroupBy[Normal[data],#["method"]&,Map[{#["k_lat"],Around[#["G"],#["dG"]]}&]];*)
(**)


(* ::Input:: *)
(*ListLinePlot[{curves["multiL_4-8-16-32"],curves["multiL_4-8-16"],curves["multiL_4-8"],curves["HMC_L64"]},IntervalMarkers->"Fences"(*,IntervalMarkersStyle-><|"FenceWidth"->0.00035|>*)*)
(*,ScalingFunctions->{"Log","Log"},PlotStyle->{Automatic,Automatic,Automatic,Directive[Black,Dashed]},PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"\\mathrm{DM,\\;L}\\in [4,8,16,32]","\\mathrm{DM,\\;L}\\in [4,8,16]","\\mathrm{DM,\\;L}\\in [4,8]","\\mathrm{HMC,\\;L} 64"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],Scaled[{0.3,0.25}]],FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"})]*)


(* ::Input:: *)
(*ListLinePlot[{curves["multiL_4-8-16-32"],curves["multiL_4-8-16"],curves["multiL_4-8"],curves["HMC_L64"]},IntervalMarkers->"Fences"(*,IntervalMarkersStyle-><|"FenceWidth"->0.00035|>*)*)
(*,ScalingFunctions->{"Log",Automatic},PlotStyle->{Automatic,Automatic,Automatic,Directive[Black,Dashed]},PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"\\mathrm{DM,\\;L}\\in [4,8,16,32]","\\mathrm{DM,\\;L}\\in [4,8,16]","\\mathrm{DM,\\;L}\\in [4,8]","\\mathrm{HMC,\\;L} 64"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],Scaled[{0.7,0.55}]],FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"}),PlotRange->All]*)


(* ::Input:: *)
(*ListLinePlot[{curves["L32_only"],curves["L16_only"],curves["L8_only"],curves["L4_only"],curves["HMC_L64"]},IntervalMarkers->"Fences"(*,IntervalMarkersStyle-><|"FenceWidth"->0.00035|>*)*)
(*,ScalingFunctions->{"Log","Log"},PlotStyle->{Automatic,Automatic,Automatic,Automatic,Directive[Black,Dashed]},PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"\\mathrm{DM,\\;L}\\in [32]","\\mathrm{DM,\\;L}\\in [16]","\\mathrm{DM,\\;L}\\in [8]","\\mathrm{DM,\\;L}\\in [4]","\\mathrm{HMC,\\;L} 64"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],Scaled[{0.3,0.25}]],FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"})]*)
