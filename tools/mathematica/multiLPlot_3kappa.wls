#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*SetDirectory[NotebookDirectory[]];*)
(*Get["academic.m"];*)
(*Needs["MaTeX`"];*)


(* ::Section:: *)
(*3D \[Phi]^4 cross-L propagator: \[Kappa] \[Element] {0.18, 0.1923, 0.2}*)


(* ::Input:: *)
(*csvDir="/data/tyywork/DM/3Dphi4/results/";*)
(*loadCurves[kStr_]:=Module[{data},*)
(*data=Import[csvDir<>"L64_crossL_3d_k"<>kStr<>"_propagator.csv","Dataset","HeaderLines"->1];*)
(*GroupBy[Normal[data],#["method"]&,Map[{#["k_lat"],Around[#["G"],#["dG"]]}&]]];*)
(*curves18=loadCurves["0.18"];*)
(*curves1923=loadCurves["0.1923"];*)
(*curves20=loadCurves["0.2"];*)


(* ::Input:: *)
(*multiLPlot[curves_,scaling_,legendPos_,label_,posSpec_,opts:OptionsPattern[ListLinePlot]]:=ListLinePlot[*)
(*{curves["multiL_4-8-16-32"],curves["multiL_4-8-16"],curves["multiL_4-8"],curves["HMC_L64"]},*)
(*IntervalMarkers->"Fences",*)
(*ScalingFunctions->scaling,*)
(*PlotStyle->{Automatic,Automatic,Automatic,Directive[Black,Dashed]},*)
(*PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"\\mathrm{DM,\\;L}\\in [4,8,16,32]","\\mathrm{DM,\\;L}\\in [4,8,16]","\\mathrm{DM,\\;L}\\in [4,8]","\\mathrm{HMC,\\;L} 64"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],legendPos],*)
(*FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"}),*)
(*Epilog->{Text[MaTeX[label,ContentPadding->False,FontSize->18],Scaled[posSpec[[1]]],posSpec[[2]]]},*)
(*opts];*)


(* ::Input:: *)
(*singleLPlot[curves_,legendPos_,label_,posSpec_]:=ListLinePlot[*)
(*{curves["L32_only"],curves["L16_only"],curves["L8_only"],curves["L4_only"],curves["HMC_L64"]},*)
(*IntervalMarkers->"Fences",*)
(*ScalingFunctions->{"Log","Log"},*)
(*PlotStyle->{Automatic,Automatic,Automatic,Automatic,Directive[Black,Dashed]},*)
(*PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"\\mathrm{DM,\\;L}\\in [32]","\\mathrm{DM,\\;L}\\in [16]","\\mathrm{DM,\\;L}\\in [8]","\\mathrm{DM,\\;L}\\in [4]","\\mathrm{HMC,\\;L} 64"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],legendPos],*)
(*FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"}),*)
(*Epilog->{Inset[MaTeX[label,ContentPadding->False,FontSize->15],Scaled[labelPos]]}];*)


(* ::Input:: *)
(*label18="\\kappa = 0.18,\\;\\mathrm{symmetric}";*)
(*label1923="\\kappa = 0.1923,\\;\\mathrm{near\\;critical}";*)
(*label20="\\kappa = 0.2,\\;\\mathrm{broken}";*)
(*posUR={{0.97,0.92},{1,1}};*)
(*posUL={{0.03,0.92},{-1,1}};*)


(* ::Section:: *)
(*\[Kappa] = 0.18*)


(* ::Input:: *)
(*multiLPlot[curves18,{"Log","Log"},Scaled[{0.3,0.25}],label18,posUR]*)


(* ::Input:: *)
(*multiLPlot[curves18,{"Log",Automatic},Scaled[{0.7,0.55}],label18,posUR,PlotRange->All]*)


(* ::Section:: *)
(*\[Kappa] = 0.1923*)


(* ::Input:: *)
(*multiLPlot[curves1923,{"Log","Log"},Scaled[{0.3,0.25}],label1923,posUR]*)


(* ::Input:: *)
(*multiLPlot[curves1923,{"Log",Automatic},Scaled[{0.7,0.55}],label1923,posUR,PlotRange->All]*)


(* ::Input:: *)
(*singleLPlot[curves1923,Scaled[{0.3,0.25}],label1923,posUL]*)


(* ::Section:: *)
(*\[Kappa] = 0.2*)


(* ::Input:: *)
(*multiLPlot[curves20,{"Log","Log"},Scaled[{0.3,0.25}],label20,posUR]*)


(* ::Input:: *)
(*multiLPlot[curves20,{"Log",Automatic},Scaled[{0.7,0.55}],label20,posUR,PlotRange->All]*)
