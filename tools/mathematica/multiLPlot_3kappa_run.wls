#!/usr/bin/env wolframscript
(* Executable: read 3 propagator CSVs, export 7 comparison plots (PDF + PNG). *)

Needs["MaTeX`"];
Get["/home/tyywork/DM/mathematica/academic.m"];

csvDir = "/data/tyywork/DM/3Dphi4/results/";
outDir = "/data/tyywork/DM/3Dphi4/results/";

loadCurves[kStr_] := Module[{data},
  data = Import[csvDir <> "L64_crossL_3d_k" <> kStr <> "_propagator.csv",
                "Dataset", "HeaderLines" -> 1];
  GroupBy[Normal[data], #["method"] &,
          Map[{#["k_lat"], Around[#["G"], #["dG"]]} &]]];

curves18   = loadCurves["0.18"];
curves1923 = loadCurves["0.1923"];
curves20   = loadCurves["0.2"];

(* posSpec = {Scaled position, anchor inside the inset}.
   anchor {1,1} = upper-right corner of label aligns to that Scaled point;
   anchor {-1,1} = upper-left corner aligns. Using corner anchors prevents
   the text from overlapping the frame regardless of its width. *)
multiLPlot[curves_, scaling_, legendPos_, label_, posSpec_,
           opts:OptionsPattern[ListLinePlot]] :=
  ListLinePlot[
    {curves["multiL_4-8-16-32"], curves["multiL_4-8-16"],
     curves["multiL_4-8"], curves["HMC_L64"]},
    IntervalMarkers -> "Fences",
    ScalingFunctions -> scaling,
    PlotStyle -> {Automatic, Automatic, Automatic, Directive[Black, Dashed]},
    PlotLegends -> Placed[
      LineLegend[
        MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
          {"\\mathrm{DM,\\;L}\\in [4,8,16,32]",
           "\\mathrm{DM,\\;L}\\in [4,8,16]",
           "\\mathrm{DM,\\;L}\\in [4,8]",
           "\\mathrm{HMC,\\;L} 64"},
        LegendMarkerSize -> {40, 12}, LegendLayout -> {"Column", 1}],
      legendPos],
    FrameLabel -> (MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
                   {"|k|", "G(|k|)"}),
    Epilog -> {Text[
      MaTeX[label, ContentPadding -> False, FontSize -> 18],
      Scaled[posSpec[[1]]], posSpec[[2]]]},
    opts];

singleLPlot[curves_, legendPos_, label_, posSpec_] :=
  ListLinePlot[
    {curves["L32_only"], curves["L16_only"], curves["L8_only"],
     curves["L4_only"], curves["HMC_L64"]},
    IntervalMarkers -> "Fences",
    ScalingFunctions -> {"Log", "Log"},
    PlotStyle -> {Automatic, Automatic, Automatic, Automatic,
                  Directive[Black, Dashed]},
    PlotLegends -> Placed[
      LineLegend[
        MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
          {"\\mathrm{DM,\\;L}\\in [32]",
           "\\mathrm{DM,\\;L}\\in [16]",
           "\\mathrm{DM,\\;L}\\in [8]",
           "\\mathrm{DM,\\;L}\\in [4]",
           "\\mathrm{HMC,\\;L} 64"},
        LegendMarkerSize -> {40, 12}, LegendLayout -> {"Column", 1}],
      legendPos],
    FrameLabel -> (MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
                   {"|k|", "G(|k|)"}),
    Epilog -> {Inset[
      MaTeX[label, ContentPadding -> False, FontSize -> 18],
      Scaled[posSpec[[1]]], posSpec[[2]]]}];

label18   = "\\kappa = 0.18,\\;\\mathrm{symmetric}";
label1923 = "\\kappa = 0.1923,\\;\\mathrm{near\\;critical}";
label20   = "\\kappa = 0.2,\\;\\mathrm{broken}";

exportBoth[name_, plot_] := (
  Export[outDir <> name <> ".pdf", plot];
  Export[outDir <> name <> ".png", plot, ImageResolution -> 200];
  Print["saved: ", name]);

(* posSpec = {Scaled-frame point, inset corner anchor}.
   posUR: align label's upper-right corner to (0.97, 0.95) inside the frame.
   posUL: align label's upper-left  corner to (0.03, 0.95). *)
posUR = {{0.97, 0.92}, {1, 1}};
posUL = {{0.03, 0.92}, {-1, 1}};

exportBoth["L64_crossL_3d_k0.18_multiL_loglog",
           multiLPlot[curves18, {"Log", "Log"}, Scaled[{0.3, 0.25}], label18, posUR]];
exportBoth["L64_crossL_3d_k0.18_multiL_loglinear",
           multiLPlot[curves18, {"Log", Automatic}, Scaled[{0.7, 0.55}], label18, posUR,
                      PlotRange -> All]];

exportBoth["L64_crossL_3d_k0.1923_multiL_loglog",
           multiLPlot[curves1923, {"Log", "Log"}, Scaled[{0.3, 0.25}], label1923, posUR]];
exportBoth["L64_crossL_3d_k0.1923_multiL_loglinear",
           multiLPlot[curves1923, {"Log", Automatic}, Scaled[{0.7, 0.55}], label1923, posUR,
                      PlotRange -> All]];
exportBoth["L64_crossL_3d_k0.1923_singleL_loglog",
           singleLPlot[curves1923, Scaled[{0.3, 0.25}], label1923, posUL]];

exportBoth["L64_crossL_3d_k0.2_multiL_loglog",
           multiLPlot[curves20, {"Log", "Log"}, Scaled[{0.3, 0.25}], label20, posUR]];
exportBoth["L64_crossL_3d_k0.2_multiL_loglinear",
           multiLPlot[curves20, {"Log", Automatic}, Scaled[{0.7, 0.55}], label20, posUR,
                      PlotRange -> All]];
