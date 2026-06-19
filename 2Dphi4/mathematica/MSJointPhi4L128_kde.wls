#!/usr/bin/env wolframscript
(* ::Package:: *)

(* DM vs HMC joint (m, S/V) density at L=128, three kappa.
   Each panel:
     - HMC: 4 mass-enclosed iso-bands filled with ThermometerColors
     - DM:  same mass-enclosed contours as dashed black lines
     - kappa label inset at top-center
     - third panel: in-plot legend at center (HMC swatch + DM dashed)
   Output: figures/mS_joint_3panels.pdf  (1x3 composite).

   Input data: results/dm_vs_hmc_L128/dat_cache/mS_pts_k{kappa}_{HMC,DM}.dat
   (raw point pairs dumped by analysis/dump_mS_joint_dat.py).

   Run from anywhere:
     wolframscript -file mathematica/MSJointPhi4L128_kde.wls *)

(* ---- Init: paths, theme, shared helpers ---- *)
SetDirectory[DirectoryName[
   If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "..", "mathematica", "academic.m"}]];
Needs["MaTeX`"];

(* MaTeX automatically caches rendered LaTeX images to
   $UserBaseDirectory/ApplicationData/MaTeX/cache — persistent across
   sessions, no extra setup needed.  First run pays the LaTeX compile
   cost; subsequent runs hit the cache and return instantly. *)

datDir = FileNameJoin[{"..", "results", "dm_vs_hmc_L128", "dat_cache"}];
figDir = FileNameJoin[{"..", "results", "dm_vs_hmc_L128", "figures"}];
If[!DirectoryQ[figDir], CreateDirectory[figDir]];

kappas    = {"0.26", "0.2705", "0.28"};
phaseStr  = <|"0.26" -> "sym", "0.2705" -> "crit", "0.28" -> "broken"|>;
massFracs = {0.2, 0.5, 0.9, 0.98};
nG        = 100;

densCF     = ColorData["ThermometerColors"];
bandColors = Prepend[
  Directive[densCF[#], Opacity[0.88]] & /@ {0.30, 0.50, 0.72, 0.92},
  None];

(* Mass-enclosed iso-density levels: sort descending, cumulative-sum the
   PDF values, pick the threshold whose region integrates to fraction f
   of total mass. Returns levels in ascending order (outermost first). *)
massLevels[Z_, fracs_] := With[{f = Reverse@Sort@Flatten[Z]},
  With[{c = Accumulate[f]/Total[f]},
    Sort[f[[Min[Count[c, x_ /; x < #] + 1, Length[f]]]] & /@ fracs]]];


(* ---- makePanel: load -> KDE grid -> plot, all in one ---- *)
Options[makePanel] = {"FontSize" -> 18, "ImageSize" -> 420,
                      "ShowLegend" -> False};
makePanel[kStr_String, OptionsPattern[]] := Module[
  {hmcPts, dmPts, skdH, skdD, mMax, svLo, svHi, xR, yR, gridZ,
   Zh, Zd, Fh, Fd, hmcLvls, dmLvls,
   fs, fracStr, annot, legendRow, legend, epi},
  fs = OptionValue["FontSize"];

  hmcPts = Import[FileNameJoin[{datDir, "mS_pts_k" <> kStr <> "_HMC.dat"}],
                  "Table", "HeaderLines" -> 1];
  dmPts  = Import[FileNameJoin[{datDir, "mS_pts_k" <> kStr <> "_DM.dat"}],
                  "Table", "HeaderLines" -> 1];

  {svLo, svHi} = Quantile[Join[hmcPts, dmPts][[All, 2]], {0.005, 0.995}];
  yR = {svLo, svHi} + 10 (svHi - svLo) {-1, 1};
  (* mMax is the larger of (a) 1.2x the |m| 99.5% quantile and (b) half of
     the total y-span — so a tight-peak panel like κ=0.26 (sym) automatically
     widens horizontally to look roughly circular under AspectRatio -> 1. *)
  mMax = Max[
    1.2 Quantile[Abs[Join[hmcPts, dmPts][[All, 1]]], 0.995],
    (yR[[2]] - yR[[1]]) / 2];
  xR = {-mMax, mMax};

  skdH = SmoothKernelDistribution[hmcPts, 0.1];
  skdD = SmoothKernelDistribution[dmPts,  0.1];

  (* Sample KDE on a coarse grid once.  Two uses:
       1. Feed massLevels to get mass-enclosed iso thresholds
       2. Wrap as a fast lookup (ListInterpolation) so ContourPlot's
          adaptive sampling does O(1) lookups instead of O(N) KDE evals. *)
  gridZ[skd_] := Table[PDF[skd, {x, y}],
    {x, Subdivide[xR[[1]], xR[[2]], nG - 1]},
    {y, Subdivide[yR[[1]], yR[[2]], nG - 1]}];
  Zh = gridZ[skdH]; Zd = gridZ[skdD];
  hmcLvls = massLevels[Zh, massFracs];
  dmLvls  = massLevels[Zd, massFracs];

  (* ListInterpolation expects mat[[i, j]] = f(xRange[i], yRange[j]) — that
     is exactly Zh's layout (Table outer iterator is x), so no Transpose. *)
  Fh = ListInterpolation[Zh, {xR, yR}, InterpolationOrder -> 1];
  Fd = ListInterpolation[Zd, {xR, yR}, InterpolationOrder -> 1];

  fracStr = "(" <> StringRiffle[ToString /@ Round[100 massFracs], "/"] <> "\\%)";

  annot = Inset[
    MaTeX["\\kappa=" <> kStr <> "\\;(\\text{" <> phaseStr[kStr] <> "})",
          FontSize -> fs + 2],
    Scaled[{0.5, 0.95}], {Center, Top},
    Background -> Directive[White, Opacity[0.85]]];

  legendRow[swatch_, label_, imgW_:32] := Row[{
    Graphics[swatch, ImageSize -> {imgW, 14}],
    MaTeX["\\text{" <> label <> "}\\;" <> fracStr, FontSize -> fs]}];

  legend = Inset[
    Framed[
      Column[{
        legendRow[{densCF[0.72], Opacity[0.88], Rectangle[{0, 0}, {1.4, 0.5}]},
                  "HMC", 32],
        legendRow[{Black,AbsoluteThickness[1.5],
                   AbsoluteDashing[{8, 4}],
                   Line[{{0, 0.5}, {2.0, 0.5}}]},
                  "DM", 32]},
        Spacings -> 0, Alignment -> Left],
      Background -> Directive[White, Opacity[0.9]],
      FrameStyle -> Directive[Gray, AbsoluteThickness[0.6]],
      RoundingRadius -> 4, FrameMargins -> 6],
    Scaled[{0.5, 0.5}]];

  epi = If[OptionValue["ShowLegend"], {annot, legend}, {annot}];

  Show[
    ContourPlot[Fh[x, y],
      {x, xR[[1]], xR[[2]]}, {y, yR[[1]], yR[[2]]},
      Contours -> hmcLvls, ContourShading -> bandColors,
      ContourLines -> False,
      PlotPoints -> 40, MaxRecursion -> 2,
      PlotRange -> All, PlotRangePadding -> None],
    ContourPlot[Fd[x, y],
      {x, xR[[1]], xR[[2]]}, {y, yR[[1]], yR[[2]]},
      Contours -> dmLvls, ContourShading -> None,
      ContourStyle -> Directive[Black, AbsoluteDashing[{8, 4}],
                                AbsoluteThickness[1.5]],
      PlotPoints -> 40, MaxRecursion -> 2,
      PlotRange -> All, PlotRangePadding -> None],
    PlotRange -> All, PlotRangePadding -> None,
    AspectRatio -> 1, Frame -> True,
    FrameLabel -> {
      MaTeX["\\bar{\\phi}=\\tfrac{1}{V}\\sum_{x}\\phi_{x}", FontSize -> fs + 4],
      MaTeX["s=S[\\phi]/V", FontSize -> fs + 4]},
    FrameTicksStyle -> Directive[FontSize -> fs],
    Epilog -> epi,
    ImageSize -> OptionValue["ImageSize"]]
];


(* ---- 1x3 composite (legend only on the last/largest-kappa panel) ---- *)
panels = MapIndexed[
  makePanel[#1, "FontSize" -> 18, "ImageSize" -> 380,
                "ShowLegend" -> (First[#2] == Length[kappas])] &,
  kappas];

composite = Grid[{panels},
  Spacings -> {1.5, 0}, Frame -> None, Alignment -> {Center, Center}];

fnAll = FileNameJoin[{figDir, "mS_joint_3panels.pdf"}];
Export[fnAll, composite];
Print["saved: ", fnAll, "  (", N@FileByteCount[fnAll]/1024., " KB)"];
