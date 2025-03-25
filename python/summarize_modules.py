#!/usr/bin/env python3

import argparse
import dataclasses
import glob
import importlib
import itertools
import numpy
import os
import random
import re
import ROOT
import sortedcontainers
import sys
import tqdm

from ruamel.yaml.scalarstring import DoubleQuotedScalarString

import constants
import utils
from utils import logging
from utils import yaml
from utils import get_grx
from utils import get_gry


@dataclasses.dataclass(init = True)
class SensorModule(utils.SensorModule) :
    
    run: int = None
    fname: str = None
    category: str = None

@dataclasses.dataclass(init = True)
class DetectorModule(utils.DetectorModule) :
    
    run: int = None
    fname: str = None
    category: str = None


def do_sm_pairing(l_sms, cat, outdir) :
    
    n_sms = len(l_sms)
    l_sms_sorted = sorted(l_sms, key = lambda _sm: _sm["pairing"])
    
    logging.info(f"Finding pairs in {n_sms} category {cat} SMs ...")
    
    l_sm_groups = [l_sms_sorted[_i: _i+2] if (_i < n_sms-1) else l_sms_sorted[_i: _i+1] for _i in range(0, n_sms, 2)]
    l_sm_pairs = [sorted(_pair, key = lambda _x: int(_x["barcode"])) for _pair in l_sm_groups if len(_pair) == 2]
    
    outfname = f"{outdir}/sm-pairs_cat-{cat}.csv"
    logging.info(f"Writing pairing results to: {outfname} ...")
    with open(outfname, "w") as fopen :
        
        print("# pair number, sm1 barcode , sm2 barcode , sm1 metric , sm2 metric, sm1 cat, sm2 cat", file = fopen)
        for ipair, pair in enumerate(l_sm_pairs) :

            print(" , ".join([
                str(ipair+1),
                pair[0]['barcode'],
                pair[1]['barcode'],
                str(pair[0]['pairing']),
                str(pair[1]['pairing']),
                pair[0]['category'],
                pair[1]['category'],
            ]), file = fopen)



def main() :
    
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class = utils.Formatter,
        description = "Plots module summary",
    )
    
    parser.add_argument(
        "--srcs",
        help = (
            "Source directories and regular expressions for each source: dir1:regexp1 dir1:regexp2 ...\n"
            "Only files that match the regular expression will be processed.\n"
            "regexp is a keyed regular expression, used to extract run and barcode from the file name.\n"
            "SM example (for cases like \"runXXXX/module_YYYY_analysis.root\"): \"run(?P<run>\\d+)/module_(?P<barcode>\\d+)_analysis.root\"\n"
            "DM example (for cases like \"runXXXX_DM-YYYY.root\"): \"run-(?P<run>\\d+)_DM-(?P<barcode>\\d+).root\""
            "\n"
        ),
        type = str,
        nargs = "+",
        required = True,
    )
    
    parser.add_argument(
        "--plotcfg",
        help = "YAML file with plot configurations.\n",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--catcfg",
        help = "YAML file with module categorization configuration.\n",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--defcfg",
        help = "YAML file with definitions of variables to be used in plots.\n",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--moduletype",
        help = "Module type.\n",
        type = str,
        required = True,
        choices = [constants.SM.KIND_OF_PART, constants.DM.KIND_OF_PART]
    )
    
    parser.add_argument(
        "--modules",
        help = "Only process this space delimited list of modules (or files with a barcode per line), unless it is in the skipmodules list.\n",
        type = str,
        nargs = "+",
        required = False,
        default = [],
    )
    
    parser.add_argument(
        "--runs",
        help = "Only process this space delimited list of runs (or files with a barcode per line), unless it is in the skipmodules list.\n",
        type = str,
        nargs = "+",
        required = False,
        default = [],
    )
    
    parser.add_argument(
        "--skipmodules",
        help = "Space delimited list of modules (or files with a barcode per line) to skip.\n",
        type = str,
        nargs = "+",
        required = False,
        default = [],
    )
    
    parser.add_argument(
        "--skipruns",
        help = "Space delimited list of runs (or files with a run per line) to skip.\n",
        type = str,
        nargs = "+",
        required = False,
        default = [],
    )
    
    parser.add_argument(
        "--runcond",
        help = "Valid python expression to filter runs. For example: \"{run}>=X and {run}<Y\".\n",
        type = str,
        required = False,
        default = None,
    )
    
    parser.add_argument(
        "--selectexpr",
        help = (
            "Module selection expression string (must be a valid python expression).\n"
            "For e.g.: \"{key1}>=X and {key2}<Y\"\n"
            "Allowed keys are: barcode (str), run (int)"
        ),
        type = str,
        required = False,
        default = "True",
    )
    
    parser.add_argument(
        "--sipminfo",
        help = "YAML file with SiPM information.\n",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--sminfo",
        help = "YAML file with module information.\n",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--dminfo",
        help = (
            "YAML file with DM information.\n"
            "Will update the file with additional DMs on the database, if --pairsms is passed.\n"
            "This is needed if one wants to omit the used (in DMs) SMs from the pairing.\n"
        ),
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--dminfoextra",
        help = (
            "Extra DM informatation as csv file.\n"
            "Syntax: <filename> <name>:<column index>:<dtype> <name>:<column index>:<dtype> ...\n"
            "N.B.\n"
            "  The first line will be treated as header and will be skipped\n"
            "  Barcode must be the first column\n"
            "  Column index starts from 0\n"
            "  \"name\" must be a valid python attribute name\n"
            "Example: path/to/AssemblesDMs.csv tec_sum_bac:76float\n"
            "Then one can access DM.extra['tec_sum_bac'].\n"
        ),
        type = str,
        nargs = "+",
        required = False,
        default = None,
    )
    
    parser.add_argument(
        "--ruinfo",
        help = (
            "YAML file with RU information.\n"
            "Will update the file with additional RUs on the database, if --groupdms is passed.\n"
            "This is needed if one wants to omit the already used (in RUs) DMs from the grouping.\n"
        ),
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--smresults",
        help = (
            "YAML file with the SM results, for example the SM categorization output file.\n"
            "This is used for reading the results (like category, light output, etc.) for SMs in a DM, when grouping DMs for a tray.\n"
        ),
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--pairsms",
        help = "Will pair SMs using the \"pairing\" metric in the categorization configuration \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--mixsmcats",
        help = "Will mix SM categories when pairing them \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--groupdms",
        help = "Will group DMs using the \"grouping\" metric in the categorization configuration \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--flipru",
        help = "Will print RU DMs [0, 1, 2, 3] row at the top (default is at the bottom) \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--seed",
        help = "Will use this to seed the DM shuffling when grouping them \n",
        type = int,
        required = False,
        default = 0
    )
    
    parser.add_argument(
        "--listmissing",
        help = "List modules that are in the info yaml but not in the src directories \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--location",
        help = "List of locations \n",
        type = str,
        nargs = "+",
        required = True,
        choices = [_loc for _loc in dir(constants.LOCATION) if not _loc.startswith("__")],
    )
    
    parser.add_argument(
        "--nodb",
        help = "Will not fetch information from the database \n",
        action = "store_true",
        default = False
    )
    
    parser.add_argument(
        "--outdir",
        help = "Output directory.\n",
        type = str,
        required = True,
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    rnd = random.Random(args.seed)
    
    l_location_ids = [vars(constants.LOCATION)[_loc] for _loc in args.location]
    
    # Create output directory
    os.system(f"mkdir -p {args.outdir}")
    
    # Get the list of files with specified pattern
    logging.info(f"Getting list of files from {len(args.srcs)} source(s) ...")
    l_srcs = [_src.split(":")[0] for _src in args.srcs]
    l_regexps = [_src.split(":")[1] for _src in args.srcs]
    l_fnames, l_regexps = utils.get_file_list(l_srcs = l_srcs, l_regexps = l_regexps)
    
    d_loaded_part_info = {
        constants.SIPM.KIND_OF_PART: {},
        constants.SM.KIND_OF_PART: {},
        constants.DM.KIND_OF_PART: {},
    }
    
    if (args.sipminfo) :
        
        d_loaded_part_info[constants.SIPM.KIND_OF_PART] = utils.load_part_info(parttype = constants.SIPM.KIND_OF_PART, yamlfile = args.sipminfo)
    
    if (args.sminfo) :
        
        d_loaded_part_info[constants.SM.KIND_OF_PART] = utils.load_part_info(parttype = constants.SM.KIND_OF_PART, yamlfile = args.sminfo, resultsyaml = args.smresults)
        
    if (args.dminfo) :
        
        assert (args.dminfoextra is None or len(args.dminfoextra) >= 2), "Invalid --dminfoextra argument."
        
        d_loaded_part_info[constants.DM.KIND_OF_PART] = utils.load_part_info(parttype = constants.DM.KIND_OF_PART, yamlfile = args.dminfo, extrainfo = args.dminfoextra)
    
    logging.info("Combining parts ...")
    utils.combine_parts(
        d_sipms = d_loaded_part_info[constants.SIPM.KIND_OF_PART],
        d_sms = d_loaded_part_info[constants.SM.KIND_OF_PART],
        d_dms = d_loaded_part_info[constants.DM.KIND_OF_PART],
    )
    logging.info("Combined parts.")
    
    ## Load SM results if provided
    #d_loaded_sm_results = {}
    #
    #if (args.smresults) :
    #    
    #    with open(args.smresults, "r") as fopen :
    #        
    #        d_loaded_sm_results = yaml.load(fopen.read())["results"]
    #    
    #    #for barcode, results in d_loaded_part_info[constants.SM.KIND_OF_PART].items() :
            
    
    # Get list of modules
    logging.info(f"Parsing {len(l_fnames)} files to get modules to process ...")
    
    l_toproc_runs = []
    l_toskip_runs = []
    
    l_toproc_modules = []
    l_toskip_modules = []
    l_skipped_modules = []
    l_duplicate_modules = []
    l_found_modules = []
    d_modules = sortedcontainers.SortedDict()
    
    for toproc in args.runs :
        
        if (os.path.isfile(toproc)) :
            
            l_tmp = numpy.loadtxt(toproc, dtype = int).flatten()
            l_toproc_runs.extend(l_tmp)
        
        else :
            
            l_toproc_runs.append(int(toproc))
    
    for toskip in args.skipruns :
        
        if (os.path.isfile(toskip)) :
            
            l_tmp = numpy.loadtxt(toskip, dtype = int).flatten()
            l_toskip_runs.extend(l_tmp)
        
        else :
            
            l_toskip_runs.append(int(toskip))
    
    for toproc in args.modules :
        
        if (os.path.isfile(toproc)) :
            
            l_tmp = numpy.loadtxt(toproc, dtype = str).flatten()
            l_toproc_modules.extend(l_tmp)
        
        else :
            
            l_toproc_modules.append(toproc)
    
    for toskip in args.skipmodules :
        
        if (os.path.isfile(toskip)) :
            
            l_tmp = numpy.loadtxt(toskip, dtype = str).flatten()
            l_toskip_modules.extend(l_tmp)
        
        else :
            
            l_toskip_modules.append(toskip)
    
    for fname, regexp in tqdm.tqdm(zip(l_fnames, l_regexps)) :
        
        parsed_result = utils.parse_string_regex(
            s = fname,
            regexp = regexp,
        )
        
        run = int(parsed_result["run"]) if ("run" in parsed_result) else -1
        barcode = parsed_result["barcode"].strip()
        
        selectexpr_tmp = args.selectexpr.format(
            run = run,
            barcode = barcode,
        )
        
        selectexpr_eval = eval(selectexpr_tmp)
        
        if barcode not in l_found_modules :
            
            l_found_modules.append(barcode)
        
        if (run in l_toskip_runs or (l_toproc_runs and run not in l_toproc_runs)) :
            
            continue
        
        if args.runcond :
            
            runcond_expr = args.runcond.format(run = run)
            
            if not eval(runcond_expr) :
                
                continue
        
        if (barcode in l_toskip_modules or not selectexpr_eval) :
            
            #print(f"Skipping module {barcode}")
            l_skipped_modules.append(barcode)
            continue
        
        if (l_toproc_modules and barcode not in l_toproc_modules) :
            
            continue
        
        # If the module is repeated, only use the latest run
        # Update: use the file with the latest timestamp
        if (barcode in d_modules) :
            
            if (run < d_modules[barcode].run) :
            #if (os.path.getmtime(fname) < os.path.getmtime(d_modules[barcode].fname)) :
                
                l_duplicate_modules.append({"run": run, "barcode": barcode, "fname": fname})
                continue
            
            else :
                
                l_duplicate_modules.append({"run": d_modules[barcode].run, "barcode": barcode, "fname": d_modules[barcode].fname})
        
        if (args.moduletype == constants.SM.KIND_OF_PART) :
            
            #d_modules[barcode] = SensorModule(
            #    barcode = barcode,
            #    lyso = d_loaded_part_info[constants.SM.KIND_OF_PART][barcode].lyso if barcode in d_loaded_part_info[constants.SM.KIND_OF_PART] else "0",
            #    sipm1 = d_loaded_part_info[constants.SM.KIND_OF_PART][barcode].sipm1 if barcode in d_loaded_part_info[constants.SM.KIND_OF_PART] else "0",
            #    sipm2 = d_loaded_part_info[constants.SM.KIND_OF_PART][barcode].sipm2 if barcode in d_loaded_part_info[constants.SM.KIND_OF_PART] else "0",
            #    run = run,
            #    fname = fname
            #)
            d_modules[barcode] = SensorModule(**{"run": run, "fname": fname, **d_loaded_part_info[constants.SM.KIND_OF_PART][barcode].__dict__})
        
        elif (args.moduletype == constants.DM.KIND_OF_PART) :
            
            #d_modules[barcode] = DetectorModule(
            #    barcode = barcode,
            #    sm1 = d_loaded_part_info[constants.DM.KIND_OF_PART][barcode].sm1 if barcode in d_loaded_part_info[constants.DM.KIND_OF_PART] else "0",
            #    sm2 = d_loaded_part_info[constants.DM.KIND_OF_PART][barcode].sm2 if barcode in d_loaded_part_info[constants.DM.KIND_OF_PART] else "0",
            #    run = run,
            #    fname = fname
            #)
            
            #d_modules[barcode] = d_loaded_part_info[constants.DM.KIND_OF_PART][barcode]
            #d_modules[barcode].run = run
            #d_modules[barcode].fname = fname
            #d_modules[barcode].category = None
            
            #d_modules[barcode] = DetectorModule(**{"run": run, "fname": fname, **d_loaded_part_info[constants.DM.KIND_OF_PART][barcode].dict()})
            d_modules[barcode] = DetectorModule(**{"run": run, "fname": fname, **d_loaded_part_info[constants.DM.KIND_OF_PART][barcode].__dict__})
            
    
    logging.info(f"Skipped {len(l_skipped_modules)} modules:")
    print("\n".join(l_skipped_modules))
    print()
    
    logging.info(f"Skipped {len(l_duplicate_modules)} duplicate modules:")
    print("#Run Barcode Filename")
    for module in l_duplicate_modules :
        
        print(f"{module['run']} {module['barcode']} {module['fname']}")
    
    print()
    
    # Read the plot config yaml
    d_plotcfgs = {}
    
    if (args.plotcfg) :
        
        with open(args.plotcfg, "r") as fopen :
            
            d_plotcfgs = yaml.load(fopen.read())
    
    # Read the category config yaml
    d_catcfgs = None
    
    with open(args.catcfg, "r") as fopen :
        
        d_catcfgs = yaml.load(fopen.read())
    
    # Read the definitions config yaml
    d_defs = {}
    
    if (args.defcfg) :
        
        with open(args.defcfg, "r") as fopen :
            
            d_defs = yaml.load(fopen.read())
    
    d_cat_results = {
        "categories": d_catcfgs["categories"],
        "counts": {_key: 0 for _key in d_catcfgs["categories"].keys()},
        "modules": {_key: [] for _key in d_catcfgs["categories"].keys()},
        "results": {}
    }
    
    
    # Process the files
    logging.info(f"Processing {len(d_modules)} modules ...")
    
    d_ones = {}
    l_modules_nodata = []
    l_modules_bad_eval = []
    
    for module in tqdm.tqdm(d_modules.values()) :
        logging.info(f"Processing {module}")
        
        rootfile = ROOT.TFile.Open(module.fname)
        isbad = False
        
        try:
            d_module_cat = utils.eval_category(
                rootfile = rootfile,
                d_catcfgs = d_catcfgs,
                barcode = module.barcode,
                d_fmt = {**module.dict(), **d_defs},
            )
        except ValueError as excpt:
            isbad = True
            l_modules_bad_eval.append((module, "category", "category", excpt))
            #continue
        
        module.category = d_module_cat["category"]
        d_cat_results["counts"][module.category] += 1
        d_cat_results["modules"][module.category].append(DoubleQuotedScalarString(module.barcode))
        d_cat_results["results"][module.barcode] = {
            "category": DoubleQuotedScalarString(module.category),
            **d_module_cat["metrics"],
            "fname": DoubleQuotedScalarString(module.fname),
        }
        
        module.results = d_cat_results["results"][module.barcode]
        
        if isbad :
            continue
        
        for plotname, plotcfg in d_plotcfgs.items() :
            
            for entryname, entrycfg in plotcfg["entries"].items() :
                
                if (isinstance(entrycfg["color"], str) and entrycfg["color"].startswith("#")) :
                    
                    entrycfg["color"] = ROOT.TColor.GetColor(entrycfg["color"])
                
                plot_arr = None
                nelements = None
                
                d_fmt = {**module.dict(), **d_defs}
                d_fmt.update(d_module_cat["metrics"])
                
                d_read_info = {}
                
                for varkey, varname in entrycfg.get("read", {}).items() :
                    
                    #varname = varname if rootfile.GetListOfKeys().Contains(varname) else f"{varname}_{module.barcode}"
                    d_read_info[varkey] = rootfile.Get(varname)
                    d_fmt[varkey] = f"d_read_info['{varkey}']"
                
                for defkey, defexpr in entrycfg.get("def", {}).items() :
                    
                    defexpr = defexpr.format(**d_fmt)
                    d_fmt[defkey] = defexpr
                
                if (plotcfg["type"] == "hist1") :
                    
                    if ("hist" not in entrycfg) :
                        
                        hist_tmp = ROOT.TH1F(
                            entryname,
                            entrycfg["label"],
                            plotcfg["nbins"],
                            plotcfg["xmin"],
                            plotcfg["xmax"],
                        )
                        
                        hist_tmp.SetDirectory(0)
                        hist_tmp.SetOption("hist")
                        hist_tmp.SetLineWidth(entrycfg.get("linewidth", 2))
                        hist_tmp.SetLineColor(entrycfg["color"])
                        hist_tmp.SetLineStyle(entrycfg.get("linestyle", 1))
                        hist_tmp.SetFillColor(entrycfg["color"])
                        hist_tmp.SetFillStyle(entrycfg["fillstyle"])
                        
                        #entrycfg["hist"] = hist_tmp
                        d_plotcfgs[plotname]["entries"][entryname]["hist"] = hist_tmp
                    
                    plot_str = entrycfg["plot"].format(**d_fmt)
                    weight_str = entrycfg.get("weight", "None").format(**d_fmt)
                    
                    try:
                        plot_eval_res = eval(plot_str)
                        weight_eval_res = eval(weight_str)
                        
                        # Skip filling the histogram if the evaluation result is None
                        # This can be used in the plot configuration to skip filling the histogram
                        if plot_eval_res is None :
                            continue
                        
                        plot_arr = numpy.array(plot_eval_res, dtype = float).flatten()
                        weight_arr = numpy.array(weight_eval_res, dtype = float).flatten() if weight_eval_res is not None else None
                    
                    except ValueError as excpt:
                        l_modules_bad_eval.append((module, plotname, entryname, excpt))
                        continue
                    
                    nelements = len(plot_arr)
                    
                    if nelements :
                        
                        if weight_arr is None :
                            
                            # Create and store arrays of ones of specific lengths; no need to recreate them everytime
                            if nelements not in d_ones :
                                
                                d_ones[nelements] = numpy.ones(nelements)
                            
                            weight_arr = d_ones[nelements]
                            
                        #print(module.barcode, plot_arr)
                        
                        entrycfg["hist"].FillN(
                            nelements,
                            plot_arr,
                            weight_arr,
                        )
                    
                    else :
                        
                        l_modules_nodata.append((module, plotname, entryname))
                
                elif (plotcfg["type"] == "graph") :
                    
                    if ("graph" not in entrycfg) :
                        
                        gr_tmp = ROOT.TGraph()
                        gr_tmp.SetName(entryname)
                        gr_tmp.SetTitle(entrycfg["label"])
                        
                        gr_tmp.SetLineWidth(2)
                        gr_tmp.SetLineColor(entrycfg["color"])
                        gr_tmp.SetMarkerColor(entrycfg["color"])
                        gr_tmp.SetMarkerSize(entrycfg["size"])
                        gr_tmp.SetMarkerStyle(entrycfg["marker"])
                        gr_tmp.SetFillStyle(0)
                        
                        d_plotcfgs[plotname]["entries"][entryname]["graph"] = gr_tmp
                    
                    plotx_str = entrycfg["plotx"].format(**d_fmt)
                    ploty_str = entrycfg["ploty"].format(**d_fmt)
                    
                    try:
                        plotx_arr = numpy.array(eval(plotx_str)).flatten()
                        ploty_arr = numpy.array(eval(ploty_str)).flatten()
                    except ValueError as excpt:
                        l_modules_bad_eval.append((module, plotname, entryname, excpt))
                        continue
                    
                    if (len(plotx_arr) != len(ploty_arr)) :
                        
                        logging.error(f"Error: Mismatch in x and y data dimensions for plot \"{plotname}\".")
                        print(f"  x[{len(plotx_arr)}]: {plotx_arr}")
                        print(f"  y[{len(ploty_arr)}]: {ploty_arr}")
                        sys.exit(1)
                    
                    if (not len(plotx_arr) or not len(ploty_arr)) :
                        
                        l_modules_nodata.append((module, plotname, entryname))
                    
                    for plotx, ploty in numpy.dstack((plotx_arr, ploty_arr))[0] :
                        
                        # Move the outliers to the outer range
                        plotx = max(plotcfg["xmin"], plotx) if (plotcfg["xmin"] is not None) else plotx
                        plotx = min(plotcfg["xmax"], plotx) if (plotcfg["xmax"] is not None) else plotx
                        
                        ploty = max(plotcfg["ymin"], ploty) if (plotcfg["ymin"] is not None) else ploty
                        ploty = min(plotcfg["ymax"], ploty) if (plotcfg["ymax"] is not None) else ploty
                        
                        entrycfg["graph"].AddPoint(plotx, ploty)
                
                else :
                    
                    logging.error(f"Error: Invalid plot type \"{plotcfg['type']}\" for plot \"{plotname}\".")
                    sys.exit(1)
        
        rootfile.Close()
    
    for plotname, plotcfg in d_plotcfgs.items() :
        
        if (plotcfg["type"] == "hist1") :
            
            l_hists = [_entrycfg["hist"] for _entrycfg in plotcfg["entries"].values()]
            
            for hist in l_hists :
                
                utils.handle_flows(hist)
                
                mean = hist.GetMean()
                mean_str = f"{round(mean)}" if mean > 100 else f"{mean:0.2f}"
                stddev = hist.GetStdDev()
                
                labelmode = plotcfg.get("labelmode", None)
                
                if (labelmode == "stddev") :
                    
                    hist.SetTitle(f"{hist.GetTitle()}#scale[0.7]{{ [#mu: {mean_str}, #sigma: {stddev:0.2f}]}}")
                
                elif (labelmode == "stddev_by_mean") :
                    
                    hist.SetTitle(f"{hist.GetTitle()}#scale[0.7]{{ [#mu: {mean_str}, #sigma: {stddev:0.2f}, #sigma/#mu: {stddev/abs(mean)*100:0.2f}%]}}")
            
            utils.root_plot1D(
                l_hist = l_hists,
                outfile = f"{args.outdir}/{plotname}.pdf",
                xrange = (plotcfg["xmin"], plotcfg["xmax"]),
                yrange = (
                    plotcfg.get("ymin", 0.5),
                    plotcfg.get("ymax", 1e3 * max([_hist.GetMaximum() for _hist in l_hists]))
                ),
                logx = plotcfg.get("logx", False),
                logy = plotcfg.get("logy", True),
                xtitle = plotcfg["xtitle"],
                ytitle = plotcfg["ytitle"],
                gridx = plotcfg.get("gridx", True),
                gridy = plotcfg.get("gridy", True),
                ndivisionsx = plotcfg.get("ndivisionsx", None),
                ndivisionsy = plotcfg.get("ndivisionsy", None),
                centerlabelx = plotcfg.get("centerlabelx", False),
                centerlabely = plotcfg.get("centerlabely", False),
                stackdrawopt = "nostack",
                legendpos = plotcfg.get("legendpos", "UR"),
                legendncol = 1,
                legendfillstyle = 0,
                legendfillcolor = 0,
                legendtextsize = 0.045,
                legendtitle = "+".join(args.location),
                legendheightscale = 1.0,
                legendwidthscale = 2.0,
                CMSextraText = "BTL Internal",
                lumiText = "Phase-2"
            )
        
        
        elif (plotcfg["type"] == "graph") :
            
            l_graphs = []
            
            xmin = plotcfg["xmin"]
            xmax = plotcfg["xmax"]
            
            ymin = plotcfg["ymin"]
            ymax = plotcfg["ymax"]
            
            for entrycfg in plotcfg["entries"].values() :
                
                gr = entrycfg["graph"]
                
                labelmode = plotcfg.get("labelmode", None)
                
                if (labelmode == "corr") :
                    
                    corr = numpy.corrcoef(get_grx(gr), get_gry(gr))[0, 1]*100
                    #corr_str = f"{corr:0.2g}"
                    gr.SetTitle(f"{gr.GetTitle()}#scale[0.7]{{ [#rho: {corr:0.2g}%]}}")
                
                for fnname, fnstr in entrycfg.get("fit", {}).items() :
                    
                    xmin_fn = min(numpy.array(gr.GetX()))
                    xmax_fn = max(numpy.array(gr.GetX()))
                    
                    f1 = ROOT.TF1(fnname, fnstr, xmin_fn, xmax_fn)
                    f1.SetLineWidth(2)
                    f1.SetLineStyle(7)
                    f1.SetLineColor(entrycfg["color"])
                    
                    fit_res = gr.Fit(
                        f1,
                        option = "SEM",
                        goption = "L",
                        xmin = xmin_fn,
                        xmax = xmax_fn
                    )
                    
                    #fn_fitted = gr.GetListOfFunctions().FindObject(fnname)
                    #fn_fitted.SetLineColor(entrycfg["color"])
                    #fn_fitted.SetLineWidth(2)
                    #fn_fitted.SetLineStyle(7)
                    #fn_fitted.SetMarkerSize(0)
                    #print("Fitted")
                    
                    fn_expr_str = utils.root_get_fn_expr(f1, "0.2g")
                    gr.SetTitle(f"{gr.GetTitle()} #scale[0.7]{{[y={fn_expr_str}]}}")
                
                gr.GetHistogram().SetOption(entrycfg["drawopt"])
                l_graphs.append(gr)
                
                arr_x_tmp = numpy.array(gr.GetX())
                arr_y_tmp = numpy.array(gr.GetY())
                
                if plotcfg["xmin"] is None :
                    
                    xmin = min(xmin, min(arr_x_tmp)) if xmin is not None else min(arr_x_tmp)
                
                if plotcfg["xmax"] is None :
                    
                    xmax = max(xmax, max(arr_x_tmp)) if xmax is not None else max(arr_x_tmp)
                
                if plotcfg["ymin"] is None :
                    
                    ymin = min(ymin, min(arr_y_tmp)) if ymin is not None else min(arr_y_tmp)
                
                if plotcfg["ymax"] is None :
                    
                    ymax = max(ymax, max(arr_y_tmp)) if ymax is not None else max(arr_y_tmp)
            
            if plotcfg["xmin"] is None and abs(xmin) > 100:
                
                xmin = 100*(numpy.floor(xmin/100)-1)
            
            if plotcfg["xmax"] is None and abs(xmax) > 100:
                
                xmax = 100*(numpy.ceil(xmax/100)+1)
            
            if plotcfg["ymin"] is None and abs(ymin) > 100:
                
                ymin = 100*(numpy.floor(ymin/100)-1)
            
            if plotcfg["ymax"] is None and abs(ymax) > 100:
                
                ymax = 100*(numpy.ceil(ymax/100)+1)
            
            
            utils.root_plot1D(
                l_hist = [ROOT.TH1F(f"h1_tmp_{plotname}", "", 1, xmin, xmax)],
                outfile = f"{args.outdir}/{plotname}.pdf",
                xrange = (xmin, xmax),
                yrange = (ymin, ymax),
                l_graph_overlay = l_graphs,
                logx = plotcfg.get("logx", False),
                logy = plotcfg.get("logy", False),
                xtitle = plotcfg["xtitle"],
                ytitle = plotcfg["ytitle"],
                gridx = plotcfg.get("gridx", True),
                gridy = plotcfg.get("gridy", True),
                ndivisionsx = plotcfg.get("ndivisionsx", None),
                ndivisionsy = plotcfg.get("ndivisionsy", None),
                centerlabelx = plotcfg.get("centerlabelx", False),
                centerlabely = plotcfg.get("centerlabely", False),
                stackdrawopt = "nostack",
                legendpos = plotcfg.get("legendpos", "UR"),
                legendncol = 1,
                legendfillstyle = 0,
                legendfillcolor = 0,
                legendtextsize = 0.045,
                legendtitle = "+".join(args.location),
                legendheightscale = 1.0,
                legendwidthscale = 1.9,
                CMSextraText = "BTL Internal",
                lumiText = "Phase-2"
            )
    
    
    # Save the categorization
    outfname = f"{args.outdir}/{args.moduletype}_categorization.yaml"
    logging.info(f"Writing categorizations to: {outfname}")
    
    d_cat_results["counts"]["total"] = sum(d_cat_results["counts"].values())
    
    with open(outfname, "w") as fopen:
        
        yaml.dump(d_cat_results, fopen)
    
    
    # Pair SMs
    if (args.pairsms) :
        
        #d_cat_pairs = {}
        
        d_produced_dms = utils.save_all_part_info(
            parttype = constants.DM.KIND_OF_PART,
            outyamlfile = args.dminfo,
            inyamlfile = args.dminfo,
            location_id = l_location_ids,
            ret = True,
            nodb = args.nodb
        )
        
        l_used_sms = list(itertools.chain(*[[_dm.sm1, _dm.sm2] for _dm in d_produced_dms.values()]))
        
        if (args.mixsmcats) :
            
            l_sms = [{
                "barcode": _sm,
                "pairing": d_cat_results["results"][_sm]["pairing"],
                "category": d_cat_results["results"][_sm]["category"],
            } for _sm in d_cat_results["results"].keys() if _sm not in l_used_sms and eval(d_catcfgs["pairing_condition"].format(**d_cat_results["results"][_sm]))]
            
            do_sm_pairing(
                l_sms = l_sms,
                cat = "mixed",
                outdir = args.outdir,
            )
        
        else :
            
            for cat in d_cat_results["categories"].keys() :
                
                #l_sms = [{_sm: d_cat_results["results"][_sm]} for _sm in d_cat_results["modules"][cat] if _sm not in l_used_sms]
                l_sms = [{
                    "barcode": _sm,
                    "pairing": d_cat_results["results"][_sm]["pairing"],
                    "category": d_cat_results["results"][_sm]["category"],
                } for _sm in d_cat_results["modules"][cat] if _sm not in l_used_sms and eval(d_catcfgs["pairing_condition"].format(**d_cat_results["results"][_sm]))]
                
                do_sm_pairing(
                    l_sms = l_sms,
                    cat = cat,
                    outdir = args.outdir,
                )
    
    
    # Group DMs
    if (args.groupdms) :
        
        d_cat_groups = {}
        
        d_produced_rus = utils.save_all_part_info(
            parttype = constants.RU.KIND_OF_PART,
            outyamlfile = args.ruinfo,
            inyamlfile = args.ruinfo,
            location_id = l_location_ids,
            ret = True,
            nodb = args.nodb
        )
        
        l_used_dms = list(itertools.chain(*[_ru.dms.values() for _ru in d_produced_rus.values()]))
        
        for cat in d_cat_results["categories"].keys() :
            
            l_dms = [{
                "barcode": _dm,
                "grouping": d_cat_results["results"][_dm]["grouping"],
                "sm_cat": d_cat_results["results"][_dm]["sm_cat"]
            } for _dm in d_cat_results["modules"][cat] if _dm not in l_used_dms and d_modules[_dm].location_id in l_location_ids]
            
            n_dms = len(l_dms)
            n_dms_ru = int(n_dms/12)*12
            n_dms_tray = int(n_dms/72)*72
            logging.info(f"Finding groups in {n_dms} category {cat} DMs ...")
            
            l_dms = l_dms[0: n_dms_tray]
            # Shuffle DMs
            rnd.shuffle(l_dms)
            # Within each tray, sirt DMs by the grouping metric
            l_dm_tray_groups = [
                sorted(l_dms[_i: _i+72], key = lambda _dm: _dm["grouping"])
            for _i in range(0, n_dms_tray, 72)]
            
            # DM positions in RU
            l_dmidx_ru = numpy.reshape(range(0, 12), (3, 4))
            l_dmidx_ru = l_dmidx_ru[::-1] if not args.flipru else l_dmidx_ru
            
            d_cat_groups[cat] = l_dm_tray_groups
            
            for itray, l_dms_tray in enumerate(l_dm_tray_groups) :
                
                outfname = f"{args.outdir}/dm-groups_cat-{cat}_tray-{itray+1}.txt"
                logging.info(f"Writing grouping results to: {outfname} ...")
                
                l_dm_ru_groups = [l_dms_tray[_i: _i+12] for _i in range(0, len(l_dms_tray), 12)]
                
                l_lines = []
                l_lines.append("[<DM position in RU> <DM barcode> <SM categories> <DM grouping metric>]\n\n")
                l_rus = []
                
                for iru, l_dms_ru in enumerate(l_dm_ru_groups) :
                    
                    # Shape into DM arrangement on an RU
                    l_dms_ru_shaped = numpy.reshape(l_dms_ru, (4, 3)).transpose()
                    l_dms_ru_shaped = l_dms_ru_shaped[::-1] if not args.flipru else l_dms_ru_shaped
                    
                    l_lines.append("="*100)
                    l_lines.append(f"RU {iru}:")
                    
                    # Create a dummy RU
                    # Required if one wants to compute RU metrics in the categorization yaml
                    ru = utils.ReadoutUnit(
                        dms = [d_modules[_dm["barcode"]] for _dm in l_dms_ru],
                    )
                    
                    d_ru_metrics = {}
                    for metric_name, metric_str in d_catcfgs["ru_metrics"].items() :
                        
                        d_ru_metrics[metric_name] = eval(metric_str.format(**d_ru_metrics))
                    
                    ru.results = d_ru_metrics
                    l_rus.append(ru)
                    
                    for irow, dm_row in enumerate(l_dms_ru_shaped) :
                        
                        l_lines.append(" ".join([f"[{l_dmidx_ru[irow][_idm]:2d} {_dm['barcode'].split('3211004000')[1]} {_dm['sm_cat']} {_dm['grouping']:0.2f}]" for _idm, _dm in enumerate(dm_row)]))
                    
                    l_lines.append("\nRU metrics:")
                    for metric_name, metric_val in d_ru_metrics.items() :
                        
                        l_lines.append(f"  {metric_name}: {metric_val:0.2f}")
                    
                    l_lines.append("="*100)
                    l_lines.append("\n")
                
                # Create a dummy Tray
                # Required if one wants to compute Tray metrics in the categorization yaml
                tray = utils.Tray(
                    rus = l_rus
                )
                
                l_lines_prefix = []
                l_lines_prefix.append(f"Category {cat}, Tray {itray+1}")
                l_lines_prefix.append("\nTray metrics:")
                
                d_tray_metrics = {}
                for metric_name, metric_str in d_catcfgs["tray_metrics"].items() :
                    
                    metric_val = eval(metric_str.format(**d_tray_metrics))
                    d_tray_metrics[metric_name] = metric_val
                    l_lines_prefix.append(f"  {metric_name}: {metric_val:0.2f}")
                
                l_lines_prefix.append("\n")
                l_lines = l_lines_prefix + l_lines
                
                with open(outfname, "w") as fopen :
                    
                    fopen.write("\n".join(l_lines)+"\n")
    
    
    # Save arguments
    outfname = f"{args.outdir}/arguments.yaml"
    logging.info(f"Writing program arguments to: {outfname} ...")
    with open(outfname, "w") as fopen:
        
        yaml.dump(vars(args), fopen)
    
    # Copy relevant files to the output directory for reference
    l_files_to_copy = [
        *args.skipmodules,
        args.plotcfg,
        args.catcfg,
        args.defcfg,
        args.sipminfo,
        args.sminfo,
        args.dminfo,
        args.dminfoextra[0] if args.dminfoextra else None,
        args.ruinfo,
        args.smresults,
    ]
    
    l_files_to_copy = [_f for _f in l_files_to_copy if _f and os.path.isfile(_f)]
    
    for fname in l_files_to_copy :
        
        logging.info(f"Copying {fname} to {args.outdir} ...")
        os.system(f"cp \"{fname}\" {args.outdir}/")
    
    if len(l_modules_nodata) :
        
        logging.info(f"No data found for the following {len(l_modules_nodata)} entries:")
        for module, plotname, entryname in l_modules_nodata:
            
            print(f"[barcode {module.barcode}] [plot {plotname}] [entry {entryname}] [file {module.fname}]")
    
    if len(l_modules_bad_eval) :
        
        logging.info(f"Bad data for the following {len(l_modules_bad_eval)} entries:")
        for module, plotname, entryname, excpt in l_modules_bad_eval:
            
            print(f"[barcode {module.barcode}] [plot {plotname}] [entry {entryname}] [file {module.fname}]")
            print(f"    Error: {excpt}")
    
    if (args.listmissing) :
        
        l_missing_modules = sorted(set(d_loaded_part_info[args.moduletype].keys()) - set(l_found_modules))
        
        if (len(l_missing_modules)) :
            
            logging.info(f"Missing run for {len(l_missing_modules)} modules:")
            for barcode in l_missing_modules :
                
                print(barcode)


if __name__ == "__main__" :
    
    main()
