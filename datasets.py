import seutils
import awkward
import uproot, os.path as osp, numpy as np, logging

DEFAULT_LOGGING_LEVEL = logging.DEBUG
def setup_logger(name='datasets'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[33m%(levelname)7s:%(asctime)s:%(module)s:%(lineno)s\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(DEFAULT_LOGGING_LEVEL)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

def bytes_to_human_readable(num, suffix='B'):
    """
    Convert number of bytes to a human readable string
    """
    for unit in ['','k','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return '{0:3.1f} {1}b'.format(num, unit)
        num /= 1024.0
    return '{0:3.1f} {1}b'.format(num, 'Y')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    from tqdm.notebook import tqdm
    logger.info('Using tqdm notebook')
else:
    import tqdm.tqdm as tqdm

ttjets = [
    # TTJets
    # 'Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',  # <-- All combined prob?
    'Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
    ]

qcd = [
    # QCD Pt
    'Autumn18.QCD_Pt_80to120_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_120to170_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_170to300_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
    'Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8',
    ]

wjets = [ 
    # WJetsToLNu
    'Autumn18.WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
    ]

zjets = [
    # ZJetsToNuNu
    'Autumn18.ZJetsToNuNu_HT-100To200_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-200To400_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-400To600_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-600To800_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-800To1200_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-1200To2500_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph',
    ]

all_bkgs = ttjets + qcd + wjets + zjets


# Function to get rootfiles for the bkg
class GetBkgRootfiles():
    def __init__(self, path):
        self.path = path
        self.all_rootfiles = None

    def read_all_rootfiles(self):
        if self.all_rootfiles is None:
            self.all_rootfiles = seutils.ls_root(self.path)

    def __call__(self, name):
        self.read_all_rootfiles()
        return [ f for f in self.all_rootfiles if osp.basename(f).startswith(name) ]

get_bkg_rootfiles = GetBkgRootfiles('root://cmseos.fnal.gov//store/user/klijnsma/semivis/treemaker_bkg_May18')


TRIGGER_TITLES = None
def get_trigger_titles():
    global TRIGGER_TITLES
    if TRIGGER_TITLES is None:
        # Just get a random root file of the dataset
        rootfile = get_bkg_rootfiles('')[0]
        title_bstring = uproot.open(rootfile).get('TreeMaker2/PreSelection')[ b'TriggerPass'].title
        TRIGGER_TITLES = title_bstring.decode('utf-8').split(',')
    return TRIGGER_TITLES

class Dataset(object):
    def __init__(self, name, rootfiles, treename='TreeMaker2/PreSelection', is_signal=False, bkgname=None):
        super().__init__()
        self.name = name
        self.bkgname = bkgname
        self.rootfiles = rootfiles
        self.treename = treename
        self.shortname = self.name[:20]
        self.is_signal = is_signal
        self.cache = None

    def __repr__(self):
        return super().__repr__().replace('Dataset', 'Dataset {0}'.format(self.shortname))

    def iterate(self, progressbar=True, n_files=None, **kwargs):
        rootfiles = self.rootfiles[:]
        if n_files: rootfiles = rootfiles[:n_files]
        default_kwargs = {
            'branches' : [b'JetsAK15_softDropMass'],
            # 'reportpath' : True,
            # 'reportfile' : True,
            # 'reportentries' : True
            }
        default_kwargs.update(kwargs)
        iterator = uproot.iterate(rootfiles, self.treename, **default_kwargs)
        if progressbar:
            iterator = tqdm(iterator, total=len(rootfiles), desc='files in {0}'.format(self.shortname))
        for elements in iterator:
            yield elements

    def iterate_branches(self, branches, **kwargs):
        # for path, file, start, stop, arrays in self.iterate(branches=branches, **kwargs):
        for arrays in self.iterate(branches=branches, **kwargs):
            yield arrays

    def cache_branches(self, branches, **kwargs):
        if not(self.cache is None): logger.info('Overwriting cache for %s', self)
        self.cache = []
        self.sizeof_cache = 0
        self.numentries_cache = 0
        for arrays in self.iterate_branches(branches, **kwargs):
            self.cache.append(arrays)
            self.sizeof_cache += sum([ v.nbytes for v in arrays.values() ])
            self.numentries_cache += arrays[branches[0]].shape[0]
        logger.info(
            'Cached ~%s (%s entries, %s branches)',
            bytes_to_human_readable(self.sizeof_cache), self.numentries_cache, len(branches)
            )

    def clear_cache(self):
        del self.cache

    def get_branches(self, branches, **kwargs):
        """
        Returns the branches merged, i.e. not per file
        Useful for when multiple root files as from the same sample
        """
        # Build output dict
        out = { b : [] for b in branches }
        for array in self.iterate_branches(branches, **kwargs):
            for b in branches:
                print(type(array[b]))
                out[b].append(array[b])
        # "Stack" all the arrays from individual files
        for b in branches:
            out[b] = awkward.concatenate(tuple(out[b]))
        return out

    def debug(self):
        print('test')
        for path, file, start, stop, arrays in self.iterate(reportpath=True, reportfile=True, reportentries=True):
            print(path, file, start, stop, len(arrays))    

    # Cross sections
    def get_xs(self):
        return self.get_xs_sig() if self.is_signal else self.get_xs_bkg()

    def get_xs_sig(self):
        if not hasattr(self, 'xs'): self.xs = 100.
        return self.xs
        
    def get_xs_bkg(self):
        if hasattr(self, 'xs'): return self.xs
        import crosssections
        datasetname = self.name.split('.', 1)[1]
        if '_ext' in datasetname: datasetname = datasetname.split('_ext')[0]
        self.xs = crosssections.get_xs(datasetname)
        if self.xs is None:
            raise RuntimeError('No cross section for {0}'.format(self.name))
        return self.xs

    def get_numentries(self, use_cache=True):
        if use_cache:
            return self.numentries_cache
        else:
            if not hasattr(self, 'numentries'):
                logger.info(
                    'Calculating numentries for %s rootfiles in %s',
                    len(self.rootfiles), self.shortname
                    )
                self.numentries = 0
                for rootfile in self.rootfiles:
                    self.numentries += uproot.open(rootfile).get(self.treename).numentries
            return self.numentries

    def get_weight(self, use_cache=True):
        numentries = self.get_numentries(use_cache)
        if numentries == 0: return 0.0
        return self.get_xs() / float(numentries)


# Shortcuts to make the datasets

def init_ttjets_datasets():
    return [ Dataset(name, get_bkg_rootfiles(name), bkgname='ttjets') for name in ttjets ]

def init_qcd_datasets():
    return [ Dataset(name, get_bkg_rootfiles(name), bkgname='qcd') for name in qcd ]

def init_wjets_datasets():
    return [ Dataset(name, get_bkg_rootfiles(name), bkgname='wjets') for name in wjets ]

def init_zjets_datasets():
    return [ Dataset(name, get_bkg_rootfiles(name), bkgname='zjets') for name in zjets ]

def init_bkgs(make_cache=False, cache_branches=None):
    ttjets = init_ttjets_datasets()
    qcd = init_qcd_datasets()
    wjets = init_wjets_datasets()
    zjets = init_zjets_datasets()
    if make_cache:
        cache_branches = [b'JetsAK15', b'JetsAK15_softDropMass', b'TriggerPass', b'MET', b'METPhi']
        # For qcd and ttjets 1 file contain more than enough events
        for dataset in ttjets + qcd:
            dataset.cache_branches(cache_branches, n_files=1)
        # For wjets and zjets we need a little more files for a reasonable test set
        for dataset in wjets:
            dataset.cache_branches(cache_branches, n_files=2)
        for dataset in zjets:
            dataset.cache_branches(cache_branches, n_files=4)
    return ttjets, qcd, wjets, zjets

def iterate_bkg(bkgs, use_cache=True, branches=None, **kwargs):
    """
    Yields arrays from lists of datasets
    'arrays' is like the TTree per file
    Expects: bkgs = [ [ Dataset, Dataset, ... ], ... ]
    """
    if branches is None: branches = [b'JetsAK15_softDropMass', b'TriggerPass']
    if use_cache: logger.warning('Using cached data (limited)')
    for bkg in bkgs:
        for dataset in bkg:
            arrays_iterator = dataset.cache if use_cache else dataset.iterate_branches(branches=branches, **kwargs)
            # Weight takes into account the cross section and the number of events
            weight = dataset.get_weight(use_cache=use_cache)
            for arrays in arrays_iterator:
                yield arrays, weight, dataset


def iterate_sig(datasets, use_cache=True, branches=None, **kwargs):
    """
    Yields arrays from a list of datasets
    'arrays' is like the TTree per file
    Expects: datasets = [ Dataset, Dataset, ... ]
    """
    if branches is None: branches = [b'JetsAK15_softDropMass', b'TriggerPass']
    if use_cache: logger.warning('Using cached data (limited)')
    for signal in datasets:
        arrays_iterator = signal.cache if use_cache else signal.iterate_branches(branches=branches, **kwargs)
        weight = signal.get_weight(use_cache=use_cache)
        for arrays in arrays_iterator:
            yield arrays, weight, signal

def numentries(arrays):
    """
    Counts the number of entries in a typical arrays from a ROOT file,
    by looking at the length of the first key
    """
    return arrays[list(arrays.keys())[0]].shape[0]
