
import ast
import numpy as np
from functools import cache
import json
from .filesystem_constants import DATA_DIR

####################
# incremental AWARE 
####################

def load_AWARE_basin_data():
    with open(DATA_DIR / "metadata" / "AWAREbas_faces.json", "r") as f:
        basin_intersections = json.load(f)
    with open(DATA_DIR / "metadata" / "AWARE_working_point.json", "r") as f:
        basin_working_point = json.load(f)
    return basin_intersections, BasinMemory(basin_working_point)

class BasinMemory:
    """
    Track basin-specific cumulative water consumption and calculate corresponding impact.

    This helper stores basin memory loaded from AWARE working-point data. For each
    basin it keeps:
    - a history of cumulative hydrological water consumption (`hwc`)
    - the basin-specific constant used for impact calculations
    - the remaining water availability before human water consumption
    - saved characterisation factors (CFs) for incremental calculations

    Parameters
    ----------
    initial : dict
        Mapping of basin IDs to initial state dictionaries. Each value should
        contain at minimum:
        - `"hwc"`: iterable of initial ()"default") basin inventory expressed as HWC values
        - `"basin_constant"`: iterable of basin-constants (AMDwa*area)
        - `"remaining_before_hwc"`: iterable of discharge remaining after subtracting EWR, but not human consumption

    Attributes
    ----------
    _hwc_invent : dict[int, list]
        Cumulative HWC history for each basin.
    _bas_const : dict[int, list]
        Basin constant values for each basin.
    _remaining_before_hwc : dict[int, list]
        For each basin, discharge remaining after subtracting EWR, but not human consumption
    incr_cfs : dict[int, list]
        Recorded CF values for each basin.
    cfs_calculated_cumulative : int
        Total number of times a CF value was saved.
    MBCs_with_cf_calculation : int
        Count of month-basin combinations with at least one CF calculation saved.
    """

    def __init__(self, initial):
        self.Mon_List = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        initial = {ast.literal_eval(key):val for key,val in initial.items()}
        self._hwc_invent = {(int(b[0]), str(b[1])): [ y["hwc"] ] for b, y in initial.items()}
        self._bas_const = {(int(b[0]), str(b[1])): y["basin_constant"] for b, y in initial.items()}
        self._remaining_before_hwc = {(int(b[0]), str(b[1])): y["remaining_before_hwc"] for b, y in initial.items()}
        self.incr_cfs = {(int(k[0]), str(k[1])): [] for k in initial.keys()}
        self.cfs_calculated_cumulative = 0
        self.MBCs_with_cf_calculation = 0

    def __str__(self):
        return (
            f"<BasinMemory\nmonth-basin-combinations={len(self._hwc_invent)}\n"
            f"cf calculation iterations={self.cfs_calculated_cumulative}\n"
            f"number of touched MBCs={self.MBCs_with_cf_calculation}>"
        )

    def __repr__(self):
        return self.__str__()

    def add(self, mbc, hwc):
        if mbc not in self._hwc_invent:
            raise KeyError(f"No basin memory for basin {mbc}")
        self._hwc_invent[mbc].append(self._hwc_invent[mbc][-1] + hwc)
    
    def add_seasonal(self, basin_id, hwc, season):
        if season == "annual":
            hwc = hwc/12
            for month in self.Mon_List:
                self.add((basin_id,month), hwc)
        elif isinstance(season, list):
            for month in season:
                self.add((basin_id,month), hwc/len(season))
        elif season in self.Mon_List:
                self.add((basin_id,season), hwc)
        else:
            raise ValueError("season not correctly defined") 

    def get_current(self, basin_id, month):
        return self._hwc_invent[(basin_id,month)][-1]

    def get_default(self, basin_id, month):
        return self._hwc_invent[(basin_id,month)][0]

    def total_lci(self, basin_id, month):
        return self.get_current(basin_id, month) - self.get_default(basin_id, month)
    
    def get_increments(self, basin_id, month):
        return self._hwc_invent[(basin_id,month)]
    
    def save_cf(self, basin_id, month, cf):
        if len(self.incr_cfs[(basin_id,month)]) ==0:
            self.MBCs_with_cf_calculation += 1
        self.incr_cfs[(basin_id,month)].append(cf)
        self.cfs_calculated_cumulative +=1
    
    def get_increm_cfs(self, index):
        if isinstance(index, tuple):
            return self.incr_cfs[(index[0],index[1])]
        elif isinstance(index, int):
            return {m:self.incr_cfs[(index,m)] for m in self.Mon_List}
        
    def impact(self, mbc, which):
        """get y value (impact) on impact curve for either default or current HWC"""
        if which == "current":
            return AWARE_impact_flexible_type(self.get_current(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
        elif which == "default":
            return AWARE_impact_flexible_type(self.get_default(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
        else:
            raise ValueError
        
    def reset(self):
        self._hwc_invent = {key: hwc_list[:1] for key, hwc_list in self._hwc_invent.items()}
        self.incr_cfs = {key: [] for key in self._hwc_invent.keys()}
        self.cfs_calculated_cumulative = 0
        self.MBCs_with_cf_calculation = 0
        
    def incremental_CF(self, basin_id, month):
        """
        calculates the incremental CF for a certain basin ID, always from default background to cumulative current background.
        if the current background is lower than the default, the slope is nevertheless returned as positive slope.
        """
        mbc = (basin_id, month)
        i1 = self.impact(mbc, "default")
        i2 = self.impact(mbc, "current")
        # calculate slope
        lci = self.total_lci(basin_id, month)
        if lci!=float(0):
            cf = (i2-i1)/lci
        else:
            # fallback to marginal cf for negligible inventory
            cf = self._bas_const[mbc]/(self._remaining_before_hwc[mbc]-self.get_current(*mbc))
            if cf<0: cf=100
            cf = max(0.1,min(100, cf))
        self.save_cf(basin_id, month, cf)
        return cf
    
    def incremental_CF_seasonal(self, basin_id, season):
        """
        calculates the incremental CF for a certain month.
        Alternatively, the annual average of incremental CFs, assuming that the inventory flow is equally distributed between months.
        """
        if season == "annual":
            return sum([self.incremental_CF(basin_id, month) for month in self.Mon_List])/12
        
        elif isinstance(season, list):
            return sum([self.incremental_CF(basin_id, month) for month in season])/len(season)
        
        elif season in self.Mon_List:
            return self.incremental_CF(basin_id, season)
        else:
            raise ValueError("season not correctly defined") 


def impact_funct(hwc, f_value, dis_zerocons_minus_ewr):
        return f_value*np.log((dis_zerocons_minus_ewr)/(dis_zerocons_minus_ewr-hwc))
    
@cache
def HWC_and_impact_for_cutoff_at(f_value,cutoff,dis_zerocons_minus_ewr):
    hwc = -f_value/cutoff+dis_zerocons_minus_ewr
    impact = impact_funct(hwc, f_value, dis_zerocons_minus_ewr)
    return hwc, impact

def impact_function_with_cutoff(hwc, basin_constant, dis_zerocons_minus_ewr):
    hwc_100co, impact_100co = HWC_and_impact_for_cutoff_at(basin_constant, 100, dis_zerocons_minus_ewr)
    hwc_01co, impact_01co = HWC_and_impact_for_cutoff_at(basin_constant, 0.1, dis_zerocons_minus_ewr)
    if hwc>hwc_100co: # impact at upper cf cutoff
        return impact_100co+100*(hwc-hwc_100co)
    elif hwc<hwc_01co: # impact at lower cf cutoff
        return impact_01co-0.1*(hwc-hwc_01co)
    else:
        return impact_funct(hwc, basin_constant, dis_zerocons_minus_ewr)

def AWARE_impact_flexible_type(x_vals, basin_constant, dis_zerocons_minus_ewr):
    try:
        y_with_co = [impact_function_with_cutoff(x,basin_constant, dis_zerocons_minus_ewr) for x in x_vals]
    except TypeError:
        y_with_co = impact_function_with_cutoff(x_vals,basin_constant, dis_zerocons_minus_ewr)
        assert isinstance(y_with_co, float)
    return y_with_co