"""Utilities for incremental AWARE impact calculations.

This module provides basin-memory helpers for tracking hydrological water
consumption and computing incremental characterization factors for AWARE.
"""

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
    impacts : dict[tuple[int, str], list[float]]
        Impact values recorded during the progression.
    
    """

    def __init__(self, initial):
        self.MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        initial = {ast.literal_eval(key):val for key,val in initial.items()}
        self._hwc_invent = {(int(b[0]), str(b[1])): [ y["hwc"] ] for b, y in initial.items()}
        self._hwc_default = {(int(b[0]), str(b[1])): y["hwc"] for b, y in initial.items()}
        self._bas_const = {(int(b[0]), str(b[1])): y["basin_constant"] for b, y in initial.items()}
        self._remaining_before_hwc = {(int(b[0]), str(b[1])): y["remaining_before_hwc"] for b, y in initial.items()}
        self.incr_cfs = {(int(k[0]), str(k[1])): [] for k in initial.keys()}
        self.impacts = {(int(k[0]), str(k[1])): [] for k in initial.keys()}
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

    def _add(self, mbc, hwc):
        """
        Append a new cumulative HWC value for a basin-month combination.
        """

        if mbc not in self._hwc_invent:
            raise KeyError(f"No basin memory for basin {mbc}")
        self._hwc_invent[mbc].append(self._hwc_invent[mbc][-1] + hwc)
    
    def _normalize_season(self, season):
        if season == "annual":
            return list(self.MONTHS)
        if isinstance(season, str):
            if season not in self.MONTHS:
                raise ValueError(f"Unknown month: {season}")
            return [season]
        if isinstance(season, (list, tuple)):
            return list(season)
        raise TypeError("season must be 'annual', a month name, or a list/tuple of months")

    def add_seasonal(self, basin_id, hwc, season):
        """
        Add water consumption for one month or a seasonal set of months.

        Parameters
        ----------
        basin_id : int
            Basin identifier.
        hwc : float
            Total water consumption to distribute across the selected months.
        season : str or list[str]
            Either a single month name, a list of month names, or "annual".

        Notes
        -----
        If `season` contains multiple months, the provided `hwc` is split evenly
        across them.
        """
        months = self._normalize_season(season) 
        
        hwc_monthly = hwc/len(months)
        for month in months:
            mbc = (basin_id,month)
            self._add(mbc, hwc_monthly)
            if len(self.impacts[mbc])==0:
                self.impacts[mbc].append(
                AWARE_impact_flexible_type(self._get_init_hwc(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
                )
            self.impacts[mbc].append(
                AWARE_impact_flexible_type(self._get_current(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
            )

    def _get_current(self, basin_id, month):
        """
        Return the current (latest) cumulative HWC value for a basin-month.
        """
        return self._hwc_invent[(basin_id,month)][-1]

    def _get_init_hwc(self, basin_id, month):
        """
        Return the initial HWC reference value (at default background pressure).
        """
        try:
            return self._hwc_invent[(basin_id,month)][0]
        except KeyError:
            raise KeyError(f"No basin memory for basin-month {basin_id}-{month}")

    def total_lci(self, basin_id, month):
        """
        Return the change in HWC relative to the inital background pressure.

        Parameters
        ----------
        basin_id : int
            Basin identifier.
        month : str
            Month name.

        Returns
        -------
        float
            Difference between current and initial HWC.
        """
        return self._get_current(basin_id, month) - self._get_init_hwc(basin_id, month)
    
    def _get_increments(self, basin_id, month):
        return self._hwc_invent[(basin_id,month)]
    
    def save_cf(self, basin_id, month, cf):
        """
        Store a newly calculated incremental CF.

        Parameters
        ----------
        basin_id : int
            Basin identifier.
        month : str
            Month name.
        cf : float
            Incremental characterization factor to store.
        """
        if len(self.incr_cfs[(basin_id,month)]) ==0:
            self.MBCs_with_cf_calculation += 1
        self.incr_cfs[(basin_id,month)].append(cf)
        self.cfs_calculated_cumulative +=1
    
    def _get_increm_cfs(self, index):
        """
        gets incremental CFs for a certain index tuple (basin, month)
        """
        if isinstance(index, tuple):
            return self.incr_cfs[(index[0],index[1])]
        elif isinstance(index, int):
            return {m:self.incr_cfs[(index,m)] for m in self.MONTHS}
        else:
            raise KeyError("wrong index")
        
    def _impact(self, mbc, which):
        """get y value (impact) on impact curve for either default or current HWC"""
        if which == "current":
            return AWARE_impact_flexible_type(self._get_current(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
        elif which == "default":
            return AWARE_impact_flexible_type(self._get_init_hwc(*mbc), self._bas_const[mbc], self._remaining_before_hwc[mbc])
        else:
            raise ValueError
        
    def reset(self):
        """
        Reset all stored state to the initial reference values.

        This clears all accumulated HWC history, calculated CFs, and impact
        progression data while preserving the original basin parameters.
        Changes made to the initial state are reset too.
        """
        self._hwc_invent = {key: [self._hwc_default[key]] for key in self._hwc_invent.keys()}
        self.incr_cfs = {key: [] for key in self._hwc_invent.keys()}
        self.impacts = {key: [] for key in self._hwc_invent.keys()}
        self.cfs_calculated_cumulative = 0
        self.MBCs_with_cf_calculation = 0
        
    def incremental_CF(self, basin_id, month):
        """
        calculates the incremental CF for a certain basin ID, always from default background to cumulative current background.
        if the current background is lower than the default, the slope is nevertheless returned as positive slope.
        """
        mbc = (basin_id, month)
        i1 = self._impact(mbc, "default")
        i2 = self._impact(mbc, "current")
        # calculate slope
        lci = self.total_lci(basin_id, month)
        if lci!=float(0):
            cf = (i2-i1)/lci
        else:
            # fallback to marginal cf for negligible inventory
            cf = self._bas_const[mbc]/(self._remaining_before_hwc[mbc]-self._get_current(*mbc))
            if cf<0: cf=100
            cf = max(0.1,min(100, cf))
        self.save_cf(basin_id, month, cf)
        return cf
    
    def incremental_CF_seasonal(self, basin_id, season):
        """
        calculates the incremental CF for a certain month.
        Alternatively, the annual or seasonal average of incremental CFs,
        assuming that the inventory flow is equally distributed between months.
        """
        if season == "annual":
            return sum([self.incremental_CF(basin_id, month) for month in self.MONTHS])/12
        
        elif isinstance(season, list):
            return sum([self.incremental_CF(basin_id, month) for month in season])/len(season)
        
        elif season in self.MONTHS:
            return self.incremental_CF(basin_id, season)
        else:
            raise ValueError("season not correctly defined")

    def set_initial_state(
        self,
        basin_id,
        month,
        hwc,
        *,
        basin_constant=None,
        remaining_before_hwc=None
    ):
        """
        Replace the initial HWC reference point for one basin-month combination.

        Parameters
        ----------
        basin_id : int
            Basin identifier.
        month : str
            Month name.
        hwc : float
            New initial HWC value.
        basin_constant : float, optional
            Optional updated basin-specific constant.
        remaining_before_hwc : float, optional
            Optional updated remaining discharge value.
        """
        mbc = (basin_id, month)
        self._hwc_invent[mbc] = [hwc]

        if basin_constant is not None:
            self._bas_const[mbc] = float(basin_constant)
        if remaining_before_hwc is not None:
            self._remaining_before_hwc[mbc] = float(remaining_before_hwc)
        self.incr_cfs[mbc] = []
        self.impacts[mbc] = []
        return
    
    def scale_initial_hwc(self, basin_id, month, multiplier):
        """
        Wrapper for set_initial_state, to adjust the hwc of a basin-month-combination
        """
        if max([len(l) for l in self._hwc_invent.values()])>1:
            raise ValueError("BASIN_MEMORY is not fresh, please consider resetting it. Did not implement change.")
        default = self._hwc_default[(basin_id, month)]
        self.set_initial_state(basin_id, month, default*multiplier)

    def get_marginal_cf_at(self, basin_id, month, hwc):
        """
        Return the marginal CF at a given background pressure value for one basin-month.

        Parameters
        ----------
        basin_id : int
        month : str
        hwc : float
            Background pressure value at which to evaluate the marginal CF.

        Returns
        -------
        float
            Marginal characterization factor at the requested point.
        """
        mbc = (int(basin_id), str(month))
        if mbc not in self._hwc_invent:
            raise KeyError(f"No basin memory for basin-month {mbc}")

        basin_constant = self._bas_const[mbc]
        remaining_before_hwc = self._remaining_before_hwc[mbc]

        # This is the local slope of the impact curve at the requested HWC
        # and is a natural proxy for the marginal CF.
        impact_current = AWARE_impact_flexible_type(
            hwc,
            basin_constant,
            remaining_before_hwc,
        )
        impact_delta = AWARE_impact_flexible_type(
            hwc + 1, # we are working with large numbers; 1 should be fine as "infinitesimal"
            basin_constant,
            remaining_before_hwc,
        )

        return impact_delta - impact_current


    def plot_impact_progress(
        self,
        mbc,
        ax=None,
        *,
        cmap="viridis_r",
        annotate=True,
        show=True,
        figsize=(8, 4),
        label_start=True,
        plot_used_part=False
    ):
        """Plot the basin HWC/impact progression with ordered arrows.

        Parameters
        ----------
        mbc : tuple[int, str]
            Basin-month combination, for example ``(31536, "May")``.
        ax : matplotlib.axes.Axes | None
            Optional axis to draw on. If ``None``, a new figure is created.
        cmap : str | matplotlib.colors.Colormap
            Colormap for marker order.
        annotate : bool
            Whether to draw horizontal arrows and vertical connectors.
        show : bool
            Whether to call ``plt.show()`` before returning.
        figsize : tuple[int, int]
            Figure size when a new figure is created.
        label_start : bool
            Whether to label the starting point marker.
        plot_used_part : bool, optional
            Whether to highlight the part of the curve that was actually used.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if mbc not in self._hwc_invent:
            raise KeyError(f"No basin memory for basin-month {mbc}")

        x_all = self._hwc_invent[mbc]
        x = np.asarray(remove_last_static_positions(x_all))
        y = np.asarray(self.impacts[mbc][:len(x)])
        if x.size == 0 or y.size == 0:
            raise ValueError(f"No plot data available for basin-month {mbc}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        color_order = np.arange(len(x))
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=color_order.min(), vmax=color_order.max())
        colors = cmap_obj(norm(color_order))

        #ax.plot(x, y, color="grey", alpha=0.7, label="impact function (interpolated)")
        ax.scatter(x, y, marker="o", c=colors, s=15, zorder=1)
        if label_start:
            ax.scatter([x[0]], [y[0]], marker="*", color="black", label="start (default background)", s=[40], zorder=2)
        xmin, xmax = ax.get_xlim()
        x_ifun = np.linspace(xmin,xmax,100)
        y_ifun = AWARE_impact_flexible_type(x_ifun, self._bas_const[mbc], self._remaining_before_hwc[mbc])
        ax.plot(x_ifun, y_ifun, color="grey", alpha=0.7, label="impact function")
        if plot_used_part:
            x_ifun = np.linspace(x[0],x[-1],50)
            y_ifun = AWARE_impact_flexible_type(x_ifun, self._bas_const[mbc], self._remaining_before_hwc[mbc])
            ax.plot(x_ifun, y_ifun, color="dimgrey", alpha=1, linestyle="--", label="used part of curve function")



        if annotate and len(x) > 1:
            ymin, ymax = ax.get_ylim()
            yrange = ymax - ymin
            y_pos = ymin + 0.04 * yrange
            y_pos_offset = 0.02 * yrange

            ax.vlines(
                    x[0],
                    ymin,
                    y[0],
                    colors="black",
                    linestyles=":",
                    linewidth=0.8,
                    alpha=0.7,
                )
                

            for i in range(len(x) - 1):
                x0, x1 = x[i], x[i + 1]
                y0, y1 = y[i], y[i + 1]
                ax.annotate(
                    "",
                    xy=(x1, y_pos),
                    xytext=(x0, y_pos),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=colors[i+1],
                        lw=1.2,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )
                ax.vlines(
                    x1,
                    y_pos,
                    y1,
                    colors=[colors[i+1]],
                    linestyles=":",
                    linewidth=0.8,
                    alpha=0.5,
                )
                #mid = 0.5 * (x0 + x1)
                ax.text(
                    x1,
                    y1 - y_pos_offset,
                    f"{i + 1}",
                    ha="center",
                    va="top",
                    fontsize=8,
                )
                y_pos += y_pos_offset
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin)
        ax.set_xlabel("background pressure (hwc, m³/month)")
        ax.set_ylabel("impact m³ world-eq./month")
        ax.set_title(f"Basin impact progression {mbc}")
        ax.legend(framealpha=0.4)

        if show:
            plt.show()
        return ax
        
def impact_funct(hwc, f_value, dis_zerocons_minus_ewr):
        return f_value*np.log((dis_zerocons_minus_ewr)/(dis_zerocons_minus_ewr-hwc))
    
@cache
def HWC_and_impact_for_cutoff_at(f_value,cutoff,dis_zerocons_minus_ewr):
    """
    inverts the AWARE CF function to obtain the hwc that would lead to a specific marginal CF.
    This function is used to identify the background pressure at the cut-offs of 0.1 or 100.
    """
    hwc = -f_value/cutoff+dis_zerocons_minus_ewr
    if hwc<0:
        return hwc, np.nan
    else:
        impact = impact_funct(hwc, f_value, dis_zerocons_minus_ewr)
        return hwc, impact

def impact_function_with_cutoff(hwc, basin_constant, dis_zerocons_minus_ewr):
    hwc_100co, _ = HWC_and_impact_for_cutoff_at(basin_constant, 100, dis_zerocons_minus_ewr)
    hwc_01co, _ = HWC_and_impact_for_cutoff_at(basin_constant, 0.1, dis_zerocons_minus_ewr)
    return handle_impact_curve_sections(hwc, hwc_100co, hwc_01co, basin_constant, dis_zerocons_minus_ewr)

def handle_impact_curve_sections(h, h_co, h_co01, basin_constant, dis_zerocons_minus_ewr):
    """Calculate impact values across AWARE impact curve sections.

    The impact function has three regions defined by cutoff points for the
    marginal characterization factor (CF):
    - lower section: slope 0.1 below the lower cutoff
    - middle section: nonlinear logarithmic section between the lower and
      upper cutoff
    - upper section: slope 100 above the upper cutoff

    The function also handles edge cases for negative water consumption and
    negative cutoff thresholds.

    Parameters
    ----------
    h : float
        Water consumption or background pressure value for which the impact is evaluated.
    h_co : float
        The upper cutoff value of water consumption where the marginal CF caps at 100.
    h_co01 : float
        The lower cutoff value of water consumption where the marginal CF floors at 0.1.
    basin_constant : float
        Basin-specific constant used in the AWARE logarithmic impact function.
    dis_zerocons_minus_ewr : float
        Available discharge after subtracting environmental water requirements, but without effects of human water consumption.

    Returns
    -------
    float
        The computed impact for the given water consumption level.
    """
    if h>h_co:
        # we are in cut-off part of curve            
        if h_co>0 and h_co01<0:
            # we are in the cut-off part of the curve but the cut-off point is at positive background pressure
            middle_part = impact_funct(h_co, basin_constant, dis_zerocons_minus_ewr)
            return middle_part+100*(h-h_co)
        
        elif h_co>0 and h_co01>0:
            # we need to take into account the constant part at the lower cut-off
            lower_part = 0.1*h_co01
            middle_part = impact_funct(h_co-h_co01, basin_constant, dis_zerocons_minus_ewr-h_co01)
            return lower_part+middle_part+100*(h-h_co)
        
        elif h_co<0:
            # we are in the cut-off part, but the cut-off happens at negative background pressure.
            # we keep zero as the reference point and integrate from there.
            # if h is negative, the integral will be negative as well, the impact simply is negative
            # we simply integrate from zero to the desired background pressure (integral is a rectangle)
            return  100*h
    elif h_co01<=h<=h_co:
        # we are in the middle section of the impact curve  
        if h>=0:
            if h_co01>0:
                # positive but affected by lower cutoff
                lower_part = 0.1*h_co01
                middle_part = impact_funct(h-h_co01, basin_constant, dis_zerocons_minus_ewr-h_co01)
                return lower_part+middle_part
            elif h_co01<=0:
                # standard case not affected by cut-offs
                return impact_funct(h, basin_constant, dis_zerocons_minus_ewr)
        elif h<0:
            if h_co>=0:
                # although h is negative, the normal logarithm should work.
                return impact_funct(h, basin_constant, dis_zerocons_minus_ewr)
            elif h_co<0:
                # first integrate down to cutoff, then the remaining part to h
                to_cutoff = 100*h_co
                i_curve = impact_funct(h-h_co, basin_constant, dis_zerocons_minus_ewr-h_co)
                return i_curve+to_cutoff
    elif h_co01>h:
        # we are in the low section of the impact curve
        if h_co01>=0:
            return 0.1*h
        elif h_co01<0:
            lower_part = 0.1*(h-h_co01)
            upper_part = 100*min(h_co,0)
            middle_part = impact_funct(h_co01-min(0,h_co), basin_constant, dis_zerocons_minus_ewr-min(0,h_co))
            return lower_part+middle_part+upper_part
    else:
        raise NameError("Could not calculate impact.")

def AWARE_impact_flexible_type(x_vals, basin_constant, dis_zerocons_minus_ewr):
    if np.isscalar(x_vals):
        y_with_co = impact_function_with_cutoff(x_vals,basin_constant, dis_zerocons_minus_ewr)
        assert isinstance(y_with_co, float)
        return y_with_co
    else:
        return [impact_function_with_cutoff(x,basin_constant, dis_zerocons_minus_ewr) for x in x_vals]

def remove_last_static_positions(u):
    """
    we consider printing OK since this is called in context of creating a figure in Jupyter Notebook
    """
    for i in range(len(u) - 1, 0, -1):
        if u[i] != u[i - 1]:
            u_trimmed = u[:i + 1]
            break
        else:
            u_trimmed = u[:1]
    print(f"removed last {len(u)-len(u_trimmed)} positions")
    return u_trimmed