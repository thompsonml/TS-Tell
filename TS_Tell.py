# suppress KNOWN warnings
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.simplefilter('ignore', InterpolationWarning)

# typing
import typing
from typing import Optional, Tuple

# data analysis
import pandas as pd
import numpy as np
from scipy.stats import yeojohnson
from pandas.plotting import autocorrelation_plot

from scipy.signal import periodogram
from scipy.signal import welch

from scipy.signal import savgol_filter
from whittaker_eilers import WhittakerSmoother


# graphics
from matplotlib import pyplot as plt
import seaborn as sns

# statistics, modeling
from arch.unitroot import PhillipsPerron

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm

# time series
from sktime.utils.plotting import plot_series
from sktime.utils.plotting import plot_correlations

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sktime.forecasting.sarimax import SARIMAX

from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS

from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoCES
from sktime.forecasting.statsforecast import StatsForecastAutoTheta
from sktime.forecasting.statsforecast import StatsForecastAutoTBATS

from sktime.forecasting.naive import NaiveForecaster

from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.tbats import TBATS

from sktime.forecasting.pykan_forecaster import PyKANForecaster
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.forecasting.neuralforecast import NeuralForecastRNN
from sktime.forecasting.timesfm_forecaster import TimesFMForecaster

"""
--------------------------------------------------------------------------------
"""
class TS_Tell():
    def __init__(self, 
                 input_ts: pd.Series, 
                 max_season: int=52
                ):
        """Default constructor of the TS_Tell class

        Parameters
        ----------        
        input_ts : pd.Series, float
            The input time series - Index should be `DatetimeIndex` (see Notes)
        max_season : int, default 52
            The maximum value of the season / cycle (12 for monthly, 52 for 
            weekly, 4 for quarterly, 7 for daily, 168 for hourly, etc.)

        Notes
        -----
        `input_ts`: expected to be a pd.Series with a DatetimeIndex - checked 
            and will fail if not. If not a DatetimeIndex it will be attempted
            to be converted but will fail if unsuccessful

        Raises
        ------
        TypeError 
            If the input type is not a pd.Series

        ValueError
            If the index is not a DatetimeIndex and cannot be converted to one
        
        """
        self.input_ts = input_ts
        self.max_season = max_season

        # additional objects
        self.n_obs = len(self.input_ts)

        # Exceptions
        if not isinstance(self.input_ts, pd.Series):
            raise TypeError("Expected a Pandas Series, but got a {}".format(
                #type(series).__name__))
                type(self.input_ts).__name__))
            
        if not isinstance(self.input_ts.index, pd.DatetimeIndex):
            try:
                 self.input_ts.index = pd.to_datetime(self.input_ts.index)
                 self.data_freq = pd.infer_freq(self.input_ts.index)[:1]
            except:
                raise ValueError("The index of the pd.Series should be a "
                    "DatetimeIndex. The conversion of the Index on the "
                    "input time series using `pd.to_datetime()` was "
                    "attempted but failed.")

    # PRIVATE methods
    def _get_data_freq(self) -> str:
        """Private method to get the frequency of the input time series
        
        """
        self.data_freq = pd.infer_freq(self.input_ts.index)[:1]
        return self.data_freq


    def _get_trend_dataframe(self) -> pd.DataFrame:
        """Private method to obtain a trend dataframe for multiple methods

        """
        df = pd.DataFrame(self.input_ts)
        df.columns = ['y']
        return df


    def _get_hp_lambda(self) -> int:
        """Private method to get the value for the Hodrick-Prescott Lambda

        References
        -----
        [1] https://www.stata.com/manuals/tstsfilterhp.pdf
            - P. 8, first paragraph
        """
        hp_lambda_dict = {'A': 1600/4**4,
                          "6M": 100,
                          'Q': 1600,
                          'M': 1600 * 3**4, 
                          'W': 1600 * 12**4,
                          'D': 1600 * (365/4)**4,
                         }
        hp_lambda = hp_lambda_dict[self._get_data_freq()]
        return hp_lambda

    
    # PUBLIC methods
    def get_stationarity_tests(self) -> tuple:
        """Get stationarity tests to examine for presence of a unit root

        Notes
        -----
        
        Augmented Dickey-Fuller (ADF) Test
            Ho: The time series has a unit root (non-stationary)

        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
            Ho: The time series is stationary around a deterministic trend

        Phillips-Perron (PP) Test
            Ho: The time series has a unit root (non-stationary)

        References
        ----------
        [1] https://www.google.com/search?q=augmented+dickey+full+vs+kpss+vs+phillips-perone+in+python&client=ubuntu-chr&hs=YKR&sca_esv=a7528baf20b27cbb&sxsrf=AHTn8zprgoczwz8YQCAMbyiRqxMw3TitcQ%3A1744806711117&ei=N6P_Z-vwBofT5NoPuuqdgQU&ved=0ahUKEwir6N7Bx9yMAxWHKVkFHTp1J1AQ4dUDCBA&uact=5&oq=augmented+dickey+full+vs+kpss+vs+phillips-perone+in+python&gs_lp=Egxnd3Mtd2l6LXNlcnAiOmF1Z21lbnRlZCBkaWNrZXkgZnVsbCB2cyBrcHNzIHZzIHBoaWxsaXBzLXBlcm9uZSBpbiBweXRob24yCBAAGIAEGKIEMgUQABjvBTIIEAAYogQYiQVIjihQnwRY4yRwAXgBkAEBmAGsAaAB-RGqAQQ4LjE0uAEDyAEA-AEBmAIDoAKRAsICChAAGLADGNYEGEfCAgQQIRgKmAMA4gMFEgExIECIBgGQBgiSBwMxLjKgB5orsgcDMC4yuAf9AQ&sclient=gws-wiz-serp
        
        """
        adf_test = adfuller(self.input_ts)
        kpss_test = kpss(self.input_ts)
        pp_test = PhillipsPerron(self.input_ts)

        d_input_ts = self.input_ts.diff()
        #d_input_ts = d_input_ts.fillna(d_input_ts.mean())

        d_adf_test = adfuller(d_input_ts.dropna())
        d_kpss_test = kpss(d_input_ts.dropna())
        d_pp_test = PhillipsPerron(d_input_ts.dropna())

        stationarity_tests = (adf_test[1], kpss_test[1].item(), pp_test.pvalue,
                              d_adf_test[1], d_kpss_test[1].item(), d_pp_test.pvalue)
        
        return stationarity_tests


    def get_yeo_johnson(self) -> Tuple[pd.Series, float]:
        """Get Yeo Johnson

        Returns
        -------
        Tuple(pd.Series, float)
            [0] The first element is the input series raised to the power 
                returned by the `yeojohnson` function
            [1] The second element is the input series raised to that power
                        
        """
        series, _lambda = yeojohnson(self.input_ts)
        series_yj = pd.Series(series, name="yj")
        series_yj.index = self.input_ts.index
        return (series_yj, _lambda)
        

    def get_sample_facts(self, 
                         print_messages: bool=True, 
                         return_results: bool=True
                            ) -> Optional[pd.Series]:
        """Get basic facts about the sample:
        
            - N-related (N, N Miss)
            - Moment-related (Max, mean, min, std, skew, kurtosis)
            - Transformation/TICS-related
            - Stationarity (as-is and differenced)

        Parameters
        ----------
        print_messages : bool default True
            Whether or not to print the output messages
        return_results : bool default True
            Whether or not to return a pd.Series of the results
        
        Returns
        -------
        Optional, pd.Series of the metrics

        Notes
        -----
        YJ Lambda: the Yeo-Johnson Lambda builds upon Box-Cox Lambda by allow-
            ing negative values. Lambda is the power the time series is raised 
            to in order to obtain a new distribution as closely related to a
            Normal distrubtion as possible. Some important values (approximate)
            would be as follows:
                | Value | Transform |
                | ----- | --------- |
                |  0.0  | LN |
                |  0.5  | Square Root |
                |  1.0  | Normal |

            Stepping back, the whole reason data are transformed are an attempt
            to remedy due to "ill-behaving" data. 
            @TODO
                
        Ljung-Box: Ljung-Box tests indepence amongst the residuals of a time
            series model by examining autocorrelation. A significant p-value 
            implies something has not been accounted for in modeling the time 
            series. A lack of independence may be from one or more of TICS:
            Trend, Irregularity, Cyclicality, Seasonality.

        ADF / KPSS / PP
            
        """
        if print_messages:
            print("\n##### {:^17} #####".format("SAMPLE FACTS"))
        n_miss = self.input_ts.isnull().sum()
        max = self.input_ts.max()
        min = self.input_ts.min()
        mean= self.input_ts.mean()        
        std = self.input_ts.std()
        skew= self.input_ts.skew()
        kurt= self.input_ts.kurtosis()
        yj_lambda = self.get_yeo_johnson()[1]
        lb_df = acorr_ljungbox(self.input_ts, lags=self.max_season)
        lb_pval = lb_df.loc[self.max_season][1:].item()
        adf, kpss, pp, d_adf, d_kpss, d_pp = self.get_stationarity_tests()
        metrics = [self.n_obs, n_miss, self._get_data_freq(), max, min, 
                   mean, std, skew, kurt, yj_lambda, lb_pval,
                   adf, kpss, pp, d_adf, d_kpss, d_pp]
        if print_messages:
            print("\nN / Freq\n" \
                  "--------\n" \
                  "Total N: {:>20d}\n" \
                  "N Miss: {:>21d}\n" \
                  "TS Freq: {:>20s}\n\n" \
                  "Extremes\n" \
                  "--------\n" \
                  "Max: {:>24.4f}\n" \
                  "Min: {:>24.4f}\n\n" \
                  "Moments\n" \
                  "-------\n" \
                  "Mean: {:>23.4f}\n" \
                  "Std: {:>24.4f}\n" \
                  "Skew: {:>23.4f}\n" \
                  "Kurtosis: {:>19.4f}\n\n" \
                  "Distribution / Transform\n" \
                  "------------------------\n" \
                  "YJ Lambda: {:>18.4f}\n\n" \
                  "AutoCorr of Residuals\n" \
                  "---------------------\n" \
                  "Ljung-Box: {:>18.4f}\n\n" \
                  "Original Series\n" \
                  "---------------\n" \
                  "ADF Test: {:>19.4f}\n" \
                  "KPSS Test: {:>18.4f}\n" \
                  "PP Test: {:>20.4f}\n\n" \
                  "Differenced Series\n" \
                  "------------------\n" \
                  "ADF Test: {:>19.4f}\n" \
                  "KPSS Test: {:>18.4f}\n" \
                  "PP Test: {:>20.4f}\n".format(*metrics)
                 )
        
        if return_results:
            index_vals = ["Total N", "N Miss", "TS Freq", "Max", "Min", 
                          "Mean", "Std Dev", "Skew", "Kurtosis", "YJ Lambda", 
                          "Ljung-Box", "ADF Test", "KPSS Test", "PP Test", 
                          "D ADF Test", "D KPSS Test", "D PP Test"]
            sample_facts = pd.Series(metrics, index=index_vals)
            return sample_facts


    def get_missing_imputation(self, print_graph: bool=True) -> pd.DataFrame:
        """Get time series missing imputation

        Use this method when a value is missing to impute

        Parameters
        ----------
        print_graph : bool default True
            Whether or not to print out the graph 

        Returns
        -------
        input_ts : float
            The input time series imputed to fill in missing values

        Notes
        -----
        Utilizes bi-directional interpolation to fill in missing data
        points. Since time series are temporal (i.e., adjacent values are
        more correlated / alike than non-adjacent values), a typical 
        `random` imputation will miss this nuance.

        """
        df = pd.DataFrame(self.input_ts)
        df.columns = ['y']
        df["missing"] = np.where(df['y'].isnull()==True, 1, 0)
        df['y'] = df['y'].interpolate(limit_direction="both")

        if print_graph:
            ax = df['y'].plot(alpha=0.75, label="Input Time Series")
            ax = df.query("missing==1")['y'].plot(ax=ax, marker='o', ls='', 
                                                  label="Imputed Value")
            plt.legend()
            plt.show()
            
        return df
        
    
    def get_autocorr_plots(self) -> None:
        """Get AutoCorr Plots
        
        Get the combo plots: Time Series, PACF, and ACF

        Notes
        -----
        For more info, see Dr. Nau's lecture notes via References below.

        PACF: the number of possible AR term(s)
        ACF: the number of possible MA term(s)
        Early Lags: generally 0, 1, 2
        Late Lags: typically 12, 24 - will also sometimes oscillate between 
            positive and negative (i.e., across the x-axis where 12 may be
            negative and 24 is positive)
        ARIMA: typically sees an Early Lag on either PACF or ACF
        SARIMA: generally exhibits both and Early Lag and a Late Lag
        - Note: this does not mean `multiple` seasonalities, where a signal
            may demonstrate (as an example) both hourly and weekly 
            seasonalities

        Use these as a `guide` - it's great when everything works `out-of-the-
        box` but it does not always work out that way. For instance: it is not 
        uncommon to fit a model using the PACF and ACF results and then find 
        that a term is suddenly insignificant. One such example is when both AR 
        and MA terms are specified simultaneously which can mathematically lead 
        to cancelling of terms. 
    
        References
        ----------
        [1] https://people.duke.edu/~rnau/411arim3.htm

        [2] https://people.duke.edu/~rnau/Mathematical_structure_of_ARIMA_models--Robert_Nau.pdf
        
        """    
        fig = plt.figure(figsize=(12, 5))
        
        # time series
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("Input Time Series")
        ax1.grid()
        ax1.plot(self.input_ts, marker='o', markersize=3)
        
        # PACF
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.set_xlabel("Lag / AR Term")
        ax2.grid()
        sm.graphics.tsa.plot_pacf(self.input_ts.values.squeeze(), lags=40, ax=ax2)
        
        # ACF
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.set_xlabel("Lag / MA Term")
        ax3.grid()
        sm.graphics.tsa.plot_acf(self.input_ts.values.squeeze(), lags=40, ax=ax3)
        
        plt.tight_layout()
        plt.show()
        
    
    def get_hist_box(self, n_hist_bins: int=20) -> None:
        """Get a combo Histogram + Boxplot

        Parameters
        ----------
        n_hist_bins : int default 20
            The number of bins to use for the histogram

        Notes
        -----
        The Histogram is useful to examine the shape of the distribution, 
        specifically to examine symmetry and determine whether a trans-
        formation would better aid in modeling efforts.

        The Boxplot is to examine Central Tendency and Spread as well as 
        to examine for possible Extrema.
        
        """
        plt.figure(figsize=(12,4))
        plt.suptitle("Histogram and Boxplot")
        
        plt.subplot(1, 2, 1)
        plt.hist(self.input_ts, bins=n_hist_bins, density=True, 
                 facecolor="lightsteelblue", edgecolor='k')
        self.input_ts.plot(kind="density", ls='--', color='navy')
        plt.title("Distribution Symmetry")
        plt.xlabel('')
        plt.grid()
        
        plt.subplot(1, 2, 2)
        result = plt.boxplot(self.input_ts)
        outliers = result['fliers'][0].get_ydata()
        sns.boxplot(self.input_ts)
        plt.title("Central Tendency, Spread, and Possible Extrema")
        outlier_vals  = ", ".join([str(np.round(x, 4)) for x in outliers])
        plt.xlabel("Possible Extrema:\n[{}]".format(outlier_vals))
        plt.grid()
        
        plt.tight_layout()
        plt.show()

    #--------------------------------------------------------------------------------
    def _get_WMA(self, s, period):
       return s.rolling(period).apply(lambda x: 
                        ((np.arange(period)+1)*x).sum() / 
                                      (np.arange(period)+1).sum(), raw=True)

    def _get_HMA(self, s, period):
       return self._get_WMA(self._get_WMA(s, period // 2
                ).multiply(2).sub(self._get_WMA(s, period)), int(np.sqrt(period)))


    def get_smoothed_imputation(self, 
                                smoother="WE",
                                smooth_order=3,
                                std_points=4,
                                extrema_std=3,
                                critical_z=2.326,
                                return_results=True) -> Optional[pd.DataFrame]:
        """Get a smoothed, imputed version of the input Time Series

        Use the parameters to create a smoother, imputed series to further
        analyze and/or model.
        
        Uses smoothing to examine possible cases of extrema/outliers/anomalies. 
        Essentially, these smoothers estimate the center of mass of the input 
        time series, meaning imputation will be affected by where the value 
        occurs along the curve. Given the temporal nature of time series data, 
        this is ideal - the probability of the value being imputed should be 
        influenced by where along the curve it occurs and the values around it.

        
        
        Parameters
        ----------
        smoother : str {"WE", "SG", "HMA"} default "WE"
            The smoother to use, either Whittaker-Eilers ("WE") <default>
            or Savitsky-Golay ("SG") or Hull's Moving Average ("HMA")
        smooth_order : int default 3
            The order (power) to raise the function for smoothing. For *most* 
            series, the default value (3) is a good starting value - much
            lower and you just get the original series and much higher results
            in a approximates nearly a straight line            
        std_points : int default 4
            The number of consecutive data points used to contruct the moving
            standard deviation
        extrema_std : float default 4
            The std dev beyond which values may be extrema
        critical_z : float, default 2.326
            The Z-value denoting the top and bottom % to consider for extrema/
            outliers/anomolies (2.326 is top/bottom 1% - see a Critical Z chart
            for more values
        return_results : bool default True
            Whether or not to return the results via a pd.DataFrame

        Returns
        -------
        input_ts : float
            The original input time series
        std_roll : float
            The rolling standard deviation of the time series
        z : float
            The standardized input time series
        smooth_ts : float
            The smoothed filter value
        hi_bound : float
            The smoothed filter value plus `extrema_std` std devs
        lo_bound : float
            The smoothed filter value minus `extrema_std` std devs
        examine : int
            1 if an observation to examine, otherwise 0
        imp : float
            The original series imputed with smoothed values

        Raises
        ------
        raiseValueError
            If the value passed to `smoother` is not:
                - 'WE'
                - 'SG'
                - 'HMA'
            ..then a ValueError is raised

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

        [2]

        [3]

        @TODO
        -----
        [1] https://chemometrics.readthedocs.io/en/stable/examples/whittaker.html
        
        """
        df = self._get_trend_dataframe()
        df["std_roll"] = df["y"].rolling(std_points).std()
        df["std_roll"] = df["std_roll"].interpolate(limit_direction="both")
        df['z'] = (df['y'] - df['y'].mean()) / df['y'].std()

        if smoother=="WE":
            df["smooth_ts"] = WhittakerSmoother(self.max_season, 
                                                smooth_order, 
                                                self.n_obs).smooth(self.input_ts)
            smooth_kind = "Whittaker-Eilters"
        elif smoother=="SG":
            df["smooth_ts"] = savgol_filter(self.input_ts, 
                                            self.max_season, 
                                            smooth_order)
            smooth_kind = "Savitsky-Golay"
        elif smoother=="HMA":
            df["smooth_ts"] = self._get_HMA(self.input_ts, int(self.max_season / 2))
            smooth_kind = "HMA"
        else:
        #--------------------------------------------------------------------------------
            raise ValueError("The type of smoother passed must be either " + \
                    "``WE`` or ``SG`` or ``HMA``")
            
        df["hi_bound"] = df["smooth_ts"] + df["smooth_ts"].std() * extrema_std
        df["lo_bound"] = df["smooth_ts"] - df["smooth_ts"].std() * extrema_std

        df["examine"] = np.where(abs(df['z']) > critical_z, 1, 0)

        df["imp"] = np.where(
            ((df["examine"]==1) & (df['y'] > 0)) & (df['hi_bound'] < df['y']), 
            df["hi_bound"],
            np.where(
                (df["examine"]==1) & (df['y'] < 0) & (df['lo_bound'] > df['y']), 
                df["lo_bound"],
                df['y']
            )
        )

        df['y'].plot(figsize=(13, 4), alpha=0.6, label="Input Series")
        df["smooth_ts"].plot(c='g', label=smooth_kind)
        styling = {"color": "peru", "ls": '--', "marker": 'o', 
                   "markersize": 1.5, "alpha": 0.3}
        df["hi_bound"].plot(**styling, label=f"Hi Bound + {extrema_std}$\sigma$")
        df["lo_bound"].plot(**styling, label=f"Lo Bound - {extrema_std}$\sigma$")
        plt.axhline(y=critical_z, color='r', linestyle='--', alpha=0.5, label="Critical Z Value")
        plt.axhline(y=-critical_z, color='r', linestyle='--', alpha=0.5)
        plt.suptitle("Curve Shape and Possible Extrema", fontsize=11)
        plt.title(f"Critical Z: {critical_z}", fontsize=9)
        plt.legend(fontsize=8, ncol=5)
        plt.grid()
        plt.show()

        if return_results:
            return df


    def get_lag_tests(self, sig_pval: float=0.05) -> pd.Series:
        """Get lag test

        Parameters
        ----------
        sig_pval : float default 0.05
            The p-value about which significance is determined

        Notes
        -----
        @TODO
        """
        df = self._get_trend_dataframe()
        lag_dict = {}
        for i in range(1, self.max_season + 1):
            ols = sm.OLS.from_formula('y ~ y.shift(' + str(i) + ')', df).fit()
            lag_dict[i] = ols.pvalues[1:].item()
        lag_sig_dict = {key: value for key, value in lag_dict.items() if 
                            value < sig_pval}
        lag_series = pd.Series(lag_dict, name="p_value")
        lag_series.index.name = "Lag"
        print("### LAG TESTS ###")

        if len(lag_sig_dict) == 0:
            print("\t*** There were NO statistically significant " \
                    "Lags detected. ***")
        else:
            return lag_series

    
    def get_trend_test(self, 
                       sig_pval: float=0.05, 
                       print_messages: bool=True) -> None:
        """Get results from a Linear Trend test

        Parameters
        ----------
        sig_pval : float default 0.05
            The p-value about which significance is determined
        print_messages : bool default True
            Whether or not to print the output messages

        Notes
        -----        
        In Econometrics, testing for Trend is easily accomplished via Ordinary
        Least Squares (OLS) regression. An ever-increasing integer at time (t)
        is constructed as follows:
            t1 = 1
            t2 = 2
            t3 = 3
            ...
            tn = n
        This is regressed against the input time series ('y'): therefore, if it
        is significant, then there is evidence of Trend in y.

        The residuals from this regression now represent the input time series,
        as the, "de-trended," series.
        
        """
        print("### LINEAR TREND TEST ###")
        df = self._get_trend_dataframe()
        df['t'] = range(len(df))
        ols = sm.OLS(df.t, sm.add_constant(df.y)).fit()
        print(ols.summary())
        linear_trend_pval = ols.pvalues[1:].item()
        self.linear_trend = linear_trend_pval < sig_pval
        if print_messages:
            print("\n#####  RESULTS: TEST FOR LINEAR TREND  #####")
            print("- With a p-value of {:0.4f}, there ".format(
                linear_trend_pval), end='')
            trend_msg = np.where(linear_trend_pval > 0.05, 
                                 "is *NOT ENOUGH* evidence for a linear trend", 
                                 "*APPEARS* to be evidence for a linear trend")
            print("{}".format(trend_msg))


    def get_trend_plot(self) -> None:
        """Get a visual plot of the Hodrick-Prescott filter for Trend
    
        Notes
        -----
        @TODO
        The Hodrick-Prescott filter 

        """
        df = self._get_trend_dataframe()
        cycle, trend = sm.tsa.filters.hpfilter(self.input_ts, 
                                               self._get_hp_lambda())
        df["cycle"] = cycle
        df["trend"] = trend
        fig, ax = plt.subplots(figsize=(12, 4))
        df['y'].plot(ax=ax, alpha=0.75)
        df["trend"].plot(ax=ax, ls='--', lw=2)
        plt.title("Trend: Hodrick-Prescott Filter ($\lambda$=" \
                "{:,d}, TS Frequency='{}')".format(self._get_hp_lambda(), 
                                                   self._get_data_freq()),
                                                   fontsize=11)
        plt.grid()
        plt.show()


    def get_spectral_graphs(self) -> None:
        """Get spectral graphs for possible seasonality 

        @TODO
            Return values
            Return DFs

        Notes
        -----
        @TODO
        The Periodogram is more granular

        Welch's reduces the Noise

        References
        ----------
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        
        """
        # Compute periodogram
        freq_spec, power_spec = periodogram(self.input_ts, self.max_season)        
        spec_df = pd.DataFrame({"freq": freq_spec, "power": power_spec})
        peak = spec_df.query("power.eq(power.max())")["freq"].item()
        spec_top2 = spec_df.sort_values("power", ascending=False)[:2]["freq"]
        
        # Compute Welch's
        freq_welch, power_welch = welch(self.input_ts, self.max_season)        
        welch_df = pd.DataFrame({"freq": freq_welch, "power": power_welch})
        peak_welch = welch_df.query("power.eq(power.max())")["freq"].item()
        welch_top2 = welch_df.sort_values("power", ascending=False)[:2]["freq"]
        
        plt.figure(figsize=(12, 4))
        plt.suptitle("Possible Seasonality Points")
        
        plt.subplot(1, 2, 1)
        ax1 = spec_df["power"].plot()
        ax1 = spec_df["power"].sort_values(ascending=
                                           False)[:2].plot(ax=ax1, 
                                                           marker='o', 
                                                           ls='')
        top_spec_vals  = ", ".join([str(np.round(x, 2)) 
                                        for x in spec_top2])
        plt.xlabel("Top 2 Candidates at: x=[{}]".format(top_spec_vals))
        plt.title("Periodogram")
        plt.grid()

        plt.subplot(1, 2, 2)
        ax2 = welch_df["power"].plot()
        ax2 = welch_df["power"].sort_values(ascending=
                                            False)[:2].plot(ax=ax2, 
                                                            marker='o', 
                                                            ls='')
        top_welch_vals  = ", ".join([str(np.round(x, 2)) 
                                         for x in welch_top2])
        plt.xlabel("Top 2 Candidates at: x=[{}]".format(top_welch_vals))
        plt.title("Welch's")
        plt.grid()
        
        plt.tight_layout()
        plt.show()


    def get_model_seasonalities(self, sig_pval: float=0.05) -> pd.DataFrame:
        """Get model seasonality tests

        Parameters
        ----------
        sig_pval : float default 0.05
            The p-value about which significance is determined

        Notes
        -----
        In Econometrics, seasonality is commonly tested by using dummy var-
        iables as regressors using Ordinary Least Squares (OLS) against the 
        time series. To test for Quarterly seasonality, for example, create
        4 dummy variables and regress *3 of the dummies against the input time 
        series, y. If any of the dummies are significant, then we can say there
        is a Quarterly seasonality effect.

        * The held out dummy is the "base" or "reference" category, to which
            the estimate for all dummies in the model are compared against. The
            held out dummy is still represented where the included quarter dum-
            mies are all equal to 0.

        The residuals of the regression now represent the, "de-seasoned" values.

        Another advantage of testing this way is in the case where multiple 
        seasonalities exist. Whereas using a procedure (Statsmodels' MSTL, 
        e.g.) you would have to know the seasonalities in advance to specify 
        them, this method can identify multiple seasonalities independently.
        
        """
        df = self._get_trend_dataframe()
        df["week"] = df.index.isocalendar().week.astype(float)
        df["month"] = df.index.month.astype(float)
        df["quarter"] = df.index.quarter.astype(float)
        df["year"] = df.index.year.astype(float)
        
        seasonalities_df = pd.DataFrame()
        
        for metric in ["week", "month", "quarter", "year"]:
            ols = sm.OLS.from_formula("y ~ C(" + metric + ")", df).fit()
            pval = pd.Series(ols.pvalues[1:], name="p_value")
            sig_pvals = pval[pval < sig_pval]
            seasonalities_df = pd.concat([seasonalities_df, sig_pvals])
            
        seasonalities_df.index.name = "SEASON"
        
        print("### SEASONALITY TESTS ###\n")

        if len(seasonalities_df) == 0:
            print("\t*** There were NO statistically significant " \
                    "Seasonalities detected. ***")
        else:
            print(seasonalities_df)
            return seasonalities_df

    
    # @TODO: Kruskal Seasonality Tests

    # @TODO: CH test
    
    
    def get_seasonal_autocorr(self) -> None:
        """Get an Autocorrelation plot to examine for possible Seasonality
        """
        plt.figure(figsize=(12, 4))
        plt.title("Visual Examination for Seasonal Component", fontsize=11)
        autocorrelation_plot(self.input_ts.tolist())


    def get_train_test(self, train_pct_or_n: float=0.95) -> Optional[Tuple]:
        """Get Train/Test split Plot

        Parameters
        ----------
        train_pct_or_n : float default 0.95
            The pct or n to apply to the Train set. If [0, 1], then treated as
            a percentage. If > 1, then treated as N.

        Returns
        -------
        Tuple of the resultant Train/Test split pd.Series

        Raises
        ------
        ValueError
            If the value passed for `train_pct_or_n` is equal to 1 or less than
            or equal to 0

        Notes
        -----
        95% may seem like a large (or very large) portion to dedicate to the 
        Train set, but consider daily data for two years. With 730 data points, 
        the Test set would contain 36/37 records (i.e., over one month). This
        would be *typically* far more than necessary to obtain a robust model 
        to forecast several days (i.e., more than a month) into the future.
        
        """
        if train_pct_or_n > 0 and train_pct_or_n < 1:
            total_n = len(self.input_ts)
            train_n = int(np.round(total_n * train_pct_or_n))
        elif train_pct_or_n > 1:
            train_n = train_pct_or_n
        else:
            raise ValueError("The value for `train_pct_or_n` cannot be equal " \
                                "to 1 or less than or equal to 0")
        train, test = self.input_ts[:train_n], self.input_ts[train_n:]
        train.index.freq = self._get_data_freq()
        test.index.freq = self._get_data_freq()
        
        return train, test


    def get_traintest_plot(self):
        """Get a plot of Train + Test

        """
        train, test = self.get_train_test()
        fig, ax = plot_series(train, test, labels=["train", "test"])
        ax.grid()
        plt.show()


    def get_profile_trend(self):
        """Get profile for Trend

        Show all the metrics and graphs to assess Trend
        """
        self.get_trend_plot()
        print()
        print()
        self.get_trend_test()
        print()
        print()


    def get_profile_seasonality(self):
        """Get profile for Seasonality

        Show all the metrics and graphs to assess Seasonality
        """
        self.get_seasonal_autocorr()
        print()
        print()
        self.get_spectral_graphs()
        print()
        print()
        self.get_model_seasonalities()
        print()


    def get_profile_ts(self):
        """Get Profile TS

        Run a logical sequence of TS Tell methods to profile the Time Series

        #try:
        #    self.get_trend_test()
        #    print()
        #except Exception as e:
        #    print(f"***WARNING*** An error occurred on get_trend_test(). " \
        #            f"The message was as follows:\n\t- `{e}`\n\n")
        
        """
        self.get_sample_facts()
        print()
        self.get_autocorr_plots()
        print()
        self.get_hist_box()
        print()
        self.get_smoothed_imputation()
        print()
        self.get_trend_plot()
        print()
        print()
        self.get_trend_test()
        print()
        print()
        self.get_lag_tests()
        print()
        self.get_seasonal_autocorr()
        print()
        self.get_spectral_graphs()
        print()
        self.get_model_seasonalities()
        print()
        self.get_traintest_plot()
        print()